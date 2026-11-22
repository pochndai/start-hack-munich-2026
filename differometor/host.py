"""
NumPy implementation of the simulation, wrapped as one JAX operation.

`differometor.simulate.simulate_jax` expresses the whole simulation in JAX, so XLA has
to trace and compile every component function and its derivative before the first
evaluation. On CPU that compilation is most of a first run. Here the component
functions, the assembly of the systems, the sparse solves and the reverse pass run on
the host, in NumPy and SciPy, and JAX only sees two callbacks. There is almost nothing
left to compile.

NumPy cannot differentiate, so two things are done by hand:

- `Dual` carries a value together with its derivatives with respect to the inputs of a
  component function. Running the unchanged component functions on `Dual` inputs
  yields their Jacobians (forward-mode differentiation).
- `_backward` and `_reverse` are the reverse pass of `_assemble` and of the solves,
  written out step by step. It follows JAX's convention for complex cotangents: a
  C-linear map y = M x pulls back as x_bar = M^T y_bar (no conjugation), y = conj(x) as
  x_bar = conj(y_bar), and a real input collects Re(y_bar * dy/dx).

The signal systems of all frequencies are solved together (`_Factors`): they are
stacked into one block-diagonal matrix, which SciPy's SuperLU factorises in a single
call. That one factorisation serves every solve of an evaluation, the gradient's
included (see `_forward`).

`simulate_host` has the signature and results of `simulate_jax` and supports reverse-
mode differentiation with respect to `optimized_parameters`.
"""

import importlib
import importlib.machinery
import operator
import os
import sys
import threading
import types

import jax
import jax.numpy as jnp
import numpy as np

from differometor import components
from differometor.components import F0, FUNCTIONS, H_PLANCK, UNIT_VACUUM

NOISE_SCALE = UNIT_VACUUM * H_PLANCK * F0 * 0.25


# -----------------------------------------------------------------------------
# Component functions on NumPy arrays
# -----------------------------------------------------------------------------


def _numpy_functions() -> list:
    """
    Copies of the component functions that compute with NumPy and return raw coefficients.

    The functions look up `jnp` and their helpers in their module globals when called, so
    each one is rebound to a copy of the module namespace in which `jnp` is NumPy. This
    leaves the real module untouched (nothing global is switched while JAX may be tracing).
    """
    namespace = dict(vars(components))
    namespace["jnp"] = np
    namespace["_RAW_OUTPUTS"] = True
    for name, value in vars(components).items():
        if isinstance(value, types.FunctionType) and value.__module__ == components.__name__:
            namespace[name] = types.FunctionType(
                value.__code__, namespace, name, value.__defaults__, value.__closure__
            )
    return [namespace[function.__name__] for function in FUNCTIONS]


NUMPY_FUNCTIONS = _numpy_functions()


class Dual:
    """
    A value together with its derivatives with respect to the seven function inputs.

    `value` has the shape of the data, `tangent` that shape with a leading axis of length
    seven (or anything that broadcasts to it). Only the operations the component functions
    use are implemented.
    """

    __slots__ = ("value", "tangent")
    __array_priority__ = 1000

    def __init__(self, value, tangent):
        self.value = value
        self.tangent = tangent

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value + other.value, self.tangent + other.tangent)
        return Dual(self.value + other, self.tangent)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value - other.value, self.tangent - other.tangent)
        return Dual(self.value - other, self.tangent)

    def __rsub__(self, other):
        return Dual(other - self.value, -self.tangent)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value * other.value, self.tangent * other.value + self.value * other.tangent)
        return Dual(self.value * other, self.tangent * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            quotient = self.value / other.value
            return Dual(quotient, (self.tangent - quotient * other.tangent) / other.value)
        return Dual(self.value / other, self.tangent / other)

    def __rtruediv__(self, other):
        quotient = other / self.value
        return Dual(quotient, -quotient * self.tangent / self.value)

    def __neg__(self):
        return Dual(-self.value, -self.tangent)

    def __pow__(self, exponent):
        return Dual(self.value**exponent, exponent * self.value ** (exponent - 1) * self.tangent)

    def __lt__(self, other):
        return self.value < _value(other)

    def __le__(self, other):
        return self.value <= _value(other)

    def __gt__(self, other):
        return self.value > _value(other)

    def __ge__(self, other):
        return self.value >= _value(other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            return NotImplemented
        if ufunc in _BINARY:
            left, right = inputs
            if isinstance(left, Dual):
                return _BINARY[ufunc](left, right)
            return _REFLECTED[ufunc](right, left)
        if ufunc in _UNARY:
            return _UNARY[ufunc](self)
        return NotImplemented

    def __array_function__(self, function, types_, args, kwargs):
        if function is np.where and not kwargs:
            condition, first, second = args
            return Dual(
                np.where(condition, _value(first), _value(second)),
                np.where(condition, _tangent(first), _tangent(second)),
            )
        return NotImplemented


def _value(x):
    return x.value if isinstance(x, Dual) else x


def _tangent(x):
    return x.tangent if isinstance(x, Dual) else 0.0


def _absolute(x):
    magnitude = np.abs(x.value)
    if np.iscomplexobj(x.value):
        return Dual(magnitude, np.real(np.conj(x.value) * x.tangent) / magnitude)
    return Dual(magnitude, np.sign(x.value) * x.tangent)


def _sqrt(x):
    root = np.sqrt(x.value)
    return Dual(root, x.tangent / (2 * root))


def _exp(x):
    exponential = np.exp(x.value)
    return Dual(exponential, exponential * x.tangent)


_BINARY = {
    np.add: operator.add,
    np.subtract: operator.sub,
    np.multiply: operator.mul,
    np.true_divide: operator.truediv,
    np.power: operator.pow,
}
_REFLECTED = {
    np.add: lambda dual, other: dual + other,
    np.subtract: lambda dual, other: dual.__rsub__(other),
    np.multiply: lambda dual, other: dual * other,
    np.true_divide: lambda dual, other: dual.__rtruediv__(other),
}
_UNARY = {
    np.negative: operator.neg,
    np.exp: _exp,
    np.sqrt: _sqrt,
    np.absolute: _absolute,
    np.sin: lambda x: Dual(np.sin(x.value), np.cos(x.value) * x.tangent),
    np.cos: lambda x: Dual(np.cos(x.value), -np.sin(x.value) * x.tangent),
    np.sinh: lambda x: Dual(np.sinh(x.value), np.cosh(x.value) * x.tangent),
    np.cosh: lambda x: Dual(np.cosh(x.value), np.sinh(x.value) * x.tangent),
    np.arcsin: lambda x: Dual(np.arcsin(x.value), x.tangent / np.sqrt(1 - x.value * x.value)),
    np.radians: lambda x: Dual(np.radians(x.value), np.radians(x.tangent)),
    np.deg2rad: lambda x: Dual(np.radians(x.value), np.radians(x.tangent)),
    np.conjugate: lambda x: Dual(np.conj(x.value), np.conj(x.tangent)),
    np.sign: lambda x: Dual(np.sign(x.value), 0.0),
}


class DualInputs:
    """
    The seven inputs of a component function as `Dual` values.

    Only the inputs listed in `seeded` get a derivative of their own; the tangent axis has
    one entry per seeded input, and every other input carries zeros there.
    """

    def __init__(self, inputs, seeded):
        self.inputs = inputs
        self.seeded = seeded

    def __getitem__(self, index):
        tangent = np.zeros((len(self.seeded),) + (1,) * (self.inputs.ndim - 1))
        tangent[self.seeded == index] = 1.0
        return Dual(self.inputs[index], tangent)


def _last_writes(*indices) -> np.ndarray:
    """Mark, among writes to possibly repeated positions, the one NumPy keeps (the last)."""
    flat = indices[0] if len(indices) == 1 else np.ravel_multi_index(indices, [int(i.max()) + 1 for i in indices])
    keep = np.zeros(len(flat), dtype=bool)
    if len(flat):
        keep[len(flat) - 1 - np.unique(flat[::-1], return_index=True)[1]] = True
    return keep


def _evaluate(parameters, function_input_indices, output_indices, function_indices, active):
    """
    NumPy counterpart of `differometor.simulate.evaluate`.

    Returns the selected outputs, shape (V, M), and, unless `active` is None, a function
    that maps cotangents of those outputs to cotangents of `parameters`, shape (V, P).

    `active` marks the parameters that depend on the optimized values, shape (P,).
    Derivatives are taken only with respect to function inputs that read such a parameter
    somewhere; the cotangent of every other parameter is discarded by the caller, so it is
    left at zero here.
    """
    output_rows, output_coefficients = output_indices
    value_count = parameters.shape[0]
    outputs = np.zeros((value_count, len(output_rows)), dtype=np.complex128)
    blocks = []
    for function_index in np.unique(function_indices):
        inputs = np.nonzero(function_indices == function_index)[0]
        function_inputs = np.moveaxis(parameters[:, function_input_indices[inputs]], -1, 0)
        seeded = () if active is None else np.nonzero(active[function_input_indices[inputs]].any(axis=0))[0]
        coefficients = NUMPY_FUNCTIONS[function_index](
            DualInputs(function_inputs, seeded) if len(seeded) else function_inputs
        )
        from_this_function = np.isin(output_rows, inputs)
        for coefficient in np.unique(output_coefficients[from_this_function]):
            selected = np.nonzero(from_this_function & (output_coefficients == coefficient))[0]
            positions = np.searchsorted(inputs, output_rows[selected])
            entry = coefficients[coefficient] if coefficient < len(coefficients) else 0
            block = np.broadcast_to(_value(entry), (value_count, len(inputs)))
            outputs[:, selected] = block[:, positions]
            if isinstance(entry, Dual):
                tangent = np.broadcast_to(entry.tangent, (len(seeded), value_count, len(inputs)))
                blocks.append((inputs, selected, positions, tangent, seeded))
    if active is None:
        return outputs, None

    def pull_back(output_cotangents):
        parameter_cotangents = np.zeros(parameters.shape)
        everything = slice(None)
        for inputs, selected, positions, tangent, seeded in blocks:
            per_input = np.zeros((value_count, len(inputs)), dtype=np.complex128)
            np.add.at(per_input, (everything, positions), output_cotangents[:, selected])
            contributions = np.real(per_input * tangent)
            for contribution, argument in zip(contributions, seeded):
                np.add.at(
                    parameter_cotangents,
                    (everything, function_input_indices[inputs, argument]),
                    contribution,
                )
        return parameter_cotangents

    return outputs, pull_back


def _signal_size(a):
    """SN, the size of the signal system, read from an array that has it as a dimension."""
    return a["noise_selection_vectors"].shape[-1]


def _expand(array, indices, values):
    """NumPy counterpart of `differometor.simulate.expand_parameters`."""
    expanded = np.tile(array, (values.shape[1], 1))
    expanded[np.arange(values.shape[1])[:, None], indices[None, :]] = values.T
    return expanded


def _assemble(optimized_parameters, a, static, differentiate, carrier=None):
    """
    Everything in `simulate` up to the signal solves.

    Returns a dictionary with the carrier solution, the entries and right-hand side of the
    signal system, the detector selections and the noise entries, plus what `_backward`
    needs. If `carrier` is given it is used instead of solving the carrier system again.
    """
    s = {}
    parameters = np.array(a["parameters"], dtype=np.float64)
    parameters[0, a["optimized_parameter_indices"]] = optimized_parameters[a["optimized_value_indices"]]
    # Parameter linking. Only identity links reach this function (see `supported`).
    parameters[0, a["indices_to_link"]] = parameters[0, a["linked_indices"]]
    # Which parameters depend on the optimized values: the two assignments above, on flags.
    # The swept parameters stay flagged; `_backward` drops their cotangents, as before.
    active = None
    if differentiate:
        active = np.zeros(parameters.shape[1], dtype=bool)
        active[a["optimized_parameter_indices"]] = True
        active[a["indices_to_link"]] = active[a["linked_indices"]]

    # Quadrature-homodyne-detector phases, multiplied into the selection vectors.
    s["phases"] = np.exp(1j * np.radians(parameters[0, a["qhd_parameter_indices"]]))
    s["qhd_places"] = tuple(a["qhd_placing_indices"].reshape(-1, 3).T)
    selections = np.array(a["noise_selection_vectors"], dtype=np.complex128)
    s["plain_selections"] = selections[s["qhd_places"]]
    selections[s["qhd_places"]] *= s["phases"]

    # Carrier system (dense, small).
    parameters = _expand(parameters, a["carrier_changing_parameter_indices"], a["carrier_changing_values"])
    carrier_outputs, s["pull_carrier"] = _evaluate(
        parameters,
        a["carrier_function_input_indices"],
        static["carrier_output_indices"],
        static["carrier_function_indices"],
        active,
    )
    carrier_count = parameters.shape[0]
    carrier_matrix = np.broadcast_to(
        np.asarray(a["carrier_matrix"], dtype=np.complex128), (carrier_count, *a["carrier_matrix"].shape)
    ).copy()
    carrier_matrix[:, a["carrier_matrix_indices"][0], a["carrier_matrix_indices"][1]] = carrier_outputs
    s["carrier_matrix"] = carrier_matrix[:, :, :-1]
    if carrier is None:
        carrier = np.linalg.solve(s["carrier_matrix"], carrier_matrix[:, :, -1][:, :, None])[:, :, 0]
    s["carrier"] = carrier
    carrier_size = carrier.shape[1]

    # Signal system: only the entries that can be non-zero, then the right-hand side.
    parameters = _expand(parameters, a["signal_changing_parameter_indices"], a["signal_changing_values"])
    value_count = parameters.shape[0]
    layout = a["signal_sparse_indices"]
    rows, columns, right_hand_side_rows = layout["rows"], layout["columns"], layout["right_hand_side_rows"]
    base = layout["base"]
    values = np.broadcast_to(base, (value_count, len(base))).copy()
    signal_outputs, s["pull_signal"] = _evaluate(
        parameters,
        a["signal_function_input_indices"],
        static["signal_output_indices"],
        static["signal_function_indices"],
        active,
    )
    values[:, layout["update_slots"]] = signal_outputs
    placing_slots = layout["placing_slots"]
    carrier_at_places = carrier[:, a["signal_carrier_indices"].ravel()]
    s["before_placing"] = values[:, placing_slots].copy()
    values[:, placing_slots] = -s["before_placing"] * carrier_at_places
    np.conjugate(values, out=values, where=layout["conjugated"])
    right_hand_side = np.zeros((value_count, _signal_size(a)), dtype=np.complex128)
    right_hand_side[:, right_hand_side_rows] = values[:, len(rows) :]

    # Detector selections scaled by the carrier (upper sideband) and its conjugate (lower).
    selections = np.tile(selections, (1, carrier_count, 1))
    s["upper_selections"] = selections[:, :, :carrier_size].copy()
    s["lower_selections"] = selections[:, :, carrier_size : 2 * carrier_size].copy()
    selections[:, :, :carrier_size] = s["upper_selections"] * carrier
    selections[:, :, carrier_size : 2 * carrier_size] = np.conj(s["lower_selections"] * carrier)
    selections = np.moveaxis(
        np.broadcast_to(selections, (selections.shape[0], value_count, selections.shape[2])), 0, -1
    )

    # Noise matrix entries.
    noise_layout = a["noise_sparse_indices"]
    noise_base = noise_layout["base"]
    noise_values = np.broadcast_to(noise_base, (value_count, len(noise_base))).copy()
    noise_outputs, s["pull_noise"] = _evaluate(
        parameters,
        a["noise_function_input_indices"],
        static["noise_output_indices"],
        static["noise_function_indices"],
        active,
    )
    noise_values[:, noise_layout["update_slots"]] = noise_outputs

    s.update(
        signal_values=values[:, : len(rows)],
        right_hand_side=right_hand_side,
        selections=np.ascontiguousarray(selections),
        noise_values=noise_values,
        carrier_at_places=carrier_at_places,
        value_count=value_count,
        carrier_count=carrier_count,
    )
    return s


# -----------------------------------------------------------------------------
# Sparse solves
# -----------------------------------------------------------------------------


def _load_superlu():
    """
    SciPy's own entry point to SuperLU (`gstrf`), without the rest of `scipy.sparse.linalg`.

    `scipy.sparse.linalg.splu` is the public way in. It is not used for two reasons. For
    a column order that is already fixed it switches SuperLU to pivoting on the
    diagonal, which these badly scaled matrices do not tolerate. And importing
    `scipy.sparse.linalg` also loads SciPy's eigenvalue and iterative solvers,
    `scipy.linalg` and `scipy.sparse`, which takes longer than every other import of this
    package together, in a benchmark that counts imports. Only the compiled SuperLU
    module is needed, so it is loaded from its file. It is not registered in
    `sys.modules`, so a later `import scipy.sparse.linalg` behaves as usual.

    `gstrf` is private to SciPy and present in the pinned scipy 1.18.1. If loading it
    this way fails, the normal import is tried; if that fails too, `_factorize` uses
    `splu`.
    """
    name = "scipy.sparse.linalg._dsolve._superlu"
    if name in sys.modules:
        return sys.modules[name].gstrf
    try:
        # Where SciPy is installed, without importing it.
        package = importlib.machinery.PathFinder.find_spec("scipy")
        if os.name == "nt":
            import scipy  # noqa: F401  on Windows this registers the folder of SciPy's bundled libraries

        folder = os.path.join(os.path.dirname(package.origin), "sparse", "linalg", "_dsolve")
        for entry in sorted(os.listdir(folder)):
            if entry.startswith("_superlu") and entry.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                loader = importlib.machinery.ExtensionFileLoader(name, os.path.join(folder, entry))
                module = loader.create_module(importlib.machinery.ModuleSpec(name, loader, origin=loader.path))
                loader.exec_module(module)
                return module.gstrf
    except Exception:
        pass
    try:
        return importlib.import_module(name).gstrf
    except Exception:
        return None


_gstrf = _load_superlu()


def _no_factor_matrices(*arguments):
    raise NotImplementedError("the L and U factors are only used for solves here, never read as matrices")


REFINEMENT_STEPS = 1
"""
Rounds of iterative refinement after a solve: compute the residual b - A x, solve for
the correction with the same factors, add it.

The signal matrices have entries from 1e-22 to 1e13. A plain LU solve is then accurate in
the large entries of the solution but not always in the small ones, which the noise
depends on, and how accurate depends on the column order SuperLU happens to use. One
round removes that dependence (measured against exact solutions in tests/solver_experiment.py).
"""

SIGNAL_REFINEMENT_STEPS = 0
"""
The same for the solve that gives the signal and the adjoint of the noise in `_forward`.

Neither needs the small entries: the signal is read out where it is large, and the adjoint
only enters the gradient. Without refinement the signal agrees with the JAX version to
1e-10 or better on the test setups and the gradients are unchanged at the level the tests
resolve (tests/test_host.py), for one triangular solve and one matrix product less per
right-hand side. The noise weights keep `REFINEMENT_STEPS`.
"""

_LAYOUTS = {}
"""Sparsity pattern and sweep length -> what `_Factors` needs to know about them (`_solver_layout`)."""


def _factorize(size, data, indices, pointers, ordering="NATURAL"):
    """
    LU factors (SuperLU) of the matrix given column by column.

    The supernode settings are turned down to one: SuperLU pads its factors to form dense
    blocks, which only costs time on matrices with a handful of entries per column.
    """
    data = np.ascontiguousarray(data)
    if _gstrf is None:
        from scipy.sparse import csc_array
        from scipy.sparse.linalg import splu

        return splu(csc_array((data, indices, pointers), shape=(size, size)), permc_spec="MMD_ATA", relax=1, panel_size=1)
    options = dict(ColPerm=ordering, DiagPivotThresh=None, PanelSize=1, Relax=1, SymmetricMode=False)
    return _gstrf(size, len(data), data, indices, pointers, csc_construct_func=_no_factor_matrices, ilu=False, options=options)


def _solver_layout(layout, size, value_count, entries):
    """
    Index arrays for the stacked matrix of `_Factors`, found once per sparsity pattern.

    The column order that keeps the factors sparse depends on the pattern only, and the
    pattern is the same at every frequency. So SuperLU orders one block once (its MMD_ATA
    order, which gave the smallest factors and the most accurate solves on these
    matrices), and every later factorisation reuses that order.
    """
    rows, columns, order = layout["rows"], layout["columns"], layout["transposed_order"]
    key = (size, value_count, rows.tobytes(), columns.tobytes())
    known = _LAYOUTS.get(key)
    if known is not None:
        return known
    column_starts = np.concatenate([[0], np.cumsum(np.bincount(columns, minlength=size))])

    def block(column_order):
        """One block, column by column in the given order: where its entries are, their rows, the column starts."""
        gather = np.concatenate([order[column_starts[column] : column_starts[column + 1]] for column in column_order])
        pointers = np.concatenate([[0], np.cumsum(np.diff(column_starts)[column_order])]).astype(np.intc)
        return gather, rows[gather].astype(np.intc), pointers

    column_order = np.arange(size)
    if _gstrf is not None:
        middle = entries[value_count // 2]
        gather, indices, pointers = block(column_order)
        chosen = np.asarray(_factorize(size, middle[gather], indices, pointers, ordering="MMD_ATA").perm_c)
        # SuperLU reports where each column goes (Pr A Pc = L U with Pc[i, perm_c[i]] = 1), so the
        # columns in their new order are the inverse permutation. Any order gives correct factors;
        # this one reproduces SuperLU's own, the sparsest.
        column_order = np.argsort(chosen)
    gather, indices, pointers = block(column_order)
    blocks = np.arange(value_count, dtype=np.int64)[:, None]
    row_starts = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=size))])
    if (np.diff(row_starts) == 0).any() or (np.diff(column_starts) == 0).any():
        raise ValueError("The signal matrix has an empty row or column; it is singular.")
    known = {
        "column_order": column_order,
        "gather": gather,
        # The stacked matrix column by column, for the factorisation.
        "indices": (indices[None, :] + size * blocks).ravel().astype(np.intc),
        "pointers": np.concatenate([[0], np.cumsum(np.tile(np.diff(pointers), value_count))]).astype(np.intc),
        # For products with the matrix (`_Factors._apply`): the entries of one block row by row, as they
        # are stored, and column by column, and where each row or column starts in the stacked entries.
        "columns": columns,
        "by_column": order,
        "rows_by_column": rows[order],
        "row_starts": (row_starts[None, :-1] + len(rows) * blocks).ravel(),
        "column_starts": (column_starts[None, :-1] + len(rows) * blocks).ravel(),
    }
    if len(_LAYOUTS) >= 8:
        _LAYOUTS.clear()
    _LAYOUTS[key] = known
    return known


class _Factors:
    """
    One LU factorisation for the signal matrices of all frequencies.

    The matrices are independent, so they form one block-diagonal matrix of size
    (frequencies x system size). Factorising and solving that matrix is the same
    arithmetic as doing it frequency by frequency, in one call to SuperLU instead of one
    per frequency. The factors serve solves with the matrix (`"N"`), its transpose (`"T"`)
    and its conjugate transpose (`"H"`).

    Parameters
    ----------
    layout :
        `signal_sparse_indices` of the build step.
    entries :
        The non-zero entries of every matrix, row by row, shape (SV, NNZ).
    size :
        SN, the size of one matrix.
    """

    def __init__(self, layout, entries, size):
        self.known = _solver_layout(layout, size, entries.shape[0], entries)
        self.entries, self.total = entries, entries.shape[0] * size
        self.factors = _factorize(self.total, entries[:, self.known["gather"]].ravel(), self.known["indices"], self.known["pointers"])
        self.entries_by_column = None

    def _once(self, right_hand_side, transpose):
        order = self.known["column_order"]
        ordered = right_hand_side if transpose == "N" else right_hand_side[:, order]
        flat = np.ascontiguousarray(ordered).reshape(self.total, *right_hand_side.shape[2:])
        solution = self.factors.solve(flat, transpose).reshape(right_hand_side.shape)
        if transpose != "N":
            return solution
        restored = np.empty_like(solution)
        restored[:, order] = solution
        return restored

    def _apply(self, vectors, transpose):
        """A x, A^T x or A^H x for every frequency: multiply entry by entry, then sum each row (or column)."""
        known = self.known
        if transpose == "H":
            return np.conj(self._apply(np.conj(vectors), "T"))
        if transpose == "N":
            entries, sources, starts = self.entries, known["columns"], known["row_starts"]
        else:
            if self.entries_by_column is None:
                self.entries_by_column = self.entries[:, known["by_column"]]
            entries, sources, starts = self.entries_by_column, known["rows_by_column"], known["column_starts"]
        # One right-hand side at a time, so that every array in the loop is flat and contiguous.
        columns = vectors.reshape(*vectors.shape[:2], -1)
        result = np.empty_like(columns)
        for index in range(columns.shape[2]):
            products = np.take(columns[:, :, index], sources, axis=1)
            np.multiply(products, entries, out=products)
            result[:, :, index] = np.add.reduceat(products.ravel(), starts).reshape(vectors.shape[:2])
        return result.reshape(vectors.shape)

    def solve(self, right_hand_side, transpose="N", refinement_steps=None):
        """Solve for every frequency. `right_hand_side` has shape (SV, SN) or (SV, SN, D)."""
        right_hand_side = np.asarray(right_hand_side, dtype=np.complex128)
        solution = self._once(right_hand_side, transpose)
        for _ in range(REFINEMENT_STEPS if refinement_steps is None else refinement_steps):
            solution = solution + self._once(right_hand_side - self._apply(solution, transpose), transpose)
        return solution


_KEPT = {}
"""Ticket -> (optimized parameters, result of `_assemble`) for the latest forward passes with a pullback."""
_KEPT_LOCK = threading.Lock()


def _keep(optimized_parameters, s):
    """
    Keep what `_assemble` produced in a forward pass for the reverse pass that follows.

    The reverse pass needs the same matrices and the derivatives of the component functions
    at the same parameters. They are not JAX arrays (they include functions), so they stay
    here, and the forward pass hands JAX a ticket for them. Two are kept; if a reverse pass
    comes with a ticket that is gone, it assembles again, as it would without this.
    """
    ticket = int.from_bytes(os.urandom(6), "little")  # random, so a ticket from another process finds nothing
    with _KEPT_LOCK:
        while len(_KEPT) >= 2:
            _KEPT.pop(next(iter(_KEPT)))
        _KEPT[ticket] = (np.array(optimized_parameters), s)
    return np.int64(ticket)


def _kept(ticket, optimized_parameters):
    with _KEPT_LOCK:
        entry = _KEPT.get(int(ticket))
    if entry is not None and np.array_equal(entry[0], optimized_parameters):
        return entry[1]
    return None


def _weight_products(noise_layout, noise_values, weights):
    """The factors conj(w[r_e]) and N_e w[k_e] of the noise quadratic form, each of shape (SV, E, D)."""
    return np.conj(weights[:, noise_layout["rows"]]), noise_values[:, :, None] * weights[:, noise_layout["columns"]]


def _forward(optimized_parameters, a, static, adjoint):
    """
    The forward simulation: carrier (CV, CN), signal (SV, SN), noise weights (SV, SN, D), noise (D, SV).

    With `adjoint`, also the solve the gradient of the noise will need, shape (SV, SN, D),
    and a ticket for the assembled systems (`_keep`), which are then built with the
    derivatives of the component functions so that the reverse pass does not build them again.
    The noise is sqrt(2 K q) with q = Re(w^H N w). Its reverse pass has to solve
    conj(A) mu = w_bar, and w_bar is a real factor per frequency and detector (the
    cotangent of q, known only later) times a vector known now. So the solve is done
    here, with the factors that exist anyway, and the gradient only scales the result.
    """
    a = _numpy(a)
    optimized_parameters = np.asarray(optimized_parameters)
    s = _assemble(optimized_parameters, a, static, differentiate=adjoint)
    factors = _Factors(a["signal_sparse_indices"], s["signal_values"], _signal_size(a))
    weights = factors.solve(s["selections"], "H")
    noise_layout = a["noise_sparse_indices"]
    left, right = _weight_products(noise_layout, s["noise_values"], weights)
    noise = np.sqrt(2 * np.real(np.sum(left * right, axis=1)).T * NOISE_SCALE)
    if not adjoint:
        return s["carrier"], factors.solve(s["right_hand_side"]), weights, noise
    everything = slice(None)
    weight_cotangent = np.zeros_like(weights)  # of q, for a cotangent of one
    np.add.at(weight_cotangent, (everything, noise_layout["columns"]), left * s["noise_values"][:, :, None])
    np.add.at(weight_cotangent, (everything, noise_layout["rows"]), np.conj(right))
    # The signal and the noise adjoint are solves with the same matrix: one call with several right-hand sides.
    both = factors.solve(np.concatenate([s["right_hand_side"][:, :, None], np.conj(weight_cotangent)], axis=2), refinement_steps=SIGNAL_REFINEMENT_STEPS)
    return s["carrier"], np.ascontiguousarray(both[:, :, 0]), weights, noise, np.conj(both[:, :, 1:]), _keep(optimized_parameters, s)


def _signal_adjoint(a, s, weights, cotangent):
    """
    lambda with A^T lambda = x_bar, the reverse pass of the signal solve, shape (SV, SN).

    The noise weights solve A^H w = c for the detector vectors c, so conj(w) solves
    A^T conj(w) = conj(c). When x_bar is a combination of the conj(c), as it is for a signal
    read out at the noise detector, lambda is the same combination of the conj(w) and no
    solve is needed. Otherwise the signal matrices are factorised again here.
    """
    if not cotangent.any():
        return np.zeros_like(cotangent)
    selections = s["selections"]
    try:
        # Least squares for x_bar = sum_d coefficients[d] conj(c[d]), per frequency.
        gram = np.einsum("fnd,fne->fde", selections, np.conj(selections))
        coefficients = np.linalg.solve(gram, np.einsum("fnd,fn->fd", selections, cotangent)[:, :, None])[:, :, 0]
        remainder = cotangent - np.einsum("fnd,fd->fn", np.conj(selections), coefficients)
        if np.all(np.abs(remainder).max(axis=1) <= 1e-12 * np.abs(cotangent).max(axis=1)):
            return np.einsum("fnd,fd->fn", np.conj(weights), coefficients)
    except np.linalg.LinAlgError:
        pass
    factors = _Factors(a["signal_sparse_indices"], s["signal_values"], _signal_size(a))
    return factors.solve(cotangent, "T")


def _backward(optimized_parameters, a, carrier, signal, weights, noise, noise_adjoint, ticket, carrier_cotangent, signal_cotangent, noise_cotangent, static):
    """
    Reverse pass of the whole simulation: the cotangents of the three results in, the
    cotangent of `optimized_parameters` out.

    `noise_adjoint` and `ticket` are the last two results of `_forward`.
    """
    a = _numpy(a)
    carrier, signal, weights, noise = (np.asarray(array) for array in (carrier, signal, weights, noise))
    optimized_parameters = np.asarray(optimized_parameters)
    s = _kept(ticket, optimized_parameters)
    if s is None:
        s = _assemble(optimized_parameters, a, static, differentiate=True, carrier=carrier)
    # noise = sqrt(2 K q): cotangent of q, shape (SV, D). A noise of zero has no derivative.
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_factor = (np.asarray(noise_cotangent).reshape(noise.shape) * NOISE_SCALE / noise).T
    weight_adjoint = noise_factor[:, None, :] * np.asarray(noise_adjoint)
    signal_adjoint = _signal_adjoint(a, s, weights, np.asarray(signal_cotangent, dtype=np.complex128))
    return _reverse(optimized_parameters, a, s, carrier, signal, weights, signal_adjoint, weight_adjoint, noise_factor, np.asarray(carrier_cotangent))


def _reverse(optimized_parameters, a, s, carrier, signal, weights, signal_adjoint, weight_adjoint, noise_factor, carrier_cotangent):
    """
    Reverse pass after the adjoint solves.

    Parameters
    ----------
    s :
        Result of `_assemble` with `differentiate=True`.
    carrier, signal, weights :
        Forward results: carrier solution (CV, CN), signal solution (SV, SN) and noise
        weights (SV, SN, D).
    signal_adjoint :
        Solution of A^T lambda = signal cotangent, shape (SV, SN).
    weight_adjoint :
        Solution of conj(A) mu = weight cotangent, shape (SV, SN, D).
    noise_factor :
        Cotangent of the noise quadratic form, shape (SV, D).
    carrier_cotangent :
        Cotangent of the carrier solution itself, shape (CV, CN).

    Returns
    -------
    Cotangent of `optimized_parameters`.
    """
    value_count, carrier_count, carrier_size = s["value_count"], s["carrier_count"], carrier.shape[1]
    everything = slice(None)

    def to_carrier_values(array):
        # Sum over the signal sweep when one carrier solution serves every signal value.
        return array.reshape(value_count // carrier_count, carrier_count, *array.shape[1:]).sum(axis=0)

    # Noise quadratic form  q = Re(sum_e conj(w[r_e]) N_e w[k_e])  ->  noise matrix entries.
    noise_layout = a["noise_sparse_indices"]
    noise_cotangents = np.sum(
        np.asarray(noise_factor)[:, None, :]
        * np.conj(weights[:, noise_layout["rows"]])
        * weights[:, noise_layout["columns"]],
        axis=2,
    )
    noise_updates = noise_layout["update_slots"]
    signal_parameter_cotangents = s["pull_noise"](noise_cotangents[:, noise_updates] * _last_writes(noise_updates))

    # Signal system entries: from x = A^-1 b and from w = A^-H c.
    layout = a["signal_sparse_indices"]
    rows, columns = layout["rows"], layout["columns"]
    value_cotangents = np.empty((value_count, len(rows) + len(layout["right_hand_side_rows"])), dtype=np.complex128)
    value_cotangents[:, : len(rows)] = -signal_adjoint[:, rows] * signal[:, columns] - np.sum(
        np.conj(weights[:, rows] * weight_adjoint[:, columns]), axis=2
    )
    value_cotangents[:, len(rows) :] = signal_adjoint[:, layout["right_hand_side_rows"]]
    value_cotangents = np.where(layout["conjugated"], np.conj(value_cotangents), value_cotangents)

    # Undo the scaling of the connector entries by the carrier.
    placing_slots = layout["placing_slots"]
    placed = value_cotangents[:, placing_slots] * _last_writes(placing_slots)
    carrier_cotangents = np.array(carrier_cotangent, dtype=np.complex128)
    per_place = np.zeros((carrier_count, carrier_size), dtype=np.complex128)
    np.add.at(per_place, (everything, a["signal_carrier_indices"].ravel()), to_carrier_values(-placed * s["before_placing"]))
    carrier_cotangents += per_place
    value_cotangents[:, placing_slots] = -placed * s["carrier_at_places"]
    updates = layout["update_slots"]
    signal_parameter_cotangents += s["pull_signal"](value_cotangents[:, updates] * _last_writes(updates))

    # Detector selections c (cotangent: mu) -> carrier and homodyne phases.
    selection_cotangents = to_carrier_values(np.moveaxis(weight_adjoint, -1, 0).swapaxes(0, 1)).swapaxes(0, 1)
    upper = selection_cotangents[:, :, :carrier_size]
    lower = np.conj(selection_cotangents[:, :, carrier_size : 2 * carrier_size])
    carrier_cotangents += np.sum(upper * s["upper_selections"] + lower * s["lower_selections"], axis=0)
    plain_cotangents = selection_cotangents.copy()
    plain_cotangents[:, :, :carrier_size] = upper * carrier
    plain_cotangents[:, :, carrier_size : 2 * carrier_size] = lower * carrier
    plain_cotangents = plain_cotangents.sum(axis=1, keepdims=True)
    phase_cotangents = plain_cotangents[s["qhd_places"]] * s["plain_selections"]

    # Carrier system: carrier = M^-1 r.
    carrier_adjoint = np.linalg.solve(s["carrier_matrix"].transpose(0, 2, 1), carrier_cotangents[:, :, None])[:, :, 0]
    matrix_rows, matrix_columns = a["carrier_matrix_indices"]
    # The builder addresses the right-hand-side column as -1.
    matrix_rows, matrix_columns = matrix_rows % carrier_size, matrix_columns % (carrier_size + 1)
    in_matrix = matrix_columns < carrier_size
    carrier_output_cotangents = np.where(
        in_matrix,
        -carrier_adjoint[:, matrix_rows] * carrier[:, np.where(in_matrix, matrix_columns, 0)],
        carrier_adjoint[:, matrix_rows],
    ) * _last_writes(matrix_rows, matrix_columns)
    carrier_parameter_cotangents = s["pull_carrier"](carrier_output_cotangents)

    # Back through the two parameter expansions, the links and the optimized values.
    signal_changes = a["signal_changing_values"].shape[1]
    signal_parameter_cotangents[np.arange(signal_changes)[:, None], a["signal_changing_parameter_indices"][None, :]] = 0
    carrier_parameter_cotangents += to_carrier_values(signal_parameter_cotangents)
    carrier_changes = a["carrier_changing_values"].shape[1]
    carrier_parameter_cotangents[np.arange(carrier_changes)[:, None], a["carrier_changing_parameter_indices"][None, :]] = 0
    parameter_cotangents = carrier_parameter_cotangents.sum(axis=0)
    np.add.at(
        parameter_cotangents,
        a["qhd_parameter_indices"],
        np.real(phase_cotangents * 1j * np.radians(1.0) * s["phases"]),
    )
    linked = parameter_cotangents[a["indices_to_link"]].copy()
    parameter_cotangents[a["indices_to_link"]] = 0
    np.add.at(parameter_cotangents, a["linked_indices"], linked)
    cotangent = np.zeros(np.shape(optimized_parameters), dtype=np.float64)
    np.add.at(cotangent, a["optimized_value_indices"], parameter_cotangents[a["optimized_parameter_indices"]])
    return cotangent


def _numpy(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


# -----------------------------------------------------------------------------
# The JAX operation
# -----------------------------------------------------------------------------

STATIC_NAMES = (
    "carrier_function_indices",
    "carrier_output_indices",
    "signal_function_indices",
    "signal_output_indices",
    "noise_function_indices",
    "noise_output_indices",
)


def supported(linking_functions) -> bool:
    """
    Whether this implementation covers the setup being traced.

    Linking functions are arbitrary user callables that would have to be differentiated
    here as well; only the identity link that the builder inserts by default is handled.
    """
    identity = (lambda x: x).__code__.co_code
    return all(getattr(function, "__code__", None) is not None and function.__code__.co_code == identity for function in linking_functions)


def simulate_host(optimized_parameters, **arrays):
    """Run `simulate` on the host. Same arguments and results as `simulate_jax`."""
    static = {name: arrays.pop(name).array for name in STATIC_NAMES}
    # The dense signal and noise matrices stay out: everything this path reads from them is in the
    # sparse layouts (`base`). They are megabytes each, and whatever is handed to the callbacks is
    # copied between JAX and the host on every call and kept for the reverse pass.
    unused = ("linking_function_indices", "signal_matrix", "noise_matrix")
    arrays = {name: value for name, value in arrays.items() if name not in unused}
    return _simulate(static_holder(static), optimized_parameters, arrays)


class static_holder:
    """Hashable wrapper so the static index arrays can be a non-differentiable argument."""

    def __init__(self, static):
        self.static = static
        self._key = tuple((name, value.shape, value.tobytes()) for name, value in sorted(static.items()))

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, static_holder) and self._key == other._key


def _shapes(a):
    carrier_count = a["carrier_changing_values"].shape[1]
    value_count = carrier_count * a["signal_changing_values"].shape[1]
    return carrier_count, value_count, a["carrier_matrix"].shape[0], _signal_size(a), a["noise_selection_vectors"].shape[0]


def _primal(holder, optimized_parameters, a, adjoint):
    """`_forward` as one JAX operation. Returns the three results of `simulate` and what the reverse pass keeps."""
    carrier_count, value_count, carrier_size, signal_size, detectors = _shapes(a)
    complex_array = lambda *shape: jax.ShapeDtypeStruct(shape, jnp.complex128)
    results = [
        complex_array(carrier_count, carrier_size),
        complex_array(value_count, signal_size),
        complex_array(value_count, signal_size, detectors),
        jax.ShapeDtypeStruct((detectors, value_count), jnp.float64),
    ]
    if adjoint:
        results += [complex_array(value_count, signal_size, detectors), jax.ShapeDtypeStruct((), jnp.int64)]
    carrier, signal, weights, noise, *for_reverse = jax.pure_callback(
        lambda parameters, arrays: _forward(parameters, arrays, holder.static, adjoint),
        tuple(results),
        optimized_parameters,
        a,
        vmap_method="sequential",
    )
    return (carrier.T, signal.T, noise.squeeze()), (carrier, signal, weights, noise, *for_reverse)


def _simulate_primal(holder, optimized_parameters, a):
    return _primal(holder, optimized_parameters, a, adjoint=False)[0]


# The static index arrays are not differentiated; they reach both rules as first argument.
_simulate_differentiable = jax.custom_vjp(_simulate_primal, nondiff_argnums=(0,))


def _simulate_forward(holder, optimized_parameters, a):
    outputs, residuals = _primal(holder, optimized_parameters, a, adjoint=True)
    return outputs, (optimized_parameters, a, *residuals)


def _simulate_backward(holder, residuals, cotangents):
    optimized_parameters, a, carrier, signal, weights, noise, noise_adjoint, ticket = residuals
    carrier_cotangent, signal_cotangent, noise_cotangent = cotangents
    cotangent = jax.pure_callback(
        lambda *arguments: _backward(*arguments, holder.static),
        jax.ShapeDtypeStruct(optimized_parameters.shape, jnp.float64),
        optimized_parameters,
        a,
        carrier,
        signal,
        weights,
        noise,
        noise_adjoint,
        ticket,
        carrier_cotangent.T,
        signal_cotangent.T,
        noise_cotangent,
        vmap_method="sequential",
    )
    # Only `optimized_parameters` is differentiated here. If a caller differentiates with
    # respect to one of the other arrays, NaN makes that visible; zeros would look like a
    # valid gradient. When those arrays are plain inputs, as usual, this is never computed.
    unsupported = jax.tree_util.tree_map(
        lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.inexact) else np.zeros(x.shape, dtype=jax.dtypes.float0),
        a,
    )
    return cotangent, unsupported


_simulate_differentiable.defvjp(_simulate_forward, _simulate_backward)


def _simulate(holder, optimized_parameters, a):
    return _simulate_differentiable(holder, optimized_parameters, a)
