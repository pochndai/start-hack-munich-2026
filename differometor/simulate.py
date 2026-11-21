"""
JAX-based optical simulation runner.

This module builds linear systems from a `differometor` `Setup` and solves:
- a carrier (steady-state) system,
- a signal (sideband) system,
- and a quantum noise propagation system.

The main public entry points are:
- `run`: build once and simulate once
- `run_setups`: run multiple setups over a shared frequency grid
- `run_with_parameter_sets`: run the same setup over many optimized-parameter sets

Conventions used throughout:
- Arrays are JAX arrays (`jax.numpy`).
- Shapes are documented explicitly for every function.
- All indices are integer arrays that index into parameter vectors and matrix entries.
"""

import jax
from time import time
import numpy as np
import klujax
import jax.numpy as jnp
from jax import lax, vmap, jit

from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0, numpy_backend


jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class Static:
    """
    Build-time NumPy array that `jax.jit` treats as part of the program, not as an input.

    `simulate` needs to know which component function fills which matrix entry while it
    is being traced. Passing that as a normal array hides the values from the tracer and
    forces every function to be compiled for every entry.
    """

    def __init__(self, array):
        self.array = np.asarray(array)
        self._key = (self.array.shape, self.array.dtype.str, self.array.tobytes())

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, Static) and self._key == other._key


def solve(matrix: jnp.ndarray, right_hand_side: jnp.ndarray) -> jnp.ndarray:
    """
    Solve a dense linear system for each batch item.

    Parameters
    ----------
    matrix : jnp.ndarray
        System matrix with shape (..., system_size, system_size).
    right_hand_side : jnp.ndarray
        Right-hand side vector with shape (..., system_size).

    Returns
    -------
    solution : jnp.ndarray
        Solution vector with shape (..., system_size).

    Notes
    -----
    Both inputs are cast to complex128 to reduce numerical issues during solving.

    The function intentionally returns the raw complex solution rather than derived
    quantities (such as phase or amplitude), because some downstream transforms
    (notably `jnp.angle`) can produce invalid gradients in some configurations.
    """
    right_hand_side = right_hand_side.astype(jnp.complex128)
    matrix = matrix.astype(jnp.complex128)
    return jnp.linalg.solve(matrix, right_hand_side)


def evaluate(
    parameters: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: Static,
) -> jnp.ndarray:
    """
    Evaluate component functions and select the outputs that go into a system matrix.

    Parameters
    ----------
    parameters : jnp.ndarray
        Parameter values, shape (P,) for one simulation instance or (V, P) for V of them.
    function_input_indices : jnp.ndarray
        Indices into `parameters` selecting inputs to the component functions,
        shape (K, 7).
    output_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting entries from the evaluated output matrix.
        Each element has shape (M,). Together they define M output positions.
    function_indices : Static
        Indices selecting which function to use for each input, shape (K,).
        Values index into the global `FUNCTIONS` list. Known at trace time.

    Returns
    -------
    outputs : jnp.ndarray
        Selected function outputs, shape (M,) or (V, M).

    Notes
    -----
    The function indices are known while tracing, so each component function is
    traced once and only for the inputs that use it. Selecting with `lax.switch`
    under `vmap` instead evaluates every function for every input and compiles
    several times slower.

    The component functions work elementwise on their seven inputs, so each one is
    called once on arrays of shape (7, ..., K_f) instead of under `vmap`, which is
    slower to trace.
    """
    function_indices = function_indices.array
    output_matrix = jnp.zeros((*parameters.shape[:-1], len(function_indices), 4), dtype=jnp.complex128)
    for function_index in np.unique(function_indices):
        inputs = np.nonzero(function_indices == function_index)[0]
        function_inputs = jnp.moveaxis(parameters[..., function_input_indices[inputs]], -1, 0)
        function_outputs = jnp.moveaxis(FUNCTIONS[function_index](function_inputs), 0, -1)
        output_matrix = output_matrix.at[..., inputs, :].set(function_outputs)
    return output_matrix[..., output_indices[0], output_indices[1]]


def update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: Static,
) -> jnp.ndarray:
    """
    Update selected entries of a dense system matrix by evaluating component functions.

    Parameters
    ----------
    parameters, function_input_indices, output_indices, function_indices :
        See `evaluate`. `parameters` has shape (P,) here.
    matrix : jnp.ndarray
        Base matrix to update, shape (N, N + 1) for augmented systems, or (N, N).
    matrix_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting where to write into `matrix`.
        Each element has shape (M,).

    Returns
    -------
    updated_matrix : jnp.ndarray
        Matrix with updated entries, same shape as `matrix`.
    """
    outputs = evaluate(parameters, function_input_indices, output_indices, function_indices)
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def expand_parameters(array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Expand a single parameter row into multiple rows by replacing selected entries.

    Parameters
    ----------
    array : jnp.ndarray
        Base parameter array with shape (1, P).
    indices : jnp.ndarray
        Column indices to replace, shape (V,).
    values : jnp.ndarray
        Replacement values stacked row-wise for each index, shape (V, R),
        where R is the number of expanded rows.

        Example:
        - array shape (1, 5)
        - indices shape (2,)
        - values shape (2, 3)
        Output has shape (3, 5), where the two indexed columns are filled from
        `values[:, row]`.

    Returns
    -------
    expanded : jnp.ndarray
        Expanded parameter array with shape (R, P).

    Notes
    -----
    This implementation uses indexed updates that remain compatible with JAX
    transformations and compilation.
    """
    tiled_array = jnp.tile(array, (values.shape[1], 1))
    row_indices = jnp.repeat(jnp.arange(values.shape[1]), len(indices))
    column_indices = jnp.tile(indices, values.shape[1])
    flat_values = values.T.flatten()
    return tiled_array.at[row_indices, column_indices].set(flat_values)


def simulate(
    # prepared arrays
    optimized_parameters: jnp.ndarray,
    optimized_parameter_indices: jnp.ndarray,
    optimized_value_indices: jnp.ndarray,
    carrier_changing_parameter_indices: jnp.ndarray,
    carrier_changing_values: jnp.ndarray,
    signal_changing_parameter_indices: jnp.ndarray,
    signal_changing_values: jnp.ndarray,
    # matrices
    parameters: jnp.ndarray,
    carrier_matrix: jnp.ndarray,
    signal_matrix: jnp.ndarray,
    noise_matrix: jnp.ndarray,
    noise_selection_vectors: jnp.ndarray,
    qhd_parameter_indices: jnp.ndarray,
    qhd_placing_indices: jnp.ndarray,
    linked_indices: jnp.ndarray,
    linking_function_indices: jnp.ndarray,
    indices_to_link: jnp.ndarray,
    # carrier arrays
    carrier_function_input_indices: jnp.ndarray,
    carrier_function_indices: Static,
    carrier_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    carrier_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    # signal arrays
    signal_function_input_indices: jnp.ndarray,
    signal_function_indices: Static,
    signal_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_carrier_indices: jnp.ndarray,
    signal_carrier_placing_indices: tuple[jnp.ndarray, jnp.ndarray],
    # noise arrays
    noise_function_input_indices: jnp.ndarray,
    noise_function_indices: Static,
    noise_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    noise_system_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    # sparse layouts
    signal_sparse_indices: dict[str, jnp.ndarray],
    noise_sparse_indices: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run one full simulation pass: carrier, signal, and quantum noise.

    Parameters
    ----------
    optimized_parameters : jnp.ndarray
        Optimized parameter values provided by an optimizer or sampling routine,
        shape (OPV,) or (num_ranges, OPV) in outer vectorization contexts.
    optimized_parameter_indices : jnp.ndarray
        Indices into `parameters` for parameters that are driven by optimized values,
        shape (OP,).
    optimized_value_indices : jnp.ndarray
        Indices into `optimized_parameters` mapping each optimized parameter to
        its source value, shape (OP,). This supports shared optimized values.
    carrier_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the carrier sweep, shape (CCP,).
    carrier_changing_values : jnp.ndarray
        Values for those carrier-changing parameters, shape (CCP, CV).
    signal_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the signal sweep, shape (SCP,).
    signal_changing_values : jnp.ndarray
        Values for those signal-changing parameters, shape (SCP, SV).

    parameters : jnp.ndarray
        Base parameter array, shape (1, P). It is expanded internally.
    carrier_matrix : jnp.ndarray
        Base carrier augmented matrix, shape (CN, CN + 1).
    signal_matrix : jnp.ndarray
        Base signal augmented matrix, shape (SN, SN + 1).
    noise_matrix : jnp.ndarray
        Noise source covariance matrix, shape (NN, NN) or broadcastable for vmap.
    noise_selection_vectors : jnp.ndarray
        Detector selection vectors, shape (D, 1, setup_size).
    qhd_parameter_indices : jnp.ndarray
        Indices of quadrature-homodyne-detector phase parameters in `parameters`,
        shape (Q,).
    qhd_placing_indices : jnp.ndarray
        Indices describing where detector phases are applied into
        `noise_selection_vectors`, shape (Q, 3). Each row contains:
        (detector_index, row_index, column_index).
    linked_indices : jnp.ndarray
        Parameter indices used as inputs to linking functions, shape (L,).
    linking_function_indices : jnp.ndarray
        Function selectors for linking functions, shape (L,).
    indices_to_link : jnp.ndarray
        Parameter indices to receive outputs of linking functions, shape (L,).

    carrier_function_input_indices, carrier_function_indices, carrier_output_indices,
    carrier_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the carrier matrix.

    signal_function_input_indices, signal_function_indices, signal_output_indices,
    signal_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the signal matrix.

    signal_carrier_indices : jnp.ndarray
        Indices selecting the carrier solution entries used to scale signal connectors.
    signal_carrier_placing_indices : tuple[jnp.ndarray, jnp.ndarray]
        Matrix indices locating the signal connector entries that must be scaled by carrier.

    noise_function_input_indices, noise_function_indices, noise_output_indices,
    noise_system_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the noise matrix.

    signal_sparse_indices : dict[str, jnp.ndarray]
        Output of `sparse_layout` for the signal matrix: where its possibly non-zero
        entries are and which of them each index array above refers to.
    noise_sparse_indices : dict[str, jnp.ndarray]
        Output of `sparse_layout` for the noise matrix.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, transposed to shape (CN, CV).
    signal : jnp.ndarray
        Signal solution, transposed to shape (SN, SV).
    noise : jnp.ndarray
        Quantum noise amplitude spectral density, shape (D, SV) after squeeze.

    Notes
    -----
    - Linking functions are applied before parameter expansion so they operate on a single row.
    - The signal path is always executed (even if effectively empty) to avoid conditionals
      that would force compilation of branches with empty shapes.
    - Conjugation and block manipulation follow the conventions used for upper and lower
      sidebands in frequency-domain interferometer models.
    """
    # Insert optimized parameter values into the base parameter row.
    parameters = parameters.at[[0], optimized_parameter_indices].set(
        optimized_parameters[optimized_value_indices]
    )

    # Apply parameter linking (derived parameters).
    inputs_to_link = parameters[0, linked_indices]
    outputs_to_link = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(
        linking_function_indices, inputs_to_link
    )
    parameters = parameters.at[[0], indices_to_link].set(outputs_to_link)

    # Quadrature-homodyne-detector phase factors (unit magnitude complex phase).
    qhd_phase_values = jnp.exp(1j * jnp.radians(parameters[0, qhd_parameter_indices]))

    # --------------------
    # Carrier solve
    # --------------------
    parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
    carrier_outputs = evaluate(
        parameters, carrier_function_input_indices, carrier_output_indices, carrier_function_indices
    )
    carrier_matrix = jnp.broadcast_to(
        carrier_matrix.astype(jnp.complex128), (parameters.shape[0], *carrier_matrix.shape)
    )
    carrier_matrix = carrier_matrix.at[:, carrier_matrix_indices[0], carrier_matrix_indices[1]].set(carrier_outputs)
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve
    # --------------------
    parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
    signal_value_count = parameters.shape[0]

    # The signal matrix is ~0.5% filled. Only its possibly non-zero entries are stored,
    # shape (SV, NNZ + R): first the matrix entries row by row, then the right-hand side.
    rows, columns = signal_sparse_indices["rows"], signal_sparse_indices["columns"]
    right_hand_side_rows = signal_sparse_indices["right_hand_side_rows"]
    signal_values = jnp.concatenate(
        [signal_matrix[rows, columns], signal_matrix[right_hand_side_rows, -1]]
    ).astype(jnp.complex128)
    signal_values = jnp.broadcast_to(signal_values, (signal_value_count, signal_values.shape[0]))
    signal_values = signal_values.at[:, signal_sparse_indices["update_slots"]].set(
        evaluate(parameters, signal_function_input_indices, signal_output_indices, signal_function_indices)
    )

    # Scale signal connector entries by the carrier solution at the respective spaces.
    # The minus sign matches the sign convention used by the referenced modeling tools.
    placing_slots = signal_sparse_indices["placing_slots"]
    signal_values = signal_values.at[:, placing_slots].set(
        -signal_values[:, placing_slots] * carrier[:, signal_carrier_indices.flatten()]
    )

    # Apply conjugation rules to enforce upper and lower sideband symmetry.
    signal_values = jnp.where(signal_sparse_indices["conjugated"], jnp.conjugate(signal_values), signal_values)

    right_hand_side = jnp.zeros((signal_value_count, signal_matrix.shape[0]), dtype=jnp.complex128)
    right_hand_side = right_hand_side.at[:, right_hand_side_rows].set(signal_values[:, rows.shape[0] :])
    signal_values = signal_values[:, : rows.shape[0]]
    signal = klujax.solve(rows, columns, signal_values, right_hand_side)

    # --------------------
    # Quantum noise solve
    # --------------------
    # Apply quadrature-homodyne-detector phases into the selection vectors.
    # `lax.scatter_mul` is used (rather than `.at[...].set`) so empty index arrays
    # remain valid without conditionals.
    noise_selection_vectors = lax.scatter_mul(
        noise_selection_vectors,
        qhd_placing_indices.reshape(-1, 3),
        qhd_phase_values,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2),
        ),
        unique_indices=True,
    )

    # Broadcast selection vectors across the carrier sweep dimension.
    noise_selection_vectors = jnp.tile(noise_selection_vectors, (1, carrier.shape[0], 1))

    # Select necessary carrier entries and build upper and lower sideband selections.
    upper_carrier_selections = noise_selection_vectors[:, :, : carrier.shape[1]] * carrier
    upper_carrier_selections = noise_selection_vectors.at[:, :, : carrier.shape[1]].set(upper_carrier_selections)

    lower_carrier_selections = jnp.conjugate(
        upper_carrier_selections[:, :, carrier.shape[1] : carrier.shape[1] * 2] * carrier
    )
    carrier_selections = upper_carrier_selections.at[
        :, :, carrier.shape[1] : carrier.shape[1] * 2
    ].set(lower_carrier_selections)

    # Back-propagate detector weights by solving the transposed conjugated signal system.
    # Detectors become the trailing right-hand-side axis: shape (SV, SN, D).
    carrier_selections = jnp.moveaxis(
        jnp.broadcast_to(
            carrier_selections,
            (carrier_selections.shape[0], signal_value_count, carrier_selections.shape[2]),
        ),
        0,
        -1,
    )
    transposed_order = signal_sparse_indices["transposed_order"]
    noise_weights = klujax.solve(
        columns[transposed_order],
        rows[transposed_order],
        jnp.conjugate(signal_values[:, transposed_order]),
        carrier_selections,
    )

    # Update noise source matrix (may depend on current parameters), again only its
    # possibly non-zero entries.
    noise_rows, noise_columns = noise_sparse_indices["rows"], noise_sparse_indices["columns"]
    noise_values = noise_matrix[noise_rows, noise_columns].astype(jnp.complex128)
    noise_values = jnp.broadcast_to(noise_values, (signal_value_count, noise_values.shape[0]))
    noise_values = noise_values.at[:, noise_sparse_indices["update_slots"]].set(
        evaluate(parameters, noise_function_input_indices, noise_output_indices, noise_function_indices)
    )

    # Propagating the noise sources N w through the signal matrix A and projecting on the
    # selections c = A^H w gives Re(c^H A^-1 N w) = Re(w^H N w), so no further solve is needed.
    temporary_sum = jnp.real(
        jnp.sum(
            noise_values[:, :, None] * noise_weights[:, noise_columns] * jnp.conjugate(noise_weights[:, noise_rows]),
            axis=1,
        )
    ).T
    noise = jnp.sqrt(2 * temporary_sum * UNIT_VACUUM * H_PLANCK * F0 * 0.25)

    return carrier.T, signal.T, noise.squeeze()


def sparse_layout(matrix, update_indices, placing_indices=None, carrier_size=None) -> dict[str, jnp.ndarray]:
    """
    Describe which entries of a system matrix can ever be non-zero.

    `simulate` stores only these entries ("slots"): first the square part row by row,
    then, for an augmented matrix, the entries of the right-hand-side column.

    Parameters
    ----------
    matrix : jnp.ndarray
        Base matrix, shape (N, N), or (N, N + 1) with the right-hand side as last column.
    update_indices : tuple[jnp.ndarray, jnp.ndarray]
        (rows, columns) that the component functions write into.
    placing_indices : tuple[jnp.ndarray, jnp.ndarray] | None
        (rows, columns) of the signal entries that are scaled by the carrier.
    carrier_size : int | None
        Size of the carrier system. Given for the signal matrix, whose lower-sideband
        blocks are conjugated.

    Returns
    -------
    layout : dict[str, jnp.ndarray]
        rows, columns : slots of the square part, row-major and unique, shape (NNZ,).
        transposed_order : order that makes the transposed slots row-major, shape (NNZ,).
        right_hand_side_rows : rows of the right-hand-side slots, shape (R,).
        update_slots : slot of each entry of `update_indices`.
        placing_slots : slot of each entry of `placing_indices` (if given).
        conjugated : whether each slot lies in a conjugated block, shape (NNZ + R,) (if
        `carrier_size` is given).
    """
    base = np.asarray(matrix)
    size = base.shape[0]
    index_pairs = [tuple(np.asarray(index) for index in pair) for pair in (update_indices, placing_indices) if pair is not None]
    pattern = base != 0
    for pair in index_pairs:
        pattern[pair] = True
    rows, columns = np.nonzero(pattern[:, :size])
    right_hand_side_rows = np.nonzero(pattern[:, size:].any(axis=1))[0]
    slot_rows = np.concatenate([rows, right_hand_side_rows])
    slot_columns = np.concatenate([columns, np.full(len(right_hand_side_rows), size)])
    slots = np.zeros(pattern.shape, dtype=np.int64)
    slots[slot_rows, slot_columns] = np.arange(len(slot_rows))
    layout = {
        "rows": rows,
        "columns": columns,
        "transposed_order": np.lexsort((rows, columns)),
        "right_hand_side_rows": right_hand_side_rows,
        "update_slots": slots[index_pairs[0]],
    }
    if placing_indices is not None:
        layout["placing_slots"] = slots[index_pairs[1]]
    if carrier_size is not None:
        # Same blocks as in Finesse: lower-sideband rows from the lower-sideband column on
        # (right-hand side included), and the rows below them in the upper-sideband columns.
        lower_rows = (slot_rows >= carrier_size) & (slot_rows < 2 * carrier_size)
        layout["conjugated"] = (lower_rows & (slot_columns >= carrier_size)) | (
            (slot_rows >= 2 * carrier_size) & (slot_columns < carrier_size)
        )
    return {key: jnp.asarray(value) for key, value in layout.items()}


def run_build_step(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
) -> tuple:
    """
    Build a `Setup` into simulation-ready arrays and matrices.

    Parameters
    ----------
    setup : Setup
        The optical setup to build.
    changing_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are swept during simulation. If None, no sweep is applied.
    changing_values : jnp.ndarray | None
        Values for the parameters described by `changing_pairs`.
        Expected shape is (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are treated as optimized inputs (external values injected into
        the parameter vector before solving). If None, no optimized parameters exist.
    timeit : bool
        If True, return an additional timing measurement (seconds) for the build.

    Returns
    -------
    If `timeit` is False
        simulation_arrays : dict[str, jnp.ndarray]
            Mapping from `simulate` argument name to the array value.
        metadata : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (detector_indices, mirror_indices, beamsplitter_indices, isolator_indices)

    If `timeit` is True
        build_seconds : float
            Time spent building and preparing arrays.
        (simulation_arrays, metadata) : tuple
            The same outputs as the non-timed case, packed into a tuple.

    Notes
    -----
    The returned dictionary keys are derived from the `simulate` function signature.
    """
    start = time()

    with numpy_backend():
        instructions, matrices, metadata = build(setup)
    arrays_to_prepare, carrier_arrays, signal_arrays, noise_arrays = pairs_to_arrays(
        *instructions, optimization_pairs, changing_pairs
    )
    prepared_arrays = prepare_arrays(*arrays_to_prepare, changing_values, parameters=matrices[0])

    arg_count = simulate.__code__.co_argcount
    arg_names = simulate.__code__.co_varnames[:arg_count]

    all_arrays = (*prepared_arrays, *matrices, *carrier_arrays, *signal_arrays, *noise_arrays)
    named = dict(zip(arg_names, all_arrays))
    all_arrays = tuple(
        Static(array) if name in ("carrier_function_indices", "signal_function_indices", "noise_function_indices")
        else array
        for name, array in named.items()
    )
    all_arrays += (
        sparse_layout(
            named["signal_matrix"],
            named["signal_matrix_indices"],
            named["signal_carrier_placing_indices"],
            carrier_size=named["carrier_matrix"].shape[0],
        ),
        sparse_layout(named["noise_matrix"], named["noise_system_matrix_indices"]),
    )
    if len(arg_names) != len(all_arrays):
        raise ValueError(
            "simulate signature does not match the number of prepared arrays and matrices. "
            f"simulate expects {len(arg_names)} arguments, but build produced {len(all_arrays)} arrays."
        )

    simulation_arrays = {arg_names[i]: array for i, array in enumerate(all_arrays)}

    if timeit:
        for array in simulation_arrays.values():
            jax.block_until_ready(array)
        return time() - start, (simulation_arrays, metadata)

    return simulation_arrays, *metadata


def run_simulation_step(
    simulation_arrays: dict,
    jit_simulation: bool = False,
    timeit: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Run `simulate` using a pre-built set of arrays.

    Parameters
    ----------
    simulation_arrays : dict[str, jnp.ndarray]
        Dictionary mapping argument names to arrays. This is the first return value
        of `run_build_step`.
    jit_simulation : bool
        If True, compile `simulate` using `jax.jit` and run one warmup call to
        trigger compilation before measuring runtime.
    timeit : bool
        If True, return the simulation runtime (seconds) alongside the results.

    Returns
    -------
    If `timeit` is False
        carrier : jnp.ndarray
        signal : jnp.ndarray
        noise : jnp.ndarray

    If `timeit` is True
        simulation_seconds : float
            Time spent executing the simulation (excluding warmup compilation).
        results : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (carrier, signal, noise)

    Notes
    -----
    The warmup run uses the same inputs and forces compilation when `jit_simulation`
    is enabled.
    """
    simulation_function = simulate

    if jit_simulation:
        simulation_function = jit(simulate)
        _ = simulation_function(**simulation_arrays)  # warmup compilation

    start = time()
    results = simulation_function(**simulation_arrays)

    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results

    return results


def run(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convenience wrapper: build a setup and immediately run the simulation.

    Parameters
    ----------
    setup : Setup
        The setup to simulate.
    changing_pairs : list | None
        Sweep definition as (component_name, parameter_name) pairs.
    changing_values : jnp.ndarray | None
        Sweep values, shape (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Optimized parameter definition as (component_name, parameter_name) pairs.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, shape (num_ports_carrier, num_carrier_values).
    signal : jnp.ndarray
        Signal solution, shape (num_ports_signal, num_signal_values).
    noise : jnp.ndarray
        Noise amplitude spectral density, shape (num_detectors, num_signal_values) after squeeze.
    detector_indices : jnp.ndarray
    mirror_indices : jnp.ndarray
    beamsplitter_indices : jnp.ndarray
    isolator_indices : jnp.ndarray
        Port indices in the order defined by the setup.
    """
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list[Setup], frequencies: jnp.ndarray):
    """
    Run a list of setups over a shared frequency sweep.

    Parameters
    ----------
    setups : list[Setup]
        Setups to simulate.
    frequencies : jnp.ndarray
        Frequency values. This is passed as the sweep values for the pair ("f", "frequency").
        Expected shape is (1, num_frequencies) or broadcastable to the builder expectations.

    Returns
    -------
    results : list
        Each element is the return value of `run(setup, [("f", "frequency")], frequencies)`.
    """
    results = []
    for setup in setups:
        results.append(run(setup, [("f", "frequency")], frequencies))
    return results
