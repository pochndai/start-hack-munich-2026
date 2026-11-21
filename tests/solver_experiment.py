#!/usr/bin/env python3
"""Could SciPy's sparse solver, with one factorisation per frequency, replace KLU?

    python tests/solver_experiment.py supplied
    python tests/solver_experiment.py homodyne-readout --exact --frequencies 5
    python tests/solver_experiment.py voyager --exact --frequencies 5 --json voyager.json

Per frequency the simulation solves four systems with one matrix A: A x = b,
A^H w = c, A^T lambda = x_bar and conj(A) mu = w_bar. Today each is a separate
klujax.solve, which factorises A every time. This script takes the assembled
systems of one scenario from `differometor.host._forward`, solves them with KLU
as today and with several ways of using SciPy's SuperLU (`splu`), factorising
once per frequency, and prints for each:

- how far the results are from a reference: the signal at the detector, the
  noise, ln(sensitivity) and the four solution vectors. The reference is the
  KLU result, or with --exact the correctly rounded solution (`refined` from
  tests/refined_solves.py, slow: use --frequencies to keep the run short);
- the time to factorise and to solve, and what the four solves would cost with
  four factorisations per frequency (today), two (one per stage) and one.

It changes nothing in the simulator and never runs the original code. A
scenario is a file in tests/variations/ (without .json) or `voyager`. Not for
scenarios with linking functions (amplitude modulation).
"""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import klujax
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import differometor as df
from differometor import host
from differometor.build import LINKING_FUNCTIONS
from differometor.setups import constrain_inter_grid_cell_spaces, voyager

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
# name: (options for splu, scale rows and columns first, one refinement step)
VARIANTS = {
    "SuperLU": ({}, False, False),
    "SuperLU, MMD_AT_PLUS_A ordering": ({"permc_spec": "MMD_AT_PLUS_A"}, False, False),
    "SuperLU, MMD_ATA ordering": ({"permc_spec": "MMD_ATA"}, False, False),
    "SuperLU, natural ordering": ({"permc_spec": "NATURAL"}, False, False),
    "SuperLU, Equil option": ({"options": {"Equil": True}}, False, False),
    "SuperLU, scaled": ({}, True, False),
    "SuperLU, refined": ({}, False, True),
    "SuperLU, scaled and refined": ({}, True, True),
}


def load(name, count):
    """Build a scenario; return its simulation arrays, detector ports, design values and frequencies."""
    if name == "voyager":
        setup, frequencies = voyager()[0], np.logspace(1, 3.5, 60)
        pairs = [[component, prop] for component, prop in setup.parameters if prop in PROPERTIES]
    else:
        case = json.loads((ROOT / "tests" / "variations" / f"{name}.json").read_text())
        if "setup" not in case:
            raise SystemExit(f"{name} has linking functions and runs on the JAX path; there is no host system to take.")
        setup, frequencies = df.Setup.from_data(case["setup"]), np.asarray(case["frequencies"], dtype=np.float64)
        pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    if count and count < len(frequencies):
        frequencies = frequencies[np.unique(np.round(np.linspace(0, len(frequencies) - 1, count)).astype(int))]
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(frequencies), pairs)
    if not host.supported(LINKING_FUNCTIONS):
        raise SystemExit(f"{name} has linking functions and runs on the JAX path; there is no host system to take.")
    values = np.zeros(int(np.max(arrays["optimized_value_indices"])) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"])[
        0, np.asarray(arrays["optimized_parameter_indices"])
    ].real
    return arrays, np.asarray(ports), values, frequencies


def expand(vector, like):
    """`vector` with trailing axes so that it scales the rows of `like`."""
    return vector.reshape(-1, *([1] * (like.ndim - 1)))


class Factor:
    """One SuperLU factorisation of A, serving A, A^T and A^H."""

    def __init__(self, data, indices, pointers, size, columns, options, scale, refine):
        self.refine, self.scale = refine, scale
        self.matrix = csc_matrix((data, indices, pointers), shape=(size, size))
        if scale:
            # Rows, then columns, to a largest entry of one. A = R^-1 (R A C) C^-1.
            magnitude = np.abs(data)
            self.rows = np.zeros(size)
            np.maximum.at(self.rows, indices, magnitude)
            self.rows = 1.0 / self.rows
            self.columns = 1.0 / np.maximum.reduceat(magnitude * self.rows[indices], pointers[:-1])
            data = data * self.rows[indices] * self.columns[columns]
        self.factors = splu(csc_matrix((data, indices, pointers), shape=(size, size)), **options)
        self.operators = {}

    def once(self, right_hand_side, transpose):
        if not self.scale:
            return self.factors.solve(right_hand_side, transpose)
        before, after = (self.rows, self.columns) if transpose == "N" else (self.columns, self.rows)
        return expand(after, right_hand_side) * self.factors.solve(expand(before, right_hand_side) * right_hand_side, transpose)

    def solve(self, right_hand_side, transpose="N"):
        solution = self.once(right_hand_side, transpose)
        if self.refine:
            if transpose not in self.operators:
                self.operators[transpose] = {"N": lambda m: m, "T": lambda m: m.T.tocsr(), "H": lambda m: m.conj().T.tocsr()}[transpose](self.matrix)
            solution = solution + self.once(right_hand_side - self.operators[transpose] @ solution, transpose)
        return solution


def superlu(system, options, scale, refine, repeats):
    """Solve the four systems of every frequency with one factorisation each; return solutions and best times."""
    order, size = system["order"], system["size"]
    indices = system["rows"][order].astype(np.int32)
    columns = system["columns"][order]
    pointers = np.concatenate([[0], np.cumsum(np.bincount(system["columns"], minlength=size))]).astype(np.int32)
    best = [np.inf, np.inf]
    for _ in range(repeats):
        start = perf_counter()
        factors = [Factor(entries[order], indices, pointers, size, columns, options, scale, refine) for entries in system["values"]]
        factored = perf_counter()
        stack = lambda right_hand_sides, transpose: np.stack([f.solve(b, transpose) for f, b in zip(factors, right_hand_sides)])
        solutions = {
            "x": stack(system["b"], "N"),
            "w": stack(system["c"], "H"),
            "lambda": stack(system["x_bar"], "T"),
            "mu": np.conj(stack(np.conj(system["w_bar"]), "N")),
        }
        best = [min(best[0], factored - start), min(best[1], perf_counter() - factored)]
    return solutions, best


def four_solves(solve, system):
    """The four systems written as host.py writes them for klujax.solve (also the form `refined` takes)."""
    rows, columns, order, values = system["rows"], system["columns"], system["order"], system["values"]
    return {
        "x": lambda: solve(rows, columns, values, system["b"]),
        "w": lambda: solve(columns[order], rows[order], np.conj(values[:, order]), system["c"]),
        "lambda": lambda: solve(columns[order], rows[order], values[:, order], system["x_bar"]),
        "mu": lambda: np.conj(solve(rows, columns, values, np.conj(system["w_bar"]))),
    }


def klu_solve(*arguments):
    return np.asarray(klujax.solve(*(jnp.asarray(argument) for argument in arguments)))


def klu(system, repeats):
    """Today's solves: klujax.solve, which factorises on every call. Returns solutions and the best total time."""
    calls = four_solves(klu_solve, system)
    solutions, best = {key: call() for key, call in calls.items()}, np.inf  # the first calls also warm klujax up
    for _ in range(repeats):
        start = perf_counter()
        for call in calls.values():
            call()
        best = min(best, perf_counter() - start)
    return solutions, best


def readout(solutions, system):
    """Signal at the detector, noise and sensitivity, as benchmark.py's `readout` computes them."""
    carrier, ports, size = system["carrier"][0], system["ports"], system["carrier"].shape[1]
    signals = np.conj(carrier[ports]) * solutions["x"][:, ports] + carrier[ports] * solutions["x"][:, size + ports]
    detected = signals[:, 0] if len(ports) == 1 else signals[:, 0] - signals[:, 1]
    weights = solutions["w"]
    form = np.real(np.sum(system["noise_values"][:, :, None] * weights[:, system["noise_columns"]] * np.conj(weights[:, system["noise_rows"]]), axis=1))
    noise = np.sqrt(2 * form * host.NOISE_SCALE)
    return detected, noise, noise[:, 0] / np.abs(detected)


def distances(solutions, reference, system):
    """Largest differences from the reference, each relative to the reference's largest value at that frequency."""
    per_frequency = lambda a, b: float(np.max(np.abs(a - b).reshape(len(a), -1).max(axis=1) / np.abs(b).reshape(len(b), -1).max(axis=1)))
    ours, theirs = readout(solutions, system), readout(reference, system)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = {
            "signal": float(np.max(np.abs(ours[0] - theirs[0]) / np.abs(theirs[0]))),
            "noise": float(np.max(np.abs(ours[1] - theirs[1]) / np.abs(theirs[1]))),
            "ln_sensitivity": float(np.max(np.abs(np.log(ours[2]) - np.log(theirs[2])))),
        }
    result.update({key: per_frequency(solutions[key], reference[key]) for key in ("x", "w", "lambda", "mu")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", help="a file in tests/variations/ without .json, or voyager")
    parser.add_argument("--frequencies", type=int, default=0, help="use only this many frequencies, spread over the band (default: all)")
    parser.add_argument("--exact", action="store_true", help="compare with the correctly rounded solution instead of the KLU result (slow)")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats; the best one is reported")
    parser.add_argument("--json", type=Path, help="also write the table to this file")
    args = parser.parse_args()

    arrays, ports, values, frequencies = load(args.scenario, args.frequencies)
    static = {name: arrays[name].array for name in host.STATIC_NAMES}
    a = host._numpy({key: value for key, value in arrays.items() if key not in (*host.STATIC_NAMES, "linking_function_indices", "optimized_parameters")})
    carrier, signal_values, right_hand_side, selections, noise_values = host._forward(values, a, static)
    layout, noise_layout = a["signal_sparse_indices"], a["noise_sparse_indices"]
    system = {
        "rows": layout["rows"], "columns": layout["columns"], "order": layout["transposed_order"], "size": a["signal_matrix"].shape[0],
        "values": signal_values, "b": right_hand_side, "c": selections, "carrier": carrier, "ports": ports,
        "noise_values": noise_values, "noise_rows": noise_layout["rows"], "noise_columns": noise_layout["columns"],
    }
    # Right-hand sides of the two gradient solves, as they arise from the readout: the signal cotangent sits on the
    # detector entries, and the weight cotangent is that of the noise quadratic form (host._simulate_backward).
    system["x_bar"], system["w_bar"] = np.conj(selections[:, :, 0]), np.zeros_like(selections)
    weights = klu_solve(system["columns"][system["order"]], system["rows"][system["order"]], np.conj(signal_values[:, system["order"]]), selections)
    np.add.at(system["w_bar"], (slice(None), system["noise_columns"]), np.conj(weights[:, system["noise_rows"]]) * noise_values[:, :, None])
    np.add.at(system["w_bar"], (slice(None), system["noise_rows"]), np.conj(noise_values[:, :, None] * weights[:, system["noise_columns"]]))
    today, klu_seconds = klu(system, args.repeats)

    magnitude = np.abs(signal_values[signal_values != 0])
    print(f"{args.scenario}: {len(frequencies)} frequencies from {frequencies[0]:g} to {frequencies[-1]:g} Hz, matrix {system['size']}x{system['size']} "
          f"with {signal_values.shape[1]} entries from {magnitude.min():.0e} to {magnitude.max():.0e}, {selections.shape[2]} noise detector(s), "
          f"{len(ports)} signal port(s)")
    print(f"Machine: {platform.platform()}, Python {platform.python_version()}, jax {jax.__version__}. Times are the best of {args.repeats}.")

    reference, reference_name = today, "the KLU result"
    if args.exact:
        from refined_solves import refined  # 100-digit residuals; slow

        start = perf_counter()
        reference = {key: call() for key, call in four_solves(refined, system).items()}
        reference_name = f"the exact solution (computed in {perf_counter() - start:.1f} s)"

    table = []
    if args.exact:
        table.append({"variant": "KLU, as today", **distances(today, reference, system), "factor_ms": None, "solve_ms": None, "today_ms": klu_seconds * 1e3})
    for name, (options, scale, refine) in VARIANTS.items():
        try:
            solutions, (factor_seconds, solve_seconds) = superlu(system, options, scale, refine, args.repeats)
        except Exception as error:  # a variant that SciPy rejects is a result, not a reason to stop
            print(f"{name}: failed with {type(error).__name__}: {error}")
            continue
        table.append({"variant": name, **distances(solutions, reference, system), "factor_ms": factor_seconds * 1e3, "solve_ms": solve_seconds * 1e3})

    print(f"\nDistance from {reference_name} (largest over the frequencies, relative; ln(sensitivity) absolute)")
    print(f"{'variant':34}{'signal':>9}{'noise':>9}{'ln(sens)':>10}{'x':>9}{'w':>9}{'lambda':>9}{'mu':>9}")
    for row in table:
        print(f"{row['variant']:34}{row['signal']:9.1e}{row['noise']:9.1e}{row['ln_sensitivity']:10.1e}{row['x']:9.1e}{row['w']:9.1e}{row['lambda']:9.1e}{row['mu']:9.1e}")

    print(f"\nTime for all {len(frequencies)} frequencies, in ms. KLU today (four klujax.solve calls, four factorisations): {klu_seconds * 1e3:.1f}")
    print(f"{'variant':34}{'factorise':>11}{'4 solves':>10}{'4 factor.':>11}{'2 factor.':>11}{'1 factor.':>11}")
    for row in table:
        if row["factor_ms"] is not None:
            factor, solve = row["factor_ms"], row["solve_ms"]
            print(f"{row['variant']:34}{factor:11.1f}{solve:10.1f}{4 * factor + solve:11.1f}{2 * factor + solve:11.1f}{factor + solve:11.1f}")

    if args.json:
        args.json.write_text(json.dumps({
            "scenario": args.scenario, "frequencies": len(frequencies), "size": system["size"], "entries": int(signal_values.shape[1]),
            "reference": "exact" if args.exact else "klu", "klu_today_ms": klu_seconds * 1e3,
            "machine": platform.platform(), "python": platform.python_version(), "jax": jax.__version__, "rows": table,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
