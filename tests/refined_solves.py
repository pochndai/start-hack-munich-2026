#!/usr/bin/env python3
"""When ours and the original disagree on a scenario: which one is right?

    python tests/refined_solves.py supplied
    python tests/refined_solves.py band-20-5000

Our calculation is repeated with every sparse solve A x = b done by iterative
refinement: solve in double precision, compute the residual b - A x exactly
(decimal arithmetic, 100 digits), solve for the correction, repeat until the
correction is below 1e-20 of the solution. That gives the correctly rounded solution of each system,
so the result is free of solver rounding. The script then prints how far our
normal result (KLU) and the original's stored result (dense solver,
bench/reference/variation-<scenario>.npz) are from it, and lists the gradient
entries on which ours and the original differ by more than 0.01%.

The last column repeats the refined calculation with every design variable
moved by one unit in its last digit. It shows how much of a difference is
simply the limit of double precision for this setup.

Takes about a minute per scenario. Not for scenarios with linking functions
(amplitude modulation): those do not use the NumPy path that is patched here.
"""
import decimal
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import differometor as df
from differometor import host
from differometor.setups import constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
decimal.getcontext().prec = 100
D = decimal.Decimal
STATS = {"iterations": [], "first_correction": []}


def refined(Ai, Aj, Ax, b):
    """klujax.solve with exact residuals. Shapes as in klujax: Ax [n_lhs, n_nz], b [n_lhs, n_col(, n_rhs)]."""
    Ai, Aj, Ax, b = np.asarray(Ai), np.asarray(Aj), np.asarray(Ax), np.asarray(b)
    single = b.ndim == 2
    b = b[..., None] if single else b
    out, n = np.empty(b.shape, dtype=np.complex128), b.shape[1]
    for k in range(Ax.shape[0]):
        factors = spla.splu(sp.coo_matrix((Ax[k], (Ai, Aj)), shape=(n, n)).tocsc())
        entries = list(zip(Ai.tolist(), Aj.tolist(), [D(float(v)) for v in Ax[k].real], [D(float(v)) for v in Ax[k].imag]))
        for c in range(b.shape[2]):
            rhs = b[k, :, c]
            start = factors.solve(rhs)
            xr, xi = [D(float(v)) for v in start.real], [D(float(v)) for v in start.imag]
            scale = max(float(np.abs(start).max()), 1e-300)
            for iteration in range(1, 13):
                rr, ri = [D(float(v)) for v in rhs.real], [D(float(v)) for v in rhs.imag]
                for i, j, vr, vi in entries:
                    rr[i] -= vr * xr[j] - vi * xi[j]
                    ri[i] -= vr * xi[j] + vi * xr[j]
                correction = factors.solve(np.array([complex(float(u), float(v)) for u, v in zip(rr, ri)]))
                for j in range(n):
                    xr[j] += D(float(correction.real[j]))
                    xi[j] += D(float(correction.imag[j]))
                size = float(np.abs(correction).max()) / scale
                if iteration == 1:
                    STATS["first_correction"].append(size)
                if size <= 1e-20:  # ten thousand times below the last digit of a double
                    break
            else:
                raise SystemExit(f"refinement did not converge (last correction {size:.1e}); the result would not be trustworthy")
            STATS["iterations"].append(iteration)
            out[k, :, c] = [complex(float(u), float(v)) for u, v in zip(xr, xi)]
    return out[..., 0] if single else out


class RefinedSolver:
    @staticmethod
    def solve(Ai, Aj, Ax, b):
        return jax.pure_callback(refined, jax.ShapeDtypeStruct(b.shape, jnp.complex128), Ai, Aj, Ax, b)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    case = json.loads((ROOT / "tests" / "variations" / f"{name}.json").read_text())
    if "uifo" in case:
        raise SystemExit("This scenario has linking functions and runs on the JAX path, which this script does not patch.")
    reference = np.load(ROOT / "bench" / "reference" / f"variation-{name}.npz")
    setup = df.Setup.from_data(case["setup"])
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    values = {key: item["properties"] for kind in ("nodes", "edges") for key, item in case["setup"][kind].items()}
    first = lambda pair: pair if isinstance(pair[0], str) else pair[0]
    parameters = jnp.asarray([values[first(pair)[0]][first(pair)[1]] for pair in pairs], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(case["frequencies"], dtype=jnp.float64), pairs)
    ports = jnp.asarray(ports)

    def loss(p):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=p))
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity)), (signals, noise, sensitivity)

    run = lambda p: jax.value_and_grad(loss, has_aux=True)(p)
    (_, ours), ours_gradient = run(parameters)
    normal_solver, host.klujax = host.klujax, RefinedSolver
    try:
        (_, exact), exact_gradient = run(parameters)
        last_digit = np.random.default_rng(0).choice([-1.0, 0.0, 1.0], size=parameters.shape)
        (_, moved), moved_gradient = run(parameters * (1 + 2e-16 * last_digit))
    finally:
        host.klujax = normal_solver

    print(f"{name}: {len(STATS['iterations'])} solves refined, at most {max(STATS['iterations'])} rounds each. "
          f"The first correction, which is the error of a plain double-precision solve, was {np.median(STATS['first_correction']):.0e} of the solution "
          f"in the median and {max(STATS['first_correction']):.0e} at worst.\n")
    relative = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())
    in_log = lambda a, b: float(np.abs(np.log(np.asarray(a)) - np.log(np.asarray(b))).max())
    gk, ge, go, gm = (np.asarray(g) for g in (ours_gradient, exact_gradient, reference["gradient"], moved_gradient))
    allowed = 1e-12 * np.abs(ge).max() + 1e-4 * np.abs(ge)  # the gradient tolerance of bench/validate.py
    outside = lambda g: int((np.abs(g - ge) > allowed).sum())
    print(f"{'distance from the refined result':36}{'ours (KLU)':>12}{'original (dense)':>18}{'refined, inputs moved':>24}")
    print(f"{'readout signal':36}{relative(ours[0], exact[0]):12.1e}{relative(reference['signal'], exact[0]):18.1e}{relative(moved[0], exact[0]):24.1e}")
    print(f"{'noise':36}{relative(ours[1], exact[1]):12.1e}{relative(reference['noise'], exact[1]):18.1e}{relative(moved[1], exact[1]):24.1e}")
    print(f"{'ln(sensitivity)':36}{in_log(ours[2], exact[2]):12.1e}{in_log(reference['sensitivity'], exact[2]):18.1e}{in_log(moved[2], exact[2]):24.1e}")
    print(f"{'gradient entries off by over 0.01%':36}{outside(gk):12d}{outside(go):18d}{outside(gm):24d}   (of {ge.size})")
    disputed = np.where(np.abs(gk - go) > 1e-12 * np.abs(go).max() + 1e-4 * np.abs(go))[0]
    print(f"\nGradient entries on which ours and the original differ by more than 0.01%: {len(disputed)}")
    off = lambda g, i: f"{g[i]: .9e} (off by {abs(g[i] - ge[i]) / abs(ge[i]):.0e})"
    for i in disputed:
        print(f"  entry {i} {'.'.join(first(pairs[i]))}: refined {ge[i]: .9e} | ours {off(gk, i)} | original {off(go, i)} | refined with inputs moved {off(gm, i)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
