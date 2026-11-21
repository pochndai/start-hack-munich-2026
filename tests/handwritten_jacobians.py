"""
Hand-derived derivatives of the 19 component functions, checked against JAX.

Run from the repository root:  python tests/handwritten_jacobians.py

`JACOBIANS[name](p)` returns the matrix dz_j/dp_i (rows: the coefficients the function
returns, columns: its inputs, in the order of the function's documentation) for one set
of real inputs `p`. The formulas are the ones derived in
docs/component-gradient-rules.pdf; this file is their executable form and their test.
The simulator itself obtains the same derivatives automatically (`differometor.host.Dual`).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from differometor.components import C_LIGHT, EPSILON_0C, F0, LAMBDA, UNIT_VACUUM, X_SCALE

RHO = np.pi / 180  # degrees to radians
K0 = 2 * np.pi / LAMBDA  # carrier wave number
LOG = 20 * np.log10(np.e)  # decibels per unit of squeezing parameter


def kappa(x):
    """cos(arcsin(x)) = sqrt(1 - x^2), continued beyond |x| = 1 as -i sign(x) sqrt(x^2 - 1)."""
    inside = 1 - x * x
    root = np.sqrt(abs(inside))
    return root if inside >= 0 else -1j * np.sign(x) * root


def jacobian(rows, inputs):
    """Matrix from a list of rows, each a dict {input index: derivative}."""
    matrix = np.zeros((len(rows), inputs), dtype=np.complex128)
    for j, row in enumerate(rows):
        for i, value in row.items():
            matrix[j, i] = value
    return matrix


def laser(p):
    power, phase = p[:2]
    z = np.sqrt(complex(2 * power / EPSILON_0C)) * np.exp(1j * RHO * phase)
    return jacobian([{0: z / (2 * power), 1: 1j * RHO * z}], 2)


def signal_function(p):
    amplitude, phase = p[:2]
    z = amplitude * np.exp(1j * RHO * phase)
    return jacobian([{0: np.exp(1j * RHO * phase), 1: 1j * RHO * z}], 2)


def constant(p):
    return jacobian([{}], 1)


def laser_frequency_modulation(p):
    f = p[0]
    return jacobian([{0: -EPSILON_0C / (2 * f * f)}], 1)


def laser_frequency_modulation_lower(p):
    f = p[0]
    return jacobian([{0: EPSILON_0C / (2 * f * f)}], 1)


def loss_quantum_noise(p):
    return jacobian([{0: 0.5}], 1)


def space(p, sign=1):
    f, length, n = p[:3]
    theta = 2 * np.pi * f * length * n / C_LIGHT
    z = -np.exp(-1j * sign * theta)
    return jacobian([{0: -1j * sign * z * theta / f, 1: -1j * sign * z * theta / length, 2: -1j * sign * z * theta / n}], 3)


def space_modulation(p, sign=1):
    f, length, n = p[:3]
    psi = np.pi * f * length * n / C_LIGHT
    z = -sign * F0 / (4 * f) * (1 - np.exp(-2j * sign * psi))
    slope = -1j * F0 / (2 * f) * np.exp(-2j * sign * psi)  # dz/dpsi
    return jacobian([{0: -z / f + slope * psi / f, 1: slope * psi / length, 2: slope * psi / n}], 3)


def surface(p):
    loss, r, tuning, f, n1, n2, alpha = p
    R, T = (1 - loss) * r, (1 - loss) * (1 - r)
    g = 2 * RHO * tuning * (1 + f / F0)
    g_tuning, g_f = 2 * RHO * (1 + f / F0), 2 * RHO * tuning / F0
    c, s = np.cos(RHO * alpha), np.sin(RHO * alpha)
    x_t, x_r = n1 * s / n2, n2 * s / n1
    k_t, k_r = kappa(x_t), kappa(x_r)
    left = np.sqrt(complex(R))
    z0 = -left * np.exp(1j * g * n1 * c)
    z1 = -np.sqrt(complex(T)) * np.exp(1j * (np.pi / 2 + 0.5 * (g * n1 * c - g * n2 * k_t)))
    z2 = -left * np.exp(-1j * g * n2 * k_r)
    # phases: Phi_L = g n1 cos, Phi_t = g n2 kappa_t, Phi_r = g n2 kappa_r; columns tuning, f, n1, n2, alpha
    d_left = {2: g_tuning * n1 * c, 3: g_f * n1 * c, 4: g * c, 5: 0.0, 6: -g * n1 * RHO * s}
    d_t = {2: g_tuning * n2 * k_t, 3: g_f * n2 * k_t, 4: -g * s * x_t / k_t, 5: g / k_t, 6: -g * n1 * RHO * c * x_t / k_t}
    d_r = {2: g_tuning * n2 * k_r, 3: g_f * n2 * k_r, 4: g * n2 * x_r**2 / (n1 * k_r), 5: g * (1 - 2 * x_r**2) / k_r, 6: -g * n2**2 * RHO * c * x_r / (n1 * k_r)}
    row0 = {0: -z0 / (2 * (1 - loss)), 1: z0 / (2 * r)}
    row1 = {0: -z1 / (2 * (1 - loss)), 1: -z1 / (2 * (1 - r))}
    row2 = {0: -z2 / (2 * (1 - loss)), 1: z2 / (2 * r)}
    for i in range(2, 7):
        row0[i] = 1j * z0 * d_left[i]
        row1[i] = 0.5j * z1 * (d_left[i] - d_t[i])
        row2[i] = -1j * z2 * d_r[i]
    return jacobian([row0, row1, row2], 7)


def susceptibility(p):
    f, mass = p[:2]
    chi = 1 / (mass * (2 * np.pi * f) ** 2)
    return jacobian([{0: -2 * chi / f, 1: -chi / mass}], 2)


def force_calculation_left(p):
    return jacobian([{0: RHO * np.sin(RHO * p[0]) / (C_LIGHT * X_SCALE)}], 1)


def force_calculation_right(p):
    alpha, n1, n2 = p[:3]
    c, s = np.cos(RHO * alpha), np.sin(RHO * alpha)
    x = n1 * s / n2
    k = kappa(x)
    scale = 1 / (C_LIGHT * X_SCALE)
    return jacobian([{0: -scale * x / k * n1 * RHO * c / n2, 1: -scale * x * x / (k * n1), 2: scale * x * x / (k * n2)}], 3)


def corrected_optomechanical_phase_left(p):
    f, tuning, r, loss, n1, alpha = p[:6]
    c, s = np.cos(RHO * alpha), np.sin(RHO * alpha)
    spread = 1 + f / F0
    theta = RHO * tuning * f / F0 + 2 * RHO * tuning * n1 * spread * c
    z = 1j * X_SCALE * K0 * c * np.sqrt(complex((1 - loss) * r)) * np.exp(1j * theta)
    return jacobian(
        [
            {
                0: 1j * z * RHO * tuning / F0 * (1 + 2 * n1 * c),
                1: 1j * z * RHO * (f / F0 + 2 * n1 * spread * c),
                2: z / (2 * r),
                3: -z / (2 * (1 - loss)),
                4: 1j * z * 2 * RHO * tuning * spread * c,
                5: -z * RHO * s / c - 1j * z * 2 * RHO * tuning * n1 * spread * RHO * s,
            }
        ],
        6,
    )


def corrected_optomechanical_phase_right(p):
    f, tuning, r, loss, n1, n2, alpha = p
    c, s = np.cos(RHO * alpha), np.sin(RHO * alpha)
    spread = 1 + f / F0
    g = 2 * RHO * tuning * spread
    x_o, x_r = n1 * s / n2, n2 * s / n1
    k_o, k_r = kappa(x_o), kappa(x_r)
    theta = RHO * tuning * f / F0 + g * n2 * k_r
    z = -1j * X_SCALE * K0 * k_o * np.sqrt(complex((1 - loss) * r)) * np.exp(-1j * theta)
    return jacobian(
        [
            {
                0: -1j * z * RHO * tuning / F0 * (1 + 2 * n2 * k_r),
                1: -1j * z * RHO * (f / F0 + 2 * n2 * spread * k_r),
                2: z / (2 * r),
                3: -z / (2 * (1 - loss)),
                4: -z * x_o**2 / (k_o**2 * n1) - 1j * z * g * n2 * x_r**2 / (k_r * n1),
                5: z * x_o**2 / (k_o**2 * n2) - 1j * z * g * (1 - 2 * x_r**2) / k_r,
                6: -z * x_o / k_o**2 * n1 * RHO * c / n2 + 1j * z * g * n2 * x_r / k_r * n2 * RHO * c / n1,
            }
        ],
        7,
    )


def squeezer(p):
    level, angle = p[:2]
    unit = UNIT_VACUUM / 2
    r = level / LOG
    phase = np.exp(2j * RHO * angle)
    z1 = unit * np.sinh(2 * r) * phase
    d_level_diagonal = 2 * unit * np.sinh(2 * r) / LOG
    d_level_off = 2 * unit * np.cosh(2 * r) * phase / LOG
    return jacobian(
        [
            {0: d_level_diagonal},
            {0: d_level_off, 1: 2j * RHO * z1},
            {0: d_level_diagonal},
            {0: np.conj(d_level_off), 1: -2j * RHO * np.conj(z1)},
        ],
        2,
    )


JACOBIANS = {
    "force_calculation_right": force_calculation_right,
    "laser": laser,
    "surface": surface,
    "space": space,
    "space_modulation": space_modulation,
    "signal_function": signal_function,
    "dummy_function": constant,
    "vacuum_quantum_noise": constant,
    "loss_quantum_noise": loss_quantum_noise,
    "space_lower": lambda p: space(p, sign=-1),
    "laser_amplitude_modulation": constant,
    "laser_frequency_modulation": laser_frequency_modulation,
    "susceptibility": susceptibility,
    "force_calculation_left": force_calculation_left,
    "corrected_optomechanical_phase_left": corrected_optomechanical_phase_left,
    "corrected_optomechanical_phase_right": corrected_optomechanical_phase_right,
    "space_modulation_lower": lambda p: space_modulation(p, sign=-1),
    "squeezer": squeezer,
    "laser_frequency_modulation_lower": laser_frequency_modulation_lower,
}


def main():
    import jax
    import jax.numpy as jnp

    from differometor import components

    assert [f.__name__ for f in components.FUNCTIONS] == list(JACOBIANS), "one rule per component function, same order"
    rng = np.random.default_rng(0)
    # physical values; the last two sets are beyond total internal reflection for one or both angles
    cases = {
        "typical": lambda: dict(n1=1.0, n2=1.45, alpha=rng.uniform(5, 60)),
        "equal indices, 45 degrees": lambda: dict(n1=1.0, n2=1.0, alpha=45.0),
        "beyond total internal reflection": lambda: dict(n1=1.0, n2=1.45, alpha=rng.uniform(50, 80)),
        "reflectivity above 1": lambda: dict(n1=1.2, n2=1.1, alpha=rng.uniform(5, 40), r=1.0004),
    }
    worst_overall, failed = 0.0, False
    print(f"{'function':38} {'inputs':>6} {'outputs':>7}   worst relative difference from JAX, per case")
    for function in components.FUNCTIONS:
        name = function.__name__
        worst = []
        for label, make in cases.items():
            c = make()
            f, length, mass = rng.uniform(50, 3000), rng.uniform(1, 4000), rng.uniform(0.1, 200)
            loss, r, tuning = rng.uniform(1e-6, 0.2), c.get("r", rng.uniform(0.05, 0.95)), rng.uniform(-170, 170)
            values = {
                "laser": [rng.uniform(1, 200), rng.uniform(0, 180)],
                "signal_function": [rng.uniform(0.5, 2), rng.uniform(0, 180)],
                "laser_frequency_modulation": [f],
                "laser_frequency_modulation_lower": [f],
                "loss_quantum_noise": [loss],
                "space": [f, length, c["n1"]],
                "space_lower": [f, length, c["n1"]],
                "space_modulation": [f, length, c["n1"]],
                "space_modulation_lower": [f, length, c["n1"]],
                "surface": [loss, r, tuning, f, c["n1"], c["n2"], c["alpha"]],
                "susceptibility": [f, mass],
                "force_calculation_left": [c["alpha"]],
                "force_calculation_right": [c["alpha"], c["n1"], c["n2"]],
                "corrected_optomechanical_phase_left": [f, tuning, r, loss, c["n1"], c["alpha"]],
                "corrected_optomechanical_phase_right": [f, tuning, r, loss, c["n1"], c["n2"], c["alpha"]],
                "squeezer": [rng.uniform(0, 15), rng.uniform(0, 360)],
            }.get(name, [1.0])
            p = np.zeros(7)
            p[: len(values)] = values
            hand = JACOBIANS[name](p)

            def stacked(x):
                with components.raw_outputs():
                    out = function(x)
                return jnp.stack([jnp.asarray(o, dtype=jnp.complex128) for o in out])

            reference = np.asarray(jax.jacfwd(stacked)(jnp.asarray(p)))[: hand.shape[0], : hand.shape[1]]
            rest = np.asarray(jax.jacfwd(stacked)(jnp.asarray(p)))[:, hand.shape[1] :]
            scale = max(np.abs(reference).max(), 1e-300)
            difference = max(np.abs(hand - reference).max() / scale, np.abs(rest).max(initial=0) / scale)
            worst.append(difference)
        worst_overall = max(worst_overall, max(worst))
        failed |= max(worst) > 1e-9
        print(f"{name:38} {hand.shape[1]:6d} {hand.shape[0]:7d}   " + "  ".join(f"{w:.1e}" for w in worst) + ("   FAIL" if max(worst) > 1e-9 else ""))
    print("cases:", ", ".join(cases))
    print("ALL RULES AGREE WITH JAX" if not failed else "SOME RULES DISAGREE", f"(worst {worst_overall:.1e})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
