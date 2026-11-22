"""
Checks of the NumPy host simulation (`differometor.host`) against the JAX simulation.

Run from the repository root:  python tests/test_host.py

Every case compares all three outputs and the reverse-mode gradients for random
cotangents on each output. The cases cover: the two saved UIFO setups (one with a
homodyne detector), aLIGO, Voyager (balanced homodyne detection), a cavity with a
carrier sweep, randomly generated UIFO layouts, shifted parameters (reflectivities
above 1), `jit`, `vmap`, the fallback for linking functions, finite differences for
one design variable of every kind, and two situations no public setup has: a second
noise detector, and matrix cells that are written twice.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib

import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import components, host
from differometor.setups import aligo, constrain_inter_grid_cell_spaces, uifo, voyager

simulate_module = importlib.import_module("differometor.simulate")
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
FAILURES = []


def check(name, value, limit):
    ok = np.isfinite(value) and value <= limit
    if not ok:
        FAILURES.append(f"{name}: {value:.2e} > {limit:.0e}")
    return f"{value:.1e}" + ("" if ok else " FAIL")


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def prepare(setup, changing_pairs, values, pairs=None):
    """Build a setup with all design variables of the usual kinds; return arrays, ports, start values."""
    if pairs is None:
        changing = {tuple(pair) for pair in changing_pairs}
        pairs = [[c, p] for c, p in setup.parameters if p in PROPERTIES and (c, p) not in changing]
    arrays, ports, *_ = df.run_build_step(setup, changing_pairs, values, pairs)
    start = np.zeros(int(np.max(arrays["optimized_value_indices"])) + 1)
    start[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"])[
        0, np.asarray(arrays["optimized_parameter_indices"])
    ].real
    return arrays, jnp.asarray(ports), jnp.asarray(start), pairs


def compare(name, arrays, start, shift_seed=None):
    """Host against JAX: outputs, and gradients for random cotangents on each output."""
    if shift_seed is not None:
        start = start * (1 + 1e-3 * jnp.asarray(np.random.default_rng(shift_seed).standard_normal(start.shape)))
    host_f = lambda p: simulate_module.simulate(**dict(arrays, optimized_parameters=p))
    jax_f = lambda p: simulate_module.simulate_jax(**dict(arrays, optimized_parameters=p))
    assert host.supported(simulate_module.LINKING_FUNCTIONS)
    host_out, host_pull = jax.vjp(host_f, start)
    jax_out, jax_pull = jax.vjp(jax_f, start)
    rng = np.random.default_rng(0)
    worst = 0.0
    for which in ({0}, {1}, {2}, {0, 1, 2}):
        cotangents = []
        for index, output in enumerate(jax_out):
            if index not in which or output.size == 0:
                cotangents.append(jnp.zeros_like(output))
                continue
            noise = rng.standard_normal(output.shape)
            if jnp.iscomplexobj(output):
                noise = noise + 1j * rng.standard_normal(output.shape)
            cotangents.append(jnp.asarray(noise) / max(float(jnp.abs(output).max()), 1e-300))
        worst = max(worst, relative(host_pull(tuple(cotangents))[0], jax_pull(tuple(cotangents))[0]))
    sizes = f"{start.size} variables, carrier {tuple(jax_out[0].shape)}, signal {tuple(jax_out[1].shape)}, noise {tuple(np.shape(jax_out[2]))}"
    print(
        f"{name:44} carrier {check(name + ' carrier', relative(host_out[0], jax_out[0]), 1e-9):8} "
        f"signal {check(name + ' signal', relative(host_out[1], jax_out[1]) if jax_out[1].size else 0.0, 1e-3):8} "
        f"noise {check(name + ' noise', relative(host_out[2], jax_out[2]) if np.size(jax_out[2]) else 0.0, 1e-6):8} "
        f"gradients {check(name + ' gradients', worst, 1e-5):8} | {sizes}"
    )
    return host_f, jax_f, start


def directional(name, f, start):
    """Gradient of a random scalar of the outputs against central differences in random directions."""
    rng = np.random.default_rng(7)
    weights = []
    for output in f(start):
        weight = rng.standard_normal(np.shape(output))
        if jnp.iscomplexobj(output):
            weight = weight + 1j * rng.standard_normal(np.shape(output))
        weights.append(jnp.asarray(weight) / max(float(jnp.abs(output).max()), 1e-300) if np.size(output) else jnp.asarray(weight))
    scalar = jax.jit(lambda p: sum(jnp.sum(jnp.real(o * w)) for o, w in zip(f(p), weights)))
    gradient = np.asarray(jax.grad(scalar)(start))
    worst, judged = 0.0, 0
    for _ in range(4):
        direction = jnp.asarray(rng.standard_normal(start.shape)) * jnp.maximum(jnp.abs(start), 1e-6)
        slope = lambda h: float(scalar(start + h * direction) - scalar(start - h * direction)) / (2 * h)
        for exponent in range(-3, -10, -1):  # shrink the step until two sizes agree within 1%
            coarse, fine = slope(10.0**exponent), slope(10.0**exponent / 2)
            if fine != 0 and abs(coarse - fine) <= 0.01 * abs(fine):
                worst, judged = max(worst, abs(float(gradient @ np.asarray(direction)) - fine) / abs(fine)), judged + 1
                break
    if not judged:
        FAILURES.append(f"{name}: finite differences never settled")
    return f"{check(name, worst, 0.02)} ({judged} of 4 directions settled)"


def written_twice(arrays):
    """Make one cell of the carrier, signal and noise systems be written a second time, by another formula."""
    arrays = dict(arrays)
    twice = lambda indices: np.concatenate([np.asarray(indices), np.asarray(indices)[..., :1]], axis=-1)
    again = lambda static: simulate_module.Static(np.concatenate([static.array, static.array[:, -1:]], axis=1))
    arrays["carrier_matrix_indices"] = jnp.asarray(twice(arrays["carrier_matrix_indices"]))
    for system in ("signal", "noise"):
        layout = dict(arrays[f"{system}_sparse_indices"])
        layout["update_slots"] = jnp.asarray(twice(layout["update_slots"]))
        arrays[f"{system}_sparse_indices"] = layout
    for system in ("carrier", "signal", "noise"):
        # the extra write goes to the cell of the first write and carries the value of the last one
        arrays[f"{system}_output_indices"] = again(arrays[f"{system}_output_indices"])
    return arrays


def saved_setup(file, low, high):
    fixture = json.loads((ROOT / "examples" / "data" / file).read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    return prepare(setup, [("f", "frequency")], jnp.logspace(np.log10(low), np.log10(high), 101), pairs)


def main():
    print("1. Every component function: NumPy value and Dual derivative against JAX")
    rng = np.random.default_rng(1)
    worst = 0.0
    for low, high in ((0.1, 0.9), (0.2, 1.6), (-1.5, 2.5)):  # in range, partly out of range, wild
        inputs = rng.uniform(low, high, size=(7, 3, 5))
        for function, numpy_function in zip(components.FUNCTIONS, host.NUMPY_FUNCTIONS):

            def stacked(x):
                with components.raw_outputs():
                    out = function(x)
                return jnp.stack([jnp.broadcast_to(jnp.asarray(o, dtype=jnp.complex128), x.shape[1:]) for o in out])

            reference = np.asarray(stacked(jnp.asarray(inputs)))
            one_point = jax.jacfwd(lambda x: stacked(x[:, None, None])[:, 0, 0])
            jacobian = np.asarray(jax.vmap(jax.vmap(one_point, in_axes=1, out_axes=1), in_axes=1, out_axes=1)(jnp.asarray(inputs)))
            with np.errstate(all="ignore"):
                dual = numpy_function(host.DualInputs(inputs, np.arange(7)))
            for index, entry in enumerate(dual):
                value = np.broadcast_to(host._value(entry), (3, 5))
                tangent = np.broadcast_to(host._tangent(entry), (7, 3, 5))
                expected = np.moveaxis(jacobian[index], -1, 0)
                finite = np.isfinite(reference[index])
                if not np.array_equal(finite, np.isfinite(value)):
                    FAILURES.append(f"{function.__name__}[{index}]: different NaN pattern")
                scale = max(1.0, np.abs(reference[index][finite]).max(initial=0))
                worst = max(worst, np.abs(value - reference[index])[finite].max(initial=0) / scale)
                finite = np.isfinite(expected)
                worst = max(worst, (np.abs(tangent - expected)[finite] / np.maximum(1, np.abs(expected)[finite])).max(initial=0))
    print(f"   {len(components.FUNCTIONS)} functions, worst relative error {check('component functions', worst, 1e-8)}")

    print("2. Host simulation against JAX simulation (largest difference relative to largest value)")
    arrays, ports, start, pairs = saved_setup("uifo_800_3000.json", 800, 3000)
    main_case = compare("main example (UIFO 800-3000)", arrays, start)
    compare("  same, parameters shifted by 0.1%", arrays, start, shift_seed=1)
    arrays2, _, start2, _ = saved_setup("uifo_20_5000.json", 20, 5000)
    compare("second UIFO (20-5000, homodyne detector)", arrays2, start2)
    compare("  same, parameters shifted by 0.1%", arrays2, start2, shift_seed=2)
    frequencies = jnp.logspace(1, 3.5, 60)
    for label, maker in (("aLIGO", aligo), ("Voyager (balanced homodyne)", voyager)):
        case = prepare(maker()[0], [("f", "frequency")], frequencies)
        compare(label, case[0], case[2])
        compare("  same, parameters shifted by 0.1%", case[0], case[2], shift_seed=3)
    cavity = df.Setup()
    cavity.add("laser", "l0", power=1)
    cavity.add("mirror", "m0", reflectivity=0.99, loss=0)
    cavity.add("mirror", "m1", reflectivity=0.991, loss=0)
    cavity.space("l0", "m0", length=1)
    cavity.space("m0", "m1", length=1)
    for detector, target, port, direction in (("refl", "m0", "left", "out"), ("circ", "m1", "left", "in"), ("trns", "m1", "right", "out")):
        cavity.add("detector", detector, target=target, port=port, direction=direction)
    case = prepare(cavity, [("m0", "tuning")], jnp.linspace(-180, 180, 50))
    compare("cavity, carrier sweep over a tuning", case[0], case[2])
    for size, seed in ((2, 1), (2, 2), (3, 3)):
        case = prepare(uifo(size, random=True, random_seed=seed)[0], [("f", "frequency")], jnp.logspace(2, 3.5, 40))
        compare(f"random UIFO, {size}x{size} grid, seed {seed}", case[0], case[2])

    print("3. jit, vmap and the fallback (main example)")
    host_f, jax_f, start = main_case
    weights = [jnp.asarray(np.random.default_rng(5).standard_normal(np.shape(o))) / max(float(jnp.abs(o).max()), 1e-300) for o in jax_f(start)]
    scalar = lambda f: (lambda p: sum(jnp.sum(jnp.real(o * w)) for o, w in zip(f(p), weights)))
    value_h, grad_h = jax.jit(jax.value_and_grad(scalar(host_f)))(start)
    value_j, grad_j = jax.jit(jax.value_and_grad(scalar(jax_f)))(start)
    print(f"   jit(value_and_grad): value {check('jit value', abs(float(value_h - value_j)) / abs(float(value_j)), 1e-6)}, gradient {check('jit gradient', relative(grad_h, grad_j), 1e-5)}")
    batch = jnp.stack([start, start * 1.0001])
    mapped = jax.vmap(host_f)(batch)
    single = [host_f(p) for p in batch]
    worst = max(relative(mapped[i][k], single[k][i]) for i in range(3) for k in range(2))
    print(f"   vmap over two parameter sets against one at a time: {check('vmap', worst, 1e-12)}")
    simulate_module.LINKING_FUNCTIONS.append(lambda x: 2 * x)
    used_host = host.supported(simulate_module.LINKING_FUNCTIONS)
    simulate_module.LINKING_FUNCTIONS.pop()
    print(f"   a non-identity linking function switches to the JAX version: {not used_host}")
    if used_host:
        FAILURES.append("linking functions did not trigger the fallback")
    other = jax.grad(lambda v: jnp.sum(jnp.real(simulate_module.simulate(**dict(arrays, optimized_parameters=start, signal_changing_values=v))[2])))(arrays["signal_changing_values"])
    print(f"   gradient with respect to another input is NaN, not a silent zero: {bool(jnp.isnan(other).all())}")
    if not bool(jnp.isnan(other).all()):
        FAILURES.append("unsupported gradient was not NaN")

    print("4. Finite differences, one design variable of every kind (main example, benchmark loss)")
    reference = json.loads((ROOT / "examples" / "data" / "uifo_800_3000_test_data.json").read_text())
    arrays, ports, start, pairs = saved_setup("uifo_800_3000.json", 800, 3000)
    arrays, ports, *_ = df.run_build_step(df.Setup.from_data(json.loads((ROOT / "examples" / "data" / "uifo_800_3000.json").read_text())), [("f", "frequency")], jnp.asarray(reference["frequencies"]), pairs)
    ports, expected = jnp.asarray(ports), jnp.asarray(reference["sensitivities"])

    def loss(p):
        carrier, signal, noise = simulate_module.simulate(**dict(arrays, optimized_parameters=p))
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        return jnp.mean(jnp.log10((noise / jnp.abs(detected)).squeeze() / expected))

    loss_jit = jax.jit(loss)
    gradient = np.asarray(jax.jit(jax.grad(loss))(start))
    # For each kind of design variable, try the ones with the largest gradients until a
    # central difference settles: two step sizes, one half the other, must agree within 1%.
    # This system is very sensitive, so the usable step differs a lot between variables.
    by_kind = {}
    for index, pair in enumerate(pairs):
        by_kind.setdefault("tied lengths" if isinstance(pair[0], list) else pair[1], []).append(index)

    def central(index, step):
        return (float(loss_jit(start.at[index].add(step))) - float(loss_jit(start.at[index].add(-step)))) / (2 * step)

    for kind, indices in by_kind.items():
        result = None
        for index in sorted(indices, key=lambda i: -abs(gradient[i]))[:4]:
            scale = max(abs(float(start[index])), 1e-6)
            for exponent in range(-3, -9, -1):
                coarse, fine = central(index, scale * 10.0**exponent), central(index, scale * 10.0**exponent / 2)
                if abs(coarse - fine) <= 0.01 * abs(fine) and fine != 0:
                    result = (index, fine)
                    break
            if result:
                break
        name = lambda i: str(pairs[i]) if kind != "tied lengths" else f"{len(pairs[i])} tied spaces"
        if result is None:
            print(f"   {kind:13} finite differences did not settle for the {min(4, len(indices))} variables tried, not judged")
            continue
        index, estimate = result
        error = abs(gradient[index] - estimate) / abs(estimate)
        print(f"   {kind:13} {name(index):30} gradient {gradient[index]: .6e}  finite difference {estimate: .6e}  relative difference {check('finite difference ' + kind, error, 0.02)}")

    print("5. Situations no public setup has (aLIGO as the base)")
    two_detectors = aligo()[0]
    two_detectors.add("qnoised", "noise-inside", target="srm", port="left", direction="in")
    case = prepare(two_detectors, [("f", "frequency")], frequencies)
    host_f, jax_f, start = compare("two noise detectors", case[0], case[2])
    print(f"   two noise detectors, gradient against finite differences: {directional('two detectors, finite differences', host_f, start)}")
    case = prepare(aligo()[0], [("f", "frequency")], frequencies)
    doubled = written_twice(case[0])
    host_f, jax_f, start = compare("cells written twice", doubled, case[2])
    untouched = simulate_module.simulate(**dict(case[0], optimized_parameters=start))
    changed = max(relative(a, b) for a, b in zip(host_f(start), untouched))
    print(f"   cells written twice: the second write changes the result by {changed:.1e} (so the case is not empty)")
    if changed < 1e-6:
        FAILURES.append("the written-twice case did not change anything")
    print(f"   cells written twice, gradient against finite differences: {directional('written twice, finite differences', host_f, start)}")

    print()
    if FAILURES:
        print("FAILED:")
        for failure in FAILURES:
            print("  ", failure)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
