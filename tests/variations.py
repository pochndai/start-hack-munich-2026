#!/usr/bin/env python3
"""Check and time the simulator on setups other than the supplied one.

The jury scores "a slight variation of the supplied example", and benchmark.py
runs the supplied example only. Each file in tests/variations/ is one such
variation: a complete setup and its frequencies. For every case this script

  1. runs the original code once and keeps its readout, sensitivity and
     gradient as the reference (bench/reference/variation-<case>.npz);
  2. runs the working tree in a fresh process, with the timed phases of
     benchmark.py, and compares the results with that reference;
  3. checks the working tree's gradient against finite differences;
  4. stores the outcome under bench/results/<machine>/variations/.

    python tests/variations.py                 # every case
    python tests/variations.py NAME [NAME ...] # some cases
    python tests/variations.py --list
    python tests/variations.py --make-reference NAME   # rerun the original for a case
    python tests/variations.py --generate      # rewrite the case files (same seeds, same files)

A case's verdict is PASS, DEVIATION (inside the judges' limit of 1e-5 in log
sensitivity, but some value differs from the original by more than the
tolerances of bench/validate.py), FAIL or NO_REFERENCE.
"""
from time import perf_counter

STARTED = perf_counter()
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).resolve().parent / "variations"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]  # the design variables of benchmark.py
JUDGES_LOG_LIMIT = 1e-5


def members_of(pair):
    """A design variable is one [name, property] pair, or a group of pairs for tied lengths."""
    return [pair] if isinstance(pair[0], str) else pair


# --------------------------------------------------------------------------
# worker: one fresh process per run. Mirrors benchmark.py phase by phase; only
# the inputs differ. Runs on any checkout, the original included.
# --------------------------------------------------------------------------
def worker(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    if args.cpus is not None:  # as in benchmark.py: pin before the numerical libraries start
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpus])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(args.cpus)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces, uifo

    if Path(df.__file__).resolve().parent != repo / "differometor":
        raise SystemExit(f"imported differometor from {df.__file__}, not from {repo}")
    phases = {"imports": perf_counter() - STARTED}

    def timed(key, label, function):
        start = perf_counter()
        result = function()
        jax.block_until_ready(result)
        phases[key] = perf_counter() - start
        print(f"{label}: {phases[key]:.6f} s", flush=True)
        return result

    def compile_phase(key, label, function, *values):
        lowered = timed(key + "_lower", label + ": trace / lower", lambda: jax.jit(function).lower(*values))
        return timed(key + "_compile", label + ": compile", lowered.compile)

    def forward_with_pullback(parameters, arrays):
        return jax.vjp(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)), parameters)

    def readout(outputs, detector_ports, reference):
        carrier, signal, noise = outputs
        signals = df.signal_detector(carrier, signal)[detector_ports]
        detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        loss = jnp.mean(jnp.log10(sensitivity / reference))
        return loss, (sensitivity, df.power_detector(carrier))

    def sensitivity_with_pullback(outputs, detector_ports, reference):
        return jax.vjp(lambda values: readout(values, detector_ports, reference), outputs, has_aux=True)

    def backward(simulation_pullback, readout_pullback, cotangent):
        return simulation_pullback(readout_pullback(cotangent)[0])[0]

    def initialize():
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_compilation_cache", False)
        return jax.devices("cpu")

    timed("initialize", "Initialize CPU backend", initialize)
    case = timed("load", "Read and parse inputs", lambda: json.loads(args.worker.read_bytes()))

    def build_inputs():
        if "uifo" in case:  # a setup with linking functions cannot be stored as JSON, so it is rebuilt
            recipe = case["uifo"]
            setup = uifo(recipe["size"], centers={key: tuple(value) for key, value in recipe["centers"].items()},
                         boundaries=recipe["boundaries"], mode=recipe["mode"])[0]
            values = case["values"]
        else:
            setup = df.Setup.from_data(case["setup"])
            values = {name: item["properties"] for kind in ("nodes", "edges") for name, item in case["setup"][kind].items()}
        pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        initial = [values[members_of(pair)[0][0]][members_of(pair)[0][1]] for pair in pairs]
        parameters = jnp.asarray(initial, dtype=jnp.float64)
        frequencies = jnp.asarray(case["frequencies"], dtype=jnp.float64)
        arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
        arrays["optimized_parameters"] = parameters
        # No public reference exists for a variation, so the loss is mean(log10(sensitivity)).
        return arrays, jnp.asarray(ports), parameters, frequencies, jnp.ones_like(frequencies), pairs, jnp.asarray(1.0, dtype=jnp.float64)

    arrays, ports, parameters, frequencies, reference, pairs, one = timed("build", "Build optical setup and prepare inputs", build_inputs)
    forward = compile_phase("forward", "Forward", forward_with_pullback, parameters, arrays)
    outputs, simulation_pullback = timed("forward", "Forward simulation + AD residuals", lambda: forward(parameters, arrays))
    sensitivity_fn = compile_phase("sensitivity", "Sensitivity", sensitivity_with_pullback, outputs, ports, reference)
    loss, readout_pullback, (sensitivity, powers) = timed("sensitivity", "Sensitivity, powers and scalar loss", lambda: sensitivity_fn(outputs, ports, reference))
    gradient_fn = compile_phase("gradient", "Gradient", backward, simulation_pullback, readout_pullback, one)
    gradient = timed("gradient", "Reverse-mode gradient", lambda: gradient_fn(simulation_pullback, readout_pullback, one))
    host = timed("host_transfer", "Transfer results to host", lambda: jax.device_get((outputs, sensitivity, powers, loss, gradient)))
    total = sum(phases.values())
    print(f"Complete first-run total (excluding validation): {total:.6f} s", flush=True)

    # Everything below is outside the timed run.
    (carrier, signal, noise), sensitivity, powers, loss, gradient = host
    detected = np.asarray(df.signal_detector(outputs[0], outputs[1])[ports])
    host_module, simulate_module = sys.modules.get("differometor.host"), sys.modules["differometor.simulate"]
    if host_module is None:  # the original code has no NumPy path
        path = "original"
    else:
        path = "NumPy" if simulate_module.HOST_SIMULATION and host_module.supported(simulate_module.LINKING_FUNCTIONS) else "JAX fallback"
    report = {
        "first_run_total": total, "phases": phases, "path": path, "loss": float(loss),
        "frequencies": int(frequencies.size), "design_variables": int(parameters.size),
        "signal_system_size": int(np.shape(signal)[0]), "detector_ports": int(ports.shape[0]),
        "finite": bool(all(np.isfinite(value).all() for value in (carrier, signal, noise, sensitivity, powers, gradient))),
        "positive_sensitivity": bool((np.asarray(sensitivity) > 0).all()),
        "gradient_nonzero": bool(np.any(gradient)),
    }
    if args.check_gradient:
        scalar = jax.jit(lambda p: readout(df.simulate(**dict(arrays, optimized_parameters=p)), ports, reference)[0])
        rng, worst, settled = np.random.default_rng(7), 0.0, 0
        # A reflectivity within 1e-6 of 0 or 1 stays fixed: sqrt(R) and sqrt(1 - R) have no finite slope there, so a
        # finite difference says nothing. Those gradient entries are still compared with the original's.
        start = np.asarray(parameters)
        moved = np.array([members_of(pair)[0][1] != "reflectivity" or 1e-6 < value < 1 - 1e-6 for pair, value in zip(pairs, start)])
        for _ in range(4):
            direction = jnp.asarray(rng.standard_normal(start.shape) * np.maximum(np.abs(start), 1e-6) * moved)
            slope = lambda h: (float(scalar(parameters + h * direction)) - float(scalar(parameters - h * direction))) / (2 * h)
            # The step grows from tiny until two step sizes agree within 1%. Growing, not shrinking: a cavity
            # resonance is so narrow that large steps agree with each other on the wrong, averaged slope.
            for exponent in range(-9, -2):
                coarse, fine = slope(10.0**exponent), slope(10.0**exponent / 2)
                if fine != 0 and abs(coarse - fine) <= 0.01 * abs(fine):
                    worst, settled = max(worst, abs(float(np.asarray(gradient) @ np.asarray(direction)) - fine) / abs(fine)), settled + 1
                    break
        report["finite_differences"] = {"worst_relative_error": worst, "directions_settled": settled, "directions": 4, "limit": 0.02,
                                         "variables_moved": int(moved.sum()), "variables": int(moved.size)}
    # The readout signal is stored, not the full field vector: it is what the sensitivity is made of, and it keeps the reference files small.
    np.savez_compressed(args.arrays, carrier=carrier, signal=detected, noise=noise, sensitivity=sensitivity, powers=powers,
                        gradient=gradient, parameters=np.asarray(parameters), frequencies=np.asarray(frequencies))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------
def generate():
    import copy
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from differometor.setups import Setup, constrain_inter_grid_cell_spaces, uifo

    data_dir = ROOT / "examples" / "data"
    supplied = json.loads((data_dir / "uifo_800_3000.json").read_text())
    second = json.loads((data_dir / "uifo_20_5000.json").read_text())
    band = json.loads((data_dir / "uifo_800_3000_test_data.json").read_text())["frequencies"]
    wide = np.geomspace(20, 5000, 101).tolist()
    # The ranges examples/uifo_optimization.py lets the optimizer move in.
    bounds = {"db": (0, 10), "angle": (-360, 360), "power": (0, 200), "tuning": (-360, 360), "mass": (0.01, 200), "length": (0.1, 4000), "reflectivity": (0, 1)}

    def revalued(data, new_value, seed):
        """Give every design variable a new value; tied lengths keep sharing one."""
        data, rng = copy.deepcopy(data), np.random.default_rng(seed)
        place = lambda name: data["nodes" if name in data["nodes"] else "edges"][name]["properties"]
        for pair in constrain_inter_grid_cell_spaces(Setup.from_data(copy.deepcopy(data)).parameters, PROPERTIES):
            members = members_of(pair)
            value = float(new_value(rng, members[0][1], place(members[0][0])[members[0][1]]))
            for name, prop in members:
                place(name)[prop] = value
        return data

    shifted = lambda rng, prop, old: old * (1 + 0.01 * rng.standard_normal())
    drawn = lambda rng, prop, old: rng.uniform(*bounds[prop])

    def layout(size, centers=None, boundaries=None, values=None, **options):
        """Build a grid with uifo(); parts that also exist in `values` (same name, same kind) take its numbers."""
        setup = uifo(size, centers={key: tuple(value) for key, value in centers.items()} if centers else None, boundaries=boundaries, **options)[0]
        data = json.loads(json.dumps(setup.to_data()))
        for kind in ("nodes", "edges"):
            for name, item in data[kind].items():
                old = (values or {}).get(kind, {}).get(name)
                if old and old.get("component") == item.get("component"):
                    item["properties"].update({key: value for key, value in old["properties"].items() if key in item["properties"]})
        return data

    def changed(boundaries=None, centers=None):
        return layout(3, centers=dict(supplied["centers"], **(centers or {})), boundaries=dict(supplied["boundaries"], **(boundaries or {})), values=supplied)

    cases = {
        "supplied": ("The supplied example, unchanged (control: must reproduce benchmark.py).", supplied, band),
        "values-shifted": ("Supplied layout; every design variable moved by about 1%.", revalued(supplied, shifted, 1), band),
        "values-random": ("Supplied layout; every design variable drawn at random from the optimizer's ranges.", revalued(supplied, drawn, 2), band),
        "band-20-5000": ("Supplied setup at 101 frequencies from 20 to 5000 Hz.", supplied, wide),
        # shortcut: fewer frequencies than 101, not more, because the original needs about 140 MB per frequency; add a 201 case on a machine with 32 GB or more.
        "frequencies-51": ("Supplied setup at 51 frequencies from 800 to 3000 Hz.", supplied, np.geomspace(800, 3000, 51).tolist()),
        "uifo-20-5000": ("The second public setup (other layout, balanced homodyne readout) at 101 frequencies from 20 to 5000 Hz.", second, wide),
        "boundary-swap": ("Supplied layout with the laser at boundary 10 and the squeezer at boundary 20 exchanged.", changed(boundaries={"10": "squeezer", "20": "laser"}), band),
        "center-swap": ("Supplied layout with center 22 made a directional beamsplitter and center 11 a plain one.", changed(centers={"22": ["directional_beamsplitter", "left"], "11": ["beamsplitter", "left"]}), band),
        "detector-moved": ("Supplied layout with the detector moved from boundary 03 to boundary 30.", changed(boundaries={"03": "laser", "30": "detector"}), band),
        "homodyne-readout": ("Supplied layout with the detector replaced by a balanced homodyne readout.", changed(boundaries={"03": "balanced_homodyne"}), band),
    }
    for seed in (1, 2, 3):
        cases[f"random-layout-{seed}"] = (f"A random 3x3 layout (uifo with random_seed={seed}) with random design values.",
                                          revalued(layout(3, random=True, random_seed=seed), drawn, 10 + seed), band)
    cases["size-2"] = ("A random 2x2 layout with random design values.", revalued(layout(2, random=True, random_seed=4), drawn, 14), band)
    # shortcut: 21 frequencies, because the original needs about 140 MB per frequency at this size; use all 101 on a machine with 64 GB.
    cases["size-4"] = ("A random 4x4 layout with random design values, at 21 frequencies (the original would not fit in 16 GB with 101).",
                       revalued(layout(4, random=True, random_seed=5), drawn, 15), np.geomspace(800, 3000, 21).tolist())
    CASES.mkdir(exist_ok=True)
    for name, (description, data, frequencies) in cases.items():
        setup = {key: data[key] for key in ("parameters", "nodes", "edges")}
        (CASES / f"{name}.json").write_text(json.dumps({"description": description, "frequencies": frequencies, "setup": setup}, indent=1) + "\n")
    values = {name: item["properties"] for kind in ("nodes", "edges") for name, item in supplied[kind].items()}
    (CASES / "amplitude-modulation.json").write_text(json.dumps({
        "description": "Supplied layout and design values in amplitude-modulation mode, which has linking functions, so our code takes its JAX fallback.",
        "frequencies": band, "uifo": {"size": 3, "centers": supplied["centers"], "boundaries": supplied["boundaries"], "mode": "amplitude_modulation"},
        "values": values}, indent=1) + "\n")
    print(f"Wrote {len(cases) + 1} cases to {CASES}")

    # The scenario sheet is written from the case files themselves, so it cannot drift from them.
    import re
    short = {"beamsplitter": "B", "directional_beamsplitter": "D", "laser": "laser", "squeezer": "squeezer"}
    rows, layouts = [], []
    for path in sorted(CASES.glob("*.json")):
        case = json.loads(path.read_text())
        if "uifo" in case:
            recipe = case["uifo"]
            setup = uifo(recipe["size"], centers={key: tuple(value) for key, value in recipe["centers"].items()}, boundaries=recipe["boundaries"], mode=recipe["mode"])[0]
            nodes, edges, mode = setup.to_data()["nodes"], setup.to_data()["edges"], "amplitude modulation"
        else:
            setup = Setup.from_data(copy.deepcopy(case["setup"]))
            nodes, edges, mode = case["setup"]["nodes"], case["setup"]["edges"], "space modulation"
        variables = len(constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES))
        centers = {name[6:]: short[item["component"]] for name, item in nodes.items() if re.fullmatch(r"center\d\d", name)}
        size = max(int(key[0]) for key in centers)
        boundaries = {}
        for position in sorted({match.group(1) for name in nodes if (match := re.match(r"boundary(\d\d)", name))}):
            boundaries[position] = ("detector" if f"boundary{position}detector" in nodes else "balanced homodyne" if f"boundary{position}bhbs" in nodes
                                    else short[nodes[f"boundary{position}"]["component"]])
        readout = "; ".join(f"{kind} at {position}" for position, kind in boundaries.items() if kind in ("detector", "balanced homodyne"))
        count = lambda kind: sum(value == kind for value in boundaries.values())
        f = case["frequencies"]
        rows.append(f"| `{path.stem}` | {size}x{size} | {len(f)} | {f[0]:.0f} to {f[-1]:.0f} Hz | {variables} | {readout} | {count('laser')} | {count('squeezer')} | "
                    f"{sum(v == 'B' for v in centers.values())} / {sum(v == 'D' for v in centers.values())} | {mode} | {len(nodes)} / {len(edges)} |")
        grid = " / ".join(" ".join(centers[f"{x}{y}"] for y in range(1, size + 1)) for x in range(1, size + 1))
        layouts.append(f"| `{path.stem}` | {grid} | " + ", ".join(f"{position} {kind}" for position, kind in boundaries.items()) + " |")
    descriptions = "\n".join(f"| `{path.stem}` | {json.loads(path.read_text())['description']} |" for path in sorted(CASES.glob("*.json")))
    bound_rows = "\n".join(f"| {name} | {low} | {high} |" for name, (low, high) in bounds.items())
    (CASES / "README.md").write_text(SCENARIO_SHEET.format(descriptions=descriptions, settings="\n".join(rows), layouts="\n".join(layouts), bounds=bound_rows))
    print(f"Wrote {CASES / 'README.md'}")


SCENARIO_SHEET = """# Test scenarios: setups other than the supplied one

Written by `python tests/variations.py --generate` from the case files in this folder. Do not edit by hand: change `generate()` in `tests/variations.py` and run that command.

## Why these exist

The jury scores "a slight variation of the supplied example". `benchmark.py` can only run the supplied example, so each file here is one variation we thought the jury might use: a complete optical setup and the frequencies to simulate. `tests/variations.py` runs every one on the original code and on ours, compares the results and gradients, and times our code.

## Run them

```bash
python tests/variations.py --list                  # the scenarios
python tests/variations.py                         # run all, compare with the stored results of the original
python tests/variations.py size-4 uifo-20-5000     # run some
python tests/variations.py --make-reference size-4 # run the original again for one scenario (slow, about 14 GB)
python tests/refined_solves.py band-20-5000        # when ours and the original disagree: which one is right
```

On a new machine, run `python bench/run.py calibrate` first. On Linux the runs are then pinned to one core (`--cpus 1`), as in the official benchmark; on a Mac every numerical library is limited to one thread. The stored results of the original (`bench/reference/variation-<scenario>.npz`) are in the repository, so the original only has to run again for a new or changed scenario.

Results: one session file per run in `bench/results/<machine>/variations/`, and the table of the latest session in `bench/results/<machine>/VARIATIONS.md`.

## The scenarios

| Scenario | What it is |
| --- | --- |
{descriptions}

## Settings of each scenario

"Design variables" are the reflectivities, tunings, squeezing levels and angles, laser powers, masses and tied lengths that `benchmark.py` differentiates. "Beamsplitters" counts plain / directional ones at the grid centers. "Nodes / edges" is the number of components and connections.

| Scenario | Grid | Frequencies | Band | Design variables | Readout | Lasers | Squeezers | Beamsplitters | Signal mode | Nodes / edges |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
{settings}

All frequency lists are spaced evenly on a logarithmic scale, except the band of the supplied example, which is taken from `examples/data/uifo_800_3000_test_data.json`.

## Layout of each scenario

Centers are listed row by row (B = beamsplitter, D = directional beamsplitter). Boundaries are named by their grid position as in `differometor.setups.uifo`: `0y` and `(size+1)y` are the two sides in one direction, `x0` and `x(size+1)` the two in the other. The orientation of each center is in the case file.

| Scenario | Centers | Boundaries |
| --- | --- | --- |
{layouts}

## Where the numbers in a scenario come from

- **Supplied values:** the optimized values stored in `examples/data/uifo_800_3000.json` (or `uifo_20_5000.json` for `uifo-20-5000`).
- **Shifted values** (`values-shifted`): every design variable is multiplied by `1 + 0.01 * z`, with `z` from a standard normal distribution, seed 1. Tied lengths share one factor.
- **Random values** (`values-random`, `random-layout-*`, `size-*`): every design variable is drawn evenly from the range the optimizer in `examples/uifo_optimization.py` works in (table below). Seeds: `values-random` 2, `random-layout-N` 10 + N, `size-2` 14, `size-4` 15.
- **Random layouts:** `uifo(size, random=True, random_seed=N)` with N = 1, 2, 3 for `random-layout-N`, 4 for `size-2` and 5 for `size-4`.
- **Changed layouts** (`boundary-swap`, `center-swap`, `detector-moved`, `homodyne-readout`): the supplied layout with the named parts exchanged. Parts that keep their name and kind keep the supplied values; new parts get the defaults of `differometor.components`.
- Everything that is not a design variable (losses, refractive indices, beamsplitter angles, lengths next to a center or a boundary) stays as supplied or at its default.

| Design variable | Lowest | Highest |
| --- | ---: | ---: |
{bounds}

## What is checked, and the limits

| Check | Limit | Verdict if outside |
| --- | --- | --- |
| Our run finishes; all results finite; sensitivities positive; gradient not all zero | | `FAIL` |
| ln(sensitivity), ours against the original | 1e-5 (the judges' limit) | `FAIL` |
| Our gradient against finite differences in four random directions | 2% | `FAIL` |
| Carrier, readout signal, noise, sensitivity, powers against the original, relative to the largest value | 1e-6 | `DEVIATION` |
| ln(sensitivity) against the original | 1e-6 | `DEVIATION` |
| Each gradient entry against the original | 0.01% of the entry, plus 1e-12 of the largest entry | `DEVIATION` |

The `DEVIATION` limits are those of `bench/validate.py`. The loss is `mean(log10(sensitivity))`; there is no public reference for these setups. The scenario `supplied` also has to reproduce the public reference sensitivities and the arrays that `benchmark.py` itself produced with the original code.

A `DEVIATION` does not say which code is wrong. `python tests/refined_solves.py <scenario>` answers that: it repeats our calculation with every sparse solve refined to a near-exact result and shows how far ours and the original are from it.

## Add a scenario

1. In `generate()` in `tests/variations.py`, add one entry to `cases`: a name, one sentence, the setup and the frequencies. Use `changed(...)` for a different layout, `revalued(...)` for different values, `layout(...)` for a new grid.
2. `python tests/variations.py --generate` writes the case file and this sheet. The existing case files come out byte for byte the same.
3. `python tests/variations.py <name>` runs the original once, stores its results, and compares.
4. Commit the case file, this sheet, `bench/reference/variation-<name>.*` and the new session under `bench/results/`.

## Not covered yet

- More than 101 frequencies, and a 4x4 grid at 101 frequencies: the original needs about 140 MB per frequency at 3x3 and more at 4x4, which does not fit in 16 GB. They need a machine with 32 GB or more.
- One-core Linux times for these scenarios (the stored times are from the Mac).
- Frequency-modulation mode, grids larger than 4x4, and a different set of design variables (for example refractive indices or beamsplitter angles).
- Setups that are not grids (Voyager, aLIGO, the cavity sweep): `tests/test_host.py` compares our two code paths on them, but they are not timed here.
"""


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_worker(common, case_path, repo, tag, limits, check_gradient, timeout):
    work = common.CACHE / "variations"
    work.mkdir(parents=True, exist_ok=True)
    stem = work / f"{case_path.stem}-{tag}"
    arrays, report, log = stem.with_suffix(".npz"), stem.with_suffix(".json"), stem.with_suffix(".log")
    for path in (arrays, report):
        path.unlink(missing_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(case_path), "--repo", str(repo), "--arrays", str(arrays), "--output", str(report)]
    command += ["--check-gradient"] if check_gradient else []
    command += ["--cpus", "1"] if limits.get("strategy") == "affinity" else []
    try:
        done = subprocess.run(command, env=common.child_env(limits), capture_output=True, text=True, timeout=timeout, **common.spawn_options(limits))
        log.write_text(done.stdout + done.stderr)
        failure = None if done.returncode == 0 and report.exists() else f"exit code {done.returncode}: " + (done.stderr.strip().splitlines() or ["no output"])[-1][:300]
    except subprocess.TimeoutExpired:
        failure = f"no result after {timeout} s"
    return (json.loads(report.read_text()) if failure is None else None), arrays, failure


def drive(args):
    import hashlib
    import shutil
    import numpy as np
    sys.path.insert(0, str(ROOT / "bench"))
    import common
    import validate

    registry = common.load_registry()
    machine = common.machine_info()
    machine_dir = common.RESULTS / machine["machine_id"]
    limits, calibrated = common.load_limits(machine_dir)
    tolerances = {**validate.DEFAULT_TOLERANCES, **(registry.get("tolerances") or {})}
    names = args.cases or sorted(path.stem for path in CASES.glob("*.json"))
    results = []
    for name in names:
        case_path = CASES / f"{name}.json"
        case = json.loads(case_path.read_text())
        case_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
        reference = common.REFERENCE / f"variation-{name}.npz"
        provenance = reference.with_suffix(".json")
        entry = {"case": name, "description": case["description"], "case_sha256": case_sha, "problems": [], "deviations": []}
        if reference.exists() and common.read_json(provenance)["case_sha256"] != case_sha:
            raise SystemExit(f"{name}: the case file changed since its reference was made. Rerun with --make-reference {name}.")
        if args.make_reference or not reference.exists():
            print(f"[{name}] original code (slow, memory-hungry)...", flush=True)
            baseline = common.resolve_version(registry["baseline"], registry)
            report, arrays, failure = run_worker(common, case_path, Path(baseline["snapshot"]), "original", limits, False, args.timeout)
            usable = failure is None and report["finite"] and report["positive_sensitivity"]
            if usable:
                common.REFERENCE.mkdir(exist_ok=True)
                shutil.copyfile(arrays, reference)
                common.write_json(provenance, {"case": name, "case_sha256": case_sha, "source_version": registry["baseline"], "source_id": baseline["id"],
                                               "created_utc": common.utc_now().isoformat(), "machine_id": machine["machine_id"], "limits": limits, "run": report})
            else:
                entry["original_failure"] = failure or "the original's results are not finite and positive"
        ours, arrays, failure = None, None, None
        print(f"[{name}] working tree...", flush=True)
        ours, arrays, failure = run_worker(common, case_path, ROOT, "worktree", limits, True, args.timeout)
        entry["ours"] = ours
        if failure:
            entry["problems"].append(f"our run failed: {failure}")
        else:
            for key in ("finite", "positive_sensitivity", "gradient_nonzero"):
                if not ours[key]:
                    entry["problems"].append(f"our run: {key.replace('_', ' ')} is false")
            check = ours["finite_differences"]
            if not check["directions_settled"]:
                entry["problems"].append("finite differences never settled")
            elif check["worst_relative_error"] > check["limit"]:
                entry["problems"].append(f"gradient is {check['worst_relative_error']:.1e} from finite differences (limit {check['limit']})")
        if reference.exists():
            entry["original"] = common.read_json(provenance)
            if failure is None:
                entry["metrics"], problems, entry["deviations"] = validate.compare_arrays(arrays, reference, tolerances)
                entry["problems"] += problems
                log_error = entry["metrics"].get("sensitivity", {}).get("max_log_abs")
                if log_error is None or log_error > JUDGES_LOG_LIMIT:
                    entry["problems"].append(f"log sensitivity is {log_error} from the original (judges' limit {JUDGES_LOG_LIMIT})")
        if name == "supplied" and failure is None:  # the script checks itself: this case must reproduce benchmark.py
            public = json.loads((ROOT / "examples" / "data" / "uifo_800_3000_test_data.json").read_text())["sensitivities"]
            with np.load(arrays) as mine:
                entry["log_error_vs_public_reference"] = float(np.abs(np.log(mine["sensitivity"]) - np.log(public)).max())
            if entry["log_error_vs_public_reference"] > JUDGES_LOG_LIMIT:
                entry["problems"].append("the supplied case is not within 1e-5 of the public reference")
            official = common.REFERENCE / "baseline_arrays.npz"
            if reference.exists() and official.exists():
                with np.load(reference) as mine, np.load(official) as theirs:
                    entry["reference_vs_benchmark"] = {key: float(np.abs(mine[key] - theirs[key]).max() / np.abs(theirs[key]).max()) for key in ("sensitivity", "gradient")}
                # The sensitivity must be identical. The original's gradient is only reproducible to about 5e-9 of its
                # largest entry when the loss is rescaled by a constant (measured), hence the looser bound there.
                if entry["reference_vs_benchmark"]["sensitivity"] > 1e-12 or entry["reference_vs_benchmark"]["gradient"] > 1e-6:
                    entry["problems"].append("this script's run of the original differs from benchmark.py's run of it")
        entry["verdict"] = ("FAIL" if entry["problems"] else "NO_REFERENCE" if not reference.exists() else "DEVIATION" if entry["deviations"] else "PASS")
        print(f"[{name}] {entry['verdict']}" + "".join(f"\n    {line}" for line in entry["problems"] + entry["deviations"]), flush=True)
        results.append(entry)

    stamp = common.utc_stamp()
    session = {"schema": 1, "id": stamp, "git_head": common.git_head(), "dirty": bool(common.git("status", "--porcelain", "--", ".", ":(exclude)bench")),
               "machine": machine, "limits": limits, "calibrated": calibrated, "tolerances": tolerances, "judges_log_limit": JUDGES_LOG_LIMIT, "cases": results}
    out = machine_dir / "variations"
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / f"{stamp}.json", session)
    (machine_dir / "VARIATIONS.md").write_text(table(session, f"variations/{stamp}.json"))
    print(f"\nStored {out / (stamp + '.json')}\nTable  {machine_dir / 'VARIATIONS.md'}")
    counts = {verdict: sum(entry["verdict"] == verdict for entry in results) for verdict in ("PASS", "DEVIATION", "FAIL", "NO_REFERENCE")}
    print("  ".join(f"{verdict}: {count}" for verdict, count in counts.items()))
    return 1 if counts["FAIL"] or counts["NO_REFERENCE"] else 0


def table(session, source):
    machine = session["machine"]
    lines = [
        "# Other setups: results and times", "",
        f"Generated by `python tests/variations.py` from `{source}`. Do not edit by hand.", "",
        f"Machine: {machine['cpu_model']}, {round((machine.get('ram_bytes') or 0) / 2**30)} GB, {machine['system']}. "
        f"CPU limit: {session['limits'].get('strategy')}. Code: `{str(session['git_head'])[:12]}`{' plus uncommitted changes' if session['dirty'] else ''}.", "",
        "Each time is one complete first run in a fresh process, with the phases of `benchmark.py`. The original's time is the single run that made the reference; "
        "on a 16 GB machine it is inflated by swapping. \"Log sensitivity\" is the largest difference from the original in ln(sensitivity); the judges' limit is 1e-5. "
        "\"Gradient\" counts the entries outside the tolerance of `bench/validate.py` and gives the largest difference relative to the largest entry. "
        "\"Finite differences\" is the largest relative error of our gradient in four random directions (limit 0.02).", "",
        "| Case | Frequencies | Design variables | Path | Ours | Original | Faster | Log sensitivity | Gradient | Finite differences | Verdict |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for entry in session["cases"]:
        ours, original, metrics = entry.get("ours") or {}, (entry.get("original") or {}).get("run") or {}, entry.get("metrics") or {}
        seconds = lambda run: f"{run['first_run_total']:.2f} s" if run else "-"
        gradient = metrics.get("gradient")
        check = ours.get("finite_differences")
        lines.append("| " + " | ".join([
            f"`{entry['case']}`", str(ours.get("frequencies", "-")), str(ours.get("design_variables", "-")), ours.get("path", "-"), seconds(ours), seconds(original),
            f"{original['first_run_total'] / ours['first_run_total']:.0f}x" if ours and original else "-",
            f"{metrics['sensitivity']['max_log_abs']:.1e}" if "max_log_abs" in metrics.get("sensitivity", {}) else "-",
            f"{gradient['entries_outside']} of {gradient['entries']} outside, {gradient['scale_rel']:.1e}" if gradient else "-",
            f"{check['worst_relative_error']:.1e}" if check and check["directions_settled"] else "-", entry["verdict"]]) + " |")
    lines += ["", "## What each case is", ""] + [f"- `{entry['case']}`: {entry['description']}" for entry in session["cases"]]
    notes = [f"- `{entry['case']}`: {line}" for entry in session["cases"] for line in entry["problems"] + entry["deviations"] + ([f"original: {entry['original_failure']}"] if "original_failure" in entry else [])]
    return "\n".join(lines + (["", "## Problems and deviations", ""] + notes if notes else [])) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--generate", action="store_true", help="rewrite the case files in tests/variations/")
    parser.add_argument("--make-reference", action="store_true", help="rerun the original code for the named cases")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repo", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arrays", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--check-gradient", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cpus", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.generate:
        return generate()
    if args.list:
        for path in sorted(CASES.glob("*.json")):
            print(f"{path.stem:22} {json.loads(path.read_text())['description']}")
        return 0
    if args.make_reference and not args.cases:
        parser.error("--make-reference needs case names: the original is slow, so name what you mean.")
    return drive(args)


if __name__ == "__main__":
    sys.exit(main())
