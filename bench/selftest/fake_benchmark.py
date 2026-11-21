#!/usr/bin/env python3
"""Stand-in for benchmark.py that needs no JAX: same CLI, same output lines.

Used only by `python bench/run.py selftest` to exercise the harness. Behaviour
comes from a `fake_config.json` next to this file:

    {"scale": 1.0, "fail": false, "perturb": 0.0, "scale_gradient_entry": null}

`scale` stretches every phase, `fail` makes it crash before the total line,
`perturb` shifts all saved arrays and `scale_gradient_entry` (`[index, factor]`)
multiplies a single gradient entry (both test the correctness comparison).
"""
from time import perf_counter, sleep

IMPORT_START = perf_counter()
import argparse
import json
import sys
from pathlib import Path

CONFIG = {"scale": 1.0, "fail": False, "perturb": 0.0, "scale_gradient_entry": None}
config_path = Path(__file__).with_name("fake_config.json")
if config_path.exists():
    CONFIG.update(json.loads(config_path.read_text(encoding="utf-8")))
SCALE = float(CONFIG["scale"])
sleep(0.04 * SCALE)
IMPORT_SECONDS = perf_counter() - IMPORT_START

# (key, label exactly as benchmark.py prints it, nominal seconds)
FIRST_RUN = [
    ("initialize", "Initialize CPU backend", 0.004),
    ("load", "Read and parse public inputs", 0.002),
    ("build", "Build optical setup and prepare inputs", 0.02),
    ("forward_lower", "Forward: trace / lower", 0.03),
    ("forward_compile", "Forward: compile", 0.16),
    ("forward", "Forward simulation + AD residuals", 0.14),
    ("sensitivity_lower", "Sensitivity: trace / lower", 0.004),
    ("sensitivity_compile", "Sensitivity: compile", 0.008),
    ("sensitivity", "Sensitivity, powers and scalar loss", 0.002),
    ("gradient_lower", "Gradient: trace / lower", 0.008),
    ("gradient_compile", "Gradient: compile", 0.06),
    ("gradient", "Reverse-mode gradient", 0.05),
    ("host_transfer", "Transfer results to host", 0.002),
]


def timed(label, nominal):
    start = perf_counter()
    sleep(nominal * SCALE)
    seconds = perf_counter() - start
    print(f"{label}: {seconds:.6f} s", flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--save-arrays", type=Path)
    args = parser.parse_args()

    phases = [{"key": "imports", "label": "Python / library imports", "category": "setup", "seconds": IMPORT_SECONDS}]
    print("CPU: fake; affinity: unknown; devices: [FakeDevice(id=0)]", flush=True)
    for index, (key, label, nominal) in enumerate(FIRST_RUN):
        if CONFIG["fail"] and index == 4:
            print("Traceback (most recent call last):\nRuntimeError: simulated failure", flush=True)
            sys.exit(2)
        phases.append({"key": key, "label": label, "category": "fake", "seconds": timed(label, nominal)})
    total = sum(phase["seconds"] for phase in phases)
    print(f"Complete first-run total (excluding validation): {total:.6f} s", flush=True)

    # A timing-only sample is killed right here. Lingering proves that it was.
    sleep(1.0)
    samples = []
    for index in range(args.repeats):
        start = perf_counter()
        sleep(0.02 * SCALE)
        samples.append(perf_counter() - start)
        print(f"Repeat {index + 1:02d}/{args.repeats}: {samples[-1]:.6f} s", flush=True)
    print("Validating outputs, reference sensitivity and finite-difference gradient...", flush=True)

    if args.save_arrays:
        import numpy as np

        grid = np.linspace(0.0, 1.0, 101)
        shift = float(CONFIG["perturb"])
        # Like the real gradient, the entries span many orders of magnitude (about 1e4 down to 1e-5).
        gradient = np.sin(np.arange(193.0)) * 10.0 ** np.linspace(4.0, -5.0, 193) * (1 + shift)
        if CONFIG["scale_gradient_entry"]:
            index, factor = CONFIG["scale_gradient_entry"]
            gradient[int(index)] *= factor
        np.savez_compressed(
            args.save_arrays,
            carrier=(np.outer(np.arange(1, 7), np.ones(1)) + 1j) * (1 + shift),
            signal=np.outer(np.arange(1, 9), np.exp(1j * grid)) * (1 + shift),
            noise=1e-9 * (1 + grid), sensitivity=1e-22 * (1 + grid) * (1 + shift),
            powers=np.arange(1.0, 7.0), gradient=gradient,
            parameters=np.arange(193.0), frequencies=800 + 22 * np.arange(101.0),
        )
    report = {
        "benchmark": "fake", "phases": phases,
        "timing_seconds": {"end_to_end": total, "runtime_median": sorted(samples)[len(samples) // 2], "runtime_samples": samples},
        "validation": {
            "max_absolute_log_sensitivity_error": 5.4e-06, "loss": 9.8e-08, "gradient_norm": 70517.1,
            "gradient_check": {"parameter": "mb11.tuning", "passed": True},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: reference sensitivity and gradient spot check; full runtime median {report['timing_seconds']['runtime_median']:.6f} s")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
