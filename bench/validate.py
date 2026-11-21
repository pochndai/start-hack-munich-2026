#!/usr/bin/env python3
"""Correctness check for one version. Deterministic, so it runs once per code state.

    python bench/validate.py v00-baseline --make-reference   # once: the single baseline run
    python bench/validate.py WORKTREE                        # check the uncommitted candidate
    python bench/validate.py v01-something --allow-deviations   # accept deviations you have inspected

Two layers:
1. The version's own benchmark.py runs to completion and must print PASS
   (reference sensitivity within 1e-5 in log, finite outputs, one
   finite-difference gradient spot check).
2. Every output and each of the 193 gradient entries is compared with the
   arrays the unmodified baseline produced (`bench/reference/baseline_arrays.npz`).

Verdicts: PASS, FAIL (also when an array differs from the baseline by more
than the tolerance), PASS_WITH_DEVIATIONS (such a difference, explicitly
accepted with --allow-deviations), PASS_NO_REFERENCE.

`--make-reference` is the one and only run of the baseline. It stores the
oracle arrays, the baseline's first-run total for this machine
(`results/<machine>/baseline.json`) and its official report and plot.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

REFERENCE_ARRAYS = common.REFERENCE / "baseline_arrays.npz"  # provenance sits next to it as .json
DEFAULT_TOLERANCES = {
    "outputs_scale_rel": 1e-6,     # carrier, signal, noise, sensitivity, powers: max|a-b| / max|b|
    "log_sensitivity_abs": 1e-6,   # max |ln(sensitivity) - ln(baseline)|; the official bar is 1e-5
    # Gradient, entry by entry: |a-b| <= floor + gradient_rtol * |b|, where
    # floor = gradient_floor_scale_rel * max|b| stands for "numerically zero".
    "gradient_rtol": 1e-4,
    "gradient_floor_scale_rel": 1e-12,
}
OUTPUT_KEYS = ("carrier", "signal", "noise", "sensitivity", "powers")
INPUT_KEYS = ("parameters", "frequencies")


def compare_arrays(candidate_path, reference_path, tolerances):
    """Return ``(metrics, problems, deviations)`` for a candidate .npz against the reference."""
    import numpy as np

    metrics, problems, deviations = {}, [], []
    with np.load(candidate_path) as candidate, np.load(reference_path) as reference:
        for key in INPUT_KEYS:
            if key not in candidate.files or not np.array_equal(candidate[key], reference[key]):
                problems.append(f"input '{key}' differs from the baseline's")
        for key in (*OUTPUT_KEYS, "gradient"):
            if key not in candidate.files:
                problems.append(f"array '{key}' missing")
                continue
            ours, theirs = candidate[key], reference[key]
            if ours.shape != theirs.shape:
                problems.append(f"'{key}' has shape {ours.shape}, baseline {theirs.shape}")
                continue
            if not np.isfinite(ours).all():
                problems.append(f"'{key}' contains NaN or infinity")
                continue
            difference, scale = np.abs(ours - theirs), float(np.max(np.abs(theirs)))
            floor = (tolerances["gradient_floor_scale_rel"] if key == "gradient" else 1e-9) * scale
            significant = np.abs(theirs) > floor
            entry = {
                "max_abs": float(difference.max()),
                "scale_rel": float(difference.max() / scale) if scale > 0 else float(difference.max()),
                "max_elementwise_rel": float((difference[significant] / np.abs(theirs[significant])).max()) if significant.any() else 0.0,
            }
            if key == "gradient":
                # A tolerance relative to the largest entry would leave small entries
                # unchecked: the gradient spans many orders of magnitude.
                allowed = floor + tolerances["gradient_rtol"] * np.abs(theirs)
                excess = (difference / np.maximum(allowed, 1e-300)).ravel()
                worst = int(np.argmax(excess))
                entry.update(
                    floor=floor, entries=int(theirs.size), entries_outside=int((difference > allowed).sum()),
                    worst_index=worst, worst_ours=float(ours.ravel()[worst]), worst_baseline=float(theirs.ravel()[worst]),
                    worst_error_over_allowed=float(excess[worst]),
                )
                if entry["entries_outside"]:
                    deviations.append(
                        f"gradient: {entry['entries_outside']} of {entry['entries']} entries outside "
                        f"|a-b| <= {floor:.1e} + {tolerances['gradient_rtol']:.0e}*|baseline|; worst is entry {worst}: "
                        f"{entry['worst_ours']:.6e} vs baseline {entry['worst_baseline']:.6e}")
            elif entry["scale_rel"] > tolerances["outputs_scale_rel"]:
                deviations.append(f"{key}: scale-relative difference {entry['scale_rel']:.3e} > {tolerances['outputs_scale_rel']:.1e}")
            if key == "sensitivity" and (ours > 0).all():
                entry["max_log_abs"] = float(np.abs(np.log(ours) - np.log(theirs)).max())
                if entry["max_log_abs"] > tolerances["log_sensitivity_abs"]:
                    deviations.append(f"sensitivity: log difference {entry['max_log_abs']:.3e} > {tolerances['log_sensitivity_abs']:.1e}")
            metrics[key] = entry
    return metrics, problems, deviations


def parameter_name(pair):
    """'mb11.tuning', or 'a.length = b.length' for tied parameters (benchmark.py's optimization_pairs)."""
    members = pair if isinstance(pair[0], list) else [pair]
    return " = ".join(".".join(member) for member in members)


def record_path(machine_dir, info):
    return Path(machine_dir) / "validation" / f"{common.safe_name(info['name'])}@{info['id'][:12]}.json"


def load_record(machine_dir, info):
    path = record_path(machine_dir, info)
    return common.read_json(path) if path.exists() else None


def baseline_timing_path(machine_dir):
    return Path(machine_dir) / "baseline.json"


def load_baseline_timing(machine_dir, registry):
    """The baseline's single stored run on this machine, or None if it has not been made."""
    path = baseline_timing_path(machine_dir)
    stored = common.read_json(path) if path.exists() else None
    return stored if stored and stored.get("label") == registry.get("baseline") else None


def accept_deviations(record):
    """Turn a FAIL that rests only on deviations into PASS_WITH_DEVIATIONS. True if it did."""
    if record["verdict"] != "FAIL" or not record["official"]["passed"] or record["problems"] or not record["deviations"]:
        return False
    record["verdict"], record["deviations_allowed"] = "PASS_WITH_DEVIATIONS", True
    return True


def validate_version(info, registry, machine_dir, limits, *, repeats=3, timeout=7200.0, make_reference=False,
                     allow_deviations=False, official_dir=None, reference_arrays=None):
    """Run the official benchmark to completion, compare arrays, store and return the record."""
    reference_arrays = Path(reference_arrays or REFERENCE_ARRAYS)
    tolerances = {**DEFAULT_TOLERANCES, **(registry.get("tolerances") or {})}
    work = common.CACHE / "validation" / f"{common.safe_name(info['name'])}@{info['id'][:12]}"
    shutil.rmtree(work, ignore_errors=True)
    arrays, report_path = work / "arrays.npz", work / "report.json"
    # The baseline is never rerun for timing: this run also supplies its stored first-run total.
    timing_path = baseline_timing_path(machine_dir)
    keep_timing = info["name"] == registry.get("baseline") and (make_reference or not timing_path.exists())
    sample = {"label": info["name"], "round": 0}
    if keep_timing:
        sample.update(canary={"seconds": common.run_canary(sys.executable, limits), "attempts": 1, "waited_seconds": 0.0},
                      load_before=common.load_average(), memory_before=common.memory_state())
    print(f"[validate] {info['name']}: full official run ({repeats} warm repeats); this takes a while...", flush=True)
    run = common.run_benchmark(
        info["snapshot"], python=info["python"], limits=limits, log_path=work / "log.txt", output_json=report_path,
        timeout=timeout, stop_after_first_run=False, repeats=repeats, save_arrays=arrays, extra_env=info["env"],
    )
    record = {
        "schema": common.SCHEMA, "name": info["name"], "id": info["id"], "kind": info["kind"],
        "validated_utc": common.utc_now().isoformat(), "machine_id": Path(machine_dir).name,
        "benchmark_sha256": info["benchmark_sha256"], "limits": limits, "repeats": repeats, "tolerances": tolerances,
        "official": {"passed": run["status"] == "ok", "status": run["status"], "returncode": run["returncode"]},
        "run": {key: run.get(key) for key in ("end_to_end", "wall_total", "cpu_per_wall", "peak_rss_bytes", "peak_footprint_bytes")},
        "arrays": None, "problems": [], "deviations": [], "deviations_allowed": False,
    }
    parameters = None
    if run["status"] != "ok":
        record["problems"].append("official benchmark did not finish with PASS")
        record["log_tail"] = run.get("log_tail")
    elif report_path.exists():
        report = common.read_json(report_path)
        parameters = (report.get("inputs") or {}).get("optimization_pairs")
        record["official"].update(
            end_to_end=report["timing_seconds"]["end_to_end"], warm_median=report["timing_seconds"].get("runtime_median"),
            max_log_sensitivity_error=report["validation"].get("max_absolute_log_sensitivity_error"),
            gradient_check=report["validation"].get("gradient_check"), loss=report["validation"].get("loss"),
            gradient_norm=report["validation"].get("gradient_norm"),
        )

    if run["status"] == "ok" and make_reference:
        reference_arrays.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arrays, reference_arrays)
        common.write_json(reference_arrays.with_suffix(".json"), {
            "source_version": info["name"], "source_id": info["id"], "created_utc": record["validated_utc"],
            "machine_id": record["machine_id"], "official": record["official"],
        })
        print(f"[validate] reference arrays written to {reference_arrays}")
    if run["status"] == "ok" and keep_timing:
        sample.update(run, memory_after=common.memory_state())
        common.write_json(timing_path, {
            "schema": common.SCHEMA, "label": info["name"], "id": info["id"], "measured_utc": record["validated_utc"],
            "machine_id": record["machine_id"], "limits": limits, "python": platform.python_version(),
            "note": "One complete official run. end_to_end is its scored first-run total; CPU time, peak resident "
                    "size and memory_after also cover the warm repeats and validation that followed in the same process.",
            "sample": sample,
        })
        print(f"[validate] baseline first-run total ({run['end_to_end']:.3f} s) stored in {timing_path}")

    if run["status"] != "ok":
        verdict = "FAIL"
    elif not reference_arrays.exists():
        verdict = "PASS_NO_REFERENCE"
    else:
        record["arrays"], record["problems"], record["deviations"] = compare_arrays(arrays, reference_arrays, tolerances)
        worst = (record["arrays"].get("gradient") or {}).get("worst_index")
        if parameters and worst is not None and worst < len(parameters):
            record["arrays"]["gradient"]["worst_parameter"] = parameter_name(parameters[worst])
        verdict = "FAIL" if record["problems"] or record["deviations"] else "PASS"
    record["verdict"] = verdict
    if allow_deviations:
        accept_deviations(record)

    if official_dir and run["status"] == "ok":
        official_dir = Path(official_dir)
        official_dir.mkdir(parents=True, exist_ok=True)
        for source in (report_path, report_path.with_suffix(".png")):
            if source.exists():
                shutil.copy2(source, official_dir / f"{common.safe_name(info['name'])}{source.suffix}")
    common.write_json(record_path(machine_dir, info), record)
    return record


def print_record(record):
    print(f"\n{record['verdict']}: {record['name']} @ {record['id'][:12]}")
    official = record["official"]
    if official.get("passed"):
        warm = official.get("warm_median")
        print(f"  official benchmark: PASS, first-run {official.get('end_to_end', float('nan')):.3f} s"
              + (f", warm median {warm:.3f} s" if warm is not None else "")
              + f", max log sensitivity error vs public reference {official.get('max_log_sensitivity_error')}")
    else:
        print(f"  official benchmark: {official['status']} (exit code {official['returncode']})")
        for line in record.get("log_tail") or []:
            print("    | " + line)
    for key, entry in (record.get("arrays") or {}).items():
        if "worst_index" in entry:
            name = f" ({entry['worst_parameter']})" if entry.get("worst_parameter") else ""
            print(f"  {key:12s} vs baseline: {entry['entries_outside']} of {entry['entries']} entries outside tolerance, worst relative error "
                  f"{entry['max_elementwise_rel']:.2e}; closest to the limit: entry {entry['worst_index']}{name} at "
                  f"{entry['worst_error_over_allowed']:.2g}x the allowed error")
            continue
        extra = f", log {entry['max_log_abs']:.2e}" if "max_log_abs" in entry else ""
        print(f"  {key:12s} vs baseline: scale-relative {entry['scale_rel']:.2e}, worst element {entry['max_elementwise_rel']:.2e}{extra}")
    for line in record["problems"]:
        print("  PROBLEM: " + line)
    for line in record["deviations"]:
        print("  DEVIATION: " + line)
    if record["deviations"] and record["verdict"] == "FAIL" and not record["problems"]:
        print("  Deviations fail the check. If you have inspected them and they are acceptable, rerun with --allow-deviations.")
    if record["verdict"] == "PASS_NO_REFERENCE":
        print("  No reference arrays yet: run `python bench/validate.py v00-baseline --make-reference`.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="registry name, WORKTREE, path:<dir> or any git ref")
    parser.add_argument("--make-reference", action="store_true",
                        help="the single baseline run: store its arrays as the oracle, its first-run total and its official report")
    parser.add_argument("--allow-deviations", action="store_true",
                        help="accept arrays that differ from the baseline by more than the tolerance (verdict PASS_WITH_DEVIATIONS)")
    parser.add_argument("--force", action="store_true", help="re-run even if this exact code state was validated before")
    parser.add_argument("--repeats", type=int, default=3, help="warm repeats of the official run (minimum 3)")
    parser.add_argument("--timeout", type=float, default=7200.0, help="seconds before the run is killed")
    args = parser.parse_args()

    registry = common.load_registry()
    info = common.resolve_version(args.version, registry)
    machine_dir = common.RESULTS / common.machine_info()["machine_id"]
    limits, _ = common.load_limits(machine_dir)
    common.isolate_harness(limits)
    if args.make_reference and info["name"] != registry.get("baseline"):
        print(f"WARNING: making the reference from {info['name']}, which is not the registered baseline {registry.get('baseline')}.")
    record = None if (args.force or args.make_reference) else load_record(machine_dir, info)
    if record:
        print(f"(cached result from {record['validated_utc']}; use --force to re-run)")
        if args.allow_deviations and accept_deviations(record):  # the arrays are unchanged, so no new run is needed
            common.write_json(record_path(machine_dir, info), record)
    else:
        record = validate_version(info, registry, machine_dir, limits, repeats=args.repeats, timeout=args.timeout,
                                  make_reference=args.make_reference, allow_deviations=args.allow_deviations,
                                  official_dir=machine_dir / "official" if args.make_reference else None)
    print_record(record)
    sys.exit(1 if record["verdict"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
