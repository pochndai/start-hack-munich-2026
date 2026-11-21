#!/usr/bin/env python3
"""Benchmark orchestrator: compare versions fairly on a machine you do not trust.

    python bench/run.py doctor        check the machine before producing data
    python bench/run.py calibrate     verify the one-CPU limit on this machine (once)
    python bench/run.py quick         anchor + latest version + WORKTREE, 3 rounds
    python bench/run.py full          every registered version except the baseline, 5 rounds
    python bench/run.py aa            one version against itself: the machine's noise floor
    python bench/run.py run --versions a,b,<git ref> --rounds N
    python bench/run.py add-version v01-<slug> --summary "..."
    python bench/run.py official --versions <final>
    python bench/run.py selftest      test the harness itself with a stub (no JAX needed)

A session runs R rounds. Each round measures every version once, in a new
random order, each in a fresh process. Speed-ups are computed per round and
then summarized. Every raw sample is stored; see bench/README.md.

The unmodified baseline is not part of any session. It runs once per machine
(`python bench/validate.py v00-baseline --make-reference`); sessions divide its
stored first-run total by each new sample.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import report
import validate

DEFAULT_ROUNDS = {"quick": 3, "full": 5, "aa": 5, "run": 5}
GATE_TOLERANCE, GATE_WAIT, GATE_MAX_WAIT = 0.05, 10.0, 120.0
VERSION_NAME = re.compile(r"^v\d\d-[a-z0-9][a-z0-9-]*$")


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------
def build_entries(specs, registry):
    official = common.official_benchmark_sha256(registry)
    seen, entries = {}, []
    for spec in specs:
        info = common.resolve_version(spec, registry)
        info["label"] = info["name"]
        if specs.count(spec) > 1:  # the same version twice (A/A): tell the copies apart
            seen[spec] = seen.get(spec, 0) + 1
            info["label"] += "#" + "ABCDEFGH"[seen[spec] - 1]
        info["official_benchmark"] = None if official is None else info["benchmark_sha256"] == official
        entries.append(info)
    labels = [entry["label"] for entry in entries]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Version labels must be unique, got: {', '.join(labels)}")
    return entries


def next_order(labels, previous, rng):
    order = list(labels)
    rng.shuffle(order)
    if len(order) > 1 and order == previous:  # never the same order twice in a row
        order = order[1:] + order[:1]
    return order


def canary_gate(limits, state, enabled):
    """Run the canary; while the machine is clearly slower than its best, wait (bounded)."""
    waited, attempts = 0.0, 0
    while True:
        seconds = common.run_canary(sys.executable, limits)
        attempts += 1
        if seconds is None:
            break
        state["best"] = min(seconds, state.get("best") or seconds)
        if not enabled or seconds <= state["best"] * (1 + GATE_TOLERANCE) or waited >= GATE_MAX_WAIT:
            break
        time.sleep(GATE_WAIT)
        waited += GATE_WAIT
    return {"seconds": seconds, "attempts": attempts, "waited_seconds": waited}


def warm_up(entries, limits, warnings):
    """One discarded import per version: writes .pyc files and warms the file cache."""
    for entry in entries:
        if not (Path(entry["snapshot"]) / "differometor").is_dir():
            continue
        try:
            subprocess.run(
                [entry["python"], "-c", "import jax, differometor"], cwd=entry["snapshot"],
                env=common.child_env(limits, entry["env"]), timeout=600, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **common.spawn_options(limits),
            )
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", b"") or b""
            warnings.append(f"warm-up import failed for {entry['label']}: {detail.decode('utf-8', 'replace').strip()[-300:] or error}")


def describe_sample(sample):
    if sample["status"] != "ok":
        tail = (sample.get("log_tail") or ["(no output)"])[-1]
        return f"{sample['status'].upper()}: {tail[:100]}"
    peak = sample.get("peak_footprint_bytes") or sample.get("peak_rss_bytes")
    cores = sample.get("cpu_per_wall")
    return (f"{sample['end_to_end']:9.3f} s   peak {common.format_bytes(peak)}"
            + (f"   cpu/wall {cores:.2f}" if cores else "")
            + (f"   waited {sample['canary']['waited_seconds']:.0f} s for a calm machine" if sample["canary"]["waited_seconds"] else ""))


def run_session(*, mode, specs, rounds, registry, results_root=None, cooldown=None, timeout=1800.0, seed=None,
                gate=True, validate_first=False, note=""):
    results_root = Path(results_root or common.RESULTS)
    machine = common.machine_info()
    machine_dir = results_root / machine["machine_id"]
    limits, calibrated = common.load_limits(machine_dir)
    common.isolate_harness(limits)
    cooldown = (15.0 if common.IS_MAC else 5.0) if cooldown is None else cooldown
    seed = random.SystemRandom().randrange(2**32) if seed is None else seed
    rng = random.Random(seed)
    entries = build_entries(specs, registry)
    warnings = []
    if not calibrated:
        warnings.append("CPU limit not calibrated on this machine: run `python bench/run.py calibrate`")
    power = common.power_state()
    if power.get("on_ac_power") is False:
        warnings.append("running on battery: plug the machine in")
    if power.get("low_power_mode"):
        warnings.append("Low Power Mode is on: it throttles the CPU")

    excluded = []
    for entry in entries:
        record = validate.load_record(machine_dir, entry)
        if record is None and validate_first:
            record = validate.validate_version(entry, registry, machine_dir, limits)
            validate.print_record(record)
        entry["validation"] = {"verdict": record["verdict"], "validated_utc": record["validated_utc"]} if record else None
        if validate_first and record and record["verdict"] == "FAIL":
            excluded.append(entry["label"])
    timed = [entry for entry in entries if entry["label"] not in excluded]
    if not timed:
        raise SystemExit("No version left to time: all failed validation.")
    labels = [entry["label"] for entry in timed]
    baseline = registry.get("baseline")
    # The baseline is not rerun: unless this session times it explicitly, its stored run is the reference.
    stored = None
    if mode != "aa" and baseline not in labels:
        stored = validate.load_baseline_timing(machine_dir, registry)
        if stored is None:
            warnings.append(f"no stored baseline run on this machine, so speed-ups are shown against {labels[0]}: "
                            f"run `python bench/validate.py {baseline} --make-reference` once")
        elif stored.get("limits") != limits:
            warnings.append("the stored baseline ran under a different CPU limit than this session: factors against it are not comparable")

    started = common.utc_now()
    session_id = f"{common.utc_stamp(started)}_{mode}"
    session_path = machine_dir / "sessions" / f"{session_id}.json"
    session = {
        "schema": common.SCHEMA, "id": session_id, "mode": mode, "note": note, "complete": False,
        "started_utc": started.isoformat(), "finished_utc": None,
        "git_head": common.git_head(), "machine": machine, "python": platform.python_version(),
        "limits": limits, "calibrated": calibrated, "power": power,
        "config": {"rounds": rounds, "cooldown_seconds": cooldown, "timeout_seconds": timeout, "seed": seed,
                   "canary_gate": {"enabled": gate, "tolerance": GATE_TOLERANCE, "max_wait_seconds": GATE_MAX_WAIT}},
        "registry": {"baseline": baseline, "anchor": registry.get("anchor")},
        "reference_label": baseline if (baseline in labels or stored) else labels[0],
        "stored_reference": stored,
        "entries": [{key: entry.get(key) for key in ("label", "spec", "name", "kind", "ref", "id", "dirty", "parent",
                                                    "benchmark_sha256", "official_benchmark", "validation")} for entry in timed],
        "excluded_after_failed_validation": excluded, "warnings": warnings, "orders": [], "samples": [],
    }
    print(f"Session {session_id} on {machine['machine_id']}: {len(timed)} versions x {rounds} rounds")
    print("CPU limit: " + (f"pinned to CPU {limits['bench_cpu']} (--cpus 1)" if limits.get("strategy") == "affinity"
                           else ", ".join(f"{key}={value}" for key, value in sorted(limits.get("env", {}).items())) or "none"))
    for entry in timed:
        verdict = (entry["validation"] or {}).get("verdict", "not validated")
        print(f"  {entry['label']:24s} {entry['kind']:8s} {entry['id'][:12]}  {verdict}"
              + ("" if entry["official_benchmark"] is not False else "  [benchmark.py differs from baseline]"))
    if stored:
        print(f"Baseline: not rerun; stored run of {stored['label']} from {stored['measured_utc'][:10]}, {stored['sample']['end_to_end']:.3f} s")
    for line in warnings:
        print("WARNING: " + line)
    warm_up(timed, limits, warnings)

    by_label = {entry["label"]: entry for entry in timed}
    scratch = common.CACHE / "tmp" / session_id
    canary_state, previous = {}, None
    try:
        for round_index in range(rounds):
            previous = next_order(labels, previous, rng)
            session["orders"].append(previous)
            for position, label in enumerate(previous):
                entry = by_label[label]
                tag = f"r{round_index + 1:02d}_{position + 1}_{common.safe_name(label)}"
                sample = {
                    "label": label, "round": round_index, "position": position, "started_utc": common.utc_now().isoformat(),
                    "canary": canary_gate(limits, canary_state, gate),
                    "load_before": common.load_average(), "memory_before": common.memory_state(),
                }
                log_path = common.CACHE / "logs" / session_id / f"{tag}.log"
                sample.update(common.run_benchmark(
                    entry["snapshot"], python=entry["python"], limits=limits, log_path=log_path,
                    output_json=scratch / tag / "out.json", timeout=timeout, extra_env=entry["env"],
                ))
                sample["memory_after"] = common.memory_state()
                sample["log"] = log_path.relative_to(common.BENCH_DIR).as_posix()
                session["samples"].append(sample)
                common.write_json(session_path, session)  # crash-safe: every sample is on disk at once
                print(f"  [round {round_index + 1}/{rounds}, {position + 1}/{len(labels)}] {label:24s} {describe_sample(sample)}", flush=True)
                time.sleep(cooldown)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial session kept (marked incomplete): {session_path}")
        raise SystemExit(130)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    session["finished_utc"] = common.utc_now().isoformat()
    session["complete"] = True
    common.write_json(session_path, session)
    noise, _ = report.noise_floor_from(report.load_sessions(machine_dir))
    session["summary"] = report.summarize_session(session, noise)
    common.write_json(session_path, session)
    summary_path = report.write_outputs(machine_dir)
    print("\n" + "\n\n".join(filter(None, [report.session_table(session["summary"]), report.stored_reference_note(session["summary"]),
                                           "Reading the table\n" + report.LEGEND])))
    if noise is not None:
        print(f"\nNoise floor from the latest A/A session: +/- {noise * 100:.1f} %")
    print(f"\nSession: {session_path}\nSummary: {summary_path}")
    return session, session_path


def unique(items):
    return list(dict.fromkeys(item for item in items if item))


def cmd_session(args):
    registry = common.load_registry()
    mode = args.command
    # The baseline ran once and is stored; only `run --versions` or `aa --version` can time it again.
    baseline = registry.get("baseline")
    improved = [entry["name"] for entry in registry["versions"] if entry.get("enabled", True) and entry["name"] != baseline]
    if mode == "quick":
        specs = unique([registry.get("anchor"), improved[-1] if improved else None, common.WORKTREE])
        specs = [spec for spec in specs if spec != baseline]
    elif mode == "full":
        specs = improved + ([common.WORKTREE] if args.with_worktree else [])
        if not specs:
            raise SystemExit("No version besides the baseline is registered yet, and the baseline is not rerun.\n"
                             "Time the working tree with `python bench/run.py quick` or `full --with-worktree`.")
    elif mode == "aa":
        version = args.version or registry.get("anchor") or (improved[-1] if improved else None)
        if not version:
            raise SystemExit("No version besides the baseline is registered yet. Name one explicitly, for example\n"
                             "`python bench/run.py aa --version WORKTREE`, or wait for the first registered speed-up.")
        specs = [version, version]
    else:
        specs = [item.strip() for item in args.versions.split(",") if item.strip()]
    if not specs:
        raise SystemExit("No versions to run.")
    run_session(
        mode=mode, specs=specs, rounds=args.rounds or DEFAULT_ROUNDS[mode], registry=registry, cooldown=args.cooldown,
        timeout=args.timeout, seed=args.seed, gate=not args.no_gate, note=args.note,
        validate_first=(mode == "full" and not args.no_validate),
    )


# --------------------------------------------------------------------------
# doctor, calibrate
# --------------------------------------------------------------------------
def pinned_requirements():
    pins = {}
    path = common.REPO / "requirements.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, version = line.partition("#")[0].strip().partition("==")
            if name and version:
                pins[name] = version
    return pins


def cmd_doctor(args):
    worst = 0
    levels = ("ok  ", "WARN", "FAIL")

    def check(level, title, detail=""):
        nonlocal worst
        worst = max(worst, level)
        print(f"[{levels[level]}] {title}" + (f": {detail}" if detail else ""))

    machine = common.machine_info()
    machine_dir = common.RESULTS / machine["machine_id"]
    ram = machine.get("ram_bytes")
    print(f"Machine id: {machine['machine_id']}\n")
    check(0 if sys.version_info[:2] == (3, 12) else 1, "Python", f"{platform.python_version()} (organizers used 3.12)")

    mismatched, missing = [], []
    for name, wanted in pinned_requirements().items():
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if found != wanted:
            mismatched.append(f"{name} {found} (pinned {wanted})")
    critical = [name for name in missing if name in ("jax", "jaxlib", "numpy", "scipy", "matplotlib")]
    check(2 if critical else 1 if (missing or mismatched) else 0, "Pinned packages",
          "; ".join(filter(None, [f"missing: {', '.join(missing)}" if missing else "", ", ".join(mismatched)])) or "all match requirements.txt")
    try:
        check(0, "differometor installed", importlib.metadata.version("differometor"))
    except importlib.metadata.PackageNotFoundError:
        check(2, "differometor installed", "no: run `python -m pip install -e .` (benchmark.py needs it for its report)")

    registry = common.load_registry()
    baseline = common.registry_entry(registry, registry.get("baseline"))
    try:
        commit = common.git("rev-parse", "--verify", "--quiet", baseline["ref"] + "^{commit}")
        check(0, "Baseline version", f"{baseline['name']} -> {commit[:12]}")
    except (TypeError, OSError, subprocess.SubprocessError):
        check(2, "Baseline version", "registry baseline does not resolve to a commit in this clone")

    check(0, "CPU", f"{machine.get('cpu_model')}, {machine.get('logical_cpus')} logical CPUs")
    if ram:
        check(0 if ram >= 24 * common.GIB else 1, "RAM", f"{ram / common.GIB:.0f} GiB"
              + ("" if ram >= 24 * common.GIB else ": the unmodified baseline peaks near 14.4 GiB, so its one stored run will be memory-pressured (flagged, not hidden)"))
    memory = common.memory_state()
    if memory.get("available_bytes") is not None:
        low = bool(ram) and memory["available_bytes"] < 0.5 * ram
        check(1 if low else 0, "Free memory now", common.format_bytes(memory["available_bytes"]) + (": close other apps before a session" if low else ""))
    else:
        check(1, "Memory probe", "not available on this system; memory flags will be missing")
    power = common.power_state()
    if power:
        check(0 if power.get("on_ac_power") else 1, "Power", "on AC" if power.get("on_ac_power") else "on battery or unknown: plug in")
        check(1 if power.get("low_power_mode") else 0, "Low Power Mode", "ON: turn it off, it throttles the CPU" if power.get("low_power_mode") else "off")
    if common.IS_MAC:
        check(0 if common.mac_footprint_is_sane() else 1, "Peak-footprint probe",
              "working" if common.mac_footprint_is_sane() else "unavailable: falling back to peak resident size, which undercounts under memory pressure")
    limits, calibrated = common.load_limits(machine_dir)
    detail = f"pinned to CPU {limits['bench_cpu']} + official --cpus 1" if limits.get("strategy") == "affinity" else "thread-count limits incl. NPROC=1 (no CPU pinning on this OS)"
    check(0 if calibrated else 1, "One-CPU limit", detail + ("" if calibrated else "; not verified here yet: run `python bench/run.py calibrate`"))
    canary = common.run_canary(sys.executable, limits)
    check(0 if canary else 1, "Canary", f"{canary:.3f} s" if canary else "failed to run")
    check(0 if validate.REFERENCE_ARRAYS.exists() else 1, "Reference arrays",
          "present" if validate.REFERENCE_ARRAYS.exists() else "missing: run `python bench/validate.py v00-baseline --make-reference`")
    stored = validate.load_baseline_timing(machine_dir, registry)
    check(0 if stored else 1, "Stored baseline run",
          f"{stored['sample']['end_to_end']:.2f} s, measured {stored['measured_utc'][:10]}" if stored
          else "missing on this machine: run `python bench/validate.py v00-baseline --make-reference` once")
    sessions = report.load_sessions(machine_dir)
    noise, _ = report.noise_floor_from(sessions)
    check(0 if noise is not None else 1, "Noise floor", f"+/- {noise * 100:.1f} %" if noise is not None else "not measured: run `python bench/run.py aa`")
    print(f"\n{len(sessions)} session(s) on record for this machine.")
    sys.exit(1 if worst == 2 else 0)


PROBE = r'''
import json, os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "true"
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def measure(function, *arguments, repeats=3):
    jax.block_until_ready(function(*arguments))
    cpu_before, wall_before = os.times(), time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(function(*arguments))
    wall = time.perf_counter() - wall_before
    cpu_after = os.times()
    cpu = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    return {"wall": wall, "cpu": cpu, "cpu_per_wall": cpu / wall}

key_a, key_b, key_c = jax.random.split(jax.random.PRNGKey(0), 3)
square = jax.random.normal(key_a, (1200, 1200)) + 1j * jax.random.normal(key_b, (1200, 1200))
wide = jnp.tile(square, (4, 1))
stack = jax.random.normal(key_c, (24, 600, 600)) + 0j + 50 * jnp.eye(600)
right = jax.random.normal(key_a, (24, 600)) + 0j
print("PROBE " + json.dumps({
    "matmul": measure(jax.jit(lambda x: x @ x), square),
    "elementwise": measure(jax.jit(lambda x: jnp.sum(jnp.exp(1j * x.real) * jnp.conj(x) + x * x)), wide, repeats=20),
    "batched_solve": measure(jax.jit(lambda m, b: jax.vmap(jnp.linalg.solve)(m, b)), stack, right),
}))
'''


def cmd_calibrate(args):
    """Verify that the chosen limit really confines JAX to one core, and store it."""
    machine = common.machine_info()
    machine_dir = common.RESULTS / machine["machine_id"]
    limited = common.default_limits()
    results = {}
    for name, limits in (("unlimited", {"strategy": "none", "env": {}}), ("limited", limited)):
        print(f"Probing '{name}' (small JAX workload, well under 1 GiB)...", flush=True)
        done = subprocess.run([sys.executable, "-c", PROBE], env=common.child_env(limits), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, **common.spawn_options(limits))
        line = next((item for item in done.stdout.splitlines() if item.startswith("PROBE ")), None)
        if line is None:
            raise SystemExit(f"Probe failed under '{name}':\n{done.stderr.strip()[-800:]}")
        results[name] = json.loads(line[6:])
        print("  " + "   ".join(f"{key}: {value['cpu_per_wall']:.2f} CPU-s per wall-s ({value['wall']:.2f} s)" for key, value in results[name].items()))
    worst = max(value["cpu_per_wall"] for value in results["limited"].values())
    single_core = worst <= report.MULTI_CORE
    common.write_json(machine_dir / "machine.json", {
        "machine": machine, "limits": limited,
        "calibration": {"utc": common.utc_now().isoformat(), "python": platform.python_version(), "results": results,
                        "worst_cpu_per_wall": worst, "single_core": single_core},
    })
    print(f"\nLimit stored in {machine_dir / 'machine.json'}")
    if single_core:
        print(f"OK: the limited run used at most {worst:.2f} CPU-seconds per wall-second, i.e. one core.")
    else:
        print(f"WARNING: the limited run still used {worst:.2f} CPU-seconds per wall-second. Timings here will not\n"
              "represent the judged one-CPU run until that is fixed. Samples get the `multi-core` flag.")
        sys.exit(1)


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------
VERSION_STUB = """
## {name}

- **Date:** {date}
- **Commit:** `{sha}` (tag `{name}`)
- **Builds on:** {parent}
- **Summary:** {summary}

**What changed**

TODO

**Why it is faster**

TODO

**Why it is still correct**

TODO (paste the `validate.py` verdict and deviations)

**Result**

TODO (paste the row from `results/<machine>/SUMMARY.md` and name the session)

**Dead ends and notes**

TODO
"""


def cmd_add_version(args):
    registry = common.load_registry()
    if not VERSION_NAME.match(args.name):
        raise SystemExit("Version names look like v01-short-slug (two digits, lowercase letters, digits, dashes).")
    if common.registry_entry(registry, args.name):
        raise SystemExit(f"{args.name} is already registered.")
    if common.git("tag", "--list", args.name):
        raise SystemExit(f"A git tag {args.name} already exists.")
    if args.ref == "HEAD" and common.git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)bench"):
        raise SystemExit("Uncommitted changes outside bench/: commit the speed-up first, then add the version.")
    sha = common.git("rev-parse", "--verify", f"{args.ref}^{{commit}}")
    parent = common.latest_version(registry)
    common.git("tag", "-a", args.name, "-m", args.summary, sha)
    registry["versions"].append({"name": args.name, "ref": args.name, "parent": parent, "summary": args.summary, "enabled": True})
    if args.anchor:
        registry["anchor"] = args.name
    common.write_json(common.REGISTRY_PATH, registry)
    with open(common.BENCH_DIR / "VERSIONS.md", "a", encoding="utf-8") as handle:
        handle.write(VERSION_STUB.format(name=args.name, date=common.utc_now().date().isoformat(), sha=sha[:12], parent=parent or "-", summary=args.summary))
    print(f"Tagged {sha[:12]} as {args.name} (parent: {parent}).\n"
          "Next: fill in bench/VERSIONS.md, then\n"
          f"  python bench/validate.py {args.name}\n"
          "  python bench/run.py full\n"
          "and commit bench/versions.json, bench/VERSIONS.md and the new results. Push the tag with `git push <remote> --tags`.")


def cmd_official(args):
    """Complete, unmodified benchmark.py runs whose JSON and plot go into the submission."""
    registry = common.load_registry()
    machine_dir = common.RESULTS / common.machine_info()["machine_id"]
    limits, _ = common.load_limits(machine_dir)
    common.isolate_harness(limits)
    failed = False
    for spec in [item.strip() for item in args.versions.split(",") if item.strip()]:
        info = common.resolve_version(spec, registry)
        record = validate.validate_version(info, registry, machine_dir, limits, repeats=args.repeats, timeout=args.timeout,
                                           official_dir=machine_dir / "official")
        validate.print_record(record)
        failed |= record["verdict"] == "FAIL"
    print(f"\nOfficial reports and plots: {machine_dir / 'official'}")
    if limits.get("strategy") != "affinity":
        print("Note: this OS cannot pin a CPU, so these reports were made without `--cpus 1`. For the submission,\n"
              "repeat this command once on a Linux machine if you can.")
    sys.exit(1 if failed else 0)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def cmd_selftest(args):
    """Exercise the whole harness with a stub benchmark: no JAX, a few minutes."""
    stub = common.BENCH_DIR / "selftest" / "fake_benchmark.py"
    with tempfile.TemporaryDirectory(prefix="bench-selftest-") as temporary:
        root = Path(temporary)

        def make(name, **config):
            folder = root / "versions" / name
            folder.mkdir(parents=True)
            shutil.copy2(stub, folder / "benchmark.py")
            (folder / "fake_config.json").write_text(json.dumps(config), encoding="utf-8")
            return {"name": name, "ref": f"path:{folder}", "enabled": True}

        registry = {"baseline": "slow", "anchor": None, "tolerances": {}, "versions": [
            {**make("slow", scale=2.0), "parent": None},
            {**make("fast", scale=1.0), "parent": "slow"},
            {**make("broken", fail=True), "parent": "fast"},
            {**make("wrong", scale=1.0, perturb=1e-3), "parent": "fast"},
            # One small gradient entry is 50 % off; every large entry is exact.
            {**make("subtle", scale=1.0, scale_gradient_entry=[190, 1.5]), "parent": "fast"},
            # One mid-sized gradient entry is 0.05 % off (inside the 0.1 % limit), then 0.2 % off (outside).
            {**make("within", scale=1.0, scale_gradient_entry=[100, 1.0005]), "parent": "fast"},
            {**make("beyond", scale=1.0, scale_gradient_entry=[100, 1.002]), "parent": "fast"},
            # The largest gradient entry is 0.01 % off: inside the per-entry limit, outside the scale-relative one.
            {**make("large", scale=1.0, scale_gradient_entry=[1, 1.0001]), "parent": "fast"},
        ]}
        results = root / "results"
        options = dict(registry=registry, results_root=results, cooldown=0.0, timeout=60.0)

        print("== selftest 1/4: three versions, three shuffled rounds ==")
        session, path = run_session(mode="run", specs=["slow", "fast", "broken"], rounds=3, seed=7, **options)
        machine_dir = path.parent.parent
        stored = common.read_json(path)
        assert stored["complete"] and len(stored["samples"]) == 9, "every sample must be stored"
        assert all(sorted(order) == ["broken", "fast", "slow"] for order in stored["orders"]), "each round runs every version once"
        assert all(a != b for a, b in zip(stored["orders"], stored["orders"][1:])), "consecutive rounds must differ in order"
        for sample in stored["samples"]:
            if sample["label"] == "broken":
                assert sample["status"] == "failed" and any("simulated failure" in line for line in sample["log_tail"]), "failures are recorded with their log"
                continue
            assert sample["status"] == "ok", sample
            assert len(sample["phases"]) == 14 and sample["phases"][0]["seconds"] > 0, "14 phases with imports recovered"
            assert abs(sum(phase["seconds"] for phase in sample["phases"]) - sample["end_to_end"]) < 1e-5, "phases sum to the total"
            assert abs(sum(sample["groups"].values()) - sample["end_to_end"]) < 1e-5, "groups sum to the total"
            assert sample["wall_total"] - sample["wall_to_first_run"] < 0.9, "process must be stopped right after the first run"
            assert sample["canary"]["seconds"], "canary must run"
            if not common.IS_WINDOWS:  # resource usage of the killed child comes from wait4
                assert sample["cpu_seconds"] is not None and sample["peak_rss_bytes"] > 2**20, "CPU time and peak memory must be recorded"
            if common.IS_MAC or common.IS_LINUX:
                assert "swapouts" in sample["memory_before"] and "swapouts" in sample["memory_after"], "memory probe must work"
        rows = {row["label"]: row for row in stored["summary"]["rows"]}
        speedup = rows["fast"]["ratios"]["reference"]
        assert speedup["reference"] == "slow" and 1.6 < speedup["median"] < 2.4, f"expected about 2x, got {speedup['median']:.2f}x"
        assert not speedup["within_noise"], "a 2x difference must not be reported as noise"
        assert rows["broken"]["n_ok"] == 0 and rows["broken"]["flags"] == {"failed": 3}, "failed version is reported, not dropped"
        assert report.summarize_session(stored, None) == stored["summary"], "summary must be reproducible from the stored session"
        before = [(machine_dir / name).read_bytes() for name in ("SUMMARY.md", "history.csv")]
        report.write_outputs(machine_dir)
        assert before == [(machine_dir / name).read_bytes() for name in ("SUMMARY.md", "history.csv")], "reports must regenerate identically"

        print("\n== selftest 2/4: A/A session gives about 1.00x and sets the noise floor ==")
        session, path = run_session(mode="aa", specs=["fast", "fast"], rounds=3, seed=11, **options)
        twin = {row["label"]: row for row in session["summary"]["rows"]}["fast#B"]["ratios"]["reference"]
        assert twin["reference"] == "fast#A" and 0.8 < twin["median"] < 1.25, f"A/A should be about 1.00x, got {twin['median']:.2f}x"
        noise, source = report.noise_floor_from(report.load_sessions(machine_dir))
        assert noise is not None and source == session["id"], "noise floor comes from the A/A session"

        print("\n== selftest 3/4: correctness checks ==")
        try:
            import numpy  # noqa: F401
        except ImportError:
            print("numpy not installed: skipping the array-comparison part")
        else:
            limits, _ = common.load_limits(machine_dir)
            reference = root / "reference" / "baseline_arrays.npz"
            verdicts = {}
            for name, extra in (("slow", {"make_reference": True}), ("fast", {}), ("wrong", {}), ("wrong-allowed", {"allow_deviations": True}),
                                ("subtle", {}), ("within", {}), ("beyond", {}), ("large", {}), ("broken", {})):
                info = common.resolve_version(name.split("-")[0], registry)
                record = validate.validate_version(info, registry, machine_dir, limits, timeout=60.0, reference_arrays=reference, **extra)
                verdicts[name] = record["verdict"]
                if name == "subtle":
                    gradient = record["arrays"]["gradient"]
                    assert gradient["entries_outside"] == 1 and gradient["worst_index"] == 190, "the one wrong gradient entry must be named"
                    assert gradient["scale_rel"] < 1e-8, "this error is invisible relative to the largest entry: only a per-entry check finds it"
                if name == "within":
                    gradient = record["arrays"]["gradient"]
                    assert 4e-4 < gradient["max_elementwise_rel"] < 6e-4, "the entry must be 0.05 % off"
                    assert gradient["entries_outside"] == 0, "an entry 0.05 % off is inside the per-entry limit of 0.1 %"
                    assert gradient["scale_rel"] < 1e-6, "and a mid-sized entry is far inside the scale-relative limit"
                if name == "large":
                    gradient = record["arrays"]["gradient"]
                    assert gradient["entries_outside"] == 0, "0.01 % off is inside the per-entry limit"
                    assert 5e-5 < gradient["scale_rel"] < 2e-4, "this must be the largest entry, 0.01 % off"
                    assert any("scale-relative" in line for line in record["deviations"]), "only the scale-relative limit finds an error in the largest entry"
                if name == "beyond":
                    gradient = record["arrays"]["gradient"]
                    assert gradient["entries_outside"] == 1 and gradient["worst_index"] == 100, "an entry 0.2 % off is outside the per-entry limit of 0.1 %"
            expected = {"slow": "PASS", "fast": "PASS", "wrong": "FAIL", "wrong-allowed": "PASS_WITH_DEVIATIONS", "subtle": "FAIL",
                        "within": "PASS", "beyond": "FAIL", "large": "FAIL", "broken": "FAIL"}
            assert verdicts == expected, f"validation verdicts {verdicts}, expected {expected}"
            print("validation verdicts: " + ", ".join(f"{key}={value}" for key, value in verdicts.items()))

            print("\n== selftest 4/4: the baseline is not rerun; its stored run is the reference ==")
            assert validate.load_baseline_timing(machine_dir, registry)["sample"]["end_to_end"] > 0, "the baseline's one run must store its total"
            session, path = run_session(mode="run", specs=["fast"], rounds=3, seed=13, **options)
            assert [entry["label"] for entry in session["entries"]] == ["fast"], "only the named version is timed"
            against = session["summary"]["rows"][0]["ratios"]["reference"]
            assert against["reference"] == "slow" and against["stored_reference"], "speed-up must come from the stored baseline run"
            assert 1.6 < against["median"] < 2.4, f"expected about 2x against the stored baseline, got {against['median']:.2f}x"
            stored = common.read_json(path)
            assert report.summarize_session(stored, stored["summary"]["noise_floor"]) == stored["summary"], "summary must be reproducible from the stored session"
        if args.keep:
            shutil.copytree(machine_dir, args.keep, dirs_exist_ok=True)
            print(f"Kept selftest results in {args.keep}")
    print("\nSELFTEST PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check the machine and environment").set_defaults(run=cmd_doctor)
    commands.add_parser("calibrate", help="verify and store the one-CPU limit for this machine").set_defaults(run=cmd_calibrate)
    for mode, text in (("quick", "anchor + latest version + WORKTREE"), ("full", "all enabled versions except the baseline"),
                       ("aa", "one version against itself"), ("run", "custom list of versions")):
        sub = commands.add_parser(mode, help=text)
        sub.add_argument("--rounds", type=int, help=f"rounds (default {DEFAULT_ROUNDS[mode]})")
        sub.add_argument("--cooldown", type=float, help="seconds to rest after each sample (default 15 on macOS, 5 elsewhere)")
        sub.add_argument("--timeout", type=float, default=1800.0, help="seconds before a sample is killed")
        sub.add_argument("--seed", type=int, help="seed for the run order (default: random, stored in the session)")
        sub.add_argument("--note", default="", help="free text stored with the session")
        sub.add_argument("--no-gate", action="store_true", help="do not wait for a calm machine before samples")
        if mode == "full":
            sub.add_argument("--with-worktree", action="store_true", help="also time the uncommitted working tree")
            sub.add_argument("--no-validate", action="store_true", help="skip validating versions that have no cached verdict")
        if mode == "aa":
            sub.add_argument("--version", help="version to test against itself (default: anchor, else baseline)")
        if mode == "run":
            sub.add_argument("--versions", required=True, help="comma-separated: registry names, WORKTREE, git refs")
        sub.set_defaults(run=cmd_session)
    sub = commands.add_parser("add-version", help="tag a commit as a named version and register it")
    sub.add_argument("name")
    sub.add_argument("--summary", required=True)
    sub.add_argument("--ref", default="HEAD", help="commit to tag (default HEAD)")
    sub.add_argument("--anchor", action="store_true", help="make this the anchor: the reference for quick runs")
    sub.set_defaults(run=cmd_add_version)
    sub = commands.add_parser("official", help="complete official benchmark.py runs for the submission")
    sub.add_argument("--versions", required=True)
    sub.add_argument("--repeats", type=int, default=10)
    sub.add_argument("--timeout", type=float, default=14400.0)
    sub.set_defaults(run=cmd_official)
    sub = commands.add_parser("selftest", help="test the harness with a stub benchmark")
    sub.add_argument("--keep", type=Path, help="copy the selftest result files here for inspection")
    sub.set_defaults(run=cmd_selftest)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
