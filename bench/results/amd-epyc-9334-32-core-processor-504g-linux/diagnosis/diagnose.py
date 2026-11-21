"""Why is the current version sometimes several times slower on a busy Linux machine?

Runs the official benchmark many times, alternating settings, in three conditions (quiet machine, a
noisy neighbour churning memory, right after a large memory user). For every run it records the
benchmark's own phase times, our process's CPU time split into user and system, its page faults,
and the kernel's large-page and compaction counters before and after.
"""
import json, os, random, subprocess, sys, time
from pathlib import Path
TOP = Path(__file__).resolve().parent; REPO = TOP / "repo"; OUT = TOP / "results"; OUT.mkdir(exist_ok=True)
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LINUX = sys.platform.startswith("linux")
cpus = sorted(os.sched_getaffinity(0)) if LINUX else [0]
CONFIGS = {
    "default": ({}, None),
    "no-large-pages": ({"NUMPY_MADVISE_HUGEPAGE": "0"}, None),                       # NumPy does not ask for large pages
    "keep-memory": ({"MALLOC_MMAP_THRESHOLD_": "4294967296", "MALLOC_TRIM_THRESHOLD_": "17179869184", "MALLOC_TOP_PAD_": "1073741824"}, None),  # freed blocks are kept and reused
    "last-cpu": ({}, cpus[-1]),                                                       # pinned like the harness did (last CPU)
}
WATCH = ("thp_fault_alloc", "thp_fault_fallback", "compact_stall", "compact_fail", "compact_success", "pgfault", "allocstall_normal", "allocstall_movable")
def vmstat():
    if not LINUX: return {}
    return {k: int(v) for k, v in (line.split() for line in Path("/proc/vmstat").read_text().splitlines()) if k in WATCH}
def run(config, condition, index):
    env_extra, cpu = CONFIGS[config]
    env = dict(os.environ, **env_extra); env.pop("XLA_FLAGS", None)
    marks, report = OUT / "marks.json", OUT / "report.json"
    command = [sys.executable, "-u", str(TOP / "probe.py"), str(marks), "--repeats", "3", "--output", str(report)] + (["--cpus", "1"] if LINUX else [])
    if cpu is not None and LINUX: command = ["taskset", "-c", str(cpu)] + command
    before, load = vmstat(), os.getloadavg()[0]
    done = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    after = vmstat()
    record = {"condition": condition, "config": config, "index": index, "returncode": done.returncode, "load": load, "kernel": {k: after[k] - before[k] for k in before}}
    if done.returncode == 0:
        rows = json.loads(marks.read_text()); first = next(i for i, m in enumerate(rows) if m["label"].startswith("Complete first-run"))
        record["phases"] = {p["key"]: p["seconds"] for p in json.loads(report.read_text())["phases"]}
        record["total"] = sum(record["phases"].values())
        record["steps"] = [{"label": b["label"], **{k: b[k] - a[k] for k in a if k != "label"}} for a, b in zip(rows[:first + 1], rows[1:first + 1])]
        record["first_run_user"], record["first_run_system"] = rows[first]["user"], rows[first]["system"]
        record["first_run_minor_faults"] = rows[first]["minor_faults"]; record["first_run_involuntary_switches"] = rows[first]["involuntary_switches"]
    else:
        record["error"] = (done.stderr or done.stdout)[-400:]
    with open(OUT / "runs.jsonl", "a") as handle: handle.write(json.dumps(record) + "\n")
    return record
def block(condition, before_each=None):
    print(f"-- {condition}", flush=True)
    for index in range(ROUNDS):
        order = list(CONFIGS); random.shuffle(order)
        for config in order:
            if before_each: before_each()
            r = run(config, condition, index)
            p = r.get("phases", {})
            print(f"   {config:15} total {r.get('total', float('nan')):5.2f} build {p.get('build', 0):.2f} forward {p.get('forward', 0):.2f} gradient {p.get('gradient', 0):.2f} | user {r.get('first_run_user', 0):.2f} system {r.get('first_run_system', 0):.2f} | large pages given {r['kernel'].get('thp_fault_alloc', 0)}, refused {r['kernel'].get('thp_fault_fallback', 0)}, compaction stalls {r['kernel'].get('compact_stall', 0)}" + (f" | FAILED {r['error'][-120:]}" if "error" in r else ""), flush=True)
def big_user(gb=14):  # allocate, touch and free a large block, like the original does
    subprocess.run([sys.executable, "-c", f"import numpy as np; a = np.ones(int({gb} * 2**30 // 8)); a[::512] += 1; print(a[0])"], capture_output=True)
random.seed(1)
block("quiet")
if os.environ.get("DIAGNOSE_DRY"): raise SystemExit("dry run: stopped after the quiet block")
others = [str(c) for c in cpus[:-1][-8:]] or ["0"]
neighbours = [subprocess.Popen((["taskset", "-c", cpu] if LINUX else []) + [sys.executable, str(TOP / "churn.py"), "6"]) for cpu in others[:4]]
time.sleep(20)
try: block("noisy neighbour churning memory")
finally:
    for n in neighbours: n.kill()
time.sleep(5)
block("right after a large memory user", before_each=big_user)
# summary
import statistics
runs = [json.loads(line) for line in open(OUT / "runs.jsonl")]
print("\n== summary (complete first run, seconds)")
print(f"{'condition':34}{'setting':17}{'runs':>5}{'fastest':>9}{'median':>8}{'slowest':>9}{'slow runs':>11}{'system s':>10}{'stalls':>8}")
best = min(r["total"] for r in runs if "total" in r)
for condition in dict.fromkeys(r["condition"] for r in runs):
    for config in CONFIGS:
        g = [r for r in runs if r["condition"] == condition and r["config"] == config and "total" in r]
        if g:
            t = sorted(r["total"] for r in g)
            print(f"{condition:34}{config:17}{len(g):5d}{t[0]:9.2f}{statistics.median(t):8.2f}{t[-1]:9.2f}{sum(v > 1.3 * best for v in t):8d}/{len(t):<2d}{statistics.mean(r['first_run_system'] for r in g):10.2f}{sum(r['kernel'].get('compact_stall', 0) for r in g):8d}")
