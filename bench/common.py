#!/usr/bin/env python3
"""Shared machinery for the benchmark harness: paths, versions, snapshots,
machine probes and the process runner. Standard library only.

Nothing in here measures time itself. Every timing comes from the output of the
version's own ``benchmark.py``; this module only launches it, reads what it
prints and records the surrounding machine state.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
CACHE = BENCH_DIR / ".cache"
RESULTS = BENCH_DIR / "results"
REFERENCE = BENCH_DIR / "reference"
REGISTRY_PATH = BENCH_DIR / "versions.json"
WORKTREE = "WORKTREE"
SCHEMA = 1

SYSTEM = platform.system()
IS_LINUX = SYSTEM == "Linux"
IS_MAC = SYSTEM == "Darwin"
IS_WINDOWS = SYSTEM == "Windows"
HAS_AFFINITY = hasattr(os, "sched_setaffinity")
GIB = 2**30


# --------------------------------------------------------------------------
# benchmark.py output protocol
# --------------------------------------------------------------------------
# benchmark.py prints one "<label>: <seconds> s" line per timed phase (six
# decimals) and then the scored total. The import phase is timed but not
# printed, so it is recovered as total minus the printed phases.
TOTAL_RE = re.compile(r"^Complete first-run total \(excluding validation\): (\d+\.\d+) s\s*$")
PHASE_RE = re.compile(r"^(.+?): (\d+\.\d{6}) s\s*$")
PASS_RE = re.compile(r"^PASS\b")

LABEL_TO_KEY = {
    "Initialize CPU backend": "initialize",
    "Read and parse public inputs": "load",
    "Build optical setup and prepare inputs": "build",
    "Forward: trace / lower": "forward_lower",
    "Forward: compile": "forward_compile",
    "Forward simulation + AD residuals": "forward",
    "Sensitivity: trace / lower": "sensitivity_lower",
    "Sensitivity: compile": "sensitivity_compile",
    "Sensitivity, powers and scalar loss": "sensitivity",
    "Gradient: trace / lower": "gradient_lower",
    "Gradient: compile": "gradient_compile",
    "Reverse-mode gradient": "gradient",
    "Transfer results to host": "host_transfer",
}

# Same five groups as plot_timings.GROUPS, so our numbers line up with the
# official phase plot. Phases a modified benchmark.py invents land in "Other".
GROUPS = (
    ("Setup", ("imports", "initialize", "load", "build")),
    ("Compilation", (
        "forward_lower", "forward_compile", "sensitivity_lower",
        "sensitivity_compile", "gradient_lower", "gradient_compile",
    )),
    ("Forward simulation", ("forward",)),
    ("Sensitivity", ("sensitivity", "host_transfer")),
    ("Gradient", ("gradient",)),
)
GROUP_NAMES = tuple(name for name, _ in GROUPS) + ("Other",)
_KEY_TO_GROUP = {key: name for name, keys in GROUPS for key in keys}


def parse_first_run(lines):
    """Return ``(total_seconds, phases)`` once the total line has been printed, else None."""
    phases = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        match = TOTAL_RE.match(line)
        if match:
            total = float(match.group(1))
            imports = total - sum(phase["seconds"] for phase in phases)
            phases.insert(0, {"key": "imports", "label": "Python / library imports", "seconds": imports})
            return total, phases
        match = PHASE_RE.match(line)
        if match:
            label = match.group(1)
            phases.append({"key": LABEL_TO_KEY.get(label, label), "label": label, "seconds": float(match.group(2))})
    return None


def group_phases(phases):
    """Sum phases into the five official groups (plus "Other" for unknown ones)."""
    totals = {name: 0.0 for name in GROUP_NAMES}
    for phase in phases:
        totals[_KEY_TO_GROUP.get(phase["key"], "Other")] += phase["seconds"]
    return totals


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc)


def utc_stamp(moment=None):
    return (moment or utc_now()).strftime("%Y-%m-%dT%H-%M-%SZ")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write atomically so an interrupted session never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalized_sha256(data):
    """Hash with CRLF folded to LF, so a Windows checkout hashes like a Unix one."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._@-]+", "-", text).strip("-") or "unnamed"


def format_bytes(count):
    return "-" if not count else f"{count / GIB:.2f} GiB"


def _capture(command, timeout=15):
    try:
        done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        return done.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------
# git, registry, versions
# --------------------------------------------------------------------------
def git(*args, binary=False, cwd=REPO):
    done = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return done.stdout if binary else done.stdout.decode("utf-8", "replace").strip()


def git_head():
    try:
        return git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None


def load_registry(path=REGISTRY_PATH):
    registry = read_json(path)
    registry.setdefault("versions", [])
    registry.setdefault("baseline", registry["versions"][0]["name"] if registry["versions"] else None)
    registry.setdefault("anchor", None)
    registry.setdefault("tolerances", {})
    return registry


def registry_entry(registry, name):
    return next((entry for entry in registry["versions"] if entry["name"] == name), None)


def latest_version(registry):
    enabled = [entry for entry in registry["versions"] if entry.get("enabled", True)]
    return enabled[-1]["name"] if enabled else None


def official_benchmark_sha256(registry):
    """Hash of benchmark.py as shipped in the baseline version (None if unknown)."""
    entry = registry_entry(registry, registry.get("baseline"))
    if not entry:
        return None
    try:
        if entry["ref"].startswith("path:"):
            return normalized_sha256((Path(entry["ref"][5:]) / "benchmark.py").read_bytes())
        return normalized_sha256(git("show", f"{entry['ref']}:benchmark.py", binary=True))
    except (OSError, subprocess.SubprocessError):
        return None


def _extract_commit(sha):
    destination = CACHE / "snapshots" / sha[:12]
    if (destination / ".complete").exists():
        return destination
    temporary = destination.with_name(f"{destination.name}.tmp{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    # autocrlf off: snapshots must be byte-exact, benchmark.py checks the fixtures' hashes.
    with tarfile.open(fileobj=io.BytesIO(git("-c", "core.autocrlf=false", "archive", "--format=tar", sha, binary=True))) as archive:
        try:
            archive.extractall(temporary, filter="data")
        except TypeError:  # Python without extraction filters
            archive.extractall(temporary)
    (temporary / ".complete").write_text(sha, encoding="utf-8")
    shutil.rmtree(destination, ignore_errors=True)
    temporary.rename(destination)
    return destination


def _worktree_files():
    listed = git("ls-files", "-co", "--exclude-standard", "-z", binary=True).split(b"\0")
    names = sorted(name.decode("utf-8") for name in listed if name)
    # The harness and its results are not part of what is being measured.
    return [name for name in names if not name.startswith("bench/") and (REPO / name).is_file()]


def _snapshot_worktree():
    files = _worktree_files()
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((REPO / name).read_bytes()).digest())
    identity = digest.hexdigest()
    destination = CACHE / "snapshots" / f"wt-{identity[:12]}"
    if not (destination / ".complete").exists():
        temporary = destination.with_name(f"{destination.name}.tmp{os.getpid()}")
        shutil.rmtree(temporary, ignore_errors=True)
        for name in files:
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        (temporary / ".complete").write_text(identity, encoding="utf-8")
        shutil.rmtree(destination, ignore_errors=True)
        temporary.rename(destination)
    return destination, identity


def resolve_version(spec, registry):
    """Turn a version spec into a runnable, immutable snapshot.

    A spec is a registry name, ``WORKTREE`` (the uncommitted working tree),
    ``path:<dir>`` (any directory holding a benchmark.py) or any git ref.
    A registry entry's ``ref`` may itself be a git ref or a ``path:<dir>``.
    """
    entry = registry_entry(registry, spec)
    target = entry["ref"] if entry else spec
    info = {
        "spec": spec, "name": spec, "ref": None, "dirty": False,
        "parent": entry.get("parent") if entry else None,
        "python": (entry or {}).get("python") or sys.executable,
        "env": dict((entry or {}).get("env") or {}),
    }
    if target == WORKTREE:
        snapshot, identity = _snapshot_worktree()
        status = git("status", "--porcelain", "--", ".", ":(exclude)bench")
        info.update(kind="worktree", id=identity, ref=git_head(), dirty=bool(status), parent=latest_version(registry))
    elif target.startswith("path:"):
        snapshot = Path(target[5:]).resolve()
        digest = hashlib.sha256()
        for item in sorted(snapshot.iterdir()):
            if item.is_file():
                digest.update(item.name.encode("utf-8") + b"\0" + item.read_bytes())
        info.update(kind="path", id=digest.hexdigest(), name=spec if entry else snapshot.name)
    else:
        try:
            sha = git("rev-parse", "--verify", "--quiet", f"{target}^{{commit}}")
        except subprocess.CalledProcessError:
            raise SystemExit(f"Unknown version or git ref: {spec!r} (ref {target!r} does not resolve)") from None
        snapshot = _extract_commit(sha)
        info.update(kind="commit", id=sha, ref=target)
    script = Path(snapshot) / "benchmark.py"
    if not script.is_file():
        raise SystemExit(f"{spec}: no benchmark.py in {snapshot}")
    info["snapshot"] = str(snapshot)
    info["benchmark_sha256"] = normalized_sha256(script.read_bytes())
    return info


# --------------------------------------------------------------------------
# machine probes (all fail-soft: a probe that cannot run returns None/{})
# --------------------------------------------------------------------------
def _sysctl(name):
    return _capture(["sysctl", "-n", name])


def _windows_memory():
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual",
            )
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return status.ullTotalPhys, status.ullAvailPhys


def _int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def machine_info():
    info = {
        "system": SYSTEM, "arch": platform.machine(), "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "cpu_model": None, "ram_bytes": None,
    }
    try:
        if IS_MAC:
            info["cpu_model"] = _sysctl("machdep.cpu.brand_string") or None
            info["model"] = _sysctl("hw.model") or None
            info["ram_bytes"] = _int(_sysctl("hw.memsize"))
            info["performance_cpus"] = _int(_sysctl("hw.perflevel0.logicalcpu"))
            info["efficiency_cpus"] = _int(_sysctl("hw.perflevel1.logicalcpu"))
        elif IS_LINUX:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    info["ram_bytes"] = int(line.split()[1]) * 1024
        elif IS_WINDOWS:
            info["cpu_model"] = platform.processor() or None
            info["ram_bytes"] = _windows_memory()[0]
    except (OSError, ValueError):
        pass
    info["cpu_model"] = info["cpu_model"] or platform.processor() or platform.machine()
    ram = f"{round(info['ram_bytes'] / GIB)}g" if info["ram_bytes"] else "ram-unknown"
    info["machine_id"] = safe_name(f"{info['cpu_model']}-{ram}-{SYSTEM}".lower().replace("(r)", "").replace("(tm)", ""))
    return info


def memory_state():
    """System memory snapshot. Counters are cumulative, so two snapshots give a delta."""
    state = {}
    try:
        if IS_MAC:
            text = _capture(["vm_stat"])
            page = int(re.search(r"page size of (\d+) bytes", text).group(1))
            pages = {name.strip('"').lower(): int(value) for name, value in re.findall(r'^("?[^":\n]+"?):\s+(\d+)\.?\s*$', text, re.M)}
            free = sum(pages.get(key, 0) for key in ("pages free", "pages inactive", "pages speculative", "pages purgeable"))
            state = {
                "available_bytes": free * page,
                "compressor_bytes": pages.get("pages occupied by compressor", 0) * page,
                "compressed_bytes_total": pages.get("compressions", 0) * page,
                "swapouts": pages.get("swapouts", 0),
                "swapins": pages.get("swapins", 0),
            }
            used = re.search(r"used = ([\d.]+)([KMG])", _sysctl("vm.swapusage"))
            if used:
                state["swap_used_bytes"] = int(float(used.group(1)) * {"K": 2**10, "M": 2**20, "G": 2**30}[used.group(2)])
        elif IS_LINUX:
            meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[1:2]}
            vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
            state = {
                "available_bytes": meminfo.get("MemAvailable"),
                "swap_used_bytes": meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0),
                "swapouts": int(vmstat.get("pswpout", 0)),
                "swapins": int(vmstat.get("pswpin", 0)),
            }
        elif IS_WINDOWS:
            state = {"available_bytes": _windows_memory()[1]}
    except (OSError, ValueError, AttributeError, KeyError):
        return {}
    return state


def load_average():
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def power_state():
    """macOS only: AC power and Low Power Mode both change single-core speed."""
    if not IS_MAC:
        return {}
    battery = _capture(["pmset", "-g", "batt"])
    settings = _capture(["pmset", "-g"])
    low = re.search(r"^\s*lowpowermode\s+(\d+)", settings, re.M)
    return {
        "on_ac_power": ("AC Power" in battery) if battery else None,
        "low_power_mode": bool(int(low.group(1))) if low else None,
    }


_RUSAGE_V4_FIELDS = (
    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
    "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
    "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
    "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
    "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
    "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
    "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
    "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time",
)


def mac_footprint(pid):
    """Peak "memory footprint" of a live process on macOS (rusage_info_v4).

    Unlike resident size this also counts compressed pages, so it shows what a
    run really needs even when the machine is short of RAM. Returns None when
    the probe is unavailable; `run.py doctor` sanity-checks it.
    """
    if not IS_MAC:
        return None
    try:
        import ctypes

        class RusageInfoV4(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _RUSAGE_V4_FIELDS]

        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib")
        except OSError:
            library = ctypes.CDLL(None)
        buffer = RusageInfoV4()
        if library.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(4), ctypes.byref(buffer)) != 0:
            return None
        return {
            "peak_footprint_bytes": int(buffer.lifetime_max_phys_footprint),
            "footprint_bytes": int(buffer.phys_footprint),
            "resident_bytes": int(buffer.resident_size),
        }
    except (OSError, AttributeError, ValueError):
        return None


def mac_footprint_is_sane():
    probe = mac_footprint(os.getpid())
    if not probe:
        return False
    return 2**20 < probe["footprint_bytes"] < 2**40 and probe["peak_footprint_bytes"] >= probe["footprint_bytes"] * 0.99 and 2**20 < probe["resident_bytes"] < 2**40


_FOOTPRINT_OK = None


def live_peak_memory(pid):
    """Best available peak-memory reading of a still-running child."""
    global _FOOTPRINT_OK
    if IS_MAC:
        if _FOOTPRINT_OK is None:
            _FOOTPRINT_OK = mac_footprint_is_sane()
        probe = mac_footprint(pid) if _FOOTPRINT_OK else None
        return probe["peak_footprint_bytes"] if probe else None
    if IS_LINUX:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------
# CPU limit: emulate the judged "one logical CPU"
# --------------------------------------------------------------------------
# Where affinity is unavailable (macOS), these emulate one CPU. Measured with
# jaxlib 0.9.0: the BLAS variables alone leave XLA's own thread pool on all
# cores (matmul ~9 CPU-seconds per wall-second), and the often-quoted
# XLA_FLAGS switches change nothing. NPROC=1 sizes that pool to one thread;
# together they give ~1.0 for matmul, elementwise work and batched solves.
# `run.py calibrate` re-checks this on the machine that produces the data.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NPROC": "1",
}
# Removed from the child environment unless a limit or version sets them again.
SCRUBBED_ENV = ("PYTHONPATH", "XLA_FLAGS", "NPROC", "JAX_COMPILATION_CACHE_DIR")


def _thread_siblings(cpu):
    try:
        text = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
    except OSError:
        return {cpu}
    siblings = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        siblings.update(range(int(low), int(high or low) + 1))
    return siblings


def default_limits():
    """Linux: real affinity plus the official --cpus 1. Elsewhere: thread-count limits."""
    if HAS_AFFINITY:
        allowed = sorted(os.sched_getaffinity(0))
        return {"strategy": "affinity", "bench_cpu": allowed[-1], "env": {}}
    return {"strategy": "thread-env", "env": dict(THREAD_ENV)}


def load_limits(machine_dir):
    """Return ``(limits, calibrated)`` from machine.json, falling back to defaults."""
    path = Path(machine_dir) / "machine.json"
    if path.exists():
        stored = read_json(path).get("limits")
        if stored:
            return stored, True
    return default_limits(), False


def isolate_harness(limits):
    """Linux: keep the harness itself off the physical core used for measuring."""
    if limits.get("strategy") != "affinity":
        return
    rest = set(os.sched_getaffinity(0)) - _thread_siblings(limits["bench_cpu"])
    if rest:
        os.sched_setaffinity(0, rest)


def child_env(limits, extra=None):
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED_ENV}
    env.update(limits.get("env") or {})
    env.update(extra or {})
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn_options(limits):
    """Popen keyword arguments: own process group, and the CPU pin where affinity exists."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    options = {"start_new_session": True}
    if limits.get("strategy") == "affinity":
        cpu = limits["bench_cpu"]
        options["preexec_fn"] = lambda: os.sched_setaffinity(0, {cpu})
    return options


def _kill(process):
    try:
        if IS_WINDOWS:
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _reap(process, deadline):
    """Wait for the child and return its resource usage (None on Windows)."""
    if IS_WINDOWS:
        try:
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return None
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return usage
        if time.monotonic() > deadline:
            _kill(process)
            deadline = float("inf")
        time.sleep(0.02)


CANARY = """
import hashlib, time
block = bytes(range(256)) * 4096
start = time.perf_counter()
digest = hashlib.sha256()
for _ in range(96):
    digest.update(block)
total = 0
for index in range(3_000_000):
    total += (index * index) % 7
print(time.perf_counter() - start)
"""


def run_canary(python, limits):
    """Fixed single-thread workload (~0.5 s). Tracks machine speed drift; never rescales timings."""
    try:
        done = subprocess.run(
            [python, "-c", CANARY], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=child_env(limits), timeout=120, **spawn_options(limits),
        )
        return float(done.stdout.decode().strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_benchmark(snapshot, *, python, limits, log_path, output_json, timeout,
                  stop_after_first_run=True, repeats=3, save_arrays=None, extra_env=None):
    """Run one fresh ``benchmark.py`` process and collect what it reports.

    With ``stop_after_first_run`` the process is killed as soon as it has
    printed the scored first-run total: warm repeats and validation are not
    scored, so skipping them buys roughly three times more samples per hour.
    """
    snapshot = Path(snapshot)
    log_path, output_json = Path(log_path), Path(output_json)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [str(python), "-u", str(snapshot / "benchmark.py"), "--repeats", str(repeats), "--output", str(output_json)]
    if limits.get("strategy") == "affinity":
        command += ["--cpus", "1"]
    if save_arrays:
        command += ["--save-arrays", str(save_arrays)]

    lines, first, passed, status = [], None, False, "failed"
    wall_first = peak_live = None
    start = time.monotonic()
    deadline = start + timeout
    process = subprocess.Popen(
        command, cwd=output_json.parent, env=child_env(limits, extra_env),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1, **spawn_options(limits),
    )
    feed = queue.Queue()

    def pump():
        for line in process.stdout:
            feed.put(line)
        feed.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    stopped_early = False
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                try:
                    line = feed.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
                if line is None:
                    break
                lines.append(line)
                log.write(line)
                if first is None and TOTAL_RE.match(line):
                    wall_first = time.monotonic() - start
                    first = parse_first_run(lines)
                    peak_live = live_peak_memory(process.pid)
                    if stop_after_first_run:
                        stopped_early = True
                        break
                elif PASS_RE.match(line):
                    passed = True
            if stopped_early or status == "timeout":
                _kill(process)
            usage = _reap(process, deadline)
        except BaseException:  # Ctrl-C included: the child has its own process group
            _kill(process)
            raise
        wall_total = time.monotonic() - start
        reader.join(timeout=5)
        process.stdout.close()
        if status == "timeout":
            log.write(f"\n[harness] killed after {timeout:.0f} s timeout\n")

    if status != "timeout":
        finished = first is not None and (stopped_early or (process.returncode == 0 and passed))
        status = "ok" if finished else "failed"
    result = {
        "status": status, "returncode": None if stopped_early else process.returncode, "passed": passed,
        "wall_to_first_run": wall_first, "wall_total": wall_total,
        "cpu_seconds": None, "cpu_per_wall": None, "peak_rss_bytes": None, "peak_footprint_bytes": peak_live,
    }
    if usage is not None:
        result["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        result["peak_rss_bytes"] = usage.ru_maxrss * (1 if IS_MAC else 1024)
        measured_wall = wall_first if stopped_early else wall_total
        if measured_wall:
            result["cpu_per_wall"] = result["cpu_seconds"] / measured_wall
    if IS_LINUX and peak_live:  # VmHWM is peak resident size, not a footprint
        result["peak_rss_bytes"], result["peak_footprint_bytes"] = peak_live, None
    if first is not None:
        total, phases = first
        result.update(end_to_end=total, phases=phases, groups=group_phases(phases), untimed_overhead=wall_first - total)
    if status != "ok":
        result["log_tail"] = [line.rstrip("\r\n") for line in lines[-25:]]
    return result
