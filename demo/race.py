#!/usr/bin/env python3
"""A live race for the demo: the original code against ours, on the same benchmark.

    python demo/race.py            # then open http://localhost:8811 and press Start
    python demo/race.py 9000       # the same on another port, if 8811 is taken

Both panels run the unchanged `benchmark.py`, each in a fresh process, started at the
same moment. The page follows the lines the benchmark prints and fills in each step
as it finishes. A run is stopped once it has printed its complete first-run total,
which is the number the challenge scores; the repeat runs and the validation that
follow are not part of the race.

The original is commit 02ec741, unpacked on first use into demo/.cache/original.
It needs about 14 GB of memory, so on a 16 GB laptop close other apps first.
Standard library only.
"""
import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(__file__).resolve().parent / ".cache" / "original"
ORIGINAL_COMMIT = "02ec741d8a1bf1a2278294ea74f9fc1d68f08f1f"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8811
# The step a printed line completes. Lines in between (tracing, the sensitivity readout) count towards the next one.
STEPS = [
    ("Start Python and the libraries", "Initialize CPU backend"),
    ("Build the detector setup", "Build optical setup"),
    ("Compile the simulation", "Forward: compile"),
    ("Simulate 101 frequencies", "Forward simulation"),
    ("Compile the gradient", "Gradient: compile"),
    ("Gradient for 193 design variables", "Reverse-mode gradient"),
]
FINISH = "Complete first-run total"
# One thread per numerical library, as the challenge scores one logical CPU (a Mac cannot pin a core).
ONE_THREAD = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NPROC")}
LOCK = threading.Lock()  # one race at a time


def original_checkout():
    if not (ORIGINAL / "benchmark.py").exists():
        ORIGINAL.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "archive", ORIGINAL_COMMIT], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(ORIGINAL)], input=archive, check=True)
    return ORIGINAL


def runner(side, folder, events, started):
    """Run one benchmark and report each finished step with the time since the common start."""
    command = [sys.executable, "-u", str(folder / "benchmark.py"), "--repeats", "3", "--output", str(folder / "benchmark-race.json")]
    if hasattr(os, "sched_setaffinity"):
        command += ["--cpus", "1"]
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "XLA_FLAGS")} | ONE_THREAD
    process = subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    step, last, total = 0, 0.0, None
    for line in process.stdout:
        now = time.perf_counter() - started
        if step < len(STEPS) and line.startswith(STEPS[step][1]):
            events.put({"side": side, "type": "step", "index": step, "seconds": now - last, "at": now})
            step, last = step + 1, now
        elif line.startswith(FINISH):
            total = float(line.split(":")[1].split()[0])
            break
    try:  # the scored part is over: stop the repeats and the validation
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    events.put({"side": side, "type": "done", "total": total, "at": time.perf_counter() - started})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.replace("__STEPS__", json.dumps([name for name, _ in STEPS])).replace("__MACHINE__", machine()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/race":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not LOCK.acquire(blocking=False):
                self.send({"type": "busy"})
                return
            try:
                events, started = queue.Queue(), time.perf_counter()
                self.send({"type": "start"})
                for side, folder in (("original", original_checkout()), ("ours", ROOT)):
                    threading.Thread(target=runner, args=(side, folder, events, started), daemon=True).start()
                finished = 0
                while finished < 2:
                    event = events.get()
                    finished += event["type"] == "done"
                    self.send(event)
                self.send({"type": "end"})
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page was closed; the runners stop on their own at the finish line
            finally:
                LOCK.release()
        else:
            self.send_error(404)

    def send(self, event):
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


def machine():
    name = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or name
    elif Path("/proc/cpuinfo").exists():
        name = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), name)
    limit = "pinned to one logical CPU" if hasattr(os, "sched_setaffinity") else "numerical libraries limited to one thread"
    return f"{name}, {limit}"


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faster Physics: the race</title>
<style>
  :root { --paper: #f6f4ef; --ink: #1c1b19; --soft: #6f6b63; --line: #dcd8ce; --card: #fffdf8; --ours: #1f7a4d; --theirs: #9a3b2e; --track: #ece8dd; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 18px/1.45 "Avenir Next", "Segoe UI", system-ui, sans-serif; }
  main { max-width: 1280px; margin: 0 auto; padding: 20px 28px 14px; }
  header { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
  h1 { font-size: 28px; line-height: 1.15; margin: 0; letter-spacing: -0.01em; }
  .verdict { color: var(--soft); font-size: 17px; line-height: 1.3; min-height: 46px; margin: 6px 0 12px; display: flex; align-items: center; }
  .verdict.live { color: var(--ink); font-size: 24px; font-weight: 600; } .verdict b { color: var(--ours); font-size: 34px; margin-right: 0.25em; }
  .lanes { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .lane { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 22px 10px; }
  .lane h2 { font-size: 19px; margin: 0; font-weight: 600; display: inline; }
  .lane .what { color: var(--soft); font-size: 14px; margin: 0 0 8px 10px; display: inline; }
  .clock { margin-top: 8px; font: 600 60px/1 ui-monospace, "SF Mono", Menlo, monospace; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
  .clock small { font-size: 24px; font-weight: 500; color: var(--soft); margin-left: 6px; }
  .original .clock.done { color: var(--theirs); } .ours .clock.done { color: var(--ours); }
  .state { height: 20px; font-size: 14px; color: var(--soft); margin: 2px 0 8px; }
  .row { display: grid; grid-template-columns: 1fr 86px; gap: 2px 12px; align-items: baseline; padding: 5px 0 7px; border-top: 1px solid var(--line); }
  .row .name { font-size: 16px; color: var(--soft); } .row.running .name, .row.finished .name { color: var(--ink); }
  .row .secs { text-align: right; font: 15px ui-monospace, "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; color: var(--soft); }
  .row.finished .secs { color: var(--ink); }
  .bar { grid-column: 1 / -1; height: 7px; background: var(--track); border-radius: 5px; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; border-radius: 5px; }
  .original .bar i { background: var(--theirs); } .ours .bar i { background: var(--ours); }
  .row.running .bar i { opacity: 0.55; }
  button { font: inherit; font-weight: 600; white-space: nowrap; padding: 11px 26px; border-radius: 8px; border: 0; background: var(--ink); color: var(--paper); cursor: pointer; }
  button:disabled { opacity: 0.35; cursor: default; } button:focus-visible { outline: 3px solid var(--ours); outline-offset: 3px; }
  footer { margin-top: 10px; color: var(--soft); font-size: 13px; line-height: 1.35; }
  @media (max-width: 860px) { .lanes { grid-template-columns: 1fr; } .clock { font-size: 48px; } h1 { font-size: 22px; } }
  @media (prefers-reduced-motion: no-preference) { .bar i { transition: width 80ms linear; } }
</style></head><body><main>
  <header><h1>One complete evaluation of a gravitational-wave detector design</h1><button id="go">Start the race</button></header>
  <div class="verdict" id="verdict" aria-live="polite">Both sides run the same official benchmark from a cold start: load, build, compile, simulate 101 frequencies, take the gradient for 193 design variables.</div>
  <div class="lanes">
    <section class="lane original" id="original"><h2>Original Differometor</h2><p class="what">The code as the organizers supplied it</p></section>
    <section class="lane ours" id="ours"><h2>Our version</h2><p class="what">Same results and gradients, twelve changes</p></section>
  </div>
  <footer>This machine: __MACHINE__. The clock shows time since both were started; the number kept at the end is the complete first-run total the benchmark itself reports. Official comparison on one pinned Linux core: 35.62 s against 0.715 s.</footer>
</main>
<script>
const STEPS = __STEPS__, lanes = {}, button = document.getElementById("go"), verdict = document.getElementById("verdict");
for (const side of ["original", "ours"]) {
  const el = document.getElementById(side);
  el.insertAdjacentHTML("beforeend", `<div class="clock" data-clock>0.00<small>s</small></div><div class="state" data-state>ready</div>` +
    STEPS.map(name => `<div class="row"><span class="name">${name}</span><span class="secs"></span><div class="bar"><i></i></div></div>`).join(""));
  lanes[side] = { el, rows: [...el.querySelectorAll(".row")], clock: el.querySelector("[data-clock]"), state: el.querySelector("[data-state]") };
}
let t0 = 0, scale = 6, timer = null;
const seconds = () => (performance.now() - t0) / 1000;
const show = (lane, value) => lane.clock.firstChild.textContent = value.toFixed(2);
function reset() {
  for (const lane of Object.values(lanes)) {
    Object.assign(lane, { step: 0, stepStart: 0, total: null, at: null });
    lane.clock.classList.remove("done"); show(lane, 0); lane.state.textContent = "running";
    lane.rows.forEach((row, i) => { row.className = "row" + (i ? "" : " running"); row.querySelector(".secs").textContent = ""; row.querySelector("i").style.width = "0"; });
  }
  verdict.className = "verdict live"; verdict.textContent = "Running\u2026"; scale = 6;
}
function tick() {
  const now = seconds();
  for (const lane of Object.values(lanes)) {
    if (lane.total !== null) continue;
    show(lane, now);
    const row = lane.rows[lane.step];
    if (row) {
      const running = now - lane.stepStart;
      if (running > scale * 0.9) rescale(running / 0.7);
      row.querySelector("i").style.width = Math.min(100, 100 * running / scale) + "%";
      row.querySelector(".secs").textContent = running.toFixed(1) + " s";
    }
  }
  const [o, u] = [lanes.original, lanes.ours];
  if (u.total !== null && o.total === null) verdict.innerHTML = `<span>Ours is done. The original has now run <b>${(now / u.at).toFixed(0)}×</b>as long.</span>`;
}
function rescale(next) {
  scale = next;
  for (const lane of Object.values(lanes)) lane.rows.forEach(row => { if (row.dataset.secs) row.querySelector("i").style.width = Math.min(100, 100 * row.dataset.secs / scale) + "%"; });
}
function onEvent(e) {
  const lane = lanes[e.side];
  if (e.type === "start") { reset(); t0 = performance.now(); timer = setInterval(tick, 50); }
  else if (e.type === "step") {
    const row = lane.rows[e.index]; row.className = "row finished"; row.dataset.secs = e.seconds;
    row.querySelector(".secs").textContent = e.seconds.toFixed(2) + " s";
    row.querySelector("i").style.width = Math.max(0.6, Math.min(100, 100 * e.seconds / scale)) + "%";
    lane.step = e.index + 1; lane.stepStart = e.at;
    if (lane.rows[lane.step]) lane.rows[lane.step].className = "row running";
  } else if (e.type === "done") {
    lane.total = e.total; lane.at = e.at; lane.clock.classList.add("done");
    if (lane.rows[lane.step]) lane.rows[lane.step].className = "row";
    if (e.total === null) { lane.state.textContent = "did not finish"; return; }
    show(lane, e.total); lane.state.textContent = "complete first run, as reported by the benchmark";
  } else if (e.type === "end") {
    clearInterval(timer); button.disabled = false; button.textContent = "Run again";
    const [o, u] = [lanes.original.total, lanes.ours.total];
    verdict.innerHTML = o && u ? `<span><b>${(o / u).toFixed(0)}× faster</b>on this machine: ${u.toFixed(2)} s against ${o.toFixed(2)} s.</span>` : "One side did not finish. See the terminal.";
  } else if (e.type === "busy") { verdict.textContent = "A race is already running in another window."; button.disabled = false; }
}
button.addEventListener("click", () => {
  button.disabled = true; button.textContent = "Running";
  const stream = new EventSource("/race");
  stream.onmessage = m => { const e = JSON.parse(m.data); onEvent(e); if (e.type === "end" || e.type === "busy") stream.close(); };
  stream.onerror = () => { stream.close(); clearInterval(timer); button.disabled = false; button.textContent = "Run again"; };
});
</script></body></html>
"""

if __name__ == "__main__":
    original_checkout()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is taken. Start it on another one, for example: python demo/race.py {PORT + 1}")
    print(f"Open http://127.0.0.1:{PORT} and press Start. Stop this with Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
