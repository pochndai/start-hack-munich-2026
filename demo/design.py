#!/usr/bin/env python3
"""A detector being designed live, for the demo.

    python demo/design.py          # then open http://127.0.0.1:8812 and press Start
    python demo/design.py 9000     # the same on another port

The page starts from a random detector (the supplied 3x3 layout with every design
variable drawn at random inside the optimizer's ranges) and improves it step by step
with the gradient, the way examples/uifo_optimization.py does: Adam with a learning
rate of 0.1 and gradient clipping, on mean(log10(sensitivity / published design)).
Every step is one complete simulation of 101 frequencies plus the gradient for 193
design variables, and the sensitivity curve is redrawn after each one.

This is a demonstration of what the speed is for, not a design tool: the loss has no
penalty on optical power, and nothing here is part of the scored benchmark.

For comparison the page says how many steps the original code would have managed in
the same time. That uses the original's measured time per evaluation on this machine
(the repeat-run median in bench/results/<machine>/validation/v00-baseline@*.json); if
that file does not exist, the comparison is left out.
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8812
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
# The ranges examples/uifo_optimization.py lets the optimizer move in.
BOUNDS = {"db": [0, 10], "angle": [-360, 360], "power": [0, 200], "tuning": [-360, 360], "mass": [0.01, 200], "length": [0.1, 4000], "reflectivity": [0, 1]}
MAX_STEPS = 4000
LOCK = threading.Lock()  # one optimisation at a time


def prepare():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import optax
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces
    from differometor.utils import inverse_sigmoid_bounding, sigmoid_bounding

    data = ROOT / "examples" / "data"
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    public = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    kind = lambda pair: pair[1] if isinstance(pair[0], str) else pair[0][1]
    bounds = jnp.asarray(np.array([[BOUNDS[kind(pair)][0] for pair in pairs], [BOUNDS[kind(pair)][1] for pair in pairs]], dtype=float))
    frequencies, published = jnp.asarray(public["frequencies"]), jnp.asarray(public["sensitivities"])
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)

    def objective(unbounded):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=sigmoid_bounding(unbounded, bounds)))
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity / published)), sensitivity

    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))

    def start(seed):
        values = np.random.default_rng(seed).uniform(np.asarray(bounds[0]), np.asarray(bounds[1]))
        return jnp.asarray(inverse_sigmoid_bounding(values, bounds))

    evaluate(start(0))  # compile now, so the first press of Start is not a wait
    return {"np": np, "optax": optax, "evaluate": evaluate, "start": start, "variables": len(pairs),
            "frequencies": public["frequencies"], "published": [round(float(v), 4) for v in np.log10(public["sensitivities"])]}


def original_seconds_per_step():
    """The original's measured time for one evaluation on this machine, or None."""
    try:
        import common
        folder = common.RESULTS / common.machine_info()["machine_id"] / "validation"
        record = json.loads(sorted(folder.glob("v00-baseline@*.json"))[-1].read_text())
        return float(record["official"]["warm_median"])
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            setup = {"frequencies": MODEL["frequencies"], "published": MODEL["published"], "variables": MODEL["variables"], "original": ORIGINAL_SECONDS}
            body = PAGE.replace("__SETUP__", json.dumps(setup)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/run":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not LOCK.acquire(timeout=5):  # the previous run stops as soon as its page closes the stream
                return
            try:
                self.optimise(int(parse_qs(url.query).get("seed", ["2"])[0]))
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page pressed Stop or was closed
            finally:
                LOCK.release()
        else:
            self.send_error(404)

    def optimise(self, seed):
        np, optax = MODEL["np"], MODEL["optax"]
        unbounded = MODEL["start"](seed)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=0.1))
        state = optimizer.init(unbounded)
        began = time.perf_counter()
        for step in range(MAX_STEPS):
            (loss, sensitivity), gradient = MODEL["evaluate"](unbounded)
            if not (np.isfinite(float(loss)) and np.isfinite(np.asarray(gradient)).all()):
                self.send({"type": "stopped", "reason": "the simulation returned a value that is not a number"})
                return
            self.send({"type": "step", "step": step, "seconds": time.perf_counter() - began, "loss": float(loss),
                       "curve": [round(float(v), 4) for v in np.log10(np.asarray(sensitivity))]})
            updates, state = optimizer.update(gradient, state, unbounded)
            unbounded = optax.apply_updates(unbounded, updates)
        self.send({"type": "stopped", "reason": f"{MAX_STEPS} steps done"})

    def send(self, event):
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faster Physics: a detector designed live</title>
<style>
  :root { --paper: #f6f4ef; --ink: #1c1b19; --soft: #6f6b63; --line: #dcd8ce; --card: #fffdf8; --ours: #1f7a4d; --theirs: #9a3b2e; --faint: #c9c4b6; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 17px/1.4 "Avenir Next", "Segoe UI", system-ui, sans-serif; }
  main { max-width: 1280px; margin: 0 auto; padding: 18px 28px 12px; }
  header { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; }
  h1 { font-size: 28px; line-height: 1.15; margin: 0; letter-spacing: -0.01em; }
  .lead { color: var(--soft); margin: 5px 0 12px; max-width: 86ch; font-size: 16px; }
  .buttons { display: flex; gap: 10px; }
  button { font: inherit; font-weight: 600; white-space: nowrap; padding: 10px 22px; border-radius: 8px; border: 1px solid var(--ink); background: var(--ink); color: var(--paper); cursor: pointer; }
  button.quiet { background: transparent; color: var(--ink); }
  button:disabled { opacity: 0.35; cursor: default; } button:focus-visible { outline: 3px solid var(--ours); outline-offset: 3px; }
  .stage { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
  .plot { padding: 10px 12px 6px; } canvas { display: block; width: 100%; height: 430px; }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: 14px; color: var(--soft); padding: 2px 6px 6px 54px; }
  .legend span::before { content: ""; display: inline-block; width: 26px; margin-right: 7px; vertical-align: middle; border-top: 3px solid var(--ours); }
  .legend .now::before { border-top-width: 1.5px; opacity: 0.6; } .legend .first::before { border-top: 2px solid var(--faint); } .legend .goal::before { border-top: 2px dashed var(--ink); }
  .side { padding: 14px 20px 10px; display: flex; flex-direction: column; }
  .figure { border-top: 1px solid var(--line); padding: 9px 0 10px; } .figure:first-child { border-top: 0; padding-top: 2px; }
  .figure .label { font-size: 14px; color: var(--soft); }
  .figure .value { font: 600 38px/1.1 ui-monospace, "SF Mono", Menlo, monospace; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
  .figure .value small { font: 500 16px "Avenir Next", system-ui, sans-serif; color: var(--soft); letter-spacing: 0; margin-left: 5px; }
  .figure.ours .value { color: var(--ours); } .figure.theirs .value { color: var(--theirs); }
  .figure .note { font-size: 13px; color: var(--soft); line-height: 1.3; margin-top: 2px; }
  footer { margin-top: 9px; color: var(--soft); font-size: 13px; line-height: 1.35; }
  @media (max-width: 900px) { .stage { grid-template-columns: 1fr; } canvas { height: 320px; } h1 { font-size: 22px; } }
</style></head><body><main>
  <header><h1>A detector being designed, live</h1>
    <div class="buttons"><button class="quiet" id="again">New random start</button><button id="go">Start</button></div></header>
  <p class="lead">It starts as a random detector. Each step is one full simulation plus the gradient for <span id="variables"></span> design variables, and nudges every mirror, laser and mass a little towards a better one.</p>
  <div class="stage">
    <section class="card plot"><canvas id="plot" role="img" aria-label="Sensitivity of the detector against signal frequency, redrawn after every step"></canvas>
      <div class="legend"><span>best design so far</span><span class="now">this step</span><span class="first">random start</span><span class="goal">published design</span></div></section>
    <section class="card side">
      <div class="figure"><div class="label">Design steps done</div><div class="value" id="steps">0</div></div>
      <div class="figure ours"><div class="label">Our speed</div><div class="value"><span id="rate">0</span><small>steps per second</small></div></div>
      <div class="figure"><div class="label">Sensitivity gained since the start</div><div class="value"><span id="gain">1</span><small>times better</small></div><div class="note" id="gap"></div></div>
      <div class="figure theirs" id="theirs"><div class="label">The original code, in the same time</div><div class="value"><span id="original">0</span><small>steps</small></div><div class="note" id="pace"></div></div>
    </section>
  </div>
  <footer>Lower on the chart is better: the curve is the smallest signal the detector can tell from its own noise, at each frequency. The dashed line is the design published with the challenge. A demonstration, not a design tool: the goal here has no limit on optical power.</footer>
</main>
<script>
const SETUP = __SETUP__, $ = id => document.getElementById(id);
const canvas = $("plot"), ctx = canvas.getContext("2d");
const X = SETUP.frequencies.map(Math.log10), X0 = X[0], X1 = X[X.length - 1], Y0 = -25.6, Y1 = -18.4;
let stream = null, seed = 2, first = null, best = null, bestLoss = Infinity, firstLoss = null, current = null;
$("variables").textContent = SETUP.variables;
if (SETUP.original) $("pace").textContent = `It needs ${SETUP.original.toFixed(1)} s for each step on this machine (measured).`; else $("theirs").hidden = true;
function draw() {
  const ratio = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * ratio) { canvas.width = w * ratio; canvas.height = h * ratio; }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, w, h);
  const left = 54, right = 12, top = 10, bottom = 34, px = x => left + (x - X0) / (X1 - X0) * (w - left - right), py = y => top + (Y1 - y) / (Y1 - Y0) * (h - top - bottom);
  const style = getComputedStyle(document.documentElement), color = name => style.getPropertyValue(name).trim();
  ctx.font = '13px "Avenir Next", system-ui, sans-serif'; ctx.lineWidth = 1;
  for (let e = -25; e <= -19; e++) {
    ctx.strokeStyle = color("--line"); ctx.beginPath(); ctx.moveTo(left, py(e)); ctx.lineTo(w - right, py(e)); ctx.stroke();
    ctx.fillStyle = color("--soft"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText("10", left - 22, py(e));
    ctx.font = '10px "Avenir Next", system-ui, sans-serif'; ctx.textAlign = "left"; ctx.fillText(String(e).replace("-", "−"), left - 21, py(e) - 6); ctx.font = '13px "Avenir Next", system-ui, sans-serif';
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const f of [800, 1000, 1500, 2000, 3000]) { ctx.strokeStyle = color("--line"); ctx.beginPath(); ctx.moveTo(px(Math.log10(f)), top); ctx.lineTo(px(Math.log10(f)), h - bottom); ctx.stroke(); ctx.fillStyle = color("--soft"); ctx.textAlign = f === 3000 ? "right" : f === 800 ? "left" : "center"; ctx.fillText(f + " Hz", px(Math.log10(f)), h - bottom + 8); }
  const line = (curve, stroke, width, dash, alpha) => {
    if (!curve) return;
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, w - left - right, h - top - bottom); ctx.clip();
    ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.setLineDash(dash); ctx.lineJoin = "round"; ctx.beginPath();
    curve.forEach((y, i) => i ? ctx.lineTo(px(X[i]), py(y)) : ctx.moveTo(px(X[i]), py(y))); ctx.stroke(); ctx.restore();
  };
  line(SETUP.published, color("--ink"), 2, [7, 5], 1); line(first, color("--faint"), 2, [], 1);
  line(current, color("--ours"), 1.5, [], 0.55); line(best, color("--ours"), 3.5, [], 1);
}
const times = value => value >= 100 ? Math.round(value).toLocaleString("en") : value.toFixed(1);
function onStep(e) {
  current = e.curve;
  if (e.step === 0) { first = e.curve; firstLoss = e.loss; }
  if (e.loss < bestLoss) { bestLoss = e.loss; best = e.curve; }
  $("steps").textContent = e.step + 1;
  if (e.seconds > 0.3) $("rate").textContent = ((e.step + 1) / e.seconds).toFixed(0);
  $("gain").textContent = times(Math.pow(10, firstLoss - bestLoss));
  $("gap").textContent = bestLoss > 0 ? `Still ${times(Math.pow(10, bestLoss))} times short of the published design.` : "Better than the published design, on this goal.";
  if (SETUP.original) $("original").textContent = Math.floor(e.seconds / SETUP.original);
  draw();
}
function stop() { if (stream) stream.close(); stream = null; $("go").textContent = "Start"; $("again").disabled = false; }
function start() {
  first = best = current = null; bestLoss = Infinity; $("go").textContent = "Stop"; $("again").disabled = true;
  stream = new EventSource("/run?seed=" + seed);
  stream.onmessage = m => { const e = JSON.parse(m.data); if (e.type === "step") onStep(e); else stop(); };
  stream.onerror = stop;
}
$("go").addEventListener("click", () => stream ? stop() : start());
$("again").addEventListener("click", () => { seed += 1; start(); });
addEventListener("resize", draw); draw();
</script></body></html>
"""

if __name__ == "__main__":
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is taken. Start it on another one, for example: python demo/design.py {PORT + 1}")
    print("Getting the simulator ready (a few seconds)...", flush=True)
    MODEL, ORIGINAL_SECONDS = prepare(), original_seconds_per_step()
    print(f"Open http://127.0.0.1:{PORT} and press Start. Stop this with Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
