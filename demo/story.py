#!/usr/bin/env python3
"""The one-minute demo: a collision in space, the detector that hears it, and the design search, live.

    python demo/story.py           # then open http://127.0.0.1:8813 and press the space bar
    python demo/story.py 9000      # the same on another port

Keys on the page: space = play / pause, 1 to 4 = jump to a scene, R = record the whole minute.

What is real and what is drawing:
- Scenes 1 and 2 (the collision, the ripples, the stretching arms) are illustrations, exaggerated
  so they can be seen.
- Scenes 3 and 4 are driven by the simulator. Every beam in the grid is drawn at the light power
  the simulation computes for it, and scene 4 runs the same search as demo/design.py live: each
  step is one complete simulation of 101 frequencies plus the gradient for 193 design variables.
- "How far it hears" uses a real relation (a detector twice as sensitive hears twice as far) on a
  logarithmic scale, with the design published with the challenge as the outermost ring.
- The original code's ring follows the same search at the original's measured time per step on
  this machine (bench/results/<machine>/validation/v00-baseline@*.json), if that record exists.

A recording (key R) is sent back to this program, saved in demo/recordings/ and converted to MP4
if ffmpeg is installed. Standard library plus the simulator's own dependencies.
"""
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8813
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
# The ranges examples/uifo_optimization.py lets the optimizer move in.
BOUNDS = {"db": [0, 10], "angle": [-360, 360], "power": [0, 200], "tuning": [-360, 360], "mass": [0.01, 200], "length": [0.1, 4000], "reflectivity": [0, 1]}
MAX_STEPS = 6000
LOCK = threading.Lock()  # one search at a time


def place(name, size):
    """Where a part of the grid sits on the drawing: (x, y, kind of part, 'h' or 'v' for a mirror)."""
    match = re.fullmatch(r"(center|ml|mr|mt|mb|m|boundary)(\d)(\d)(detector)?", name)
    if not match:
        return None
    kind, row, column = match.group(1), int(match.group(2)), int(match.group(3))
    x, y, gap = float(column), float(row), 0.34
    if kind in ("ml", "mr"):
        return x + (gap if kind == "mr" else -gap), y, "mirror", "v"
    if kind in ("mt", "mb"):
        return x, y + (gap if kind == "mb" else -gap), "mirror", "h"
    if kind == "center":
        return x, y, "center", ""
    edge = 0.36 if kind == "m" else -0.02  # boundary mirrors sit just inside the sources
    if column == 0:
        x = edge
    elif column == size + 1:
        x = size + 1 - edge
    elif row == 0:
        y = edge
    else:
        y = size + 1 - edge
    side = "v" if column in (0, size + 1) else "h"
    return x, y, ("mirror" if kind == "m" else "source"), side


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
    first = lambda pair: pair if isinstance(pair[0], str) else pair[0]
    bounds = jnp.asarray(np.array([[BOUNDS[first(pair)[1]][0] for pair in pairs], [BOUNDS[first(pair)[1]][1] for pair in pairs]], dtype=float))
    frequencies, published = jnp.asarray(public["frequencies"]), jnp.asarray(public["sensitivities"])
    arrays, ports, *_, port_index = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)

    # The drawing: parts with a position, and one beam per connection with the ports whose light it carries.
    size = max(int(name[6]) for name in fixture["nodes"] if re.fullmatch(r"center\d\d", name))
    nodes, beams = {}, []
    for name, node in fixture["nodes"].items():
        spot = place(name, size)
        if spot:
            kind = {"laser": "laser", "squeezer": "squeezer", "detector": "detector", "beamsplitter": "splitter", "directional_beamsplitter": "director"}.get(node["component"], spot[2])
            nodes[name] = {"x": spot[0], "y": spot[1], "kind": kind, "turn": spot[3]}
    for name, edge in fixture["edges"].items():
        a, b = name.split("_", 1)
        if a in nodes and b in nodes:
            carried = [port_index.get(f"{a}.{edge['source_port']}.out"), port_index.get(f"{b}.{edge['target_port']}.out")]
            beams.append({"a": a, "b": b, "ports": [index for index in carried if index is not None]})
    for name, node in fixture["nodes"].items():  # the detector looks at one port of a mirror
        if node["component"] == "detector" and name in nodes:
            beams.append({"a": node["target"], "b": name, "ports": [port_index[f"{node['target']}.{node['port']}.{node['direction']}"]]})
    knobs = []
    for pair in pairs:  # one mark per design variable, at the part or connection it belongs to
        name = first(pair)[0].removesuffix("sus")  # a mass belongs to the mirror it hangs
        if name in nodes:
            knobs.append([nodes[name]["x"], nodes[name]["y"]])
        elif "_" in name:
            a, b = name.split("_", 1)
            if a in nodes and b in nodes:
                knobs.append([(nodes[a]["x"] + nodes[b]["x"]) / 2, (nodes[a]["y"] + nodes[b]["y"]) / 2])

    def objective(unbounded):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=sigmoid_bounding(unbounded, bounds)))
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity / published)), (sensitivity, df.power_detector(carrier))

    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))

    def light(powers):
        """log10 of the light power in each beam, from the simulator's power at every port."""
        powers = np.asarray(powers).ravel()
        return [round(float(np.log10(max(sum(powers[index] for index in beam["ports"]), 1e-12))), 2) for beam in beams]

    def start(seed):
        values = np.random.default_rng(seed).uniform(np.asarray(bounds[0]), np.asarray(bounds[1]))
        return jnp.asarray(inverse_sigmoid_bounding(values, bounds))

    supplied = {name: item["properties"] for kind in ("nodes", "edges") for name, item in fixture[kind].items()}
    design = jnp.asarray(inverse_sigmoid_bounding(np.clip(np.array([supplied[first(pair)[0]][first(pair)[1]] for pair in pairs], dtype=float),
                                                          np.asarray(bounds[0]) + 1e-9, np.asarray(bounds[1]) - 1e-9), bounds))
    (_, (_, powers)), _ = evaluate(design)  # compiles, and gives the light in the published design
    scene = {"size": size, "nodes": nodes, "beams": [{"a": beam["a"], "b": beam["b"]} for beam in beams], "knobs": knobs,
             "variables": len(pairs), "published_light": light(powers)}
    return {"np": np, "optax": optax, "evaluate": evaluate, "start": start, "light": light, "scene": scene}


def original_seconds_per_step():
    """The original's measured time for one evaluation on this machine, or None."""
    try:
        import common
        folder = common.RESULTS / common.machine_info()["machine_id"] / "validation"
        return float(json.loads(sorted(folder.glob("v00-baseline@*.json"))[-1].read_text())["official"]["warm_median"])
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.reply((HERE / "story.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/scene.json":
            self.reply(json.dumps(dict(MODEL["scene"], original=ORIGINAL_SECONDS)).encode(), "application/json")
        elif url.path == "/run":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not LOCK.acquire(timeout=5):  # the previous search stops as soon as its page closes the stream
                return
            try:
                self.search(int(parse_qs(url.query).get("seed", ["2"])[0]))
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page stopped listening
            finally:
                LOCK.release()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/recording":
            return self.send_error(404)
        folder = HERE / "recordings"
        folder.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        raw = folder / f"story-{stamp}.raw.{'mp4' if 'mp4' in self.headers.get('Content-Type', '') else 'webm'}"
        raw.write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
        saved = raw
        if shutil.which("ffmpeg"):  # a clean constant-rate MP4 that any player and any slide program opens
            film = folder / f"story-{stamp}.mp4"
            done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "19", "-preset", "medium",
                                   "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(film)], capture_output=True, text=True)
            if done.returncode == 0:
                raw.unlink()
                saved = film
        print(f"Recording saved: {saved}", flush=True)
        self.reply(json.dumps({"saved": str(saved)}).encode(), "application/json")

    def search(self, seed):
        np, optax = MODEL["np"], MODEL["optax"]
        unbounded = MODEL["start"](seed)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=0.1))
        state = optimizer.init(unbounded)
        began = time.perf_counter()
        for step in range(MAX_STEPS):
            (loss, (_, powers)), gradient = MODEL["evaluate"](unbounded)
            if not (np.isfinite(float(loss)) and np.isfinite(np.asarray(gradient)).all()):
                return
            self.wfile.write(f"data: {json.dumps({'step': step, 'seconds': time.perf_counter() - began, 'loss': float(loss), 'light': MODEL['light'](powers)})}\n\n".encode())
            self.wfile.flush()
            updates, state = optimizer.update(gradient, state, unbounded)
            unbounded = optax.apply_updates(unbounded, updates)


if __name__ == "__main__":
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is taken. Start it on another one, for example: python demo/story.py {PORT + 1}")
    print("Getting the simulator ready (a few seconds)...", flush=True)
    MODEL, ORIGINAL_SECONDS = prepare(), original_seconds_per_step()
    print(f"Open http://127.0.0.1:{PORT} and press the space bar. Stop this with Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
