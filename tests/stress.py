"""Stress test of the NumPy path (written for v12). Run from the repository root:  python tests/stress.py [quick]

1. Random layouts over a wide band, NumPy path against JAX path (outputs and gradients).
2. Pullbacks used out of order and from several threads (the kept assembly and its tickets).
3. vmap of a gradient against a loop.
4. Many evaluations in one process: memory must level off.
"""
import os
import resource
import sys
import threading

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "true"
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quick = len(sys.argv) > 1
sys.path.insert(0, repo)
sys.path.insert(0, repo + "/tests")

import importlib

import jax
import jax.numpy as jnp
import numpy as np

import test_host as t
from differometor import host
from differometor.setups import uifo, voyager

sim = importlib.import_module("differometor.simulate")
frequency = [("f", "frequency")]


def rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


print("1. Random layouts, 1 Hz to 10 kHz, NumPy path against JAX path")
seeds = range(10, 14) if quick else range(10, 22)
for seed in seeds:
    size = 2 if seed % 2 else 3
    case = t.prepare(uifo(size, random=True, random_seed=seed)[0], frequency, jnp.logspace(0, 4, 24))
    t.compare(f"random UIFO {size}x{size}, seed {seed}", case[0], case[2])
    t.compare(f"  same, parameters shifted, seed {seed}", case[0], case[2], shift_seed=seed)

print("2. Pullbacks out of order and from threads (main example)")
arrays, ports, start, pairs = t.saved_setup("uifo_800_3000.json", 800, 3000)
f = lambda p: sim.simulate(**dict(arrays, optimized_parameters=p))
rng = np.random.default_rng(3)
points = [start * (1 + 1e-3 * jnp.asarray(rng.standard_normal(start.shape))) for _ in range(6)]
outputs0, _ = jax.vjp(f, start)
cotangent = tuple(jnp.asarray(rng.standard_normal(o.shape) + (1j * rng.standard_normal(o.shape) if jnp.iscomplexobj(o) else 0)) / float(jnp.abs(o).max()) for o in outputs0)
expected = [np.asarray(jax.vjp(f, p)[1](cotangent)[0]) for p in points]  # forward and pullback back to back
pulls = [jax.vjp(f, p)[1] for p in points]  # six forwards first: only the last two are still kept
worst = 0.0
for index in rng.permutation(len(points)).tolist() * 2:
    worst = max(worst, t.relative(pulls[index](cotangent)[0], expected[index]))
print(f"   12 pullbacks in random order after 6 forward passes: {t.check('out of order', worst, 0)}  (0 = identical)")
print(f"   kept assemblies: {len(host._KEPT)} (at most 2)")
if len(host._KEPT) > 2:
    t.FAILURES.append("more than two assemblies kept")

grad = jax.jit(lambda p: jax.vjp(f, p)[1](cotangent)[0])
grad(start)
results = [None] * len(points)


def work(index):
    for _ in range(4):
        results[index] = np.asarray(grad(points[index]))


threads = [threading.Thread(target=work, args=(i,)) for i in range(len(points))]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
worst = max(t.relative(results[i], expected[i]) for i in range(len(points)))
print(f"   6 threads, 4 jitted gradients each: {t.check('threads', worst, 0)}")

print("3. vmap of a gradient over 5 parameter sets against a loop")
batch = jnp.stack(points[:5])
mapped = jax.vmap(lambda p: jax.vjp(f, p)[1](cotangent)[0])(batch)
worst = max(t.relative(mapped[i], expected[i]) for i in range(5))
print(f"   {t.check('vmap gradient', worst, 0)}")

print("4. Memory over many evaluations")
count = 60 if quick else 300
marks = []
for index in range(count):
    grad(points[index % len(points)])
    if index in (count // 3, 2 * count // 3, count - 1):
        marks.append(rss_mb())
print(f"   peak memory after {count // 3}, {2 * count // 3} and {count} gradients: " + ", ".join(f"{m:.0f} MB" for m in marks))
if marks[-1] > marks[0] * 1.10 + 20:
    t.FAILURES.append(f"memory keeps growing: {marks}")

print("\nFAILURES:" if t.FAILURES else "\nSTRESS TEST PASSED")
for failure in t.FAILURES:
    print("  " + failure)
