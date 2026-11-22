"""Does the simulator come back from every way of calling it when the process has one CPU?

Run from the repository root, on Linux:

    python tests/one_cpu_calls.py                    all cases, as the package sets JAX up
    python tests/one_cpu_calls.py --async-dispatch   the same with JAX's asynchronous dispatch left on,
                                                     which is how the code behaved before change 22

Each case builds the supplied setup the way benchmark.py does and then calls the simulator in
one particular order. It runs in a process of its own that is pinned to one CPU before JAX is
imported. If its calls have not returned after a time limit, Python's faulthandler prints where
every thread is and ends that process; the case is then reported as FROZE.

Cases:
  waited         benchmark.py's order: forward, readout, gradient, waiting for each result
  not-awaited    the same three compiled programs, but the gradient is started while the readout is still running
  one-program    loss and gradient as one compiled program, jax.jit(jax.value_and_grad(loss))
  loop           five optimizer steps (that program, then a compiled optax update) with no waiting in between
  eager          jax.grad(loss) without jax.jit

Exit code 0 if every case finishes. With --async-dispatch the cases `not-awaited` and `loop` are
expected to freeze on one CPU (`loop` not every time), and the exit code says so.
"""
import os
import sys

CASES = ("waited", "not-awaited", "one-program", "loop", "eager")
LIMITS = {"eager": 120.0}  # seconds; 45 for the others
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_all(arguments):
    import subprocess

    if not hasattr(os, "sched_setaffinity"):
        print("This test pins a process to one CPU, which needs Linux. Nothing was run.")
        return 0
    keep_async = "--async-dispatch" in arguments
    cpu = max(os.sched_getaffinity(0))
    environment = dict(os.environ)
    environment.pop("JAX_CPU_ENABLE_ASYNC_DISPATCH", None)
    if keep_async:
        environment["JAX_CPU_ENABLE_ASYNC_DISPATCH"] = "1"  # the package leaves a setting made here alone
    print(f"One CPU (logical CPU {cpu}), asynchronous dispatch {'left on' if keep_async else 'as the package sets it'}.")
    froze, failed = [], []
    for case in CASES:
        result = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", case, "--cpu", str(cpu)],
                                capture_output=True, text=True, env=environment, cwd=repo)
        text = result.stdout + result.stderr
        finished = [line for line in text.splitlines() if "FINISHED" in line]
        if finished:
            print(f"  {case:12} {finished[0].split(': ', 1)[1]}")
            print(f"  {'':12} {[line.strip() for line in text.splitlines() if 'calls into NumPy' in line][0]}")
        elif "Timeout (" in text:
            froze.append(case)
            steps = [line.strip() for line in text.splitlines() if " entered on " in line or " returned" in line]
            print(f"  {case:12} FROZE: nothing returned within {LIMITS.get(case, 45.0):.0f} s. Last step: {steps[-1] if steps else 'none'}")
        else:
            failed.append(case)
            print(f"  {case:12} ERROR (exit code {result.returncode}): {text.strip().splitlines()[-1] if text.strip() else 'no output'}")
    if froze or failed:
        print(f"FROZE: {', '.join(froze) or 'none'}; errors: {', '.join(failed) or 'none'}")
        return 1
    print("ALL CASES FINISHED")
    return 0


def run_case(case, cpu):
    os.sched_setaffinity(0, {cpu})  # before JAX is imported, as benchmark.py --cpus 1 does
    import faulthandler
    import json
    import threading
    from pathlib import Path
    from time import perf_counter

    sys.path.insert(0, repo)
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import differometor as df
    from differometor import host
    from differometor.setups import constrain_inter_grid_cell_spaces

    root = Path(repo)
    assert Path(df.__file__).resolve().parent == root / "differometor", f"imported {df.__file__}, not this checkout"
    print(f"case {case}: {len(os.sched_getaffinity(0))} CPU for this process, jax {jax.__version__}, "
          f"asynchronous dispatch {'on' if jax.config.values['jax_cpu_enable_async_dispatch'] else 'off'}", flush=True)

    # Note on which thread each call into NumPy runs: the calling (main) thread or one of XLA's own.
    calls = []

    def watch(name):
        original = getattr(host, name)

        def watched(*arguments, **keywords):
            thread = threading.current_thread()
            if thread is threading.main_thread():
                where = "main thread"
            else:
                where = "XLA thread " + Path(f"/proc/self/task/{threading.get_native_id()}/comm").read_text().strip()
            calls.append(f"{name} on the {where}")
            print(f"   {name} entered on the {where}", flush=True)
            result = original(*arguments, **keywords)
            print(f"   {name} returned", flush=True)
            return result

        setattr(host, name, watched)

    watch("_forward")
    watch("_backward")

    properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
    fixture = json.loads((root / "examples" / "data" / "uifo_800_3000.json").read_text())
    reference = json.loads((root / "examples" / "data" / "uifo_800_3000_test_data.json").read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, properties)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        initial.append(fixture["nodes" if members[0][0] in fixture["nodes"] else "edges"][members[0][0]]["properties"][members[0][1]])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    ports, one = jnp.asarray(ports), jnp.asarray(1.0, dtype=jnp.float64)
    jax.block_until_ready((arrays, ports, parameters, expected))

    def forward_with_pullback(parameters, arrays):
        return jax.vjp(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)), parameters)

    def readout(outputs, detector_ports, reference):
        carrier, signal, noise = outputs
        signals = df.signal_detector(carrier, signal)[detector_ports]
        detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity / reference)), (sensitivity, df.power_detector(carrier))

    def sensitivity_with_pullback(outputs, detector_ports, reference):
        return jax.vjp(lambda values: readout(values, detector_ports, reference), outputs, has_aux=True)

    def backward(simulation_pullback, readout_pullback, cotangent):
        return simulation_pullback(readout_pullback(cotangent)[0])[0]

    def loss(parameters, arrays):
        return readout(df.simulate(**dict(arrays, optimized_parameters=parameters)), ports, expected)[0]

    def three_programs(wait_for_readout, repeats=3):
        forward = jax.jit(forward_with_pullback).lower(parameters, arrays).compile()
        outputs, simulation_pullback = jax.block_until_ready(forward(parameters, arrays))
        sensitivity_fn = jax.jit(sensitivity_with_pullback).lower(outputs, ports, expected).compile()
        value, readout_pullback, _ = jax.block_until_ready(sensitivity_fn(outputs, ports, expected))
        gradient_fn = jax.jit(backward).lower(simulation_pullback, readout_pullback, one).compile()
        gradient = jax.block_until_ready(gradient_fn(simulation_pullback, readout_pullback, one))
        print("   compiled and ran once with waits; now the order under test", flush=True)
        for _ in range(repeats):
            outputs, simulation_pullback = jax.block_until_ready(forward(parameters, arrays))
            value, readout_pullback, _ = sensitivity_fn(outputs, ports, expected)
            if wait_for_readout:
                jax.block_until_ready((value, readout_pullback))
            gradient = jax.block_until_ready(gradient_fn(simulation_pullback, readout_pullback, one))
        return value, gradient

    def one_program(repeats=3):
        value_and_gradient = jax.jit(jax.value_and_grad(loss))
        for _ in range(repeats):
            value, gradient = jax.block_until_ready(value_and_gradient(parameters, arrays))
        return value, gradient

    def loop(steps=5):
        import optax

        # Learning rate zero: the update is still a compiled program whose result feeds the next
        # step, but no parameter moves, so every step stays inside the valid range.
        optimizer = optax.adam(0.0)
        value_and_gradient = jax.jit(jax.value_and_grad(loss))

        @jax.jit
        def update(gradient, state, current):
            updates, state = optimizer.update(gradient, state, current)
            return optax.apply_updates(current, updates), state

        current, state = parameters, optimizer.init(parameters)
        value, gradient = jax.block_until_ready(value_and_gradient(current, arrays))
        jax.block_until_ready(update(gradient, state, current))
        print("   compiled; now five steps without waiting", flush=True)
        for _ in range(steps):
            value, gradient = value_and_gradient(current, arrays)
            current, state = update(gradient, state, current)
        jax.block_until_ready((value, current))
        return value, gradient

    def eager():
        return jax.block_until_ready((loss(parameters, arrays), jax.grad(loss)(parameters, arrays)))

    faulthandler.dump_traceback_later(LIMITS.get(case, 45.0), exit=True)
    start = perf_counter()
    value, gradient = {
        "waited": lambda: three_programs(True),
        "not-awaited": lambda: three_programs(False),
        "one-program": one_program,
        "loop": loop,
        "eager": eager,
    }[case]()
    faulthandler.cancel_dump_traceback_later()
    print(f"case {case}: FINISHED in {perf_counter() - start:.1f} s, loss {float(value):.6e}, gradient norm {float(jnp.linalg.norm(gradient)):.6e}")
    print(f"   calls into NumPy: {len(calls)} ({'; '.join(sorted(set(calls)))})")


if __name__ == "__main__":
    if "--case" in sys.argv:
        run_case(sys.argv[sys.argv.index("--case") + 1], int(sys.argv[sys.argv.index("--cpu") + 1]))
    else:
        sys.exit(run_all(sys.argv[1:]))
