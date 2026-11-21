import os
from importlib.metadata import version

# Compiling is most of a first evaluation on CPU, and the compiled program then runs in a
# fraction of a second, so ask XLA to compile quickly rather than to produce fast code.
# This has to be set before JAX starts its backend; flags the user set are left alone.
_FAST_COMPILE_FLAGS = {
    "xla_backend_optimization_level": "0",
    "xla_llvm_disable_expensive_passes": "true",
}
if version("jaxlib").startswith("0.9."):
    # XLA aborts on flags it does not know, and these two come and go between releases.
    _FAST_COMPILE_FLAGS |= {
        "xla_cpu_use_fusion_emitters": "false",
        "xla_cpu_parallel_codegen_split_count": "1",
    }
_xla_flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = " ".join(
    [_xla_flags] + [f"--{name}={value}" for name, value in _FAST_COMPILE_FLAGS.items() if name not in _xla_flags]
).strip()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.plot import visualize_setup
from differometor.components import power_detector, signal_detector