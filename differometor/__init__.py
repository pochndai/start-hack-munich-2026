import os

# Compiling is most of a first evaluation on CPU, and the compiled program then runs in a
# fraction of a second. Unoptimised kernels compile about twice as fast. This has to be
# set before JAX starts its backend; a level the user chose themselves is left alone.
if "xla_backend_optimization_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=0").strip()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.plot import visualize_setup
from differometor.components import power_detector, signal_detector