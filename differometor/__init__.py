import os


def _jaxlib_version():
    """The installed jaxlib's version. Read from jaxlib itself, which JAX loads anyway; package metadata is slow to query."""
    try:
        import jaxlib.version

        return jaxlib.version.__version__
    except Exception:
        from importlib.metadata import version

        return version("jaxlib")


# Compiling is most of a first evaluation on CPU, and the compiled program then runs in a
# fraction of a second, so ask XLA to compile quickly rather than to produce fast code.
# This has to be set before JAX starts its backend; flags the user set are left alone.
_FAST_COMPILE_FLAGS = {
    "xla_backend_optimization_level": "0",
    "xla_llvm_disable_expensive_passes": "true",
}
if _jaxlib_version().startswith("0.9."):
    # XLA aborts on flags it does not know, and these two come and go between releases.
    _FAST_COMPILE_FLAGS |= {
        "xla_cpu_use_fusion_emitters": "false",
        "xla_cpu_parallel_codegen_split_count": "1",
    }
_xla_flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = " ".join(
    [_xla_flags] + [f"--{name}={value}" for name, value in _FAST_COMPILE_FLAGS.items() if name not in _xla_flags]
).strip()



def _reuse_freed_memory():
    """
    On Linux, ask the C library to keep freed memory and hand it out again.

    The simulation creates and frees many NumPy arrays of a megabyte or more. By default
    glibc gets each of them from the operating system and gives it back when it is freed,
    so every one of them is paid for again in page faults. Only where memory comes from
    changes, no number does. The price is that freed memory stays with the process.
    Settings the user made through the environment are left alone.
    """
    import sys

    settings = ("MALLOC_MMAP_MAX_", "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_TOP_PAD_")
    if not sys.platform.startswith("linux") or any(name in os.environ for name in settings) or "glibc.malloc" in os.environ.get("GLIBC_TUNABLES", ""):
        return
    try:
        import ctypes

        mallopt = ctypes.CDLL(None).mallopt
        mallopt(-4, 0)  # M_MMAP_MAX: no separate mapping per large block
        mallopt(-1, 1 << 30)  # M_TRIM_THRESHOLD: do not return the top of the heap
        mallopt(-2, 64 << 20)  # M_TOP_PAD: grow the heap in large steps
    except Exception:
        pass


_reuse_freed_memory()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name):
    # `visualize_setup` needs plotly, which a simulation does not; load it when it is first asked for.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
