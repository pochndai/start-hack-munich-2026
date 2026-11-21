# Changes and why

Everything we changed in Differometor for the Feyer "Faster Physics" challenge, in the order we made the changes, with the reason for each one.

## Result so far

Official benchmark (`python benchmark.py --cpus 1`), complete first run, one CPU core:

| Machine | Original | Ours |
| --- | ---: | ---: |
| Intel Xeon Platinum 8468 (Feyer's published baseline) | 44.2 s | not measured yet |
| AMD EPYC 7763, GitHub Codespace, Python 3.12 | does not fit in 16 GB of RAM | **5.8 s** |

Same Codespace, one core, both versions in the style of `start_here.py` (one compile, one run):

| | Original | Ours |
| --- | ---: | ---: |
| Setup | 1.4 s | 1.2 s |
| Compile | 22.6 s | 4.0 s |
| First run | 11.7 s | 0.3 s |
| **Total** | **35.8 s** | **5.6 s** |

A repeat run, once compiled, takes 0.34 s in the official benchmark (the published baseline is 17.7 s).

The report and chart of our run are in `benchmarks/codespace-candidate.json` and `benchmarks/codespace-candidate.png`.

## The four changes

All code changes are in `differometor/simulate.py` and `differometor/__init__.py`. The physics functions in `components.py` and the builder in `build.py` are untouched.

### 1. Solve the signal system with a sparse solver

- **What:** the three solves of the signal matrix now call `klujax.solve` instead of `jnp.linalg.solve`. klujax is a JAX wrapper around KLU, a solver for matrices that are mostly zeros.
- **Why:** the signal matrix is 751×751, but only 2,597 of its 564,001 entries (0.46%) can be non-zero. A dense solver does arithmetic on all of them, for each of the 101 frequencies, three times. KLU only touches the non-zero ones. Finesse, the simulator Differometor is modelled on, uses KLU for the same reason.
- **How it stays general:** the build step reads the positions of the non-zero entries from whatever setup it is given. Nothing is specific to the example setup.
- **Cost:** one new dependency, `klujax==0.5.3`, added to `requirements.txt` and `pyproject.toml`.

### 2. Compile each physics function once, only where it is used

- **What:** the build step now passes the "which function fills which entry" arrays to `simulate` as build-time data (the small `Static` class). `evaluate()` uses them to group the entries by function.
- **Why:** before, the function was chosen with `lax.switch` inside `vmap`, on an input JAX could not see while compiling. JAX therefore compiled all 19 physics functions for every entry, three times (carrier, signal, noise). Now each function is compiled once, for the entries that use it.
- **Effect:** outputs are bit-for-bit the same as after change 1. Compile time fell by about a third.

### 3. Build only the non-zero entries

- **What:** `simulate` no longer creates the full 751×752 signal matrix for each frequency, or the full noise matrix. The new `sparse_layout()` in the build step numbers the entries that can be non-zero, and `simulate` fills, scales, conjugates and solves on that list only.
- **Why:** after change 1 the solver was fast, but the code still filled about 57 million numbers per matrix copy to use 0.5% of them, and the gradient had to remember all of it.
- **Effect:** a repeat run went from about 3 s to 0.07 s on an Apple M5. Peak memory went from 3.8 GiB to 0.5 GiB in our test script (the original needs about 14.4 GiB in the official benchmark).

### 4. Ask the compiler to compile quickly

- **What:** `differometor/__init__.py` adds four settings to `XLA_FLAGS` before JAX starts, unless the user has already set them:
  - `--xla_backend_optimization_level=0`: do not optimise the generated code.
  - `--xla_cpu_use_fusion_emitters=false`: use the simpler of XLA's two code generators.
  - `--xla_llvm_disable_expensive_passes=true`: skip slow optimisation passes.
  - `--xla_cpu_parallel_codegen_split_count=1`: do not split the program for parallel compiling, which only costs time on one core.
- **Why:** after changes 1–3, compiling is about 85% of a first run and the compiled program then runs in a fraction of a second. Code that is quick to produce matters more than code that is quick to run.
- **Effect:** measured on the Codespace with the official benchmark, complete first run:

  | Settings | Total | Compile |
  | --- | ---: | ---: |
  | None (XLA defaults) | 17.8 s | 15.9 s |
  | Optimisation level 0 | 10.9 s | 9.2 s |
  | All four | 5.8 s | 4.2 s |

  The repeat run went from 0.23 s to 0.34 s.
- **Safety:** XLA stops with an error on a flag it does not know. The second and fourth flag exist in the pinned jaxlib 0.9.0 but not in every release, so they are only added when jaxlib is a 0.9 version.
- **Transparency:** the benchmark records the flags in its report under `environment.xla_flags`.

## Step by step (Apple M5, official benchmark without CPU pinning)

| Version | Total | Compile | Forward | Gradient |
| --- | ---: | ---: | ---: | ---: |
| Original | 12.44 s | 3.69 s | 4.42 s | 3.70 s |
| + change 1 | 8.20 s | 3.73 s | 1.76 s | 1.57 s |
| + change 2 | 7.37 s | 2.39 s | 1.96 s | 1.78 s |
| + change 3 | 3.61 s | 2.41 s | 0.04 s | 0.06 s |

Change 4 was measured on Linux only (table above).

## Correctness checks

- **Official checks:** `benchmark.py` prints `PASS` on macOS and on Linux with one CPU: positive sensitivities within 1e-5 of the public reference, all outputs finite, non-zero gradient, finite-difference gradient spot check.
- **Distance from the public reference:** 5.374e-06 in log sensitivity, the same as the original code (limit 1e-5).
- **Original against ours, same Linux machine, all outputs:**

  | Output | Largest difference, relative to its largest value |
  | --- | ---: |
  | Carrier, powers | below 1e-19 |
  | Signal | 5.7e-08 |
  | Noise | 1.5e-08 |
  | Sensitivity | 2.0e-09 |
  | Gradient (193 values) | 2.4e-08 |

  The remaining differences come from using a different solver on a matrix that is very sensitive to rounding.
- **Other public setups:** `examples/aligo.py` still passes its own comparison with Finesse (limit 1e-6). `examples/cavity.py` and `examples/voyager.py` run.
- **What is computed:** all 101 frequencies, all 193 design variables, double precision. No answers are stored and no calculation is skipped.

## Reproduce

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python benchmark.py --cpus 1 --output benchmark-candidate.json
```

For the original, run the same commands on the `main` branch. It needs about 16 GB of free RAM.

## Known limits

- No run yet on the contest's Intel Xeon; the Linux numbers are from an AMD EPYC Codespace.
- The original could not complete the official benchmark on the 16 GB Codespace, so the official-style "before" on that machine is missing.
- Setups with several detectors or a homodyne detector use code paths the example does not exercise; we have not tested one.
- The carrier system (286×286, solved once) still uses the dense solver.
