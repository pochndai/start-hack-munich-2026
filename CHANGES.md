# Changes and why

Everything we changed in Differometor for the Feyer "Faster Physics" challenge, in the order we made the changes, with the reason for each one.

## Result so far

Official benchmark, one CPU core (`--cpus 1`), original and current version on the same Linux machine (AMD EPYC 9334 on the AMD Radeon Cloud, 55 GB RAM, Python 3.12.15, 10 repeats). Both print `PASS`. This is the comparison to quote:

| Step | Original | Ours (changes 1–12) |
| --- | ---: | ---: |
| Imports | 0.29 s | 0.28 s |
| Build the setup | 0.70 s | 0.08 s |
| Trace and compile | 19.28 s | 0.17 s |
| Forward simulation | 12.35 s | 0.07 s |
| Gradient | 3.00 s | 0.10 s |
| **Complete first run** | **35.62 s** | **0.72 s** |
| Repeat run, once compiled (median of 10) | 15.67 s | 0.12 s |
| Distance from the public reference, in log (limit 1e-5) | 5.374e-06 | 5.375e-06 |

That is about 50 times faster for the scored first run. A second run of our version on the same machine gave 0.716 s against 0.715 s. On that machine `tests/test_host.py` ends with `ALL CHECKS PASSED` and `tests/handwritten_jacobians.py` with all 19 rules agreeing with JAX. The reports and charts are `benchmarks/amd-epyc-9334-original.*` and `benchmarks/amd-epyc-9334-candidate.*`; the file fingerprints inside them match the upstream code and the code on `main`. The saved output arrays of the two runs were compared on the machine but not brought back; the array comparisons recorded below are from the Apple M5 and the Codespace.

How the total came down on that machine: an earlier run of changes 1–11 there took 2.13 s (printed timings only, no report saved), so change 12 accounts for the step from 2.13 s to 0.72 s.

Change 12 on a second machine. Official benchmark, one core, Linux Codespace (AMD EPYC 7763), three runs each, all `PASS`:

| Version | Complete first run | Trace and compile | Repeat run |
| --- | ---: | ---: | ---: |
| Changes 1–11 | 3.23, 3.52, 4.22 s | about 2.5 s | 0.19 s |
| Changes 1–12 | 1.01, 1.05, 1.33 s | about 0.2 s | 0.16 s |

The numbers in the rest of this section are for changes 1–11.

Official benchmark on the same machine, both versions with `PASS` (Apple M5, 10 repeats; a Mac cannot pin to one core, so all cores are used):

| | Original | Ours |
| --- | ---: | ---: |
| Complete first run | 13.1 s | **1.01 s** |
| Repeat run, once compiled | 6.00 s | 0.054 s |

A second pair of runs on the same Mac gave 10.5 s and 1.28 s. Reports: `benchmarks/mac-original.json` and `benchmarks/mac-candidate.json`.

The original's Mac times are inflated by a shortage of memory. This Mac has 16 GB, and in the official benchmark the original asks for 18.4 GB at its peak (ours: 0.31 GB), so macOS was compressing and swapping throughout: about half of the original's processor time was spent inside the operating system. Its times also vary more from run to run (10.5 to 13.1 s). The one-core Linux comparison below is not affected, because that machine has no swap and the starter-script style fits in its memory; quote that one.

Official benchmark (`python benchmark.py --cpus 1`), complete first run, one CPU core:

| Machine | Original | Ours |
| --- | ---: | ---: |
| Intel Xeon Platinum 8468 (Feyer's published baseline) | 44.2 s | not measured yet |
| AMD EPYC 7763, GitHub Codespace, Python 3.12 | does not fit in 16 GB of RAM | **3.2 s** |
| AMD EPYC 9334, AMD Radeon Cloud, Python 3.12 | 35.6 s | **2.13 s** |

Same Codespace, one core, both versions in the style of `start_here.py` (one compile, one run):

| | Original | Ours |
| --- | ---: | ---: |
| Setup | 1.4 s | 0.5 s |
| Compile | 22.6 s | 2.4 s |
| First run | 11.7 s | 0.2 s |
| **Total** | **35.8 s** | **3.1 s** |

A repeat run, once compiled, takes 0.20 s in the official benchmark (the published baseline is 17.7 s).

The machines in the table above are different, so 44.2 s and 3.2 s are not a same-hardware comparison. The AMD EPYC 9334 table at the top and the Codespace table just above are.

The report and chart of our run are in `benchmarks/codespace-candidate.json` and `benchmarks/codespace-candidate.png`.

## The twelve changes

Most of the code changes are in `differometor/simulate.py`. In `components.py` the physics formulas are the same; only the mirror function was rewritten in an equivalent, shorter form (change 10).

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

### 5. Drop the third signal solve

- **What:** the noise calculation solved the signal system three times per frequency. The third solve is gone; the noise is computed directly from the result of the second.
- **Why:** it was algebraically redundant. With `A` the signal matrix, `N` the noise matrix, `c` the detector selection and `w` the solution of `Aᴴw = c`, the code computed `y = A⁻¹Nw` and then `Re(cᴴy)`. Substituting gives `Re(wᴴA·A⁻¹Nw) = Re(wᴴNw)`, which needs no solve.
- **What it is not:** nothing is skipped or stored. The signal and noise systems are still assembled and solved for every frequency from the current parameters; one internal solve whose result cancels out is removed.
- **Step-by-step derivation:** [`docs/third-solve-algebra.pdf`](docs/third-solve-algebra.pdf) walks through the algebra with a small example that can be checked by hand.

### 6. Call each physics function once on whole arrays

- **What:** `evaluate()` calls each physics function once with all its inputs as arrays, instead of wrapping it in `vmap` per entry. `standardize_output()` in `components.py` now broadcasts its entries so this works; for single numbers it behaves as before.
- **Why:** tracing through two layers of `vmap` is slow in Python. All 19 functions work elementwise and give identical results when called this way (checked for each one).

### 7. Fill in the build step's constants with NumPy

- **What:** `components.numpy_backend()` makes the physics functions use NumPy while the build step evaluates them on plain numbers to fill in constant matrix entries.
- **Why:** with JAX, each of those scalar operations was sent to the device and compiled on first use: 65 small compilations, about half the build time. The built arrays are bit-for-bit the same.

### Changes 5–7 measured one at a time

Official benchmark on the Codespace, one core, each version run twice:

| Version | Total | Build | Forward trace | Repeat run |
| --- | ---: | ---: | ---: | ---: |
| Changes 1–4 | 5.75 s, 5.70 s | 0.77 s | 1.2 s | 0.34 s |
| + change 7 | 5.36 s, 5.35 s | 0.40 s | 1.2 s | 0.35 s |
| + change 5 | 5.28 s, 5.20 s | 0.39 s | 1.2 s | 0.29 s |
| + change 6 | 5.15 s, 5.09 s | 0.40 s | 1.0 s | 0.27 s |

### 8. Keep the parameters real

- **What:** the parameter vector is stored as real numbers (`float64`) instead of complex ones. Matrix entries and function outputs stay `complex128`.
- **Why:** every parameter is a real quantity (mass, reflectivity, length, angle). Stored as complex, they turned all the arithmetic on them, including sines, cosines and hyperbolic functions, into complex arithmetic, which is much more code to compile and to differentiate.
- **Out-of-range values:** with complex parameters, a reflectivity pushed slightly above 1 gave an imaginary square root; with real ones it would give NaN. So the three square roots of a reflectivity, transmissivity or power are still taken as complex (`as_complex`). Our perturbed-parameter check caught this.
- **Snell's law:** `cos(arcsin(x))` is computed by `cos_arcsin`, a closed form with one square root that returns the same values, also beyond total internal reflection (checked on 6,000 points from -3 to 3, largest difference 1e-14). The complex arcsine it replaces is one of the slowest functions to compile.
- **The same closed form in the helper `incidence_angle_right`** (added with change 12): the radiation-pressure force and the displacement coupling on the right side of a surface use it. With real parameters and the real arcsine they would return NaN beyond total internal reflection, where the original returns a complex number.

### 9. Only build the coefficients that are read

- **What:** which coefficient of which function goes where is now build-time data too, and `evaluate()` reads the functions' coefficients directly (`raw_outputs`).
- **Why:** every function returned four values, most of them padding zeros. They were broadcast, stacked, differentiated, written into a table and mostly never read.
- **Effect:** bit-for-bit the same outputs as after change 8.

### 10. Share the common steps of the mirror function

- **What:** `surface()` computes its two reflection coefficients and its transmission coefficient from shared phases, angles and square roots, instead of calling three helpers that each recompute them.
- **Why:** it is the largest function and is traced twice. Its compiled size with gradient fell by a third.
- **Effect:** bit-for-bit the same outputs, and gradients equal to 6e-17.

### 11. Convert build results on the host

- **What:** the build step converts its lists to JAX arrays through NumPy (`device_array`) and reads the parameter vector as a NumPy array.
- **Why:** `jnp.array(list, dtype=...)` and indexing a JAX array each ran, and at first use compiled, a small device program. That was 36 more small compilations.
- **Effect:** the build step went from 0.42 s to 0.11 s on the Codespace. Outputs are bit-for-bit the same.

### 12. Do the work in NumPy, keep JAX as the interface

- **What:** `differometor/host.py` computes the simulation with NumPy: the physics functions, the assembly of the systems, the carrier solve and the reverse pass for the gradient. `simulate` is still a JAX function with the same arguments and results, and it works under `jax.jit`, `jax.grad` and `jax.vjp`. Inside, JAX now only sees two calls into NumPy, the sparse solves and a handful of array operations.
- **Why:** after changes 1–11, most of a first run was JAX tracing and compiling the physics functions and their derivatives. NumPy needs no compiling, so that cost disappears.
- **How the gradient is obtained without JAX:**
  - The physics functions are the same code. They run on a small `Dual` type, a number that carries its own derivatives through each arithmetic step, which yields their derivatives automatically (checked against JAX for all 19 functions).
  - The reverse pass through the assembly and the solves is written out by hand in `_backward`. Each forward solve needs one adjoint solve, so the gradient stage now does two sparse solves, not four.
- **Fallback:** the previous implementation is kept as `simulate_jax`. It is used when `differometor.simulate.HOST_SIMULATION` is set to `False`, or when a setup has linking functions other than the identity, which the NumPy version does not differentiate.
- **Limit:** the NumPy version supports reverse-mode differentiation only. Forward mode (`jax.jvp`, `jax.jacfwd`) raises an error; use the fallback for that.
- **Only the design variables are differentiated.** A gradient with respect to any other input array (for example the swept frequencies) comes back as NaN in the NumPy version, so that it cannot be mistaken for a valid zero; use the fallback for that.
- **Checks, NumPy version against the JAX version:**

  | Setup | Outputs | Gradients for random directions on each output |
  | --- | ---: | ---: |
  | Main example | 5e-10 | 2e-09 |
  | Main example, every parameter shifted by 0.1% (reflectivities above 1) | 2e-12 | 1e-11 |
  | Second UIFO setup (homodyne detector, two signal detectors) | 6e-05 in the signal vector, 4e-15 in the noise | 3e-07 |
  | Second UIFO setup, shifted | 2e-10 | 7e-10 |

- **Hand-derived rules for the 19 physics formulas:** [`docs/component-gradient-rules.pdf`](docs/component-gradient-rules.pdf) derives, in closed form, the derivative of every output of every component function with respect to every input, with the conventions, the singular points and the branch beyond total internal reflection. `python tests/handwritten_jacobians.py` implements those formulas and compares them with JAX at four kinds of input; the worst relative difference is 7e-16. The simulator obtains the same derivatives automatically through `Dual`; the closed forms are an independent derivation.
- **Test file:** `python tests/test_host.py` repeats these checks and adds more. It compares the NumPy version with the JAX version (all outputs, and gradients for random directions on each output) on 13 cases: the two saved UIFO setups, aLIGO, Voyager with balanced homodyne detection, a cavity with a carrier sweep, three randomly generated UIFO layouts (two 2×2 grids and one 3×3), and shifted parameters for four of them. Gradients agree to 3e-07 or better in every case. It also checks `jit`, `vmap`, the switch to the fallback for linking functions, and finite differences for one design variable of every kind (reflectivity, tuning, mass, power, length, squeezing level, squeezing angle and a group of tied lengths), which agree with the gradient to 0.5% or better. Two situations that no public setup has are built artificially from aLIGO: a second noise detector, and one cell of each system written a second time by another formula. In both, the NumPy and JAX versions agree (gradients to 1e-10) and the gradient matches finite differences in random directions to 0.3% or better.

  Against the original code: Voyager, aLIGO and the cavity sweep agree as before (sensitivity to 5e-11 and 1e-11, cavity to 1e-14), and the official benchmark's saved arrays agree to 1.2e-07 (signal) and 9e-08 (gradient). The distance from the public reference is 5.376e-06 (limit 1e-5).

### Changes 8–11 measured one at a time

Official benchmark on the Codespace, one core, each version run three times:

| Version | Total (three runs) | Build | Forward trace | Forward compile |
| --- | ---: | ---: | ---: | ---: |
| Changes 1–7 | 5.61, 5.45, 5.53 s | 0.46 s | 1.05 s | 2.14 s |
| + change 8, without the out-of-range handling | 4.70, 4.49, 4.33 s | 0.45 s | 1.04 s | 1.19 s |
| + change 9 | 3.60, 3.69, 3.50 s | 0.46 s | 0.88 s | 0.70 s |
| + change 10 | 3.30, 3.29, 3.44 s | 0.42 s | 0.80 s | 0.66 s |
| + change 11 | 3.01, 3.01, 3.05 s | 0.11 s | 0.82 s | 0.66 s |
| + out-of-range handling with a complex arcsine | 3.33, 3.38, 3.33 s | 0.11 s | 0.82 s | 0.97 s |
| + `cos_arcsin` in its place (final) | 3.16, 3.30, 3.24 s | 0.11 s | 0.85 s | 0.82 s |

The Codespace was a little slower in this session than when changes 5–7 were measured, which is why the first row reads 5.5 s and not 5.1 s.

### Tried and not adopted

- **`cos(x) + 1j*sin(x)` instead of `exp(1j*x)`:** no measurable difference in compile time (about 3 ms per use either way).
- **Reusing one factorisation for the solves:** change 12 already cut the sparse factorisations from six per frequency to four (two forward, two in the gradient). Going further was measured on the main example (Apple M5): the two forward solves take 12.2 ms when each factors from scratch and 7.4 ms when they share one factorisation. Over both stages that is about 10 ms per evaluation, roughly 3% of the complete first run. It needs the solver's factorisations to be kept in memory that Python does not manage and released by hand, where a mistake means a crash or a leak, not a wrong number. Not adopted.
- **SciPy's sparse solver (SuperLU) in place of KLU** (branch `scipy-host`): it would make klujax unnecessary except for the fallback, and it factors each matrix once per stage. The official benchmark passes, a little slower (Apple M5: 0.42 s against 0.33 s). But `tests/test_host.py` fails on the two setups with a homodyne detector: the noise differs from the KLU result by 2e-05 (second UIFO) and 6e-05 (Voyager), and the Voyager example's sensitivity ends up 7e-03 away from the original code in log, where KLU is 6e-11 away. The matrices there have entries from 5e-20 to 1e+13; SuperLU matches the dense reference solver in the large entries of the solution (1e-09) but is less accurate in the small ones that the noise depends on. Not adopted.

## Step by step, changes 1–3 (Apple M5, official benchmark without CPU pinning)

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
  | Signal | 5.9e-08 |
  | Noise | 1.5e-08 |
  | Sensitivity | 2.6e-09 |
  | Gradient (193 values) | 2.7e-08 |

  The remaining differences come from using a different solver on a matrix that is very sensitive to rounding.
- **Other parameter values:** at three parameter sets with every design variable moved randomly by about 0.1%, original and ours agree to 2e-07 or better in log sensitivity and 5e-08 or better in the gradient (Apple M5).
- **All saved arrays of the official benchmark, original against ours (Apple M5, `--save-arrays`):** carrier and powers agree to 2e-15, sensitivity to 6e-09, noise to 4e-08, signal to 1.2e-07, and all 193 gradient values to 9e-08 of the largest one.
- **A second UIFO setup:** `examples/data/uifo_20_5000.json` has a different layout, two signal detectors, a homodyne detector and 192 design variables, at 101 frequencies from 20 to 5000 Hz. Original and ours agree to 2.5e-07 in log sensitivity and 1e-08 in the gradient. Inside the full signal vector the largest difference is 2.4e-05 of its largest entry; this setup is even more sensitive to solver rounding at low frequencies.
- **Other public setups, compared array by array with the original:** Voyager, which uses balanced homodyne detection (sensitivity to 5e-11), aLIGO (sensitivity to 1e-11, and it still passes its own comparison with Finesse) and the cavity sweep (1e-14).
- **The "original" is the upstream code:** the report of the original run lists file fingerprints identical to those in `UPSTREAM.json`.
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

- Change 12 is new code for the whole reverse pass. It is verified against the JAX version on the setups listed above; a setup that uses a feature none of them has would be new ground, and `HOST_SIMULATION = False` switches back.
- No run yet on the contest's Intel Xeon; the Linux numbers are from two AMD EPYC machines.
- The original could not complete the official benchmark on the 16 GB Codespace, so the official-style "before" on that machine is missing.
- On the AMD EPYC 9334 both versions have saved reports, but their output arrays were not brought back, so there is no array-by-array comparison from that machine. The comparisons on the Mac and the Codespace cover it.
- The carrier system (286×286, solved once) still uses the dense solver.
- `cos_arcsin` returns the same values as the complex arcsine route, but beyond total internal reflection its derivative has the opposite sign in the imaginary part. That only matters when a refractive index or an incidence angle is itself a design variable in such a setup; the benchmark's design variables are reflectivity, tuning, squeezing, angle of squeezing, power, mass and length.
