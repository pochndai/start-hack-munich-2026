# Versions

One section per named version, oldest first. A version is a git tag plus an
entry in [`versions.json`](versions.json); `python bench/run.py add-version`
creates both and appends a stub here. This file doubles as the submission note
("approach, why it helps, what we learned"), so write it for someone who has
not seen the code.

Numbers quoted here come from a named session in
`results/<machine>/sessions/`. The baseline ran once
(`results/<machine>/baseline.json`); speed-ups over it divide that stored total
by a session's samples. Speed-ups between two other versions are per-round
ratios within one session.

## v00-baseline

- **Commit:** `02ec741d8a1b` (unmodified organizer repository; simulator = upstream Differometor `b0dc7c7`)
- **Builds on:** -
- **Summary:** The starting point. Nothing changed.

**Reference numbers from the organizers** (Intel Xeon Platinum 8468, one logical CPU, 32 GB; `benchmarks/cpu-baseline.json`)

| Phase group | Seconds |
| --- | ---: |
| Setup (imports 2.61, build 1.03) | 3.66 |
| Compilation (forward 16.96, gradient 5.55, sensitivity 0.40) | 22.92 |
| Forward simulation + AD residuals | 13.29 |
| Sensitivity + host transfer | 0.00 |
| Gradient | 4.34 |
| **Complete first-run total** | **44.20** |

Warm median 17.68 s. Peak memory about 14.4 GiB.

**Result on our machine**

One run, 21 November 2026, Apple M5 MacBook Air, 16 GB, thread limits set to one (a Mac cannot pin a CPU). Raw data: `results/apple-m5-16g-darwin/baseline.json`, report `results/apple-m5-16g-darwin/official/v00-baseline.json`.

| Phase group | Seconds |
| --- | ---: |
| Setup | 0.55 |
| Compilation | 3.20 |
| Forward simulation + AD residuals | 8.61 |
| Gradient | 4.05 |
| **Complete first-run total** | **16.41** |

Warm median 9.67 s (3 repeats). `PASS`. The process peaked at 14.5 GB on a 16 GB machine with other apps open, so macOS was compressing and swapping: the whole process took 59.7 s of wall time for 16.4 s of timed phases. The seconds are inflated and vary from run to run (earlier hand-made runs on the same Mac gave 10.5 to 13.1 s with all cores, see `RUNS.md`). The one-core Linux comparison in `CHANGES.md` is the one to quote.

## v01-klujax-solver

- **Date:** 2026-11-21
- **Commit:** `25051f259a15` (tag `v01-klujax-solver`)
- **Builds on:** v00-baseline
- **Summary:** Solve the signal system with the sparse KLU solver (klujax)

**What changed**

The three solves of the signal matrix per frequency call `klujax.solve` (the sparse solver KLU) in place of `jnp.linalg.solve`. One new pinned dependency, `klujax==0.5.3`. Change 1 in `CHANGES.md`.

**Why it is faster**

The signal matrix is 751×751 in the example, but only 0.46% of its entries can be non-zero. A dense solver works on all of them, for every frequency, three times. KLU only touches the non-zero ones. The positions are read from the setup in the build step, so nothing is tied to the example.

**Why it is still correct**

`python bench/validate.py v01-klujax-solver` (Apple M5, `results/apple-m5-16g-darwin/validation/v01-klujax-solver@25051f259a15.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01).

**The one gradient value.** The example's gradient spans thirteen orders of magnitude: the largest entry is 5.8e+04, and entry 158 is 4.9e-04, a hundred-millionth of that. It belongs to a mirror with reflectivity 4.5e-06 in front of a laser of 1e-12 W. On this entry the original and ours differ by 0.03%; on the other 192 they agree within 0.01%. What we know: (1) every version from v01 to v07 gives the same value to 8 digits, although v01–v06 differentiate with JAX and v07 with a hand-written reverse pass; (2) the experiment with SciPy's SuperLU solver, a third independent solver, gave -4.890100920e-04, also the same to 8 digits; (3) only the original's dense solver gives -4.8885e-04. That points to rounding in the dense solve of a badly scaled matrix, not to an error in ours. (4) the original's own value moves in the fifth digit (-4.888495e-04 to -4.888760e-04) when its loss is shifted by a constant, which leaves the true gradient unchanged, while ours stays the same to 7 digits (`tests/variations.py`, case `supplied`). What we do not know: a finite-difference check cannot settle it, because the loss changes by only 1e-9 over the usable step and both codes are too noisy at that level (estimates scatter between -3e-04 and -9e-04).

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 7.07 s, warm median 3.32 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

The carrier system (286×286, solved once) stays dense; it is a small part of the run.

## v02-group-by-function

- **Date:** 2026-11-21
- **Commit:** `59705c9af4bb` (tag `v02-group-by-function`)
- **Builds on:** v01-klujax-solver
- **Summary:** Trace each component function once, only for the entries that use it

**What changed**

The build step hands `simulate` the list of which physics function fills which matrix entry as build-time data (the small `Static` class), and `evaluate()` groups the entries by function. Change 2 in `CHANGES.md`.

**Why it is faster**

Before, the function was picked with `lax.switch` inside `vmap` from a value JAX could not see while compiling, so all 19 physics functions were compiled for every entry, three times. Now each is compiled once, for the entries that use it.

**Why it is still correct**

Outputs are bit-for-bit those of v01. `python bench/validate.py v02-group-by-function` (Apple M5, `results/apple-m5-16g-darwin/validation/v02-group-by-function@59705c9af4bb.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01).

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 5.36 s, warm median 3.28 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

None.

## v03-sparse-assembly

- **Date:** 2026-11-21
- **Commit:** `7938724845d8` (tag `v03-sparse-assembly`)
- **Builds on:** v02-group-by-function
- **Summary:** Assemble the signal and noise systems from their non-zero entries only

**What changed**

`simulate` no longer builds the full signal and noise matrices for each frequency. `sparse_layout()` in the build step numbers the entries that can be non-zero, and `simulate` fills, scales, conjugates and solves on that list. Change 3 in `CHANGES.md`.

**Why it is faster**

After v01 the solver was fast, but the code still filled 57 million numbers per matrix copy to use 0.5% of them, and the gradient had to keep all of it. The warm run fell from 3.3 s to 0.08 s and peak memory from several GB to under 1 GB.

**Why it is still correct**

`python bench/validate.py v03-sparse-assembly` (Apple M5, `results/apple-m5-16g-darwin/validation/v03-sparse-assembly@7938724845d8.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01).

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 2.30 s, warm median 0.08 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

None.

## v04-fast-compile-flags

- **Date:** 2026-11-21
- **Commit:** `73680c8eb289` (tag `v04-fast-compile-flags`)
- **Builds on:** v03-sparse-assembly
- **Summary:** Lower the XLA optimisation level and add three fast-compile flags

**What changed**

`differometor/__init__.py` adds four settings to `XLA_FLAGS` before JAX starts, unless the user set them: optimisation level 0, no expensive LLVM passes, and, for jaxlib 0.9 only, the simpler code generator and no parallel code splitting. Change 4 in `CHANGES.md`.

**Why it is faster**

After v03, compiling is about 85% of a first run and the compiled program runs in a fraction of a second. Code that is quick to produce matters more than code that is quick to run. The warm run gets slower (0.08 s to 0.12 s here), which is not scored.

**Why it is still correct**

`python bench/validate.py v04-fast-compile-flags` (Apple M5, `results/apple-m5-16g-darwin/validation/v04-fast-compile-flags@73680c8eb289.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01). The flags are recorded in every report under `environment.xla_flags`.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 1.63 s, warm median 0.12 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

XLA stops with an error on a flag it does not know, so the two version-specific flags are only added when jaxlib is a 0.9 release.

## v05-two-solves-whole-arrays

- **Date:** 2026-11-21
- **Commit:** `0a42262cb682` (tag `v05-two-solves-whole-arrays`)
- **Builds on:** v04-fast-compile-flags
- **Summary:** Drop the redundant noise solve, call each function on whole arrays, build constants with NumPy

**What changed**

Three changes (5, 6 and 7 in `CHANGES.md`): the third signal solve per frequency is removed, each physics function is called once on whole arrays instead of through `vmap`, and the build step fills its constants with NumPy.

**Why it is faster**

The third solve computed `Re(cᴴA⁻¹Nw)` with `c = Aᴴw`, which equals `Re(wᴴNw)` and needs no solve (derivation in `docs/third-solve-algebra.pdf`). Tracing through `vmap` is slow in Python. The build step ran 65 small device compilations for plain scalar arithmetic.

**Why it is still correct**

`python bench/validate.py v05-two-solves-whole-arrays` (Apple M5, `results/apple-m5-16g-darwin/validation/v05-two-solves-whole-arrays@0a42262cb682.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.9e-08 (relative to the largest value), sensitivity 6.2e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01). Nothing is skipped or stored: both systems are assembled and solved for every frequency.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 1.46 s, warm median 0.09 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

Each of the three was measured on its own on the Linux Codespace (table "Changes 5–7 measured one at a time" in `CHANGES.md`).

## v06-real-parameters

- **Date:** 2026-11-21
- **Commit:** `490a15f91da7` (tag `v06-real-parameters`)
- **Builds on:** v05-two-solves-whole-arrays
- **Summary:** Real parameters, compact function outputs, shared mirror terms, host-side build

**What changed**

Four changes (8 to 11 in `CHANGES.md`): parameters are stored as real numbers, functions return only the coefficients that are read, the mirror function shares its common terms, and build results are converted to arrays on the host.

**Why it is faster**

Complex parameters turned every sine, cosine and square root into complex arithmetic, which is far more code to compile and differentiate. Most function outputs were padding zeros that were stacked, differentiated and never read.

**Why it is still correct**

`python bench/validate.py v06-real-parameters` (Apple M5, `results/apple-m5-16g-darwin/validation/v06-real-parameters@490a15f91da7.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 4.0e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01). Square roots of reflectivity, transmissivity and power are still taken as complex, so a reflectivity slightly above 1 gives the same result as in the original.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 0.96 s, warm median 0.07 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

`cos(x) + 1j*sin(x)` in place of `exp(1j*x)` made no measurable difference. A first version without the complex square roots returned NaN for reflectivities above 1; the shifted-parameter check caught it.

## v07-numpy-host

- **Date:** 2026-11-21
- **Commit:** `3d6516f91829` (tag `v07-numpy-host`)
- **Builds on:** v06-real-parameters
- **Summary:** Do the simulation in NumPy behind one JAX callback, with a hand-written reverse pass

**What changed**

`differometor/host.py` does the simulation in NumPy: the physics functions, the assembly, the carrier solve and the reverse pass for the gradient. `simulate` is still a JAX function with the same arguments; inside, JAX sees two calls into NumPy and the sparse solves. The previous implementation stays as `simulate_jax` and is used for setups with linking functions. Change 12 in `CHANGES.md`.

**Why it is faster**

After v06 most of a first run was JAX tracing and compiling the physics functions and their derivatives. NumPy needs no compiling. The gradient stage also needs two sparse solves where it needed four.

**Why it is still correct**

`python bench/validate.py v07-numpy-host` (Apple M5, `results/apple-m5-16g-darwin/validation/v07-numpy-host@3d6516f91829.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 4.0e-08 (relative to the largest value), sensitivity 5.8e-08 in log (harness limit 1e-6). One gradient value out of 193 is outside the harness tolerance: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original, 1.5 times the allowed difference, so the harness verdict is `FAIL` until the team decides on it (see "The one gradient value" under v01). The reverse pass is new hand-written code; `tests/test_host.py` compares it with the JAX version on 15 cases and `tests/handwritten_jacobians.py` checks the 19 derivative rules.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 0.36 s, warm median 0.05 s. Shuffled-round session: not run yet, because `python bench/run.py full` leaves out versions whose validation is `FAIL`.

**Dead ends and notes**

SciPy's SuperLU in place of KLU passed the official benchmark but was 7e-03 off on the Voyager setup: not adopted. Sharing one factorisation between solves was worth about 3% of a first run and needs memory managed by hand: not adopted. Forward-mode differentiation is not supported on this path.
