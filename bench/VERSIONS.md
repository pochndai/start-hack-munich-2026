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

`python bench/validate.py v01-klujax-solver` (Apple M5, `results/apple-m5-16g-darwin/validation/v01-klujax-solver@25051f259a15.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01).

**The one gradient value.** The example's gradient spans thirteen orders of magnitude: the largest entry is 5.8e+04, and entry 158 is 4.9e-04, a hundred-millionth of that. It belongs to a mirror with reflectivity 4.5e-06 in front of a laser of 1e-12 W. On this entry the original and ours differ by 0.03%; on the other 192 they agree within 0.01%. What we know: (1) every version from v01 to v07 gives the same value to 8 digits, although v01–v06 differentiate with JAX and v07 with a hand-written reverse pass; (2) the experiment with SciPy's SuperLU solver, a third independent solver, gave -4.890100920e-04, also the same to 8 digits; (3) only the original's dense solver gives -4.8885e-04. (4) the original's own value moves in the fifth digit (-4.888495e-04 to -4.888760e-04) when its loss is shifted by a constant, which leaves the true gradient unchanged, while ours stays the same to 7 digits; (5) the decisive check: `python tests/refined_solves.py supplied` repeats the calculation with every sparse solve refined to a high-accuracy reference (iterative refinement with residuals computed in 100-digit arithmetic, stopped when the correction is below 1e-20 of the solution). This is strong numerical evidence, not a formal proof that every digit is right. The refined value is -4.890100922e-04. Ours is 1e-08 away from it, the original 3e-04. With refined solves no gradient entry of ours is off by more than 0.01% and one of the original's is. **The deviation is rounding in the original's dense solver; the harness reports it as a failure of ours because its reference arrays come from the original.** Finite differences could not settle this (the loss changes by only 1e-9 over the usable step), the refined solves can.

**Decision (21 November 2026).** The team raised the harness's per-entry gradient limit from 0.01% to 0.1% of the entry (`gradient_rtol` in `bench/validate.py`). A second limit now also applies (largest difference relative to the largest entry below 1e-6, `gradient_scale_rel`). All seven versions were validated again with these limits (`python bench/validate.py <version> --force`, Apple M5, 21 November, 21:50 UTC): `PASS` for `v01` to `v07`. Entry 158 is at 0.29 of the allowed difference, and the largest difference relative to the largest entry is between 6.6e-08 and 9.2e-08.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 7.07 s, warm median 3.32 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 8.62 s, fastest 8.12 s; 1.90x (1.55 to 2.02) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: -. The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

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

Outputs are bit-for-bit those of v01. `python bench/validate.py v02-group-by-function` (Apple M5, `results/apple-m5-16g-darwin/validation/v02-group-by-function@59705c9af4bb.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01).

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 5.36 s, warm median 3.28 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 6.86 s, fastest 5.97 s; 2.39x (2.25 to 2.75) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: 1.40x (1.16 to 1.54). The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

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

`python bench/validate.py v03-sparse-assembly` (Apple M5, `results/apple-m5-16g-darwin/validation/v03-sparse-assembly@7938724845d8.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01).

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 2.30 s, warm median 0.08 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 2.68 s, fastest 2.64 s; 6.12x (6.02 to 6.21) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: 2.56x (2.22 to 2.71). The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

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

`python bench/validate.py v04-fast-compile-flags` (Apple M5, `results/apple-m5-16g-darwin/validation/v04-fast-compile-flags@73680c8eb289.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.8e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01). The flags are recorded in every report under `environment.xla_flags`.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 1.63 s, warm median 0.12 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 1.69 s, fastest 1.65 s; 9.74x (8.28 to 9.96) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: 1.58x (1.35 to 1.63). The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

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

`python bench/validate.py v05-two-solves-whole-arrays` (Apple M5, `results/apple-m5-16g-darwin/validation/v05-two-solves-whole-arrays@0a42262cb682.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 3.9e-08 (relative to the largest value), sensitivity 6.2e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01). Nothing is skipped or stored: both systems are assembled and solved for every frequency.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 1.46 s, warm median 0.09 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 1.79 s, fastest 1.49 s; 9.15x (8.94 to 11.03) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: 0.94x (0.90 to 1.33), not distinguishable from noise on this machine. The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

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

`python bench/validate.py v06-real-parameters` (Apple M5, `results/apple-m5-16g-darwin/validation/v06-real-parameters@490a15f91da7.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 4.0e-08 (relative to the largest value), sensitivity 6.1e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01). Square roots of reflectivity, transmissivity and power are still taken as complex, so a reflectivity slightly above 1 gives the same result as in the original.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 0.96 s, warm median 0.07 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 1.27 s, fastest 0.92 s; 12.93x (12.03 to 17.89) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: 1.34x (1.17 to 1.94). The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

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

`python bench/validate.py v07-numpy-host` (Apple M5, `results/apple-m5-16g-darwin/validation/v07-numpy-host@3d6516f91829.json`): official benchmark `PASS`, distance from the public reference 5.37e-06 (limit 1e-5). Against the original's arrays: carrier and powers identical, signal 1.2e-07, noise 4.0e-08 (relative to the largest value), sensitivity 5.8e-08 in log (harness limit 1e-6). Harness verdict: `PASS` (rerun at 21:50 UTC, after the team set the per-entry gradient tolerance to 0.1%). At the earlier tolerance of 0.01% it was `FAIL` because of one value out of 193: entry 158 (`m01.reflectivity`) is -4.890101e-04 here and -4.888495e-04 in the original. The cause is rounding in the original, not in this version (see "The one gradient value" under v01). The reverse pass is new hand-written code; `tests/test_host.py` compares it with the JAX version on 15 cases and `tests/handwritten_jacobians.py` checks the 19 derivative rules.

**Result**

Validation run, one sample (Apple M5, 16 GB, thread limits set to one): complete first run 0.36 s, warm median 0.05 s. Session `results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json` (5 shuffled rounds, same machine and limits): median 0.50 s, fastest 0.35 s; 33.11x (23.97 to 46.83) against the stored baseline run of 16.41 s, which was memory-pressured, so that factor is overstated; against the version before it: 2.03x (1.91 to 3.81). The noise floor of this machine in that hour was ±16% (`results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`).

**Dead ends and notes**

SciPy's SuperLU in place of KLU passed the official benchmark but was 7e-03 off on the Voyager setup: not adopted. Sharing one factorisation between solves was worth about 3% of a first run and needs memory managed by hand: not adopted. Forward-mode differentiation is not supported on this path.

## v08-no-large-pages

**Withdrawn on 22 November 2026: this version is not the submitted code.** The submitted simulator is `v07-numpy-host`. The switch costs 4% on a quiet machine and its benefit showed in one of two test sessions; the decision and the way to reinstate it are in `CHANGES.md` ("Withdrawn"). The entry is disabled in `versions.json`, so timing sessions leave it out; the tag stays.

- **Date:** 2026-11-21
- **Commit:** `487e6efc9f04` (tag `v08-no-large-pages`)
- **Builds on:** v07-numpy-host
- **Summary:** Switch off NumPy's request for large memory pages, for a steady first-run time on a busy Linux machine

**What changed**

At import the package switches off NumPy's request for large memory pages (`numpy._core.multiarray._set_madvise_hugepage(False)` in `differometor/__init__.py`), unless the user has set `NUMPY_MADVISE_HUGEPAGE` themselves. No calculation changes. Change 13 in `CHANGES.md`.

**Why it is faster**

It is not faster; it is steadier. NumPy asks Linux for 2 MB pages for every array above 4 MB. On a host whose memory other programs keep busy, the kernel assembles those pages inside our process when an array is first touched. In the test on the AMD cloud (`results/amd-epyc-9334-32-core-processor-504g-linux/diagnosis/`, 96 runs of the unchanged benchmark, the request toggled with the equivalent environment variable) a step of the NumPy path took more than twice its normal time in 8 of 24 runs with the request on and in 0 of 8 with it off. On a quiet machine the complete first run is 0.74 s with the request off against 0.71 s with it on. The scored number is a single run, so an outlier costs more than 0.03 s.

**Why it is still correct**

Only where memory comes from changes, so the outputs are bit-for-bit those of v07. `python bench/validate.py v08-no-large-pages` (Apple M5): `PASS`, with the same differences from the original as v07 (sensitivity 5.8e-08 in log, gradient 9.2e-08 of the largest entry, entry 158 at 0.29 of the allowed difference). `tests/test_host.py`: `ALL CHECKS PASSED`. `tests/handwritten_jacobians.py`: all 19 rules agree. `tests/variations.py`: 18 `PASS` and the one known `DEVIATION` on 19 scenarios.

**Result**

On the Mac the switch has no effect (macOS has no such request), so there is no Mac timing to quote: the validation run took 0.37 s, like v07. A second AMD test used the switch itself (`results/amd-epyc-9334-32-core-processor-504g-linux/diagnosis-change-13/`, 72 runs, one pinned core): quiet machine 0.74 s with the request off against 0.71 s with it on (median of 24 each); next to memory-churning processes 1.16 s against 1.11 s (median of 12 each). No large pages were handed to the process with the switch off, so it works. The outliers this version is meant to prevent did not occur with either setting in that session, so its benefit rests on the first test alone (8 of 24 runs against 0 of 8).

**Dead ends and notes**

The switch is private to NumPy and exists in the pinned 2.5.5; if it is missing the call is skipped. Setting the environment variable inside the package would be too late, because `benchmark.py` imports NumPy before this package. Not shown: the worst state we observed (0.74 s becoming 2.8 s on a busy shared host) did not occur again during the test, so this version has not been measured against it. Two other settings were tried in the same test and did not remove the outliers: keeping freed memory in the allocator, and pinning to a different CPU.

## v09-active-derivatives

- **Date:** 2026-11-22
- **Commit:** `9c1d7f3351e3` (tag `v09-active-derivatives`)
- **Builds on:** v07-numpy-host
- **Summary:** Take derivatives of the physics formulas only for inputs that depend on a design variable

**What changed**

In `differometor/host.py`, `_assemble` marks which entries of the parameter vector depend on the design variables: the entries the optimized values are written into, carried through the parameter links by the same two assignments that carry the values. `_evaluate` then asks a physics formula for derivatives only with respect to those of its seven inputs that read a marked entry somewhere. A formula none of whose inputs is marked runs on plain numbers, with no derivative at all. Change 14 in `CHANGES.md`.

**Why it is faster**

Before, every formula carried seven derivatives through every arithmetic step, one per input. Most of them were never used: the frequency and the fixed refractive indices are not design variables, and whole formulas (force calculations, signal sources, vacuum and loss noise) have no design variable among their inputs. Their derivatives were computed and then thrown away at the end of the reverse pass, where only the design variables' entries are read.

**Why it is still correct**

No requested derivative is skipped: all 193 gradient entries are computed as before. What is left out is arithmetic whose result was discarded. Each remaining derivative goes through exactly the same operations, so the gradient is bit-for-bit the same: checked on the supplied setup, the second UIFO setup (homodyne), aLIGO, Voyager and a random 3×3 layout, for all outputs and for gradients with random weights on every output (20 arrays compared byte by byte, script not in the repository). `python bench/validate.py v09-active-derivatives` (Apple M5, `results/apple-m5-16g-darwin/validation/v09-active-derivatives@9c1d7f3351e3.json`): `PASS`, with the same differences from the original as v07 (sensitivity 5.8e-08 in log, entry 158 at 0.29 of the allowed difference). `tests/test_host.py`: `ALL CHECKS PASSED` (it includes the tied lengths, which are parameter links). The rule is general: the marks come from the setup's own index arrays, nothing is tied to the supplied example.

**Result**

Session `results/apple-m5-16g-darwin/sessions/2026-11-21T23-38-07Z_run.json` (Apple M5, 16 GB, thread limits set to one, 10 shuffled rounds of `v07`, `v09` and `v10`, a fresh process per run; the Mac was busy with a running demo page, so 2 to 3 samples per version carry the `memory` flag): the gradient phase of a first run has a median of 30.6 ms against 38.3 ms for v07. The complete first run, 0.43 s against 0.44 s (medians), is not distinguishable from noise there: 1.01x (0.94 to 1.37) against v07. Compiled and warm, one forward evaluation plus gradient of the supplied setup takes 38.6 ms against 44.1 ms (median of 30 in one process, by hand). One pinned Linux core (GitHub Codespace, AMD EPYC 7763, `results/amd-epyc-7763-64-core-processor-16g-linux/sessions/2026-11-22T00-14-30Z_run.json`, 10 shuffled rounds of `v07`, `v09` and `v10`): complete first run 0.999 s against 1.026 s for v07 (medians), 1.02x (0.99 to 1.05), not distinguishable from noise on its own; gradient phase 109.9 ms against 130.5 ms.

**Dead ends and notes**

The marks are per input of a group of cells that share a formula, not per cell: an input is differentiated for the whole group if any cell of the group needs it. Going down to single cells would mean calling a formula several times on parts of its cells; not tried. Precomputing the index bookkeeping of `_evaluate` once per setup was profiled and not done: it is 1 to 2 ms per evaluation on the Apple M5.

## v10-host-swept-values

- **Date:** 2026-11-22
- **Commit:** `49f887148515` (tag `v10-host-swept-values`)
- **Builds on:** v09-active-derivatives
- **Summary:** Reshape and copy the swept values in NumPy, which removes two small compilations from the build step

**What changed**

`prepare_arrays` in `differometor/build.py` receives the swept values (in the benchmark: the 101 frequencies) as a JAX array, gives them the shape (1, 101) and copies them. Both steps now happen in NumPy and the result is turned into a JAX array once. Change 15 in `CHANGES.md`.

**Why it is faster**

On a JAX array, `reshape` and `copy` each run a small device program, and JAX compiles such a program the first time it meets a new shape. With compilation logging switched on (`JAX_LOG_COMPILES=1`) the build step showed five small compilations; these two were ours, the other three come from `benchmark.py` itself. Same idea as change 11, which removed 36 of them; these two were left over.

**Why it is still correct**

The values are the same numbers in the same shape and type; only where the reshape and the copy happen has changed. All outputs and gradients are bit-for-bit those of v09 on the five setups named there. `python bench/validate.py v10-host-swept-values` (Apple M5, `results/apple-m5-16g-darwin/validation/v10-host-swept-values@49f887148515.json`): `PASS`, same differences from the original as v07. `tests/test_host.py`: `ALL CHECKS PASSED`; `tests/handwritten_jacobians.py`: all 19 rules agree; `tests/variations.py` (session `results/apple-m5-16g-darwin/variations/2026-11-21T23-37-37Z.json`): 18 `PASS` and the one known `DEVIATION` (20 to 5000 Hz) on 19 scenarios, as for v07. Swept values that are NumPy arrays, or values being traced by JAX, take the same path as before.

**Result**

Same session as v09 (`results/apple-m5-16g-darwin/sessions/2026-11-21T23-38-07Z_run.json`): the build phase has a median of 37.7 ms against 41.8 ms for v09; the complete first run is not distinguishable from noise, 1.00x (0.77 to 1.75) against v09. Three runs of `benchmark.py` by hand on a quieter machine gave 30.7, 30.9 and 30.7 ms for the build phase, against 41.4 to 42.6 ms in three runs before the change. One pinned Linux core (GitHub Codespace, AMD EPYC 7763, `results/amd-epyc-7763-64-core-processor-16g-linux/sessions/2026-11-22T00-14-30Z_run.json`, 10 shuffled rounds of `v07`, `v09` and `v10`): complete first run 0.976 s against 0.999 s for v09 and 1.026 s for v07 (medians); build phase 98.8 ms against 117.5 ms for v09. Against v07, v09 and v10 together are 1.04x (1.03 to 1.08): faster in every round. `bench/validate.py v10-host-swept-values` on that machine: `PASS` (`results/amd-epyc-7763-64-core-processor-16g-linux/validation/`).

**Dead ends and notes**

The rest of the build step was profiled (Apple M5, about 10 ms once warm): 3.5 ms in `set_instructions`, 2 ms scanning the dense matrices for non-zero cells in `sparse_layout`. Not changed: a few milliseconds at most for edits to code that decides the sparsity pattern. Importing klujax takes 12 ms, of which 10 ms is `jaxtyping`, which klujax needs only for one type annotation; replacing that import with a stand-in was measured (12 ms to 2 ms) and not done, because it means interfering with another package's imports.

## v11-scipy-stacked-solver

- **Date:** 2026-11-22
- **Commit:** `f3f14542ea40` (tag `v11-scipy-stacked-solver`)
- **Builds on:** v10-host-swept-values
- **Summary:** Solve with SciPy's sparse LU inside the host callbacks: all frequencies in one factorisation, one factorisation per evaluation

**What changed**

On the NumPy path the sparse signal systems are now solved with SciPy's sparse LU (SuperLU) instead of klujax. SciPy was already in the organizers' `requirements.txt`, so no package is added. Changes 16 to 18 in `CHANGES.md`. In `differometor/host.py`:

- **All frequencies in one matrix.** The signal matrices of all frequencies are stacked into one block-diagonal matrix and factorised in a single call (`_Factors`). The column order is found once per sparsity pattern (`_solver_layout`: SuperLU's own ordering of the middle frequency's matrix) and reused for every frequency and every later evaluation.
- **One factorisation per evaluation**, where v10 factorised four times per frequency (two solves in the forward pass, two in the gradient). The forward pass now also does the noise's adjoint solve, which is a vector known there times a real factor known only later. The signal's adjoint is a combination of the noise weights whenever the signal is read at the noise detector; `_signal_adjoint` checks that at run time and factorises again if it does not hold.
- **One step of iterative refinement after every solve** (`REFINEMENT_STEPS`).
- **Forward and gradient are one host callback each**, so no solver is left in the JAX program. The dense signal and noise matrices no longer travel into the callbacks: `sparse_layout` in `simulate.py` carries the entries the host needs (`base`). The arrays kept for the reverse pass shrink from 26.7 MB to 5.2 MB on the supplied setup.
- **Imports.** SuperLU's compiled module is loaded from its file, without `import scipy.sparse.linalg` (`_load_superlu`); the jaxlib version is read from jaxlib itself; plotly is loaded when `visualize_setup` is first asked for (`differometor/__init__.py`); klujax is imported only when the JAX fallback runs.

**Why it is faster**

Three phases of a first run shrink (one pinned core of the Intel Xeon Platinum 8468, medians, session named under "Result"): tracing and compiling the forward and gradient functions takes 66 ms where v10 needs 158 ms, because the solver is no longer part of the program JAX compiles; the gradient run takes 58 ms against 109 ms, because it no longer factorises; imports take 313 ms against 335 ms. The forward run gets slower, 112 ms against 87 ms: it now carries the one factorisation, three solves and their refinement.

**Why it is still correct**

Every frequency and every design variable is still computed, in double precision; only the solver and the order of the solves changed. The numbers are not bit-for-bit those of v10 (a different solver). All on the Intel Xeon VM, commit `f3f1454`, logs in `results/intel-xeon-platinum-8468-197g-linux/no-klujax-scipy-logs/`:

- `python bench/validate.py` (`results/intel-xeon-platinum-8468-197g-linux/validation/no-klujax-scipy@f3f14542ea40.json`): `PASS`. Against the original's arrays from the same VM: signal 1.3e-07, noise 4.0e-08, sensitivity 6.8e-08 in log, gradient 9.1e-08 of its largest entry, 0 of 193 gradient entries outside the tolerance. Distance from the public reference: 5.374e-06 in log (limit 1e-5), as for the original.
- `tests/test_host.py`: `ALL CHECKS PASSED` (NumPy path against JAX path on 13 cases, `jit`, `vmap`, finite differences). `tests/handwritten_jacobians.py`: all 19 rules agree.
- `tests/variations.py`, 19 scenarios against the original (`results/intel-xeon-platinum-8468-197g-linux/variations/2026-11-22T00-23-50Z.json`): 18 `PASS`, 1 `DEVIATION`, 0 `FAIL`. The deviation is `band-20-5000`, the one known since v07. v10 in the same hour: 17 `PASS`, 2 `DEVIATION`; its second one (a signal difference of 1.6e-06 on `uifo-20-5000`) does not occur here, where that scenario agrees with the original to 5e-11 in log sensitivity.
- `benchmark_aligo.py`: `PASS` in all four runs.
- On the Apple M5 (a teammate's session, from a separate checkout): `validate.py` `PASS`, `tests/test_host.py` all passed, `tests/variations.py` 18 `PASS` and the known `DEVIATION`.

Nothing is tied to the supplied setup: the column order, the stacked layout and the check behind the one-factorisation shortcut are all derived from the setup the code is given, and a setup whose signal is read elsewhere takes the second factorisation.

**Result**

The sessions on the Xeon VM list this version under its branch name, `no-klujax-scipy`; it is the same commit, `f3f1454`. Session `results/intel-xeon-platinum-8468-197g-linux/sessions/2026-11-22T00-14-25Z_run.json` (Intel Xeon Platinum 8468, pinned to logical CPU 8 with the official `--cpus 1`, 32 GiB memory cap, swap off, 5 shuffled rounds, a fresh process per run):

| Version | Median first run | Fastest | Slowest | Against the stored original (43.14 s) |
| --- | ---: | ---: | ---: | ---: |
| `v07-numpy-host` | 0.88 s | 0.85 s | 0.91 s | 49x |
| `v10-host-swept-values` | 0.84 s | 0.82 s | 0.87 s | 51x |
| `v11-scipy-stacked-solver` | 0.71 s | 0.68 s | 0.72 s | 61x |

Against v10 within each round: 1.20x (1.17 to 1.21), faster in all five rounds. Against v07: 1.24x (1.22 to 1.29). The harness flagged a slower control workload in 2 to 3 of 5 rounds for every version, and the noise floor of this VM has not been measured; the per-round ratios are the evidence. An earlier session the same night, before the merge with v10 (`sessions/2026-11-22T00-08-43Z_run.json`), gave 0.67 s against 0.85 s for v07 (1.27x, 1.26 to 1.29).

Two other setups on the same VM, by hand (`CHANGES.md`, "Changes 16 to 18 measured"): aLIGO 0.50 s against 0.62 s for v10, `random-layout-1` 0.69 s against 0.83 s. Apple M5, thread limits set to one, 10 shuffled rounds, machine busy (`results/apple-m5-16g-darwin/sessions/2026-11-22T00-31-37Z_run.json`): 0.35 s against 0.41 s for v10, 1.20x (0.93 to 1.54), faster in 9 of 10 rounds.

**Dead ends and notes**

- The factorisation calls SciPy's internal `gstrf` and loads its module from a file. Both exist in the pinned `scipy==1.18.1`. If either fails, the code falls back to the normal import and then to `splu`, which is slower but gives the same results.
- klujax stays a requirement, for the JAX fallback (setups with linking functions). That path is not faster than in v10: 2.57 s against 2.47 s on `amplitude-modulation`, one run each.
- Switching the refinement step off saves 13 ms of a first run (658 to 645 ms, VM, four alternating runs) and costs accuracy on the hardest scenario (`uifo-20-5000`: 5e-11 becomes 2e-08 in log sensitivity against the original). Kept on.
- Tried and not adopted, with the measurements in `docs/no-klujax-scipy.md`: Krylov methods (no convergence in 2,000 iterations), a banded LAPACK solver (three times slower), scaling rows and columns without refinement, plain SuperLU without refinement (up to 5e-04 off on Voyager), loading SciPy's sparse-product module directly (49 ms to load).
- Known and unchanged from v07: on one pinned CPU, starting the gradient before the previous JAX call has finished hangs inside the host callback. `benchmark.py` waits after every phase and is not affected.
- The harness marks this version and v10 `unofficial-benchmark` because their `benchmark.py` is the organizers' newer file (with aLIGO), which differs from the one at the baseline commit. Nobody on the team edited it.

## v12-kept-assembly

- **Date:** 2026-11-22
- **Commit:** `f4f5c40d53d8` (tag `v12-kept-assembly`)
- **Builds on:** v11-scipy-stacked-solver
- **Summary:** Refine only the noise weights' solve, keep the forward pass's assembled systems for the reverse pass, and reuse freed memory on Linux

**What changed**

Three changes on top of v11, changes 19 to 21 in `CHANGES.md`:

- **Refinement only where it is needed** (`0960d15`, `differometor/host.py`). v11 refined every sparse solve once. The solve for the noise weights keeps that step (`REFINEMENT_STEPS`); the solve that gives the signal and the noise's adjoint is now a plain solve (`SIGNAL_REFINEMENT_STEPS = 0`). The column order of the factorisation is taken from SuperLU's permutation by its documented convention, so the two trial factorisations that chose between the two readings are gone.
- **The forward pass keeps its assembled systems for the reverse pass** (`f1bc7a7`). The reverse pass used to assemble the carrier, signal and noise systems a second time to get the derivatives of the component functions. A forward pass with a pullback now assembles them with those derivatives and keeps them on the host (`_keep`, at most two); JAX carries a ticket. A reverse pass whose ticket is gone assembles again. The signal entries are conjugated in place and no longer copied.
- **On Linux, freed memory is kept for reuse** (`f4f5c40`, `differometor/__init__.py`): three `mallopt` settings of the C library at import, unless the user has set the corresponding `MALLOC_*` variables or `GLIBC_TUNABLES`.

**Why it is faster**

The first forward call does one triangular solve and one matrix product less per right-hand side and two small factorisations less. The gradient call no longer evaluates the component functions and rebuilds three systems. On Linux, glibc returned every freed array of a megabyte or more to the operating system and fetched the next one fresh, so each was paid for again in page faults; the gradient call, which runs after the forward call has freed its arrays, gains most from keeping them.

**Why it is still correct**

- The second change is bit-for-bit: all outputs and gradients of five setups (supplied, second UIFO, aLIGO, Voyager, a random 3x3 layout) are byte-identical before and after it (20 arrays). The third changes only where memory comes from.
- The first change alters rounding. Against v11 the gradients for random weights move by at most 1.5e-09 of their largest entry on those five setups. In `tests/test_host.py` (NumPy path against the JAX path) the noise is unchanged from v11 (5e-10 on the supplied setup, 1e-12 and 2e-13 on the two homodyne setups), the gradients are at the same level (6e-08 on the second UIFO), and the signal of the random layouts is within 1.4e-10 where v11 had 2e-15; the judges' limit is 1e-5 in log sensitivity.
- Apple M5, on the state with the first two changes: `bench/validate.py WORKTREE` `PASS` (sensitivity 6.1e-08 in log from the original, entry 158 at 0.29 of the allowed difference, 5.374e-06 from the public reference); `tests/test_host.py` `ALL CHECKS PASSED`, including a new case in which a pullback is used after three later forward passes; `tests/refined_solves.py supplied`: 0 gradient entries off by over 0.1%; `tests/variations.py`: 18 `PASS` and the one known `DEVIATION` (20 to 5000 Hz); `benchmark_aligo.py` `PASS`.
- Intel Xeon Platinum 8468 (second Nebius VM), on the tagged commit: `bench/validate.py --force` `PASS` (`results/intel-xeon-platinum-8468-197g-linux/validation/f4f5c40@f4f5c40d53d8.json`).

**Result**

Intel Xeon Platinum 8468 (second Nebius VM, pinned to logical CPU 15 with the official `--cpus 1`, 32 GiB memory cap, swap off), 10 shuffled rounds, a fresh process per run, `results/intel-xeon-platinum-8468-197g-linux/sessions/2026-11-22T10-05-52Z_run.json`; the session lists this version under its commit, `f4f5c40`:

| Version | Complete first run (median) | Fastest | Slowest | Forward run | Gradient run |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v11` | 0.666 s | 0.650 s | 0.681 s | 107.8 ms | 55.1 ms |
| first two changes (`f1bc7a7`) | 0.631 s | 0.620 s | 0.640 s | 95.3 ms | 33.7 ms |
| `v12` (`f4f5c40`) | 0.626 s | 0.612 s | 0.640 s | 98.1 ms | 27.1 ms |

`v12` against `v11` within each round: 1.07x (1.03 to 1.11), faster in 10 of 10 rounds; the noise floor of that machine is +/-2.4% (`sessions/2026-11-22T09-54-18Z_aa.json`). The memory setting on its own (`f4f5c40` against `f1bc7a7`) is 1.01x (0.98 to 1.03), inside that noise for the total; its effect shows in the gradient run, 27 ms against 34 ms. One pinned core of an AMD EPYC 7763 (GitHub Codespace), 8 rounds per session, `results/amd-epyc-7763-64-core-processor-16g-linux/sessions/2026-11-22T09-42-13Z_run.json` and `...T09-52-29Z_run.json` (labels `cand2` = `f1bc7a7`, `cand4` = this version): 0.791 s for v11, 0.748 s with the first two changes, 0.739 s with the memory setting as first written; in the second session 0.789, 0.755 and 0.751 s for this version; gradient run 56.2, 36.3 and 27.2 ms. `validate.py` `PASS` there (`validation/cand4@f4f5c40d53d8.json`). Apple M5, by hand, alternating fresh runs: forward run 36.5 to 28.8 ms with the first change (8 runs each), gradient run 14.6 to 9.8 ms with the second (10 runs each).

**Dead ends and notes**

Tried on the way and not adopted, all on the Apple M5: dropping refinement for every solve (all tests pass, but the homodyne setup's noise goes from 1e-12 to 1e-08 and its gradients from 6e-08 to 2e-06, for 4 ms more); a hand-written gradient rule for `signal_detector` (about 2 ms of the benchmark's readout compilation, inside the noise, and it would lose forward-mode differentiation); five more XLA flags (no effect, and `xla_disable_all_hlo_passes` does not compile). The memory setting is process-wide: freed memory stays with the process.
