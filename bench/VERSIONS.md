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
