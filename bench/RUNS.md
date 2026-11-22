# Every run so far

The log of all runs of the benchmark made for this project, with the result of each and where its raw data is. Times are UTC on 21 November 2026 (Munich time is one hour later). "First run" is the scored number, the complete first-run total that `benchmark.py` prints. Numbers from different machines must not be compared with each other.

There are three kinds of run:

1. **Through the harness** (from 19:28 UTC on). These are stored by the harness itself; this file only points to them.
2. **By hand, with a saved report.** The report files were collected into `benchmarks/runs/`.
3. **By hand, printed only.** The terminal output was read but no report file was kept. They are listed with what was printed.

A fourth section points to the runs on other setups, and the last one is for runs made on other machines by the rest of the team.

## 1. Through the harness (Apple M5 MacBook Air, 16 GB, thread limits set to one)

Raw data: `bench/results/apple-m5-16g-darwin/`. A Mac cannot pin a process to one core; the harness limits every numerical library to one thread and `machine.json` records the check that this holds (1.00 CPU-seconds per second).

| Time | What | Result | Raw data |
| --- | --- | --- | --- |
| 19:28 | `run.py selftest`, `doctor`, `calibrate` | `SELFTEST PASS`; doctor warned about 2.8 GB of free memory (other apps were open) | `machine.json` |
| 19:30 | The one baseline run: `validate.py v00-baseline --make-reference` | First run 16.41 s, repeat 9.67 s, `PASS`. Peak memory 14.5 GB, so it swapped | `baseline.json`, `official/v00-baseline.json`, `bench/reference/baseline_arrays.npz` |
| 19:31 | `validate.py` on `v01` to `v07`, one full run each | First runs 7.07, 5.36, 2.30, 1.63, 1.46, 0.96 and 0.36 s. Official `PASS` for all; harness verdict `FAIL` for all on one gradient value out of 193 (see `VERSIONS.md`, v01) | `validation/<version>@<commit>.json` |
| 19:57 | `run.py full --no-validate`, first try | Stopped after 19 of 35 runs: Chrome and other apps were busy and `v01` took 16.7 and 19.7 s. Kept, marked incomplete, ignored by the summary | `sessions/2026-11-21T19-57-35Z_full.json` |
| 20:21 | `run.py full --no-validate`: `v01` to `v07`, 5 shuffled rounds | Medians 8.62, 6.86, 2.68, 1.69, 1.79, 1.27 and 0.50 s | `sessions/2026-11-21T20-21-55Z_full.json`, `SUMMARY.md` |
| 20:38 | `run.py aa`: `v07` against itself, 5 rounds | Noise floor ±16%; medians 0.66 and 0.69 s, fastest 0.34 s | `sessions/2026-11-21T20-38-57Z_aa.json` |
| 20:42 | `run.py official --versions v07-numpy-host` | First run 0.66 s, repeat 0.051 s, `PASS` | `official/v07-numpy-host.json` |
| 21:50 | `validate.py --force` on `v01` to `v07`, after the per-entry gradient tolerance was set to 0.1% | `PASS` for all seven; entry 158 is at 0.29 of the allowed difference | `validation/<version>@<commit>.json` |
| 23:31 | `run.py quick --rounds 8`: `v07` and the working tree with change 14 | Medians 0.35 and 0.35 s; not distinguishable (1.03x, 0.34 to 1.19) | `sessions/2026-11-21T23-31-32Z_quick.json` |
| 23:37 | `tests/variations.py` on the working tree with changes 14 and 15 | 18 `PASS`, 1 `DEVIATION` (20 to 5000 Hz, known) | `variations/2026-11-21T23-37-37Z.json`, `VARIATIONS.md` |
| 23:38 | `run.py run`: `v07`, `v09`, `v10`, 10 shuffled rounds, machine busy with a demo page | Medians 0.44, 0.43 and 0.43 s; gradient phase 38.3, 30.6, 30.7 ms; build phase 43.1, 41.8, 37.7 ms | `sessions/2026-11-21T23-38-07Z_run.json`, `SUMMARY.md` |
| 00:31 (22 Nov) | `run.py run`: `v10` and the branch `no-klujax-scipy` at `f3f1454` (SciPy SuperLU inside the host callbacks), 10 shuffled rounds, machine busy with a demo page | Medians 0.41 and 0.35 s, fastest 0.33 and 0.28 s; branch 1.20x (0.93 to 1.54), faster in 9 of 10 rounds. Compile 70 against 42 ms, forward 26 against 39 ms, gradient 30 against 15 ms. On the branch: `validate.py` `PASS`, `tests/test_host.py` all passed, `tests/variations.py` 18 `PASS` and the known `DEVIATION` (run from a separate checkout, files not stored here) | `sessions/2026-11-22T00-31-37Z_run.json` |

Validation was switched off for the timing session because the harness leaves out versions with a `FAIL` verdict; the verdicts themselves were not changed. `SUMMARY.md` is regenerated from the sessions.

### Through the harness on the second AMD cloud instance (AMD EPYC 9334, 128 logical CPUs, 504 GB, Linux, one pinned core)

Raw data: `bench/results/amd-epyc-9334-32-core-processor-504g-linux/`; the terminal logs of every step are in its `logs/` folder. The host was shared: load average about 20 from other users throughout.

| Time | What | Result | Raw data |
| --- | --- | --- | --- |
| 20:52 | `run.py selftest`, `doctor`, `calibrate` | `SELFTEST PASS`; one core verified (1.00 CPU-seconds per second) | `machine.json` |
| 20:54 | `validate.py v00-baseline`: the original, once | First run 40.79 s, repeat 19.09 s, `PASS`; against the original's arrays from the Mac: signal 1.2e-07, entry 158 at 0.34 of the allowed difference | `baseline.json`, `official/v00-baseline.json` |
| 20:55 | `validate.py` on `v01` to `v07` | `PASS` for all seven; first runs 31.06, 21.77, 13.45, 4.27, 3.93, 2.22 and 2.89 s | `validation/` |
| 20:58 | `run.py full --no-validate`: 5 shuffled rounds | Medians 30.03, 21.49, 13.50, 4.66, 4.32, 2.23 and 1.46 s. `v07` ranged from 0.74 to 2.75 s | `sessions/2026-11-21T20-58-56Z_full.json`, `SUMMARY.md` |
| 21:12 | `run.py aa`: `v07` against itself | 0.73 to 2.77 s; the "noise floor" of ±236% is this instability, not ordinary noise | `sessions/2026-11-21T21-12-08Z_aa.json` |
| 21:15 | `run.py official --versions v07-numpy-host` | First run 2.45 s (a slow run), repeat 0.14 s, `PASS` | `official/v07-numpy-host.json` |
| 21:15 | `tests/variations.py --time-original`: 18 scenarios, original and ours | 16 `PASS`, 2 `DEVIATION`; originals 24 to 73 s; ours 0.63 to 2.82 s (four slow runs) | `variations/2026-11-21T21-28-38Z.json` |
| 21:29 | `tests/variations.py`: 18 scenarios, ours only | Same verdicts; ours 0.63 to 0.93 s, and 2.11 s on the JAX fallback | `variations/2026-11-21T21-29-17Z.json`, `VARIATIONS.md` |
| 21:30 | `tests/test_host.py` | `ALL CHECKS PASSED` | `logs/test-host.log` |

Why `v07` was unstable there is described in `CHANGES.md` ("The same on one pinned Linux core").

Follow-up on a third instance of the same kind (21:55 UTC; a container with 16 CPUs and 55 GiB on the same 128-CPU host type), by hand, not through the harness:

| Time | What | Result | Raw data |
| --- | --- | --- | --- |
| 21:58 | 96 runs of `benchmark.py --cpus 1` on the current code (`e44b489`): four settings, 8 rounds each, on a quiet machine, next to four memory-churning processes, and right after a 14 GB memory user | Quiet and after the memory user: 0.69 to 0.75 s in all 64 runs. Next to the churning processes: 0.93 to 1.51 s; lumps in the NumPy steps in 8 of 24 runs with NumPy's large-page request on, in 0 of 8 with it off | `diagnosis/runs.jsonl`, `diagnosis/diagnose.log`, `diagnosis/machine.txt` |
| 22:35 | 72 runs of `benchmark.py --cpus 1` on `8ff051a` (change 13): NumPy's large-page request off (as committed) and on, 12 rounds each, quiet, next to four memory-churning processes, quiet again | Quiet: 0.74 s off, 0.71 s on. Next to the churning processes: 1.16 s off, 1.11 s on (medians). No step took more than twice its normal time with either setting | `diagnosis-change-13/runs.jsonl`, `diagnosis-change-13/diagnose.log` |

### Through the harness on the GitHub Codespace (AMD EPYC 7763, 4 logical CPUs, 16 GB, Linux, one pinned core)

22 November, times in UTC. The repository was copied there as a git bundle of `main` at `bf8d047`. Files under `bench/results/amd-epyc-7763-64-core-processor-16g-linux/`. The original has no stored run on this machine (it does not fit in 16 GB), so factors are against `v07`.

| Time | What | Result | Raw data |
| --- | --- | --- | --- |
| 00:13 | `run.py calibrate` | The one-CPU limit holds: at most 1.00 CPU-seconds per wall-second | `machine.json` |
| 00:14 | `run.py run`: `v07`, `v09`, `v10`, 10 shuffled rounds | Medians 1.026, 0.999 and 0.976 s; `v10` against `v07` 1.04x (1.03 to 1.08) | `sessions/2026-11-22T00-14-30Z_run.json`, `SUMMARY.md` |
| 00:27 | `validate.py` on `v10` and `v07` | `PASS` for both; first runs 0.997 and 1.031 s | `validation/` |
| 09:42 | `run.py run`: `v11`, `cand2` (`f1bc7a7`, changes 19 and 20), `tmp-cand3` (change 21 as first written), 8 shuffled rounds | Medians 0.791, 0.748 and 0.739 s; gradient run 54.9, 35.1, 26.2 ms | `sessions/2026-11-22T09-42-13Z_run.json` |
| 09:50 | `validate.py cand4` (`f4f5c40` = `v12`), `tests/test_host.py`, `benchmark_aligo.py --cpus 1` | `PASS`, `ALL CHECKS PASSED`, `PASS` | `validation/cand4@f4f5c40d53d8.json` |
| 09:52 | `run.py run`: `v11`, `cand2`, `cand4` (= `v12`), 8 shuffled rounds | Medians 0.789, 0.755 and 0.751 s; gradient run 56.2, 36.3, 27.2 ms | `sessions/2026-11-22T09-52-29Z_run.json`, `SUMMARY.md` |

## 2. By hand, with a saved report

All are complete runs of the unchanged `benchmark.py`. "Repeat" is the median of the repeat runs, with their number in brackets. "Reference" is the distance from the public reference sensitivities in log (limit 1e-5).

| Time | Machine | CPU limit | Code | First run | Repeat | Reference | Check | Raw data |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 13:54 | Apple M5 MacBook Air, 16 GB | none (all cores) | Change 1 (working tree, later commit `25051f2`) | 8.20 s | 2.871 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1354_change-1.json` |
| 13:55 | Apple M5 MacBook Air, 16 GB | none (all cores) | Original | 12.44 s | 6.369 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1355_original.json` |
| 14:02 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–2 (working tree, later commit `59705c9`) | 7.37 s | 3.369 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1402_changes-1-2.json` |
| 14:06 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–3 (working tree, later commit `7938724`) | 3.61 s | 0.066 s (5) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1406_changes-1-3.json` |
| 14:34 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–3 plus XLA optimisation level 0 (working tree, later `946b5b8`) | 2.72 s | 0.079 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1434_changes-1-3-opt-level-0.json` |
| 14:44 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–4, all four XLA flags (working tree, later `73680c8`) | 2.06 s | 0.078 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1444_changes-1-4.json` |
| 15:26 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–11 (working tree, later commit `490a15f`) | 0.92 s | 0.052 s (3) | 5.375e-06 | PASS | `benchmarks/runs/apple-m5_1526_changes-1-11.json` |
| 15:44 | Apple M5 MacBook Air, 16 GB | none (all cores) | Original | 12.53 s | 6.374 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1544_original.json` |
| 15:44 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–11 | 1.06 s | 0.054 s (3) | 5.375e-06 | PASS | `benchmarks/runs/apple-m5_1544_changes-1-11.json` |
| 15:49 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–11, run by Jia in their own terminal | 1.28 s | 0.054 s (10) | 5.375e-06 | PASS | `benchmarks/runs/apple-m5_1549_changes-1-11-own-terminal.json` |
| 15:51 | Apple M5 MacBook Air, 16 GB | none (all cores) | Original, run by Jia in their own terminal | 10.55 s | 5.367 s (10) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1551_original-own-terminal.json` |
| 16:00 | Apple M5 MacBook Air, 16 GB | none (all cores) | Original, 10 repeats, arrays saved (same run as `benchmarks/mac-original.json`) | 13.11 s | 6.003 s (10) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1600_original-10-repeats.json` |
| 16:00 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–11, 10 repeats, arrays saved (same run as `benchmarks/mac-candidate.json`) | 1.01 s | 0.054 s (10) | 5.375e-06 | PASS | `benchmarks/runs/apple-m5_1600_changes-1-11-10-repeats.json` |
| 16:09 | Apple M5 MacBook Air, 16 GB | none (all cores) | Original, while its memory use was being measured (peak 18.4 GB) | 12.90 s | 6.910 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1609_original-memory-measurement.json` |
| 16:09 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–11, while its memory use was being measured (peak 0.31 GB) | 1.06 s | 0.054 s (3) | 5.375e-06 | PASS | `benchmarks/runs/apple-m5_1609_changes-1-11-memory-measurement.json` |
| 17:20 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–12, NumPy version (working tree on branch `numpy-host`); second of two runs | 0.33 s | 0.050 s (10) | 5.376e-06 | PASS | `benchmarks/runs/apple-m5_1720_changes-1-12.json` |
| 17:33 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–12 after adding the test file | 0.42 s | 0.051 s (3) | 5.376e-06 | PASS | `benchmarks/runs/apple-m5_1733_changes-1-12.json` |
| 17:54 | Apple M5 MacBook Air, 16 GB | none (all cores) | SciPy SuperLU experiment (branch `scipy-host`, not adopted); second of two runs | 0.42 s | 0.084 s (5) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1754_scipy-superlu-experiment.json` |
| 18:42 | Apple M5 MacBook Air, 16 GB | none (all cores) | Changes 1–12 after the fix to the refraction helper | 0.42 s | 0.052 s (3) | 5.376e-06 | PASS | `benchmarks/runs/apple-m5_1842_changes-1-12.json` |
| 18:53 | Apple M5 MacBook Air, 16 GB | none (all cores) | `main` as it was on GitHub with the SciPy solver merged by mistake (copy without git data) | 0.52 s | 0.084 s (3) | 5.374e-06 | PASS | `benchmarks/runs/apple-m5_1853_main-with-scipy-merged.json` |
| 18:58 | Apple M5 MacBook Air, 16 GB | none (all cores) | `main` with the KLU solver restored (working tree, commit `3d6516f`) | 0.37 s | 0.051 s (5) | 5.376e-06 | PASS | `benchmarks/runs/apple-m5_1858_main-klujax-restored.json` |
| 19:01 | AMD EPYC 9334, AMD Radeon Cloud, 55 GB | one core (`--cpus 1`) | Changes 1–12 (`43a4609`), warm-up run, not counted | 0.75 s | 0.140 s (3) | 5.375e-06 | PASS | `benchmarks/runs/amd-epyc-9334_1901_changes-1-12-warmup.json` |
| 19:02 | AMD EPYC 9334, AMD Radeon Cloud, 55 GB | one core (`--cpus 1`) | Changes 1–12 (`43a4609`), first timed run (same run as `benchmarks/amd-epyc-9334-candidate.json`) | 0.72 s | 0.124 s (10) | 5.375e-06 | PASS | `benchmarks/runs/amd-epyc-9334_1902_changes-1-12.json` |
| 19:02 | AMD EPYC 9334, AMD Radeon Cloud, 55 GB | one core (`--cpus 1`) | Changes 1–12 (`43a4609`), second timed run | 0.72 s | 0.120 s (10) | 5.375e-06 | PASS | `benchmarks/runs/amd-epyc-9334_1902_changes-1-12-second.json` |
| 19:08 | AMD EPYC 9334, AMD Radeon Cloud, 55 GB | one core (`--cpus 1`) | Original (`2bb437a`) (same run as `benchmarks/amd-epyc-9334-original.json`) | 35.62 s | 15.673 s (10) | 5.374e-06 | PASS | `benchmarks/runs/amd-epyc-9334_1908_original.json` |
| 15:57 | AMD EPYC 7763, GitHub Codespace, 16 GB | one core (`--cpus 1`) | Changes 1–11 (`490a15f`, clean checkout), second of three runs | 3.24 s | 0.191 s (10) | 5.375e-06 | PASS | `benchmarks/codespace-candidate.json` |

On the Mac the original is short of memory (it asks for up to 18.4 GB on a 16 GB machine), so its times are inflated and vary between 10.5 and 13.1 s. Where two runs were made in a row and the report was overwritten, the first one is in section 3.

For comparison, the organizers' own run of the original (not ours): Intel Xeon Platinum 8468, one core, 44.20 s, `benchmarks/cpu-baseline.json`.

## 3. By hand, printed only

### GitHub Codespace (AMD EPYC 7763, 4 cores, 16 GB, Linux, `--cpus 1`)

| Time | Code | First run | Repeat | Notes |
| --- | --- | ---: | ---: | --- |
| 14:15 | Changes 1–3 | 21.51 s | 0.232 s | First run after installing the packages. The environment was Python 3.14 by mistake; redone below. `PASS` |
| 14:17 | Changes 1–3 | 17.80 s | 0.229 s | Python 3.12, pinned packages. `PASS` |
| 14:17 | Original | no result | | Killed by the system for lack of memory (exit code 143). The original cannot finish the official benchmark on 16 GB without swap |
| 14:33 | Changes 1–3 plus XLA optimisation level 0 | 41.56 s | 0.327 s | First run after copying new files to the machine (cold start). `PASS`. Not used |
| 14:34 | Same code | 10.97 s, 10.95 s | 0.325 s, 0.329 s | Two runs. `PASS` |
| 14:43 | Changes 1–4 (all four flags) | 5.85 s | 0.338 s | `PASS` |
| 14:59 | Changes 1–7, first try | 6.67 s | 0.269 s | Slower than expected; measuring each change on its own showed it was noise |
| about 15:00 | Changes 1–4, then adding 7, 5 and 6 one at a time | 5.75/5.70, 5.36/5.35, 5.28/5.20, 5.15/5.09 s | 0.34 to 0.27 s | Two runs each; table "Changes 5–7 measured one at a time" in `CHANGES.md` |
| 15:02 | Changes 1–7 | 5.06 s | 0.251 s | `PASS` |
| about 15:15 | Changes 1–7, then adding 8, 9, 10, 11 and the out-of-range handling one at a time | 5.5, 4.5, 3.6, 3.3, 3.0, 3.3, 3.2 s | | Three runs each; table "Changes 8–11 measured one at a time" in `CHANGES.md` |
| 15:22 | Changes 1–11 | 3.37 s | 0.195 s | `PASS` |
| 15:40 | Original and changes 1–11 in the style of `start_here.py` (one compile, one run), two runs each | 35.8 s and 3.1 s | | Not the official benchmark; this style fits in 16 GB, so it is the only same-machine pair on the Codespace. Outputs compared array by array |
| 15:40 | Changes 1–11 | 3.20 s | 0.192 s | Official benchmark. `PASS` |
| 15:50 | Changes 1–11 after emptying the file cache | 6.19 s, 4.29 s, then 4.19 s | 0.192 s, 0.173 s | Cold starts: loading the packages from disk adds 1 to 3 s for any version |
| 15:57 | Changes 1–11, clean checkout of `490a15f` | 4.37 s, 3.24 s, 3.11 s | 0.251 s, 0.191 s, 0.191 s | Three runs; the second is the saved report in section 2. `PASS` |
| about 17:25 | Changes 1–11 and changes 1–12, three runs each | 3.23, 3.52, 4.22 s and 1.01, 1.05, 1.33 s | 0.19 s and 0.16 s | `PASS` |

### AMD Radeon Cloud (AMD EPYC 9334, 55 GB, Linux, `--cpus 1`)

| Time | Code | First run | Repeat | Notes |
| --- | --- | ---: | ---: | --- |
| about 16:30 | Original | 35.58 s | 15.70 s | Validation raised no error, but the run script had not installed the package, so `benchmark.py` stopped while writing its report. No file saved |
| about 16:30 | Changes 1–11 (`490a15f`) | 2.13 s | 0.13 s | Same; no file saved |

The second AMD session (19:01 to 19:08) has saved reports and is in section 2. On that machine `tests/test_host.py` ended with `ALL CHECKS PASSED` and `tests/handwritten_jacobians.py` with all 19 rules agreeing.

### Apple M5 MacBook Air (16 GB, all cores)

| Time | Code | First run | Repeat | Notes |
| --- | --- | ---: | ---: | --- |
| 15:19 | Changes 1–11, working tree | 0.88 s | 0.052 s | Report overwritten by the 15:26 run. `PASS` |
| 15:46 | Run by Jia in their own terminal; version not recorded | 8.51 s | | Its forward stage took 0.02 s, so the code was changes 1–3 or later; the total fits a first run right after installing |
| 17:20 | Changes 1–12, first of two runs | 0.40 s | 0.049 s | Report overwritten by the second run (0.33 s). `PASS` |
| 17:54 | SciPy SuperLU experiment, first of two runs | 0.51 s | 0.083 s | Report overwritten by the second run (0.42 s). `PASS` |

### On the Nebius VM (Intel Xeon Platinum 8468, the organizers' processor type, 197 GB, Linux, one pinned core)

21 and 22 November, times in UTC. Every run is pinned to logical CPU 8 with the official `--cpus 1`, inside a 32 GiB memory cap with swap off. Files under `bench/results/intel-xeon-platinum-8468-197g-linux/`. The repository reaches the VM by `git push` over SSH, because the VM cannot reach GitHub. The sessions of 22 November list version `v11-scipy-stacked-solver` under its branch name, `no-klujax-scipy`.

| Time | What | Result | Raw data |
| --- | --- | --- | --- |
| 22:42 (21 Nov) | The original (`02ec741`), once, by hand with the official command; imported as this machine's stored baseline | First run 43.14 s, `PASS`, peak memory 15.3 GB | `baseline.json`, `bench/results/nebius-xeon-8468-200g-linux/original-reproduction/` |
| 22:51 | `validate.py` on `v01` to `v08` | `PASS` for all eight | `validation/` |
| 22:53 | `run.py run`: `v01` to `v08`, 5 shuffled rounds | Medians 33.39, 23.66, 15.66, 5.17, 4.61, 2.77, 0.90 and 0.91 s | `sessions/2026-11-21T22-53-03Z_run.json` |
| 00:08 (22 Nov) | `run.py run`: `v07` and the SciPy branch at `05da4b9` (before changes 14 and 15 were merged in), 5 shuffled rounds; `validate.py` on the branch | Medians 0.85 and 0.67 s; branch 1.27x (1.26 to 1.29). `PASS` | `sessions/2026-11-22T00-08-43Z_run.json`, `validation/no-klujax-scipy@05da4b9818e9.json` |
| 00:10 | `run.py aa` (noise floor) | Stopped by hand before it finished; kept, marked incomplete. The noise floor of this machine is not measured | `sessions/2026-11-22T00-10-37Z_aa.json` |
| 00:14 | `run.py run`: `v07`, `v10` and `v11` (`f3f1454`), 5 shuffled rounds | Medians 0.88, 0.84 and 0.71 s; `v11` against `v10` 1.20x (1.17 to 1.21), against `v07` 1.24x (1.22 to 1.29). Slower control workload flagged in 2 to 3 of 5 rounds for every version | `sessions/2026-11-22T00-14-25Z_run.json`, `SUMMARY.md` |
| 00:17 | `benchmark_aligo.py --cpus 1` by hand: the original once, then `v10` and `v11` alternately, 4 runs each | Original 26.25 s; `v10` 0.61 to 0.63 s (median 0.62); `v11` 0.50 to 0.53 s (median 0.50). `PASS` in every run | `aligo/` |
| 00:18 | `tests/variations.py random-layout-1` by hand: the original once, then `v10` and `v11` alternately | Original 41.55 s; `v10` 0.83 s; `v11` 0.69 to 0.71 s. `PASS` | `variations/2026-11-22T00-18-*.json`, `variations-v10/2026-11-22T00-18-*.json` |
| 00:20 | `tests/variations.py` on `v11`, five scenarios | 3 `PASS`, 1 `DEVIATION`, and a `FAIL` on `supplied` that is not about the code: the script found that its own run of the original differs from the stored reference arrays, because the VM held its own baseline arrays in place of the committed ones (from the Mac). Rerun at 00:23 with the committed arrays | `variations/2026-11-22T00-20-20Z.json` |
| 00:23 | `validate.py --force` on `v11` and on `v10` | `PASS` for both; first runs 0.67 and 0.81 s; `v11` against the original's arrays from this VM: signal 1.3e-07, sensitivity 6.8e-08 in log, gradient 9.1e-08, 0 of 193 entries outside | `validation/no-klujax-scipy@f3f14542ea40.json`, `validation/v10-host-swept-values@49f887148515.json` |
| 00:23 | `tests/variations.py` on `v11`, all 19 scenarios | 18 `PASS`, 1 `DEVIATION` (`band-20-5000`, known), 0 `FAIL`; NumPy path 0.59 to 0.88 s, JAX fallback 2.57 s | `variations/2026-11-22T00-23-50Z.json`, `VARIATIONS.md` |
| 00:24 | `tests/variations.py` on `v10`, all 19 scenarios | 17 `PASS`, 2 `DEVIATION` (`band-20-5000` and `uifo-20-5000`); NumPy path 0.68 to 1.05 s, JAX fallback 2.47 s | `variations-v10/2026-11-22T00-24-58Z.json` |
| between 00:14 and 00:30, time not noted | `tests/test_host.py` and `tests/handwritten_jacobians.py` on `v11` | `ALL CHECKS PASSED`; all 19 rules agree | `no-klujax-scipy-logs/` |

The sessions up to 22:53 were made by a teammate's session; the ones of 22 November are those behind changes 16 to 18 in `CHANGES.md`. The noise-floor session and the official report and chart of `v11` that were still owed here were made on a second VM of the same type; see the next table.

### On a second Nebius VM of the same type (Intel Xeon Platinum 8468, 197 GB, Linux, one pinned core)

22 November, times in UTC. A new instance (`89.169.127.49`), set up from nothing with `bench/setup_mac.sh`, because the first one could not be reached. It has the same machine id, so its files are in the same folder, `bench/results/intel-xeon-platinum-8468-197g-linux/`. Every run is inside a 32 GiB memory cap with swap off. `run.py calibrate` chose logical CPU 15 on this instance, so the harness runs are pinned there; the original was pinned to CPU 8 by hand, as on the first VM. `selftest` and `doctor` passed first.

| Time | What | Result | Raw data |
| --- | --- | --- | --- |
| 09:45 | The original (`02ec741`), once, by hand with the official `benchmark.py --cpus 1` (10 warm repeats). Not imported as the stored baseline, which is still the first VM's 43.14 s | First run 41.31 s, repeat median 17.69 s, `PASS` | `bench/results/nebius-xeon-8468-200g-linux/original-reproduction/original-1cpu-32g-vm2.*` |
| 09:50 | `run.py run`: `v07`, `v10`, `v11` and `main` at `f1bc7a7`, 5 shuffled rounds | Medians 0.83, 0.81, 0.67 and 0.64 s. `v11` 0.67 to 0.69 s, which is 61.5 times faster than the original's run on this instance (60.2 to 61.8). `v11` against `v10` 1.20x (1.17 to 1.22), against `v07` 1.23x (1.22 to 1.26); `f1bc7a7` against `v11` 1.06x (1.04 to 1.11), faster in 5 of 5 rounds. The factors the harness prints divide the first VM's 43.14 s and are not the ones to quote | `sessions/2026-11-22T09-50-17Z_run.json` |
| 09:54 | `run.py aa`: `v11` against itself, 5 rounds | Medians 0.65 and 0.65 s; noise floor ±2.4% | `sessions/2026-11-22T09-54-18Z_aa.json`, `SUMMARY.md` |
| 09:59 | `run.py official --versions v11-scipy-stacked-solver` | First run 0.667 s, repeat median 0.132 s, `PASS` | `official/v11-scipy-stacked-solver.json` and `.png` |
| 10:00 | `validate.py --force` on `v11` and on `f1bc7a7`, against the committed reference arrays | `PASS` for both; first runs 0.671 and 0.645 s; `v11`: signal 1.2e-07, sensitivity 6.0e-08 in log, 0 of 193 gradient entries outside | `validation/v11-scipy-stacked-solver@f3f14542ea40.json`, `validation/f1bc7a7@f1bc7a7654f9.json` |
| 10:01 | `benchmark_aligo.py --cpus 1` on `v11` by hand, 4 runs, pinned to CPU 15 | 0.486 to 0.490 s; `PASS` in every run. The original was not run on this setup here | `aligo/v11-vm2-*.json`, `aligo/v11-vm2-steps.log` |
| 10:05 | `run.py run`: `v11` against `main` without the Linux memory setting (`f1bc7a7`) and with it (`f4f5c40`), 10 shuffled rounds; then `validate.py --force` on `f4f5c40` | Medians 0.666, 0.631 and 0.626 s. Against `v11` within each round: `f1bc7a7` 1.06x (1.03 to 1.09), `f4f5c40` 1.07x (1.03 to 1.11), each faster in 10 of 10 rounds. `f4f5c40` against `f1bc7a7`: 1.01x (0.98 to 1.03), faster in 7 of 10, which is inside the ±2.4% noise floor; its gradient step is 27 ms against 34 ms. `PASS` for `f4f5c40`, first run 0.625 s. The harness waited for a calm machine before 20 of the 30 samples | `sessions/2026-11-22T10-05-52Z_run.json`, `validation/f4f5c40@f4f5c40d53d8.json` |

Not run on this instance: `tests/variations.py`, `tests/test_host.py`, and the original on the aLIGO setup.

## 4. Other setups

`python tests/variations.py` runs 16 varied setups on the original code and on ours, compares them and times them. Its sessions are in `bench/results/<machine>/variations/`, the table of the latest one in `bench/results/<machine>/VARIATIONS.md`, and the original's results in `bench/reference/variation-<case>.npz`. See "Other setups" in `README.md`.

Earlier comparisons with the original on other setups (the second UIFO, Voyager, aLIGO, the cavity sweep, shifted parameter values) were made with scripts that are not in the repository; their results are in "Correctness checks" in `CHANGES.md`.

## 4b. The organizers' second benchmark, aLIGO (22 November)

By hand, Apple M5 MacBook Air, 16 GB, all cores, after merging the organizers' update `83f8d2f`:

| Time (UTC) | Code | First run | Repeat | Reference | Check | Raw data |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 00:40 | Ours (`main` after the merge), `benchmark_aligo.py` | 0.38 s | 0.007 s (10) | 1.4e-10 | PASS | `benchmarks/mac-aligo-candidate.json` |
| 00:40 | Original (`upstream/main` at `83f8d2f`), `benchmark_aligo.py` | 4.10 s | 0.021 s (10) | 1.0e-10 | PASS | `benchmarks/mac-aligo-original.json` |
| 00:42 | Ours, `benchmark.py` (the grid) with the organizers' new shared harness | 0.40 s | 0.051 s (10) | | PASS | not saved |

`bench/validate.py WORKTREE` also gives `PASS` after the merge. Note for the harness: `benchmark.py` now differs from the one in the registered baseline (`02ec741`), because the organizers changed it; sessions will flag later versions as `unofficial-benchmark` until the baseline reference is moved to the organizers' new commit.

## 5. Runs on other machines

Runs by other team members on their own machines are not in this file yet. Add them here in the same form: time, machine, CPU limit, code, first run, check, raw data.
