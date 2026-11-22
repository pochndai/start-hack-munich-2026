# Changes and why

Everything we changed in Differometor for the Feyer "Faster Physics" challenge, in the order we made the changes, with the reason for each one.

## Result so far

**The current version is `v12` (changes 1–21): 0.626 s on the processor type the organizers use.** Intel Xeon Platinum 8468 (second Nebius VM, pinned to one logical CPU with the official `benchmark.py --cpus 1`, 32 GiB memory cap, swap off), median of 10 fresh runs, 0.612 to 0.640 s. In the same session `v11` took 0.666 s; `v12` was faster in all ten rounds, by 1.07 times (1.03 to 1.11), and `bench/validate.py` gives `PASS` for it there. The original was not run on that VM; on the first VM it took 43.14 s and `v11` 0.71 s (next table). Details: "Changes 19 to 21 measured".

**The version before it (changes 1–18, `v11`) with the original on the same VM.** Intel Xeon Platinum 8468 (a teammate's Nebius machine, 197 GB RAM, Python 3.12.3), the official `benchmark.py --cpus 1`, pinned to one logical CPU, inside a 32 GiB memory cap with swap off. The original ran once there. Our versions ran five times each in shuffled order, a fresh process per run, and the table gives medians (`bench/results/intel-xeon-platinum-8468-197g-linux/sessions/2026-11-22T00-14-25Z_run.json`). This is the comparison to quote:

| Step | Original (one run) | Changes 1–12 (`v07`) | Changes 1–15 (`v10`) | Changes 1–18 (`v11`) |
| --- | ---: | ---: | ---: | ---: |
| Imports | 0.33 s | 0.33 s | 0.34 s | 0.31 s |
| Build the setup | 0.87 s | 0.11 s | 0.09 s | 0.09 s |
| Trace and compile | 23.64 s | 0.22 s | 0.22 s | 0.13 s |
| Forward simulation | 13.82 s | 0.09 s | 0.09 s | 0.11 s |
| Gradient | 4.47 s | 0.13 s | 0.11 s | 0.06 s |
| **Complete first run** | **43.14 s** | 0.88 s | 0.84 s | **0.71 s** |
| Fastest and slowest of the five runs | | 0.85 and 0.91 s | 0.82 and 0.87 s | 0.68 and 0.72 s |

That is about 61 times faster than the original for the scored first run (60 to 64 over the five runs). The rows are medians of each step, so a column need not add up exactly to its total. On that machine, for `v11`: `bench/validate.py` gives `PASS` (every output and all 193 gradient entries against the original's arrays from the same machine), the distance from the public reference is 5.374e-06 in log sensitivity (limit 1e-5), `tests/test_host.py` ends with `ALL CHECKS PASSED`, and of 19 other setups 18 get `PASS` and one the known `DEVIATION` ("Changes 16 to 18 measured" and "Correctness checks" below).

What this table does not have: the original ran once, the noise of that machine has not been measured (within the session `v11` was faster than `v10` in every round, by 1.17 to 1.21 times), and the official report file and chart of a `v11` run on that machine have not been saved yet. The original's report and chart are in `bench/results/nebius-xeon-8468-200g-linux/original-reproduction/`.

**Repeated on a second VM of the same type** (a new Nebius instance, same processor, 22 November, 09:45 to 10:01 UTC; rows in `bench/RUNS.md`). The original took 41.31 s in one run there and `v11` 0.67 s (median of 5, 0.67 to 0.69 s), 61.5 times faster (60.2 to 61.8), with `PASS` from `bench/validate.py`. `v11` against `v10` within each round: 1.20 times (1.17 to 1.22). This instance also has what the first one lacks: its noise floor, ±2.4% (`v11` against itself, `sessions/2026-11-22T09-54-18Z_aa.json`), and the official report and chart of `v11` (`official/v11-scipy-stacked-solver.json` and `.png`, first run 0.667 s). On this instance the original was pinned to logical CPU 8 and our versions to CPU 15.

The tables in the rest of this section are for earlier states of the code, named in each heading. They are kept because they are complete pairs of official reports, and because they show how the total came down.

**Changes 1–12 (`v07`), with the official reports of both sides.** Official benchmark, one CPU core (`--cpus 1`), original and changes 1–12 on the same Linux machine (AMD EPYC 9334 on the AMD Radeon Cloud, 55 GB RAM, Python 3.12.15, 10 repeats). Both print `PASS`:

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

The same pair on the processor type the organizers use (Intel Xeon Platinum 8468, a teammate's Nebius machine, 197 GB RAM, one logical CPU with the official `--cpus 1`): the original took **43.14 s** in its one run there, changes 1–12 (`v07`) took **0.90 s** (median of 5 fresh runs, fastest 0.89 s), about 48 times faster, with `PASS` from `bench/validate.py`. Data: `bench/results/intel-xeon-platinum-8468-197g-linux/sessions/2026-11-21T22-53-03Z_run.json`. That was the first session on that machine, on 21 November; the table at the top of this section is from a later one, in which `v07` took 0.88 s.

How the total came down on that machine (the AMD EPYC 9334): an earlier run of changes 1–11 there took 2.13 s (printed timings only, no report saved), so change 12 accounts for the step from 2.13 s to 0.72 s.

Change 12 on a second machine. Official benchmark, one core, Linux Codespace (AMD EPYC 7763), three runs each, all `PASS`:

| Version | Complete first run | Trace and compile | Repeat run |
| --- | ---: | ---: | ---: |
| Changes 1–11 | 3.23, 3.52, 4.22 s | about 2.5 s | 0.19 s |
| Changes 1–12 | 1.01, 1.05, 1.33 s | about 0.2 s | 0.16 s |

The numbers in the rest of this section are for changes 1–11.

Official benchmark on the same machine, both versions with `PASS` (Apple M5, 10 repeats; a Mac cannot pin to one core, so all cores are used):

| | Original | Ours (changes 1–11) |
| --- | ---: | ---: |
| Complete first run | 13.1 s | **1.01 s** |
| Repeat run, once compiled | 6.00 s | 0.054 s |

A second pair of runs on the same Mac gave 10.5 s and 1.28 s. Reports: `benchmarks/mac-original.json` and `benchmarks/mac-candidate.json`.

The original's Mac times are inflated by a shortage of memory. This Mac has 16 GB, and in the official benchmark the original asks for 18.4 GB at its peak (ours: 0.31 GB), so macOS was compressing and swapping throughout: about half of the original's processor time was spent inside the operating system. Its times also vary more from run to run (10.5 to 13.1 s). The one-core Linux comparison below is not affected, because that machine has no swap and the starter-script style fits in its memory; quote that one.

Official benchmark (`python benchmark.py --cpus 1`), complete first run, one CPU core:

| Machine | Original | Ours (changes 1–11, an earlier state) |
| --- | ---: | ---: |
| Intel Xeon Platinum 8468 (Feyer's published baseline) | 44.2 s | not measured |
| AMD EPYC 7763, GitHub Codespace, Python 3.12 | does not fit in 16 GB of RAM | **3.2 s** |
| AMD EPYC 9334, AMD Radeon Cloud, Python 3.12 | 35.6 s | **2.13 s** |

Same Codespace, one core, both versions in the style of `start_here.py` (one compile, one run):

| | Original | Ours (changes 1–11) |
| --- | ---: | ---: |
| Setup | 1.4 s | 0.5 s |
| Compile | 22.6 s | 2.4 s |
| First run | 11.7 s | 0.2 s |
| **Total** | **35.8 s** | **3.1 s** |

A repeat run, once compiled, takes 0.20 s in the official benchmark (the published baseline is 17.7 s).

The machines in the table above are different, so 44.2 s and 3.2 s are not a same-hardware comparison. The AMD EPYC 9334 table at the top and the Codespace table just above are.

The report and chart of our run are in `benchmarks/codespace-candidate.json` and `benchmarks/codespace-candidate.png`.

All seven registered versions side by side, through the team's harness (`bench/README.md`): Apple M5 MacBook Air, 16 GB, every numerical library limited to one thread (a Mac cannot pin a core), each version run 5 times in shuffled order, a fresh process per run. Raw data: `bench/results/apple-m5-16g-darwin/sessions/2026-11-21T20-21-55Z_full.json`.

| Version | Changes | Median first run | Fastest | Against the version before |
| --- | --- | ---: | ---: | --- |
| `v00-baseline` (the original, one stored run) | | 16.41 s | | |
| `v01-klujax-solver` | 1 | 8.62 s | 8.12 s | |
| `v02-group-by-function` | 2 | 6.86 s | 5.97 s | 1.40x |
| `v03-sparse-assembly` | 3 | 2.68 s | 2.64 s | 2.56x |
| `v04-fast-compile-flags` | 4 | 1.69 s | 1.65 s | 1.58x |
| `v05-two-solves-whole-arrays` | 5–7 | 1.79 s | 1.49 s | 0.94x, inside the noise |
| `v06-real-parameters` | 8–11 | 1.27 s | 0.92 s | 1.34x |
| `v07-numpy-host` (the latest at the time) | 12 | 0.50 s | 0.35 s | 2.03x |

Read this table for the order of the versions, not for exact factors. The noise floor of the Mac in that hour was ±16% (the same version against itself, `bench/results/apple-m5-16g-darwin/sessions/2026-11-21T20-38-57Z_aa.json`), the original's single run was short of memory, and `v01` to `v03` used more than one core in spite of the thread limits (flagged by the harness). Changes 5–7 were each measured on their own on one pinned Linux core, where they add up to about 10% (table further down). The official "after" report from this machine is `bench/results/apple-m5-16g-darwin/official/v07-numpy-host.json` (0.66 s first run, 0.051 s repeat, `PASS`; one sample on the same noisy machine). The one-core AMD comparison at the top stays the one to quote.

### The same on one pinned Linux core (AMD cloud), and a timing problem found there

A second AMD cloud instance (AMD EPYC 9334, 128 logical CPUs, 504 GB, Linux, a shared host with a load average of about 20 from other users), one core pinned by the harness plus the official `--cpus 1`. Raw data: `bench/results/amd-epyc-9334-32-core-processor-504g-linux/`, logs in its `logs/` folder. All seven versions and the original get `PASS` from `bench/validate.py` there, and `tests/test_host.py` ends with `ALL CHECKS PASSED`.

| Version | Changes | Median first run | Fastest | Against the version before |
| --- | --- | ---: | ---: | --- |
| `v00-baseline` (the original, one run) | | 40.79 s | | |
| `v01-klujax-solver` | 1 | 30.03 s | 29.13 s | |
| `v02-group-by-function` | 2 | 21.49 s | 20.94 s | 1.39x |
| `v03-sparse-assembly` | 3 | 13.50 s | 12.68 s | 1.60x |
| `v04-fast-compile-flags` | 4 | 4.66 s | 4.03 s | 2.88x |
| `v05-two-solves-whole-arrays` | 5–7 | 4.32 s | 3.74 s | 1.09x, inside the noise |
| `v06-real-parameters` | 8–11 | 2.23 s | 2.15 s | 1.92x |
| `v07-numpy-host` (the latest at the time) | 12 | 1.46 s | 0.74 s | see below |

**The time of `v07`, the latest version at the time, was not stable on this machine.** Its five runs took 0.74, 0.78, 2.31, 1.46 and 2.75 s, and ten more in the noise check took between 0.73 and 2.77 s. The earlier AMD instance gave 0.715 and 0.716 s for the same code. What the stored phase times show:

- In a slow run, three steps carry extra time and nothing else does: the build step (0.09 s becomes 0.87 to 0.92 s), the forward simulation (0.08 s becomes 0.52 to 0.59 s) and the gradient (0.10 s becomes 0.38 to 0.83 s). Imports, tracing and compiling are identical to the millisecond.
- The extra time is processor time of our own process, not waiting (CPU seconds equal wall seconds).
- The older versions show the same thing in their build step only: `v03`, `v04` and `v05` take about 0.7 s longer there in some runs (for example `v05`: 0.32, 0.34, 1.02, 1.01, 1.03 s). Their forward and gradient steps, which run as compiled JAX code, never do.
- It comes and goes in time: the runs before 21:02 UTC and after about 21:20 were all fast, including all 18 scenarios in the second pass below (0.63 to 0.93 s).
- It is not Python's garbage collector (measured on the Mac: 0.02 s per run).

So the steps that run in NumPy and plain Python can become several times slower for minutes at a time on that machine, while compiled JAX code is unaffected.

**A follow-up test on the same kind of instance** (`bench/results/amd-epyc-9334-32-core-processor-504g-linux/diagnosis/`: the scripts as they ran, `machine.txt`, and `runs.jsonl` with 96 runs of the unchanged `benchmark.py --cpus 1`, each with its phase times, its user and system CPU time, its page faults and the kernel's large-page counters). What it established:

- The "machine" is a container with a quota of 16 CPUs and 55 GiB of memory on a 128-CPU host that other users keep busy (load average 18 before we started). Linux hands out large memory pages only to programs that ask for them (`transparent_hugepage/enabled` and `defrag` are both `madvise`). NumPy asks for every array above 4 MB; JAX does not.
- On a quiet machine every run was fast: 0.69 to 0.75 s in 32 runs, also right after a program that used and freed 14 GB.
- Next to four processes that keep allocating and freeing memory, every run got slower, and all of the extra time is the operating system's (user time is unchanged, system time goes from 0.16 to 0.59 s). Loading the packages goes from 0.29 to 0.48 s: under memory pressure every fresh page costs more, for any program.
- On top of that, the lumps from the earlier run appeared again in the three NumPy steps, and only when NumPy asks for large pages:

  | Setting, next to the memory-churning processes | The three NumPy steps together, 8 runs each (seconds, sorted) |
  | --- | --- |
  | Default | 0.42, 0.42, 0.43, 0.45, 0.52, 0.55, 0.58, 0.67 |
  | Freed memory kept and reused (allocator setting) | 0.37, 0.39, 0.40, 0.41, 0.43, 0.45, 0.65, 0.75 |
  | Pinned to the last CPU, as the harness does | 0.40, 0.41, 0.41, 0.41, 0.45, 0.71, 0.92, 0.96 |
  | NumPy does not ask for large pages (`NUMPY_MADVISE_HUGEPAGE=0`) | 0.39, 0.46, 0.48, 0.51, 0.52, 0.52, 0.53, 0.54 |

  With the request on, 8 of 24 runs have a step that takes two to five times its normal time (up to 0.51 s for the build step alone). With it off, 0 of 8 do, and no step exceeds 0.20 s.
- The price of switching it off: memory then comes in small pages, about 6,000 more page faults per step. Quiet machine: 0.74 s against 0.71 s for the complete first run (4%). Under memory pressure: a median of 1.16 s against 1.14 s, with the worst run at 1.23 s against 1.51 s.

So the cause of the lumps is the large-page request: when free large pages are scarce, the kernel tries to assemble them inside our process, at the moment NumPy first touches a new array. This fits the first observation (only the NumPy steps, only at times, compiled JAX code never). It is supported by 32 runs, not proven: the slow stretch of the first evening, where a step took ten times as long, did not come back on its own, and 0 of 8 against 8 of 24 would happen by chance about one time in 25. Switching the request off inside the package was tried as change 13 and withdrawn again; see "Withdrawn" further down for the measurements and the decision.

The 18 scenarios on the same instance (`bench/results/amd-epyc-9334-32-core-processor-504g-linux/VARIATIONS.md`). "Ours" is from the second pass, when only our code ran; "ours, first pass" is the run made right after the original had run; the original's time is one run on this machine:

| Case | Frequencies | Design variables | Path | Ours | Ours, first pass | Original | Faster | Sensitivity | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `amplitude-modulation` | 101 | 193 | JAX fallback | 2.11 s | 2.81 s | 36.8 s | 17x | 7.1e-09 | PASS |
| `band-20-5000` | 101 | 193 | NumPy | 0.74 s | 0.74 s | 39.9 s | 54x | 6.8e-08 | DEVIATION |
| `boundary-swap` | 101 | 193 | NumPy | 0.74 s | 2.82 s | 40.0 s | 54x | 3.3e-11 | PASS |
| `center-swap` | 101 | 193 | NumPy | 0.73 s | 2.03 s | 39.3 s | 54x | 1.4e-10 | PASS |
| `detector-moved` | 101 | 193 | NumPy | 0.73 s | 2.53 s | 39.8 s | 54x | 1.5e-07 | PASS |
| `frequencies-201` | 201 | 193 | NumPy | 0.86 s | 0.96 s | 58.3 s | 68x | 6.6e-08 | PASS |
| `frequencies-51` | 51 | 193 | NumPy | 0.67 s | 0.67 s | 29.8 s | 44x | 6.5e-08 | PASS |
| `homodyne-readout` | 101 | 195 | NumPy | 0.74 s | 0.74 s | 37.4 s | 51x | 8.9e-08 | PASS |
| `random-layout-1` | 101 | 193 | NumPy | 0.74 s | 0.80 s | 35.8 s | 48x | 3.0e-09 | PASS |
| `random-layout-2` | 101 | 192 | NumPy | 0.73 s | 0.77 s | 35.5 s | 49x | 6.6e-10 | PASS |
| `random-layout-3` | 101 | 197 | NumPy | 0.74 s | 0.80 s | 35.4 s | 48x | 2.3e-09 | PASS |
| `size-2` | 101 | 102 | NumPy | 0.63 s | 0.63 s | 24.1 s | 38x | 2.1e-08 | PASS |
| `size-4` | 21 | 315 | NumPy | 0.75 s | 0.84 s | 31.1 s | 41x | 1.4e-10 | PASS |
| `size-4-full` | 101 | 315 | NumPy | 0.93 s | 1.10 s | 72.5 s | 78x | 1.4e-10 | PASS |
| `supplied` | 101 | 193 | NumPy | 0.74 s | 0.73 s | 35.6 s | 48x | 6.5e-08 | PASS |
| `uifo-20-5000` | 101 | 192 | NumPy | 0.73 s | 0.73 s | 36.6 s | 50x | 8.0e-07 | DEVIATION |
| `values-random` | 101 | 193 | NumPy | 0.73 s | 0.73 s | 35.5 s | 49x | 7.3e-08 | PASS |
| `values-shifted` | 101 | 193 | NumPy | 0.74 s | 0.74 s | 35.6 s | 48x | 8.8e-09 | PASS |

The two scenarios that do not fit on the Mac (`frequencies-201` and `size-4-full`) got their reference here and pass. `uifo-20-5000` is `DEVIATION` here and `PASS` on the Mac: its readout signal is 1.4e-06 from the stored original, a hair over the 1e-6 limit, because the stored original was computed on the Mac and ours here on Linux. The original itself is not identical across the two machines either: on `band-20-5000`, the original on this machine and the original on the Mac disagree on two gradient entries by more than 0.1%.

### The organizers' second benchmark: aLIGO

On 21 November the organizers added `benchmark_aligo.py`: the public simplified Advanced LIGO setup with mirror mechanics and squeezed light, 101 frequencies from 20 to 5000 Hz, 29 design variables with independent lengths, and a tighter check (sensitivity within 1e-6 of the reference in log, where the grid benchmark allows 1e-5). They say the final evaluation will be on a setup like the larger grid; this one is for testing whether a speed-up carries over to a smaller setup. Their update (`83f8d2f`) is merged into `main`; it also changes `benchmark.py` so that both benchmarks share one harness, and the grid benchmark still prints `PASS` with it.

Both versions on the same machine (Apple M5 MacBook Air, all cores, since a Mac cannot pin one; 10 repeats; reports `benchmarks/mac-aligo-original.*` and `benchmarks/mac-aligo-candidate.*`):

| Step | Original | Ours |
| --- | ---: | ---: |
| Imports | 0.17 s | 0.27 s |
| Build the setup | 0.36 s | 0.03 s |
| Trace and compile | 3.51 s | 0.07 s |
| Forward simulation | 0.038 s | 0.004 s |
| Gradient | 0.025 s | 0.005 s |
| **Complete first run** | **4.10 s** | **0.38 s** |
| Repeat run, once compiled (median of 10) | 0.021 s | 0.007 s |
| Distance from the reference, in log (limit 1e-6) | 1.0e-10 | 1.4e-10 |
| Official check | `PASS` | `PASS` |

About 11 times faster on the scored number. The setup is small (the original needs 0.02 s per evaluation once compiled), so almost all of the original's time is compiling, and that is what our NumPy path avoids; what is left of ours is mostly loading the packages. Ours against the original, every saved array: carrier and powers identical, noise 2e-12, sensitivity 1.3e-10 in log, signal 1.2e-08, and all 29 gradient entries within 1.1e-09 of their own size. No deviation at the harness tolerances.

Not measured: a one-core Linux pair for this benchmark. The organizers' own run of the original on their Xeon took 25.31 s (`benchmarks/aligo-cpu-baseline.json`); that is a different machine and must not be compared with our Mac numbers.

## The changes

The code changes of 1–11 are mostly in `differometor/simulate.py`; from change 12 on they are mostly in `differometor/host.py`. In `components.py` the physics formulas are the same; only the mirror function was rewritten in an equivalent, shorter form (change 10).

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

### 14. Take derivatives only where a design variable is involved

- **What:** in the NumPy version (change 12), the reverse pass marks which entries of the parameter vector depend on the design variables, and asks each physics formula for derivatives only with respect to the inputs that read a marked entry. A formula with no such input runs without derivatives.
- **Why:** each formula has seven inputs, and the `Dual` type carried seven derivatives through every arithmetic step. Most were never used. The frequency and the fixed refractive indices are not design variables, and force calculations, signal sources, vacuum noise and loss noise have no design variable among their inputs at all. Those derivatives were computed and discarded.
- **Nothing requested is skipped:** all 193 gradient entries are computed. Only arithmetic whose result was thrown away is gone.
- **Effect:** the gradient phase of a first run went from 130.5 to 109.9 ms on one pinned Linux core, and from 38.3 to 30.6 ms on the Apple M5 (medians of 10; tables below). All outputs and gradients are bit-for-bit the same as before on five setups (supplied, second UIFO, aLIGO, Voyager, a random 3×3 layout).
- **General:** the marks are derived from the setup's index arrays, including the parameter links (tied lengths).

### 15. Reshape and copy the swept values on the host

- **What:** the build step reshapes and copies the swept values (the frequencies) in NumPy and converts the result to a JAX array once.
- **Why:** on a JAX array each of the two steps compiled a small device program at first use. They were the last two such compilations in our build step; the three that remain in that phase are made by `benchmark.py` itself.
- **Effect:** the build phase went from 117.5 to 98.8 ms on one pinned Linux core (medians of 10; tables below). On the Apple M5: from 41.8 to 37.7 ms in the session below, and from about 42 to 31 ms in runs by hand on a quieter machine. Outputs are bit-for-bit the same.

### Changes 14 and 15 measured

One pinned Linux core (GitHub Codespace, AMD EPYC 7763, 16 GB, Python 3.12.3, official `--cpus 1`), 10 shuffled rounds, a fresh process per run (`bench/results/amd-epyc-7763-64-core-processor-16g-linux/sessions/2026-11-22T00-14-30Z_run.json`). This is the comparison to quote for these two changes:

| Version | Changes | Complete first run (median) | Fastest | Slowest | Imports | Build phase | Gradient phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v07-numpy-host` | 1–12 | 1.026 s | 0.998 s | 1.056 s | 414 ms | 116.7 ms | 130.5 ms |
| `v09-active-derivatives` | + 14 | 0.999 s | 0.987 s | 1.034 s | 410 ms | 117.5 ms | 109.9 ms |
| `v10-host-swept-values` | + 15 | 0.976 s | 0.964 s | 0.998 s | 411 ms | 98.8 ms | 111.1 ms |

`v10` against `v07`: 1.04 times faster as the median of the per-round ratios, between 1.03 and 1.08 in every one of the 10 rounds, so it is outside the noise of that machine. That is about 50 ms of the scored first run: 20 ms from the gradient phase (change 14) and 18 ms from the build phase (change 15). Each change on its own (1.02 times) is not distinguishable from noise there. Imports did not change. `bench/validate.py` on that machine gives `PASS` for `v10` and for `v07`, with the same differences from the original's arrays (sensitivity 6.0e-08 in log; gradient entry 158 at 0.29 of the allowed difference); the files are in `bench/results/amd-epyc-7763-64-core-processor-16g-linux/validation/`. These seconds are not comparable with the 0.90 s on the Intel Xeon or the 0.72 s on the AMD EPYC 9334: a different machine.

The same three versions on the Mac:

Apple M5 MacBook Air, 16 GB, thread limits set to one, 10 shuffled rounds, a fresh process per run (`bench/results/apple-m5-16g-darwin/sessions/2026-11-21T23-38-07Z_run.json`). The Mac was busy during the session, so the totals are higher than in earlier tables.

| Version | Changes | Complete first run (median) | Fastest | Build phase | Gradient phase |
| --- | --- | ---: | ---: | ---: | ---: |
| `v07-numpy-host` | 1–12 | 0.44 s | 0.42 s | 43.1 ms | 38.3 ms |
| `v09-active-derivatives` | + 14 | 0.43 s | 0.34 s | 41.8 ms | 30.6 ms |
| `v10-host-swept-values` | + 15 | 0.43 s | 0.33 s | 37.7 ms | 30.7 ms |

The two phases moved as on Linux. The complete first run did not move by more than the noise of the Mac (the harness marks both steps as not distinguishable from noise); the Linux table above is the evidence for the total. Together the two changes remove about 12 ms of work on this Mac, where a first run is about 330 ms on a quiet machine and half of that is `import jax`.

### 16. Solve with SciPy's sparse LU: all frequencies in one matrix, one factorisation per evaluation

- **What:** on the NumPy path (change 12) the sparse signal systems are solved by SciPy's sparse LU solver (SuperLU), called from NumPy code, where klujax was called from the JAX program before. Three things change with it (`_Factors`, `_forward` and `_signal_adjoint` in `differometor/host.py`):
  - **All frequencies in one matrix.** Every frequency has its own signal matrix with the same pattern of non-zero entries. They are stacked into one block-diagonal matrix, which is factorised in a single call and solved in a single call per right-hand side. The order in which the solver eliminates the columns is found once per pattern and reused for every frequency and every later evaluation.
  - **One factorisation per evaluation.** An LU factorisation splits a matrix into two triangular factors; once they exist, a solve with the matrix, its transpose or its conjugate transpose is cheap. Before, each of the four solves per frequency (signal and noise in the forward pass, their two adjoints in the gradient) factorised again: 404 factorisations per evaluation at 101 frequencies. Now there is one. The gradient needs none of its own:
    - the noise's adjoint solve is a vector known in the forward pass times a real factor that is only known in the reverse pass, so the solve is done in the forward pass and scaled later;
    - the signal's adjoint is a combination of the noise weights whenever the signal is read out at the noise detector. The code checks this at run time, for the setup it is given, and factorises a second time if it does not hold.
  - **One step of iterative refinement after every solve:** compute the residual, solve for the correction, add it.
- **Why:** three costs went away at once. The solver was the last large piece of the program that JAX had to trace and compile. The factorisation was repeated four times per frequency. And klujax had to be imported on every run. SciPy is already in the organizers' `requirements.txt`, so no package is added.
- **Why the earlier attempt failed and this one does not:** SciPy's solver was tried once before (branch `scipy-host`) and dropped: it was slower than KLU and up to 6e-05 off in the noise on setups with a homodyne detector. Both had causes that could be removed. It was slower because of SciPy's wrapper and SuperLU's default settings, which are tuned for much larger matrices; calling the factorisation directly, with the column order fixed beforehand and its block sizes set to one, cut the time for 101 matrices from 86.7 to 38.5 ms (development laptop under Windows, a guide for the decision and not a recorded result). It was less accurate because these matrices have entries from 5e-20 to 1e+13; the refinement step repairs that.
- **This change alters the numbers**, because the solver is a different one. Against the original's arrays on the same machine (Intel Xeon, `bench/validate.py`), before and after:

  | Largest difference from the original | Changes 1–15 (`v10`) | Changes 1–18 (`v11`) |
  | --- | ---: | ---: |
  | Signal, relative to its largest value | 1.36e-07 | 1.33e-07 |
  | Noise, relative to its largest value | 4.00e-08 | 4.04e-08 |
  | Sensitivity, in log (judges' limit 1e-5) | 7.1e-08 | 6.8e-08 |
  | Gradient, relative to its largest entry | 9.15e-08 | 9.11e-08 |
  | Gradient entries outside the harness tolerance | 0 of 193 | 0 of 193 |

  So the new solver is as close to the original as the old one. On the hardest of the other setups (`uifo-20-5000`) it is closer: 5e-11 in log sensitivity, where `v10` shows a signal difference of 1.6e-06 and a `DEVIATION` verdict.
- **Effect** (one pinned core of the Intel Xeon, medians of five, table below): tracing and compiling the forward and gradient functions went from 158 to 66 ms, the gradient run from 109 to 58 ms. The forward run went from 87 to 112 ms, because it now carries the factorisation, three solves and their refinement.
- **General:** the stacked layout, the column order and the check behind the single factorisation are derived from the setup the code is given. No size and no pattern is written into the code. The fallback (`simulate_jax`, used for setups with linking functions) still uses klujax and is unchanged.

### 17. Keep the dense matrices out of the calls into NumPy

- **What:** with the solver gone from the JAX program (change 16), the forward stage and the gradient stage are each a single call from JAX into NumPy. The two dense system matrices that the build step prepares no longer travel through those calls: the sparse layouts carry the entries the NumPy code needs (`base` in `sparse_layout`, `differometor/simulate.py`).
- **Why:** what is handed to such a call, and what the forward stage keeps for the gradient, is carried along by the JAX program. Most of that volume was two dense matrices of which only the non-zero entries are read.
- **Effect:** the arrays kept for the reverse pass shrink from 26.7 MB to 5.2 MB on the supplied setup. The same values are read, from a different place, so no result should change; that was not checked separately from change 16. It was timed together with parts of change 18 (table below).

### 18. Load less when the package is imported

- **What:**
  - SuperLU's compiled module is loaded directly from its file. `import scipy.sparse.linalg` would load the whole of SciPy's sparse and linear-algebra packages for one function.
  - klujax is imported only when the fallback runs.
  - The jaxlib version, which the package checks at import, is read from jaxlib itself, not from the installed-packages database.
  - plotly, needed only for `visualize_setup`, is loaded when that function is first asked for.
- **Why:** on a first run of 0.7 s, importing is the largest single step, and `benchmark.py` counts it.
- **Effect:** the import step went from 335 to 313 ms (one pinned core of the Intel Xeon, medians of five, `v10` against `v11`). No calculation is touched.
- **What it does not do:** it does not move work out of the timed region. Importing JAX before `benchmark.py` starts its timer (through a start-up hook) would look faster and was rejected for that reason.

### Changes 16 to 18 measured

All on the Nebius machine with the organizers' processor type: Intel Xeon Platinum 8468, pinned to logical CPU 8 with the official `--cpus 1`, 32 GiB memory cap, swap off, 22 November 2026. Raw data: `bench/results/intel-xeon-platinum-8468-197g-linux/`. The sessions there list `v11` under its branch name, `no-klujax-scipy`; it is the same commit (`f3f1454`).

The supplied setup, through the harness, five shuffled rounds, a fresh process per run (`sessions/2026-11-22T00-14-25Z_run.json`), medians in milliseconds:

| Step | `v07` (changes 1–12) | `v10` (1–15) | `v11` (1–18) | `v11` minus `v10` |
| --- | ---: | ---: | ---: | ---: |
| Imports | 329 | 335 | 313 | −22 |
| Build the setup | 108 | 87 | 92 | +5 |
| Forward: trace and compile | 61 | 61 | 19 | −42 |
| Forward: run | 88 | 87 | 112 | +25 |
| Readout: trace and compile (the benchmark's own function) | 59 | 60 | 63 | +3 |
| Gradient: trace and compile | 99 | 97 | 47 | −50 |
| Gradient: run | 129 | 109 | 58 | −51 |
| **Complete first run** | **877** | **840** | **705** | **−135** |

`v11` against `v10`, as the ratio within each round: 1.20 times faster (1.17 to 1.21), in all five rounds. Against `v07`: 1.24 times (1.22 to 1.29). `v10` against `v07`: 1.04 times (1.02 to 1.06), which is the first timing of changes 14 and 15 on this machine and agrees with the Codespace figure above. The harness noted a slower control workload in 2 to 3 of the 5 rounds for every version, and the noise floor of this machine has not been measured, so the per-round ratios are the evidence, not the seconds. An earlier session the same night, before changes 14 and 15 were merged in (`sessions/2026-11-22T00-08-43Z_run.json`): 0.67 s against 0.85 s for `v07`, 1.27 times (1.26 to 1.29).

Two other setups, because the final evaluation uses a different one. Complete first run, by hand, versions run alternately, medians:

| Setup | Original (one run) | `v10` | `v11` | Runs | Raw data |
| --- | ---: | ---: | ---: | --- | --- |
| aLIGO, the organizers' second benchmark (`benchmark_aligo.py`, 29 design variables) | 26.25 s | 0.62 s | 0.50 s | 4 each | `aligo/` |
| `random-layout-1`: another 3×3 arrangement with a balanced homodyne readout (`tests/variations.py`) | 41.55 s | 0.83 s | 0.69 s | 4 and 6 | `variations/`, `variations-v10/` |

Both benchmarks print `PASS` in every one of these runs. Over all 18 scenarios of `tests/variations.py` that take the NumPy path, `v11` needs 593 to 880 ms where `v10` needs 675 to 1045 ms (one run each, `variations/2026-11-22T00-23-50Z.json` and `variations-v10/2026-11-22T00-24-58Z.json`). The scenario that takes the JAX fallback (`amplitude-modulation`) is not faster: 2.57 s against 2.47 s, one run each.

How the total moved while the three changes were built. Quick comparisons by hand, three or four alternating runs per version, medians. The machine's speed drifted during the night, so only the two numbers of one row can be compared:

| State of the code | `v07` in the same minutes | New code |
| --- | ---: | ---: |
| Change 16, with SuperLU's module loaded directly (`b0b902a`) | 835 ms | 725 ms |
| + change 17, + the jaxlib version and plotly parts of change 18 | 812 ms | 658 ms |
| + the build step fixed and the matrix product of the refinement rewritten (`05da4b9`) | 834 ms | 667 ms |
| + changes 14 and 15 merged in (`f3f1454`, which is `v11`); `v10` itself: 879 ms | 913 ms | 720 ms |

The same pair on the Mac, by a teammate (Apple M5 MacBook Air, 16 GB, thread limits set to one, 10 shuffled rounds, the machine busy with a demo page, `bench/results/apple-m5-16g-darwin/sessions/2026-11-22T00-31-37Z_run.json`): `v11` 0.35 s against 0.41 s for `v10` (medians), 1.20 times (0.93 to 1.54), faster in 9 of 10 rounds. `bench/validate.py`, `tests/test_host.py` and `tests/variations.py` pass there too (18 `PASS` and the known `DEVIATION`).

The longer record of this work, with every experiment, is [`docs/no-klujax-scipy.md`](docs/no-klujax-scipy.md); a summary with charts is [`docs/no-klujax-scipy-results.pdf`](docs/no-klujax-scipy-results.pdf).

### 19. Refine only the solve that needs it

- **What:** change 16 refined every sparse solve once (solve, compute the residual, solve again for the correction). The solve for the noise weights keeps that step. The solve that gives the signal and the adjoint of the noise is now a plain solve. The column order of the factorisation is taken from SuperLU's permutation by its documented convention, without the two trial factorisations that chose between the two readings.
- **Why:** refinement is there for the small entries of a solution, which the noise depends on. The signal is read out where it is large, and the adjoint only enters the gradient.
- **Effect:** the first forward call went from 36.5 to 28.8 ms on the Apple M5 (8 alternating fresh runs each).
- **Accuracy:** this changes rounding. NumPy path against JAX path (`tests/test_host.py`): noise unchanged (5e-10 on the supplied setup, 1e-12 and 2e-13 on the two homodyne setups), gradients at the same level (6e-08 on the second UIFO), signal of the random layouts within 1.4e-10 where it was 2e-15. Against the original, `bench/validate.py` gives sensitivity 6.1e-08 in log (limit 1e-5) and `PASS`.

### 20. Keep the assembled systems of the forward pass for the reverse pass

- **What:** a forward pass that will be followed by a gradient builds the carrier, signal and noise systems together with the derivatives of the component functions and keeps them on the host. JAX carries a ticket for them. The reverse pass used to build all of it a second time.
- **Why:** building the systems was a third of the gradient call.
- **Safe by construction:** the ticket is a random number, at most two sets are kept, and a reverse pass whose ticket is gone, or whose parameters do not match, builds the systems again as before. `tests/test_host.py` has that case.
- **Effect:** the gradient call went from 14.6 to 9.8 ms on the Apple M5 (10 alternating fresh runs each), for 1.4 ms more in the forward call. Outputs and gradients are bit-for-bit the same (20 arrays of five setups compared byte by byte).

### 21. On Linux, keep freed memory for reuse

- **What:** at import the package sets three options of the C library's memory manager (`mallopt`): large blocks come from the heap, and the heap is not handed back when they are freed. Skipped if the user has set the corresponding `MALLOC_*` variables or `GLIBC_TUNABLES`.
- **Why:** the simulation creates and frees many arrays of a megabyte or more. By default glibc returns each to the operating system and fetches the next one fresh, and every fresh page costs a page fault.
- **Effect:** the gradient call went from 34 to 27 ms on the Intel Xeon Platinum 8468 and from 35 to 26 ms on an AMD EPYC 7763. No number changes. On macOS it does nothing.
- **Cost:** freed memory stays with the process.
- **How this differs from the withdrawn change 13:** that one switched off something NumPy does and cost 4% on a quiet machine; this one is faster on a quiet machine.

### Changes 19 to 21 measured

Intel Xeon Platinum 8468 (second Nebius VM, pinned to logical CPU 15 with the official `--cpus 1`, 32 GiB memory cap, swap off), 10 shuffled rounds, a fresh process per run (`bench/results/intel-xeon-platinum-8468-197g-linux/sessions/2026-11-22T10-05-52Z_run.json`), medians in milliseconds. The session lists `v12` under its commit, `f4f5c40`:

| Step | `v11` (changes 1–18) | + changes 19 and 20 (`f1bc7a7`) | `v12` (changes 1–21) | `v12` minus `v11` |
| --- | ---: | ---: | ---: | ---: |
| Imports | 294 | 292 | 292 | −2 |
| Build the setup | 80 | 80 | 82 | +2 |
| Forward: trace and compile | 17 | 18 | 17 | 0 |
| Forward: run | 108 | 95 | 98 | −10 |
| Readout: trace and compile (the benchmark's own function) | 61 | 61 | 61 | 0 |
| Gradient: trace and compile | 43 | 42 | 43 | 0 |
| Gradient: run | 55 | 34 | 27 | −28 |
| **Complete first run** | **666** | **631** | **626** | **−40** |
| Fastest and slowest of the ten runs | 650 and 681 | 620 and 640 | 612 and 640 | |

`v12` against `v11`, as the ratio within each round: 1.07 times faster (1.03 to 1.11), in all ten rounds; the noise floor of that machine is ±2.4%. Change 21 on its own is 1.01 times (0.98 to 1.03) for the total, inside the noise; its effect is the gradient run. `bench/validate.py` on that machine gives `PASS` for `v12`. `v11` took 0.71 s in the session of the night before on the first VM and 0.67 s here: a different VM, so compare within a session only.

One pinned core of an AMD EPYC 7763 (GitHub Codespace, official `--cpus 1`, 8 shuffled rounds per session, `bench/results/amd-epyc-7763-64-core-processor-16g-linux/sessions/`; the sessions list the commits as `cand2` = `f1bc7a7`, `tmp-cand3` = change 21 as first written, `cand4` = `f4f5c40` = `v12`):

| Session | `v11` | + changes 19 and 20 | + change 21 | Gradient run, the same three |
| --- | ---: | ---: | ---: | --- |
| `2026-11-22T09-42-13Z_run.json` | 0.791 s | 0.748 s | 0.739 s | 54.9, 35.1, 26.2 ms |
| `2026-11-22T09-52-29Z_run.json` | 0.789 s | 0.755 s | 0.751 s (`v12`) | 56.2, 36.3, 27.2 ms |

`bench/validate.py` on that machine gives `PASS` for `v12` (`bench/results/amd-epyc-7763-64-core-processor-16g-linux/validation/cand4@f4f5c40d53d8.json`), and `tests/test_host.py` and `benchmark_aligo.py` pass there.

### Tried and not adopted

- **No refinement in any solve:** all tests pass, but on the second UIFO setup (homodyne) the noise moves from 1e-12 to 1e-08 of the JAX result and the gradients from 6e-08 to 2e-06. Change 19 keeps the refinement where those numbers come from.
- **A hand-written gradient rule for `signal_detector`**, to shorten the benchmark's readout compilation: about 2 ms on the Apple M5, inside the noise, and forward-mode differentiation of that function would be lost.
- **More compiler flags** (`xla_cpu_enable_fast_math`, `xla_cpu_copy_insertion_use_region_analysis`, `xla_dump_disable_metadata`, `xla_disable_all_hlo_passes`): no effect on the Apple M5; the last one does not compile the programs at all.

From the work on changes 16 to 18 (measurements and machines in [`docs/no-klujax-scipy.md`](docs/no-klujax-scipy.md); "development laptop" means short runs under Windows that guided a decision and are not recorded results):

- **Switching the refinement step of change 16 off:** the forward run goes from 109 to 84 ms and the complete first run from 658 to 645 ms (Intel Xeon, one pinned core, four alternating runs). Agreement with the original on `uifo-20-5000` falls from 5e-11 to 2e-08 in log sensitivity. Kept on: 13 ms are not worth the margin on a setup we have not seen.
- **SuperLU without refinement:** up to 5e-04 off in log sensitivity on Voyager with the default column order (development laptop, against a rounding-free solution). Not adopted.
- **Scaling rows and columns before factorising:** no help without refinement, sometimes worse (development laptop). Not adopted.
- **Iterative (Krylov) solvers, GMRES and BiCGSTAB,** also with the LU factors of the neighbouring frequency as a preconditioner: no convergence in 2,000 iterations (development laptop). Not adopted.
- **A banded LAPACK solver after reordering the matrix into a band:** three times slower than SuperLU and 9e-06 away from the klujax result (development laptop). Not adopted.
- **Loading SciPy's compiled sparse-product module directly,** the way change 18 loads SuperLU: the module took 49 ms to load on the laptop, more than the products it would speed up. Not adopted; the products are done in NumPy.
- **Importing JAX before `benchmark.py` starts its timer** (through a start-up hook): not tried. It would move work out of the timed region instead of removing it.
- **A different sparse solver package:** not tried. The aim was to use only packages the organizers' `requirements.txt` already lists.

Older entries:

- **Precomputing the index bookkeeping of the NumPy evaluation** (which cells belong to which formula, where each result goes): profiled at 1 to 2 ms per evaluation on the Apple M5, under half a percent of a first run. Not done.
- **A stand-in for `jaxtyping` while klujax is imported:** klujax imports `jaxtyping` for one type annotation, and that import is 10 of the 12 ms klujax takes to load (Apple M5), about 3% of a first run. It would mean interfering with another package's imports. Measured, not done. No longer relevant since change 18: klujax is only imported when the fallback runs.

- **`cos(x) + 1j*sin(x)` instead of `exp(1j*x)`:** no measurable difference in compile time (about 3 ms per use either way).
- **Reusing one factorisation for the solves:** change 12 already cut the sparse factorisations from six per frequency to four (two forward, two in the gradient). Going further was measured on the main example (Apple M5): the two forward solves take 12.2 ms when each factors from scratch and 7.4 ms when they share one factorisation. Over both stages that is about 10 ms per evaluation, roughly 3% of the complete first run. It needs the solver's factorisations to be kept in memory that Python does not manage and released by hand, where a mistake means a crash or a leak, not a wrong number. Not adopted in this form. **Superseded by change 16**, which reuses one factorisation through SciPy, where Python manages that memory.
- **One extra refinement step in every sparse solve** (solve, compute the residual in double precision, solve again for the correction): it brings entry 17 of the 20 to 5000 Hz scenario from 0.1% to 0.002% of the refined value, and the supplied setup's gradient from 1e-08 to 1e-09. It costs a second solve per system: a compiled evaluation of loss and gradient goes from 52 to 79 ms on the supplied setup (Apple M5), about 8% of a complete first run. The judges' limit is 1e-5 in log sensitivity, where ours is at 3e-08 without it, and the supplied setup's gradient is already within 2e-07. Not adopted with KLU; it was the change to make if a gradient accurate to 0.01% outside the design band was ever required. **Change 16 does refine once**, with SciPy's solver: there the second solve reuses the factors, and the step is what makes that solver accurate enough.
- **SciPy's sparse solver (SuperLU) in place of KLU** (branch `scipy-host`): it would make klujax unnecessary except for the fallback, and it factors each matrix once per stage. The official benchmark passes, a little slower (Apple M5: 0.42 s against 0.33 s). But `tests/test_host.py` fails on the two setups with a homodyne detector: the noise differs from the KLU result by 2e-05 (second UIFO) and 6e-05 (Voyager), and the Voyager example's sensitivity ends up 7e-03 away from the original code in log, where KLU is 6e-11 away. The matrices there have entries from 5e-20 to 1e+13; SuperLU matches the dense reference solver in the large entries of the solution (1e-09) but is less accurate in the small ones that the noise depends on. Not adopted at the time. **Superseded by change 16**, which removes both causes (the solver's settings for the speed, a refinement step for the accuracy).

### Withdrawn: do not ask Linux for large memory pages (was change 13, version `v08`)

**Decision (22 November 2026): not in the submitted code.** This change was added, measured twice on Linux and then taken out again. Its cost is reproducible (about 0.03 s, 4%, on a quiet machine); its benefit appeared in one test session and not in the next; and the severe slow state it was aimed at has never been reproduced, so we cannot claim it protects against that. The remaining uncertainty is about the scoring machine, not something more runs of ours would settle. The measurements stay here, and it can be put back in either of two ways if the scoring machine turns out to be a shared, busy host:

- without touching the code: set `NUMPY_MADVISE_HUGEPAGE=0` in the environment before starting `benchmark.py`;
- in the code: `git cherry-pick 487e6ef` (tag `v08-no-large-pages` holds that state).

What it was, as written while it was in:


- **What:** when the package was imported it switched off NumPy's request for large memory pages (`numpy._core.multiarray._set_madvise_hugepage(False)` in `differometor/__init__.py`). No calculation changes; every output is bit-for-bit the same.
- **Why:** this is a change for a steady time, not for a faster one. By default NumPy asks Linux for 2 MB pages for every array above 4 MB. Where large pages are only given on request (the usual Linux setting, and the one on the AMD cloud host), the kernel then tries to assemble such a page inside our process at the moment the array is first touched. On a machine whose memory other programs keep busy, that made single steps of our NumPy path take two to five times as long in a third of the runs, and up to ten times in one session ("The same on one pinned Linux core" above). JAX's own memory never makes that request, which is why only the NumPy steps were affected.
- **Effect, measured with the equivalent environment variable** (`NUMPY_MADVISE_HUGEPAGE=0`, AMD EPYC 9334 container, one core, 8 runs per setting and condition, `bench/results/amd-epyc-9334-32-core-processor-504g-linux/diagnosis/`):

  | | Request on (before) | Request off (this change) |
  | --- | ---: | ---: |
  | Complete first run, quiet machine, median | 0.71 s | 0.74 s |
  | Complete first run next to memory-churning processes, median | 1.14 s | 1.16 s |
  | Same, slowest run | 1.51 s | 1.23 s |
  | Runs with a NumPy step at more than twice its normal time | 8 of 24 | 0 of 8 |

  So it costs about 0.03 s (4%) on a quiet machine and removes the outliers on a busy one. The scored number is a single run, where an outlier costs far more than 0.03 s.
- **Second test, with the switch itself** (commit `8ff051a`, same kind of AMD container, 72 runs, `bench/results/amd-epyc-9334-32-core-processor-504g-linux/diagnosis-change-13/`). The package as committed against the same package with `NUMPY_MADVISE_HUGEPAGE=1`, which leaves NumPy's request on:

  | | Request on | Request off (this change) |
  | --- | ---: | ---: |
  | Large pages handed to our process per run, quiet machine | 54 to 64 | 0 |
  | Complete first run, quiet machine, median of 24 | 0.71 s | 0.74 s |
  | Complete first run next to memory-churning processes, median of 12 | 1.11 s | 1.16 s |
  | Same, slowest run | 1.15 s | 1.26 s |
  | Runs with a NumPy step at more than twice its normal time | 0 of 12 | 0 of 12 |

  The switch works (no large pages are handed out any more) and its cost is confirmed: 0.03 s on a quiet machine, about 0.05 s under memory pressure. **The benefit did not show in this session:** the outliers did not occur with either setting, so there was nothing for the switch to remove. Both tests together give outliers in 8 of 36 runs with the request on and 0 of 20 with it off, which would happen by chance about one time in 50.
- **A caution on the statistics:** all eight outliers came from one session. Counting every run as independent, as the "one time in 50" figure does, makes the evidence look stronger than it is at the level of sessions: one session showed the effect and one did not.
- **What is not shown:** the worst state we saw (0.74 s becoming 2.8 s) did not occur again during the test, so there is no measurement of this change against it. 0 of 8 against 8 of 24 would happen by chance about one time in 25.
- **Why the run-time switch and not the environment variable:** `benchmark.py` imports NumPy (through JAX) before it imports this package, and NumPy reads `NUMPY_MADVISE_HUGEPAGE` only when it is imported. The switch does the same thing afterwards.
- **A dependency to know about:** the switch is private to NumPy. It exists in the pinned `numpy==2.5.5`; if a future NumPy removes it, the call is skipped and the behaviour is as before this change. A user who sets `NUMPY_MADVISE_HUGEPAGE` themselves keeps their choice.
- **How it stays general:** it does not depend on the setup. On macOS and Windows it has no effect.
- **Checks after the change (Apple M5):** official benchmark `PASS`; `bench/validate.py WORKTREE` `PASS` with the same differences from the original as before the change (sensitivity 5.8e-08 in log, gradient 9.2e-08 of the largest entry); `tests/test_host.py` ends with `ALL CHECKS PASSED`; `tests/handwritten_jacobians.py` agrees on all 19 rules; `tests/variations.py` gives 18 `PASS` and the one known `DEVIATION` on 19 scenarios. The Linux timing comparison with the switch itself is the second test above.

## Step by step, changes 1–3 (Apple M5, official benchmark without CPU pinning)

| Version | Total | Compile | Forward | Gradient |
| --- | ---: | ---: | ---: | ---: |
| Original | 12.44 s | 3.69 s | 4.42 s | 3.70 s |
| + change 1 | 8.20 s | 3.73 s | 1.76 s | 1.57 s |
| + change 2 | 7.37 s | 2.39 s | 1.96 s | 1.78 s |
| + change 3 | 3.61 s | 2.41 s | 0.04 s | 0.06 s |

Change 4 was measured on Linux only (table above).

## Correctness checks

- **The current version (`v11`, changes 1–18), on the Intel Xeon machine, 22 November 2026.** Change 16 replaced the solver, so every check was repeated for it. Logs: `bench/results/intel-xeon-platinum-8468-197g-linux/no-klujax-scipy-logs/`.

  | Check | Result |
  | --- | --- |
  | Official `benchmark.py` and `benchmark_aligo.py` | `PASS` in every run; distance from the public reference 5.374e-06 in log sensitivity (limit 1e-5), as for the original |
  | `bench/validate.py`: every output and all 193 gradient entries against the original's arrays from the same machine | `PASS`: signal 1.3e-07, noise 4.0e-08, sensitivity 6.8e-08 in log, gradient 9.1e-08 of its largest entry, 0 entries outside the tolerance (`validation/no-klujax-scipy@f3f14542ea40.json`) |
  | `tests/test_host.py`: NumPy path against JAX path on 13 cases, `jit`, `vmap`, finite differences | `ALL CHECKS PASSED` |
  | `tests/handwritten_jacobians.py` | all 19 rules agree with JAX |
  | `tests/variations.py`: 19 other setups against the original | 18 `PASS`, 1 `DEVIATION`, 0 `FAIL` (`variations/2026-11-22T00-23-50Z.json`) |

  The one `DEVIATION` is `band-20-5000`, the scenario discussed further down, where the original's own gradient is the less accurate one. `v10` in the same hour got 17 `PASS` and 2 `DEVIATION`; its second one, a signal difference of 1.6e-06 on `uifo-20-5000`, does not occur with `v11`. A teammate repeated `validate.py`, `tests/test_host.py` and `tests/variations.py` on the Apple M5 with the same verdicts.

  The remaining entries of this section were made with earlier versions and are kept as they were written.
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
- **Sixteen other setups, original against ours (`python tests/variations.py`, Apple M5 MacBook Air, 16 GB, thread limits set to one; session `bench/results/apple-m5-16g-darwin/variations/2026-11-21T19-46-00Z.json`):** the jury scores a slight variation of the supplied example, so each case changes something a variation could change. The original code ran once per case and its results are stored in `bench/reference/variation-<case>.npz`. "Sensitivity" is the largest difference from the original in ln(sensitivity) (judges' limit 1e-5). "Gradient outside" counts the entries that differ from the original's by more than 0.01% of their own size. "Finite differences" is the largest relative error of our gradient in four random directions. The times are one first run each on this Mac; the original's are inflated by swapping.

  | Case | What it is | Path | Ours | Original | Sensitivity | Gradient outside | Finite differences | Verdict |
  | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | `amplitude-modulation` | Supplied layout and design values in amplitude-modulation mode, which has linking functions, so our code takes its JAX fallback | JAX fallback | 1.00 s | 14.7 s | 8.0e-09 | 0 of 193 | 3.5e-06 | PASS |
  | `band-20-5000` | Supplied setup at 101 frequencies from 20 to 5000 Hz | NumPy | 0.34 s | 15.0 s | 6.7e-08 | 3 of 193 | 3.6e-04 | DEVIATION |
  | `boundary-swap` | Supplied layout with the laser at boundary 10 and the squeezer at boundary 20 exchanged | NumPy | 0.33 s | 14.0 s | 2.7e-11 | 0 of 193 | 2.8e-06 | PASS |
  | `center-swap` | Supplied layout with center 22 made a directional beamsplitter and center 11 a plain one | NumPy | 0.34 s | 16.3 s | 6.3e-11 | 0 of 193 | 8.1e-04 | PASS |
  | `detector-moved` | Supplied layout with the detector moved from boundary 03 to boundary 30 | NumPy | 0.33 s | 17.6 s | 1.5e-07 | 0 of 193 | 1.0e-04 | PASS |
  | `frequencies-51` | Supplied setup at 51 frequencies from 800 to 3000 Hz | NumPy | 0.31 s | 7.5 s | 5.8e-08 | 0 of 193 | 3.6e-06 | PASS |
  | `homodyne-readout` | Supplied layout with the detector replaced by a balanced homodyne readout | NumPy | 0.34 s | 13.7 s | 8.0e-08 | 0 of 195 | 2.8e-05 | PASS |
  | `random-layout-1` | A random 3x3 layout (uifo with random_seed=1) with random design values | NumPy | 0.34 s | 12.5 s | 3.0e-09 | 0 of 193 | 8.9e-07 | PASS |
  | `random-layout-2` | A random 3x3 layout (uifo with random_seed=2) with random design values | NumPy | 0.33 s | 12.3 s | 6.6e-10 | 0 of 192 | 9.2e-06 | PASS |
  | `random-layout-3` | A random 3x3 layout (uifo with random_seed=3) with random design values | NumPy | 0.34 s | 13.0 s | 2.3e-09 | 0 of 197 | 1.5e-07 | PASS |
  | `size-2` | A random 2x2 layout with random design values | NumPy | 0.30 s | 4.6 s | 2.1e-08 | 0 of 102 | 3.6e-07 | PASS |
  | `size-4` | A random 4x4 layout with random design values, at 21 frequencies (the original would not fit in 16 GB with 101) | NumPy | 0.32 s | 8.0 s | 1.4e-10 | 0 of 315 | 7.2e-06 | PASS |
  | `supplied` | The supplied example, unchanged (control: must reproduce benchmark.py) | NumPy | 0.34 s | 12.6 s | 5.8e-08 | 1 of 193 | 9.5e-06 | DEVIATION |
  | `uifo-20-5000` | The second public setup (other layout, balanced homodyne readout) at 101 frequencies from 20 to 5000 Hz | NumPy | 0.33 s | 14.3 s | 1.7e-07 | 0 of 192 | 1.1e-04 | PASS |
  | `values-random` | Supplied layout; every design variable drawn at random from the optimizer's ranges | NumPy | 0.34 s | 13.4 s | 7.3e-08 | 0 of 193 | 6.8e-07 | PASS |
  | `values-shifted` | Supplied layout; every design variable moved by about 1% | NumPy | 0.34 s | 20.5 s | 8.8e-09 | 0 of 193 | 2.1e-07 | PASS |

  All sixteen are far inside the judges' limit. The two `DEVIATION` verdicts are about single gradient values, described next.
- **The stricter gradient check of the harness, the values it flags, and which code is right:** `bench/validate.py` compares each gradient entry with the original's. Its tolerance was 0.01% of the entry's own size when this was investigated; the team has since set it to 0.1%, and with that all seven versions get `PASS` from the harness (rerun 21 November, 21:50 UTC). At 0.01% the picture was this. On the supplied setup 192 of 193 entries pass and one does not: entry 158 (`m01.reflectivity`), -4.8901e-04 in ours and -4.8885e-04 in the original, 0.03% apart. On the supplied setup at 20 to 5000 Hz three entries differ, by 0.03%, 0.3% and 0.65%. These entries are tiny next to the largest gradient entry (between one part in 100,000 and one part in a billion), and two of them belong to a laser set to 1e-12 W and the mirror in front of it.

  To find out which code is right, `python tests/refined_solves.py <scenario>` repeats our calculation with every sparse solve refined by iterative refinement: solve, compute the residual in 100-digit arithmetic, correct, repeat until the correction is below 1e-20 of the solution. That gives a reference far more accurate than either code's own solve. It is numerical evidence, not a formal proof that every digit of the reference is right; the last column of the script's output (inputs moved by one unit in the last digit) shows how far the reference itself can be trusted. Results (Apple M5):

  | Distance from the refined result | Ours (KLU) | Original (dense solver) |
  | --- | ---: | ---: |
  | Supplied setup: readout signal | 2.2e-08 | 8.9e-08 |
  | Supplied setup: noise | 7.0e-10 | 4.0e-08 |
  | Supplied setup: ln(sensitivity) | 2.4e-08 | 6.0e-08 |
  | Supplied setup: gradient entries off by more than 0.01% | 0 of 193 | 1 of 193 |
  | Supplied setup: entry 158 (refined value -4.890100922e-04) | 1e-08 | 3e-04 |
  | 20 to 5000 Hz: ln(sensitivity) | 3.2e-08 | 6.7e-08 |
  | 20 to 5000 Hz: gradient entries off by more than 0.01% | 1 of 193 | 3 of 193 |
  | 20 to 5000 Hz: entry 17, `mr12.reflectivity` | 0.1% | 0.6% |
  | 20 to 5000 Hz: entries 158 and 161 | 3e-07 and 1e-06 | 0.3% and 0.03% |

  **So the cause of the harness `FAIL` is rounding in the original's dense solver, not an error in ours.** On the supplied setup ours is within 2e-07 of the refined result on every gradient entry, and closer than the original on every output. Moving every input by one unit in its last digit changes the refined values by at most 4e-06, so the comparison is meaningful at this level. The harness's reference arrays come from the original, which is why it reported a failure at 0.01%. One value remains where ours is not within 0.01% either: entry 17 at 20 to 5000 Hz, 0.1% off (the original: 0.6%), in a setup that was optimized for 800 to 3000 Hz and is used here outside that band.

  **Decision (21 November 2026):** the harness's per-entry gradient limit is now 0.1% of the entry instead of 0.01%, because the original's own values are not reproducible to 0.01% on such entries. The counts in the table above were produced with 0.01%. Rerun with the new limits (Apple M5, 21 November, 21:50 UTC): `bench/validate.py` gives `PASS` for all seven versions, and of the scenarios in `bench/results/apple-m5-16g-darwin/VARIATIONS.md` the supplied setup is now `PASS` too. Entry 158 of the supplied setup, 0.03% apart, is inside the new limit; at 20 to 5000 Hz two entries are still outside it (0.65% and 0.3% apart; on both, ours is the closer one to the refined result), so that scenario stays `DEVIATION`. A second limit was added with it: the largest difference in the gradient, relative to its largest entry, must stay below 1e-6. That one guards the large entries, which 0.1% of their own size would leave loosely checked. The sixteen setups in the table above are all inside it; the largest value is 2.7e-07, for 20 to 5000 Hz (the second number in the "Gradient" column of `bench/results/apple-m5-16g-darwin/VARIATIONS.md`).
- **The "original" is the upstream code:** the report of the original run lists file fingerprints identical to those in `UPSTREAM.json`.
- **What is computed:** all 101 frequencies, all 193 design variables, double precision. No answers are stored and no calculation is skipped.

## Where the measurements are

- `bench/RUNS.md` lists every run made so far, on every machine, with its result and its raw data.
- `bench/VERSIONS.md` describes the registered versions (`v01` to `v12`; `v08` is withdrawn) and the baseline; `bench/results/` holds what the harness stored.
- `docs/no-klujax-scipy.md` is the full record of changes 16 to 18, and `docs/no-klujax-scipy-results.pdf` its summary with charts.
- The branch `legacy-klujax` holds `main` as it was before change 16 (version `v10` plus records), in case the klujax solver is needed again.
- `benchmarks/runs/` holds the report files of the runs made by hand before the harness was used.

## Reproduce

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python benchmark.py --cpus 1 --output benchmark-candidate.json
```

For the original, run the same commands on commit `02ec741`. It needs about 16 GB of free RAM.

```bash
python bench/validate.py WORKTREE     # official benchmark, then every output and gradient entry against the original's
python tests/variations.py            # nineteen other setups against the original, with times
python tests/test_host.py             # NumPy version against the JAX version
```

## Known limits

- Change 12 is new code for the whole reverse pass. It is verified against the JAX version on the setups listed above; a setup that uses a feature none of them has would be new ground, and `HOST_SIMULATION = False` switches back.
- On the contest's processor type (Intel Xeon Platinum 8468) the original ran once, the noise floor of that machine has not been measured (a session for it was started and stopped; `sessions/2026-11-22T00-10-37Z_aa.json` is incomplete), and `v07` was not timed on the two extra setups. The official report file and chart of a `v11` run there are not saved yet. Both gaps are closed on a second VM of the same type, where the noise floor is ±2.4% and the official `v11` report is saved next to a run of the original on the same instance (`bench/RUNS.md`, "On a second Nebius VM of the same type"); `v07` is still not timed on the two extra setups.
- Change 16 calls a private part of SciPy: the internal function `gstrf`, in a compiled module that is loaded from its file. Both exist in the pinned `scipy==1.18.1`. If either is missing, the code falls back to the normal import and then to the public `splu`, which is slower and gives the same results.
- Change 16 is new code for the solves, like change 12 was for the reverse pass. It refines every solve exactly once; a setup whose matrices are worse than any we tested could need more (`REFINEMENT_STEPS` in `differometor/host.py`). `HOST_SIMULATION = False` switches back to the JAX path with klujax.
- klujax is still a requirement, because the JAX fallback uses it.
- On one pinned CPU, starting the gradient before the previous JAX call has finished hangs inside the call into NumPy. `benchmark.py` waits after every phase and is not affected. This is not new: `v07` hangs in the same test. A caller who runs on one CPU has to wait for each result (`jax.block_until_ready`) before starting the next call.
- The harness marks `v10` and `v11` as `unofficial-benchmark`, because their `benchmark.py` is the organizers' newer file (with aLIGO), which differs from the one at the baseline commit. Nobody on the team edited it.
- The original could not complete the official benchmark on the 16 GB Codespace, so the official-style "before" on that machine is missing.
- On the AMD EPYC 9334 both versions have saved reports, but their output arrays were not brought back, so there is no array-by-array comparison from that machine. The comparisons on the Mac and the Codespace cover it.
- The carrier system (286×286, solved once) still uses the dense solver.
- A setup in amplitude-modulation mode has linking functions, so our code takes its JAX path there: correct, but 1.0 s where the NumPy path takes 0.34 s (Apple M5, case `amplitude-modulation`, changes 1–12). Changes 16 to 18 do not speed that path up: 2.57 s on the Intel Xeon, where the NumPy path takes 0.6 to 0.9 s.
- A negative reflectivity gives NaN in ours and a number in the original (seen while taking finite differences with too large a step). Reflectivities are bounded to 0..1 in the optimisation examples, so no setup we know has one.
- **The first-run time of `v07` was unstable on a busy shared Linux host; `v11` has not been run on such a host** (0.74 to 2.8 s on the second AMD instance; see "The same on one pinned Linux core"). The likely cause is NumPy's request for large memory pages. Switching it off was tried and withdrawn ("Withdrawn" above): it costs 4% on a quiet machine and its benefit showed in only one of two test sessions. On a shared scoring machine, `NUMPY_MADVISE_HUGEPAGE=0` is the setting to try. Under ordinary memory pressure every program gets slower, ours included (about 1.1 s against 0.71 s in the tests).
- `cos_arcsin` returns the same values as the complex arcsine route, but beyond total internal reflection its derivative has the opposite sign in the imaginary part. That only matters when a refractive index or an incidence angle is itself a design variable in such a setup; the benchmark's design variables are reflectivity, tuning, squeezing, angle of squeezing, power, mass and length.
