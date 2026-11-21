# Benchmarking

Everything about measuring speed-ups lives in this folder: how to run, the
scripts, and every stored result.

**What is measured.** The challenge scores the *complete first-run total*
printed by `benchmark.py`: imports, initialization, input build, all JIT
compilation, forward simulation, sensitivity, gradient and host transfer, on
one logical CPU. Warm runtime is reported separately and is not scored. The
harness never times anything itself; it launches each version's own
`benchmark.py` and records what that script prints.

**The method in five rules**

1. **One fresh process per sample.** The scored number includes imports and compilation, so in-process repeats would measure the wrong thing.
2. **The baseline runs once.** The unmodified baseline is run a single time on the benchmark machine. That run stores its output arrays (the correctness oracle) and its first-run total. No session reruns it: it is slow, it swaps on 16 GB, and every later version is far faster. "x faster than baseline" is the stored total divided by a new sample.
3. **Rounds in shuffled order.** A session runs R rounds; each round measures every version in it once, in a new random order.
4. **Compare within a round.** Versions timed in the same session (anchor, parent, candidate) are compared as per-round ratios of samples measured minutes apart, summarized as the median with its lowest and highest round. Use these columns for close calls; the baseline column rests on one stored sample.
5. **Nothing is dropped.** Every raw sample is stored, including failures. Suspicious samples are flagged, not removed.

## Where to run

All benchmarking runs on the **MacBook Air (Apple Silicon, 16 GB)**. The
Windows development laptop cannot run the workload (the unmodified baseline
peaks near 14.4 GiB). The harness also runs on Linux, where it pins a CPU and
uses the official `--cpus 1`.

Results are stored per machine under `results/<machine-id>/` and are never
compared across machines.

## First-time setup on the benchmark machine

Run from the repository root.

```bash
bash bench/setup_mac.sh          # Python 3.12 venv, pinned packages, editable install
source .venv/bin/activate
python bench/run.py selftest     # tests the harness itself with a stub; must end with SELFTEST PASS
python bench/run.py doctor       # checks Python, packages, power, memory, probes
python bench/run.py calibrate    # verifies the one-CPU limit and stores it for this machine
```

`selftest` comes first on purpose: it is the first time the macOS code paths
(process control, memory and power probes) execute. If it fails, fix that
before producing data.

Then run the baseline, once:

```bash
git tag v00-baseline 02ec741d8a1bf1a2278294ea74f9fc1d68f08f1f   # optional, for readable diffs
python bench/validate.py v00-baseline --make-reference
```

This is the only time the baseline executes. The one run stores:

- `reference/baseline_arrays.npz`: the correct outputs and gradient, which every later version is checked against;
- `results/<machine>/baseline.json`: its first-run total and phases, the fixed reference for "x faster than baseline";
- `results/<machine>/official/v00-baseline.json` and `.png`: the "before" report and plot for the submission.

Commit `bench/reference/` and `bench/results/`. Measure the noise floor
(`python bench/run.py aa`) once the first faster version is registered; it
runs that version against itself, not the baseline.

Before every session: plug the Mac in, turn Low Power Mode off, quit other
apps, and leave the machine alone until the session ends.

## Workflow for each speed-up

1. Edit the code, then `python bench/run.py quick` until the candidate holds up. `WORKTREE` is the uncommitted working tree; it is copied before the session starts, so you can keep editing while it runs.
2. `python bench/validate.py WORKTREE` to confirm it is still correct. A difference from the baseline arrays beyond the tolerance is a `FAIL`.
3. Commit, then `python bench/run.py add-version v01-<slug> --summary "..."`. This tags the commit, registers it in `versions.json` and appends a stub to `VERSIONS.md`. Fill the stub in.
4. `python bench/run.py full` reruns every registered version except the baseline in shuffled rounds. Commit the new session file and the regenerated summary.

## Commands

| Command | What it does |
| --- | --- |
| `python bench/run.py doctor` | Checks the machine before producing data. Exit code 1 on a blocking problem. |
| `python bench/run.py calibrate` | Runs a small JAX workload with and without the CPU limit and stores the limit in `results/<machine>/machine.json`. |
| `python bench/run.py quick` | Anchor + latest version + `WORKTREE` (never the baseline), 3 rounds. The inner loop. |
| `python bench/run.py full` | Every enabled version except the baseline, 5 rounds. Validates not-yet-validated versions first and leaves out any that fail. `--with-worktree` adds the candidate. |
| `python bench/run.py aa` | One version against itself. The spread is the machine's noise floor. Defaults to the anchor, else the latest version; `--version NAME` picks another. |
| `python bench/run.py run --versions a,b,c` | Custom session. Each item is a registry name, `WORKTREE` or any git ref. The only way to time the baseline again, should you ever want to. |
| `python bench/run.py add-version NAME --summary "..."` | Tags HEAD (or `--ref`) and registers the version. `--anchor` makes it the anchor. |
| `python bench/run.py official --versions NAME` | Complete official `benchmark.py` run (10 repeats, PASS, JSON + plot) into `results/<machine>/official/` for the submission. The baseline's report is already there from its one run. |
| `python bench/validate.py NAME` | Correctness check, cached per code state. `--allow-deviations`, `--force`. |
| `python bench/validate.py v00-baseline --make-reference` | The single baseline run: stores the oracle arrays, the baseline's first-run total and its official report. |
| `python bench/report.py` | Rebuilds `SUMMARY.md`, `history.csv` and `summary.png` from the stored sessions. No simulation. |
| `python bench/run.py selftest` | Tests the harness with a stub benchmark. No JAX needed, a few minutes. |

Session options: `--rounds N`, `--cooldown SECONDS`, `--timeout SECONDS`,
`--seed N`, `--note "text"`, `--no-gate`.

## How one sample is measured

1. **Snapshot.** A registered version is extracted from its git commit into `.cache/snapshots/<sha>/`; `WORKTREE` is copied from the tracked and untracked files. Each snapshot runs its own `benchmark.py` against its own `differometor/`.
2. **Machine state.** Load, free and compressed memory and swap counters are recorded before and after.
3. **Canary gate.** A fixed half-second single-thread workload runs first. While it is more than 5 % slower than the session's best, the harness waits (up to 2 minutes), so a throttled or busy machine gets time to recover.
4. **Run.** `python -u benchmark.py --repeats 3 --output <tmp>` in a fresh process, under the one-CPU limit.
5. **Record and stop.** Once the script prints `Complete first-run total (excluding validation)`, the harness stores the total and every phase and kills the process. Warm repeats and validation are not scored, and skipping them gives about three times more samples per hour.

Before the first round, each version is imported once and the result thrown
away, so bytecode files exist and the file cache is warm for every sample.

## The one-CPU limit

The judged run is pinned to one logical CPU. macOS cannot pin a process, and
`benchmark.py --cpus` refuses to run there, so on the Mac the harness sets
thread-count limits instead:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NPROC=1
```

`NPROC=1` is the one that matters for JAX itself: it sizes XLA's thread pool.
Measured with jaxlib 0.9.0, the BLAS variables alone left a matrix product at
about 9 CPU-seconds per wall-second, and the often-quoted `XLA_FLAGS` switches
changed nothing. With `NPROC=1` added, every probe workload ran at about 1.0.
`calibrate` repeats this check on the benchmark machine, and every sample
stores its own CPU-seconds per wall-second (`cpu/wall`).

This matters for decisions, not just bookkeeping: without the limit a laptop
runs the workload on many cores and ranks the bottlenecks differently from the
judged run.

## Correctness

`validate.py` runs the version's official `benchmark.py` to completion and
requires `PASS`. It then compares carrier, signal, noise, sensitivity, powers
and the gradient with `reference/baseline_arrays.npz`, the arrays the
unmodified baseline produced in its one run.

The official check alone is not enough for the gradient: it requires finite,
nonzero values and compares a single entry (`mb11.tuning`) with a finite
difference at about 2 %. The other 192 entries are not compared with anything.

| Verdict | Meaning |
| --- | --- |
| `PASS` | Official PASS and every array within tolerance of the baseline. |
| `FAIL` | No official PASS, wrong shapes, non-finite values, changed inputs, or an array that differs from the baseline by more than the tolerance (a deviation). |
| `PASS_WITH_DEVIATIONS` | Official PASS with deviations that someone inspected and accepted with `--allow-deviations`. The acceptance is stored in the cached verdict. |
| `PASS_NO_REFERENCE` | Official PASS; no reference arrays exist yet, so nothing was compared. |

Tolerances:

| Key | Default | Check |
| --- | --- | --- |
| `gradient_rtol` | `1e-3` | Every gradient entry on its own: `abs(a - b) <= floor + gradient_rtol * abs(b)`. |
| `gradient_floor_scale_rel` | `1e-12` | `floor` = this × the largest baseline gradient entry: what counts as numerically zero. |
| `gradient_scale_rel` | `1e-6` | The gradient as a whole: largest difference relative to the largest baseline entry. |
| `log_sensitivity_abs` | `1e-6` | Largest `abs(ln(sensitivity) - ln(baseline))`. The official bar against the public reference is `1e-5`, and the baseline already uses `5.4e-6` of it. |
| `outputs_scale_rel` | `1e-6` | Carrier, signal, noise, sensitivity, powers: largest difference relative to the array's largest magnitude. |

The gradient is checked entry by entry because its entries span many orders of
magnitude (norm about 7e4, `mb11.tuning` about 8e-5). A tolerance relative to
the largest entry would let the small ones be wrong by any factor.

The per-entry limit alone would in turn let the largest entries be wrong by
0.1%, and those decide the direction an optimizer moves in. `gradient_scale_rel`
covers them: it is the stricter of the two for every entry larger than a
thousandth of the largest one, and the per-entry limit is the stricter one
below that. A gradient has to meet both.

`gradient_rtol` was `1e-4` until 21 November 2026 and was raised to `1e-3` by
team decision. On very small gradient entries the original's own values are
only reproducible to about `1e-4`, so the stricter limit failed correct
versions (see "The one gradient value" in `VERSIONS.md`). A stored record
keeps the tolerances it was judged with in its `"tolerances"` field and is not
judged again on its own: rerun the version with `--force`.

`gradient_scale_rel` was added on the same day at `1e-6`, the value already
used for the other arrays. The largest value measured so far is 2.7e-07
(scenario `band-20-5000`, Apple M5, session
`results/apple-m5-16g-darwin/variations/2026-11-21T19-46-00Z.json`).

The other three numbers are first guesses. A reordered but correct calculation can
differ from the baseline by rounding noise larger than a guess; the printed
record shows the worst entry, its parameter name and how far past the limit it
is. If the deviation is noise, either accept that version
(`--allow-deviations`) or change the value in `versions.json` under
`"tolerances"` and rerun with `--force`.

## Reading the results

`results/<machine>/SUMMARY.md` shows the latest session, each version's
speed-up in every past session, and a chart.

- `vs X`: `3.58x (3.51-3.66)` is the median of the per-round ratios with the lowest and highest round.
- `vs v00-baseline (stored)`: the baseline was not rerun. Every round is divided into the baseline's one stored total, so the bracket shows only the version's own scatter, and the line under the table names the stored number and its date.
- `~`: not distinguishable from noise (the range includes 1.00x, or the gain is below the A/A noise floor).
- `*`: the reference ran under memory pressure, so the factor is overstated.
- `spread`: (max − min) / median of a version's totals.
- `cpu/wall`: about 1.00 means the run used one core.

| Flag | Meaning |
| --- | --- |
| `memory` | Swapping or heavy memory compression during the run, or a peak footprint above 80 % of RAM. |
| `slow-machine` | The canary was more than 5 % slower than the session's best, even after waiting. |
| `multi-core` | More than 1.15 CPU-seconds per wall-second: the limit did not hold. |
| `failed`, `timeout` | The run produced no total. The last output lines are stored in the session file; the full log is in `.cache/logs/`. |
| `unofficial-benchmark` | This version's `benchmark.py` differs from the baseline's. |

### Files

| Path | Content | In git |
| --- | --- | --- |
| `results/<machine>/sessions/<utc>_<mode>.json` | One session: configuration, machine, seed, run orders, every raw sample, summary. Never edited afterwards. | yes |
| `results/<machine>/machine.json` | The calibrated CPU limit and the evidence for it. | yes |
| `results/<machine>/baseline.json` | The baseline's one run on this machine: first-run total, phases, memory state. The reference for every "vs baseline" factor. | yes |
| `results/<machine>/validation/<name>@<id>.json` | Cached correctness verdicts. | yes |
| `results/<machine>/official/` | Official reports and plots for the submission. | yes |
| `results/<machine>/SUMMARY.md`, `history.csv`, `summary.png` | Generated from the sessions; rebuild with `report.py`. | yes |
| `reference/baseline_arrays.npz` (+ `.json`) | The correctness oracle and where it came from. | yes |
| `results/<machine>/variations/<utc>.json`, `VARIATIONS.md` | Runs on other setups (`tests/variations.py`) and the table of the latest one. | yes |
| `reference/variation-<case>.npz` (+ `.json`) | The original code's results on each other setup, and the one run that made them. | yes |
| `RUNS.md` | The log of every run, including those made by hand before the harness existed. | yes |
| `../benchmarks/runs/` | The report files of those hand-made runs. | yes |
| `.cache/` | Snapshots, per-sample logs, temporary files. | no |

## Other setups

The jury scores "a slight variation of the supplied example", and everything above runs the supplied example only: `benchmark.py` checks its input file's fingerprint and must not be edited. `tests/variations.py` covers the gap. It is a separate script, not part of the harness, because it has to import the simulator and time its phases itself; it does both exactly as `benchmark.py` does, with the same phase names.

```bash
python tests/variations.py --list           # the cases and what each one changes
python tests/variations.py                  # run them all
python tests/variations.py uifo-20-5000     # run some
```

Each case is a file in `tests/variations/`: a complete setup and its frequencies. For every case the script

1. runs the **original code once** and stores its readout signal, noise, sensitivity, powers and gradient in `reference/variation-<case>.npz`. This happens only when that file is missing (or with `--make-reference <case>`), because the original is slow and needs about 14 GB;
2. runs the **working tree** in a fresh process and compares it with that reference, using `compare_arrays` and the tolerances of `validate.py`;
3. checks the working tree's gradient against **finite differences** in four random directions;
4. writes a session file to `results/<machine>/variations/` and regenerates `results/<machine>/VARIATIONS.md`.

Verdicts: `PASS`; `DEVIATION` (inside the judges' limit of 1e-5 in log sensitivity, but some value differs from the original by more than the `validate.py` tolerance; the line names it); `FAIL`; `NO_REFERENCE` (the original could not produce a result). The case `supplied` is the script's check of itself: it must reproduce the public reference sensitivities and the harness's own baseline arrays.

A `DEVIATION` does not say which code is wrong. `python tests/refined_solves.py <case>` does: it repeats our calculation with every sparse solve refined to a rounding-free result and prints how far ours and the original are from it. The settings of every case (layout, frequencies, values, seeds, limits) and how to add one are in `tests/variations/README.md`.

Things to know when reading the table:

- The time of "ours" is one sample per case. For a clean number, run the script a second time once the references exist: in the session that creates them, our run starts right after the original has filled the memory.
- The original's time is the single run that made the reference, on whatever machine that was.
- There is no public reference for these setups, so the loss is `mean(log10(sensitivity))`. Its gradient is the same as the benchmark's, whose loss differs by a constant.
- A reflectivity within 1e-6 of 0 or 1 is left out of the finite-difference directions, because the square roots in the mirror formulas have no finite slope there. Those gradient entries are still compared with the original's.
- The cases are our guess at "a slight variation": other values, other frequencies, another layout, another readout, another grid size, and one mode in which our code falls back to its JAX path. Add a case by adding it to `generate()` in the script and running `python tests/variations.py --generate`.

## Runs made by hand

`RUNS.md` lists every run made so far, including the ones from before the harness was used, with the result of each and where its raw data is. Add a row there for any run made outside the harness (another machine, a cloud instance), and put its report in `../benchmarks/runs/`.

## Versions, baseline and anchor

`versions.json` lists the versions in order. Each has a `name`, a `ref` (git
tag or commit), a `parent`, a `summary`, and `enabled` (set it to `false` to
leave a dead end out of `full` runs).

- **`baseline`** is the unmodified starting point. It ran once; its stored total is the reference for speed-ups in every session, and it is left out of `quick`, `full` and `aa`.
- **`anchor`** is a version that is rerun in every `quick` session, so the candidate has a reference measured minutes apart. Set it to the first version whose samples no longer carry the `memory` flag (`add-version --anchor`, or edit `versions.json`). Until then `quick` times the latest version and `WORKTREE` only.

## Known limits

- **The baseline swaps on the 16 GB Mac.** Its seconds are inflated, so "x faster than baseline" measured here overstates what the judges will see on a 32 GB machine. Those factors are marked `*`. Comparisons against the anchor are clean. Cutting peak memory is therefore a sensible first optimization.
- **The baseline factor rests on one sample.** The baseline ran once, at a different time than the versions compared with it, so machine drift between that day and a session does not cancel out. This is fine while the factors are large. To decide between two close versions, use `vs anchor` and `vs parent`, which are measured within a round.
- **No CPU pinning on macOS.** The thread limits emulate one CPU; they cannot choose which core. The canary catches the resulting drift.
- **ARM and Apple's math library are not the judges' x86 Linux.** Relative gains may not transfer one-to-one. Before submitting, run `python bench/run.py official` once on a Linux x86 machine if one is available.
- **Bytecode and file cache are warm** for every sample (see the discarded import above). A first-ever run on a freshly cloned repository is slightly slower.

## Rules that keep the numbers honest

- No state may survive between runs: no persistent JAX compilation cache, no precomputed results on disk. `benchmark.py` disables the cache; the harness clears `JAX_COMPILATION_CACHE_DIR`, `XLA_FLAGS`, `NPROC` and `PYTHONPATH` before each run.
- A version that needs extra environment variables declares them in its `versions.json` entry (`"env": {...}`), where they are visible and recorded.
- The harness records wall-clock time from process start next to the script's own total (`untimed_overhead` in the session file), so work cannot quietly move outside the timers.

## Troubleshooting

- **`selftest` fails:** read the assertion message; it names the property that broke. Nothing else is trustworthy until it passes.
- **A sample shows `FAILED`:** open the log named in the session file under `.cache/logs/<session>/`.
- **`multi-core` flags:** run `calibrate` again and check that nothing in the version sets its own thread counts.
- **Frequent "waited ... for a calm machine":** something else is using the CPU, or the laptop is hot. Close apps, wait, or raise `--cooldown`.
- **The baseline run takes very long:** it swaps on 16 GB and only has to succeed once. Nothing reruns it afterwards.
- **A correct version gets `FAIL` with a deviation:** see the last paragraph of [Correctness](#correctness).
- **Speed-ups are shown against another version instead of the baseline:** `results/<machine>/baseline.json` is missing on this machine. Run `python bench/validate.py v00-baseline --make-reference`.
- **Interrupted session:** the partial file stays, marked `"complete": false`, and reports ignore it. Delete it or keep it; then start a new session.
