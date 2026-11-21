# CLAUDE.md

Guidelines for working in this repository. The project is our entry for the
Feyer "Faster Physics" challenge at START Hack Munich 2026: make the
Differometor simulator faster while it keeps producing the same physical
results and gradients.

## What decides the result

- **The scored number** is the complete first-run total that `benchmark.py`
  prints: imports, setup, all JIT compilation, forward simulation,
  sensitivity, gradient and host transfer, on one logical CPU. Compilation is
  most of it. Warm runtime is reported but not scored.
- **The final evaluation uses a different optical setup**, a slight variation
  of the supplied example. A speed-up only counts if it carries over to other
  setups, and results and gradients must be correct there too.
- **Only correct submissions are ranked.** A faster version that changes the
  physics or the gradient is worth nothing.

## Rules that must never be broken

These come from the organizers' [README.md](README.md) and the judging criteria.

- Compute every frequency and every design variable, in double precision.
- No precomputed answers, no skipped calculations, and no state that survives
  between runs (no persistent JAX compilation cache, no results on disk).
- Nothing may depend on the supplied setup in particular: no hard-coded sizes
  (751, 193, 101), component names or sparsity patterns. Derive them from the
  setup the code is given.
- Do not edit `benchmark.py`, `benchmark_aligo.py`, `plot_timings.py`,
  `UPSTREAM.json` or the fixtures in `examples/data/`. `benchmark.py` checks the fixtures' hashes,
  and the harness flags any version whose `benchmark.py` differs.

## Repository map

| Path | What it is |
| --- | --- |
| `differometor/` | The simulator. All speed work happens here, mostly in `simulate.py`, `build.py` and `components.py`. |
| `benchmark.py` | The official benchmark and correctness checks. Read-only. |
| `CHANGES.md` | Every change we made, in order, with the reason, the measurement and the correctness evidence. This is also our submission note. |
| `bench/` | Our harness for comparing versions. `bench/README.md` is the manual; read it before timing or validating anything. |
| `bench/VERSIONS.md`, `bench/versions.json` | The registry of named versions the harness compares. |
| `benchmarks/` | Official reports and plots, named `<machine>-original` and `<machine>-candidate`. |
| `docs/` | Longer explanations, such as the derivation behind a change. |
| `examples/` | Public setups and scripts. Use them to check that a change generalizes. |

## Git

- **Push directly to `main`** unless the request explicitly says otherwise.
- **Create a new branch only after asking the user.** This covers every way
  of making one: `git branch`, `git switch -c`, `git checkout -b`, a new
  worktree on a new branch, or pushing to a new remote branch. If a branch
  seems useful (a risky experiment, a change that would break `main` for the
  others), say why and wait for a yes. The same holds for pull requests: do
  not open one on your own initiative.
- `origin` is the team repository (`pochndai/start-hack-munich-2026`).
  `upstream` is the organizers' repository; we cannot push there, and nothing
  of ours belongs there.
- **Pull as often as possible.** Several of us are coding on `main` in
  parallel, so a checkout goes stale within minutes. Run
  `git pull origin main` when starting a session, before starting each new
  task, before reading files you are about to change, before every commit and
  again right before pushing. A pull costs seconds; working on a stale
  checkout costs a merge conflict or a measurement of the wrong code.
- Push as soon as a change is committed and checked, so the others can pull
  it. Do not let finished commits sit locally.
- If a pull brings in changes to files you are working on, read what changed
  before continuing, and say so. Resolve conflicts by keeping both sides'
  intent; never discard a teammate's change to make a merge go through.
- Never force-push `main` and never rewrite pushed history.
- The organizers still change the rules. Run `git fetch upstream` when
  starting a session. If `upstream/main` has moved, read what changed, tell
  the user, and merge it into `main`.
- One commit per coherent change. The subject is a plain sentence that says
  what changed, for example "Drop the redundant noise solve". Do not commit
  `.venv/`, `bench/.cache/`, `*.npz` outside `bench/reference/`, or
  `benchmark*.json` scratch reports.

## Task: speeding up the simulator

1. Find where the time goes before changing anything: the phase lines
   `benchmark.py` prints, or a profile. State the expected gain first.
2. Make one change at a time, so each one can be measured and, if needed,
   dropped on its own.
3. Prefer changes whose output is bit-for-bit identical. If a change alters
   the numbers (a different solver, reordered arithmetic), say so and quantify
   the difference.
4. Check correctness, then measure, then record (the next three sections).
   A change is not done until all three are.
5. New dependencies are pinned in both `requirements.txt` and
   `pyproject.toml`.
6. Environment switches (such as XLA flags) are set inside the package or
   declared in the version's `env` entry in `bench/versions.json`, where they
   are visible and recorded. Never rely on a variable set by hand in a shell.

## Task: checking correctness

- `python bench/validate.py WORKTREE` is the check. It runs the official
  benchmark to `PASS` and then compares every output and each of the 193
  gradient entries with the arrays the original code produced.
- The official `PASS` alone is not enough: it compares one gradient entry out
  of 193 against a finite difference, at about 2 %.
- A deviation beyond tolerance is a `FAIL`. Do not reach for
  `--allow-deviations` or loosen a tolerance to get a pass. Explain the
  deviation first (which entry, how large, why), then let the user decide.
- `PASS_NO_REFERENCE` means nothing was compared, because
  `bench/reference/baseline_arrays.npz` does not exist yet. Do not report it
  as a pass.
- `validate.py` covers the supplied setup only. For a change that touches the
  physics formulas, the solver or the build step, also run
  `python tests/variations.py`. It compares the working tree with the original
  code on 16 other setups (other values, frequencies, layouts, readouts and
  grid sizes) and checks the gradient against finite differences; see "Other
  setups" in `bench/README.md`. Commit the session it writes under
  `bench/results/` and quote its table in `CHANGES.md`.
  The scenarios and their settings are listed in `tests/variations/README.md`.
  If ours and the original disagree, `python tests/refined_solves.py <case>`
  shows which one is right.
- The original code is commit `02ec741` (version `v00-baseline`).

## Task: measuring speed

- Use the harness: `python bench/run.py quick` while iterating,
  `python bench/run.py full` after registering a version. Each sample is a
  fresh process, because the scored number includes imports and compilation.
- **The original baseline is not rerun.** It ran once
  (`python bench/validate.py v00-baseline --make-reference`); sessions divide
  its stored total by each new sample. Rerun it only if the user asks.
- Compare two close versions with the within-round columns (`vs anchor`,
  `vs parent`), never with seconds from different sessions or machines.
- A difference smaller than the noise floor (`python bench/run.py aa`) is not
  a speed-up. Say so instead of reporting it as one.
- Numbers that get recorded come from the Mac through the harness, or from a
  Linux machine with the official `--cpus 1`. Run `python bench/run.py doctor`
  first on any machine that produces data. The Windows development laptop has
  no simulator environment (JAX is not installed there); use it for editing
  and `python bench/run.py selftest` only.
- Quote the one-CPU Linux comparison when one exists. A Mac cannot pin a CPU,
  and the original swaps on 16 GB there, which inflates its time.

## Task: recording a change

1. Commit the change.
2. `python bench/run.py add-version vNN-<slug> --summary "..."` tags the
   commit and registers it. Fill in the stub it appends to `bench/VERSIONS.md`.
3. Add a numbered section to `CHANGES.md` in the existing form: **What**,
   **Why**, **Effect** with measured numbers, and how it stays general. Update
   the result tables at the top if the headline number moved.
4. Commit the session files under `bench/results/` together with the text
   that quotes them.

Every number in `CHANGES.md` or `VERSIONS.md` names its machine, its CPU limit
and where the raw data is. Do not write a number that was not measured.

## Task: experiments that did not work

Record them under "Tried and not adopted" in `CHANGES.md` with the measurement
that ruled them out, so nobody repeats them. Remove the code; do not leave
dead switches behind.

## Task: changing the harness (`bench/`)

- Standard library only, apart from NumPy for array comparison and matplotlib
  for the chart. It must keep working without JAX installed.
- `python bench/run.py selftest` must end with `SELFTEST PASS` before any
  harness change is pushed. Add a selftest case for new behaviour.
- The harness never times anything itself and never drops a sample. Keep both
  properties.
- Update `bench/README.md` in the same commit as the behaviour it describes.

## Task: notes, documentation and the submission

- Write for someone who has not seen the code: a judge from the Feyer team.
  Plain sentences, the reason before the detail, terms defined on first use.
- State limits and missing measurements openly (see "Known limits" in
  `CHANGES.md`). An honest gap is better than an unqualified claim.
- The submission needs: the code, before and after JSON reports and plots
  (`python bench/run.py official --versions <final>`; the "before" comes from
  the single baseline run), reproduction commands, and the note on approach
  and correctness checks, which is `CHANGES.md`.

## Reporting back

- Report what was run and what it printed. If a check failed or was skipped,
  say that first.
- Separate what was measured from what is expected.
- If a step needs a machine you are not on (the Mac, a Linux box), do
  everything else and name the exact command that is still owed.
