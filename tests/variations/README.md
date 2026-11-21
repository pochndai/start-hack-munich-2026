# Test scenarios: setups other than the supplied one

Written by `python tests/variations.py --generate` from the case files in this folder. Do not edit by hand: change `generate()` in `tests/variations.py` and run that command.

## Why these exist

The jury scores "a slight variation of the supplied example". `benchmark.py` can only run the supplied example, so each file here is one variation we thought the jury might use: a complete optical setup and the frequencies to simulate. `tests/variations.py` runs every one on the original code and on ours, compares the results and gradients, and times our code.

## Run them

```bash
python tests/variations.py --list                  # the scenarios
python tests/variations.py                         # run all, compare with the stored results of the original
python tests/variations.py size-4 uifo-20-5000     # run some
python tests/variations.py --make-reference size-4 # run the original again for one scenario (slow, about 14 GB)
python tests/refined_solves.py band-20-5000        # when ours and the original disagree: which one is right
```

On a new machine, run `python bench/run.py calibrate` first. On Linux the runs are then pinned to one core (`--cpus 1`), as in the official benchmark; on a Mac every numerical library is limited to one thread. The stored results of the original (`bench/reference/variation-<scenario>.npz`) are in the repository, so the original only has to run again for a new or changed scenario.

Results: one session file per run in `bench/results/<machine>/variations/`, and the table of the latest session in `bench/results/<machine>/VARIATIONS.md`.

## The scenarios

| Scenario | What it is |
| --- | --- |
| `amplitude-modulation` | Supplied layout and design values in amplitude-modulation mode, which has linking functions, so our code takes its JAX fallback. |
| `band-20-5000` | Supplied setup at 101 frequencies from 20 to 5000 Hz. |
| `boundary-swap` | Supplied layout with the laser at boundary 10 and the squeezer at boundary 20 exchanged. |
| `center-swap` | Supplied layout with center 22 made a directional beamsplitter and center 11 a plain one. |
| `detector-moved` | Supplied layout with the detector moved from boundary 03 to boundary 30. |
| `frequencies-51` | Supplied setup at 51 frequencies from 800 to 3000 Hz. |
| `homodyne-readout` | Supplied layout with the detector replaced by a balanced homodyne readout. |
| `random-layout-1` | A random 3x3 layout (uifo with random_seed=1) with random design values. |
| `random-layout-2` | A random 3x3 layout (uifo with random_seed=2) with random design values. |
| `random-layout-3` | A random 3x3 layout (uifo with random_seed=3) with random design values. |
| `size-2` | A random 2x2 layout with random design values. |
| `size-4` | A random 4x4 layout with random design values, at 21 frequencies (the original would not fit in 16 GB with 101). |
| `supplied` | The supplied example, unchanged (control: must reproduce benchmark.py). |
| `uifo-20-5000` | The second public setup (other layout, balanced homodyne readout) at 101 frequencies from 20 to 5000 Hz. |
| `values-random` | Supplied layout; every design variable drawn at random from the optimizer's ranges. |
| `values-shifted` | Supplied layout; every design variable moved by about 1%. |

## Settings of each scenario

"Design variables" are the reflectivities, tunings, squeezing levels and angles, laser powers, masses and tied lengths that `benchmark.py` differentiates. "Beamsplitters" counts plain / directional ones at the grid centers. "Nodes / edges" is the number of components and connections.

| Scenario | Grid | Frequencies | Band | Design variables | Readout | Lasers | Squeezers | Beamsplitters | Signal mode | Nodes / edges |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `amplitude-modulation` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | amplitude modulation | 132 / 71 |
| `band-20-5000` | 3x3 | 101 | 20 to 5000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `boundary-swap` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `center-swap` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `detector-moved` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 30 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `frequencies-51` | 3x3 | 51 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `homodyne-readout` | 3x3 | 101 | 800 to 3000 Hz | 195 | balanced homodyne at 03 | 7 | 4 | 6 / 3 | space modulation | 203 / 73 |
| `random-layout-1` | 3x3 | 101 | 800 to 3000 Hz | 193 | balanced homodyne at 20 | 6 | 5 | 5 / 4 | space modulation | 202 / 73 |
| `random-layout-2` | 3x3 | 101 | 800 to 3000 Hz | 192 | detector at 41 | 5 | 6 | 5 / 4 | space modulation | 195 / 71 |
| `random-layout-3` | 3x3 | 101 | 800 to 3000 Hz | 197 | detector at 41 | 3 | 8 | 6 / 3 | space modulation | 196 / 71 |
| `size-2` | 2x2 | 101 | 800 to 3000 Hz | 102 | balanced homodyne at 23 | 2 | 5 | 2 / 2 | space modulation | 106 / 37 |
| `size-4` | 4x4 | 21 | 800 to 3000 Hz | 315 | detector at 45 | 7 | 8 | 10 / 6 | space modulation | 323 / 119 |
| `supplied` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `uifo-20-5000` | 3x3 | 101 | 20 to 5000 Hz | 192 | balanced homodyne at 42 | 4 | 7 | 4 / 5 | space modulation | 201 / 73 |
| `values-random` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |
| `values-shifted` | 3x3 | 101 | 800 to 3000 Hz | 193 | detector at 03 | 7 | 4 | 6 / 3 | space modulation | 196 / 71 |

All frequency lists are spaced evenly on a logarithmic scale, except the band of the supplied example, which is taken from `examples/data/uifo_800_3000_test_data.json`.

## Layout of each scenario

Centers are listed row by row (B = beamsplitter, D = directional beamsplitter). Boundaries are named by their grid position as in `differometor.setups.uifo`: `0y` and `(size+1)y` are the two sides in one direction, `x0` and `x(size+1)` the two in the other. The orientation of each center is in the case file.

| Scenario | Centers | Boundaries |
| --- | --- | --- |
| `amplitude-modulation` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `band-20-5000` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `boundary-swap` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 squeezer, 14 laser, 20 laser, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `center-swap` | B B D / B D B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `detector-moved` | D B D / B B B / D B B | 01 laser, 02 laser, 03 laser, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 detector, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `frequencies-51` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `homodyne-readout` | D B D / B B B / D B B | 01 laser, 02 laser, 03 balanced homodyne, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `random-layout-1` | B D B / D B B / D B D | 01 squeezer, 02 laser, 03 laser, 10 squeezer, 14 squeezer, 20 balanced homodyne, 24 squeezer, 30 laser, 34 laser, 41 laser, 42 laser, 43 squeezer |
| `random-layout-2` | B B B / D D D / D B B | 01 squeezer, 02 laser, 03 squeezer, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 squeezer, 34 squeezer, 41 detector, 42 laser, 43 squeezer |
| `random-layout-3` | B D B / B D B / D B B | 01 laser, 02 squeezer, 03 squeezer, 10 squeezer, 14 squeezer, 20 laser, 24 laser, 30 squeezer, 34 squeezer, 41 detector, 42 squeezer, 43 squeezer |
| `size-2` | D D / B B | 01 laser, 02 squeezer, 10 squeezer, 13 squeezer, 20 squeezer, 23 balanced homodyne, 31 squeezer, 32 laser |
| `size-4` | B D D B / D B B B / B D B B / B D B D | 01 laser, 02 squeezer, 03 laser, 04 squeezer, 10 squeezer, 15 squeezer, 20 squeezer, 25 laser, 30 squeezer, 35 squeezer, 40 squeezer, 45 detector, 51 laser, 52 laser, 53 laser, 54 laser |
| `supplied` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `uifo-20-5000` | B D B / D D D / B B D | 01 squeezer, 02 squeezer, 03 laser, 10 laser, 14 squeezer, 20 squeezer, 24 squeezer, 30 laser, 34 squeezer, 41 squeezer, 42 balanced homodyne, 43 laser |
| `values-random` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |
| `values-shifted` | D B D / B B B / D B B | 01 laser, 02 laser, 03 detector, 10 laser, 14 laser, 20 squeezer, 24 laser, 30 laser, 34 squeezer, 41 squeezer, 42 squeezer, 43 laser |

## Where the numbers in a scenario come from

- **Supplied values:** the optimized values stored in `examples/data/uifo_800_3000.json` (or `uifo_20_5000.json` for `uifo-20-5000`).
- **Shifted values** (`values-shifted`): every design variable is multiplied by `1 + 0.01 * z`, with `z` from a standard normal distribution, seed 1. Tied lengths share one factor.
- **Random values** (`values-random`, `random-layout-*`, `size-*`): every design variable is drawn evenly from the range the optimizer in `examples/uifo_optimization.py` works in (table below). Seeds: `values-random` 2, `random-layout-N` 10 + N, `size-2` 14, `size-4` 15.
- **Random layouts:** `uifo(size, random=True, random_seed=N)` with N = 1, 2, 3 for `random-layout-N`, 4 for `size-2` and 5 for `size-4`.
- **Changed layouts** (`boundary-swap`, `center-swap`, `detector-moved`, `homodyne-readout`): the supplied layout with the named parts exchanged. Parts that keep their name and kind keep the supplied values; new parts get the defaults of `differometor.components`.
- Everything that is not a design variable (losses, refractive indices, beamsplitter angles, lengths next to a center or a boundary) stays as supplied or at its default.

| Design variable | Lowest | Highest |
| --- | ---: | ---: |
| db | 0 | 10 |
| angle | -360 | 360 |
| power | 0 | 200 |
| tuning | -360 | 360 |
| mass | 0.01 | 200 |
| length | 0.1 | 4000 |
| reflectivity | 0 | 1 |

## What is checked, and the limits

| Check | Limit | Verdict if outside |
| --- | --- | --- |
| Our run finishes; all results finite; sensitivities positive; gradient not all zero | | `FAIL` |
| ln(sensitivity), ours against the original | 1e-5 (the judges' limit) | `FAIL` |
| Our gradient against finite differences in four random directions | 2% | `FAIL` |
| Carrier, readout signal, noise, sensitivity, powers against the original, relative to the largest value | 1e-6 | `DEVIATION` |
| ln(sensitivity) against the original | 1e-6 | `DEVIATION` |
| Each gradient entry against the original | 0.1% of the entry, plus 1e-12 of the largest entry | `DEVIATION` |

The `DEVIATION` limits are those of `bench/validate.py`. The loss is `mean(log10(sensitivity))`; there is no public reference for these setups. The scenario `supplied` also has to reproduce the public reference sensitivities and the arrays that `benchmark.py` itself produced with the original code.

A `DEVIATION` does not say which code is wrong. `python tests/refined_solves.py <scenario>` answers that: it repeats our calculation with every sparse solve refined to a near-exact result and shows how far ours and the original are from it.

## Add a scenario

1. In `generate()` in `tests/variations.py`, add one entry to `cases`: a name, one sentence, the setup and the frequencies. Use `changed(...)` for a different layout, `revalued(...)` for different values, `layout(...)` for a new grid.
2. `python tests/variations.py --generate` writes the case file and this sheet. The existing case files come out byte for byte the same.
3. `python tests/variations.py <name>` runs the original once, stores its results, and compares.
4. Commit the case file, this sheet, `bench/reference/variation-<name>.*` and the new session under `bench/results/`.

## Not covered yet

- More than 101 frequencies, and a 4x4 grid at 101 frequencies: the original needs about 140 MB per frequency at 3x3 and more at 4x4, which does not fit in 16 GB. They need a machine with 32 GB or more.
- One-core Linux times for these scenarios (the stored times are from the Mac).
- Frequency-modulation mode, grids larger than 4x4, and a different set of design variables (for example refractive indices or beamsplitter angles).
- Setups that are not grids (Voyager, aLIGO, the cavity sweep): `tests/test_host.py` compares our two code paths on them, but they are not timed here.
