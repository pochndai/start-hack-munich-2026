# Benchmark summary: intel-xeon-platinum-8468-197g-linux

- Machine: Intel(R) Xeon(R) Platinum 8468, 197 GiB RAM, Linux-6.11.0-1016-nvidia-x86_64-with-glibc2.39
- CPU limit: pinned to logical CPU 15 with the official `--cpus 1`
- Noise floor: +/- 2.4 % (A/A session `2026-11-22T09-54-18Z_aa`). Smaller differences are not speed-ups.
- Sessions on record: 5

## Latest session: `2026-11-22T09-54-18Z_aa`

5 rounds, shuffled order per round, one fresh process per sample.

| Version                    | n   | median s | min s | spread | vs v11-scipy-stacked-solver#A | vs parent | Setup | Compile | Forward | Sens. | Grad. | peak mem | cpu/wall | valid | flags                |
| -------------------------- | --- | -------- | ----- | ------ | ----------------------------- | --------- | ----- | ------- | ------- | ----- | ----- | -------- | -------- | ----- | -------------------- |
| v11-scipy-stacked-solver#A | 5/5 | 0.65     | 0.65  | 1.3%   | -                             | -         | 0.37  | 0.12    | 0.11    | 0.00  | 0.05  | 0.32 GiB | 1.01     | -     | unofficial-benchmark |
| v11-scipy-stacked-solver#B | 5/5 | 0.65     | 0.64  | 3.2%   | 1.01x (0.99-1.02)~            | -         | 0.37  | 0.12    | 0.10    | 0.00  | 0.05  | 0.32 GiB | 1.01     | -     | unofficial-benchmark |

## Speed-up over the session's reference, across sessions

One column per session. A version whose factor stays put from column to column is a trustworthy result.
Against the stored baseline every session divides into the same number, so those factors are directly comparable.

| Version                     | 11-21T22-53 run | 11-22T00-08 run | 11-22T00-14 run | 11-22T09-50 run |
| --------------------------- | --------------- | --------------- | --------------- | --------------- |
| v01-klujax-solver           | 1.29x           |                 |                 |                 |
| v02-group-by-function       | 1.82x           |                 |                 |                 |
| v03-sparse-assembly         | 2.75x           |                 |                 |                 |
| v04-fast-compile-flags      | 8.35x           |                 |                 |                 |
| v05-two-solves-whole-arrays | 9.36x           |                 |                 |                 |
| v06-real-parameters         | 15.55x          |                 |                 |                 |
| v07-numpy-host              | 48.10x          | 50.47x          | 49.21x          | 51.71x          |
| v08-no-large-pages          | 47.30x          |                 |                 |                 |
| no-klujax-scipy             |                 | 64.05x          | 61.17x          |                 |
| v10-host-swept-values       |                 |                 | 51.32x          | 53.13x          |
| v11-scipy-stacked-solver    |                 |                 |                 | 64.25x          |
| f1bc7a7                     |                 |                 |                 | 67.54x          |

![First-run total per version, session 2026-11-22T09-50-17Z_run](summary.png)

## Reading the tables

- `median s` / `min s`: complete first-run total printed by the version's own benchmark.py (the scored number).
- `vs X`: speed-up over X, median of the per-round ratios, with the lowest and highest round in brackets.
  `~` = not distinguishable from noise. `*` = X ran under memory pressure, so the factor is overstated.
  `(stored)` = X is the baseline, which ran once and is not rerun: every round is divided into that one
  stored total, so the bracket shows only this version's own scatter.
- `spread`: (max - min) / median of the totals. `cpu/wall`: CPU-seconds per wall-second, about 1.00 means one core.
- Flags (count / samples): `memory` = swapping or heavy compression during the run, `slow-machine` = the canary
  ran more than 5 % slower than the session's best, `multi-core` = more than one core was used,
  `failed` / `timeout` = no result, `unofficial-benchmark` = benchmark.py differs from the baseline's.
