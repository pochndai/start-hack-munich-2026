# Benchmark summary: apple-m5-16g-darwin

- Machine: Apple M5, 16 GiB RAM, macOS-26.6-arm64-arm-64bit
- CPU limit: thread limits `MKL_NUM_THREADS=1`, `NPROC=1`, `NUMEXPR_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`
- Noise floor: +/- 16.2 % (A/A session `2026-11-21T20-38-57Z_aa`). Smaller differences are not speed-ups.
- Sessions on record: 2

## Latest full session: `2026-11-21T20-21-55Z_full`

5 rounds, shuffled order per round, one fresh process per sample.

| Version                     | n   | median s | min s | spread | vs v00-baseline (stored) | vs parent           | Setup | Compile | Forward | Sens. | Grad. | peak mem | cpu/wall | valid | flags                        |
| --------------------------- | --- | -------- | ----- | ------ | ------------------------ | ------------------- | ----- | ------- | ------- | ----- | ----- | -------- | -------- | ----- | ---------------------------- |
| v01-klujax-solver           | 5/5 | 8.62     | 8.12  | 28.7%  | 1.90x (1.55-2.02)*       | -                   | 0.93  | 2.93    | 2.42    | 0.00  | 2.19  | 8.58 GiB | 1.76     | FAIL  | memory 5/5, multi-core 5/5   |
| v02-group-by-function       | 5/5 | 6.86     | 5.97  | 19.3%  | 2.39x (2.25-2.75)*       | 1.40x (1.16-1.54)*  | 0.87  | 1.64    | 2.19    | 0.00  | 2.09  | 9.24 GiB | 1.56     | FAIL  | memory 5/5, multi-core 5/5   |
| v03-sparse-assembly         | 5/5 | 2.68     | 2.64  | 3.0%   | 6.12x (6.02-6.21)*       | 2.56x (2.22-2.71)*  | 0.93  | 1.64    | 0.06    | 0.00  | 0.05  | 0.49 GiB | 2.75     | FAIL  | memory 5/5, multi-core 5/5   |
| v04-fast-compile-flags      | 5/5 | 1.69     | 1.65  | 19.8%  | 9.74x (8.28-9.96)*       | 1.58x (1.35-1.63)*  | 0.48  | 1.08    | 0.07    | 0.00  | 0.07  | 0.46 GiB | 1.14     | FAIL  | memory 3/5                   |
| v05-two-solves-whole-arrays | 5/5 | 1.79     | 1.49  | 19.3%  | 9.15x (8.94-11.03)*      | 0.94x (0.90-1.33)~* | 0.60  | 1.06    | 0.06    | 0.00  | 0.05  | 0.38 GiB | 1.01     | FAIL  | memory 4/5, slow-machine 1/5 |
| v06-real-parameters         | 5/5 | 1.27     | 0.92  | 35.2%  | 12.93x (12.03-17.89)*    | 1.34x (1.17-1.94)*  | 0.50  | 0.71    | 0.04    | 0.00  | 0.04  | 0.25 GiB | 0.97     | FAIL  | memory 2/5, multi-core 1/5   |
| v07-numpy-host              | 5/5 | 0.50     | 0.35  | 67.4%  | 33.11x (23.97-46.83)*    | 2.03x (1.91-3.81)   | 0.35  | 0.07    | 0.03    | 0.00  | 0.04  | 0.22 GiB | 0.79     | FAIL  | slow-machine 1/5             |

Stored baseline: v00-baseline took 16.41 s in its single run on 2026-11-21, under memory pressure. It is not rerun: each factor against it is this number divided by a new sample.

## Latest session: `2026-11-21T20-38-57Z_aa`

5 rounds, shuffled order per round, one fresh process per sample.

| Version          | n   | median s | min s | spread | vs v07-numpy-host#A | vs parent | Setup | Compile | Forward | Sens. | Grad. | peak mem | cpu/wall | valid | flags      |
| ---------------- | --- | -------- | ----- | ------ | ------------------- | --------- | ----- | ------- | ------- | ----- | ----- | -------- | -------- | ----- | ---------- |
| v07-numpy-host#A | 5/5 | 0.66     | 0.34  | 56.3%  | -                   | -         | 0.48  | 0.10    | 0.03    | 0.00  | 0.04  | 0.22 GiB | 0.70     | FAIL  | memory 1/5 |
| v07-numpy-host#B | 5/5 | 0.69     | 0.34  | 64.2%  | 0.97x (0.84-1.02)~  | -         | 0.51  | 0.11    | 0.03    | 0.00  | 0.04  | 0.22 GiB | 0.67     | FAIL  | memory 3/5 |

## Speed-up over the session's reference, across sessions

One column per session. A version whose factor stays put from column to column is a trustworthy result.
Against the stored baseline every session divides into the same number, so those factors are directly comparable.

| Version                     | 11-21T20-21 full |
| --------------------------- | ---------------- |
| v01-klujax-solver           | 1.90x*           |
| v02-group-by-function       | 2.39x*           |
| v03-sparse-assembly         | 6.12x*           |
| v04-fast-compile-flags      | 9.74x*           |
| v05-two-solves-whole-arrays | 9.15x*           |
| v06-real-parameters         | 12.93x*          |
| v07-numpy-host              | 33.11x*          |

![First-run total per version, session 2026-11-21T20-21-55Z_full](summary.png)

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
