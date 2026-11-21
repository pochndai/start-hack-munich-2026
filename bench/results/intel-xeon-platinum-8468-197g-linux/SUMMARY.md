# Benchmark summary: intel-xeon-platinum-8468-197g-linux

- Machine: Intel(R) Xeon(R) Platinum 8468, 197 GiB RAM, Linux-6.11.0-1016-nvidia-x86_64-with-glibc2.39
- CPU limit: pinned to logical CPU 8 with the official `--cpus 1`
- Noise floor: not measured yet. Run `python bench/run.py aa`.
- Sessions on record: 1

## Latest session: `2026-11-21T22-53-03Z_run`

5 rounds, shuffled order per round, one fresh process per sample.

| Version                     | n   | median s | min s | spread | vs v00-baseline (stored) | vs parent         | Setup | Compile | Forward | Sens. | Grad. | peak mem | cpu/wall | valid | flags            |
| --------------------------- | --- | -------- | ----- | ------ | ------------------------ | ----------------- | ----- | ------- | ------- | ----- | ----- | -------- | -------- | ----- | ---------------- |
| v01-klujax-solver           | 5/5 | 33.39    | 33.25 | 1.4%   | 1.29x (1.28-1.30)        | -                 | 1.33  | 23.49   | 4.51    | 0.00  | 4.08  | 9.30 GiB | 1.01     | PASS  | -                |
| v02-group-by-function       | 5/5 | 23.66    | 23.62 | 1.5%   | 1.82x (1.80-1.83)        | 1.41x (1.39-1.43) | 1.34  | 14.15   | 4.29    | 0.00  | 3.90  | 8.67 GiB | 1.01     | PASS  | slow-machine 1/5 |
| v03-sparse-assembly         | 5/5 | 15.66    | 15.59 | 1.2%   | 2.75x (2.73-2.77)        | 1.51x (1.50-1.54) | 1.37  | 13.98   | 0.16    | 0.00  | 0.15  | 1.00 GiB | 1.00     | PASS  | -                |
| v04-fast-compile-flags      | 5/5 | 5.17     | 5.15  | 0.5%   | 8.35x (8.33-8.37)        | 3.03x (3.02-3.05) | 1.10  | 3.68    | 0.20    | 0.00  | 0.18  | 0.56 GiB | 1.00     | PASS  | -                |
| v05-two-solves-whole-arrays | 5/5 | 4.61     | 4.60  | 1.7%   | 9.36x (9.23-9.38)        | 1.12x (1.10-1.13) | 0.75  | 3.55    | 0.17    | 0.00  | 0.14  | 0.51 GiB | 1.00     | PASS  | -                |
| v06-real-parameters         | 5/5 | 2.77     | 2.75  | 2.2%   | 15.55x (15.36-15.70)     | 1.67x (1.65-1.68) | 0.45  | 2.11    | 0.09    | 0.00  | 0.12  | 0.35 GiB | 1.00     | PASS  | -                |
| v07-numpy-host              | 5/5 | 0.90     | 0.89  | 1.7%   | 48.10x (47.78-48.63)     | 3.10x (3.05-3.14) | 0.45  | 0.22    | 0.09    | 0.00  | 0.13  | 0.37 GiB | 1.01     | PASS  | -                |
| v08-no-large-pages          | 5/5 | 0.91     | 0.89  | 2.4%   | 47.30x (47.21-48.36)     | -                 | 0.45  | 0.22    | 0.10    | 0.00  | 0.13  | 0.35 GiB | 1.01     | PASS  | -                |

Stored baseline: v00-baseline took 43.14 s in its single run on 2026-11-21. It is not rerun: each factor against it is this number divided by a new sample.

## Speed-up over the session's reference, across sessions

One column per session. A version whose factor stays put from column to column is a trustworthy result.
Against the stored baseline every session divides into the same number, so those factors are directly comparable.

| Version                     | 11-21T22-53 run |
| --------------------------- | --------------- |
| v01-klujax-solver           | 1.29x           |
| v02-group-by-function       | 1.82x           |
| v03-sparse-assembly         | 2.75x           |
| v04-fast-compile-flags      | 8.35x           |
| v05-two-solves-whole-arrays | 9.36x           |
| v06-real-parameters         | 15.55x          |
| v07-numpy-host              | 48.10x          |
| v08-no-large-pages          | 47.30x          |

![First-run total per version, session 2026-11-21T22-53-03Z_run](summary.png)

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
