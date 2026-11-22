# Benchmark summary: amd-epyc-7763-64-core-processor-16g-linux

- Machine: AMD EPYC 7763 64-Core Processor, 16 GiB RAM, Linux-6.8.0-1070-azure-x86_64-with-glibc2.39
- CPU limit: pinned to logical CPU 3 with the official `--cpus 1`
- Noise floor: not measured yet. Run `python bench/run.py aa`.
- Sessions on record: 3

## Latest session: `2026-11-22T09-52-29Z_run`

8 rounds, shuffled order per round, one fresh process per sample.

| Version                  | n   | median s | min s | spread | vs v11-scipy-stacked-solver | vs parent | Setup | Compile | Forward | Sens. | Grad. | peak mem | cpu/wall | valid | flags                                  |
| ------------------------ | --- | -------- | ----- | ------ | --------------------------- | --------- | ----- | ------- | ------- | ----- | ----- | -------- | -------- | ----- | -------------------------------------- |
| v11-scipy-stacked-solver | 8/8 | 0.79     | 0.78  | 3.9%   | -                           | -         | 0.48  | 0.14    | 0.11    | 0.00  | 0.06  | 0.31 GiB | 1.02     | -     | unofficial-benchmark                   |
| cand2                    | 8/8 | 0.76     | 0.74  | 6.4%   | 1.05x (1.01-1.06)           | -         | 0.47  | 0.14    | 0.10    | 0.00  | 0.04  | 0.30 GiB | 1.02     | -     | unofficial-benchmark                   |
| cand4                    | 8/8 | 0.75     | 0.74  | 7.2%   | 1.05x (0.99-1.08)~          | -         | 0.48  | 0.14    | 0.10    | 0.00  | 0.03  | 0.30 GiB | 1.02     | PASS  | slow-machine 1/8, unofficial-benchmark |

## Speed-up over the session's reference, across sessions

One column per session. A version whose factor stays put from column to column is a trustworthy result.
Against the stored baseline every session divides into the same number, so those factors are directly comparable.

| Version                  | 11-22T00-14 run | 11-22T09-42 run | 11-22T09-52 run |
| ------------------------ | --------------- | --------------- | --------------- |
| v07-numpy-host           | ref             |                 |                 |
| v09-active-derivatives   | 1.02x           |                 |                 |
| v10-host-swept-values    | 1.04x           |                 |                 |
| v11-scipy-stacked-solver |                 | ref             | ref             |
| cand2                    |                 | 1.07x           | 1.05x           |
| tmp-cand3                |                 | 1.07x           |                 |
| cand4                    |                 |                 | 1.05x           |

![First-run total per version, session 2026-11-22T09-52-29Z_run](summary.png)

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
