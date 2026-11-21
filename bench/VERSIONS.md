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

TODO after its one run (`python bench/validate.py v00-baseline --make-reference`);
copy the total from `results/<machine>/baseline.json`. Expect memory pressure:
the benchmark machine has 16 GB, so the baseline swaps there and its seconds
are inflated.
