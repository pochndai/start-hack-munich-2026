# Experiment: the simulator without klujax (branch `no-klujax-scipy`)

This note records an experiment that lives on the branch `no-klujax-scipy` and is not merged into `main`.
The readable summary with charts is [no-klujax-scipy-results.pdf](no-klujax-scipy-results.pdf); this file is the record behind it.

Every result below was measured on the Nebius VM with the judges' CPU: Intel Xeon Platinum 8468, pinned to logical CPU 8 with the official `--cpus 1`, inside a 32 GiB memory cap with swap off, on 22 November 2026.
Raw data: `bench/results/intel-xeon-platinum-8468-197g-linux/`.
Numbers marked "development laptop" are from short runs on Windows with a newer JAX; they guided decisions and count for nothing else.

## What the branch does

The host path of change 12 solved the signal systems with klujax, four times per frequency, each time factorising the matrix again. The branch changes that as follows.

- **SciPy's sparse LU (SuperLU) instead of klujax.** SciPy is already in the organizers' `requirements.txt`. klujax is still imported by the JAX fallback (`simulate_jax`), which setups with linking functions use.
- **One matrix for all frequencies.** The signal matrices are stacked into one block-diagonal matrix and factorised in a single call (`_Factors` in `differometor/host.py`). The column order is found once per sparsity pattern, and SuperLU's supernode settings are set to one.
- **One factorisation per evaluation instead of four.** The gradient needs none of its own:
  - the adjoint solve of the noise is a real factor, known only in the reverse pass, times a vector known in the forward pass, so the solve is done in the forward pass (`_forward`);
  - the adjoint of the signal is a combination of the noise weights whenever the signal is read out at the noise detector. `_signal_adjoint` checks this at run time and factorises again otherwise.
- **One step of iterative refinement after every solve** (`REFINEMENT_STEPS`).
- **Forward and gradient are one host callback each.** Nothing of the solver is in the JAX program.
- **The dense signal and noise matrices stay out of the callbacks.** The sparse layouts carry the entries the host path needs (`base`). The arrays kept for the reverse pass shrink from 26.7 MB to 5.2 MB.
- **Cheaper imports of the package:** SuperLU's compiled module is loaded from its file without importing `scipy.sparse.linalg` (`_load_superlu`), the jaxlib version is read from jaxlib, and plotly is loaded when `visualize_setup` is first asked for.

The branch is merged with `main` at `v10`, so it contains changes 14 and 15 in `main`'s version.

## Result

Complete first run, the scored number.

| Setup | How it was timed | Original | `v07` | `v10` (latest `main`) | Branch |
| --- | --- | ---: | ---: | ---: | ---: |
| Supplied grid, 101 frequencies, 193 design variables | Harness session, 5 shuffled rounds, medians (`sessions/2026-11-22T00-14-25Z_run.json`) | 43.14 s (one stored run) | 0.88 s | 0.84 s | **0.71 s** |
| aLIGO, the organizers' second benchmark, 29 design variables | `benchmark_aligo.py`, 4 alternating runs each, medians (`aligo/`) | 26.25 s (one run) | not timed | 0.62 s | **0.50 s** |
| `random-layout-1`: another 3×3 arrangement, balanced homodyne readout | `tests/variations.py`, 4 and 6 runs, medians (`variations/`, `variations-v10/`) | 41.55 s (one run) | not timed | 0.83 s | **0.69 s** |

In the harness session the five runs ranged from 0.85 to 0.91 s (`v07`), 0.82 to 0.87 s (`v10`) and 0.68 to 0.72 s (branch). The machine was not perfectly quiet: the harness flagged a slower control workload in 2 to 3 of 5 rounds for every version. Versions are therefore compared within one session only. An earlier session the same night, before the merge with `main`, gave `v07` 0.85 s and the branch 0.67 s (`sessions/2026-11-22T00-08-43Z_run.json`).

Phases of the supplied setup in the same session, medians in milliseconds:

| Phase | `v07` | `v10` | Branch | Branch minus `v10` |
| --- | ---: | ---: | ---: | ---: |
| Imports | 329 | 335 | 313 | −22 |
| Build the setup | 108 | 87 | 92 | +5 |
| Forward: trace and compile | 61 | 61 | 19 | −42 |
| Forward: run | 88 | 87 | 112 | +25 |
| Readout: trace and compile (the benchmark's own function) | 59 | 60 | 63 | +3 |
| Gradient: trace and compile | 99 | 97 | 47 | −50 |
| Gradient: run | 129 | 109 | 58 | −51 |

Compiling forward and gradient takes a third of the time because the solver is no longer in the JAX program. The gradient run is twice as fast because it no longer factorises. The forward run is slower because it now does the one factorisation, three solves and their refinement.

## Correctness

All on the VM, for commit `f3f1454`. Logs: `no-klujax-scipy-logs/`.

- **Official benchmarks:** `benchmark.py` and `benchmark_aligo.py` print `PASS` in every run. Distance from the public reference on the supplied setup: 5.374e-06 in log sensitivity (limit 1e-5), the same as the original.
- **`bench/validate.py no-klujax-scipy`:** `PASS`. Against the original's arrays from this VM: signal 1.3e-07, noise 4.0e-08, sensitivity 6.8e-08 in log, gradient 9.1e-08 of its largest entry, 0 of 193 gradient entries outside the tolerance (`validation/no-klujax-scipy@f3f14542ea40.json`).
- **`tests/test_host.py`:** `ALL CHECKS PASSED` (host path against JAX path on 13 cases, `jit`, `vmap`, finite differences).
- **`tests/handwritten_jacobians.py`:** all 19 rules agree with JAX.
- **`tests/variations.py`, 19 scenarios against the original:** 18 `PASS`, 1 `DEVIATION`, 0 `FAIL` (`variations/2026-11-22T00-23-50Z.json`). The deviation is `band-20-5000`, the one already recorded for `main`. `main` at `v10` in the same hour: 17 `PASS`, 2 `DEVIATION`; its second deviation, a signal difference of 1.6e-06 on `uifo-20-5000`, does not occur on the branch, where that scenario agrees with the original to 5e-11 in log sensitivity.

Per scenario the branch needs 593 to 880 ms where `v10` needs 675 to 1045 ms, on the 18 scenarios that take the NumPy path. `amplitude-modulation` takes the JAX fallback on both; the branch does not make it faster (2565 ms against 2472 ms, one run each).

## How the result was reached

Quick comparisons while developing: the official benchmark, three or four alternating runs per version, medians. They are not harness sessions, and the machine's speed drifted during the night, so only the two numbers of one row are comparable.

| State of the branch | `v07` in the same minutes | Branch |
| --- | ---: | ---: |
| First version: SciPy solver, one factorisation, callbacks only, SuperLU loaded directly (`b0b902a`) | 835 ms | 725 ms |
| Dense matrices kept out of the callbacks, imports trimmed | 812 ms | 658 ms |
| Build step fixed, matrix product rewritten (`05da4b9`) | 834 ms | 667 ms |
| Merged with `main` at `v10` (`f3f1454`); `v10` itself: 879 ms | 913 ms | 720 ms |

## Experiments and decisions

| Tried | Measured | Decision |
| --- | --- | --- |
| Switching the refinement step off | Forward run 109 ms to 84 ms, complete first run 658 ms to 645 ms (VM, four alternating runs). Agreement with the original on `uifo-20-5000` falls from 5e-11 to 2e-08 in log sensitivity | Kept on |
| Tuning SuperLU: direct call, fixed column order, supernode settings of one | Factorising 101 matrices: 86.7 ms to 38.5 ms (development laptop) | Adopted |
| Stacking all frequencies into one matrix | Solve time halved, factorisation unchanged (development laptop) | Adopted |
| Plain SuperLU without refinement | Up to 5e-04 off in log sensitivity on Voyager with the default column order (development laptop, against a rounding-free solution) | Not adopted |
| Scaling rows and columns before factorising | No help without refinement, sometimes worse (development laptop) | Not adopted |
| Krylov methods (GMRES, BiCGSTAB), also preconditioned with the LU of the neighbouring frequency | No convergence in 2,000 iterations (development laptop) | Not adopted |
| Banded LAPACK solver after reverse Cuthill-McKee reordering | Three times slower than SuperLU, 9e-06 away from the klujax result (development laptop) | Not adopted |
| Loading SciPy's compiled sparse-product module directly, like SuperLU | The module took 49 ms to load on the laptop, more than the products it would speed up | Not adopted |
| Importing jax before `benchmark.py` starts its timer (a start-up hook) | Not tried | Rejected: it moves work out of the timed region instead of removing it |

## Limits and open points

- **A private SciPy entry point.** The factorisation calls SciPy's internal `gstrf` and loads its module from a file. Both exist in the pinned scipy 1.18.1. If either fails, the code falls back to the normal import and then to `splu`, which is slower.
- **klujax is still a requirement** because of the JAX fallback.
- **A stall that `main` has too.** On one pinned CPU, starting the gradient before the previous JAX call has finished hangs inside the host callback. `benchmark.py` waits after every phase and is not affected. `v07` hangs in the same test.
- **The harness flags `v10` and the branch as `unofficial-benchmark`** because their `benchmark.py` is the organizers' newer file (with aLIGO), which differs from the one at the baseline commit. Nobody on the team edited it.
- **Not done:** a noise-floor session on the VM in its present state (one was started and stopped: `sessions/2026-11-22T00-10-37Z_aa.json` is incomplete), and timings of `v07` on the two extra setups.
- **Not registered as a version.** If the branch is merged, it needs `add-version`, a numbered section in `CHANGES.md` and a harness session of its own.

## Reproduce

```bash
git checkout no-klujax-scipy
python benchmark.py --cpus 1 --output benchmark-candidate.json
python benchmark_aligo.py --cpus 1 --output benchmark-aligo.json
python bench/run.py run --versions v07-numpy-host,v10-host-swept-values,no-klujax-scipy --rounds 5
python bench/validate.py no-klujax-scipy --force
python tests/test_host.py
python tests/variations.py
python docs/make_no_klujax_report.py      # rebuilds the PDF; needs matplotlib and reportlab
```
