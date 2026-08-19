# Experiment scripts

The main user-facing entry point remains [`benchmark_runner.py`](../benchmark_runner.py).
This directory contains experiment-specific helpers that are useful for reproducing
the thesis runs but are not part of the core package.

## Layout

- `experiments/` contains standalone experiment runners.
- `hpc/` contains Imperial CX3 PBS jobs and the repository sync helper. These files
  include site-specific paths and must be adapted for other clusters.
- `verification/` contains offline checks for experiment instrumentation.

Run Python helpers from the repository root with `PYTHONPATH=.`. For example:

```bash
PYTHONPATH=. python3.9 scripts/verification/verify_ablation_counters.py
```
