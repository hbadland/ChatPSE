# Repeatability and component-ablation interpretation

## Repeatability

Five independent single-sample executions were analysed for each of four
validation cases. `VAL_01`, `VAL_06`, and `VAL_09` were invariant across all
five executions. `VAL_04` produced two matched-stream subsets: three runs
matched 3/27 streams and two matched 8/27. Across all five runs, temperature
MAPE was 19.64 ± 2.72% and pressure MAPE was 49.93 ± 5.38% (mean ± sample SD).
All five `VAL_04` flowsheets fully solved, although only one returned `PASS`;
the other four terminated at `MAX_ITER`.

## Component ablation

The targeted study compared `full_ccs`, `no_physics`, and `no_coupling` using
one best-of-three execution for each of four cases. No arm produced a `PASS`.
`VAL_01`, `VAL_03`, and `VAL_06` were identical across all three arms.
`VAL_02` changed from `MAX_ITER` with 9/13 matched streams under `full_ccs` to
`HUMAN` without a scored comparison under both removal arms. Because only one
stochastic best-of-three execution was retained per case-arm, this difference
cannot be attributed causally to either removed component.

Where activation statistics were present, bubble-point call sites were reached
15–30 times but produced no phase-derived repair candidate, and the coupling
map was never queried. Code inspection verifies that coupling is queried only
after a candidate repair creates a beam state and at least one unfixed error
remains. The traces therefore establish that the coupling gate was not reached;
they do not establish a single shared reason for non-activation in every case.
