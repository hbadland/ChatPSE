# ChatPSE thesis data

This deposit contains the selected machine-readable records underlying the
reported ChatPSE thesis results. It is a curated release, not a dump of the
mixed development-results directory.

Canonical repository:
<https://github.com/hbadland/multiAgentFlowsheet/tree/main>

The records are separated into six analytical cohorts across five top-level
groups:

- `capability`: the 20-case capability panel selected on 10 August 2026.
- `validation/main_panel`: the ten-case validation panel selected on
  10 August 2026.
- `validation/repeatability`: five independent single-sample executions for
  each of `VAL_01`, `VAL_04`, `VAL_06`, and `VAL_09`. These runs used one
  generated candidate per execution and are not best-of-five samples.
- `specification_density`: L0, L1, and L2 records for `VAL_01`, `VAL_03`, and
  `VAL_04`.
- `model_substitution`: the three L1 cases evaluated with the baseline model
  and four substitute models.
- `ablation/component_targeted`: matched `full_ccs`, `no_physics`, and
  `no_coupling` records for `VAL_01`, `VAL_02`, `VAL_03`, and `VAL_06`.

The component ablation produced no improvement under either removal arm.
`VAL_01`, `VAL_03`, and `VAL_06` were invariant, whereas `VAL_02` changed from
`MAX_ITER` with 9/13 matched streams under `full_ccs` to `HUMAN` without a
scored comparison in both removal arms. Because only one stochastic
best-of-three execution was retained per case-arm, this difference cannot be
assigned causally to either component. Activation counters show that
bubble-point call sites did not supply a selected phase-derived repair, while
the coupling map was never queried. Code inspection confirms that coupling is
consulted only after a candidate fix creates a beam state and another unfixed
error remains. The records establish that this gate was not reached; they do
not establish one common case-level reason for every non-activation.

`selection_manifest.tsv` is the authoritative mapping from archived source
records to analytical groups. Some baseline records legitimately appear in
more than one group. `SHA256SUMS` will identify the deposited copies after
staging and validation.

## Exclusions

Exploratory, interrupted, superseded, and unreported records are excluded.
In particular, the earlier eight-case `no_physics` batch from 17 August is not
part of the reported four-case component ablation.

`FOS_01` is also excluded. It was adapted from FOSSEE during development but
was not included in the reported ten-case validation panel or any aggregate
reported in the thesis.

## Licence

The generated research data, derived summaries, metadata, manifests, and
redacted logs in this directory are released under CC BY 4.0. See
`DATA_LICENSE.md`. ChatPSE source code is separately licensed under the MIT
licence in the repository root. Third-party material is not included here.
