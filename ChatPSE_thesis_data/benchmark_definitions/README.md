# Benchmark Definitions

All benchmark inputs that are independent of model or pipeline execution.

## `cases/`

JSON case files for all experimental groups. Each file follows the schema defined in
`benchmark/case_schema.py`. Copy from the repository:

```
benchmark/cases/capability.json       → cases/capability.json
benchmark/cases/easy.json             → cases/easy.json
benchmark/cases/sanity.json           → cases/sanity.json
benchmark/cases/generalisation.json   → cases/generalisation.json
benchmark/cases/val_l0.json           → cases/val_l0.json
benchmark/cases/val_l1.json           → cases/val_l1.json
benchmark/cases/val_specified.json    → cases/val_specified.json
```

The validation L2 (fully specified) cases `VALS_01`, `VALS_03`, `VALS_04` are in `val_specified.json`.

## `references/`

Reference flowsheet JSON files. Each is an expert-specified DWSIM solution used for
post-execution comparison. Copy from the repository:

```
benchmark/reference_flowsheets/*.json        → references/capability/
benchmark/reference_flowsheets_v2/*.json     → references/validation/
```

See `PROVENANCE.md` in this directory for construction methodology and per-case notes.

## `PROVENANCE.md`

Copy of `benchmark/reference_flowsheets/PROVENANCE.md`. Documents:
- Construction methodology (expert-specified DWSIM solutions)
- Validity criteria (set-point vs computed quantities)
- Minimum-match gate for MAPE reporting
- Vapour-fraction sensitivity analysis
- Per-case construction notes and corrections
- Self-consistency verification results
