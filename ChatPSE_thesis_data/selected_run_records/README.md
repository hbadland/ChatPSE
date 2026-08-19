# Selected Run Records

Per-run JSON records included in the reported thesis aggregate, organised by experimental group.
Only the records cited in `thesis_2026_manifest.json` belong here. Development runs, discarded
repeats, and runs from earlier code versions are excluded.

## File naming convention

Records follow the logger naming pattern:

```
<case_id>_<mode>_<model_slug>_<timestamp>.json
```

Example: `VAL_06_full_ccs_qwen3_30b-a3b_20260815_143022.json`

## Subdirectory contents

| Directory | Experimental group | N per case | max_iter | USE_LANGGRAPH | BEST_OF_N |
|---|---|---|---|---|---|
| `capability/` | 20-case headline benchmark | 5 | 6 (+15 beam) | 1 | 5 |
| `validation/` | VAL_01–VAL_10 panel | 5 | 6 (+15 beam) | 1 | 5 |
| `specification_density/` | L0/L1/L2 on VAL_01/03/04 | 5 | 15 | 1 | 5 |
| `model_substitution/` | 5 models × 3 L1 cases | 5 | 6 (+15 beam) | 1 | 5 |
| `ablation/` | Capability (N=1) + targeted validation (N=5) | see manifest | see manifest | varies | varies |

## Key fields in each record

See `PIPELINE.md` in the root repository for full field definitions. Primary fields:
- `case_id`, `ablation_mode`, `model`, `timestamp`, `outcome`
- `iterations` — per-iteration repair history
- `system_streams` — DWSIM stream conditions (used for reference comparison)
- `ablation_stats` — bubble-point and coupling activation counters (instrumented runs only)

## To populate from HPC

Transfer the selected records from `results/per_run/` on the HPC filesystem:

```bash
# From the HPC login node — adapt paths
rsync -av \
    hgb25@login.cx3.hpc.ic.ac.uk:/rds/general/user/hgb25/home/multiAgentFlowsheet/results/per_run/ \
    selected_run_records/
```

Then sort records into the correct subdirectory by `case_id` prefix and update
`thesis_2026_manifest.json` with the exact filenames.
