# ChatPSE Thesis Data Archive

This directory is the curated research data archive for the MSc thesis:

> **[THESIS TITLE]** — Harry Badland, Imperial College London, 2026

It is structured to satisfy the Imperial College London Research Data Management policy. All data required to reproduce the reported results, or to verify the reported conclusions without re-running inference, is collected here.

---

## Archive contents

### `thesis_2026_manifest.json`

Machine-readable index of every result record included in the reported aggregate. This is the authoritative record for reproducing the published tables; a date-filtered query of `results/per_run/` is approximate and should not be used as a substitute. See the manifest for the exact record list and associated metadata.

### `code/`

A snapshot of the source code at the reported results commit. The canonical version is the tagged release at `[THESIS RELEASE TAG]` on GitHub (`https://github.com/hbadland/multiAgentFlowsheet`). The snapshot here is provided for archival self-containment.

### `benchmark_definitions/`

All inputs to the benchmark that are independent of the model or pipeline run:
- `cases/` — JSON case files (natural-language descriptions + tier assignments) for all experimental groups
- `references/` — Reference flowsheet JSON files (expert-specified DWSIM solutions)
- `PROVENANCE.md` — Construction methodology, validity criteria, per-case notes, and integrity checks for all reference flowsheets

### `selected_run_records/`

Per-run JSON records selected for inclusion in the reported aggregate, organised by experimental group. Each subdirectory contains only the records used to compute the reported results. Development runs and discarded repeats are excluded.

- `capability/` — Main 20-case capability benchmark (headline results)
- `validation/` — Ten-case validation panel (VAL_01–VAL_10)
- `specification_density/` — L0/L1/L2 density study (VAL_01, VAL_03, VAL_04)
- `model_substitution/` — Substitution study (qwen3:32b, deepseek-r1-distill-qwen-32b, gemma3:27b, mistral-small3.2:24b on three L1 cases)
- `ablation/` — Component ablation runs (capability set N=1; VAL_01_L1, VAL_03_L1, VAL_04_L1, VAL_05, VAL_06 at N=5)

### `analysis/`

- `scripts/` — Aggregation and figure-generation scripts used to produce the reported tables and figures
- `outputs/` — Final figure and table input data (CSVs, JSON summaries) as submitted to the manuscript

### `environment/`

Everything needed to reconstruct the execution environment:
- `requirements.txt` — Python dependencies (direct, without version pins — see `software_versions.txt` for known pinned versions)
- `Dockerfile` — Container definition (Ubuntu 20.04, Python 3.9, .NET 8, DWSIM 9.0.4)
- `software_versions.txt` — Recorded and unrecorded software versions with provenance notes

---

## How to use this archive

**To verify the reported aggregate without re-running inference:**

```bash
PYTHONPATH=<code_snapshot_path> python3.9 analysis/scripts/aggregate_from_manifest.py \
    --manifest thesis_2026_manifest.json \
    --records selected_run_records/ \
    --out analysis/outputs/aggregate_verification.json
```

*(Requires only `networkx`, `scipy`, `numpy` — no container, DWSIM, or model inference needed.)*

**To inspect individual run records:** each JSON file in `selected_run_records/` is self-contained and human-readable. See `PIPELINE.md` in the root repository for field definitions.

**To re-run inference:** follow the instructions in the root `README.md`. The container definition and dependency list here match the environment used for the reported runs.

---

## Archival deposit

This directory and its contents are deposited at `[ARCHIVE DOI]`. The deposit was created on `[DEPOSIT DATE]`.
