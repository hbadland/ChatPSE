# Analysis

Scripts and outputs used to produce the reported tables and figures.

## `scripts/`

Aggregation and figure-generation scripts. The primary aggregation script is
`benchmark/aggregate_per_run.py` from the code snapshot. Copy it here and verify it
accepts a `--manifest` flag for manifest-based (not date-filtered) recalculation:

```bash
python3.9 scripts/aggregate_from_manifest.py \
    --manifest ../thesis_2026_manifest.json \
    --records ../selected_run_records/ \
    --out outputs/aggregate_verification.json
```

Place any figure-generation or table-formatting scripts here as well.

## `outputs/`

Final input data for figures and tables as submitted to the manuscript. Each file should
be named to correspond to the figure or table number in the thesis. Include:
- Aggregate metric summaries (JSON or CSV)
- Per-case result tables (CSV)
- Figure input data (CSV/JSON) for all plots in the thesis
- A `figure_index.md` mapping each output file to the thesis figure/table it generates

These files should be exactly the data used to produce the submitted figures — not
recalculated from scratch, unless recalculation produces an identical result.
