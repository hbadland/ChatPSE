"""
sort_records.py

Step 2 of the data-transfer workflow.
Reads per-run JSON records from a staging directory, classifies each into an
experimental group, filters by date, and moves selected records into the
appropriate ChatPSE_thesis_data/selected_run_records/ subdirectory.

Also updates thesis_2026_manifest.json with the filenames of deposited records.

Usage:
    python3.9 sort_records.py \\
        --staging   /path/to/_staging/per_run \\
        --archive   /path/to/ChatPSE_thesis_data \\
        --manifest  /path/to/ChatPSE_thesis_data/thesis_2026_manifest.json \\
        --since     20260810

Dry-run (print decisions without moving files):
    python3.9 sort_records.py ... --dry-run
"""

import argparse
import json
import re
import shutil
from pathlib import Path

# ── Known case IDs by group ────────────────────────────────────────────────────

CAPABILITY_CASES = {
    "P1","P2","P3",
    "F1","F2","F3","F4",
    "S1","S2",
    "C1","C2","C3",
    "M1",
    "EASY_01","EASY_02","EASY_04",
    "SAN_03","SAN_04",
    "GEN_01","GEN_03",
}

VALIDATION_CASES = {
    "VAL_01","VAL_02","VAL_03","VAL_04","VAL_05",
    "VAL_06","VAL_07","VAL_08","VAL_09","VAL_10",
}

# L0/L1 density variants and fully-specified variants
DENSITY_CASES = {
    "VAL_01_L0","VAL_01_L1",
    "VAL_03_L0","VAL_03_L1",
    "VAL_04_L0","VAL_04_L1",
    "VALS_01","VALS_03","VALS_04",
}

# Targeted ablation cases (VAL_01_L1, VAL_03_L1, VAL_04_L1 also appear here at
# no_physics/no_coupling — they go to ablation/, not specification_density/)
TARGETED_ABLATION_CASES = {"VAL_01_L1","VAL_03_L1","VAL_04_L1","VAL_05","VAL_06"}

ABLATION_MODES = {"no_physics","no_coupling","no_rule_store","greedy"}

PRIMARY_MODEL_SLUG = "qwen3_30b-a3b"  # as it appears in filenames

# ── Filename parser ────────────────────────────────────────────────────────────
# Expected format: <case_id>_<mode>_<model_slug>_<YYYYMMDD>_<HHMMSS>.json
# case_id and mode may contain underscores, so we parse from the right.
_TS_PATTERN = re.compile(r"_(\d{8})_(\d{6})\.json$")

def parse_filename(name: str):
    """Return (case_id, mode, model_slug, date_str) or None on parse failure."""
    m = _TS_PATTERN.search(name)
    if not m:
        return None
    date_str = m.group(1)
    stem = name[:m.start()]  # everything before _YYYYMMDD_HHMMSS.json

    # Known modes and model slugs to guide splitting
    known_modes = {"full_ccs","no_physics","no_coupling","no_rule_store","greedy"}
    known_slugs = {
        "qwen3_30b-a3b","qwen3_32b","deepseek-r1-distill-qwen-32b",
        "gemma3_27b","mistral-small3.2_24b",
    }

    # Split stem into parts and find mode and slug from the right
    parts = stem.split("_")
    model_slug = None
    mode = None

    for slug in known_slugs:
        slug_parts = slug.replace("-", "_").split("_")
        # Try matching the tail of parts against the slug
        if len(parts) >= len(slug_parts):
            if "_".join(parts[-len(slug_parts):]).lower() == slug.lower().replace("-","_"):
                model_slug = slug
                parts = parts[:-len(slug_parts)]
                break
    if model_slug is None:
        # Fall back: last part is model slug
        model_slug = parts[-1]
        parts = parts[:-1]

    for m_name in known_modes:
        m_parts = m_name.split("_")
        if len(parts) >= len(m_parts) and parts[-len(m_parts):] == m_parts:
            mode = m_name
            parts = parts[:-len(m_parts)]
            break
    if mode is None:
        mode = parts[-1]
        parts = parts[:-1]

    case_id = "_".join(parts)
    return case_id, mode, model_slug, date_str


def classify(case_id: str, mode: str, model_slug: str) -> str:
    """Return the target subdirectory name, or 'skip' to exclude."""

    # Ablation mode takes priority regardless of case
    if mode in ABLATION_MODES:
        return "ablation"

    # Non-primary model → model_substitution
    if PRIMARY_MODEL_SLUG not in model_slug:
        return "model_substitution"

    if case_id in CAPABILITY_CASES:
        return "capability"

    if case_id in VALIDATION_CASES:
        # full_ccs validation runs at full spec → validation
        return "validation"

    if case_id in DENSITY_CASES:
        return "specification_density"

    # VAL_05/06 at full_ccs (targeted ablation baseline) → ablation
    if case_id in {"VAL_05","VAL_06"}:
        return "ablation"

    return "skip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging",  required=True, help="Directory of pulled per-run JSONs")
    ap.add_argument("--archive",  required=True, help="ChatPSE_thesis_data/ root")
    ap.add_argument("--manifest", required=True, help="Path to thesis_2026_manifest.json")
    ap.add_argument("--since",    default="20260810", help="Include records on or after YYYYMMDD")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    staging  = Path(args.staging)
    archive  = Path(args.archive)
    manifest = Path(args.manifest)
    since    = args.since

    with open(manifest) as f:
        mf = json.load(f)

    records_dir = archive / "selected_run_records"
    counts = {"capability": 0, "validation": 0, "specification_density": 0,
              "model_substitution": 0, "ablation": 0, "skip": 0, "unparseable": 0}

    for record_path in sorted(staging.glob("*.json")):
        parsed = parse_filename(record_path.name)
        if parsed is None:
            print(f"[SKIP unparseable] {record_path.name}")
            counts["unparseable"] += 1
            continue

        case_id, mode, model_slug, date_str = parsed

        if date_str < since:
            print(f"[SKIP pre-{since}] {record_path.name}")
            counts["skip"] += 1
            continue

        group = classify(case_id, mode, model_slug)

        if group == "skip":
            print(f"[SKIP unclassified] {record_path.name} (case={case_id}, mode={mode})")
            counts["skip"] += 1
            continue

        dest_dir = records_dir / group
        dest_path = dest_dir / record_path.name

        print(f"[{group}] {record_path.name}")
        counts[group] += 1

        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record_path, dest_path)
            # Register in manifest
            group_key = group
            if group_key in mf["experimental_groups"]:
                if record_path.name not in mf["experimental_groups"][group_key].get("records", []):
                    mf["experimental_groups"][group_key].setdefault("records", []).append(record_path.name)
            elif group_key == "ablation":
                mf["experimental_groups"]["ablation"].setdefault("records", []).append(record_path.name)

    print("\n── Summary ──────────────────────────────────────────────────────")
    for k, v in counts.items():
        print(f"  {k:<25} {v}")

    if not args.dry_run:
        with open(manifest, "w") as f:
            json.dump(mf, f, indent=2)
        print(f"\nManifest updated: {manifest}")
    else:
        print("\n[dry-run] No files moved; manifest not updated.")


if __name__ == "__main__":
    main()
