#!/usr/bin/env bash
# populate_benchmark_definitions.sh
#
# Step 4 of the data-transfer workflow.
# Copies benchmark case files and reference flowsheets from the repository
# into ChatPSE_thesis_data/benchmark_definitions/.
#
# Before running: confirm that FOS_01_reference.json may be redistributed
# under the FOSSEE licence. If not, remove it from the references/ directory
# after this script runs, and add a citation instead (see DEPOSIT_POLICY.md).
#
# Run from the repository root:
#   bash ChatPSE_thesis_data/analysis/scripts/populate_benchmark_definitions.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BD="${REPO_ROOT}/ChatPSE_thesis_data/benchmark_definitions"

# ── Case JSON files ────────────────────────────────────────────────────────────
echo "[cases] Copying case files..."
mkdir -p "${BD}/cases"
for f in capability easy sanity generalisation val_l0 val_l1 val_specified; do
    src="${REPO_ROOT}/benchmark/cases/${f}.json"
    if [[ -f "$src" ]]; then
        cp "$src" "${BD}/cases/"
        echo "  copied ${f}.json"
    else
        echo "  WARNING: ${src} not found"
    fi
done

# ── Reference flowsheets ───────────────────────────────────────────────────────
echo ""
echo "[references] Copying reference flowsheets..."
mkdir -p "${BD}/references/capability" "${BD}/references/validation"

# Capability references (benchmark/reference_flowsheets/)
if [[ -d "${REPO_ROOT}/benchmark/reference_flowsheets" ]]; then
    for f in "${REPO_ROOT}/benchmark/reference_flowsheets"/*.json; do
        fname="$(basename "$f")"
        cp "$f" "${BD}/references/capability/"
        echo "  copied capability/${fname}"
    done
fi

# Validation references (benchmark/reference_flowsheets_v2/)
if [[ -d "${REPO_ROOT}/benchmark/reference_flowsheets_v2" ]]; then
    for f in "${REPO_ROOT}/benchmark/reference_flowsheets_v2"/*.json; do
        fname="$(basename "$f")"
        cp "$f" "${BD}/references/validation/"
        echo "  copied validation/${fname}"
    done
fi

# ── Provenance document ────────────────────────────────────────────────────────
echo ""
echo "[provenance] Copying PROVENANCE.md..."
cp "${REPO_ROOT}/benchmark/reference_flowsheets/PROVENANCE.md" "${BD}/PROVENANCE.md"
echo "  copied PROVENANCE.md"

# ── FOS_01 licence check reminder ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ACTION REQUIRED: FOS_01 licence check"
echo ""
echo "FOS_01_reference.json was adapted from the FOSSEE process-simulation"
echo "repository (https://fossee.in). Before this archive is deposited:"
echo ""
echo "  1. Check whether the FOSSEE materials permit redistribution."
echo "  2. If YES: leave FOS_01_reference.json in references/capability/ and"
echo "     add a citation in PROVENANCE.md."
echo "  3. If NO or UNCLEAR: remove it now:"
echo "       rm ${BD}/references/capability/FOS_01_reference.json"
echo "     and add a citation-only note in PROVENANCE.md."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Done. Review ${BD}/ before committing."
