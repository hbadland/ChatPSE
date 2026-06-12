"""
Validation script for hard_benchmark_mu.json.
Run from project root: PYTHONPATH=. python3.9 benchmark/validate_hard_mu.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "benchmark" / "cases"
MU_FILE   = CASES_DIR / "hard_benchmark_mu.json"
DWSIM_TXT = ROOT / "rag" / "sources" / "dwsim_compounds.txt"
BIP_FILE  = ROOT / "rag" / "sources" / "binary_parameters.json"

# ── Load reference data ───────────────────────────────────────────────────────

with open(DWSIM_TXT) as f:
    dwsim_lower = {line.strip().lower() for line in f if line.strip()}

with open(BIP_FILE) as f:
    _bips = json.load(f)
bip_pairs: set[frozenset] = set()
for b in _bips:
    bip_pairs.add(frozenset([b.get("compound_a", "").lower(),
                              b.get("compound_b", "").lower()]))

with open(MU_FILE) as f:
    cases = json.load(f)

# ── Required fields (from existing case schema) ───────────────────────────────
REQUIRED = {"id", "name", "tier", "difficulty", "coupling_level", "perturbation",
            "domain", "description", "compounds", "reference_structure", "expected", "notes"}

VALID_DOMAINS     = {"polar", "hydrocarbon", "azeotrope", "mixed"}
VALID_DIFFICULTIES = {"easy", "medium", "hard", "very_hard"}
VALID_PKG_CLASSES  = {"activity_coefficient", "eos", "ideal"}
REQUIRED_EXPECTED  = {"property_package_class", "n_units_min", "n_units_max",
                      "convergence_expected", "physics_checks"}

# Track errors
errors:   list[str] = []
warnings: list[str] = []

print("=" * 70)
print(f"Validating: {MU_FILE.name}  ({len(cases)} cases)")
print("=" * 70)

# ── Per-case checks ───────────────────────────────────────────────────────────
id_set:        set[str]        = set()
compound_pairs: list[frozenset] = []
domain_counts:  dict[str, int] = {}
tier_counts:    dict[str, int] = {}
unit_count_dist: dict[int, int] = {}
repair_3plus   = 0

for c in cases:
    cid = c.get("id", "<NO ID>")

    # 1. Required fields present
    missing = REQUIRED - set(c.keys())
    if missing:
        errors.append(f"{cid}: missing fields {missing}")

    # 2. No duplicate IDs
    if cid in id_set:
        errors.append(f"Duplicate ID: {cid}")
    id_set.add(cid)

    # 3. Domain valid
    dom = c.get("domain", "")
    if dom not in VALID_DOMAINS:
        errors.append(f"{cid}: invalid domain '{dom}'")
    domain_counts[dom] = domain_counts.get(dom, 0) + 1

    # 4. Difficulty valid
    if c.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append(f"{cid}: invalid difficulty '{c.get('difficulty')}'")

    # 5. Tier consistent
    tier = c.get("tier", "")
    tier_counts[tier] = tier_counts.get(tier, 0) + 1
    if tier != "multi_unit":
        errors.append(f"{cid}: tier must be 'multi_unit', got '{tier}'")

    # 6. Compounds in DWSIM
    compounds = c.get("compounds", [])
    for comp in compounds:
        if comp.lower() not in dwsim_lower:
            errors.append(f"{cid}: compound '{comp}' NOT in dwsim_compounds.txt")

    # 7. No duplicate compound pairs across cases
    fp = frozenset(c_.lower() for c_ in compounds)
    if fp in compound_pairs:
        errors.append(f"{cid}: compound pair {sorted(fp)} already used in a previous case")
    compound_pairs.append(fp)

    # 8. reference_structure present and consistent
    rs = c.get("reference_structure", {})
    n_units = rs.get("n_units", 0)
    unit_types = rs.get("unit_types", [])
    if len(unit_types) != n_units:
        errors.append(f"{cid}: n_units={n_units} but len(unit_types)={len(unit_types)}")
    unit_count_dist[n_units] = unit_count_dist.get(n_units, 0) + 1
    if n_units < 3 or n_units > 5:
        warnings.append(f"{cid}: n_units={n_units} — expected 3–5 for multi_unit tier")

    # 9. expected block
    exp = c.get("expected", {})
    missing_exp = REQUIRED_EXPECTED - set(exp.keys())
    if missing_exp:
        errors.append(f"{cid}: expected block missing {missing_exp}")
    pkg = exp.get("property_package_class", "")
    if pkg not in VALID_PKG_CLASSES:
        errors.append(f"{cid}: property_package_class '{pkg}' invalid")

    # 10. BIP check: if activity_coefficient, check BIPs exist
    if pkg == "activity_coefficient":
        # collect all compound pairs
        comps_lower = [c_.lower() for c_ in compounds]
        for i in range(len(comps_lower)):
            for j in range(i + 1, len(comps_lower)):
                pair = frozenset([comps_lower[i], comps_lower[j]])
                if pair not in bip_pairs:
                    warnings.append(
                        f"{cid}: NRTL pair {sorted(pair)} NOT in binary_parameters.json "
                        f"(missing_bip case or oversight)"
                    )

    # 11. source field present (new required field)
    if "source" not in c:
        errors.append(f"{cid}: missing 'source' field")

    # 12. expected_repair_iterations
    eri = c.get("expected_repair_iterations", 0)
    if eri >= 3:
        repair_3plus += 1

    # 13. At least one mass_balance physics check
    checks = exp.get("physics_checks", [])
    check_types = {ch.get("type") for ch in checks}
    if "mass_balance" not in check_types:
        warnings.append(f"{cid}: no mass_balance physics check")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nTotal cases  : {len(cases)}")
print(f"Unique IDs   : {len(id_set)}")

print(f"\nDomain distribution:")
for dom, n in sorted(domain_counts.items()):
    bar = "█" * n
    print(f"  {dom:<15s}: {n:2d}  {bar}")

print(f"\nUnit count distribution:")
for nu, n in sorted(unit_count_dist.items()):
    bar = "█" * n
    print(f"  {nu} units: {n:2d}  {bar}")

print(f"\nCases with expected_repair_iterations >= 3: {repair_3plus}")

print(f"\nCompound pair uniqueness: {'PASS — all pairs unique' if len(set(map(tuple, map(sorted, compound_pairs)))) == len(cases) else 'FAIL — duplicates found'}")

print(f"\nPhysics check types used across all cases:")
all_check_types: dict[str, int] = {}
for c in cases:
    for ch in c.get("expected", {}).get("physics_checks", []):
        t = ch.get("type", "?")
        all_check_types[t] = all_check_types.get(t, 0) + 1
for t, n in sorted(all_check_types.items(), key=lambda x: -x[1]):
    print(f"  {t:<40s}: {n}")

# ── Errors and warnings ───────────────────────────────────────────────────────

if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠  {w}")

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  ✗  {e}")
    print(f"\nValidation FAILED — {len(errors)} error(s)")
    sys.exit(1)
else:
    print(f"\n✓  Validation PASSED — 0 errors, {len(warnings)} warning(s)")
