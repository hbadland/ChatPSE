"""
Validation script for hard_benchmark_mb.json.
Run: PYTHONPATH=. python3.9 benchmark/validate_hard_mb.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
MB_FILE   = ROOT / "benchmark" / "cases" / "hard_benchmark_mb.json"
DWSIM_TXT = ROOT / "rag" / "sources" / "dwsim_compounds.txt"
BIP_FILE  = ROOT / "rag" / "sources" / "binary_parameters.json"

with open(DWSIM_TXT) as f:
    dwsim_lower = {l.strip().lower() for l in f if l.strip()}
with open(BIP_FILE) as f:
    _bips = json.load(f)
bip_pairs: set[frozenset] = {
    frozenset([b.get("compound_a","").lower(), b.get("compound_b","").lower()])
    for b in _bips
}
with open(MB_FILE) as f:
    cases = json.load(f)

REQUIRED = {"id","name","tier","difficulty","coupling_level","perturbation",
            "domain","description","compounds","reference_structure","expected",
            "notes","source","missing_bip_pairs"}
VALID_DOMAINS = {"polar","hydrocarbon","azeotrope","mixed"}

errors, warnings = [], []
id_set = set()
domain_counts: dict[str,int] = {}
unit_count_dist: dict[int,int] = {}
convergence_by_domain: dict[str,list[bool]] = {}

print("=" * 70)
print(f"Validating: {MB_FILE.name}  ({len(cases)} cases)")
print("=" * 70)

for c in cases:
    cid = c.get("id","<NO ID>")
    missing = REQUIRED - set(c.keys())
    if missing:
        errors.append(f"{cid}: missing fields {missing}")
    if cid in id_set:
        errors.append(f"Duplicate ID: {cid}")
    id_set.add(cid)

    dom = c.get("domain","")
    if dom not in VALID_DOMAINS:
        errors.append(f"{cid}: invalid domain '{dom}'")
    domain_counts[dom] = domain_counts.get(dom,0)+1

    if c.get("tier") != "missing_bip":
        errors.append(f"{cid}: tier must be 'missing_bip', got '{c.get('tier')}'")

    compounds = c.get("compounds",[])
    for comp in compounds:
        if comp.lower() not in dwsim_lower:
            errors.append(f"{cid}: '{comp}' NOT in dwsim_compounds.txt")

    # Confirm each declared missing pair is genuinely missing
    for mp in c.get("missing_bip_pairs",[]):
        a = mp.get("compound_a","").lower()
        b = mp.get("compound_b","").lower()
        pair = frozenset([a,b])
        if pair in bip_pairs:
            errors.append(f"{cid}: declared missing pair {sorted(pair)} IS in BIP corpus")
        if a not in dwsim_lower:
            errors.append(f"{cid}: missing_bip compound '{a}' NOT in DWSIM")
        if b not in dwsim_lower:
            errors.append(f"{cid}: missing_bip compound '{b}' NOT in DWSIM")

    rs = c.get("reference_structure",{})
    n_u = rs.get("n_units",0)
    if len(rs.get("unit_types",[])) != n_u:
        errors.append(f"{cid}: n_units mismatch")
    unit_count_dist[n_u] = unit_count_dist.get(n_u,0)+1

    if "source" not in c:
        errors.append(f"{cid}: missing 'source' field")
    if "missing_bip_pairs" not in c or not c["missing_bip_pairs"]:
        errors.append(f"{cid}: missing_bip_pairs must be non-empty")

    conv = c.get("expected",{}).get("convergence_expected")
    convergence_by_domain.setdefault(dom,[]).append(bool(conv))

# Summary
print(f"\nTotal cases  : {len(cases)}")
print(f"\nDomain distribution:")
for dom, n in sorted(domain_counts.items()):
    print(f"  {dom:<15s}: {n:2d}  {'█'*n}")

print(f"\nUnit count distribution:")
for nu, n in sorted(unit_count_dist.items()):
    print(f"  {nu} units: {n:2d}  {'█'*n}")

print(f"\nConvergence by domain:")
for dom, convs in sorted(convergence_by_domain.items()):
    n_true  = sum(convs)
    n_false = len(convs)-n_true
    print(f"  {dom:<15s}: {n_true} expected-converge, {n_false} expected-fail")

# Verify no pair appears in both MB file and BIP corpus (redundant)
print(f"\nMissing BIP pair verification:")
all_clean = True
for c in cases:
    for mp in c.get("missing_bip_pairs",[]):
        a = mp.get("compound_a","").lower()
        b = mp.get("compound_b","").lower()
        pair = frozenset([a,b])
        in_bip = pair in bip_pairs
        status = "MISSING ✓" if not in_bip else "IN CORPUS ✗"
        if in_bip:
            all_clean = False
        print(f"  {c['id']}: {a}/{b} → {status}")

if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings: print(f"  ⚠  {w}")
if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors: print(f"  ✗  {e}")
    print(f"\nValidation FAILED — {len(errors)} error(s)")
    sys.exit(1)
else:
    print(f"\n✓  Validation PASSED — 0 errors, {len(warnings)} warning(s)")
