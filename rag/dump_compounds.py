#!/usr/bin/env python3
"""
Dump all DWSIM compound names to rag/sources/dwsim_compounds.txt.

Must be run INSIDE the DWSIM Singularity/Docker container where pythonnet
and the DWSIM .dlls are present at /usr/local/lib/dwsim/.

Usage (on HPC, inside singularity shell):
    cd /rds/general/user/hgb25/home/multiAgentFlowsheet
    singularity exec --bind /rds /path/to/dwsim.sif \
        python3.9 rag/dump_compounds.py

Output is written to rag/sources/dwsim_compounds.txt (one canonical name per
line, sorted, de-duplicated).  The file is read by agents/basis.py at startup
to power fuzzy compound name validation.

Also runs a quick probe on known-ambiguous compound names and prints a report
so you can update compound_database.md if DWSIM uses a different name.
"""

import os
import sys
from typing import Optional

# ── DWSIM bootstrap (identical to dwsim_wrapper.py) ──────────────────────────
os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")

import pythonnet
try:
    pythonnet.load("coreclr")
except RuntimeError as _e:
    if "already" not in str(_e).lower():
        raise

import clr
sys.path.append("/usr/local/lib/dwsim/")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Automation.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Interfaces.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.SharedClasses.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Thermodynamics.dll")

from DWSIM.Automation import Automation3

# ── Get full compound list ────────────────────────────────────────────────────

print("Initialising DWSIM Automation3 ...", flush=True)
auto = Automation3()
sim  = auto.CreateFlowsheet()

# Automation3 exposes all registered compounds via AvailableCompounds
# (a Dictionary<string, ConstantProperties> populated from chemsep1.xml etc.)
available = getattr(sim, "AvailableCompounds", None)
if available is None:
    # Fallback: older DWSIM versions may expose it differently
    available = getattr(auto, "GetAvailableCompounds", lambda: None)()

if available is None:
    print("ERROR: could not retrieve AvailableCompounds — check DWSIM version",
          file=sys.stderr)
    sys.exit(1)

names = sorted(str(k) for k in available.Keys)
print(f"Found {len(names)} compounds in DWSIM database.", flush=True)

# ── Write output file ─────────────────────────────────────────────────────────
out_path = os.path.join(
    os.path.dirname(__file__), "sources", "dwsim_compounds.txt")

with open(out_path, "w") as f:
    for name in names:
        f.write(name + "\n")

print(f"Written: {out_path}", flush=True)

# ── Probe ambiguous benchmark compounds ───────────────────────────────────────
# Tests names that compound_database.md marks as ✓ or ? but may be wrong.
# Creates a fresh flowsheet for each probe to avoid cross-contamination.

PROBES = {
    # key = canonical name we EXPECT; value = names to try in priority order
    # Light gases
    "Water":              ["Water"],
    "Carbon Dioxide":     ["Carbon Dioxide", "Carbon dioxide", "CO2"],
    "Hydrogen Sulfide":   ["Hydrogen Sulfide", "Hydrogen sulfide", "H2S"],
    "Ammonia":            ["Ammonia"],
    "Hydrogen Chloride":  ["Hydrogen Chloride", "Hydrogen chloride", "HCl"],
    "Nitrogen":           ["Nitrogen"],
    "Oxygen":             ["Oxygen"],
    "Methane":            ["Methane"],
    "Ethane":             ["Ethane"],
    "Propane":            ["Propane"],
    "n-Butane":           ["n-Butane", "n-butane", "Butane"],
    "n-Pentane":          ["n-Pentane", "n-pentane", "Pentane"],
    "n-Hexane":           ["n-Hexane", "n-hexane", "Hexane"],
    "n-Heptane":          ["n-Heptane", "n-heptane", "Heptane"],
    "n-Octane":           ["n-Octane", "n-octane", "Octane"],
    # Olefins
    "Ethylene":           ["Ethylene"],
    "Propylene":          ["Propylene"],
    "1-Butene":           ["1-Butene", "1-butene", "But-1-ene"],
    "Isobutylene":        ["Isobutylene", "Isobutene", "IsoButene", "2-Methylpropene"],
    # Aromatics
    "Benzene":            ["Benzene"],
    "Toluene":            ["Toluene"],
    "o-Xylene":           ["o-Xylene", "o-xylene", "O-xylene"],
    "m-Xylene":           ["m-Xylene", "m-xylene", "M-xylene"],
    "p-Xylene":           ["p-Xylene", "p-xylene", "P-xylene"],
    "Styrene":            ["Styrene"],
    "Cumene":             ["Cumene"],
    "Ethylbenzene":       ["Ethylbenzene"],
    "Naphthalene":        ["Naphthalene"],
    # Alcohols (suspected lowercase in this build)
    "1-Propanol":         ["1-Propanol", "1-propanol", "n-Propanol", "n-propanol"],
    "2-Propanol":         ["2-Propanol", "Isopropanol", "isopropanol", "2-propanol"],
    "1-Butanol":          ["1-Butanol", "1-butanol", "n-Butanol"],
    "2-Butanol":          ["2-Butanol", "2-butanol", "sec-Butanol"],
    "2-Methyl-1-propanol":["2-Methyl-1-propanol", "2-methyl-1-propanol", "Isobutanol"],
    "2-Methyl-2-propanol":["2-Methyl-2-propanol", "2-methyl-2-propanol", "tert-Butanol"],
    "Methanol":           ["Methanol"],
    "Ethanol":            ["Ethanol"],
    "Ethylene Glycol":    ["Ethylene Glycol", "Ethylene glycol"],
    "Glycerol":           ["Glycerol"],
    # Ketones / aldehydes
    "Acetone":            ["Acetone"],
    "Methyl Ethyl Ketone":["Methyl Ethyl Ketone", "Methyl ethyl ketone", "2-Butanone"],
    "Methyl Isobutyl Ketone": ["Methyl Isobutyl Ketone", "Methyl isobutyl ketone", "MIBK"],
    "Cyclohexanone":      ["Cyclohexanone"],
    "Acetaldehyde":       ["Acetaldehyde"],
    "Formaldehyde":       ["Formaldehyde"],
    # Acids
    "Acetic Acid":        ["Acetic Acid", "Acetic acid", "Ethanoic Acid"],
    "Formic Acid":        ["Formic Acid", "Formic acid"],
    "Acrylic Acid":       ["Acrylic Acid", "Acrylic acid"],
    "Benzoic Acid":       ["Benzoic Acid", "Benzoic acid"],
    # Esters
    "Ethyl Acetate":      ["Ethyl Acetate", "Ethyl acetate"],
    "Methyl Acetate":     ["Methyl Acetate", "Methyl acetate"],
    # Ethers
    "Diethyl Ether":      ["Diethyl Ether", "Diethyl ether"],
    "Tetrahydrofuran":    ["Tetrahydrofuran", "THF"],
    "Methyl tert-Butyl Ether": ["Methyl tert-Butyl Ether", "Methyl tert-butyl ether", "MTBE"],
    # Halogenated
    "Dichloromethane":    ["Dichloromethane"],
    "Chloroform":         ["Chloroform"],
    "Carbon Tetrachloride":["Carbon Tetrachloride", "Carbon tetrachloride"],
    "Vinyl Chloride":     ["Vinyl Chloride", "Vinyl chloride"],
    # N / S compounds
    "Acetonitrile":       ["Acetonitrile"],
    "Acrylonitrile":      ["Acrylonitrile"],
    "Dimethyl Sulfoxide": ["Dimethyl Sulfoxide", "Dimethyl sulfoxide", "DMSO"],
    "Dimethylformamide":  ["Dimethylformamide", "DMF"],
    # Cyclic
    "Cyclohexane":        ["Cyclohexane", "CycloHexane"],
    "Cyclopentane":       ["Cyclopentane"],
    # Epoxides (newly added)
    "Ethylene Oxide":     ["Ethylene Oxide", "Ethylene oxide"],
    "Propylene Oxide":    ["Propylene Oxide", "Propylene oxide", "1,2-propylene oxide"],
}

print("\n─── Compound name probe results ───────────────────────────────────────")
print(f"{'Compound':<28}  {'DWSIM accepted as':<32}  Result")
print("─" * 72)

def _try_add(flowsheet, name: str) -> "Optional[str]":
    """
    Try AddCompound(name). Returns the accepted compound name on success, None on failure.
    DWSIM raises KeyNotFoundException for unrecognised names — catch gracefully.
    """
    try:
        flowsheet.AddCompound(name)
        added = list(flowsheet.SelectedCompounds.Keys)
        return added[0] if added else None
    except Exception:
        return None


results = {}
for canonical, candidates in PROBES.items():
    accepted = None
    for name in candidates:
        probe = auto.CreateFlowsheet()
        result = _try_add(probe, name)
        if result:
            accepted = result
            break
    results[canonical] = accepted
    status = f"✓ \"{accepted}\"" if accepted else "✗ NOT FOUND"
    mismatch = " ← RENAME NEEDED" if accepted and accepted != canonical else ""
    print(f"{canonical:<28}  {status:<38}{mismatch}")

print()
rename_needed = {c: a for c, a in results.items() if a and a != c}
not_found     = [c for c, a in results.items() if a is None]

if rename_needed:
    print("compound_database.md entries that need renaming:")
    for old, new in rename_needed.items():
        print(f"  '{old}'  →  '{new}'")

if not_found:
    print("Compounds NOT found in this DWSIM installation:")
    for c in not_found:
        print(f"  '{c}'  — mark as unsupported or find correct name manually")

print("\nDone. Update compound_database.md with any renames shown above.")
