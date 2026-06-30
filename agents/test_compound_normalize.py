"""
Tests for deterministic compound-name normalisation.

Run: PYTHONPATH=. python3.9 agents/test_compound_normalize.py

Covers the Probe-1 offenders that caused hard DWSIM "could not add compound" /
"key not present" failures and run-to-run flakiness.  All target keys verified
present in rag/sources/dwsim_compounds.txt.
"""
from agents.compound_normalize import (
    canonicalize_compound, canonicalize_reaction, canonicalize_list)


def test_offender_compounds():
    cases = {
        "Carbon Monoxide":    "Carbon monoxide",     # casing (VAL_04/10 hard blocker)
        "Isobutylene":        "Isobutene",           # synonym (HARD_03/PERT flaky)
        "Methyl Ethyl Ketone": "Methyl ethyl ketone", # casing (GEN_04 flaky)
        "1-Propanol":         "1-propanol",          # casing (EASY_03 flaky)
        "1,2-Dichloroethane": "1,2-dichloroethane",  # casing (VAL_05 reaction key)
        "1,2-dichloroethane": "1,2-dichloroethane",
        "2-Butanone":         "Methyl ethyl ketone", # synonym → DWSIM's MEK key
        "P-Xylene":           "p-Xylene",            # preserve VAL_09 entrainer isomer
        "o-Xylene":           "o-Xylene",
        "Ethanol":            "Ethanol",             # unchanged
        "Water":              "Water",
    }
    for raw, expect in cases.items():
        got, _ = canonicalize_compound(raw)
        assert got == expect, f"{raw!r} → {got!r}, expected {expect!r}"
    print("OK offender compounds normalise to exact DWSIM keys")


def test_bare_xylene_defaults_with_warning():
    got, warn = canonicalize_compound("Xylene")
    assert got == "m-Xylene", got
    assert warn and "ambiguous" in warn, "bare Xylene must warn"
    print("OK bare 'Xylene' defaults to m-Xylene with a warning:", got)


def test_reaction_canonicalisation():
    r1, _ = canonicalize_reaction("Ethylene + Chlorine -> 1,2-Dichloroethane")
    assert r1 == "Ethylene + Chlorine -> 1,2-dichloroethane", r1
    # coefficient on '3 Hydrogen' preserved; comma in compound not read as coeff
    r2, _ = canonicalize_reaction("Methane + Water -> Carbon Monoxide + 3 Hydrogen")
    assert r2 == "Methane + Water -> Carbon monoxide + 3 Hydrogen", r2
    print("OK reaction stoichiometry compounds canonicalised (coefficients kept)")


def test_list_dedup():
    out, _ = canonicalize_list(["Carbon Monoxide", "carbon monoxide", "Hydrogen"])
    assert out == ["Carbon monoxide", "Hydrogen"], out
    print("OK list canonicalises and de-duplicates:", out)


if __name__ == "__main__":
    test_offender_compounds()
    test_bare_xylene_defaults_with_warning()
    test_reaction_canonicalisation()
    test_list_dedup()
    print("\nALL COMPOUND-NORMALIZE TESTS PASSED")
