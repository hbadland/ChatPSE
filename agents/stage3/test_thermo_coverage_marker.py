"""Option-B: BIP-coverage MISSING marker — behavioural tests.

Covers the graph-level provenance marker `graph.metadata["thermo_coverage"]`
written by ThermoMapper.assign when a polar/azeotropic system has no NRTL/UNIQUAC
coverage. The marker records the gap WITHOUT changing control flow — the run still
proceeds to a substitute package.

Five assertions:
  1. marker set to "MISSING" (with the uncovered pairs) on a no-coverage system;
  2. marker ABSENT on a covered system and on a near-ideal (non-polar) system;
  3. NO raise by default (THERMO_COVERAGE_GUARD unset) — outcome/control unchanged;
  4. STILL raises ThermoCoverageGuard under THERMO_COVERAGE_GUARD=1 (raise condition
     preserved);
  5. the marker SURVIVES graph.copy().

Assertion 5 is not hygiene: §3.1's composition property (each pipeline stage returns
a deep copy of the graph) only carries provenance if every provenance-bearing field
travels inside the copied object. `thermo_coverage` is a metadata field, so this test
certifies it obeys the same invariant §3.2 proves for the per-unit T/P source tags.

Run: PYTHONPATH=. python3.9 agents/stage3/test_thermo_coverage_marker.py
"""
import os

# Deterministic + offline: skip the LLM tiebreak so ambiguous candidate lists
# resolve to candidates[0] instead of calling out to a model.
os.environ["THERMO_TIEBREAK"] = "deterministic"
os.environ.pop("THERMO_COVERAGE_GUARD", None)

from ir.graph import FlowsheetGraph, Source
from rag.retriever import Retriever, ThermoCoverageGuard
from agents.stage3.thermo_mapper import ThermoMapper

_ALLOWED = {
    "Raoult's Law", "NRTL", "UNIQUAC",
    "Peng-Robinson", "Soave-Redlich-Kwong", "Lee-Kesler-Plöcker",
}

# Candidate polar pairs to search for a genuine corpus gap. 1,4-dioxane is polar
# (ETHERS) and pairs uncovered across the corpus, so at least one should have no
# NRTL/UNIQUAC coverage. Discovered at runtime so the test does not hard-code a
# corpus assumption that a future BIP addition could silently invalidate.
_CANDIDATE_POLAR_PAIRS = [
    ["1,4-dioxane", "1-butanol"], ["1,4-dioxane", "1-propanol"],
    ["1,4-dioxane", "2-propanol"], ["1,4-dioxane", "acetonitrile"],
]
_COVERED_PAIR   = ["Acetone", "Chloroform"]   # polar + in-corpus
_NEAR_IDEAL     = ["Benzene", "Toluene"]       # non-polar → coverage branch not entered

_ret = Retriever()


def _graph(compounds):
    g = FlowsheetGraph()
    g.compounds = list(compounds)
    return g


def _find_uncovered_pair():
    for pair in _CANDIDATE_POLAR_PAIRS:
        hn = _ret.bip.has_full_coverage(pair, "NRTL")
        hu = _ret.bip.has_full_coverage(pair, "UNIQUAC")
        if not hn and not hu:
            return pair
    raise AssertionError(
        "no uncovered polar pair found in the corpus — cannot exercise the MISSING "
        "marker; add a polar pair the BIP corpus does not cover to _CANDIDATE_POLAR_PAIRS")


# Fresh mapper per test; all share the one Retriever so the selector's per-call
# coverage observation is what ThermoMapper reads back.
def _mapper():
    return ThermoMapper(model="offline-unused", retriever=_ret)


def test_vocabulary_has_missing():
    assert Source.MISSING.name == "MISSING"
    assert int(Source.MISSING) == -1          # lowest authority — a recorded gap
    print("OK: Source.MISSING in provenance vocabulary (=-1)")


def test_marker_set_and_no_raise_on_missing_coverage():
    """(1) marker set with pairs, and (3) no raise with the guard off (default)."""
    pair = _find_uncovered_pair()
    try:
        g = _mapper().assign(_graph(pair), description="separate the mixture")
    except ThermoCoverageGuard:
        raise AssertionError("guard raised while disabled — control flow changed!")
    assert g.metadata.get("thermo_coverage") == "MISSING", g.metadata
    assert g.metadata.get("thermo_coverage_missing_pairs"), \
        "expected the uncovered pairs to be recorded"
    assert g.property_package in _ALLOWED, g.property_package
    print(f"OK: MISSING marker set on {pair}, no raise, package="
          f"{g.property_package!r}, pairs={g.metadata['thermo_coverage_missing_pairs']}")


def test_marker_absent_on_covered_and_near_ideal():
    """(2) no false positives."""
    g_cov = _mapper().assign(_graph(_COVERED_PAIR), description="separate")
    assert "thermo_coverage" not in g_cov.metadata, g_cov.metadata
    g_ni = _mapper().assign(_graph(_NEAR_IDEAL), description="distill")
    assert "thermo_coverage" not in g_ni.metadata, g_ni.metadata
    print(f"OK: no marker on covered {_COVERED_PAIR} (pkg={g_cov.property_package!r}) "
          f"or near-ideal {_NEAR_IDEAL} (pkg={g_ni.property_package!r})")


def test_guard_enabled_still_raises():
    """(4) raise condition preserved when the guard is enabled."""
    pair = _find_uncovered_pair()
    os.environ["THERMO_COVERAGE_GUARD"] = "1"
    try:
        raised = False
        try:
            _mapper().assign(_graph(pair), description="separate")
        except ThermoCoverageGuard:
            raised = True
        assert raised, "guard enabled but did not raise on missing coverage"
    finally:
        os.environ.pop("THERMO_COVERAGE_GUARD", None)
    print("OK: THERMO_COVERAGE_GUARD=1 still raises on the same missing case")


def test_marker_survives_copy():
    """(5) provenance travels inside the transformed object — ties to §3.1/§3.2."""
    pair = _find_uncovered_pair()
    g = _mapper().assign(_graph(pair), description="separate")
    assert g.metadata.get("thermo_coverage") == "MISSING"
    gc = g.copy()
    assert gc.metadata.get("thermo_coverage") == "MISSING", \
        "coverage marker lost across graph.copy() — violates the §3.1 composition property"
    assert gc.metadata.get("thermo_coverage_missing_pairs") == \
        g.metadata.get("thermo_coverage_missing_pairs")
    print("OK: MISSING marker survives graph.copy() (composition property holds)")


def _run_all():
    test_vocabulary_has_missing()
    test_marker_set_and_no_raise_on_missing_coverage()
    test_marker_absent_on_covered_and_near_ideal()
    test_guard_enabled_still_raises()
    test_marker_survives_copy()
    print("\nALL THERMO COVERAGE-MARKER TESTS PASSED")


if __name__ == "__main__":
    _run_all()
