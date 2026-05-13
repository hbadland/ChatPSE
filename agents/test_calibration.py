"""
Unit tests for CalibrationAgent.

Tests (no DWSIM required — runs on host):
  1. Pair in corpus → success=True, correct A12/A21/alpha12
  2. Pair not in corpus → success=False, pair in pairs_missing
  3. Pair found but T out of range → success=True + warning in notes
  4. Non-NRTL/UNIQUAC package → success=False immediately
  5. Multi-compound flowsheet — all pairs found → success=True
  6. Multi-compound flowsheet — one pair missing → success=False

Run:
    python3.9 -m pytest agents/test_calibration.py -v
    # or
    python3.9 agents/test_calibration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.calibration import CalibrationAgent, _lookup_pair


def _base_flowsheet(**kwargs) -> dict:
    base = {
        "compounds": ["Ethanol", "Water"],
        "property_package": "NRTL",
        "streams": [{"tag": "FEED", "type": "feed", "T_K": 353.15, "P_Pa": 101325.0}],
    }
    base.update(kwargs)
    return base


def test_pair_in_corpus_success():
    """Ethanol/Water NRTL is in corpus — should succeed with correct parameters."""
    agent = CalibrationAgent()
    result = agent.run(_base_flowsheet())
    assert result.success, f"Expected success=True but got False. Notes: {result.notes}"
    assert ("Ethanol", "Water") in result.pairs_found or ("Water", "Ethanol") in result.pairs_found
    assert len(result.parameters_injected) == 1
    rec = result.parameters_injected[0]
    assert rec.model == "NRTL"
    assert abs(rec.A12 - 586.1) < 1.0, f"Expected A12≈586.1, got {rec.A12}"
    assert abs(rec.A21 - (-195.0)) < 1.0, f"Expected A21≈-195.0, got {rec.A21}"
    assert abs(rec.alpha12 - 0.5765) < 0.01, f"Expected alpha12≈0.5765, got {rec.alpha12}"
    bp = result.updated_flowsheet["binary_parameters"]
    assert len(bp) == 1
    assert bp[0]["model"] == "NRTL"
    assert "alpha12" in bp[0]
    assert "source" in bp[0]
    print(f"  PASS: A12={rec.A12}, A21={rec.A21}, alpha12={rec.alpha12}, source={rec.source!r}")


def test_pair_not_in_corpus_failure():
    """A pair not in the corpus should return success=False."""
    agent = CalibrationAgent()
    # Benzene/Isobutane is not in corpus
    fs = _base_flowsheet(compounds=["Benzene", "Isobutane"])
    result = agent.run(fs)
    assert not result.success, "Expected success=False for pair not in corpus"
    assert len(result.pairs_missing) > 0
    assert "binary_parameters" not in result.updated_flowsheet
    print(f"  PASS: pairs_missing={result.pairs_missing}")


def test_temperature_range_warning():
    """Feed T 10-20% outside fit range → success=True with warning note (mild extrapolation)."""
    agent = CalibrationAgent()
    # Ethanol/Water fit range 333-373 K (range=40 K).
    # T=379 K → overshoot=6 K = 15% → warning only (>10% but <20%).
    fs = _base_flowsheet(
        streams=[{"tag": "FEED", "type": "feed", "T_K": 379.0, "P_Pa": 101325.0}]
    )
    result = agent.run(fs)
    assert result.success, f"Expected success=True for mild extrapolation. Notes: {result.notes}"
    has_warning = any("above" in n.lower() or "below" in n.lower() or "extrapolation" in n.lower()
                      for n in result.notes)
    assert has_warning, f"Expected mild-extrapolation warning in notes but got: {result.notes}"
    print(f"  PASS: warning note: {[n for n in result.notes if 'extrapolation' in n.lower()]}")


def test_temperature_range_hard_block():
    """Feed T >20% outside fit range → success=False (hard block)."""
    agent = CalibrationAgent()
    # Ethanol/Water fit range 333-373 K (range=40 K).
    # T=500 K → overshoot=127 K = 317% → hard block.
    fs = _base_flowsheet(
        streams=[{"tag": "FEED", "type": "feed", "T_K": 500.0, "P_Pa": 101325.0}]
    )
    result = agent.run(fs)
    assert not result.success, f"Expected success=False for >20% extrapolation. Notes: {result.notes}"
    has_block = any("hard block" in n.lower() for n in result.notes)
    assert has_block, f"Expected 'hard block' in notes but got: {result.notes}"
    print(f"  PASS: hard block note: {[n for n in result.notes if 'hard block' in n.lower()][:1]}")


def test_non_activity_package_skipped():
    """Non-NRTL/UNIQUAC packages return success=False immediately."""
    agent = CalibrationAgent()
    fs = _base_flowsheet(property_package="Peng-Robinson")
    result = agent.run(fs)
    assert not result.success
    assert any("not NRTL or UNIQUAC" in n for n in result.notes)
    print(f"  PASS: Peng-Robinson skipped with note: {result.notes[0]!r}")


def test_uniquac_pair_found():
    """Ethanol/Water UNIQUAC is in corpus."""
    agent = CalibrationAgent()
    fs = _base_flowsheet(property_package="UNIQUAC")
    result = agent.run(fs)
    assert result.success, f"Expected success=True for Ethanol/Water UNIQUAC. Notes: {result.notes}"
    rec = result.parameters_injected[0]
    assert rec.model == "UNIQUAC"
    bp = result.updated_flowsheet["binary_parameters"]
    assert "alpha12" not in bp[0], "alpha12 should be omitted for UNIQUAC entries"
    print(f"  PASS: UNIQUAC A12={rec.A12}, A21={rec.A21}, source={rec.source!r}")


def test_multi_compound_all_found():
    """Three-compound system where all pairs are in corpus → success=True."""
    agent = CalibrationAgent()
    # Ethanol/Water and Ethanol/Benzene and Benzene/Water all in corpus
    fs = _base_flowsheet(compounds=["Ethanol", "Water", "Benzene"])
    result = agent.run(fs)
    assert result.success, (
        f"Expected all 3 pairs found. pairs_missing={result.pairs_missing}. "
        f"Notes={result.notes}"
    )
    assert len(result.parameters_injected) == 3
    print(f"  PASS: 3 pairs found: {result.pairs_found}")


def test_multi_compound_one_missing():
    """Three-compound system with one missing pair → success=False."""
    agent = CalibrationAgent()
    # Ethanol/Water ✓, Ethanol/Isobutane ✗ → should fail
    fs = _base_flowsheet(compounds=["Ethanol", "Water", "Isobutane"])
    result = agent.run(fs)
    assert not result.success, "Expected success=False when one pair is missing"
    assert len(result.pairs_missing) >= 1
    print(f"  PASS: pairs_missing={result.pairs_missing}")


def test_methanol_water_nrtl():
    """Methanol/Water NRTL — verify another known pair."""
    agent = CalibrationAgent()
    fs = _base_flowsheet(compounds=["Methanol", "Water"])
    result = agent.run(fs)
    assert result.success, f"Methanol/Water NRTL not found. Notes: {result.notes}"
    rec = result.parameters_injected[0]
    assert abs(rec.A12 - 254.2) < 1.0, f"Expected A12≈254.2, got {rec.A12}"
    print(f"  PASS: Methanol/Water NRTL A12={rec.A12}, A21={rec.A21}")


if __name__ == "__main__":
    tests = [
        ("Pair in corpus → success=True",              test_pair_in_corpus_success),
        ("Pair not in corpus → success=False",          test_pair_not_in_corpus_failure),
        ("T 10-20% out of range → warning only",        test_temperature_range_warning),
        ("T >20% out of range → hard block",            test_temperature_range_hard_block),
        ("Non-NRTL/UNIQUAC skipped",                    test_non_activity_package_skipped),
        ("UNIQUAC pair found",                          test_uniquac_pair_found),
        ("Multi-compound all found → success=True",     test_multi_compound_all_found),
        ("Multi-compound one missing → success=False",  test_multi_compound_one_missing),
        ("Methanol/Water NRTL",                         test_methanol_water_nrtl),
    ]

    passed, failed = 0, []
    for name, fn in tests:
        print(f"\n{'─'*55}")
        print(f"  {name}")
        print(f"{'─'*55}")
        try:
            fn()
            passed += 1
            print("  PASS")
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  FAIL: {exc}")

    print(f"\n{'═'*55}")
    print(f"  {passed}/{len(tests)} passed")
    if failed:
        print("  Failed:")
        for name, msg in failed:
            print(f"    ✗ {name}")
            print(f"      {msg}")
    else:
        print("  All CalibrationAgent tests passed.")
    print(f"{'═'*55}")
