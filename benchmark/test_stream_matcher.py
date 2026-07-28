"""
Tests for cross-tag reference stream matching.

Run: PYTHONPATH=. python3.9 benchmark/test_stream_matcher.py
"""
import json
import math
from benchmark.stream_matcher import match_streams
from benchmark.physics_eval import _do_reference_comparison
from agents.executor import StreamResult


def test_unit_tags_match_process_tags_by_composition():
    """Unit-derived system tags match process-derived reference tags by content."""
    sys = {
        "FEED":    {"T_K": 298.0, "P_Pa": 101325.0, "vapor_fraction": 1.0,
                    "composition": {"Methane": 0.6, "Carbon dioxide": 0.4}, "is_feed": True},
        "WATER-IN":{"T_K": 300.0, "P_Pa": 200000.0, "vapor_fraction": 0.0,
                    "composition": {"Water": 1.0}, "is_feed": True},
        "H2OUT":   {"T_K": 310.0, "P_Pa": 2000000.0, "vapor_fraction": 1.0,
                    "composition": {"Hydrogen": 0.98, "Carbon monoxide": 0.02}, "is_feed": False},
        "INTERIOR":{"T_K": 800.0, "P_Pa": 2000000.0, "vapor_fraction": 1.0,
                    "composition": {"Methane": 0.3, "Hydrogen": 0.4, "Carbon monoxide": 0.3}, "is_feed": False},
    }
    ref = {
        "BIOGAS": {"T_K": 298.15, "P_Pa": 101325.0, "vapor_fraction": 1.0,
                   "composition": {"Methane": 0.6, "Carbon dioxide": 0.4}},
        "WATER":  {"T_K": 298.15, "P_Pa": 200000.0, "vapor_fraction": 0.0,
                   "composition": {"Water": 1.0}},
        "H2":     {"T_K": 305.0,  "P_Pa": 2000000.0, "vapor_fraction": 1.0,
                   "composition": {"Hydrogen": 0.98, "Carbon monoxide": 0.02}},
    }
    m = match_streams(sys, ref)
    pairs = {p["ref_tag"]: p["sys_tag"] for p in m["pairs"]}
    assert pairs.get("BIOGAS") == "FEED", pairs
    assert pairs.get("WATER")  == "WATER-IN", pairs
    assert pairs.get("H2")     == "H2OUT", pairs
    assert m["n_matched"] == 3
    assert m["n_reference_unmatched"] == 0
    assert m["n_system_unmatched"] == 1 and m["system_unmatched"] == ["INTERIOR"]
    assert all(p["confidence"] >= 0.55 for p in m["pairs"])
    print("OK unit-tags↔process-tags matched by composition:",
          {k: (v, next(p['confidence'] for p in m['pairs'] if p['ref_tag']==k)) for k,v in pairs.items()})


def test_conservative_no_forced_match():
    """A reference stream with no compositional counterpart stays UNMATCHED."""
    sys = {"S1": {"T_K": 300, "P_Pa": 1e5, "vapor_fraction": 1.0,
                  "composition": {"Nitrogen": 1.0}, "is_feed": True}}
    ref = {"R1": {"T_K": 300, "P_Pa": 1e5, "vapor_fraction": 1.0,
                  "composition": {"Ammonia": 1.0}}}
    m = match_streams(sys, ref)
    assert m["n_matched"] == 0 and m["n_reference_unmatched"] == 1
    print("OK conservative: incompatible streams left unmatched (no forced match)")


def test_end_to_end_against_real_reference():
    """_do_reference_comparison computes MAPE + emits matching detail on VAL_04 ref."""
    ref_file = "benchmark/reference_flowsheets/VAL_04_reference.json"
    ref = json.load(open(ref_file))
    # Build system streams from every FINITE reference stream (copy composition,
    # perturb T +3 K / P +1%). DWSIM dead-end ports (S-022: flow=-inf, vf=NaN) are
    # skipped as system streams but remain in the reference — so this exercises the
    # non-finite-cost path the matcher must survive.
    def _fin(v):
        return isinstance(v, (int, float)) and math.isfinite(v)
    sys_sr = {}
    i = 0
    for rtag, rs in ref["streams"].items():
        T = rs.get("T_K")
        P = rs.get("P_Pa") or (rs.get("P_bar") or 0) * 1e5
        comp = {k: v for k, v in (rs.get("composition") or {}).items()
                if _fin(v) and v > 0}
        if not _fin(T) or not P or not comp:
            continue
        vf = rs.get("vapor_fraction")
        sys_sr[f"UNITTAG-{i}"] = StreamResult(
            tag=f"UNITTAG-{i}",
            T_K=float(T) + 3.0,
            P_Pa=float(P) * 1.01,
            flow_mol_s=rs.get("flow_mol_s"),
            composition=dict(comp),
            vapor_fraction=vf if _fin(vf) else 0.0,
            is_feed=bool(rs.get("is_feed")))
        i += 1

    class MockPR:
        class final_execution:
            stream_results = sys_sr
    checks, mape_T, mape_P, mape_vf = _do_reference_comparison(ref_file, MockPR())
    detail = next(c for c in checks if c["check"] == "reference_stream_matching")
    # The matcher COMPLETES on a reference containing a DWSIM dead-end port
    # (S-022: flow=-inf, vf=NaN) instead of throwing on the NaN cost, and matches the
    # finite streams (each perturbed stream to its identical-composition source/twin).
    assert detail["n_matched"] >= 3, detail
    assert mape_T > 0.0, mape_T            # +3 K → non-zero T MAPE
    assert any(c["check"] == "reference_match_T" for c in checks)
    print(f"OK end-to-end: n_matched={detail['n_matched']} "
          f"MAPE_T={mape_T}% MAPE_P={mape_P}% MAE_vf={mape_vf}")
    print("   sample matched pair:", detail["matches"][0])


if __name__ == "__main__":
    test_unit_tags_match_process_tags_by_composition()
    test_conservative_no_forced_match()
    test_end_to_end_against_real_reference()
    print("\nALL STREAM-MATCHER TESTS PASSED")
