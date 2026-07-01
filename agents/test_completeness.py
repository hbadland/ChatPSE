"""
Tests for the completeness-verification loop (deterministic — critic is mocked).

Run: PYTHONPATH=. python3.9 agents/test_completeness.py
"""
import json
import agents.stage1.completeness as C
from agents.stage1.unit_extractor import SemanticUnit, SemanticUnits

DESC = ("Toluene and hydrogen are reacted. The hot effluent is cooled to condense "
        "water, which is knocked out in separators, before the dry gas is purified "
        "in a pressure swing adsorption unit.")


def _units(*specs):
    return SemanticUnits(units=[SemanticUnit(tag=t, type=ty, role=r) for t, ty, r in specs])


def _mock(responses):
    """Return a fake chat() that yields queued JSON responses in order."""
    seq = list(responses)
    def fake_chat(*a, **k):
        return seq.pop(0) if seq else json.dumps({"missing": []})
    return fake_chat


def test_span_grounding_accepts_and_rejects():
    """Grounded spans accepted; hallucinated span + unsupported type rejected."""
    C.chat = _mock([json.dumps({"missing": [
        # accepted — span occurs in DESC
        {"tag": "CL-01", "type": "Cooler", "role": "cool effluent",
         "span": "cooled to condense water"},
        # accepted — span occurs
        {"tag": "V-01", "type": "Vessel", "role": "knockout",
         "span": "knocked out in separators"},
        # REJECTED — domain hallucination, span not in text
        {"tag": "V-99", "type": "Vessel", "role": "amine absorber",
         "span": "absorbed in an amine unit"},
        # REJECTED — unsupported type
        {"tag": "X-01", "type": "MembraneUnit", "role": "membrane",
         "span": "purified in a pressure swing adsorption unit"},
    ]}), json.dumps({"missing": []})])   # iter 1: nothing → stop

    base = _units(("RX-01", "ConversionReactor", "react"))
    out, log = C.run_completeness_loop(DESC, base, model="x")

    assert log["pre_loop_n_units"] == 1
    assert log["post_loop_n_units"] == 3, log["post_loop_n_units"]
    it0 = log["iterations"][0]
    assert len(it0["accepted"]) == 2, it0["accepted"]
    reasons = {r["reject_reason"] for r in it0["rejected"]}
    assert any("span not found" in r for r in reasons), reasons
    assert any("unsupported type" in r for r in reasons), reasons
    assert {u.type for u in out.units} == {"ConversionReactor", "Cooler", "Vessel"}
    print("OK span-grounding: accepted 2, rejected hallucination + unsupported type")


def test_no_inflation_when_all_hallucinated():
    """Regression guard (VAL_09-style over-extraction): only ungrounded claims → no adds."""
    C.chat = _mock([json.dumps({"missing": [
        {"tag": "V-50", "type": "Vessel", "role": "extra column",
         "span": "a third distillation column is added for polishing"},
    ]})])
    base = _units(("V-01", "Vessel", "col1"), ("V-02", "Vessel", "col2"),
                  ("CO-01", "Cooler", "cond"))
    out, log = C.run_completeness_loop(DESC, base, model="x")
    assert log["pre_loop_n_units"] == log["post_loop_n_units"] == 3
    assert len(log["iterations"][0]["accepted"]) == 0
    print("OK no inflation: all-hallucinated claims rejected, count unchanged")


def test_stops_at_max_iters():
    """Never exceeds MAX_ITERS even if the critic keeps returning grounded units."""
    # Always claim the same grounded span; dedup should stop it after 1 accept.
    C.chat = _mock([json.dumps({"missing": [
        {"tag": "CL-01", "type": "Cooler", "role": "cool",
         "span": "cooled to condense water"}]})] * 5)
    base = _units(("RX-01", "ConversionReactor", "react"))
    out, log = C.run_completeness_loop(DESC, base, model="x", max_iters=3)
    assert len(log["iterations"]) <= 3
    # second iteration re-claims same (type,role) → dedup reject → stop
    assert log["post_loop_n_units"] == 2, log["post_loop_n_units"]
    print("OK bounded: dedup + max_iters prevent runaway, final count =", log["post_loop_n_units"])


if __name__ == "__main__":
    test_span_grounding_accepts_and_rejects()
    test_no_inflation_when_all_hallucinated()
    test_stops_at_max_iters()
    print("\nALL COMPLETENESS TESTS PASSED")
