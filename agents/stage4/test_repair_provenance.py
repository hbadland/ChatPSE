"""Repair-agent provenance: no silent overwrite + guard re-threading.

After wiring RepairAgent / DeterministicRepair through NodeIR.correct_param, two
properties must hold:

  A. A repair that overrides an existing value RETAGS the origin honestly
     (correct_param → "computed", clears the description sentinel) rather than
     leaving a stale tag — the no-silent-overwrite property extended to repair.

  B. The rule-learning guard (_record_repairs_in_store) still refuses to learn a
     rule from a repair that overrode a DESCRIPTION value. Because repair now
     retags, the *live* tag reads "computed" post-repair; the guard therefore
     decides on the PRE-repair snapshot (params_before). Without the snapshot the
     honest retag would defeat the guard — test_snapshot_is_load_bearing pins that.

Run: PYTHONPATH=. python3.9 agents/stage4/test_repair_provenance.py
"""
import os
os.environ.setdefault("THERMO_TIEBREAK", "deterministic")

from ir.graph import FlowsheetGraph, HeaterNode
from ir.repair import DeterministicRepair
from ir.types import SimError, ErrorTarget, ErrorType, RepairStrategy, ErrorSeverity
from agents.rule_store import FailureRuleStore
from agents.orchestrator_v2 import _record_repairs_in_store


def _heater_graph(tag="H-01", params=None):
    g = FlowsheetGraph()
    g.compounds = ["Acetone", "Water"]
    g.add_unit(HeaterNode(tag=tag, params=dict(params or {})), strict=False)
    return g


def _condition_fix_change(tag="H-01", param="T_out", new=360.0):
    # Matches the CONDITION_FIX regex in _record_repairs_in_store.
    return [f"CONDITION_FIX[flash]: {tag}.{param} 350.0->{new}".replace("->", "→")]


# ── A. no silent overwrite: repair retags honestly ────────────────────────────

def test_repair_retags_honestly_not_silent():
    g = _heater_graph(params={
        "T_out": 25.0,                    # looks like °C (< 100) → triggers conversion
        "_temperature_source": "specified",
        "_desc_T_out": True,
    })
    err = SimError(
        error_type      = ErrorType.UNPHYSICAL_VALUES,
        target          = ErrorTarget.unit("H-01", "T_out"),
        evidence        = "T_out looks like celsius",
        repair_strategy = RepairStrategy.UNIT_CONVERSION,
        severity        = ErrorSeverity.CRITICAL,
    )
    g2, _ = DeterministicRepair().fix_unit_conversions(g, err)
    n = g2.unit("H-01")
    assert n.params["T_out"] == 298.15, n.params["T_out"]
    assert n.params["_temperature_source"] == "computed", n.params      # honest retag
    assert "_desc_T_out" not in n.params, "description sentinel must be cleared"
    print("OK: repair override retags specified→computed + clears sentinel "
          "(no silent overwrite)")


# ── B. guard re-threading: decide on PRE-repair provenance ────────────────────

def test_guard_refuses_learning_from_overridden_description_value():
    # live (post-repair) node: value applied and retagged "computed" by correct_param
    g = _heater_graph(params={"T_out": 360.0, "_temperature_source": "computed"})
    params_before = {"H-01": {"T_out": 350.0,
                              "_temperature_source": "specified",
                              "_desc_T_out": True}}
    store = FailureRuleStore()
    assert store.num_patterns() == 0
    _record_repairs_in_store([], _condition_fix_change(), g, store,
                             compounds=["Acetone", "Water"],
                             params_before=params_before)
    assert store.num_patterns() == 0, "must NOT learn from an overridden description value"
    print("OK: guard reads pre-repair snapshot → refuses to learn from description override")


def test_guard_learns_from_genuine_estimate():
    g = _heater_graph(params={"T_out": 360.0, "_temperature_source": "computed"})
    params_before = {"H-01": {"T_out": 350.0, "_temperature_source": "computed"}}
    store = FailureRuleStore()
    _record_repairs_in_store([], _condition_fix_change(), g, store,
                             compounds=["Acetone", "Water"],
                             params_before=params_before)
    assert store.num_patterns() == 1, "should learn from a genuinely-estimated param"
    print("OK: guard still learns from genuinely-estimated params")


def test_snapshot_is_load_bearing():
    """Without params_before the guard falls back to the live (retagged) node, which
    now reads 'computed' — masking the description origin. Documents WHY the snapshot
    re-threading is required once repair retags honestly."""
    g = _heater_graph(params={"T_out": 360.0, "_temperature_source": "computed"})
    store = FailureRuleStore()
    _record_repairs_in_store([], _condition_fix_change(), g, store,
                             compounds=["Acetone", "Water"], params_before=None)
    assert store.num_patterns() == 1, (
        "without the snapshot the live 'computed' tag masks the description origin — "
        "this is exactly the regression the params_before re-threading prevents")
    print("OK: snapshot is load-bearing (absent → live tag masks origin, would learn)")


def _run_all():
    test_repair_retags_honestly_not_silent()
    test_guard_refuses_learning_from_overridden_description_value()
    test_guard_learns_from_genuine_estimate()
    test_snapshot_is_load_bearing()
    print("\nALL REPAIR PROVENANCE TESTS PASSED")


if __name__ == "__main__":
    _run_all()
