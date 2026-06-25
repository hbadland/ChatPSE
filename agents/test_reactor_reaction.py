"""
Tests for the ConversionReactor reaction-string fix (and the classifier guard).

Fix #1: a reaction extracted into SemanticUnit must survive to to_dwsim, and
        ParamMapper's estimator must NOT overwrite a populated reaction with "".
Fix #2: a reactor with an empty reaction must classify as INVALID_UNIT_CONFIG
        targeting <tag>.reaction — not a generic downstream-condition fix.

Run: PYTHONPATH=. python3.9 agents/test_reactor_reaction.py
"""
from __future__ import annotations

from agents.stage1.unit_extractor import SemanticUnits, SemanticUnit
from agents.stage1.stream_extractor import SemanticTopology, SemanticStream
from agents.stage2 import GraphBuilder
from agents.stage3.param_mapper import _estimate_params
from agents.stage4.error_classifier import ErrorClassifier
from ir import to_dwsim
from ir.graph import FlowsheetGraph, make_node
from ir.types import ErrorType, RepairStrategy

_passed = _failed = 0


def check(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"  PASS: {msg}")
    else:
        _failed += 1; print(f"  FAIL: {msg}")


_RXN = "Toluene + Hydrogen -> Benzene + Methane"


class _Exec:
    solved = False
    solver_errors: list = []
    errors: list = []
    _critic_report = None


def test_reaction_threads_to_dwsim():
    print("\n[Fix 1] reaction: SemanticUnit -> node.params -> to_dwsim")
    units = SemanticUnits([SemanticUnit("RX-01", "ConversionReactor", "react", _RXN)])
    topo = SemanticTopology([
        SemanticStream("FEED", None, "RX-01", True, T=600.0, P=101325.0, flow=1.0,
                       composition={"Toluene": 0.5, "Hydrogen": 0.5}),
        SemanticStream("PROD", "RX-01", None, False),
    ])
    g = GraphBuilder().build(units, topo, ["Toluene", "Hydrogen", "Benzene", "Methane"])
    check(g.unit("RX-01").params.get("reaction") == _RXN,
          "GraphBuilder threads reaction into node.params")
    g.property_package = "Peng-Robinson"
    d = to_dwsim(g)
    rx = next(u for u in d["units"] if u["tag"] == "RX-01")
    check(rx.get("reaction") == _RXN, f"to_dwsim serialises the reaction (got {rx.get('reaction')!r})")


def test_empty_reaction_not_threaded():
    print("\n[Fix 1] no reaction extracted -> no reaction param (estimator fills later)")
    units = SemanticUnits([SemanticUnit("RX-01", "ConversionReactor", "react", "")])
    topo = SemanticTopology([
        SemanticStream("FEED", None, "RX-01", True, T=600.0, P=101325.0, flow=1.0,
                       composition={"Toluene": 1.0}),
        SemanticStream("PROD", "RX-01", None, False),
    ])
    g = GraphBuilder().build(units, topo, ["Toluene", "Benzene"])
    check("reaction" not in g.unit("RX-01").params,
          "empty reaction is not threaded (ParamMapper estimator owns the default)")


def test_parammapper_preserves_reaction():
    print("\n[Fix 1] ParamMapper estimator preserves a populated reaction")
    node = make_node("ConversionReactor", "RX-01", params={})
    est = _estimate_params(node, {"reaction": _RXN}, None, None, None)
    check("reaction" not in est, "reaction already set -> estimator does NOT overwrite it")
    est2 = _estimate_params(node, {}, None, None, None)
    check(est2.get("reaction") == "", "reaction absent -> estimator fills \"\" (the old bug's source)")


def _graph_with_reactor(reaction: str) -> FlowsheetGraph:
    g = FlowsheetGraph()
    g.compounds = ["Toluene", "Benzene"]
    g.add_unit(make_node("ConversionReactor", "RX-01",
                         params={"reaction": reaction, "conversion": 0.9,
                                 "temperature_K": 598.15, "pressure_Pa": 101325.0}))
    g.add_unit(make_node("Cooler", "CL-01", params={"T_out": 250.0}))
    return g


def test_classifier_targets_reactor_on_empty_reaction():
    print("\n[Fix 2] empty reaction -> INVALID_UNIT_CONFIG on RX-01.reaction (not the cooler)")
    g = _graph_with_reactor("")            # the bug shape
    errs = ErrorClassifier(model="qwen3:14b").classify(_Exec(), g)
    check(len(errs) == 1, f"one error produced (got {len(errs)})")
    e = errs[0]
    check(str(e.target) == "unit:RX-01.reaction",
          f"target is the reactor's reaction (got {e.target})")
    check(e.error_type == ErrorType.INVALID_UNIT_CONFIG,
          "error_type INVALID_UNIT_CONFIG")
    check(e.is_terminal, "terminal (HUMAN) — fail-fast with the real cause, no wasted iterations")
    check("CL-01" not in str(e.target), "does NOT target the downstream cooler")


def test_classifier_skips_when_reaction_present():
    print("\n[Fix 2] reaction present -> reactor guard does NOT fire (normal path)")
    g = _graph_with_reactor(_RXN)
    errs = ErrorClassifier(model="qwen3:14b").classify(_Exec(), g)
    check(not any(str(e.target) == "unit:RX-01.reaction" for e in errs),
          "no spurious reaction error when the reaction is defined")


def main():
    test_reaction_threads_to_dwsim()
    test_empty_reaction_not_threaded()
    test_parammapper_preserves_reaction()
    test_classifier_targets_reactor_on_empty_reaction()
    test_classifier_skips_when_reaction_present()
    print(f"\n{'='*60}\nRESULT: {_passed} passed, {_failed} failed\n{'='*60}")
    raise SystemExit(1 if _failed else 0)


if __name__ == "__main__":
    main()
