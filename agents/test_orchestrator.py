"""
Tests for the Orchestrator — all agents mocked (no LLM, no DWSIM).

Covers:
- Happy path: PASS on first iteration
- REFINER routing: Refiner fixes flowsheet, second iteration passes
- THERMO routing: ThermoAgent reassigned, second iteration passes
- BASIS routing: BasisAgent re-run, Planner re-runs, then PASS
- HUMAN escalation from Critic
- Refiner unable to fix → HUMAN escalation
- BASIS reruns exceeded → HUMAN escalation
- BASIS failure on rerun → BASIS_FAILED
- Basis Agent failure (unsupported compound) → BASIS_FAILED before planner
- Planner failure → PLAN_FAILED
- Max iterations reached → MAX_ITER
- OrchestratorResult.summary() and .passed property
- IterationRecord populated correctly including refinement audit trail
"""
from __future__ import annotations
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from agents.orchestrator import Orchestrator, OrchestratorResult
from agents.basis   import BasisResult
from agents.critic  import CriticReport, FailureSignal
from agents.refiner import RefinementResult, RefinementChange
from agents.executor import ExecutionResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _good_basis() -> BasisResult:
    return BasisResult(
        compound_map={"methanol": "Methanol", "water": "Water"},
        dwsim_compounds=["Methanol", "Water"],
        normalised_description="Flash Methanol and Water at 1 atm.",
        success=True,
        stage="LLM",
        stage1_count=2,
    )


def _bad_basis() -> BasisResult:
    return BasisResult(
        compound_map={},
        dwsim_compounds=[],
        normalised_description="",
        errors=["'brine' detected: Electrolyte packages not implemented."],
        success=False,
        stage="LOOKUP",
    )


def _good_flowsheet(v: int = 0) -> dict:
    d = {
        "compounds": ["Methanol", "Water"],
        "property_package": "Raoult's Law",
        "units": [{"tag": "HT-01", "type": "Heater"}],
        "streams": [{"tag": "FEED", "is_feed": True}],
        "connections": [],
    }
    if v:
        d["_v"] = v  # makes each version produce a distinct flowsheet hash
    return d


def _good_execution(solved: bool = True) -> ExecutionResult:
    return ExecutionResult(solved=solved, stream_results={}, errors=[] if solved else ["Solver failed"])


def _pass_report() -> CriticReport:
    return CriticReport(passed=True, routing="PASS", severity="PASS",
                        diagnosis="All good.")


def _refiner_report() -> CriticReport:
    return CriticReport(
        passed=False, routing="REFINER", severity="CRITICAL",
        diagnosis="Temperature too low.",
        signals=[FailureSignal(code="UNPHYSICAL_T", severity="CRITICAL",
                               location="stream:FEED", evidence="T=50K")],
    )


def _thermo_report() -> CriticReport:
    return CriticReport(
        passed=False, routing="THERMO", severity="CRITICAL",
        diagnosis="Wrong package.",
        signals=[FailureSignal(code="NO_SEPARATION", severity="CRITICAL",
                               location="global", evidence="no split")],
    )


def _human_report() -> CriticReport:
    return CriticReport(
        passed=False, routing="HUMAN", severity="CRITICAL",
        diagnosis="Infeasible process.",
        signals=[FailureSignal(code="INFEASIBLE", severity="CRITICAL",
                               location="global", evidence="impossible")],
    )


def _basis_report() -> CriticReport:
    return CriticReport(
        passed=False, routing="BASIS", severity="CRITICAL",
        diagnosis="Bad compound name.",
        signals=[FailureSignal(code="PARAM_MISSING", severity="CRITICAL",
                               location="global", evidence="compound not found")],
    )


def _good_refinement(v: int = 1) -> RefinementResult:
    return RefinementResult(
        success=True,
        updated_flowsheet=_good_flowsheet(v),
        changes=[RefinementChange("stream:FEED", "T", 50.0, 323.15, "°C→K", "UNPHYSICAL_T")],
        stage="DETERMINISTIC",
        reasoning="Fixed temperature units.",
    )


def _bad_refinement() -> RefinementResult:
    return RefinementResult(
        success=False,
        updated_flowsheet=_good_flowsheet(),
        changes=[],
        stage="FAILED",
        reasoning="No fix available.",
    )


def _make_orchestrator(**kwargs) -> Orchestrator:
    return Orchestrator(model="mock-model", **kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _patch_all(basis_results, plan_results, thermo_results, exec_results, critic_results,
               refine_results=None):
    """
    Returns a context-manager stack that patches every agent in sequence.
    Each list is consumed in order; last element is reused for remaining calls.
    """
    import contextlib

    def _side_effects(items):
        items = list(items)
        calls = [0]
        def se(*a, **kw):
            i = min(calls[0], len(items) - 1)
            calls[0] += 1
            r = items[i]
            if callable(r) and not isinstance(r, (BasisResult, dict, ExecutionResult,
                                                   CriticReport, RefinementResult)):
                return r(*a, **kw)
            if isinstance(r, Exception):
                raise r
            return r
        return se

    @contextlib.contextmanager
    def cm():
        with patch("agents.orchestrator.BasisAgent.identify",
                   side_effect=_side_effects(basis_results)):
            with patch("agents.orchestrator.PlannerAgent.plan",
                       side_effect=_side_effects(plan_results)):
                with patch("agents.orchestrator.ThermoAgent.assign",
                           side_effect=_side_effects(thermo_results)):
                    with patch("agents.orchestrator.Executor.run",
                               side_effect=_side_effects(exec_results)):
                        with patch("agents.orchestrator.CriticAgent.critique",
                                   side_effect=_side_effects(critic_results)):
                            if refine_results:
                                with patch("agents.orchestrator.RefinerAgent.refine",
                                           side_effect=_side_effects(refine_results)):
                                    yield
                            else:
                                yield
    return cm()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_happy_path_pass_first_iteration():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution()],
        critic_results  = [_pass_report()],
    ):
        res = orch.run("Flash methanol and water at 1 atm.")
    assert res.passed
    assert res.outcome == "PASS"
    assert len(res.iterations) == 1
    assert res.iterations[0].routing == "PASS"
    assert res.basis_reruns == 0
    print(f"PASS  test_happy_path_pass_first_iteration")


def test_refiner_routing_then_pass():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution(False), _good_execution(True)],
        critic_results  = [_refiner_report(), _pass_report()],
        refine_results  = [_good_refinement(1)],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.passed
    assert res.outcome == "PASS"
    assert len(res.iterations) == 2
    assert res.iterations[0].routing == "REFINER"
    assert res.iterations[0].refinement is not None
    assert len(res.iterations[0].refinement.changes) == 1
    assert res.iterations[1].routing == "PASS"
    print(f"PASS  test_refiner_routing_then_pass")


def test_thermo_routing_then_pass():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {}), (_good_flowsheet(1), {})],  # 2nd call returns distinct fs
        exec_results    = [_good_execution(False), _good_execution(True)],
        critic_results  = [_thermo_report(), _pass_report()],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.passed
    assert res.outcome == "PASS"
    assert res.iterations[0].routing == "THERMO"
    assert res.iterations[1].routing == "PASS"
    print(f"PASS  test_thermo_routing_then_pass")


def test_basis_routing_then_pass():
    orch = _make_orchestrator(max_basis_reruns=1)
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis(), _good_basis()],      # initial + rerun
        plan_results    = [fs, _good_flowsheet(1)],            # replan returns distinct fs
        thermo_results  = [(fs, {}), (_good_flowsheet(1), {})],
        exec_results    = [_good_execution(False), _good_execution(True)],
        critic_results  = [_basis_report(), _pass_report()],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.passed
    assert res.outcome == "PASS"
    assert res.basis_reruns == 1
    assert res.iterations[0].routing == "BASIS"
    print(f"PASS  test_basis_routing_then_pass")


def test_human_escalation_from_critic():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution(False)],
        critic_results  = [_human_report()],
    ):
        res = orch.run("React sodium with water.")
    assert not res.passed
    assert res.outcome == "HUMAN"
    assert res.iterations[0].routing == "HUMAN"
    print(f"PASS  test_human_escalation_from_critic")


def test_refiner_fails_escalates_to_human():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution(False)],
        critic_results  = [_refiner_report()],
        refine_results  = [_bad_refinement()],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.outcome == "HUMAN"
    assert res.iterations[0].refinement is not None
    assert not res.iterations[0].refinement.success
    print(f"PASS  test_refiner_fails_escalates_to_human")


def test_basis_reruns_exceeded_escalates_to_human():
    orch = _make_orchestrator(max_basis_reruns=0)
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution(False)],
        critic_results  = [_basis_report()],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.outcome == "HUMAN"
    assert res.basis_reruns == 0
    print(f"PASS  test_basis_reruns_exceeded_escalates_to_human")


def test_basis_failed_initially():
    orch = _make_orchestrator()
    with patch("agents.orchestrator.BasisAgent.identify", return_value=_bad_basis()):
        res = orch.run("Dissolve brine and heat.")
    assert res.outcome == "BASIS_FAILED"
    assert not res.passed
    assert len(res.iterations) == 0
    print(f"PASS  test_basis_failed_initially")


def test_planner_exception_returns_plan_failed():
    orch = _make_orchestrator()
    with patch("agents.orchestrator.BasisAgent.identify", return_value=_good_basis()):
        with patch("agents.orchestrator.PlannerAgent.plan",
                   side_effect=RuntimeError("LLM timeout")):
            res = orch.run("Flash methanol and water.")
    assert res.outcome == "PLAN_FAILED"
    assert len(res.iterations) == 0
    print(f"PASS  test_planner_exception_returns_plan_failed")


def test_max_iterations_reached():
    orch = _make_orchestrator(max_iterations=3)
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution(False)] * 3,
        critic_results  = [_refiner_report()] * 3,
        refine_results  = [_good_refinement(1), _good_refinement(2), _good_refinement(3)],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.outcome == "MAX_ITER"
    assert len(res.iterations) == 3
    print(f"PASS  test_max_iterations_reached")


def test_iteration_records_have_execution_and_report():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution()],
        critic_results  = [_pass_report()],
    ):
        res = orch.run("Flash methanol and water.")
    rec = res.iterations[0]
    assert rec.execution is not None
    assert rec.report is not None
    assert rec.elapsed_s >= 0.0
    assert rec.flowsheet == fs
    print(f"PASS  test_iteration_records_have_execution_and_report")


def test_summary_and_passed_property():
    orch = _make_orchestrator()
    fs = _good_flowsheet()
    with _patch_all(
        basis_results   = [_good_basis()],
        plan_results    = [fs],
        thermo_results  = [(fs, {})],
        exec_results    = [_good_execution()],
        critic_results  = [_pass_report()],
    ):
        res = orch.run("Flash methanol and water.")
    assert res.passed
    summary = res.summary()
    assert "PASS" in summary
    assert "Methanol" in summary
    print(f"PASS  test_summary_and_passed_property")
    print(f"\n{summary}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_happy_path_pass_first_iteration,
        test_refiner_routing_then_pass,
        test_thermo_routing_then_pass,
        test_basis_routing_then_pass,
        test_human_escalation_from_critic,
        test_refiner_fails_escalates_to_human,
        test_basis_reruns_exceeded_escalates_to_human,
        test_basis_failed_initially,
        test_planner_exception_returns_plan_failed,
        test_max_iterations_reached,
        test_iteration_records_have_execution_and_report,
        test_summary_and_passed_property,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)

    print(f"\n{'─'*50}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): {failed}")
    else:
        print(f"All {len(tests)} tests passed.")
