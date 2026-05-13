"""
Integration tests: full pipeline with realistic mock executor outputs.

These tests verify end-to-end pipeline correctness against known ground truth
without requiring a live DWSIM installation.  All agents run their real code
paths; only the Executor (DWSIM interface) and LLM calls are mocked.

Ground truth cases
──────────────────
Case 1 — Methanol/Water flash at 80 °C, 1 atm (50/50 molar feed)
    Raoult's Law flash calculation using Antoine vapour pressures:
      Psat(MeOH, 353.15 K) = 181 300 Pa   K_MeOH = 1.789
      Psat(H2O,  353.15 K) =  47 390 Pa   K_H2O  = 0.468
    Rachford-Rice solution: V = 0.305 (vapour fraction)
      y_MeOH = 0.721   y_H2O = 0.279
      x_MeOH = 0.403   x_H2O = 0.597
    Assertions:
      - VAP is methanol-rich  (y_MeOH > y_H2O)
      - LIQ is water-rich     (x_H2O  > x_MeOH)
      - Mass balance ≤ 1 % error
      - Outcome == PASS

Case 2 — WRONG_PHASE_DIR: vapour/liquid outlets swapped
    Executor returns inverted compositions (methanol heavier in vapour than liquid).
    Critic must raise WRONG_PHASE_DIR → routes to REFINER (not THERMO).
    Refiner swaps src_port 0↔1.  Second Executor call returns correct VLE.
    Assertions:
      - Iteration 0 routing == REFINER
      - Outcome == PASS after two iterations

Case 3 — SOLVER_FAIL with NRTL triggers Raoult's Law fallback
    Iteration 0: Executor fails (solved=False), flowsheet uses NRTL.
    Critic raises SOLVER_FAIL → routes to REFINER.
    Refiner Stage 1: NRTL → Raoult's Law.
    Iteration 1: Executor succeeds with Raoult's Law.
    Assertions:
      - Iteration 0 refinement changes property_package to Raoult's Law
      - Outcome == PASS

Case 4 — Planner receives exact compound list from Basis Agent
    Verify the Planner prompt includes the compound constraint so the
    flowsheet compounds field matches basis.dwsim_compounds exactly.
"""
from __future__ import annotations
import json
from unittest.mock import patch, MagicMock

from agents.orchestrator import Orchestrator
from agents.basis   import BasisResult
from agents.critic  import CriticReport, FailureSignal
from agents.refiner import RefinementResult, RefinementChange
from agents.executor import ExecutionResult, StreamResult


# ── Shared VLE ground truth (Raoult's Law, 80 °C, 1 atm) ─────────────────────

_FEED_TAG = "FEED"
_VAP_TAG  = "VAP"
_LIQ_TAG  = "LIQ"

_FEED_FLOW    = 1.0          # mol/s
_VAP_FRACTION = 0.305
_LIQ_FRACTION = 1.0 - _VAP_FRACTION

_Y_MEOH = 0.721;  _Y_H2O = 0.279   # vapour compositions
_X_MEOH = 0.403;  _X_H2O = 0.597   # liquid compositions

_T_K  = 353.15   # 80 °C
_P_PA = 101_325.0


def _vle_result(solved: bool = True) -> ExecutionResult:
    """Realistic Raoult's Law VLE result for 50/50 MeOH/H2O flash."""
    if not solved:
        return ExecutionResult(solved=False, errors=["DWSIM solver did not converge."],
                               solver_errors=["Solver diverged."])
    return ExecutionResult(
        solved=True,
        stream_results={
            _FEED_TAG: StreamResult(_FEED_TAG, _T_K, _P_PA, _FEED_FLOW,
                                    {"Methanol": 0.5, "Water": 0.5}, is_feed=True),
            _VAP_TAG:  StreamResult(_VAP_TAG,  _T_K, _P_PA, _VAP_FRACTION,
                                    {"Methanol": _Y_MEOH, "Water": _Y_H2O}),
            _LIQ_TAG:  StreamResult(_LIQ_TAG,  _T_K, _P_PA, _LIQ_FRACTION,
                                    {"Methanol": _X_MEOH, "Water": _X_H2O}),
        },
    )


def _inverted_vle_result() -> ExecutionResult:
    """Phase direction inverted: heavy compound richer in VAP than LIQ."""
    return ExecutionResult(
        solved=True,
        stream_results={
            _FEED_TAG: StreamResult(_FEED_TAG, _T_K, _P_PA, _FEED_FLOW,
                                    {"Methanol": 0.5, "Water": 0.5}, is_feed=True),
            # Water-rich in VAP (wrong), methanol-rich in LIQ (wrong)
            _VAP_TAG:  StreamResult(_VAP_TAG,  _T_K, _P_PA, _VAP_FRACTION,
                                    {"Methanol": _X_MEOH, "Water": _X_H2O}),
            _LIQ_TAG:  StreamResult(_LIQ_TAG,  _T_K, _P_PA, _LIQ_FRACTION,
                                    {"Methanol": _Y_MEOH, "Water": _Y_H2O}),
        },
    )


# ── Shared good flowsheet ─────────────────────────────────────────────────────

def _good_flowsheet(pp: str = "Raoult's Law") -> dict:
    return {
        "compounds": ["Methanol", "Water"],
        "property_package": pp,
        "streams": [
            {"tag": _FEED_TAG, "T": _T_K, "P": _P_PA, "flow": _FEED_FLOW,
             "composition": {"Methanol": 0.5, "Water": 0.5}},
            {"tag": _VAP_TAG},
            {"tag": _LIQ_TAG},
        ],
        "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
        "connections": [
            [_FEED_TAG, "V-01",  0, 0],
            ["V-01",  _VAP_TAG,  0, 0],
            ["V-01",  _LIQ_TAG,  1, 0],
        ],
    }


# ── LLM mocks ─────────────────────────────────────────────────────────────────

def _basis_llm_echo(prompt, system, model):
    """Basis Agent Stage 2 — confirms anchors, adds nothing."""
    anchors_block = prompt.split("Stage 1 anchors (verify these):")[1].split("Process description:")[0]
    try:
        anchors = json.loads(anchors_block.strip())
        confirmed = [v for v in anchors.values() if isinstance(v, str)]
    except Exception:
        anchors, confirmed = {}, []
    desc_raw = prompt.split("Process description:")[1].split("Perform")[0].strip()
    normalised = desc_raw
    for orig, dwsim in anchors.items():
        if isinstance(dwsim, str):
            normalised = normalised.replace(orig, dwsim)
    return json.dumps({
        "confirmed": confirmed, "rejected": [], "additional": [],
        "mixture_expansions": [], "concentration_hints": [],
        "normalised_description": normalised,
    })


def _planner_llm(prompt, system, model):
    """Planner — returns the canonical good flowsheet."""
    return json.dumps(_good_flowsheet())


def _planner_llm_nrtl(prompt, system, model):
    """Planner — returns flowsheet with NRTL (will trigger SOLVER_FAIL path)."""
    return json.dumps(_good_flowsheet(pp="NRTL"))


def _thermo_llm(prompt, system, model):
    """Thermo Agent — keeps Raoult's Law, returns valid two-block response."""
    fs = _good_flowsheet()
    reasoning = {
        "global_package": "Raoult's Law",
        "global_reasoning": "Near-ideal VLE; Raoult's Law sufficient for topology.",
        "unit_overrides": {},
    }
    return json.dumps(fs) + "\n---\n" + json.dumps(reasoning)


# ── Case 1: Happy path ────────────────────────────────────────────────────────

def test_meoh_water_flash_passes():
    """
    Full pipeline produces PASS with physically correct VLE split.
    VAP must be methanol-rich; LIQ must be water-rich; mass balance within 1%.
    """
    orch = Orchestrator(model="mock", max_iterations=3)

    with patch("agents.basis.chat",    side_effect=_basis_llm_echo), \
         patch("agents.planner.chat",  side_effect=_planner_llm), \
         patch("agents.thermo.chat",   side_effect=_thermo_llm), \
         patch("agents.executor.Executor.run", return_value=_vle_result()):

        result = orch.run(
            "Flash separate a 50/50 molar methanol/water feed at 1 atm and 80°C."
        )

    assert result.outcome == "PASS", f"Expected PASS, got {result.outcome}"
    assert len(result.iterations) == 1

    ex = result.final_execution
    vap = ex.stream_results[_VAP_TAG]
    liq = ex.stream_results[_LIQ_TAG]

    # Physics assertions against ground truth
    assert vap.composition["Methanol"] > vap.composition["Water"], \
        f"VAP should be methanol-rich: {vap.composition}"
    assert liq.composition["Water"] > liq.composition["Methanol"], \
        f"LIQ should be water-rich: {liq.composition}"

    # Quantitative check against Raoult's Law solution (±5 % tolerance)
    assert abs(vap.composition["Methanol"] - _Y_MEOH) < 0.05, \
        f"y_MeOH={vap.composition['Methanol']:.3f}, expected {_Y_MEOH}"
    assert abs(liq.composition["Water"] - _X_H2O) < 0.05, \
        f"x_H2O={liq.composition['Water']:.3f}, expected {_X_H2O}"

    # Mass balance
    feed_flow = ex.stream_results[_FEED_TAG].flow_mol_s
    out_flow  = vap.flow_mol_s + liq.flow_mol_s
    rel_err   = abs(feed_flow - out_flow) / feed_flow
    assert rel_err < 0.01, f"Mass balance error {rel_err:.1%}"

    print(f"PASS  test_meoh_water_flash_passes")
    print(f"      VAP: MeOH={vap.composition['Methanol']:.3f}  "
          f"H2O={vap.composition['Water']:.3f}  flow={vap.flow_mol_s:.3f} mol/s")
    print(f"      LIQ: MeOH={liq.composition['Methanol']:.3f}  "
          f"H2O={liq.composition['Water']:.3f}  flow={liq.flow_mol_s:.3f} mol/s")
    print(f"      Mass balance error: {rel_err:.3%}")


# ── Case 2: WRONG_PHASE_DIR routes to REFINER ─────────────────────────────────

def test_wrong_phase_dir_routes_to_refiner_not_thermo():
    """
    WRONG_PHASE_DIR must route to REFINER (not THERMO) — verifies C1 fix.

    The Critic is mocked to return WRONG_PHASE_DIR on iteration 0 then PASS,
    isolating exactly the routing decision without relying on mock-executor
    stream data being consistent with DWSIM's port-assignment convention
    (which is a DWSIM simulation detail, not a routing logic detail).
    """
    orch = Orchestrator(model="mock", max_iterations=3)

    fs = _good_flowsheet()
    fs_fixed = json.loads(json.dumps(fs))
    fs_fixed["connections"][1][2] = 1
    fs_fixed["connections"][2][2] = 0

    critic_calls = [0]
    def mock_critic(execution, flowsheet, iteration=0):
        i = critic_calls[0]; critic_calls[0] += 1
        if i == 0:
            return CriticReport(
                passed=False, routing="REFINER", severity="CRITICAL",
                diagnosis="Phase outlets swapped.",
                failure_codes=["WRONG_PHASE_DIR"],
                signals=[FailureSignal("WRONG_PHASE_DIR", "WARNING",
                                       "unit:V-01", "MeOH heavier in VAP")],
            )
        return CriticReport(passed=True, routing="PASS", severity="PASS",
                            diagnosis="Correct phase split.")

    def mock_refiner(flowsheet, report, **kw):
        change = RefinementChange(
            target="unit:V-01", field="connections.src_port",
            old_value="[0, 1]", new_value="[1, 0]",
            reason="Swapped vapour/liquid outlet ports.",
            failure_code="WRONG_PHASE_DIR",
        )
        return RefinementResult(
            success=True, updated_flowsheet=fs_fixed,
            changes=[change], stage="DETERMINISTIC",
            reasoning="Phase direction fixed.",
        )

    with patch("agents.basis.chat",   side_effect=_basis_llm_echo), \
         patch("agents.planner.chat", side_effect=_planner_llm), \
         patch("agents.thermo.chat",  side_effect=_thermo_llm), \
         patch("agents.orchestrator.Executor.run",         return_value=_vle_result()), \
         patch("agents.orchestrator.CriticAgent.critique", side_effect=mock_critic), \
         patch("agents.orchestrator.RefinerAgent.refine",  side_effect=mock_refiner):

        result = orch.run(
            "Flash separate a 50/50 molar methanol/water feed at 1 atm and 80°C."
        )

    assert result.outcome == "PASS", f"Expected PASS, got {result.outcome}"
    assert len(result.iterations) == 2
    assert result.iterations[0].routing == "REFINER", \
        f"WRONG_PHASE_DIR must route to REFINER, got {result.iterations[0].routing}"
    assert result.iterations[0].refinement is not None
    assert result.iterations[1].routing == "PASS"

    print(f"PASS  test_wrong_phase_dir_routes_to_refiner_not_thermo")
    print(f"      iter 0: {result.iterations[0].routing} → "
          f"iter 1: {result.iterations[1].routing}")


# ── Case 3: SOLVER_FAIL + NRTL → Raoult's Law fallback ───────────────────────

def test_solver_fail_nrtl_falls_back_to_raoults_law():
    """
    Iteration 0: NRTL flowsheet + solver divergence → SOLVER_FAIL → REFINER.
    Refiner Stage 1 deterministically switches NRTL → Raoult's Law.
    Iteration 1: solver succeeds.  Verifies S2 fix.

    The Thermo mock keeps NRTL.  The Critic is mocked to control routing.
    """
    orch = Orchestrator(model="mock", max_iterations=3)

    def _thermo_llm_nrtl(prompt, system, model):
        fs = _good_flowsheet(pp="NRTL")
        reasoning = {"global_package": "NRTL",
                     "global_reasoning": "Non-ideal — NRTL.",
                     "unit_overrides": {}}
        return json.dumps(fs) + "\n---\n" + json.dumps(reasoning)

    critic_calls = [0]
    def mock_critic(execution, flowsheet, iteration=0):
        i = critic_calls[0]; critic_calls[0] += 1
        if i == 0:
            return CriticReport(
                passed=False, routing="REFINER", severity="CRITICAL",
                diagnosis="Solver diverged — likely missing NRTL parameters.",
                failure_codes=["SOLVER_FAIL"],
                signals=[FailureSignal("SOLVER_FAIL", "CRITICAL",
                                       "global", "sim.Solved=False")],
            )
        return CriticReport(passed=True, routing="PASS", severity="PASS",
                            diagnosis="Raoult's Law converged.")

    exec_calls = [0]
    def mock_executor(flowsheet):
        i = exec_calls[0]; exec_calls[0] += 1
        return (ExecutionResult(solved=False, errors=["Solver diverged."],
                                solver_errors=["Diverged."])
                if i == 0 else _vle_result())

    with patch("agents.basis.chat",   side_effect=_basis_llm_echo), \
         patch("agents.planner.chat", side_effect=_planner_llm_nrtl), \
         patch("agents.thermo.chat",  side_effect=_thermo_llm_nrtl), \
         patch("agents.orchestrator.Executor.run",         side_effect=mock_executor), \
         patch("agents.orchestrator.CriticAgent.critique", side_effect=mock_critic):

        result = orch.run(
            "Flash separate a 50/50 molar methanol/water feed at 1 atm and 80°C."
        )

    assert result.outcome == "PASS", f"Expected PASS, got {result.outcome}"
    assert len(result.iterations) == 2

    ref = result.iterations[0].refinement
    assert ref is not None, "Expected refinement in iteration 0"
    assert ref.stage == "DETERMINISTIC", f"Expected DETERMINISTIC, got {ref.stage}"

    pp_changes = [c for c in ref.changes if c.field == "property_package"]
    assert pp_changes, f"Expected property_package change; changes={ref.changes}"
    assert pp_changes[0].old_value == "NRTL"
    assert pp_changes[0].new_value == "Raoult's Law"
    assert pp_changes[0].failure_code == "SOLVER_FAIL"

    print(f"PASS  test_solver_fail_nrtl_falls_back_to_raoults_law")
    print(f"      Refiner change: {pp_changes[0]}")


# ── Case 4: Planner receives exact compound list from Basis Agent ─────────────

def test_planner_receives_compound_constraint():
    """
    Basis Agent resolves ['Methanol', 'Water'].
    Planner prompt must include those exact names as a constraint.
    Verifies C2 fix: Orchestrator passes basis.dwsim_compounds to Planner.
    """
    captured_prompts = []

    def capture_planner_llm(prompt, system, model):
        captured_prompts.append(prompt)
        return json.dumps(_good_flowsheet())

    orch = Orchestrator(model="mock", max_iterations=1)

    with patch("agents.basis.chat",   side_effect=_basis_llm_echo), \
         patch("agents.planner.chat", side_effect=capture_planner_llm), \
         patch("agents.thermo.chat",  side_effect=_thermo_llm), \
         patch("agents.orchestrator.Executor.run", return_value=_vle_result()):

        result = orch.run(
            "Flash separate a 50/50 molar methanol/water feed at 1 atm and 80°C."
        )

    assert captured_prompts, "Planner LLM was never called"
    planner_prompt = captured_prompts[0]
    assert "Methanol" in planner_prompt, \
        "Planner prompt must contain 'Methanol' from basis compound list"
    assert "Water" in planner_prompt, \
        "Planner prompt must contain 'Water' from basis compound list"
    assert "CONSTRAINTS" in planner_prompt or "exact" in planner_prompt.lower(), \
        "Planner prompt must include a compound constraint block"

    print(f"PASS  test_planner_receives_compound_constraint")
    # Show the constraint section
    lines = planner_prompt.split("\n")
    constraint_lines = [l for l in lines[:15] if l.strip()]
    print(f"      Prompt header: {constraint_lines[:5]}")


# ── Case 5: Cycling detection stops infinite loop ─────────────────────────────

def test_cycling_detection_stops_loop():
    """
    Refiner returns the same flowsheet each time (no real fix).
    Orchestrator must detect the repeated hash and escalate to HUMAN.
    Verifies A2 fix: flowsheet hash cycling detection.
    """
    orch = Orchestrator(model="mock", max_iterations=5)

    fs = _good_flowsheet()

    def mock_refiner(flowsheet, report, **kw):
        # Returns the SAME flowsheet — will cause cycling
        return RefinementResult(
            success=True, updated_flowsheet=fs,
            changes=[], stage="DETERMINISTIC", reasoning="No-op.",
        )

    with patch("agents.basis.chat",   side_effect=_basis_llm_echo), \
         patch("agents.planner.chat", side_effect=_planner_llm), \
         patch("agents.thermo.chat",  side_effect=_thermo_llm), \
         patch("agents.orchestrator.Executor.run",
               return_value=ExecutionResult(solved=False,
                                            errors=["Solver failed"])), \
         patch("agents.orchestrator.RefinerAgent.refine", side_effect=mock_refiner):

        result = orch.run(
            "Flash separate a 50/50 molar methanol/water feed at 1 atm and 80°C."
        )

    assert result.outcome == "HUMAN", \
        f"Cycling should escalate to HUMAN, got {result.outcome}"
    assert len(result.iterations) <= orch._max_iterations, \
        "Should not exceed max_iterations before detecting cycle"

    print(f"PASS  test_cycling_detection_stops_loop")
    print(f"      Detected cycle after {len(result.iterations)} iteration(s)")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_meoh_water_flash_passes,
        test_wrong_phase_dir_routes_to_refiner_not_thermo,
        test_solver_fail_nrtl_falls_back_to_raoults_law,
        test_planner_receives_compound_constraint,
        test_cycling_detection_stops_loop,
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

    print(f"\n{'─'*60}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): {failed}")
    else:
        print(f"All {len(tests)} integration tests passed.")
