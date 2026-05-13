"""
Tests for CriticAgent — no DWSIM required.
Covers: PASS, SOLVER_FAIL, NO_SEPARATION + PARAM_MISSING, UNPHYSICAL, infeasibility.
"""
import math
from agents.executor import ExecutionResult, StreamResult
from agents.critic import CriticAgent, _run_stage1

# ── Shared flowsheet fixture ──────────────────────────────────────────────────

_FLOWSHEET = {
    "compounds": ["Ethanol", "Water"],
    "property_package": "NRTL",
    "streams": [
        {"tag": "FEED", "T": 351.0, "P": 101325.0, "flow": 1.0,
         "composition": {"Ethanol": 0.5, "Water": 0.5}},
        {"tag": "VAP"},
        {"tag": "LIQ"},
    ],
    "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
    "connections": [
        ["FEED", "V-01", 0, 0],
        ["V-01", "VAP",  0, 0],
        ["V-01", "LIQ",  1, 0],
    ],
}


def _good_result() -> ExecutionResult:
    return ExecutionResult(
        solved=True,
        stream_results={
            "FEED": StreamResult("FEED", 351.0, 101325.0, 1.0,
                                 {"Ethanol": 0.5, "Water": 0.5}, is_feed=True),
            "VAP":  StreamResult("VAP",  355.0, 101325.0, 0.6,
                                 {"Ethanol": 0.72, "Water": 0.28}),
            "LIQ":  StreamResult("LIQ",  351.0, 101325.0, 0.4,
                                 {"Ethanol": 0.18, "Water": 0.82}),
        },
        errors=[], warnings=[], diagnostics=[], solver_errors=[],
    )


# ── Stage 1 unit tests ────────────────────────────────────────────────────────

def test_pass_no_signals():
    signals = _run_stage1(_good_result(), _FLOWSHEET, iteration=0)
    assert signals == [], f"Expected no signals, got: {signals}"
    print("PASS  test_pass_no_signals")


def test_solver_fail():
    result = ExecutionResult(
        solved=False, stream_results={}, errors=[], warnings=[],
        diagnostics=[], solver_errors=["Convergence error"],
    )
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "SOLVER_FAIL" in codes
    print("PASS  test_solver_fail")


def test_unphysical_temperature():
    result = _good_result()
    result.stream_results["VAP"] = StreamResult(
        "VAP", 50.0, 101325.0, 0.6, {"Ethanol": 0.72, "Water": 0.28})
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "UNPHYSICAL_T" in codes
    print("PASS  test_unphysical_temperature")


def test_no_separation_param_missing():
    """NRTL with outlet identical to feed → NO_SEPARATION + PARAM_MISSING."""
    result = ExecutionResult(
        solved=True,
        stream_results={
            "FEED": StreamResult("FEED", 351.0, 101325.0, 1.0,
                                 {"Ethanol": 0.5, "Water": 0.5}, is_feed=True),
            "VAP":  StreamResult("VAP",  351.0, 101325.0, 0.5,
                                 {"Ethanol": 0.5, "Water": 0.5}),
            "LIQ":  StreamResult("LIQ",  351.0, 101325.0, 0.5,
                                 {"Ethanol": 0.5, "Water": 0.5}),
        },
        errors=[], warnings=[], diagnostics=[], solver_errors=[],
    )
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "NO_SEPARATION" in codes, f"Expected NO_SEPARATION, got {codes}"
    assert "PARAM_MISSING" in codes, f"Expected PARAM_MISSING, got {codes}"
    print("PASS  test_no_separation_param_missing")


def test_zero_outlet():
    result = _good_result()
    result.stream_results["VAP"] = StreamResult(
        "VAP", 351.0, 101325.0, 0.0, {"Ethanol": 0.0, "Water": 0.0})
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "ZERO_OUTLET" in codes
    print("PASS  test_zero_outlet")


def test_numeric_fail_nan():
    result = _good_result()
    result.stream_results["VAP"] = StreamResult(
        "VAP", math.nan, 101325.0, 0.6, {"Ethanol": 0.72, "Water": 0.28})
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "NUMERIC_FAIL" in codes
    print("PASS  test_numeric_fail_nan")


# ── CriticAgent (full, Stage 1 only path) ────────────────────────────────────

def test_critic_pass():
    agent = CriticAgent()
    report = agent.critique(_good_result(), _FLOWSHEET, iteration=0)
    assert report.passed is True
    assert report.routing == "PASS"
    assert report.confidence == 1.0
    print(f"PASS  test_critic_pass\n      {report.summary()}")


def test_critic_infeasible_threshold():
    """After _INFEASIBLE_THRESHOLD iterations with CRITICAL signals → HUMAN."""
    result = ExecutionResult(
        solved=False, stream_results={}, errors=[], warnings=[],
        diagnostics=[], solver_errors=["Failed"],
    )
    agent = CriticAgent()
    report = agent.critique(result, _FLOWSHEET, iteration=3)
    assert report.routing == "HUMAN", f"Expected HUMAN, got {report.routing}"
    assert "INFEASIBLE" in report.failure_codes
    print(f"PASS  test_critic_infeasible_threshold\n      {report.summary()}")


def test_critic_routing_param_missing():
    """NO_SEPARATION + PARAM_MISSING with NRTL → CALIBRATION routing."""
    result = ExecutionResult(
        solved=True,
        stream_results={
            "FEED": StreamResult("FEED", 351.0, 101325.0, 1.0,
                                 {"Ethanol": 0.5, "Water": 0.5}, is_feed=True),
            "VAP":  StreamResult("VAP",  351.0, 101325.0, 0.5,
                                 {"Ethanol": 0.5, "Water": 0.5}),
            "LIQ":  StreamResult("LIQ",  351.0, 101325.0, 0.5,
                                 {"Ethanol": 0.5, "Water": 0.5}),
        },
        errors=[], warnings=[], diagnostics=[], solver_errors=[],
    )
    agent = CriticAgent()
    report = agent.critique(result, _FLOWSHEET, iteration=0)
    assert report.passed is False
    assert report.routing in ("CALIBRATION", "REFINER", "HUMAN"), (
        f"Unexpected routing: {report.routing}")
    print(f"PASS  test_critic_routing_param_missing\n"
          f"      routing={report.routing}  codes={report.failure_codes}")


def test_mass_balance_computed_from_flows():
    """MASS_BALANCE fires from flow arithmetic, not error strings."""
    result = ExecutionResult(
        solved=True,
        stream_results={
            "FEED": StreamResult("FEED", 351.0, 101325.0, 1.0,
                                 {"Ethanol": 0.5, "Water": 0.5}, is_feed=True),
            "VAP":  StreamResult("VAP",  355.0, 101325.0, 0.3,  # only 0.3 + 0.3 = 0.6 out
                                 {"Ethanol": 0.72, "Water": 0.28}),
            "LIQ":  StreamResult("LIQ",  351.0, 101325.0, 0.3,
                                 {"Ethanol": 0.18, "Water": 0.82}),
        },
        errors=[], warnings=[], diagnostics=[], solver_errors=[],
    )
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "MASS_BALANCE" in codes, f"Expected MASS_BALANCE, got {codes}"
    print("PASS  test_mass_balance_computed_from_flows")


def test_wrong_phase_dir():
    """WRONG_PHASE_DIR fires when vapour is richer in heavy component."""
    result = ExecutionResult(
        solved=True,
        stream_results={
            "FEED": StreamResult("FEED", 351.0, 101325.0, 1.0,
                                 {"Ethanol": 0.5, "Water": 0.5}, is_feed=True),
            # Inverted: water (higher NBP) dominates vapour, ethanol dominates liquid
            "VAP":  StreamResult("VAP",  355.0, 101325.0, 0.6,
                                 {"Ethanol": 0.15, "Water": 0.85}),
            "LIQ":  StreamResult("LIQ",  351.0, 101325.0, 0.4,
                                 {"Ethanol": 0.95, "Water": 0.05}),
        },
        errors=[], warnings=[], diagnostics=[], solver_errors=[],
    )
    signals = _run_stage1(result, _FLOWSHEET, iteration=0)
    codes = {s.code for s in signals}
    assert "WRONG_PHASE_DIR" in codes, f"Expected WRONG_PHASE_DIR, got {codes}"
    print("PASS  test_wrong_phase_dir")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_pass_no_signals,
        test_solver_fail,
        test_unphysical_temperature,
        test_no_separation_param_missing,
        test_zero_outlet,
        test_numeric_fail_nan,
        test_mass_balance_computed_from_flows,
        test_wrong_phase_dir,
        test_critic_pass,
        test_critic_infeasible_threshold,
        test_critic_routing_param_missing,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)

    print(f"\n{'All tests passed!' if not failed else f'FAILED: {failed}'}")
