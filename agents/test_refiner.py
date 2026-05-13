"""
Tests for RefinerAgent — no DWSIM required, no LLM calls for Stage 1 tests.
"""
import copy
from agents.critic import CriticReport, FailureSignal
from agents.refiner import (
    RefinerAgent, RefinementResult,
    _apply_deterministic_fixes,
    _fix_temperatures, _fix_pressures, _fix_compositions,
    _fix_phase_direction,
)

# ── Base flowsheet ────────────────────────────────────────────────────────────

_FLOWSHEET = {
    "compounds": ["Ethanol", "Water"],
    "property_package": "NRTL",
    "streams": [
        {"tag": "FEED", "T": 78.0, "P": 1.013, "flow": 1.0,
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


def _report(codes, signals=None) -> CriticReport:
    return CriticReport(
        passed=False,
        routing="REFINER",
        failure_codes=codes,
        severity="CRITICAL",
        diagnosis="Test diagnosis.",
        suggested_fixes=[],
        confidence=0.8,
        iteration=0,
        signals=signals or [],
    )


# ── Temperature fix ───────────────────────────────────────────────────────────

def test_fix_temperature_celsius_to_kelvin():
    fs = copy.deepcopy(_FLOWSHEET)
    fixed, changes = _fix_temperatures(fs)
    assert fixed["streams"][0]["T"] == 78.0 + 273.15
    assert len(changes) == 1
    assert changes[0].field == "T"
    assert changes[0].old_value == 78.0
    print("PASS  test_fix_temperature_celsius_to_kelvin")


def test_fix_temperature_unit_with_t_out():
    fs = copy.deepcopy(_FLOWSHEET)
    fs["units"] = [{"tag": "HT-01", "type": "Heater", "T_out": 80.0, "dP": 0.0}]
    fixed, changes = _fix_temperatures(fs)
    assert fixed["units"][0]["T_out"] == 80.0 + 273.15
    assert len(changes) == 2  # stream FEED + unit HT-01
    print("PASS  test_fix_temperature_unit_with_t_out")


def test_fix_temperature_already_kelvin():
    """Temperatures already in K should not be changed."""
    fs = copy.deepcopy(_FLOWSHEET)
    fs["streams"][0]["T"] = 351.0
    fixed, changes = _fix_temperatures(fs)
    assert fixed["streams"][0]["T"] == 351.0
    assert len(changes) == 0
    print("PASS  test_fix_temperature_already_kelvin")


# ── Pressure fix ──────────────────────────────────────────────────────────────

def test_fix_pressure_bar_to_pa():
    fs = copy.deepcopy(_FLOWSHEET)
    fixed, changes = _fix_pressures(fs)
    # Use tolerance — 1.013 * 1e5 has floating-point representation error
    assert abs(fixed["streams"][0]["P"] - 1.013 * 1e5) < 1.0
    assert len(changes) == 1
    assert "bar" in changes[0].reason
    print("PASS  test_fix_pressure_bar_to_pa")


def test_fix_pressure_kpa_to_pa():
    fs = copy.deepcopy(_FLOWSHEET)
    fs["streams"][0]["P"] = 500.0   # 500 kPa
    fixed, changes = _fix_pressures(fs)
    assert fixed["streams"][0]["P"] == 500_000.0
    assert "kPa" in changes[0].reason
    print("PASS  test_fix_pressure_kpa_to_pa")


def test_fix_pressure_already_pa():
    """Pressures already in Pa should not be changed."""
    fs = copy.deepcopy(_FLOWSHEET)
    fs["streams"][0]["P"] = 101325.0
    fixed, changes = _fix_pressures(fs)
    assert fixed["streams"][0]["P"] == 101325.0
    assert len(changes) == 0
    print("PASS  test_fix_pressure_already_pa")


# ── Composition fix ───────────────────────────────────────────────────────────

def test_fix_composition_sum():
    fs = copy.deepcopy(_FLOWSHEET)
    fs["streams"][0]["composition"] = {"Ethanol": 0.6, "Water": 0.6}  # sums to 1.2
    fixed, changes = _fix_compositions(fs)
    comp = fixed["streams"][0]["composition"]
    assert abs(sum(comp.values()) - 1.0) < 0.01
    assert abs(comp["Ethanol"] - 0.5) < 0.01
    print("PASS  test_fix_composition_sum")


# ── PARAM_MISSING → Raoult's Law ──────────────────────────────────────────────

def test_param_missing_falls_back_to_raoults():
    report = _report(["PARAM_MISSING", "NO_SEPARATION"])
    # n-Hexane/n-Heptane: near-ideal hydrocarbons — Raoult's Law is physically valid
    # (no azeotrope, no LLE), so physics_validate passes and the fallback applies.
    fs = {
        "compounds": ["n-Hexane", "n-Heptane"],
        "property_package": "NRTL",
        "streams": [
            {"tag": "FEED", "T": 351.0, "P": 101325.0, "flow": 1.0,
             "composition": {"n-Hexane": 0.5, "n-Heptane": 0.5}},
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
    fixed, changes = _apply_deterministic_fixes(fs, report)
    assert fixed["property_package"] == "Raoult's Law"
    assert any(c.field == "property_package" for c in changes)
    print("PASS  test_param_missing_falls_back_to_raoults")


# ── Phase direction fix ───────────────────────────────────────────────────────

def test_fix_phase_direction_swaps_ports():
    report = _report(
        ["WRONG_PHASE_DIR"],
        signals=[FailureSignal(
            code="WRONG_PHASE_DIR", severity="WARNING",
            location="unit:V-01", evidence="water in vap > liq")],
    )
    fs = copy.deepcopy(_FLOWSHEET)
    # Before: V-01 → VAP at port 0, V-01 → LIQ at port 1
    fixed, changes = _fix_phase_direction(fs, report)
    # After: ports should be swapped
    vap_conn = next(c for c in fixed["connections"] if c[1] == "VAP")
    liq_conn = next(c for c in fixed["connections"] if c[1] == "LIQ")
    assert vap_conn[2] == 1  # was 0, now 1
    assert liq_conn[2] == 0  # was 1, now 0
    assert len(changes) == 1
    print("PASS  test_fix_phase_direction_swaps_ports")


# ── Full deterministic pipeline ───────────────────────────────────────────────

def test_deterministic_t_and_p_and_comp():
    """Combined T, P, composition fixes in one pass."""
    report = _report(["UNPHYSICAL_T", "UNPHYSICAL_P", "COMP_SUM"])
    fs = {
        "compounds": ["Ethanol", "Water"],
        "property_package": "Raoult's Law",
        "streams": [
            {"tag": "FEED", "T": 78.0, "P": 1.013,
             "composition": {"Ethanol": 0.6, "Water": 0.6}},
        ],
        "units": [],
        "connections": [],
    }
    fixed, changes = _apply_deterministic_fixes(fs, report)
    assert abs(fixed["streams"][0]["T"] - 351.15) < 0.1
    assert abs(fixed["streams"][0]["P"] - 101300.0) < 1.0
    comp_sum = sum(fixed["streams"][0]["composition"].values())
    assert abs(comp_sum - 1.0) < 0.01
    assert len(changes) == 3
    print("PASS  test_deterministic_t_and_p_and_comp")


# ── RefinerAgent Stage 1 (no LLM) ────────────────────────────────────────────

def test_refiner_agent_deterministic_param_missing():
    """Full RefinerAgent call resolves PARAM_MISSING deterministically."""
    report = _report(["PARAM_MISSING", "NO_SEPARATION"])
    # n-Hexane/n-Heptane: near-ideal hydrocarbons — Raoult's Law passes physics
    # validation (no azeotrope, no LLE), so deterministic fallback is applied.
    flowsheet = {
        "compounds": ["n-Hexane", "n-Heptane"],
        "property_package": "NRTL",
        "streams": [
            {"tag": "FEED", "T": 351.0, "P": 101325.0, "flow": 1.0,
             "composition": {"n-Hexane": 0.5, "n-Heptane": 0.5}},
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
    agent = RefinerAgent()
    result = agent.refine(flowsheet, report)
    assert result.success
    assert result.stage == "DETERMINISTIC"
    assert result.updated_flowsheet["property_package"] == "Raoult's Law"
    print(f"PASS  test_refiner_agent_deterministic_param_missing")
    print(f"      {result.summary()}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_fix_temperature_celsius_to_kelvin,
        test_fix_temperature_unit_with_t_out,
        test_fix_temperature_already_kelvin,
        test_fix_pressure_bar_to_pa,
        test_fix_pressure_kpa_to_pa,
        test_fix_pressure_already_pa,
        test_fix_composition_sum,
        test_param_missing_falls_back_to_raoults,
        test_fix_phase_direction_swaps_ports,
        test_deterministic_t_and_p_and_comp,
        test_refiner_agent_deterministic_param_missing,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)

    print(f"\n{'All tests passed!' if not failed else f'FAILED: {failed}'}")
