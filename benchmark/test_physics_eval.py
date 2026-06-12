"""
Tests for benchmark/physics_eval.py — Task 4.

Runs without DWSIM or networkx. All graph/execution objects are duck-typed mocks.

Three test scenarios
────────────────────
1. known_correct   — Heater + Vessel (benzene/toluene heat+flash at 100°C).
                     All checks should pass. Verifies the happy path and
                     that CRITICAL execution-grounded checks fire correctly.

2. known_incorrect — Heater whose outlet is cooler than its inlet (T_out < T_in).
                     temp_increases_across must fail at CRITICAL severity.
                     mass_balance must still pass (flow is conserved).

3. borderline      — Flash vessel at the bubble point of ethanol/water.
                     The vapour outlet carries only 0.09% of feed flow —
                     below the 0.1% threshold used by flash_vapor_fraction.
                     Expected outcome:
                       two_phase_outlet   → PASS (VF thresholds met)
                       flash_vapor_fraction → FAIL (flow < threshold)
                     This documents the intentional behaviour: a near-bubble-point
                     flash is flagged by the stricter flow-fraction check.

Audit table (from module docstring, reproduced here for quick reference)
────────────────────────────────────────────────────────────────────────
check                      | source      | severity | FP notes
unit_type_present          | IR          | WARNING  | safe
n_units_of_type            | IR          | WARNING  | safe
property_package_class     | IR          | WARNING  | safe
temp_increases_across      | exec → IR   | CRITICAL | exec path verifies simulation
temp_decreases_across      | exec → IR   | CRITICAL | exec path verifies simulation
pressure_increases_across  | exec → IR   | CRITICAL | fixed: compares vs actual P_in
outlet_t_range             | exec → IR   | WARNING  | exec avoids consistency-pass drift
two_phase_outlet           | exec → IR   | CRITICAL | fixed: vessel outlets only
single_phase_vapor_ok      | —           | INFO     | trivial pass by design
bip_injected               | IR          | CRITICAL | safe
separation_quality_below   | exec → skip | WARNING  | requires execution data
temp_consistency_inlet_out | exec → IR   | CRITICAL | plausibility guard (> 200 K)
convergence                | pipeline    | CRITICAL | safe
mass_balance               | execution   | CRITICAL | NEW — flow conservation
flash_vapor_fraction       | execution   | CRITICAL | NEW — degenerate flash detection
energy_balance_heater      | execution   | WARNING  | NEW — ΔT sanity check
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.physics_eval import evaluate_check, run_physics_checks, CheckSeverity
from benchmark.case_schema import PhysicsCheck


# ── Mock primitives ────────────────────────────────────────────────────────────

class _Stream:
    def __init__(self, tag, T_K, P_Pa, flow_mol_s, vapor_fraction=0.0,
                 composition=None):
        self.tag            = tag
        self.T_K            = T_K
        self.P_Pa           = P_Pa
        self.flow_mol_s     = flow_mol_s
        self.vapor_fraction = vapor_fraction
        self.composition    = composition or {}


class _Execution:
    def __init__(self, stream_results: dict, solved: bool = True):
        self.stream_results = stream_results
        self.solved         = solved


class _Unit:
    def __init__(self, tag: str, unit_type: str, params: dict = None):
        self.tag       = tag
        self.unit_type = unit_type
        self.params    = params or {}


class _EdgeIR:
    """Minimal EdgeIR duck-type (tag only; T/P read from IR topology helpers)."""
    def __init__(self, tag: str, T=None, P=None):
        self.tag = tag
        self.T   = T
        self.P   = P


class _Graph:
    """
    Duck-typed FlowsheetGraph.

    topology maps unit_tag → {"inlets": [stream_tags], "outlets": [stream_tags]}.
    Any stream not listed as a unit's outlet has no upstream unit (= feed stream).
    Any stream not listed as a unit's inlet  has no downstream unit (= terminal).
    """

    def __init__(self, units: list[_Unit], topology: dict,
                 property_package: str = "NRTL",
                 binary_parameters: list = None):
        self._units              = units
        self._topology           = topology          # {unit_tag: {inlets, outlets}}
        self.property_package    = property_package
        self.binary_parameters   = binary_parameters or []

    # FlowsheetGraph API subset needed by physics_eval

    def units(self):
        return self._units

    def streams(self):
        seen, result = set(), []
        for data in self._topology.values():
            for tag in data.get("inlets", []) + data.get("outlets", []):
                if tag not in seen:
                    seen.add(tag)
                    result.append(_EdgeIR(tag))
        return result

    def stream(self, tag: str):
        return _EdgeIR(tag)

    def inlet_streams(self, unit_tag: str):
        return [_EdgeIR(t) for t in self._topology.get(unit_tag, {}).get("inlets", [])]

    def outlet_streams(self, unit_tag: str):
        return [_EdgeIR(t) for t in self._topology.get(unit_tag, {}).get("outlets", [])]

    def stream_source(self, stream_tag: str):
        """Return the unit whose outlets list contains stream_tag, else None."""
        for unit_tag, data in self._topology.items():
            if stream_tag in data.get("outlets", []):
                return unit_tag
        return None

    def stream_dest(self, stream_tag: str):
        """Return the unit whose inlets list contains stream_tag, else None."""
        for unit_tag, data in self._topology.items():
            if stream_tag in data.get("inlets", []):
                return unit_tag
        return None


class _PipelineResult:
    def __init__(self, graph, execution=None, outcome="PASS"):
        self.final_graph     = graph
        self.final_execution = execution
        self.outcome         = outcome
        self.converged       = (outcome == "PASS")


class _CaseSpec:
    class _Expected:
        def __init__(self, checks):
            self.physics_checks = checks
    def __init__(self, checks):
        self.expected = self._Expected(checks)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _chk(check_type: str, **params) -> PhysicsCheck:
    return PhysicsCheck(check_type=check_type, params=params)


def _run(graph, execution, checks):
    pr = _PipelineResult(graph, execution)
    results = []
    for c in checks:
        results.append(evaluate_check(c, graph, pr))
    return results


def _assert(result, passed: bool, severity: str = None, label: str = ""):
    tag = f"[{label}] " if label else ""
    assert result["passed"] == passed, (
        f"{tag}check={result['check']!r}: expected passed={passed}, "
        f"got {result['passed']}. detail={result['detail']!r}")
    if severity is not None:
        assert result["severity"] == severity, (
            f"{tag}check={result['check']!r}: expected severity={severity!r}, "
            f"got {result['severity']!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — known-correct: Heater + Vessel (benzene/toluene, heat to 100°C + flash)
#
# Flowsheet:
#   FEED (298 K, 1 atm, equimolar BT) → HT-01 → HT-OUT (373 K) → VES-01
#                                                                   ├─ VAP-OUT (VF=1.0, 42% flow)
#                                                                   └─ LIQ-OUT (VF=0.0, 58% flow)
#
# All checks should pass.  Mass balance: 1.0 in, 0.42+0.58 = 1.00 out.
# ══════════════════════════════════════════════════════════════════════════════

def test_known_correct():
    units = [
        _Unit("HT-01",  "Heater",  {"T_out": 373.15}),
        _Unit("VES-01", "Vessel",  {}),
    ]
    topology = {
        "HT-01":  {"inlets": ["FEED"],   "outlets": ["HT-OUT"]},
        "VES-01": {"inlets": ["HT-OUT"], "outlets": ["VAP-OUT", "LIQ-OUT"]},
    }
    graph = _Graph(units, topology,
                   property_package="NRTL",
                   binary_parameters=[{"model": "NRTL", "compound_a": "Benzene",
                                        "compound_b": "Toluene",
                                        "A12": 0.0, "A21": 0.0, "alpha12": 0.3}])

    streams = {
        "FEED":    _Stream("FEED",    T_K=298.15, P_Pa=101325, flow_mol_s=1.0,
                           vapor_fraction=0.0,
                           composition={"Benzene": 0.5, "Toluene": 0.5}),
        "HT-OUT":  _Stream("HT-OUT",  T_K=373.15, P_Pa=101325, flow_mol_s=1.0,
                           vapor_fraction=0.42,
                           composition={"Benzene": 0.5, "Toluene": 0.5}),
        "VAP-OUT": _Stream("VAP-OUT", T_K=373.15, P_Pa=101325, flow_mol_s=0.42,
                           vapor_fraction=1.0,
                           composition={"Benzene": 0.60, "Toluene": 0.40}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=373.15, P_Pa=101325, flow_mol_s=0.58,
                           vapor_fraction=0.0,
                           composition={"Benzene": 0.42, "Toluene": 0.58}),
    }
    execution = _Execution(streams)

    checks = [
        _chk("unit_type_present",      unit_type="Heater"),
        _chk("unit_type_present",      unit_type="Vessel"),
        _chk("temp_increases_across",  unit_tag_pattern="HT"),
        _chk("two_phase_outlet",       unit_tag_pattern="VES"),
        _chk("outlet_t_range",         unit_tag_pattern="HT", T_min_K=355, T_max_K=395),
        _chk("bip_injected",           model="NRTL"),
        _chk("mass_balance"),
        _chk("flash_vapor_fraction",   unit_tag_pattern="VES"),
        _chk("energy_balance_heater",  unit_tag_pattern="HT", max_dT_K=500),
    ]
    results = _run(graph, execution, checks)

    for r in results:
        _assert(r, passed=True, label="known_correct")

    # Verify simulation-grounded checks use execution data
    temp_r = next(r for r in results if r["check"] == "temp_increases_across")
    assert temp_r["source"] == "execution", "temp_increases_across should use execution data"
    assert temp_r["severity"] == CheckSeverity.CRITICAL

    two_phase_r = next(r for r in results if r["check"] == "two_phase_outlet")
    assert two_phase_r["source"] == "execution"
    assert two_phase_r["severity"] == CheckSeverity.CRITICAL

    mb_r = next(r for r in results if r["check"] == "mass_balance")
    assert mb_r["source"] == "execution"
    assert mb_r["severity"] == CheckSeverity.CRITICAL

    flash_r = next(r for r in results if r["check"] == "flash_vapor_fraction")
    assert flash_r["source"] == "execution"

    # BIP check is still IR-based
    bip_r = next(r for r in results if r["check"] == "bip_injected")
    assert bip_r["source"] == "IR"

    print("PASS  test_known_correct")
    for r in results:
        print(f"      [{r['severity'][:4]}] [{r['source']:9s}] "
              f"{r['check']:35s} passed={r['passed']}  {r['detail']}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — known-incorrect: Heater with T_out < T_in (cold outlet from heater)
#
# Feed at 400 K enters HT-01; DWSIM somehow reports outlet at 350 K.
# This is physically impossible for a heater and must be flagged as CRITICAL.
#
# Expected failures:
#   temp_increases_across  → FAIL, CRITICAL, source=execution
#   energy_balance_heater  → PASS (ΔT = -50 K < 500 K limit — the check only
#                            guards against implausibly large positive rises,
#                            not negative ones; direction is caught by the above)
# Expected passes:
#   mass_balance           → PASS (flows conserved)
#   unit_type_present      → PASS
# ══════════════════════════════════════════════════════════════════════════════

def test_known_incorrect():
    units    = [_Unit("HT-01", "Heater", {"T_out": 350.0})]
    topology = {"HT-01": {"inlets": ["FEED"], "outlets": ["PRODUCT"]}}
    graph    = _Graph(units, topology, property_package="Peng-Robinson")

    streams = {
        "FEED":    _Stream("FEED",    T_K=400.0, P_Pa=101325, flow_mol_s=1.0,
                           vapor_fraction=0.0,
                           composition={"Benzene": 0.5, "Toluene": 0.5}),
        "PRODUCT": _Stream("PRODUCT", T_K=350.0, P_Pa=101325, flow_mol_s=1.0,
                           vapor_fraction=0.0,
                           composition={"Benzene": 0.5, "Toluene": 0.5}),
    }
    execution = _Execution(streams)

    results = _run(graph, execution, [
        _chk("unit_type_present",     unit_type="Heater"),
        _chk("temp_increases_across", unit_tag_pattern="HT"),
        _chk("mass_balance"),
        _chk("energy_balance_heater", unit_tag_pattern="HT", max_dT_K=500),
    ])

    r_by_check = {r["check"]: r for r in results}

    _assert(r_by_check["unit_type_present"],     passed=True,  label="known_incorrect")
    _assert(r_by_check["mass_balance"],          passed=True,  label="known_incorrect")

    # THE CRITICAL ASSERTION: temp direction failure detected from execution data
    _assert(r_by_check["temp_increases_across"], passed=False,
            severity=CheckSeverity.CRITICAL, label="known_incorrect")
    assert r_by_check["temp_increases_across"]["source"] == "execution", (
        "temp_increases_across must use execution data, not IR fallback")

    # energy_balance_heater does not detect wrong direction (by design — see docstring)
    assert r_by_check["energy_balance_heater"]["passed"] is True, (
        "energy_balance_heater only guards against large positive ΔT; "
        "direction error is caught by temp_increases_across")

    print("PASS  test_known_incorrect")
    for r in results:
        status = "FAIL" if not r["passed"] else "pass"
        print(f"  {status:4s}  [{r['severity'][:4]}] [{r['source']:9s}] "
              f"{r['check']:35s}  {r['detail']}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — borderline: flash at bubble point (ethanol/water, 78°C, 1 atm)
#
# At the bubble point virtually all feed is liquid. The vapour outlet carries
# only 0.09% of the feed (0.0009 mol/s out of 1.0 mol/s) — just below the
# 0.1% flow threshold used by flash_vapor_fraction.
#
# Expected behaviour (documented, not asserted as errors):
#   two_phase_outlet     → PASS: the vapour stream VF=1.0 > 0.05 ✓
#                                the liquid stream VF=0.0 < 0.95 ✓
#                                both outlet flows > min_flow threshold ✗
#                                   vapour flow = 0.0009 < 0.001 (threshold)
#                                   → FAIL because both_nonzero_flow=False
#   flash_vapor_fraction → FAIL: vapour flow 0.0009 < 0.001 threshold
#
# Interpretation: both checks correctly flag this as a degenerate case.
# The flash is at the bubble point — operating conditions are marginally
# in the two-phase region.  The CRITICAL failures are intentional and correct.
# ══════════════════════════════════════════════════════════════════════════════

def test_borderline_bubble_point():
    units    = [_Unit("VES-01", "Vessel", {})]
    topology = {"VES-01": {"inlets": ["FEED"], "outlets": ["VAP-OUT", "LIQ-OUT"]}}
    graph    = _Graph(units, topology,
                      property_package="NRTL",
                      binary_parameters=[{"model": "NRTL", "compound_a": "Ethanol",
                                           "compound_b": "Water",
                                           "A12": 586.1, "A21": -195.0, "alpha12": 0.5765}])

    # Vapour flow = 0.0009 mol/s = 0.09% of feed = just below 0.1% threshold
    streams = {
        "FEED":    _Stream("FEED",    T_K=351.0, P_Pa=101325, flow_mol_s=1.0,
                           vapor_fraction=0.0,
                           composition={"Ethanol": 0.5, "Water": 0.5}),
        "VAP-OUT": _Stream("VAP-OUT", T_K=351.0, P_Pa=101325, flow_mol_s=0.0009,
                           vapor_fraction=1.0,
                           composition={"Ethanol": 0.68, "Water": 0.32}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=351.0, P_Pa=101325, flow_mol_s=0.9991,
                           vapor_fraction=0.0,
                           composition={"Ethanol": 0.499, "Water": 0.501}),
    }
    execution = _Execution(streams)

    checks = [
        _chk("two_phase_outlet",     unit_tag_pattern="VES"),
        _chk("flash_vapor_fraction", unit_tag_pattern="VES"),
        _chk("bip_injected",         model="NRTL"),
        _chk("mass_balance"),
    ]
    results = _run(graph, execution, checks)
    r_by_check = {r["check"]: r for r in results}

    # Both two_phase_outlet and flash_vapor_fraction flag the degenerate case
    _assert(r_by_check["two_phase_outlet"],     passed=False,
            severity=CheckSeverity.CRITICAL, label="borderline")
    _assert(r_by_check["flash_vapor_fraction"], passed=False,
            severity=CheckSeverity.CRITICAL, label="borderline")

    # BIP and mass balance still pass
    _assert(r_by_check["bip_injected"], passed=True,  label="borderline")
    _assert(r_by_check["mass_balance"], passed=True,  label="borderline")

    # Confirm the threshold is exactly 0.1% of feed
    from benchmark.physics_eval import _FLOW_FRAC_MIN
    threshold = 1.0 * _FLOW_FRAC_MIN   # feed=1.0, so threshold = _FLOW_FRAC_MIN exactly
    assert 0.0009 < threshold, (
        f"vapour flow (0.0009 mol/s) must be below threshold ({threshold:.4f} mol/s) "
        f"for this test to exercise the near-bubble-point case")

    print("PASS  test_borderline_bubble_point")
    print(f"      (vapour_flow=0.0009 mol/s, threshold={threshold:.4f} mol/s "
          f"— just below; both flash checks correctly FAIL)")
    for r in results:
        status = "FAIL" if not r["passed"] else "pass"
        print(f"  {status:4s}  [{r['severity'][:4]}] [{r['source']:9s}] "
              f"{r['check']:35s}  {r['detail']}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — fallback behaviour: no execution data (IR-only mode)
#
# Verifies that when pipeline_result.final_execution is None, all checks
# that depend on execution data fall back gracefully:
#   - temp_increases_across   → falls back to IR (WARNING, not CRITICAL)
#   - two_phase_outlet        → falls back to IR structural check (INFO)
#   - mass_balance            → skipped (INFO)
#   - flash_vapor_fraction    → skipped (INFO)
# ══════════════════════════════════════════════════════════════════════════════

def test_ir_only_fallback():
    units    = [_Unit("HT-01", "Heater", {"T_out": 373.15})]
    topology = {"HT-01": {"inlets": ["FEED"], "outlets": ["PRODUCT"]}}
    graph    = _Graph(units, topology, property_package="Peng-Robinson")

    pr = _PipelineResult(graph, execution=None, outcome="PASS")

    checks = [
        _chk("temp_increases_across", unit_tag_pattern="HT"),
        _chk("two_phase_outlet",      unit_tag_pattern="HT"),
        _chk("mass_balance"),
        _chk("flash_vapor_fraction",  unit_tag_pattern="HT"),
    ]
    results = []
    for c in checks:
        results.append(evaluate_check(c, graph, pr))

    r_by_check = {r["check"]: r for r in results}

    # temp_increases_across: falls back to IR, severity WARNING (373 > 298.15 heuristic)
    assert r_by_check["temp_increases_across"]["source"]   == "IR"
    assert r_by_check["temp_increases_across"]["severity"] == CheckSeverity.WARNING
    assert r_by_check["temp_increases_across"]["passed"]   is True

    # two_phase_outlet: no execution → INFO fallback
    assert r_by_check["two_phase_outlet"]["severity"] == CheckSeverity.INFO

    # mass_balance: no execution → INFO / skipped
    assert r_by_check["mass_balance"]["source"]   == "none"
    assert r_by_check["mass_balance"]["severity"] == CheckSeverity.INFO
    assert r_by_check["mass_balance"]["passed"]   is True

    # flash_vapor_fraction: no execution → INFO / skipped
    assert r_by_check["flash_vapor_fraction"]["source"] == "none"

    print("PASS  test_ir_only_fallback")
    for r in results:
        print(f"      [{r['severity'][:4]}] [{r['source']:9s}] "
              f"{r['check']:35s} passed={r['passed']}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — vle_bubble_point_spot_check
#
# Uses benzene/toluene (both in Antoine database).
# bubble_point_K(["Benzene","Toluene"], 101325) ≈ 368 K (equimolar Raoult's Law).
#
# Sub-test A: flash at 373 K (above bubble point) with VF=0.42 — consistent, should PASS.
# Sub-test B: flash at 340 K (below bubble point, 28 K below) with VF=0.4 — inconsistent.
# Sub-test C: flash at 420 K (50 K above bubble) with VF=0.0 — inconsistent (all-liquid
#             well above bubble).
# Sub-test D: compounds not in database (Polymer A) — should skip (INFO, passed=True).
# ══════════════════════════════════════════════════════════════════════════════

def test_vle_bubble_point_spot_check():
    # Shared graph structure: feed → VES-01 → [VAP-OUT, LIQ-OUT]
    units    = [_Unit("VES-01", "Vessel", {})]
    topology = {"VES-01": {"inlets": ["FEED"], "outlets": ["VAP-OUT", "LIQ-OUT"]}}

    def _make_pr(T_flash, VF_overall, compounds=None):
        g = _Graph(units, topology, property_package="Peng-Robinson")
        g.compounds = compounds or ["Benzene", "Toluene"]
        vap_flow = VF_overall
        liq_flow = 1.0 - VF_overall
        streams = {
            "FEED":    _Stream("FEED",    T_K=T_flash,  P_Pa=101325, flow_mol_s=1.0,
                               vapor_fraction=VF_overall),
            "VAP-OUT": _Stream("VAP-OUT", T_K=T_flash,  P_Pa=101325, flow_mol_s=vap_flow,
                               vapor_fraction=1.0),
            "LIQ-OUT": _Stream("LIQ-OUT", T_K=T_flash,  P_Pa=101325, flow_mol_s=liq_flow,
                               vapor_fraction=0.0),
        }
        return _PipelineResult(g, _Execution(streams))

    chk = _chk("vle_bubble_point_spot_check", unit_tag_pattern="VES", T_margin_K=10.0)

    # A: 373 K, VF=0.42 — above bubble point, VF consistent → PASS
    r_A = evaluate_check(chk, _make_pr(373.0, 0.42).final_graph, _make_pr(373.0, 0.42))
    _assert(r_A, passed=True, label="vle_spot A")
    assert r_A["source"] == "execution"

    # B: 340 K (28 K below ~368 K bubble point), VF=0.40 — inconsistent → FAIL/WARNING
    r_B = evaluate_check(chk, _make_pr(340.0, 0.40).final_graph, _make_pr(340.0, 0.40))
    _assert(r_B, passed=False, severity="WARNING", label="vle_spot B")
    assert "BELOW" in r_B["detail"], f"expected 'BELOW' in detail: {r_B['detail']}"

    # C: 420 K (52 K above bubble), VF=0.0 — inconsistent (above bubble, all liquid) → FAIL/WARNING
    r_C = evaluate_check(chk, _make_pr(420.0, 0.0).final_graph, _make_pr(420.0, 0.0))
    _assert(r_C, passed=False, severity="WARNING", label="vle_spot C")
    assert "ABOVE" in r_C["detail"], f"expected 'ABOVE' in detail: {r_C['detail']}"

    # D: unknown compound — bubble_point_K returns None → INFO skip
    pr_D = _make_pr(373.0, 0.42, compounds=["PolymerX", "PolymerY"])
    r_D  = evaluate_check(chk, pr_D.final_graph, pr_D)
    assert r_D["passed"] is True
    assert r_D["severity"] == "INFO", f"expected INFO for unknown compounds, got {r_D['severity']}"

    print("PASS  test_vle_bubble_point_spot_check")
    for label, r in [("A:consistent", r_A), ("B:below_bubble", r_B),
                     ("C:above_no_vap", r_C), ("D:unknown_comps", r_D)]:
        status = "FAIL" if not r["passed"] else "pass"
        print(f"  {status:4s} [{label:<18s}] [{r['severity'][:4]}] "
              f"[{r['source']:9s}]  {r['detail'][:90]}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — separation_achieved
#
# Three scenarios for a flash vessel:
#
# A: Good separation — benzene/toluene vapour enriched in benzene.
#    VAP: {Bz: 0.65, Tol: 0.35}  LIQ: {Bz: 0.40, Tol: 0.60}
#    rel_diff(Bz) = |0.65-0.40|/0.65 = 0.385 → > 0.10 → PASS
#
# B: No separation (NRTL-without-BIPs failure mode) — outlet ≈ feed for both phases.
#    VAP: {Bz: 0.502, Tol: 0.498}  LIQ: {Bz: 0.498, Tol: 0.502}
#    rel_diff(Bz) = |0.502-0.498|/0.502 ≈ 0.008 → < 0.10 → FAIL/CRITICAL
#
# C: Single-component system (pure propane) — compositions identical by definition.
#    → INFO skip (not a failure)
# ══════════════════════════════════════════════════════════════════════════════

def test_separation_achieved():
    units    = [_Unit("VES-01", "Vessel", {})]
    topology = {"VES-01": {"inlets": ["FEED"], "outlets": ["VAP-OUT", "LIQ-OUT"]}}
    graph_bt = _Graph(units, topology, property_package="Peng-Robinson")
    graph_bt.compounds = ["Benzene", "Toluene"]

    chk = _chk("separation_achieved", unit_tag_pattern="VES", min_relative_diff=0.10)

    # A: good separation
    streams_A = {
        "FEED":    _Stream("FEED",    T_K=373, P_Pa=101325, flow_mol_s=1.0),
        "VAP-OUT": _Stream("VAP-OUT", T_K=373, P_Pa=101325, flow_mol_s=0.45,
                           vapor_fraction=1.0,
                           composition={"Benzene": 0.65, "Toluene": 0.35}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=373, P_Pa=101325, flow_mol_s=0.55,
                           vapor_fraction=0.0,
                           composition={"Benzene": 0.40, "Toluene": 0.60}),
    }
    pr_A = _PipelineResult(graph_bt, _Execution(streams_A))
    r_A  = evaluate_check(chk, graph_bt, pr_A)
    _assert(r_A, passed=True, severity="CRITICAL", label="sep A")
    assert r_A["source"] == "execution"

    # B: no separation (NRTL-without-BIPs)
    streams_B = {
        "FEED":    _Stream("FEED",    T_K=373, P_Pa=101325, flow_mol_s=1.0),
        "VAP-OUT": _Stream("VAP-OUT", T_K=373, P_Pa=101325, flow_mol_s=0.5,
                           vapor_fraction=1.0,
                           composition={"Benzene": 0.502, "Toluene": 0.498}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=373, P_Pa=101325, flow_mol_s=0.5,
                           vapor_fraction=0.0,
                           composition={"Benzene": 0.498, "Toluene": 0.502}),
    }
    pr_B = _PipelineResult(graph_bt, _Execution(streams_B))
    r_B  = evaluate_check(chk, graph_bt, pr_B)
    _assert(r_B, passed=False, severity="CRITICAL", label="sep B")
    assert "FAIL" in r_B["detail"] or "identical" in r_B["detail"].lower() \
        or "compositions nearly identical" in r_B["detail"], \
        f"detail should mention failure reason: {r_B['detail']}"

    # Quantitative check: the rel_diff value should be visible in the detail
    assert "rel_diff=" in r_B["detail"], f"expected rel_diff in detail: {r_B['detail']}"

    # C: single-component
    graph_c   = _Graph([_Unit("VES-01", "Vessel", {})], topology, property_package="Peng-Robinson")
    graph_c.compounds = ["Propane"]
    streams_C = {
        "FEED":    _Stream("FEED",    T_K=300, P_Pa=101325, flow_mol_s=1.0),
        "VAP-OUT": _Stream("VAP-OUT", T_K=300, P_Pa=101325, flow_mol_s=0.3,
                           vapor_fraction=1.0,
                           composition={"Propane": 1.0}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=300, P_Pa=101325, flow_mol_s=0.7,
                           vapor_fraction=0.0,
                           composition={"Propane": 1.0}),
    }
    pr_C = _PipelineResult(graph_c, _Execution(streams_C))
    r_C  = evaluate_check(chk, graph_c, pr_C)
    assert r_C["passed"] is True
    assert r_C["severity"] == "INFO", f"single-component should be INFO: {r_C}"

    # ── D: near-azeotrope guard ────────────────────────────────────────────────
    # Ethanol-water flash very close to the 89.4 mol% azeotrope.
    # VAP: {EtOH: 0.890, Water: 0.110}
    # LIQ: {EtOH: 0.885, Water: 0.115}
    #
    # rel_diff by compound:
    #   Ethanol: |0.890 - 0.885| / 0.890 ≈ 0.0056  (< 1%)
    #   Water:   |0.110 - 0.115| / 0.115 ≈ 0.043   (4.3%)
    # max_rel_diff = 0.043 — Water is a minor component so even tiny
    # absolute shifts look large in relative terms.
    #
    # Without guard (threshold=0.10): 0.043 < 0.10 → FAIL (false positive)
    # With guard (NRTL + domain=azeotrope → threshold=0.03): 0.043 > 0.03 → PASS ✓
    #
    # Also verify BIP-missing failure (rel_diff≈0.008) still fails under the guard.

    graph_ew = _Graph(
        [_Unit("VES-01", "Vessel", {})], topology,
        property_package="NRTL",
        binary_parameters=[{"model": "NRTL", "compound_a": "Ethanol",
                             "compound_b": "Water",
                             "A12": 586.1, "A21": -195.0, "alpha12": 0.5765}],
    )
    graph_ew.compounds = ["Ethanol", "Water"]

    streams_D_good = {
        "FEED":    _Stream("FEED",    T_K=351, P_Pa=101325, flow_mol_s=1.0),
        # Very close to azeotrope: 89.0/88.5 mol% ethanol
        # rel_diff(Water) = |0.110-0.115|/0.115 ≈ 0.043 — between 3% and 10%
        "VAP-OUT": _Stream("VAP-OUT", T_K=351, P_Pa=101325, flow_mol_s=0.45,
                           vapor_fraction=1.0,
                           composition={"Ethanol": 0.890, "Water": 0.110}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=351, P_Pa=101325, flow_mol_s=0.55,
                           vapor_fraction=0.0,
                           composition={"Ethanol": 0.885, "Water": 0.115}),
    }
    pr_D_good = _PipelineResult(graph_ew, _Execution(streams_D_good))

    streams_D_fail = {
        "FEED":    _Stream("FEED",    T_K=351, P_Pa=101325, flow_mol_s=1.0),
        "VAP-OUT": _Stream("VAP-OUT", T_K=351, P_Pa=101325, flow_mol_s=0.5,
                           vapor_fraction=1.0,
                           composition={"Ethanol": 0.502, "Water": 0.498}),
        "LIQ-OUT": _Stream("LIQ-OUT", T_K=351, P_Pa=101325, flow_mol_s=0.5,
                           vapor_fraction=0.0,
                           composition={"Ethanol": 0.498, "Water": 0.502}),
    }
    pr_D_fail = _PipelineResult(graph_ew, _Execution(streams_D_fail))

    # D1: near-azeotrope, no domain set → standard threshold → FAIL (false positive)
    chk_no_domain = _chk("separation_achieved", unit_tag_pattern="VES",
                          min_relative_diff=0.10)
    r_D1 = evaluate_check(chk_no_domain, graph_ew, pr_D_good)
    _assert(r_D1, passed=False, severity="CRITICAL", label="sep D1 (no guard, false pos)")

    # D2: near-azeotrope, domain=azeotrope set → lowered threshold → PASS ✓
    chk_azeo = _chk("separation_achieved", unit_tag_pattern="VES",
                     min_relative_diff=0.10, domain="azeotrope")
    r_D2 = evaluate_check(chk_azeo, graph_ew, pr_D_good)
    _assert(r_D2, passed=True, severity="CRITICAL", label="sep D2 (guard active)")
    assert "near-azeotrope guard active" in r_D2["detail"], (
        f"guard note must appear in detail: {r_D2['detail']}")
    assert "threshold=0.03" in r_D2["detail"], (
        f"lowered threshold must appear in detail: {r_D2['detail']}")

    # D3: BIP-missing failure on same azeotrope case — must still FAIL under guard
    r_D3 = evaluate_check(chk_azeo, graph_ew, pr_D_fail)
    _assert(r_D3, passed=False, severity="CRITICAL", label="sep D3 (BIP-miss still fails)")
    assert "near-azeotrope guard active" in r_D3["detail"]

    # D4: domain=azeotrope but EOS package (not NRTL/UNIQUAC) → guard NOT applied
    graph_pr = _Graph(
        [_Unit("VES-01", "Vessel", {})], topology, property_package="Peng-Robinson")
    graph_pr.compounds = ["Ethanol", "Water"]
    r_D4 = evaluate_check(chk_azeo, graph_pr, pr_D_good)
    _assert(r_D4, passed=False, severity="CRITICAL", label="sep D4 (EOS no guard)")
    assert "near-azeotrope guard active" not in r_D4["detail"], (
        f"guard must NOT fire for EOS packages: {r_D4['detail']}")

    print("PASS  test_separation_achieved")
    for label, r in [
        ("A:good_sep",         r_A),
        ("B:no_sep(BIP_miss)", r_B),
        ("C:single_comp",      r_C),
        ("D1:azeo_no_guard",   r_D1),
        ("D2:azeo_guard_PASS", r_D2),
        ("D3:azeo_BIP_FAIL",   r_D3),
        ("D4:EOS_no_guard",    r_D4),
    ]:
        status = "FAIL" if not r["passed"] else "pass"
        print(f"  {status:4s} [{label:<22s}] [{r['severity'][:4]}] "
              f"[{r['source']:9s}]  {r['detail'][:100]}")


# ══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Physics eval test suite")
    print("=" * 70)

    print("\n--- Test 1: known_correct ---")
    test_known_correct()

    print("\n--- Test 2: known_incorrect ---")
    test_known_incorrect()

    print("\n--- Test 3: borderline_bubble_point ---")
    test_borderline_bubble_point()

    print("\n--- Test 4: ir_only_fallback ---")
    test_ir_only_fallback()

    print("\n--- Test 5: vle_bubble_point_spot_check ---")
    test_vle_bubble_point_spot_check()

    print("\n--- Test 6: separation_achieved ---")
    test_separation_achieved()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)
