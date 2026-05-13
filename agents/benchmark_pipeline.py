"""
Pipeline-level benchmark for the multi-agent flowsheet system.

Measures end-to-end success rate across a diverse process description set,
enabling the paper's evaluation table.

Usage
─────
# Dry run — Basis/Planner/Thermo use real LLM calls; DWSIM is mocked:
    python agents/benchmark_pipeline.py

# Full run — requires a live DWSIM container:
    python agents/benchmark_pipeline.py --real-executor

# Specific model:
    python agents/benchmark_pipeline.py --model claude-sonnet-4-6
"""
from __future__ import annotations
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator import Orchestrator, OrchestratorResult
from agents.executor import ExecutionResult, StreamResult
from agents.llm import reset_call_count, get_call_count

# ── Property package families ─────────────────────────────────────────────────

_IDEAL_PKGS    = {"Raoult's Law"}
_ACTIVITY_PKGS = {"NRTL", "UNIQUAC"}
_EOS_PKGS      = {"Peng-Robinson", "Soave-Redlich-Kwong", "Lee-Kesler-Plöcker"}

_PKG_FAMILIES: dict[str, set[str]] = {
    "ideal":    _IDEAL_PKGS,
    "activity": _ACTIVITY_PKGS,
    "eos":      _EOS_PKGS,
}


# ── Test cases ────────────────────────────────────────────────────────────────

@dataclass
class PipelineTestCase:
    name:                          str
    description:                   str
    expected_outcome:              str        # "PASS" | "HUMAN" | "BASIS_FAILED" | "MAX_ITER"
    expected_compounds:            list[str]
    expected_property_package_family: str     # "ideal" | "activity" | "eos"
    max_iterations_allowed:        int = 4


PIPELINE_TEST_CASES: list[PipelineTestCase] = [
    PipelineTestCase(
        name="Ideal binary flash — methanol/water",
        description=(
            "Flash separate a 50/50 molar methanol/water feed at 1 atm and 25°C "
            "with a molar flow of 1 mol/s. The feed is first heated to 80°C then "
            "sent to a flash vessel to separate vapour and liquid phases."
        ),
        expected_outcome="PASS",
        expected_compounds=["Methanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Non-ideal azeotropic flash — ethanol/water",
        description=(
            "Heat a 60/40 molar ethanol/water feed at 1 atm and 298 K "
            "to 353 K, then flash it in a separator to obtain a vapour "
            "enriched in ethanol and a liquid lean in ethanol."
        ),
        expected_outcome="PASS",
        expected_compounds=["Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Hydrocarbon flash — methane/ethane high pressure",
        description=(
            "Flash separate a 70/30 molar methane/ethane mixture at 50 bar "
            "and 200 K to obtain a methane-rich vapour and an ethane-rich liquid."
        ),
        expected_outcome="PASS",
        expected_compounds=["Methane", "Ethane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Multi-unit — heater + flash (n-hexane/n-heptane)",
        description=(
            "Heat a 40/60 molar feed of n-hexane and n-heptane at 1 atm and 298 K "
            "to 360 K using a heater, then flash the heated stream in a vessel "
            "to separate the more volatile n-hexane into the vapour phase."
        ),
        expected_outcome="PASS",
        expected_compounds=["n-Hexane", "n-Heptane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Mixer + flash — two feeds combined before separation",
        description=(
            "Mix a pure methanol stream (1 mol/s, 298 K, 1 atm) with a pure water "
            "stream (1 mol/s, 298 K, 1 atm) in a mixer, heat the combined stream "
            "to 360 K, then flash it to separate vapour and liquid."
        ),
        expected_outcome="PASS",
        expected_compounds=["Methanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Simple heater only — propane heating",
        description=(
            "Heat a pure propane stream at 10 bar and 250 K with a molar flow "
            "of 2 mol/s to 320 K using a heater. No separation required."
        ),
        expected_outcome="PASS",
        expected_compounds=["Propane"],
        expected_property_package_family="eos",
        max_iterations_allowed=3,
    ),
    PipelineTestCase(
        name="High-pressure natural gas flash — methane/ethane/propane",
        description=(
            "Flash separate a natural gas stream containing 70 mol% methane, "
            "20 mol% ethane, and 10 mol% propane at 80 bar and 220 K "
            "to recover a liquid condensate."
        ),
        expected_outcome="PASS",
        expected_compounds=["Methane", "Ethane", "Propane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Cooler + flash — benzene/toluene separation",
        description=(
            "Cool a 50/50 molar benzene/toluene vapour feed at 1 atm and 400 K "
            "to 355 K, then flash it to obtain a benzene-rich vapour and a "
            "toluene-rich liquid."
        ),
        expected_outcome="PASS",
        expected_compounds=["Benzene", "Toluene"],
        expected_property_package_family="ideal",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Three-component flash — acetone/methanol/water",
        description=(
            "Flash a ternary mixture of 40 mol% acetone, 30 mol% methanol, and "
            "30 mol% water at 1 atm and 330 K to partially separate the acetone "
            "into the vapour phase."
        ),
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Methanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Unsupported compound — electrolyte (NaCl)",
        description=(
            "Dissolve sodium chloride in water at 1 atm and 298 K, then heat "
            "the brine solution to 373 K."
        ),
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="ideal",  # not reached
        max_iterations_allowed=2,
    ),
    PipelineTestCase(
        name="Unsupported compound — NaOH electrolyte",
        description=(
            "Neutralise a caustic stream (NaOH in water) with acetic acid "
            "at 1 atm and 298 K."
        ),
        expected_outcome="BASIS_FAILED",
        expected_compounds=[],
        expected_property_package_family="ideal",  # not reached
        max_iterations_allowed=2,
    ),
    PipelineTestCase(
        name="Ambiguous abbreviation — IPA/water flash",
        description=(
            "Flash separate an IPA/water mixture at 1 atm and 355 K. "
            "The feed is 40 mol% IPA and 60 mol% water at a flow of 1 mol/s."
        ),
        expected_outcome="PASS",
        expected_compounds=["Water"],   # IPA should resolve to a propanol isomer
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="Diethyl ether / ethanol separation",
        description=(
            "Separate a 60/40 molar mixture of diethyl ether and ethanol at "
            "1 atm by partial condensation: cool the feed from 340 K to 310 K "
            "in a cooler, then flash the two-phase stream."
        ),
        expected_outcome="PASS",
        expected_compounds=["Diethyl Ether", "Ethanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    # ── CALIBRATION test cases — exercise PARAM_MISSING → CALIBRATION path ──────
    PipelineTestCase(
        name="[CALIBRATION] Acetone / chloroform flash — negative-deviation azeotrope",
        description=(
            "Flash separate an equimolar mixture of acetone and chloroform at "
            "343 K and 1 atm. Feed flow is 1 mol/s. Use NRTL to capture the "
            "negative deviations from Raoult's Law. Report vapour and liquid "
            "phase compositions."
        ),
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Chloroform"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[CALIBRATION] n-Propanol / water flash — alcohol-water azeotrope",
        description=(
            "Partial vaporisation of a 50/50 mol/mol n-propanol and water "
            "mixture at 370 K and 1 atm. Feed flow is 1 mol/s. Use NRTL to "
            "account for non-ideal liquid-phase interactions."
        ),
        expected_outcome="PASS",
        expected_compounds=["1-Propanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[CALIBRATION] Diethyl ether / methanol flash — activity-coefficient system",
        description=(
            "Flash separate diethyl ether and methanol (50/50 molar feed) at "
            "308 K and 1 atm. Feed flow is 1 mol/s. Use NRTL for accurate "
            "VLE representation."
        ),
        expected_outcome="PASS",
        expected_compounds=["Diethyl Ether", "Methanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    # Note: the three cases below require --real-executor to trigger PARAM_MISSING;
    # the mock executor always returns solved=True and never visits CALIBRATION routing.
    PipelineTestCase(
        name="[CALIBRATION] 2-Propanol / water flash — IPA azeotrope",
        description=(
            "Flash separate a 40/60 molar 2-propanol and water feed at 355 K "
            "and 1 atm. Feed flow is 1 mol/s. Use NRTL to capture the "
            "minimum-boiling azeotrope at 87.7 mol% IPA."
        ),
        expected_outcome="PASS",
        expected_compounds=["2-Propanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[CALIBRATION] n-Hexane / ethanol flash — heterogeneous azeotrope",
        description=(
            "Flash separate an equimolar n-hexane and ethanol feed at 331 K "
            "and 1 atm. Feed flow is 1 mol/s. Use NRTL to account for the "
            "minimum-boiling heterogeneous azeotrope."
        ),
        expected_outcome="PASS",
        expected_compounds=["n-Hexane", "Ethanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[CALIBRATION] Tetrahydrofuran / water flash — cyclic ether system",
        description=(
            "Flash separate a 50/50 molar tetrahydrofuran and water feed at "
            "340 K and 1 atm. Feed flow is 1 mol/s. Use NRTL to model the "
            "azeotropic behaviour of this cyclic ether/water system."
        ),
        expected_outcome="PASS",
        expected_compounds=["Tetrahydrofuran", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[CALIBRATION] Ethyl acetate / ethanol flash — ester/alcohol system",
        description=(
            "Flash separate a 50/50 molar ethyl acetate and ethanol feed at "
            "345 K and 1 atm. Feed flow is 1 mol/s. Use NRTL to model "
            "non-ideal interactions between the ester and alcohol."
        ),
        expected_outcome="PASS",
        expected_compounds=["Ethyl Acetate", "Ethanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    # ── Stress tests ──────────────────────────────────────────────────────────
    PipelineTestCase(
        name="[STRESS] Sub-bubble-point cooler — acetone/water",
        description=(
            "Cool a 50/50 molar acetone/water vapour feed at 1 atm and 400 K "
            "to 290 K using a cooler, then flash the cooled stream in a vessel "
            "to recover a vapour fraction enriched in acetone."
        ),
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[STRESS] Near-pure feed — minimal separation",
        description=(
            "Flash a 95/5 molar methanol/water feed at 1 atm and 340 K. "
            "Feed flow is 1 mol/s. Recover the small water-enriched liquid fraction."
        ),
        expected_outcome="PASS",
        expected_compounds=["Methanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="[STRESS] Temperature inversion — cool then flash",
        description=(
            "A hot 40/60 molar methanol/water vapour stream exits a reactor at "
            "420 K and 1 atm at 2 mol/s. Cool it to 355 K then flash the cooled "
            "stream to obtain a methanol-enriched vapour and a water-enriched liquid."
        ),
        expected_outcome="PASS",
        expected_compounds=["Methanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="[STRESS] Ambiguous unit — condenser description",
        description=(
            "Condense a pure propane vapour stream at 10 bar and 320 K "
            "by removing heat until the stream is fully liquid. "
            "Feed flow is 1 mol/s."
        ),
        expected_outcome="PASS",
        expected_compounds=["Propane"],
        expected_property_package_family="eos",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="[STRESS] Partial BIP coverage — ternary acetone/methanol/benzene",
        description=(
            "Flash separate a ternary 40/30/30 molar feed of acetone, methanol, "
            "and benzene at 1 atm and 330 K. Feed flow is 1 mol/s. "
            "Use NRTL to model all binary interactions."
        ),
        expected_outcome="PASS",
        expected_compounds=["Acetone", "Methanol", "Benzene"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[STRESS] Implicit multi-stage — heat then flash ethanol/water",
        description=(
            "Take an equimolar ethanol and water stream at 298 K and 1 atm "
            "flowing at 2 mol/s. First mix with a pure ethanol stream of 1 mol/s "
            "at the same conditions, heat the blend to 365 K, then flash to "
            "recover an ethanol-enriched vapour."
        ),
        expected_outcome="PASS",
        expected_compounds=["Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
    PipelineTestCase(
        name="[STRESS] Recycle stream — unsupported topology",
        description=(
            "React methanol vapour at 400 K and 1 atm, recycle the unreacted "
            "methanol from the reactor outlet back to the feed, and purge a "
            "small bleed stream to prevent accumulation."
        ),
        expected_outcome="PLAN_FAILED",   # schema rejects cyclic topology at plan time
        expected_compounds=["Methanol"],
        expected_property_package_family="activity",
        max_iterations_allowed=4,
    ),
    PipelineTestCase(
        name="[STRESS] Mixed-phase biphasic feed — ethanol/water",
        description=(
            "A partially vaporised 30/70 molar ethanol/water feed at 1 atm "
            "and 358 K with a vapour fraction of approximately 0.4 and a "
            "total flow of 1 mol/s is fed directly to a flash vessel to "
            "complete the phase separation."
        ),
        expected_outcome="PASS",
        expected_compounds=["Ethanol", "Water"],
        expected_property_package_family="activity",
        max_iterations_allowed=6,
    ),
]


# ── Per-case metrics ──────────────────────────────────────────────────────────

@dataclass
class CaseMetrics:
    name:                    str
    outcome:                 str
    outcome_correct:         bool
    compound_match:          float       # fraction of expected compounds found
    package_family_correct:  bool
    llm_calls:               int
    iterations:              int
    elapsed_s:               float
    warnings:                list[str]   = field(default_factory=list)
    error:                   Optional[str] = None
    routing_trace:           list[str]   = field(default_factory=list)


# ── Mock executor ─────────────────────────────────────────────────────────────

def _mock_execution_result(flowsheet: dict) -> ExecutionResult:
    """Return a minimal solved ExecutionResult without touching DWSIM.

    Produces results that pass all Critic physics checks:
    - Mass balance: terminal outlet flows sum to total feed flow
    - NO_SEPARATION: each terminal outlet is enriched in a different compound
    - UNPHYSICAL_T/P: uses stream-defined T/P or safe defaults
    """
    streams   = flowsheet.get("streams", [])
    conns     = flowsheet.get("connections", [])
    compounds = flowsheet.get("compounds", [])
    n         = max(len(compounds), 1)

    # Connection format: [src_tag, dst_tag, src_port, dst_port]
    has_incoming = {c[1] for c in conns if len(c) >= 2}  # destinations
    has_outgoing = {c[0] for c in conns if len(c) >= 2}  # sources

    feed_comp = {c: 1.0 / n for c in compounds} if compounds else {}

    # Feed streams: not in has_incoming (nothing flows into them)
    feed_streams = [s for s in streams if s["tag"] not in has_incoming]
    total_feed_flow = sum(s.get("flow", 1.0) for s in feed_streams) or 1.0

    # Terminal outlets: receive flow from a unit, don't feed into another unit/stream.
    # Sorted by src_port (port 0 = vapour first, port 1 = liquid second) so that
    # composition enrichment below aligns with physical VLE convention.
    _outlet_port: dict[str, int] = {}
    for c in conns:
        if len(c) >= 3 and c[1] in has_incoming and c[1] not in has_outgoing:
            _outlet_port[c[1]] = c[2]  # src_port
    terminal_outlets = sorted(
        [s["tag"] for s in streams
         if s["tag"] in has_incoming and s["tag"] not in has_outgoing],
        key=lambda t: _outlet_port.get(t, 99),
    )
    n_terminals = max(len(terminal_outlets), 1)
    outlet_flow = total_feed_flow / n_terminals

    # Sort compounds by NBP (most volatile first) so outlet[0]/port-0 (vapour)
    # is always enriched in the most volatile compound — prevents WRONG_PHASE_DIR.
    from agents.critic import _NBP_K as _nbp
    _volatility = {c: _nbp.get(c.lower(), 500.0) for c in compounds}
    compounds_by_volatility = sorted(compounds, key=lambda c: _volatility[c])

    # Propagate Heater/Cooler T_out to their immediate outlet streams so the
    # Critic's LLM stage sees physically consistent temperatures in mock results.
    unit_outlet_T: dict[str, float] = {}   # stream_tag → T_K from upstream unit
    stream_by_tag = {s["tag"]: s for s in streams}
    for u in flowsheet.get("units", []):
        t_out = u.get("T_out")
        if t_out and u.get("type") in ("Heater", "Cooler"):
            for conn in conns:
                if len(conn) >= 2 and conn[0] == u["tag"] and conn[1] in stream_by_tag:
                    unit_outlet_T[conn[1]] = float(t_out)

    stream_results: dict[str, StreamResult] = {}
    for s in streams:
        tag     = s["tag"]
        is_feed = tag not in has_incoming

        # Flow: feeds keep defined flow; terminal outlets split feed total evenly;
        # intermediate streams carry full feed flow
        if is_feed:
            flow = s.get("flow", 1.0)
        elif tag in terminal_outlets:
            flow = outlet_flow
        else:
            flow = total_feed_flow

        # Composition: terminal outlets get differentiated comps to pass NO_SEPARATION.
        # Uses volatility-sorted compound order so the vapour outlet (port 0, idx 0)
        # is enriched in the most volatile compound — prevents WRONG_PHASE_DIR.
        if not is_feed and tag in terminal_outlets and n >= 2 and n_terminals >= 2:
            idx   = terminal_outlets.index(tag)
            comp  = {c: 1.0 / n for c in compounds}
            rich  = compounds_by_volatility[idx % n]
            lean  = compounds_by_volatility[(idx + 1) % n]
            delta = 0.3
            comp[rich] = min(1.0, comp[rich] + delta)
            comp[lean] = max(0.0, comp[lean] - delta)
            total = sum(comp.values())
            comp  = {c: v / total for c, v in comp.items()}
        else:
            comp = s.get("composition") or feed_comp

        # Temperature: use unit T_out if stream is a Heater/Cooler outlet,
        # else use flowsheet-defined T, else fall back to 298.15 K
        T_K = unit_outlet_T.get(tag) or s.get("T", 298.15)

        stream_results[tag] = StreamResult(
            tag=tag,
            T_K=T_K,
            P_Pa=s.get("P", 101325.0),
            flow_mol_s=flow,
            composition=comp,
            is_feed=is_feed,
        )
    return ExecutionResult(solved=True, stream_results=stream_results)


# ── Package family checker ────────────────────────────────────────────────────

def _package_family(pkg: str) -> str:
    for family, pkgs in _PKG_FAMILIES.items():
        if pkg in pkgs:
            return family
    return "unknown"


def _family_correct(assigned_pkg: str, expected_family: str) -> bool:
    return _package_family(assigned_pkg) == expected_family


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(
    model: str = "claude-sonnet-4-6",
    use_real_executor: bool = False,
    inter_case_delay: float = 10.0,
) -> list[CaseMetrics]:
    """
    Run all PIPELINE_TEST_CASES through the Orchestrator.

    Parameters
    ----------
    model             : LLM model for all agents
    use_real_executor : if False, Executor.run is patched to avoid DWSIM
    inter_case_delay  : seconds to sleep between cases (default 10) — prevents
                        sustained rate-limit exhaustion on Gemini's RPM quota
    """
    metrics: list[CaseMetrics] = []

    for i, tc in enumerate(PIPELINE_TEST_CASES):
        if i > 0 and inter_case_delay > 0:
            print(f"  [benchmark] sleeping {inter_case_delay:.0f}s before next case...")
            time.sleep(inter_case_delay)

        reset_call_count()
        t0 = time.time()

        try:
            orch = Orchestrator(model=model, max_iterations=tc.max_iterations_allowed)

            if use_real_executor:
                result: OrchestratorResult = orch.run(tc.description)
            else:
                with patch(
                    "agents.executor.Executor.run",
                    side_effect=lambda fs: _mock_execution_result(fs),
                ):
                    result = orch.run(tc.description)

        except Exception as exc:
            elapsed = time.time() - t0
            metrics.append(CaseMetrics(
                name=tc.name,
                outcome="ERROR",
                outcome_correct=False,
                compound_match=0.0,
                package_family_correct=False,
                llm_calls=get_call_count(),
                iterations=0,
                elapsed_s=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue

        elapsed   = time.time() - t0
        llm_calls = get_call_count()

        # Compound match
        found = (set(result.basis_result.dwsim_compounds)
                 if result.basis_result else set())
        if tc.expected_compounds:
            compound_match = sum(
                1 for c in tc.expected_compounds if c in found
            ) / len(tc.expected_compounds)
        else:
            compound_match = 1.0  # BASIS_FAILED cases — no compounds expected

        # Property package family
        assigned_pkg = ""
        pkg_family_ok = False
        if result.final_flowsheet:
            assigned_pkg = result.final_flowsheet.get("property_package", "")
            pkg_family_ok = _family_correct(assigned_pkg, tc.expected_property_package_family)
        elif tc.expected_outcome == "BASIS_FAILED":
            pkg_family_ok = True   # package never assigned — not counted against score

        routing_trace = [f"i{i}:{rec.routing}" for i, rec in enumerate(result.iterations)]

        metrics.append(CaseMetrics(
            name=tc.name,
            outcome=result.outcome,
            outcome_correct=(result.outcome == tc.expected_outcome),
            compound_match=compound_match,
            package_family_correct=pkg_family_ok,
            llm_calls=llm_calls,
            iterations=len(result.iterations),
            elapsed_s=elapsed,
            warnings=result.warnings,
            routing_trace=routing_trace,
        ))

    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(metrics: list[CaseMetrics], model: str) -> None:
    print(f"\n## Pipeline Benchmark — model: {model}\n")

    header = (f"| {'Test case':<48} | {'Outcome':<12} | {'OK':>2} "
              f"| {'CmpMatch':>8} | {'PkgFam':>6} | {'LLM':>4} | {'Iter':>4} | {'Time(s)':>7} |")
    sep    = "|" + "-" * 50 + "|" + "-" * 14 + "|" + "-" * 4 + "|" + "-" * 10 + \
             "|" + "-" * 8 + "|" + "-" * 6 + "|" + "-" * 6 + "|" + "-" * 9 + "|"
    print(header)
    print(sep)

    for m in metrics:
        ok_mark  = "✓" if m.outcome_correct        else "✗"
        pkg_mark = "✓" if m.package_family_correct  else "✗"
        name_short = m.name[:48]
        row = (f"| {name_short:<48} | {m.outcome:<12} | {ok_mark:>2} "
               f"| {m.compound_match:>8.0%} | {pkg_mark:>6} | {m.llm_calls:>4} "
               f"| {m.iterations:>4} | {m.elapsed_s:>7.1f} |")
        print(row)
        if m.error:
            print(f"|   ERROR: {m.error}")
        if m.outcome == "HUMAN" and m.routing_trace:
            print(f"|   trace: {' → '.join(m.routing_trace)}")
        for w in m.warnings:
            print(f"|   ⚠ {w[:90]}")

    print()

    n  = len(metrics)
    n_outcome_ok  = sum(1 for m in metrics if m.outcome_correct)
    n_pkg_ok      = sum(1 for m in metrics if m.package_family_correct)
    mean_llm      = sum(m.llm_calls  for m in metrics) / n if n else 0
    mean_time     = sum(m.elapsed_s  for m in metrics) / n if n else 0
    mean_compound = sum(m.compound_match for m in metrics) / n if n else 0

    print(f"**Outcome accuracy:**  {n_outcome_ok}/{n}  ({n_outcome_ok/n:.0%})")
    print(f"**Pkg family accuracy:** {n_pkg_ok}/{n}  ({n_pkg_ok/n:.0%})")
    print(f"**Mean compound match:** {mean_compound:.0%}")
    print(f"**Mean LLM calls:**    {mean_llm:.1f}")
    print(f"**Mean time:**         {mean_time:.1f}s")
    print(f"**Total cases:**       {n}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline-level benchmark")
    p.add_argument("--model",             default="claude-sonnet-4-6",
                   help="LLM model to use for all agents")
    p.add_argument("--real-executor",     action="store_true",
                   help="Use live DWSIM executor (requires DWSIM container)")
    p.add_argument("--inter-case-delay",  type=float, default=10.0,
                   help="Seconds to sleep between cases (default 10, set 0 to disable)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_benchmark(
        model=args.model,
        use_real_executor=args.real_executor,
        inter_case_delay=args.inter_case_delay,
    )
    _print_report(results, model=args.model)
