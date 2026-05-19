"""
Benchmark dataset and runner for the v2 pipeline (Item 9).

Provides:
  BENCHMARK_CASES   — 15 hardcoded test cases covering the major flowsheet
                      archetypes expected in the publication experiments.
  run_benchmark()   — run all cases through an orchestrator instance.
  run_baseline()    — single-agent baseline: one LLM call → JSON → validate.
  compare()         — compute delta metrics vs baseline.
  run_statistical() — repeat N runs of same case, report mean ± std.

Case difficulty tiers:
  easy   — 2-unit, single phase, ambient conditions
  medium — 3-5 unit, flash separation, BIP-requiring thermo
  hard   — recycle, azeotrope, high-pressure, multi-package
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from eval.metrics import CaseResult, BenchmarkMetrics, compute_metrics


# ── Benchmark case definitions ─────────────────────────────────────────────────

@dataclass
class BenchmarkCase:
    case_id:      str
    description:  str
    compounds:    list[str]
    tier:         str         # "easy" | "medium" | "hard"
    expected_pkg: str         # property package that should be selected
    expected_units: list[str] # unit types that should appear (subset check)
    notes:        str = ""


BENCHMARK_CASES: list[BenchmarkCase] = [
    # ── Easy: single-phase, ambient ──────────────────────────────────────────
    BenchmarkCase(
        case_id      = "E01",
        description  = "Heat a feed of methanol and water from 25°C to 80°C",
        compounds    = ["methanol", "water"],
        tier         = "easy",
        expected_pkg = "NRTL",
        expected_units = ["Heater"],
        notes        = "Minimal case: one unit, NRTL for methanol-water",
    ),
    BenchmarkCase(
        case_id      = "E02",
        description  = "Compress a pure methane stream from 1 bar to 5 bar",
        compounds    = ["methane"],
        tier         = "easy",
        expected_pkg = "Peng-Robinson",
        expected_units = ["Compressor"],
        notes        = "Single-compound gas compression",
    ),
    BenchmarkCase(
        case_id      = "E03",
        description  = "Pump liquid water from 1 atm to 10 atm",
        compounds    = ["water"],
        tier         = "easy",
        expected_pkg = "Peng-Robinson",
        expected_units = ["Pump"],
        notes        = "Liquid pumping, pure component",
    ),
    BenchmarkCase(
        case_id      = "E04",
        description  = "Cool a hot acetone stream from 150°C to 40°C using a condenser",
        compounds    = ["acetone"],
        tier         = "easy",
        expected_pkg = "Peng-Robinson",
        expected_units = ["Cooler"],
        notes        = "Single cooler",
    ),
    BenchmarkCase(
        case_id      = "E05",
        description  = "Heat a benzene-toluene mixture to 100°C then flash to "
                       "separate vapour and liquid phases",
        compounds    = ["benzene", "toluene"],
        tier         = "easy",
        expected_pkg = "Peng-Robinson",
        expected_units = ["Heater", "Vessel"],
        notes        = "Classic non-polar VLE; no BIPs needed",
    ),
    # ── Medium: flash separation, BIP-requiring thermo ───────────────────────
    BenchmarkCase(
        case_id      = "M01",
        description  = "Heat an ethanol-water feed to 78°C, flash separate vapour "
                       "from liquid",
        compounds    = ["ethanol", "water"],
        tier         = "medium",
        expected_pkg = "NRTL",
        expected_units = ["Heater", "Vessel"],
        notes        = "NRTL azeotrope; BIP injection required",
    ),
    BenchmarkCase(
        case_id      = "M02",
        description  = "Separate acetone and methanol by heating to 60°C then flashing",
        compounds    = ["acetone", "methanol"],
        tier         = "medium",
        expected_pkg = "NRTL",
        expected_units = ["Heater", "Vessel"],
        notes        = "Polar mixture; acetone-methanol azeotrope",
    ),
    BenchmarkCase(
        case_id      = "M03",
        description  = "Cool a methane-ethane-propane natural gas mixture from 50°C "
                       "to -30°C at 20 bar to liquefy heavier components, "
                       "then flash to remove liquid",
        compounds    = ["methane", "ethane", "propane"],
        tier         = "medium",
        expected_pkg = "Peng-Robinson",
        expected_units = ["Cooler", "Vessel"],
        notes        = "High-pressure light gases; PR appropriate",
    ),
    BenchmarkCase(
        case_id      = "M04",
        description  = "Pump liquid ethyl acetate and ethanol mixture to 3 bar, "
                       "heat to 80°C, then flash",
        compounds    = ["ethyl acetate", "ethanol"],
        tier         = "medium",
        expected_pkg = "NRTL",
        expected_units = ["Pump", "Heater", "Vessel"],
        notes        = "Pump + heater + flash with ester/alcohol mixture",
    ),
    BenchmarkCase(
        case_id      = "M05",
        description  = "Mix two feed streams of isopropanol and water, heat the "
                       "combined stream to 85°C, then flash",
        compounds    = ["isopropanol", "water"],
        tier         = "medium",
        expected_pkg = "NRTL",
        expected_units = ["Mixer", "Heater", "Vessel"],
        notes        = "Mixer insertion; isopropanol-water azeotrope",
    ),
    # ── Hard: recycle, azeotrope, high-pressure, multi-step ──────────────────
    BenchmarkCase(
        case_id      = "H01",
        description  = "Separate a ternary acetone-methanol-water mixture: "
                       "heat feed to 65°C, flash to remove low-boiling vapour, "
                       "cool and recycle the liquid back to the feed",
        compounds    = ["acetone", "methanol", "water"],
        tier         = "hard",
        expected_pkg = "NRTL",
        expected_units = ["Heater", "Vessel", "Cooler"],
        notes        = "Ternary polar; recycle path expected",
    ),
    BenchmarkCase(
        case_id      = "H02",
        description  = "Compress a CO2-rich gas stream from 1 bar to 80 bar in two "
                       "stages with intercooling between stages",
        compounds    = ["carbon dioxide"],
        tier         = "hard",
        expected_pkg = "Peng-Robinson",
        expected_units = ["Compressor", "Cooler", "Compressor"],
        notes        = "Two-stage compression with intercooler; topology test",
    ),
    BenchmarkCase(
        case_id      = "H03",
        description  = "Cryogenic separation of air: compress nitrogen-oxygen mixture "
                       "to 50 bar, cool to -150°C, expand through a turbine, "
                       "then flash to separate liquid oxygen",
        compounds    = ["nitrogen", "oxygen"],
        tier         = "hard",
        expected_pkg = "Lee-Kesler-Plöcker",
        expected_units = ["Compressor", "Cooler", "Expander", "Vessel"],
        notes        = "Cryogenic; Lee-Kesler-Plöcker required",
    ),
    BenchmarkCase(
        case_id      = "H04",
        description  = "Separate tetrahydrofuran and water by heating to 70°C, "
                       "flashing, then cooling the vapour to condense THF",
        compounds    = ["tetrahydrofuran", "water"],
        tier         = "hard",
        expected_pkg = "NRTL",
        expected_units = ["Heater", "Vessel", "Cooler"],
        notes        = "THF-water azeotrope; BIP injection required",
    ),
    BenchmarkCase(
        case_id      = "H05",
        description  = "Process an n-hexane and ethanol mixture: pump to 5 bar, "
                       "heat to 70°C, flash to split vapour and liquid, "
                       "then split the liquid into two product streams",
        compounds    = ["n-hexane", "ethanol"],
        tier         = "hard",
        expected_pkg = "NRTL",
        expected_units = ["Pump", "Heater", "Vessel", "Splitter"],
        notes        = "Splitter insertion; hexane-ethanol heterogeneous azeotrope",
    ),
]


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark(
    orchestrator,
    cases:     list[BenchmarkCase] | None = None,
    ablation:  str = "full",
    verbose:   bool = False,
) -> tuple[list[CaseResult], BenchmarkMetrics]:
    """
    Run all benchmark cases through an OrchestratorV2 instance.
    Returns (case_results, aggregate_metrics).
    """
    from agents import llm as _llm

    cases   = cases or BENCHMARK_CASES
    results: list[CaseResult] = []

    for case in cases:
        t0 = time.time()
        call_before = getattr(_llm, "get_call_count", lambda: 0)()
        try:
            pipeline_result = orchestrator.run(
                description = case.description,
                compounds   = case.compounds,
            )
            elapsed = time.time() - t0
            call_after = getattr(_llm, "get_call_count", lambda: 0)()

            result = CaseResult(
                case_id           = case.case_id,
                outcome           = pipeline_result.outcome,
                valid_ir          = pipeline_result.ir_valid,
                valid_json        = pipeline_result.json_valid,
                converged         = getattr(pipeline_result, "converged", False),
                repair_iterations = len(pipeline_result.iterations),
                llm_calls         = call_after - call_before,
                elapsed_s         = elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            result  = CaseResult(
                case_id           = case.case_id,
                outcome           = "EXCEPTION",
                valid_ir          = False,
                valid_json        = False,
                converged         = False,
                repair_iterations = 0,
                llm_calls         = 0,
                elapsed_s         = elapsed,
                warnings          = [str(exc)],
            )
        results.append(result)
        if verbose:
            print(f"[{case.case_id}/{case.tier}] {result.outcome} "
                  f"ir={result.valid_ir} json={result.valid_json} "
                  f"conv={result.converged} repairs={result.repair_iterations} "
                  f"t={result.elapsed_s:.1f}s")

    metrics = compute_metrics(results, ablation)
    return results, metrics


# ── Baseline runner ────────────────────────────────────────────────────────────

def run_baseline(
    cases:   list[BenchmarkCase] | None = None,
    model:   str | None = None,
    verbose: bool = False,
) -> tuple[list[CaseResult], BenchmarkMetrics]:
    """
    Single-agent baseline: one LLM call produces the full flowsheet JSON,
    which is then validated directly (no IR graph, no repair loop).

    Used to measure the delta between our multi-agent system and naive LLM.
    """
    from agents.llm import chat, DEFAULT_MODEL
    from agents import schema as _schema

    model  = model or DEFAULT_MODEL
    cases  = cases or BENCHMARK_CASES
    results: list[CaseResult] = []

    _BASELINE_SYSTEM = """\
Generate a DWSIM flowsheet JSON for the given chemical process.
Return ONLY a JSON object matching this schema — no explanation, no markdown.

{
  "compounds": ["name1", "name2"],
  "property_package": "<Raoult's Law|NRTL|UNIQUAC|Peng-Robinson|Soave-Redlich-Kwong|Lee-Kesler-Plöcker>",
  "binary_parameters": [],
  "units": [
    {"tag": "HT-01", "type": "Heater", "T_out": 373.0}
  ],
  "streams": [
    {"tag": "FEED", "src": null, "dst": "HT-01",
     "T": 298.15, "P": 101325.0, "flow": 1.0,
     "composition": {"compound": 1.0}, "is_feed": true}
  ],
  "connections": [["FEED", "HT-01", 0, 0]]
}

All temperatures in Kelvin. All pressures in Pascals."""

    for case in cases:
        t0 = time.time()
        prompt = (
            f"Process: {case.description}\n"
            f"Compounds: {', '.join(case.compounds)}"
        )
        valid_json = False
        try:
            raw  = chat(prompt, system=_BASELINE_SYSTEM, model=model, max_tokens=2048)
            data = _parse_json(raw)
            errs = _schema.validate(data) if hasattr(_schema, "validate") else []
            valid_json = len(errs) == 0
        except Exception:
            valid_json = False

        elapsed = time.time() - t0
        results.append(CaseResult(
            case_id           = case.case_id,
            outcome           = "PASS" if valid_json else "INVALID_JSON",
            valid_ir          = False,   # baseline has no IR layer
            valid_json        = valid_json,
            converged         = False,   # no execution in baseline
            repair_iterations = 0,
            llm_calls         = 1,
            elapsed_s         = elapsed,
        ))
        if verbose:
            print(f"[{case.case_id}] baseline valid_json={valid_json} t={elapsed:.1f}s")

    metrics = compute_metrics(results, ablation_mode="baseline")
    return results, metrics


# ── Statistical runner ─────────────────────────────────────────────────────────

def run_statistical(
    orchestrator,
    case:      BenchmarkCase,
    n_runs:    int = 5,
) -> dict:
    """
    Run one case N times through the orchestrator.
    Returns a dict with mean, std, and min/max for key metrics.
    """
    case_results = []
    for _ in range(n_runs):
        results, _ = run_benchmark(orchestrator, cases=[case])
        if results:
            case_results.append(results[0])

    def _stat(values):
        return {
            "mean": statistics.mean(values),
            "std":  statistics.stdev(values) if len(values) > 1 else 0.0,
            "min":  min(values),
            "max":  max(values),
        }

    return {
        "case_id":   case.case_id,
        "n_runs":    n_runs,
        "converged": _stat([int(r.converged) for r in case_results]),
        "valid_ir":  _stat([int(r.valid_ir)  for r in case_results]),
        "repairs":   _stat([r.repair_iterations for r in case_results]),
        "llm_calls": _stat([r.llm_calls       for r in case_results]),
        "elapsed_s": _stat([r.elapsed_s        for r in case_results]),
    }


# ── Comparison helper ──────────────────────────────────────────────────────────

def compare(
    system_metrics:   BenchmarkMetrics,
    baseline_metrics: BenchmarkMetrics,
) -> dict:
    """
    Compute delta metrics: system − baseline.
    Positive values mean the multi-agent system is better.
    """
    return {
        "delta_valid_json":  system_metrics.pct_valid_json  - baseline_metrics.pct_valid_json,
        "delta_converged":   system_metrics.pct_converged   - baseline_metrics.pct_converged,
        "delta_repair_iter": system_metrics.avg_repair_iters - 0,  # baseline has 0 by design
        "system_valid_json": system_metrics.pct_valid_json,
        "baseline_valid_json": baseline_metrics.pct_valid_json,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)
