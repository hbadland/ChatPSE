"""
Benchmark runner for the v2 pipeline.

Provides:
  BENCHMARK_CASES   — 60 test cases (imported from benchmark_cases.py)
  BenchmarkCase     — dataclass (imported from benchmark_cases.py)
  run_benchmark()   — run cases through an OrchestratorV2 instance
  run_baseline()    — single-agent baseline: one LLM call → JSON → validate
  compare()         — compute delta metrics vs baseline
  run_statistical() — repeat N runs of one case, report mean ± std
  check_expected()  — verify a pipeline result meets BenchmarkCase expectations
"""
from __future__ import annotations

import json
import re
import statistics
import time
from typing import Optional

from eval.metrics import CaseResult, BenchmarkMetrics, compute_metrics
from eval.benchmark_cases import (
    BenchmarkCase, BENCHMARK_CASES, CASES_BY_ID, CASE_TIERS,   # noqa: F401
    EASY_CASES, MEDIUM_CASES, HARD_CASES,
    UNDERSPEC_CASES, AMBIGUOUS_CASES, ADVERSARIAL_CASES, EDGE_CASES,
)


# ── Expected-property check ────────────────────────────────────────────────────

def check_expected(
    case:            BenchmarkCase,
    pipeline_result,
    graph = None,
) -> dict:
    """
    Verify a pipeline result against the case's expected properties.
    Returns a dict of {check_name: bool}.
    """
    checks: dict[str, bool] = {}

    # Package check (skip if expected_pkg is empty)
    if case.expected_pkg and graph is not None:
        checks["pkg_correct"] = (
            getattr(graph, "property_package", "") == case.expected_pkg)

    # Unit subset check
    if case.expected_units and graph is not None:
        actual_types = {u.unit_type for u in graph.units()}
        # For repeated types (e.g., two Compressors) check count
        from collections import Counter
        expected_counter = Counter(case.expected_units)
        actual_counter   = Counter(actual_types)   # set: at-least-one check
        checks["units_present"] = all(
            actual_counter.get(ut, 0) >= 1
            for ut in expected_counter
        )
    elif case.expected_units:
        checks["units_present"] = False   # no graph available

    # Adversarial cases: expect the pipeline to flag issues, not crash
    if case.expect_failure:
        outcome = getattr(pipeline_result, "outcome", "UNKNOWN")
        checks["flagged_gracefully"] = outcome not in ("PASS",)

    return checks


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark(
    orchestrator,
    cases:    list[BenchmarkCase] | None = None,
    ablation: str  = "full",
    verbose:  bool = False,
    check_expectations: bool = False,
) -> tuple[list[CaseResult], BenchmarkMetrics]:
    """
    Run benchmark cases through an OrchestratorV2 instance.
    Returns (case_results, aggregate_metrics).
    """
    from agents import llm as _llm

    cases   = cases or BENCHMARK_CASES
    results: list[CaseResult] = []

    for case in cases:
        t0          = time.time()
        call_before = getattr(_llm, "get_call_count", lambda: 0)()
        warnings: list[str] = []

        try:
            pr = orchestrator.run(
                description = case.description,
                compounds   = case.compounds,
            )
            elapsed    = time.time() - t0
            call_after = getattr(_llm, "get_call_count", lambda: 0)()

            # Expected-property verification
            if check_expectations:
                graph = getattr(pr, "graph", None)
                chks  = check_expected(case, pr, graph)
                for k, v in chks.items():
                    if not v:
                        warnings.append(f"EXPECTED_FAIL:{k}")

            result = CaseResult(
                case_id            = case.case_id,
                outcome            = pr.outcome,
                valid_ir           = getattr(pr, "ir_valid",   False),
                valid_json         = getattr(pr, "json_valid", False),
                converged          = getattr(pr, "converged",  False),
                repair_iterations  = len(getattr(pr, "iterations", [])),
                llm_calls          = call_after - call_before,
                elapsed_s          = elapsed,
                warnings           = warnings,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            result  = CaseResult(
                case_id            = case.case_id,
                outcome            = "EXCEPTION",
                valid_ir           = False,
                valid_json         = False,
                converged          = False,
                repair_iterations  = 0,
                llm_calls          = 0,
                elapsed_s          = elapsed,
                warnings           = [f"EXCEPTION: {exc}"],
            )

        results.append(result)
        if verbose:
            warn_str = f" [{'; '.join(results[-1].warnings[:2])}]" if results[-1].warnings else ""
            print(f"[{case.case_id}/{case.tier}] {result.outcome}"
                  f" ir={result.valid_ir} json={result.valid_json}"
                  f" conv={result.converged} repairs={result.repair_iterations}"
                  f" t={result.elapsed_s:.1f}s{warn_str}")

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
    validated directly.  No IR graph, no repair loop.
    """
    from agents.llm import chat, DEFAULT_MODEL
    from agents import schema as _schema

    model  = model or DEFAULT_MODEL
    cases  = cases or BENCHMARK_CASES
    results: list[CaseResult] = []

    _BASELINE_SYSTEM = """\
Generate a DWSIM flowsheet JSON for the given chemical process.
Return ONLY a JSON object — no explanation, no markdown.

{
  "compounds": ["name1", "name2"],
  "property_package": "<Raoult's Law|NRTL|UNIQUAC|Peng-Robinson|Soave-Redlich-Kwong|Lee-Kesler-Plöcker>",
  "binary_parameters": [],
  "units": [{"tag": "HT-01", "type": "Heater", "T_out": 373.0}],
  "streams": [
    {"tag": "FEED", "src": null, "dst": "HT-01",
     "T": 298.15, "P": 101325.0, "flow": 1.0,
     "composition": {"compound": 1.0}, "is_feed": true}
  ],
  "connections": [["FEED", "HT-01", 0, 0]]
}

All temperatures in Kelvin. All pressures in Pascals."""

    for case in cases:
        t0         = time.time()
        valid_json = False
        warnings: list[str] = []
        try:
            raw  = chat(
                f"Process: {case.description}\nCompounds: {', '.join(case.compounds)}",
                system=_BASELINE_SYSTEM, model=model, max_tokens=2048)
            data = _parse_json(raw)
            errs = _schema.validate(data) if hasattr(_schema, "validate") else []
            valid_json = len(errs) == 0
            if errs:
                warnings = [str(e)[:80] for e in errs[:3]]
        except Exception as exc:
            warnings = [f"EXCEPTION: {exc}"]

        elapsed = time.time() - t0
        results.append(CaseResult(
            case_id           = case.case_id,
            outcome           = "PASS" if valid_json else "INVALID_JSON",
            valid_ir          = False,
            valid_json        = valid_json,
            converged         = False,
            repair_iterations = 0,
            llm_calls         = 1,
            elapsed_s         = elapsed,
            warnings          = warnings,
        ))
        if verbose:
            print(f"[{case.case_id}] baseline json={valid_json} t={elapsed:.1f}s")

    return results, compute_metrics(results, "baseline")


# ── Statistical runner ─────────────────────────────────────────────────────────

def run_statistical(
    orchestrator,
    case:   BenchmarkCase,
    n_runs: int = 5,
) -> dict:
    """Run one case N times and return mean±std statistics."""
    case_results = []
    for _ in range(n_runs):
        rs, _ = run_benchmark(orchestrator, cases=[case])
        if rs:
            case_results.append(rs[0])

    def _stat(values):
        m = statistics.mean(values) if values else 0.0
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": m, "std": s, "min": min(values, default=0),
                "max": max(values, default=0)}

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
    return {
        "delta_valid_json":  system_metrics.pct_valid_json  - baseline_metrics.pct_valid_json,
        "delta_converged":   system_metrics.pct_converged   - baseline_metrics.pct_converged,
        "system_valid_json": system_metrics.pct_valid_json,
        "baseline_valid_json": baseline_metrics.pct_valid_json,
        "n_system":          system_metrics.n_total,
        "n_baseline":        baseline_metrics.n_total,
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
