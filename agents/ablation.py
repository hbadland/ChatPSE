"""
Ablation harness for the multi-agent flowsheet system.

Runs the pipeline with individual agents/stages disabled, measuring each
component's contribution to success rate and LLM cost.  Supports the
paper's claim about the two-stage deterministic-before-LLM design.

Conditions
──────────
  FULL             Baseline — full pipeline, no mocking
  NO_BASIS         BasisAgent skipped; description passed raw to Planner
  NO_PHYSICS       physics_validate returns [] everywhere
  LLM_ONLY_CRITIC  Critic Stage 1 skipped; LLM always called for critique
  LLM_ONLY_REFINER Refiner Stage 1 skipped; LLM always called for fixes
  NO_CALIBRATION   CalibrationAgent always fails; PARAM_MISSING → ThermoAgent only

Usage
─────
    python agents/ablation.py
    python agents/ablation.py --model claude-sonnet-4-6
    python agents/ablation.py --real-executor
"""
from __future__ import annotations
import sys
import time
import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.basis import BasisResult
from agents.orchestrator import Orchestrator, OrchestratorResult
from agents.executor import ExecutionResult, StreamResult
from agents.llm import reset_call_count, get_call_count
from agents.benchmark_pipeline import PIPELINE_TEST_CASES, PipelineTestCase

# ── Ablation condition names ───────────────────────────────────────────────────

CONDITIONS = [
    "FULL",
    "NO_BASIS",
    "NO_PHYSICS",
    "LLM_ONLY_CRITIC",
    "LLM_ONLY_REFINER",
    "NO_CALIBRATION",
]


# ── Condition context managers ────────────────────────────────────────────────

@contextmanager
def _condition_full() -> Generator:
    yield


@contextmanager
def _condition_no_basis() -> Generator:
    """Pass description through as-is; no compound normalisation."""
    def _passthrough_identify(self, description: str,
                              feedback=None) -> BasisResult:
        return BasisResult(
            compound_map={},
            dwsim_compounds=[],
            normalised_description=description,
            stage="LOOKUP",
            stage1_count=0,
            success=True,
            warnings=["[ABLATION] BasisAgent disabled — raw description forwarded."],
        )

    with patch("agents.basis.BasisAgent.identify", _passthrough_identify):
        yield


@contextmanager
def _condition_no_physics() -> Generator:
    """physics_validate returns [] — all static physics pre-checks suppressed."""
    with patch("agents.physics_check.physics_validate", return_value=[]):
        # Also patch the imports inside critic and refiner (they import the name
        # directly, so we need to patch the reference in each module).
        with patch("agents.critic.physics_validate", return_value=[]):
            with patch("agents.refiner.physics_validate", return_value=[]):
                yield


@contextmanager
def _condition_llm_only_critic() -> Generator:
    """Critic Stage 1 returns no signals — always escalates to Stage 2 LLM."""
    with patch("agents.critic._run_stage1", return_value=[]):
        yield


@contextmanager
def _condition_llm_only_refiner() -> Generator:
    """Refiner Stage 1 applies no fixes — always escalates to Stage 2 LLM."""
    with patch("agents.refiner._apply_deterministic_fixes",
               side_effect=lambda fs, report: (copy.deepcopy(fs), [])):
        yield


@contextmanager
def _condition_no_calibration() -> Generator:
    """CalibrationAgent always reports not_found — forces THERMO fallback.

    Measures the contribution of BIP retrieval + injection by comparing FULL
    vs NO_CALIBRATION on test cases that require NRTL/UNIQUAC parameters not
    in DWSIM's built-in database (PARAM_MISSING scenario).
    """
    from agents.calibration import CalibrationResult
    with patch(
        "agents.orchestrator.CalibrationAgent.run",
        side_effect=lambda fs: CalibrationResult(
            success=False,
            updated_flowsheet=fs,
            pairs_found=[],
            pairs_missing=[],
            parameters_injected=[],
            notes=["[ABLATION:NO_CALIBRATION] BIP retrieval disabled"],
        ),
    ):
        yield


_CONDITION_CTX = {
    "FULL":              _condition_full,
    "NO_BASIS":          _condition_no_basis,
    "NO_PHYSICS":        _condition_no_physics,
    "LLM_ONLY_CRITIC":   _condition_llm_only_critic,
    "LLM_ONLY_REFINER":  _condition_llm_only_refiner,
    "NO_CALIBRATION":    _condition_no_calibration,
}


# ── Mock executor (mirrors benchmark_pipeline) ────────────────────────────────

def _mock_execution_result(flowsheet: dict) -> ExecutionResult:
    streams  = flowsheet.get("streams", [])
    conns    = flowsheet.get("connections", [])
    has_in   = {c[1] for c in conns if len(c) >= 2}
    compounds = flowsheet.get("compounds", [])
    comp = {c: 1.0 / len(compounds) for c in compounds} if compounds else {}

    stream_results: dict[str, StreamResult] = {}
    for s in streams:
        tag = s["tag"]
        stream_results[tag] = StreamResult(
            tag=tag,
            T_K=s.get("T", 298.15),
            P_Pa=s.get("P", 101325.0),
            flow_mol_s=s.get("flow", 1.0),
            composition=s.get("composition", comp) or comp,
            is_feed=(tag not in has_in),
        )
    return ExecutionResult(solved=True, stream_results=stream_results)


# ── Per-case result ────────────────────────────────────────────────────────────

@dataclass
class AblationCaseResult:
    name:        str
    outcome:     str
    success:     bool          # outcome == expected_outcome
    llm_calls:   int
    iterations:  int
    elapsed_s:   float
    error:       str | None = None


# ── Per-condition summary ──────────────────────────────────────────────────────

@dataclass
class ConditionSummary:
    condition:       str
    n_cases:         int
    success_rate:    float
    mean_llm_calls:  float
    mean_iterations: float
    mean_time_s:     float
    case_results:    list[AblationCaseResult] = field(default_factory=list)


# ── Single-condition runner ────────────────────────────────────────────────────

def _run_condition(
    condition: str,
    test_cases: list[PipelineTestCase],
    model: str,
    use_real_executor: bool,
) -> ConditionSummary:
    case_results: list[AblationCaseResult] = []
    ctx = _CONDITION_CTX[condition]

    for tc in test_cases:
        reset_call_count()
        t0 = time.time()
        outcome = "ERROR"
        error_msg: str | None = None

        try:
            orch = Orchestrator(model=model,
                                max_iterations=tc.max_iterations_allowed)
            with ctx():
                if use_real_executor:
                    result: OrchestratorResult = orch.run(tc.description)
                else:
                    with patch(
                        "agents.executor.Executor.run",
                        side_effect=lambda fs: _mock_execution_result(fs),
                    ):
                        result = orch.run(tc.description)
            outcome = result.outcome
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        case_results.append(AblationCaseResult(
            name=tc.name,
            outcome=outcome,
            success=(outcome == tc.expected_outcome),
            llm_calls=get_call_count(),
            iterations=0 if error_msg else len(result.iterations),  # type: ignore[possibly-undefined]
            elapsed_s=time.time() - t0,
            error=error_msg,
        ))

    n = len(case_results)
    return ConditionSummary(
        condition=condition,
        n_cases=n,
        success_rate=sum(1 for r in case_results if r.success) / n if n else 0.0,
        mean_llm_calls=sum(r.llm_calls  for r in case_results) / n if n else 0.0,
        mean_iterations=sum(r.iterations for r in case_results) / n if n else 0.0,
        mean_time_s=sum(r.elapsed_s  for r in case_results) / n if n else 0.0,
        case_results=case_results,
    )


# ── Main runner ────────────────────────────────────────────────────────────────

def run_ablation(
    model: str = "gemini-2.5-flash",
    test_cases: list[PipelineTestCase] | None = None,
    use_real_executor: bool = False,
) -> list[ConditionSummary]:
    """
    Run all ablation conditions on the provided test cases.

    Parameters
    ----------
    model             : LLM model for all agents
    test_cases        : defaults to PIPELINE_TEST_CASES from benchmark_pipeline
    use_real_executor : if False, Executor.run is patched to avoid DWSIM
    """
    if test_cases is None:
        test_cases = PIPELINE_TEST_CASES

    summaries: list[ConditionSummary] = []
    for condition in CONDITIONS:
        print(f"  Running condition: {condition} ...", flush=True)
        summary = _run_condition(condition, test_cases, model, use_real_executor)
        summaries.append(summary)
        print(f"    success={summary.success_rate:.0%}  "
              f"llm={summary.mean_llm_calls:.1f}  "
              f"time={summary.mean_time_s:.1f}s")

    return summaries


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary_table(summaries: list[ConditionSummary], model: str) -> None:
    print(f"\n## Ablation Study — model: {model}\n")
    header = (f"| {'Condition':<20} | {'Success':>8} | {'LLM calls':>10} "
              f"| {'Iterations':>10} | {'Time(s)':>8} |")
    sep    = "|" + "-" * 22 + "|" + "-" * 10 + "|" + "-" * 12 + \
             "|" + "-" * 12 + "|" + "-" * 10 + "|"
    print(header)
    print(sep)
    for s in summaries:
        print(f"| {s.condition:<20} | {s.success_rate:>8.0%} | {s.mean_llm_calls:>10.1f} "
              f"| {s.mean_iterations:>10.1f} | {s.mean_time_s:>8.1f} |")
    print()


def _print_per_case_breakdown(summaries: list[ConditionSummary]) -> None:
    """Per-case breakdown: one table per condition, rows = test cases."""
    for s in summaries:
        print(f"\n### {s.condition}\n")
        print(f"| {'Test case':<48} | {'Outcome':<12} | {'OK':>2} "
              f"| {'LLM':>4} | {'Iter':>4} | {'Time':>6} |")
        print("|" + "-" * 50 + "|" + "-" * 14 + "|" + "-" * 4 +
              "|" + "-" * 6 + "|" + "-" * 6 + "|" + "-" * 8 + "|")
        for r in s.case_results:
            ok = "✓" if r.success else "✗"
            name_short = r.name[:48]
            print(f"| {name_short:<48} | {r.outcome:<12} | {ok:>2} "
                  f"| {r.llm_calls:>4} | {r.iterations:>4} | {r.elapsed_s:>6.1f} |")
            if r.error:
                print(f"|   ERROR: {r.error}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study harness")
    p.add_argument("--model",         default="gemini-2.5-flash",
                   help="LLM model to use for all agents")
    p.add_argument("--real-executor", action="store_true",
                   help="Use live DWSIM executor (requires DWSIM container)")
    p.add_argument("--verbose",       action="store_true",
                   help="Print per-case breakdown for each condition")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"Running ablation study with model={args.model}, "
          f"real_executor={args.real_executor}\n")
    summaries = run_ablation(
        model=args.model,
        use_real_executor=args.real_executor,
    )
    _print_summary_table(summaries, model=args.model)
    if args.verbose:
        _print_per_case_breakdown(summaries)
