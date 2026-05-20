"""
CCS Benchmark Suite — evaluation infrastructure for the NeurIPS paper.

Quick start:
    from benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner(model="qwen3:14b")

    # Run all tiers, full CCS system
    results = runner.run_all()
    print(results.to_markdown())

    # Run ablation study (compares all 5 modes)
    ablation = runner.run_ablation(tiers=["easy", "medium", "hard"])
    from benchmark.runner import ablation_table
    print(ablation_table(ablation))

    # Run single case
    r = runner.run_case("HARD_01")
    print(r.metrics)
"""
from benchmark.case_schema import (
    BenchmarkCaseSpec, load_all, load_tier, load_by_id, case_index, summary,
    TIERS,
)
from benchmark.metrics import RunMetrics, AggregateMetrics, extract_metrics, aggregate
from benchmark.logger import RunLog, extract_run_log
from benchmark.physics_eval import run_physics_checks
from benchmark.ablation import AblationConfig, CONFIGS, ABLATION_MODES, apply_ablation
from benchmark.runner import BenchmarkRunner, BenchmarkRunSet, CaseRunResult, ablation_table

__all__ = [
    "BenchmarkCaseSpec", "load_all", "load_tier", "load_by_id",
    "case_index", "summary", "TIERS",
    "RunMetrics", "AggregateMetrics", "extract_metrics", "aggregate",
    "RunLog", "extract_run_log",
    "run_physics_checks",
    "AblationConfig", "CONFIGS", "ABLATION_MODES", "apply_ablation",
    "BenchmarkRunner", "BenchmarkRunSet", "CaseRunResult", "ablation_table",
]
