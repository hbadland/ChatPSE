"""
Train/test splits for generalisation evaluation (Item 3).

Split design rationale:
  TRAIN — processes with common unit type combinations seen in standard
          chemical engineering textbooks (heater, flash, compressor basics).
          Models may have encountered these during pre-training.
  TEST  — novel combinations or less common configurations the model is
          unlikely to have seen together (multi-stage compression, ternary
          azeotrope, cryogenic, adversarial, edge thermodynamic).

The split is defined by case_id, not by random sampling, so results
are reproducible without a random seed.

Generalisation gap = test_metric - train_metric.
A negative gap indicates the system generalises poorly to novel cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from eval.benchmark_cases import BenchmarkCase, BENCHMARK_CASES, CASES_BY_ID
from eval.metrics import CaseResult, BenchmarkMetrics, compute_metrics


# ── Split definition ────────────────────────────────────────────────────────────
# Train: common processes (easy + straightforward medium)
# Test:  novel/hard/underspec/adversarial/edge

_TRAIN_IDS: frozenset[str] = frozenset({
    # Easy (all)
    "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10",
    # Medium (common flash processes)
    "M01", "M02", "M03", "M05", "M07", "M10",
})

_TEST_IDS: frozenset[str] = frozenset({
    # Medium (less common)
    "M04", "M06", "M08", "M09", "M11", "M12",
    # Hard (all)
    "H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10",
    # Underspecified (all)
    "U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08",
    # Ambiguous (all)
    "AMB01", "AMB02", "AMB03", "AMB04", "AMB05", "AMB06",
    # Adversarial (all)
    "ADV01", "ADV02", "ADV03", "ADV04", "ADV05", "ADV06", "ADV07",
    # Edge thermodynamic (all)
    "EDGE01", "EDGE02", "EDGE03", "EDGE04", "EDGE05", "EDGE06", "EDGE07",
})


@dataclass
class BenchmarkSplit:
    train: list[BenchmarkCase]
    test:  list[BenchmarkCase]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def __repr__(self) -> str:
        return f"BenchmarkSplit(train={self.n_train}, test={self.n_test})"


@dataclass
class SplitResult:
    """Metrics computed on each partition plus the generalisation gap."""
    train_metrics: BenchmarkMetrics
    test_metrics:  BenchmarkMetrics
    split:         BenchmarkSplit

    # Generalisation gaps (test - train; negative = harder on test set)
    gap_valid_ir:   float = 0.0
    gap_valid_json: float = 0.0
    gap_converged:  float = 0.0

    def __post_init__(self) -> None:
        self.gap_valid_ir   = (self.test_metrics.pct_valid_ir
                               - self.train_metrics.pct_valid_ir)
        self.gap_valid_json = (self.test_metrics.pct_valid_json
                               - self.train_metrics.pct_valid_json)
        self.gap_converged  = (self.test_metrics.pct_converged
                               - self.train_metrics.pct_converged)

    def summary(self) -> str:
        lines = [
            "=== Generalisation Evaluation ===",
            f"  train cases  : {self.split.n_train}",
            f"  test cases   : {self.split.n_test}",
            "",
            "             TRAIN        TEST       GAP",
            f"  valid IR : {self.train_metrics.pct_valid_ir:.1%}       "
            f"{self.test_metrics.pct_valid_ir:.1%}      "
            f"{self.gap_valid_ir:+.1%}",
            f"  valid JSON: {self.train_metrics.pct_valid_json:.1%}       "
            f"{self.test_metrics.pct_valid_json:.1%}      "
            f"{self.gap_valid_json:+.1%}",
            f"  converged: {self.train_metrics.pct_converged:.1%}       "
            f"{self.test_metrics.pct_converged:.1%}      "
            f"{self.gap_converged:+.1%}",
        ]
        return "\n".join(lines)


def get_split(cases: list[BenchmarkCase] | None = None) -> BenchmarkSplit:
    """Return the standard train/test split."""
    cases = cases or BENCHMARK_CASES
    train = [c for c in cases if c.case_id in _TRAIN_IDS]
    test  = [c for c in cases if c.case_id in _TEST_IDS]
    # Any case not in either set goes to test
    assigned = _TRAIN_IDS | _TEST_IDS
    test += [c for c in cases if c.case_id not in assigned]
    return BenchmarkSplit(train=train, test=test)


def evaluate_split(
    orchestrator,
    split:   BenchmarkSplit | None = None,
    verbose: bool = False,
) -> SplitResult:
    """
    Run train and test partitions separately and compute generalisation metrics.
    """
    from eval.benchmark import run_benchmark

    split   = split or get_split()
    _, tm   = run_benchmark(orchestrator, cases=split.train,
                            ablation="split_train", verbose=verbose)
    _, testm = run_benchmark(orchestrator, cases=split.test,
                             ablation="split_test", verbose=verbose)
    return SplitResult(train_metrics=tm, test_metrics=testm, split=split)


def tier_breakdown(
    results: list[CaseResult],
    cases:   list[BenchmarkCase] | None = None,
) -> dict[str, BenchmarkMetrics]:
    """
    Compute metrics stratified by tier (easy/medium/hard/…).
    Returns {tier: BenchmarkMetrics}.
    """
    cases    = cases or BENCHMARK_CASES
    id_to_tier = {c.case_id: c.tier for c in cases}
    by_tier: dict[str, list[CaseResult]] = {}
    for r in results:
        t = id_to_tier.get(r.case_id, "unknown")
        by_tier.setdefault(t, []).append(r)
    return {tier: compute_metrics(rs, ablation_mode=tier)
            for tier, rs in by_tier.items()}
