"""
Structured failure mode logging.

Every failed (or partially failed) benchmark case is classified into one or
more FailureMode categories.  FailureLog aggregates these records and supports
per-split breakdowns, worst-case ranking, and formatted output.

Failure taxonomy (ordered by root cause priority):
  EXCEPTION          — unhandled Python exception; pipeline crashed
  INVALID_IR         — schema or graph-level validation error (no property_package,
                       disconnected stream, port violation)
  INVALID_JSON       — IR cannot be serialised to DWSIM JSON
  PHYSICS_VIOLATION  — physics-level critical issue (T out of range, wrong phase,
                       infeasible unit config)
  MISSING_PARAMS     — required unit params still unset after Stage 3
  REPAIR_EXHAUSTED   — max iterations reached without converging
  ESCALATED          — RepairAgent returned HUMAN (cannot auto-repair)
  NO_CONVERGENCE     — DWSIM execution did not solve
  PASS               — no failure (all checks passed)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from eval.metrics import CaseResult
from eval.benchmark_cases import BenchmarkCase
from eval.dataset import split_of, is_frozen


# ── Failure taxonomy ───────────────────────────────────────────────────────────

FAILURE_MODES = [
    "EXCEPTION",
    "INVALID_IR",
    "INVALID_JSON",
    "PHYSICS_VIOLATION",
    "MISSING_PARAMS",
    "REPAIR_EXHAUSTED",
    "ESCALATED",
    "NO_CONVERGENCE",
    "PASS",
]


def classify_failure(result: CaseResult, case: BenchmarkCase) -> list[str]:
    """
    Classify a CaseResult into one or more FailureMode strings.
    Returns ["PASS"] for a fully successful result.
    """
    if result.outcome == "EXCEPTION":
        return ["EXCEPTION"]

    modes: list[str] = []

    if not result.valid_ir:
        modes.append("INVALID_IR")

    if not result.valid_json:
        modes.append("INVALID_JSON")

    # Infer physics violations from warnings
    combined_warnings = " ".join(result.warnings).upper()
    if "PHYSICS" in combined_warnings or "UNPHYSICAL" in combined_warnings:
        modes.append("PHYSICS_VIOLATION")

    if "MISSING_PARAM" in combined_warnings or "REQUIRED_PARAM" in combined_warnings:
        modes.append("MISSING_PARAMS")

    if result.outcome == "HUMAN" or "ESCALATED" in combined_warnings:
        modes.append("ESCALATED")
    elif result.outcome in ("MAX_ITER", "MAX_ITERATIONS"):
        modes.append("REPAIR_EXHAUSTED")
    elif not result.converged and result.outcome not in ("PASS",):
        modes.append("NO_CONVERGENCE")

    return modes if modes else ["PASS"]


# ── FailureRecord ──────────────────────────────────────────────────────────────

@dataclass
class FailureRecord:
    case_id:      str
    split:        str
    tier:         str
    frozen:       bool
    modes:        list[str]   # one or more from FAILURE_MODES
    outcome:      str
    repair_iters: int
    warnings:     list[str]

    @property
    def is_pass(self) -> bool:
        return self.modes == ["PASS"]

    @property
    def primary_mode(self) -> str:
        for m in FAILURE_MODES:   # ordered by severity
            if m in self.modes:
                return m
        return "PASS"


# ── FailureLog ────────────────────────────────────────────────────────────────

class FailureLog:
    """Collects and analyses failure records across a benchmark run."""

    def __init__(self) -> None:
        self._records: list[FailureRecord] = []

    def add(self, result: CaseResult, case: BenchmarkCase) -> None:
        modes = classify_failure(result, case)
        self._records.append(FailureRecord(
            case_id      = result.case_id,
            split        = split_of(result.case_id),
            tier         = case.tier,
            frozen       = is_frozen(result.case_id),
            modes        = modes,
            outcome      = result.outcome,
            repair_iters = result.repair_iterations,
            warnings     = result.warnings[:5],
        ))

    def records(self, split: Optional[str] = None) -> list[FailureRecord]:
        if split is None:
            return list(self._records)
        return [r for r in self._records if r.split == split]

    def failures(self, split: Optional[str] = None) -> list[FailureRecord]:
        return [r for r in self.records(split) if not r.is_pass]

    # ── Aggregate counts ───────────────────────────────────────────────────────

    def mode_counts(
        self, split: Optional[str] = None
    ) -> dict[str, int]:
        """Count primary failure mode per case (one bucket per case)."""
        counts: dict[str, int] = Counter()
        for r in self.records(split):
            counts[r.primary_mode] += 1
        return dict(counts)

    def all_mode_counts(
        self, split: Optional[str] = None
    ) -> dict[str, int]:
        """Count all failure modes (a case can contribute to multiple)."""
        counts: dict[str, int] = Counter()
        for r in self.records(split):
            for m in r.modes:
                counts[m] += 1
        return dict(counts)

    def by_split(self) -> dict[str, dict[str, int]]:
        """Return mode counts broken down by split."""
        splits = ["dev", "holdout", "stress"]
        return {s: self.mode_counts(s) for s in splits}

    def by_tier(self) -> dict[str, dict[str, int]]:
        """Return mode counts broken down by tier."""
        by_tier: dict[str, list[FailureRecord]] = defaultdict(list)
        for r in self._records:
            by_tier[r.tier].append(r)
        return {
            tier: dict(Counter(rec.primary_mode for rec in recs))
            for tier, recs in by_tier.items()
        }

    def frozen_failures(self) -> list[FailureRecord]:
        """Failures in frozen cases — always investigated immediately."""
        return [r for r in self._records if r.frozen and not r.is_pass]

    def worst_cases(self, n: int = 10) -> list[FailureRecord]:
        """Return n cases with the most repair iterations among failures."""
        failures = self.failures()
        return sorted(failures, key=lambda r: r.repair_iters, reverse=True)[:n]

    def pass_rate(self, split: Optional[str] = None) -> float:
        recs = self.records(split)
        if not recs:
            return 0.0
        return sum(1 for r in recs if r.is_pass) / len(recs)

    def summary(self) -> "FailureSummary":
        return FailureSummary(log=self)


@dataclass
class FailureSummary:
    """Pre-computed summary of failure log for reporting."""
    log: FailureLog

    def overall_counts(self) -> dict[str, int]:
        return self.log.mode_counts()

    def by_split(self) -> dict[str, dict[str, int]]:
        return self.log.by_split()

    def by_tier(self) -> dict[str, dict[str, int]]:
        return self.log.by_tier()

    def n_frozen_failures(self) -> int:
        return len(self.log.frozen_failures())

    def pass_rates(self) -> dict[str, float]:
        return {
            "all":     self.log.pass_rate(),
            "dev":     self.log.pass_rate("dev"),
            "holdout": self.log.pass_rate("holdout"),
            "stress":  self.log.pass_rate("stress"),
        }
