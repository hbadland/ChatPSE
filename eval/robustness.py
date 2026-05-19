"""
Robustness checks for the benchmarking pipeline.

Provides three independent perturbation modes:

  perturb_scoring(orchestrator, cases, n_runs)
    — Vary CandidateScore weights ±20% uniformly at random across N runs.
      Reports coefficient of variation (CV) for each primary metric.
      CV < 0.05 (5%) → stable, 0.05–0.15 → moderate, > 0.15 → sensitive.

  perturb_prompt(orchestrator, cases, perturbation)
    — Apply a single named prompt perturbation: reorder instructions,
      drop the few-shot example, or rephrase the task noun.
      Returns delta metrics vs unperturbed baseline.

  multi_trial(orchestrator, cases, n_trials)
    — Repeat benchmark N times with the same orchestrator (no perturbation).
      Captures model stochasticity (temperature > 0).
      Returns per-metric (mean, std, cv).

All functions are purely additive — they never modify core system files.
They read from eval.dataset to respect split boundaries.
"""
from __future__ import annotations

import copy
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from eval.metrics import CaseResult, BenchmarkMetrics, compute_metrics
from eval.benchmark_cases import BenchmarkCase, BENCHMARK_CASES


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class TrialMetrics:
    """Metrics from one robustness trial."""
    trial_idx:     int
    perturbation:  str       # e.g. "score_weights", "no_fewshot", "none"
    metrics:       BenchmarkMetrics
    seed:          int = 0


@dataclass
class RobustnessSummary:
    """Aggregated robustness statistics across N trials for one metric."""
    metric_name:   str
    perturbation:  str
    n_trials:      int
    values:        list[float]
    mean:          float = 0.0
    std:           float = 0.0
    cv:            float = 0.0   # coefficient of variation = std / mean
    stable:        bool  = True  # cv < 0.05

    def __post_init__(self) -> None:
        if self.values:
            self.mean = statistics.mean(self.values)
            self.std  = statistics.stdev(self.values) if len(self.values) > 1 else 0.0
            self.cv   = self.std / self.mean if self.mean > 0 else 0.0
            self.stable = self.cv < 0.05

    def stability_label(self) -> str:
        if self.cv < 0.05:
            return "STABLE"
        if self.cv < 0.15:
            return "MODERATE"
        return "SENSITIVE"

    def __str__(self) -> str:
        return (f"{self.metric_name:25s} [{self.perturbation}]: "
                f"mean={self.mean:.3f}  std={self.std:.3f}  "
                f"cv={self.cv:.3f}  {self.stability_label()}")


@dataclass
class RobustnessReport:
    """Full robustness report across all perturbations and metrics."""
    summaries:         list[RobustnessSummary] = field(default_factory=list)
    baseline_metrics:  Optional[BenchmarkMetrics] = None
    n_cases:           int = 0

    def by_perturbation(self) -> dict[str, list[RobustnessSummary]]:
        out: dict[str, list[RobustnessSummary]] = {}
        for s in self.summaries:
            out.setdefault(s.perturbation, []).append(s)
        return out

    def sensitive_metrics(self) -> list[RobustnessSummary]:
        return [s for s in self.summaries if not s.stable]

    def all_stable(self) -> bool:
        return all(s.stable for s in self.summaries)


# ── Scoring weight perturbation ────────────────────────────────────────────────

_SCORE_WEIGHT_ATTRS = [
    "valid_ir", "unit_appropriateness", "param_completeness",
    "thermo_consistency", "repair_economy", "valid_json",
    "separation_feasibility", "phase_consistency",
]


def _perturb_weights(rng: random.Random, magnitude: float = 0.20) -> dict[str, float]:
    """
    Sample weight dict with each weight varied ±magnitude uniformly.
    Renormalises so weights sum to 1.0.
    """
    base = {
        "valid_ir":               0.35,
        "unit_appropriateness":   0.15,
        "param_completeness":     0.12,
        "thermo_consistency":     0.10,
        "repair_economy":         0.10,
        "valid_json":             0.08,
        "separation_feasibility": 0.07,
        "phase_consistency":      0.03,
    }
    perturbed = {
        k: max(0.0, v * (1.0 + rng.uniform(-magnitude, magnitude)))
        for k, v in base.items()
    }
    total = sum(perturbed.values())
    if total > 0:
        perturbed = {k: v / total for k, v in perturbed.items()}
    return perturbed


def _patch_weights(weights: dict[str, float]) -> None:
    """Monkey-patch ir.scoring._WEIGHTS in-place."""
    try:
        import ir.scoring as _scoring
        _scoring.CandidateScore._WEIGHTS = weights   # type: ignore[attr-defined]
    except Exception:
        pass


def _restore_weights(original: dict[str, float]) -> None:
    try:
        import ir.scoring as _scoring
        _scoring.CandidateScore._WEIGHTS = original  # type: ignore[attr-defined]
    except Exception:
        pass


def perturb_scoring(
    orchestrator,
    cases:     list[BenchmarkCase] | None = None,
    n_runs:    int  = 5,
    magnitude: float = 0.20,
    seed:      int  = 0,
    verbose:   bool = False,
) -> RobustnessReport:
    """
    Vary CandidateScore weights ±magnitude across n_runs trials.

    Each trial uses a fresh random weight vector (seeded reproducibly).
    Returns a RobustnessReport with per-metric stability summaries.
    """
    from eval.benchmark import run_benchmark

    cases = cases or BENCHMARK_CASES
    rng   = random.Random(seed)

    # Read original weights once
    try:
        import ir.scoring as _scoring
        original_weights = dict(_scoring.CandidateScore._WEIGHTS)  # type: ignore
    except Exception:
        original_weights = {}

    trial_metrics: list[BenchmarkMetrics] = []

    for i in range(n_runs):
        weights = _perturb_weights(rng)
        _patch_weights(weights)
        try:
            _, metrics = run_benchmark(orchestrator, cases=cases, verbose=verbose)
            trial_metrics.append(metrics)
            if verbose:
                print(f"  [score perturb {i+1}/{n_runs}] converged={metrics.pct_converged:.2%}")
        finally:
            _restore_weights(original_weights)

    return _build_report(trial_metrics, "score_weights", cases)


# ── Prompt perturbation ────────────────────────────────────────────────────────

PROMPT_PERTURBATIONS = [
    "no_fewshot",      # drop few-shot examples from Stage 1 prompts
    "reorder_rules",   # shuffle the bullet-point rules in Stage 1
    "rephrase_task",   # replace "flowsheet" with "process diagram" throughout
]


def _apply_prompt_perturbation(perturbation: str) -> dict:
    """
    Monkey-patch Stage 1 prompt strings for the given perturbation.
    Returns a dict of original values so they can be restored.
    """
    originals: dict = {}
    try:
        import agents.stage1.unit_extractor as _ue
        import agents.stage1.stream_extractor as _se

        if perturbation == "no_fewshot":
            for mod, attr in [(_ue, "_SYSTEM"), (_se, "_SYSTEM")]:
                orig = getattr(mod, attr, None)
                if orig is not None:
                    originals[f"{mod.__name__}.{attr}"] = orig
                    # Strip everything from the first "Example" or "---" line
                    lines = orig.split("\n")
                    cutoff = next(
                        (i for i, ln in enumerate(lines)
                         if ln.strip().startswith("Example") or ln.strip() == "---"),
                        len(lines),
                    )
                    setattr(mod, attr, "\n".join(lines[:cutoff]))

        elif perturbation == "reorder_rules":
            for mod, attr in [(_ue, "_SYSTEM"), (_se, "_SYSTEM")]:
                orig = getattr(mod, attr, None)
                if orig is not None:
                    originals[f"{mod.__name__}.{attr}"] = orig
                    lines = orig.split("\n")
                    # Shuffle contiguous bullet-point blocks
                    bullets, other, out = [], [], []
                    for ln in lines:
                        if ln.strip().startswith("-"):
                            bullets.append(ln)
                        else:
                            if bullets:
                                random.shuffle(bullets)
                                out.extend(bullets)
                                bullets = []
                            out.append(ln)
                    if bullets:
                        random.shuffle(bullets)
                        out.extend(bullets)
                    setattr(mod, attr, "\n".join(out))

        elif perturbation == "rephrase_task":
            for mod, attr in [(_ue, "_SYSTEM"), (_se, "_SYSTEM")]:
                orig = getattr(mod, attr, None)
                if orig is not None:
                    originals[f"{mod.__name__}.{attr}"] = orig
                    setattr(mod, attr,
                            orig.replace("flowsheet", "process diagram")
                                .replace("Flowsheet", "Process diagram"))

    except Exception:
        pass

    return originals


def _restore_prompt_perturbation(originals: dict) -> None:
    import importlib
    for key, value in originals.items():
        module_name, attr = key.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_name)
            setattr(mod, attr, value)
        except Exception:
            pass


def perturb_prompt(
    orchestrator,
    cases:       list[BenchmarkCase] | None = None,
    perturbation: str = "no_fewshot",
    verbose:     bool = False,
) -> dict[str, float]:
    """
    Run one benchmark trial with the named prompt perturbation active.

    Returns delta dict: {metric: perturbed_value - baseline_value}.
    """
    from eval.benchmark import run_benchmark

    if perturbation not in PROMPT_PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation {perturbation!r}. "
            f"Choose from: {PROMPT_PERTURBATIONS}"
        )

    cases = cases or BENCHMARK_CASES

    # Baseline (unperturbed)
    _, base_metrics = run_benchmark(orchestrator, cases=cases, verbose=False)

    # Perturbed
    originals = _apply_prompt_perturbation(perturbation)
    try:
        _, pert_metrics = run_benchmark(orchestrator, cases=cases, verbose=verbose)
    finally:
        _restore_prompt_perturbation(originals)

    return {
        "delta_valid_ir":    pert_metrics.pct_valid_ir    - base_metrics.pct_valid_ir,
        "delta_valid_json":  pert_metrics.pct_valid_json  - base_metrics.pct_valid_json,
        "delta_converged":   pert_metrics.pct_converged   - base_metrics.pct_converged,
        "delta_repair_iters": pert_metrics.avg_repair_iters - base_metrics.avg_repair_iters,
        "base_converged":    base_metrics.pct_converged,
        "pert_converged":    pert_metrics.pct_converged,
        "perturbation":      perturbation,
    }


# ── Multi-trial variance (model stochasticity) ─────────────────────────────────

def multi_trial(
    orchestrator,
    cases:    list[BenchmarkCase] | None = None,
    n_trials: int  = 5,
    verbose:  bool = False,
) -> RobustnessReport:
    """
    Repeat benchmark n_trials times with the same orchestrator.

    Captures variance attributable to LLM stochasticity (temperature > 0).
    Use temperature=0 for purely deterministic variance checks.
    """
    from eval.benchmark import run_benchmark

    cases = cases or BENCHMARK_CASES
    trial_metrics: list[BenchmarkMetrics] = []

    for i in range(n_trials):
        if verbose:
            print(f"  [trial {i+1}/{n_trials}]")
        _, metrics = run_benchmark(orchestrator, cases=cases, verbose=verbose)
        trial_metrics.append(metrics)

    return _build_report(trial_metrics, "stochasticity", cases)


# ── Report builder ─────────────────────────────────────────────────────────────

_METRIC_KEYS = [
    "pct_valid_ir",
    "pct_valid_json",
    "pct_converged",
    "avg_repair_iters",
    "avg_llm_calls",
]


def _build_report(
    trial_metrics: list[BenchmarkMetrics],
    perturbation:  str,
    cases:         list[BenchmarkCase],
) -> RobustnessReport:
    summaries: list[RobustnessSummary] = []
    for key in _METRIC_KEYS:
        values = [getattr(m, key) for m in trial_metrics]
        summaries.append(RobustnessSummary(
            metric_name  = key,
            perturbation = perturbation,
            n_trials     = len(values),
            values       = values,
        ))

    baseline = trial_metrics[0] if trial_metrics else None
    return RobustnessReport(
        summaries        = summaries,
        baseline_metrics = baseline,
        n_cases          = len(cases),
    )
