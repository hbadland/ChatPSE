"""
Experiment runner.

Run inside the VSCode Dev Container terminal:

    PYTHONPATH=. python3.9 run_experiments.py                        # all experiments, qwen3:14b
    PYTHONPATH=. python3.9 run_experiments.py --exp 1                # single experiment
    PYTHONPATH=. python3.9 run_experiments.py --model gemini-2.5-flash --exp 1  # different model

Provider / model selection
    Ollama  (default): qwen3:14b, llama3:8b, mistral, phi3, ...
    Google:            gemini-2.5-flash  (needs GOOGLE_API_KEY)
    Anthropic:         claude-sonnet-4-6 (needs ANTHROPIC_API_KEY)
    Groq:              llama-3.3-70b-versatile (needs GROQ_API_KEY)

For Ollama the container talks to Ollama on the host via host.docker.internal.
No API key is needed.

Experiments
-----------
1  Baseline vs System  — all 60 cases; compares single-LLM baseline to full pipeline
2  Split breakdown     — system on dev / holdout / stress separately
3  Robustness          — scoring perturbation (N=10), prompt perturbation, multi-trial (N=5)
4  Real cases          — 3 real industrial descriptions

Results are written to results/exp<N>_<timestamp>.txt and printed to stdout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_OLLAMA_PREFIXES = ("qwen", "llama", "mistral", "phi", "deepseek", "gemma")
_KEY_MAP = {
    "google":    "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "groq":      "GROQ_API_KEY",
}


def _check_api_key(model: str) -> None:
    m = model.lower()
    if any(m.startswith(p) for p in _OLLAMA_PREFIXES):
        return   # Ollama needs no API key
    if m.startswith("gemini"):
        provider = "google"
    elif m.startswith("claude"):
        provider = "anthropic"
    elif m.startswith(("gpt", "o1", "o3")):
        provider = "openai"
    elif any(m.startswith(p) for p in ("llama-3", "qwen-qwq", "gemma2-", "mixtral-8")):
        provider = "groq"
    else:
        return
    env_var = _KEY_MAP[provider]
    if not os.environ.get(env_var):
        sys.exit(
            f"ERROR: {env_var} is not set (required for model '{model}').\n"
            f"  export {env_var}=..."
        )


# ── OrchestratorAdapter ───────────────────────────────────────────────────────

class OrchestratorAdapter:
    """
    Wraps the existing Orchestrator so it satisfies the interface expected
    by run_benchmark():

        orchestrator.run(description, compounds) -> result
        result.outcome        — "PASS" | "HUMAN" | "MAX_ITER" | ...
        result.ir_valid       — bool
        result.json_valid     — bool
        result.converged      — bool
        result.iterations     — list (length = repair count)
        result.warnings       — list[str]
    """

    def __init__(self, model: str = "qwen3:14b",
                 max_iterations: int = 6) -> None:
        from agents.orchestrator import Orchestrator
        from agents import schema as _schema
        self._orch   = Orchestrator(model=model, max_iterations=max_iterations)
        self._schema = _schema

    def run(self, description: str, compounds: list[str] | None = None):
        pr = self._orch.run(description)
        # Derive json_valid from schema validation of the final flowsheet
        json_valid = False
        if pr.final_flowsheet:
            try:
                errs = self._schema.validate(pr.final_flowsheet)
                json_valid = len(errs) == 0
            except Exception:
                json_valid = False

        # Derive converged from execution result
        converged = False
        if pr.final_execution is not None:
            converged = getattr(pr.final_execution, "solved", False)

        pr.ir_valid   = json_valid   # v1 has no IR layer; use json_valid as proxy
        pr.json_valid = json_valid
        pr.converged  = converged
        return pr


# ── Output helpers ─────────────────────────────────────────────────────────────

_RESULTS_DIR = ROOT / "results"


def _save(name: str, content: str) -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = _RESULTS_DIR / f"{name}_{ts}.txt"
    path.write_text(content)
    print(f"\n[saved → {path}]")
    return path


def _section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


# ── Experiment 1: Baseline vs System ──────────────────────────────────────────

def exp1_baseline_vs_system(model: str, verbose: bool) -> None:
    _section("EXPERIMENT 1 — Baseline vs System  (split=all)")

    from eval.benchmark import run_benchmark, run_baseline, compare
    from eval.report import format_report
    from eval.dataset import get_cases

    cases = get_cases("all")
    print(f"Running baseline on {len(cases)} cases …")
    baseline_results, baseline_metrics = run_baseline(cases=cases, verbose=verbose)
    print(baseline_metrics)

    print(f"\nRunning full system on {len(cases)} cases …")
    orch = OrchestratorAdapter(model=model)
    run  = run_benchmark(orch, cases=cases, split="all", ablation="full", verbose=verbose)
    print(run.metrics_all)

    delta = compare(run.metrics_all, baseline_metrics)
    print(f"\nDelta (system - baseline):")
    print(f"  Δ valid_json : {delta['delta_valid_json']:+.1%}")
    print(f"  Δ converged  : {delta['delta_converged']:+.1%}")

    report = format_report(run)
    print(report)
    _save("exp1_baseline_vs_system", report)


# ── Experiment 2: Split breakdown ─────────────────────────────────────────────

def exp2_split_breakdown(model: str, verbose: bool) -> None:
    _section("EXPERIMENT 2 — System per split  (dev / holdout / stress)")

    from eval.benchmark import run_benchmark
    from eval.report import format_report, BenchmarkRunResult
    from eval.metrics import compute_metrics

    orch = OrchestratorAdapter(model=model)
    lines: list[str] = []

    for split in ("dev", "holdout", "stress"):
        print(f"\n--- Running split={split} ---")
        run = run_benchmark(orch, split=split, ablation="full", verbose=verbose)
        print(run.metrics_all)
        report = format_report(run)
        lines.append(report)

    combined = "\n\n".join(lines)
    print(combined)
    _save("exp2_split_breakdown", combined)


# ── Experiment 3: Robustness ──────────────────────────────────────────────────

def exp3_robustness(model: str, verbose: bool,
                    n_scoring: int = 10, n_prompt: int = 10, n_trial: int = 5) -> None:
    _section("EXPERIMENT 3 — Robustness checks")

    from eval.robustness import (
        perturb_scoring, perturb_prompt, multi_trial,
        PROMPT_PERTURBATIONS,
    )
    from eval.dataset import get_cases

    # Use dev split only for robustness (never touch holdout)
    cases = get_cases("dev")
    orch  = OrchestratorAdapter(model=model)
    lines: list[str] = []

    # 3a: Scoring weight perturbation
    print(f"\n[3a] Scoring perturbation  (n={n_scoring}, dev cases={len(cases)}) …")
    rob_score = perturb_scoring(orch, cases=cases, n_runs=n_scoring, verbose=verbose)
    section = ["=== 3a: Scoring weight perturbation ==="]
    for s in rob_score.summaries:
        section.append(f"  {s}")
    print("\n".join(section))
    lines.extend(section)

    # 3b: Prompt perturbations
    for pert in PROMPT_PERTURBATIONS:
        print(f"\n[3b] Prompt perturbation: {pert}  (dev cases={len(cases)}) …")
        delta = perturb_prompt(orch, cases=cases, perturbation=pert, verbose=verbose)
        line = (f"  {pert:<18} "
                f"base={delta['base_converged']:.1%}  "
                f"pert={delta['pert_converged']:.1%}  "
                f"Δconverged={delta['delta_converged']:+.1%}")
        print(line)
        lines.append(f"\n=== 3b: Prompt perturbation: {pert} ===\n{line}")

    # 3c: Multi-trial (stochasticity)
    print(f"\n[3c] Multi-trial  (n={n_trial}, dev cases={len(cases)}) …")
    rob_multi = multi_trial(orch, cases=cases, n_trials=n_trial, verbose=verbose)
    section = ["\n=== 3c: Multi-trial stochasticity ==="]
    for s in rob_multi.summaries:
        section.append(f"  {s}")
    if rob_multi.all_stable():
        section.append("\n  Overall: ALL STABLE")
    else:
        section.append(f"\n  Sensitive: {[s.metric_name for s in rob_multi.sensitive_metrics()]}")
    print("\n".join(section))
    lines.extend(section)

    _save("exp3_robustness", "\n".join(lines))


# ── Experiment 4: Real cases ───────────────────────────────────────────────────

def exp4_real_cases(model: str, verbose: bool) -> None:
    _section("EXPERIMENT 4 — Real industrial cases")

    from eval.benchmark import run_benchmark
    from eval.real_cases import load_real_cases
    from eval.report import format_report

    cases = load_real_cases()
    print(f"Running {len(cases)} real cases …\n")
    for c in cases:
        print(f"  {c.case_id}: {c.description[:70]}…")
        print(f"           source: {c.source}")
    print()

    orch = OrchestratorAdapter(model=model)
    run  = run_benchmark(orch, cases=cases, split="all", ablation="real", verbose=True)
    report = format_report(run)
    print(report)
    _save("exp4_real_cases", report)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments (must be run inside the Dev Container).")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4],
                        help="Run only this experiment number (default: all)")
    parser.add_argument("--model", default="qwen3:14b",
                        help="LLM model to use (default: qwen3:14b via Ollama)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-case progress")
    parser.add_argument("--n-scoring", type=int, default=10,
                        help="Scoring perturbation trials (exp 3)")
    parser.add_argument("--n-prompt",  type=int, default=10,
                        help="Prompt perturbation repeats (exp 3)")
    parser.add_argument("--n-trial",   type=int, default=5,
                        help="Multi-trial repeats (exp 3)")
    args = parser.parse_args()

    _check_api_key(args.model)

    run_all = args.exp is None
    if run_all or args.exp == 1:
        exp1_baseline_vs_system(args.model, args.verbose)
    if run_all or args.exp == 2:
        exp2_split_breakdown(args.model, args.verbose)
    if run_all or args.exp == 3:
        exp3_robustness(args.model, args.verbose,
                        n_scoring=args.n_scoring,
                        n_prompt=args.n_prompt,
                        n_trial=args.n_trial)
    if run_all or args.exp == 4:
        exp4_real_cases(args.model, args.verbose)


if __name__ == "__main__":
    main()
