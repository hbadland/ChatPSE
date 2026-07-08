"""
BenchmarkRunner — main entry point for the CCS benchmark suite.

Usage:
    from benchmark.runner import BenchmarkRunner
    from benchmark.ablation import CONFIGS

    runner = BenchmarkRunner(model="qwen3:14b")

    # Full suite
    results = runner.run_all()

    # Single tier
    results = runner.run_tier("hard")

    # Ablation study
    all_results = runner.run_ablation(tiers=["easy", "medium", "hard"])

    # Single case
    result = runner.run_case("HARD_01")

BenchmarkRunner returns BenchmarkRunSet, which serialises to JSON
and produces publication-ready tables via .to_markdown() and .to_latex().
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from benchmark.case_schema import (
    BenchmarkCaseSpec, load_all, load_tier, load_by_id, load_from_files, TIERS
)
from benchmark.metrics import (
    RunMetrics, extract_metrics, aggregate, AggregateMetrics
)
from benchmark.logger import RunLog, extract_run_log, extract_system_streams
from benchmark.physics_eval import run_physics_checks, run_reference_comparison, CheckSeverity
from benchmark.ablation import AblationConfig, CONFIGS, apply_ablation, make_orchestrator

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results"
)


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class CaseRunResult:
    case:            BenchmarkCaseSpec
    metrics:         RunMetrics
    run_log:         RunLog
    log_path:        str = ""
    final_execution: Optional[object] = None  # ExecutionResult from DWSIM solver
    final_graph:     Optional[object] = None  # FlowsheetGraph after last repair


@dataclass
class BenchmarkRunSet:
    """Full results from one benchmark run (one ablation mode, one or more tiers)."""
    ablation_mode:  str
    model:          str
    tiers:          list[str]
    timestamp:      str
    case_results:   list[CaseRunResult] = field(default_factory=list)
    aggregate:      Optional[AggregateMetrics] = None

    # Per-tier aggregates
    tier_aggregates: dict[str, AggregateMetrics] = field(default_factory=dict)

    def metrics_list(self) -> list[RunMetrics]:
        return [r.metrics for r in self.case_results]

    def to_dict(self) -> dict:
        return {
            "ablation_mode":        self.ablation_mode,
            "model":                self.model,
            "tiers":                self.tiers,
            "timestamp":            self.timestamp,
            "n_cases":              len(self.case_results),
            "aggregate":            self.aggregate.__dict__ if self.aggregate else {},
            "tier_aggregates":      {k: v.__dict__ for k, v in self.tier_aggregates.items()},
            "failure_mode_summary": _failure_mode_summary(self.case_results),
            "case_results": [
                {
                    "case_id":   r.case.id,
                    "tier":      r.case.tier,
                    "difficulty": r.case.difficulty,
                    "domain":    r.case.domain,
                    **r.metrics.to_dict(),
                }
                for r in self.case_results
            ],
        }

    def save(self, results_dir: str | None = None) -> str:
        from benchmark.metrics import aggregate as _aggregate

        d = results_dir or os.path.join(_RESULTS_DIR, "summaries")
        os.makedirs(d, exist_ok=True)
        tiers_str = "_".join(self.tiers[:3])
        fname = f"{self.ablation_mode}_{tiers_str}_{self.timestamp}.json"
        path  = os.path.join(d, fname)

        # Write final DWSIM execution results.
        dwsim_path = path.replace(".json", "_dwsim_results.json")
        with open(dwsim_path, "w", encoding="utf-8") as f:
            json.dump(_execution_results_dict(self.case_results),
                      f, indent=2, default=str)

        # Write one flowsheet file per case to results/flowsheets/.
        flowsheets_dir = os.path.join(_RESULTS_DIR, "flowsheets")
        os.makedirs(flowsheets_dir, exist_ok=True)
        for r in self.case_results:
            fs = _flowsheet_dict(r)
            fs_path = os.path.join(flowsheets_dir, f"{r.case.id}_flowsheet.json")
            with open(fs_path, "w", encoding="utf-8") as f:
                json.dump(fs, f, indent=2, default=str)
            txt_path = os.path.join(flowsheets_dir, f"{r.case.id}_flowsheet.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(_flowsheet_report(fs, r))

            # For validation cases: compare against the ground-truth reference.
            ref_file = getattr(r.case, "reference_file", None)
            if ref_file:
                cr = _write_comparison(r.case, fs, ref_file, flowsheets_dir)
                if cr is not None:
                    r.metrics.match_score     = cr.match_score
                    r.metrics.validation_pass = cr.overall_pass
                    r.metrics.failure_modes   = cr.failure_modes
                    r.metrics.mape_T_pct      = cr.mape_T_pct

        # Re-aggregate after comparison scores are written onto metrics, then
        # write summary JSON so match_score / validation_pass are included.
        all_metrics = self.metrics_list()
        self.aggregate = _aggregate(all_metrics, self.ablation_mode)
        for tier in self.tiers:
            tier_m = [m for m in all_metrics if m.tier == tier]
            if tier_m:
                self.tier_aggregates[tier] = _aggregate(tier_m, self.ablation_mode)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return path

    def to_markdown(self) -> str:
        lines = [
            f"## Benchmark Results — {self.ablation_mode}  ({self.timestamp})\n",
            f"Model: `{self.model}`  |  Tiers: {', '.join(self.tiers)}\n",
        ]
        agg = self.aggregate
        if agg:
            lines += [
                "### Aggregate\n",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Success rate | {agg.success_rate:.1%} |",
                f"| Valid IR | {agg.valid_ir_rate:.1%} |",
                f"| Valid JSON | {agg.valid_json_rate:.1%} |",
                f"| Physics checks pass (all) | {agg.physics_pass_rate:.1%} |",
                f"| Physics checks pass (CRITICAL) | {agg.critical_physics_pass_rate:.1%} |",
                f"| Mean iterations | {agg.mean_iterations:.2f} |",
                f"| Mean sim calls | {agg.mean_sim_calls:.2f} |",
                f"| Mean candidates | {agg.mean_candidates:.1f} |",
                f"| Explore/exploit ratio | {agg.mean_explore_ratio:.2f} |",
                f"| Score improved (%) | {agg.pct_score_improved:.1%} |",
                f"| Score oscillated (%) | {agg.pct_score_oscillated:.1%} |",
                f"| BIP injected (%) | {agg.pct_bip_injected:.1%} |",
            ]
            if agg.recovery_rate is not None:
                lines.append(f"| Recovery rate | {agg.recovery_rate:.1%} |")
            if agg.mean_match_score > 0.0:
                lines += [
                    f"| Mean match score (val) | {agg.mean_match_score:.3f} |",
                    f"| Validation pass rate | {agg.pct_validation_pass:.1%} |",
                ]
            if agg.mean_mape_T_pct > 0.0:
                lines.append(
                    f"| Mean MAPE T (val) | {agg.mean_mape_T_pct:.2f}% |")
            if agg.ref_match_rate > 0.0 or agg.mean_ref_mape_T > 0.0:
                lines += [
                    f"| **Ref match rate** | **{agg.ref_match_rate:.1%}** |",
                    f"| Ref MAPE cases (sufficient/total) | {agg.n_mape_sufficient}/{agg.n_ref_cases} (mean n_matched {agg.mean_n_matched}) |",
                    f"| Ref MAPE T (±5 K) | {agg.mean_ref_mape_T:.2f}% |",
                    f"| Ref MAPE P (±5%) | {agg.mean_ref_mape_P:.2f}% |",
                    f"| Ref VF MAE (±0.05) | {agg.mean_ref_mape_vf:.4f} |",
                ]
            lines.append("")

        if self.tier_aggregates:
            lines += ["### By Tier\n",
                      "| Tier | Success | Valid IR | Physics | Mean iters |",
                      "|------|---------|----------|---------|-----------|"]
            for tier, ta in sorted(self.tier_aggregates.items()):
                lines.append(
                    f"| {tier} | {ta.success_rate:.1%} | {ta.valid_ir_rate:.1%} "
                    f"| {ta.physics_pass_rate:.1%} | {ta.mean_iterations:.1f} |"
                )
            lines.append("")

        fms = _failure_mode_summary(self.case_results)
        if fms.get("by_mode"):
            n_cmp = fms["n_compared"]
            lines += [
                "### Validation Failure Modes\n",
                f"Compared: {n_cmp}  |  Overall pass: {fms['n_overall_pass']}/{n_cmp}\n",
                "| Failure mode | Count | % of compared |",
                "|---|---|---|",
            ]
            for mode, info in sorted(fms["by_mode"].items(),
                                     key=lambda kv: -kv[1]["count"]):
                diag = " *(diagnostic)*" if mode == "flow_fail" else ""
                lines.append(
                    f"| {mode}{diag} | {info['count']} | {info['pct']:.0%} |")
            lines.append("")

        lines += ["### Per-case Results\n",
                  "| ID | Tier | Difficulty | Domain | Success | Match | Val pass | MAPE T% | Ref✓ | Ref T% | Recycle | Iter | Phys ✓ |",
                  "|----|------|------------|--------|---------|-------|----------|---------|------|--------|---------|------|--------|"]
        for r in self.case_results:
            m = r.metrics
            match_str = f"{m.match_score:.2f}" if m.match_score > 0.0 else "—"
            vpass_str = ("✓" if m.validation_pass else "✗") if m.match_score > 0.0 else "—"
            mape_str  = f"{m.mape_T_pct:.1f}%" if m.mape_T_pct > 0.0 else "—"
            if getattr(m, "reference_excluded", False):
                ref_pass_str = "phys-only"          # excluded-invalid reference
                ref_t_str    = "excluded"
            elif m.has_reference:
                ref_pass_str = "✓" if m.reference_match_pass else "✗"
                # Never a bare MAPE — always with its match count; insufficient
                # matches are marked, not shown as a (misleading) number.
                ref_t_str = (f"{m.reference_mape_T:.1f}% (n={m.reference_n_matched})"
                             if m.reference_mape_sufficient
                             else f"insuf(n={m.reference_n_matched})")
            else:
                ref_pass_str = ref_t_str = "—"
            graph     = getattr(r, "final_graph", None)
            rec_str   = ("✓" if getattr(graph, "has_recycles", False) else "✗") if graph is not None else "?"
            lines.append(
                f"| {r.case.id} | {r.case.tier} | {r.case.difficulty} "
                f"| {r.case.domain} | {'✓' if m.success else '✗'} "
                f"| {match_str} | {vpass_str} "
                f"| {mape_str} | {ref_pass_str} | {ref_t_str} | {rec_str} "
                f"| {m.n_iterations} "
                f"| {m.physics_checks_passed}/{m.physics_checks_run} |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        agg = self.aggregate
        if not agg:
            return ""
        rows = []
        data = [
            ("Success rate",       f"{agg.success_rate:.1%}"),
            ("Valid IR",           f"{agg.valid_ir_rate:.1%}"),
            ("Physics check pass", f"{agg.physics_pass_rate:.1%}"),
            ("Mean iterations",    f"{agg.mean_iterations:.2f}"),
            ("Mean sim calls",     f"{agg.mean_sim_calls:.2f}"),
            ("Mean candidates",    f"{agg.mean_candidates:.1f}"),
            ("Explore ratio",      f"{agg.mean_explore_ratio:.2f}"),
            ("Score improved",     f"{agg.pct_score_improved:.1%}"),
            ("Score oscillated",   f"{agg.pct_score_oscillated:.1%}"),
            ("BIP injected",       f"{agg.pct_bip_injected:.1%}"),
        ]
        if agg.recovery_rate is not None:
            data.append(("Recovery rate", f"{agg.recovery_rate:.1%}"))
        for label, val in data:
            rows.append(f"  {label} & {val} \\\\")
        return (
            "\\begin{tabular}{lc}\n"
            "\\hline\n"
            "Metric & Value \\\\ \\hline\n"
            + "\n".join(rows)
            + "\n\\hline\n\\end{tabular}"
        )


# ── BenchmarkRunner ────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs benchmark cases through OrchestratorV2 with full metrics collection.

    Parameters
    ----------
    model           : LLM model name (default: qwen3:14b via Ollama)
    max_iterations  : Stage 4 repair loop limit per case
    save_logs       : write per-run trajectory JSON to results/per_run/
    verbose         : print per-case progress
    """

    def __init__(
        self,
        model:          str  = "qwen3:14b",
        max_iterations: int  = 6,
        save_logs:      bool = True,
        verbose:        bool = True,
    ) -> None:
        self._model     = model
        self._max_iter  = max_iterations
        self._save_logs = save_logs
        self._verbose   = verbose

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_case(
        self,
        case_id:       str,
        ablation_mode: str = "full_ccs",
    ) -> CaseRunResult:
        """Run a single case by ID."""
        case   = load_by_id(case_id)
        config = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        return self._run_one(case, config)

    def run_tier(
        self,
        tier:          str,
        ablation_mode: str = "full_ccs",
    ) -> BenchmarkRunSet:
        """Run all cases in one tier."""
        cases  = load_tier(tier)
        config = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        return self._run_set(cases, config, tiers=[tier])

    def run_all(
        self,
        tiers:         list[str] | None = None,
        ablation_mode: str = "full_ccs",
    ) -> BenchmarkRunSet:
        """Run all cases across specified tiers (default: all)."""
        selected = tiers or TIERS
        cases    = load_all(tiers=selected)
        config   = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        return self._run_set(cases, config, tiers=selected)

    def run_case_files(
        self,
        file_paths:    list[str],
        ablation_mode: str = "full_ccs",
    ) -> BenchmarkRunSet:
        """Run all cases loaded from specific JSON files, bypassing the tier system.

        Useful for running the extended hard-benchmark files directly:

            runner.run_case_files([
                "benchmark/cases/hard_benchmark_mu.json",
                "benchmark/cases/hard_benchmark_adv.json",
            ])
        """
        cases  = load_from_files(file_paths)
        config = CONFIGS.get(ablation_mode, CONFIGS["full_ccs"])
        # Derive tier labels from the loaded cases for run-set metadata.
        tiers  = sorted({c.tier for c in cases})
        return self._run_set(cases, config, tiers=tiers)

    def run_targeted_ablation(
        self,
        case_ids: list[str],
        modes:    list[str] | None = None,
        verbose:  bool = True,
    ) -> dict[str, BenchmarkRunSet]:
        """
        Run ablation only on the specified subset of case IDs.

        Useful for isolating cases that exercise the repair loop (>1 iteration)
        where physics/coupling/rule_store components actually differentiate.

        Returns {ablation_mode: BenchmarkRunSet}.
        """
        selected_modes = modes or list(CONFIGS.keys())
        results: dict[str, BenchmarkRunSet] = {}

        for mode in selected_modes:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Ablation (targeted {len(case_ids)} cases): {mode}")
                print(f"{'='*60}")

            config = CONFIGS.get(mode, CONFIGS["full_ccs"])
            cases  = [load_by_id(cid) for cid in case_ids]
            tiers  = sorted({c.tier for c in cases})

            run_set = self._run_set(cases, config, tiers=tiers)
            results[mode] = run_set

            if verbose:
                print(run_set.aggregate)

        return results

    def run_ablation(
        self,
        tiers:   list[str] | None = None,
        modes:   list[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, BenchmarkRunSet]:
        """
        Run the full ablation study across all modes.

        Returns {ablation_mode: BenchmarkRunSet}.
        """
        selected_modes = modes or list(CONFIGS.keys())
        selected_tiers = tiers or ["easy", "medium", "hard"]
        results: dict[str, BenchmarkRunSet] = {}

        for mode in selected_modes:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Ablation: {mode}")
                print(f"{'='*60}")
            run_set = self.run_all(tiers=selected_tiers, ablation_mode=mode)
            results[mode] = run_set

            if verbose:
                print(run_set.aggregate)

        return results

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_one(
        self,
        case:   BenchmarkCaseSpec,
        config: AblationConfig,
    ) -> CaseRunResult:
        from agents.llm import get_call_count, reset_call_count

        orch, _ = make_orchestrator(config, self._model, self._max_iter)
        reset_call_count()

        import sys as _sys
        print(f"[ABLATION] case={case.id} mode={config.mode} "
              f"disable_physics={config.disable_physics} "
              f"disable_rules={config.disable_rules} "
              f"disable_coupling={config.disable_coupling} "
              f"beam_width={config.beam_width} "
              f"rule_store_patterns={orch._rule_store.num_patterns()} "
              f"active_rules={orch._rule_store.num_active()}",
              flush=True, file=_sys.stderr)

        # Verify no_rule_store: all retriever references must be NullRetriever.
        # ThermoMapper stores the retriever in sub-components, not self._retriever.
        if config.disable_rules:
            def _check_ret(label: str, obj) -> None:
                if obj is None:
                    return
                ret = getattr(obj, "_retriever", None)
                if ret is None:
                    return  # attribute absent — skip silently
                status = "NullRetriever=OK" if type(ret).__name__ == "NullRetriever" \
                         else f"PATCH_FAILED({type(ret).__name__})"
                print(f"[ABLATION]   no_rule_store VERIFY: {label}._retriever = {status}",
                      flush=True, file=_sys.stderr)

            _check_ret("orch._params",   getattr(orch, "_params",  None))
            _check_ret("orch._repair",   getattr(orch, "_repair",  None))
            _thermo = getattr(orch, "_thermo", None)
            if _thermo is not None:
                for _comp in ("_selector", "_injector", "_llm"):
                    _check_ret(f"orch._thermo.{_comp}", getattr(_thermo, _comp, None))

        if config.beam_width == 1:
            _bw = getattr(getattr(orch, "_repair", None), "_beam_width", "NO_ATTR")
            print(f"[ABLATION]   greedy VERIFY: orch._repair._beam_width={_bw} "
                  f"(1=OK)", flush=True, file=_sys.stderr)

        if self._verbose:
            print(f"  [{case.id}/{case.tier}] {case.name[:50]} …", end=" ", flush=True)

        # Use the short benchmark prompt when available; fall back to description.
        nl_input = getattr(case, "prompt", None) or case.description

        t0 = time.time()
        try:
            with apply_ablation(config):
                pr = orch.run(
                    nl_input,
                    reference_file=getattr(case, "reference_file", None),
                    tier=case.tier,
                )
        except Exception as exc:
            import sys as _sys, traceback as _tb
            print(f"[RUNNER] EXCEPTION in {case.id}: {exc}", flush=True, file=_sys.stderr)
            if self._verbose:
                print(f"\n  [EXCEPTION] {exc}")
                _tb.print_exc()
            pr = _make_failed_result(str(exc))

        llm_calls = get_call_count()

        # Physics checks
        checks = run_physics_checks(case, pr)
        n_checks_run    = len(checks)
        n_checks_passed = sum(1 for c in checks if c.get("passed", False))
        n_critical_run    = sum(1 for c in checks
                                if c.get("severity") == CheckSeverity.CRITICAL)
        n_critical_passed = sum(1 for c in checks
                                if c.get("severity") == CheckSeverity.CRITICAL
                                and c.get("passed", False))

        # Attach physics check results to pr so metrics extractor can read them
        pr._physics_checks = checks   # type: ignore[attr-defined]

        # ── Reference-match comparison (validation cases only) ─────────────────
        # Runs live during the benchmark with stricter ±5 K / ±5% / ±0.05 thresholds.
        # Distinct from BenchmarkRunSet.save()'s _write_comparison() which uses looser
        # ±10 K tolerances for archival reporting and runs after all cases complete.
        ref_mape_T = ref_mape_P = ref_mape_vf = 0.0
        ref_match_pass = False
        _ref_checks: list = []
        _ref_n_matched  = 0
        _ref_sufficient = False   # >= _MIN_MATCH_FOR_MAPE streams matched
        has_reference  = bool(getattr(case, "reference_file", None))

        # Physics-only exclusion: a reference flagged excluded-invalid-reference
        # (e.g. VAL_10 carbon-balance violation) still runs converged + physics
        # checks, but reference-MAPE is skipped — its data can't be ground truth.
        ref_excluded = False
        ref_excluded_reason = ""
        if has_reference:
            from benchmark.comparison import load_reference
            _refdata = load_reference(getattr(case, "reference_file", "")) or {}
            if "excluded-invalid-reference" in str(_refdata.get("reference_validity", "")):
                ref_excluded = True
                ref_excluded_reason = _refdata.get("reference_validity_reason", "") \
                    or _refdata.get("reference_validity", "")
                print(f"[BENCH] {getattr(case,'id','?')}: reference excluded "
                      f"(physics-only) — {ref_excluded_reason[:80]}", flush=True)

        if has_reference and not ref_excluded:
            _ref_checks, ref_mape_T, ref_mape_P, ref_mape_vf = \
                run_reference_comparison(case, pr)
            # n_matched from the stream-matching detail; MAPE is None when the
            # match count is below _MIN_MATCH_FOR_MAPE (insufficient_match).
            _match_detail = next((c for c in _ref_checks
                                  if c.get("check") == "reference_stream_matching"), {})
            _ref_n_matched  = _match_detail.get("n_matched", 0)
            _ref_sufficient = ref_mape_T is not None
            if _ref_checks:
                checks.extend(_ref_checks)
                pr._physics_checks = checks
                n_checks_run    += len(_ref_checks)
                n_checks_passed += sum(1 for c in _ref_checks if c.get("passed", False))
                _crit = [c for c in _ref_checks
                         if c.get("severity") == CheckSeverity.CRITICAL]
                n_critical_run    += len(_crit)
                n_critical_passed += sum(1 for c in _crit if c.get("passed", False))
                _active_crit = [c for c in _crit if c.get("source") != "none"]
                # Gate on BOTH a sufficient match count AND the MAPE thresholds.
                ref_match_pass = (_ref_sufficient and bool(_active_crit)
                                  and all(c["passed"] for c in _active_crit))

        # Derive compatible attributes for metrics extractor
        pr.ir_valid   = getattr(pr, "ir_valid",   False) or (
            getattr(pr, "ir_report", None) is not None and
            getattr(getattr(pr, "ir_report", None), "valid", False))
        pr.json_valid = getattr(pr, "json_valid", False) or (
            getattr(pr, "final_flowsheet", None) is not None)
        pr.converged  = getattr(pr, "converged", False) or (
            getattr(pr, "outcome", "") == "PASS")

        # Extract metrics
        m = extract_metrics(pr, case, config.mode, llm_calls)
        m.physics_checks_run              = n_checks_run
        m.physics_checks_passed           = n_checks_passed
        m.physics_check_details           = checks
        m.critical_physics_checks_run     = n_critical_run
        m.critical_physics_checks_passed  = n_critical_passed
        # Excluded references are physics-only: drop has_reference so the MAPE
        # aggregates skip them, but record the exclusion for the per-case report.
        m.has_reference                   = has_reference and not ref_excluded
        # MAPE is None when insufficient_match — store 0.0 as an internal
        # placeholder and rely on reference_mape_sufficient to EXCLUDE it from
        # aggregates (never averaged/reported as a real value).
        m.reference_mape_T                = ref_mape_T if _ref_sufficient else 0.0
        m.reference_mape_P                = ref_mape_P if _ref_sufficient else 0.0
        m.reference_mape_vf               = ref_mape_vf if _ref_sufficient else 0.0
        m.reference_match_pass            = ref_match_pass
        m.reference_n_matched             = _ref_n_matched
        m.reference_mape_sufficient       = _ref_sufficient
        m.reference_excluded              = ref_excluded
        m.reference_excluded_reason       = ref_excluded_reason

        # Extract trajectory log
        run_log  = extract_run_log(pr, case.id, config.mode, self._model)

        # ── Additive evidence (logging only — does NOT affect PASS gating) ─────
        # Converged system stream table + the already-computed reference
        # comparison, persisted per-run so the produced values are inspectable
        # independent of the PASS decision.
        run_log.system_streams = extract_system_streams(pr)

        # Property-package family selection scored against the case's expected
        # family label — persisted so family-selection accuracy is measurable
        # from the ACTIVE pipeline's per-run JSONs (not only the offline harness).
        _exp_cls = getattr(case.expected, "property_package_class", None)
        if _exp_cls:
            from benchmark.package_family import score_family
            _fgs = run_log.final_graph_summary or {}
            run_log.package_family = score_family(
                _fgs.get("property_package"), _exp_cls,
                _fgs.get("n_binary_params"))

        if has_reference:
            from benchmark.physics_eval import _MIN_MATCH_FOR_MAPE
            _ins = "insufficient_match"      # never a bare MAPE without its count
            run_log.reference_comparison = {
                # n_matched is reported prominently and ALWAYS alongside the MAPE.
                "n_matched":                 _ref_n_matched,
                "min_match_threshold":       _MIN_MATCH_FOR_MAPE,
                "mape_status":               "computed" if _ref_sufficient else _ins,
                "reference_match_pass":      ref_match_pass,
                "reference_mape_T":          ref_mape_T  if _ref_sufficient else _ins,
                "reference_mape_P":          ref_mape_P  if _ref_sufficient else _ins,
                "reference_mape_vf":         ref_mape_vf if _ref_sufficient else _ins,
                "reference_excluded":        ref_excluded,
                "reference_excluded_reason": ref_excluded_reason,
                # Preserve ALL keys per check (severity serialised) so the
                # 'reference_stream_matching' check's matches/unmatched detail
                # is persisted, not just check/passed/severity/source/detail.
                "checks": [
                    {**{k: v for k, v in c.items() if k != "severity"},
                     "severity": getattr(c.get("severity"), "value",
                                         str(c.get("severity")))}
                    for c in _ref_checks
                ],
            }

        log_path = ""
        if self._save_logs:
            log_path = run_log.save(os.path.join(_RESULTS_DIR, "per_run"))

        elapsed = time.time() - t0
        if self._verbose:
            outcome = getattr(pr, "outcome", "?")
            print(f"{outcome}  iter={m.n_iterations}  "
                  f"phys={n_checks_passed}/{n_checks_run}  {elapsed:.1f}s")

        return CaseRunResult(
            case            = case,
            metrics         = m,
            run_log         = run_log,
            log_path        = log_path,
            final_execution = getattr(pr, "final_execution", None),
            final_graph     = getattr(pr, "final_graph", None),
        )

    def _run_set(
        self,
        cases:  list[BenchmarkCaseSpec],
        config: AblationConfig,
        tiers:  list[str],
    ) -> BenchmarkRunSet:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_set = BenchmarkRunSet(
            ablation_mode = config.mode,
            model         = self._model,
            tiers         = tiers,
            timestamp     = timestamp,
        )

        for case in cases:
            result = self._run_one(case, config)
            run_set.case_results.append(result)

        # Aggregates
        all_metrics = run_set.metrics_list()
        run_set.aggregate = aggregate(all_metrics, config.mode)

        for tier in tiers:
            tier_m = [m for m in all_metrics if m.tier == tier]
            if tier_m:
                run_set.tier_aggregates[tier] = aggregate(tier_m, config.mode)

        return run_set


# ── Helpers ────────────────────────────────────────────────────────────────────

_FAILURE_MODES = [
    "pkg_mismatch", "unit_count_out_of_range", "stream_not_found",
    "temperature_fail", "pressure_fail", "composition_fail",
    "vapor_fraction_fail", "flow_fail",
]


def _failure_mode_summary(case_results: list) -> dict:
    """Aggregate comparison failure modes across all validation cases that were compared."""
    compared = [r for r in case_results
                if getattr(r.case, "reference_file", None) and r.metrics.match_score > 0.0]
    if not compared:
        return {}
    n = len(compared)
    counts = {m: 0 for m in _FAILURE_MODES}
    for r in compared:
        for mode in r.metrics.failure_modes:
            if mode in counts:
                counts[mode] += 1
    return {
        "n_compared":     n,
        "n_overall_pass": sum(1 for r in compared if r.metrics.validation_pass),
        "by_mode": {
            k: {"count": v, "pct": round(v / n, 3)}
            for k, v in counts.items() if v > 0
        },
    }


def _write_comparison(
    case,
    system_fs:      dict,
    reference_file: str,
    out_dir:        str,
):
    """Load reference, run comparison, write JSON + txt diff report.

    Returns the ComparisonResult so the caller can pull match_score and
    overall_pass back onto the RunMetrics; returns None if skipped.
    """
    from benchmark.comparison import (
        load_reference, compare_flowsheets, comparison_report)

    ref = load_reference(reference_file)
    if ref is None:
        return None

    n_min = getattr(getattr(case, "expected", None), "n_units_min", None)
    n_max = getattr(getattr(case, "expected", None), "n_units_max", None)

    cr = compare_flowsheets(
        system_fs, ref,
        reference_file=reference_file,
        n_units_min=n_min,
        n_units_max=n_max,
    )

    case_id      = getattr(case, "id", str(case))
    cr_json_path = os.path.join(out_dir, f"{case_id}_comparison.json")
    cr_txt_path  = os.path.join(out_dir, f"{case_id}_comparison.txt")

    with open(cr_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "case_id":              cr.case_id,
            "case_name":            cr.case_name,
            "reference_file":       cr.reference_file,
            "overall_pass":         cr.overall_pass,
            "match_score":          cr.match_score,
            "mape_T_pct":           cr.mape_T_pct,
            "pkg_match":            cr.pkg_match,
            "unit_type_match":      cr.unit_type_match,
            "unit_count_in_range":  cr.unit_count_in_range,
            "unit_types_system":    cr.unit_types_system,
            "unit_types_reference": cr.unit_types_reference,
            "failure_modes":        cr.failure_modes,
            "warnings":             cr.warnings,
            "field_results": [
                {"stream": fr.stream, "field": fr.field,
                 "system": fr.system, "reference": fr.reference,
                 "diff": fr.diff, "passed": fr.passed,
                 "tolerance": fr.tolerance,
                 "is_diagnostic": fr.is_diagnostic}
                for fr in cr.field_results
            ],
        }, f, indent=2)

    with open(cr_txt_path, "w", encoding="utf-8") as f:
        f.write(comparison_report(cr))

    return cr


def _flowsheet_dict(r: CaseRunResult) -> dict:
    """
    Build a self-contained flowsheet record for one case combining:
      - unit topology (tags, types, parameters) from the final IR graph
      - stream conditions (T, P, composition, vapour fraction) from DWSIM

    Saved to results/flowsheets/{case_id}_flowsheet.json for direct
    comparison against ground-truth flowsheets converted from .dwxml files.
    """
    graph = getattr(r, "final_graph", None)
    ex    = r.final_execution

    # ── Unit topology from IR graph ───────────────────────────────────────────
    units:       list[dict] = []
    connections: list[list] = []

    if graph is not None:
        for node in graph.units():
            units.append({
                "tag":    node.tag,
                "type":   node.unit_type,
                "params": {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                           for k, v in node.params.items()},
            })
        # Reconstruct connections from graph edges (stream-mediated)
        try:
            for stream in graph.streams() if hasattr(graph, "streams") else []:
                src = graph.stream_source(stream.tag)
                dst = graph.stream_dest(stream.tag)
                if src and dst:
                    connections.append([src, stream.tag, dst])
        except Exception:
            pass  # connection extraction is best-effort

    # ── Stream conditions from DWSIM execution ────────────────────────────────
    streams: dict = {}
    if ex is not None:
        for tag, s in (getattr(ex, "stream_results", {}) or {}).items():
            streams[tag] = {
                "T_K":            round(float(s.T_K), 4),
                "T_C":            round(float(s.T_K) - 273.15, 4),
                "P_Pa":           round(float(s.P_Pa), 2),
                "P_bar":          round(float(s.P_Pa) / 1e5, 5),
                "flow_mol_s":     round(float(s.flow_mol_s), 6),
                "vapor_fraction": round(float(getattr(s, "vapor_fraction", 0.0)), 6),
                "composition":    {k: round(float(v), 6)
                                   for k, v in s.composition.items()},
            }

    has_recycles   = graph.has_recycles if graph is not None else False
    recycle_blocks = graph.metadata.get("recycle_blocks", []) if graph is not None else []

    return {
        "case_id":          r.case.id,
        "case_name":        r.case.name,
        "tier":             r.case.tier,
        "compounds":        r.case.compounds,
        "property_package": graph.property_package if graph is not None else "",
        "solved":           bool(getattr(ex, "solved", False)) if ex else False,
        "has_recycles":     has_recycles,
        "recycle_blocks":   recycle_blocks,
        "units":            units,
        "connections":      connections,
        "streams":          streams,
    }


def _flowsheet_report(fs: dict, r: CaseRunResult) -> str:
    """
    Produce a human-readable plain-text report of the converged flowsheet.
    Saved alongside the JSON as {case_id}_flowsheet.txt.
    """
    W = 60
    bar  = "═" * W
    line = "─" * W

    solved_str = "YES" if fs["solved"] else "NO"
    n_iter     = getattr(r.metrics, "n_iterations", "?")

    has_rec    = fs.get("has_recycles", False)
    rec_blocks = fs.get("recycle_blocks", [])
    if has_rec and rec_blocks:
        rec_str = "YES — " + ", ".join(
            f"{rb['tag']}({rb['inlet_stream']}→{rb['outlet_stream']})"
            for rb in rec_blocks)
    elif has_rec:
        rec_str = "YES"
    else:
        rec_str = "NO"

    lines = [
        bar,
        f"FLOWSHEET REPORT — {fs['case_id']}",
        fs["case_name"],
        bar,
        f"Compounds        : {', '.join(fs['compounds'])}",
        f"Property package : {fs['property_package']}",
        f"Solved           : {solved_str}  ({n_iter} iterations)",
        f"Recycle streams  : {rec_str}",
        "",
    ]

    # ── Units ─────────────────────────────────────────────────────────────────
    lines += ["UNITS", line]
    if fs["units"]:
        col_tag  = max(len(u["tag"])  for u in fs["units"])
        col_type = max(len(u["type"]) for u in fs["units"])
        col_tag  = max(col_tag,  5)
        col_type = max(col_type, 6)
        lines.append(f"{'Tag':<{col_tag}}  {'Type':<{col_type}}  Parameters")
        for u in fs["units"]:
            param_str = "   ".join(
                f"{k} = {v:.4g} K" if k == "T_out"
                else f"{k} = {v:.4g} Pa" if k in ("P_out", "dP")
                else f"{k} = {v:.4g}"
                for k, v in u["params"].items()
                if isinstance(v, (int, float))
            )
            lines.append(f"{u['tag']:<{col_tag}}  {u['type']:<{col_type}}  {param_str}")
    else:
        lines.append("  (no unit data — graph not available)")
    lines.append("")

    # ── Connections ───────────────────────────────────────────────────────────
    lines += ["CONNECTIONS", line]
    if fs["connections"]:
        for conn in fs["connections"]:
            lines.append("  " + "  →  ".join(str(c) for c in conn))
    else:
        lines.append("  (no connection data)")
    lines.append("")

    # ── Streams ───────────────────────────────────────────────────────────────
    lines += ["STREAMS", line]
    streams = fs.get("streams", {})
    if streams:
        compounds = fs["compounds"]
        # Header
        comp_hdrs = "  ".join(f"{c[:8]:>8}" for c in compounds)
        lines.append(
            f"{'Stream':<12}  {'T (°C)':>7}  {'P (bar)':>7}  "
            f"{'Flow':>8}  {'VF':>5}  {comp_hdrs}"
        )
        lines.append(line)
        for tag, s in streams.items():
            comp_vals = "  ".join(
                f"{s['composition'].get(c, 0.0):>8.4f}" for c in compounds)
            lines.append(
                f"{tag:<12}  {s['T_C']:>7.2f}  {s['P_bar']:>7.4f}  "
                f"{s['flow_mol_s']:>8.4f}  {s['vapor_fraction']:>5.3f}  {comp_vals}"
            )
    else:
        lines.append("  (no stream data — DWSIM did not converge)")
    lines.append("")
    lines.append(bar)

    return "\n".join(lines) + "\n"


def _execution_results_dict(case_results: list[CaseRunResult]) -> dict:
    """
    Serialise the final DWSIM execution result for every case.

    Stream conditions (T_K, P_Pa, flow_mol_s, composition, vapor_fraction)
    come directly from the DWSIM solver via ExecutionResult.stream_results.
    solved=True with non-empty stream_results proves DWSIM ran and converged.
    """
    out: dict = {}
    for r in case_results:
        ex = r.final_execution
        if ex is None:
            out[r.case.id] = {
                "case_id": r.case.id, "case_name": r.case.name,
                "tier": r.case.tier, "compounds": r.case.compounds,
                "solved": False, "stream_results": {},
                "note": "no execution recorded",
            }
            continue

        stream_results = {}
        for tag, s in (getattr(ex, "stream_results", {}) or {}).items():
            stream_results[tag] = {
                "T_K":            round(float(s.T_K), 4),
                "T_C":            round(float(s.T_K) - 273.15, 4),
                "P_Pa":           round(float(s.P_Pa), 2),
                "P_bar":          round(float(s.P_Pa) / 1e5, 5),
                "flow_mol_s":     round(float(s.flow_mol_s), 6),
                "vapor_fraction": round(float(getattr(s, "vapor_fraction", 0.0)), 6),
                "composition":    {k: round(float(v), 6)
                                   for k, v in s.composition.items()},
            }

        out[r.case.id] = {
            "case_id":        r.case.id,
            "case_name":      r.case.name,
            "tier":           r.case.tier,
            "compounds":      r.case.compounds,
            "solved":         bool(getattr(ex, "solved", False)),
            "solver_errors":  list(getattr(ex, "solver_errors", []) or []),
            "stream_results": stream_results,
        }
    return out


def _make_failed_result(error_msg: str):
    """Minimal PipelineResult-like object for exception cases."""
    class _FailedResult:
        outcome       = "EXCEPTION"
        ir_valid      = False
        json_valid    = False
        converged     = False
        ir_report     = None
        final_graph   = None
        final_flowsheet = None
        final_execution = None
        iterations    = []
        warnings      = []
        basis_result  = None
        total_time_s  = 0.0

    r = _FailedResult()
    r.warnings = [f"EXCEPTION: {error_msg}"]
    return r


def find_repair_cases(
    results_dir: str | None = None,
    ablation_mode: str = "full_ccs",
    min_iterations: int = 2,
) -> list[str]:
    """
    Parse per-run JSON logs to find case IDs that required multiple repair
    iterations in the given ablation mode.

    These are the cases where physics/coupling/rule_store actually differentiate
    between ablation modes — run targeted_ablation on them to see a real signal.

    Parameters
    ----------
    results_dir    : directory containing per-run JSON files (default: results/per_run/)
    ablation_mode  : which mode to analyse (default: "full_ccs")
    min_iterations : minimum iteration count to include (default: 2)

    Returns
    -------
    Sorted list of case IDs with n_iterations >= min_iterations.
    """
    from benchmark.logger import load_all_logs

    d = results_dir or os.path.join(_RESULTS_DIR, "per_run")
    logs = load_all_logs(d)
    if not logs:
        print(f"[find_repair_cases] no logs found in {d}")
        return []

    repair_cases: dict[str, int] = {}
    for log in logs:
        if log.get("ablation_mode") != ablation_mode:
            continue
        case_id  = log.get("case_id", "")
        n_iters  = len(log.get("iterations", []))
        # Keep the latest run for each case (logs are sorted by filename/timestamp)
        if n_iters >= min_iterations:
            repair_cases[case_id] = n_iters

    result = sorted(repair_cases.keys())
    print(f"[find_repair_cases] mode={ablation_mode} min_iter={min_iterations}: "
          f"{len(result)} cases — {result}")
    for cid, n in sorted(repair_cases.items()):
        print(f"  {cid}: {n} iterations")
    return result


def ablation_table(results: dict[str, BenchmarkRunSet]) -> str:
    """
    Format ablation comparison table as Markdown.
    results: {ablation_mode: BenchmarkRunSet}
    """
    modes   = list(results.keys())
    headers = ["Mode", "Success", "Valid IR", "Physics", "Mean iter", "Candidates",
               "Explore%", "Oscillated%"]
    rows    = [headers, ["---"] * len(headers)]

    for mode, rs in results.items():
        agg = rs.aggregate
        if agg is None:
            continue
        rows.append([
            mode,
            f"{agg.success_rate:.1%}",
            f"{agg.valid_ir_rate:.1%}",
            f"{agg.physics_pass_rate:.1%}",
            f"{agg.mean_iterations:.2f}",
            f"{agg.mean_candidates:.1f}",
            f"{agg.mean_explore_ratio:.1%}",
            f"{agg.pct_score_oscillated:.1%}",
        ])

    col_w = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = []
    for row in rows:
        cells = [cell.ljust(col_w[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
