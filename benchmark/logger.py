"""
Trajectory logger for benchmark runs.

Each RunLog captures:
  - the full iteration trajectory (states, candidates, scores per step)
  - constraint violations per iteration
  - scoring breakdown per candidate
  - explore/exploit phase labels per step

Designed for:
  - paper figures (score convergence curves)
  - ablation analysis (beam vs greedy trajectories)
  - debugging (what was tried at each step)

Logs are stored as JSON in results/per_run/<case_id>_<ablation>_<timestamp>.json.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ── Iteration-level record ─────────────────────────────────────────────────────

@dataclass
class IterationLog:
    iteration:           int
    n_errors_before:     int
    n_errors_after:      int
    changes:             list[str]
    candidates_tried:    list[dict]   # {source, param, value, score}
    constraint_violations: list[str]  # human-readable constraint violations
    explore_exploit:     str          # "explore" | "exploit" | "unknown"
    coupling_boosts:     list[str]    # parameters that got coupling boosts
    cache_hits:          int
    elapsed_s:           float
    # Fix 2: coordinate descent instrumentation
    coord_descent_triggered:      bool          = False
    coord_descent_level:          int           = -1
    coord_descent_improvement:    float         = 0.0
    # Fix 3: per-stage error counts from REPAIR_STAGE_LOG
    stage_after_beam_errors:      Optional[int] = None
    stage_after_local_opt_errors: Optional[int] = None


@dataclass
class RunLog:
    """Full trajectory log for one pipeline run."""
    case_id:        str
    ablation_mode:  str
    model:          str
    timestamp:      str
    outcome:        str
    total_elapsed_s: float

    iterations:     list[IterationLog] = field(default_factory=list)
    ir_report_json: Optional[dict]     = None
    final_graph_summary: Optional[dict] = None
    warnings:       list[str]          = field(default_factory=list)
    # Completeness-verification loop diagnostic (None unless the loop ran):
    # {pre_loop_n_units, post_loop_n_units, iterations:[{claimed, accepted,
    #  rejected(+reason), n_before, n_after}]} — claimed-missing units + spans.
    completeness_critic: Optional[dict] = None
    # Converged stream conditions from the system's SOLVED flowsheet — evidence of
    # what the system produced, independent of the PASS decision.  Matches the
    # reference_flowsheets/*.json stream format for direct comparison.  None when
    # DWSIM produced no stream results.
    system_streams: Optional[dict] = None
    # Reference-comparison results (also in the aggregate metrics; persisted here
    # per-run too): match pass + MAPEs + per-check detail.  None when no reference.
    reference_comparison: Optional[dict] = None
    # Property-package family selection scored against the case's expected family
    # label (expected.property_package_class).  None when the case has no label.
    package_family: Optional[dict] = None
    # Solve completeness — fully_solved is False when any non-feed stream is left
    # at uncomputed default values (a downstream unit failed).  Gates reference-MAPE
    # so a partial-solve MAPE is not reported as valid correctness.
    fully_solved:   Optional[bool] = None
    n_units_solved: Optional[int]  = None
    n_units_total:  Optional[int]  = None
    # Best-of-N sampling audit (None when best-of-N is off / N=1). Records N, the
    # selected sample, the reference-BLIND selection reason, and every sample's
    # reference-blind signals — so the selection is auditable and demonstrably
    # never uses reference/MAPE data.
    best_of_n:      Optional[dict] = None

    @property
    def score_curve(self) -> list[int]:
        """n_errors_before per iteration — the convergence curve."""
        return [it.n_errors_before for it in self.iterations]

    @property
    def n_explore(self) -> int:
        return sum(1 for it in self.iterations if it.explore_exploit == "explore")

    @property
    def n_exploit(self) -> int:
        return sum(1 for it in self.iterations if it.explore_exploit == "exploit")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score_curve"]   = self.score_curve
        d["n_explore"]     = self.n_explore
        d["n_exploit"]     = self.n_exploit
        return d

    def save(self, results_dir: str = "results/per_run") -> str:
        os.makedirs(results_dir, exist_ok=True)
        fname = f"{self.case_id}_{self.ablation_mode}_{self.timestamp}.json"
        path  = os.path.join(results_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path


# ── Extractor ──────────────────────────────────────────────────────────────────

def extract_system_streams(pipeline_result) -> Optional[dict]:
    """
    Converged material-stream table from the system's SOLVED flowsheet, in the
    reference_flowsheets/*.json stream format (T_K/T_C/P_Pa/P_bar/flow_mol_s/
    vapor_fraction/composition) so it is directly comparable to the reference.

    Reads pipeline_result.final_execution.stream_results (dict[tag → StreamResult]).
    Returns None when DWSIM produced no converged stream results.  Pure logging.
    """
    execution = getattr(pipeline_result, "final_execution", None)
    sr = getattr(execution, "stream_results", None) if execution is not None else None
    if not sr:
        return None
    out: dict = {}
    for tag, s in sr.items():
        T = getattr(s, "T_K", None)
        P = getattr(s, "P_Pa", None)
        out[str(tag)] = {
            "T_K":            (round(float(T), 3) if T is not None else None),
            "T_C":            (round(float(T) - 273.15, 3) if T is not None else None),
            "P_Pa":           (round(float(P), 2) if P is not None else None),
            "P_bar":          (round(float(P) / 1e5, 5) if P is not None else None),
            "flow_mol_s":     getattr(s, "flow_mol_s", None),
            "vapor_fraction": getattr(s, "vapor_fraction", None),
            "composition":    dict(getattr(s, "composition", {}) or {}),
            "is_feed":        getattr(s, "is_feed", None),
        }
    return out


def extract_run_log(
    pipeline_result,
    case_id:      str,
    ablation_mode: str = "full",
    model:        str  = "",
) -> RunLog:
    """
    Build a RunLog from an OrchestratorV2 PipelineResult.

    Reads iteration records and extracts as much search-behaviour
    detail as the PipelineResult exposes.
    """
    pr        = pipeline_result
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outcome   = getattr(pr, "outcome", "UNKNOWN")
    elapsed   = getattr(pr, "total_time_s", 0.0)
    warnings  = list(getattr(pr, "warnings", []))

    iter_logs: list[IterationLog] = []
    raw_iters = getattr(pr, "iterations", [])

    for i, rec in enumerate(raw_iters):
        errors      = list(getattr(rec, "errors", []))
        changes     = list(getattr(rec, "changes", []))
        n_before    = len(errors)
        n_after     = len(getattr(raw_iters[i+1], "errors", []))  \
                      if i + 1 < len(raw_iters) else 0
        it_elapsed  = getattr(rec, "elapsed_s", 0.0)

        # Parse structured log entries BEFORE truncating changes
        cd_triggered        = False
        cd_level            = -1
        cd_improvement      = 0.0
        stage_after_beam    = None
        stage_after_local   = None
        for chg in changes:
            if not isinstance(chg, str):
                continue
            if chg.startswith("LOCAL_OPT_LOG:"):
                try:
                    p = json.loads(chg[len("LOCAL_OPT_LOG:"):])
                    cd_triggered   = p.get("triggered", False)
                    cd_level       = p.get("level", -1)
                    cd_improvement = p.get("improvement", 0.0)
                except Exception:
                    pass
            elif chg.startswith("REPAIR_STAGE_LOG:"):
                try:
                    p = json.loads(chg[len("REPAIR_STAGE_LOG:"):])
                    stage_after_beam  = p.get("after_beam")
                    stage_after_local = p.get("after_local_opt")
                except Exception:
                    pass

        # Parse explore/exploit from change log
        phase = "unknown"
        for chg in changes:
            if isinstance(chg, str):
                if "EXPLORE" in chg.upper():
                    phase = "explore"
                elif "EXPLOIT" in chg.upper():
                    phase = "exploit"

        # Constraint violations from errors
        violations = [str(e) for e in errors]

        # Coupling boosts
        boosts = [c for c in changes
                  if isinstance(c, str) and "COUPLING" in c.upper()]

        # Cache hits
        cache_hits = sum(1 for c in changes
                         if isinstance(c, str) and "CACHE" in c.upper())

        # Candidate info (best-effort parse)
        candidates: list[dict] = []
        for chg in changes:
            if isinstance(chg, str) and ("→" in chg or ":=" in chg or "->" in chg):
                candidates.append({"change": chg, "source": "parsed"})
            elif isinstance(chg, dict):
                candidates.append(chg)

        iter_logs.append(IterationLog(
            iteration            = i,
            n_errors_before      = n_before,
            n_errors_after       = n_after,
            changes              = [str(c) for c in changes[:30]],
            candidates_tried     = candidates[:20],
            constraint_violations = violations[:20],
            explore_exploit      = phase,
            coupling_boosts      = boosts[:10],
            cache_hits           = cache_hits,
            elapsed_s            = it_elapsed,
            coord_descent_triggered      = cd_triggered,
            coord_descent_level          = cd_level,
            coord_descent_improvement    = cd_improvement,
            stage_after_beam_errors      = stage_after_beam,
            stage_after_local_opt_errors = stage_after_local,
        ))

    # Completeness-loop diagnostic (None unless the loop ran)
    completeness = getattr(pr, "completeness", None)

    # Graph summary
    graph = getattr(pr, "final_graph", None)
    graph_summary: Optional[dict] = None
    if graph is not None:
        try:
            units  = list(graph.units()) if hasattr(graph, "units") else []
            graph_summary = {
                "property_package": getattr(graph, "property_package", ""),
                "n_units": len(units),
                "unit_types": [getattr(u, "unit_type", str(u)) for u in units],
                "n_binary_params": len(getattr(graph, "binary_parameters", [])),
            }
            # Pre-loop unit count (before completeness augmentation), when available.
            if isinstance(completeness, dict):
                graph_summary["n_units_pre_loop"] = completeness.get("pre_loop_n_units")
                graph_summary["n_units_post_loop"] = completeness.get("post_loop_n_units")

            # Per-unit temperature provenance so every unit's operating condition
            # is auditable: temperature_source ∈ {specified, extracted, template,
            # computed, inherited, default_fallback, unknown}.  Reactors also carry
            # reaction_type + basis.
            unit_conditions = []
            for u in units:
                p = getattr(u, "params", {}) or {}
                ut = getattr(u, "unit_type", "")
                entry = {
                    "tag":                getattr(u, "tag", ""),
                    "type":               ut,
                    "T_K":                p.get("temperature_K", p.get("T_out")),
                    "temperature_source": p.get("_temperature_source", "unknown"),
                }
                if ut == "ConversionReactor":
                    entry["reaction_type"] = p.get("_reaction_type")
                    entry["basis"]         = p.get("_reactor_T_basis")
                unit_conditions.append(entry)
            if unit_conditions:
                graph_summary["unit_conditions"] = unit_conditions
        except Exception:
            pass

    # IR report summary
    ir_report = getattr(pr, "ir_report", None)
    ir_json: Optional[dict] = None
    if ir_report is not None:
        try:
            issues = getattr(ir_report, "issues", [])
            ir_json = {
                "valid": getattr(ir_report, "valid", False),
                "n_issues": len(issues),
                "issue_summaries": [str(i)[:120] for i in issues[:10]],
            }
        except Exception:
            pass

    return RunLog(
        case_id          = case_id,
        ablation_mode    = ablation_mode,
        model            = model,
        timestamp        = timestamp,
        outcome          = outcome,
        total_elapsed_s  = elapsed,
        iterations       = iter_logs,
        ir_report_json   = ir_json,
        final_graph_summary = graph_summary,
        warnings         = warnings[:20],
        completeness_critic = completeness if isinstance(completeness, dict) else None,
    )


# ── Log loading ────────────────────────────────────────────────────────────────

def load_run_log(path: str) -> dict:
    """Load a saved run log JSON as a plain dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_logs(results_dir: str = "results/per_run") -> list[dict]:
    """Load all run log JSON files from a directory."""
    logs = []
    if not os.path.isdir(results_dir):
        return logs
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            try:
                logs.append(load_run_log(os.path.join(results_dir, fname)))
            except Exception:
                pass
    return logs
