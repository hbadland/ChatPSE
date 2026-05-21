"""
benchmark/visualisation.py

Generates plot-ready data structures from DiagnosticReport.

All functions return plain dicts/lists that serialise directly to JSON
and map onto standard matplotlib / plotly calls.  No rendering is done
here — call save_plot_data() to write everything to one JSON file, then
consume with your plotting library of choice.

Expected matplotlib usage (illustrative):

    import json, matplotlib.pyplot as plt
    data = json.load(open("results/diagnostics/plots_20250521.json"))

    # Convergence curves
    curves = data["convergence_curves"]["full_ccs"]
    for case_id, curve in curves.items():
        plt.plot(curve, label=case_id, alpha=0.6)
    plt.xlabel("Iteration"); plt.ylabel("IR errors"); plt.legend(); plt.show()

    # Tier heatmap
    hm = data["tier_heatmap"]
    Z  = hm["data"]   # shape: [n_modes][n_tiers]
    plt.imshow(Z, aspect="auto")
    plt.xticks(range(len(hm["tiers"])), hm["tiers"], rotation=30)
    plt.yticks(range(len(hm["modes"])), hm["modes"])
    plt.colorbar(label="Success rate"); plt.show()
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Optional

from benchmark.diagnostics import DiagnosticReport, CaseDiagnostic


# ══════════════════════════════════════════════════════════════════════════════
# Individual plot-data generators
# ══════════════════════════════════════════════════════════════════════════════

def convergence_curves(report: DiagnosticReport) -> dict:
    """
    Returns:
        {
          "<mode>": {
            "<case_id>": [n_errors_0, n_errors_1, ...],
            ...
          },
          ...
        }

    x-axis: iteration index
    y-axis: IR error count at that iteration (from score_trajectory)

    Suitable for: line plots per mode, one trace per case.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        result[mode] = {
            d.case_id: list(d.metrics.score_trajectory)
            for d in diags
            if d.metrics.score_trajectory
        }
    return result


def ablation_bar_data(report: DiagnosticReport) -> dict:
    """
    Returns:
        {
          "<metric>": {
            "<mode>": <value>,
            ...
          }
        }

    Metrics exported: success_rate, physics_pass_rate, valid_ir_rate,
    mean_iterations, mean_candidates, oscillation_rate, recurrence_rate,
    error_reduction_per_candidate, mean_explore_ratio.

    Suitable for: grouped bar charts (modes on x-axis, one group per metric).
    """
    data: dict = {
        "success_rate":                   {},
        "physics_pass_rate":              {},
        "valid_ir_rate":                  {},
        "mean_iterations":                {},
        "mean_candidates":                {},
        "oscillation_rate":               {},
        "recurrence_rate":                {},
        "error_reduction_per_candidate":  {},
        "mean_explore_ratio":             {},
    }

    for mode, rs in report.run_sets.items():
        agg = rs.aggregate
        diags = report.diagnostics.get(mode, [])
        if agg is None:
            continue
        data["success_rate"][mode]                   = round(agg.success_rate, 4)
        data["physics_pass_rate"][mode]              = round(agg.physics_pass_rate, 4)
        data["valid_ir_rate"][mode]                  = round(agg.valid_ir_rate, 4)
        data["mean_iterations"][mode]                = round(agg.mean_iterations, 3)
        data["mean_candidates"][mode]                = round(agg.mean_candidates, 3)
        data["oscillation_rate"][mode]               = round(agg.pct_score_oscillated, 4)
        data["recurrence_rate"][mode]                = round(agg.mean_recurrence_rate, 4)
        data["error_reduction_per_candidate"][mode]  = round(
            agg.mean_error_reduction_per_candidate, 4)
        data["mean_explore_ratio"][mode]             = round(agg.mean_explore_ratio, 4)

    return data


def tier_heatmap(report: DiagnosticReport) -> dict:
    """
    Returns:
        {
          "tiers": ["sanity", "easy", ...],
          "modes": ["full_ccs", "no_physics", ...],
          "data":  [[<success_rate>, ...], ...],   # indexed [mode_idx][tier_idx]
          "metric": "success_rate"
        }

    Suitable for: imshow / seaborn heatmap.
    """
    all_tiers: list = []
    for rs in report.run_sets.values():
        for t in rs.tier_aggregates:
            if t not in all_tiers:
                all_tiers.append(t)

    modes  = report.modes
    matrix = []
    for mode in modes:
        rs  = report.run_sets.get(mode)
        row = []
        for tier in all_tiers:
            ta = rs.tier_aggregates.get(tier) if rs else None
            row.append(round(ta.success_rate, 4) if ta else None)
        matrix.append(row)

    return {"tiers": all_tiers, "modes": modes, "data": matrix, "metric": "success_rate"}


def oscillation_frequency(report: DiagnosticReport) -> dict:
    """
    Returns:
        {
          "<mode>": {
            "<case_id>": <oscillation_event_count>,
            ...
          }
        }

    Suitable for: bar chart per mode; x-axis = case id, y = event count.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        result[mode] = {d.case_id: d.search_trace.oscillation_events for d in diags}
    return result


def beam_diversity_distribution(report: DiagnosticReport) -> dict:
    """
    Returns:
        {
          "<mode>": {
            "diversity_rates": [<float>, ...],   # one per case
            "case_ids":        ["EASY_01", ...],
            "mean":            <float>,
            "std":             <float>,
          }
        }

    Suitable for: histogram or violin plot per mode.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        rates = [d.search_trace.diversity_acceptance_rate for d in diags]
        ids   = [d.case_id for d in diags]
        result[mode] = {
            "diversity_rates": rates,
            "case_ids":        ids,
            "mean": round(statistics.mean(rates), 4) if rates else 0.0,
            "std":  round(statistics.stdev(rates), 4) if len(rates) >= 2 else 0.0,
        }
    return result


def margin_stability_over_time(report: DiagnosticReport) -> dict:
    """
    Returns per-case margin proxy data (score change magnitude per iteration):
        {
          "<mode>": {
            "<case_id>": {
              "magnitudes": [<abs_delta_per_iter>, ...],
              "drift": <bool>,
              "variance": <float>,
            }
          }
        }

    Suitable for: line plot of magnitude over iteration.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        result[mode] = {}
        for d in diags:
            traj = d.metrics.score_trajectory
            if len(traj) >= 2:
                mags = [abs(traj[i] - traj[i - 1]) for i in range(1, len(traj))]
            else:
                mags = []
            result[mode][d.case_id] = {
                "magnitudes": mags,
                "drift":      d.margin_model.drift_detected,
                "variance":   d.margin_model.margin_variance,
            }
    return result


def coupling_effectiveness(report: DiagnosticReport) -> dict:
    """
    Returns:
        {
          "<mode>": {
            "case_ids":              [...],
            "corrections_triggered": [...],
            "ping_pong_events":      [...],
            "settler_resolved":      [...],
            "propagation_lag_steps": [...],
          }
        }

    Suitable for: stacked bar or grouped bar chart.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        result[mode] = {
            "case_ids":              [d.case_id for d in diags],
            "corrections_triggered": [d.coupling.corrections_triggered for d in diags],
            "ping_pong_events":      [d.coupling.ping_pong_events for d in diags],
            "settler_resolved":      [d.coupling.coupled_settler_resolved for d in diags],
            "propagation_lag_steps": [d.coupling.propagation_lag_steps for d in diags],
        }
    return result


def repair_source_breakdown(report: DiagnosticReport) -> dict:
    """
    Returns per-mode aggregate repair source success rates:
        {
          "<mode>": {
            "deterministic": <mean_success_rate>,
            "physics":        <mean_success_rate>,
            "llm":            <mean_success_rate>,
          }
        }

    Suitable for: stacked or grouped bar.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        def _mean_rate(attr: str) -> float:
            vals = [getattr(d.repair_dynamics, attr) for d in diags]
            return round(statistics.mean(vals), 4) if vals else 0.0
        result[mode] = {
            "deterministic": _mean_rate("success_rate_deterministic"),
            "physics":       _mean_rate("success_rate_physics"),
            "llm":           _mean_rate("success_rate_llm"),
        }
    return result


def generalisation_split(report: DiagnosticReport) -> dict:
    """
    Returns known vs unseen compound success rates per mode:
        {
          "<mode>": {
            "known":  {"n": ..., "success_rate": ..., "avg_ir_reduction": ...},
            "unseen": {...},
            "gap":    <float>,
          }
        }
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        known  = [d for d in diags if d.compound_familiarity == "known"]
        unseen = [d for d in diags if d.compound_familiarity == "unseen"]

        def _stats(subset) -> dict:
            if not subset:
                return {"n": 0, "success_rate": None, "avg_ir_reduction": None}
            succ = sum(d.metrics.success for d in subset) / len(subset)
            reds = [(t[0] - t[-1]) for d in subset
                    if (t := d.metrics.score_trajectory) and len(t) >= 2]
            return {
                "n":               len(subset),
                "success_rate":    round(succ, 4),
                "avg_ir_reduction": round(statistics.mean(reds), 3) if reds else 0.0,
            }

        ks = _stats(known)
        us = _stats(unseen)
        gap = ((ks["success_rate"] or 0) - (us["success_rate"] or 0))
        result[mode] = {"known": ks, "unseen": us, "gap": round(gap, 4)}

    return result


def credit_label_distribution(report: DiagnosticReport) -> dict:
    """
    Returns distribution of trajectory credit labels per mode:
        {
          "<mode>": {
            "CREDIT_ALIGNMENT_GOOD": <count>,
            "CREDIT_COLLAPSE":       <count>,
            "CREDIT_DIFFUSION":      <count>,
            "MIXED":                 <count>,
            "INSUFFICIENT_DATA":     <count>,
            "NO_IMPROVEMENT":        <count>,
          }
        }

    Suitable for: pie chart or bar chart.
    """
    result: dict = {}
    for mode, diags in report.diagnostics.items():
        counts: dict = {}
        for d in diags:
            lbl = d.trajectory_credit.credit_label
            counts[lbl] = counts.get(lbl, 0) + 1
        result[mode] = counts
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Combined save
# ══════════════════════════════════════════════════════════════════════════════

def all_plot_data(report: DiagnosticReport) -> dict:
    """
    Bundle all plot-data generators into a single dict keyed by plot name.
    Call this then json.dump() the result for offline plotting.
    """
    return {
        "convergence_curves":         convergence_curves(report),
        "ablation_bar_data":          ablation_bar_data(report),
        "tier_heatmap":               tier_heatmap(report),
        "oscillation_frequency":      oscillation_frequency(report),
        "beam_diversity_distribution": beam_diversity_distribution(report),
        "margin_stability_over_time": margin_stability_over_time(report),
        "coupling_effectiveness":     coupling_effectiveness(report),
        "repair_source_breakdown":    repair_source_breakdown(report),
        "generalisation_split":       generalisation_split(report),
        "credit_label_distribution":  credit_label_distribution(report),
    }


def save_plot_data(
    report: DiagnosticReport,
    out_dir: str = "results/diagnostics",
) -> str:
    """
    Write all plot data to a single JSON file.

    Returns the file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"plots_{report.timestamp}.json"
    path  = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_plot_data(report), f, indent=2, default=str)
    return path
