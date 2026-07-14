"""
Best-of-N sampling for extraction — mitigate run-to-run nondeterminism (MoE
routing variance drops units / hallucinates unseparable compounds). Run the
pipeline N times and select the best VALID build.

CRITICAL — selection is strictly REFERENCE-BLIND. It must NEVER rank on
reference-MAPE or any reference comparison (that would select on the test answer,
invalidating the benchmark). The two functions here are deliberately isolated:

  sample_signals(pr, system_streams)  — extracts ONLY produced-flowsheet signals
      from a single run (solve state, outcome, IR errors, iterations). It does not
      receive, load, or read any reference file / reference_comparison / MAPE.
  select_best(signals_list)           — ranks per (a)>(b)>(c) below. It receives
      ONLY the list of signal dicts — no pipeline result, no reference — so it is
      structurally incapable of reading the answer.

Priority (all reference-independent), higher is better:
  (a) solve tier:  fully_solved > partial_solve > failed
  (b) among equal: converged (outcome PASS), then fewest CRITICAL IR errors
  (c) among equal: fewest repair iterations needed
"""
from __future__ import annotations

from benchmark.solve_status import compute_solve_status

_SOLVE_TIER = {"failed": 0, "partial_solve": 1, "fully_solved": 2}


def sample_signals(pr, system_streams: dict | None) -> dict:
    """Reference-BLIND signals from ONE pipeline run. Reads only the produced
    flowsheet / solve state — never any reference or MAPE data."""
    solve = compute_solve_status(system_streams, None)
    if not system_streams:
        tier = "failed"
    elif solve["fully_solved"]:
        tier = "fully_solved"
    else:
        tier = "partial_solve"
    outcome  = str(getattr(pr, "outcome", "") or "")
    warnings = [str(w) for w in (getattr(pr, "warnings", []) or [])]
    n_critical_ir = sum(1 for w in warnings if "CRITICAL" in w)
    n_iterations  = len(getattr(pr, "iterations", []) or [])
    return {
        "solve_tier":           tier,
        "outcome":              outcome,
        "passed":               outcome == "PASS",
        "n_critical_ir_errors": n_critical_ir,
        "n_iterations":         n_iterations,
        "n_streams_at_default": solve["n_streams_at_default"],
    }


def _rank_key(sig: dict) -> tuple:
    """Sort key (higher is better). Reference-blind priority a > b > c."""
    return (
        _SOLVE_TIER.get(sig.get("solve_tier"), 0),   # (a) solve tier
        1 if sig.get("passed") else 0,               # (b) converged (PASS)
        -(sig.get("n_critical_ir_errors") or 0),     # (b) fewest CRITICAL IR errors
        -(sig.get("n_iterations") or 0),             # (c) fewest repair iterations
    )


def select_best(signals_list: list[dict]) -> tuple[int, str]:
    """Return (best_index, human-readable reason) from per-sample signals ONLY.

    This function receives no pipeline result and no reference data — it cannot
    read the test answer. Ranks by _rank_key; ties keep the earliest sample.
    """
    if not signals_list:
        return 0, "no samples"
    best_i = 0
    for i in range(1, len(signals_list)):
        if _rank_key(signals_list[i]) > _rank_key(signals_list[best_i]):
            best_i = i
    b = signals_list[best_i]
    reason = (
        f"reference-blind argmax over {len(signals_list)} samples on "
        f"(solve_tier > passed > -critical_ir_errors > -iterations): "
        f"selected sample {best_i} with solve_tier={b['solve_tier']}, "
        f"passed={b['passed']}, critical_ir_errors={b['n_critical_ir_errors']}, "
        f"iterations={b['n_iterations']}"
    )
    return best_i, reason
