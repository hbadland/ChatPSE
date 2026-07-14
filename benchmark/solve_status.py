"""
Solve-completeness detection for reference-MAPE reporting integrity.

A MAPE computed over a PARTIALLY-solved flowsheet is misleading: the matched
streams are the ones that converged, while units downstream of a failure sit at
their uncomputed default values, so a low MAPE is a partial-solve artifact — as
misleading as a MAPE over 1 matched stream. This module detects partial solves
from the persisted system_streams so reference-MAPE can be gated on fully_solved.

Detection is stream-based (works retroactively on stored JSONs, which carry no
per-unit solved flag): a NON-feed stream still sitting at the executor's
uncomputed default — T=298.15 K, flow=1.0 mol/s, exactly equimolar composition —
never solved. A feed legitimately has those values (is_feed excludes it), and a
genuinely-solved equimolar pass-through has a computed T != 298.15 (verified on
VAL_04, whose solved COMP/MIX are flow=1.0/equimolar at T=352 K, NOT flagged).
"""
from __future__ import annotations
import re

_DEFAULT_T   = 298.15
_DEFAULT_FLOW = 1.0


def stream_is_default(s: dict) -> bool:
    """True if a stream is a non-feed stream still at uncomputed default values."""
    if not isinstance(s, dict) or s.get("is_feed"):
        return False
    T    = s.get("T_K")
    flow = s.get("flow_mol_s")
    comp = s.get("composition") or {}
    if T != _DEFAULT_T or flow != _DEFAULT_FLOW or not comp:
        return False
    n = len(comp)
    return all(abs(v - 1.0 / n) < 1e-6 for v in comp.values())


def _unit_prefix(tag: str) -> str | None:
    """Source-unit tag for a stream tag like 'COL-02-BOT'→'COL-02', 'RX-05-OUT1'→
    'RX-05'. Semantic stream names (TOL, WATER, RECYCLE) don't map — returns None."""
    m = re.match(r"([A-Za-z]+-\d+)", tag or "")
    return m.group(1) if m else None


def compute_solve_status(system_streams: dict | None,
                         n_units_total: int | None = None) -> dict:
    """
    Returns a solve-status block:
      fully_solved          — True iff no non-feed stream is at default (exact)
      n_streams_total / n_streams_at_default
      streams_at_default    — the offending tags
      n_units_total         — passed through (from final_graph_summary.n_units)
      n_units_solved        — best-effort: n_units_total when fully solved; else
                              n_units_total minus units implicated by default
                              stream tags (approximate — semantic-named default
                              streams may not map to a unit, so this is a floor).
    """
    if not system_streams:
        return {
            "fully_solved": False, "reason": "no_streams",
            "n_streams_total": 0, "n_streams_at_default": 0,
            "streams_at_default": [],
            "n_units_total": n_units_total,
            "n_units_solved": (0 if n_units_total is not None else None),
        }
    defaults = [t for t, s in system_streams.items() if stream_is_default(s)]
    fully = (len(defaults) == 0)
    n_units_solved = None
    if n_units_total is not None:
        if fully:
            n_units_solved = n_units_total
        else:
            implicated = {p for t in defaults if (p := _unit_prefix(t)) is not None}
            n_unsolved = len(implicated) if implicated else 1
            n_units_solved = max(n_units_total - n_unsolved, 0)
    return {
        "fully_solved": fully,
        "n_streams_total": len(system_streams),
        "n_streams_at_default": len(defaults),
        "streams_at_default": defaults,
        "n_units_total": n_units_total,
        "n_units_solved": n_units_solved,
    }


def gate_mape_status(fully_solved: bool, ref_sufficient: bool) -> str:
    """Precedence: a flowsheet that didn't fully solve can't yield a correctness
    MAPE regardless of match count; then the match-count gate; else computed."""
    if not fully_solved:
        return "partial_solve"
    if not ref_sufficient:
        return "insufficient_match"
    return "computed"
