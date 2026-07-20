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
never solved. A feed legitimately has those values (is_feed excludes it).

Value-sniffing alone false-positives when a genuinely-solved OUTPUT coincides
with the defaults: a cooler outlet specified to 25 C (=298.15 K), or a pure /
equimolar feed passed through, reads exactly T=298.15 / flow=1.0 / "equimolar"
(trivially true for a single component). To disambiguate we compare against the
flowsheet's SPECIFIED unit outlet temperatures: a stream whose temperature
matches a specified setpoint is that unit's solved output, not an uncomputed
default. (VAL_04's real partial solve is unaffected — its failed outlet sits at
298.15 while every specified reformer/shift outlet is high-T, so no match.)
"""
from __future__ import annotations
import re

_DEFAULT_T   = 298.15
_DEFAULT_FLOW = 1.0
_T_MATCH_TOL = 0.5   # K — tolerance for matching a stream T to a specified setpoint


def specified_outlet_temps(unit_conditions) -> list[float]:
    """Collect the specified/expected outlet temperatures [K] from a run's
    final_graph_summary.unit_conditions, for disambiguating default coincidences."""
    out: list[float] = []
    for u in unit_conditions or []:
        t = u.get("T_K") if isinstance(u, dict) else None
        if isinstance(t, (int, float)):
            out.append(float(t))
    return out


def stream_is_default(s: dict, specified_temps: "list[float] | None" = None) -> bool:
    """True if a stream is a non-feed stream still at uncomputed default values.

    specified_temps: specified unit outlet temperatures [K]. A non-feed stream
    at the default temperature that matches one of these is a solved output that
    merely coincides with the default (not uncomputed), so it is NOT flagged.
    """
    if not isinstance(s, dict) or s.get("is_feed"):
        return False
    T    = s.get("T_K")
    flow = s.get("flow_mol_s")
    comp = s.get("composition") or {}
    if T != _DEFAULT_T or flow != _DEFAULT_FLOW or not comp:
        return False
    n = len(comp)
    if not all(abs(v - 1.0 / n) < 1e-6 for v in comp.values()):
        return False
    # Solved output coinciding with the default T (e.g. a 25 C cooler outlet):
    # if T matches a specified unit setpoint, it is solved, not an uncomputed default.
    if specified_temps:
        for st in specified_temps:
            if st is not None and abs(float(st) - float(T)) <= _T_MATCH_TOL:
                return False
    return True


def _unit_prefix(tag: str) -> str | None:
    """Source-unit tag for a stream tag like 'COL-02-BOT'→'COL-02', 'RX-05-OUT1'→
    'RX-05'. Semantic stream names (TOL, WATER, RECYCLE) don't map — returns None."""
    m = re.match(r"([A-Za-z]+-\d+)", tag or "")
    return m.group(1) if m else None


def compute_solve_status(system_streams: dict | None,
                         n_units_total: int | None = None,
                         specified_temps: "list[float] | None" = None) -> dict:
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
    defaults = [t for t, s in system_streams.items()
                if stream_is_default(s, specified_temps)]
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


def exact_units_solved(graph, system_streams: dict | None):
    """
    EXACT (n_units_solved, n_units_total) from the live graph's unit↔stream
    connectivity — for fresh runs, where the FlowsheetGraph is available. A unit
    is solved iff NONE of its outlet streams is at default. This is precise where
    compute_solve_status()'s tag-prefix mapping is only a floor (it cannot
    attribute semantic-named default streams like TOL/WATER to their source unit).

    Returns (None, None) if the graph or streams are unavailable, so the caller
    keeps the stream-based approximation. fully_solved / n_streams_at_default are
    always stream-based and exact regardless.
    """
    if graph is None or not system_streams:
        return None, None
    try:
        units = list(graph.units())
    except Exception:
        return None, None
    n_solved = 0
    for u in units:
        try:
            outs = graph.outlet_streams(u.tag)
        except Exception:
            outs = []
        # Compare each outlet against THIS unit's own specified outlet temperature
        # (exact) rather than the coarse union — so a 25 C cooler outlet is solved
        # while a failed unit's 298.15 default (with a non-default spec) is flagged.
        p = getattr(u, "params", {}) or {}
        utemp = p.get("temperature_K", p.get("T_out"))
        temps = [float(utemp)] if isinstance(utemp, (int, float)) else None
        unsolved = any(stream_is_default(system_streams.get(s.tag, {}), temps)
                       for s in outs)
        if not unsolved:
            n_solved += 1
    return n_solved, len(units)


def gate_mape_status(fully_solved: bool, ref_sufficient: bool) -> str:
    """Precedence: a flowsheet that didn't fully solve can't yield a correctness
    MAPE regardless of match count; then the match-count gate; else computed."""
    if not fully_solved:
        return "partial_solve"
    if not ref_sufficient:
        return "insufficient_match"
    return "computed"
