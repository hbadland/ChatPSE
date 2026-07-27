"""
Cross-tag stream matcher for reference-MAPE.

System stream tags are unit-derived (FEED, COMP, MIX, REFORM, SHIFT); reference
tags are process-derived (BIOGAS, WATER, S-01…S-09, H2).  Exact-tag matching
yields ZERO pairs, so the reference-MAPE previously ran over nothing.

This matches streams by CONTENT, not tag:
  * composition-vector similarity (cosine) with compound names canonicalised to
    DWSIM keys (so 'Carbon Monoxide'/'Carbon monoxide' etc. align) — the primary,
    feed/product-anchoring signal, since feeds and products carry the most
    distinctive compositions and therefore the highest, least-ambiguous scores;
  * T, P, vapour-fraction closeness as tiebreakers;
  * an is_feed agreement bonus when both sides label feeds.

Global greedy assignment by confidence, each stream used once.  CONSERVATIVE:
pairs below `threshold` are left UNMATCHED — no forced matches.  MAPE is computed
by the caller over the confident pairs only.

Topological-position alignment (walking feed→product on both graphs) is only
possible when the reference ships populated `connections`; almost all current
references ship empty connections, so composition-anchoring carries the match.
When `ref_depths`/`sys_depths` are supplied, a position-similarity term is added.
"""
from __future__ import annotations

import math
from typing import Optional

from agents.compound_normalize import canonicalize_compound

# Weights — composition dominates; T/P/vf are tiebreakers; feed-agreement a nudge.
_W_COMP, _W_T, _W_P, _W_VF, _W_FEED, _W_POS = 0.55, 0.15, 0.15, 0.08, 0.04, 0.10
DEFAULT_THRESHOLD = 0.55


def _canon_comp(comp: Optional[dict]) -> dict:
    out: dict = {}
    for k, v in (comp or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        ck, _ = canonicalize_compound(k)
        out[ck] = out.get(ck, 0.0) + v
    return out


def _cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _diff(sv, rv):
    return (float(sv) - float(rv)) if (sv is not None and rv is not None) else None


def _rel_sim(sv, rv, floor):
    if sv is None or rv is None:
        return None
    return max(0.0, 1.0 - abs(float(sv) - float(rv)) / max(abs(float(rv)), floor))


def _pair_confidence(sys_s: dict, ref_s: dict,
                     sys_depth: Optional[float], ref_depth: Optional[float]):
    """Return (confidence, dT, dP, dvf, comp_cosine)."""
    comp = _cosine(_canon_comp(sys_s.get("composition")),
                   _canon_comp(ref_s.get("composition")))
    dT  = _diff(sys_s.get("T_K"),  ref_s.get("T_K"))
    dP  = _diff(sys_s.get("P_Pa"), ref_s.get("P_Pa"))
    dvf = _diff(sys_s.get("vapor_fraction"), ref_s.get("vapor_fraction"))

    terms = [(comp, _W_COMP)]
    t_sim = _rel_sim(sys_s.get("T_K"),  ref_s.get("T_K"),  50.0)
    p_sim = _rel_sim(sys_s.get("P_Pa"), ref_s.get("P_Pa"), 1e4)
    vf_sim = (1.0 - min(abs(dvf), 1.0)) if dvf is not None else None
    feed_sim = 1.0 if (sys_s.get("is_feed") and ref_s.get("is_feed")) else None
    pos_sim = (1.0 - min(abs(sys_depth - ref_depth), 1.0)
               if (sys_depth is not None and ref_depth is not None) else None)
    for s, w in ((t_sim, _W_T), (p_sim, _W_P), (vf_sim, _W_VF),
                 (feed_sim, _W_FEED), (pos_sim, _W_POS)):
        if s is not None:
            terms.append((s, w))
    wsum = sum(w for _, w in terms)
    conf = sum(s * w for s, w in terms) / wsum if wsum else comp
    return conf, dT, dP, dvf, comp


def _assign_greedy(rts, sts, detail, threshold):
    """Highest-confidence-first, each stream used once (the historical matcher).
    Retained as the fallback when SciPy is unavailable."""
    scored = sorted(
        ((detail[(rt, st)][0], rt, st) for rt in rts for st in sts),
        key=lambda x: x[0], reverse=True)
    used_ref: set = set()
    used_sys: set = set()
    out = []
    for conf, rt, st in scored:
        if conf < threshold:
            break
        if rt in used_ref or st in used_sys:
            continue
        used_ref.add(rt)
        used_sys.add(st)
        _c, dT, dP, dvf, comp = detail[(rt, st)]
        out.append((_c, rt, st, dT, dP, dvf, comp))
    return out


def _assign_global(rts, sts, detail, threshold):
    """Hungarian global assignment maximising total confidence (SciPy
    linear_sum_assignment). Sub-threshold assignments are dropped, so the result
    is never a forced match. Falls back to greedy if SciPy is missing — never
    worse than the historical behaviour."""
    if not rts or not sts:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _assign_greedy(rts, sts, detail, threshold)
    cost = np.array([[-detail[(rt, st)][0] for st in sts] for rt in rts])
    ri, ci = linear_sum_assignment(cost)
    out = []
    for i, j in zip(ri, ci):
        conf, dT, dP, dvf, comp = detail[(rts[i], sts[j])]
        if conf < threshold:
            continue
        out.append((conf, rts[i], sts[j], dT, dP, dvf, comp))
    return out


def match_streams(
    sys_streams: dict,
    ref_streams: dict,
    threshold: float = DEFAULT_THRESHOLD,
    sys_depths: Optional[dict] = None,
    ref_depths: Optional[dict] = None,
) -> dict:
    """
    Match system streams to reference streams by content.

    sys_streams / ref_streams: {tag: {T_K, P_Pa, vapor_fraction, composition,
                                       is_feed?}}
    Returns:
      {pairs:[{ref_tag, sys_tag, confidence, comp_cosine, dT, dP, dvf}...],
       n_matched, n_system_unmatched, n_reference_unmatched,
       system_unmatched:[...], reference_unmatched:[...], threshold}
    """
    sys_depths = sys_depths or {}
    ref_depths = ref_depths or {}

    # Score every (ref, sys) pair once; both assignment strategies read this cache.
    rts = list(ref_streams)
    sts = list(sys_streams)
    detail = {
        (rt, st): _pair_confidence(sys_streams[st], ref_streams[rt],
                                   sys_depths.get(st), ref_depths.get(rt))
        for rt in rts for st in sts
    }

    # GLOBAL assignment (Hungarian): maximise total match confidence, so a small
    # local gain never forces a large loss elsewhere — the greedy matcher's crossed
    # pairings (identical-composition streams matched by the wrong T/P) are avoided.
    # Conservative: assignments below `threshold` are dropped (left UNMATCHED).
    raw = _assign_global(rts, sts, detail, threshold)
    # Confidence-descending order — identical to the greedy output ordering, so a
    # case with no crossing (same pair set) serialises byte-for-byte as before.
    raw.sort(key=lambda x: x[0], reverse=True)

    used_ref: set = set()
    used_sys: set = set()
    pairs: list = []
    for conf, rt, st, dT, dP, dvf, comp in raw:
        used_ref.add(rt)
        used_sys.add(st)
        pairs.append({
            "ref_tag": rt, "sys_tag": st,
            "confidence": round(conf, 4), "comp_cosine": round(comp, 4),
            "dT":  (round(dT, 2)  if dT  is not None else None),
            "dP":  (round(dP, 1)  if dP  is not None else None),
            "dvf": (round(dvf, 4) if dvf is not None else None),
        })

    sys_un = [t for t in sys_streams if t not in used_sys]
    ref_un = [t for t in ref_streams if t not in used_ref]
    return {
        "pairs": pairs,
        "n_matched": len(pairs),
        "n_system_unmatched": len(sys_un),
        "n_reference_unmatched": len(ref_un),
        "system_unmatched": sys_un,
        "reference_unmatched": ref_un,
        "threshold": threshold,
    }
