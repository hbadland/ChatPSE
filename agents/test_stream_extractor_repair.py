"""StreamExtractor deterministic-repair tests.

The LLM extraction step itself is not tested here (non-deterministic, needs a model).
These cover the deterministic repair helpers that make extraction robust to malformed
LLM output — all pure functions, no LLM:

  _best_match          — snap a near-miss unit reference to a real tag
  _reconcile_unit_refs — snap near-misses, null true phantoms, drop isolated streams
  _dedupe_stream_tags  — suffix duplicate stream tags
  _lenient_loads       — recover trailing-comma and truncated JSON

Run: PYTHONPATH=. python3.9 agents/test_stream_extractor_repair.py
"""
import json
from agents.stage1.stream_extractor import (
    SemanticStream, _best_match, _reconcile_unit_refs,
    _dedupe_stream_tags, _lenient_loads,
)


def _s(tag, src, dst, is_feed=False):
    return SemanticStream(tag=tag, src=src, dst=dst, is_feed=is_feed)


# ── _best_match ───────────────────────────────────────────────────────────────

def test_best_match_normalises_separators():
    assert _best_match("HT01", {"HT-01", "CL-02"}) == "HT-01"   # separator/case
    assert _best_match("V_01", {"V-01"}) == "V-01"
    print("OK: _best_match snaps separator/case variants (HT01 → HT-01)")


def test_best_match_unique_numeric_index():
    # no normalised match, but exactly one tag shares the numeric suffix
    assert _best_match("HEATER-03", {"HT-03", "CL-05"}) == "HT-03"
    print("OK: _best_match snaps on a unique numeric index (HEATER-03 → HT-03)")


def test_best_match_ambiguous_or_absent_returns_none():
    assert _best_match("REACTOR-01", {"HT-01", "CL-01"}) is None   # two share '01'
    assert _best_match("PUMP-09", {"HT-01"}) is None               # nothing matches
    print("OK: _best_match refuses ambiguous/absent matches (None, no wrong guess)")


# ── _reconcile_unit_refs ──────────────────────────────────────────────────────

def test_reconcile_snaps_near_miss():
    kept = _reconcile_unit_refs([_s("S1", "HT01", "V-01")], {"HT-01", "V-01"})
    assert len(kept) == 1
    assert kept[0].src == "HT-01" and kept[0].dst == "V-01"
    print("OK: reconcile snaps a near-miss endpoint to the real tag")


def test_reconcile_nulls_single_phantom_keeps_boundary():
    # 'COLUMN' is a true phantom (not in the unit list) → nulled; V-01 real → kept
    kept = _reconcile_unit_refs([_s("S1", "COLUMN", "V-01")], {"V-01"})
    assert len(kept) == 1
    assert kept[0].src is None and kept[0].dst == "V-01"   # becomes a boundary stream
    print("OK: reconcile nulls a single phantom endpoint, keeps the boundary stream")


def test_reconcile_drops_double_phantom():
    kept = _reconcile_unit_refs(
        [_s("S1", "COLUMN", "ABSORBER"), _s("S2", "HT-01", "V-01")],
        {"HT-01", "V-01"})
    assert [s.tag for s in kept] == ["S2"]   # S1 (both endpoints phantom) dropped
    print("OK: reconcile drops a stream whose both endpoints are phantom")


# ── _dedupe_stream_tags ───────────────────────────────────────────────────────

def test_dedupe_suffixes_duplicates_only():
    out = _dedupe_stream_tags([_s("LIQUID", "V-01", None),
                               _s("LIQUID", "V-02", None),
                               _s("VAPOUR", "V-01", None)])
    assert [s.tag for s in out] == ["LIQUID-1", "LIQUID-2", "VAPOUR"]
    print("OK: dedupe suffixes only colliding tags, leaves uniques intact")


def test_dedupe_noop_when_unique():
    out = _dedupe_stream_tags([_s("FEED", None, "HT-01"), _s("PROD", "V-01", None)])
    assert [s.tag for s in out] == ["FEED", "PROD"]
    print("OK: dedupe is a no-op when all tags are unique")


# ── _lenient_loads ────────────────────────────────────────────────────────────

def test_lenient_loads_wellformed_unchanged():
    assert _lenient_loads('{"streams": [{"tag": "S1"}]}') == {"streams": [{"tag": "S1"}]}
    print("OK: lenient_loads passes well-formed JSON through unchanged")


def test_lenient_loads_strips_trailing_comma():
    assert _lenient_loads('{"streams": [{"tag": "S1"},]}') == {"streams": [{"tag": "S1"}]}
    print("OK: lenient_loads strips a trailing comma")


def test_lenient_loads_recovers_truncation():
    # token cutoff left the second object and the array/object unclosed:
    truncated = '{"streams": [{"tag": "S1"}, {"tag": "S2"'
    assert _lenient_loads(truncated) == {"streams": [{"tag": "S1"}]}
    print("OK: lenient_loads recovers truncated JSON up to the last complete object")


def test_lenient_loads_unrecoverable_raises():
    raised = False
    try:
        _lenient_loads("not json at all")
    except json.JSONDecodeError:
        raised = True
    assert raised, "unrecoverable JSON must raise JSONDecodeError"
    print("OK: lenient_loads raises on unrecoverable input")


def _run_all():
    test_best_match_normalises_separators()
    test_best_match_unique_numeric_index()
    test_best_match_ambiguous_or_absent_returns_none()
    test_reconcile_snaps_near_miss()
    test_reconcile_nulls_single_phantom_keeps_boundary()
    test_reconcile_drops_double_phantom()
    test_dedupe_suffixes_duplicates_only()
    test_dedupe_noop_when_unique()
    test_lenient_loads_wellformed_unchanged()
    test_lenient_loads_strips_trailing_comma()
    test_lenient_loads_recovers_truncation()
    test_lenient_loads_unrecoverable_raises()
    print("\nALL STREAM-EXTRACTOR REPAIR TESTS PASSED")


if __name__ == "__main__":
    _run_all()
