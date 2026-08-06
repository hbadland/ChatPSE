"""Recycle-guard tests: deterministic fuzzy resolution of a recycle target.

The Stage-1 recycle guard resolves a natural-language `recycle_target` (whatever the
LLM wrote — "the first column", "second reactor", "V-01") to an exact unit tag, or
None when it cannot. `_resolve_recycle_target` (agents/orchestrator_v2.py) is the
deterministic core of the recycle handling; these tests exercise every resolution
path and the no-misfire / None cases against the real SemanticUnit type.

Not covered here (they run inline in OrchestratorV2.run and are not extracted into
callable functions): the multi-recycle deduplication guard and the trigger-phrase
guard. Testing those would require either a refactor to extract them or an
integration run; flagged rather than silently skipped.

Run: PYTHONPATH=. python3.9 agents/test_recycle_guards.py
"""
from agents.orchestrator_v2 import _resolve_recycle_target
from agents.stage1.unit_extractor import SemanticUnit


def _u(tag, type, role=""):
    return SemanticUnit(tag=tag, type=type, role=role)


def test_exact_tag_match_wins():
    units = [_u("RX-01", "ConversionReactor"), _u("RX-02", "ConversionReactor")]
    # an exact tag is returned verbatim, ahead of any fuzzy ordinal resolution
    assert _resolve_recycle_target("RX-01", units) == "RX-01"
    print("OK: exact tag match returned verbatim")


def test_ordinal_plus_type_keyword():
    reactors = [_u("RX-01", "ConversionReactor"), _u("RX-02", "ConversionReactor")]
    assert _resolve_recycle_target("second reactor", reactors) == "RX-02"
    heaters = [_u("HT-01", "Heater"), _u("HT-02", "Heater")]
    assert _resolve_recycle_target("first heater", heaters) == "HT-01"
    print("OK: ordinal + type keyword picks the ordinal-th unit of that type")


def test_type_keyword_only_defaults_to_first():
    units = [_u("RX-01", "ConversionReactor")]
    assert _resolve_recycle_target("the reactor", units) == "RX-01"
    print("OK: bare type keyword resolves to the first unit of that type")


def test_ordinal_clamps_to_last():
    reactors = [_u("RX-01", "ConversionReactor"), _u("RX-02", "ConversionReactor")]
    # "fifth" (index 4) with only two reactors clamps to the last match
    assert _resolve_recycle_target("fifth reactor", reactors) == "RX-02"
    print("OK: out-of-range ordinal clamps to the last matching unit")


def test_type_matches_but_no_such_unit_returns_none():
    # "reactor" matches the reactor pattern; with no ConversionReactor present the
    # break stops it falling through to a later pattern, so it will NOT misfire onto
    # the Heater. No partial-tag / role match exists either -> None.
    units = [_u("HT-01", "Heater")]
    assert _resolve_recycle_target("reactor", units) is None
    print("OK: type keyword with no unit of that type resolves to None (no misfire)")


def test_partial_tag_substring():
    units = [_u("V-01", "Vessel")]
    assert _resolve_recycle_target("v-01", units) == "V-01"   # case-folded substring
    print("OK: partial / lowercased tag substring resolves")


def test_role_keyword_match():
    units = [_u("MX-01", "Mixer", role="absorber recycle return")]
    # tag and type do not resolve "absorber overhead"; the role keyword does
    assert _resolve_recycle_target("absorber overhead", units) == "MX-01"
    print("OK: role-keyword fallback resolves when tag/type do not")


def test_unresolvable_and_empty_return_none():
    units = [_u("HT-01", "Heater")]
    assert _resolve_recycle_target("nowhere valid", units) is None
    assert _resolve_recycle_target("", units) is None
    assert _resolve_recycle_target("   ", units) is None
    print("OK: unresolvable and empty/whitespace targets return None")


def _run_all():
    test_exact_tag_match_wins()
    test_ordinal_plus_type_keyword()
    test_type_keyword_only_defaults_to_first()
    test_ordinal_clamps_to_last()
    test_type_matches_but_no_such_unit_returns_none()
    test_partial_tag_substring()
    test_role_keyword_match()
    test_unresolvable_and_empty_return_none()
    print("\nALL RECYCLE-GUARD TESTS PASSED")


if __name__ == "__main__":
    _run_all()
