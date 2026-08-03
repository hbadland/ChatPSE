"""
Provenance invariant tests (Option B, Phase 1).

Establishes the property 3.2 claims: for a provenance-tracked parameter, a value can
only change via (a) a priority-guarded merge that rejects equal-or-lower authority, or
(b) a sanctioned physical correction that re-tags honestly. No silent overwrite.

Also proves set_param(Source.RULE) reproduces the legacy FailureRuleStore whitelist
exactly — the guarantee that Phase 2 (migrating rule_store) is behaviour-preserving.

Run: PYTHONPATH=. python3.9 -m ir.test_provenance
     (run as a module — ir/ on sys.path[0] shadows stdlib `types` via ir/types.py)
"""
from ir.graph import HeaterNode, Source


def test_downgrade_rejected_upgrade_applied():
    n = HeaterNode(tag="H-01")
    assert n.set_param("T_out", 350.0, Source.SPECIFIED) is True
    # every strictly-lower authority is rejected, value untouched:
    for s in (Source.EXTRACTED, Source.INHERITED, Source.RULE,
              Source.COMPUTED, Source.FALLBACK, Source.DEFAULT):
        assert n.set_param("T_out", 999.0, s) is False, s
        assert n.params["T_out"] == 350.0
    # equal authority is also rejected (no silent ties):
    assert n.set_param("T_out", 999.0, Source.SPECIFIED) is False
    assert n.params["T_out"] == 350.0
    # a strictly-higher write would apply — build the ladder upward:
    m = HeaterNode(tag="H-02")
    assert m.set_param("T_out", 1.0, Source.COMPUTED) is True
    assert m.set_param("T_out", 2.0, Source.RULE) is True      # RULE > COMPUTED
    assert m.set_param("T_out", 3.0, Source.EXTRACTED) is True  # EXTRACTED > RULE
    assert m.set_param("T_out", 4.0, Source.COMPUTED) is False  # downgrade rejected
    assert m.params["T_out"] == 3.0
    print("OK: downgrades/ties rejected, upgrades applied")


def test_correction_overrides_and_retags():
    n = HeaterNode(tag="H-03")
    n.set_param("T_out", 350.0, Source.SPECIFIED)
    n.params["_desc_T_out"] = True
    # a physical correction overrides even SPECIFIED, but re-tags + clears the sentinel
    n.correct_param("T_out", 360.0, Source.COMPUTED)
    assert n.params["T_out"] == 360.0
    assert n.params["_temperature_source"] == "computed"
    assert "_desc_T_out" not in n.params
    # now a rule can act on it (it is genuinely an estimate again)
    assert n.set_param("T_out", 370.0, Source.RULE) is True
    print("OK: correction overrides, downgrades tag honestly, clears sentinel")


def test_reproduces_legacy_rule_guard():
    """Legacy guard (rule_store.py): a RULE overwrites {computed, default_fallback,
    fallback} or untagged-without-sentinel; is suppressed on {specified, extracted,
    inherited, rule} or untagged-with-sentinel. set_param(RULE) must match exactly."""
    overwritable = ["computed", "default_fallback", "fallback"]
    protected    = ["specified", "extracted", "inherited", "rule"]
    for tag in overwritable:
        n = HeaterNode(tag="X"); n.params["T_out"] = 300.0
        n.params["_temperature_source"] = tag
        assert n.set_param("T_out", 400.0, Source.RULE) is True, tag
    for tag in protected:
        n = HeaterNode(tag="X"); n.params["T_out"] = 300.0
        n.params["_temperature_source"] = tag
        assert n.set_param("T_out", 400.0, Source.RULE) is False, tag
    # untagged + no sentinel -> overwritable
    n = HeaterNode(tag="X"); n.params["T_out"] = 300.0
    assert n.set_param("T_out", 400.0, Source.RULE) is True
    # untagged + description sentinel -> protected
    n = HeaterNode(tag="X"); n.params["T_out"] = 300.0; n.params["_desc_T_out"] = True
    assert n.set_param("T_out", 400.0, Source.RULE) is False
    print("OK: set_param(RULE) reproduces the legacy whitelist exactly")


def test_untracked_param_unconditional():
    n = HeaterNode(tag="H-04")
    # a param with no provenance field is written unconditionally (nothing to protect)
    assert n.set_param("duty", 100.0, Source.DEFAULT) is True
    assert n.set_param("duty", 200.0, Source.DEFAULT) is True
    assert n.params["duty"] == 200.0
    print("OK: untracked params write unconditionally")


if __name__ == "__main__":
    test_downgrade_rejected_upgrade_applied()
    test_correction_overrides_and_retags()
    test_reproduces_legacy_rule_guard()
    test_untracked_param_unconditional()
    print("\nALL PROVENANCE TESTS PASSED")
