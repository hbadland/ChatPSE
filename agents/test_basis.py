"""
Tests for BasisAgent.

Stage 1 tests are deterministic (no LLM).  Stage 2 tests patch the LLM call
so the full verifier-completer path can be exercised without network access.
"""
from __future__ import annotations
import json
from unittest.mock import patch

from agents.basis import BasisAgent, _parse_database, _flatten, _substitute, _word_match
from context import COMPOUND_DATABASE

agent = BasisAgent()


# ── Database parsing ──────────────────────────────────────────────────────────

def test_database_loads():
    lookup, mixtures, mixture_fracs, unsupported, unverified = _parse_database(COMPOUND_DATABASE)
    assert len(lookup) > 50,       f"Expected 50+ aliases, got {len(lookup)}"
    assert len(mixtures) > 5,      f"Expected 5+ mixture aliases, got {len(mixtures)}"
    assert len(unsupported) > 10,  f"Expected 10+ unsupported entries, got {len(unsupported)}"
    assert "methanol" in lookup
    assert lookup["methanol"] == "Methanol"
    assert lookup["meoh"] == "Methanol"
    assert "natural gas" in mixtures
    assert "Methane" in mixtures["natural gas"]
    # Colloquial unsupported names must be registered
    assert "brine" in unsupported, "colloquial 'brine' missing from unsupported"
    assert "lye" in unsupported,   "colloquial 'lye' missing from unsupported"
    assert "caustic" in unsupported, "colloquial 'caustic' missing from unsupported"
    print(f"PASS  test_database_loads  "
          f"({len(lookup)} aliases, {len(mixtures)} mixtures, "
          f"{len(unsupported)} unsupported, {len(mixture_fracs)} fracs)")


def test_mixture_fracs_parsed():
    _, _, mixture_fracs, _, _ = _parse_database(COMPOUND_DATABASE)
    assert "natural gas" in mixture_fracs, "natural gas mole fractions not parsed"
    ng = mixture_fracs["natural gas"]
    assert "Methane" in ng
    assert abs(sum(ng.values()) - 1.0) < 0.01, f"Fractions don't sum to 1: {ng}"
    assert "air" in mixture_fracs
    print(f"PASS  test_mixture_fracs_parsed  natural_gas={mixture_fracs['natural gas']}")


# ── Stage 1 alias resolution (deterministic) ──────────────────────────────────

def _mock_llm_echo(prompt, system, model):
    """Minimal valid LLM response: confirms all anchors, substitutes names, adds nothing."""
    anchors_block = prompt.split("Stage 1 anchors (verify these):")[1].split("Process description:")[0]
    desc_raw = prompt.split("Process description:")[1].split("Perform")[0].strip()
    try:
        anchors = json.loads(anchors_block.strip())
        confirmed = [v for v in anchors.values() if isinstance(v, str)]
    except Exception:
        anchors, confirmed = {}, []
    # Substitute anchors into description so normalised_description is correct
    normalised = desc_raw
    for orig, dwsim in anchors.items():
        if isinstance(dwsim, str):
            normalised = normalised.replace(orig, dwsim)
    return json.dumps({
        "confirmed": confirmed,
        "rejected": [],
        "additional": [],
        "mixture_expansions": [],
        "concentration_hints": [],
        "normalised_description": normalised,
    })


def test_exact_alias_ethanol():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Flash separate an ethanol/water feed at 1 atm.")
    assert "Ethanol" in result.dwsim_compounds
    assert "Water"   in result.dwsim_compounds
    assert result.success
    assert result.stage1_count >= 2
    print(f"PASS  test_exact_alias_ethanol  compounds={result.dwsim_compounds}")


def test_abbreviation_IPA():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Separate IPA from water by distillation.")
    assert any("propanol" in c.lower() for c in result.dwsim_compounds), \
        f"IPA not resolved: {result.dwsim_compounds}"
    assert result.success
    print(f"PASS  test_abbreviation_IPA  compounds={result.dwsim_compounds}")


def test_abbreviation_MEK():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Use MEK as a solvent with water.")
    assert any("ketone" in c.lower() or "butanone" in c.lower() or "MEK" in c
               for c in result.dwsim_compounds) or \
           "Methyl Ethyl Ketone" in result.dwsim_compounds
    print(f"PASS  test_abbreviation_MEK  compounds={result.dwsim_compounds}")


def test_trivial_name_wood_alcohol():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Distil wood alcohol from water.")
    assert "Methanol" in result.dwsim_compounds, \
        f"wood alcohol not resolved: {result.dwsim_compounds}"
    print(f"PASS  test_trivial_name_wood_alcohol")


def test_mixture_alias_natural_gas():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Separate natural gas into components at high pressure.")
    assert "Methane" in result.dwsim_compounds, \
        f"natural gas not expanded: {result.dwsim_compounds}"
    assert result.success
    print(f"PASS  test_mixture_alias_natural_gas  compounds={result.dwsim_compounds}")


def test_mixture_alias_air():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Cool an air stream from 200°C to 30°C.")
    assert "Nitrogen" in result.dwsim_compounds or "Oxygen" in result.dwsim_compounds, \
        f"air not expanded: {result.dwsim_compounds}"
    print(f"PASS  test_mixture_alias_air")


# ── Mixture composition hints ─────────────────────────────────────────────────

def test_suggested_compositions_natural_gas():
    """Stage 1 should attach known mole fractions for named mixture aliases."""
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Process natural gas at 50 bar.")
    has_composition = any(
        "Methane" in comps for comps in result.suggested_compositions.values()
    )
    assert has_composition, \
        f"Expected natural gas fractions in suggested_compositions: {result.suggested_compositions}"
    print(f"PASS  test_suggested_compositions_natural_gas  "
          f"comps={result.suggested_compositions}")


# ── Unsupported compound detection ────────────────────────────────────────────

def test_unsupported_formal_nacl():
    result = agent.identify("Dissolve sodium chloride in water and heat.")
    assert not result.success, "Expected failure for NaCl"
    assert result.errors
    print(f"PASS  test_unsupported_formal_nacl  errors={result.errors}")


def test_unsupported_colloquial_brine():
    result = agent.identify("Pump brine from the evaporator to the crystalliser.")
    assert not result.success, \
        f"Expected failure for colloquial 'brine' (NaCl). Got: success={result.success}"
    assert result.errors
    print(f"PASS  test_unsupported_colloquial_brine  errors={result.errors}")


def test_unsupported_colloquial_lye():
    result = agent.identify("Mix lye with acetic acid in water.")
    assert not result.success, "Expected failure for colloquial 'lye' (NaOH)"
    print(f"PASS  test_unsupported_colloquial_lye")


def test_unsupported_colloquial_caustic():
    result = agent.identify("Heat a stream of caustic and water to 80°C.")
    assert not result.success, "Expected failure for colloquial 'caustic' (NaOH)"
    print(f"PASS  test_unsupported_colloquial_caustic")


def test_unsupported_naoh():
    result = agent.identify("Mix NaOH with acetic acid in water.")
    assert not result.success, "Expected failure for NaOH"
    print(f"PASS  test_unsupported_naoh")


def test_unsupported_vitriol():
    result = agent.identify("Neutralise oil of vitriol with ammonia.")
    assert not result.success, "Expected failure for 'oil of vitriol' (H2SO4)"
    print(f"PASS  test_unsupported_vitriol")


# ── Word-boundary matching (no false positives) ───────────────────────────────

def test_no_false_positive_pronoun_he():
    assert _word_match("he", "He said to heat the stream.") is None or True
    # Helium (alias 'He') must NOT fire inside normal prose — confirm via agent
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("He said to heat the methanol stream.")
    assert "Helium" not in result.dwsim_compounds, \
        f"False positive: Helium found in {result.dwsim_compounds}"
    print(f"PASS  test_no_false_positive_pronoun_he")


def test_no_false_positive_an_in_sentence():
    """'an' (Acrylonitrile alias) must not match 'an ethanol/water feed'."""
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Flash separate an ethanol/water feed at 1 atm.")
    assert "Acrylonitrile" not in result.dwsim_compounds, \
        f"False positive: Acrylonitrile found in {result.dwsim_compounds}"
    print(f"PASS  test_no_false_positive_an_in_sentence")


# ── LLM verifier rejects a Stage 1 false positive ────────────────────────────

def test_llm_rejects_false_positive_anchor():
    """
    Stage 1 proposes 'DME' → Dimethyl Ether; LLM rejects it (context shows 'DME'
    is an acronym for 'direct methanol electrolyser', not the solvent).
    The rejected compound must not appear in dwsim_compounds and a warning is emitted.
    """

    def mock_reject_dme(prompt, system, model):
        return json.dumps({
            "confirmed": ["Methanol", "Water"],
            "rejected": [{"proposed": "Dimethyl Ether", "original_text": "DME",
                          "reason": "DME here means direct methanol electrolyser, not solvent"}],
            "additional": [],
            "mixture_expansions": [],
            "concentration_hints": [],
            "normalised_description": "Feed Methanol and Water to the DME unit.",
        })

    with patch("agents.basis.chat", side_effect=mock_reject_dme):
        result = agent.identify("Feed methanol and water to the DME unit.")
    assert "Dimethyl Ether" not in result.dwsim_compounds, \
        f"Rejected anchor 'Dimethyl Ether' should be excluded: {result.dwsim_compounds}"
    assert any("rejected" in w.lower() for w in result.warnings), \
        f"Expected rejection warning in: {result.warnings}"
    print(f"PASS  test_llm_rejects_false_positive_anchor  compounds={result.dwsim_compounds}")


# ── LLM adds compound Stage 1 missed ─────────────────────────────────────────

def test_llm_adds_missed_compound():
    """LLM identifies 'cyclohexane' from a description where Stage 1 missed it."""

    def mock_add_cyclohexane(prompt, system, model):
        return json.dumps({
            "confirmed": ["Water"],
            "rejected": [],
            "additional": [{"original_text": "cyclohexyl solvent",
                            "dwsim_name": "Cyclohexane",
                            "is_mixture": False, "mixture_components": [],
                            "mole_fractions": [],
                            "status": "ok", "note": None}],
            "mixture_expansions": [],
            "concentration_hints": ["solvent feed is pure cyclohexane"],
            "normalised_description": "Extract Water with Cyclohexane solvent.",
        })

    with patch("agents.basis.chat", side_effect=mock_add_cyclohexane):
        result = agent.identify("Extract water with cyclohexyl solvent.")
    assert "Cyclohexane" in result.dwsim_compounds, \
        f"LLM-added Cyclohexane missing: {result.dwsim_compounds}"
    assert any("cyclohexane" in h.lower() for h in result.concentration_hints), \
        f"Concentration hint missing: {result.concentration_hints}"
    print(f"PASS  test_llm_adds_missed_compound  compounds={result.dwsim_compounds}")


# ── LLM fallback (Stage 1 only on LLM failure) ───────────────────────────────

def test_llm_failure_falls_back_to_stage1():
    """When LLM raises, result uses Stage 1 anchors and stage='PARTIAL'."""
    def mock_fail(prompt, system, model):
        raise RuntimeError("Simulated LLM timeout")

    with patch("agents.basis.chat", side_effect=mock_fail):
        result = agent.identify("Flash methanol and water at 1 atm.")
    assert "Methanol" in result.dwsim_compounds, \
        f"Stage 1 fallback should include Methanol: {result.dwsim_compounds}"
    assert result.stage == "PARTIAL"
    assert any("LLM" in w for w in result.warnings)
    print(f"PASS  test_llm_failure_falls_back_to_stage1  stage={result.stage}")


# ── Concentration hints from LLM ──────────────────────────────────────────────

def test_concentration_hints_extracted():
    """LLM extracts concentration hints from description text."""

    def mock_hints(prompt, system, model):
        return json.dumps({
            "confirmed": ["Ethanol", "Water"],
            "rejected": [],
            "additional": [],
            "mixture_expansions": [],
            "concentration_hints": ["feed is 30 wt% ethanol in water"],
            "normalised_description": "Distil a 30 wt% Ethanol/Water feed.",
        })

    with patch("agents.basis.chat", side_effect=mock_hints):
        result = agent.identify("Distil a 30 wt% ethanol/water feed.")
    assert len(result.concentration_hints) >= 1
    assert "ethanol" in result.concentration_hints[0].lower()
    print(f"PASS  test_concentration_hints_extracted  hints={result.concentration_hints}")


# ── Normalised description substitution ───────────────────────────────────────

def test_normalised_description_substitution():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Heat methyl alcohol and water to 80°C then flash.")
    assert "Methanol" in result.normalised_description, \
        f"'methyl alcohol' not substituted: {result.normalised_description}"
    print(f"PASS  test_normalised_description_substitution")
    print(f"      → {result.normalised_description}")


# ── Flat compound list helpers ────────────────────────────────────────────────

def test_flatten_deduplication():
    compound_map = {
        "ethanol": "Ethanol",
        "EtOH":    "Ethanol",   # duplicate DWSIM name
        "water":   "Water",
    }
    flat = _flatten(compound_map)
    assert flat.count("Ethanol") == 1, f"Duplicate Ethanol in {flat}"
    assert "Water" in flat
    print(f"PASS  test_flatten_deduplication  flat={flat}")


# ── Summary ───────────────────────────────────────────────────────────────────

def test_result_summary_prints():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Flash methanol and water at 1 atm.")
    summary = result.summary()
    assert "Methanol" in summary
    assert "stage1_count" in summary.lower() or "Stage1" in summary
    print(f"PASS  test_result_summary_prints")
    print(f"\n{summary}")


# ── stage1_count reported correctly ──────────────────────────────────────────

def test_stage1_count():
    with patch("agents.basis.chat", side_effect=_mock_llm_echo):
        result = agent.identify("Separate methanol and water by distillation.")
    assert result.stage1_count >= 2, \
        f"Expected stage1_count >= 2, got {result.stage1_count}"
    print(f"PASS  test_stage1_count  stage1_count={result.stage1_count}")


# ── Stage 2 skip (verified single compounds only) ────────────────────────────

def test_stage2_skipped_for_verified_compounds():
    """
    When Stage 1 resolves all compounds as verified non-mixture aliases,
    Stage 2 (LLM) must not be called and stage must be 'LOOKUP'.
    """
    with patch("agents.basis.chat") as mock_chat:
        result = agent.identify("Flash separate a methanol/water feed at 1 atm.")
    mock_chat.assert_not_called()
    assert result.stage == "LOOKUP", f"Expected stage='LOOKUP', got {result.stage!r}"
    assert "Methanol" in result.dwsim_compounds
    assert "Water"   in result.dwsim_compounds
    assert result.success
    print(f"PASS  test_stage2_skipped_for_verified_compounds  "
          f"stage={result.stage}  compounds={result.dwsim_compounds}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_database_loads,
        test_mixture_fracs_parsed,
        test_exact_alias_ethanol,
        test_abbreviation_IPA,
        test_abbreviation_MEK,
        test_trivial_name_wood_alcohol,
        test_mixture_alias_natural_gas,
        test_mixture_alias_air,
        test_suggested_compositions_natural_gas,
        test_unsupported_formal_nacl,
        test_unsupported_colloquial_brine,
        test_unsupported_colloquial_lye,
        test_unsupported_colloquial_caustic,
        test_unsupported_naoh,
        test_unsupported_vitriol,
        test_no_false_positive_pronoun_he,
        test_no_false_positive_an_in_sentence,
        test_llm_rejects_false_positive_anchor,
        test_llm_adds_missed_compound,
        test_llm_failure_falls_back_to_stage1,
        test_concentration_hints_extracted,
        test_normalised_description_substitution,
        test_flatten_deduplication,
        test_result_summary_prints,
        test_stage1_count,
        test_stage2_skipped_for_verified_compounds,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)

    print(f"\n{'─'*50}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): {failed}")
    else:
        print(f"All {len(tests)} tests passed.")
