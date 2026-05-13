"""
Thermodynamics Agent: analyses each unit operation in a flowsheet and assigns
the most appropriate property package, unit-by-unit where needed.

Fully provider-agnostic — pass any model string supported by agents/llm.py.
"""
from __future__ import annotations
import json
import re

from agents import schema
from agents.llm import chat, DEFAULT_MODEL
from agents.physics_check import physics_validate, has_errors, format_issues
from context import DWSIM_KNOWLEDGE

_SYSTEM = """\
You are a thermodynamics expert in chemical process engineering. Your job is to
analyse a flowsheet and assign the most appropriate thermodynamic property
package to each unit operation, overriding the global default where necessary.

For each unit operation decide:
1. What phase equilibrium behaviour dominates (VLE, LLE, VLLE, single-phase)?
2. How non-ideal is the liquid phase?
3. What is the pressure regime?

Only add a "property_package" field to a unit if it should DIFFER from the
flowsheet default. Leave units fine with the default unchanged.

─── COMPOUND CLASSES (memorise — used in HARD RULES below) ──────────────────
ALCOHOLS   : methanol, ethanol, 1-propanol, 2-propanol, 1-butanol, 2-butanol,
             isobutanol, tert-butanol, 1-pentanol, ethylene glycol, glycerol
KETONES    : acetone, methyl ethyl ketone (MEK), cyclohexanone, methyl isobutyl
             ketone (MIBK), acetophenone
ESTERS     : ethyl acetate, methyl acetate, butyl acetate, isopropyl acetate
ETHERS     : diethyl ether, methyl tert-butyl ether (MTBE), tetrahydrofuran (THF),
             1,4-dioxane, diisopropyl ether
CHLORINATED: chloroform, dichloromethane (DCM), carbon tetrachloride,
             1,2-dichloroethane, chlorobenzene
AROMATICS  : benzene, toluene, o/m/p-xylene, ethylbenzene, styrene, naphthalene
ALKANES    : methane, ethane, propane, n-butane, isobutane, n-pentane, n-hexane,
             n-heptane, n-octane, cyclohexane, methylcyclohexane
LIGHT GASES: methane, ethane, propane, butane, nitrogen, oxygen, CO₂, H₂S, H₂, Ar
POLAR_OTHER: acetic acid, formic acid, acetonitrile, dimethyl sulfoxide (DMSO),
             ammonia, hydrogen sulfide

─── KNOWN AZEOTROPES (these systems REQUIRE NRTL or UNIQUAC) ────────────────
ethanol/water         (min-boiling, 95.6 mol% EtOH at 1 atm)
methanol/water        (near-azeotropic; strong non-ideality)
1-propanol/water      (min-boiling azeotrope)
2-propanol/water      (min-boiling azeotrope, 87.7 mol% IPA)
1-butanol/water       (heteroazeotrope, two liquid phases)
ethyl acetate/ethanol (min-boiling azeotrope)
ethyl acetate/water   (heteroazeotrope)
acetone/chloroform    (MAX-boiling azeotrope — negative deviation)
acetone/methanol      (min-boiling azeotrope)
diethyl ether/water   (near-immiscible + azeotrope)
n-hexane/ethanol      (min-boiling heteroazeotrope)
benzene/cyclohexane   (min-boiling azeotrope)
THF/water             (min-boiling azeotrope)
chloroform/acetone    (same as acetone/chloroform, max-boiling)
acetonitrile/water    (min-boiling azeotrope)

─── SELECTION RULES ─────────────────────────────────────────────────────────
- Ideal / near-ideal, chemically-similar at low P  → Raoult's Law
- Polar / H-bonding non-ideal VLE                  → NRTL (preferred) or UNIQUAC
- Azeotropic systems (see list above)              → NRTL or UNIQUAC
- Partially miscible / LLE                         → UNIQUAC (preferred) or NRTL
- Light gases / hydrocarbons, any pressure         → Peng-Robinson
- High-pressure VLE (>10 bar), non-polar           → Peng-Robinson or SRK
- Cryogenic (<200 K) or natural gas                → Lee-Kesler-Plöcker

─── HARD RULES (override all rules above) ───────────────────────────────────
1. Raoult's Law is ONLY valid for chemically-similar compounds:
   ALLOWED  : all-alkane mixtures, all-aromatic mixtures of similar MW,
              alkane/alkane, aromatic/aromatic (e.g. benzene+toluene,
              hexane+heptane, methane+ethane AT LOW PRESSURE)
   FORBIDDEN when the mixture contains:
   • any ALCOHOL    with water, a hydrocarbon, another alcohol class, or a ketone
   • any KETONE     with water or a chlorinated compound
   • any ESTER      with water, an alcohol, or a hydrocarbon
   • any ETHER      with water or an alcohol
   • any CHLORINATED compound with a ketone, ether, or aromatic
   • any compound pair in the KNOWN AZEOTROPES list above
   → For all forbidden pairs: use NRTL.

2. Peng-Robinson / SRK are for GAS-PHASE and HIGH-PRESSURE systems only.
   NEVER assign them to ambient-pressure (<3 bar) polar liquid mixtures
   (alcohols, ketones, esters, ethers mixed with water) — liquid-phase
   activity coefficients are not captured by cubic EOS at low pressure.

3. Lee-Kesler-Plöcker is for CRYOGENIC systems (T < 200 K) and natural gas.
   Do NOT use it for ambient-temperature separations.

─── TRIAL HISTORY INTERPRETATION ────────────────────────────────────────────
If a TRIAL HISTORY block appears in the prompt, apply these rules:
- If an ACTIVITY MODEL (NRTL or UNIQUAC) failed with "outlet≈feed":
    DWSIM has no binary interaction parameters for this compound pair.
    → If the system is NON-POLAR (alkane/alkane, aromatic/aromatic):
      switch to Peng-Robinson or SRK.
    → If the system is POLAR (alcohols, ketones, esters, ethers, or any pair
      in the KNOWN AZEOTROPES list):
      EXCEPTION TO HARD RULE 1 — select Raoult's Law as a topology-
      verification placeholder. This exception is valid ONLY when NRTL or
      UNIQUAC has already failed with outlet≈feed in the trial history.
      Reason: NRTL/UNIQUAC without BIPs produces outlet = feed (identity
      mapping — no separation at all). Raoult's Law, despite being
      theoretically imprecise for non-ideal pairs, at least produces a
      first-order VLE split to confirm the topology is correct.
      Do NOT switch to the other activity model (e.g. UNIQUAC if NRTL failed)
      — both have the same BIP requirement and will fail identically.
- If an EOS (Peng-Robinson, SRK) failed with "solver_diverged" at low pressure:
    EOS is inappropriate for this polar system at these conditions.
    → Switch to NRTL (or Raoult's Law if BIPs are also likely absent).
- If the SAME PACKAGE appears more than once in history with the same failure:
    Do NOT select it again. Choose a fundamentally different model class.
- Never select a package that appears in the trial history unless ALL other
  options have also been tried.

─── OUTPUT FORMAT ───────────────────────────────────────────────────────────
Output two JSON objects separated by a line containing only ---

First: updated flowsheet JSON (same structure, property_package fields added
where needed, global property_package updated to best overall default).

Second: reasoning object:
{
  "global_package": "<name>",
  "global_reasoning": "<one sentence>",
  "unit_overrides": {
    "<unit_tag>": {"package": "<name>", "reasoning": "<one sentence>"}
  }
}

Output ONLY the two JSON blocks separated by ---, no other text.\
"""

_FEW_SHOT = """\
━━━ EXAMPLE 1 — alcohol/water (non-ideal, known azeotrope) ━━━━━━━━━━━━━━━━━━
Input:
{"compounds":["Ethanol","Water"],"property_package":"Raoult's Law",
 "streams":[{"tag":"FEED","T":351.0,"P":101325.0,"flow":1.0,
             "composition":{"Ethanol":0.5,"Water":0.5}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
Output:
{"compounds":["Ethanol","Water"],"property_package":"NRTL",
 "streams":[{"tag":"FEED","T":351.0,"P":101325.0,"flow":1.0,
             "composition":{"Ethanol":0.5,"Water":0.5}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
---
{"global_package":"NRTL",
 "global_reasoning":"Ethanol/water forms a minimum-boiling azeotrope — HARD RULE: alcohol+water forbidden for Raoult's Law.",
 "unit_overrides":{}}

━━━ EXAMPLE 2 — similar aromatics (ideal, Raoult's Law correct) ━━━━━━━━━━━━━
Input:
{"compounds":["Benzene","Toluene"],"property_package":"NRTL",
 "streams":[{"tag":"FEED","T":360.0,"P":101325.0,"flow":1.0,
             "composition":{"Benzene":0.4,"Toluene":0.6}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
Output:
{"compounds":["Benzene","Toluene"],"property_package":"Raoult's Law",
 "streams":[{"tag":"FEED","T":360.0,"P":101325.0,"flow":1.0,
             "composition":{"Benzene":0.4,"Toluene":0.6}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
---
{"global_package":"Raoult's Law",
 "global_reasoning":"Benzene and toluene are chemically similar aromatics with near-ideal VLE — NRTL is unnecessary and overcomplicated.",
 "unit_overrides":{}}

━━━ EXAMPLE 3 — light hydrocarbon gas, moderate pressure ━━━━━━━━━━━━━━━━━━━
Input:
{"compounds":["Methane","Ethane","Propane"],"property_package":"Raoult's Law",
 "streams":[{"tag":"FEED","T":250.0,"P":2000000.0,"flow":1.0,
             "composition":{"Methane":0.7,"Ethane":0.2,"Propane":0.1}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
Output:
{"compounds":["Methane","Ethane","Propane"],"property_package":"Peng-Robinson",
 "streams":[{"tag":"FEED","T":250.0,"P":2000000.0,"flow":1.0,
             "composition":{"Methane":0.7,"Ethane":0.2,"Propane":0.1}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
---
{"global_package":"Peng-Robinson",
 "global_reasoning":"Light hydrocarbon gas mixture at 20 bar — cubic EOS required; Raoult's Law is invalid for gas-phase systems.",
 "unit_overrides":{}}

━━━ EXAMPLE 4 — cryogenic natural gas ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
{"compounds":["Methane","Ethane"],"property_package":"Peng-Robinson",
 "streams":[{"tag":"FEED","T":150.0,"P":5000000.0,"flow":1.0,
             "composition":{"Methane":0.8,"Ethane":0.2}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
Output:
{"compounds":["Methane","Ethane"],"property_package":"Lee-Kesler-Plöcker",
 "streams":[{"tag":"FEED","T":150.0,"P":5000000.0,"flow":1.0,
             "composition":{"Methane":0.8,"Ethane":0.2}},
            {"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
---
{"global_package":"Lee-Kesler-Plöcker",
 "global_reasoning":"Cryogenic methane/ethane (T=150 K) at high pressure — Lee-Kesler-Plöcker is required for accurate VLE at these conditions.",
 "unit_overrides":{}}

━━━ EXAMPLE 5 — ketone/water (azeotrope, non-ideal) ━━━━━━━━━━━━━━━━━━━━━━━━
Input:
{"compounds":["Acetone","Water"],"property_package":"Raoult's Law",
 "streams":[{"tag":"FEED","T":340.0,"P":101325.0,"flow":1.0,
             "composition":{"Acetone":0.5,"Water":0.5}},
            {"tag":"HOT"},{"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"HT-01","type":"Heater","T_out":360.0,"dP":0.0},
          {"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","HT-01",0,0],["HT-01","HOT",0,0],
                ["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
Output:
{"compounds":["Acetone","Water"],"property_package":"NRTL",
 "streams":[{"tag":"FEED","T":340.0,"P":101325.0,"flow":1.0,
             "composition":{"Acetone":0.5,"Water":0.5}},
            {"tag":"HOT"},{"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"HT-01","type":"Heater","T_out":360.0,"dP":0.0},
          {"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","HT-01",0,0],["HT-01","HOT",0,0],
                ["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}
---
{"global_package":"NRTL",
 "global_reasoning":"Acetone/water is a polar ketone+water system with strong non-ideality — HARD RULE: ketone+water forbidden for Raoult's Law.",
 "unit_overrides":{}}\
"""


class ThermoAgent:
    """
    Assigns property packages based on thermodynamic reasoning.
    Returns (updated_flowsheet, reasoning_dict).

    Args:
        model: any model string supported by agents/llm.py
               e.g. "gemini-2.5-flash", "claude-opus-4-7", "gpt-4o"
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model
        self._system = _SYSTEM + "\n\n---\n" + DWSIM_KNOWLEDGE

    # ── Pre-planning package selection ────────────────────────────────────────

    def pre_select(
            self,
            compounds: list[str],
            description: str,
            max_retries: int = 2,
    ) -> tuple[str, str]:
        """
        Select the property package BEFORE topology is drawn.

        Called by the Orchestrator between BasisAgent and PlannerAgent so the
        Planner receives the package as a hard constraint and never has to
        reason about thermodynamics independently.

        Uses the same hard rules and compound classifications as assign() but
        asks for a single JSON decision rather than a full flowsheet update.

        Returns:
            (package_name, one_sentence_reasoning)
            Falls back to "Raoult's Law" on any error — the pipeline continues
            and ThermoAgent.assign() will correct it after planning.
        """
        from agents.schema import SUPPORTED_PROPERTY_PACKAGES
        pkg_list = ", ".join(sorted(SUPPORTED_PROPERTY_PACKAGES))

        prompt = (
            f"{_FEW_SHOT}\n\n"
            "─────────────────────────────────────────────────────────────────\n"
            "PRE-PLANNING PACKAGE SELECTION — no flowsheet yet.\n"
            "Apply COMPOUND CLASSES + HARD RULES above to pick the SINGLE best\n"
            "global property package for these compounds and process description.\n\n"
            f"Compounds      : {compounds}\n"
            f"Description    : {description}\n\n"
            f"Supported packages: {pkg_list}\n\n"
            "Output ONLY this JSON object — no flowsheet, no other text:\n"
            '{"package": "<one of the supported packages>", '
            '"reasoning": "<one sentence citing the specific rule that applies>"}'
        )

        for _attempt in range(max_retries):
            try:
                raw = chat(prompt, system=self._system,
                           model=self._model, temperature=0)
                raw = raw.strip()
                # Strip markdown fences
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    raw = "\n".join(
                        ln for ln in lines if not ln.strip().startswith("```"))
                    raw = raw.strip()
                # Extract the first JSON object
                match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                if match:
                    raw = match.group(0)
                parsed    = json.loads(raw)
                package   = parsed.get("package", "").strip()
                reasoning = parsed.get("reasoning", "").strip()
                if package in SUPPORTED_PROPERTY_PACKAGES:
                    return package, reasoning
            except Exception:
                pass  # retry

        return "Raoult's Law", "pre_select fallback — could not determine package"

    def assign(self, flowsheet: dict, max_retries: int = 3,
               exclude_packages: set[str] | None = None,
               trial_history: list | None = None) -> tuple[dict, dict]:
        """Analyse flowsheet and return (updated_flowsheet, reasoning)."""
        exclusion_note = ""
        if exclude_packages:
            names = ", ".join(sorted(exclude_packages))
            exclusion_note = (
                f"IMPORTANT: The following property packages have already been "
                f"tried and failed for this system — do NOT select them: {names}\n"
                f"Choose a different package from the supported list.\n\n"
            )
        history_note = _format_trial_history(trial_history or [])
        base_prompt = (
            f"{history_note}{exclusion_note}{_FEW_SHOT}\n\n"
            f"Now analyse this flowsheet:\n{schema.to_json(flowsheet)}"
        )
        prompt = base_prompt
        last_errors: list[str] = []

        for attempt in range(max_retries):
            raw = chat(prompt, system=self._system, model=self._model, temperature=0)

            try:
                updated_fs, reasoning = _parse_response(raw)
            except Exception as e:
                last_errors = [str(e)]
                prompt = _retry_prompt(base_prompt, raw, last_errors)
                continue

            errors = schema.validate(updated_fs)
            if errors:
                last_errors = errors
                prompt = _retry_prompt(base_prompt, raw, errors)
                continue

            # Physics compatibility check — retry on errors, warn but pass on warnings
            physics_issues = physics_validate(updated_fs)
            physics_errors = [str(i) for i in physics_issues
                              if i.severity.value == "ERROR"]
            if physics_errors:
                last_errors = physics_errors
                prompt = _retry_prompt(base_prompt, raw, physics_errors)
                continue

            return updated_fs, reasoning

        raise ValueError(
            f"ThermoAgent failed after {max_retries} attempts "
            f"using {self._model}. Last errors: {last_errors}")


def _format_trial_history(history: list) -> str:
    if not history:
        return ""
    lines = [
        "TRIAL HISTORY — these have already been attempted and failed. "
        "Do NOT select any package listed below. "
        "Use the failure reasons to infer the correct choice:\n"
    ]
    for rec in history:
        line = (
            f"  - Iteration {rec.iteration}: {rec.property_package} → "
            f"{rec.execution_summary}  [{', '.join(rec.failure_codes)}]"
        )
        if rec.diagnosis:
            line += f"\n    Diagnosis: {rec.diagnosis[:120]}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_response(raw: str) -> tuple[dict, dict]:
    """
    Split on the separator between the two JSON blocks.

    The system prompt and DWSIM_KNOWLEDGE both contain '---' lines, so the LLM
    response may also contain them as decoration. We anchor on a line that
    contains ONLY '---' (possibly with surrounding whitespace) and use the
    LAST such line as the separator — the reasoning JSON always comes second.
    """
    # Match a line that is solely '---' (the JSON separator the LLM was told to use)
    sep_pattern = re.compile(r'\n[ \t]*---[ \t]*\n')
    matches = list(sep_pattern.finditer(raw))
    if not matches:
        raise ValueError(
            "Response missing '---' separator between flowsheet and reasoning JSON.")

    # Use the last match — reasoning block always follows
    last = matches[-1]
    flowsheet_raw = raw[:last.start()]
    reasoning_raw = raw[last.end():]

    def clean(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))
        return text.strip()

    def extract_json(text: str) -> dict:
        text = clean(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # LLM appended trailing text after the JSON block — extract the object
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    flowsheet = extract_json(flowsheet_raw)
    reasoning = extract_json(reasoning_raw)
    return flowsheet, reasoning


def _retry_prompt(base: str, bad: str, errors: list[str]) -> str:
    error_msg = "\n".join(f"- {e}" for e in errors)
    return (
        f"{base}\n\n"
        f"Your previous attempt had errors:\n{error_msg}\n"
        f"Previous output:\n{bad}\n\n"
        "Fix the errors and output only the two JSON blocks separated by ---."
    )
