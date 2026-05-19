"""
PlannerAgent — converts a natural language description into a validated
flowsheet JSON (agents/schema.py format).

The generation is decomposed into four focused stages:

  1. TopologyAgent   — unit sequence (zero LLM if topology_library matches)
  2. ConnectionAgent — stream / connection graph
  3. ConditionAgent  — numerical T/P/flow/composition/unit parameters
  4. Assembler       — deterministic JSON assembly (no LLM)

After assembly, schema validation and physics validation are applied.
Retry feedback is targeted: ConditionAgent is re-run with the specific
error, not the whole chain.

Image input (Gemini models) uses a legacy single-shot path that bypasses
the sub-agents — multi-modal decomposition can be added later.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agents import schema
from agents.llm              import chat, DEFAULT_MODEL
from agents.planner_types    import TopologyPlan, ConnectionPlan, ConditionPlan
from agents.topology_agent   import TopologyAgent
from agents.connection_agent import ConnectionAgent
from agents.condition_agent  import ConditionAgent
from agents.assembler        import Assembler
from agents.topology_library import TopologyHint
from context                 import DWSIM_KNOWLEDGE


class PlannerAgent:
    """
    Converts text (and optionally image) input to a validated flowsheet dict.

    Args:
        model: any model string supported by agents/llm.py for text input.
               Image input requires a Gemini model (multimodal capability).
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model      = model
        self._topology   = TopologyAgent(model=model)
        self._connection = ConnectionAgent(model=model)
        self._condition  = ConditionAgent(model=model)
        self._assembler  = Assembler()

        # Legacy system prompt kept for the image path and revise()
        self._system = _SYSTEM_LEGACY + "\n\n---\n" + DWSIM_KNOWLEDGE

    # ── Primary entry point ────────────────────────────────────────────────────

    def plan(
            self,
            description:            str = "",
            image_path:             str | None = None,
            max_retries:            int = 3,
            compounds:              list[str] | None = None,
            suggested_compositions: dict | None = None,
            topology_feedback:      str | None = None,
            condition_feedback:     str | None = None,
            package_feedback:       str | None = None,
            compound_feedback:      str | None = None,
            property_package:       str | None = None,
            condition_estimate:     str | None = None,
            topology_hint:          TopologyHint | None = None,
    ) -> dict:
        """
        Generate a flowsheet JSON from text and/or image.

        Args:
            description            : natural language process description
            image_path             : optional P&ID sketch (Gemini only)
            max_retries            : retry limit passed to sub-agents
            compounds              : exact DWSIM compound names from BasisAgent
            suggested_compositions : {alias → {dwsim_name: mole_fraction}}
            topology_feedback      : Critic's topology diagnosis (TOPOLOGY_REBUILD)
            condition_feedback     : quantitative T/P guidance from prior failures
            package_feedback       : packages tried and why each failed
            compound_feedback      : wrong compound names from a BASIS re-run
            property_package       : package pre-selected by ThermoAgent.pre_select()
            condition_estimate     : bubble-point T/P hint from orchestrator
            topology_hint          : pattern-matched topology from topology_library
        """
        if not description and not image_path:
            raise ValueError("Provide at least a text description or an image.")

        if image_path:
            return self._plan_with_image(
                description, image_path, max_retries,
                _build_legacy_constraint(
                    compounds, suggested_compositions,
                    topology_feedback, condition_feedback,
                    package_feedback, compound_feedback,
                    property_package, condition_estimate,
                    topology_hint,
                ))

        return self._plan_decomposed(
            description=description,
            compounds=compounds or [],
            property_package=property_package or "Raoult's Law",
            topology_hint=topology_hint,
            condition_estimate=condition_estimate,
            suggested_compositions=suggested_compositions,
            condition_feedback=condition_feedback,
            topology_feedback=topology_feedback,
            package_feedback=package_feedback,
            max_retries=max_retries,
        )

    # ── Decomposed text path ───────────────────────────────────────────────────

    def _plan_decomposed(
            self,
            description:            str,
            compounds:              list[str],
            property_package:       str,
            topology_hint:          TopologyHint | None,
            condition_estimate:     str | None,
            suggested_compositions: dict | None,
            condition_feedback:     str | None,
            topology_feedback:      str | None,
            package_feedback:       str | None,
            max_retries:            int,
    ) -> dict:
        """
        Chain: TopologyAgent → ConnectionAgent → ConditionAgent → Assembler
               with targeted retries on validation failure.
        """
        # Stage 1 — topology (free if library matches)
        topology: TopologyPlan = self._topology.plan(
            description=description,
            compounds=compounds,
            property_package=property_package,
            hint=topology_hint,
            topology_feedback=topology_feedback,
        )

        # Stage 2 — connection graph
        connections: ConnectionPlan = self._connection.plan(
            description=description,
            topology=topology,
            compounds=compounds,
        )

        # Stage 3+4 — conditions + assembly with targeted retries
        # Only ConditionAgent is re-run on physics/schema failures;
        # topology and connection plans are stable across retries.
        extra_feedback = condition_feedback or ""
        if package_feedback:
            extra_feedback = (
                f"PACKAGE FEEDBACK (why prior thermodynamic packages failed):\n"
                f"{package_feedback}\n"
                f"Use this to choose operating conditions appropriate for "
                f"the current package ({property_package}).\n\n"
                + extra_feedback
            ).strip()

        last_errors: list[str] = []

        for attempt in range(max_retries):
            try:
                conditions: ConditionPlan = self._condition.plan(
                    description=description,
                    compounds=compounds,
                    property_package=property_package,
                    topology=topology,
                    connections=connections,
                    condition_estimate=condition_estimate,
                    suggested_compositions=suggested_compositions,
                    condition_feedback=extra_feedback or None,
                )
            except (ValueError, TypeError) as exc:
                last_errors = [str(exc)]
                extra_feedback = (
                    f"CONDITION ERROR — conditions could not be generated "
                    f"(attempt {attempt + 1}):\n"
                    f"  {exc}\n"
                    "Review the units and feed streams and return valid conditions."
                )
                continue

            flowsheet = self._assembler.assemble(
                compounds=compounds,
                property_package=property_package,
                topology=topology,
                connections=connections,
                conditions=conditions,
            )

            # ── Schema validation ──────────────────────────────────────────────
            errors = schema.validate(flowsheet)
            if errors:
                primary = errors[0]
                last_errors = errors
                extra_feedback = (
                    f"SCHEMA ERROR — fix this specific issue:\n"
                    f"  {primary}\n"
                    f"  ({len(errors) - 1} further error(s) will be shown after this is fixed.)"
                )
                continue

            # ── Physics validation ─────────────────────────────────────────────
            physics_issues = schema.physics_validate(flowsheet)
            physics_errors = [i for i in physics_issues if i.severity.value == "ERROR"]
            if physics_errors:
                primary = physics_errors[0]
                last_errors = [str(i) for i in physics_errors]
                extra_feedback = (
                    f"PHYSICS ERROR — fix this specific issue:\n"
                    f"  Location : {primary.location}\n"
                    f"  Problem  : {primary.message}\n"
                    f"  Fix      : {primary.fix}"
                )
                continue

            return flowsheet

        raise ValueError(
            f"PlannerAgent failed after {max_retries} attempts. "
            f"Topology: {[u.type for u in topology.units]} | "
            f"Last errors: {last_errors}"
        )

    # ── Surgical revision (UNIT_PATCH — keeps existing approach) ──────────────

    def revise(
            self,
            flowsheet:              dict,
            description:            str,
            compounds:              list[str],
            broken_unit_name:       str,
            reason:                 str,
            suggested_compositions: dict | None = None,
    ) -> dict:
        """
        Surgical revision: fix exactly one unit in an existing flowsheet.

        Returns a patch for the broken unit's parameters only — the model
        never sees or reproduces the full flowsheet, eliminating silent
        field corruption from small models.
        """
        from agents.chem_data import estimate_bubble_point

        # Find the broken unit to show its current parameters
        broken_unit = next(
            (u for u in flowsheet.get("units", []) if u["tag"] == broken_unit_name),
            None,
        )
        current_params = (
            {k: v for k, v in broken_unit.items() if k not in ("tag", "type")}
            if broken_unit else {}
        )
        unit_type = broken_unit.get("type", "unknown") if broken_unit else "unknown"

        # Build compact feed context for bubble-point reasoning
        feed_summary = []
        for s in flowsheet.get("streams", []):
            if "T" in s:
                comp = s.get("composition", {})
                comp_str = ", ".join(f"{k}: {v}" for k, v in comp.items())
                feed_summary.append(
                    f"  {s['tag']}: T={s['T']}K  P={s['P']}Pa  {{{comp_str}}}")

        # Compute bubble point and inject concrete target temperatures
        bubble_hint = ""
        feed_stream = next(
            (s for s in flowsheet.get("streams", [])
             if "T" in s and s.get("composition")),
            None,
        )
        if feed_stream:
            t_bub = estimate_bubble_point(
                compounds,
                feed_stream.get("composition", {}),
                feed_stream.get("P", 101_325.0),
            )
            if t_bub is not None:
                bubble_hint = (
                    f"\nEstimated mixture bubble point: {t_bub} K "
                    f"at {feed_stream.get('P', 101325.0):.0f} Pa"
                )
                if unit_type in ("Heater", "Cooler"):
                    lo, hi = round(t_bub + 15, 0), round(t_bub + 25, 0)
                    bubble_hint += (
                        f"\n→ {broken_unit_name} T_out target: "
                        f"{lo}–{hi} K (bubble_point + 15–25 K)"
                    )

        _SYSTEM_REVISE = (
            "You are a chemical process unit parameter expert.\n"
            "Fix ONLY the parameters of the specified broken unit.\n\n"
            "Output ONLY this JSON object (no markdown, no explanation):\n"
            '{"new_params": {<corrected unit parameters>}, '
            '"reasoning": "<one sentence>"}\n\n'
            "Unit parameter formats:\n"
            "  Heater/Cooler  : {\"T_out\": <K>, \"dP\": 0.0}\n"
            "  Pump/Compressor: {\"P_out\": <Pa>, \"efficiency\": 0.75}\n"
            "  Expander       : {\"P_out\": <Pa>, \"efficiency\": 0.75}\n"
            "  Vessel/Mixer   : {\"dP\": 0.0}\n\n"
            "MUST: temperatures in Kelvin, pressures in Pascals.\n"
            "MUST: Heater/Cooler T_out must be above the mixture bubble point + 15 K.\n"
            "MUST: Compressor/Pump P_out must exceed feed pressure.\n"
            "MUST: Expander P_out must be below feed pressure.\n\n"
            "━━━ EXAMPLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "UNIT TO FIX: HT-01 (Heater)\n"
            'Current parameters: {"T_out": 310.0, "dP": 0.0}\n'
            "PROBLEM: T_out=310 K is below the methanol/water bubble point.\n"
            "Feed stream conditions:\n"
            "  FEED: T=298.15K  P=101325Pa  {Methanol: 0.5, Water: 0.5}\n"
            "Estimated mixture bubble point: 355.0 K at 101325 Pa\n"
            "→ HT-01 T_out target: 370–380 K (bubble_point + 15–25 K)\n"
            'Output: {"new_params": {"T_out": 372.0, "dP": 0.0}, '
            '"reasoning": "HT-01 T_out raised 310→372K — above methanol/water bubble point ~355K."}\n'
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        base_prompt = (
            f"UNIT TO FIX: {broken_unit_name} ({unit_type})\n"
            f"Current parameters: {json.dumps(current_params)}\n\n"
            f"PROBLEM: {reason}\n\n"
            f"Feed stream conditions:\n"
            + ("\n".join(feed_summary) if feed_summary else "  (none specified)")
            + bubble_hint
            + f"\n\nCompounds: {compounds}\n"
            f"Description: {description}\n\n"
            f"Output ONLY: "
            '{"new_params": {...}, "reasoning": "<one sentence>"}'
        )

        import copy as _copy
        last_error = ""
        prompt = base_prompt
        for attempt in range(3):
            try:
                raw = chat(prompt, system=_SYSTEM_REVISE, model=self._model,
                           temperature=0.0 if attempt == 0 else 0.3)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = "\n".join(
                        ln for ln in raw.splitlines()
                        if not ln.strip().startswith("```")).strip()
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if not m:
                    raise ValueError("No JSON object in response")
                parsed = json.loads(m.group(0))
                new_params = parsed.get("new_params")
                if not isinstance(new_params, dict) or not new_params:
                    raise ValueError(
                        f'"new_params" must be a non-empty dict, got: {new_params}')
                updated = _copy.deepcopy(flowsheet)
                for u in updated.get("units", []):
                    if u["tag"] == broken_unit_name:
                        for k, v in new_params.items():
                            u[k] = v
                        break
                errors = schema.validate(updated)
                if errors:
                    raise ValueError(f"Schema errors after patch: {errors[0]}")
                return updated
            except Exception as exc:
                last_error = str(exc)
                prompt = (
                    f"CORRECTION REQUIRED — previous output had an error:\n"
                    f"  {last_error}\n\n"
                    f"Return ONLY: "
                    f'{{\"new_params\": {{...}}, \"reasoning\": \"<one sentence>\"}}\n\n'
                    + base_prompt
                )

        raise ValueError(
            f"Planner.revise() failed after 3 attempts for unit '{broken_unit_name}'. "
            f"Last error: {last_error}")

    # ── Image path (Gemini only — legacy single-shot) ─────────────────────────

    def _plan_with_image(self, description: str, image_path: str,
                         max_retries: int, constraint: str = "") -> dict:
        if not self._model.lower().startswith("gemini"):
            raise ValueError(
                f"Image input requires a Gemini model. Got '{self._model}'. "
                "Pass model='gemini-2.5-flash' or similar.")

        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set.")

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(system_instruction=self._system)

        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        suffix = img.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".webp": "image/webp"}.get(suffix, "image/png")

        base_prompt = f"{constraint}{_LEGACY_FEW_SHOT}\n\nNow generate the JSON for:\n{description}"
        initial_parts = [
            types.Part(text=base_prompt),
            types.Part(inline_data=types.Blob(mime_type=mime, data=img.read_bytes())),
        ]
        contents = [types.Content(role="user", parts=initial_parts)]
        last_errors: list[str] = []

        for _ in range(max_retries):
            response = client.models.generate_content(
                model=self._model, contents=contents, config=config)
            raw = response.text
            _, flowsheet, last_errors = self._legacy_validate(base_prompt, raw)
            if flowsheet is not None:
                return flowsheet

            error_msg = "\n".join(f"- {e}" for e in last_errors)
            contents = contents + [
                types.Content(role="model", parts=[types.Part(text=raw)]),
                types.Content(role="user", parts=[types.Part(text=(
                    f"Errors in your output:\n{error_msg}\n"
                    "Fix them and output valid JSON only."))]),
            ]

        raise ValueError(
            f"Planner (image) failed after {max_retries} attempts. "
            f"Last errors: {last_errors}")

    # ── Shared legacy validation (used by revise() and image path) ────────────

    def _legacy_validate(self, base_prompt: str, raw: str
                         ) -> tuple[str, dict | None, list[str]]:
        try:
            flowsheet = schema.from_json(raw)
        except json.JSONDecodeError as e:
            errors = [f"JSON parse error: {e}"]
            return _legacy_json_retry(base_prompt, raw, str(e)), None, errors

        errors = schema.validate(flowsheet)
        if errors:
            primary = errors[0]
            return _legacy_schema_retry(base_prompt, raw, primary, errors), None, errors

        physics_issues = schema.physics_validate(flowsheet)
        physics_errors = [i for i in physics_issues if i.severity.value == "ERROR"]
        if physics_errors:
            error_strs = [str(i) for i in physics_errors]
            return _legacy_physics_retry(base_prompt, raw, physics_errors), None, error_strs

        return base_prompt, flowsheet, []

    @staticmethod
    def _last_generated_json(raw: str) -> str:
        try:
            schema.from_json(raw)
            return raw
        except Exception:
            import re as _re
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            return m.group(0) if m else raw


# ── Legacy constraint builder (for revise / image path) ───────────────────────

def _build_legacy_constraint(
        compounds:              list[str] | None = None,
        suggested_compositions: dict | None = None,
        topology_feedback:      str | None = None,
        condition_feedback:     str | None = None,
        package_feedback:       str | None = None,
        compound_feedback:      str | None = None,
        property_package:       str | None = None,
        condition_estimate:     str | None = None,
        topology_hint:          TopologyHint | None = None,
) -> str:
    lines = []

    if property_package:
        lines += [
            "═══ THERMODYNAMICS — pre-selected by ThermoAgent (DO NOT CHANGE) ═══",
            f'  "property_package": "{property_package}"',
            "You MUST use exactly this value. Any other package will be rejected.",
            "---", "",
        ]

    if topology_hint:
        lines += [topology_hint.as_constraint_block(), ""]

    if topology_feedback:
        lines += [
            "TOPOLOGY CORRECTION — previous attempt had unfixable errors:",
            topology_feedback, "",
            "Rules: each src_port → exactly ONE stream; Vessel port 0=vapour, 1=liquid;",
            "every intermediate stream must appear as both src and dst.",
            "---", "",
        ]

    if condition_feedback:
        lines += [
            "CONDITION FEEDBACK — previous flowsheet had infeasible conditions:",
            condition_feedback, "",
            "Do not repeat the same T/P. T_out must be ABOVE the mixture bubble point.",
            "---", "",
        ]

    if condition_estimate:
        lines += [
            "CONDITION ESTIMATE — starting point for T/P values:",
            condition_estimate, "---", "",
        ]

    if package_feedback:
        lines += [
            "PACKAGE FEEDBACK — models tried and why each failed:",
            package_feedback, "", "---", "",
        ]

    if compound_feedback:
        lines += [
            "COMPOUND CORRECTION — previous flowsheet used incorrect names:",
            compound_feedback, "",
            "Use ONLY the exact DWSIM names in CONSTRAINTS below.",
            "---", "",
        ]

    if compounds or suggested_compositions:
        lines.append("CONSTRAINTS — override anything in the description:\n")
        if compounds:
            lines.append(
                f"Compounds (exact spelling/case):\n  {compounds}\n")
        if suggested_compositions:
            lines.append("Feed compositions:")
            for alias, comp in suggested_compositions.items():
                frac_str = ", ".join(f"{k}: {v:.4f}" for k, v in comp.items())
                lines.append(f"  '{alias}' → {{{frac_str}}}")
            lines.append("")
        lines.append("---\n")

    return "\n".join(lines)


# ── Legacy retry prompts (revise / image path) ─────────────────────────────────

def _legacy_json_retry(base: str, bad: str, error: str) -> str:
    return (f"{base}\n\nJSON SYNTAX ERROR — fix syntax only:\n  {error}\n"
            f"Previous output:\n{bad}\n\nReturn valid JSON only.")


def _legacy_schema_retry(base: str, bad: str, primary: str, all_errors: list[str]) -> str:
    tail = (f"\n  ({len(all_errors)-1} further error(s) after this is fixed.)"
            if len(all_errors) > 1 else "")
    return (f"{base}\n\nSCHEMA ERROR — fix this specific issue:{tail}\n  {primary}\n"
            f"Previous output:\n{bad}\n\nFix only the issue above. Return JSON only.")


def _legacy_physics_retry(base: str, bad: str, errors: list) -> str:
    p = errors[0]
    tail = (f"\n  ({len(errors)-1} further error(s) after this is fixed.)"
            if len(errors) > 1 else "")
    return (f"{base}\n\nPHYSICS ERROR — fix this specific issue:{tail}\n"
            f"  Location : {p.location}\n"
            f"  Problem  : {p.message}\n"
            f"  Fix      : {p.fix}\n"
            f"Previous output:\n{bad}\n\nApply only the fix above. Return JSON only.")


# ── Legacy prompts (revise / image path) ──────────────────────────────────────

_SYSTEM_LEGACY = """\
You are a chemical process design expert. Your job is to convert a process
description or a hand-drawn P&ID sketch into a structured flowsheet definition.

Output ONLY valid JSON matching this schema — no explanation, no markdown fences:

{
  "compounds": ["<name>", ...],
  "property_package": "<name>",
  "streams": [
    {"tag": "<TAG>", "T": <K>, "P": <Pa>, "flow": <mol/s>,
     "composition": {"<compound>": <mole_fraction>, ...}}
  ],
  "units": [
    {"tag": "<TAG>", "type": "<type>", <type-specific params>}
  ],
  "connections": [
    ["<src_tag>", "<dst_tag>", <src_port>, <dst_port>]
  ]
}

Rules:
- Every unit operation must connect through MaterialStreams — never link two
  unit ops directly.
- Intermediate streams linking unit ops need only a "tag" field.
- Mole fractions in each feed stream must sum to 1.0.
- SI units throughout: K, Pa, mol/s.
- src_port 0 = primary/vapour outlet, src_port 1 = liquid/secondary outlet.
- Supported unit types: Heater, Cooler, Vessel, Mixer, Splitter, Pump, Compressor, Expander.
- Supported property packages: Raoult's Law, NRTL, UNIQUAC, Peng-Robinson,
  Soave-Redlich-Kwong, Lee-Kesler-Plöcker.
"""

_LEGACY_FEW_SHOT = """\
Example — "Heat a 50/50 methanol/water stream at 1 atm and 25°C to 77°C then flash it":
{
  "compounds": ["Methanol", "Water"],
  "property_package": "NRTL",
  "streams": [
    {"tag": "FEED", "T": 298.15, "P": 101325.0, "flow": 1.0,
     "composition": {"Methanol": 0.5, "Water": 0.5}},
    {"tag": "HOT"}, {"tag": "VAP"}, {"tag": "LIQ"}
  ],
  "units": [
    {"tag": "HT-01", "type": "Heater", "T_out": 350.0, "dP": 0.0},
    {"tag": "V-01",  "type": "Vessel", "dP": 0.0}
  ],
  "connections": [
    ["FEED","HT-01",0,0],["HT-01","HOT",0,0],["HOT","V-01",0,0],
    ["V-01","VAP",0,0],["V-01","LIQ",1,0]
  ]
}
"""

_REVISE_EXAMPLE = """\
── REVISION TASK ──────────────────────────────────────────────────────────────
Fix EXACTLY ONE broken unit in the flowsheet below.
Keep ALL other units, streams, and connections unchanged.

━━━ EXAMPLE — wrong T_out (below bubble point) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BROKEN:
{"compounds":["Methanol","Water"],"property_package":"NRTL",
 "streams":[{"tag":"FEED","T":298.15,"P":101325.0,"flow":1.0,
             "composition":{"Methanol":0.5,"Water":0.5}},
            {"tag":"HOT"},{"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"HT-01","type":"Heater","T_out":310.0,"dP":0.0},
          {"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","HT-01",0,0],["HT-01","HOT",0,0],
                ["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}

PROBLEM: HT-01 T_out=310 K is below the methanol/water bubble point (~355 K).
FIX: raise HT-01 T_out to 365 K (above bubble point).

CORRECTED:
{"compounds":["Methanol","Water"],"property_package":"NRTL",
 "streams":[{"tag":"FEED","T":298.15,"P":101325.0,"flow":1.0,
             "composition":{"Methanol":0.5,"Water":0.5}},
            {"tag":"HOT"},{"tag":"VAP"},{"tag":"LIQ"}],
 "units":[{"tag":"HT-01","type":"Heater","T_out":365.0,"dP":0.0},
          {"tag":"V-01","type":"Vessel","dP":0.0}],
 "connections":[["FEED","HT-01",0,0],["HT-01","HOT",0,0],
                ["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]]}

Changed: HT-01 T_out 310.0 → 365.0.  All else unchanged.

── END OF EXAMPLE ─────────────────────────────────────────────────────────────

"""
