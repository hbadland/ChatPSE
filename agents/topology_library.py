"""
Topology Library — template matching for common multi-unit process flowsheets.

Converts a natural language description into a canonical unit-operation sequence
BEFORE the PlannerAgent runs.  This is a pure pattern-matching / retrieval
problem, not a reasoning problem: the LLM should not have to invent topology
for processes that follow well-known design patterns.

The library provides:
  match(description, compounds)
      → TopologyHint | None

  TopologyHint.as_constraint_block()
      → str   (appended to the Planner constraint so the model sees a template)

Design principles
─────────────────
• Zero LLM calls — keyword scoring only.
• Returns the best-scoring template above a confidence threshold.
• Negative keywords disqualify a template regardless of positive score
  (e.g., "no phase separation" blocks all flash-vessel templates).
• Templates are structural: they describe unit types and their connection
  sequence but NOT stream conditions or property packages (those come from
  ThermoAgent and the condition estimator).
• The Planner is still responsible for generating valid JSON; the hint
  reduces its search space by providing a recommended skeleton.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Template definition ────────────────────────────────────────────────────────

@dataclass
class TopologyTemplate:
    name:        str
    description: str          # human-readable label for the template
    units:       list[str]    # unit types in processing order
    keywords:    list[str]    # scoring keywords (case-insensitive substring/regex)
    bonus_compounds: list[str] = field(default_factory=list)
    # If ANY negative keyword matches, the template is disqualified (score → 0)
    negative_keywords: list[str] = field(default_factory=list)
    # Additional clarification notes for the Planner constraint block
    notes:       list[str]    = field(default_factory=list)


@dataclass
class TopologyHint:
    template:    TopologyTemplate
    score:       float    # normalised 0–1

    def as_constraint_block(self) -> str:
        """
        Return a Planner-ready constraint string describing the recommended
        unit sequence.  The Planner must still supply all stream conditions,
        compound assignments, and connections — this is a skeleton, not a
        complete flowsheet.
        """
        units_str = " → ".join(self.template.units)
        lines = [
            "TOPOLOGY TEMPLATE — pattern-matched from process description:",
            f"  Template : {self.template.name}",
            f"  Unit sequence: {units_str}",
        ]
        for note in self.template.notes:
            lines.append(f"  Note: {note}")
        lines += [
            "",
            "Follow this unit sequence exactly unless the description explicitly",
            "requires a different arrangement.  Add intermediate streams between",
            "every pair of consecutive units.",
            "---",
        ]
        return "\n".join(lines)


# ── Template library ───────────────────────────────────────────────────────────
# Each template covers one canonical design pattern found in the benchmark.
# Keywords are scored against description + compound names.
#
# Keyword weights:  1× for single words, 2× for multi-word / regex patterns.
# Negative keywords: any match → template score becomes 0.0 (hard disqualifier).

_NO_FLASH_NEGATIVES = [
    "no phase separ", "no flash", "no separation requir",
    "without flash", "without phase", "no vessel",
]

_TEMPLATES: list[TopologyTemplate] = [

    # ── Gas compression with aftercooling and flash ────────────────────────────
    TopologyTemplate(
        name="Compressor + Aftercooler + Flash Vessel",
        description="Gas compressed to elevated pressure, cooled, then phase-separated",
        units=["Compressor", "Cooler", "Vessel"],
        keywords=[
            "compress", "compressor", "aftercool", "after-cool", "flash",
            "high pressure", "high-pressure",
            "cool the compressed", "cool and flash",
            "compress.*cool", "compress.*flash",   # compound patterns
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Compressor outlet → intermediate stream → Cooler → intermediate stream → Vessel.",
            "Vessel src_port=0 → vapour product, src_port=1 → liquid product.",
            "Set Compressor P_out in Pa (1 bar = 100000 Pa).",
            "Set Cooler T_out below the dew point of the compressed stream.",
        ],
    ),

    # ── Heater + flash (single stage) ─────────────────────────────────────────
    TopologyTemplate(
        name="Heater + Flash Vessel",
        description="Feed heated then partially vaporised in a flash drum",
        units=["Heater", "Vessel"],
        keywords=[
            "heat", "heater", "preheat", "flash", "partial vapori",
            "partial vapour", "two-phase", "two phase",
            "vapour fraction", "vapour product",
            "heat.*flash", "heater.*vessel",       # compound patterns
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Heater T_out MUST exceed the mixture bubble point to produce vapour.",
            "Vessel src_port=0 → vapour, src_port=1 → liquid.",
        ],
    ),

    # ── Cooler + flash ────────────────────────────────────────────────────────
    TopologyTemplate(
        name="Cooler + Flash Vessel",
        description="Hot vapour feed cooled into the two-phase region then flashed",
        units=["Cooler", "Vessel"],
        keywords=[
            "cool", "cooler", "condense", "partial condens",
            "hot vapour", "hot vapor", "flash",
            "vapour feed", "vapor feed", "two-phase",
            "cool.*flash", "cooler.*vessel",       # compound patterns
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Cooler T_out must be ABOVE the bubble point to produce two phases.",
            "A T_out below the dew point will produce only liquid (zero vapour).",
            "Vessel src_port=0 → vapour, src_port=1 → liquid.",
        ],
    ),

    # ── Three-stream mixer + heater + flash ───────────────────────────────────
    TopologyTemplate(
        name="3-Stream Mixer + Heater + Flash Vessel",
        description="Three separate feed streams mixed, heated, then flashed",
        units=["Mixer", "Heater", "Vessel"],
        keywords=[
            "three stream", "three-stream", "3 stream", "three feed",
            "three inlet", "mix three", "blend three",
            "mix.*heat", "heat.*mix", "mixer.*heater",
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Mixer accepts 3 inlet streams (dst_ports 0, 1, 2).",
            "Mixer outlet → Heater → Vessel.",
            "All three feed streams must have full T, P, flow, and composition.",
        ],
    ),

    # ── Two-stream mixer + heater + flash ─────────────────────────────────────
    TopologyTemplate(
        name="2-Stream Mixer + Heater + Flash Vessel",
        description="Two feed streams mixed, heated, then flashed",
        units=["Mixer", "Heater", "Vessel"],
        keywords=[
            "mix.*heat", "heat.*mix", "blend.*heat",
            "two stream", "two feed", "two inlet",
            "mixer", "blend", "combine",
            "mix.*flash", "blend.*flash",          # compound patterns
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Mixer accepts 2 inlet streams (dst_ports 0 and 1).",
            "Mixer outlet → Heater → Vessel.",
        ],
    ),

    # ── Two-stage flash ────────────────────────────────────────────────────────
    TopologyTemplate(
        name="Two-Stage Flash",
        description="Feed flashed twice at successively lower pressures",
        units=["Vessel", "Vessel"],
        keywords=[
            "two-stage flash", "two stage flash", "double flash",
            "first flash", "second flash",
            "stage 1.*flash", "stage 2.*flash",
            "sequential flash", "cascade flash",
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "First Vessel liquid outlet → Second Vessel.",
            "Each Vessel: src_port=0 vapour, src_port=1 liquid.",
            "Second Vessel operates at lower pressure than the first.",
            "Use separate tags for both vessels (e.g. V-01 and V-02).",
        ],
    ),

    # ── Heater + compressor (gas-phase heating before compression) ─────────────
    TopologyTemplate(
        name="Heater + Compressor",
        description="Gas pre-heated before compression (prevents liquefaction)",
        units=["Heater", "Compressor"],
        keywords=[
            "preheat.*compress", "heat.*compress",
            "superheat.*compres",
            "before compress", "prior to compress",
        ],
        notes=[
            "Heater raises feed T before compression — prevents two-phase flow in Compressor.",
            "No Vessel needed unless the description explicitly requires a flash step.",
        ],
    ),

    # ── Pump + flash (liquid pressurisation then flash) ───────────────────────
    TopologyTemplate(
        name="Pump + Flash Vessel",
        description="Liquid pumped to higher pressure then flashed at reduced pressure",
        units=["Pump", "Vessel"],
        keywords=[
            "pump", "pressurise", "pressurize",
            "flash.*pump", "pump.*flash",
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Pump P_out must exceed inlet pressure.",
            "Vessel src_port=0 → vapour, src_port=1 → liquid.",
        ],
    ),

    # ── Simple flash vessel only (no pre-conditioning) ────────────────────────
    TopologyTemplate(
        name="Single Flash Vessel",
        description="Feed at two-phase conditions flashed directly in a drum",
        units=["Vessel"],
        keywords=[
            "flash", "flash separ", "flash separate",
            "flash drum", "flash vessel",
            "partial vapori", "phase separ",
            "flash at", "flash the feed",
        ],
        negative_keywords=_NO_FLASH_NEGATIVES + [
            # avoid stealing descriptions that explicitly mention a pre-conditioner
            "heat.*flash", "cool.*flash", "compress.*flash",
            "heater.*flash", "cooler.*flash",
        ],
        notes=[
            "Feed stream must already be in the two-phase region (T between bubble and dew point).",
            "Vessel src_port=0 → vapour, src_port=1 → liquid.",
        ],
    ),

    # ── Mixer + Cooler + flash (hot streams blended then cooled) ──────────────
    TopologyTemplate(
        name="Mixer + Cooler + Flash Vessel",
        description="Hot streams mixed, cooled into two-phase region, then flashed",
        units=["Mixer", "Cooler", "Vessel"],
        keywords=[
            "mix.*cool", "cool.*mix",
            "blend.*cool", "combine.*cool",
            "hot.*mix.*flash", "mix.*cool.*flash",
        ],
        negative_keywords=_NO_FLASH_NEGATIVES,
        notes=[
            "Mixer → Cooler → Vessel.",
            "Cooler T_out must be between the bubble and dew points.",
        ],
    ),
]

# Minimum normalised score to emit a hint (prevents low-confidence noise).
# Set lower than naive intuition suggests because compound regex patterns boost
# scores for clear descriptions while negative keywords block false positives.
_CONFIDENCE_THRESHOLD = 0.20


# ── Public interface ───────────────────────────────────────────────────────────

def match(description: str, compounds: list[str] | None = None) -> TopologyHint | None:
    """
    Score every template against the process description (and optionally the
    compound list).  Return the best-matching TopologyHint above the confidence
    threshold, or None if no template matches well.

    Args:
        description: natural language process description (normalised form preferred)
        compounds  : list of DWSIM compound names from BasisAgent (optional)

    Returns:
        TopologyHint with the best-matching template, or None.
    """
    desc_lower = description.lower()
    compound_str = " ".join(compounds or []).lower()
    search_text  = desc_lower + " " + compound_str

    best_score    = 0.0
    best_template = None

    for tpl in _TEMPLATES:
        score = _score_template(tpl, search_text)
        if score > best_score:
            best_score    = score
            best_template = tpl

    if best_template is not None and best_score >= _CONFIDENCE_THRESHOLD:
        return TopologyHint(template=best_template, score=best_score)
    return None


def _score_template(tpl: TopologyTemplate, text: str) -> float:
    """
    Normalised keyword hit rate for one template.

    Negative keywords are checked first — any match returns 0.0.
    Positive score = matched_weight / total_weight, capped at 1.0.
    Multi-word / regex keywords get 2× weight on a hit.
    """
    # ── Negative keyword check (hard disqualifier) ────────────────────────────
    for nkw in tpl.negative_keywords:
        try:
            if re.search(nkw, text):
                return 0.0
        except re.error:
            if nkw in text:
                return 0.0

    # ── Positive keyword scoring ───────────────────────────────────────────────
    total_weight = 0.0
    hit_weight   = 0.0

    for kw in tpl.keywords:
        weight = 2.0 if " " in kw or ".*" in kw else 1.0
        total_weight += weight
        try:
            if re.search(kw, text):
                hit_weight += weight
        except re.error:
            if kw in text:
                hit_weight += weight

    if total_weight == 0:
        return 0.0
    return min(hit_weight / total_weight, 1.0)


def format_hint(hint: TopologyHint) -> str:
    """Return a one-line summary of the matched template (for logging)."""
    units_str = " → ".join(hint.template.units)
    return (f"Topology match '{hint.template.name}' "
            f"(score={hint.score:.2f}): {units_str}")
