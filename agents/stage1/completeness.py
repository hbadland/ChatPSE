"""
Completeness-verification loop around unit extraction.

Targets systematic UNDER-extraction on large flowsheets (the single-pass
extractor captures the reactive core but drops separation/purification/downstream
units that ARE described in the text).  It does NOT invent domain-typical units.

Loop:  extract (baseline) → [critic finds text-described-but-missing units →
augment] × up to 3, stopping when the critic returns nothing new.

Anti-hallucination guard (deterministic): every critic-claimed unit must quote a
`span` — a phrase from the description — and that span MUST actually occur in the
description text.  Claims whose span is absent are REJECTED and logged, never
added.  This is what separates "the text says 'purified in a PSA unit'" (accept)
from "hydrogen plants usually have a PSA" (reject).

Toggle: LOOP_ENABLED (set by the 'completeness' ablation mode) or the
COMPLETENESS_LOOP env var.  Off by default — baseline extraction is unchanged.
"""
from __future__ import annotations

import json
import os
import re
import sys

from agents.llm import chat, retry_temperature
from agents.stage1.unit_extractor import (
    SemanticUnit, SemanticUnits, SUPPORTED_UNIT_TYPES, _TAG_ABBREV, _parse_json)

# Flipped True by benchmark.ablation.apply_ablation for the 'completeness' mode.
LOOP_ENABLED = False
MAX_ITERS = 3


def enabled() -> bool:
    return LOOP_ENABLED or os.environ.get(
        "COMPLETENESS_LOOP", "").strip().lower() in ("1", "true", "yes")


_SYSTEM = """\
/no_think
You are a COMPLETENESS CHECKER for chemical-process unit extraction.
Given a process description and a list of ALREADY-extracted unit operations,
find unit operations that are EXPLICITLY described in the text but MISSING from
the list.

HARD RULES:
- Report a unit ONLY if the description explicitly names or clearly describes it,
  and quote the EXACT phrase from the description as "span".
- NEVER report a unit you infer from domain knowledge. If the text does not
  describe it, do not report it.
    REPORT:        text says "the gas is purified in a PSA unit" and no such unit
                   is present  → missing separation/purification unit.
    DO NOT REPORT: "hydrogen plants usually have a PSA, so add one"  → not in the
                   text; that is hallucination.
- type MUST be one of: Heater Cooler Vessel Mixer Splitter Pump Compressor
  Expander ConversionReactor.
- If nothing described is missing, return {"missing": []}.

Return ONLY JSON:
{"missing": [{"tag": "V-02", "type": "Vessel", "role": "...", "span": "exact phrase from description"}]}

Example
Description: "... the shifted gas is cooled to condense water, which is knocked
out in separators, before the dry gas is purified in a pressure swing adsorption
unit."
Already-extracted: RX-01 (ConversionReactor), RX-02 (ConversionReactor)
Output:
{"missing": [
  {"tag": "CL-01", "type": "Cooler", "role": "cool shifted gas to condense water",
   "span": "the shifted gas is cooled to condense water"},
  {"tag": "V-01", "type": "Vessel", "role": "knock out condensed water",
   "span": "knocked out in separators"},
  {"tag": "V-02", "type": "Vessel", "role": "pressure swing adsorption purification",
   "span": "purified in a pressure swing adsorption unit"}
]}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _span_in_text(span: str, description: str) -> bool:
    """The justifying phrase must actually occur in the description (normalised)."""
    span_n = _norm(span)
    if len(span_n) < 4:
        return False
    return span_n in _norm(description)


def _critic_call(description: str, sem_units: SemanticUnits, model: str,
                 attempt: int) -> list[dict]:
    unit_lines = "\n".join(
        f"  {u.tag} ({u.type}) — {u.role}" for u in sem_units.units) or "  (none)"
    prompt = (f"Process description:\n{description}\n\n"
              f"Already-extracted units:\n{unit_lines}\n\n"
              "List units explicitly described in the text but missing from the "
              "list above. Quote the justifying span for each.")
    raw = chat(prompt, system=_SYSTEM, model=model,
               temperature=retry_temperature(attempt), max_tokens=2048)
    try:
        return _parse_json(raw).get("missing", []) or []
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return []


def _alloc_tag(utype: str, taken: set[str]) -> str:
    abbrev = _TAG_ABBREV.get(utype, "U")
    i = 1
    while f"{abbrev}-{i:02d}" in taken:
        i += 1
    tag = f"{abbrev}-{i:02d}"
    taken.add(tag)
    return tag


def run_completeness_loop(
    description: str,
    sem_units:   SemanticUnits,
    model:       str,
    max_iters:   int = MAX_ITERS,
) -> tuple[SemanticUnits, dict]:
    """
    Returns (possibly-augmented SemanticUnits, completeness_log).

    completeness_log = {
      "pre_loop_n_units": int, "post_loop_n_units": int,
      "iterations": [{"iteration", "claimed":[...], "accepted":[...],
                      "rejected":[{...,"reject_reason"}], "n_before", "n_after"}]
    }
    Every accepted unit carries a `span` verified to occur in the description.
    """
    units = list(sem_units.units)
    log: dict = {"pre_loop_n_units": len(units), "iterations": []}

    for it in range(max_iters):
        claimed = _critic_call(description, SemanticUnits(units=units), model, it)
        taken = {u.tag for u in units}
        existing_type_roles = {(u.type, _norm(u.role)) for u in units}
        accepted, rejected = [], []

        for c in claimed:
            utype = c.get("type", "")
            span = c.get("span", "")
            role = c.get("role", "")
            entry = {"tag": c.get("tag"), "type": utype, "role": role, "span": span}
            if utype not in SUPPORTED_UNIT_TYPES:
                rejected.append({**entry, "reject_reason": f"unsupported type {utype!r}"})
            elif not _span_in_text(span, description):
                rejected.append({**entry, "reject_reason": "span not found in description"})
            elif (utype, _norm(role)) in existing_type_roles:
                rejected.append({**entry, "reject_reason": "duplicate of existing unit"})
            else:
                tag = _alloc_tag(utype, taken)
                accepted.append({**entry, "tag": tag})
                existing_type_roles.add((utype, _norm(role)))
                units.append(SemanticUnit(tag=tag, type=utype, role=role))

        log["iterations"].append({
            "iteration": it, "claimed": claimed,
            "accepted": accepted, "rejected": rejected,
            "n_before": len(units) - len(accepted), "n_after": len(units),
        })
        print(f"[COMPLETENESS] iter {it}: claimed={len(claimed)} "
              f"accepted={len(accepted)} rejected={len(rejected)} "
              f"n_units {len(units) - len(accepted)}→{len(units)}",
              flush=True, file=sys.stderr)
        if not accepted:
            break

    log["post_loop_n_units"] = len(units)
    return SemanticUnits(units=units, raw_json=sem_units.raw_json), log
