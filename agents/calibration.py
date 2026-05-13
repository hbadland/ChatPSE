"""
CalibrationAgent — deterministic binary interaction parameter retrieval.

Intercepts PARAM_MISSING failure signals, queries a curated corpus of literature
NRTL/UNIQUAC parameters via exact normalised compound-name lookup, and returns an
updated flowsheet JSON with binary_parameters populated.

Architecture:
  - Corpus: rag/sources/binary_parameters.json (curated, versioned in git)
  - Lookup: O(1) dict keyed by (norm_a, norm_b, model) expanding all aliases
  - Temperature guard: hard block >20% outside fit range; warning at 10–20%
  - All-or-nothing: success=True only if every compound pair is covered and
    every pair passes the temperature guard
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Optional

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "sources", "binary_parameters.json")

# Module-level lookup: (norm_a, norm_b, model) -> ParameterRecord, built once on
# first CalibrationAgent.run() call via _build_lookup().
_LOOKUP: dict[tuple[str, str, str], "ParameterRecord"] = {}
_LOOKUP_BUILT = False


@dataclass
class ParameterRecord:
    compound_a:  str
    compound_b:  str
    model:       str          # "NRTL" | "UNIQUAC"
    A12:         float
    A21:         float
    alpha12:     float        # 0.0 for UNIQUAC
    B12:         float
    B21:         float
    source:      str
    T_min_K:     Optional[float]
    T_max_K:     Optional[float]
    confidence:  str


@dataclass
class CalibrationResult:
    success:             bool
    updated_flowsheet:   dict
    pairs_found:         list[tuple[str, str]]       = field(default_factory=list)
    pairs_missing:       list[tuple[str, str]]       = field(default_factory=list)
    pairs_blocked:       list[tuple[str, str]]       = field(default_factory=list)
    parameters_injected: list[ParameterRecord]       = field(default_factory=list)
    notes:               list[str]                   = field(default_factory=list)


def _norm(s: str) -> str:
    return s.strip().lower()


def _build_lookup() -> None:
    """
    Load binary_parameters.json and build _LOOKUP once.

    For each corpus entry, every (alias_a × alias_b) combination is inserted in
    both orderings so that query order does not matter.  Later entries silently
    lose to earlier ones on collision (first-found wins), which keeps the most
    authoritative duplicates — the corpus should be deduplicated over time.
    """
    global _LOOKUP, _LOOKUP_BUILT
    if _LOOKUP_BUILT:
        return
    _LOOKUP_BUILT = True  # set before loading so re-entrant calls don't loop

    try:
        with open(_CORPUS_PATH, encoding="utf-8") as fh:
            corpus: list[dict] = json.load(fh)
    except FileNotFoundError:
        return

    for entry in corpus:
        names_a: list[str] = [entry["compound_a"]] + entry.get("aliases_a", [])
        names_b: list[str] = [entry["compound_b"]] + entry.get("aliases_b", [])
        model: str = entry["model"]

        record = ParameterRecord(
            compound_a = entry["compound_a"],
            compound_b = entry["compound_b"],
            model      = model,
            A12        = float(entry["A12"]),
            A21        = float(entry["A21"]),
            alpha12    = float(entry.get("alpha12", 0.0)),
            B12        = float(entry.get("B12", 0.0)),
            B21        = float(entry.get("B21", 0.0)),
            source     = entry["source"],
            T_min_K    = float(entry["T_min_K"]) if entry.get("T_min_K") is not None else None,
            T_max_K    = float(entry["T_max_K"]) if entry.get("T_max_K") is not None else None,
            confidence = entry.get("confidence", "medium"),
        )

        for na in names_a:
            for nb in names_b:
                key_fwd = (_norm(na), _norm(nb), model)
                key_rev = (_norm(nb), _norm(na), model)
                if key_fwd not in _LOOKUP:
                    _LOOKUP[key_fwd] = record
                if key_rev not in _LOOKUP:
                    _LOOKUP[key_rev] = record


def _lookup_pair(compound_a: str, compound_b: str, model: str) -> Optional[ParameterRecord]:
    _build_lookup()
    return _LOOKUP.get((_norm(compound_a), _norm(compound_b), model))


def _check_temperature(
    record: ParameterRecord,
    feed_T_K: float,
    pair_str: str,
) -> tuple[bool, list[str]]:
    """
    Evaluate whether feed_T_K is within the parameter fit range.

    Returns (ok, notes):
      ok=False  when extrapolation exceeds 20% of the fit interval (hard block).
      ok=True   with a warning note when extrapolation is 10–20%.
      ok=True   with no note when T is inside the range or within 10%.

    The 20% threshold is intentionally conservative: NRTL/UNIQUAC parameters
    fitted to VLE data can diverge significantly outside the regression window,
    and injecting bad BIPs is worse than falling back to ThermoAgent.
    """
    if record.T_min_K is None or record.T_max_K is None:
        return True, []

    T_range = record.T_max_K - record.T_min_K
    if T_range <= 0:
        return True, []

    notes: list[str] = []

    if feed_T_K > record.T_max_K:
        fraction = (feed_T_K - record.T_max_K) / T_range
        if fraction > 0.20:
            return False, [
                f"{pair_str} {record.model}: feed T={feed_T_K:.1f} K exceeds fit range "
                f"[{record.T_min_K:.0f}–{record.T_max_K:.0f} K] by {fraction*100:.0f}% of the "
                f"interval — hard block (>20% extrapolation). Route to HUMAN or provide a wider corpus entry."
            ]
        if fraction > 0.10:
            notes.append(
                f"{pair_str} {record.model}: feed T={feed_T_K:.1f} K is {fraction*100:.0f}% above "
                f"fit range [{record.T_min_K:.0f}–{record.T_max_K:.0f} K] — mild extrapolation, verify results."
            )

    elif feed_T_K < record.T_min_K:
        fraction = (record.T_min_K - feed_T_K) / T_range
        if fraction > 0.20:
            return False, [
                f"{pair_str} {record.model}: feed T={feed_T_K:.1f} K is below fit range "
                f"[{record.T_min_K:.0f}–{record.T_max_K:.0f} K] by {fraction*100:.0f}% of the "
                f"interval — hard block (>20% extrapolation). Route to HUMAN or provide a wider corpus entry."
            ]
        if fraction > 0.10:
            notes.append(
                f"{pair_str} {record.model}: feed T={feed_T_K:.1f} K is {fraction*100:.0f}% below "
                f"fit range [{record.T_min_K:.0f}–{record.T_max_K:.0f} K] — mild extrapolation, verify results."
            )

    if record.confidence == "low":
        notes.append(
            f"{pair_str} {record.model}: confidence is 'low' — verify against primary literature before use."
        )

    return True, notes


def _extract_feed_temperature(flowsheet: dict) -> Optional[float]:
    """Return the feed stream temperature [K], or None if not found.

    Feed streams have no incoming connections.  Schema field is 'T' (K).
    """
    conns = flowsheet.get("connections", [])
    has_incoming = {c[1] for c in conns if len(c) >= 2}
    for stream in flowsheet.get("streams", []):
        tag = stream.get("tag", "")
        if tag not in has_incoming:
            t = stream.get("T") or stream.get("T_K") or stream.get("temperature_K")
            if t is not None:
                return float(t)
    return None


class CalibrationAgent:
    """
    Retrieves literature binary interaction parameters from the curated JSON corpus
    via exact normalised compound-name lookup and injects them into the flowsheet.

    Zero LLM calls — deterministic O(1) retrieval per pair.
    All-or-nothing: success=True only if ALL pairs are found in the corpus AND
    all pass the temperature guard (no hard-block extrapolations).
    """

    def __init__(self, corpus_path: Optional[str] = None) -> None:
        global _CORPUS_PATH, _LOOKUP_BUILT
        if corpus_path is not None:
            _CORPUS_PATH = corpus_path
            _LOOKUP_BUILT = False   # force rebuild from new path

    def has_coverage(self, flowsheet: dict) -> bool:
        """
        Return True iff the BIP corpus covers every compound pair for the
        flowsheet's current property package (NRTL or UNIQUAC).

        Zero-cost check — no temperature guard, no CalibrationResult allocation.
        Use this to decide whether to route to CALIBRATION before committing an
        iteration to a full run() call.
        """
        _build_lookup()
        model = flowsheet.get("property_package", "")
        if model not in ("NRTL", "UNIQUAC"):
            return False
        compounds = flowsheet.get("compounds", [])
        if len(compounds) < 2:
            return False
        for ca, cb in itertools.combinations(compounds, 2):
            if _lookup_pair(ca, cb, model) is None:
                return False
        return True

    def run(self, flowsheet: dict) -> CalibrationResult:
        """
        Retrieve parameters for every compound pair in the flowsheet.

        Args:
            flowsheet: Current flowsheet dict (must include 'compounds' list and
                       optionally 'property_package').

        Returns:
            CalibrationResult with success=True iff all pairs found and T-validated.
        """
        compounds = flowsheet.get("compounds", [])
        model = flowsheet.get("property_package", "NRTL")

        if model not in ("NRTL", "UNIQUAC"):
            return CalibrationResult(
                success=False,
                updated_flowsheet=flowsheet,
                notes=[
                    f"CalibrationAgent: property package '{model}' is not NRTL or UNIQUAC"
                    f" — no BIP injection needed."
                ],
            )

        if len(compounds) < 2:
            return CalibrationResult(
                success=False,
                updated_flowsheet=flowsheet,
                notes=["CalibrationAgent: fewer than 2 compounds — no binary pairs to retrieve."],
            )

        feed_T    = _extract_feed_temperature(flowsheet)
        pairs     = list(itertools.combinations(compounds, 2))

        pairs_found:   list[tuple[str, str]] = []
        pairs_missing: list[tuple[str, str]] = []
        pairs_blocked: list[tuple[str, str]] = []
        records:       list[ParameterRecord] = []
        all_notes:     list[str]             = []

        for ca, cb in pairs:
            record = _lookup_pair(ca, cb, model)

            if record is None:
                pairs_missing.append((ca, cb))
                continue

            pair_str = f"{ca}/{cb}"
            if feed_T is not None:
                ok, t_notes = _check_temperature(record, feed_T, pair_str)
                all_notes.extend(t_notes)
                if not ok:
                    pairs_blocked.append((ca, cb))
                    continue

            pairs_found.append((ca, cb))
            records.append(record)

        if pairs_missing or pairs_blocked:
            msgs: list[str] = []
            if pairs_missing:
                missing_str = ", ".join(f"{a}/{b}" for a, b in pairs_missing)
                msgs.append(
                    f"CalibrationAgent: parameters not found in corpus for: {missing_str}"
                )
            if pairs_blocked:
                blocked_str = ", ".join(f"{a}/{b}" for a, b in pairs_blocked)
                msgs.append(
                    f"CalibrationAgent: temperature hard block for: {blocked_str}"
                    f" — expand corpus T-range or route to HUMAN."
                )
            return CalibrationResult(
                success=False,
                updated_flowsheet=flowsheet,
                pairs_found=pairs_found,
                pairs_missing=pairs_missing,
                pairs_blocked=pairs_blocked,
                parameters_injected=records,
                notes=all_notes + msgs,
            )

        # All pairs found and T-validated — inject into flowsheet['binary_parameters']
        binary_parameters: list[dict] = []
        for rec in records:
            entry: dict = {
                "model":      rec.model,
                "compound_a": rec.compound_a,
                "compound_b": rec.compound_b,
                "A12":        rec.A12,
                "A21":        rec.A21,
                "source":     rec.source,
                "confidence": rec.confidence,
            }
            if rec.model == "NRTL":
                entry["alpha12"] = rec.alpha12
            if rec.B12 != 0.0:
                entry["B12"] = rec.B12
            if rec.B21 != 0.0:
                entry["B21"] = rec.B21
            if rec.T_min_K is not None:
                entry["T_min_K"] = rec.T_min_K
            if rec.T_max_K is not None:
                entry["T_max_K"] = rec.T_max_K
            binary_parameters.append(entry)

        updated = {**flowsheet, "binary_parameters": binary_parameters}

        return CalibrationResult(
            success=True,
            updated_flowsheet=updated,
            pairs_found=pairs_found,
            pairs_missing=[],
            parameters_injected=records,
            notes=all_notes,
        )
