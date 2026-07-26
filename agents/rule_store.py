"""
Failure-driven rule synthesis.

FailureRuleStore tracks repeated repair patterns across benchmark cases.
When a (unit_type, error_code, downstream_type) pattern occurs ≥ RULE_THRESHOLD
times with a consistent fix, a deterministic rule is synthesized and applied
automatically in future cases via apply_to_graph().

Typical lifecycle in a benchmark run:
    store = FailureRuleStore()
    store.load(RULES_PATH)               # restore rules from previous runs

    # After each repair iteration:
    store.record_fix(...)

    # Before the next case (or iteration):
    graph, changes = store.apply_to_graph(graph, compounds)

    store.save(RULES_PATH)               # persist learned rules
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

RULE_THRESHOLD = 3  # minimum occurrences before a rule is synthesized

# Persistent rule store written after every repair; loaded at orchestrator init.
RULES_PATH = str(Path(__file__).resolve().parent.parent / "results" / "rule_store.json")

# param → (provenance source field, description sentinel). A synthesized rule may
# overwrite a param only when its source is estimated ({computed, default_fallback}
# or untagged/non-sentinel); description-specified values are protected. Shared by
# apply()'s guard and the synthesis-exclusion in the record_fix caller.
_PROV_FIELDS = {
    "T_out":         ("_temperature_source", "_desc_T_out"),
    "temperature_K": ("_temperature_source", "_desc_T_out"),
    "P_out":         ("_pressure_source",     "_desc_P_out"),
}

# ── Compound classification ────────────────────────────────────────────────────

_COMPOUND_CLASSES: dict[str, set[str]] = {
    "alcohol":     {"Methanol", "Ethanol", "n-Propanol", "1-Propanol",
                    "Isopropanol", "2-Propanol", "n-Butanol", "1-Butanol"},
    "hydrocarbon": {"Methane", "Ethane", "Propane", "n-Butane", "i-Butane",
                    "Isobutane", "n-Pentane", "n-Hexane", "n-Heptane"},
    "ketone":      {"Acetone"},
    "aromatic":    {"Benzene", "Toluene"},
    "ester":       {"Ethyl Acetate"},
    "nitrile":     {"Acetonitrile"},
    "light_gas":   {"Methane", "Ethane", "Nitrogen", "Oxygen", "Hydrogen",
                    "Carbon Dioxide", "Hydrogen Sulfide", "Ammonia"},
    "water":       {"Water"},
    "halogenated": {"Chloroform", "Dichloromethane"},
}


def classify_compounds(compounds: list[str]) -> frozenset[str]:
    """Return the set of abstract compound classes present in the mixture."""
    classes: set[str] = set()
    for comp in compounds:
        for cls, members in _COMPOUND_CLASSES.items():
            if comp in members:
                classes.add(cls)
    return frozenset(classes) if classes else frozenset({"unknown"})


@dataclass
class FailurePattern:
    unit_type:        str
    error_code:       str
    downstream_type:  Optional[str]         # direct downstream unit type, or None
    param:            str
    fixes:            list[float]           = field(default_factory=list)
    # Abstract compound classes present when each fix was recorded.
    # Empty set = rule applies regardless of compound system.
    compound_classes: frozenset[str]        = field(default_factory=frozenset)

    def record(self, value: float) -> None:
        self.fixes.append(value)

    @property
    def count(self) -> int:
        return len(self.fixes)

    def median_fix(self) -> Optional[float]:
        if not self.fixes:
            return None
        s = sorted(self.fixes)
        return s[len(s) // 2]

    def matches_compounds(self, current_classes: frozenset[str]) -> bool:
        """
        True when the rule is applicable to the current compound system.

        A rule matches if:
          - It has no compound class constraint (applies universally), OR
          - At least one compound class overlaps with the current system
            (generalise across similar systems, not just exact match)
        """
        if not self.compound_classes:
            return True
        return bool(self.compound_classes & current_classes)

    def to_dict(self) -> dict:
        return {
            "unit_type":        self.unit_type,
            "error_code":       self.error_code,
            "downstream_type":  self.downstream_type,
            "param":            self.param,
            "fixes":            self.fixes,
            "compound_classes": list(self.compound_classes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailurePattern":
        return cls(
            unit_type        = d["unit_type"],
            error_code       = d["error_code"],
            downstream_type  = d.get("downstream_type"),
            param            = d["param"],
            fixes            = d.get("fixes", []),
            compound_classes = frozenset(d.get("compound_classes", [])),
        )


class FailureRuleStore:
    """
    Tracks failure patterns and synthesizes deterministic repair rules.

    Thread-safe for read access; writes should be serialized per-run.
    """

    def __init__(self) -> None:
        # Key: (unit_type, error_code, downstream_type or "")
        self._patterns: dict[tuple, FailurePattern] = {}

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_fix(
        self,
        unit_type:        str,
        error_code:       str,
        downstream_type:  Optional[str],
        param:            str,
        applied_value:    float,
        compounds:        Optional[list[str]] = None,
    ) -> None:
        """
        Record one repair attempt.

        Compound classes are inferred from `compounds` if provided and merged
        into the pattern's class set so the rule generalises across similar
        compound systems.
        """
        key = (unit_type, error_code, downstream_type or "")
        if key not in self._patterns:
            classes = classify_compounds(compounds) if compounds else frozenset()
            self._patterns[key] = FailurePattern(
                unit_type        = unit_type,
                error_code       = error_code,
                downstream_type  = downstream_type,
                param            = param,
                compound_classes = classes,
            )
        else:
            # Accumulate compound classes from each recorded case
            if compounds:
                existing = self._patterns[key].compound_classes
                self._patterns[key].compound_classes = (
                    existing | classify_compounds(compounds))
        self._patterns[key].record(applied_value)

    # ── Active rules ───────────────────────────────────────────────────────────

    def active_rules(self) -> list[FailurePattern]:
        """Patterns that have hit the synthesis threshold."""
        return [p for p in self._patterns.values() if p.count >= RULE_THRESHOLD]

    def num_patterns(self) -> int:
        return len(self._patterns)

    def num_active(self) -> int:
        return len(self.active_rules())

    # ── Application ───────────────────────────────────────────────────────────

    def apply_to_graph(
        self,
        graph:     Any,     # FlowsheetGraph — avoid circular import at module level
        compounds: list[str],
    ) -> tuple[Any, list[str]]:
        """
        Apply all active synthesized rules to the graph.
        Returns (modified_graph, change_log).

        Rules are applied only when:
          • node.unit_type matches the pattern's unit_type
          • if pattern has downstream_type, at least one direct downstream
            unit has that type
          • the current param value differs from the rule value by > 5 K / 5% Pa
        """
        rules = self.active_rules()
        if not rules:
            return graph, []

        current_classes = classify_compounds(compounds)
        g = graph.copy()
        changes: list[str] = []

        for rule in rules:
            # Skip rules that don't generalise to the current compound system
            if not rule.matches_compounds(current_classes):
                continue
            target_val = rule.median_fix()
            if target_val is None:
                continue

            for node in g.units():
                if node.unit_type != rule.unit_type:
                    continue

                # Check downstream type constraint
                if rule.downstream_type:
                    outlet_types = _outlet_unit_types(g, node.tag)
                    if rule.downstream_type not in outlet_types:
                        continue

                # Only apply if current value is meaningfully different
                current = node.params.get(rule.param)
                if current is not None:
                    diff = abs(float(current) - target_val)
                    if rule.param == "T_out" and diff <= 5.0:
                        continue
                    if rule.param == "P_out" and diff / max(abs(target_val), 1.0) <= 0.05:
                        continue

                # Provenance guard: a synthesized rule must NOT overwrite a value
                # that came from the description. Overwrite ONLY genuinely-estimated
                # / absent / untrusted values — whitelist {computed, default_fallback,
                # fallback}, or untagged and not the description sentinel. 'fallback'
                # is a pooled desc-list guess (structured attribution absent), so it
                # is untrusted and overwritable like an estimate. 'specified' (flat
                # description) and 'extracted' (structured per-unit attribution) and
                # inherited/template are protected by construction. Applies to
                # temperature (T_out/temperature_K) AND pressure (P_out). Log the
                # suppression so this bug's blast radius is measurable per run.
                _prov = _PROV_FIELDS.get(rule.param)   # (source_field, desc_sentinel)
                if _prov is not None:
                    _src = node.params.get(_prov[0])
                    _overwritable = (_src in ("computed", "default_fallback", "fallback")
                                     or (_src is None
                                         and not node.params.get(_prov[1])))
                    if not _overwritable:
                        changes.append(
                            f"[rule] SUPPRESSED RULE[{rule.unit_type}→"
                            f"{rule.downstream_type or '*'}/{rule.error_code}]: "
                            f"{node.tag}.{rule.param} {current} "
                            f"({_src or 'desc-specified'}, would have set "
                            f"{target_val:.2f})")
                        continue

                old = node.params.get(rule.param, "?")
                node.params[rule.param] = target_val
                # Keep provenance truthful: the value is now a learned median, not
                # whatever it was tagged before (fixes the self-certifying tag bug).
                if _prov is not None:
                    node.params[_prov[0]] = "rule"
                changes.append(
                    f"RULE[{rule.unit_type}→{rule.downstream_type or '*'}/"
                    f"{rule.error_code}]: {node.tag}.{rule.param} "
                    f"{old}→{target_val:.2f} (synthesized from {rule.count} observations)"
                )

        return g, changes

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            f"{k[0]}|{k[1]}|{k[2]}": p.to_dict()
            for k, p in self._patterns.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for v in data.values():
            p   = FailurePattern.from_dict(v)
            key = (p.unit_type, p.error_code, p.downstream_type or "")
            self._patterns[key] = p

    def summary(self) -> str:
        lines = [f"FailureRuleStore: {self.num_patterns()} patterns, "
                 f"{self.num_active()} active rules (threshold={RULE_THRESHOLD})"]
        for p in sorted(self._patterns.values(), key=lambda x: x.count, reverse=True):
            marker = "✓" if p.count >= RULE_THRESHOLD else " "
            med = p.median_fix()
            med_str = f"{med:.1f}" if med is not None else "?"
            lines.append(
                f"  [{marker}] {p.unit_type}→{p.downstream_type or '*'} "
                f"/{p.error_code}: {p.param}={med_str} "
                f"(n={p.count})"
            )
        return "\n".join(lines)


# ── Graph helpers ──────────────────────────────────────────────────────────────

def _outlet_unit_types(graph: Any, unit_tag: str) -> set[str]:
    types: set[str] = set()
    for s in graph.outlet_streams(unit_tag):
        dst_tag = graph.stream_dest(s.tag)
        if dst_tag:
            dst = graph.unit(dst_tag)
            if dst:
                types.add(dst.unit_type)
    return types
