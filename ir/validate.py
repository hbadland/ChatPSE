"""
Multi-level validation for FlowsheetGraph.

Level 1 — Schema:   required fields, supported types, composition sums
Level 2 — Graph:    connectivity, port constraints, acyclicity, duplicate ports
Level 3 — Physics:  T/P physical ranges, flow positivity, phase consistency

All checks are deterministic.  No LLM calls.
Returns a ValidationReport; agents check report.valid before proceeding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ir.graph import FlowsheetGraph, PORT_SPECS, SUPPORTED_UNIT_TYPES


# ── Issue + Report ─────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    level:    str   # "SCHEMA" | "GRAPH" | "PHYSICS"
    code:     str
    location: str   # unit/stream tag, or "global"
    message:  str
    severity: str   # "ERROR" | "WARNING"

    def __str__(self) -> str:
        return f"[{self.level}/{self.severity}] {self.code} @ {self.location}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "ERROR" for i in self.issues)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def by_level(self, level: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == level]

    def summary(self) -> str:
        if not self.issues:
            return "ValidationReport: VALID (no issues)"
        lines = [f"ValidationReport: {'VALID' if self.valid else 'INVALID'} "
                 f"({len(self.errors())} errors, {len(self.warnings())} warnings)"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# ── Public entry point ─────────────────────────────────────────────────────────

def validate(graph: FlowsheetGraph) -> ValidationReport:
    issues: list[ValidationIssue] = []
    issues += _schema_validate(graph)
    issues += _graph_validate(graph)
    issues += _physics_validate(graph)
    return ValidationReport(issues=issues)


# ── Level 1: Schema ────────────────────────────────────────────────────────────

def _schema_validate(graph: FlowsheetGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not graph.compounds:
        issues.append(ValidationIssue(
            "SCHEMA", "MISSING_COMPOUNDS", "global",
            "compounds list is empty", "ERROR"))

    if not graph.property_package:
        issues.append(ValidationIssue(
            "SCHEMA", "MISSING_PROPERTY_PACKAGE", "global",
            "property_package is not set", "ERROR"))

    if not graph.units():
        issues.append(ValidationIssue(
            "SCHEMA", "NO_UNITS", "global",
            "flowsheet has no unit operations", "ERROR"))

    if not graph.streams():
        issues.append(ValidationIssue(
            "SCHEMA", "NO_STREAMS", "global",
            "flowsheet has no streams", "ERROR"))

    for node in graph.units():
        if node.unit_type not in SUPPORTED_UNIT_TYPES:
            issues.append(ValidationIssue(
                "SCHEMA", "UNSUPPORTED_UNIT_TYPE", node.tag,
                f"unknown type '{node.unit_type}'; "
                f"supported: {sorted(SUPPORTED_UNIT_TYPES)}", "ERROR"))

    for stream in graph.streams():
        if not stream.tag:
            issues.append(ValidationIssue(
                "SCHEMA", "MISSING_STREAM_TAG", "global",
                "a stream has no tag", "ERROR"))
            continue

        comp = stream.composition
        if comp:
            total = sum(comp.values())
            if abs(total - 1.0) > 0.01:
                issues.append(ValidationIssue(
                    "SCHEMA", "COMPOSITION_SUM", stream.tag,
                    f"mole fractions sum to {total:.4f}, not 1.0", "ERROR"))
            for name, frac in comp.items():
                if name not in graph.compounds:
                    issues.append(ValidationIssue(
                        "SCHEMA", "UNKNOWN_COMPOUND", stream.tag,
                        f"compound '{name}' not in compounds list", "ERROR"))
                if not isinstance(frac, (int, float)) or frac < 0.0:
                    issues.append(ValidationIssue(
                        "SCHEMA", "INVALID_FRACTION", stream.tag,
                        f"'{name}' fraction {frac} is not a non-negative number", "ERROR"))

        if stream.flow is not None and stream.flow <= 0:
            issues.append(ValidationIssue(
                "SCHEMA", "NON_POSITIVE_FLOW", stream.tag,
                f"flow={stream.flow} mol/s must be > 0", "ERROR"))

    return issues


# ── Level 2: Graph ─────────────────────────────────────────────────────────────

def _graph_validate(graph: FlowsheetGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # Acyclicity
    if not graph.is_acyclic():
        issues.append(ValidationIssue(
            "GRAPH", "CYCLE", "global",
            "connection graph contains a cycle (recycle loops not supported)", "ERROR"))

    unit_tags   = graph.unit_tags()
    stream_tags = graph.stream_tags()

    # Every stream must connect to at least one unit
    for stream in graph.streams():
        src = graph.stream_source(stream.tag)
        dst = graph.stream_dest(stream.tag)
        if src is None and dst is None:
            issues.append(ValidationIssue(
                "GRAPH", "ISOLATED_STREAM", stream.tag,
                "stream is not connected to any unit", "ERROR"))

    # Port constraint enforcement per unit
    for node in graph.units():
        specs = PORT_SPECS.get(node.unit_type, [])
        if not specs:
            continue

        required_inlets  = [s for s in specs if s.direction == "inlet"  and s.required]
        required_outlets = [s for s in specs if s.direction == "outlet" and s.required]
        actual_inlets    = graph.inlet_streams(node.tag)
        actual_outlets   = graph.outlet_streams(node.tag)

        if len(actual_inlets) < len(required_inlets):
            issues.append(ValidationIssue(
                "GRAPH", "MISSING_INLET", node.tag,
                f"{node.unit_type} needs ≥{len(required_inlets)} inlet(s), "
                f"has {len(actual_inlets)}", "ERROR"))

        if len(actual_outlets) < len(required_outlets):
            issues.append(ValidationIssue(
                "GRAPH", "MISSING_OUTLET", node.tag,
                f"{node.unit_type} needs ≥{len(required_outlets)} outlet(s), "
                f"has {len(actual_outlets)}", "ERROR"))

        # Vessel must have exactly 2 outlets (vapour + liquid)
        if node.unit_type == "Vessel" and len(actual_outlets) != 2:
            issues.append(ValidationIssue(
                "GRAPH", "VESSEL_OUTLET_COUNT", node.tag,
                f"Vessel must have exactly 2 outlets (vapour port=0, liquid port=1), "
                f"has {len(actual_outlets)}", "ERROR"))

    # Duplicate src_port: two outlet streams on the same port of the same unit
    for node in graph.units():
        seen_ports: set[int] = set()
        for stream in graph.outlet_streams(node.tag):
            port = stream.src_port
            if port in seen_ports:
                issues.append(ValidationIssue(
                    "GRAPH", "DUPLICATE_SRC_PORT", node.tag,
                    f"two outlet streams both use src_port={port}", "ERROR"))
            seen_ports.add(port)

    return issues


# ── Level 3: Physics ───────────────────────────────────────────────────────────

_T_MIN_K  = 50.0
_T_MAX_K  = 2000.0
_P_MIN_PA = 100.0
_P_MAX_PA = 1e8

def _physics_validate(graph: FlowsheetGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for stream in graph.streams():
        if stream.T is not None and not (_T_MIN_K < stream.T < _T_MAX_K):
            issues.append(ValidationIssue(
                "PHYSICS", "UNPHYSICAL_T", stream.tag,
                f"T={stream.T} K outside valid range ({_T_MIN_K}–{_T_MAX_K} K). "
                "Ensure all temperatures are in Kelvin (25°C = 298.15 K).", "ERROR"))

        if stream.P is not None and not (_P_MIN_PA < stream.P < _P_MAX_PA):
            issues.append(ValidationIssue(
                "PHYSICS", "UNPHYSICAL_P", stream.tag,
                f"P={stream.P} Pa outside valid range ({_P_MIN_PA}–{_P_MAX_PA:.0e} Pa). "
                "Ensure all pressures are in Pascals (1 atm = 101325 Pa).", "ERROR"))

    for node in graph.units():
        t_out = node.params.get("T_out")
        if t_out is not None and not (_T_MIN_K < t_out < _T_MAX_K):
            issues.append(ValidationIssue(
                "PHYSICS", "UNPHYSICAL_T_OUT", node.tag,
                f"T_out={t_out} K outside valid range. "
                "Ensure T_out is in Kelvin.", "ERROR"))

        p_out = node.params.get("P_out")
        if p_out is not None and not (_P_MIN_PA < p_out < _P_MAX_PA):
            issues.append(ValidationIssue(
                "PHYSICS", "UNPHYSICAL_P_OUT", node.tag,
                f"P_out={p_out} Pa outside valid range.", "ERROR"))

        eff = node.params.get("efficiency")
        if eff is not None and not (0.0 < eff <= 1.0):
            issues.append(ValidationIssue(
                "PHYSICS", "INVALID_EFFICIENCY", node.tag,
                f"efficiency={eff} must be in (0, 1]", "ERROR"))

    # Warn on Vessel with no feed temperature defined — may cause ZERO_OUTLET
    for node in graph.units():
        if node.unit_type == "Vessel":
            inlets = graph.inlet_streams(node.tag)
            if inlets and all(s.T is None for s in inlets):
                issues.append(ValidationIssue(
                    "PHYSICS", "VESSEL_NO_FEED_T", node.tag,
                    "Vessel has no feed stream with T defined — "
                    "may produce zero vapour if feed is sub-bubble-point", "WARNING"))

    return issues
