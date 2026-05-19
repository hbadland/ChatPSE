"""
Multi-level validation for FlowsheetGraph.

Level 1 — Schema:   required fields, supported types, composition sums
Level 2 — Graph:    connectivity, port constraints, acyclicity, duplicate ports
Level 3 — Physics:  T/P ranges, phase consistency, thermodynamic feasibility,
                    mass balance (when flows are defined)

All checks are deterministic. No LLM calls.
Issues are typed via ir.types enums; the ValidationReport feeds directly
into DeterministicRepair for structured, targeted fixing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ir.graph import (
    FlowsheetGraph, PORT_SPECS, SUPPORTED_UNIT_TYPES,
    SeparatorNode, PumpNode, CompressorNode, ExpanderNode,
    HeaterNode, CoolerNode, SplitterNode,
)
from ir.types import ErrorType, RepairStrategy, ErrorSeverity, ErrorTarget, SimError


# ── ValidationIssue — wraps SimError with a validation level ──────────────────

@dataclass
class ValidationIssue:
    level:  str       # "SCHEMA" | "GRAPH" | "PHYSICS"
    error:  SimError

    @property
    def severity(self) -> str:
        return self.error.severity.value

    @property
    def code(self) -> str:
        return self.error.error_type.value

    @property
    def location(self) -> str:
        return str(self.error.target)

    def __str__(self) -> str:
        return f"[{self.level}/{self.severity}] {self.error}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.error.severity == ErrorSeverity.CRITICAL for i in self.issues)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.error.severity == ErrorSeverity.CRITICAL]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.error.severity == ErrorSeverity.WARNING]

    def by_level(self, level: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == level]

    def sim_errors(self) -> list[SimError]:
        """Return the underlying SimError objects for direct use by RepairAgent."""
        return [i.error for i in self.issues]

    def summary(self) -> str:
        if not self.issues:
            return "ValidationReport: VALID (no issues)"
        lines = [f"ValidationReport: {'VALID' if self.valid else 'INVALID'} "
                 f"({len(self.errors())} errors, {len(self.warnings())} warnings)"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _err(level: str, etype: ErrorType, target: ErrorTarget, evidence: str,
         strategy: RepairStrategy,
         severity: ErrorSeverity = ErrorSeverity.CRITICAL) -> ValidationIssue:
    return ValidationIssue(level, SimError(etype, target, evidence, strategy, severity))


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
    G = ErrorTarget.global_

    if not graph.compounds:
        issues.append(_err("SCHEMA", ErrorType.MISSING_PARAM, G(),
                           "compounds list is empty", RepairStrategy.HUMAN))

    if not graph.property_package:
        issues.append(_err("SCHEMA", ErrorType.MISSING_PARAM, G(),
                           "property_package is not set", RepairStrategy.THERMO_SWITCH))

    if not graph.units():
        issues.append(_err("SCHEMA", ErrorType.INVALID_TOPOLOGY, G(),
                           "flowsheet has no unit operations", RepairStrategy.HUMAN))

    if not graph.streams():
        issues.append(_err("SCHEMA", ErrorType.INVALID_TOPOLOGY, G(),
                           "flowsheet has no streams", RepairStrategy.HUMAN))

    for node in graph.units():
        if node.unit_type not in SUPPORTED_UNIT_TYPES:
            issues.append(_err("SCHEMA", ErrorType.INVALID_TOPOLOGY,
                               ErrorTarget.unit(node.tag),
                               f"unknown type '{node.unit_type}'", RepairStrategy.HUMAN))

    for stream in graph.streams():
        if not stream.tag:
            issues.append(_err("SCHEMA", ErrorType.MISSING_PARAM, G(),
                               "stream has no tag", RepairStrategy.HUMAN))
            continue
        comp = stream.composition
        if comp:
            total = sum(comp.values())
            if abs(total - 1.0) > 0.01:
                issues.append(_err("SCHEMA", ErrorType.UNPHYSICAL_VALUES,
                                   ErrorTarget.stream(stream.tag, "composition"),
                                   f"mole fractions sum to {total:.4f}",
                                   RepairStrategy.DEFAULT_FILL))
            for name, frac in comp.items():
                if name not in graph.compounds:
                    issues.append(_err("SCHEMA", ErrorType.MISSING_PARAM,
                                       ErrorTarget.stream(stream.tag, "composition"),
                                       f"compound '{name}' not in compounds list",
                                       RepairStrategy.HUMAN))
                if not isinstance(frac, (int, float)) or frac < 0.0:
                    issues.append(_err("SCHEMA", ErrorType.UNPHYSICAL_VALUES,
                                       ErrorTarget.stream(stream.tag, "composition"),
                                       f"'{name}' fraction {frac} invalid",
                                       RepairStrategy.DEFAULT_FILL))
        if stream.flow is not None and stream.flow <= 0:
            issues.append(_err("SCHEMA", ErrorType.UNPHYSICAL_VALUES,
                               ErrorTarget.stream(stream.tag, "flow"),
                               f"flow={stream.flow} mol/s must be > 0",
                               RepairStrategy.DEFAULT_FILL))
    return issues


# ── Level 2: Graph ─────────────────────────────────────────────────────────────

def _graph_validate(graph: FlowsheetGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    G = ErrorTarget.global_

    if not graph.is_acyclic():
        issues.append(_err("GRAPH", ErrorType.INVALID_TOPOLOGY, G(),
                           "connection graph contains a cycle", RepairStrategy.HUMAN))

    for stream in graph.streams():
        src = graph.stream_source(stream.tag)
        dst = graph.stream_dest(stream.tag)
        if src is None and dst is None:
            issues.append(_err("GRAPH", ErrorType.INVALID_TOPOLOGY,
                               ErrorTarget.stream(stream.tag),
                               "stream not connected to any unit",
                               RepairStrategy.TOPOLOGY_FIX))

    for node in graph.units():
        specs  = PORT_SPECS.get(node.unit_type, [])
        req_in  = len([s for s in specs if s.direction == "inlet"  and s.required])
        req_out = len([s for s in specs if s.direction == "outlet" and s.required])
        max_out = len([s for s in specs if s.direction == "outlet"])

        actual_in  = len(graph.inlet_streams(node.tag))
        actual_out = len(graph.outlet_streams(node.tag))

        if actual_in < req_in:
            issues.append(_err("GRAPH", ErrorType.INVALID_TOPOLOGY,
                               ErrorTarget.unit(node.tag),
                               f"{node.unit_type} needs ≥{req_in} inlet(s), has {actual_in}",
                               RepairStrategy.TOPOLOGY_FIX))
        if actual_out < req_out:
            issues.append(_err("GRAPH", ErrorType.INVALID_TOPOLOGY,
                               ErrorTarget.unit(node.tag),
                               f"{node.unit_type} needs ≥{req_out} outlet(s), has {actual_out}",
                               RepairStrategy.TOPOLOGY_FIX))
        if isinstance(node, SeparatorNode) and actual_out != 2:
            issues.append(_err("GRAPH", ErrorType.INVALID_TOPOLOGY,
                               ErrorTarget.unit(node.tag),
                               f"Vessel must have exactly 2 outlets, has {actual_out}",
                               RepairStrategy.TOPOLOGY_FIX))

    # Duplicate src_port
    for node in graph.units():
        seen: set[int] = set()
        for s in graph.outlet_streams(node.tag):
            if s.src_port in seen:
                issues.append(_err("GRAPH", ErrorType.INVALID_TOPOLOGY,
                                   ErrorTarget.unit(node.tag, "src_port"),
                                   f"two outlets both use src_port={s.src_port}",
                                   RepairStrategy.PORT_REPAIR))
            seen.add(s.src_port)

    return issues


# ── Level 3: Physics ───────────────────────────────────────────────────────────

_T_MIN_K  = 50.0
_T_MAX_K  = 2000.0
_P_MIN_PA = 100.0
_P_MAX_PA = 1e8
_FLOW_TOL = 0.05   # 5% relative tolerance for mass balance checks

def _physics_validate(graph: FlowsheetGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # ── 3a: T/P range checks ──────────────────────────────────────────────────
    for s in graph.streams():
        if s.T is not None and not (_T_MIN_K < s.T < _T_MAX_K):
            issues.append(_err("PHYSICS", ErrorType.UNPHYSICAL_VALUES,
                               ErrorTarget.stream(s.tag, "T"),
                               f"T={s.T} K out of range — use Kelvin (25°C=298.15 K)",
                               RepairStrategy.UNIT_CONVERSION))
        if s.P is not None and not (_P_MIN_PA < s.P < _P_MAX_PA):
            issues.append(_err("PHYSICS", ErrorType.UNPHYSICAL_VALUES,
                               ErrorTarget.stream(s.tag, "P"),
                               f"P={s.P} Pa out of range — use Pascals (1 atm=101325 Pa)",
                               RepairStrategy.UNIT_CONVERSION))

    for node in graph.units():
        t_out = node.params.get("T_out")
        if t_out is not None and not (_T_MIN_K < float(t_out) < _T_MAX_K):
            issues.append(_err("PHYSICS", ErrorType.UNPHYSICAL_VALUES,
                               ErrorTarget.unit(node.tag, "T_out"),
                               f"T_out={t_out} K out of range",
                               RepairStrategy.UNIT_CONVERSION))
        p_out = node.params.get("P_out")
        if p_out is not None and not (_P_MIN_PA < float(p_out) < _P_MAX_PA):
            issues.append(_err("PHYSICS", ErrorType.UNPHYSICAL_VALUES,
                               ErrorTarget.unit(node.tag, "P_out"),
                               f"P_out={p_out} Pa out of range",
                               RepairStrategy.UNIT_CONVERSION))
        eff = node.params.get("efficiency")
        if eff is not None and not (0.0 < float(eff) <= 1.0):
            issues.append(_err("PHYSICS", ErrorType.UNPHYSICAL_VALUES,
                               ErrorTarget.unit(node.tag, "efficiency"),
                               f"efficiency={eff} not in (0, 1]",
                               RepairStrategy.DEFAULT_FILL))

    # ── 3b: Thermodynamic feasibility ─────────────────────────────────────────
    for node in graph.units():
        inlets = graph.inlet_streams(node.tag)
        feed_T = next((s.T for s in inlets if s.T is not None), None)
        feed_P = next((s.P for s in inlets if s.P is not None), None)

        # Cooler: T_out must be below feed T
        if isinstance(node, CoolerNode):
            t_out = node.params.get("T_out")
            if t_out is not None and feed_T is not None and float(t_out) >= feed_T:
                issues.append(_err("PHYSICS", ErrorType.INVALID_UNIT_CONFIG,
                                   ErrorTarget.unit(node.tag, "T_out"),
                                   f"Cooler T_out={t_out} K ≥ feed T={feed_T} K",
                                   RepairStrategy.CONDITION_FIX))

        # Heater: T_out must be above feed T
        if isinstance(node, HeaterNode):
            t_out = node.params.get("T_out")
            if t_out is not None and feed_T is not None and float(t_out) <= feed_T:
                issues.append(_err("PHYSICS", ErrorType.INVALID_UNIT_CONFIG,
                                   ErrorTarget.unit(node.tag, "T_out"),
                                   f"Heater T_out={t_out} K ≤ feed T={feed_T} K",
                                   RepairStrategy.CONDITION_FIX))

        # Pump: P_out must be above feed P
        if isinstance(node, PumpNode):
            p_out = node.params.get("P_out")
            if p_out is not None and feed_P is not None and float(p_out) <= feed_P:
                issues.append(_err("PHYSICS", ErrorType.INVALID_UNIT_CONFIG,
                                   ErrorTarget.unit(node.tag, "P_out"),
                                   f"Pump P_out={p_out} Pa ≤ feed P={feed_P} Pa",
                                   RepairStrategy.CONDITION_FIX))

        # Compressor/Expander: P_out in right direction
        if isinstance(node, CompressorNode):
            p_out = node.params.get("P_out")
            if p_out is not None and feed_P is not None and float(p_out) <= feed_P:
                issues.append(_err("PHYSICS", ErrorType.INVALID_UNIT_CONFIG,
                                   ErrorTarget.unit(node.tag, "P_out"),
                                   f"Compressor P_out={p_out} Pa ≤ feed P={feed_P} Pa",
                                   RepairStrategy.CONDITION_FIX))

        if isinstance(node, ExpanderNode):
            p_out = node.params.get("P_out")
            if p_out is not None and feed_P is not None and float(p_out) >= feed_P:
                issues.append(_err("PHYSICS", ErrorType.INVALID_UNIT_CONFIG,
                                   ErrorTarget.unit(node.tag, "P_out"),
                                   f"Expander P_out={p_out} Pa ≥ feed P={feed_P} Pa",
                                   RepairStrategy.CONDITION_FIX))

        # Vessel: warn if feed T is likely below bubble point
        if isinstance(node, SeparatorNode):
            if inlets and all(s.T is None for s in inlets):
                issues.append(_err("PHYSICS", ErrorType.INVALID_UNIT_CONFIG,
                                   ErrorTarget.unit(node.tag),
                                   "Vessel has no feed T — may produce zero vapour",
                                   RepairStrategy.CONDITION_FIX,
                                   ErrorSeverity.WARNING))

    # ── 3c: Phase consistency ─────────────────────────────────────────────────
    for node in graph.units():
        # Pump requires liquid inlet
        if isinstance(node, PumpNode):
            for s in graph.inlet_streams(node.tag):
                if s.phase == "vapour":
                    issues.append(_err("PHYSICS", ErrorType.PHASE_MISMATCH,
                                       ErrorTarget.unit(node.tag),
                                       f"Pump inlet stream '{s.tag}' is vapour (liquid required)",
                                       RepairStrategy.TOPOLOGY_FIX))

        # Compressor/Expander require vapour inlet
        if isinstance(node, (CompressorNode, ExpanderNode)):
            for s in graph.inlet_streams(node.tag):
                if s.phase == "liquid":
                    issues.append(_err("PHYSICS", ErrorType.PHASE_MISMATCH,
                                       ErrorTarget.unit(node.tag),
                                       f"{node.unit_type} inlet stream '{s.tag}' is liquid (vapour required)",
                                       RepairStrategy.TOPOLOGY_FIX))

        # Vessel: vapour outlet must be on port 0, liquid on port 1
        if isinstance(node, SeparatorNode):
            for s in graph.outlet_streams(node.tag):
                if s.src_port == 0 and s.phase == "liquid":
                    issues.append(_err("PHYSICS", ErrorType.INVALID_TOPOLOGY,
                                       ErrorTarget.unit(node.tag, "src_port"),
                                       f"Stream '{s.tag}' phase=liquid on port 0 (should be vapour)",
                                       RepairStrategy.PORT_REPAIR))
                if s.src_port == 1 and s.phase == "vapour":
                    issues.append(_err("PHYSICS", ErrorType.INVALID_TOPOLOGY,
                                       ErrorTarget.unit(node.tag, "src_port"),
                                       f"Stream '{s.tag}' phase=vapour on port 1 (should be liquid)",
                                       RepairStrategy.PORT_REPAIR))

    # ── 3d: Mass balance (when flows are defined) ─────────────────────────────
    for node in graph.units():
        # Skip splitters (flow split is intentional) and separators (multi-phase)
        if isinstance(node, (SplitterNode, SeparatorNode)):
            continue
        inlets  = graph.inlet_streams(node.tag)
        outlets = graph.outlet_streams(node.tag)
        in_flows  = [s.flow for s in inlets  if s.flow is not None]
        out_flows = [s.flow for s in outlets if s.flow is not None]
        if not in_flows or not out_flows:
            continue
        total_in  = sum(in_flows)
        total_out = sum(out_flows)
        if total_in > 0 and abs(total_in - total_out) / total_in > _FLOW_TOL:
            issues.append(_err("PHYSICS", ErrorType.MASS_BALANCE,
                               ErrorTarget.unit(node.tag),
                               f"inlet flow={total_in:.3f} ≠ outlet flow={total_out:.3f} mol/s "
                               f"({abs(total_in-total_out)/total_in:.1%} imbalance)",
                               RepairStrategy.TOPOLOGY_FIX,
                               ErrorSeverity.WARNING))

    return issues
