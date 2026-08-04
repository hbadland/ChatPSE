"""
Intermediate Representation (IR) for flowsheets.

NodeIR subclasses encode their own port constraints and required parameters.
FlowsheetGraph enforces port constraints at add_stream() time — invalid states
are rejected during construction, not just during validation.

Node hierarchy:
  NodeIR (base)
    ├── ConditioningNode   — Heater, Cooler
    ├── SeparatorNode      — Vessel
    ├── MixerNode          — Mixer
    ├── SplitterNode       — Splitter
    └── PressureChangerNode — Pump, Compressor, Expander
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, ClassVar


# ── Value provenance ─────────────────────────────────────────────────────────
class Source(IntEnum):
    """Provenance authority of a parameter value; higher = more authoritative.

    A priority-guarded merge (`NodeIR.set_param`) refuses to overwrite a value with
    one of equal-or-lower authority (no silent downgrade). A sanctioned physical
    correction (`NodeIR.correct_param`) may override regardless of authority, but
    re-tags the provenance so the record stays truthful — the two channels are the
    ONLY ways a tracked value changes, and both are caller-logged.
    """
    MISSING   = -1  # recorded gap: no value/coverage exists (lowest authority; a
                    # marker that the data is absent, NOT a usable value)
    DEFAULT   = 0   # bare default fill
    FALLBACK  = 1   # pooled description-list guess (no structured attribution)
    COMPUTED  = 2   # deterministic estimator / physical correction (bubble point …)
    RULE      = 3   # cross-run learned median rule
    INHERITED = 4   # propagated from a real upstream/feed value
    EXTRACTED = 5   # structured per-unit attribution from the description
    SPECIFIED = 6   # explicit setpoint stated in the description


# Backward-compatible string tags <-> Source (params dict keeps the legacy strings).
_SOURCE_TO_STR: dict = {
    Source.MISSING: "missing",
    Source.DEFAULT: "default_fallback", Source.FALLBACK: "fallback",
    Source.COMPUTED: "computed", Source.RULE: "rule", Source.INHERITED: "inherited",
    Source.EXTRACTED: "extracted", Source.SPECIFIED: "specified",
}
_STR_TO_SOURCE: dict = {v: k for k, v in _SOURCE_TO_STR.items()}

# Which source-tag field and description sentinel guard each tracked parameter.
_PARAM_SOURCE_FIELD: dict = {
    "T_out": "_temperature_source", "temperature_K": "_temperature_source",
    "P_out": "_pressure_source", "pressure_Pa": "_pressure_source",
}
_PARAM_SENTINEL: dict = {
    "T_out": "_desc_T_out", "temperature_K": "_desc_T_out",
    "P_out": "_desc_P_out", "pressure_Pa": "_desc_P_out",
}

import networkx as nx


# ── Exceptions ────────────────────────────────────────────────────────────────

class TopologyError(ValueError):
    """Raised when the non-recycle subgraph contains a cycle."""


# ── Port specification ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortSpec:
    port_id:   int
    direction: str   # "inlet" | "outlet"
    phase:     str   # "any" | "vapour" | "liquid" | "mixed"
    required:  bool = True


# ── Typed node base class ──────────────────────────────────────────────────────

class NodeIR:
    """
    Base class for all unit operation nodes.

    Subclasses declare PORT_SPECS and REQUIRED_PARAMS as class attributes.
    Construction-time validation is called by FlowsheetGraph.add_unit().
    """
    PORT_SPECS:      ClassVar[list[PortSpec]] = []
    REQUIRED_PARAMS: ClassVar[list[str]]      = []
    UNIT_TYPE:       ClassVar[str]            = ""

    def __init__(
        self,
        tag:              str,
        params:           Optional[dict] = None,
        property_package: Optional[str]  = None,
        metadata:         Optional[dict] = None,
    ) -> None:
        if not tag:
            raise ValueError("NodeIR tag must be non-empty")
        self.tag              = tag
        self.unit_type        = self.__class__.UNIT_TYPE
        self.params           = params or {}
        self.property_package = property_package
        self.metadata         = metadata or {}

    # ── Provenance-tracked parameter access (Phase 1: additive, not yet wired) ──
    def _param_source(self, name: str) -> Optional["Source"]:
        """Current provenance of a tracked param, or None if untracked/untagged.
        A description sentinel with no explicit tag is read as SPECIFIED — this
        reproduces the legacy overwrite guard exactly."""
        field_name = _PARAM_SOURCE_FIELD.get(name)
        if field_name is None:
            return None
        tag = self.params.get(field_name)
        if tag in _STR_TO_SOURCE:
            return _STR_TO_SOURCE[tag]
        sentinel = _PARAM_SENTINEL.get(name)
        if sentinel and self.params.get(sentinel):
            return Source.SPECIFIED
        return None

    def set_param(self, name: str, value, source: "Source") -> bool:
        """Priority-guarded merge. Writes value + provenance tag IFF `source` STRICTLY
        outranks the incumbent (or there is none). Returns True if written, False if
        rejected as an equal-or-lower-authority downgrade. Never silent: the caller
        logs both outcomes. Params outside the provenance map are written
        unconditionally (no provenance to protect)."""
        field_name = _PARAM_SOURCE_FIELD.get(name)
        if field_name is None:
            self.params[name] = value
            return True
        cur = self._param_source(name)
        if cur is not None and source <= cur:
            return False
        self.params[name] = value
        self.params[field_name] = _SOURCE_TO_STR[source]
        return True

    def correct_param(self, name: str, value, source: "Source" = Source.COMPUTED) -> None:
        """Sanctioned physical correction: overrides regardless of incumbent authority
        (a physically-implausible value must be fixable even if user-specified) but
        RE-TAGS provenance to `source` and clears any description sentinel, so the
        record stays truthful — the value is no longer what it was. Distinct from a
        silent overwrite: the caller records the physical reason in the change log."""
        self.params[name] = value
        field_name = _PARAM_SOURCE_FIELD.get(name)
        if field_name is not None:
            self.params[field_name] = _SOURCE_TO_STR[source]
        sentinel = _PARAM_SENTINEL.get(name)
        if sentinel:
            self.params.pop(sentinel, None)

    # Called by FlowsheetGraph.add_unit() — subclasses may override
    def validate_construction(self) -> list[str]:
        return []

    def max_inlets(self) -> int:
        return len([s for s in self.PORT_SPECS if s.direction == "inlet"])

    def max_outlets(self) -> int:
        return len([s for s in self.PORT_SPECS if s.direction == "outlet"])

    def required_inlets(self) -> int:
        return len([s for s in self.PORT_SPECS
                    if s.direction == "inlet" and s.required])

    def required_outlets(self) -> int:
        return len([s for s in self.PORT_SPECS
                    if s.direction == "outlet" and s.required])

    def outlet_phase(self, port_id: int) -> str:
        for spec in self.PORT_SPECS:
            if spec.direction == "outlet" and spec.port_id == port_id:
                return spec.phase
        return "any"

    def copy(self) -> "NodeIR":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tag={self.tag!r}, params={self.params})"


# ── Concrete typed nodes ───────────────────────────────────────────────────────

class HeaterNode(NodeIR):
    UNIT_TYPE = "Heater"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "any", required=True),
        PortSpec(0, "outlet", "any", required=True),
    ]
    REQUIRED_PARAMS = ["T_out"]

    def validate_construction(self) -> list[str]:
        errors = []
        t = self.params.get("T_out")
        if t is not None and not (50 < float(t) < 2000):
            errors.append(f"HeaterNode {self.tag}: T_out={t} K out of range (50–2000 K)")
        return errors


class CoolerNode(NodeIR):
    UNIT_TYPE = "Cooler"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "any", required=True),
        PortSpec(0, "outlet", "any", required=True),
    ]
    REQUIRED_PARAMS = ["T_out"]

    def validate_construction(self) -> list[str]:
        errors = []
        t = self.params.get("T_out")
        if t is not None and not (50 < float(t) < 2000):
            errors.append(f"CoolerNode {self.tag}: T_out={t} K out of range")
        return errors


class SeparatorNode(NodeIR):
    """
    Vessel / flash separator. Enforces exactly 2 outlets (vapour port=0, liquid port=1).
    """
    UNIT_TYPE = "Vessel"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "mixed",  required=True),
        PortSpec(0, "outlet", "vapour", required=True),
        PortSpec(1, "outlet", "liquid", required=True),
    ]
    REQUIRED_PARAMS = []

    def validate_construction(self) -> list[str]:
        return []  # outlet count enforced by add_stream


class MixerNode(NodeIR):
    UNIT_TYPE = "Mixer"
    PORT_SPECS = [
        PortSpec(0, "inlet", "any", required=True),
        PortSpec(1, "inlet", "any", required=False),  # ≥1 inlet, ≤N
        PortSpec(0, "outlet", "any", required=True),
    ]
    REQUIRED_PARAMS = []

    def max_inlets(self) -> int:
        return 16  # DWSIM supports many inlets


class SplitterNode(NodeIR):
    UNIT_TYPE = "Splitter"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "any", required=True),
        PortSpec(0, "outlet", "any", required=True),
        PortSpec(1, "outlet", "any", required=True),
    ]
    REQUIRED_PARAMS = ["split_fractions"]

    def validate_construction(self) -> list[str]:
        errors = []
        sf = self.params.get("split_fractions", {})
        if sf:
            total = sum(sf.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"SplitterNode {self.tag}: split_fractions sum to {total:.4f}, not 1.0")
        return errors


class PressureChangerNode(NodeIR):
    """Base for Pump, Compressor, Expander."""
    REQUIRED_PARAMS = ["P_out"]

    def validate_construction(self) -> list[str]:
        errors = []
        p = self.params.get("P_out")
        if p is not None and not (100 < float(p) < 1e8):
            errors.append(
                f"{self.unit_type} {self.tag}: P_out={p} Pa out of range (100–1e8 Pa)")
        return errors


class PumpNode(PressureChangerNode):
    UNIT_TYPE = "Pump"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "liquid", required=True),
        PortSpec(0, "outlet", "liquid", required=True),
    ]


class CompressorNode(PressureChangerNode):
    UNIT_TYPE = "Compressor"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "vapour", required=True),
        PortSpec(0, "outlet", "vapour", required=True),
    ]


class ExpanderNode(PressureChangerNode):
    UNIT_TYPE = "Expander"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "vapour", required=True),
        PortSpec(0, "outlet", "vapour", required=True),
    ]


class ConversionReactorNode(NodeIR):
    """
    Stoichiometric conversion reactor.  Converts a specified fraction of the
    limiting reactant.  Requires: temperature_K, pressure_Pa, conversion (0–1),
    reaction (stoichiometry string, e.g. "CH4 + H2O → CO + 3H2").
    """
    UNIT_TYPE = "ConversionReactor"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "any", required=True),
        PortSpec(0, "outlet", "any", required=True),
    ]
    REQUIRED_PARAMS = ["temperature_K", "pressure_Pa", "conversion", "reaction"]

    def validate_construction(self) -> list[str]:
        errors = []
        t = self.params.get("temperature_K")
        if t is not None and not (50 < float(t) < 3000):
            errors.append(
                f"ConversionReactorNode {self.tag}: temperature_K={t} out of range (50–3000 K)")
        p = self.params.get("pressure_Pa")
        if p is not None and not (100 < float(p) < 1e8):
            errors.append(
                f"ConversionReactorNode {self.tag}: pressure_Pa={p} out of range")
        conv = self.params.get("conversion")
        if conv is not None and not (0.0 <= float(conv) <= 1.0):
            errors.append(
                f"ConversionReactorNode {self.tag}: conversion={conv} out of range (0–1)")
        return errors


class ColumnNode(NodeIR):
    """
    Shortcut (Fenske-Underwood-Gilliland) distillation column. One material feed
    (inlet port 0) → distillate (outlet port 0) + bottoms (outlet port 1). The
    reboiler energy stream is attached at DWSIM-mapping time (input port 1), like
    the reactor's energy stream — it is NOT an IR port.

    Params (probe-validated against DWSIM ShortcutColumn):
      light_key, heavy_key                 — component names (the split keys)
      light_key_frac_bottoms               — LK mole fraction allowed in bottoms
      heavy_key_frac_distillate            — HK mole fraction allowed in distillate
      reflux_ratio                         — actual reflux as multiple of Rmin
      condenser_pressure_Pa, boiler_pressure_Pa
    """
    UNIT_TYPE = "Column"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "any", required=True),
        PortSpec(0, "outlet", "any", required=True),   # distillate
        PortSpec(1, "outlet", "any", required=True),   # bottoms
    ]
    REQUIRED_PARAMS = [
        "light_key", "heavy_key",
        "light_key_frac_bottoms", "heavy_key_frac_distillate",
        "reflux_ratio", "condenser_pressure_Pa", "boiler_pressure_Pa",
    ]

    def validate_construction(self) -> list[str]:
        errors: list[str] = []
        for k in ("light_key", "heavy_key"):
            v = self.params.get(k)
            if v is not None and not str(v).strip():
                errors.append(f"ColumnNode {self.tag}: {k} is empty")
        for k in ("light_key_frac_bottoms", "heavy_key_frac_distillate"):
            v = self.params.get(k)
            if v is not None and not (0.0 < float(v) < 1.0):
                errors.append(
                    f"ColumnNode {self.tag}: {k}={v} out of range (0–1 exclusive)")
        r = self.params.get("reflux_ratio")
        if r is not None and float(r) <= 0.0:
            errors.append(f"ColumnNode {self.tag}: reflux_ratio={r} must be > 0")
        for k in ("condenser_pressure_Pa", "boiler_pressure_Pa"):
            v = self.params.get(k)
            if v is not None and not (100 < float(v) < 1e8):
                errors.append(
                    f"ColumnNode {self.tag}: {k}={v} Pa out of range (100–1e8)")
        return errors


class DecanterNode(NodeIR):
    """
    Liquid-liquid decanter — a Vessel variant that runs an LLE/VLLE flash
    (NestedLoopsSVLLE at DWSIM-mapping time) and exposes a THIRD outlet for the
    second liquid phase. Ports: inlet 0 (mixed); outlet 0 vapour (may be empty),
    outlet 1 liquid phase 1, outlet 2 liquid phase 2.
    """
    UNIT_TYPE = "Decanter"
    PORT_SPECS = [
        PortSpec(0, "inlet",  "mixed",  required=True),
        PortSpec(0, "outlet", "vapour", required=False),  # often zero-flow
        PortSpec(1, "outlet", "liquid", required=True),
        PortSpec(2, "outlet", "liquid", required=True),
    ]
    REQUIRED_PARAMS = []

    def validate_construction(self) -> list[str]:
        return []


# ── Registry: unit_type string → class ────────────────────────────────────────

NODE_REGISTRY: dict[str, type[NodeIR]] = {
    "Heater":             HeaterNode,
    "Cooler":             CoolerNode,
    "Vessel":             SeparatorNode,
    "Mixer":              MixerNode,
    "Splitter":           SplitterNode,
    "Pump":               PumpNode,
    "ConversionReactor":  ConversionReactorNode,
    "Compressor": CompressorNode,
    "Expander":   ExpanderNode,
    "Column":     ColumnNode,
    "Decanter":   DecanterNode,
}

SUPPORTED_UNIT_TYPES: frozenset[str] = frozenset(NODE_REGISTRY.keys())

# Backward-compatible PORT_SPECS dict (used by validate.py, normalise.py)
PORT_SPECS: dict[str, list[PortSpec]] = {
    name: cls.PORT_SPECS for name, cls in NODE_REGISTRY.items()
}


def make_node(unit_type: str, tag: str, params: Optional[dict] = None,
              property_package: Optional[str] = None,
              metadata: Optional[dict] = None) -> NodeIR:
    """Factory: returns the correct typed subclass for unit_type."""
    cls = NODE_REGISTRY.get(unit_type)
    if cls is None:
        raise ValueError(
            f"Unknown unit type '{unit_type}'. "
            f"Supported: {sorted(SUPPORTED_UNIT_TYPES)}")
    return cls(tag=tag, params=params or {}, property_package=property_package,
               metadata=metadata or {})


# ── IR edge: process stream ────────────────────────────────────────────────────

@dataclass
class EdgeIR:
    tag:            str
    T:              Optional[float] = None   # K
    P:              Optional[float] = None   # Pa
    flow:           Optional[float] = None   # mol/s
    composition:    dict            = field(default_factory=dict)
    src_port:       int             = 0
    dst_port:       int             = 0
    phase:          str             = "mixed"
    metadata:       dict            = field(default_factory=dict)
    is_recycle:     bool            = False
    recycle_target: Optional[str]   = None   # unit tag this stream recycles to

    def copy(self) -> "EdgeIR":
        return copy.deepcopy(self)


# ── FlowsheetGraph ─────────────────────────────────────────────────────────────

class FlowsheetGraph:
    """
    Simulator-independent IR.  NetworkX DiGraph where:
      - unit nodes carry NodeIR in attr "ir"  (node_type="unit")
      - stream nodes carry EdgeIR in attr "ir" (node_type="stream")
      - edges: src_unit → stream → dst_unit

    Construction-time enforcement:
      add_unit()   — calls node.validate_construction(); raises on violation
      add_stream() — checks phase compatibility against outlet PortSpec
    """

    def __init__(self) -> None:
        self._g:                nx.DiGraph = nx.DiGraph()
        self.compounds:         list[str]  = []
        self.property_package:  str        = ""
        self.binary_parameters: list[dict] = []
        self.metadata:          dict       = {}

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_unit(self, node: NodeIR, strict: bool = True) -> None:
        """
        Add a unit node. Calls validate_construction() and raises ValueError
        if any construction-time constraints are violated (when strict=True).
        strict=False skips the check — used during repair when params are
        temporarily incomplete.
        """
        if strict:
            errors = node.validate_construction()
            if errors:
                raise ValueError(
                    f"Construction-time constraint violation for {node.tag}: "
                    + "; ".join(errors))
        self._g.add_node(node.tag, ir=node, node_type="unit")

    def add_stream(self, edge: EdgeIR, src_tag: Optional[str],
                   dst_tag: Optional[str],
                   enforce_phase: bool = True) -> None:
        """
        Add a stream node and connect it.

        When enforce_phase=True, checks that the stream phase is compatible
        with the src unit's outlet PortSpec (e.g. liquid stream from a
        Compressor's vapour outlet raises an error).
        """
        if not edge.tag:
            raise ValueError("Stream tag must be non-empty")

        if enforce_phase and src_tag:
            src_data = self._g.nodes.get(src_tag, {})
            src_node = src_data.get("ir")
            if isinstance(src_node, NodeIR):
                expected_phase = src_node.outlet_phase(edge.src_port)
                if (expected_phase != "any"
                        and edge.phase != "mixed"
                        and edge.phase != expected_phase):
                    raise ValueError(
                        f"Phase mismatch: stream '{edge.tag}' has phase='{edge.phase}' "
                        f"but {src_tag} port {edge.src_port} requires '{expected_phase}'")

        self._g.add_node(edge.tag, ir=edge, node_type="stream")
        if src_tag:
            self._g.add_edge(src_tag, edge.tag)
        if dst_tag:
            self._g.add_edge(edge.tag, dst_tag)

    def remove_edge_between(self, src_tag: str, dst_tag: str) -> None:
        for node in list(self._g.successors(src_tag)):
            if dst_tag in list(self._g.successors(node)):
                self._g.remove_node(node)
                return

    # ── Accessors ──────────────────────────────────────────────────────────────

    def units(self) -> list[NodeIR]:
        return [d["ir"] for _, d in self._g.nodes(data=True)
                if d.get("node_type") == "unit"]

    def unit(self, tag: str) -> Optional[NodeIR]:
        data = self._g.nodes.get(tag)
        return data["ir"] if data and data.get("node_type") == "unit" else None

    def streams(self) -> list[EdgeIR]:
        return [d["ir"] for _, d in self._g.nodes(data=True)
                if d.get("node_type") == "stream"]

    def stream(self, tag: str) -> Optional[EdgeIR]:
        data = self._g.nodes.get(tag)
        return data["ir"] if data and data.get("node_type") == "stream" else None

    def unit_tags(self) -> set[str]:
        return {n for n, d in self._g.nodes(data=True) if d.get("node_type") == "unit"}

    def stream_tags(self) -> set[str]:
        return {n for n, d in self._g.nodes(data=True) if d.get("node_type") == "stream"}

    def inlet_streams(self, unit_tag: str) -> list[EdgeIR]:
        return [self._g.nodes[p]["ir"] for p in self._g.predecessors(unit_tag)
                if self._g.nodes.get(p, {}).get("node_type") == "stream"]

    def outlet_streams(self, unit_tag: str) -> list[EdgeIR]:
        return [self._g.nodes[s]["ir"] for s in self._g.successors(unit_tag)
                if self._g.nodes.get(s, {}).get("node_type") == "stream"]

    def stream_source(self, stream_tag: str) -> Optional[str]:
        for pred in self._g.predecessors(stream_tag):
            if self._g.nodes[pred].get("node_type") == "unit":
                return pred
        return None

    def stream_dest(self, stream_tag: str) -> Optional[str]:
        for succ in self._g.successors(stream_tag):
            if self._g.nodes[succ].get("node_type") == "unit":
                return succ
        return None

    # ── Graph properties ───────────────────────────────────────────────────────

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def validate_dag(self) -> bool:
        """
        True if the graph is acyclic after removing recycle streams.

        Raises TopologyError if a cycle exists in the non-recycle subgraph —
        this indicates a real cycle that was not tagged is_recycle=True.
        """
        import sys as _sys

        # Diagnostic: dump every stream with its is_recycle flag so failures
        # are unambiguous — either the flag wasn't propagated or a guard cleared it.
        _all_streams = self.streams()
        print(
            f"[DAG] validate_dag called — {len(_all_streams)} stream(s): "
            + ", ".join(
                f"'{s.tag}'(recycle={s.is_recycle})" for s in _all_streams
            ),
            flush=True, file=_sys.stderr,
        )

        recycle_tags = {e.tag for e in self.recycle_edges()}
        print(
            f"[DAG] recycle stream(s) excluded from DAG check: "
            f"{sorted(recycle_tags) or '(none)'}",
            flush=True, file=_sys.stderr,
        )

        def _report_cycle(g: "nx.DiGraph", label: str) -> None:
            try:
                cycle_edges = nx.find_cycle(g)
                nodes_in_cycle = [u for u, _ in cycle_edges]
                print(
                    f"[DAG] cycle in {label}: "
                    + " → ".join(nodes_in_cycle)
                    + f" → {nodes_in_cycle[0]}",
                    flush=True, file=_sys.stderr,
                )
                for n in nodes_in_cycle:
                    ir = self._g.nodes.get(n, {}).get("ir")
                    if ir:
                        flag = getattr(ir, "is_recycle", "N/A")
                        print(
                            f"[DAG]   node '{n}': is_recycle={flag}",
                            flush=True, file=_sys.stderr,
                        )
            except nx.NetworkXNoCycle:
                pass

        if not recycle_tags:
            if not nx.is_directed_acyclic_graph(self._g):
                _report_cycle(self._g, "full graph (no recycle tags found)")
                raise TopologyError(
                    "Cycle detected in non-recycle streams — "
                    "possible missing is_recycle tag"
                )
            return True

        sub = self._g.subgraph([n for n in self._g.nodes if n not in recycle_tags])
        print(
            f"[DAG] non-recycle subgraph nodes ({len(sub.nodes)}): "
            + ", ".join(sorted(sub.nodes)),
            flush=True, file=_sys.stderr,
        )
        if not nx.is_directed_acyclic_graph(sub):
            _report_cycle(sub, "non-recycle subgraph")
            raise TopologyError(
                "Cycle detected in non-recycle streams — "
                "possible missing is_recycle tag"
            )
        return True

    def recycle_edges(self) -> list[EdgeIR]:
        return [s for s in self.streams() if s.is_recycle]

    @property
    def has_recycles(self) -> bool:
        return any(s.is_recycle for s in self.streams())

    def unit_graph(self) -> nx.DiGraph:
        ug = nx.DiGraph()
        for node in self.units():
            ug.add_node(node.tag)
        for stream in self.streams():
            src = self.stream_source(stream.tag)
            dst = self.stream_dest(stream.tag)
            if src and dst:
                ug.add_edge(src, dst, stream_tag=stream.tag)
        return ug

    def feed_streams(self) -> list[EdgeIR]:
        return [s for s in self.streams()
                if s.metadata.get("is_feed") or self.stream_source(s.tag) is None]

    def product_streams(self) -> list[EdgeIR]:
        return [s for s in self.streams()
                if self.stream_dest(s.tag) is None and self.stream_source(s.tag) is not None]

    # ── Copy ──────────────────────────────────────────────────────────────────

    def copy(self) -> "FlowsheetGraph":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (f"FlowsheetGraph("
                f"units={[u.tag for u in self.units()]}, "
                f"streams={[s.tag for s in self.streams()]}, "
                f"package={self.property_package!r})")


def inlet_traces_to_pressure_raiser(graph: "FlowsheetGraph", unit_tag: str) -> bool:
    """True when a unit's PROPAGATED inlet temperature is unreliable because it
    traces upstream to a Compressor/Pump with no intervening temperature-setting
    unit (Heater/Cooler/ConversionReactor).

    The consistency pass propagates a pressure-raiser's inlet T unchanged to its
    outlet (it has no compression-heating model), so a cooler placed downstream
    sees the cold FEED temperature rather than the real hot discharge. Its
    legitimate setpoint (e.g. cool a 250 C discharge to 40 C, above the 25 C feed)
    must therefore NOT be 'corrected' by the monotonic check, nor flagged by the
    validate feasibility check, against that unreliable inlet. A temperature-setting
    unit between the pressure-raiser and this unit resets the propagated T, so the
    walk stops at the first such unit on each branch.
    """
    ug = graph.unit_graph()
    if unit_tag not in ug:
        return False
    seen: set = set()
    stack = list(ug.predecessors(unit_tag))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        node = graph.unit(p)
        if node is None:
            continue
        if isinstance(node, (CompressorNode, PumpNode)):
            return True                        # pressure-raiser upstream → inlet T unreliable
        if isinstance(node, (HeaterNode, CoolerNode, ConversionReactorNode)):
            continue                           # T reset here → this branch is reliable; stop
        stack.extend(ug.predecessors(p))       # pass-through (mixer/splitter) → keep walking up
    return False
