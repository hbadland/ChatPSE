"""
Intermediate Representation (IR) for flowsheets.

All agents read and write FlowsheetGraph — a NetworkX DiGraph wrapper.
DWSIM JSON is generated only at the final stage via ir/to_dwsim.py.

Node types in the graph:
  "unit"   — a unit operation (NodeIR stored in node attr "ir")
  "stream" — a process stream (EdgeIR stored in edge attr "ir")

Edges always connect: unit → stream → unit
i.e. every edge in the DiGraph is (unit_tag, stream_tag) or (stream_tag, unit_tag).
Stream nodes carry the EdgeIR; unit nodes carry NodeIR.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


# ── Port specification ─────────────────────────────────────────────────────────

@dataclass
class PortSpec:
    port_id:   int
    direction: str   # "inlet" | "outlet"
    phase:     str   # "any" | "vapour" | "liquid" | "mixed"
    required:  bool = True


# Canonical port constraints per unit type.
# normalise.py and validate.py read these; LLMs never see them.
PORT_SPECS: dict[str, list[PortSpec]] = {
    "Heater": [
        PortSpec(0, "inlet",  "any"),
        PortSpec(0, "outlet", "any"),
    ],
    "Cooler": [
        PortSpec(0, "inlet",  "any"),
        PortSpec(0, "outlet", "any"),
    ],
    "Vessel": [
        PortSpec(0, "inlet",  "mixed"),
        PortSpec(0, "outlet", "vapour"),
        PortSpec(1, "outlet", "liquid"),
    ],
    "Mixer": [
        PortSpec(0, "inlet", "any"),
        PortSpec(1, "inlet", "any"),
        PortSpec(0, "outlet", "any"),
    ],
    "Splitter": [
        PortSpec(0, "inlet",  "any"),
        PortSpec(0, "outlet", "any"),
        PortSpec(1, "outlet", "any"),
    ],
    "Pump": [
        PortSpec(0, "inlet",  "liquid"),
        PortSpec(0, "outlet", "liquid"),
    ],
    "Compressor": [
        PortSpec(0, "inlet",  "vapour"),
        PortSpec(0, "outlet", "vapour"),
    ],
    "Expander": [
        PortSpec(0, "inlet",  "vapour"),
        PortSpec(0, "outlet", "vapour"),
    ],
}

SUPPORTED_UNIT_TYPES: frozenset[str] = frozenset(PORT_SPECS.keys())


# ── IR node: unit operation ────────────────────────────────────────────────────

@dataclass
class NodeIR:
    tag:              str
    unit_type:        str
    params:           dict = field(default_factory=dict)
    property_package: Optional[str] = None
    metadata:         dict = field(default_factory=dict)

    def copy(self) -> "NodeIR":
        return copy.deepcopy(self)


# ── IR edge: process stream ────────────────────────────────────────────────────

@dataclass
class EdgeIR:
    tag:         str
    T:           Optional[float] = None   # K
    P:           Optional[float] = None   # Pa
    flow:        Optional[float] = None   # mol/s
    composition: dict            = field(default_factory=dict)
    src_port:    int             = 0
    dst_port:    int             = 0
    phase:       str             = "mixed"
    metadata:    dict            = field(default_factory=dict)

    def copy(self) -> "EdgeIR":
        return copy.deepcopy(self)


# ── FlowsheetGraph ─────────────────────────────────────────────────────────────

class FlowsheetGraph:
    """
    Simulator-independent IR.  Internally a NetworkX DiGraph where:
      - unit nodes carry NodeIR in node attr "ir"
      - stream nodes carry EdgeIR in node attr "ir"
      - edges are (src_unit_tag, stream_tag) and (stream_tag, dst_unit_tag)

    External feed/product streams that have no upstream/downstream unit are
    represented as stream-only nodes with degree < 2.
    """

    def __init__(self) -> None:
        self._g:                nx.DiGraph  = nx.DiGraph()
        self.compounds:         list[str]   = []
        self.property_package:  str         = ""
        self.binary_parameters: list[dict]  = []
        self.metadata:          dict        = {}

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_unit(self, node: NodeIR) -> None:
        self._g.add_node(node.tag, ir=node, node_type="unit")

    def add_stream(self, edge: EdgeIR, src_tag: str, dst_tag: str) -> None:
        """
        Add a stream node and connect it: src_tag → stream_tag → dst_tag.
        src_tag and dst_tag may be unit tags or None (for feed/product terminals).
        """
        self._g.add_node(edge.tag, ir=edge, node_type="stream")
        if src_tag:
            self._g.add_edge(src_tag, edge.tag)
        if dst_tag:
            self._g.add_edge(edge.tag, dst_tag)

    def remove_edge_between(self, src_tag: str, dst_tag: str) -> None:
        """Remove the stream node (and its edges) that sits between src and dst."""
        for node in list(self._g.successors(src_tag)):
            if dst_tag in list(self._g.successors(node)):
                self._g.remove_node(node)
                return

    # ── Accessors ──────────────────────────────────────────────────────────────

    def units(self) -> list[NodeIR]:
        return [
            d["ir"]
            for _, d in self._g.nodes(data=True)
            if d.get("node_type") == "unit"
        ]

    def unit(self, tag: str) -> Optional[NodeIR]:
        data = self._g.nodes.get(tag)
        return data["ir"] if data and data.get("node_type") == "unit" else None

    def streams(self) -> list[EdgeIR]:
        return [
            d["ir"]
            for _, d in self._g.nodes(data=True)
            if d.get("node_type") == "stream"
        ]

    def stream(self, tag: str) -> Optional[EdgeIR]:
        data = self._g.nodes.get(tag)
        return data["ir"] if data and data.get("node_type") == "stream" else None

    def unit_tags(self) -> set[str]:
        return {
            n for n, d in self._g.nodes(data=True)
            if d.get("node_type") == "unit"
        }

    def stream_tags(self) -> set[str]:
        return {
            n for n, d in self._g.nodes(data=True)
            if d.get("node_type") == "stream"
        }

    def inlet_streams(self, unit_tag: str) -> list[EdgeIR]:
        """Streams flowing INTO unit_tag."""
        result = []
        for pred in self._g.predecessors(unit_tag):
            data = self._g.nodes.get(pred, {})
            if data.get("node_type") == "stream":
                result.append(data["ir"])
        return result

    def outlet_streams(self, unit_tag: str) -> list[EdgeIR]:
        """Streams flowing OUT OF unit_tag."""
        result = []
        for succ in self._g.successors(unit_tag):
            data = self._g.nodes.get(succ, {})
            if data.get("node_type") == "stream":
                result.append(data["ir"])
        return result

    def stream_source(self, stream_tag: str) -> Optional[str]:
        """Unit tag upstream of this stream, or None if it's a feed."""
        for pred in self._g.predecessors(stream_tag):
            if self._g.nodes[pred].get("node_type") == "unit":
                return pred
        return None

    def stream_dest(self, stream_tag: str) -> Optional[str]:
        """Unit tag downstream of this stream, or None if it's a product."""
        for succ in self._g.successors(stream_tag):
            if self._g.nodes[succ].get("node_type") == "unit":
                return succ
        return None

    # ── Graph properties ───────────────────────────────────────────────────────

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def unit_graph(self) -> nx.DiGraph:
        """Projected DiGraph over unit nodes only (streams collapsed to edges)."""
        ug = nx.DiGraph()
        for node in self.units():
            ug.add_node(node.tag)
        for stream in self.streams():
            src = self.stream_source(stream.tag)
            dst = self.stream_dest(stream.tag)
            if src and dst:
                ug.add_edge(src, dst, stream_tag=stream.tag)
        return ug

    # ── Copy ──────────────────────────────────────────────────────────────────

    def copy(self) -> "FlowsheetGraph":
        return copy.deepcopy(self)

    # ── Debug repr ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FlowsheetGraph("
            f"units={[u.tag for u in self.units()]}, "
            f"streams={[s.tag for s in self.streams()]}, "
            f"package={self.property_package!r})"
        )
