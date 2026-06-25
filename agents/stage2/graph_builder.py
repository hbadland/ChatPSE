"""
Agent C — Graph Builder.

Input : SemanticUnits + SemanticTopology (from Stage 1) + compound list
Output: FlowsheetGraph (the canonical IR)

This is a deterministic assembly step — no LLM call.
It translates the flat semantic dicts into the graph IR, then hands off
to ir/normalise.py and ir/validate.py.

The graph is NOT yet simulation-ready: property_package and unit params
are populated by Stage 3 agents (ThermoMapper, ParamMapper).
"""
from __future__ import annotations

from ir.graph import FlowsheetGraph, EdgeIR, make_node
from agents.stage1.unit_extractor import SemanticUnits
from agents.stage1.stream_extractor import SemanticTopology


class GraphBuilder:
    """Assembles a FlowsheetGraph from Stage 1 semantic output. Zero LLM calls."""

    def build(
        self,
        units:     SemanticUnits,
        topology:  SemanticTopology,
        compounds: list[str],
    ) -> FlowsheetGraph:
        graph = FlowsheetGraph()
        graph.compounds = list(compounds)

        # Add all unit nodes
        for sem_unit in units.units:
            # Seed the reactor stoichiometry into params so it survives to
            # to_dwsim (ParamMapper preserves existing params; its estimator only
            # fills reaction="" when absent).  Without this the reactor reaches
            # DWSIM with an empty reaction and converts nothing.
            _params = {}
            if sem_unit.type == "ConversionReactor" and getattr(sem_unit, "reaction", ""):
                _params["reaction"] = sem_unit.reaction
            node = make_node(
                unit_type = sem_unit.type,
                tag       = sem_unit.tag,
                params    = _params,
                metadata  = {"role": sem_unit.role},
            )
            graph.add_unit(node)

        # Case-insensitive lookup so LLM-written "water" matches canonical "Water".
        _compound_canon: dict[str, str] = {c.lower(): c for c in compounds}

        # Add all stream nodes and edges.
        # Track next available outlet port per unit so multi-outlet units
        # (Vessel: port 0 = vapour, port 1 = liquid) get distinct src_ports.
        # _repair_vessel_ports in normalise.py then swaps them if naming heuristics
        # indicate the LLM listed liquid before vapour.
        _next_outlet_port: dict[str, int] = {}

        for sem_stream in topology.streams:
            src_port = 0
            if sem_stream.src is not None:
                src_node = graph.unit(sem_stream.src)
                if src_node is not None and src_node.max_outlets() > 1:
                    src_port = _next_outlet_port.get(sem_stream.src, 0)
                    _next_outlet_port[sem_stream.src] = src_port + 1

            raw_comp = dict(sem_stream.composition)
            composition = {
                _compound_canon.get(k.lower(), k): v for k, v in raw_comp.items()
            }

            edge = EdgeIR(
                tag            = sem_stream.tag,
                T              = sem_stream.T,
                P              = sem_stream.P,
                flow           = sem_stream.flow,
                composition    = composition,
                src_port       = src_port,
                metadata       = {"is_feed": sem_stream.is_feed},
                is_recycle     = sem_stream.is_recycle,
                recycle_target = sem_stream.recycle_target,
            )
            graph.add_stream(edge, sem_stream.src, sem_stream.dst)

        _insert_recycle_blocks(graph)

        # Assertion 1: is_recycle flag must survive SemanticStream → EdgeIR.
        # Fires when graph_builder.py fails to propagate the flag — should never
        # trigger, but confirms the propagation invariant on every build.
        import sys as _sys
        for _sem in topology.streams:
            if _sem.is_recycle:
                _ir = graph.stream(_sem.tag)
                if _ir is None or not _ir.is_recycle:
                    _actual = _ir.is_recycle if _ir else "MISSING"
                    print(
                        f"[BUILDER] BUG: SemanticStream '{_sem.tag}' is_recycle=True "
                        f"but built EdgeIR has is_recycle={_actual}",
                        flush=True, file=_sys.stderr,
                    )
                    raise AssertionError(
                        f"GraphBuilder failed to propagate is_recycle=True for "
                        f"stream '{_sem.tag}' (built EdgeIR has is_recycle={_actual})"
                    )
                else:
                    print(
                        f"[BUILDER] recycle stream '{_sem.tag}' "
                        f"is_recycle=True correctly propagated to EdgeIR",
                        flush=True, file=_sys.stderr,
                    )

        # Assertion 2: every recycle edge must reference a real unit tag.
        _unit_tags = graph.unit_tags()
        for _edge in graph.recycle_edges():
            if not _edge.recycle_target or _edge.recycle_target not in _unit_tags:
                raise ValueError(
                    f"Recycle stream '{_edge.tag}' has recycle_target="
                    f"{_edge.recycle_target!r} which is not a valid unit tag. "
                    f"Valid tags: {sorted(_unit_tags)}"
                )
            _src = graph.stream_source(_edge.tag)
            if _src not in _unit_tags:
                raise ValueError(
                    f"Recycle stream '{_edge.tag}' source unit '{_src}' "
                    "not found in graph units."
                )

        return graph


def _insert_recycle_blocks(graph: FlowsheetGraph) -> None:
    """
    Record recycle convergence block tags in graph.metadata["recycle_blocks"].

    Each recycle edge gets a deterministic REC-01 / REC-02 … tag.  The actual
    DWSIM convergence blocks are wired up during ir/to_dwsim.py conversion via
    add_recycle_block(); storing the mapping here makes it inspectable without
    inserting a physical unit node (which would break SUPPORTED_UNIT_TYPES
    validation and unit-count benchmark checks).
    """
    recycle_edges = graph.recycle_edges()
    if not recycle_edges:
        return
    blocks = []
    for i, edge in enumerate(recycle_edges, start=1):
        blocks.append({
            "tag":           f"REC-{i:02d}",
            "inlet_stream":  edge.tag,
            "outlet_stream": f"{edge.tag}-INIT",
        })
    graph.metadata["recycle_blocks"] = blocks
