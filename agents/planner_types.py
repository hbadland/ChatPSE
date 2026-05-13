"""
Shared data classes for the decomposed Planner pipeline.

PlannerAgent chains four stages:
  TopologyAgent   → TopologyPlan   (unit sequence)
  ConnectionAgent → ConnectionPlan (stream graph)
  ConditionAgent  → ConditionPlan  (numerical values)
  Assembler       → dict           (final flowsheet JSON, deterministic)
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class UnitSpec:
    tag:  str   # e.g. "HT-01"
    type: str   # e.g. "Heater"


@dataclass
class TopologyPlan:
    units:     list[UnitSpec]
    n_feeds:   int             # number of external feed streams required
    reasoning: str = ""
    source:    str = "llm"    # "topology_library" | "llm"


@dataclass
class ConnectionPlan:
    feed_tags:         list[str]    # streams that need full T/P/composition
    intermediate_tags: list[str]    # streams between units (no conditions)
    product_tags:      list[str]    # final outlet streams (no conditions)
    connections:       list[list]   # [src_tag, dst_tag, src_port, dst_port]


@dataclass
class StreamCondition:
    T:           float
    P:           float
    flow:        float
    composition: dict[str, float]


@dataclass
class ConditionPlan:
    # stream_tag → full specification for each feed stream
    feed_conditions:  dict[str, StreamCondition]
    # unit_tag → parameter dict (T_out, dP, P_out, efficiency, split_fractions …)
    unit_parameters:  dict[str, dict]
