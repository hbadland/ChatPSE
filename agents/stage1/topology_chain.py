"""
TopologyChain — 4-call LangChain pipeline for validation-tier flowsheet extraction.

Replaces the UnitExtractor + StreamExtractor pair when tier="validation", using a
sequential chain so each call can build on verified intermediate output rather than
attempting the full topology in one shot.

Call 1: Unit identification   → {units: [{tag, type}]}
Call 2: Stream connections    → {streams: [{tag, src, dst, T, P, composition}]}
Call 3: Recycle detection     → {recycles: [{stream_tag, target_unit}]}
Call 4: Topology validation   → {valid, issues, corrected_units, corrected_streams}

Returns (SemanticUnits, SemanticTopology) — same interface as UnitExtractor /
StreamExtractor so the orchestrator recycle guards and GraphBuilder are unchanged.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Optional

# ── LangChain availability check — install if absent ─────────────────────────
try:
    import langchain  # noqa: F401
except ImportError:
    subprocess.run(
        ["pip", "install", "langchain", "langchain-community",
         "--break-system-packages", "--quiet"],
        check=False,
    )

from agents.llm import DEFAULT_MODEL, _provider, _OLLAMA_BASE_URL
from agents.stage1.unit_extractor import SemanticUnit, SemanticUnits, SUPPORTED_UNIT_TYPES
from agents.stage1.stream_extractor import SemanticStream, SemanticTopology

# ── Prompt templates ──────────────────────────────────────────────────────────

_UNIT_SYSTEM = """\
/no_think
You are a chemical process engineer. List ALL unit operations in the process description.
For each unit assign a tag (K-01 for compressors, HT-01 for heaters, CL-01 for coolers,
V-01 for vessels, MX-01 for mixers, SP-01 for splitters, P-01 for pumps, EX-01 for expanders,
RX-01 for reactors).
Output JSON: {"units": [{"tag": "...", "type": "..."}]}
Types: Heater Cooler Vessel Mixer Splitter Pump Compressor Expander ConversionReactor"""

_UNIT_USER = """\
Process description: {description}
Compounds: {compounds}

List ALL unit operations in this process."""


_STREAM_SYSTEM = """\
/no_think
You are a chemical process engineer. Given these unit operations, identify all material
streams connecting them. For each stream: source unit tag, destination unit tag, and any
T/P/composition conditions.
Output JSON: {"streams": [{"tag": "...", "src": "...", "dst": "...", "T": ..., "P": ..., "composition": {...}}]}
- src=null for feed streams, dst=null for product streams
- T in Kelvin, P in Pascals
- composition: mole fractions summing to 1.0 for feed streams"""

_STREAM_USER = """\
Process description: {description}
Unit operations:
{units_json}

Identify all material streams connecting these units."""


_RECYCLE_SYSTEM = """\
/no_think
You are a chemical process engineer. Identify which streams return material to an earlier
point in the process (recycle streams).
Only tag as recycle if the description explicitly uses: recycled back, returned to,
fed back to, recirculated to.
Output JSON: {"recycles": [{"stream_tag": "...", "target_unit": "..."}]}
- stream_tag: exact tag of the recycle stream from the stream list
- target_unit: exact tag of the unit the stream recycles back to
If no recycles, return: {"recycles": []}"""

_RECYCLE_USER = """\
Process description: {description}
Streams:
{streams_json}

Identify all recycle streams."""


_VALIDATE_SYSTEM = """\
/no_think
You are a chemical process engineer. Review this flowsheet topology against the description.
Check:
1. Every unit mentioned is included
2. Every stream connection makes physical sense
3. Feed streams have T, P, composition specified
Output JSON: {"valid": true/false, "issues": ["..."], "corrected_units": [...], "corrected_streams": [...]}
- corrected_units and corrected_streams must always be populated (copy from input if no changes needed)
- issues: list any problems found (empty list if none)"""

_VALIDATE_USER = """\
Process description: {description}
Compounds: {compounds}
Units: {units_json}
Streams (with recycle annotations applied): {topology_json}

Review this topology and return the corrected version."""


# ── LLM factory ──────────────────────────────────────────────────────────────

def _make_langchain_llm(model: str):
    """Return a LangChain chat model configured for the given model name."""
    provider = _provider(model)

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        # _OLLAMA_BASE_URL is the OpenAI-compat endpoint (ends in /v1).
        # ChatOllama uses Ollama's native API so strip the /v1 suffix.
        native_url = _OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
        return ChatOllama(
            model=model, base_url=native_url,
            temperature=0.0, num_predict=16384,
        )

    if provider == "openai":
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.0, max_tokens=16384)

    if provider == "anthropic":
        from langchain_community.chat_models import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0.0, max_tokens=16384)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.0, max_output_tokens=16384)

    if provider == "groq":
        from langchain_community.chat_models import ChatOpenAI
        from agents.llm import _GROQ_BASE_URL
        return ChatOpenAI(
            model=model,
            openai_api_base=_GROQ_BASE_URL,
            openai_api_key=os.environ.get("GROQ_API_KEY", ""),
            temperature=0.0, max_tokens=16384,
        )

    raise ValueError(f"TopologyChain: unsupported provider for model '{model}'")


# ── TopologyChain ─────────────────────────────────────────────────────────────

class TopologyChain:
    """
    4-call LangChain sequential chain for validation-tier topology extraction.
    Returns (SemanticUnits, SemanticTopology) — same interface as the individual extractors.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        self._model = model
        llm    = _make_langchain_llm(model)
        parser = StrOutputParser()

        self._unit_chain = (
            ChatPromptTemplate.from_messages([("system", _UNIT_SYSTEM), ("human", _UNIT_USER)])
            | llm | parser
        )
        self._stream_chain = (
            ChatPromptTemplate.from_messages([("system", _STREAM_SYSTEM), ("human", _STREAM_USER)])
            | llm | parser
        )
        self._recycle_chain = (
            ChatPromptTemplate.from_messages([("system", _RECYCLE_SYSTEM), ("human", _RECYCLE_USER)])
            | llm | parser
        )
        self._validate_chain = (
            ChatPromptTemplate.from_messages([("system", _VALIDATE_SYSTEM), ("human", _VALIDATE_USER)])
            | llm | parser
        )

    def extract(
        self,
        description: str,
        compounds:   list[str],
    ) -> tuple[list[SemanticUnit], list[SemanticStream]]:
        """Run the 4-call chain and return (units list, streams list)."""
        compounds_str = ", ".join(compounds)

        # ── Call 1: unit identification ───────────────────────────────────────
        print("[TC] call 1/4: unit identification", flush=True, file=sys.stderr)
        raw1       = _strip(self._unit_chain.invoke({"description": description,
                                                      "compounds": compounds_str}))
        units_data = _parse_json(raw1)
        units_json = json.dumps(units_data, indent=2)
        n_units    = len(units_data.get("units", []))
        print(f"[TC] call 1 → {n_units} units", flush=True, file=sys.stderr)

        # ── Call 2: stream connections ────────────────────────────────────────
        print("[TC] call 2/4: stream connections", flush=True, file=sys.stderr)
        raw2         = _strip(self._stream_chain.invoke({"description": description,
                                                          "units_json": units_json}))
        streams_data = _parse_json(raw2)
        streams_json = json.dumps(streams_data, indent=2)
        n_streams    = len(streams_data.get("streams", []))
        print(f"[TC] call 2 → {n_streams} streams", flush=True, file=sys.stderr)

        # ── Call 3: recycle detection ─────────────────────────────────────────
        print("[TC] call 3/4: recycle detection", flush=True, file=sys.stderr)
        raw3          = _strip(self._recycle_chain.invoke({"description": description,
                                                            "streams_json": streams_json}))
        recycles_data = _parse_json(raw3)
        n_recycles    = len(recycles_data.get("recycles", []))
        print(f"[TC] call 3 → {n_recycles} recycle annotation(s)", flush=True, file=sys.stderr)

        # Merge recycle annotations into the stream list so Node 4 sees a complete picture
        _recycle_map: dict[str, str] = {
            r["stream_tag"]: r["target_unit"]
            for r in recycles_data.get("recycles", [])
            if "stream_tag" in r and "target_unit" in r
        }
        for s in streams_data.get("streams", []):
            if s.get("tag") in _recycle_map:
                s["is_recycle"]     = True
                s["recycle_target"] = _recycle_map[s["tag"]]

        topology_for_validation = {
            "units":    units_data.get("units", []),
            "streams":  streams_data.get("streams", []),
            "recycles": recycles_data.get("recycles", []),
        }
        topology_json = json.dumps(topology_for_validation, indent=2)

        # ── Call 4: topology validation ───────────────────────────────────────
        print("[TC] call 4/4: topology validation", flush=True, file=sys.stderr)
        raw4       = _strip(self._validate_chain.invoke({"description": description,
                                                          "compounds": compounds_str,
                                                          "units_json": units_json,
                                                          "topology_json": topology_json}))
        final_data = _parse_json(raw4)
        issues     = final_data.get("issues", [])
        if issues:
            print(f"[TC] call 4 issues: {issues}", flush=True, file=sys.stderr)
        print(f"[TC] call 4 → valid={final_data.get('valid')}", flush=True, file=sys.stderr)

        # Use corrected_units / corrected_streams from Node 4; fall back to Node 1/2 output
        final_units   = final_data.get("corrected_units")   or units_data.get("units", [])
        final_streams = final_data.get("corrected_streams") or streams_data.get("streams", [])

        # Re-apply recycle annotations to corrected streams (Node 4 may have dropped them)
        for s in final_streams:
            if s.get("tag") in _recycle_map and not s.get("is_recycle"):
                s["is_recycle"]     = True
                s["recycle_target"] = _recycle_map[s["tag"]]

        combined = {"units": final_units, "streams": final_streams}
        return (
            _to_semantic_units(combined).units,
            _to_semantic_topology(combined).streams,
        )


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _strip(text: str) -> str:
    """Strip <think> blocks and markdown fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict:
    text = _strip(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)


def _to_semantic_units(data: dict) -> SemanticUnits:
    units = []
    for u in data.get("units", []):
        utype = u.get("type", "")
        if utype not in SUPPORTED_UNIT_TYPES:
            print(f"[TC] WARNING: unknown unit type '{utype}' for tag '{u.get('tag')}' — skipping",
                  flush=True, file=sys.stderr)
            continue
        units.append(SemanticUnit(
            tag  = u["tag"],
            type = utype,
            role = u.get("role", ""),
        ))
    return SemanticUnits(units=units, raw_json=data)


def _to_semantic_topology(data: dict) -> SemanticTopology:
    streams = []
    for s in data.get("streams", []):
        raw_comp = s.get("composition") or {}
        streams.append(SemanticStream(
            tag            = s["tag"],
            src            = s.get("src"),
            dst            = s.get("dst"),
            is_feed        = bool(s.get("is_feed", s.get("src") is None)),
            T              = s.get("T"),
            P              = s.get("P"),
            flow           = s.get("flow"),
            composition    = {k: float(v) for k, v in raw_comp.items()},
            is_recycle     = bool(s.get("is_recycle", False)),
            recycle_target = s.get("recycle_target"),
        ))
    return SemanticTopology(streams=streams, raw_json=data)
