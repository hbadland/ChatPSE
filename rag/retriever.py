"""
RAG retriever — unified knowledge query interface.

Wraps three corpora:
  1. BIP corpus       (rag/sources/binary_parameters.json)
  2. Thermo models    (rag/sources/thermo_models.json)
  3. Unit specs       (rag/sources/unit_specs.json)

All lookups are deterministic (no embeddings, no vector search).
Agents call this instead of encoding knowledge in their prompts.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Optional

_DIR = os.path.dirname(__file__)

def _load(name: str) -> object:
    with open(os.path.join(_DIR, "sources", name), encoding="utf-8") as f:
        return json.load(f)


# ── BIP retrieval ──────────────────────────────────────────────────────────────

@dataclass
class BIPRecord:
    compound_a:  str
    compound_b:  str
    model:       str
    A12:         float
    A21:         float
    alpha12:     float
    B12:         float = 0.0
    B21:         float = 0.0
    source:      str   = ""
    T_min_K:     Optional[float] = None
    T_max_K:     Optional[float] = None
    confidence:  str   = ""

    def as_dwsim_dict(self) -> dict:
        """Convert to the binary_parameters entry format used by executor.py."""
        d: dict = {
            "model":      self.model,
            "compound_a": self.compound_a,
            "compound_b": self.compound_b,
            "A12":        self.A12,
            "A21":        self.A21,
            "source":     self.source,
        }
        if self.model == "NRTL":
            d["alpha12"] = self.alpha12
        if self.B12: d["B12"] = self.B12
        if self.B21: d["B21"] = self.B21
        if self.T_min_K is not None: d["T_min_K"] = self.T_min_K
        if self.T_max_K is not None: d["T_max_K"] = self.T_max_K
        return d


def _norm(s: str) -> str:
    return s.strip().lower()


class BIPRetriever:
    """
    O(1) BIP lookup by (compound_a, compound_b, model), alias-aware.
    Covers all orderings: (a,b) and (b,a) are both indexed.
    """

    def __init__(self) -> None:
        self._lookup: dict[tuple[str, str, str], BIPRecord] = {}
        self._built = False

    def _build(self) -> None:
        if self._built:
            return
        self._built = True
        corpus: list[dict] = _load("binary_parameters.json")  # type: ignore
        for entry in corpus:
            names_a = [entry["compound_a"]] + entry.get("aliases_a", [])
            names_b = [entry["compound_b"]] + entry.get("aliases_b", [])
            model   = entry["model"]
            rec = BIPRecord(
                compound_a = entry["compound_a"],
                compound_b = entry["compound_b"],
                model      = model,
                A12        = entry["A12"],
                A21        = entry["A21"],
                alpha12    = entry.get("alpha12", 0.0),
                B12        = entry.get("B12", 0.0),
                B21        = entry.get("B21", 0.0),
                source     = entry.get("source", ""),
                T_min_K    = entry.get("T_min_K"),
                T_max_K    = entry.get("T_max_K"),
                confidence = entry.get("confidence", ""),
            )
            for na, nb in itertools.product(names_a, names_b):
                key_fwd = (_norm(na), _norm(nb), model)
                key_rev = (_norm(nb), _norm(na), model)
                if key_fwd not in self._lookup:
                    self._lookup[key_fwd] = rec
                if key_rev not in self._lookup:
                    # Reversed entry: swap A12↔A21 (DWSIM requires both)
                    rev = BIPRecord(
                        compound_a = rec.compound_b,
                        compound_b = rec.compound_a,
                        model=model, A12=rec.A21, A21=rec.A12,
                        alpha12=rec.alpha12, B12=rec.B21, B21=rec.B12,
                        source=rec.source, T_min_K=rec.T_min_K,
                        T_max_K=rec.T_max_K, confidence=rec.confidence,
                    )
                    self._lookup[key_rev] = rev

    def query(
        self,
        compounds:  list[str],
        model:      str,
        T_K:        Optional[float] = None,
    ) -> tuple[list[dict], list[tuple[str, str]]]:
        """
        Return (bip_dicts, missing_pairs).
        bip_dicts is the list ready for flowsheet["binary_parameters"].
        missing_pairs lists (a, b) pairs not in the corpus.

        Temperature guard: if T_K is given, records whose fit range
        is >20% outside are excluded (pair added to missing_pairs).
        """
        self._build()
        pairs = list(itertools.combinations(compounds, 2))
        found:   list[dict]            = []
        missing: list[tuple[str, str]] = []

        for a, b in pairs:
            key = (_norm(a), _norm(b), model)
            rec = self._lookup.get(key)
            if rec is None:
                missing.append((a, b))
                continue
            if T_K is not None and rec.T_min_K and rec.T_max_K:
                span = rec.T_max_K - rec.T_min_K
                if T_K < rec.T_min_K - 0.2 * span or T_K > rec.T_max_K + 0.2 * span:
                    missing.append((a, b))
                    continue
            found.append(rec.as_dwsim_dict())
            # Also add the reverse entry (DWSIM requires both orderings)
            rev_key = (_norm(b), _norm(a), model)
            rev_rec = self._lookup.get(rev_key)
            if rev_rec:
                found.append(rev_rec.as_dwsim_dict())

        return found, missing

    def has_full_coverage(self, compounds: list[str], model: str,
                          T_K: Optional[float] = None) -> bool:
        _, missing = self.query(compounds, model, T_K)
        return len(missing) == 0


# ── Thermo model retrieval ─────────────────────────────────────────────────────

class ThermoRetriever:
    """
    Rule-based property package selection.
    Returns a ranked list of candidate packages for a given compound set.
    """

    _COMPOUND_CLASSES = {
        "ALCOHOLS":    {"methanol","ethanol","1-propanol","2-propanol","n-propanol",
                        "isopropanol","1-butanol","n-butanol","isobutanol",
                        "2-butanol","1-pentanol","ethylene glycol","glycerol"},
        "KETONES":     {"acetone","methyl ethyl ketone","mek","cyclohexanone",
                        "methyl isobutyl ketone","mibk","acetophenone"},
        "ESTERS":      {"ethyl acetate","methyl acetate","butyl acetate",
                        "isopropyl acetate"},
        "ETHERS":      {"diethyl ether","methyl tert-butyl ether","mtbe",
                        "tetrahydrofuran","thf","1,4-dioxane","diisopropyl ether"},
        "CHLORINATED": {"chloroform","dichloromethane","dcm","carbon tetrachloride",
                        "1,2-dichloroethane","chlorobenzene"},
        "AROMATICS":   {"benzene","toluene","o-xylene","m-xylene","p-xylene",
                        "ethylbenzene","styrene","naphthalene"},
        "ALKANES":     {"methane","ethane","propane","n-butane","isobutane",
                        "n-pentane","n-hexane","n-heptane","n-octane",
                        "cyclohexane","methylcyclohexane"},
        "LIGHT_GASES": {"methane","ethane","propane","n-butane","isobutane",
                        "nitrogen","oxygen","carbon dioxide","co2",
                        "hydrogen sulfide","h2s","hydrogen","argon"},
        "POLAR_OTHER": {"acetic acid","formic acid","acetonitrile","dimethyl sulfoxide",
                        "dmso","ammonia","hydrogen sulfide","water"},
        "WATER":       {"water"},
    }

    _AZEOTROPES: set[frozenset] = {
        frozenset({"ethanol","water"}),
        frozenset({"methanol","water"}),
        frozenset({"1-propanol","water"}), frozenset({"n-propanol","water"}),
        frozenset({"2-propanol","water"}), frozenset({"isopropanol","water"}),
        frozenset({"1-butanol","water"}),  frozenset({"n-butanol","water"}),
        frozenset({"ethyl acetate","ethanol"}),
        frozenset({"ethyl acetate","water"}),
        frozenset({"acetone","chloroform"}),
        frozenset({"acetone","methanol"}),
        frozenset({"diethyl ether","water"}),
        frozenset({"n-hexane","ethanol"}),
        frozenset({"benzene","cyclohexane"}),
        frozenset({"tetrahydrofuran","water"}), frozenset({"thf","water"}),
        frozenset({"acetonitrile","water"}),
    }

    def _classify(self, compounds: list[str]) -> set[str]:
        """Return the set of compound class labels present."""
        norm = {c.strip().lower() for c in compounds}
        classes: set[str] = set()
        for cls, members in self._COMPOUND_CLASSES.items():
            if norm & members:
                classes.add(cls)
        return classes

    def _has_azeotrope(self, compounds: list[str]) -> bool:
        norm = [c.strip().lower() for c in compounds]
        for a, b in itertools.combinations(norm, 2):
            if frozenset({a, b}) in self._AZEOTROPES:
                return True
        return False

    def select(
        self,
        compounds:       list[str],
        description:     str = "",
        pressure_pa:     float = 101_325.0,
        temperature_k:   float = 300.0,
        exclude:         set[str] | None = None,
        bip_retriever:   Optional["BIPRetriever"] = None,
    ) -> list[str]:
        """
        Return a ranked list of candidate packages, best first.
        Applies hard rules in order; excludes already-tried packages.
        """
        exclude = exclude or set()
        classes = self._classify(compounds)
        has_azeo = self._has_azeotrope(compounds)
        is_polar = bool(classes & {"ALCOHOLS","KETONES","ESTERS","ETHERS",
                                   "POLAR_OTHER","WATER"})

        # ── Gas-phase vs liquid-activity routing ────────────────────────────
        # Thermodynamic basis: a mixture DOMINATED by light gases / hydrocarbons
        # is governed by the vapour phase and is correctly modelled with a cubic
        # EOS (PR/SRK) across the whole mixture. A MINORITY of polar species —
        # steam, or a trace of NH3 — does not turn it into a liquid system where
        # activity coefficients govern: the polar fraction is too small to create
        # activity-dominated liquid non-ideality, and the associating species are
        # themselves in the vapour phase. Only when polar/associating species are
        # a substantial fraction does the liquid become activity-controlled and an
        # activity model (NRTL/UNIQUAC) become necessary. So gas-phase routing
        # tolerates a minority polar species as long as gas/HC species dominate.
        cc = self._COMPOUND_CLASSES
        _comps_l = {c.strip().lower() for c in compounds}
        _acid_gases = {"hydrogen sulfide", "h2s", "carbon dioxide", "co2",
                       "sulfur dioxide", "so2"}
        _amines = {"monoethanolamine", "mea", "diethanolamine", "dea",
                   "methyldiethanolamine", "mdea"}
        _glycols = {"ethylene glycol", "diethylene glycol", "triethylene glycol",
                    "glycerol"}
        # Polar species that DO force a liquid activity model. Water and glycol
        # dehydration solvents are excluded (they ride along in a gas EOS); acid
        # gases are gas-phase; amines are chemical solvents that keep the activity
        # path (so they are counted here as activity-forcing).
        _activity_organics = (((cc["ALCOHOLS"] | cc["KETONES"] | cc["ESTERS"]
                                | cc["ETHERS"] | (cc["POLAR_OTHER"] - {"water"}))
                               - _glycols - _acid_gases) | _amines)
        _n_gas_like       = len(_comps_l & (cc["LIGHT_GASES"] | cc["ALKANES"]))
        _n_activity_polar = len(_comps_l & _activity_organics)
        # Gas-like species present (>=2) AND any activity-forcing polar species
        # are a strict minority → the stream is gas-phase, use an EOS.
        _gas_dominated = _n_gas_like >= 2 and _n_gas_like > _n_activity_polar

        _desc_l = (description or "").lower()
        _steam_kw = ("steam", "reform", "syngas", "combust", "flue gas", "gasif",
                     "high temperature", "high-temperature", "furnace",
                     "cracker", "cracking", "pyrolysis")

        # Steam: hot gas-phase water in a gas-dominated stream is not liquid water
        # (high T, or a steam/reforming/combustion/pyrolysis context) → EOS.
        water_is_steam = (
            "WATER" in classes and "LIGHT_GASES" in classes and _gas_dominated
            and (temperature_k >= 400.0 or any(k in _desc_l for k in _steam_kw))
        )
        # Acid gas (H2S/CO2/SO2) in a gas-dominated stream (sour/natural gas,
        # sweetening, glycol dehydration) → EOS. Amine acid-gas capture is NOT
        # gas-dominated (amine counted as activity-forcing; typically one gas)
        # so it correctly keeps the activity path.
        acid_gas_system = bool(_comps_l & _acid_gases) and _gas_dominated

        if water_is_steam or acid_gas_system:
            is_polar = False      # gas-dominated phase → EOS, not activity/ideal
            has_azeo = False

        is_light_gas  = "LIGHT_GASES" in classes and not is_polar
        is_cryogenic  = temperature_k < 200.0
        is_high_press = pressure_pa > 3e5
        has_water = "WATER" in classes

        candidates: list[str] = []

        # Cryogenic → Lee-Kesler-Plöcker first
        if is_cryogenic and is_light_gas:
            candidates.append("Lee-Kesler-Plöcker")

        # High-pressure non-polar / light gas
        if is_light_gas or (is_high_press and not is_polar):
            candidates.append("Peng-Robinson")
            candidates.append("Soave-Redlich-Kwong")

        # Polar / azeotropic
        if is_polar or has_azeo:
            if bip_retriever:
                # T_K=None here: check corpus coverage regardless of temperature.
                # The temperature guard is enforced during BIP injection, not selection.
                has_nrtl    = bip_retriever.has_full_coverage(compounds, "NRTL")
                has_uniquac = bip_retriever.has_full_coverage(compounds, "UNIQUAC")
            else:
                has_nrtl = has_uniquac = False  # unknown — try anyway

            if has_nrtl or not bip_retriever:
                candidates.append("NRTL")
            if has_uniquac:
                candidates.append("UNIQUAC")
            # Fallback when BIPs are unavailable
            if is_high_press:
                candidates.append("Peng-Robinson")
            candidates.append("Raoult's Law")

        # Non-polar, non-gas, low pressure
        if not is_polar and not is_light_gas:
            candidates.append("Peng-Robinson")
            candidates.append("Raoult's Law")

        # Deduplicate, preserve order, apply exclusions
        seen: set[str] = set()
        result: list[str] = []
        for pkg in candidates:
            if pkg not in seen and pkg not in exclude:
                seen.add(pkg)
                result.append(pkg)

        if not result:
            result = [p for p in ["Peng-Robinson", "NRTL", "Raoult's Law"]
                      if p not in exclude]

        return result

    def context_for_prompt(self, compounds: list[str]) -> str:
        """
        Return a short, structured selection guide for inclusion in LLM prompts.
        Replaces the 200-line thermo system prompt with a targeted snippet.
        """
        classes = self._classify(compounds)
        has_azeo = self._has_azeotrope(compounds)
        lines = [f"Compound classes detected: {', '.join(sorted(classes)) or 'unknown'}"]
        if has_azeo:
            lines.append("Known azeotrope pair detected — NRTL or UNIQUAC required.")
        if "WATER" in classes and classes & {"ALCOHOLS","KETONES","ESTERS"}:
            lines.append("Polar/water system — Raoult's Law forbidden. Use NRTL.")
        if "LIGHT_GASES" in classes:
            lines.append("Light gases present — prefer Peng-Robinson or Lee-Kesler-Plöcker.")
        return "\n".join(lines)


# ── Unit spec retrieval ────────────────────────────────────────────────────────

class UnitSpecRetriever:
    """Returns parameter spec and default values for a given unit type."""

    def __init__(self) -> None:
        self._specs: dict[str, dict] = {}
        self._built = False

    def _build(self) -> None:
        if self._built:
            return
        self._built = True
        specs: list[dict] = _load("unit_specs.json")  # type: ignore
        for s in specs:
            self._specs[s["unit_type"]] = s

    def get(self, unit_type: str) -> Optional[dict]:
        self._build()
        return self._specs.get(unit_type)

    def defaults(self, unit_type: str) -> dict:
        """Return a dict of {param_name: default_value} for optional params."""
        self._build()
        spec = self._specs.get(unit_type, {})
        return {
            p["name"]: p["default"]
            for p in spec.get("optional_params", [])
            if "default" in p
        }

    def context_for_prompt(self, unit_type: str) -> str:
        """Short structured spec string for LLM prompt injection."""
        self._build()
        spec = self._specs.get(unit_type)
        if not spec:
            return f"{unit_type}: no spec available."
        lines = [f"{unit_type}: {spec.get('description','')}"]
        req = spec.get("required_params", [])
        if req:
            lines.append("  Required: " + ", ".join(
                f"{p['name']} [{p['unit']}]" for p in req))
        opt = spec.get("optional_params", [])
        if opt:
            lines.append("  Optional: " + ", ".join(
                f"{p['name']}={p.get('default','?')} [{p['unit']}]" for p in opt))
        if spec.get("notes"):
            lines.append(f"  Note: {spec['notes']}")
        return "\n".join(lines)


# ── Unified retriever ─────────────────────────────────────────────────────────

class Retriever:
    """Single object passed to all agents that need knowledge retrieval."""

    def __init__(self) -> None:
        self.bip   = BIPRetriever()
        self.thermo = ThermoRetriever()
        self.units  = UnitSpecRetriever()

    def query_bips(
        self,
        compounds: list[str],
        model:     str,
        T_K:       Optional[float] = None,
    ) -> tuple[list[dict], list[tuple[str, str]]]:
        return self.bip.query(compounds, model, T_K)

    def select_package(
        self,
        compounds:     list[str],
        description:   str   = "",
        pressure_pa:   float = 101_325.0,
        temperature_k: float = 300.0,
        exclude:       set[str] | None = None,
    ) -> list[str]:
        return self.thermo.select(
            compounds, description, pressure_pa, temperature_k,
            exclude, self.bip)

    def unit_context(self, unit_type: str) -> str:
        return self.units.context_for_prompt(unit_type)

    def thermo_context(self, compounds: list[str]) -> str:
        return self.thermo.context_for_prompt(compounds)
