"""
Deterministic compound-name normalisation → exact DWSIM keys.

LLM compound output is non-deterministic in casing/synonym (e.g. 'Carbon Monoxide'
vs DWSIM's 'Carbon monoxide'; 'Isobutylene' vs DWSIM's 'Isobutene'), which causes
hard DWSIM failures ("could not add compounds" / "key not present in the
dictionary") and run-to-run flakiness.  This module maps any variant to the exact
DWSIM key, deterministically, so the same input always yields the same key.

Two layers, explicit wins:
  1. SYNONYM_MAP — hand-verified synonym/abbreviation → exact DWSIM key (for
     non-casing aliases the case-insensitive fallback can't catch).
  2. case-insensitive fallback against the real DWSIM compound list
     (rag/sources/dwsim_compounds.txt) — fixes ANY pure casing variant.

All target keys below were verified to exist in rag/sources/dwsim_compounds.txt.
"""
from __future__ import annotations
import os
from typing import Optional

# ── Layer 1: explicit synonyms/abbreviations (lowercased key → exact DWSIM key) ──
# Verified against rag/sources/dwsim_compounds.txt.
SYNONYM_MAP: dict[str, str] = {
    # casing-only (also caught by the fallback, kept explicit for clarity/safety)
    "carbon monoxide":    "Carbon monoxide",
    "1-propanol":         "1-propanol",
    "1,2-dichloroethane": "1,2-dichloroethane",
    "methyl ethyl ketone": "Methyl ethyl ketone",
    # true synonyms (NOT casing variants — fallback would miss these)
    "isobutylene":        "Isobutene",
    "isobutene":          "Isobutene",
    "2-butanone":         "Methyl ethyl ketone",   # DWSIM has no 'Butanone' key
    "mek":                "Methyl ethyl ketone",
    "n-propanol":         "1-propanol",
    "propan-1-ol":        "1-propanol",
    "ethylene dichloride": "1,2-dichloroethane",
    "edc":                "1,2-dichloroethane",
    # chemical-formula variants: extraction sometimes emits the formula instead of
    # the name, which the DWSIM-name fallback cannot resolve. Without these, the
    # chlorination/EDC-pyrolysis stoichiometry signatures silently miss and the
    # reactor falls to the generic 623 K template.
    "cl2":                "Chlorine",
    "hcl":                "Hydrogen chloride",
    "hydrogen chloride":  "Hydrogen chloride",
    "vinyl chloride":     "Vinyl chloride",
    "vcm":                "Vinyl chloride",
    "ethene":             "Ethylene",
    # xylene isomers (DWSIM has no bare 'Xylene' key — only o/m/p)
    "o-xylene":           "o-Xylene",
    "ortho-xylene":       "o-Xylene",
    "m-xylene":           "m-Xylene",
    "meta-xylene":        "m-Xylene",
    "p-xylene":           "p-Xylene",
    "para-xylene":        "p-Xylene",
    # STEP 2: corpus BIP-record spellings → DWSIM 9.0.4 keys, verified by CAS against
    # the runtime compound DB. DWSIM stores many common solvents under machine-
    # generated names; without these the corpus name no-ops at BIP injection AND
    # fails at compound addition. All targets verified present in dwsim_compounds.txt.
    "hexane":             "n-Hexane",              # 110-54-3
    "2-propanol":         "Isopropanol",           # 67-63-0
    "dimethylformamide":  "N,n-dimethylformamide", # 68-12-2
    "propyl acetate":     "N-propyl acetate",      # 109-60-4
    "n-butanol":          "1-butanol",             # 71-36-3
    "butyl acetate":      "N-butyl acetate",       # 123-86-4
    "n-butyl acetate":    "N-butyl acetate",       # 123-86-4
    "dioxane":            "1,4-dioxane",           # 123-91-1
    "caprolactam":        "Azepan-2-One",          # 105-60-2
    "propylene glycol":   "1,2-propylene glycol",  # 57-55-6
    "isooctane":          "2,2,4-trimethylpentane",# 540-84-1
    "tert-butanol":       "2-methyl-2-propanol",   # 75-65-0
    "chlorobenzene":      "Monochlorobenzene",     # 108-90-7
    "dichloromethane":    "Bis(Chloranyl)Methane", # 75-09-2
    "butyric acid":       "N-butyric acid",        # 107-92-6
    # Morpholine (110-91-8) and Furfuryl alcohol (98-00-0) are genuinely absent from
    # DWSIM 9.0.4's compound DB — no mapping possible; their BIP records are dead.
}

# Bare 'xylene' is ambiguous; default to m-Xylene (dominant isomer in mixed
# xylenes) so the run does not hard-crash, and warn the caller to disambiguate.
_AMBIGUOUS_DEFAULT = {"xylene": "m-Xylene"}

_DWSIM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rag", "sources", "dwsim_compounds.txt")

_dwsim_lower: Optional[dict[str, str]] = None


def _dwsim_index() -> dict[str, str]:
    """lower(name) → exact DWSIM key (first occurrence wins). Cached."""
    global _dwsim_lower
    if _dwsim_lower is None:
        idx: dict[str, str] = {}
        try:
            with open(_DWSIM_PATH, encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if not name or name.startswith("#"):
                        continue
                    idx.setdefault(name.lower(), name)
        except OSError:
            pass
        _dwsim_lower = idx
    return _dwsim_lower


def canonicalize_compound(name: str) -> tuple[str, Optional[str]]:
    """
    Return (canonical_name, warning).  Unchanged + None warning if no rule applies.

    Resolution order: explicit synonym → case-insensitive DWSIM match → ambiguous
    default (with warning) → unchanged.
    """
    if not name:
        return name, None
    key = name.strip()
    low = key.lower()

    if low in SYNONYM_MAP:
        return SYNONYM_MAP[low], None

    exact = _dwsim_index().get(low)
    if exact is not None:
        return exact, None

    if low in _AMBIGUOUS_DEFAULT:
        canon = _AMBIGUOUS_DEFAULT[low]
        return canon, (f"compound '{name}' is ambiguous → defaulted to '{canon}'; "
                       "specify o-/m-/p- in the description for the exact isomer")

    return key, None


def canonicalize_list(names) -> tuple[list, list[str]]:
    """Canonicalise a list of compound names, de-duplicating while preserving order."""
    out: list = []
    warns: list[str] = []
    seen: set = set()
    for n in names:
        canon, w = canonicalize_compound(n)
        if w:
            warns.append(w)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out, warns


def canonicalize_reaction(reaction: str) -> tuple[str, list[str]]:
    """
    Canonicalise the compound names inside a stoichiometry string, e.g.
      'Ethylene + Chlorine -> 1,2-Dichloroethane'
      'Methane + Water -> Carbon Monoxide + 3 Hydrogen'
    Coefficients ('3 Hydrogen') and the '+' / '->' / '→' separators are preserved.
    """
    if not reaction:
        return reaction, []
    warns: list[str] = []

    def fix_term(term: str) -> str:
        t = term.strip()
        if not t:
            return term
        # Optional leading integer coefficient followed by a SPACE (so the comma
        # in '1,2-dichloroethane' is never mistaken for a coefficient).
        coeff = ""
        rest = t
        i = 0
        while i < len(rest) and rest[i].isdigit():
            i += 1
        if i > 0 and i < len(rest) and rest[i] == " ":
            coeff, rest = rest[:i + 1], rest[i + 1:]
        canon, w = canonicalize_compound(rest)
        if w:
            warns.append(w)
        return coeff + canon

    # Normalise the arrow, then split sides and terms.
    arrow = "->" if "->" in reaction else ("→" if "→" in reaction else None)
    if arrow is None:
        return reaction, warns
    lhs, rhs = reaction.split(arrow, 1)

    def fix_side(side: str) -> str:
        return " + ".join(fix_term(p) for p in side.split("+"))

    return f"{fix_side(lhs).strip()} {arrow} {fix_side(rhs).strip()}", warns
