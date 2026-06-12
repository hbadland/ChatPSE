#!/usr/bin/env python3
"""
Enrich rag/sources/compound_synonyms.json from PubChem REST API.

For every compound in dwsim_compounds.txt:
  1. Query PubChem /compound/name/{name}/synonyms/JSON
  2. Filter to human-readable names (common names, IUPAC, abbreviations, CAS numbers)
  3. Merge with existing hand-curated entries (existing entries are never removed)

Results are cached in rag/sources/.pubchem_cache.json so re-runs are free.
Rate-limits to 5 req/s as required by PubChem policy. ~315 compounds ≈ 1-2 min.

Usage:
    python3 rag/enrich_synonyms_pubchem.py                         # enrich all
    python3 rag/enrich_synonyms_pubchem.py --dry-run               # print changes, no write
    python3 rag/enrich_synonyms_pubchem.py --compound "Water,THF"  # specific compounds only
    python3 rag/enrich_synonyms_pubchem.py --force                 # re-fetch even cached entries
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

ROOT           = Path(__file__).resolve().parent.parent
COMPOUNDS_FILE = ROOT / "rag" / "sources" / "dwsim_compounds.txt"
SYNONYMS_FILE  = ROOT / "rag" / "sources" / "compound_synonyms.json"
CACHE_FILE     = ROOT / "rag" / "sources" / ".pubchem_cache.json"

_RATE_LIMIT_RPS  = 5    # PubChem policy: ≤ 5 requests/second
_MAX_SYNONYMS    = 200  # per compound — keeps lookup table small

_CAS_RE      = re.compile(r'^\d{1,7}-\d{2}-\d$')
_INCHI_RE    = re.compile(r'^InChI=')
_INCHIKEY_RE = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')
_REGISTRY_RE = re.compile(
    r'^(EINECS|ELINCS|NLP|EU\s|HSNO|UN\s|CAS\s|RTECS|NIOSH|EPA\s|'
    r'NSC\s|CCRIS|CPDB|FEMA\s|IARC|ACGIH)',
    re.IGNORECASE,
)
# Trade/brand name patterns — contain digits embedded in names in unusual ways
_BRANDNAME_RE = re.compile(r'[®™©]|\b(grade|sol\.|solution|anhydrous|reagent|'
                            r'technical|certified|acs|bp|usp|nf|ph\s*eur)\b',
                            re.IGNORECASE)


def _is_useful(s: str) -> bool:
    """
    Return False for non-human-readable synonyms.
    Keeps: common names, IUPAC names ≤ 50 chars, abbreviations, CAS numbers.
    Drops: InChI/InChIKey strings, registry IDs, brand/grade descriptors.
    """
    s = s.strip()
    if not s or len(s) > 50:
        return False
    if _INCHI_RE.match(s):
        return False
    if _INCHIKEY_RE.match(s):
        return False
    if _REGISTRY_RE.match(s):
        return False
    if _BRANDNAME_RE.search(s):
        return False
    if s.replace(' ', '').isdigit():
        return False
    return True


def _fetch_synonyms(name: str) -> Optional[list[str]]:
    """Query PubChem REST API. Returns synonym list or None (404 / network error)."""
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(name, safe="")
        + "/synonyms/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        return data["InformationList"]["Information"][0]["Synonym"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        print(f"  WARNING: HTTP {exc.code} for '{name}': {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  WARNING: fetch failed for '{name}': {exc}", file=sys.stderr)
        return None


def _load_compounds() -> list[str]:
    compounds = []
    with open(COMPOUNDS_FILE) as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                compounds.append(name)
    return compounds


def enrich(
    dry_run: bool = False,
    only: list[str] | None = None,
    force: bool = False,
) -> None:
    existing: dict[str, list[str]] = {}
    if SYNONYMS_FILE.exists():
        with open(SYNONYMS_FILE) as f:
            existing = json.load(f)

    cache: dict[str, list[str] | None] = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    compounds = _load_compounds()
    if only:
        only_set = {c.lower() for c in only}
        compounds = [c for c in compounds if c.lower() in only_set]
        if not compounds:
            print(f"No matching compounds found for: {only}")
            return

    updated = dict(existing)
    stats = {"fetched": 0, "cached": 0, "not_found": 0, "added": 0}

    for i, compound in enumerate(compounds):
        use_cache = compound in cache and not force
        if use_cache:
            raw = cache[compound]
            stats["cached"] += 1
        else:
            time.sleep(1.0 / _RATE_LIMIT_RPS)
            raw = _fetch_synonyms(compound)
            cache[compound] = raw
            if not dry_run:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f, indent=2)
            stats["fetched"] += 1

        if raw is None:
            print(f"  [{i+1:3d}/{len(compounds)}] {compound:<42} NOT IN PUBCHEM")
            stats["not_found"] += 1
            if compound not in updated:
                updated[compound] = []
            continue

        # Filter to useful synonyms, exclude the canonical name itself
        canonical_lower = compound.lower()
        useful = [
            s.strip() for s in raw
            if _is_useful(s.strip()) and s.strip().lower() != canonical_lower
        ]

        # Separate CAS numbers (put first for clarity)
        cas_list   = [s for s in useful if _CAS_RE.match(s)]
        other_list = [s for s in useful if not _CAS_RE.match(s)]

        # Merge: keep all existing hand-curated entries, add new PubChem ones
        existing_syns = existing.get(compound, [])
        existing_lower = {s.lower() for s in existing_syns}

        new_entries: list[str] = []
        seen_lower = set(existing_lower)

        # Add CAS first if not already present
        for s in cas_list:
            if s.lower() not in seen_lower:
                new_entries.append(s)
                seen_lower.add(s.lower())

        for s in other_list:
            if s.lower() not in seen_lower:
                new_entries.append(s)
                seen_lower.add(s.lower())

        combined = sorted(existing_syns + new_entries, key=str.lower)
        # Always keep CAS numbers; truncate others to keep lookup table bounded
        cas_entries   = [s for s in combined if _CAS_RE.match(s)]
        other_entries = [s for s in combined if not _CAS_RE.match(s)]
        combined = cas_entries + other_entries[:_MAX_SYNONYMS - len(cas_entries)]

        stats["added"] += len(new_entries)
        updated[compound] = combined

        tag = use_cache and "cached" or "fetched"
        note = f"+{len(new_entries)} new" if new_entries else "no change"
        print(
            f"  [{i+1:3d}/{len(compounds)}] {compound:<42} "
            f"{len(combined):3d} synonyms  ({note}, {tag})"
        )

    print(
        f"\nSummary: {stats['fetched']} fetched, {stats['cached']} cached, "
        f"{stats['not_found']} not in PubChem, {stats['added']} synonyms added"
    )

    if not dry_run:
        with open(SYNONYMS_FILE, "w") as f:
            json.dump(dict(sorted(updated.items())), f, indent=2)
        print(f"Written: {SYNONYMS_FILE}")
    else:
        print("Dry run — no files written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing files")
    parser.add_argument("--compound", metavar="NAMES",
                        help="Comma-separated list of specific compounds to enrich")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch from PubChem even if cached")
    args = parser.parse_args()

    only = [c.strip() for c in args.compound.split(",")] if args.compound else None
    enrich(dry_run=args.dry_run, only=only, force=args.force)
