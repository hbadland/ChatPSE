"""
Hash-based state cache for IR validation results (Item 8).

Avoids redundant validation calls when beam search branches produce identical
parameter states.  The cache key is an MD5 hash of all unit parameter values
(topology is fixed during repair, so only params need to be hashed).

Usage:
    cache = StateCache()
    report = cache.cached_validate(graph)   # hits cache on second call
    cache.clear()                           # between benchmark runs
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ir.graph import FlowsheetGraph
from ir.validate import validate, ValidationReport


class StateCache:
    """
    Validation cache keyed by a hash of unit parameter states.

    Thread-safety: single-threaded access assumed (beam search is sequential).
    """

    def __init__(self) -> None:
        self._cache: dict[str, ValidationReport] = {}
        self._hits:   int = 0
        self._misses: int = 0

    def cached_validate(self, graph: FlowsheetGraph) -> ValidationReport:
        """Return cached ValidationReport if available, otherwise compute + cache."""
        h = _hash_params(graph)
        cached = self._cache.get(h)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        report = validate(graph)
        self._cache[h] = report
        return report

    def clear(self) -> None:
        self._cache.clear()
        self._hits   = 0
        self._misses = 0

    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> str:
        total = self._hits + self._misses
        rate  = self._hits / total if total > 0 else 0.0
        return (f"StateCache: {self._hits} hits / {total} total "
                f"({rate:.0%} hit rate, {self.size()} entries)")


def _hash_params(graph: FlowsheetGraph) -> str:
    """
    Stable hash of {unit_tag: {param: rounded_value}} for all units.
    Rounded to 3 decimal places to treat near-identical floats as equal.
    """
    state: dict[str, Any] = {}
    for unit in sorted(graph.units(), key=lambda u: u.tag):
        state[unit.tag] = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in sorted(unit.params.items())
        }
    raw = json.dumps(state, sort_keys=True, default=str)
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()
