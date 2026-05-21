"""
Learned margin model for thermodynamic offsets (Item 4 — Issue 2 hardened).

Replaces fixed heuristics (e.g. bubble_point + 15 K) with distributions
learned from successful repair outcomes.

Issue-2 mitigations:
  Sliding window  — only the last MAX_OBS observations contribute.
                    Older experience decays naturally as the model encounters
                    new compound systems; prevents indefinite drift.
  Trimmed mean    — drop TRIM_FRAC from each tail before computing mean + 0.5σ.
                    Guards against outlier-induced overfitting when n is small.
  Hard bounds     — per-parameter floor/ceiling on the returned margin.
  Confidence gate — returns default when n < MIN_OBS; returns trimmed estimate
                    only once sufficient data exists.

Lifecycle:
    model = MarginModel()
    model.load("data/margins.json")

    model.record("Heater", "Vessel", compound_classes, margin=17.5)
    margin = model.get_margin("Heater", "Vessel", compound_classes)

    model.save("data/margins.json")
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Optional

# ── Hyperparameters ────────────────────────────────────────────────────────────

MAX_OBS    = 20      # sliding window — keep only the most recent observations
TRIM_FRAC  = 0.10   # drop this fraction from each tail (trimmed mean)
MIN_OBS    = 3      # minimum observations before overriding default

# Hard bounds by param type
_T_MARGIN_MIN = 3.0
_T_MARGIN_MAX = 70.0
_P_RATIO_MIN  = 0.05
_P_RATIO_MAX  = 4.0


class MarginModel:
    """
    Adaptive trimmed-mean margin distribution with sliding-window forgetting.
    """

    def __init__(self) -> None:
        self._data: dict[tuple, list[float]] = defaultdict(list)

    # ── Recording ──────────────────────────────────────────────────────────────

    def record(
        self,
        unit_type:        str,
        downstream_type:  Optional[str],
        compound_classes: frozenset,
        margin:           float,
        param:            str = "T_out",
    ) -> None:
        if not math.isfinite(margin):
            return

        key = _make_key(unit_type, downstream_type, compound_classes, param)
        buf = self._data[key]
        buf.append(margin)
        # Enforce sliding window
        if len(buf) > MAX_OBS:
            self._data[key] = buf[-MAX_OBS:]

        # Wildcard fallback also updated (for sparse lookups)
        if downstream_type is not None:
            wkey = _make_key(unit_type, None, compound_classes, param)
            wbuf = self._data[wkey]
            wbuf.append(margin)
            if len(wbuf) > MAX_OBS:
                self._data[wkey] = wbuf[-MAX_OBS:]

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_margin(
        self,
        unit_type:        str,
        downstream_type:  Optional[str],
        compound_classes: frozenset,
        param:            str   = "T_out",
        default:          float = 15.0,
    ) -> float:
        """
        Return trimmed-mean + 0.5*trimmed-std.

        Falls back to `default` when fewer than MIN_OBS observations.
        Hard-clamps the result to [floor, ceiling] regardless of history.
        """
        key  = _make_key(unit_type, downstream_type, compound_classes, param)
        vals = list(self._data.get(key, []))

        if len(vals) < MIN_OBS:
            wkey = _make_key(unit_type, None, compound_classes, param)
            vals = list(self._data.get(wkey, []))

        if len(vals) < MIN_OBS:
            return default

        raw = _trimmed_mean_plus_half_std(vals, TRIM_FRAC)

        if param in ("T_out",):
            return max(_T_MARGIN_MIN, min(_T_MARGIN_MAX, raw))
        if param in ("P_out", "P_out_ratio"):
            return max(_P_RATIO_MIN, min(_P_RATIO_MAX, raw))
        return raw

    def n_records(self) -> int:
        return sum(len(v) for v in self._data.values())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        serialisable = {"|".join(k): v for k, v in self._data.items()}
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            raw = json.load(f)
        for k_str, vals in raw.items():
            parts = tuple(k_str.split("|"))
            if len(parts) == 4:
                # Enforce sliding window on load too
                self._data[parts] = [float(v) for v in vals[-MAX_OBS:]]

    def snapshot(self) -> dict:
        """Export current margin estimates as a JSON-serialisable dict."""
        out = {}
        for k, vals in self._data.items():
            if not vals:
                continue
            n    = len(vals)
            mean = sum(vals) / n
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n >= 2 else 0.0
            out["|".join(k)] = {"mean": round(mean, 4), "std": round(std, 4), "n_obs": n}
        return out

    def summary(self) -> str:
        lines = [f"MarginModel: {len(self._data)} entries, {self.n_records()} records "
                 f"(window={MAX_OBS}, trim={TRIM_FRAC:.0%})"]
        for k, vals in sorted(self._data.items(), key=lambda x: -len(x[1])):
            if len(vals) >= MIN_OBS:
                est = _trimmed_mean_plus_half_std(vals, TRIM_FRAC)
                lines.append(
                    f"  {k[0]}→{k[1]} [{k[3]}] cc={k[2]}: "
                    f"n={len(vals)} est={est:.1f}"
                )
        return "\n".join(lines)


# ── Module-level singleton ─────────────────────────────────────────────────────

_GLOBAL_MARGIN_MODEL = MarginModel()


def get_global_margin_model() -> MarginModel:
    return _GLOBAL_MARGIN_MODEL


# ── Internal helpers ──────────────────────────────────────────────────────────

def _trimmed_mean_plus_half_std(vals: list[float], trim: float) -> float:
    """
    Trimmed mean + 0.5 * trimmed std.

    With n < 4, trim is skipped (not enough data to drop anything).
    """
    n = len(vals)
    if n < 4:
        # Use full set
        mean = sum(vals) / n
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        return mean + 0.5 * std

    k = max(1, int(n * trim))
    trimmed = sorted(vals)[k: n - k]
    if not trimmed:
        trimmed = vals

    mean = sum(trimmed) / len(trimmed)
    std  = math.sqrt(sum((v - mean) ** 2 for v in trimmed) / len(trimmed))
    return mean + 0.5 * std


def _make_key(
    unit_type:        str,
    downstream_type:  Optional[str],
    compound_classes: frozenset,
    param:            str,
) -> tuple:
    return (
        unit_type,
        downstream_type or "*",
        ",".join(sorted(compound_classes)) if compound_classes else "*",
        param,
    )
