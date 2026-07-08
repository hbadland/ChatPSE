"""
Property-package family classification + scoring against expected labels.

Maps a concrete DWSIM property package to its thermodynamic family so the
pipeline's selection can be scored against a case's expected family label
(BenchmarkCaseSpec.expected.property_package_class, one of
"ideal" | "eos" | "activity_coefficient").

  Raoult's Law                                  -> ideal
  NRTL, UNIQUAC                                 -> activity
  Peng-Robinson, Soave-Redlich-Kwong, LKP       -> eos

Kept deliberately tiny and dependency-free so both the active runner and the
standalone scorer share ONE mapping (previously this logic lived only in the
offline adversarial harness as _package_family / _family_correct).
"""
from __future__ import annotations

_PKG_FAMILY: dict[str, str] = {
    "Raoult's Law":         "ideal",
    "NRTL":                 "activity",
    "UNIQUAC":              "activity",
    "Peng-Robinson":        "eos",
    "Soave-Redlich-Kwong":  "eos",
    "Lee-Kesler-Plöcker":   "eos",
}

# Expected-label aliases -> canonical family. The active case schema uses
# "activity_coefficient"; the offline harness used "activity". Accept both.
_EXPECTED_ALIAS: dict[str, str] = {
    "ideal":                "ideal",
    "eos":                  "eos",
    "activity":             "activity",
    "activity_coefficient": "activity",
}


def package_to_family(pkg: str | None) -> str:
    return _PKG_FAMILY.get((pkg or "").strip(), "unknown")


def normalize_expected(label: str | None) -> str | None:
    return _EXPECTED_ALIAS.get((label or "").strip().lower())


def family_correct(pkg: str | None, expected_label: str | None) -> bool:
    exp = normalize_expected(expected_label)
    return exp is not None and package_to_family(pkg) == exp


def score_family(selected_pkg: str | None, expected_label: str | None,
                 n_binary_params: int | None = None) -> dict:
    """Return the persisted package-family block for a per-run JSON."""
    return {
        "selected_package": selected_pkg or "",
        "selected_family":  package_to_family(selected_pkg),
        "expected_label":   expected_label,
        "expected_family":  normalize_expected(expected_label),
        "correct":          family_correct(selected_pkg, expected_label),
        "n_binary_params":  n_binary_params,
    }
