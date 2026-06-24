"""
Reference reactor consistency verifier (auditable, re-runnable).

For each repaired reactive-case reference, this re-checks that the reactor's
reaction + conversion (read from the reference file's unit params) reproduces the
composition change already present in that reference's own stream table — the
"mass-balance consistency gate" used when the reactors were authored.

Conversion and reaction are read from the reference; the reactor→stream mapping
(which annotated streams are a reactor's inlet/outlet) is the audit's explicit
assumption, declared in _REACTOR_STREAMS below.

  VAL_03  RX-01 toluene HDA            — checkable
  VAL_04  RX-01/02/03 SMR + HT/LT WGS  — checkable
  VAL_05  RX-02 EDC pyrolysis          — checkable
          RX-01 chlorination           — OUTLET-ONLY (no ethylene/chlorine
                                         inlet stream in the reference)
  VAL_10  excluded-invalid-reference   — physics-only, no reactor check

Run: PYTHONPATH=. python3.9 benchmark/verify_reference_reactors.py
Exit code 0 iff every checkable reactor passes (<= TOL abs error).
"""
from __future__ import annotations

import json
import os
import re
import sys

TOL = 0.002  # absolute mole-fraction tolerance

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REF_DIR = os.path.join(_REPO_ROOT, "benchmark", "reference_flowsheets")

# reactor tag -> (inlet stream, outlet stream, key reactant)
_REACTOR_STREAMS: dict[str, dict[str, tuple[str, str, str]]] = {
    "VAL_03": {"RX-01": ("Hot_Mixed_Stream", "MSTR-008", "Toluene")},
    "VAL_04": {"RX-01": ("S-05",  "S-06",  "Methane"),
               "RX-02": ("S-08",  "S-09",  "Carbon monoxide"),
               "RX-03": ("S-011", "S-012", "Carbon monoxide")},
    "VAL_05": {"RX-02": ("S-04",  "S-07",  "1,2-dichloroethane")},
}
# Reactors that exist in the reference but cannot be checked against a composition
# change (no inlet stream in the reference) — verified outlet-only when authored.
_OUTLET_ONLY = {"VAL_05": {"RX-01": "no ethylene/chlorine inlet stream; outlet pure EDC consistent with 100%"}}


def _parse_rxn(s: str) -> tuple[dict, dict]:
    """'A + 2 B -> C + 3 D' -> ({A:1, B:2}, {C:1, D:3}). A coefficient is a number
    followed by whitespace, so names that start with a digit (1,2-dichloroethane)
    are preserved."""
    def side(t: str) -> dict:
        d = {}
        for term in t.split("+"):
            term = term.strip()
            m = re.match(r"^(?:(\d+(?:\.\d+)?)\s+)?(.+)$", term)
            d[m.group(2).strip()] = float(m.group(1)) if m.group(1) else 1.0
        return d
    left, right = re.split(r"->|→", s)
    return side(left), side(right)


def _check_reactor(ref: dict, rx_tag: str, inlet: str, outlet: str, key: str) -> tuple[bool, list[str]]:
    comps = ref["compounds"]
    unit = next(u for u in ref["units"] if u["tag"] == rx_tag)
    reaction = unit["params"]["reaction"]
    X = float(unit["params"]["conversion"])
    react, prod = _parse_rxn(reaction)

    sin, sout = ref["streams"][inlet], ref["streams"][outlet]
    Fin = sin["flow_mol_s"]
    nin = {c: Fin * sin["composition"].get(c, 0.0) for c in comps}
    xout = {c: sout["composition"].get(c, 0.0) for c in comps}

    xi = X * nin[key] / react[key]
    nout = {c: nin[c] + (prod.get(c, 0.0) - react.get(c, 0.0)) * xi for c in comps}
    tot = sum(nout.values())
    pred = {c: nout[c] / tot for c in comps}

    max_err = max(abs(pred[c] - xout[c]) for c in comps)
    ok = max_err <= TOL
    lines = [f"  {rx_tag}: {reaction}  (conv {X:.4f}, key {key})",
             f"    {inlet} -> {outlet}   max abs err {max_err:.4f}  "
             f"=> {'PASS' if ok else 'MISMATCH'}"]
    if not ok:
        for c in comps:
            e = abs(pred[c] - xout[c])
            if e > TOL:
                lines.append(f"      {c}: pred {pred[c]:.4f} vs ref {xout[c]:.4f} (err {e:.4f})")
    return ok, lines


def main() -> None:
    all_ok = True
    for cid in ("VAL_03", "VAL_04", "VAL_05", "VAL_10"):
        ref = json.load(open(os.path.join(_REF_DIR, f"{cid}_reference.json")))
        print(f"=== {cid} ===")
        if "excluded-invalid-reference" in str(ref.get("reference_validity", "")):
            print(f"  EXCLUDED-INVALID-REFERENCE (physics-only): "
                  f"{ref.get('reference_validity_reason','')[:90]}")
            continue
        for rx_tag, (inlet, outlet, key) in _REACTOR_STREAMS.get(cid, {}).items():
            ok, lines = _check_reactor(ref, rx_tag, inlet, outlet, key)
            all_ok = all_ok and ok
            print("\n".join(lines))
        for rx_tag, why in _OUTLET_ONLY.get(cid, {}).items():
            print(f"  {rx_tag}: OUTLET-ONLY (not checkable) — {why}")
    print(f"\n{'='*60}\n{'ALL CHECKABLE REACTORS PASS' if all_ok else 'CONSISTENCY FAILURE'}\n{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
