"""
STEP 2 addition — corpus-wide resolution assertion (offline; no DWSIM runtime).

Every compound in every BIP record must resolve to a real DWSIM compound name via
the SAME deterministic normaliser the pipeline uses at compound addition
(agents.compound_normalize.canonicalize_compound). If a record cannot be resolved,
its parameters would silently no-op at injection (set_nrtl/uniquac_parameters) — the
per-case fail-loud only fires when a case happens to use that record, whereas this
catches all of them at once and prevents silent regression when records are added.

Run: PYTHONPATH=. python3.9 agents/test_bip_corpus_resolves.py
"""
import json
import os

from agents.compound_normalize import canonicalize_compound, _dwsim_index

_CORPUS = os.path.join(os.path.dirname(__file__), "..", "rag", "sources",
                       "binary_parameters.json")


def _resolves(name: str) -> bool:
    """True iff `name` normalises to a real DWSIM compound key."""
    canon = canonicalize_compound(name)[0]
    return canon.lower() in _dwsim_index()


def audit():
    corpus = json.load(open(_CORPUS))
    failures = []  # (model, compound_a, compound_b, unresolved_names)
    for r in corpus:
        bad = [c for c in (r["compound_a"], r["compound_b"]) if not _resolves(c)]
        if bad:
            failures.append((r["model"], r["compound_a"], r["compound_b"], bad))
    return corpus, failures


# Compounds genuinely absent from DWSIM 9.0.4's compound DB (verified by CAS: no
# entry under any synonym). Their BIP records cannot be simulated regardless of the
# name fix, so they are expected failures — the test guards against NEW unresolvable
# records slipping in, not against these known dead ones.
KNOWN_ABSENT = {"Morpholine", "Furfuryl Alcohol"}


def test_all_bip_records_resolve():
    corpus, failures = audit()
    passed = len(corpus) - len(failures)
    print(f"BIP corpus resolution: {passed}/{len(corpus)} records resolve "
          f"({len(failures)} fail)")
    unexpected = []
    for model, a, b, bad in failures:
        known = all(c in KNOWN_ABSENT for c in bad)
        tag = "known-absent (DWSIM lacks compound)" if known else "REGRESSION"
        print(f"  [{model}] {a} / {b}  -> UNRESOLVED {bad}  [{tag}]")
        if not known:
            unexpected.append((model, a, b, bad))
    assert not unexpected, (
        f"{len(unexpected)} BIP records contain a compound that does not resolve to a "
        f"DWSIM name and is NOT in KNOWN_ABSENT — these silently no-op at injection.")


if __name__ == "__main__":
    test_all_bip_records_resolve()
    print("PASS")
