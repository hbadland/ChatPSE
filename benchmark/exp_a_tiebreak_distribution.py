"""
EXPERIMENT A — LLM thermo-tiebreak non-determinism, multi-model sweep (stage 3, no DWSIM).

For each MODEL x reported case, run the REAL LLM tiebreak
(agents.stage3.thermo_components.ThermoLLMFallback, post-CHANGE-2 candidate
validation) N times and record, per case:
  - the pick distribution over N samples;
  - agreement with candidates[0]          (rule agreement);
  - agreement with the case's reference    (expected.property_package_class, pulled
    from the case definition — not hardcoded — via benchmark.package_family);
  - the directional ideal-bias split (more-ideal / less-ideal / lateral);
  - determinism: whether the N greedy calls returned identical picks.
Per model it also records LLM call-count and wall-clock. Reports per model and
aggregated across models, so architecture effects separate from a general tendency.

Why the tiebreak can vary at all: attempt-0 is temperature 0 (greedy), but
qwen3:30b-a3b's MoE routing is non-deterministic at temp 0 and not seedable
(agents/llm.py:retry_seed). Dense small models (e.g. qwen3:14b) are more
deterministic there — which is exactly the architecture effect this sweep isolates.

Run (where OLLAMA_BASE_URL serves the models):
  EXP_A_MODELS="qwen3:30b-a3b,qwen3:14b" EXP_A_N=10 \
    PYTHONPATH=. python3.9 benchmark/exp_a_tiebreak_distribution.py
"""
from __future__ import annotations

import os, re, json, glob, time
from collections import Counter
from types import SimpleNamespace

from rag.retriever import ThermoRetriever, BIPRetriever, Retriever
from agents.stage3.thermo_components import ThermoLLMFallback
from agents.llm import DEFAULT_MODEL, get_call_count, reset_call_count
from benchmark.package_family import package_to_family, family_correct

N = int(os.environ.get("EXP_A_N", "10"))
MODELS = [m.strip() for m in
          os.environ.get("EXP_A_MODELS", os.environ.get("EXP_A_MODEL", DEFAULT_MODEL)).split(",")
          if m.strip()]

REPORTED_30 = (["C1", "C2", "C3", "EASY_01", "EASY_02", "EASY_04", "F1", "F2", "F3",
                "F4", "GEN_01", "GEN_03", "M1", "P1", "P2", "P3", "S1", "S2",
                "SAN_03", "SAN_04"] + [f"VAL_{i:02d}" for i in range(1, 11)])


def ideality(pkg: str) -> int:
    # Raoult's Law is the ideal-solution model (most ideal); every non-ideal model
    # (activity or EOS) ranks below it. Used only for the directional bias split.
    return 2 if pkg == "Raoult's Law" else 1


def load_cases():
    cases = {}
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "cases", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for c in (d if isinstance(d, list) else d.get("cases", [])):
            if isinstance(c, dict) and c.get("compounds"):
                cases.setdefault(c["id"], c)
    return cases


def ref_class(case) -> str | None:
    return (case.get("expected") or {}).get("property_package_class")


def feed_pressure(desc: str) -> float:
    P = 101_325.0
    m = re.search(r'(\d+(?:\.\d+)?)\s*bar', desc or "", re.I)
    if m:
        P = float(m.group(1)) * 1e5
    m = re.search(r'(\d+(?:\.\d+)?)\s*atm', desc or "", re.I)
    if m:
        P = float(m.group(1)) * 101_325
    return P


def run_model(model, cases, tr, bip):
    """Return a stats dict for one model over the reported cases."""
    llm = ThermoLLMFallback(model=model, retriever=Retriever())
    s = dict(model=model, samples=0, rule_agree=0,
             ref_total=0, ref_agree=0,
             more=[], less=[], lateral=[],
             calls=0, wall=0.0, det_cases=0, det_consistent=0)
    print(f"\n===== MODEL: {model}  (N={N}/case) =====")
    print(f"{'case':9} {'candidates':38} {'det[0]':13} {'ref':10} {'greedy-consistent':17} distribution")
    for cid in REPORTED_30:
        if cid not in cases:
            continue
        c = cases[cid]
        cand = tr.select(c["compounds"], c.get("description", ""),
                         feed_pressure(c.get("description", "")), 300.0, None, bip)
        det0 = cand[0] if cand else None
        rc = ref_class(c)
        if len(cand) <= 1:
            print(f"{cid:9} {str(cand):38} {str(det0):13} {str(rc):10} {'(unambiguous)':17} LLM not called")
            continue
        picks = []
        for _ in range(N):
            reset_call_count()
            g = SimpleNamespace(compounds=c["compounds"])
            t0 = time.time()
            pick = llm.select(g, c.get("description", ""), cand) or f"{det0} (fallback)"
            s["wall"] += time.time() - t0
            s["calls"] += get_call_count()
            picks.append(pick)
        s["det_cases"] += 1
        consistent = len(set(picks)) == 1
        if consistent:
            s["det_consistent"] += 1
        for p in picks:
            base = p.replace(" (fallback)", "")
            s["samples"] += 1
            if base == det0:
                s["rule_agree"] += 1
            elif ideality(base) > ideality(det0):
                s["more"].append((cid, det0, base))
            elif ideality(base) < ideality(det0):
                s["less"].append((cid, det0, base))
            else:
                s["lateral"].append((cid, det0, base))
            if rc:
                s["ref_total"] += 1
                s["ref_agree"] += int(family_correct(base, rc))
        print(f"{cid:9} {str(cand):38} {str(det0):13} {str(rc):10} "
              f"{str(consistent):17} {dict(Counter(picks))}")
    return s


def summarise(s, label):
    smp = max(s["samples"], 1)
    print(f"\n--- {label} ---")
    print(f"  samples={s['samples']}  rule-agree(candidates[0]) = {s['rule_agree']} "
          f"({100*s['rule_agree']/smp:.1f}%)")
    if s["ref_total"]:
        print(f"  reference-family agree = {s['ref_agree']}/{s['ref_total']} "
              f"({100*s['ref_agree']/s['ref_total']:.1f}%)")
    print(f"  ideal-bias: MORE-ideal={len(s['more'])}  LESS-ideal={len(s['less'])}  "
          f"lateral={len(s['lateral'])}   "
          f"(bias only if MORE>LESS; report says {'MORE' if len(s['more'])>len(s['less']) else ('LESS' if len(s['less'])>len(s['more']) else 'NEITHER')})")
    if s["more"]:
        print(f"    more-ideal divergences: {Counter((f,t) for _,f,t in s['more'])}")
    if s["less"]:
        print(f"    less-ideal divergences: {Counter((f,t) for _,f,t in s['less'])}")
    if s["lateral"]:
        print(f"    lateral divergences:    {Counter((f,t) for _,f,t in s['lateral'])}")
    if s["det_cases"]:
        print(f"  greedy determinism: {s['det_consistent']}/{s['det_cases']} ambiguous cases "
              f"returned identical picks across all {N} calls")
    print(f"  cost: {s['calls']} LLM calls, {s['wall']:.1f}s wall "
          f"({s['wall']/max(s['det_cases']*N,1):.2f}s per tiebreak call)")


def main():
    cases = load_cases()
    tr, bip = ThermoRetriever(), BIPRetriever()
    all_stats = [run_model(m, cases, tr, bip) for m in MODELS]

    print("\n" + "=" * 70)
    print("PER-MODEL SUMMARY")
    for s in all_stats:
        summarise(s, s["model"])

    if len(all_stats) > 1:
        agg = dict(samples=0, rule_agree=0, ref_total=0, ref_agree=0,
                   more=[], less=[], lateral=[], calls=0, wall=0.0,
                   det_cases=0, det_consistent=0)
        for s in all_stats:
            for k in ("samples", "rule_agree", "ref_total", "ref_agree", "calls",
                      "wall", "det_cases", "det_consistent"):
                agg[k] += s[k]
            for k in ("more", "less", "lateral"):
                agg[k] += s[k]
        print("\n" + "=" * 70)
        summarise(agg, f"AGGREGATED ACROSS {len(all_stats)} MODELS")
        print("\n  (Compare per-model determinism + ideal-bias above to separate an "
              "architecture effect from a general tendency.)")


if __name__ == "__main__":
    main()
