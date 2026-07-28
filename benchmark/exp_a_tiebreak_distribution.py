"""
EXPERIMENT A — quantify the LLM thermo-tiebreak non-determinism (stage 3 only, no DWSIM).

For each reported case: compute the deterministic candidate list and candidates[0],
then run the REAL LLM tiebreak (agents.stage3.thermo_components.ThermoLLMFallback,
post-CHANGE-2 candidate validation) N times and record the distribution of picks.

Why HPC only: the tiebreak's attempt-0 call is temperature 0 (greedy), but
qwen3:30b-a3b's MoE routing is non-deterministic at temp 0 (agents/llm.py:retry_seed
docstring). The local 14B is dense and would understate the variance. Run this where
OLLAMA_BASE_URL points at the 30B.

Run:  EXP_A_N=10 EXP_A_MODEL=qwen3:30b-a3b PYTHONPATH=. python3.9 benchmark/exp_a_tiebreak_distribution.py
"""
import os, re, json, glob, time
from collections import Counter
from types import SimpleNamespace

from rag.retriever import ThermoRetriever, BIPRetriever, Retriever
from agents.stage3.thermo_components import ThermoLLMFallback
from agents.llm import DEFAULT_MODEL, get_call_count, reset_call_count

N     = int(os.environ.get("EXP_A_N", "10"))
MODEL = os.environ.get("EXP_A_MODEL", DEFAULT_MODEL)

REPORTED_30 = (["C1", "C2", "C3", "EASY_01", "EASY_02", "EASY_04", "F1", "F2", "F3",
                "F4", "GEN_01", "GEN_03", "M1", "P1", "P2", "P3", "S1", "S2",
                "SAN_03", "SAN_04"] + [f"VAL_{i:02d}" for i in range(1, 11)])

# Ideality rank for the bias test: Raoult's Law is the ideal-solution model (most
# ideal); every non-ideal model (activity or EOS) ranks below it. A divergence is
# "more ideal" if the LLM moved toward Raoult's, "less ideal" if away, "lateral" if
# it swapped between two non-ideal models (NRTL<->UNIQUAC, PR<->SRK, NRTL<->PR).
def ideality(pkg: str) -> int:
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


def feed_pressure(desc: str) -> float:
    P = 101_325.0
    m = re.search(r'(\d+(?:\.\d+)?)\s*bar', desc or "", re.I)
    if m:
        P = float(m.group(1)) * 1e5
    m = re.search(r'(\d+(?:\.\d+)?)\s*atm', desc or "", re.I)
    if m:
        P = float(m.group(1)) * 101_325
    return P


def main():
    cases = load_cases()
    tr, bip = ThermoRetriever(), BIPRetriever()
    llm = ThermoLLMFallback(model=MODEL, retriever=Retriever())

    agree = 0
    diverge_more, diverge_less, diverge_lateral = [], [], []
    n_ambiguous_calls = 0
    total_calls, total_wall = 0, 0.0

    print(f"EXPERIMENT A — tiebreak distribution  (model={MODEL}, N={N}/case)\n")
    print(f"{'case':9} {'candidates':40} {'det[0]':14} pick distribution (N samples)")
    print("-" * 100)
    for cid in REPORTED_30:
        if cid not in cases:
            continue
        c = cases[cid]
        cand = tr.select(c["compounds"], c.get("description", ""),
                         feed_pressure(c.get("description", "")), 300.0, None, bip)
        det0 = cand[0] if cand else None
        if len(cand) <= 1:
            print(f"{cid:9} {str(cand):40} {str(det0):14} (unambiguous — LLM not called)")
            continue
        picks = []
        for _ in range(N):
            reset_call_count()
            g = SimpleNamespace(compounds=c["compounds"])
            t0 = time.time()
            pick = llm.select(g, c.get("description", ""), cand) or f"{det0} (fallback)"
            total_wall += time.time() - t0
            total_calls += get_call_count()
            n_ambiguous_calls += 1
            picks.append(pick)
        dist = Counter(picks)
        print(f"{cid:9} {str(cand):40} {str(det0):14} {dict(dist)}")
        for p in picks:
            base = p.replace(" (fallback)", "")
            if base == det0:
                agree += 1
            elif ideality(base) > ideality(det0):
                diverge_more.append((cid, det0, base))
            elif ideality(base) < ideality(det0):
                diverge_less.append((cid, det0, base))
            else:
                diverge_lateral.append((cid, det0, base))

    total_samples = agree + len(diverge_more) + len(diverge_less) + len(diverge_lateral)
    print("\n" + "=" * 60)
    print(f"AGGREGATE over {total_samples} ambiguous-case samples")
    print(f"  agree with candidates[0]      : {agree} ({100*agree/max(total_samples,1):.1f}%)")
    print(f"  diverge toward MORE ideal      : {len(diverge_more)}  {Counter((f,t) for _,f,t in diverge_more)}")
    print(f"  diverge toward LESS ideal      : {len(diverge_less)}  {Counter((f,t) for _,f,t in diverge_less)}")
    print(f"  diverge LATERAL (non-ideal<->) : {len(diverge_lateral)}  {Counter((f,t) for _,f,t in diverge_lateral)}")
    print(f"\n  IDEAL-BIAS TEST: more-ideal={len(diverge_more)} vs less-ideal={len(diverge_less)} "
          f"(if ~equal or less>more, no ideal bias)")
    print(f"\nCOST: {total_calls} LLM calls, {total_wall:.1f}s wall "
          f"({total_wall/max(n_ambiguous_calls,1):.2f}s per tiebreak, "
          f"{n_ambiguous_calls} tiebreaks). Removing the tiebreak (THERMO_TIEBREAK="
          f"deterministic) saves all of this.")


if __name__ == "__main__":
    main()
