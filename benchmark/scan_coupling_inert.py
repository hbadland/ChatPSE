"""
Scan results/per_run/ for the COUPLING-INERT signature, to size the coupling fix.

Signature (per run):
  * outcome is a non-converged failure (MAX_ITER / HUMAN),
  * the terminal (last-iteration) constraint violations include a DWSIM-convergence
    failure on a Compressor / Cooler / Expander setpoint, AND
  * the coupled-repair component never engaged (coupling_boosts == [] in EVERY iter).

i.e. the repair loop adjusted thermodynamically-coupled pressure-changer/cooler
setpoints INDEPENDENTLY (beam search) without the coupled settler ever firing,
and never converged — the VAL_01 / HB_ADV_07 / HB_AMB_03 pattern.

Reports: how many CASES, which tiers, consistent (same case fails this way every
run) vs flaky (also PASSes sometimes), and a breakdown by ablation mode (the
no_coupling ablation has coupling OFF by design, so empty boosts there is expected
and is reported separately — only coupling-ENABLED modes size the fix).

Usage:  PYTHONPATH=. python3.9 benchmark/scan_coupling_inert.py
"""
import glob, json, os
from collections import defaultdict, Counter

COUPLED_TYPES = ("(Compressor)", "(Cooler)", "(Expander)")
FAIL_OUTCOMES = ("MAX_ITER", "HUMAN")


def tier_map():
    """case_id -> tier (authoritative, from the case specs)."""
    m = {}
    try:
        import benchmark.case_schema as C
        for t in C.TIERS:
            try:
                for c in C.load_tier(t):
                    m[c.id] = t
            except Exception:
                pass
    except Exception:
        pass
    return m


def is_coupled_conv_violation(v):
    s = str(v)
    return ("DWSIM convergence" in s) and any(t in s for t in COUPLED_TYPES)


def coupled_params(violations):
    """Extract the coupled-type setpoint tags from violation strings."""
    out = []
    for v in violations:
        s = str(v)
        if is_coupled_conv_violation(s):
            # e.g. 'unit:CP-01.P_out'
            for tok in s.split():
                if tok.startswith("unit:") and "." in tok:
                    out.append(tok[len("unit:"):])
                    break
    return out


def scan_run(d):
    """Return (is_signature, coupling_ever, coupled_params) for one run dict."""
    iters = d.get("iterations") or []
    outcome = (d.get("outcome") or "").upper()
    coupling_ever = any((it.get("coupling_boosts") or []) for it in iters)
    last_cv = (iters[-1].get("constraint_violations") if iters else []) or []
    cps = coupled_params(last_cv)
    sig = (outcome in FAIL_OUTCOMES) and bool(cps) and (not coupling_ever)
    return sig, coupling_ever, cps


def main():
    files = glob.glob("results/per_run/*.json")
    tiers = tier_map()

    runs = []
    bad = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            bad += 1
            continue
        sig, coup_ever, cps = scan_run(d)
        runs.append({
            "case": d.get("case_id") or "?",
            "tier": tiers.get(d.get("case_id"), "?"),
            "mode": d.get("ablation_mode") or "?",
            "model": d.get("model") or "?",
            "ts": d.get("timestamp") or "",
            "outcome": (d.get("outcome") or "").upper(),
            "sig": sig,
            "coupling_ever": coup_ever,
            "coupled_params": cps,
        })

    coupling_enabled = [r for r in runs if r["mode"] != "no_coupling"]
    n_sig_runs = sum(1 for r in coupling_enabled if r["sig"])

    print("=" * 96)
    print("COUPLING-INERT SIGNATURE SCAN")
    print("=" * 96)
    print(f"files scanned        : {len(files)} ({bad} unreadable)")
    print(f"coupling-ENABLED runs: {len(coupling_enabled)}  (excludes no_coupling ablation)")
    print(f"runs matching signature: {n_sig_runs}")

    # ── Per-case aggregation (coupling-enabled runs only) ────────────────────────
    by_case = defaultdict(list)
    for r in coupling_enabled:
        by_case[r["case"]].append(r)

    rows = []
    for case, rs in by_case.items():
        n = len(rs)
        n_sig = sum(1 for r in rs if r["sig"])
        if n_sig == 0:
            continue
        n_pass = sum(1 for r in rs if r["outcome"] == "PASS")
        modes = sorted({r["mode"] for r in rs if r["sig"]})
        params = sorted({p for r in rs if r["sig"] for p in r["coupled_params"]})
        if n_pass > 0:
            label = f"FLAKY (+{n_pass} PASS)"
        elif n_sig == n:
            label = "CONSISTENT (all runs)"
        else:
            label = f"PARTIAL ({n_sig}/{n} sig, rest other-fail)"
        rows.append({"case": case, "tier": rs[0]["tier"], "n": n, "n_sig": n_sig,
                     "n_pass": n_pass, "label": label, "modes": modes, "params": params})

    rows.sort(key=lambda x: (x["tier"], x["case"]))
    print("\n" + "-" * 96)
    print(f"CASES WITH THE SIGNATURE (>=1 coupling-enabled run): {len(rows)}")
    print("-" * 96)
    print(f"{'case':14}{'tier':14}{'runs':6}{'sig':5}{'pass':6}{'classification':24}coupled setpoints")
    for x in rows:
        print(f"  {x['case']:12}{x['tier']:14}{x['n']:<6}{x['n_sig']:<5}{x['n_pass']:<6}"
              f"{x['label']:24}{','.join(x['params'])}")

    # ── Summaries ────────────────────────────────────────────────────────────────
    print("\n" + "-" * 96)
    by_tier = Counter(x["tier"] for x in rows)
    consistent = [x for x in rows if x["label"].startswith("CONSISTENT")]
    flaky = [x for x in rows if x["label"].startswith("FLAKY")]
    partial = [x for x in rows if x["label"].startswith("PARTIAL")]
    print(f"distinct cases hitting signature : {len(rows)}")
    print(f"  consistent (every run)         : {len(consistent)}  {sorted(c['case'] for c in consistent)}")
    print(f"  partial (all-fail, mixed kind) : {len(partial)}  {sorted(c['case'] for c in partial)}")
    print(f"  flaky (also PASSes)            : {len(flaky)}  {sorted(c['case'] for c in flaky)}")
    print(f"by tier                          : {dict(by_tier)}")

    # no_coupling reference (coupling OFF by design)
    nc = [r for r in runs if r["mode"] == "no_coupling"]
    nc_cases = {r["case"] for r in nc
                if r["outcome"] in FAIL_OUTCOMES and r["coupled_params"]}
    print(f"\n[ref] no_coupling-ablation cases with coupled-conv failures (coupling OFF "
          f"by design): {len(nc_cases)}  {sorted(nc_cases)}")
    print("      -> overlap with above sizes how many the coupling component COULD "
          "help if it engaged.")


if __name__ == "__main__":
    main()
