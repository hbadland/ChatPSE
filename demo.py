"""
End-to-end pipeline demo using the Orchestrator.

Usage (inside container):
    PYTHONPATH=. python3 demo.py
    PYTHONPATH=. python3 demo.py --description "separate methane/ethane at 50 bar"
    PYTHONPATH=. python3 demo.py --model gemini-2.5-flash
"""
import sys
import argparse
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.orchestrator import Orchestrator

DEFAULT_DESCRIPTION = (
    "Flash separate a 50/50 molar methanol/water feed at 1 atm and 25°C. "
    "First heat the feed to 80°C, then flash it in a vessel. "
    "Report the vapour and liquid outlet compositions."
)

def _banner(text: str) -> None:
    print(f"\n{'─'*60}\n  {text}\n{'─'*60}")

def run_demo(description: str, model: str) -> None:
    print(f"\nMulti-Agent Flowsheet Pipeline")
    print(f"Model  : {model}")
    print(f"Process: {description}\n")

    orch   = Orchestrator(model=model, max_iterations=4, max_basis_reruns=1)
    result = orch.run(description)

    _banner("RESULT")
    print(result.summary())

    if result.passed and result.final_execution:
        _banner("FINAL STREAM RESULTS")
        ex = result.final_execution
        fs = result.final_flowsheet or {}
        conn_srcs = {c[0] for c in fs.get("connections", [])}
        for tag, s in ex.stream_results.items():
            marker = " [FEED]" if s.is_feed else (
                " [OUTLET]" if tag not in conn_srcs else "")
            print(f"  {tag}{marker}: "
                  f"T={s.T_C:.1f}°C  P={s.P_bar:.3f} bar  "
                  f"flow={s.flow_mol_s:.4f} mol/s  "
                  f"comp={{{', '.join(f'{k}: {v:.3f}' for k, v in s.composition.items())}}}")

    if result.basis_result and result.basis_result.concentration_hints:
        print(f"\n  Concentration hints: {result.basis_result.concentration_hints}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()
    run_demo(args.description, args.model)
