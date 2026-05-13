"""
End-to-end test: Planner → Thermo → Executor
First full natural language → DWSIM results pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner  import PlannerAgent
from agents.thermo   import ThermoAgent
from agents.executor import Executor, pre_execution_check
from agents import schema

planner  = PlannerAgent()
thermo   = ThermoAgent()
executor = Executor()

description = (
    "Heat a 50/50 methanol/water stream at 1 atm and 25°C "
    "with a molar flow of 1 mol/s to 80°C, then flash it "
    "to separate vapour and liquid phases."
)

print("── Step 1: Planner ──────────────────────────────────")
flowsheet = planner.plan(description=description)
print(f"  Package: {flowsheet['property_package']}")
print(f"  Units:   {[u['tag'] for u in flowsheet['units']]}")
print(f"  Streams: {[s['tag'] for s in flowsheet['streams']]}")

print("\n── Step 2: Thermodynamics Agent ─────────────────────")
flowsheet, reasoning = thermo.assign(flowsheet)
print(f"  Package (final): {flowsheet['property_package']}")
print(f"  Reasoning: {reasoning['global_reasoning']}")

print("\n── Step 3: Pre-execution check ──────────────────────")
pre_errors = pre_execution_check(flowsheet)
if pre_errors:
    print("  ERRORS — aborting:")
    for e in pre_errors:
        print(f"    - {e}")
    sys.exit(1)
print("  PASS")

print("\n── Step 4: Execute ──────────────────────────────────")
result = executor.run(flowsheet)
print(f"  Solved: {result.solved}")

if result.errors:
    print(f"  Errors:")
    for e in result.errors:
        print(f"    - {e}")
if result.warnings:
    print(f"  Warnings:")
    for w in result.warnings:
        print(f"    - {w}")

print("\n── Stream Results ───────────────────────────────────")
for tag, s in result.stream_results.items():
    marker = " [FEED]" if s.is_feed else ""
    print(f"  {tag}{marker}")
    print(f"    T = {s.T_C:.2f} °C   P = {s.P_bar:.4f} bar   "
          f"flow = {s.flow_mol_s:.4f} mol/s")
    print(f"    composition = {s.composition}")
