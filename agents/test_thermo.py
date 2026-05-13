import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner import PlannerAgent
from agents.thermo import ThermoAgent
from agents import schema

planner = PlannerAgent()
thermo  = ThermoAgent()

description = (
    "Design a process to separate a 60/40 ethanol/water mixture fed at "
    "1 atm and 25°C with a molar flow of 2 mol/s. "
    "Heat it to 80°C then flash it."
)

print("── Planner ──")
flowsheet = planner.plan(description=description)
print(f"  Global package (initial): {flowsheet['property_package']}")

print("\n── Thermodynamics Agent ──")
updated, reasoning = thermo.assign(flowsheet)

print(f"  Global package (revised): {updated['property_package']}")
print(f"  Global reasoning: {reasoning['global_reasoning']}")

overrides = reasoning.get("unit_overrides", {})
if overrides:
    print("  Unit overrides:")
    for tag, info in overrides.items():
        print(f"    {tag}: {info['package']} — {info['reasoning']}")
else:
    print("  No unit-level overrides needed.")

print("\n── Updated flowsheet ──")
print(schema.to_json(updated))

errors = schema.validate(updated)
print(f"\nValidation: {'PASS' if not errors else 'FAIL'}")
for e in errors:
    print(f"  - {e}")
