import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner import PlannerAgent
from agents import schema

agent = PlannerAgent()

description = (
    "Design a process to separate a 60/40 ethanol/water mixture fed at "
    "1 atm and 25°C with a molar flow of 2 mol/s. "
    "Heat it to 80°C then flash it to separate vapour and liquid phases."
)

print("Running planner...")
flowsheet = agent.plan(description=description)
print("\nGenerated flowsheet:")
print(schema.to_json(flowsheet))

errors = schema.validate(flowsheet)
print(f"\nValidation: {'PASS' if not errors else 'FAIL'}")
for e in errors:
    print(f"  - {e}")
