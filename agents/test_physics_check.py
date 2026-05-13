import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.physics_check import physics_validate, format_issues

cases = [
    {
        "name": "Ethanol/Water with Raoult's Law (should flag azeotrope ERROR)",
        "flowsheet": {
            "compounds": ["Ethanol", "Water"],
            "property_package": "Raoult's Law",
            "streams": [{"tag": "FEED", "T": 351.0, "P": 101325.0, "flow": 1.0,
                         "composition": {"Ethanol": 0.5, "Water": 0.5}}],
            "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
            "connections": [],
        }
    },
    {
        "name": "Methane/Ethane with NRTL (should flag light gas ERROR)",
        "flowsheet": {
            "compounds": ["Methane", "Ethane"],
            "property_package": "NRTL",
            "streams": [{"tag": "FEED", "T": 200.0, "P": 5000000.0, "flow": 1.0,
                         "composition": {"Methane": 0.7, "Ethane": 0.3}}],
            "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
            "connections": [],
        }
    },
    {
        "name": "n-Butanol/Water with Raoult's Law (azeotrope + LLE ERROR)",
        "flowsheet": {
            "compounds": ["n-Butanol", "Water"],
            "property_package": "Raoult's Law",
            "streams": [{"tag": "FEED", "T": 298.15, "P": 101325.0, "flow": 1.0,
                         "composition": {"n-Butanol": 0.5, "Water": 0.5}}],
            "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
            "connections": [],
        }
    },
    {
        "name": "Ethanol/Water with NRTL (should PASS)",
        "flowsheet": {
            "compounds": ["Ethanol", "Water"],
            "property_package": "NRTL",
            "streams": [{"tag": "FEED", "T": 351.0, "P": 101325.0, "flow": 1.0,
                         "composition": {"Ethanol": 0.5, "Water": 0.5}}],
            "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
            "connections": [],
        }
    },
    {
        "name": "Compressor with Raoult's Law (should WARNING)",
        "flowsheet": {
            "compounds": ["Methanol", "Water"],
            "property_package": "Raoult's Law",
            "streams": [{"tag": "FEED", "T": 298.15, "P": 101325.0, "flow": 1.0,
                         "composition": {"Methanol": 0.5, "Water": 0.5}}],
            "units": [{"tag": "C-01", "type": "Compressor", "P_out": 500000.0}],
            "connections": [],
        }
    },
]

for case in cases:
    print(f"\n{'='*60}")
    print(f"Test: {case['name']}")
    issues = physics_validate(case["flowsheet"])
    errors   = [i for i in issues if i.severity.value == "ERROR"]
    warnings = [i for i in issues if i.severity.value == "WARNING"]
    print(f"Result: {len(errors)} error(s), {len(warnings)} warning(s)")
    print(format_issues(issues))
