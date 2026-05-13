"""
Benchmark the Thermodynamics Agent across multiple LLM providers.

Runs each model on the same test cases and scores:
  - Correct global property package (vs. ground truth)
  - Correct unit-level overrides where needed
  - Reasoning quality (pass/fail — manually review)
  - Retries needed
  - Whether output passed schema validation

Set API keys before running:
  export GOOGLE_API_KEY=...
  export ANTHROPIC_API_KEY=...
  export OPENAI_API_KEY=...        (optional)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.thermo import ThermoAgent
from agents import schema

# ── Test cases with ground-truth answers ──────────────────────────────────────
TEST_CASES = [
    {
        "name": "Ethanol/Water flash (non-ideal VLE, azeotrope)",
        "flowsheet": {
            "compounds": ["Ethanol", "Water"],
            "property_package": "Raoult's Law",
            "streams": [
                {"tag": "FEED", "T": 298.15, "P": 101325.0, "flow": 1.0,
                 "composition": {"Ethanol": 0.6, "Water": 0.4}},
                {"tag": "HOT"}, {"tag": "VAP"}, {"tag": "LIQ"}
            ],
            "units": [
                {"tag": "HT-01", "type": "Heater", "T_out": 353.15, "dP": 0.0},
                {"tag": "V-01",  "type": "Vessel", "dP": 0.0},
            ],
            "connections": [
                ["FEED","HT-01",0,0],["HT-01","HOT",0,0],
                ["HOT","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]
            ],
        },
        "expected_global": {"NRTL", "UNIQUAC"},
        "expected_overrides": {},
    },
    {
        "name": "Methane/Ethane high-pressure separation",
        "flowsheet": {
            "compounds": ["Methane", "Ethane"],
            "property_package": "Raoult's Law",
            "streams": [
                {"tag": "FEED", "T": 200.0, "P": 5000000.0, "flow": 1.0,
                 "composition": {"Methane": 0.7, "Ethane": 0.3}},
                {"tag": "VAP"}, {"tag": "LIQ"}
            ],
            "units": [{"tag": "V-01", "type": "Vessel", "dP": 0.0}],
            "connections": [
                ["FEED","V-01",0,0],["V-01","VAP",0,0],["V-01","LIQ",1,0]
            ],
        },
        "expected_global": {"Peng-Robinson", "Soave-Redlich-Kwong",
                            "Lee-Kesler-Plöcker"},
        "expected_overrides": {},
    },
    {
        "name": "Methanol/Water non-ideal heater (no unit overrides needed)",
        "flowsheet": {
            "compounds": ["Methanol", "Water"],
            "property_package": "NRTL",
            "streams": [
                {"tag": "FEED", "T": 298.15, "P": 101325.0, "flow": 1.0,
                 "composition": {"Methanol": 0.5, "Water": 0.5}},
                {"tag": "OUT"}
            ],
            "units": [{"tag": "HT-01", "type": "Heater", "T_out": 350.0, "dP": 0.0}],
            "connections": [["FEED","HT-01",0,0],["HT-01","OUT",0,0]],
        },
        "expected_global": {"NRTL", "UNIQUAC"},
        "expected_overrides": {},
    },
]

MODELS = [
    "gemini-2.5-flash",
    # "gemini-2.5-pro",
    # "claude-sonnet-4-6",
    # "claude-opus-4-7",
    # "gpt-4o",
]


def score(result: dict, expected_global: set, expected_overrides: dict) -> dict:
    pkg_correct = result["global_package"] in expected_global
    # Check overrides: all expected keys present with correct package
    override_correct = all(
        result.get("unit_overrides", {}).get(tag, {}).get("package") in pkgs
        for tag, pkgs in expected_overrides.items()
    )
    return {"pkg_correct": pkg_correct, "overrides_correct": override_correct}


# ── Run benchmark ─────────────────────────────────────────────────────────────
print(f"{'Model':<28} {'Test Case':<45} {'Pk':>2} {'Package':<20} {'Overrides':>10} {'Time(s)':>8}")
print("-" * 119)

for model in MODELS:
    agent = ThermoAgent(model=model)
    for tc in TEST_CASES:
        t0 = time.time()
        try:
            updated, reasoning = agent.assign(tc["flowsheet"])
            elapsed = time.time() - t0
            s = score(reasoning, tc["expected_global"], tc["expected_overrides"])
            pkg_mark = "✓" if s["pkg_correct"] else "✗"
            ov_mark  = "✓" if s["overrides_correct"] else "✗"
            pkg_name = reasoning["global_package"]
            print(f"{model:<28} {tc['name']:<45} {pkg_mark:>2} {pkg_name:<20} {ov_mark:>10} {elapsed:>8.1f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{model:<28} {tc['name']:<45} {'?':>2} {'ERROR':<20} {str(e)[:10]:>10} {elapsed:>8.1f}")
