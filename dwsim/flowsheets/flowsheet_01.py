"""
Flowsheet 01: FEED → Heater → Flash
Topology: FEED → HT-01 → HOT → V-01 → VAP (vapour)
                                      → LIQ (liquid)
Validates dwsim_wrapper.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dwsim.dwsim_wrapper import DWSIMFlowsheet

sim = DWSIMFlowsheet()
sim.add_compounds(["Methanol", "Water"])
sim.set_property_package("Raoult's Law")

# Objects (x, y positions are cosmetic only)
sim.add_stream("FEED",  x=50,  y=150)
sim.add_unit  ("HT-01", "Heater", x=200, y=150)
sim.add_stream("HOT",   x=350, y=150)
sim.add_unit  ("V-01",  "Vessel", x=500, y=150)
sim.add_stream("VAP",   x=650, y=50)
sim.add_stream("LIQ",   x=650, y=250)

# Connections
sim.connect("FEED",  "HT-01")
sim.connect("HT-01", "HOT")
sim.connect("HOT",   "V-01")
sim.connect("V-01",  "VAP", src_port=0)
sim.connect("V-01",  "LIQ", src_port=1)

# Conditions
sim.set_stream("FEED", T=298.15, P=101325.0, flow=1.0,
               composition={"Methanol": 0.5, "Water": 0.5})
sim.set_heater("HT-01", T_out=350.0, dP=0.0)
sim.set_vessel("V-01",  dP=0.0)

# Solve
results = sim.solve()
print(f"Solved: {results['solved']}")
print(f"Errors: {results['errors'] or 'none'}")

for tag in ("FEED", "HOT", "VAP", "LIQ"):
    s = results[tag]
    print(f"\n── {tag} ──")
    print(f"  T [K]:      {s['T_K']:.2f}")
    print(f"  P [Pa]:     {s['P_Pa']:.1f}")
    print(f"  Flow [mol/s]: {s['flow_mol_s']:.4f}")
    print(f"  Composition: {s['composition']}")
