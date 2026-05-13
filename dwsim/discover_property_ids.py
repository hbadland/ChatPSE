"""
Discover the correct DWSIM property IDs for Splitter, Pump, Compressor, Expander.

Run inside the Docker container:
    python3.9 dwsim/discover_property_ids.py

Prints all available property IDs for each unit type so dwsim_wrapper.py
can be updated with correct, empirically verified strings.
"""
import sys
sys.path.append("/usr/local/lib/dwsim/")
import pythonnet
pythonnet.load("coreclr")
import clr

clr.AddReference("/usr/local/lib/dwsim/DWSIM.Automation.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Interfaces.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.FlowsheetBase.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.SharedClasses.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Thermodynamics.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.UnitOperations.dll")

from DWSIM.Automation import Automation3
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

auto = Automation3()
sim = auto.CreateFlowsheet()
sim.AddCompound("Water")
sim.CreateAndAddPropertyPackage("Raoult's Law")

UNIT_TYPES = {
    "Splitter":   ObjectType.Splitter,
    "Pump":       ObjectType.Pump,
    "Compressor": ObjectType.Compressor,
    "Expander":   ObjectType.Expander,
}


def probe_unit(name: str, ot):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    sim.AddObject(ot, 100, 100, name)
    obj = sim.GetFlowsheetSimulationObject(name)

    # ── Method 1: GetProperties() ─────────────────────────────────────────
    try:
        props = obj.GetProperties(
            getattr(obj, "GetPropertyType", lambda: None)()
            if hasattr(obj, "GetPropertyType") else None
        )
        if props:
            print("GetProperties():", list(props))
    except Exception:
        pass

    # ── Method 2: try PROP_* IDs directly ────────────────────────────────
    candidates = [
        # Splitter guesses
        "PROP_SPLIT_0", "PROP_SPLIT_1", "PROP_SPLIT_2", "PROP_SPLIT_3",
        # Pump guesses
        "PROP_PU_0", "PROP_PU_1", "PROP_PU_2",
        # Compressor guesses
        "PROP_CO_0", "PROP_CO_1", "PROP_CO_2",
        # Expander guesses
        "PROP_EX_0", "PROP_EX_1", "PROP_EX_2",
        # Generic alternates
        "PROP_BP_0", "PROP_BP_1",  # centrifugal pump
        "PROP_CM_0", "PROP_CM_1",  # compressor
        "PROP_EX_0", "PROP_EX_1",  # expander / turbine
        "PROP_SP_0", "PROP_SP_1", "PROP_SP_2",  # splitter alternate
    ]
    print("\nProbing SetPropertyValue / GetPropertyValue:")
    for pid in candidates:
        try:
            val = obj.GetPropertyValue(pid)
            print(f"  {pid:20s} GET → {val}")
        except Exception as e:
            err = str(e)
            if "not" not in err.lower() and "invalid" not in err.lower():
                print(f"  {pid:20s} GET → ERROR: {err[:60]}")

    # ── Method 3: .NET reflection — list all public properties ───────────
    print("\n.NET public properties (reflection):")
    t = obj.GetType()
    for p in t.GetProperties():
        try:
            val = p.GetValue(obj)
            print(f"  {p.Name:40s} = {str(val)[:60]}")
        except Exception:
            print(f"  {p.Name:40s} = <unreadable>")


for unit_name, ot in UNIT_TYPES.items():
    probe_unit(unit_name, ot)

print("\nDone.")
