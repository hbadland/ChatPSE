"""
Third-pass NRTL discovery: inspect NRTL_IPData and test live parameter injection.

Run inside the Docker container:
    python3.9 dwsim/discover_nrtl_params3.py
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

import System
from DWSIM.Automation import Automation3
from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import NRTL_IPData

auto = Automation3()
sim  = auto.CreateFlowsheet()
sim.AddCompound("Ethanol")
sim.AddCompound("Water")
pkg  = sim.CreateAndAddPropertyPackage("NRTL")

muni     = pkg.GetType().GetProperty("m_uni").GetValue(pkg)
ip_prop  = muni.GetType().GetProperty("InteractionParameters")
ip       = ip_prop.GetValue(muni)

# ── 1. Show all outer keys (compound names as DWSIM knows them) ───────────────
print("=" * 70)
print("  InteractionParameters outer keys (first 30)")
print("=" * 70)
keys = list(ip.Keys)
for k in keys[:30]:
    inner = ip[k]
    inner_keys = list(inner.Keys)[:5]
    print(f"  '{k}' → inner keys: {inner_keys}")

print(f"\n  Total outer keys: {len(keys)}")

# ── 2. Check if Ethanol / Water appear as keys ────────────────────────────────
print("\n── Ethanol / Water key search ──")
for name in ["Ethanol", "Water", "ethanol", "water", "ETHANOL", "WATER",
             "Etanol", "Agua"]:
    found = ip.ContainsKey(name)
    print(f"  '{name}': {'FOUND' if found else 'not found'}")

# ── 3. Inspect NRTL_IPData fields ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  NRTL_IPData structure")
print("=" * 70)
# Grab one existing entry
first_key  = keys[0]
inner_dict = ip[first_key]
inner_key  = list(inner_dict.Keys)[0]
entry      = inner_dict[inner_key]
et         = entry.GetType()
print(f"  Sample entry: [{first_key}][{inner_key}]")
print(f"  Type: {et.FullName}")

print("\n  Properties:")
for p in et.GetProperties():
    try:
        val = p.GetValue(entry)
        print(f"    {p.Name:<30} CanWrite={p.CanWrite}  val={val}")
    except Exception as e:
        print(f"    {p.Name:<30} <error: {e}>")

print("\n  Fields:")
for f in et.GetFields():
    try:
        val = f.GetValue(entry)
        print(f"    {f.Name:<30} val={val}")
    except Exception as e:
        print(f"    {f.Name:<30} <error: {e}>")

# ── 4. Construct and inject a new NRTL_IPData entry ───────────────────────────
print("\n" + "=" * 70)
print("  Injection test: add Ethanol/Water parameters")
print("=" * 70)

# Create a new NRTL_IPData object
new_entry = NRTL_IPData()
print(f"  Created NRTL_IPData(): {new_entry}")

# Inspect default field values
print("\n  Default field values on new entry:")
for p in et.GetProperties():
    try:
        val = p.GetValue(new_entry)
        print(f"    {p.Name:<30} = {val}")
    except Exception:
        pass

# Set plausible NRTL parameters for Ethanol/Water from literature
# τ_12 = 3.4578, τ_21 = -0.8009, α = 0.3 (source: Gmehling, 1977)
print("\n  Setting parameters on new entry...")
for p in et.GetProperties():
    if not p.CanWrite:
        continue
    name = p.Name
    if name in ("A12", "alpha12", "Alpha12", "NRTL_A12", "tau12", "Tau12", "a12"):
        p.SetValue(new_entry, System.Double(3.4578))
        print(f"    Set {name} = 3.4578")
    elif name in ("A21", "alpha21", "Alpha21", "NRTL_A21", "tau21", "Tau21", "a21"):
        p.SetValue(new_entry, System.Double(-0.8009))
        print(f"    Set {name} = -0.8009")
    elif name in ("alpha", "Alpha", "alpha12", "cij", "C12", "C"):
        p.SetValue(new_entry, System.Double(0.3))
        print(f"    Set {name} = 0.3")

# Inject into the InteractionParameters dictionary
print("\n  Injecting into m_uni.InteractionParameters...")
import System.Collections.Generic as SCG

# Ensure outer key "Ethanol" exists
if not ip.ContainsKey("Ethanol"):
    inner_new = System.Collections.Generic.Dictionary[System.String,
        DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData]()
    ip["Ethanol"] = inner_new
    print("  Created outer key 'Ethanol'")

ethanol_dict = ip["Ethanol"]
ethanol_dict["Water"] = new_entry
print("  Set ip['Ethanol']['Water'] = new_entry")

# Read back
readback = ip["Ethanol"]["Water"]
print(f"  Read back type: {readback.GetType().FullName}")
for p in et.GetProperties():
    try:
        val = p.GetValue(readback)
        print(f"    {p.Name} = {val}")
    except Exception:
        pass

# ── 5. Now solve a flash and see if the injected params are used ──────────────
print("\n" + "=" * 70)
print("  Flash test with injected parameters")
print("=" * 70)
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

sim.AddObject(ObjectType.MaterialStream, 0, 0, "FEED")
sim.AddObject(ObjectType.MaterialStream, 200, 0, "VAP")
sim.AddObject(ObjectType.MaterialStream, 200, 100, "LIQ")
sim.AddObject(ObjectType.Vessel, 100, 0, "V-01")

def go(tag):
    go_dict = sim.GraphicObjects
    for k in go_dict.Keys:
        if str(go_dict[k].Tag) == tag:
            return go_dict[k]
    raise KeyError(tag)

sim.ConnectObjects(go("FEED"), go("V-01"), 0, 0)
sim.ConnectObjects(go("V-01"), go("VAP"),  0, 0)
sim.ConnectObjects(go("V-01"), go("LIQ"),  1, 0)

feed = sim.GetFlowsheetSimulationObject("FEED")
feed.SetPropertyValue("PROP_MS_0", 353.15)   # 80°C
feed.SetPropertyValue("PROP_MS_1", 101325.0)
feed.SetPropertyValue("PROP_MS_2", 1.0)
ic = feed.GetType().GetProperty("InputComposition").GetValue(feed)
ic["Ethanol"] = 0.5
ic["Water"]   = 0.5

# Disable auto-estimation so only our injected params are used
pkg.GetType().GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(
    pkg, System.Boolean(False))

errors = auto.CalculateFlowsheet2(sim)
solved = bool(sim.Solved)
print(f"  Solved: {solved}  Errors: {list(errors) if errors else []}")

if solved:
    for tag in ("VAP", "LIQ"):
        obj = sim.GetFlowsheetSimulationObject(tag)
        flow = float(obj.GetPropertyValue("PROP_MS_2"))
        phases = obj.GetType().GetProperty("Phases").GetValue(obj)
        ph0 = phases[0]
        comps = ph0.GetType().GetProperty("Compounds").GetValue(ph0)
        comp_dict = {}
        for cname in ["Ethanol", "Water"]:
            try:
                c    = comps[cname]
                mf   = c.GetType().GetProperty("MoleFraction").GetValue(c)
                comp_dict[cname] = float(mf)
            except Exception:
                pass
        print(f"  {tag}: flow={flow:.4f} mol/s  comp={comp_dict}")

print("\nDone.")
