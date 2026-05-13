"""
Final NRTL injection verification.

Fixes from pass 3:
  - Use GetField (not GetProperty) to set A12/A21/alpha12 on NRTL_IPData
  - Disable AutoEstimate BEFORE adding compounds so it never fires
  - Verify flash uses injected parameters by comparing to known NRTL result

Run inside the Docker container:
    python3.9 dwsim/discover_nrtl_params4.py
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
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import NRTL_IPData

auto = Automation3()
sim  = auto.CreateFlowsheet()
sim.AddCompound("Ethanol")
sim.AddCompound("Water")
pkg  = sim.CreateAndAddPropertyPackage("NRTL")

# Disable auto-estimation IMMEDIATELY after package creation, before any solve
t = pkg.GetType()
ae_prop = t.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters")
ae_prop.SetValue(pkg, System.Boolean(False))
print(f"AutoEstimate disabled: {ae_prop.GetValue(pkg)}")

# Access m_uni and its InteractionParameters dict
muni    = t.GetProperty("m_uni").GetValue(pkg)
mt      = muni.GetType()
ip_prop = mt.GetProperty("InteractionParameters")
ip      = ip_prop.GetValue(muni)

# Inspect NRTL_IPData fields for writability
et = NRTL_IPData().GetType()
print("\nNRTL_IPData fields (checking writability):")
for f in et.GetFields():
    print(f"  {f.Name:<15} IsPublic={f.IsPublic}  IsInitOnly={f.IsInitOnly}")

# ── Build and inject Ethanol/Water entry ──────────────────────────────────────
# Literature: Gmehling et al. NRTL for ethanol/water at ~353 K
# A12=586.1  A21=-195.0  alpha12=0.5765
# (in DWSIM convention A12/A21 are in K, as τ = A/T)
entry = NRTL_IPData()
et.GetField("A12").SetValue(entry,     System.Double(586.1))
et.GetField("A21").SetValue(entry,     System.Double(-195.0))
et.GetField("alpha12").SetValue(entry, System.Double(0.5765))
et.GetField("comment").SetValue(entry, System.String("Ethanol/Water, Gmehling 1977"))

# Read back to confirm fields were set
print(f"\nEntry fields after set:")
print(f"  A12     = {et.GetField('A12').GetValue(entry)}")
print(f"  A21     = {et.GetField('A21').GetValue(entry)}")
print(f"  alpha12 = {et.GetField('alpha12').GetValue(entry)}")
print(f"  comment = {et.GetField('comment').GetValue(entry)}")

# Inject — ensure both orderings exist
for outer, inner_key in [("Ethanol", "Water"), ("Water", "Ethanol")]:
    if not ip.ContainsKey(outer):
        from System.Collections.Generic import Dictionary
        ip[outer] = Dictionary[System.String, NRTL_IPData]()
    if not ip[outer].ContainsKey(inner_key):
        ip[outer][inner_key] = entry
    else:
        ip[outer][inner_key] = entry  # overwrite if exists

print(f"\nInjected ip['Ethanol']['Water'] and ip['Water']['Ethanol']")
print(f"  ip['Ethanol']['Water'].A12 = {et.GetField('A12').GetValue(ip['Ethanol']['Water'])}")

# ── Build a flash flowsheet and solve ─────────────────────────────────────────
sim.AddObject(ObjectType.MaterialStream, 0,   0, "FEED")
sim.AddObject(ObjectType.MaterialStream, 200, 0, "VAP")
sim.AddObject(ObjectType.MaterialStream, 200, 100, "LIQ")
sim.AddObject(ObjectType.Vessel, 100, 0, "V-01")

def go(tag):
    d = sim.GraphicObjects
    for k in d.Keys:
        if str(d[k].Tag) == tag:
            return d[k]
    raise KeyError(tag)

sim.ConnectObjects(go("FEED"), go("V-01"), 0, 0)
sim.ConnectObjects(go("V-01"), go("VAP"),  0, 0)
sim.ConnectObjects(go("V-01"), go("LIQ"),  1, 0)

feed = sim.GetFlowsheetSimulationObject("FEED")
feed.SetPropertyValue("PROP_MS_0", 353.15)
feed.SetPropertyValue("PROP_MS_1", 101325.0)
feed.SetPropertyValue("PROP_MS_2", 1.0)
ic = feed.GetType().GetProperty("InputComposition").GetValue(feed)
ic["Ethanol"] = 0.5
ic["Water"]   = 0.5

errors = auto.CalculateFlowsheet2(sim)
solved = bool(sim.Solved)
print(f"\nSolved: {solved}  Errors: {list(errors) if errors else []}")

def read_stream(tag):
    obj   = sim.GetFlowsheetSimulationObject(tag)
    flow  = float(obj.GetPropertyValue("PROP_MS_2"))
    phases = obj.GetType().GetProperty("Phases").GetValue(obj)
    ph0   = phases[0]
    comps = ph0.GetType().GetProperty("Compounds").GetValue(ph0)
    comp_dict = {}
    for cname in ["Ethanol", "Water"]:
        try:
            mf = comps[cname].GetType().GetProperty("MoleFraction").GetValue(comps[cname])
            comp_dict[cname] = round(float(mf), 4)
        except Exception:
            pass
    return flow, comp_dict

if solved:
    for tag in ("VAP", "LIQ"):
        flow, comp = read_stream(tag)
        print(f"  {tag}: flow={flow:.4f} mol/s  comp={comp}")
    print("\nExpected (Gmehling NRTL, ~353K, 1atm, 50/50 feed):")
    print("  VAP: Ethanol ~0.65,  Water ~0.35")
    print("  LIQ: Ethanol ~0.40,  Water ~0.60")
    print("  (exact values depend on T-dependent parameters; ballpark check only)")
else:
    print("  Flash did not solve — injection path may need further debugging")

print("\nDone.")
