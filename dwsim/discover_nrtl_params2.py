"""
Second-pass NRTL parameter discovery.

Probes:
  1. m_uni (the auxiliary NRTL object) — properties and mutation
  2. ParametersXMLString — is it writable, what format does it expect?
  3. AutoEstimate flag — can we disable it?
  4. What happens when we add compounds and check if parameters appear?

Run inside the Docker container:
    python3.9 dwsim/discover_nrtl_params2.py
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

auto = Automation3()
sim  = auto.CreateFlowsheet()
sim.AddCompound("Ethanol")
sim.AddCompound("Water")
pkg  = sim.CreateAndAddPropertyPackage("NRTL")
t    = pkg.GetType()

# ── 1. ParametersXMLString — readable? writable? what does it look like? ──────
print("=" * 70)
print("  ParametersXMLString")
print("=" * 70)
xml_prop = t.GetProperty("ParametersXMLString")
print(f"  CanRead:  {xml_prop.CanRead}")
print(f"  CanWrite: {xml_prop.CanWrite}")
xml_val = xml_prop.GetValue(pkg)
print(f"  Current value (first 300 chars): '{str(xml_val)[:300]}'")

# Trigger parameter loading by running a flash — parameters may populate after use
print("\n  Triggering parameter load via GetInteractionParameter call...")
try:
    # Some DWSIM packages expose GetInteractionParameter directly
    m = t.GetMethod("GetInteractionParameter")
    if m:
        result = m.Invoke(pkg, System.Array[System.Object](["Ethanol", "Water", "A12"]))
        print(f"  GetInteractionParameter(Ethanol, Water, A12) = {result}")
    else:
        print("  GetInteractionParameter method not found")
except Exception as e:
    print(f"  GetInteractionParameter error: {e}")

# Read ParametersXMLString again after potential load
xml_val2 = xml_prop.GetValue(pkg)
print(f"  After probe (first 500 chars): '{str(xml_val2)[:500]}'")


# ── 2. m_uni — the auxiliary NRTL object ─────────────────────────────────────
print("\n" + "=" * 70)
print("  m_uni (Auxiliary.NRTL)")
print("=" * 70)
muni_prop = t.GetProperty("m_uni")
muni = muni_prop.GetValue(pkg)
mt = muni.GetType()
print(f"  Type: {mt.FullName}")
print(f"\n  .NET public properties:")
for p in mt.GetProperties():
    try:
        val = p.GetValue(muni)
        val_str = str(val)
        # For collections, show count and first item
        if hasattr(val, 'Count'):
            inner = f"[count={val.Count}]"
            try:
                keys = list(val.Keys)[:3]
                inner += f" keys={keys}"
            except Exception:
                pass
            print(f"    {p.Name:<40} = {inner}")
        else:
            print(f"    {p.Name:<40} = {val_str[:80]}")
    except Exception as e:
        print(f"    {p.Name:<40} = <error: {str(e)[:60]}>")

print(f"\n  .NET public fields:")
for f in mt.GetFields():
    try:
        val = f.GetValue(muni)
        val_str = str(val)
        if hasattr(val, 'Count'):
            inner = f"[count={val.Count}]"
            try:
                keys = list(val.Keys)[:3]
                inner += f" keys={keys}"
            except Exception:
                pass
            print(f"    {f.Name:<40} = {inner}")
        else:
            print(f"    {f.Name:<40} = {val_str[:80]}")
    except Exception as e:
        print(f"    {f.Name:<40} = <error: {str(e)[:60]}>")

print(f"\n  Methods:")
for m in mt.GetMethods():
    if not m.Name.startswith("get_") and not m.Name.startswith("set_"):
        params = ", ".join(str(p.ParameterType) for p in m.GetParameters())
        print(f"    {m.Name}({params})")


# ── 3. AutoEstimate flag — can we disable it? ─────────────────────────────────
print("\n" + "=" * 70)
print("  AutoEstimateMissingNRTLUNIQUACParameters")
print("=" * 70)
ae_prop = t.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters")
print(f"  CanWrite: {ae_prop.CanWrite}")
print(f"  Current:  {ae_prop.GetValue(pkg)}")
if ae_prop.CanWrite:
    ae_prop.SetValue(pkg, System.Boolean(False))
    print(f"  After set False: {ae_prop.GetValue(pkg)}")
    ae_prop.SetValue(pkg, System.Boolean(True))   # restore


# ── 4. Probe via IP dictionary on m_uni ───────────────────────────────────────
print("\n" + "=" * 70)
print("  m_uni IP / interaction parameter dictionary probe")
print("=" * 70)
for field_name in ["IP", "InteractionParameters", "ip", "BIP", "kij"]:
    try:
        fi = mt.GetField(field_name)
        if fi:
            val = fi.GetValue(muni)
            print(f"  Field '{field_name}': {val.GetType().FullName}, count={val.Count}")
            # drill one level
            for key in val.Keys:
                inner = val[key]
                print(f"    [{key}] type={inner.GetType().FullName}")
                inner_t = inner.GetType()
                for p in inner_t.GetProperties():
                    try:
                        v = p.GetValue(inner)
                        print(f"      .{p.Name} = {v}")
                    except Exception:
                        pass
                break
        else:
            pi = mt.GetProperty(field_name)
            if pi:
                val = pi.GetValue(muni)
                print(f"  Property '{field_name}': {val.GetType().FullName}, count={val.Count}")
            else:
                print(f"  '{field_name}': not found")
    except Exception as e:
        print(f"  '{field_name}': error — {e}")

print("\nDone.")
