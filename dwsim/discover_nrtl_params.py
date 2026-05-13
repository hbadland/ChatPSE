"""
Discover whether DWSIM's NRTL property package exposes binary interaction
parameters for runtime mutation via the automation API.

Run inside the Docker container:
    python3.9 dwsim/discover_nrtl_params.py

Prints:
  1. All .NET properties on the NRTL package object (reflection)
  2. The structure of InteractionParameters (if it exists)
  3. A mutation probe — tries to set tau_12 for a known pair and reads it back
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

print("=" * 70)
print("  NRTL package type:", pkg.GetType().FullName)
print("=" * 70)

# ── 1. All public .NET properties ────────────────────────────────────────────
print("\n── .NET public properties (reflection) ──")
t = pkg.GetType()
for p in t.GetProperties():
    try:
        val = p.GetValue(pkg)
        print(f"  {p.Name:<45} = {str(val)[:80]}")
    except Exception as e:
        print(f"  {p.Name:<45} = <error: {str(e)[:50]}>")

# ── 2. Probe InteractionParameters ───────────────────────────────────────────
print("\n── InteractionParameters probe ──")
ip_prop = t.GetProperty("InteractionParameters")
if ip_prop is None:
    print("  InteractionParameters property: NOT FOUND")
else:
    ip = ip_prop.GetValue(pkg)
    print(f"  InteractionParameters type: {ip.GetType().FullName}")
    print(f"  Keys count: {ip.Count}")
    for key in ip.Keys:
        print(f"  Key: '{key}'")
        inner = ip[key]
        print(f"    inner type: {inner.GetType().FullName}")
        print(f"    inner keys: {list(inner.Keys)[:10]}")
        for key2 in inner.Keys:
            entry = inner[key2]
            print(f"    [{key}][{key2}] type: {entry.GetType().FullName}")
            # Inspect fields/properties of the entry
            et = entry.GetType()
            for ep in et.GetProperties():
                try:
                    ev = ep.GetValue(entry)
                    print(f"      .{ep.Name} = {ev}")
                except Exception:
                    print(f"      .{ep.Name} = <unreadable>")
            break  # one entry is enough to understand the structure
        break  # one outer key is enough

# ── 3. Mutation probe ─────────────────────────────────────────────────────────
print("\n── Mutation probe (try to set tau_12 for Ethanol/Water) ──")
ip_prop2 = t.GetProperty("InteractionParameters")
if ip_prop2 is not None:
    ip2 = ip_prop2.GetValue(pkg)
    # Discover what field names hold the interaction parameters
    try:
        outer_key = list(ip2.Keys)[0]
        inner     = ip2[outer_key]
        inner_key = list(inner.Keys)[0]
        entry     = inner[inner_key]
        et        = entry.GetType()
        print(f"  Attempting mutations on [{outer_key}][{inner_key}]:")
        for candidate_name in ["A12", "A21", "alpha12", "Alpha12",
                                "NRTL_A12", "NRTL_A21", "NRTL_alpha12",
                                "kij", "tau12", "tau21"]:
            ep = et.GetProperty(candidate_name)
            if ep is not None and ep.CanWrite:
                old_val = ep.GetValue(entry)
                ep.SetValue(entry, System.Double(99.999))
                new_val = ep.GetValue(entry)
                writable = "WRITABLE ✓" if abs(float(new_val) - 99.999) < 0.01 else "SET FAILED"
                ep.SetValue(entry, old_val)   # restore
                print(f"  {candidate_name:20s}: old={old_val}  after set={new_val}  → {writable}")
            elif ep is not None:
                print(f"  {candidate_name:20s}: exists but READ-ONLY")
            else:
                print(f"  {candidate_name:20s}: property not found")
    except Exception as e:
        print(f"  Mutation probe failed: {e}")
else:
    print("  Skipped — InteractionParameters not accessible")

print("\nDone.")
