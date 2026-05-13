"""
UNIQUAC parameter injection discovery script.

Mirrors discover_nrtl_params4.py for the UNIQUAC property package.
Confirms whether UNIQUAC binary interaction parameters are injectable
at runtime via the same reflection path used for NRTL.

Run inside the Docker container:
    python3.9 dwsim/discover_uniquac_params.py 2>&1

Reports:
  1. UNIQUAC auxiliary object type and its InteractionParameters structure
  2. The IPData class field names and writeability
  3. End-to-end injection + flash test (ethanol/water at 353 K, 1 atm)
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

auto = Automation3()
sim  = auto.CreateFlowsheet()
sim.AddCompound("Ethanol")
sim.AddCompound("Water")
pkg  = sim.CreateAndAddPropertyPackage("UNIQUAC")
t    = pkg.GetType()

print("=" * 70)
print(f"  UNIQUAC package type: {t.FullName}")
print("=" * 70)

# ── 1. Disable AutoEstimate immediately ───────────────────────────────────────
ae_prop = t.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters")
if ae_prop and ae_prop.CanWrite:
    ae_prop.SetValue(pkg, System.Boolean(False))
    print(f"  AutoEstimate disabled: {ae_prop.GetValue(pkg)}")
else:
    print("  AutoEstimate property not found or read-only")

# ── 2. Find the auxiliary model property (analogous to m_uni for NRTL) ────────
print("\n── Searching for auxiliary model property ──")
aux_obj  = None
aux_name = None
for p in t.GetProperties():
    try:
        val = p.GetValue(pkg)
        if val is None:
            continue
        type_name = val.GetType().FullName
        if "UNIQUAC" in type_name and "FlashAlgorithm" not in type_name:
            print(f"  Found: {p.Name} → {type_name}")
            aux_obj  = val
            aux_name = p.Name
            break
    except Exception:
        pass

if aux_obj is None:
    print("  No auxiliary UNIQUAC model property found via reflection.")
    print("  Trying common names: m_uni, m_uniquac, m_act ...")
    for name in ["m_uni", "m_uniquac", "m_act", "m_UNIQUAC"]:
        prop = t.GetProperty(name)
        if prop:
            val = prop.GetValue(pkg)
            print(f"  {name} → {val.GetType().FullName if val else 'None'}")
            if val:
                aux_obj  = val
                aux_name = name
                break

if aux_obj is None:
    print("\n  FATAL: Cannot locate UNIQUAC auxiliary object. Injection path unknown.")
    sys.exit(1)

# ── 3. InteractionParameters on the auxiliary object ──────────────────────────
print(f"\n── InteractionParameters on {aux_name} ──")
at = aux_obj.GetType()
ip_prop = at.GetProperty("InteractionParameters")
if ip_prop is None:
    print("  InteractionParameters property: NOT FOUND")
    print("  All properties on auxiliary object:")
    for p in at.GetProperties():
        try:
            val = p.GetValue(aux_obj)
            val_str = f"[count={val.Count}]" if hasattr(val, "Count") else str(val)[:60]
            print(f"    {p.Name:<40} = {val_str}")
        except Exception as e:
            print(f"    {p.Name:<40} = <error: {str(e)[:40]}>")
    sys.exit(1)

ip = ip_prop.GetValue(aux_obj)
print(f"  Type: {ip.GetType().FullName}")
print(f"  Count: {ip.Count}")
keys = list(ip.Keys)
print(f"  First 10 keys: {keys[:10]}")
print(f"  'Ethanol' in keys: {ip.ContainsKey('Ethanol')}")
print(f"  'Water'   in keys: {ip.ContainsKey('Water')}")

# ── 4. Inspect IPData structure ───────────────────────────────────────────────
print("\n── IPData structure ──")
if ip.Count > 0:
    outer_key  = keys[0]
    inner_dict = ip[outer_key]
    inner_key  = list(inner_dict.Keys)[0]
    entry      = inner_dict[inner_key]
    et         = entry.GetType()
    print(f"  Sample: [{outer_key}][{inner_key}]  type={et.FullName}")
    print("\n  Fields:")
    for f in et.GetFields():
        try:
            val = f.GetValue(entry)
            print(f"    {f.Name:<20} IsPublic={f.IsPublic}  IsInitOnly={f.IsInitOnly}  val={val}")
        except Exception as e:
            print(f"    {f.Name:<20} <error: {e}>")
    print("\n  Properties:")
    for p in et.GetProperties():
        try:
            val = p.GetValue(entry)
            print(f"    {p.Name:<20} CanWrite={p.CanWrite}  val={val}")
        except Exception:
            pass
else:
    print("  InteractionParameters is empty — no sample entry to inspect.")
    print("  Attempting to discover IPData type via assembly scan...")
    try:
        import clr as _clr
        from System.Reflection import Assembly
        asm = Assembly.LoadFrom("/usr/local/lib/dwsim/DWSIM.Thermodynamics.dll")
        for typ in asm.GetTypes():
            if "UNIQUAC" in typ.Name and "IPData" in typ.Name:
                print(f"  Found type: {typ.FullName}")
                inst = typ()
                for f in typ.GetFields():
                    print(f"    field: {f.Name}  IsPublic={f.IsPublic}  IsInitOnly={f.IsInitOnly}")
    except Exception as e:
        print(f"  Assembly scan failed: {e}")

# ── 5. Injection test ─────────────────────────────────────────────────────────
print("\n── Injection test: Ethanol/Water UNIQUAC ──")
try:
    from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import UNIQUAC_IPData
    ip_data_type = UNIQUAC_IPData
    new_entry = UNIQUAC_IPData()
    et = new_entry.GetType()

    # Literature UNIQUAC parameters for Ethanol/Water (source: DECHEMA Vol. 1)
    # u12-u22 = A12, u21-u11 = A21 (in K)
    param_set = {
        "A12": 291.27,   # u_12 - u_22 [K]
        "A21": -116.86,  # u_21 - u_11 [K]
    }

    fields_set = []
    for fname, fval in param_set.items():
        f = et.GetField(fname)
        if f and not f.IsInitOnly:
            f.SetValue(new_entry, System.Double(fval))
            readback = f.GetValue(new_entry)
            fields_set.append(f"{fname}={readback:.4f}")
            print(f"  Set {fname} = {fval} → readback = {readback}")
        else:
            # Try alternate field names
            for alt in [fname.lower(), fname.upper(), f"_{fname}"]:
                fa = et.GetField(alt)
                if fa and not fa.IsInitOnly:
                    fa.SetValue(new_entry, System.Double(fval))
                    fields_set.append(f"{alt}={fval}")
                    print(f"  Set {alt} = {fval} (alt name)")
                    break
            else:
                print(f"  WARNING: No writable field found for {fname}")

    # Inject both orderings
    if not ip.ContainsKey("Ethanol"):
        from System.Collections.Generic import Dictionary
        ip["Ethanol"] = Dictionary[System.String, UNIQUAC_IPData]()
    ip["Ethanol"]["Water"] = new_entry

    # Create reverse entry with A12↔A21 swapped
    rev_entry = UNIQUAC_IPData()
    a12_f = et.GetField("A12")
    a21_f = et.GetField("A21")
    if a12_f and a21_f:
        a12_f.SetValue(rev_entry, System.Double(param_set.get("A21", 0.0)))
        a21_f.SetValue(rev_entry, System.Double(param_set.get("A12", 0.0)))
    if not ip.ContainsKey("Water"):
        from System.Collections.Generic import Dictionary
        ip["Water"] = Dictionary[System.String, UNIQUAC_IPData]()
    ip["Water"]["Ethanol"] = rev_entry

    print(f"  Injected: {fields_set}")

    # ── Flash test ──────────────────────────────────────────────────────────
    print("\n── Flash test with injected UNIQUAC parameters ──")
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
    print(f"  Solved: {solved}  Errors: {list(errors) if errors else []}")

    if solved:
        for tag in ("VAP", "LIQ"):
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
            print(f"  {tag}: flow={flow:.4f} mol/s  comp={comp_dict}")
        print("\n  Expected: VAP Ethanol ~0.65, LIQ Ethanol ~0.40")
        print("  (Exact values depend on UNIQUAC parameters used)")

except Exception as e:
    import traceback
    print(f"  Injection test failed: {e}")
    traceback.print_exc()

print("\nDone.")
