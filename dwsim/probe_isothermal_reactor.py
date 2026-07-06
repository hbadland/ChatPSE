"""
STAGE 3a ISOLATED PROBE — run inside the DWSIM container.

Question: can a DWSIM ConversionReactor solve with BOTH a conversion reaction AND
an enforced outlet temperature (energy stream carrying the reaction duty)?  And
what is the real operation-mode API (does it support 'define outlet temperature',
or only Isothermic/Adiabatic)?

This is self-discovering: Phase 1 reproduces the adiabatic bug, Phase 2 introspects
the reactor's operation-mode property + enum values, Phase 3 tries to enforce the
outlet T via the mode + a wired energy stream and reports whether it solves with
conversion preserved.  No source files are modified — this only probes.

Run:
  docker exec <dwsim-container> sh -c "cd /workspaces/multiAgentFlowsheet && \
      PYTHONPATH=. python3.9 dwsim/probe_isothermal_reactor.py"
"""
import sys
from dwsim.dwsim_wrapper import DWSIMFlowsheet

SET_T = 1073.15   # target steam-methane-reforming outlet temperature (~800 C)
RXN   = "Methane + Water -> Carbon monoxide + 3 Hydrogen"
P_PA  = 2_000_000.0


def _build():
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methane", "Water", "Carbon monoxide", "Hydrogen"])
    # Package is incidental to what this probe tests (the reactor operation-mode
    # API); use a confirmed-valid DWSIM key. "Peng-Robinson" is not in this build's
    # dictionary (KeyNotFoundException); "Raoult's Law" is the wrapper default.
    sim.set_property_package("Raoult's Law")
    sim.add_stream("FEED"); sim.add_stream("OUT0"); sim.add_stream("OUT1")
    sim.add_unit("RX-01", "ConversionReactor")
    sim.connect("FEED", "RX-01", 0, 0)     # feed
    sim.connect("RX-01", "OUT0", 0, 0)     # material outlet port 0
    sim.connect("RX-01", "OUT1", 1, 0)     # material outlet port 1
    sim.set_stream("FEED", T=298.15, P=P_PA, flow=1.0,
                   composition={"Methane": 0.5, "Water": 0.5,
                                "Carbon monoxide": 0.0, "Hydrogen": 0.0})
    return sim


def _report_outlets(r):
    for tg in ("OUT0", "OUT1"):
        s = r.get(tg)
        comp = {k: round(v, 3) for k, v in (s or {}).get("composition", {}).items() if v > 0.01}
        print(f"    {tg}: T={s['T_K'] if s else None} K  comp={comp}")


# ── PHASE 1: baseline (current wrapper behaviour) ─────────────────────────────
print("### PHASE 1 — baseline (reaction + OutletTemperature property, no mode/energy) ###")
sim = _build()
sim.set_conversion_reactor("RX-01", temperature_K=SET_T, pressure_Pa=P_PA,
                           conversion=0.9, reaction=RXN)
r = sim.solve()
print("solved:", r.get("solved"), "| errors:", str(r.get("errors"))[:200])
_report_outlets(r)
print(f">> EXPECT: outlet T well below {SET_T:.0f} K (adiabatic collapse) — reproduces the bug.\n")

# ── PHASE 2: introspect the operation-mode API ────────────────────────────────
print("### PHASE 2 — reactor operation-mode API discovery ###")
try:
    import System
    obj = _build()._sim.GetFlowsheetSimulationObject("RX-01")
    t = obj.GetType()
    for pname in ("ReactorOperationMode", "OperationMode", "CalcMode", "ReactorMode"):
        p = t.GetProperty(pname)
        if p is None:
            print(f"  {pname!r}: (absent)")
            continue
        try:
            vals = list(System.Enum.GetNames(p.PropertyType))
        except Exception as e:
            vals = f"<not an enum: {e}>"
        print(f"  {pname!r}: type={p.PropertyType.Name} CanWrite={p.CanWrite} values={vals}")
except Exception as e:
    print("  probe error:", e)
print()

# ── PHASE 3: enforce outlet T via operation mode + wired energy stream ────────
print("### PHASE 3 — enforce outlet T (operation mode + energy stream) ###")
sim3 = _build()
sim3.set_conversion_reactor("RX-01", temperature_K=SET_T, pressure_Pa=P_PA,
                            conversion=0.9, reaction=RXN)
try:
    import System
    obj = sim3._sim.GetFlowsheetSimulationObject("RX-01"); t = obj.GetType()

    set_mode = None
    for pname in ("ReactorOperationMode", "OperationMode"):
        p = t.GetProperty(pname)
        if p is None or not p.CanWrite:
            continue
        for val in ("OutletTemperature", "DefineOutletTemperature",
                    "Isothermic", "Isothermal"):
            try:
                p.SetValue(obj, System.Enum.Parse(p.PropertyType, val))
                set_mode = (pname, val); break
            except Exception:
                pass
        if set_mode:
            break
    print("  operation mode set:", set_mode)

    energy_ok = None
    from dwsim.dwsim_wrapper import ObjectType
    for es_name in ("EnergyStream", "OT_EnergyStream"):
        ot = getattr(ObjectType, es_name, None)
        if ot is None:
            continue
        sim3._sim.AddObject(ot, 400, 200, "EN-01")
        eg = sim3._get_graphic_object("EN-01"); rg = sim3._get_graphic_object("RX-01")
        for (src, dst, sp, dp) in [(rg, eg, 2, 0), (eg, rg, 0, 2)]:
            try:
                sim3._sim.ConnectObjects(src, dst, sp, dp)
                energy_ok = (es_name, f"{sp}->{dp}"); break
            except Exception:
                pass
        if energy_ok:
            break
    print("  energy stream wired:", energy_ok)
except Exception as e:
    print("  setup error:", e)

r = sim3.solve()
print("  solved:", r.get("solved"), "| errors:", str(r.get("errors"))[:200])
_report_outlets(r)

print(f"""
>> SUCCESS CRITERIA for Stage 3a:
   Phase 3 solved == True, an outlet stream T ~= {SET_T:.0f} K (enforced, not adiabatic),
   AND Carbon monoxide + Hydrogen present in the outlet (conversion still applied).
   Paste this whole output back; it tells us the exact mode enum + energy-stream
   wiring to bake into set_conversion_reactor.""")
