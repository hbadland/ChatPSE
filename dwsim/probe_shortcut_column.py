"""
STAGE A PROBE — DWSIM shortcut distillation column (Fenske-Underwood-Gilliland).

Gate for the distillation-column schema expansion: does DWSIM expose a SHORTCUT
column, does feed -> column -> distillate + bottoms converge cleanly in isolation,
and what inputs does it require? Also probes a liquid-liquid (decanter) flash.
Runs inside the DWSIM container. Does NOT touch the IR / extraction.
"""
import sys
from dwsim.dwsim_wrapper import DWSIMFlowsheet   # imports clr + loads DWSIM assemblies
import System
from System.Reflection import BindingFlags
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

_F = BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public


def _connectors(go, kind):
    conns = getattr(go, kind)
    out = []
    for i in range(conns.Count):
        c = conns[i]
        out.append((i, str(getattr(c, "Type", "?")), bool(getattr(c, "IsAttached", False))))
    return out


def probe_shortcut():
    print("\n================ SHORTCUT COLUMN PROBE ================")
    fs = DWSIMFlowsheet()
    fs.add_compounds(["Benzene", "Toluene"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim

    for t in ("FEED", "DIST", "BOT"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, t)
    sim.AddObject(ObjectType.EnergyStream, 0, 0, "COND-Q")
    sim.AddObject(ObjectType.EnergyStream, 0, 0, "REB-Q")
    sim.AddObject(ObjectType.ShortcutColumn, 0, 0, "SC-01")

    fs.set_stream("FEED", 365.0, 101325.0, 1.0, {"Benzene": 0.5, "Toluene": 0.5})
    fs.set_unit_property_package("SC-01", "Peng-Robinson (PR)")

    sc_go = fs._get_graphic_object("SC-01")
    print("SC-01 InputConnectors :", _connectors(sc_go, "InputConnectors"))
    print("SC-01 OutputConnectors:", _connectors(sc_go, "OutputConnectors"))

    # Connect per the discovered ports: input 0 = material feed (ConIn),
    # input 1 = energy (ConEn); output 0 = distillate, output 1 = bottoms.
    fs.connect("FEED", "SC-01", 0, 0)      # material feed  → input port 0
    fs.connect("REB-Q", "SC-01", 0, 1)     # energy stream  → input port 1 (ConEn)
    fs.connect("SC-01", "DIST", 0, 0)      # distillate      ← output port 0
    fs.connect("SC-01", "BOT", 1, 0)       # bottoms         ← output port 1

    obj = sim.GetFlowsheetSimulationObject("SC-01")
    t = obj.GetType()
    def setf(name, val, is_str=False):
        f = t.GetField(name, _F)
        f.SetValue(obj, System.String(val) if is_str else System.Double(float(val)))
    setf("m_lightkey", "Benzene", is_str=True)
    setf("m_heavykey", "Toluene", is_str=True)
    setf("m_lightkeymolarfrac", 0.02)   # light key mole frac allowed in BOTTOMS
    setf("m_heavykeymolarfrac", 0.02)   # heavy key mole frac allowed in DISTILLATE
    setf("m_refluxratio", 1.5)          # R / Rmin
    setf("m_condenserpressure", 101325.0)
    setf("m_boilerpressure", 101325.0)
    print("inputs set: LK=Benzene HK=Toluene LKx_bot=0.02 HKx_dist=0.02 "
          "R/Rmin=1.5 Pcond=Preb=101325 Pa")

    res = fs.solve(timeout=120)
    print("solve outcome:", res.get("success", res))
    err = getattr(obj, "ErrorMessage", "")
    print("column Calculated:", getattr(obj, "Calculated", "?"), " ErrorMessage:", err or "(none)")
    # Fenske/Underwood outputs
    for fld in ("m_Nmin", "m_Rmin"):
        try: print(f"  {fld} =", t.GetField(fld, _F).GetValue(obj))
        except Exception: pass
    for s in ("DIST", "BOT"):
        try: print(f"  {s}:", fs.get_stream(s))
        except Exception as e: print(f"  {s}: <read failed: {e}>")


def probe_decanter():
    print("\n================ LIQUID-LIQUID (DECANTER) PROBE ================")
    fs = DWSIMFlowsheet()
    fs.add_compounds(["Water", "Benzene"])   # classic immiscible LLE pair (DWSIM sample)
    fs.set_property_package("UNIQUAC")
    sim = fs._sim
    for t in ("FEED", "L1", "L2", "VAP"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, t)
    sim.AddObject(ObjectType.Vessel, 0, 0, "DEC-01")
    fs.set_stream("FEED", 320.0, 101325.0, 1.0, {"Water": 0.5, "Benzene": 0.5})
    fs.set_unit_property_package("DEC-01", "UNIQUAC")

    dec_go = fs._get_graphic_object("DEC-01")
    print("DEC-01 OutputConnectors:", _connectors(dec_go, "OutputConnectors"))
    obj = sim.GetFlowsheetSimulationObject("DEC-01")
    # Try to force a VLLE / LLE flash so two liquid phases split
    for tag in ("NestedLoopsSVLLE", "GibbsMinimization3P", "NestedLoops3P",
                "NestedLoops3PV3"):
        try:
            obj.PreferredFlashAlgorithmTag = tag
            print("  set flash algorithm:", tag); break
        except Exception:
            continue
    fs.connect("FEED", "DEC-01", 0, 0)
    fs.connect("DEC-01", "VAP", 0, 0)   # output 0 = vapour
    fs.connect("DEC-01", "L1", 1, 0)    # output 1 = liquid phase 1
    fs.connect("DEC-01", "L2", 2, 0)    # output 2 = liquid phase 2
    res = fs.solve(timeout=120)
    print("solve outcome:", res.get("success", res))
    print("DEC-01 Calculated:", getattr(obj, "Calculated", "?"),
          " Error:", getattr(obj, "ErrorMessage", "") or "(none)")
    for s in ("L1", "L2"):
        try: print(f"  {s}:", fs.get_stream(s))
        except Exception as e: print(f"  {s}: <read failed: {e}>")


if __name__ == "__main__":
    try:
        probe_shortcut()
    except Exception as e:
        import traceback; print("SHORTCUT PROBE EXCEPTION:", e); traceback.print_exc()
    try:
        probe_decanter()
    except Exception as e:
        import traceback; print("DECANTER PROBE EXCEPTION:", e); traceback.print_exc()
