"""
Independent, expert-specified reference construction for Tier-A capability-demo
cases. Each flowsheet is built DIRECTLY via the DWSIM wrapper from the process
description — NOT via the pipeline's extraction/IR — so the reference is an
independent ground truth, not the system's own output. Runs in the DWSIM
container. Output matches benchmark/reference_flowsheets/VAL_*_reference.json so
the existing gated reference-MAPE scoring (fully_solved + T±5K/P±5%/vf±0.05)
applies directly.

Each case builder configures compounds/PP/units/streams/connections and returns
the case metadata; the driver solves, records per-stream conditions, and writes
the reference JSON.
"""
import json, os, sys
from dwsim.dwsim_wrapper import DWSIMFlowsheet
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

_OUT_DIR = "benchmark/reference_flowsheets"


def _stream_record(s: dict) -> dict:
    """Convert a wrapper solve() stream dict to the VAL reference stream format."""
    T = s.get("T_K"); P = s.get("P_Pa")
    return {
        "T_K":            round(float(T), 3) if T is not None else None,
        "T_C":            round(float(T) - 273.15, 3) if T is not None else None,
        "P_Pa":           round(float(P), 2) if P is not None else None,
        "P_bar":          round(float(P) / 1e5, 5) if P is not None else None,
        "flow_mol_s":     s.get("flow_mol_s"),
        "vapor_fraction": s.get("vapor_fraction"),
        "composition":    dict(s.get("composition", {}) or {}),
    }


# ── Case builders (expert-specified flowsheets) ────────────────────────────────

def build_SAN_04(fs: DWSIMFlowsheet) -> dict:
    """
    SAN_04 — "Compress propane vapour from 1 bar to 10 bar, then cool the
    compressed gas to 25 C." Pure propane; EOS (Peng-Robinson).

    FEED --> CP-01 (compressor, P_out=10 bar, eta=0.75) --> COMP
             --> CL-01 (cooler, T_out=25 C, dP=0) --> PROD
    FEED: propane vapour, 25 C, 1 bar, 1.0 mol/s.
    """
    fs.add_compounds(["Propane"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "COMP", "PROD"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    sim.AddObject(ObjectType.EnergyStream, 0, 0, "CP-Q")
    fs.add_unit("CP-01", "Compressor")
    fs.add_unit("CL-01", "Cooler")

    fs.set_stream("FEED", 298.15, 100000.0, 1.0, {"Propane": 1.0})  # 25 C, 1 bar, vapour
    fs.connect("FEED", "CP-01", 0, 0)      # feed -> compressor inlet
    try:
        fs.connect("CP-Q", "CP-01", 0, 1)  # energy -> compressor energy port (if required)
    except Exception as e:
        print(f"  [note] compressor energy connect: {e}", file=sys.stderr)
    fs.connect("CP-01", "COMP", 0, 0)      # compressor outlet -> COMP
    fs.connect("COMP", "CL-01", 0, 0)      # COMP -> cooler inlet
    fs.connect("CL-01", "PROD", 0, 0)      # cooler outlet -> PROD

    fs.set_compressor("CP-01", 1_000_000.0, efficiency=0.75)   # 10 bar, 75% adiabatic eff
    fs.set_cooler("CL-01", 298.15, dP=0.0)                     # cool to 25 C, no dP

    return {
        "case_id": "SAN_04",
        "case_name": "Compress then cool propane",
        "compounds": ["Propane"],
        "property_package": "Peng-Robinson",
        "units": [
            {"tag": "CP-01", "type": "Compressor",
             "params": {"P_out": 1_000_000.0, "efficiency": 0.75}},
            {"tag": "CL-01", "type": "Cooler",
             "params": {"T_out": 298.15, "dP": 0.0}},
        ],
        "connections": [["FEED", "CP-01"], ["CP-01", "COMP"],
                        ["COMP", "CL-01"], ["CL-01", "PROD"]],
        "material_streams": ["FEED", "COMP", "PROD"],
    }


def build_EASY_04(fs: DWSIMFlowsheet) -> dict:
    """
    EASY_04 — "Compress propylene vapour from 1 bar to 12 bar, then cool the
    high-pressure gas to 30 C." Pure propylene; Peng-Robinson.
    FEED(25 C,1 bar,vap) -> CP-01(12 bar, eta=0.75) -> COMP -> CL-01(30 C) -> PROD
    """
    fs.add_compounds(["Propylene"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "COMP", "PROD"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    sim.AddObject(ObjectType.EnergyStream, 0, 0, "CP-Q")
    fs.add_unit("CP-01", "Compressor"); fs.add_unit("CL-01", "Cooler")
    fs.set_stream("FEED", 298.15, 100000.0, 1.0, {"Propylene": 1.0})
    fs.connect("FEED", "CP-01", 0, 0)
    try: fs.connect("CP-Q", "CP-01", 0, 1)
    except Exception: pass
    fs.connect("CP-01", "COMP", 0, 0); fs.connect("COMP", "CL-01", 0, 0)
    fs.connect("CL-01", "PROD", 0, 0)
    fs.set_compressor("CP-01", 1_500_000.0, efficiency=0.75)   # 15 bar (was 12; 12 didn't condense)
    fs.set_cooler("CL-01", 303.15, dP=0.0)
    return {"case_id": "EASY_04", "case_name": "Compress then cool propylene",
            "compounds": ["Propylene"], "property_package": "Peng-Robinson",
            "units": [{"tag": "CP-01", "type": "Compressor",
                       "params": {"P_out": 1_500_000.0, "efficiency": 0.75}},
                      {"tag": "CL-01", "type": "Cooler",
                       "params": {"T_out": 303.15, "dP": 0.0}}],
            "connections": [["FEED", "CP-01"], ["CP-01", "COMP"],
                            ["COMP", "CL-01"], ["CL-01", "PROD"]],
            "material_streams": ["FEED", "COMP", "PROD"]}


def build_GEN_03(fs: DWSIMFlowsheet) -> dict:
    """
    GEN_03 — "Compress n-heptane vapour from 0.2 bar to 2 bar, then cool the
    compressed gas to 40 C to condense it." Pure n-heptane; Peng-Robinson.
    FEED(60 C,0.2 bar,vap) -> CP-01(2 bar, eta=0.75) -> COMP -> CL-01(40 C) -> PROD
    Feed at 60 C so it is a vapour at 0.2 bar (n-heptane boils ~51 C at 0.2 bar).
    """
    fs.add_compounds(["N-heptane"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "COMP", "PROD"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    sim.AddObject(ObjectType.EnergyStream, 0, 0, "CP-Q")
    fs.add_unit("CP-01", "Compressor"); fs.add_unit("CL-01", "Cooler")
    fs.set_stream("FEED", 333.15, 20000.0, 1.0, {"N-heptane": 1.0})   # 60 C, 0.2 bar
    fs.connect("FEED", "CP-01", 0, 0)
    try: fs.connect("CP-Q", "CP-01", 0, 1)
    except Exception: pass
    fs.connect("CP-01", "COMP", 0, 0); fs.connect("COMP", "CL-01", 0, 0)
    fs.connect("CL-01", "PROD", 0, 0)
    fs.set_compressor("CP-01", 200_000.0, efficiency=0.75)
    fs.set_cooler("CL-01", 313.15, dP=0.0)
    return {"case_id": "GEN_03", "case_name": "Compress then cool n-heptane",
            "compounds": ["N-heptane"], "property_package": "Peng-Robinson",
            "units": [{"tag": "CP-01", "type": "Compressor",
                       "params": {"P_out": 200_000.0, "efficiency": 0.75}},
                      {"tag": "CL-01", "type": "Cooler",
                       "params": {"T_out": 313.15, "dP": 0.0}}],
            "connections": [["FEED", "CP-01"], ["CP-01", "COMP"],
                            ["COMP", "CL-01"], ["CL-01", "PROD"]],
            "material_streams": ["FEED", "COMP", "PROD"]}


def build_EASY_02(fs: DWSIMFlowsheet) -> dict:
    """
    EASY_02 — condense propane at 10 bar, then pump the liquid. Restructured from
    the original (cool at 2 bar / 0 C did NOT condense — propane boils at -25 C at
    2 bar). Cooling now occurs at 10 bar where propane liquefies (Tsat~27 C), so
    there is genuine liquid for the pump. Pure propane; Peng-Robinson.
    FEED(50 C,10 bar,vap) -> CL-01(20 C -> liquid) -> COOL -> PM-01(10->20 bar) -> PROD
    """
    fs.add_compounds(["Propane"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "COOL", "PROD"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    sim.AddObject(ObjectType.EnergyStream, 0, 0, "PM-Q")
    fs.add_unit("CL-01", "Cooler"); fs.add_unit("PM-01", "Pump")
    fs.set_stream("FEED", 323.15, 1_000_000.0, 1.0, {"Propane": 1.0})  # 50 C, 10 bar (vapour)
    fs.connect("FEED", "CL-01", 0, 0); fs.connect("CL-01", "COOL", 0, 0)
    fs.connect("COOL", "PM-01", 0, 0)
    try: fs.connect("PM-Q", "PM-01", 0, 1)
    except Exception: pass
    fs.connect("PM-01", "PROD", 0, 0)
    fs.set_cooler("CL-01", 293.15, dP=0.0)                    # 20 C -> liquid (below 27 C bubble)
    fs.set_pump("PM-01", 2_000_000.0, efficiency=0.75)        # pump 10 -> 20 bar
    return {"case_id": "EASY_02", "case_name": "Condense then pump propane",
            "compounds": ["Propane"], "property_package": "Peng-Robinson",
            "units": [{"tag": "CL-01", "type": "Cooler",
                       "params": {"T_out": 293.15, "dP": 0.0}},
                      {"tag": "PM-01", "type": "Pump",
                       "params": {"P_out": 2_000_000.0, "efficiency": 0.75}}],
            "connections": [["FEED", "CL-01"], ["CL-01", "COOL"],
                            ["COOL", "PM-01"], ["PM-01", "PROD"]],
            "material_streams": ["FEED", "COOL", "PROD"]}


def _flash_case(fs, compounds, feed_comp, pp, T_flash_K, meta):
    """Shared heat-then-flash builder: FEED -> HT-01(T_flash) -> HOT -> V-01 ->
    VAP + LIQ. Feed at 25 C / 1 atm; the heater sets the flash temperature and the
    vessel does an adiabatic flash, so HOT carries the two-phase vapour fraction."""
    fs.add_compounds(compounds)
    fs.set_property_package(pp)
    sim = fs._sim
    for s in ("FEED", "HOT", "VAP", "LIQ"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    fs.add_unit("HT-01", "Heater"); fs.add_unit("V-01", "Vessel")
    fs.set_stream("FEED", 298.15, 101325.0, 1.0, feed_comp)
    fs.connect("FEED", "HT-01", 0, 0); fs.connect("HT-01", "HOT", 0, 0)
    fs.connect("HOT", "V-01", 0, 0)
    fs.connect("V-01", "VAP", 0, 0); fs.connect("V-01", "LIQ", 1, 0)
    fs.set_heater("HT-01", T_flash_K, dP=0.0)
    fs.set_vessel("V-01", dP=0.0)
    meta["material_streams"] = ["FEED", "HOT", "VAP", "LIQ"]
    return meta


def build_SAN_03(fs):
    """SAN_03 — heat then flash equimolar benzene/toluene to a genuine two-phase
    split. Benzene bp 80 C, toluene 111 C; bubble ~92 C / dew ~98 C at 1 atm, so a
    flash near 95 C gives vf ~ 0.5. Near-ideal aromatics; Peng-Robinson."""
    return _flash_case(fs, ["Benzene", "Toluene"], {"Benzene": 0.5, "Toluene": 0.5},
        "Peng-Robinson (PR)", 368.15,   # 95 C
        {"case_id": "SAN_03", "case_name": "Heat then flash benzene-toluene",
         "compounds": ["Benzene", "Toluene"], "property_package": "Peng-Robinson",
         "units": [{"tag": "HT-01", "type": "Heater", "params": {"T_out": 368.15, "dP": 0.0}},
                   {"tag": "V-01", "type": "Vessel", "params": {"dP": 0.0}}],
         "connections": [["FEED", "HT-01"], ["HT-01", "HOT"], ["HOT", "V-01"],
                         ["V-01", "VAP"], ["V-01", "LIQ"]]})


def build_GEN_01(fs):
    """GEN_01 — heat then flash equimolar n-hexane/n-heptane to two-phase. Hexane
    bp 69 C, heptane 98 C; bubble ~81 C / dew ~89 C at 1 atm, so flash near 85 C
    gives vf ~ 0.5. Near-ideal alkanes; Peng-Robinson."""
    return _flash_case(fs, ["N-hexane", "N-heptane"], {"N-hexane": 0.5, "N-heptane": 0.5},
        "Peng-Robinson (PR)", 358.15,   # 85 C
        {"case_id": "GEN_01", "case_name": "Heat then flash hexane-heptane",
         "compounds": ["N-hexane", "N-heptane"], "property_package": "Peng-Robinson",
         "units": [{"tag": "HT-01", "type": "Heater", "params": {"T_out": 358.15, "dP": 0.0}},
                   {"tag": "V-01", "type": "Vessel", "params": {"dP": 0.0}}],
         "connections": [["FEED", "HT-01"], ["HT-01", "HOT"], ["HOT", "V-01"],
                         ["V-01", "VAP"], ["V-01", "LIQ"]]})


def build_EASY_01(fs):
    """EASY_01 — heat then flash 50 mol% acetone / 50 mol% water to two-phase.
    Acetone bp 56 C, water 100 C; a flash at 70 C gives a robust partial vaporisation
    (vf ~ 0.68). The equimolar composition sits in a wide two-phase window with a
    gentle T-vf slope (precise-to-tolerance vf); the acetone-rich 70/30 feed instead
    flashes almost completely at 70 C and has no stable mid-range flash at 1 atm.
    Polar system -> NRTL (activity)."""
    return _flash_case(fs, ["Acetone", "Water"], {"Acetone": 0.5, "Water": 0.5},
        "NRTL", 343.15,   # 70 C
        {"case_id": "EASY_01", "case_name": "Heat then flash acetone-water",
         "compounds": ["Acetone", "Water"], "property_package": "NRTL",
         "units": [{"tag": "HT-01", "type": "Heater", "params": {"T_out": 343.15, "dP": 0.0}},
                   {"tag": "V-01", "type": "Vessel", "params": {"dP": 0.0}}],
         "connections": [["FEED", "HT-01"], ["HT-01", "HOT"], ["HOT", "V-01"],
                         ["V-01", "VAP"], ["V-01", "LIQ"]]})


def build_F1(fs):
    """
    F1 — wide-boiling n-pentane/n-octane flash. Unlike the close-boiling SAN_03/
    GEN_01, the 36-vs-126 C boiling gap gives a gentle T-vf slope, so vf is precise
    to the +/-0.05 tolerance (Delta_vf ~ 0.031 per +/-2 C). Peng-Robinson.

    FEED(50/50 pentane/octane, 25 C, 1 bar, 1 mol/s) -> HT-01(75 C) -> HOT
      -> V-01(adiabatic flash, 1 bar) -> VAP + LIQ.
    """
    fs.add_compounds(["N-pentane", "N-octane"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "HOT", "VAP", "LIQ"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    fs.add_unit("HT-01", "Heater"); fs.add_unit("V-01", "Vessel")
    fs.set_stream("FEED", 298.15, 100000.0, 1.0, {"N-pentane": 0.5, "N-octane": 0.5})
    fs.connect("FEED", "HT-01", 0, 0); fs.connect("HT-01", "HOT", 0, 0)
    fs.connect("HOT", "V-01", 0, 0)
    fs.connect("V-01", "VAP", 0, 0); fs.connect("V-01", "LIQ", 1, 0)
    fs.set_heater("HT-01", 348.15, dP=0.0)   # 75 C
    fs.set_vessel("V-01", dP=0.0)
    return {"case_id": "F1", "case_name": "Heat then flash n-pentane/n-octane",
            "compounds": ["N-pentane", "N-octane"], "property_package": "Peng-Robinson",
            "units": [{"tag": "HT-01", "type": "Heater", "params": {"T_out": 348.15, "dP": 0.0}},
                      {"tag": "V-01", "type": "Vessel", "params": {"dP": 0.0}}],
            "connections": [["FEED", "HT-01"], ["HT-01", "HOT"], ["HOT", "V-01"],
                            ["V-01", "VAP"], ["V-01", "LIQ"]],
            "material_streams": ["FEED", "HOT", "VAP", "LIQ"]}


def build_P1(fs):
    """
    P1 — two-stage nitrogen compression with intercooling. Directly targets the
    stage-collapse fault (a system that fuses 1->25 bar into one stage puts the
    intermediate streams at 25 bar instead of 5). Pure N2; Peng-Robinson.

    FEED(N2,25 C,1 bar,1 mol/s) -> CP-01(5 bar, eta=0.75) -> INT1
      -> CL-01(40 C) -> COOLED -> CP-02(25 bar, eta=0.75) -> INT2
      -> CL-02(40 C) -> PROD.  N2 stays supercritical vapour (Tc=126 K), vf=1.
    """
    fs.add_compounds(["Nitrogen"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "INT1", "COOLED", "INT2", "PROD"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    for e in ("CP1-Q", "CP2-Q"):
        sim.AddObject(ObjectType.EnergyStream, 0, 0, e)
    fs.add_unit("CP-01", "Compressor"); fs.add_unit("CL-01", "Cooler")
    fs.add_unit("CP-02", "Compressor"); fs.add_unit("CL-02", "Cooler")
    fs.set_stream("FEED", 298.15, 100000.0, 1.0, {"Nitrogen": 1.0})
    fs.connect("FEED", "CP-01", 0, 0)
    try: fs.connect("CP1-Q", "CP-01", 0, 1)
    except Exception: pass
    fs.connect("CP-01", "INT1", 0, 0); fs.connect("INT1", "CL-01", 0, 0)
    fs.connect("CL-01", "COOLED", 0, 0); fs.connect("COOLED", "CP-02", 0, 0)
    try: fs.connect("CP2-Q", "CP-02", 0, 1)
    except Exception: pass
    fs.connect("CP-02", "INT2", 0, 0); fs.connect("INT2", "CL-02", 0, 0)
    fs.connect("CL-02", "PROD", 0, 0)
    fs.set_compressor("CP-01", 500_000.0, efficiency=0.75)     # 5 bar
    fs.set_cooler("CL-01", 313.15, dP=0.0)                     # 40 C
    fs.set_compressor("CP-02", 2_500_000.0, efficiency=0.75)   # 25 bar
    fs.set_cooler("CL-02", 313.15, dP=0.0)                     # 40 C
    return {"case_id": "P1", "case_name": "Two-stage nitrogen compression with intercooling",
            "compounds": ["Nitrogen"], "property_package": "Peng-Robinson",
            "units": [{"tag": "CP-01", "type": "Compressor",
                       "params": {"P_out": 500_000.0, "efficiency": 0.75}},
                      {"tag": "CL-01", "type": "Cooler",
                       "params": {"T_out": 313.15, "dP": 0.0}},
                      {"tag": "CP-02", "type": "Compressor",
                       "params": {"P_out": 2_500_000.0, "efficiency": 0.75}},
                      {"tag": "CL-02", "type": "Cooler",
                       "params": {"T_out": 313.15, "dP": 0.0}}],
            "connections": [["FEED", "CP-01"], ["CP-01", "INT1"], ["INT1", "CL-01"],
                            ["CL-01", "COOLED"], ["COOLED", "CP-02"], ["CP-02", "INT2"],
                            ["INT2", "CL-02"], ["CL-02", "PROD"]],
            "material_streams": ["FEED", "INT1", "COOLED", "INT2", "PROD"]}


def build_C1(fs):
    """
    C1 — canonical benzene/toluene shortcut (FUG) distillation column. Light key
    benzene, heavy key toluene; 2% LK in bottoms, 2% HK in distillate; column at
    atmospheric pressure. Reflux via the two-pass logic: seed R below Rmin so
    solve() reads the computed Rmin and bumps R = 1.3 x Rmin. Peng-Robinson.

    FEED(50/50, saturated liquid at the 1 atm bubble point ~92 C, 1 mol/s)
      -> COL-01 (ShortcutColumn) -> DIST + BOT.
    Fenske Nmin ~ 8.77, Underwood Rmin ~ 1.30; mass balance closes exactly.
    """
    fs.add_compounds(["Benzene", "Toluene"])
    fs.set_property_package("Peng-Robinson (PR)")
    sim = fs._sim
    for s in ("FEED", "DIST", "BOT"):
        sim.AddObject(ObjectType.MaterialStream, 0, 0, s)
    fs.add_unit("COL-01", "Column")
    # 92.00 C = PR bubble point of 50/50 benzene/toluene at 1 atm -> saturated liquid (q=1)
    fs.set_stream("FEED", 365.15, 101325.0, 1.0, {"Benzene": 0.5, "Toluene": 0.5})
    fs.connect("FEED", "COL-01", 0, 0)
    fs.connect("COL-01", "DIST", 0, 0); fs.connect("COL-01", "BOT", 1, 0)
    # LK=benzene, HK=toluene; LK-in-bottoms 0.02, HK-in-distillate 0.02; seed R=0.5
    # (< Rmin) -> solve() two-pass bumps to 1.3 x Rmin.
    fs.set_column("COL-01", "Benzene", "Toluene", 0.02, 0.02, 0.5, 101325.0, 101325.0)
    return {"case_id": "C1", "case_name": "Benzene/toluene shortcut column",
            "compounds": ["Benzene", "Toluene"], "property_package": "Peng-Robinson",
            "units": [{"tag": "COL-01", "type": "Column",
                       "params": {"light_key": "Benzene", "heavy_key": "Toluene",
                                  "light_key_frac_bottoms": 0.02,
                                  "heavy_key_frac_distillate": 0.02,
                                  "reflux_ratio": 0.5,
                                  "condenser_pressure_Pa": 101325.0,
                                  "boiler_pressure_Pa": 101325.0}}],
            "connections": [["FEED", "COL-01"], ["COL-01", "DIST"], ["COL-01", "BOT"]],
            "material_streams": ["FEED", "DIST", "BOT"]}


_BUILDERS = {"SAN_04": build_SAN_04, "EASY_04": build_EASY_04,
             "GEN_03": build_GEN_03, "EASY_02": build_EASY_02,
             "SAN_03": build_SAN_03, "GEN_01": build_GEN_01, "EASY_01": build_EASY_01,
             "P1": build_P1, "F1": build_F1, "C1": build_C1}


def build_case(case_id: str, write: bool = False) -> dict:
    fs = DWSIMFlowsheet()
    meta = _BUILDERS[case_id](fs)
    res = fs.solve(timeout=120)
    streams = {tag: _stream_record(res[tag])
               for tag in meta["material_streams"] if tag in res}
    ref = {
        "case_id": meta["case_id"],
        "case_name": meta["case_name"],
        "source_file": "expert-specified (independent reference — built directly "
                       "via DWSIM wrapper, NOT the pipeline extraction/IR)",
        "reference_validity": "expert-constructed",
        "compounds": meta["compounds"],
        "property_package": meta["property_package"],
        "solved": bool(res.get("solved")),
        "units": meta["units"],
        "connections": meta["connections"],
        "streams": streams,
    }
    if write and ref["solved"]:
        os.makedirs(_OUT_DIR, exist_ok=True)
        path = os.path.join(_OUT_DIR, f"{case_id}_reference.json")
        with open(path, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"wrote {path}", file=sys.stderr)
    return ref


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="SAN_04")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ref = build_case(a.case, write=a.write)
    print(json.dumps(ref, indent=2))
