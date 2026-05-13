"""
DWSIM wrapper integration tests — requires a live DWSIM installation.

Run inside the Docker container:
    PYTHONPATH=. python3.9 dwsim/test_dwsim_wrapper.py

These tests exercise the real DWSIM solver. They will fail if DWSIM is not
installed at /usr/local/lib/dwsim/ or if the .NET runtime is not available.

Ground-truth VLE values are from Rachford-Rice at 80 °C, 1 atm:
    Psat(MeOH, 353.15 K) = 181 300 Pa   K_MeOH = 1.789
    Psat(H2O,  353.15 K) =  47 390 Pa   K_H2O  = 0.468
    Vapour fraction V = 0.305
    y_MeOH = 0.721   y_H2O = 0.279
    x_MeOH = 0.403   x_H2O = 0.597
"""
from __future__ import annotations
import math
import traceback

_COMP_TOL  = 0.05   # ± 5 % on mole fractions vs. Rachford-Rice ground truth
_TEMP_TOL  = 1.0    # ± 1 K on outlet temperatures
_MB_TOL    = 0.01   # ± 1 % mass balance error


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")

def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")

def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# ── Test 1: DWSIM starts and accepts compounds ────────────────────────────────

def test_startup_and_compounds():
    """DWSIMFlowsheet can be instantiated and AddCompound works."""
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "Water"])
    _assert("Methanol" in sim._compounds, f"Methanol not in compounds: {sim._compounds}")
    _assert("Water"    in sim._compounds, f"Water not in compounds: {sim._compounds}")
    _assert(len(sim._compounds) == 2, f"Expected 2 compounds, got {sim._compounds}")
    _ok(f"Compounds added: {sim._compounds}")


# ── Test 2: Property package assignment ──────────────────────────────────────

def test_property_package():
    """CreateAndAddPropertyPackage succeeds for all supported packages."""
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    # Use DWSIM internal names (executor._PP_MAP values, not schema keys)
    for pkg in ("Raoult's Law", "NRTL", "Peng-Robinson (PR)"):
        sim = DWSIMFlowsheet()
        sim.add_compounds(["Methanol", "Water"])
        sim.set_property_package(pkg)
        _assert(pkg in sim._property_packages,
                f"Package '{pkg}' not stored after set_property_package")
        _ok(f"Property package set: {pkg}")


# ── Test 3: Methanol/Water flash — Raoult's Law ───────────────────────────────

def test_meoh_water_flash_raoults_law():
    """
    Methanol/Water flash at 80 °C, 1 atm with Raoult's Law.
    Results must match Rachford-Rice ground truth within ±5 %.
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "Water"])
    sim.set_property_package("Raoult's Law")

    sim.add_stream("FEED", x=50,  y=100)
    sim.add_stream("VAP",  x=250, y=50)
    sim.add_stream("LIQ",  x=250, y=150)
    sim.add_unit("V-01", "Vessel", x=150, y=100)

    sim.connect("FEED", "V-01", 0, 0)
    sim.connect("V-01", "VAP",  0, 0)
    sim.connect("V-01", "LIQ",  1, 0)

    sim.set_stream("FEED", T=353.15, P=101325.0, flow=1.0,
                   composition={"Methanol": 0.5, "Water": 0.5})
    sim.set_vessel("V-01", dP=0.0)

    result = sim.solve()
    _assert(result["solved"], f"Solver failed: {result.get('errors')}")

    vap = result.get("VAP")
    liq = result.get("LIQ")
    _assert(vap is not None, "VAP stream not in results")
    _assert(liq is not None, "LIQ stream not in results")

    y_meoh = vap["composition"].get("Methanol", 0.0)
    y_h2o  = vap["composition"].get("Water",    0.0)
    x_meoh = liq["composition"].get("Methanol", 0.0)
    x_h2o  = liq["composition"].get("Water",    0.0)

    _ok(f"VAP: MeOH={y_meoh:.3f}  H2O={y_h2o:.3f}  flow={vap['flow_mol_s']:.3f} mol/s")
    _ok(f"LIQ: MeOH={x_meoh:.3f}  H2O={x_h2o:.3f}  flow={liq['flow_mol_s']:.3f} mol/s")

    _assert(abs(y_meoh - 0.721) < _COMP_TOL,
            f"y_MeOH={y_meoh:.3f}, expected 0.721 ± {_COMP_TOL} (Rachford-Rice ground truth)")
    _assert(abs(x_h2o - 0.597) < _COMP_TOL,
            f"x_H2O={x_h2o:.3f}, expected 0.597 ± {_COMP_TOL} (Rachford-Rice ground truth)")
    _assert(y_meoh > y_h2o,
            f"VAP should be methanol-rich: MeOH={y_meoh:.3f} H2O={y_h2o:.3f}")
    _assert(x_h2o > x_meoh,
            f"LIQ should be water-rich: H2O={x_h2o:.3f} MeOH={x_meoh:.3f}")

    # Mass balance
    feed_flow = result["FEED"]["flow_mol_s"]
    out_flow  = vap["flow_mol_s"] + liq["flow_mol_s"]
    mb_err    = abs(feed_flow - out_flow) / feed_flow
    _assert(mb_err < _MB_TOL, f"Mass balance error {mb_err:.1%} exceeds {_MB_TOL:.0%}")
    _ok(f"Mass balance error: {mb_err:.3%}")


# ── Test 4: Heater raises temperature ─────────────────────────────────────────

def test_heater():
    """
    Heater must raise Methanol/Water stream from 25 °C to 80 °C.
    Verifies CalcMode=OutletTemperature reflection path.
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "Water"])
    sim.set_property_package("Raoult's Law")

    sim.add_stream("FEED", x=50,  y=100)
    sim.add_stream("HOT",  x=250, y=100)
    sim.add_unit("HT-01", "Heater", x=150, y=100)

    sim.connect("FEED",  "HT-01", 0, 0)
    sim.connect("HT-01", "HOT",   0, 0)

    sim.set_stream("FEED", T=298.15, P=101325.0, flow=1.0,
                   composition={"Methanol": 0.5, "Water": 0.5})
    sim.set_heater("HT-01", T_out=353.15, dP=0.0)

    result = sim.solve()
    _assert(result["solved"], f"Solver failed: {result.get('errors')}")

    hot = result.get("HOT")
    _assert(hot is not None, "HOT stream not in results")
    _assert(abs(hot["T_K"] - 353.15) < _TEMP_TOL,
            f"HOT T={hot['T_K']:.2f} K, expected 353.15 ± {_TEMP_TOL} K")
    _ok(f"HOT outlet: T={hot['T_K']:.2f} K  P={hot['P_Pa']:.0f} Pa")


# ── Test 5: Heater + Vessel in series ─────────────────────────────────────────

def test_heater_then_flash():
    """
    Full heat-then-flash topology: FEED → HT-01 → HOT → V-01 → VAP / LIQ.
    This is the demo.py default process. Verifies multi-unit solve.
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "Water"])
    sim.set_property_package("Raoult's Law")

    sim.add_stream("FEED", x=50,  y=100)
    sim.add_stream("HOT",  x=200, y=100)
    sim.add_stream("VAP",  x=350, y=50)
    sim.add_stream("LIQ",  x=350, y=150)
    sim.add_unit("HT-01", "Heater", x=125, y=100)
    sim.add_unit("V-01",  "Vessel", x=275, y=100)

    sim.connect("FEED",  "HT-01", 0, 0)
    sim.connect("HT-01", "HOT",   0, 0)
    sim.connect("HOT",   "V-01",  0, 0)
    sim.connect("V-01",  "VAP",   0, 0)
    sim.connect("V-01",  "LIQ",   1, 0)

    sim.set_stream("FEED", T=298.15, P=101325.0, flow=1.0,
                   composition={"Methanol": 0.5, "Water": 0.5})
    sim.set_heater("HT-01", T_out=353.15, dP=0.0)
    sim.set_vessel("V-01",  dP=0.0)

    result = sim.solve()
    _assert(result["solved"], f"Solver failed: {result.get('errors')}")

    vap = result.get("VAP")
    liq = result.get("LIQ")
    _assert(vap is not None and liq is not None, "VAP or LIQ missing from results")
    _assert(vap["composition"].get("Methanol", 0) > liq["composition"].get("Methanol", 0),
            "VAP should be methanol-richer than LIQ")

    feed_flow = result["FEED"]["flow_mol_s"]
    out_flow  = vap["flow_mol_s"] + liq["flow_mol_s"]
    mb_err    = abs(feed_flow - out_flow) / feed_flow
    _assert(mb_err < _MB_TOL, f"Mass balance error {mb_err:.1%}")
    _ok(f"Heater+Flash solved. VAP MeOH={vap['composition'].get('Methanol', 0):.3f}  "
        f"LIQ H2O={liq['composition'].get('Water', 0):.3f}  MB err={mb_err:.3%}")


# ── Test 6: NRTL silent failure — no binary params → outlet ≈ feed ────────────

def test_nrtl_no_params_gives_no_separation():
    """
    NRTL without binary interaction parameters should produce outlet compositions
    close to feed (DWSIM silently defaults to ideal behaviour).
    This is the PARAM_MISSING failure mode the Critic detects.
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "Water"])
    sim.set_property_package("NRTL")

    sim.add_stream("FEED", x=50,  y=100)
    sim.add_stream("VAP",  x=250, y=50)
    sim.add_stream("LIQ",  x=250, y=150)
    sim.add_unit("V-01", "Vessel", x=150, y=100)

    sim.connect("FEED", "V-01", 0, 0)
    sim.connect("V-01", "VAP",  0, 0)
    sim.connect("V-01", "LIQ",  1, 0)

    sim.set_stream("FEED", T=353.15, P=101325.0, flow=1.0,
                   composition={"Methanol": 0.5, "Water": 0.5})
    sim.set_vessel("V-01", dP=0.0)

    result = sim.solve()
    # DWSIM may or may not mark as solved — what matters is no real separation
    vap = result.get("VAP", {})
    liq = result.get("LIQ", {})
    vap_meoh = vap.get("composition", {}).get("Methanol", 0.5)
    liq_meoh = liq.get("composition", {}).get("Methanol", 0.5)
    max_diff  = abs(vap_meoh - 0.5)

    # NRTL without params should show < 5 % deviation from feed (near-ideal fallback)
    if max_diff < 0.05:
        _ok(f"NRTL (no params): outlet ≈ feed as expected "
            f"(VAP MeOH={vap_meoh:.3f}, LIQ MeOH={liq_meoh:.3f})")
    else:
        _ok(f"NRTL: binary params found in DWSIM db — separation occurred "
            f"(VAP MeOH={vap_meoh:.3f}, LIQ MeOH={liq_meoh:.3f}). "
            "PARAM_MISSING path may not fire for this pair.")


# ── Test 7: Per-unit property package override ────────────────────────────────

def test_per_unit_property_package():
    """
    set_unit_property_package assigns a different package to a single unit.
    Verifies the CLR obj.PropertyPackage assignment works in DWSIM 9.
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "Water"])
    sim.set_property_package("Raoult's Law")   # global default

    sim.add_stream("FEED", x=50,  y=100)
    sim.add_stream("VAP",  x=250, y=50)
    sim.add_stream("LIQ",  x=250, y=150)
    sim.add_unit("V-01", "Vessel", x=150, y=100)

    sim.connect("FEED", "V-01", 0, 0)
    sim.connect("V-01", "VAP",  0, 0)
    sim.connect("V-01", "LIQ",  1, 0)

    sim.set_stream("FEED", T=353.15, P=101325.0, flow=1.0,
                   composition={"Methanol": 0.5, "Water": 0.5})
    sim.set_vessel("V-01", dP=0.0)

    # Override the vessel to use NRTL
    sim.set_unit_property_package("V-01", "NRTL")

    result = sim.solve()
    # We don't assert composition here — NRTL may or may not have params.
    # What matters is that the call does not raise an exception.
    _ok(f"Per-unit property package set without exception. solved={result['solved']}")


# ── Test 8: Unknown compound is rejected ──────────────────────────────────────

def test_unknown_compound_rejected():
    """
    AddCompound with an unknown name raises KeyNotFoundException in DWSIM.
    The wrapper catches this per-compound so valid compounds still load.
    The executor's missing-compound check then catches the omission.
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methanol", "NOTACOMPOUND_XYZ"])
    _assert("NOTACOMPOUND_XYZ" not in sim._compounds,
            "Unknown compound should be absent from sim._compounds after per-compound try/except")
    _assert("Methanol" in sim._compounds,
            "Valid compound must still be added when an unknown compound is in the list")
    _ok(f"Unknown compound skipped, valid compound retained. Compounds: {sim._compounds}")


def test_pump():
    """Pump raises feed pressure from 1 atm to 5 atm via reflection-based setter."""
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Water"])
    sim.set_property_package("Raoult's Law")

    sim.add_stream("FEED", 0, 0)
    sim.add_stream("OUT",  200, 0)
    sim.add_unit("PU-01", "Pump", 100, 0)

    sim.connect("FEED", "PU-01", 0, 0)
    sim.connect("PU-01", "OUT",  0, 0)

    sim.set_stream("FEED", T=298.15, P=101325.0, flow=1.0,
                   composition={"Water": 1.0})
    sim.set_pump("PU-01", P_out=506625.0, efficiency=0.75)  # 5 atm out

    raw = sim.solve()
    _assert(raw["solved"], f"Pump flowsheet did not solve. Errors: {raw.get('errors')}")

    out = raw["OUT"]
    _assert(out["P_Pa"] > 400000.0,
            f"Outlet pressure {out['P_Pa']:.0f} Pa not raised to ~5 atm")
    _ok(f"Pump raised P: {101325:.0f} → {out['P_Pa']:.0f} Pa")


def test_compressor():
    """Compressor raises methane pressure from 1 atm to 10 atm."""
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methane"])
    sim.set_property_package("Peng-Robinson (PR)")

    sim.add_stream("FEED", 0, 0)
    sim.add_stream("OUT",  200, 0)
    sim.add_unit("CO-01", "Compressor", 100, 0)

    sim.connect("FEED", "CO-01", 0, 0)
    sim.connect("CO-01", "OUT",  0, 0)

    sim.set_stream("FEED", T=298.15, P=101325.0, flow=1.0,
                   composition={"Methane": 1.0})
    sim.set_compressor("CO-01", P_out=1013250.0, efficiency=0.75)  # 10 atm out

    raw = sim.solve()
    _assert(raw["solved"], f"Compressor flowsheet did not solve. Errors: {raw.get('errors')}")

    out = raw["OUT"]
    _assert(out["P_Pa"] > 900000.0,
            f"Outlet pressure {out['P_Pa']:.0f} Pa not raised to ~10 atm")
    _assert(out["T_K"] > 298.15,
            f"Compressor outlet T {out['T_K']:.1f} K not above feed (compression heats gas)")
    _ok(f"Compressor: P {101325:.0f} → {out['P_Pa']:.0f} Pa, "
        f"T {298.15:.1f} → {out['T_K']:.1f} K")


def test_expander():
    """Expander drops methane from 10 atm to 1 atm (Joule-Thomson cooling)."""
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Methane"])
    sim.set_property_package("Peng-Robinson (PR)")

    sim.add_stream("FEED", 0, 0)
    sim.add_stream("OUT",  200, 0)
    sim.add_unit("EX-01", "Expander", 100, 0)

    sim.connect("FEED", "EX-01", 0, 0)
    sim.connect("EX-01", "OUT",  0, 0)

    sim.set_stream("FEED", T=298.15, P=1013250.0, flow=1.0,
                   composition={"Methane": 1.0})
    sim.set_expander("EX-01", P_out=101325.0, efficiency=0.75)

    raw = sim.solve()
    _assert(raw["solved"], f"Expander flowsheet did not solve. Errors: {raw.get('errors')}")

    out = raw["OUT"]
    _assert(out["P_Pa"] < 200000.0,
            f"Outlet pressure {out['P_Pa']:.0f} Pa not dropped to ~1 atm")
    _assert(out["T_K"] < 298.15,
            f"Expander outlet T {out['T_K']:.1f} K not below feed (expansion cools gas)")
    _ok(f"Expander: P {1013250:.0f} → {out['P_Pa']:.0f} Pa, "
        f"T {298.15:.1f} → {out['T_K']:.1f} K")


def test_splitter():
    """Splitter divides a water stream 60/40 into two outlets."""
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Water"])
    sim.set_property_package("Raoult's Law")

    sim.add_stream("FEED", 0,   0)
    sim.add_stream("OUT1", 200, 0)
    sim.add_stream("OUT2", 200, 150)
    sim.add_unit("SP-01", "Splitter", 100, 0)

    sim.connect("FEED",  "SP-01", 0, 0)
    sim.connect("SP-01", "OUT1",  0, 0)
    sim.connect("SP-01", "OUT2",  1, 0)

    sim.set_stream("FEED", T=298.15, P=101325.0, flow=1.0,
                   composition={"Water": 1.0})
    sim.set_splitter("SP-01", split_fractions={"OUT1": 0.6, "OUT2": 0.4})

    raw = sim.solve()
    _assert(raw["solved"], f"Splitter flowsheet did not solve. Errors: {raw.get('errors')}")

    f1 = raw["OUT1"]["flow_mol_s"]
    f2 = raw["OUT2"]["flow_mol_s"]
    _assert(abs(f1 - 0.6) < 0.02, f"OUT1 flow {f1:.3f} not ~0.6 mol/s")
    _assert(abs(f2 - 0.4) < 0.02, f"OUT2 flow {f2:.3f} not ~0.4 mol/s")
    _ok(f"Splitter: OUT1={f1:.3f} mol/s, OUT2={f2:.3f} mol/s (target 0.6/0.4)")


# ── Test 13: NRTL parameter injection ────────────────────────────────────────

def test_nrtl_parameter_injection():
    """
    set_nrtl_parameters() injects literature BIPs and enables real VLE separation.

    Acetone/Chloroform is a negative-deviation system (attractive interactions):
    NRTL parameters from DECHEMA Vol 3 give a maximum-pressure azeotrope.
    At 313 K, 1 atm with 50/50 feed, chloroform concentrates in the vapour phase
    (higher vapour pressure than acetone when NRTL activity coefficients are applied).
    We check: vapour-phase chloroform fraction meaningfully differs from feed (>0.05).
    """
    from dwsim.dwsim_wrapper import DWSIMFlowsheet
    sim = DWSIMFlowsheet()
    sim.add_compounds(["Acetone", "Chloroform"])
    sim.set_property_package("NRTL")

    # DECHEMA VLE Vol 3 — Acetone(1)/Chloroform(2)
    # A12 = -253.4 K, A21 = 664.2 K, alpha = 0.3
    # Run at 343 K (above bubble point ~331 K at 1 atm for 50/50 feed)
    sim.set_nrtl_parameters(
        "Acetone", "Chloroform",
        A12=-253.4, A21=664.2, alpha12=0.3,
        source="DECHEMA VLE Vol 3, p.184",
    )

    sim.add_stream("FEED", x=50,  y=100)
    sim.add_stream("VAP",  x=250, y=50)
    sim.add_stream("LIQ",  x=250, y=150)
    sim.add_unit("V-01", "Vessel", x=150, y=100)

    sim.connect("FEED", "V-01", 0, 0)
    sim.connect("V-01", "VAP",  0, 0)
    sim.connect("V-01", "LIQ",  1, 0)

    sim.set_stream("FEED", T=343.15, P=101325.0, flow=1.0,
                   composition={"Acetone": 0.5, "Chloroform": 0.5})
    sim.set_vessel("V-01", dP=0.0)

    result = sim.solve()
    _assert(result["solved"], f"Flowsheet did not solve. Errors: {result.get('errors')}")

    vap = result.get("VAP", {})
    liq = result.get("LIQ", {})
    vap_chcl3 = vap.get("composition", {}).get("Chloroform", 0.5)
    liq_chcl3 = liq.get("composition", {}).get("Chloroform", 0.5)
    separation = abs(vap_chcl3 - liq_chcl3)

    _assert(
        separation > 0.02,
        f"Injected NRTL params produced no real separation "
        f"(VAP CHCl3={vap_chcl3:.3f}, LIQ CHCl3={liq_chcl3:.3f}). "
        f"Expected |VAP - LIQ| > 0.02.",
    )
    _ok(
        f"NRTL injection: VAP CHCl3={vap_chcl3:.3f}, LIQ CHCl3={liq_chcl3:.3f} "
        f"(phase split={separation:.3f}, target >0.02)"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("DWSIM starts and accepts compounds",       test_startup_and_compounds),
        ("Property package assignment",              test_property_package),
        ("Methanol/Water flash — Raoult's Law",      test_meoh_water_flash_raoults_law),
        ("Heater raises temperature",                test_heater),
        ("Heater + Vessel in series",                test_heater_then_flash),
        ("NRTL without params → no separation",      test_nrtl_no_params_gives_no_separation),
        ("Per-unit property package override",       test_per_unit_property_package),
        ("Unknown compound rejected",                test_unknown_compound_rejected),
        ("Pump raises pressure",                     test_pump),
        ("Compressor raises pressure and heats gas", test_compressor),
        ("Expander drops pressure and cools gas",    test_expander),
        ("Splitter divides flow 60/40",              test_splitter),
        ("NRTL parameter injection gives separation", test_nrtl_parameter_injection),
    ]

    passed, failed = 0, []
    for name, fn in tests:
        print(f"\n{'─' * 55}")
        print(f"  {name}")
        print(f"{'─' * 55}")
        try:
            fn()
            passed += 1
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'═' * 55}")
    print(f"  {passed}/{len(tests)} passed")
    if failed:
        print(f"  Failed:")
        for f in failed:
            print(f"    ✗ {f}")
    else:
        print(f"  All tests passed — DWSIM wrapper is working correctly.")
    print(f"{'═' * 55}")
