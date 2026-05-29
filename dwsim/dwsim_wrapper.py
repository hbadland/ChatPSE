"""
DWSIM automation wrapper.

Property ID reference (discovered empirically):
  PROP_MS_0  = temperature [K]
  PROP_MS_1  = pressure [Pa]
  PROP_MS_2  = molar flow [mol/s]

  Composition is set via the .NET InputComposition dictionary (reflection).
  Composition is read from Phase[0].Compounds[name].MoleFraction (reflection).

  Heater: CalcMode defaults to HeatAdded — must set to OutletTemperature via
  reflection. Properties: CalcMode, OutletTemperature [K], PressureDrop [Pa].

  PROP_SEP_0 = vessel/flash pressure drop [Pa]
"""

import os
import sys

# Ensure coreclr is selected before pythonnet initialises (env var is read at
# import time; setting it here covers the case where nothing set it earlier).
os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")

import pythonnet
try:
    pythonnet.load("coreclr")
except RuntimeError as _pn_err:
    # "already loaded" is fine if coreclr was loaded by the env var path above.
    # Any other message means a wrong runtime (mono) was initialised first.
    if "already" not in str(_pn_err).lower():
        raise RuntimeError(
            f"pythonnet runtime conflict — coreclr could not be loaded: {_pn_err}. "
            "Make sure PYTHONNET_RUNTIME=coreclr is set before any pythonnet import, "
            "or that nothing in the environment auto-imports clr with mono."
        ) from _pn_err

import clr
import System

sys.path.append("/usr/local/lib/dwsim/")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Automation.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Interfaces.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.FlowsheetBase.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.SharedClasses.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Thermodynamics.dll")
clr.AddReference("/usr/local/lib/dwsim/DWSIM.UnitOperations.dll")

from DWSIM.Automation import Automation3
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

_UNIT_TYPE_MAP = {
    "Heater":     ObjectType.Heater,
    "Cooler":     ObjectType.Cooler,
    "Vessel":     ObjectType.Vessel,
    "Mixer":      ObjectType.Mixer,
    "Splitter":   ObjectType.Splitter,
    "Compressor": ObjectType.Compressor,
    "Expander":   ObjectType.Expander,
    "Pump":       ObjectType.Pump,
    "MaterialStream": ObjectType.MaterialStream,
}

# .NET reflection property accessors (cached on first use)
_IC_PROP   = None  # InputComposition property descriptor
_PH_PROP   = None  # Phases property descriptor


def _get_input_composition(obj):
    """Return the InputComposition .NET dict on a MaterialStream interface object."""
    global _IC_PROP
    if _IC_PROP is None:
        _IC_PROP = obj.GetType().GetProperty("InputComposition")
    return _IC_PROP.GetValue(obj)


def _get_phases(obj):
    """Return the Phases .NET dict on a MaterialStream interface object."""
    global _PH_PROP
    if _PH_PROP is None:
        _PH_PROP = obj.GetType().GetProperty("Phases")
    return _PH_PROP.GetValue(obj)


class DWSIMFlowsheet:
    """Thin wrapper around DWSIM Automation3 for headless flowsheet simulation."""

    def __init__(self):
        self._auto = Automation3()
        self._sim = self._auto.CreateFlowsheet()
        self._compounds: list[str] = []
        self._property_packages: dict[str, object] = {}

    # ── Setup ──────────────────────────────────────────────────────────────────

    def add_compounds(self, names: list[str]) -> None:
        for name in names:
            try:
                self._sim.AddCompound(name)
            except Exception:
                pass  # DWSIM raises KeyNotFoundException for unknown compound names
        self._compounds = list(self._sim.SelectedCompounds.Keys)

    def set_property_package(self, name: str = "Raoult's Law") -> None:
        if name not in self._property_packages:
            pkg = self._sim.CreateAndAddPropertyPackage(name)
            self._property_packages[name] = pkg

    def set_unit_property_package(self, unit_tag: str, pp_name: str) -> None:
        """Assign a property package to a single unit operation."""
        if pp_name not in self._property_packages:
            pkg = self._sim.CreateAndAddPropertyPackage(pp_name)
            self._property_packages[pp_name] = pkg
        obj = self._sim.GetFlowsheetSimulationObject(unit_tag)
        obj.PropertyPackage = self._property_packages[pp_name]

    # ── Binary interaction parameter injection ────────────────────────────────

    def set_nrtl_parameters(
        self,
        compound_a: str,
        compound_b: str,
        A12: float,
        A21: float,
        alpha12: float,
        B12: float = 0.0,
        B21: float = 0.0,
        source: str = "",
    ) -> None:
        """Inject NRTL binary interaction parameters into the active NRTL package.

        Both orderings are written (A12↔A21 swapped for the reverse entry),
        matching DWSIM's internal lookup convention. AutoEstimate is disabled
        so injected values are not overwritten by DWSIM's estimator.
        """
        from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import NRTL_IPData
        from System.Collections.Generic import Dictionary

        pkg = self._property_packages.get("NRTL")
        if pkg is None:
            raise ValueError("NRTL property package not initialised — call set_property_package('NRTL') first")

        t = pkg.GetType()
        t.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pkg, System.Boolean(False))

        muni = t.GetProperty("m_uni").GetValue(pkg)
        ip   = muni.GetType().GetProperty("InteractionParameters").GetValue(muni)
        et   = NRTL_IPData().GetType()

        def _make_entry(a12_val, a21_val):
            e = NRTL_IPData()
            et.GetField("A12").SetValue(e,     System.Double(a12_val))
            et.GetField("A21").SetValue(e,     System.Double(a21_val))
            et.GetField("alpha12").SetValue(e, System.Double(alpha12))
            et.GetField("B12").SetValue(e,     System.Double(B12))
            et.GetField("B21").SetValue(e,     System.Double(B21))
            et.GetField("comment").SetValue(e, System.String(source))
            return e

        for outer, inner_key, a12_v, a21_v in [
            (compound_a, compound_b, A12, A21),
            (compound_b, compound_a, A21, A12),
        ]:
            if not ip.ContainsKey(outer):
                ip[outer] = Dictionary[System.String, NRTL_IPData]()
            ip[outer][inner_key] = _make_entry(a12_v, a21_v)

    def set_uniquac_parameters(
        self,
        compound_a: str,
        compound_b: str,
        A12: float,
        A21: float,
        B12: float = 0.0,
        B21: float = 0.0,
        source: str = "",
    ) -> None:
        """Inject UNIQUAC binary interaction parameters into the active UNIQUAC package.

        Both orderings are written (A12↔A21 swapped for the reverse entry).
        AutoEstimate is disabled so injected values are not overwritten.
        """
        from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import UNIQUAC_IPData
        from System.Collections.Generic import Dictionary

        pkg = self._property_packages.get("UNIQUAC")
        if pkg is None:
            raise ValueError("UNIQUAC property package not initialised — call set_property_package('UNIQUAC') first")

        t = pkg.GetType()
        t.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pkg, System.Boolean(False))

        muni = t.GetProperty("m_uni").GetValue(pkg)
        ip   = muni.GetType().GetProperty("InteractionParameters").GetValue(muni)
        et   = UNIQUAC_IPData().GetType()

        def _make_entry(a12_val, a21_val):
            e = UNIQUAC_IPData()
            et.GetField("A12").SetValue(e,  System.Double(a12_val))
            et.GetField("A21").SetValue(e,  System.Double(a21_val))
            et.GetField("B12").SetValue(e,  System.Double(B12))
            et.GetField("B21").SetValue(e,  System.Double(B21))
            et.GetField("comment").SetValue(e, System.String(source))
            return e

        for outer, inner_key, a12_v, a21_v in [
            (compound_a, compound_b, A12, A21),
            (compound_b, compound_a, A21, A12),
        ]:
            if not ip.ContainsKey(outer):
                ip[outer] = Dictionary[System.String, UNIQUAC_IPData]()
            ip[outer][inner_key] = _make_entry(a12_v, a21_v)

    def disable_auto_estimate(self, package_name: str) -> None:
        """Disable AutoEstimateMissingNRTLUNIQUACParameters on a package.

        Must be called after set_property_package() and before solve().
        Ensures DWSIM does not silently estimate missing BIPs, which would
        produce spurious pseudo-separation and prevent PARAM_MISSING detection.
        """
        pkg = self._property_packages.get(package_name)
        if pkg is None:
            raise ValueError(
                f"Property package '{package_name}' not initialised — "
                "call set_property_package() first")
        pkg.GetType().GetProperty(
            "AutoEstimateMissingNRTLUNIQUACParameters"
        ).SetValue(pkg, System.Boolean(False))

    # ── Object management ─────────────────────────────────────────────────────

    def add_stream(self, tag: str, x: int = 0, y: int = 0) -> None:
        self._sim.AddObject(ObjectType.MaterialStream, x, y, tag)

    def add_unit(self, tag: str, unit_type: str, x: int = 0, y: int = 0) -> None:
        ot = _UNIT_TYPE_MAP.get(unit_type)
        if ot is None:
            raise ValueError(f"Unknown unit type '{unit_type}'. Valid: {list(_UNIT_TYPE_MAP)}")
        self._sim.AddObject(ot, x, y, tag)

    def connect(self, src_tag: str, dst_tag: str,
                src_port: int = 0, dst_port: int = 0) -> None:
        src = self._get_graphic_object(src_tag)
        dst = self._get_graphic_object(dst_tag)
        self._sim.ConnectObjects(src, dst, src_port, dst_port)

    # ── Condition setting ─────────────────────────────────────────────────────

    def set_stream(self, tag: str, T: float, P: float, flow: float,
                   composition: dict[str, float]) -> None:
        """Set temperature [K], pressure [Pa], molar flow [mol/s], mole fractions."""
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        obj.SetPropertyValue("PROP_MS_0", float(T))
        obj.SetPropertyValue("PROP_MS_1", float(P))
        obj.SetPropertyValue("PROP_MS_2", float(flow))
        ic = _get_input_composition(obj)
        for name, frac in composition.items():
            if name not in self._compounds:
                raise ValueError(f"Compound '{name}' not in flowsheet.")
            ic[name] = float(frac)

    def set_heater(self, tag: str, T_out: float, dP: float = 0.0) -> None:
        """Set heater outlet temperature [K] and pressure drop [Pa]."""
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        t = obj.GetType()
        # Switch to OutletTemperature spec mode
        calc_prop = t.GetProperty("CalcMode")
        mode_type = calc_prop.PropertyType
        calc_prop.SetValue(obj, System.Enum.Parse(mode_type, "OutletTemperature"))
        # Set outlet T (Nullable<Double>)
        t.GetProperty("OutletTemperature").SetValue(
            obj, System.Nullable[System.Double](float(T_out)))
        # Set pressure drop
        t.GetProperty("PressureDrop").SetValue(obj, float(dP))

    def set_cooler(self, tag: str, T_out: float, dP: float = 0.0) -> None:
        """Set cooler outlet temperature [K] and pressure drop [Pa]."""
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        t = obj.GetType()
        calc_prop = t.GetProperty("CalcMode")
        mode_type = calc_prop.PropertyType
        calc_prop.SetValue(obj, System.Enum.Parse(mode_type, "OutletTemperature"))
        t.GetProperty("OutletTemperature").SetValue(
            obj, System.Nullable[System.Double](float(T_out)))
        t.GetProperty("PressureDrop").SetValue(obj, float(dP))

    def set_vessel(self, tag: str, dP: float = 0.0) -> None:
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        obj.SetPropertyValue("PROP_SEP_0", float(dP))

    def set_splitter(self, tag: str, split_fractions: dict[str, float],
                     dP: float = 0.0) -> None:
        """Set split ratios on a Splitter via the Ratios ArrayList (reflection).

        split_fractions values must sum to 1.0; order matches outlet port order.
        dP is not settable via reflection on DWSIM's Splitter — ignored.
        """
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        t = obj.GetType()
        # Explicitly set SplitRatios mode
        op_prop = t.GetProperty("OperationMode")
        op_prop.SetValue(obj, System.Enum.Parse(op_prop.PropertyType, "SplitRatios"))
        # DWSIM keeps a fixed 3-slot Ratios ArrayList ([1.0, 0.0, 0.0] by default).
        # Never change the count — index-set our fractions and zero out remaining
        # slots. Clear() or RemoveAt() break DWSIM's internal solver assumption.
        ratios = t.GetProperty("Ratios").GetValue(obj)
        fracs = list(split_fractions.values())
        for i in range(ratios.Count):
            val = float(fracs[i]) if i < len(fracs) else 0.0
            ratios[i] = System.Double(val)

    def set_pump(self, tag: str, P_out: float, efficiency: float = 0.75) -> None:
        """Set pump outlet pressure [Pa] and efficiency (0–1 fraction).

        Switches CalcMode to OutletPressure and sets Pout + Efficiency via
        reflection. Efficiency stored as percent internally (0.75 → 75.0).
        """
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        t = obj.GetType()
        calc_prop = t.GetProperty("CalcMode")
        calc_prop.SetValue(
            obj, System.Enum.Parse(calc_prop.PropertyType, "OutletPressure"))
        t.GetProperty("Pout").SetValue(obj, float(P_out))
        t.GetProperty("Efficiency").SetValue(obj, float(efficiency * 100.0))

    def set_compressor(self, tag: str, P_out: float,
                       efficiency: float = 0.75) -> None:
        """Set compressor outlet pressure [Pa] and adiabatic efficiency (0–1).

        CalcMode defaults to OutletPressure. Sets POut + AdiabaticEfficiency
        via reflection. Efficiency stored as percent internally (0.75 → 75.0).
        """
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        t = obj.GetType()
        t.GetProperty("POut").SetValue(obj, float(P_out))
        t.GetProperty("AdiabaticEfficiency").SetValue(
            obj, float(efficiency * 100.0))

    def set_expander(self, tag: str, P_out: float,
                     efficiency: float = 0.75) -> None:
        """Set expander outlet pressure [Pa] and adiabatic efficiency (0–1).

        CalcMode defaults to OutletPressure. Sets POut + AdiabaticEfficiency
        via reflection. Efficiency stored as percent internally (0.75 → 75.0).
        """
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        t = obj.GetType()
        t.GetProperty("POut").SetValue(obj, float(P_out))
        t.GetProperty("AdiabaticEfficiency").SetValue(
            obj, float(efficiency * 100.0))

    # ── Solve & read ──────────────────────────────────────────────────────────

    def solve(self, timeout: int = 120) -> dict:
        """Run the headless solver. Returns {tag: stream_dict} for all streams.

        Uses a daemon thread for the timeout so it works inside Singularity
        containers where signal.SIGALRM is non-functional. The thread is marked
        daemon so it is reaped when the host process exits; if it stays alive
        after timeout the stuck .NET call continues in the background but the
        benchmark loop proceeds to the next case.
        """
        import threading
        import os

        _timeout = int(os.environ.get("DWSIM_SOLVER_TIMEOUT", str(timeout)))
        _result = [None]
        _exc    = [None]

        def _worker():
            try:
                # Try CalculateFlowsheet first (avoids the error-collection path
                # in CalculateFlowsheet2 which may deadlock on Linux/coreclr).
                try:
                    self._auto.CalculateFlowsheet(self._sim)
                    errors = []
                except Exception:
                    # Fall back to the standard method if CalculateFlowsheet
                    # is not available on this DWSIM build.
                    errors = self._auto.CalculateFlowsheet2(self._sim)
                    errors = list(errors) if errors else []

                solved = bool(self._sim.Solved)

                # CalculateFlowsheet() returns no error list; when unsolved,
                # harvest error messages from individual unit operation objects.
                if not solved and not errors:
                    go = self._sim.GraphicObjects
                    for key in go.Keys:
                        gobj = go[key]
                        if str(gobj.ObjectType) == "MaterialStream":
                            continue
                        try:
                            obj = self._sim.GetFlowsheetSimulationObject(
                                str(gobj.Tag))
                            msg = str(getattr(obj, "ErrorMessage", None) or "")
                            if msg and msg.lower() not in ("", "none"):
                                errors.append(f"{gobj.Tag}: {msg}")
                        except Exception:
                            pass
                r = {"solved": solved, "errors": errors}
                go = self._sim.GraphicObjects
                for key in go.Keys:
                    gobj = go[key]
                    if str(gobj.ObjectType) == "MaterialStream":
                        tag = str(gobj.Tag)
                        obj = self._sim.GetFlowsheetSimulationObject(tag)
                        r[tag] = self._read_stream(obj)
                _result[0] = r
            except Exception as e:
                _exc[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(_timeout)

        if t.is_alive():
            return {
                "solved": False,
                "errors": [
                    f"DWSIM solver timed out after {_timeout}s — "
                    "CalculateFlowsheet did not return. "
                    "Likely an infinite loop in the .NET solver for this flowsheet."
                ],
            }
        if _exc[0] is not None:
            raise _exc[0]
        return _result[0]

    def get_stream(self, tag: str) -> dict:
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        return self._read_stream(obj)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_graphic_object(self, tag: str):
        go = self._sim.GraphicObjects
        for key in go.Keys:
            if str(go[key].Tag) == tag:
                return go[key]
        raise KeyError(f"No graphic object with tag '{tag}'")

    def _read_stream(self, obj) -> dict:
        T    = float(obj.GetPropertyValue("PROP_MS_0"))
        P    = float(obj.GetPropertyValue("PROP_MS_1"))
        flow = float(obj.GetPropertyValue("PROP_MS_2"))

        # Read overall (Phase 0) mole fractions via reflection
        composition = {}
        try:
            phases = _get_phases(obj)
            ph0 = phases[0]
            pp_prop = ph0.GetType().GetProperty("Properties")
            pp = pp_prop.GetValue(ph0)
            mc_prop = pp.GetType().GetProperty("molecularWeight")
            # Use compound mole fracs from Phase 0
            comp_prop = ph0.GetType().GetProperty("Compounds")
            if comp_prop:
                comps = comp_prop.GetValue(ph0)
                for name in self._compounds:
                    try:
                        c = comps[name]
                        mf_p = c.GetType().GetProperty("MoleFraction")
                        mf = mf_p.GetValue(c) if mf_p else None
                        composition[name] = float(mf) if mf is not None else 0.0
                    except Exception:
                        composition[name] = 0.0
            else:
                composition = {n: 0.0 for n in self._compounds}
        except Exception:
            composition = {n: 0.0 for n in self._compounds}

        return {"T_K": T, "P_Pa": P, "flow_mol_s": flow, "composition": composition}
