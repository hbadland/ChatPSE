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
    "Heater":            ObjectType.Heater,
    "Cooler":            ObjectType.Cooler,
    "Vessel":            ObjectType.Vessel,
    "Mixer":             ObjectType.Mixer,
    "Splitter":          ObjectType.Splitter,
    "Compressor":        ObjectType.Compressor,
    "Expander":          ObjectType.Expander,
    "Pump":              ObjectType.Pump,
    "MaterialStream":    ObjectType.MaterialStream,
    # Member name varies by DWSIM build (e.g. RCT_Conversion in current build);
    # discovery loop in add_unit() is the runtime fallback when these are absent.
    "ConversionReactor": getattr(ObjectType, "RCT_Conversion",
                          getattr(ObjectType, "OT_React_Conversion",
                          getattr(ObjectType, "React_Conversion", None))),
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


def _parse_reaction_stoich(reaction_str: str) -> "tuple[dict, dict]":
    """Parse 'A + 2B → 3C + D' into ({A:1, B:2}, {C:3, D:1}).

    Accepts → ⇌ = -> as arrow.  Coefficient may precede name with a space
    ('3 H2') or be adjacent ('3H2').  Returns empty dicts on parse failure.
    """
    import re
    s = reaction_str.replace("→", "->").replace("⇌", "->").replace("=", "->")
    if "->" not in s:
        return {}, {}
    lhs, rhs = s.split("->", 1)

    def _side(text: str) -> dict:
        result: dict = {}
        for token in re.split(r'\s*\+\s*', text.strip()):
            token = token.strip()
            if not token:
                continue
            # "3 H2" or "3.5 CO2"
            m = re.match(r'^(\d+(?:\.\d+)?)\s+(.+)$', token)
            if m:
                result[m.group(2).strip()] = float(m.group(1))
                continue
            # "3H2" (digit-letter boundary, no space)
            m2 = re.match(r'^(\d+(?:\.\d+)?)([A-Za-z].*)$', token)
            if m2:
                result[m2.group(2).strip()] = float(m2.group(1))
                continue
            result[token] = 1.0
        return result

    return _side(lhs), _side(rhs)


def _find_clr_type(fullname: str):
    """Return a loaded .NET type by full name, or None.

    Uses Assembly.GetType(name) (no GetTypes() enumeration), which avoids the
    ReflectionTypeLoadException that some DWSIM assemblies throw when fully
    enumerated.
    """
    for asm in System.AppDomain.CurrentDomain.GetAssemblies():
        try:
            t = asm.GetType(fullname)
        except Exception:
            t = None
        if t is not None:
            return t
    return None


def _set_clr_prop(obj, name: str, value) -> bool:
    """Set a writable .NET property by name; return True on success."""
    p = obj.GetType().GetProperty(name)
    if p is not None and p.CanWrite:
        try:
            p.SetValue(obj, value)
            return True
        except Exception:
            return False
    return False


def _make_stoich(stoich_type, comp_name: str, coeff: float, is_base: bool):
    """Construct a ReactionStoichBase via its parameterised constructor.

    ctor signature: (name, stoichCoeff, isBaseReactant, directOrder, reverseOrder).
    coeff < 0 for reactants, > 0 for products. Orders are unused by conversion
    reactions but the ctor requires them, so pass |coeff|.
    """
    args = System.Array[System.Object]([
        System.String(comp_name),
        System.Double(float(coeff)),
        System.Boolean(bool(is_base)),
        System.Double(abs(float(coeff))),
        System.Double(abs(float(coeff))),
    ])
    return System.Activator.CreateInstance(stoich_type, args)


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
            # Note: FlashAlgorithm on the property package is not a settable enum
            # in this DWSIM version.  Flash algorithm selection is done per-Vessel
            # via PreferredFlashAlgorithmTag (see set_vessel).

    def set_unit_property_package(self, unit_tag: str, pp_name: str) -> None:
        """Assign a property package to a single unit operation."""
        if pp_name not in self._property_packages:
            self.set_property_package(pp_name)
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
        # ConversionReactor ObjectType varies by DWSIM build — discover at runtime.
        if ot is None and unit_type == "ConversionReactor":
            for attr_name in dir(ObjectType):
                _n = attr_name.lower()
                # Build-dependent name: match "conversion" reactor whether the
                # prefix is "react" (React_Conversion) or "rct" (RCT_Conversion).
                if "conv" in _n and ("react" in _n or "rct" in _n):
                    ot = getattr(ObjectType, attr_name, None)
                    if ot is not None:
                        _UNIT_TYPE_MAP["ConversionReactor"] = ot  # cache for next call
                        print(f"  [DWSIM] ConversionReactor ObjectType discovered: {attr_name}",
                              flush=True, file=sys.stderr)
                        break
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

        # Force NestedLoops (Rachford-Rice) flash on this Vessel.
        # PreferredFlashAlgorithmTag is a string property on the Vessel object;
        # it takes priority over the property package's default algorithm.
        # NestedLoops is more robust than BostonBrittMainProperty for polar VLE.
        _target_tag = "NestedLoops"
        try:
            # Read current value first so we can log what DWSIM defaults to.
            _current = getattr(obj, "PreferredFlashAlgorithmTag", "<no attr>")
            print(f"  [DWSIM] Vessel {tag} PreferredFlashAlgorithmTag "
                  f"before={_current!r}", flush=True, file=sys.stderr)
            # Try direct attribute assignment first.
            obj.PreferredFlashAlgorithmTag = _target_tag
            print(f"  [DWSIM] Vessel {tag} PreferredFlashAlgorithmTag "
                  f"→ {_target_tag!r} (direct)", flush=True, file=sys.stderr)
        except Exception as _direct_err:
            # Fall back to reflection if direct assignment raises.
            try:
                t    = obj.GetType()
                prop = t.GetProperty("PreferredFlashAlgorithmTag")
                if prop is not None and prop.CanWrite:
                    prop.SetValue(obj, _target_tag)
                    print(f"  [DWSIM] Vessel {tag} PreferredFlashAlgorithmTag "
                          f"→ {_target_tag!r} (reflection)", flush=True, file=sys.stderr)
                else:
                    print(f"  [DWSIM] Vessel {tag} PreferredFlashAlgorithmTag "
                          f"not writable ({_direct_err})", flush=True, file=sys.stderr)
            except Exception as _refl_err:
                print(f"  [DWSIM] Vessel {tag} flash tag not set "
                      f"({_direct_err}; {_refl_err})", flush=True, file=sys.stderr)

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

    def set_conversion_reactor(self, tag: str, temperature_K: float,
                               pressure_Pa: float, conversion: float,
                               reaction: str = "") -> None:
        """Configure a DWSIM conversion reactor (OT_React_Conversion).

        Sets outlet temperature, pressure drop to zero (feed pressure maintained),
        and wires up a conversion reaction via the flowsheet reaction manager.

        temperature_K : reactor outlet temperature [K]
        pressure_Pa   : reactor operating pressure [Pa] (used only for logging;
                        DWSIM conversion reactors maintain feed pressure by default)
        conversion    : fractional conversion of limiting reactant (0–1)
        reaction      : stoichiometry string, e.g. "CH4 + H2O → CO + 3H2"
        """
        obj    = self._sim.GetFlowsheetSimulationObject(tag)
        t_type = obj.GetType()

        # ── Temperature ───────────────────────────────────────────────────────
        # Try OutletTemperature (Nullable<Double>) then Temperature (Double)
        _set_ok = False
        for prop_name in ("OutletTemperature", "Temperature"):
            prop = t_type.GetProperty(prop_name)
            if prop is not None and prop.CanWrite:
                try:
                    prop.SetValue(obj, System.Nullable[System.Double](float(temperature_K)))
                    _set_ok = True
                    break
                except Exception:
                    try:
                        prop.SetValue(obj, System.Double(float(temperature_K)))
                        _set_ok = True
                        break
                    except Exception:
                        pass
        if not _set_ok:
            print(f"  [DWSIM] ConversionReactor {tag}: could not set OutletTemperature",
                  flush=True, file=sys.stderr)

        # ── Enforce the outlet temperature (Stage 3a — probe-validated) ───────
        # DWSIM's ConversionReactor defaults to Adiabatic, so the OutletTemperature
        # set above is INERT and an endothermic reaction (e.g. SMR) collapses the
        # whole train to a non-physical temperature.  Switch ReactorOperationMode
        # to OutletTemperature AND wire an energy stream to the reactor's energy
        # port (port 2) to carry the reaction duty — DWSIM requires both for the
        # mode to solve.  GUARD: only when a valid target temperature exists; with
        # none, leave the reactor in its default (adiabatic) mode.
        if temperature_K is not None and 50.0 < float(temperature_K) < 3000.0:
            mode_set = False
            for mode_prop in ("ReactorOperationMode", "OperationMode"):
                mp = t_type.GetProperty(mode_prop)
                if mp is None or not mp.CanWrite:
                    continue
                try:
                    mp.SetValue(obj, System.Enum.Parse(mp.PropertyType, "OutletTemperature"))
                    mode_set = True
                    break
                except Exception:
                    pass

            energy_wired = False
            en_tag = f"{tag}-EN"
            for es_name in ("EnergyStream", "OT_EnergyStream"):
                es_ot = getattr(ObjectType, es_name, None)
                if es_ot is None:
                    continue
                try:
                    self._sim.AddObject(es_ot, 0, 0, en_tag)
                    self._sim.ConnectObjects(
                        self._get_graphic_object(tag),
                        self._get_graphic_object(en_tag), 2, 0)   # reactor port 2 → energy
                    energy_wired = True
                    break
                except Exception:
                    pass

            print(f"  [DWSIM] ConversionReactor {tag}: OutletTemperature mode="
                  f"{mode_set}, energy stream {en_tag} wired={energy_wired}, "
                  f"T_out={float(temperature_K):.1f} K", flush=True, file=sys.stderr)
            if not (mode_set and energy_wired):
                print(f"  [DWSIM] WARNING: {tag} could not fully enforce outlet T "
                      f"(mode={mode_set}, energy={energy_wired}) — reactor may fall "
                      f"back to adiabatic", flush=True, file=sys.stderr)
        else:
            print(f"  [DWSIM] ConversionReactor {tag}: no valid target temperature "
                  f"({temperature_K}) — left in default (adiabatic) mode",
                  flush=True, file=sys.stderr)

        # ── Pressure drop → 0 (preserves feed pressure) ───────────────────────
        for prop_name in ("PressureDrop", "DeltaP"):
            prop = t_type.GetProperty(prop_name)
            if prop is not None and prop.CanWrite:
                try:
                    prop.SetValue(obj, System.Double(0.0))
                    break
                except Exception:
                    pass

        # ── Reaction setup ────────────────────────────────────────────────────
        rxn_id = self._setup_conversion_reaction(tag, reaction, conversion)
        if rxn_id is None:
            print(
                f"  [DWSIM] ConversionReactor {tag}: reaction setup failed — "
                f"T={temperature_K:.1f} K and P_drop=0 set; conversion not applied. "
                "Flowsheet may not solve correctly.",
                flush=True, file=sys.stderr)
            return

        # Point the reactor at the reaction set the reaction was registered in.
        # _setup_conversion_reaction added it to "DefaultSet" and encoded the
        # conversion in the reaction's Expression (percent); the reactor reads
        # the active reactions from this set at solve time.
        for set_prop in ("ReactionSetID", "ReactionSetName"):
            prop = t_type.GetProperty(set_prop)
            if prop is not None and prop.CanWrite:
                try:
                    prop.SetValue(obj, System.String("DefaultSet"))
                except Exception:
                    pass
        print(
            f"  [DWSIM] ConversionReactor {tag}: reaction {rxn_id[:8]}… registered "
            f"in DefaultSet, conversion={conversion:.4f} (Expression in %)",
            flush=True, file=sys.stderr)

    def _setup_conversion_reaction(self, reactor_tag: str,
                                   reaction_str: str,
                                   conversion: float) -> "Optional[str]":
        """Create and register a conversion reaction, returning its GUID (or None).

        Matches the DWSIM 9.x reaction API in this build:
          - Reaction class : DWSIM.Thermodynamics.BaseClasses.Reaction
          - Stoichiometry  : DWSIM.Thermodynamics.BaseClasses.ReactionStoichBase
                             (ctor: name, coeff, isBaseReactant, directOrder, reverseOrder;
                              coeff < 0 for reactants, > 0 for products)
          - Components     : Dictionary<String, IReactionStoichBase> keyed by compound
          - Conversion     : reaction.Expression in PERCENT (e.g. 90.02 ⇒ 0.9002)
          - Registration   : flowsheet.Reactions[id] = rxn, and a ReactionSetBase
                             entry in ReactionSets["DefaultSet"].Reactions
        """
        try:
            reactants, products = _parse_reaction_stoich(reaction_str)
            if not reactants or not products:
                return None

            rxn_type    = _find_clr_type("DWSIM.Thermodynamics.BaseClasses.Reaction")
            stoich_type = _find_clr_type("DWSIM.Thermodynamics.BaseClasses.ReactionStoichBase")
            rsb_type    = _find_clr_type("DWSIM.Thermodynamics.BaseClasses.ReactionSetBase")
            if rxn_type is None or stoich_type is None or rsb_type is None:
                return None

            rxn_id  = str(System.Guid.NewGuid())
            rxn_obj = System.Activator.CreateInstance(rxn_type)
            comp_lower = {c.lower(): c for c in self._compounds}

            # Limiting/base reactant = first reactant in the equation.
            base_name = comp_lower.get(next(iter(reactants)).lower(),
                                       next(iter(reactants)))

            _set_clr_prop(rxn_obj, "ID",   System.String(rxn_id))
            _set_clr_prop(rxn_obj, "Name", System.String(f"Rxn_{reactor_tag}"))
            _set_clr_prop(rxn_obj, "BaseReactant", System.String(base_name))
            _set_clr_prop(rxn_obj, "Equation",     System.String(reaction_str))

            # ReactionType = Conversion (enum on the property's own type).
            rt_prop = rxn_type.GetProperty("ReactionType")
            if rt_prop is not None and rt_prop.CanWrite:
                try:
                    rt_prop.SetValue(rxn_obj,
                                     System.Enum.Parse(rt_prop.PropertyType, "Conversion"))
                except Exception:
                    pass
            # ReactionPhase = Mixture (gas/liquid agnostic; reactor handles VLE).
            rp_prop = rxn_type.GetProperty("ReactionPhase")
            if rp_prop is not None and rp_prop.CanWrite:
                try:
                    rp_prop.SetValue(rxn_obj,
                                     System.Enum.Parse(rp_prop.PropertyType, "Mixture"))
                except Exception:
                    pass

            # Components: Dictionary<String, IReactionStoichBase> keyed by compound.
            comps = rxn_type.GetProperty("Components").GetValue(rxn_obj)
            for formula, coeff in reactants.items():
                name = comp_lower.get(formula.lower(), formula)
                comps[System.String(name)] = _make_stoich(
                    stoich_type, name, -abs(coeff), is_base=(name == base_name))
            for formula, coeff in products.items():
                name = comp_lower.get(formula.lower(), formula)
                comps[System.String(name)] = _make_stoich(
                    stoich_type, name, abs(coeff), is_base=False)

            # Conversion expression — DWSIM evaluates this as a PERCENTAGE.
            _set_clr_prop(rxn_obj, "Expression",
                          System.String(repr(float(conversion) * 100.0)))

            # Register the reaction with the flowsheet and the DefaultSet.
            self._sim.Reactions[System.String(rxn_id)] = rxn_obj
            default_set = self._sim.ReactionSets[System.String("DefaultSet")]
            rsb = System.Activator.CreateInstance(rsb_type)
            _set_clr_prop(rsb, "ReactionID", System.String(rxn_id))
            _set_clr_prop(rsb, "IsActive",   System.Boolean(True))
            _set_clr_prop(rsb, "Rank",       System.Int32(0))
            default_set.Reactions[System.String(rxn_id)] = rsb

            return rxn_id

        except Exception as exc:
            print(f"  [DWSIM] _setup_conversion_reaction for {reactor_tag}: {exc}",
                  flush=True, file=sys.stderr)
            return None

    def add_recycle_block(self, tag: str, inlet_stream: str,
                          outlet_stream: str, tol: float = 1e-4) -> None:
        """Add a DWSIM Recycle convergence block between two tear-stream endpoints.

        inlet_stream  : tag of the calculated stream (outlet of downstream unit)
        outlet_stream : tag of the assumed/initial stream (inlet of upstream unit)
        tol           : convergence tolerance applied to T, P, and flow

        Uses ObjectType.OT_Recycle when available; falls back to assembly
        reflection searching for any class with 'recycle' in its name.
        """
        import System

        # ── 1. Add the recycle block object ───────────────────────────────────
        # Must select the MATERIAL recycle (OT_Recycle), not OT_EnergyRecycle:
        # an energy-recycle block only accepts energy streams, so connecting a
        # material tear stream to it fails with "connection cannot be done".
        # A bare dir() scan picks OT_EnergyRecycle first (alphabetical), so
        # prefer OT_Recycle explicitly, then any 'recycle' name without 'energy'.
        recycle_ot = getattr(ObjectType, "OT_Recycle", None)
        if recycle_ot is None:
            for attr_name in dir(ObjectType):
                n = attr_name.lower()
                if "recycle" in n and "energy" not in n:
                    recycle_ot = getattr(ObjectType, attr_name, None)
                    if recycle_ot is not None:
                        break
        if recycle_ot is None:
            # Last resort: any recycle-like ObjectType (may be energy).
            for attr_name in dir(ObjectType):
                if "recycle" in attr_name.lower():
                    recycle_ot = getattr(ObjectType, attr_name, None)
                    if recycle_ot is not None:
                        break

        if recycle_ot is not None:
            self._sim.AddObject(recycle_ot, 0, 0, tag)
        else:
            # Reflection fallback: search all loaded assemblies for a Recycle class.
            recycle_cls = None
            for assembly in System.AppDomain.CurrentDomain.GetAssemblies():
                try:
                    matched = [t for t in assembly.GetTypes()
                               if "recycle" in t.Name.lower()
                               and t.IsClass and not t.IsAbstract]
                except Exception:
                    continue
                if matched:
                    recycle_cls = matched[0]
                    break
            if recycle_cls is None:
                raise RuntimeError(
                    "DWSIM Recycle class not found in any loaded assembly. "
                    "Ensure DWSIM.UnitOperations.dll is referenced.")
            instance = System.Activator.CreateInstance(recycle_cls)
            for pname in ("Tag", "ComponentName", "Name"):
                p = recycle_cls.GetProperty(pname)
                if p is not None and p.CanWrite:
                    p.SetValue(instance, System.String(tag))
                    break
            fs_type = self._sim.GetType()
            added = False
            for mname in ("AddSimulationObject", "AddFlowsheetObject"):
                for mi in fs_type.GetMethods():
                    if mi.Name == mname:
                        try:
                            mi.Invoke(self._sim, [instance])
                            added = True
                            break
                        except Exception:
                            pass
                if added:
                    break
            if not added:
                raise RuntimeError(
                    f"Could not register Recycle block '{tag}' with the flowsheet.")

        # ── 2. Set convergence tolerances (best-effort; try all known names) ──
        obj = self._sim.GetFlowsheetSimulationObject(tag)
        if obj is not None:
            t_type = obj.GetType()
            for pname in ("ConvergenceTolerance", "Tolerance",
                          "TemperatureTolerance", "PressureTolerance",
                          "FlowTolerance", "MaximumResidual"):
                prop = t_type.GetProperty(pname)
                if prop is not None and prop.CanWrite:
                    try:
                        prop.SetValue(obj, System.Double(tol))
                    except Exception:
                        pass

        # ── 3. Wire streams through the block ─────────────────────────────────
        inlet_go   = self._get_graphic_object(inlet_stream)
        outlet_go  = self._get_graphic_object(outlet_stream)
        recycle_go = self._get_graphic_object(tag)
        # calculated stream → recycle block inlet
        self._sim.ConnectObjects(inlet_go,   recycle_go, 0, 0)
        # recycle block outlet → assumed/init stream
        self._sim.ConnectObjects(recycle_go, outlet_go,  0, 0)

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

                # Always harvest per-unit ErrorMessage so we know which unit
                # operation failed and why — not just a generic "did not converge".
                # This runs even when CalculateFlowsheet2 already returned errors,
                # because that list is often just "Object X error" with no detail.
                if not solved:
                    go = self._sim.GraphicObjects
                    unit_msgs: list[str] = []
                    for key in go.Keys:
                        gobj = go[key]
                        if str(gobj.ObjectType) == "MaterialStream":
                            continue
                        try:
                            obj = self._sim.GetFlowsheetSimulationObject(
                                str(gobj.Tag))
                            msg = str(getattr(obj, "ErrorMessage", None) or "")
                            if msg and msg.lower() not in ("", "none", "null"):
                                unit_msgs.append(f"{gobj.Tag}: {msg}")
                        except Exception:
                            pass
                    # Prepend unit-level messages so the classifier sees them first.
                    # Deduplicate against anything CalculateFlowsheet2 already added.
                    existing = {m.lower() for m in errors}
                    for m in unit_msgs:
                        if m.lower() not in existing:
                            errors.insert(0, m)
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

        # Read vapor phase mole fraction from Phase[1].Properties.molarfraction
        vapor_fraction = 0.0
        try:
            phases = _get_phases(obj)
            ph1 = phases[1]
            ph1_pp_prop = ph1.GetType().GetProperty("Properties")
            ph1_pp = ph1_pp_prop.GetValue(ph1)
            vf_prop = ph1_pp.GetType().GetProperty("molarfraction")
            if vf_prop is not None:
                vf = vf_prop.GetValue(ph1_pp)
                vapor_fraction = float(vf) if vf is not None else 0.0
        except Exception:
            pass

        return {"T_K": T, "P_Pa": P, "flow_mol_s": flow, "composition": composition,
                "vapor_fraction": vapor_fraction}
