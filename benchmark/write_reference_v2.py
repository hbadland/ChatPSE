"""Reference-JSON writer v2.

Parse a converged DWSIM .dwxmz (zip of XML) with zipfile + xml.etree only — no
DWSIM runtime — and emit the existing reference schema WITH connectivity and
is_feed populated (both lost when the VAL references were first written).

Per material stream: T/P/molar-flow/vapour-fraction/overall composition from the
Mixture phase; is_feed / is_product from having no source / no dest unit in the
GraphicObject connector graph. Nothing is hand-edited — every value is parsed.
"""
import zipfile, io, re, json, os, math
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_xml(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        return z.read([n for n in z.namelist() if n.endswith(".xml")][0])
    return raw


def _short(t):
    return (t or "").split(".")[-1].split(",")[0].strip()


# DWSIM object type -> reference-schema unit type. The original DWSIM type is kept
# in each unit's params["_dwsim_type"] so the mapping is auditable.
_UNIT_MAP = {
    "DistillationColumn": "Column", "AbsorptionColumn": "Column", "CapeOpenUO": "Column",
    "RCT_Conversion": "ConversionReactor", "RCT_Equilibrium": "ConversionReactor",
    "RCT_Gibbs": "ConversionReactor", "RCT_PFR": "ConversionReactor", "RCT_CSTR": "ConversionReactor",
    "Heater": "Heater", "Cooler": "Cooler", "HeatExchanger": "Heater",
    "Compressor": "Compressor", "Pump": "Pump", "Expander": "Expander", "Turbine": "Expander",
    "Valve": "Expander", "Vessel": "Vessel", "ComponentSeparator": "Vessel", "Decanter": "Decanter",
    "NodeIn": "Mixer", "NodeOut": "Splitter", "Mixer": "Mixer", "Splitter": "Splitter",
}
_NON_UNIT = {"MaterialStream", "EnergyStream", "OT_Recycle", "OT_EnergyRecycle", "GO_MasterTable",
             "GO_Text", "GO_Table", "GO_FloatingTable", "GO_SpreadsheetTable", ""}


def _num(el, key):
    """Phase-level property (lowercase tag); compound props are CamelCase so a
    case-sensitive match never picks a per-compound <MolarFlow> by accident."""
    m = re.search(rf"<{key}>([^<]+)</{key}>", ET.tostring(el, encoding="unicode"))
    if not m:
        return None
    try:
        v = float(m.group(1))
        return None if math.isnan(v) else v
    except ValueError:
        return None


def extract(path):
    """Return (tag_of, type_of, streams, src, dst). streams[guid] = dict of
    T_K/P_Pa/flow_mol_s/vapor_fraction/composition. src/dst[stream_guid] = unit_guid."""
    root = ET.fromstring(load_xml(path))
    tag_of, type_of = {}, {}
    out_e, in_e = defaultdict(list), defaultdict(list)
    for g in root.find("GraphicObjects").findall("GraphicObject"):
        name = g.findtext("Name")
        tag_of[name] = g.findtext("Tag") or name
        type_of[name] = _short(g.findtext("ObjectType") or g.findtext("Type"))
        for c in (g.find("OutputConnectors") or []):
            if c.attrib.get("AttachedToObjID"):
                out_e[name].append(c.attrib["AttachedToObjID"])
        for c in (g.find("InputConnectors") or []):
            if c.attrib.get("AttachedFromObjID"):
                in_e[name].append(c.attrib["AttachedFromObjID"])
    streams = {}
    for so in root.find("SimulationObjects").findall("SimulationObject"):
        if "MaterialStream" not in (so.findtext("Type") or ""):
            continue
        name = so.findtext("Name")
        mix = vap = None
        for ph in so.find("Phases").findall("Phase"):
            if ph.findtext("Name") == "Mixture":
                mix = ph
            elif ph.findtext("Name") == "Vapor":
                vap = ph
        if mix is None:
            continue
        T, P, F = _num(mix, "temperature"), _num(mix, "pressure"), _num(mix, "molarflow")
        comp = {}
        for cp in mix.find("Compounds").findall("Compound"):
            nm, mf = cp.findtext("Name"), cp.findtext("MoleFraction")
            if nm is not None and mf is not None:
                try:
                    comp[nm] = float(mf)
                except ValueError:
                    pass
        vf = None
        if vap is not None and F:
            vfl = _num(vap, "molarflow")
            if vfl is not None:
                vf = vfl / F
        streams[name] = {"T_K": T, "P_Pa": P, "flow_mol_s": F,
                         "vapor_fraction": vf, "composition": comp}
    matset = set(streams)
    src = {s: u for u, ss in out_e.items() for s in ss if s in matset}
    dst = {s: u for u, ss in in_e.items() for s in ss if s in matset}
    return tag_of, type_of, streams, src, dst


def _property_package(path):
    blob = load_xml(path).decode("utf-8", "ignore")
    m = re.search(r"<PropertyPackages>.*?<Name>([^<]+)</Name>", blob, re.S)
    return m.group(1) if m else None


def _meta(path):
    blob = load_xml(path).decode("utf-8", "ignore")
    out = {}
    for tag in ("BuildVersion", "SimulationName"):
        m = re.search(rf"<{tag}>([^<]*)</{tag}>", blob)
        if m and m.group(1).strip():
            out[tag] = m.group(1).strip()
    return out


def build_reference(path, case_id, case_name):
    """Build the reference-schema dict from a parsed .dwxmz. Values are parsed;
    only case_id/case_name are carried in (identity, decided by content mapping)."""
    tag_of, type_of, streams, src, dst = extract(path)
    meta = _meta(path)
    units = []
    for guid, ty in type_of.items():
        if ty in _NON_UNIT:
            continue
        mapped = _UNIT_MAP.get(ty)
        if mapped is None:
            continue
        units.append({"tag": tag_of.get(guid, guid), "type": mapped,
                      "params": {"_dwsim_type": ty}})
    ref_streams = {}
    for guid, s in streams.items():
        stag = tag_of.get(guid, guid)
        T, P, F, vf = s["T_K"], s["P_Pa"], s["flow_mol_s"], s["vapor_fraction"]
        ref_streams[stag] = {
            "T_K": round(T, 3) if T is not None else None,
            "T_C": round(T - 273.15, 3) if T is not None else None,
            "P_Pa": round(P, 2) if P is not None else None,
            "P_bar": round(P / 1e5, 5) if P is not None else None,
            "flow_mol_s": round(F, 4) if F is not None else None,
            "vapor_fraction": round(vf, 4) if vf is not None else None,
            "composition": {k: round(v, 6) for k, v in s["composition"].items()
                            if not math.isnan(v)},
            "is_feed": guid not in src,
            "is_product": guid not in dst,
        }
    connections = []
    for guid in streams:
        stag = tag_of.get(guid, guid)
        if guid in src:
            connections.append([tag_of.get(src[guid], src[guid]), stag])
        if guid in dst:
            connections.append([stag, tag_of.get(dst[guid], dst[guid])])
    compounds = sorted({c for s in streams.values() for c in s["composition"]})
    return {
        "case_id": case_id,
        "case_name": case_name,
        "source_file": os.path.basename(path),
        "reference_validity": "reconstructed-v2-from-converged-dwxmz",
        "provenance": {
            "source_dwxmz": os.path.basename(path),
            "dwsim_build": meta.get("BuildVersion"),
            "sim_name": meta.get("SimulationName"),
            "note": "re-extracted from the converged .dwxmz with connectivity and "
                    "is_feed/is_product recovered from the GraphicObject connector graph; "
                    "no hand-edited values",
        },
        "compounds": compounds,
        "property_package": _property_package(path),
        "solved": True,
        "units": units,
        "connections": connections,
        "streams": ref_streams,
    }
