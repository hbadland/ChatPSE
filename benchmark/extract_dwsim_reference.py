"""
Extract converged stream/unit conditions + connectivity from a DWSIM .dwxmz.

.dwxmz = zip(XML).  SimulationObjects carry converged conditions (MaterialStream:
temperature/pressure/massflow/molarflow/enthalpy; EnergyStream: EnergyFlow).
GraphicObjects carry friendly Tags + connectors (AttachedFrom/ToObjID) that wire
units<->streams by GUID.  We join them to recover, per material stream:
T/P/flow + source/dest unit; and per unit: type + inlet/outlet streams + duty.

Usage:
  PYTHONPATH=. python3.9 benchmark/extract_dwsim_reference.py <flowsheet.dwxmz>
"""
import sys, zipfile, io
import xml.etree.ElementTree as ET
from collections import defaultdict

_STREAMISH = {"MaterialStream", "EnergyStream"}


def load_xml(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        return z.read([n for n in z.namelist() if n.endswith(".xml")][0])
    return raw


def short_type(t):
    return (t or "").split(".")[-1].split(",")[0].strip()


def parse(path):
    root = ET.fromstring(load_xml(path))

    # GUID -> (tag, type) and connectivity from GraphicObjects
    tag_of, type_of = {}, {}
    out_edges = defaultdict(list)   # unit_guid -> [stream_guid]  (unit produces)
    in_edges = defaultdict(list)    # unit_guid -> [stream_guid]  (unit consumes)
    for g in root.find("GraphicObjects").findall("GraphicObject"):
        name = g.findtext("Name")
        tag_of[name] = g.findtext("Tag") or name
        type_of[name] = short_type(g.findtext("ObjectType") or g.findtext("Type"))
        for c in (g.find("OutputConnectors") or []):
            if c.attrib.get("AttachedToConnIndex") is not None and \
               c.attrib.get("ConnType") in ("ConOut", None) and \
               c.attrib.get("AttachedToObjID"):
                out_edges[name].append(c.attrib["AttachedToObjID"])
        for c in (g.find("InputConnectors") or []):
            if c.attrib.get("AttachedFromObjID"):
                in_edges[name].append(c.attrib["AttachedFromObjID"])

    # GUID -> conditions from SimulationObjects
    cond = {}
    import re
    for so in root.find("SimulationObjects").findall("SimulationObject"):
        st = short_type(so.findtext("Type"))
        name = so.findtext("Name")
        blob = ET.tostring(so, encoding="unicode")

        def num(key):
            m = re.search(rf"<{key}>([^<]+)</{key}>", blob, re.I)
            try:
                return float(m.group(1)) if m else None
            except ValueError:
                return None
        if st == "MaterialStream":
            cond[name] = {"kind": "material", "T_K": num("temperature"),
                          "P_Pa": num("pressure"), "massflow": num("massflow"),
                          "molarflow": num("molarflow"), "enthalpy": num("enthalpy")}
        elif st == "EnergyStream":
            cond[name] = {"kind": "energy", "EnergyFlow_W": num("EnergyFlow")}

    return tag_of, type_of, in_edges, out_edges, cond


def main():
    path = sys.argv[1]
    tag_of, type_of, in_edges, out_edges, cond = parse(path)

    # source/dest unit for each material stream (inverse of unit edges)
    src_unit, dst_unit = {}, {}
    for u, streams in out_edges.items():
        for s in streams:
            src_unit[s] = u
    for u, streams in in_edges.items():
        for s in streams:
            dst_unit[s] = u

    print("=" * 90)
    print(f"MATERIAL STREAMS — {path.split('/')[-1]}")
    print(f"{'tag':16}{'T_K':>10}{'P_Pa':>14}{'massflow':>12}{'molarflow':>12}  src->dst")
    print("-" * 90)
    for guid, c in cond.items():
        if c["kind"] != "material":
            continue
        s = tag_of.get(guid, guid)
        su = tag_of.get(src_unit.get(guid), "FEED")
        du = tag_of.get(dst_unit.get(guid), "PRODUCT")
        T = f"{c['T_K']:.2f}" if c["T_K"] is not None else "?"
        P = f"{c['P_Pa']:.0f}" if c["P_Pa"] is not None else "?"
        mf = f"{c['massflow']:.5g}" if c["massflow"] is not None else "?"
        nf = f"{c['molarflow']:.5g}" if c["molarflow"] is not None else "?"
        print(f"  {s:14}{T:>10}{P:>14}{mf:>12}{nf:>12}  {su} -> {du}")

    print("\n" + "=" * 90)
    print("UNITS (+ attached energy duty)")
    print(f"{'tag':16}{'type':16}{'duty/work (W)':>16}")
    print("-" * 90)
    for guid, t in type_of.items():
        if t in _STREAMISH or not t:
            continue
        # energy stream attached to this unit
        duty = None
        for s in in_edges.get(guid, []) + out_edges.get(guid, []):
            if cond.get(s, {}).get("kind") == "energy":
                duty = cond[s]["EnergyFlow_W"]
        d = f"{duty:.2f}" if duty is not None else "-"
        print(f"  {tag_of.get(guid, guid):14}{t:16}{d:>16}")


if __name__ == "__main__":
    main()
