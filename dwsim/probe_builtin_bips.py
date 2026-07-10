"""
Probe DWSIM's BUILT-IN NRTL/UNIQUAC interaction-parameter database for specific
binary pairs. Reads the parameters DWSIM ships with (Auxiliary.NRTL()/UNIQUAC()
load the embedded database on construction) — these are established, citable
values, NOT UNIFAC estimates and NOT fabricated. Runs inside the DWSIM container.
"""
import clr, System
clr.AddReference("/usr/local/lib/dwsim/DWSIM.Thermodynamics.dll")
from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import NRTL, UNIQUAC

# Pairs still missing from our corpus (DWSIM canonical names; probe tries variants).
PAIRS = [
    ("Toluene",       "Pyridine"),
    ("Acetonitrile",  "Isopropanol"),
    ("Acetonitrile",  "Ethyleneglycol"),
    ("Isopropanol",   "Ethyleneglycol"),
    ("Ethanol",       "P-Xylene"),
]
# name variants DWSIM may key on
VARIANTS = {
    "Isopropanol":  ["Isopropanol", "2-propanol", "Isopropyl alcohol", "Propan-2-ol"],
    "Ethyleneglycol": ["Ethyleneglycol", "Ethylene glycol", "1,2-Ethanediol", "Ethylene Glycol"],
    "P-Xylene":     ["P-Xylene", "p-Xylene", "p-xylene", "1,4-Dimethylbenzene"],
    "Pyridine":     ["Pyridine"],
    "Acetonitrile": ["Acetonitrile"],
    "Toluene":      ["Toluene"],
    "Ethanol":      ["Ethanol"],
}


def _read_ip(aux, field_names):
    """Return {(outer,inner): {field:val}} from an Auxiliary DB object."""
    ip = aux.InteractionParameters
    out = {}
    for outer in ip.Keys:
        inner_d = ip[outer]
        for inner in inner_d.Keys:
            e = inner_d[inner]
            et = e.GetType()
            rec = {}
            for f in field_names:
                fld = et.GetField(f)
                if fld is not None:
                    rec[f] = fld.GetValue(e)
            out[(str(outer), str(inner))] = rec
    return out


def _find(db, a_variants, b_variants):
    """Look up a pair under any name variant, both orderings."""
    keys = {(o.lower(), i.lower()): (o, i) for (o, i) in db.keys()}
    for a in a_variants:
        for b in b_variants:
            for (o, i) in ((a, b), (b, a)):
                hit = keys.get((o.lower(), i.lower()))
                if hit:
                    return hit, db[hit]
    return None, None


def main():
    nrtl = NRTL()
    uniquac = UNIQUAC()
    nrtl_db = _read_ip(nrtl, ["A12", "A21", "alpha12", "B12", "B21"])
    uni_db = _read_ip(uniquac, ["A12", "A21", "B12", "B21"])
    print(f"DWSIM built-in DB sizes: NRTL pairs={len(nrtl_db)}  UNIQUAC pairs={len(uni_db)}")
    # sample a few keys to confirm key format (names vs CAS)
    print("sample NRTL keys:", list(nrtl_db.keys())[:5])

    for a, b in PAIRS:
        av, bv = VARIANTS.get(a, [a]), VARIANTS.get(b, [b])
        print(f"\n=== {a} + {b} ===")
        k, rec = _find(nrtl_db, av, bv)
        if rec:
            print(f"  NRTL   FOUND key={k}  {rec}")
        else:
            print(f"  NRTL   not in built-in DB")
        k2, rec2 = _find(uni_db, av, bv)
        if rec2:
            print(f"  UNIQUAC FOUND key={k2}  {rec2}")
        else:
            print(f"  UNIQUAC not in built-in DB")


if __name__ == "__main__":
    main()
