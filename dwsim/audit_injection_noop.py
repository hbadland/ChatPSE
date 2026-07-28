import sys, json, itertools
sys.path.insert(0,"/workspaces/multiAgentFlowsheet")
from dwsim.dwsim_wrapper import DWSIMFlowsheet
from rag.retriever import BIPRetriever, ThermoRetriever

# ---- DWSIM canonical-name resolver (fresh sim per compound; primary-only + any) ----
_pcache={}
def canon_primary(name):
    """DWSIM canonical Name when injection passes exactly `name` (the corpus primary)."""
    if name in _pcache: return _pcache[name]
    fs=DWSIMFlowsheet()
    try: fs._sim.AddCompound(name)
    except Exception: pass
    ks=list(fs._sim.SelectedCompounds.Keys)
    r=ks[0] if ks else None
    _pcache[name]=r; return r
def variants(n):
    n=n.strip(); yield n; yield n.lower(); yield (n[0].lower()+n[1:]) if n else n
    yield n.replace("N-","n-"); yield n.replace("n-","N-"); yield n.title(); yield n.capitalize()
_acache={}
def canon_any(name, aliases=()):
    key=(name,tuple(aliases))
    if key in _acache: return _acache[key]
    got=None
    for cand in [name,*aliases]:
        for v in variants(cand):
            fs=DWSIMFlowsheet()
            try: fs._sim.AddCompound(v)
            except Exception: pass
            ks=list(fs._sim.SelectedCompounds.Keys)
            if ks: got=ks[0]; break
        if got: break
    _acache[key]=got; return got

corpus=json.load(open("/workspaces/multiAgentFlowsheet/rag/sources/binary_parameters.json"))
# ---- 1(a): records whose primary spelling != DWSIM name (no-op at injection) ----
def rec_noop(r):
    ca=canon_primary(r["compound_a"]); cb=canon_primary(r["compound_b"])
    na=(ca!=r["compound_a"]); nb=(cb!=r["compound_b"])
    # distinguish absent vs mismatch using canon_any
    return na, nb, ca, cb
tally={"NRTL":[0,0], "UNIQUAC":[0,0]}  # [total, noop]
noop_records=[]
for r in corpus:
    m=r["model"]; tally[m][0]+=1
    na,nb,ca,cb=rec_noop(r)
    if na or nb:
        tally[m][1]+=1
        noop_records.append((m,r["compound_a"],r["compound_b"],ca,cb))
print("### STEP 1(a): corpus records that SILENTLY NO-OP at injection (primary != DWSIM Name) ###")
for m in ("NRTL","UNIQUAC"):
    print(f"  {m}: {tally[m][1]} of {tally[m][0]} records no-op")
print(f"  distinct offending primary spellings:")
from collections import Counter
off=Counter()
for m,a,b,ca,cb in noop_records:
    if canon_primary(a)!=a: off[(a,canon_primary(a))]+=1
    if canon_primary(b)!=b: off[(b,canon_primary(b))]+=1
for (spell,dw),n in off.most_common():
    print(f"     '{spell}' -> DWSIM '{dw}'  ({n} records)")

# ---- 1(b)/(c): cross-reference against benchmark cases ----
cases={}
import glob
for f in glob.glob("/workspaces/multiAgentFlowsheet/benchmark/cases/*.json"):
    try: d=json.load(open(f))
    except: continue
    for c in (d if isinstance(d,list) else d.get("cases",[])):
        if isinstance(c,dict) and c.get("compounds"): cases.setdefault(c["id"],c["compounds"])
bip=BIPRetriever(); tr=ThermoRetriever()
def classify(comps):
    # would this case inject NRTL? (polar/azeo AND full NRTL coverage)
    classes=tr._classify(comps); azeo=tr._has_azeotrope(comps)
    is_polar=bool(classes & {"ALCOHOLS","KETONES","ESTERS","ETHERS","POLAR_OTHER","WATER"})
    found,missing=bip.query(comps,"NRTL")
    if not (is_polar or azeo) or missing:
        return None  # does not inject NRTL (EOS/ideal route or no coverage)
    n=len(found); noop=0
    for bp in found:
        if canon_primary(bp["compound_a"])!=bp["compound_a"] or canon_primary(bp["compound_b"])!=bp["compound_b"]:
            noop+=1
    return (n,noop)

REPORTED_30=["C1","C2","C3","EASY_01","EASY_02","EASY_04","F1","F2","F3","F4","GEN_01",
 "GEN_03","M1","P1","P2","P3","S1","S2","SAN_03","SAN_04"]+[f"VAL_{i:02d}" for i in range(1,11)]

def summarise(ids, label):
    inj=full=part=clean=0; rows=[]
    for cid in ids:
        if cid not in cases: continue
        r=classify(cases[cid])
        if r is None: continue
        inj+=1; n,noop=r
        tag="FULLY IDEAL (all pairs no-op)" if noop==n else ("PARTIAL" if noop>0 else "clean")
        if noop==n: full+=1
        elif noop>0: part+=1
        else: clean+=1
        if noop>0: rows.append((cid,n,noop,tag))
    print(f"\n### {label}: {inj} inject NRTL | fully-ideal={full} partial={part} clean={clean} ###")
    for cid,n,noop,tag in rows: print(f"    {cid:10} pairs={n} no-op={noop}  {tag}")
summarise(REPORTED_30,"1(c) THE 30 REPORTED (20 consistency + 10 VAL)")
summarise(list(cases.keys()),"1(b) ALL 130")

# ---- refine: for the 30, show affected cases + whether DWSIM built-in covers the no-op pair ----
fs0=DWSIMFlowsheet(); fs0.add_compounds(["Ethanol","Water"]); fs0.set_property_package("NRTL")
pkg=fs0._property_packages["NRTL"]; muni=pkg.GetType().GetProperty("m_uni").GetValue(pkg)
ipD=muni.GetType().GetProperty("InteractionParameters").GetValue(muni)
DB=set(frozenset((str(o),str(i))) for o in ipD.Keys for i in ipD[o].Keys)
def builtin_has(a,b):
    ra=canon_any(a); rb=canon_any(b)
    return (ra and rb and frozenset((ra,rb)) in DB)
print("\n=== 1(c) THE 30 — affected cases in detail (no-op pairs + DWSIM built-in fallback?) ===")
for cid in REPORTED_30:
    if cid not in cases: continue
    comps=cases[cid]; found,missing=bip.query(comps,"NRTL")
    classes=tr._classify(comps); azeo=tr._has_azeotrope(comps)
    is_polar=bool(classes & {"ALCOHOLS","KETONES","ESTERS","ETHERS","POLAR_OTHER","WATER"})
    if missing or not (is_polar or azeo): continue
    bad=[]
    for bp in found:
        a,b=bp["compound_a"],bp["compound_b"]
        if canon_primary(a)!=a or canon_primary(b)!=b:
            bad.append((a,b,"DWSIM-builtin" if builtin_has(a,b) else "NO builtin -> IDEAL"))
    if bad:
        alln = len(bad)==len(found)
        print(f"  {cid}: {'ALL' if alln else 'SOME'} pairs no-op ({len(bad)}/{len(found)})")
        for a,b,fb in bad: print(f"       {a}/{b}  [{fb}]")

print("\n=== CONSOLIDATED: the 30 reported — every NRTL-injecting case ===")
for cid in REPORTED_30:
    if cid not in cases: continue
    comps=cases[cid]; found,missing=bip.query(comps,"NRTL")
    classes=tr._classify(comps); azeo=tr._has_azeotrope(comps)
    is_polar=bool(classes & {"ALCOHOLS","KETONES","ESTERS","ETHERS","POLAR_OTHER","WATER"})
    if missing or not (is_polar or azeo): continue
    n=len(found); noop=sum(1 for bp in found if canon_primary(bp["compound_a"])!=bp["compound_a"] or canon_primary(bp["compound_b"])!=bp["compound_b"])
    ideal=sum(1 for bp in found if (canon_primary(bp["compound_a"])!=bp["compound_a"] or canon_primary(bp["compound_b"])!=bp["compound_b"]) and not builtin_has(bp["compound_a"],bp["compound_b"]))
    tag="CLEAN" if noop==0 else ("ALL-NOOP" if noop==n else "PARTIAL")
    print(f"  {cid:10} compounds={comps}  found={n} noop={noop} ideal={ideal} -> {tag}")
