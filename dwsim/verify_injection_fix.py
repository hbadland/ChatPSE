import sys
sys.path.insert(0,"/workspaces/multiAgentFlowsheet")
from dwsim.dwsim_wrapper import DWSIMFlowsheet
def flash(comps, comp, T, P, inject=None):
    fs=DWSIMFlowsheet(); fs.add_compounds(comps); fs.set_property_package("NRTL")
    applied=True
    if inject:
        try: fs.set_nrtl_parameters(*inject)
        except Exception as e: applied=f"RAISED: {e}"
    fs.disable_auto_estimate("NRTL")
    fs.add_stream("FEED"); fs.add_unit("V-01","Vessel"); fs.add_stream("VAP"); fs.add_stream("LIQ")
    fs.connect("FEED","V-01"); fs.connect("V-01","VAP",0,0); fs.connect("V-01","LIQ",1,0)
    fs.set_stream("FEED",T,P,1.0,comp); fs.set_vessel("V-01")
    try: fs.solve(timeout=90)
    except Exception as e: return {"solved":False,"applied":applied}
    v=fs.get_stream("VAP")
    return {"solved":bool(fs._sim.Solved),"VAP":{k:round(x,4) for k,x in (v.get('composition') or {}).items()},"applied":applied}

print("### 3a: A/B/C replay MEK/n-hexane (WITH the STEP-2 fix) ###")
C=["Methyl ethyl ketone","n-Hexane"]; cm={"Methyl ethyl ketone":0.5,"n-Hexane":0.5}
A=flash(C,cm,333.0,101325.0,None)
B=flash(C,cm,333.0,101325.0,("Methyl Ethyl Ketone","n-Hexane",1200.0,600.0,0.3))  # corpus spelling
Cc=flash(C,cm,333.0,101325.0,("Methyl ethyl ketone","n-Hexane",1200.0,600.0,0.3)) # DWSIM name
print(f"  A no-inject : solved={A['solved']} VAP={A['VAP']}")
print(f"  B corpus-name inject: applied={B['applied']} solved={B['solved']} VAP={B['VAP']}")
print(f"  C dwsim-name inject : solved={Cc['solved']} VAP={Cc['VAP']}")
print(f"  --> B == C (fix works)? {B['VAP']==Cc['VAP']}   B != A? {B['VAP']!=A['VAP']}")

print("\n### 3c: ethanol/water (names already match) must be UNCHANGED ###")
E=["Ethanol","Water"]; em={"Ethanol":0.5,"Water":0.5}
eB=flash(E,em,351.0,101325.0,("Ethanol","Water",1200.0,600.0,0.3))
eC=flash(E,em,351.0,101325.0,("Ethanol","Water",1200.0,600.0,0.3))
print(f"  ethanol/water inject: applied={eB['applied']} solved={eB['solved']} VAP={eB['VAP']}  (B==C:{eB['VAP']==eC['VAP']})")

print("\n### 3b(pair): VAL_09 P-Xylene resolution via the fix ###")
# does corpus 'P-Xylene' now resolve to the flowsheet key at injection?
fs=DWSIMFlowsheet(); fs.add_compounds(["Ethanol","Benzene","P-Xylene"])
print(f"  SelectedCompounds: {list(fs._sim.SelectedCompounds.Keys)}")
fs.set_property_package("NRTL")
try:
    fs.set_nrtl_parameters("P-Xylene","Ethanol",500.0,300.0,0.3)  # corpus spelling
    # check the entry landed under the flowsheet key, not a phantom
    pkg=fs._property_packages["NRTL"]; muni=pkg.GetType().GetProperty("m_uni").GetValue(pkg)
    ip=muni.GetType().GetProperty("InteractionParameters").GetValue(muni)
    keys=[str(k) for k in fs._sim.SelectedCompounds.Keys]
    landed=[k for k in keys if ip.ContainsKey(k) and any("thanol" in str(i) or "Xylene" in str(i) for i in ip[k].Keys)]
    print(f"  inject('P-Xylene',...) resolved+applied under flowsheet key(s): {landed}")
except Exception as e:
    print(f"  RAISED: {e}")
