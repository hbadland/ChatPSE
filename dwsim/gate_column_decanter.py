"""
Stage C GATE — run a benzene/toluene shortcut column and a water/benzene decanter
through the COMMITTED IR->DWSIM mapping (ir.to_dwsim -> agents.executor.Executor),
NOT the standalone probe. Confirms the committed path reproduces the probe result
(≈98% purity products; clean LLE split). Runs inside the DWSIM container.
"""
from dwsim.dwsim_wrapper import DWSIMFlowsheet   # loads DWSIM assemblies
from ir.graph import FlowsheetGraph, ColumnNode, DecanterNode, EdgeIR
from ir.to_dwsim import to_dwsim
from agents.executor import Executor


def _streams(res):
    sr = getattr(res, "stream_results", None) or {}
    out = {}
    for tag, s in sr.items():
        out[str(tag)] = {
            "T_K": getattr(s, "T_K", None),
            "flow": getattr(s, "flow_mol_s", None),
            "comp": dict(getattr(s, "composition", {}) or {}),
        }
    return out


def gate_column():
    print("\n================ COLUMN GATE (committed path) ================")
    g = FlowsheetGraph(); g.compounds = ["Benzene", "Toluene"]
    g.property_package = "Peng-Robinson"
    g.add_unit(ColumnNode(tag="SC-01", params=dict(
        light_key="Benzene", heavy_key="Toluene",
        light_key_frac_bottoms=0.02, heavy_key_frac_distillate=0.02,
        reflux_ratio=1.5, condenser_pressure_Pa=101325.0,
        boiler_pressure_Pa=101325.0)), strict=False)
    g.add_stream(EdgeIR(tag="FEED", T=365.0, P=101325.0, flow=1.0,
                        composition={"Benzene": 0.5, "Toluene": 0.5},
                        metadata={"is_feed": True}), None, "SC-01")
    g.add_stream(EdgeIR(tag="DIST", src_port=0), "SC-01", None)
    g.add_stream(EdgeIR(tag="BOT", src_port=1), "SC-01", None)

    res = Executor().run(to_dwsim(g))
    print("converged:", getattr(res, "converged", "?"),
          " outcome:", getattr(res, "outcome", "?"))
    st = _streams(res)
    for s in ("DIST", "BOT"):
        print(f"  {s}: {st.get(s)}")
    dist = st.get("DIST", {}).get("comp", {})
    bot  = st.get("BOT", {}).get("comp", {})
    ok = dist.get("Benzene", 0) > 0.95 and bot.get("Toluene", 0) > 0.95
    print(f"  [{'PASS' if ok else 'FAIL'}] DIST≈98% benzene & BOT≈98% toluene "
          f"(probe parity)")


def gate_decanter():
    print("\n================ DECANTER GATE (committed path) ================")
    g = FlowsheetGraph(); g.compounds = ["Water", "Benzene"]
    g.property_package = "UNIQUAC"
    g.add_unit(DecanterNode(tag="DEC-01"), strict=False)
    g.add_stream(EdgeIR(tag="FEEDD", T=320.0, P=101325.0, flow=1.0,
                        composition={"Water": 0.5, "Benzene": 0.5},
                        metadata={"is_feed": True}), None, "DEC-01")
    g.add_stream(EdgeIR(tag="VAP", src_port=0), "DEC-01", None)
    g.add_stream(EdgeIR(tag="L1", src_port=1), "DEC-01", None)
    g.add_stream(EdgeIR(tag="L2", src_port=2), "DEC-01", None)

    res = Executor().run(to_dwsim(g))
    print("converged:", getattr(res, "converged", "?"),
          " outcome:", getattr(res, "outcome", "?"))
    st = _streams(res)
    for s in ("L1", "L2"):
        print(f"  {s}: {st.get(s)}")
    l1 = st.get("L1", {}).get("comp", {}); l2 = st.get("L2", {}).get("comp", {})
    # one liquid benzene-rich, the other water-rich
    split = (max(l1.get("Benzene", 0), l2.get("Benzene", 0)) > 0.9 and
             max(l1.get("Water", 0),   l2.get("Water", 0))   > 0.9)
    print(f"  [{'PASS' if split else 'FAIL'}] two liquid phases split "
          f"(benzene-rich + water-rich)")


if __name__ == "__main__":
    import traceback
    for fn in (gate_column, gate_decanter):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} EXCEPTION:", e); traceback.print_exc()
