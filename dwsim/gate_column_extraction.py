"""
Stage D GATE — a SIMPLE single-column case from the SEMANTIC extraction onward,
through the COMMITTED pipeline (GraphBuilder -> normalise -> ParamMapper ->
to_dwsim -> Executor). Bottoms is emitted BEFORE distillate to prove the port
repair. Confirms distillate→port0/bottoms→port1 and ~98% purity end-to-end.

The LLM unit extraction itself (does the model emit a Column for a distillation
description) needs an HPC run; here we feed the semantic units a correct
extraction would produce and verify everything downstream. Runs in the container.
"""
from dwsim.dwsim_wrapper import DWSIMFlowsheet          # loads DWSIM
from agents.stage1.unit_extractor import SemanticUnit, SemanticUnits
from agents.stage1.stream_extractor import SemanticStream, SemanticTopology
from agents.stage2.graph_builder import GraphBuilder
from ir.normalise import normalise
from agents.stage3.param_mapper import ParamMapper
from ir.to_dwsim import to_dwsim
from agents.executor import Executor


def main():
    units = SemanticUnits(units=[SemanticUnit(
        tag="SC-01", type="Column", role="distillation column", reaction="")])
    # bottoms BEFORE distillate → order-based port assignment would be wrong
    streams = SemanticTopology(streams=[
        SemanticStream(tag="FEED", src=None, dst="SC-01", is_feed=True,
                       T=365.0, P=101325.0, flow=1.0,
                       composition={"Benzene": 0.5, "Toluene": 0.5}),
        SemanticStream(tag="BOTTOMS", src="SC-01", dst=None, is_feed=False),
        SemanticStream(tag="DISTILLATE", src="SC-01", dst=None, is_feed=False),
    ])
    g = GraphBuilder().build(units, streams, ["Benzene", "Toluene"])
    g = normalise(g)
    g.property_package = "Peng-Robinson"
    g = ParamMapper().assign(g, description="separate benzene and toluene by distillation")

    res = Executor().run(to_dwsim(g))
    sr = getattr(res, "stream_results", {}) or {}
    def comp(tag):
        s = sr.get(tag)
        return dict(getattr(s, "composition", {}) or {}) if s else {}
    d, b = comp("DISTILLATE"), comp("BOTTOMS")
    print("DISTILLATE:", d)
    print("BOTTOMS   :", b)
    ok = d.get("Benzene", 0) > 0.95 and b.get("Toluene", 0) > 0.95
    print(f"[{'PASS' if ok else 'FAIL'}] single-column extraction→wiring→solve: "
          f"DIST≈98% benzene, BOT≈98% toluene")


if __name__ == "__main__":
    main()
