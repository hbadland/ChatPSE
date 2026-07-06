"""
Partition experiment — VAL_01 (clean, IR-representable, reference-EXACT topology).

Runs the FULL LangGraph pipeline with LLM extraction REPLACED by reference
injection (Variant B): reference units + connectivity + setpoints are injected,
then thermo -> DWSIM -> repair run normally.  Measures whether the case converges
end-to-end once extraction is removed — the "extraction removed" partition point.

Ground truth for VAL_01 (verified against the solved stream table):
  * 8 real units — the 8 'HT-0x' Heaters in the reference are phantom energy/duty
    artifacts (all 10 streams are consumed by the 8-unit chain), so they are dropped.
  * Straight-through chain, NO recycle: flow is constant 0.017315 mol/s across
    S-01..S-08 (nothing added upstream), and SEP-01 splits 0.010499 vapour (S-09)
    + 0.006816 liquid (S-10) = 0.017315.  (The NL description claims a recycle; the
    real flowsheet is once-through — a description-vs-flowsheet mismatch.)

Usage (HPC):
  PYTHONPATH=. OLLAMA_BASE_URL=http://localhost:11434/v1 \
      python3.9 partition_val01.py [model]
  (VARIANT_B is armed inside this script; DWSIM executor env must be available.)
"""
import os, sys, json, tempfile

# Arm reference-topology injection (read by graph_pipeline._variant_b_enabled()).
os.environ["VARIANT_B"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.case_schema import load_tier
from agents.graph_pipeline import GraphPipeline

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:30b-a3b"
REF   = "benchmark/reference_flowsheets/VAL_01_reference.json"

# The 8 real units (drop phantom HT-01..HT-08).
REAL_UNITS = {"K-01", "K-02", "K-03", "CO-01", "CO-02", "CO-03", "EX-01", "V-01"}

# Paired DWSIM connections [a, b, src_port, dst_port]:
#   [unit, stream]  -> that unit is the stream's SOURCE
#   [stream, unit]  -> that unit is the stream's DEST
# S-01 has no unit source -> feed; S-09/S-10 have no unit dest -> terminal products.
# S-09 listed before S-10 so the Vessel's first outlet (port 0) = vapour, second
# (port 1) = liquid, matching vapor_fraction in the stream table.
CONNECTIONS = [
    ["S-01", "K-01", 0, 0],                            # FEED -> K-01
    ["K-01", "S-02", 0, 0], ["S-02", "CO-01", 0, 0],
    ["CO-01", "S-03", 0, 0], ["S-03", "K-02", 0, 0],
    ["K-02", "S-04", 0, 0], ["S-04", "CO-02", 0, 0],
    ["CO-02", "S-05", 0, 0], ["S-05", "K-03", 0, 0],
    ["K-03", "S-06", 0, 0], ["S-06", "CO-03", 0, 0],
    ["CO-03", "S-07", 0, 0], ["S-07", "EX-01", 0, 0],
    ["EX-01", "S-08", 0, 0], ["S-08", "V-01", 0, 0],
    ["V-01", "S-09", 0, 0],                            # vapour outlet (terminal)
    ["V-01", "S-10", 1, 0],                            # liquid outlet (terminal)
]

ref = json.load(open(REF))
ref["units"]       = [u for u in ref["units"] if u["tag"] in REAL_UNITS]
ref["connections"] = CONNECTIONS
ref.pop("recycles", None)   # straight-through; no recycle annotations

tmp = tempfile.NamedTemporaryFile("w", suffix="_VAL_01_exact.json", delete=False)
json.dump(ref, tmp)
tmp.close()
print(f"[PARTITION] reference-exact VAL_01: {len(ref['units'])} units, "
      f"{len(CONNECTIONS)} connection entries -> {tmp.name}", flush=True)

case = next(c for c in load_tier("val_6_9") if c.id == "VAL_01")
desc = getattr(case, "prompt", None) or case.description

gp     = GraphPipeline(model=MODEL, max_iterations=10)
result = gp.run(desc, reference_file=tmp.name, tier="validation")

ladder = getattr(result, "variant_b_diag", None)
ex     = result.final_execution
print("\n" + "=" * 72)
print("PARTITION RESULT — VAL_01 (reference-exact topology, extraction removed)")
print("=" * 72)
print("outcome          :", result.outcome)
print("reached DWSIM    :", ex is not None)
print("DWSIM converged  :", bool(getattr(ex, "solved", False)))
print("total time (s)   :", round(getattr(result, "total_time_s", 0.0), 1))
print("variant_b ladder :",
      json.dumps(ladder, indent=2, default=str) if ladder else "(none)")
print("\nVALIDATION: grep the log for")
print("  [VARIANT_B] reference topology: ... topology_source=reference-exact")
print("If it shows 'reference-inferred-connections', the connections were NOT used")
print("and the result is invalid for the partition.")
