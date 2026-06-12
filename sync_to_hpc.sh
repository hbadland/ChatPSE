#!/bin/bash
HPC="hgb25@login.cx3.hpc.imperial.ac.uk"
DEST="/rds/general/user/hgb25/home/multiAgentFlowsheet"

# ── Agents ────────────────────────────────────────────────────────────────────
scp agents/orchestrator_v2.py agents/executor.py agents/llm.py agents/basis.py agents/rule_store.py "$HPC:$DEST/agents/"
scp agents/stage1/unit_extractor.py agents/stage1/stream_extractor.py "$HPC:$DEST/agents/stage1/"
scp agents/stage2/graph_builder.py "$HPC:$DEST/agents/stage2/"
scp agents/stage3/param_mapper.py agents/stage3/thermo_mapper.py "$HPC:$DEST/agents/stage3/"
scp agents/stage4/error_classifier.py agents/stage4/beam_search.py agents/stage4/repair_agent.py "$HPC:$DEST/agents/stage4/"

# ── DWSIM wrapper ─────────────────────────────────────────────────────────────
scp dwsim/dwsim_wrapper.py dwsim/test_recycle_block.py "$HPC:$DEST/dwsim/"

# ── IR layer ──────────────────────────────────────────────────────────────────
scp ir/graph.py ir/validate.py ir/normalise.py ir/thermo_estimation.py ir/consistency.py ir/to_dwsim.py ir/test_recycle.py "$HPC:$DEST/ir/"

# ── Benchmark (full directory) ───────────────────────────────────────────────
scp benchmark/__init__.py \
    benchmark/ablation.py \
    benchmark/case_schema.py \
    benchmark/comparison.py \
    benchmark/diagnostics.py \
    benchmark/logger.py \
    benchmark/metrics.py \
    benchmark/physics_eval.py \
    benchmark/runner.py \
    benchmark/test_physics_eval.py \
    benchmark/visualisation.py \
    "$HPC:$DEST/benchmark/" && \
scp run_validation_hpc.sh "$HPC:$DEST/"

# ── Benchmark case files ──────────────────────────────────────────────────────
scp benchmark/cases/validation.json \
    benchmark/cases/sanity.json \
    benchmark/cases/easy.json \
    benchmark/cases/medium.json \
    benchmark/cases/hard.json \
    benchmark/cases/val_recycle.json \
    "$HPC:$DEST/benchmark/cases/"

# ── Validation reference flowsheets (ground-truth DWSIM data) ────────────────
scp benchmark/reference_flowsheets/VAL_01_reference.json \
    benchmark/reference_flowsheets/VAL_02_reference.json \
    benchmark/reference_flowsheets/VAL_03_reference.json \
    benchmark/reference_flowsheets/VAL_04_reference.json \
    benchmark/reference_flowsheets/VAL_05_reference.json \
    benchmark/reference_flowsheets/VAL_06_reference.json \
    benchmark/reference_flowsheets/VAL_07_reference.json \
    benchmark/reference_flowsheets/VAL_08_reference.json \
    benchmark/reference_flowsheets/VAL_09_reference.json \
    benchmark/reference_flowsheets/VAL_10_reference.json \
    "$HPC:$DEST/benchmark/reference_flowsheets/"

# ── Top-level scripts ─────────────────────────────────────────────────────────
scp benchmark_runner.py run_validation_hpc.sh "$HPC:$DEST/"

# ── Context / RAG ─────────────────────────────────────────────────────────────
scp context/compound_database.md "$HPC:$DEST/context/"
scp rag/dump_compounds.py rag/enrich_synonyms_pubchem.py "$HPC:$DEST/rag/"
scp rag/sources/dwsim_compounds.txt rag/sources/compound_synonyms.json "$HPC:$DEST/rag/sources/"

# After syncing, run the dump script ONCE to replace the preliminary compound list:
#   singularity exec --bind /rds /path/to/dwsim.sif \
#       python3.9 $DEST/rag/dump_compounds.py
# Then scp the generated dwsim_compounds.txt back:
#   scp "$HPC:$DEST/rag/sources/dwsim_compounds.txt" rag/sources/dwsim_compounds.txt
