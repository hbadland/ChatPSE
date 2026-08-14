#!/bin/bash
#PBS -N comp_val
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -o /rds/general/user/hgb25/home/multiAgentFlowsheet/results/component_val.out
#PBS -e /rds/general/user/hgb25/home/multiAgentFlowsheet/results/component_val.err
#PBS -m abe
#PBS -M hgb25@ic.ac.uk

set -euo pipefail

DEST="/rds/general/user/hgb25/home/multiAgentFlowsheet"
SINGULARITY_IMG="/rds/general/user/hgb25/home/dwsim.sif"
OLLAMA_HOST="http://localhost:11434/v1"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$DEST/results/component_ablation_val_${RUN_STAMP}.log"

cd "$DEST"
mkdir -p results/per_run

{
    echo "[$(date)] Component ablation starting"
    echo "commit=$(git rev-parse HEAD)"
    echo "cases=VAL_06,VAL_05"
    echo "modes=full_ccs,no_physics,no_coupling"
    echo "model=qwen3:30b-a3b"
    echo "pipeline=GraphPipeline"
    echo "BEST_OF_N=5"
    echo "configured_max_iter=15"
    echo "rule_store=cleared before each case-mode run"
} | tee "$LOG"

echo "[$(date)] Waiting 30s for Ollama" | tee -a "$LOG"
sleep 30
if ! curl -fsS http://localhost:11434/api/tags >/dev/null; then
    echo "[$(date)] ERROR: Ollama not responding" | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Running instrumentation preflight" | tee -a "$LOG"
singularity exec --bind /rds --bind "$DEST" "$SINGULARITY_IMG" \
    bash -lc "cd '$DEST' && PYTHONPATH=. python3.9 verify_ablation_counters.py" \
    2>&1 | tee -a "$LOG"

for CASE_ID in VAL_06 VAL_05; do
    for MODE in full_ccs no_physics no_coupling; do
        echo "[$(date)] START case=$CASE_ID mode=$MODE" | tee -a "$LOG"

        # Prevent corrections learned by an earlier arm from leaking into a
        # later arm.  Each Python invocation also resets process-local caches.
        rm -f "$DEST/results/rule_store.json"

        singularity exec --bind /rds --bind "$DEST" "$SINGULARITY_IMG" \
            bash -lc "
                cd '$DEST'
                PYTHONPATH=. \
                USE_LANGGRAPH=1 \
                OLLAMA_BASE_URL='$OLLAMA_HOST' \
                PYTHONNET_RUNTIME=coreclr \
                BEST_OF_N=5 \
                python3.9 benchmark_runner.py \
                    --case '$CASE_ID' \
                    --mode '$MODE' \
                    --model 'qwen3:30b-a3b' \
                    --max-iter 15
            " 2>&1 | tee -a "$LOG"

        echo "[$(date)] END case=$CASE_ID mode=$MODE" | tee -a "$LOG"
    done
done

echo "[$(date)] Component ablation complete" | tee -a "$LOG"
