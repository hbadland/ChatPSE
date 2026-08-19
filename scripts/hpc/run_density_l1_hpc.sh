#!/bin/bash
# Imperial CX3 job for the L1 specification-density study.
#PBS -N density_l1
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o /rds/general/user/hgb25/home/multiAgentFlowsheet/results/density_l1.out
#PBS -e /rds/general/user/hgb25/home/multiAgentFlowsheet/results/density_l1.err
#PBS -m abe
#PBS -M hgb25@ic.ac.uk

DEST="/rds/general/user/hgb25/home/multiAgentFlowsheet"
SINGULARITY_IMG="/rds/general/user/hgb25/home/dwsim.sif"
OLLAMA_HOST="http://localhost:11434/v1"

cd "$DEST"
mkdir -p results/per_run

echo "[$(date)] Density-study L1 starting — commit=$(git rev-parse HEAD | head -c 12)"
echo "[$(date)] Cases: VAL_01_L1, VAL_03_L1, VAL_04_L1 — model: qwen3:30b-a3b — BEST_OF_N=5 — max-iter=15"

echo "[$(date)] Waiting 30s for Ollama..."
sleep 30
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "[$(date)] ERROR: Ollama not responding — aborting"
    exit 1
fi
echo "[$(date)] Ollama ready."

singularity exec \
    --bind /rds \
    --bind "$DEST" \
    "$SINGULARITY_IMG" \
    bash -c "
        cd $DEST
        PYTHONPATH=. \
        OLLAMA_BASE_URL=$OLLAMA_HOST \
        PYTHONNET_RUNTIME=coreclr \
        BEST_OF_N=5 \
        python3.9 benchmark_runner.py \
            --case-files benchmark/cases/val_l1.json \
            --mode full_ccs \
            --model qwen3:30b-a3b \
            --max-iter 15 \
            2>&1 | tee results/density_l1_run.log
    "

echo "[$(date)] Done. Results in $DEST/results/per_run/"
