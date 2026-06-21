#!/bin/bash
set -euo pipefail
HPC="hgb25@login.cx3.hpc.imperial.ac.uk"
DEST="/rds/general/user/hgb25/home/multiAgentFlowsheet"

rsync -avz --delete \
    --exclude 'results/' \
    --exclude 'deps/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.sif' \
    --exclude '*.dwxmz' \
    --exclude '.git/' \
    --exclude '.ollama/' \
    --exclude '*.log' \
    ./ "$HPC:$DEST/"

echo "Synced $(git branch --show-current) to HPC."
