#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HPC="hgb25@login.cx3.hpc.imperial.ac.uk"
DEST="/rds/general/user/hgb25/home/multiAgentFlowsheet"

cd "$REPO_ROOT"

rsync -avz \
    --exclude 'results/' \
    --exclude 'deps/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.sif' \
    --exclude '*.dwxmz' \
    --exclude '*.tar.gz' \
    --exclude '*.tar' \
    --exclude '*.zip' \
    --exclude '.git/' \
    --exclude '.ollama/' \
    --exclude '*.log' \
    --exclude '.DS_Store' \
    ./ "$HPC:$DEST/"

echo "Synced $(git branch --show-current) to HPC."
