# Code Snapshot

This directory will contain a snapshot of the ChatPSE source code at the reported results commit.

**Canonical version:** the tagged GitHub release at `[THESIS RELEASE TAG]`
(`https://github.com/hbadland/multiAgentFlowsheet`)

**Commit:** `[FULL COMMIT HASH]`

## To populate this directory

Copy the repository source (excluding `results_hpc/`, `ChatPSE_thesis_data/`, `.git/`, and other
data/generated directories) here:

```bash
rsync -av \
    --exclude='.git' \
    --exclude='results_hpc/' \
    --exclude='results/' \
    --exclude='ChatPSE_thesis_data/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    /path/to/multiAgentFlowsheet/ \
    ChatPSE_thesis_data/code/
```

The snapshot here is provided for archival self-containment. It is identical to the tagged
GitHub release; the GitHub URL is the primary citable reference.
