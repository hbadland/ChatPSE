"""Loads context markdown files once at import time."""
from pathlib import Path

_DIR = Path(__file__).parent

FAILURE_TAXONOMY   = (_DIR / "failure_taxonomy.md").read_text()
DWSIM_KNOWLEDGE    = (_DIR / "dwsim_knowledge.md").read_text()
COMPOUND_DATABASE  = (_DIR / "compound_database.md").read_text()
