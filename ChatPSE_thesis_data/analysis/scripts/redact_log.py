"""
redact_log.py

Step 3 of the data-transfer workflow.
Strips sensitive institutional identifiers from HPC console logs before deposit.

Redacted items:
  - Usernames (hgb25 and variants)
  - Institutional email addresses
  - Internal HPC filesystem paths (/rds/general/user/...)
  - Cluster node names (e.g. cx3-... hostnames)
  - PBS job IDs (numeric)

Preserves:
  - All pipeline output (stage transitions, repair logs, metrics, errors)
  - Timestamps
  - Case IDs and ablation mode identifiers
  - DWSIM output and stream conditions
  - Commit hashes

Usage:
    python3.9 redact_log.py \\
        --input  /path/to/raw.log \\
        --output /path/to/redacted.log

Review the redacted output manually before depositing.
"""

import argparse
import re
import sys
from pathlib import Path

# ── Redaction rules: (pattern, replacement) ────────────────────────────────────
RULES = [
    # Institutional email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@ic\.ac\.uk\b"),          "[EMAIL_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@imperial\.ac\.uk\b"),     "[EMAIL_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@\S+\.\S+\b"),             "[EMAIL_REDACTED]"),

    # HPC filesystem paths — replace with portable placeholder
    (re.compile(r"/rds/general/user/[A-Za-z0-9_]+/home/multiAgentFlowsheet"),
                                                                  "[PROJECT_PATH]"),
    (re.compile(r"/rds/general/user/[A-Za-z0-9_]+/home"),        "[HPC_HOME]"),
    (re.compile(r"/rds/general/user/[A-Za-z0-9_]+"),             "[HPC_USER_DIR]"),
    (re.compile(r"/rds/[A-Za-z0-9/_\-]+"),                       "[HPC_PATH]"),

    # Usernames (standalone word, case-insensitive)
    (re.compile(r"\bhgb25\b", re.IGNORECASE),                    "[USERNAME]"),

    # PBS job IDs (lines like "Job ID: 12345678.cx3")
    (re.compile(r"\b\d{6,9}\.cx3\b"),                            "[JOB_ID]"),

    # Cluster node hostnames (e.g. cx3-1-23.cx3.hpc.ic.ac.uk)
    (re.compile(r"\bcx3-[\w\-]+\.cx3\.hpc\.ic\.ac\.uk\b"),       "[NODE_HOSTNAME]"),
    (re.compile(r"\bcx3-[\w\-]+\b"),                              "[NODE_ID]"),

    # Generic internal hostnames ending in .ic.ac.uk or .imperial.ac.uk
    (re.compile(r"\b[\w\-]+\.ic\.ac\.uk\b"),                     "[INTERNAL_HOST]"),
    (re.compile(r"\b[\w\-]+\.imperial\.ac\.uk\b"),               "[INTERNAL_HOST]"),

    # PBS/scheduler-injected header lines (often contain paths and usernames)
    # Match common PBS header patterns
    (re.compile(r"^PBS:.*$", re.MULTILINE),                      "[PBS_HEADER_REDACTED]"),
    (re.compile(r"^#PBS.*$", re.MULTILINE),                      "[PBS_DIRECTIVE_REDACTED]"),
]


def redact(text: str) -> str:
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Raw log file from HPC")
    ap.add_argument("--output", required=True, help="Redacted output path")
    ap.add_argument("--verify", action="store_true",
                    help="After redaction, scan for possible remaining sensitive strings and warn")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        print(f"ERROR: input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    raw = inp.read_text(encoding="utf-8", errors="replace")
    redacted = redact(raw)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(redacted, encoding="utf-8")
    print(f"Redacted log written to: {out}")

    if args.verify:
        warnings = []
        # Check for remaining @ signs (possible email addresses)
        for i, line in enumerate(redacted.splitlines(), 1):
            if "@" in line and "[EMAIL_REDACTED]" not in line:
                warnings.append(f"  line {i}: possible email — {line[:120]}")
            if "/rds/" in line:
                warnings.append(f"  line {i}: possible HPC path — {line[:120]}")
            if "hgb25" in line.lower():
                warnings.append(f"  line {i}: possible username — {line[:120]}")
        if warnings:
            print(f"\nWARNING: {len(warnings)} line(s) may contain residual sensitive content:")
            for w in warnings[:20]:
                print(w)
            if len(warnings) > 20:
                print(f"  ... and {len(warnings) - 20} more. Review manually.")
        else:
            print("Verification: no residual sensitive strings detected.")

    print("\nReview the redacted file manually before depositing.")


if __name__ == "__main__":
    main()
