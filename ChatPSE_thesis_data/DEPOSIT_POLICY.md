# Deposit Policy

This archive is governed by the following deposit rules, aligned with the Imperial College
London Research Data Management policy and any applicable third-party licence obligations.

---

## What IS deposited here

- Per-run JSON result records (pipeline outputs, not model weights or simulator binaries)
- Benchmark case JSON files (natural-language descriptions authored for this project)
- Reference flowsheet JSON files authored for this project (see third-party exceptions below)
- Pipeline source code snapshot at the reported commit
- Aggregation and analysis scripts
- Figure and table input data
- Container definition (Dockerfile)
- Python dependency list (requirements.txt)
- This manifest and policy document

---

## What is NOT deposited, and how it is cited instead

### Model weights

The `qwen3:30b-a3b` model weights are not deposited. They are served via Ollama and
distributed under the Qwen licence by Alibaba Cloud. Cite as:

> Qwen Team (2025). Qwen3 Technical Report. Model available at:
> https://huggingface.co/Qwen/Qwen3-30B-A3B
>
> Ollama tag: `qwen3:30b-a3b`
> Model digest: [DIGEST — record from `ollama show qwen3:30b-a3b` on CX3 if still available]
> Ollama version: [VERSION — record from `ollama --version` on CX3 if still available]

### DWSIM installer

The DWSIM 9.0.4 `.deb` installer is not deposited. DWSIM is distributed under the
GNU General Public Licence v3 (GPL-3.0). Redistribution of the compiled binary requires
compliance with GPL-3.0, which permits redistribution but requires source availability.
Given the file size and that the official SourceForge release is the canonical distribution
point, cite as:

> DWSIM — Open-Source Chemical Process Simulator, version 9.0.4 (amd64).
> Available at: https://dwsim.inforside.com.br / SourceForge DWSIM project.
> Installed from: dwsim_9.0.4-amd64.deb (SHA256: [record if available])

The container definition (Dockerfile) is deposited; it documents the exact installation
procedure. Anyone rebuilding the container must obtain the `.deb` from the official source.

Container image SHA256 (Singularity .sif): [record from `sha256sum dwsim.sif` on CX3]

### Third-party source flowsheets

FOS_01 is adapted from the FOSSEE process-simulation repository
(https://fossee.in/simulations/chemical). Before depositing `FOS_01_reference.json`, confirm
that the FOSSEE materials are depositable under their licence terms. If redistribution is
not permitted, remove `FOS_01_reference.json` from `benchmark_definitions/references/`
and replace with a citation:

> FOS_01 case adapted from: FOSSEE Chemical Process Simulation repository,
> Indian Institute of Technology Bombay. https://fossee.in/simulations/chemical

### HPC logs

Console logs (`*.log`, `*.out`, `*.err`) from the CX3 PBS jobs are not deposited in their
raw form. They contain:
- Internal HPC paths (`/rds/general/user/hgb25/...`)
- Institutional email addresses (`hgb25@ic.ac.uk`)
- Cluster hostnames and node names

Use `analysis/scripts/redact_log.py` to produce a redacted version before any deposit.
The redacted log is deposited; the raw log is not. See that script for the redaction rules.

### Exploratory and development run records

Per-run JSON records from before 2026-08-10 and from any run not listed in
`thesis_2026_manifest.json` are not deposited. They remain on the HPC filesystem and in
the local `results/per_run/` directory (which is gitignored) for internal reference.

The selection criterion is: a record is deposited if and only if its filename is listed
in `thesis_2026_manifest.json`.

---

## Licence status

**No licence has been assigned to this software or data.** The repository is publicly
visible, but public visibility does not constitute permission to reuse, modify, or
redistribute the code or data.

Before the archive is finalised, obtain confirmation from your supervisor(s) and the
Imperial College London IP and Licensing team on:

1. Whether the software should be released under an open-source licence (e.g. MIT, Apache 2.0)
   and if so, which one.
2. Whether the benchmark definitions and result records should be released under a data
   licence (e.g. CC BY 4.0) and if so, which one.
3. Whether a software licence and a data licence need to be specified separately.
4. Whether any component of the pipeline is subject to IP claims by Imperial College London
   or any collaborating organisation.

Do not add a `LICENSE` file or claim an open-source licence in this archive until the
above has been confirmed in writing.

---

## Container integrity

To verify that a rebuilt container matches the one used for the reported experiments,
compute the SHA256 of the Singularity `.sif` file:

```bash
sha256sum dwsim.sif
```

Record this value in `environment/software_versions.txt` under "Container SHA256".
If the value on CX3 was not recorded at run time, this field is permanently unrecorded.
