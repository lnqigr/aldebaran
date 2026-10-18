# MetaCentrum HPC Guide for Aldebaran TESS Pipeline (v9.1.0)

This guide explains how to deploy and execute the **Aldebaran** exoplanet search pipeline on the Czech National Grid Organization (**MetaCentrum** / CESNET e-INFRA CZ) using PBS Pro.

---

## 1. Quick Connection & Setup

### Step 1: Connect to a MetaCentrum Frontend
Open PowerShell or your terminal and SSH into any available MetaCentrum frontend:
```bash
ssh your_username@tarkil.metacentrum.cz
# Alternatively: perun.metacentrum.cz, skirit.metacentrum.cz, minos.metacentrum.cz
```

### Step 2: Clone or Update the Project
On the frontend node, navigate to your persistent storage directory and clone or update your repository:
```bash
cd /storage/praha1/home/$USER   # or /storage/brno2/home/$USER
git clone https://github.com/lnqigr/aldebaran.git exoplanet_search
cd exoplanet_search
```

If you already have the repository cloned:
```bash
cd exoplanet_search
git pull origin main
```

Make the cluster scripts executable:
```bash
chmod +x cluster/*.pbs cluster/*.sh
```

---

## 2. Submitting Jobs via PBS Pro

### Standard Production Job (8 Cores, 24 GB RAM, 35 GB NVMe Scratch, 8h Walltime)
To submit a single-sector production run:
```bash
qsub cluster/run_aldebaran.pbs
```
PBS will return a unique Job ID, for example: `1234567.pbs-m1.metacentrum.cz`.

By default, the script scans up to 15,000 targets (`ALDEBARAN_MAX_TARGETS=15000`) on 8 CPU cores. This hardware footprint matches NASA MAST download throughput without CPU over-allocation or fairshare penalty.

### Customizing Batch Size via Environment Variables
You can override the target count without editing any files:
```bash
# Scan a smaller test batch (e.g. 500 targets):
qsub -v ALDEBARAN_MAX_TARGETS=500 cluster/run_aldebaran.pbs

# Scan a full sector (e.g. 15,000 targets):
qsub -v ALDEBARAN_MAX_TARGETS=15000 cluster/run_aldebaran.pbs
```

### High-Throughput Distributed Array (8 Nodes, 8 Sectors, ~100,000 Targets)
Array indices 1–8 map to sectors 108, 107, 98, 97, 96, 95, 94, 93, respectively, in the current `PREFERRED_SECTORS` list. Runtime depends on data availability, target count and node performance:
```bash
qsub -J 1-8 cluster/run_aldebaran.pbs
```
Each node in the array:
* Gets 8 dedicated CPU cores, 24 GB RAM, and 35 GB local NVMe scratch.
* Automatically processes its own assigned sector from `PREFERRED_SECTORS` in zero-lock isolated storage (`tess_results/sector_<sec>/`).
* Desynchronizes MAST API calls with an initial jitter delay (`sleep $(( (PBS_ARRAY_INDEX * 3) % 20 ))`).

### Scan submission and explicit reduction

**Deployment readiness:** Confirm the selected source revision, inputs, output paths and resource limits before real submission; this guide does not establish scientific readiness for a survey run.

Use `bash cluster/submit_all.sh --scan-only --array 1-8` (or a single index or
ascending range within 1-8). No arguments, automatic merge, and unknown options
are refused before submission. `afterany` does not prove input completeness.
Check shard completion, real persistent paths, run identity and hashes before
preparing the explicit manifest required by `merge_shards.py`.

Submit reduction separately from the repository root, replacing all placeholders:

```bash
qsub -v MANIFEST_PATH=/storage/USER/run/manifest.json,RUN_ID=verified-run-id,OUTPUT_DIR=/storage/USER/run/fresh-merge cluster/run_merge.pbs
```

Use a fresh output directory. Partial reduction requires explicit `ALLOW_PARTIAL=1`
and remains incomplete. `RUN_PLEIADES`, `RUN_ANTARES`, `RUN_TRICERATOPS` default
to 0. Inspect the manifest schema and `python merge_shards.py --help`; the launcher
never constructs a manifest.

`constraints-windows-py314.txt` captures installed Windows AMD64 Python 3.14.3
versions (TRICERATOPS 1.1.0). Its header lists missing transitive metadata; this
is an incomplete installed-environment constraint set, not a validated lock.
`constraints-linux-py311.txt` preserves the saved Linux Python 3.11.11 freeze
(TRICERATOPS 1.0.21), with source path and SHA256. Versions differ by environment;
neither dependency pins nor these contract tests establish scientific validity.
The pinned Linux engineering verification passed on Python 3.11.11: MetaCentrum job 24793285, 339 regression tests and 50 PBS contract tests, with no failures or skips, plus two synthetic standalone checks. Dependency checks and imports passed; the network guard recorded 19 expected blocked probes and zero unexpected attempts. This does not establish scientific validation or production-survey readiness.

Installation profile used by the isolated Linux verifier: select Python 3.11.11,
create/activate a fresh venv, and use the bootstrap and constraints below. Stop on
any install or dependency-check failure; do not run tests or jobs until resolved:

```bash
python -m venv .venv
# Activate .venv for your shell first.
python -m pip install pip==26.2.1 setuptools==80.10.2 wheel==0.45.1 || exit 1
python -m pip install --no-build-isolation -c constraints-linux-py311.txt -r requirements.txt || exit 1
python -m pip check || exit 1
ALDEBARAN_OFFLINE=1 python -m unittest tests.test_known_signals tests.test_submit_all tests.test_ephemeris_and_provenance tests.test_toi_companions_auditable
```

The Windows snapshot is for observed component testing only and is not a
supported reproducible full runtime. A fresh environment and successful
`python -m pip check` are mandatory before runtime use. PowerShell must stop on
nonzero `$LASTEXITCODE`; set `$env:ALDEBARAN_OFFLINE='1'` for offline tests. Required known-signal tests generate **SYNTHETIC** fixtures for
API/cache contracts. Optional observational replay is separately invoked with
`python -m unittest tests.historical_known_signals` and requires unchanged
`24702696_20261016_025524/audit/catalog_audit` files and original provenance.
Synthetic success is not equivalent to historical recovery.

PBS pip honors `PIP_CONSTRAINT`. Pass an absolute persistent path with `qsub -v`;
ensure that path is visible on compute nodes:

```bash
qsub -v PIP_CONSTRAINT="$PWD/constraints-linux-py311.txt" -J 1-8 cluster/run_aldebaran.pbs
```

For separate reduction include `PIP_CONSTRAINT=/storage/USER/repo/constraints-linux-py311.txt`
in its `qsub -v` list. The isolated engineering verification did not run scientific scans or validate these production submission examples against real candidates.

---


## 3. Monitoring & Managing Jobs

* **View running jobs / array summary:**
  ```bash
  qstat -u $USER
  ```
* **View all individual tasks in a running array:**
  ```bash
  qstat -t <JOB_ID>[]
  ```
* **View detailed job telemetry (node name, memory, and CPU usage):**
  ```bash
  qstat -f <JOB_ID>
  ```
* **Follow live log output of a running job:**
  ```bash
  # Single job:
  tail -f aldebaran_tess.o<JOB_ID>

  # Array task (e.g. worker index 1):
  tail -f aldebaran_tess.o<JOB_ID>-1
  ```
* **Request cancellation of a running job:**
  ```bash
  qdel <JOB_ID>
  ```
  > Cancellation or walltime termination can interrupt analysis and copying. Inspect the job exit status, sync/cleanup logs and persistent artifacts; a signal trap cannot guarantee complete output after forced termination or node failure.

---

## 4. Storage Architecture & Quotas

MetaCentrum enforces strict quotas on the shared `/storage/...` network filesystem.
* **Why `$SCRATCHDIR` is critical:** The `run_aldebaran.pbs` script automatically routes all raw FITS downloads from NASA MAST, pip cache, and astropy/matplotlib caches into the compute node's fast local NVMe (`$SCRATCHDIR`).
* **Scratch cleanup:** Cleanup is conditional on successful persistence checks. Failure paths can retain scratch for diagnosis; neither cancellation nor completion alone proves that all output reached persistent storage.
* **Periodic sync:** Background copies reduce potential loss but do not certify a complete run or protect unsynced data from a node crash. Verify the final artifacts and completion evidence.
