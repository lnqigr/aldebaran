# MetaCentrum HPC Guide for Aldebaran TESS Pipeline (v8.5)

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

Make the PBS script executable:
```bash
chmod +x run_aldebaran.pbs
```

---

## 2. Submitting Jobs via PBS Pro

### Standard Production Job (8 Cores, 24 GB RAM, 35 GB NVMe Scratch, 8h Walltime)
To submit a single-sector production run:
```bash
qsub run_aldebaran.pbs
```
PBS will return a unique Job ID, for example: `1234567.pbs-m1.metacentrum.cz`.

By default, the script scans up to 15,000 targets (`ALDEBARAN_MAX_TARGETS=15000`) on 8 CPU cores. This hardware footprint matches NASA MAST download throughput without CPU over-allocation or fairshare penalty.

### Customizing Batch Size via Environment Variables
You can override the target count without editing any files:
```bash
# Scan a smaller test batch (e.g. 500 targets):
qsub -v ALDEBARAN_MAX_TARGETS=500 run_aldebaran.pbs

# Scan a full sector (e.g. 15,000 targets):
qsub -v ALDEBARAN_MAX_TARGETS=15000 run_aldebaran.pbs
```

### High-Throughput Distributed Array (8 Nodes, 8 Sectors, ~100,000 Targets)
To scan up to 8 fresh TESS sectors (Sectors 106, 105, 104, 103, 102, 101, 100, 99) simultaneously in ~2 hours:
```bash
qsub -J 1-8 run_aldebaran.pbs
```
Each node in the array:
* Gets 8 dedicated CPU cores, 16 GB RAM, and 35 GB local NVMe scratch.
* Automatically processes its own assigned sector from `PREFERRED_SECTORS` in zero-lock isolated storage (`tess_results/sector_<sec>/`).
* Desynchronizes MAST API calls with an initial jitter delay (`sleep $(( (PBS_ARRAY_INDEX * 3) % 20 ))`).

### Fully Automated "Before Sleep" Workflow (`submit_all.sh`)
To queue the entire end-to-end pipeline (Tier-1 array screening + automatic Tier-2 Pleiades deep vetting) with a single command:
```bash
chmod +x submit_all.sh run_merge.pbs
./submit_all.sh
```

### Attaching the Auto-Reducer to an Already-Running Array
If you already submitted an array (e.g. `24683990`), schedule the reducer to wait for it:
```bash
qsub -W depend=afterany:24683990[].pbs-m1.metacentrum.cz run_merge.pbs
```
The reducer job will sit in `Hold` (`H`), and the instant the final array worker finishes, PBS will launch `run_merge.pbs`, deduplicating candidates and generating all Pleiades diagnostic dossiers.

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
* **Cancel / kill a running job gracefully:**
  ```bash
  qdel <JOB_ID>
  ```
  > **Safe Sync Protection:** Even if a job is cancelled with `qdel` or hits the 6-hour walltime limit, `run_aldebaran.pbs` traps the `SIGTERM` signal, lets `aldebaran.py` finalize its Run Summary, syncs all reports and dossiers back to `/storage/.../tess_results/`, and only then cleans local scratch.

---

## 4. Storage Architecture & Quotas

MetaCentrum enforces strict quotas on the shared `/storage/...` network filesystem.
* **Why `$SCRATCHDIR` is critical:** The `run_aldebaran.pbs` script automatically routes all raw FITS downloads from NASA MAST, pip cache, and astropy/matplotlib caches into the compute node's fast local NVMe (`$SCRATCHDIR`).
* **Zero Quota Bloat:** Upon job completion or cancellation, the local scratch disk is automatically wiped (`clean_scratch`), and only the final lightweight results (`candidates_report.csv`, `.png` plots, and `.log` files) are synced back to your persistent folder.
* **Periodic Backup:** In addition to exit/signal traps, the script performs an automatic background sync every 20 minutes to guard against hard node crashes.
