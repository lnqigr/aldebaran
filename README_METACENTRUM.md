# MetaCentrum HPC Guide for Aldebaran TESS Pipeline

This guide explains how to deploy and execute the **Aldebaran** exoplanet search pipeline on the Czech National Grid Organization (**MetaCentrum** / CESNET) using PBS Pro.

---

## 1. Quick Connection & Setup

### Step 1: Connect to a MetaCentrum Frontend
Open PowerShell or your terminal and SSH into any available MetaCentrum frontend:
```bash
ssh your_username@tarkil.metacentrum.cz
# Alternatively: perun.metacentrum.cz, skirit.metacentrum.cz, minos.metacentrum.cz
```

### Step 2: Clone or Update the Project
On the frontend node, navigate to your persistent storage directory and clone your repository:
```bash
cd /storage/praha1/home/$USER   # or /storage/brno2/home/$USER
git clone https://github.com/lnqigr/aldebaran.git exoplanet_search
cd exoplanet_search
```

Make the PBS script executable:
```bash
chmod +x run_aldebaran.pbs
```

---

## 2. Submitting Jobs via PBS Pro

### Standard Production Job (16 Cores, 32 GB RAM, 40 GB NVMe Scratch, 24h Walltime)
To submit a production run:
```bash
qsub run_aldebaran.pbs
```
PBS will return a unique Job ID, for example: `1234567.meta-pbs.metacentrum.cz`.

By default, the script scans up to 12,000 targets (`ALDEBARAN_MAX_TARGETS=12000`) on 16 CPU cores.

### Customizing Batch Size via Environment Variables
You can override the target count without editing any files:
```bash
# Scan a full sector (e.g. 15,000 targets):
qsub -v ALDEBARAN_MAX_TARGETS=15000 run_aldebaran.pbs
```

### High-Throughput Distributed Array (8 Nodes, 8 Sectors, ~100,000 Targets)
To scan up to 8 fresh TESS sectors (Sectors 99 through 106) simultaneously in ~2 hours:
```bash
qsub -J 1-8 run_aldebaran.pbs
```
Each node in the array:
* Gets 8 dedicated CPU cores, 16 GB RAM, and 35 GB local NVMe scratch.
* Automatically processes its own assigned sector in zero-lock isolated storage (`tess_results/sector_<sec>/`).
* After the array finishes, run the aggregator to merge all sectors into a unified candidate catalog and trigger Tier-2 Pleiades deep vetting:
```bash
python merge_shards.py
```

---

## 3. Monitoring & Managing Jobs

* **View running jobs:**
  ```bash
  qstat -u $USER
  ```
* **View detailed job telemetry (node name, memory and CPU usage):**
  ```bash
  qstat -f <JOB_ID>
  ```
* **Follow live log output of a running job:**
  ```bash
  tail -f aldebaran_tess.o<JOB_ID>
  ```
* **Cancel / kill a running job gracefully:**
  ```bash
  qdel <JOB_ID>
  ```
  > **Safe Sync Protection:** Even if a job is cancelled with `qdel` or hits the 24-hour walltime limit, `run_aldebaran.pbs` traps the `SIGTERM` signal, lets `aldebaran.py` finalize its Run Summary, syncs all reports and dossiers back to `/storage/.../tess_results/`, and only then cleans local scratch.

---

## 4. Storage Architecture & Quotas

MetaCentrum enforces strict quotas on the shared `/storage/...` network filesystem.
* **Why `$SCRATCHDIR` is critical:** The `run_aldebaran.pbs` script automatically routes all raw FITS downloads from NASA MAST, pip cache, and astropy/matplotlib caches into the compute node's fast local NVMe (`$SCRATCHDIR`).
* **Zero Quota Bloat:** Upon job completion or cancellation, the local scratch disk is automatically wiped (`clean_scratch`), and only the final lightweight results (`candidates_report.csv`, `.png` plots, and `.log` files) are synced back to your persistent folder.
* **Periodic Backup:** In addition to exit/signal traps, the script performs an automatic background sync every 20 minutes to guard against hard node crashes.
