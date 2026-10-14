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

### Standard Single Job (16 Cores, 32 GB RAM, 40 GB NVMe Scratch)
To submit a single run (scans 2500 targets by default):
```bash
qsub run_aldebaran.pbs
```
PBS will return a unique Job ID, for example: `1234567.meta-pbs.metacentrum.cz`.

---

## 3. High-Throughput Parallel Scanning (Job Arrays)

To process **10,000 to 50,000 stars** simultaneously across independent physical compute nodes:
```bash
# Launches 5 parallel nodes concurrently:
qsub -J 1-5 run_aldebaran.pbs
```
Each node gets its own dedicated 16 CPU cores, 32 GB RAM, and local high-speed NVMe scratch storage. Each worker node safely appends scanned TICs to `tess_results/scanned_tics.txt` and records candidates to `tess_results/candidates_report.csv`.

---

## 4. Monitoring & Managing Jobs

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
* **Cancel / kill a running job:**
  ```bash
  qdel <JOB_ID>
  ```

---

## 5. Storage Architecture & Quotas

MetaCentrum enforces strict quotas on the shared `/storage/...` network filesystem.
* **Why `$SCRATCHDIR` is critical:** The `run_aldebaran.pbs` script automatically routes all raw FITS downloads from NASA MAST into the compute node's fast local NVMe (`$SCRATCHDIR`).
* **Zero Quota Bloat:** Upon job completion or cancellation, the local scratch disk is automatically wiped (`clean_scratch`), and only the final lightweight results (`candidates_report.csv`, `.png` plots, and `.log` files) are synced back to your persistent folder.
