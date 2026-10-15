# Aldebaran: End-to-End Photometric Discovery, Vetting & Archival Forensic Pipeline for TESS (v8.5)

**Aldebaran** is an automated, high-throughput digital signal processing (DSP) and astrophysical falsification pipeline designed to detect low-SNR transiting exoplanets in archival and newly released sectors of the NASA TESS Extended Mission (2-minute SPOC cadence).

The pipeline is engineered to probe the detection grey zone ($6.0 \le \text{SDE} \le 7.5$) while automatically rejecting instrumental artifacts, background eclipsing binaries (BEBs), grazing stellar binaries (EBLMs), and rotational variables before human inspection.

---

## Pipeline Architecture

```
                    NASA TESS 2-Minute SPOC Cadence
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TIER 1: High-Throughput Screening (aldebaran.py v8.5)     │
├─────────────────────────────────────────────────────────────────────────┤
│  • Catalog Cross-Match (ExoFOP, TOIs, CTOIs, Confirmed Planets)         │
│  • Bitwise Quality Filtering (QUALITY == 0) & Gap Ramp Trimming         │
│  • Piecewise Iterative Savitzky-Golay Filtering with Transit Masking    │
│  • Whitened Signal Residue SDE (Kovács et al. 2002, 501-pt Median/MAD)  │
│  • Multi-Peak BLS Period Search (0.045 d <= Dur <= 0.18 d, Top 3 Peaks)│
│  • 9-Stage Astrophysical Falsification Cascade:                         │
│     0. Aperture Dilution: SPOC CROWDSAP >= 0.70                         │
│     1. Detection & Duty Cycle: SDE >= 6.0, T_dur / P <= 0.095           │
│     2. Transit Profile: Integrated SNR >= 5.5σ, Shoulder <= 35%,        │
│        U-shape check (rejects sharp V-shaped grazing binaries)          │
│     3. Multiplicity & Consistency: N_transits >= 3, Odd/Even < 2.8σ,   │
│        Epoch Consistency (>= 60% with negative flux deficit)            │
│     4. Dual-Zone Secondary Eclipse Scan:                                │
│        - Circular (|Δφ| <= 0.04): 1 hypothesis, SNR > 3.1σ              │
│        - Eccentric (|Δφ| > 0.04): Šidák-Bonferroni SNR > z_crit         │
│        - Depth floor: >= 100 ppm & >= 10% primary depth                 │
│     5. Astrometric Centroid Stability: Welch's t-test (< 2.0 arcsec)    │
│     6. Stellar Density & Keplerian Bounds: TIC v8.2 / Gaia DR3 RUWE     │
│        (RUWE <= 1.4, R_* <= 1.75 R_sun, 0.35*T_min <= Dur <= 1.38*T_max)│
│     7. Target Pixel Stamp (TPF): Difference image aperture localization │
│     8. Multi-Epoch Archival Check: Multi-year baseline persistence      │
│        (> 2.4σ) & multi-sector secondary eclipse verification           │
│  • Zero-Lock PBS Array Sharding & Scratch Invalidation (~0 MB Cache)   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ candidates_report.csv
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             MULTI-NODE SHARD REDUCER (merge_shards.py v8.5)             │
├─────────────────────────────────────────────────────────────────────────┤
│  • Aggregates all tess_results/sector_*/candidates_report.csv shards   │
│  • Deduplicates multi-sector detections prioritizing highest Kovács SDE │
│  • Merges scanned_tics.txt registries across parallel nodes             │
│  • Automatically triggers Pleiades Tier-2 deep vetting on merged catalog│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TIER 2: Deep Physical Vetting (pleiades.py v2.5)          │
├─────────────────────────────────────────────────────────────────────────┤
│  • Multi-Sector Detrending & Stitched Baseline Normalization            │
│  • 300-Ring Concentric Limb-Darkened Mandel-Agol Model Integration      │
│  • Export of Multi-Sector Ephemeris: refined_t0, fitted_depth, dur_hr   │
│  • Full-Orbit Eccentric Secondary Eclipse Scan with Šidák Threshold     │
│  • Astrophysical Catalog Cross-Match:                                   │
│     - VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022)         │
│     - Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Solutions (SB1/SB2) │
│  • Gaia DR3 Field Contamination & Nearby Eclipsing Binary (NEB) Scoring │
│  • 4-Panel Publication Dossiers in tess_results/deep_vetting/           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ scientific_dossier_summary.csv
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│       TIER 3: Archival Drill & CTOI Packager (antares_packager.py)      │
├─────────────────────────────────────────────────────────────────────────┤
│  • Stage 3A: Corrected FP Risk & Pre-Filtering Gate                     │
│  • Stage 3B: Decoupled Archival Drill (SPOC vs FFI Isolation)           │
│     - Earliest-Transit Anchoring (t0_anchor) avoids epoch drift         │
│     - Clean SPOC-only coherent folding and SNR_red metrics              │
│  • Stage 3C: Planetary Follow-Up Prioritization (K m/s, TSM, ESM)       │
│  • Stage 3D: ExoFOP CTOI Export with Multi-Sector Parameter Sync        │
│  • 4-Panel Archival Posters saved in tess_results/archival_vetting/     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features in v8.5

1. **Multi-Sector Ephemeris Synchronization (Pleiades v2.5)**:
   Eliminated parameter desynchronization by propagating `refined_t0_btjd`, `fitted_depth_ppm`, and `fitted_duration_hours` directly from multi-sector Mandel-Agol fits into downstream packagers, preventing single-sector BLS parameter contamination in ExoFOP CTOI exports.
2. **$\delta$ Scuti Asteroseismic Pre-Whitening (TIC 301152169)**:
   Subtracts verified orthogonal pulsation modes ($f_1=37.93\text{ c/d}, f_2=41.28, f_3=28.70, f_4=44.72, f_5=37.40$), reducing residual RMS noise by 26.5% ($1554\text{ ppm} \to 1143\text{ ppm}$) and boosting transit SNR from $20.9\sigma$ to $28.6\sigma$ ($+7.8\sigma, 1.37\times$).
3. **Decoupled Archival Drill & Earliest-Transit Anchoring (`antares_packager.py`)**:
   Anchors $T_0$ to the earliest observed transit epoch ($t_{0,\text{anchor}}$), preventing negative epoch projection drift across multi-year baselines (e.g. eliminating 45-min phase errors on TIC 350190949 and lifting SPOC stack SNR from $2.1\sigma$ to $4.35\sigma$). Isolates SPOC 2-min cadence from QLP/TGLC side-lobe interference.
4. **Distributed PBS Job Array Sharding (`qsub -J 1-8`)**:
   Enables parallel execution across 8 independent compute nodes without inter-process file locks. Workers autonomously map `PBS_ARRAY_INDEX` to sequential TESS sectors (`[106, 105, 104, 103, 102, 101, 100, 99]`) and write to isolated subdirectories (`tess_results/sector_<sec>/`).
5. **Dual-Zone Full-Orbit Secondary Eclipse Scan**:
   Differentiates between circular orbits ($|\phi - 0.50| \le 0.04$, $\text{SNR} > 3.1\sigma$) and eccentric orbits ($|\phi - 0.50| > 0.04$, Šidák-Bonferroni $\text{SNR} > z_{\text{crit}}$). Enforces a 100 ppm floor and $\ge 10\%$ primary depth.
6. **Modular Directory Architecture**:
   Structured results into dedicated functional subfolders (`tess_results/deep_vetting/`, `tess_results/archival_vetting/`, `tess_results/archive/`) with consolidated summary catalogs in root for clean automated reporting.

---

## Test Suite (`tests/`)

Aldebaran includes an automated validation and regression suite:
* `tests/test_ssl_check.py`: Tests connectivity to ExoFOP, MAST, NASA Exoplanet Archive, and Gaia TAP across certificate stores.
* `tests/test_ring_accuracy.py`: Benchmarks 300-ring Mandel-Agol numerical integration accuracy against a 20,000-ring continuous reference.
* `tests/test_binary_catalogs.py`: Verifies VizieR TESS EB and Gaia DR3 NSS catalog cross-matching against benchmark targets.
* `tests/test_kovacs_sde.py`: Validates Kovács whitened SDE noise suppression on pure white noise and planet recovery.
* `tests/test_secondary_scan.py`: Validates dual-zone full-orbit secondary eclipse detection and Šidák-Bonferroni false-alarm thresholds.
* `tests/test_tess_eb_fetch.py`: Validates direct TAPVizieR and Astroquery ingestion of the Prša et al. (2022) catalog.

Run all tests:
```bash
python -c "import subprocess, sys; [subprocess.run([sys.executable, t], check=True) for t in ['tests/test_ssl_check.py', 'tests/test_ring_accuracy.py', 'tests/test_binary_catalogs.py', 'tests/test_kovacs_sde.py', 'tests/test_secondary_scan.py', 'tests/test_tess_eb_fetch.py']]"
```

---

## HPC & Cluster Execution (MetaCentrum)

For large-scale runs across tens of thousands of stars, PBS Pro scripts are available:
* `run_aldebaran.pbs`: PBS script optimized for MetaCentrum HPC with Job Arrays (`qsub -J 1-8`) and local NVMe `$SCRATCHDIR` I/O (8 CPUs, 16 GB RAM, 35 GB local scratch, 6h walltime).
* `run_merge.pbs`: Dependent reducer PBS job that executes automatically when the array finishes.
* `submit_all.sh`: One-command automated launcher submitting both the 8-node array and the dependent reducer job.
* `merge_shards.py`: Reducer script to merge shard outputs after array execution.
* See [README_METACENTRUM.md](README_METACENTRUM.md) for full deployment and monitoring instructions.

---

## Quickstart

### Local Execution (Single Machine)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Tier-1 screening
python aldebaran.py

# 3. Run Tier-2 deep physical vetting on detected candidates
python pleiades.py

# 4. Run Tier-3 CTOI packaging, RV K calculation & atmospheric ranking
python ctoi_packager.py

# 5. Run Antares multi-year archival forensic drill on a specific candidate
python antares.py --tic 46314221
```

### HPC End-to-End Automated Array Execution (MetaCentrum)
```bash
# One-command fully automated workflow (Tier-1 array + automatic Tier-2 deep vetting reducer):
chmod +x submit_all.sh run_merge.pbs run_aldebaran.pbs
./submit_all.sh
```