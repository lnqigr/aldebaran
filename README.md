# Aldebaran: End-to-End Photometric Discovery, Vetting & Archival Forensic Pipeline for TESS (v8.x)

**Aldebaran** is an automated, high-throughput digital signal processing (DSP) and astrophysical falsification pipeline designed to detect low-SNR transiting exoplanets in archival and newly released sectors of the NASA TESS Extended Mission (2-minute SPOC cadence).

The pipeline is engineered to probe the detection grey zone ($6.0 \le \text{SDE} \le 7.5$) while automatically rejecting instrumental artifacts, background eclipsing binaries (BEBs), grazing stellar binaries (EBLMs), and rotational variables before human inspection.

---

## Pipeline Architecture

```
                    NASA TESS 2-Minute SPOC Cadence
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TIER 1: High-Throughput Screening (aldebaran.py v8.0)     │
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
│             MULTI-NODE SHARD REDUCER (merge_shards.py v8.0)             │
├─────────────────────────────────────────────────────────────────────────┤
│  • Aggregates all tess_results/sector_*/candidates_report.csv shards   │
│  • Deduplicates multi-sector detections prioritizing highest Kovács SDE │
│  • Merges scanned_tics.txt registries across parallel nodes             │
│  • Automatically triggers Pleiades Tier-2 deep vetting on merged catalog│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TIER 2: Deep Physical Vetting (pleiades.py v2.2)          │
├─────────────────────────────────────────────────────────────────────────┤
│  • Multi-Sector Detrending & Stitched Baseline Normalization            │
│  • 300-Ring Concentric Limb-Darkened Mandel-Agol Model Integration      │
│  • Full-Orbit Eccentric Secondary Eclipse Scan                          │
│  • Astrophysical Catalog Cross-Match:                                   │
│     - VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022)         │
│     - Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Solutions (SB1/SB2) │
│  • Gaia DR3 Field Contamination & Nearby Eclipsing Binary (NEB) Scoring │
│  • 4-Panel Publication Dossiers with Transit Depth Zoom & Error Bars:   │
│     - Top-Left:  Phase-folded LC zoomed to [1 - 3.5*d, 1 + 2*d] with   │
│                  analytical 300-ring fit & ±1σ binned error bars        │
│     - Top-Right: Full stitched multi-sector light curve with epochs     │
│     - Bottom-Left: Odd/Even phase overlays & full-orbit secondary scan  │
│     - Bottom-Right: Gaia DR3 120" neighbor map with NEB critical radii  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ scientific_dossier_summary.csv
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              TIER 3: CTOI Packager & Prioritizer (ctoi_packager.py)     │
├─────────────────────────────────────────────────────────────────────────┤
│  • Empirical Mass Forecasting (Chen & Kipping 2017; Wolfgang et al. 2016)│
│  • Radial Velocity Semi-Amplitude K Calculation (ESPRESSO / MAROON-X)   │
│  • Atmospheric Follow-Up Metrics (Kempton et al. 2018 TSM & ESM)        │
│  • Automated NASA ExoFOP CTOI Pipe-Delimited Batch Upload CSV           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│     HISTORICAL ARCHIVAL FORENSICS: Antares Detective (antares.py)       │
├─────────────────────────────────────────────────────────────────────────┤
│  • Multi-Year Archival Drill across SPOC (2-min), QLP (200s), TGLC (1800s)│
│  • High-Precision Coherent Ephemeris Locking Across 7+ Year Baselines   │
│  • Cross-Pipeline Gaia-PSF Deblending Consistency (TGLC vs QLP vs SPOC) │
│  • 6-Panel Diagnostic Dossier with Secular Mission-Year Stability       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features in v8.x

1. **Distributed PBS Job Array Sharding (`qsub -J 1-8`)**:
   Enables parallel execution across 8 independent compute nodes without inter-process file locks. Workers autonomously map `PBS_ARRAY_INDEX` to sequential TESS sectors (`[106, 105, 104, 103, 102, 101, 100, 99]`) and write to isolated subdirectories (`tess_results/sector_<sec>/`).
2. **Dual-Zone Full-Orbit Secondary Eclipse Scan**:
   Differentiates between circular orbits ($|\phi - 0.50| \le 0.04$) evaluated as a single hypothesis test at $\text{SNR} > 3.1\sigma$ and eccentric orbits ($|\phi - 0.50| > 0.04$) evaluated with analytical Šidák-Bonferroni correction ($\alpha = 0.001$, $\text{SNR} > z_{\text{crit}}$). Enforces a 100 ppm floor and $\ge 10\%$ of primary depth to avoid rejecting real planets due to stellar variability.
3. **Multi-Sector Archival Secondary Scan**:
   Filter 8 (`verify_multi_epoch`) now executes a secondary eclipse scan across the combined archival baseline to eliminate faint occulting stellar companions that only become statistically significant when multiple sectors are stacked.
4. **Automated Shard Aggregation (`merge_shards.py`)**:
   Gathers reports and scanned star registries from all array shards, deduplicates duplicate candidates by highest SDE, and hands off the unified catalog directly to Pleiades Tier-2.
5. **Dynamic Transit Zoom & Error Bars (Pleiades v2.2)**:
   Diagnostic dossiers now automatically zoom into the transit depth window ($[1.0 - 3.5 \cdot \text{depth}, 1.0 + 2.0 \cdot \text{depth}]$) and render phase-binned $\pm 1\sigma$ error bars, providing clear visual discrimination of transit ingress, egress, and bottom flat-floor curvature.
6. **Tier-3 CTOI Packaging & Atmospheric Ranking (`ctoi_packager.py`)**:
   Automatically produces NASA ExoFOP CTOI pipe-delimited upload tables and ranks surviving candidates by RV semi-amplitude $K$ and JWST/HST Transmission/Emission Spectroscopy Metrics (TSM, ESM).
7. **Antares Multi-Year Archival Drill (`antares.py`)**:
   Targeted forensic detective for top candidates across all historical TESS sectors (2018–2026), stitching SPOC, QLP, and TGLC PSF-deblended light curves over 7+ year baselines with coherent ephemeris locking.

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