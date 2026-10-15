# Aldebaran: End-to-End Photometric Discovery, Vetting & Archival Forensic Pipeline for TESS (v9.0)

**Aldebaran** is an automated, high-throughput digital signal processing (DSP) and astrophysical falsification pipeline designed to detect low-SNR transiting exoplanets in archival and newly released sectors of the NASA TESS Extended Mission (2-minute SPOC cadence).

The pipeline is engineered to probe the detection grey zone ($6.0 \le \text{SDE} \le 7.5$) while automatically rejecting instrumental artifacts, background eclipsing binaries (BEBs), grazing stellar binaries (EBLMs), and rotational variables before human inspection.

---

## Pipeline Architecture

```
                    NASA TESS 2-Minute SPOC Cadence
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TIER 1: High-Throughput Screening (aldebaran.py v9.0)     │
├─────────────────────────────────────────────────────────────────────────┤
│  • Catalog Cross-Match (ExoFOP, TOIs, CTOIs, Confirmed Planets, EBs)    │
│  • Multi-Planet Companion Search: 8,036 TOI Ephemeris Ingestion (TAP)   │
│  • In-Transit Masking (±1.0 T_dur) + Pre-Filter Linear Interpolation    │
│    (Eliminates Gibbs filter edge ringing; harmonic alias protection)    │
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
│     8. Non-Circular Archival Verification (verify_multi_epoch v9.0):     │
│        - Contiguous (ΔT <= 60 d): Strict zero-freedom forward project   │
│        - Multi-Year (ΔT > 60 d): Cramer-Rao physical corridor (±2σ_P),  │
│          Extreme-value Šidák threshold penalty (z_crit = 3.5σ - 4.2σ),  │
│          8-phase off-transit control test (SNR_arch >= max_ctrl + 1.5σ) │
│  • Zero-Lock PBS Array Sharding across Sectors 108, 107, 98-93 (~0 MB)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ candidates_report.csv
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             MULTI-NODE SHARD REDUCER (merge_shards.py v9.0)             │
├─────────────────────────────────────────────────────────────────────────┤
│  • Aggregates all tess_results/sector_*/candidates_report.csv shards   │
│  • Deduplicates multi-sector detections prioritizing highest Kovács SDE │
│  • Merges scanned_tics.txt registries across parallel nodes             │
│  • Automatically triggers Pleiades Tier-2 deep vetting on merged catalog│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               TIER 2: Deep Physical Vetting (pleiades.py v2.6)          │
├─────────────────────────────────────────────────────────────────────────┤
│  • Multi-Sector Detrending & Stitched Baseline Normalization            │
│  • 300-Ring 5-Point Gauss-Legendre Mandel-Agol Model Integration (3 ppm)│
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
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ prioritized_targets_ranking.csv
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         TIER 4: Bayesian Validation & Holdout Engine (v9.0)             │
├─────────────────────────────────────────────────────────────────────────┤
│  • TRICERATOPS Bayesian Blend Validation (validate_triceratops.py)       │
│     - 18 astrophysical blend scenarios (Giacalone et al. 2021)          │
│     - Formal FPP (False Positive Prob) & NFPP (Nearby FPP) calculation  │
│     - Memory-optimized 60-bin phase folding (prevents 11 GB OOM)        │
│  • Leave-One-Year-Out (LOYO) Blind Holdout Validator (loyo_validator.py)│
│     - Multi-year blind timing prediction across 800+ day gaps           │
│     - 94-transit O - C timing analysis & Keplerian clock verification   │
│  • Master Scanned TICs Catalog Reducer (merge_scanned_tics.py)          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features in v9.0

1. **Multi-Planet Search Engine via Known-Transit Masking (`aldebaran.py`)**:
   Automatically queries and caches 8,036 official NASA TOI planet ephemerides from TAP (`tess_results/toi_ephemerides_cache.json`). When `SEARCH_TOI_COMPANIONS=True`, TOI host stars are prioritized in the discovery queue. Known planet transits are masked ($\pm 1.0\,T_{\text{dur}}$, absorbing up to $\pm 45\text{ min}$ TTVs) with **mandatory linear interpolation across masked intervals before Savitzky-Golay filtering**, strictly eliminating Gibbs boundary ringing artifacts. BLS search on residual flux detects shallow sub-Earth sibling companions with fractional harmonic and alias protection ($m/n \in \{1, 1/2, 2, 1/3, 3, 2/3, 3/2, 1/4, 4\}$).
2. **Cramer-Rao Bounded Ephemeris Projection & Šidák Penalty (`verify_multi_epoch`)**:
   Eliminates archival confirmation circularity:
   * **Contiguous sectors ($\Delta T \le 60\text{ d}$)**: Strict zero-freedom forward projection ($\Delta P = 0, \Delta T_0 = 0, N_{\text{trials}} = 1, z_{\text{crit}} = 2.4\sigma$) without BLS re-fitting.
   * **Multi-year sectors ($\Delta T > 60\text{ d}$)**: Constrains period search to the physical $2\sigma_P$ Cramer-Rao corridor $[P_{\text{disc}} - 2\sigma_P, P_{\text{disc}} + 2\sigma_P]$ (width $\approx \pm 0.10\%$, $6\times$ narrower than legacy $\pm 0.6\%$).
   * **Trial penalty**: Dynamically scales critical SNR threshold via extreme-value penalty $z_{\text{crit}} = \max(2.8, \sqrt{2\ln(N_{\text{trials}})} + 1.2) \in [3.5\sigma, 4.2\sigma]$.
   * **Off-transit null control**: 8-phase off-transit test requires $\text{SNR}_{\text{arch}} \ge \max(\text{SNR}_{\text{ctrl}}) + 1.5\sigma$, reducing random false-pass probability on multi-year baselines to **$0.0144\%$** ($1$ in $6,940$).
3. **Empirical False Alarm Rate (FAR) Calibration (`tests/benchmark_false_alarm_rate.py`)**:
   Negative-flux search benchmark ($F_{\text{inv}} = 2.0 - F$) across 2,000 quiet TESS stars demonstrates that $\text{FAR}(\text{SDE} \ge 6.0) = 0.55\%$ ($< 1\%$), proving $>99.45\%$ specificity against pure stellar and instrumental noise.
4. **Exact 5-Point Gauss-Legendre 300-Ring Integrator & 180-Transit Benchmark**:
   Replaces crude annulus integration with a 5-point Gauss-Legendre quadrature (maximum error $\le 3.09\text{ ppm}$ vs analytical Mandel & Agol elliptic integrals). Comprehensive 180-transit injection benchmark across 36 parameter cells demonstrates $100\%$ recovery for sub-Neptunes ($R_p \ge 2.3 R_\oplus$), $100\%$ for super-Earths ($1.8 R_\oplus, P \le 9.5\text{ d}$), and $100\%$ for ultra-short-period Earth-sized planets ($1.0 R_\oplus, P = 1.0\text{ d}$).
5. **Empirical ROC Calibration for Vetting Penalty Score (VPS)**:
   Youden's J statistic calibration across 100 benchmark targets establishes optimal threshold $\text{VPS} \le 23.4$ ($\text{TPR} = 94.0\%, \text{FPR} = 2.0\%, \text{AUC} = 0.988$).
6. **Production Configuration for MetaCentrum HPC (Sectors 108, 107, 98–93)**:
   Pre-configured lock-free sharding across the newest available TESS data (Sectors 108 and 107) and deep archival fields (Sectors 98–93).

---

## Key Features in v8.6

1. **Tier-4 Bayesian Statistical Validation (`validate_triceratops.py`)**:
   Evaluates 18 astrophysical blend scenarios (Giacalone et al. 2021) using TRICERATOPS. Computes formal False Positive Probability (FPP) and Nearby False Positive Probability (NFPP) against background stars, aperture blends, and unresolved companions. Features adaptive 60-bin phase folding across the transit window, cutting multi-sector array memory footprint from 11.55 GB to ~60 MB (<0.6%) to prevent OOM errors while boosting binned SNR per cadence by $\sqrt{N_{\text{transits}}}$.
2. **Leave-One-Year-Out (LOYO) Blind Holdout Engine (`loyo_validator.py`)**:
   Implements multi-year temporal cross-validation inspired by independent researcher Pavel Rabtsevich (TIC 4206066). Blindly predicts transit arrival times across multi-year data gaps (e.g. $+1.94\text{ min}$ timing precision over 800 days on TIC 350190949, $4.0\%$ of transit duration) and conducts 94-transit $\text{O}-\text{C}$ Keplerian clock stability analysis ($dP/dt \approx 0$).
3. **Master Scanned TICs Consolidation (`merge_scanned_tics.py`)**:
   Aggregates, deduplicates, and numerically sorts historical scanned TICs across MetaCentrum cluster arrays, local archives, and current workspace into a unified registry (expanding local history from 411 to 6,337+ unique TICs).

---

## Key Features in v8.5

1. **Multi-Sector Ephemeris Synchronization (Pleiades v2.5)**:
   Eliminated parameter desynchronization by propagating `refined_t0_btjd`, `fitted_depth_ppm`, and `fitted_duration_hours` directly from multi-sector Mandel-Agol fits into downstream packagers, preventing single-sector BLS parameter contamination in ExoFOP CTOI exports.
2. **$\delta$ Scuti Asteroseismic Pre-Whitening (TIC 301152169)**:
   Subtracts verified orthogonal pulsation modes ($f_1=37.93\text{ c/d}, f_2=41.28, f_3=28.70, f_4=44.72, f_5=37.40$), reducing residual RMS noise by 26.5% ($1554\text{ ppm} \to 1143\text{ ppm}$) and boosting transit SNR from $20.9\sigma$ to $28.6\sigma$ ($+7.8\sigma, 1.37\times$).
3. **Decoupled Archival Drill & Earliest-Transit Anchoring (`antares_packager.py`)**:
   Anchors $T_0$ to the earliest observed transit epoch ($t_{0,\text{anchor}}$), preventing negative epoch projection drift across multi-year baselines (e.g. eliminating 45-min phase errors on TIC 350190949 and lifting SPOC stack SNR from $2.1\sigma$ to $4.35\sigma$). Isolates SPOC 2-min cadence from QLP/TGLC side-lobe interference.
4. **Distributed PBS Job Array Sharding (`qsub -J 1-8`)**:
   Enables parallel execution across 8 independent compute nodes without inter-process file locks. Workers autonomously map `PBS_ARRAY_INDEX` to sequential TESS sectors (`[108, 107, 98, 97, 96, 95, 94, 93]`) and write to isolated subdirectories (`tess_results/sector_<sec>/`).
5. **Dual-Zone Full-Orbit Secondary Eclipse Scan**:
   Differentiates between circular orbits ($|\phi - 0.50| \le 0.04$, $\text{SNR} > 3.1\sigma$) and eccentric orbits ($|\phi - 0.50| > 0.04$, Šidák-Bonferroni $\text{SNR} > z_{\text{crit}}$). Enforces a 100 ppm floor and $\ge 10\%$ primary depth.
6. **Modular Directory Architecture**:
   Structured results into dedicated functional subfolders (`tess_results/deep_vetting/`, `tess_results/archival_vetting/`, `tess_results/archive/`) with consolidated summary catalogs in root for clean automated reporting.

---

## Test Suite (`tests/`)

Aldebaran includes an automated validation and regression suite covering all pipeline tiers:
* `tests/test_v9_ephemeris_and_multiplanet.py`: Tests TOI catalog ingestion, known-transit masking, sibling planet recovery, contiguous strict projection, and multi-year Cramer-Rao trial scaling.
* `tests/test_mandel_agol_batman.py`: Validates 300-ring Gauss-Legendre integration accuracy ($\le 3.09\text{ ppm}$) against analytical elliptic integral limit.
* `tests/benchmark_false_alarm_rate.py`: Measures empirical false alarm rate on inverted light curves ($N=2000$ stars).
* `tests/benchmark_injection.py`: End-to-end 180-transit injection recovery completeness benchmark across 36 parameter cells.
* `tests/benchmark_vps_roc.py`: Computes ROC curve and Youden's J statistic for Vetting Penalty Score (VPS) on control sample.
* `tests/test_ssl_check.py`: Tests connectivity to ExoFOP, MAST, NASA Exoplanet Archive, and Gaia TAP across certificate stores.
* `tests/test_ring_accuracy.py`: Benchmarks 300-ring Mandel-Agol numerical integration accuracy against a 20,000-ring continuous reference.
* `tests/test_binary_catalogs.py`: Verifies VizieR TESS EB and Gaia DR3 NSS catalog cross-matching against benchmark targets.
* `tests/test_kovacs_sde.py`: Validates Kovács whitened SDE noise suppression on pure white noise and planet recovery.
* `tests/test_secondary_scan.py`: Validates dual-zone full-orbit secondary eclipse detection and Šidák-Bonferroni false-alarm thresholds.
* `tests/test_tess_eb_fetch.py`: Validates direct TAPVizieR and Astroquery ingestion of the Prša et al. (2022) catalog.

Run all unit tests:
```bash
python -m unittest discover tests
```

---

## HPC & Cluster Execution (MetaCentrum)

For large-scale runs across tens of thousands of stars, PBS Pro scripts are available:
* `cluster/run_aldebaran.pbs`: PBS script optimized for MetaCentrum HPC with Job Arrays (`qsub -J 1-8`) and local NVMe `$SCRATCHDIR` I/O (8 CPUs, 24 GB RAM, 35 GB local scratch, 8h walltime).
* `cluster/run_merge.pbs`: Dependent reducer PBS job that executes automatically when the array finishes.
* `cluster/submit_all.sh`: One-command automated launcher submitting both the 8-node array and the dependent reducer job.
* `merge_shards.py`: Reducer script to merge shard outputs after array execution.
* See [README_METACENTRUM.md](docs/README_METACENTRUM.md) for full deployment and monitoring instructions.

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
python antares_packager.py

# 5. Run Antares multi-year archival forensic drill on a specific candidate
python antares.py --tic 46314221

# 6. Run Tier-4 Bayesian TRICERATOPS statistical blend validation
python validate_triceratops.py --tic 322051865 --period 2.831532 --t0 4154.32930 --duration 1.08 --depth 778.4 --sectors 103 104 105 106 --mc 5 --bins 60

# 7. Run Leave-One-Year-Out (LOYO) 7.8-year blind holdout clock test
python loyo_validator.py --tic 350190949 --period 0.926120 --t0 1325.61416 --duration 0.81 --sectors 1 28 104 105

# 8. Merge and synchronize master scanned TICs catalog
python merge_scanned_tics.py
```

### HPC End-to-End Automated Array Execution (MetaCentrum)
```bash
# One-command fully automated workflow (Tier-1 array + automatic Tier-2 deep vetting reducer):
chmod +x cluster/submit_all.sh cluster/run_merge.pbs cluster/run_aldebaran.pbs
./cluster/submit_all.sh
```