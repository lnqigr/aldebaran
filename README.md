# Aldebaran: TESS Transit-Candidate Search and Vetting Pipeline (v9.2)

**Aldebaran** is an automated, high-throughput digital signal processing (DSP) pipeline designed to search archival and newly released sectors of the NASA TESS Extended Mission (2-minute SPOC cadence) for low-SNR transit candidates and route them through follow-up checks.

**Aldebaran's output is a preliminary transit candidate list that needs external secondary-eclipse or RV vetting.** In the v9.2 release measurement, 25 of 150 known eclipsing binaries and false positives (16.7 %) still ended as candidates (one-sided 95 % upper bound 22.5 %).

The search threshold is SDE $\ge 6.0$, so peaks in the detection grey zone ($6.0 \le \text{SDE} \le 7.5$) pass the search stage. The v9.2 measurement tested whether this zone adds useful sensitivity, and **the grey-zone claim is withdrawn**:
- For injected SNR 5 to 9, completeness rose by 5.9 percentage points (34 of 580 injections) from SDE $\ge 7.5$ to SDE $\ge 6.0$, above the required 5.0.
- The false-alarm condition failed. On 500 sectors of quiet stars whose light curves were scrambled in time (the first orbit reversed, the second kept forward, segments swapped and archival sectors reversed and shifted, so no real periodic signal survives but the star's noise does), 10 ended as final candidates at SDE $\ge 6.0$ (2.0 %). The one-sided 95 % upper bound is 3.4 %, above the required 1.5 %.

A candidate in the grey zone therefore needs independent follow-up before it is treated as a signal. Completeness by injected SNR is given below. The pipeline applies heuristic filters intended to flag instrumental artifacts, background eclipsing binaries (BEBs), grazing stellar binaries (EBLMs), and rotational variables for human inspection.

## Scientific status and limits

Aldebaran is a research candidate-search pipeline. Version 9.2 was measured once on 1,126 TESS hosts that were never used in its development. The protocol was preregistered and every number was recomputed independently (M-RUN-RC2-01, November 2026; details in [docs/MEASUREMENT_V9_2.md](docs/MEASUREMENT_V9_2.md)).

The measurement covers Tier 1 (`aldebaran.py`, discovery search and its vetting cascade) with the default settings. Tier 2 to 4 (Pleiades, Antares, TRICERATOPS, LOYO) were not measured, and their attrition, including the Pleiades `is_clean` and VPS cuts, is unknown. Provided the later tiers only remove Tier 1 candidates, the Tier 1 numbers overstate end-to-end recovery of planets and overstate end-to-end false rates. The declared domain:
- hosts with CROWDSAP $\ge$ 0.70, RUWE $\le$ 1.4, $R_\star \le 1.75\,R_\odot$, $T_\text{eff} \le$ 9,500 K;
- transits 180 to 30,000 ppm deep (no completeness claim below 180 ppm), $R_p \le 15\,R_\oplus$; the detector rejects $R_p < 0.8\,R_\oplus$ by design, and its depth gate accepts core depths up to 35,000 ppm;
- injected planets on circular orbits, impact parameter 0 to 0.8, periods 1 to 13 days, $a/R_\star \ge 2.6$, at least 3 transits in the discovery sector;
- one discovery sector per star, with any archival TESS sectors used for multi-epoch verification.

| Measured on fresh hosts | Result |
|---|---|
| Injected planets ending as a matched candidate, by injected SNR (about 290 injections per bin) | 4 to 6: 4.3 %; 6 to 8: 21.5 %; 8 to 10: 46.4 %; 10 to 15: 65.5 %; 15 to 25: 78.8 %; 25 to 35: 87.5 % |
| The same at SNR $\ge$ 10 | 77.1 % (675 of 875) |
| Known archival planets recovered in a single sector (in domain) | 18.1 % (36 of 199); 49.3 % (34 of 69) at catalogue SNR $\ge$ 10; 3 % (2 of 67) for transits longer than 0.18 d |
| Known eclipsing binaries and TESS false positives ending as candidates | 16.7 % (25 of 150, one-sided 95 % upper bound 22.5 %): Prša et al. EBs 6 of 100, TFOPWG false positives 19 of 50 |
| Time-scrambled quiet stars ending with a final candidate (500 star-sectors) | 2.0 % at SDE $\ge$ 6.0 (one-sided 95 % upper bound 3.4 %); the same 500 stars unscrambled: 0.6 % (upper bound 1.54 %) |
| Period error of recovered injections (886) | median 0.00001 d; 68 % between −0.0013 and +0.0009 d |

All 10 alarms on the scrambled stars, and the 3 on the unscrambled ones, had SDE below 7.5. This was observed after the fact and is not a tested threshold. No false-alarm rate is claimed for the production threshold. Purity, the fraction of real planets in a candidate list, depends on how common planets are in the searched population and is not stated here.

The same injected light curves and quiet stars were also run through SHERLOCK 1.6.1, compared at the search stage:
- at injected SNR 5 to 10, SHERLOCK recovered 13.0 percentage points more injections (95 % host-cluster interval 9.5 to 16.8);
- at SNR 10 and above, the two pipelines did not differ beyond the interval (Aldebaran −1.6 points, −3.8 to +0.6);
- on the time-scrambled quiet stars, SHERLOCK raised a signal on 39 % of them and Aldebaran's search on 11 %.

The other benchmarks below include synthetic experiments; their results must not be read as real-survey performance.

Known limitations include the BLS duration grid, which stops at 0.18 d so that longer transits are rarely recovered (see above), and small-sample odd/even statistics: a candidate whose odd and even transits cannot both be replicated is labelled `ODD_EVEN_NOT_TESTABLE` rather than tested. Tier 2 decides eclipsing binaries with the Cp secondary scan; its older raw scan, which takes only the single global maximum by orbital phase, is kept as a diagnostic and decides nothing. Numerical model accuracy requires a stated independent reference, geometry and exposure regime. VPS is an uncalibrated heuristic penalty score (0.1–99.9 points), not a calibrated probability; `TIER-2 HEURISTIC SCREEN PASS`, `TIER-3 VETTED CANDIDATE`, and historical `HIGH-CONFIDENCE` labels are heuristic screen tags, not independent validation. Pleiades `is_clean` requires unrounded clipped VPS < 5.0 joined by `AND` with `b < 0.82`, `sec_snr <= sec_snr_threshold`, `not is_sec_eb`, `not is_tess_eb`, `not is_gaia_nss`, `0.35 <= dur_ratio <= 1.38`, and `stitched_primary_snr >= 5.5`; Antares Stage 3A filters survivors with validated VPS <= 25.0 (default). The legacy `fp_risk_percent` field is a compatibility alias for VPS points (0–100), not a probability. `multi_sde` and `est_multi_sde_proxy` are heuristic proxies (`sde * sqrt(1 + 0.5 * n_archival_sectors)` on archival pass, or discovery `sde` otherwise), not a recomputed multi-sector BLS SDE. The persisted `Confirmed_in_Sectors_*` archival status is a legacy signal-match sentinel, not an independent confirmation; human-facing output calls it an archival match. TRICERATOPS FPP/NFPP are conditional on their inputs and scenario model; passing a model threshold is not independent validation. LOYO timing outputs and candidate labels alone do not establish independent confirmation. Historical candidate examples below are not validated discoveries; reported performance figures are not independent calibration.

Version 9.2 is the measured code: it is identical to release candidate RC2 (commit e474c65c2c), and the numbers above apply to that code with its default switches. Historical feature descriptions below are not calibrated scientific performance claims.

---

## Pipeline Architecture

```
                       NASA TESS 2-Minute SPOC Cadence                       
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│           TIER 1: High-Throughput Screening (aldebaran.py v9.2)           │
├───────────────────────────────────────────────────────────────────────────┤
│  • Catalog Cross-Match (ExoFOP, TOIs, CTOIs, Confirmed Planets, EBs)      │
│  • Multi-Planet Companion Search: 8,036 TOI Ephemeris Ingestion (TAP)     │
│  • In-Transit Masking (±1.0 T_dur) + Pre-Filter Linear Interpolation      │
│    (Intended to reduce filter ringing; harmonic veto is heuristic)        │
│  • Bitwise Quality Filtering (QUALITY == 0) & Gap Ramp Trimming           │
│  • Piecewise Iterative Savitzky-Golay Filtering with Transit Masking      │
│  • Whitened Signal Residue SDE (Kovács et al. 2002, 501-pt Median/MAD)    │
│  • Multi-Peak BLS Period Search (0.045 d <= Dur <= 0.18 d, Top 3 Peaks)   │
│  • 9-Stage Heuristic Candidate-Vetting Cascade:                           │
│     0. Aperture Dilution: SPOC CROWDSAP >= 0.70                           │
│     1. Detection & Duty Cycle: SDE >= 6.0, T_dur / P <= 0.095             │
│     2. Transit Profile: Integrated SNR >= 5.5σ, Shoulder <= 35%,          │
│        U-shape check (V-shaped profiles flagged as a warning only)        │
│     3. Multiplicity & Consistency: N_transits >= 3, Odd/Even v2 < 2.8σ    │
│        (unreplicated parity: ODD_EVEN_NOT_TESTABLE),                      │
│        Epoch Consistency (>= 60% with negative flux deficit)              │
│     4. Dual-Zone Secondary Eclipse Scan:                                  │
│        - Circular (|Δφ| <= 0.04): Šidák SNR > z_crit_circ                 │
│        - Eccentric (|Δφ| > 0.04): Šidák-Bonferroni SNR > z_crit           │
│        - Depth floor: >= 100 ppm & >= 10% primary depth                   │
│     5. Astrometric Centroid Stability: Welch's t-test (< 2.0 arcsec)      │
│     6. Stellar Density & Keplerian Bounds: TIC v8.2 / Gaia DR3 RUWE       │
│        (RUWE <= 1.4, R_* <= 1.75 R_sun, 0.35*T_min <= Dur <= 1.38*T_max)  │
│     7. Target Pixel Stamp (TPF): Difference image aperture localization   │
│     8. Non-Circular Archival Verification (verify_multi_epoch, Cp):       │
│        - Contiguous (ΔT <= 60 d): Strict zero-freedom forward project     │
│        - Multi-Year (ΔT > 60 d): Cramer-Rao physical corridor (±2σ_P),    │
│          Extreme-value Šidák threshold penalty (z_crit = 3.5σ - 4.2σ),    │
│          8-phase off-transit control test (SNR_arch >= max_ctrl + 1.5σ)   │
│  • PBS Array Sharding across Sectors 108, 107, 98-93                      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ candidates_report.csv
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              MULTI-NODE SHARD REDUCER (merge_shards.py v9.0)              │
├───────────────────────────────────────────────────────────────────────────┤
│  • Loads explicit manifest-declared, integrity-checked shard inputs       │
│  • Preserves signal identities and provenance across sector inputs        │
│  • Merges scanned_tics.txt registries across parallel nodes               │
│  • Optional downstream vetting requires explicit separate opt-in          │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              TIER 2: Deep Physical Vetting (pleiades.py v2.6)             │
├───────────────────────────────────────────────────────────────────────────┤
│  • Multi-Sector Detrending & Stitched Baseline Normalization              │
│  • 300-Ring Gauss-Legendre Mandel-Agol Model (accuracy unvalidated)       │
│  • Export of Multi-Sector Ephemeris: refined_t0, fitted_depth, dur_hr     │
│  • Full-Orbit Eccentric Secondary Eclipse Scan with Šidák Threshold       │
│  • Astrophysical Catalog Cross-Match:                                     │
│     - VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022)           │
│     - Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Solutions (SB1/SB2)   │
│  • Gaia DR3 Field Contamination & Nearby Eclipsing Binary (NEB) Scoring   │
│  • 4-Panel Publication Dossiers in tess_results/deep_vetting/             │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ scientific_dossier_summary.csv
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│        TIER 3: Archival Drill & CTOI Packager (antares_packager.py)       │
├───────────────────────────────────────────────────────────────────────────┤
│  • Stage 3A: Heuristic VPS Score & Pre-Filtering Gate                     │
│  • Stage 3B: Decoupled Archival Drill (SPOC vs FFI Isolation)             │
│     - Earliest-Transit Anchoring (t0_anchor) avoids epoch drift           │
│     - Clean SPOC-only coherent folding and SNR_red metrics                │
│  • Stage 3C: Planetary Follow-Up Prioritization (K m/s, TSM, ESM)         │
│  • Stage 3D: ExoFOP CTOI Export with Multi-Sector Parameter Sync          │
│  • 4-Panel Archival Posters saved in tess_results/archival_vetting/       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ prioritized_targets_ranking.csv
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│               TIER 4: Blend-Scenario and Timing Tools (v9.0)              │
├───────────────────────────────────────────────────────────────────────────┤
│  • TRICERATOPS Blend-Scenario Analysis (validate_triceratops.py)          │
│     - 18 astrophysical blend scenarios (Giacalone et al. 2021)            │
│     - Input- and scenario-dependent FPP / NFPP calculations               │
│     - Memory-optimized 60-bin phase folding (prevents 11 GB OOM)          │
│  • Leave-One-Year-Out (LOYO) Timing Analysis (loyo_validator.py)          │
│     - Multi-year blind timing prediction across 800+ day gaps             │
│     - 94-transit O - C timing analysis & Keplerian clock verification     │
│  • Master Scanned TICs Catalog Reducer (merge_scanned_tics.py)            │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features in v9.0

1. **Multi-Planet Search Engine via Known-Transit Masking (`aldebaran.py`)**:
   Automatically queries and caches 8,036 official NASA TOI planet ephemerides from TAP (`tess_results/toi_ephemerides_cache.json`). When `SEARCH_TOI_COMPANIONS=True`, TOI host stars are prioritized in the discovery queue. Known planet transits are masked ($\pm 1.0\,T_{\text{dur}}$, a duration-dependent buffer, not a guaranteed TTV bound) with **mandatory linear interpolation across masked intervals before Savitzky-Golay filtering**, intended to reduce boundary ringing; masking losses and residual artifacts require calibration. BLS searches residual flux for possible sibling companions; sensitivity to shallow sub-Earth signals has not been established. The implementation applies heuristic fractional harmonic vetoes ($m/n \in \{1, 1/2, 2, 1/3, 3, 2/3, 3/2, 1/4, 4\}$). The separate `search_toi_companions.py` CLI defaults to sectors 106, 105, 104, 103, 102, 101, 100, and 99; these defaults do not change the Tier-1 PBS array mapping documented below.
2. **Cramer-Rao Bounded Ephemeris Projection & Šidák Penalty (`verify_multi_epoch`)**:
   Implements archival search/projection restrictions; these do not by themselves establish independent confirmation or calibrated uncertainty:
   * **Contiguous sectors ($\Delta T \le 60\text{ d}$)**: Strict zero-freedom forward projection ($\Delta P = 0, \Delta T_0 = 0, N_{\text{trials}} = 1, z_{\text{crit}} = 2.4\sigma$) without BLS re-fitting.
   * **Multi-year sectors ($\Delta T > 60\text{ d}$)**: Constrains period search to a $2\sigma_P$ Cramer-Rao corridor $[P_{\text{disc}} - 2\sigma_P, P_{\text{disc}} + 2\sigma_P]$ (reported width $\approx \pm 0.10\%$, $6\times$ narrower than the legacy $\pm 0.6\%$ setting); its uncertainty coverage has not been calibrated.
   * **Trial penalty**: Dynamically scales critical SNR threshold via extreme-value penalty $z_{\text{crit}} = \max(2.8, \sqrt{2\ln(N_{\text{trials}})} + 1.2) \in [3.5\sigma, 4.2\sigma]$.
   * **Off-transit control (unvalidated)**: the intended eight-phase control is not a calibrated null test. The historical **0.0144%** random-pass claim is unsupported and must not be used as an established probability.
   * **Single archival gate (S3.3)**: the legacy archival gate and the legacy odd/even path were retired; `verify_multi_epoch` is the accepted Cp gate (PORT-CP-01: item 12 timing modes, item 18 refinement grid, item 13 Sidak penalty, item 19 refinement check). `odd_even_method` accepts only `v2`. Ranks after the first CANDIDATE are still evaluated and logged to `all_ranks_journal.jsonl` (one record per rank exit plus a host_end record); a later peak whose events coincide with an earlier CANDIDATE becomes REVIEW, and an exception after a CANDIDATE ends the host as ERROR.
   * **Mode 2 label (Argon 31.10)**: an archival pass that needed the period refinement (Mode 2, not a strict forward projection) is labelled `Multi-Epoch_Refined (Conditional on Archival Period Search)` in the candidate metadata (`archival_confirmation_label`). It is conditional on the archival period search and is not independent confirmation; no sector split or extra trials penalty is applied. Mode 2 archival diagnostics also report the circular-secondary SNR at the reference period, plus and minus one refinement grid step and plus and minus sigma_P (SCI-74), as diagnostics only.
3. **Synthetic False-Alarm Demonstration (`tests/benchmark_false_alarm_rate.py`)**:
   The reported **0.55%** is the fraction of 2,000 synthetic Gumbel draws exceeding a threshold, not a benchmark processing 2,000 real quiet TESS stars. It establishes neither empirical survey FAR nor 99.45% real-data specificity. Script labels identify the Gumbel sample and its threshold fractions as synthetic demonstrations. The rate measured in v9.2 on 500 time-scrambled sectors of real quiet stars is 2.0 % at SDE $\ge$ 6.0 (10 of 500, one-sided 95 % upper bound 3.4 %; no false-alarm rate is claimed at SDE $\ge$ 6.0; see Scientific status and limits).
4. **Numerical Integrator and Synthetic Injection Benchmark**:
   The integrator has comparison tests, but the historical **3.09 ppm** number is not a validated universal error bound. `tests/test_mandel_agol_batman.py` uses batman when available and a related annulus reference otherwise; the latter is not independent validation. `tests/benchmark_injection.py` defaults to **288 trials (36 cells × 8)** on a synthetic baseline and a BLS-focused path, not the full production cascade on real TESS curves. Historical **100% recovery** cells describe only those synthetic trials, not population completeness; results were not rerun for this documentation correction. Completeness measured in v9.2 on planets injected into real light curves through the full Tier 1 cascade is given in Scientific status and limits.
5. **Synthetic VPS ROC Demonstration (`tests/benchmark_vps_roc.py`)**:
   The reported **AUC 0.988**, TPR 94%, FPR 2% and VPS cutoff 23.4 describe artificially generated scores/parameters for 50 synthetic TOI-like and 50 synthetic EB/blend-like controls. They do not establish an optimal production threshold or independent performance on real planets/EBs.
6. **Production Configuration for MetaCentrum HPC (Sectors 108, 107, 98–93)**:
   Pre-configured lock-free sharding across the newest available TESS data (Sectors 108 and 107) and deep archival fields (Sectors 98–93).

---

## Key Features in v8.6

1. **Tier-4 Bayesian TRICERATOPS FPP/NFPP Assessment (`validate_triceratops.py`)**:
   Evaluates 18 astrophysical blend scenarios (Giacalone et al. 2021) and reports Bayesian FPP/NFPP estimates conditional on the supplied data, model priors, and scenario model; a threshold pass is exploratory and is not independent statistical validation. Adaptive 60-bin phase folding was introduced to reduce memory use (historically reported from 11.55 GB to ~60 MB for a multi-sector case); binning changes sampling and error assumptions, and a universal SNR gain is not established.
2. **Leave-One-Year-Out (LOYO) Timing Analysis (`loyo_validator.py`)**:
   Implements multi-year timing projections inspired by independent researcher Pavel Rabtsevich (TIC 4206066). Historical examples include a reported $+1.94\text{ min}$ timing offset over 800 days on TIC 350190949 and a 94-transit $\text{O}-\text{C}$ analysis. These examples have not been independently verified as blind holdout evidence or as population-level timing performance.
3. **Master Scanned TICs Consolidation (`merge_scanned_tics.py`)**:
   Aggregates, deduplicates, and numerically sorts historical scanned TICs across MetaCentrum cluster arrays, local archives, and current workspace into a unified registry (expanding local history from 411 to 6,337+ unique TICs).

---

## Key Features in v8.5

1. **Multi-Sector Ephemeris Synchronization (Pleiades v2.5)**:
   Eliminated parameter desynchronization by propagating `refined_t0_btjd`, `fitted_depth_ppm`, and `fitted_duration_hours` directly from multi-sector Mandel-Agol fits into downstream packagers, preventing single-sector BLS parameter contamination in ExoFOP CTOI exports.
2. **$\delta$ Scuti Asteroseismic Pre-Whitening Case Study (TIC 301152169)**:
   A historical single-target analysis subtracted five modeled pulsation modes ($f_1=37.93\text{ c/d}, f_2=41.28, f_3=28.70, f_4=44.72, f_5=37.40$) and reported residual RMS changing from $1554$ to $1143\text{ ppm}$ and transit SNR from $20.9\sigma$ to $28.6\sigma$. This example does not measure population-level completeness or independently establish the transit interpretation.
3. **Decoupled Archival Drill & Earliest-Transit Anchoring (`antares_packager.py`)**:
   Anchors $T_0$ to the earliest observed transit epoch ($t_{0,\text{anchor}}$) to address negative epoch projection drift. Historical TIC 350190949 output reported a 45-min phase difference and SPOC stack SNR changing from $2.1\sigma$ to $4.35\sigma$; this is a target-specific diagnostic, not independent confirmation or a general performance estimate. The workflow separates SPOC 2-min cadence from QLP/TGLC data during this analysis.
4. **Distributed PBS Job Array Sharding (`qsub -J 1-8`)**:
   Enables parallel execution across 8 independent compute nodes without inter-process file locks. Workers autonomously map `PBS_ARRAY_INDEX` to sequential TESS sectors (`[108, 107, 98, 97, 96, 95, 94, 93]`) and write to isolated subdirectories (`tess_results/sector_<sec>/`).
5. **Dual-Zone Full-Orbit Secondary Eclipse Scan**:
   Differentiates between circular orbits ($|\phi - 0.50| \le 0.04$, $\text{SNR} > 3.1\sigma$) and eccentric orbits ($|\phi - 0.50| > 0.04$, Šidák-Bonferroni $\text{SNR} > z_{\text{crit}}$). Enforces a 100 ppm floor and $\ge 10\%$ primary depth.
6. **Modular Directory Architecture**:
   Structured results into dedicated functional subfolders (`tess_results/deep_vetting/`, `tess_results/archival_vetting/`, `tess_results/archive/`) with consolidated summary catalogs in root for clean automated reporting.

---

## Test Suite (`tests/`)

Aldebaran includes tests and benchmark scripts for several pipeline components. Their presence or passing status does not establish end-to-end scientific validation:
* `tests/test_v9_ephemeris_and_multiplanet.py`: Tests TOI catalog ingestion, known-transit masking, sibling planet recovery, contiguous strict projection, and multi-year Cramer-Rao trial scaling.
* `tests/test_mandel_agol_batman.py`: Numerical comparison cases; record whether batman or the related annulus fallback supplied the reference. Not a universal production accuracy bound.
* `tests/benchmark_false_alarm_rate.py`: Synthetic Gumbel-based false-alarm demonstration ($N=2000$ draws), not measured survey FAR.
* `tests/benchmark_injection.py`: Synthetic BLS-focused injection recovery; default 288 trials across 36 cells, not full-cascade completeness.
* `tests/benchmark_vps_roc.py`: ROC/Youden demonstration on artificially generated scores, not independent empirical VPS calibration.
* `tests/test_kovacs_sde.py`: unittest; whitened Kovács SDE on synthetic white noise (no detection above SDE 6) and one injected transit (detected near P = 4 d). Seeded synthetic data only.
* `tests/test_secondary_scan.py`: unittest; `scan_full_orbit_secondary_eclipse` on one synthetic clean planet (no EB flag) and one synthetic eccentric EB at phase 0.35 (flagged). Seeded synthetic data only.
* `tests/test_ring_accuracy.py`: manual script (`python tests/test_ring_accuracy.py`), prints ring-model error versus a 20,000-ring reference for an in-file model; no assertions, not collected by unittest.
* `tests/test_ssl_check.py`: manual network script, prints TLS reachability of ExoFOP, NASA Exoplanet Archive, Gaia TAP and MAST; not collected by unittest.
* `tests/test_binary_catalogs.py`: manual network script with assertions on its own copy of the catalog cross-match (not `pleiades.check_binary_catalogs`); not collected by unittest.
* `tests/test_tess_eb_fetch.py`: manual network script that fetches the Prša et al. (2022) EB table; no assertions, not collected by unittest.

Run the offline unittest suite (a passing suite does not establish scientific validation):
```bash
ALDEBARAN_OFFLINE=1 python -m unittest discover tests
```
The live catalog download test is skipped unless `ALDEBARAN_NETWORK_TESTS=1` is set; when enabled it writes its caches and logs only to a temporary directory.

---

## HPC & Cluster Execution (MetaCentrum)

For large-scale runs across tens of thousands of stars, PBS Pro scripts are available:
* `cluster/run_aldebaran.pbs`: PBS script optimized for MetaCentrum HPC with Job Arrays (`qsub -J 1-8`) and local NVMe `$SCRATCHDIR` I/O (8 CPUs, 24 GB RAM, 35 GB local scratch, 8h walltime).
* `cluster/run_merge.pbs`: Manifest-driven reducer PBS job requiring separate explicit submission.
* `cluster/submit_all.sh`: Explicit scan-only launcher; manifest reduction is submitted separately.
* `merge_shards.py`: Reducer script to merge shard outputs after array execution.
* See [README_METACENTRUM.md](docs/README_METACENTRUM.md) for full deployment and monitoring instructions.

---

## Quickstart

### Local Execution (Single Machine)
```bash
# 1. Install dependencies
python -m pip install pip==26.2.1 setuptools==80.10.2 wheel==0.45.1 || exit 1
python -m pip install --no-build-isolation -c constraints-linux-py311.txt -r requirements.txt || exit 1
python -m pip check || exit 1

# 2. Run Tier-1 screening
python aldebaran.py

# 3. Run Tier-2 deep physical vetting on detected candidates
python pleiades.py

# 4. Run Tier-3 CTOI packaging, RV K calculation & atmospheric ranking
python antares_packager.py

# 5. Run Antares multi-year archival forensic drill on a specific candidate
python antares.py --tic 46314221

# 6. Run Tier-4 Bayesian TRICERATOPS FPP/NFPP assessment (conditional on the supplied data, scenarios, and model priors)
python validate_triceratops.py --tic 322051865 --period 2.831532 --t0 4154.32930 --duration 1.08 --depth 778.4 --sectors 103 104 105 106 --mc 5 --bins 60

# 7. Run exploratory LOYO defaults (TIC 350190949, P=0.926120 d, T0=1325.61416 BTJD, duration=0.81 h; no CLI flags)
python loyo_validator.py

# 8. Merge and synchronize master scanned TICs catalog
python merge_scanned_tics.py
```

### HPC scan submission and captured environments

**Deployment readiness:** These commands submit real cluster work. Confirm the selected source revision, inputs, output paths and resource limits before submission; this guide does not establish scientific readiness for a survey run.

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
`requirements.txt` describes the main pipeline profile (NumPy 2.4.6, SciPy 1.17.1
in `constraints-linux-py311.txt`). It pins TRICERATOPS 1.0.21, the only version the
adapter accepts, but PyTransit 2.2.0 imports `scipy.integrate.trapz`, which SciPy 1.14
removed, so the full TRICERATOPS runtime cannot be imported in this profile; its
contract tests read the installed source only. Tier 4 runs use the separate profile
`requirements-triceratops-linux-py311.in`, resolved in
`requirements-triceratops-linux-py311.lock` (NumPy 1.23.5, SciPy 1.13.1, Astropy
6.1.7). The two Linux profiles cannot share one environment. The Windows
snapshot's TRICERATOPS 1.1.0 is refused by the adapter and does not satisfy
`requirements.txt`.
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
