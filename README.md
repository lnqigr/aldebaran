# Aldebaran: Two-Tier Photometric & Pixel-Level Exoplanet Vetting Pipeline for TESS

**Aldebaran** is an automated, high-throughput digital signal processing (DSP) and astrophysical falsification pipeline designed to detect low-SNR transiting exoplanets in archival and newly released sectors of the NASA TESS Extended Mission (2-minute SPOC cadence).

The pipeline is engineered to probe the detection grey zone ($6.0 \le \text{SDE} \le 7.5$) while automatically rejecting instrumental artifacts, background eclipsing binaries (BEBs), grazing stellar binaries (EBLMs), and rotational variables before human inspection.

---

## Pipeline Architecture

### Tier 1: High-Throughput Multi-Core Screening (`aldebaran.py` v7.7)
1. **Catalog Cross-Matching**: Automatically queries Caltech/IPAC TAP servers and ExoFOP to exclude known NASA TOIs, Community TOIs (CTOIs), and confirmed exoplanets (`pscomppars`).
2. **Quality & Downlink Gap Sanitization**: Enforces strict `QUALITY == 0` bitwise filtering and trims $\pm 0.10\text{ d}$ thermal/scattered-light ramps around spacecraft momentum dumps and Earth-downlink gaps ($\Delta t > 0.04\text{ d}$).
3. **Piecewise Iterative Detrending**: Applies upper $3\sigma$ flare clipping and piecewise Savitzky-Golay high-pass filtering independently across continuous observation segments, masking candidate transit windows to prevent self-subtraction.
4. **Kovács (2002) Whitened Signal Residue SDE**: Converts Astropy quadratic log-likelihood power ($0.5 \cdot \text{SNR}^2$) into linear Signal Residue ($SR = \sqrt{2 \cdot \text{Power}}$), removes the low-frequency $1/f$ baseline ramp via a 501-point running median filter, and normalizes by MAD. Drops false-alarm rate at $\text{SDE} \ge 6.0$ on pure noise from 100% down to 1.0%.
5. **Multi-Peak Box Least Squares (BLS)**: Starts duration grid at $0.045\text{ d}$ ($1.08\text{ h}$) to prevent noise boundary collapse, evaluating the top 3 distinct peaks per star.
6. **Multi-Stage Falsification Cascade**:
   * **Cascade 0 (Aperture Crowding)**: Rejects blended fields using SPOC `CROWDSAP` ($< 0.70$) without redundant double dilution division.
   * **Cascade 1 (SDE & Duty Cycle)**: Enforces whitened $\text{SDE} \ge 6.0$ and Keplerian duty cycle bounds ($T_{\text{dur}} / P \le 0.095$).
   * **Cascade 2 (Baseline Morphology & Grazing V-Shape)**: Scales out-of-transit bin scatter to transit duration ($\text{SNR}_{\text{integrated}} \ge 5.5\sigma$), limits phase-folded shoulder bumps ($\le 35\%$), and rejects sharp V-shaped grazing profiles.
   * **Cascade 3 (Multiplicity, Odd/Even & Transit Consistency)**: Requires $\ge 3$ distinct transits, tests odd/even symmetry ($< 2.8\sigma$), and enforces individual transit consistency ($\ge 60\%$ of observed transit epochs must exhibit a real negative flux deficit).
   * **Cascade 4 (Full-Orbit Secondary Eclipse Scan)**: Scans all out-of-transit orbital phases $\phi \in [0.06, 0.94]$ with a sliding box, applying analytical Šidák-Bonferroni trials-factor correction to detect eccentric eclipsing binaries ($e > 0.1$) at $\alpha = 0.001$ false-alarm rate.
   * **Cascade 5 (Astrometric Centroid Stability)**: Performs Welch's t-test on centroid coordinates to reject off-target shifts ($> 2.0\text{ arcsec}$ at $p < 0.001$).
   * **Cascade 6 (Stellar Density, Gaia DR3 RUWE & Keplerian Duration Bounds)**: Queries TIC v8.2 and Gaia DR3 to reject evolved giants ($R_* > 1.75 R_\odot$), unresolved binaries ($\text{RUWE} > 1.4$), and non-planetary radii ($R_p > 15 R_\oplus$). Enforces physical Keplerian duration limits accounting for $1\sigma$ stellar uncertainties (`e_rad`, `e_mass`) and orbital eccentricities $e \le 0.30$ ($0.35\times$ lower bound, $1.38\times$ upper bound).
   * **Cascade 7 (Target Pixel File Difference Imaging)**: Downloads calibrated pixel stamps (`TPF`), verifies that peak flux deficit lies strictly inside the SPOC aperture mask, and rejects pointing-drift dipoles.
   * **Cascade 8 (Multi-Epoch Archival Verification)**: Queries prior TESS sectors across multi-year baselines and verifies persistent detection ($> 2.4\sigma$) in independent archival epochs.
7. **Cache & Disk Protection**: `AUTO_PURGE_CACHE = True` automatically deletes raw FITS files of non-candidates, maintaining disk usage near 0 MB. Dual-stream logging outputs milestone events to `tess_results/pipeline_events.log`.

---

### Tier 2: Deep Physical Vetting (`pleiades.py` v2.2)
Surviving candidates are processed through `pleiades.py`:
* **Tukey Biweight Detrending**: Multi-sector stitching using a Wotan-style time-windowed Tukey biweight slider.
* **300-Ring Annular Limb-Darkened Transit Model**: Sub-ppm accurate numerical integration of Mandel-Agol transit profiles using 300 concentric stellar rings with Claret quadratic limb darkening ($u_1, u_2$). Zero staircase or discretization artifacts.
* **Full-Orbit Secondary Eclipse Scan**: Exhaustive scan across all out-of-transit phases for eccentric secondary eclipses.
* **Astrophysical Binary Catalog Cross-Matching**: Automated cross-match against the VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022, `J/ApJS/258/16/tess-ebs`) and Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Table (`gaiadr3.nss_two_body_orbit`).
* **Gaia DR3 Neighborhood Flux Contamination**: Models TESS PSF overlap for all Gaia DR3 stars within $120\text{ arcsec}$, identifies nearby eclipsing binary (NEB) capable sources, and computes a composite Heuristic False Positive Risk Score.
* **4-Panel Publication Dossiers**: Saves high-resolution diagnostic plots to `tess_results/deep_vetting/dossier_TIC_*.png`.

---

## Test Suite (`tests/`)

Aldebaran includes an automated validation and regression suite:
* `tests/test_ssl_check.py`: Tests connectivity to ExoFOP, MAST, NASA Exoplanet Archive, and Gaia TAP.
* `tests/test_ring_accuracy.py`: Benchmarks 300-ring Mandel-Agol model accuracy vs. reference integration.
* `tests/test_binary_catalogs.py`: Verifies VizieR TESS EB and Gaia DR3 NSS catalog cross-matching.
* `tests/test_kovacs_sde.py`: Validates Kovács whitened SDE noise suppression and planet recovery.
* `tests/test_secondary_scan.py`: Verifies full-orbit eccentric secondary eclipse detection.

Run all tests:
```bash
python -c "import subprocess, sys; [subprocess.run([sys.executable, t], check=True) for t in ['tests/test_ssl_check.py', 'tests/test_ring_accuracy.py', 'tests/test_binary_catalogs.py', 'tests/test_kovacs_sde.py', 'tests/test_secondary_scan.py']]"
```

---

## HPC & Cluster Execution (MetaCentrum)

For large-scale runs across tens of thousands of stars, PBS Pro scripts are available:
* `run_aldebaran.pbs`: PBS script optimized for MetaCentrum HPC with Job Arrays and local NVMe `$SCRATCHDIR` I/O.
* See `README_METACENTRUM.md` for complete submission instructions.

---

## Quickstart

```bash
pip install -r requirements.txt
python aldebaran.py
python pleiades.py
```