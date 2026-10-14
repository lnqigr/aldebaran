# Aldebaran: Two-Tier Photometric & Pixel-Level Exoplanet Vetting Pipeline for TESS

**Aldebaran** is an automated, multi-process digital signal processing (DSP) and astrophysical falsification pipeline designed to detect low-SNR transiting exoplanets in archival and newly released sectors of the NASA TESS Extended Mission (2-minute SPOC cadence).

The pipeline is engineered to probe the detection grey zone ($6.0 \le \text{SDE} \le 7.5$) while automatically rejecting instrumental artifacts, background eclipsing binaries (BEBs), grazing stellar binaries (EBLMs), and rotational variables before human inspection.

---

## Pipeline Architecture

### Tier 1: High-Throughput Multi-Core Screening (`aldebaran.py`)
1. **Catalog Cross-Matching**: Automatically queries Caltech/IPAC TAP servers and ExoFOP to exclude known NASA TOIs, Community TOIs (CTOIs), and confirmed exoplanets (`pscomppars`).
2. **Quality & Downlink Gap Sanitization**: Enforces strict `QUALITY == 0` bitwise filtering and trims $\pm 0.10\text{ d}$ thermal/scattered-light ramps around spacecraft momentum dumps and Earth-downlink gaps ($\Delta t > 0.04\text{ d}$).
3. **Piecewise Iterative Detrending**: Applies upper $3\sigma$ flare clipping and piecewise Savitzky-Golay high-pass filtering independently across continuous observation segments, masking candidate transit windows to prevent self-subtraction.
4. **Multi-Peak Box Least Squares (BLS)**: Evaluates the top 3 distinct frequency peaks per target to prevent downlink gap aliases from masking true planetary signals.
5. **Seven-Stage Falsification Cascade**:
   * **Cascade 0 (Aperture Crowding)**: Rejects blended fields using the SPOC `CROWDSAP` metric ($< 0.70$) and corrects transit depth for neighbor flux dilution.
   * **Cascade 1 (SDE & Duty Cycle)**: Enforces $\text{SDE} \ge 6.0$ and Keplerian duty cycle bounds ($T_{\text{dur}} / P \le 0.095$).
   * **Cascade 2 (Baseline Morphology & Grazing V-Shape)**: Scales out-of-transit bin scatter to the transit duration ($\text{SNR}_{\text{integrated}} \ge 5.5\sigma$), limits phase-folded shoulder bumps ($\le 35\%$), and rejects sharp V-shaped grazing profiles.
   * **Cascade 3 (Multiplicity, Odd/Even & Phase 0.5 Secondary Eclipse)**: Requires $\ge 3$ distinct transits, tests odd/even depth symmetry ($< 2.8\sigma$), and screens for self-luminous secondary eclipses at orbital phase $0.5$.
   * **Cascade 4 (Astrometric Centroid Stability)**: Performs Welch's t-test on `mom_centr` minus `pos_corr` coordinates to reject off-target centroid shifts ($> 2.0\text{ arcsec}$ at $p < 0.001$).
   * **Cascade 5 (Stellar Density, Gaia DR3 RUWE & Keplerian Duration)**: Queries TIC v8.2 and Gaia DR3 to reject evolved giants ($R_* > 1.75 R_\odot$), unresolved binaries ($\text{RUWE} > 1.4$), non-planetary radii ($R_p > 15 R_\oplus$), and transit durations exceeding the analytical circular Keplerian limit ($T_{14} > 1.25 \, T_{14,\max}(\rho_*)$).
   * **Cascade 6 (Target Pixel File Difference Imaging)**: Downloads raw calibrated pixel stamps (`TPF`), computes in- versus out-of-transit difference matrices, verifies that peak flux deficit lies strictly inside the SPOC aperture mask, and rejects pointing-drift dipoles.
   * **Cascade 7 (Multi-Epoch Archival Verification)**: Queries prior TESS sectors across multi-year baselines, refines the orbital period on a dynamically scaled frequency grid, and verifies persistent transit detection ($> 2.4\sigma$) in independent archival epochs.

### Tier 2: Deep Physical Vetting (`vet_candidate.py`)
Surviving candidates are processed through `vet_candidate.py`:
* **Tukey Biweight Detrending**: Multi-sector stitching using a Wotan-style time-windowed Tukey biweight slider.
* **60-Ring Annular Limb-Darkened Transit Model**: Non-linear least-squares fitting of $R_p/R_*$, impact parameter $b$, and transit duration $T_{14}$ constrained by stellar density and Claret quadratic limb-darkening coefficients ($u_1, u_2$).
* **Gaia DR3 Neighborhood Flux Contamination**: Queries all Gaia DR3 sources within $120\text{ arcsec}$, models TESS PSF capture fractions, identifies nearby eclipsing binary (NEB) capable contaminants, and outputs a 4-panel publication dossier.

---

## Usage

```bash
pip install -r requirements.txt
python aldebaran.py
python vet_candidate.py