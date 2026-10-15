#!/usr/bin/env python3
"""
ALDEBARAN EXOPLANET SEARCH PIPELINE — MODULE: TIER-4 TRICERATOPS VALIDATOR
Formal Bayesian False Positive Probability (FPP) and Nearby False Positive Probability (NFPP)
Evaluator across 18 Astrophysical Blend Scenarios (Giacalone et al. 2021).
Optimized with Phase-Folded Adaptive Binning to prevent OOM on multi-sector datasets.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_triceratops_validation(tic_id, period, t0_btjd, duration_hours, depth_ppm, sectors, n_mc=5, n_bins=60):
    print(f"\n{'='*75}")
    print(f" ALDEBARAN TIER-4 BAYESIAN STATISTICAL VALIDATION (TRICERATOPS)")
    print(f" Target: TIC {tic_id}")
    print(f" Ephemeris: P = {period:.6f} d | T0 = {t0_btjd:.5f} BTJD | Depth = {depth_ppm:.1f} ppm")
    print(f" Sectors: {sectors} | Target Phase Bins: {n_bins} | MC Draws: {n_mc}")
    print(f"{'='*75}\n")

    try:
        import triceratops.triceratops as tr
    except ImportError as e:
        print(f"[!] TRICERATOPS import error: {e}")
        return False

    depth_frac = depth_ppm * 1e-6
    dur_days = duration_hours / 24.0

    print(f"[TRICERATOPS] Initializing target object for TIC {tic_id}...")
    try:
        target = tr.target(ID=int(tic_id), sectors=np.array(sectors, dtype=int))
        target.sectors = np.array(target.sectors, dtype=int)
        star0 = target.stars.iloc[0]
        r_star = star0.get('rad', np.nan)
        teff = star0.get('Teff', np.nan)
        print(f"   -> Target host properties: R_* = {r_star:.3f} R_sun, T_eff = {teff:.0f} K" if np.isfinite(r_star) else f"   -> Target loaded.")
        print(f"   -> Stars in aperture catalog: {len(target.stars)} stars loaded.")
    except Exception as e:
        print(f"[!] Error initializing TRICERATOPS target: {e}")
        return False

    print(f"[TRICERATOPS] Calculating transit depths on all stars in aperture (tdepth={depth_frac:.6f})...")
    try:
        target.calc_depths(tdepth=depth_frac)
        print("   -> Depths calculated successfully across all candidate host configurations.")
    except Exception as e:
        print(f"[!] Error calculating depths: {e}")
        return False

    # Fetch light curve around transit window for probability calculation
    import lightkurve as lk
    print(f"[TRICERATOPS] Fetching SPOC light curve data across sectors...")
    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        lc_coll = search.download_all()
        lc = lc_coll.stitch()
        
        # Flatten and phase fold
        lc = lc.remove_nans().remove_outliers(sigma=5)
        phase = ((lc.time.btjd - t0_btjd) / period + 0.5) % 1.0 - 0.5
        dt_days = phase * period
        f_raw = lc.flux.value
        e_raw = lc.flux_err.value
        
        # Crop to transit window: [-2.0 * duration, +2.0 * duration]
        win_mask = np.abs(dt_days) <= (2.0 * dur_days)
        dt_win = dt_days[win_mask]
        f_win = f_raw[win_mask]
        e_win = e_raw[win_mask]
        print(f"   -> Raw multi-sector cadences in transit window: {len(dt_win)} points.")
        
        # CRITICAL OPTIMIZATION: Phase binning to prevent 10.8 GB MemoryError
        # 132,169 simulated scenarios x 10,927 raw points = 10.8 GB array allocation.
        # Binning into n_bins reduces allocation to ~60 MB (<0.6% of original RAM)
        # while increasing SNR per data point by sqrt(N_transits) = sqrt(94) ~ 9.7x!
        bins = np.linspace(-2.0 * dur_days, 2.0 * dur_days, n_bins + 1)
        bin_cents = 0.5 * (bins[:-1] + bins[1:])
        b_time, b_flux, b_err = [], [], []
        
        for i in range(n_bins):
            m_bin = (dt_win >= bins[i]) & (dt_win < bins[i+1])
            n_pts = np.sum(m_bin)
            if n_pts >= 3:
                med_val = float(np.nanmedian(f_win[m_bin]))
                sem_val = float(np.nanstd(f_win[m_bin]) / np.sqrt(n_pts)) if n_pts > 1 else float(np.nanmedian(e_win[m_bin]))
                b_time.append(bin_cents[i])
                b_flux.append(med_val)
                b_err.append(sem_val)
                
        time_binned = np.array(b_time, dtype=float)
        flux_binned = np.array(b_flux, dtype=float)
        
        # Normalize out-of-transit flux to 1.0
        oot_mask = np.abs(time_binned) > (0.6 * dur_days)
        norm_factor = float(np.nanmedian(flux_binned[oot_mask])) if np.sum(oot_mask) >= 5 else float(np.nanmedian(flux_binned))
        flux_binned = flux_binned / norm_factor
        
        # Representative flux uncertainty (scalar float for TRICERATOPS)
        flux_err_scalar = float(np.nanmedian(b_err) / norm_factor)
        
        print(f"   -> Phase-folded binned profile: {len(time_binned)} points (RAM footprint: ~{132169 * len(time_binned) * 8 / (1024**2):.1f} MB)")
        print(f"   -> Effective binned photometric error: {flux_err_scalar*1e6:.1f} ppm")
        
    except Exception as e:
        print(f"[!] Warning fetching and binning light curve: {e}")
        return False

    print(f"\n[TRICERATOPS] Calculating Bayesian scenario probabilities (N_MC={n_mc} draws)...")
    fpps = []
    nfpps = []
    for i in range(n_mc):
        try:
            target.calc_probs(
                time=time_binned,
                flux_0=flux_binned,
                flux_err_0=flux_err_scalar,
                P_orb=period,
                parallel=True,
                verbose=0
            )
            fpps.append(target.FPP)
            nfpps.append(target.NFPP)
            print(f"   [Iter {i+1:2d}/{n_mc}] FPP = {target.FPP:.5f} ({target.FPP*100:.2f}%) | NFPP = {target.NFPP:.5f} ({target.NFPP*100:.2f}%)")
        except Exception as e:
            print(f"   [Iter {i+1:2d}] Failed: {e}")
            continue

    if not fpps:
        print("[!] All Monte Carlo iterations failed.")
        return False

    mean_fpp = float(np.mean(fpps))
    std_fpp = float(np.std(fpps))
    mean_nfpp = float(np.mean(nfpps))
    std_nfpp = float(np.std(nfpps))

    print(f"\n{'='*65}")
    print(f" TRICERATOPS BAYESIAN STATISTICAL VALIDATION SUMMARY")
    print(f" False Positive Probability (FPP): {mean_fpp:.4f} ± {std_fpp:.4f} ({mean_fpp*100:.2f}%)")
    print(f" Nearby False Positive Probability (NFPP): {mean_nfpp:.4f} ± {std_nfpp:.4f} ({mean_nfpp*100:.2f}%)")
    
    # Statistical validation criteria (Giacalone et al. 2021):
    # Validated: FPP < 0.015 (1.5%) and NFPP < 0.001 (0.1%)
    # High-confidence: FPP < 0.10
    # Moderate: FPP < 0.50
    if mean_fpp < 0.015 and mean_nfpp < 0.001:
        disposition = "STATISTICALLY VALIDATED PLANET (FPP < 0.015, NFPP < 0.001)"
    elif mean_fpp < 0.05:
        disposition = "VERY HIGH CONFIDENCE PLANET CANDIDATE (FPP < 5%)"
    elif mean_fpp < 0.10:
        disposition = "HIGH-CONFIDENCE PLANET CANDIDATE (FPP < 10%)"
    elif mean_fpp < 0.50:
        disposition = "MODERATE-CONFIDENCE CANDIDATE (FPP < 50%)"
    else:
        disposition = "LIKELY FALSE POSITIVE / BLEND (FPP >= 50%)"
    print(f" Scientific Disposition: {disposition}")
    print(f"{'='*65}\n")

    os.makedirs("tess_results/triceratops", exist_ok=True)

    # Plot field and scenario fits
    try:
        fits_base = f"tess_results/triceratops/triceratops_fits_TIC_{tic_id}"
        target.plot_fits(time=time_binned, flux_0=flux_binned, flux_err_0=flux_err_scalar, save=True, fname=fits_base)
        plt.savefig(fits_base + ".png", dpi=120)
        plt.close('all')
        print(f"[TRICERATOPS] Saved scenario fits plot to {fits_base}.png/.pdf")
    except Exception as e:
        print(f"[!] Notice plotting fits: {e}")

    try:
        field_base = f"tess_results/triceratops/triceratops_field_TIC_{tic_id}"
        sec_to_plot = sectors[0] if sectors else None
        target.plot_field(sector=sec_to_plot, save=True, fname=field_base)
        plt.savefig(field_base + ".png", dpi=120)
        plt.close('all')
        print(f"[TRICERATOPS] Saved aperture field plot to {field_base}.png/.pdf")
    except Exception as e:
        print(f"[!] Notice plotting field: {e}")

    # Save CSV
    prob_df = target.probs
    csv_path = f"tess_results/triceratops/triceratops_probs_TIC_{tic_id}.csv"
    prob_df.to_csv(csv_path)
    print(f"[TRICERATOPS] Saved scenario probabilities table to {csv_path}")

    # Print scenario breakdown
    print("\n[TRICERATOPS] Top 5 Most Probable Scenarios:")
    sorted_probs = prob_df.sort_values(by="prob", ascending=False)
    for idx, r in sorted_probs.head(5).iterrows():
        print(f"   {r['scenario']:<10}: P = {r['prob']:.4f} ({r['prob']*100:.2f}%)")

    return {
        'tic_id': tic_id,
        'fpp_mean': mean_fpp,
        'fpp_std': std_fpp,
        'nfpp_mean': mean_nfpp,
        'nfpp_std': std_nfpp,
        'disposition': disposition
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TRICERATOPS Bayesian Validation on a candidate.")
    parser.add_argument("--tic", type=int, default=350190949, help="TIC ID")
    parser.add_argument("--period", type=float, default=0.926120, help="Period in days")
    parser.add_argument("--t0", type=float, default=1325.61416, help="T0 BTJD")
    parser.add_argument("--duration", type=float, default=0.81, help="Duration in hours")
    parser.add_argument("--depth", type=float, default=1346.9, help="Depth in ppm")
    parser.add_argument("--sectors", type=int, nargs="+", default=[1, 28, 104, 105], help="TESS sectors")
    parser.add_argument("--mc", type=int, default=5, help="Number of Monte Carlo iterations")
    parser.add_argument("--bins", type=int, default=60, help="Number of phase bins across transit window")
    
    args = parser.parse_args()
    run_triceratops_validation(
        tic_id=args.tic,
        period=args.period,
        t0_btjd=args.t0,
        duration_hours=args.duration,
        depth_ppm=args.depth,
        sectors=args.sectors,
        n_mc=args.mc,
        n_bins=args.bins
    )
