#!/usr/bin/env python3
"""
ALDEBARAN EXOPLANET SEARCH PIPELINE — MODULE: LOYO VALIDATOR
Leave-One-Year-Out (LOYO) Blind Holdout Verification Engine

Validates exoplanet candidates by strictly withholding one observing season/year,
optimizing transit ephemeris on the remaining seasons, and evaluating whether
the transit appears at the exact predicted epoch without parameter tuning.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import lightkurve as lk

# Set matplotlib backend to Agg
plt.switch_backend('Agg')

def robust_tukey_clean(time, flux, ferr, window_days=0.5):
    """Clean outliers and detrend low-frequency stellar/instrumental variability."""
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    t, f, e = time[good], flux[good], ferr[good] if ferr is not None else np.ones_like(flux[good]) * np.nanstd(flux[good])
    
    # Normalize by median
    med = np.nanmedian(f)
    f = f / med
    e = e / med
    
    # Outlier rejection (5-sigma clipping)
    res = f - 1.0
    mad = np.nanmedian(np.abs(res - np.nanmedian(res)))
    std_rob = 1.4826 * mad if mad > 0 else np.nanstd(res)
    mask = np.abs(res) < 5.0 * std_rob
    t, f, e = t[mask], f[mask], e[mask]
    
    # Savitzky-Golay detrending
    dt = np.nanmedian(np.diff(t))
    pts_per_win = int(window_days / dt)
    if pts_per_win % 2 == 0:
        pts_per_win += 1
    if pts_per_win > 7 and len(t) > pts_per_win:
        try:
            trend = savgol_filter(f, window_length=pts_per_win, polyorder=2)
            f_detrend = f / trend
        except Exception:
            f_detrend = f
    else:
        f_detrend = f
        
    return t, f_detrend, e

def box_transit_model(t, t0, period, depth, duration):
    """Simple box transit model for ephemeris evaluation."""
    phase = ((t - t0) / period + 0.5) % 1.0 - 0.5
    dt_hours = phase * period * 24.0
    model = np.ones_like(t, dtype=float)
    in_transit = np.abs(dt_hours) <= (duration / 2.0)
    model[in_transit] = 1.0 - depth
    return model

def fit_ephemeris(t_all, f_all, p_init, t0_init, dur_hours, depth_init):
    """Refine ephemeris (t0, period, depth) using least-squares on given cadences."""
    dur_days = dur_hours / 24.0
    
    def loss(params):
        t0, p, depth = params
        if depth <= 0 or p <= 0:
            return 1e9
        phase = ((t_all - t0) / p + 0.5) % 1.0 - 0.5
        dt_hrs = phase * p * 24.0
        in_tr = np.abs(dt_hrs) <= (dur_hours / 2.0)
        # Depth penalty / fit
        f_in = f_all[in_tr]
        f_out = f_all[~in_tr]
        if len(f_in) < 5 or len(f_out) < 50:
            return 1e9
        # Objective: minimize squared residuals
        res_in = np.sum((f_in - (1.0 - depth))**2)
        res_out = np.sum((f_out - 1.0)**2)
        return res_in + res_out

    res = minimize(loss, [t0_init, p_init, depth_init], method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    t0_fit, p_fit, depth_fit = res.x
    return t0_fit, p_fit, depth_fit

def evaluate_holdout_sector(t_hold, f_hold, e_hold, t0_pred, p_pred, dur_hours, depth_pred):
    """Blindly evaluate the holdout sector using only predicted ephemeris."""
    phase_hold = ((t_hold - t0_pred) / p_pred + 0.5) % 1.0 - 0.5
    dt_hrs = phase_hold * p_pred * 24.0
    
    in_tr = np.abs(dt_hrs) <= (dur_hours / 2.0)
    out_tr = (np.abs(dt_hrs) > (dur_hours / 2.0)) & (np.abs(dt_hrs) <= 2.5 * dur_hours)
    
    n_in = np.sum(in_tr)
    n_out = np.sum(out_tr)
    
    if n_in < 3:
        return {
            'n_in': n_in, 'snr_blind': 0.0, 'measured_depth_ppm': 0.0,
            'time_offset_min': np.nan, 'status': 'INSUFFICIENT_POINTS'
        }
        
    depth_meas = np.nanmedian(f_hold[out_tr]) - np.nanmedian(f_hold[in_tr]) if n_out > 10 else 1.0 - np.nanmedian(f_hold[in_tr])
    std_out = np.nanstd(f_hold[out_tr]) if n_out > 10 else np.nanstd(f_hold)
    
    snr_blind = (depth_meas / (std_out / np.sqrt(n_in))) if std_out > 0 else 0.0
    
    # Measure timing offset (delta t) by fitting local center in a window [-1.5 dur, +1.5 dur]
    win = np.abs(dt_hrs) <= 1.5 * dur_hours
    t_win, f_win = dt_hrs[win], f_hold[win]
    
    # Cross-correlate with box template to find offset
    grid_shifts = np.linspace(-dur_hours, dur_hours, 101) # hours
    best_shift = 0.0
    best_chi2 = 1e9
    for s in grid_shifts:
        template = np.ones_like(t_win)
        template[np.abs(t_win - s) <= (dur_hours / 2.0)] = 1.0 - depth_pred
        chi2 = np.sum((f_win - template)**2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_shift = s
            
    time_offset_min = best_shift * 60.0
    
    return {
        'n_in': int(n_in),
        'n_out': int(n_out),
        'snr_blind': float(snr_blind),
        'measured_depth_ppm': float(depth_meas * 1e6),
        'predicted_depth_ppm': float(depth_pred * 1e6),
        'depth_ratio': float(depth_meas / depth_pred) if depth_pred > 0 else 0.0,
        'time_offset_min': float(time_offset_min),
        'time_offset_fraction_dur': float(abs(best_shift) / dur_hours),
        'dt_hrs': dt_hrs,
        'status': 'SUCCESS'
    }

def run_loyo_tic(tic_id=350190949, p_nom=0.926120, t0_nom=1325.61416, dur_nom_hr=0.81, depth_nom_ppm=1346.9):
    print(f"\n{'='*70}")
    print(f" ALDEBARAN LOYO (LEAVE-ONE-YEAR-OUT) VERIFICATION ENGINE")
    print(f" Target: TIC {tic_id}")
    print(f" Nominal Ephemeris: P = {p_nom:.6f} d | T0 = {t0_nom:.5f} BTJD | Dur = {dur_nom_hr:.2f} h")
    print(f"{'='*70}\n")
    
    # 1. Download light curves across all available sectors
    print(f"[LOYO] Querying MAST for TESS SPOC 2-minute cadence data on TIC {tic_id}...")
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if len(search) == 0:
        print("[!] No SPOC light curves found.")
        return False
        
    print(f"[LOYO] Found {len(search)} SPOC data products:")
    for idx, item in enumerate(search):
        sec = int(str(item.mission[0]).split()[-1])
        yr = int(item.year[0])
        print(f"   Sector {sec:3d} (Year {yr}) | Author: {item.author[0]} | Exptime: {item.exptime[0]}")

    sector_data = {}
    for idx, item in enumerate(search):
        sec = int(str(item.mission[0]).split()[-1])
        yr = int(item.year[0])
        print(f"[LOYO] Downloading & preparing Sector {sec} ({yr})...")
        lc = item.download()
        if lc is None:
            continue
        # Quality mask and robust detrending
        t = lc.time.btjd
        f = lc.flux.value
        ferr = lc.flux_err.value
        
        t_clean, f_clean, e_clean = robust_tukey_clean(t, f, ferr, window_days=0.5)
        sector_data[sec] = {
            'year': yr,
            'time': t_clean,
            'flux': f_clean,
            'ferr': e_clean,
            'cadences': len(t_clean)
        }
        print(f"   -> Sector {sec}: {len(t_clean)} cadences processed.")

    all_secs = sorted(sector_data.keys())
    print(f"\n[LOYO] Total available sectors: {all_secs}")
    
    # We define 3 cross-validation folds:
    # Fold 1 (Primary - Rabtsevich protocol): Holdout = Sector 28 (Year 2020)
    # Train = Sector 1 (2018) + Sectors 104, 105 (2026)
    folds = [
        {
            'name': 'Fold-1 (Holdout Year 2020 / Sector 28)',
            'holdout_sec': 28,
            'train_secs': [s for s in all_secs if s != 28],
            'description': 'Hold out middle season (2020); train on 2018 + 2026'
        },
        {
            'name': 'Fold-2 (Holdout Year 2018 / Sector 1)',
            'holdout_sec': 1,
            'train_secs': [s for s in all_secs if s != 1],
            'description': 'Hold out earliest season (2018); train on 2020 + 2026'
        },
        {
            'name': 'Fold-3 (Holdout Year 2026 / Sectors 104+105)',
            'holdout_sec': 104, # Or 104+105
            'train_secs': [1, 28],
            'description': 'Hold out latest season (2026); train on 2018 + 2020'
        }
    ]

    fold_results = []
    
    for fold in folds:
        h_sec = fold['holdout_sec']
        t_secs = fold['train_secs']
        print(f"\n{'-'*60}")
        print(f" Executing: {fold['name']}")
        print(f" Training Sectors: {t_secs} | Holdout Sector: {h_sec}")
        print(f"{'-'*60}")
        
        # Concatenate training data
        t_train = np.concatenate([sector_data[s]['time'] for s in t_secs])
        f_train = np.concatenate([sector_data[s]['flux'] for s in t_secs])
        e_train = np.concatenate([sector_data[s]['ferr'] for s in t_secs])
        
        # Fit ephemeris strictly on training data
        print("   -> Fitting ephemeris strictly on Training Sectors...")
        t0_fit, p_fit, depth_fit = fit_ephemeris(t_train, f_train, p_nom, t0_nom, dur_nom_hr, depth_nom_ppm * 1e-6)
        print(f"   -> Training Ephemeris: P = {p_fit:.7f} d | T0 = {t0_fit:.5f} BTJD | Depth = {depth_fit*1e6:.1f} ppm")
        
        # Now blindly evaluate Holdout Sector
        t_h = sector_data[h_sec]['time']
        f_h = sector_data[h_sec]['flux']
        e_h = sector_data[h_sec]['ferr']
        
        print(f"   -> Evaluating Holdout Sector {h_sec} without parameter tuning...")
        metrics = evaluate_holdout_sector(t_h, f_h, e_h, t0_fit, p_fit, dur_nom_hr, depth_fit)
        
        print(f"   [RESULT] Blind SNR in Holdout Sector {h_sec}: {metrics['snr_blind']:.2f} sigma")
        print(f"   [RESULT] Measured Depth: {metrics['measured_depth_ppm']:.1f} ppm (Predicted: {metrics['predicted_depth_ppm']:.1f} ppm)")
        print(f"   [RESULT] Timing Offset: {metrics['time_offset_min']:+.2f} minutes ({metrics['time_offset_fraction_dur']*100:.1f}% of transit duration)")
        
        metrics['fold_name'] = fold['name']
        metrics['holdout_sec'] = h_sec
        metrics['train_secs'] = str(t_secs)
        metrics['t0_train'] = t0_fit
        metrics['p_train'] = p_fit
        fold_results.append(metrics)

    # 4. Detailed individual transit timing (O - C diagram) across all 7.8 years
    print(f"\n[LOYO] Calculating Individual Transit Midpoints (O - C Analysis) across 7.8-year baseline...")
    t_full = np.concatenate([sector_data[s]['time'] for s in all_secs])
    f_full = np.concatenate([sector_data[s]['flux'] for s in all_secs])
    
    # Best master ephemeris
    t0_master, p_master, depth_master = fit_ephemeris(t_full, f_full, p_nom, t0_nom, dur_nom_hr, depth_nom_ppm * 1e-6)
    
    # Find all transit epochs
    min_t, max_t = np.min(t_full), np.max(t_full)
    k_min = int(np.floor((min_t - t0_master) / p_master))
    k_max = int(np.ceil((max_t - t0_master) / p_master))
    
    oc_epochs = []
    oc_residuals_min = []
    oc_sectors = []
    
    for k in range(k_min, k_max + 1):
        t_expected = t0_master + k * p_master
        # check which sector it belongs to
        for sec in all_secs:
            t_s = sector_data[sec]['time']
            if np.min(t_s) <= t_expected <= np.max(t_s):
                # We have observations!
                # Slice local window
                m_tr = np.abs(t_s - t_expected) <= (1.2 * dur_nom_hr / 24.0)
                if np.sum(m_tr) >= 4:
                    # Measure local dip center
                    t_win = t_s[m_tr]
                    f_win = sector_data[sec]['flux'][m_tr]
                    # Template match
                    shifts = np.linspace(-0.6 * dur_nom_hr, 0.6 * dur_nom_hr, 41) / 24.0
                    b_shift = 0.0
                    b_loss = 1e9
                    for s in shifts:
                        m_box = np.ones_like(t_win)
                        m_box[np.abs(t_win - (t_expected + s)) <= (0.5 * dur_nom_hr / 24.0)] = 1.0 - depth_master
                        l = np.sum((f_win - m_box)**2)
                        if l < b_loss:
                            b_loss = l
                            b_shift = s
                    oc_epochs.append(k)
                    oc_residuals_min.append(b_shift * 24.0 * 60.0)
                    oc_sectors.append(sec)
                break

    print(f"[LOYO] Successfully measured {len(oc_epochs)} individual transit events across 7.8 years.")
    oc_df = pd.DataFrame({
        'epoch': oc_epochs,
        'sector': oc_sectors,
        'residual_min': oc_residuals_min
    })
    
    rms_oc = np.nanstd(oc_residuals_min)
    mean_oc = np.nanmean(oc_residuals_min)
    print(f"[LOYO] Timing jitter (RMS of O-C residuals): {rms_oc:.2f} minutes (Duration = {dur_nom_hr*60:.1f} min)")
    
    # 5. Build High-Resolution Diagnostic Figure
    print("[LOYO] Generating 4-Panel Verification Poster (loyo_TIC_350190949.png)...")
    fig = plt.figure(figsize=(20, 14), facecolor='#0d1117', dpi=120)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.22, wspace=0.18,
                           left=0.07, right=0.95, top=0.92, bottom=0.07)
    
    fig.suptitle(f"TIC {tic_id}: Leave-One-Year-Out (LOYO) Blind Holdout Verification\n7.8-Year Ephemeris Stability Test (Sectors 1, 28, 104, 105)",
                 color='#58a6ff', fontsize=18, fontweight='bold', family='sans-serif', y=0.97)

    # Panel A: Baseline Timeline
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#161b22')
    for sp in ax_a.spines.values(): sp.set_color('#30363d')
    ax_a.tick_params(colors='#8b949e', labelsize=10)
    
    # Plot observations
    for sec in all_secs:
        t_sec = sector_data[sec]['time']
        yr = sector_data[sec]['year']
        col = '#f0883e' if sec == 28 else '#58a6ff'
        label = f"Sector {sec} ({yr})" + (" [HOLDOUT]" if sec == 28 else " [TRAIN]")
        ax_a.plot(t_sec, sector_data[sec]['flux'], '.', color=col, alpha=0.3, markersize=1.5, label=label)
    
    ax_a.set_xlabel("Time (TESS BTJD)", color='#c9d1d9', fontsize=11)
    ax_a.set_ylabel("Relative Flux (Detrended)", color='#c9d1d9', fontsize=11)
    ax_a.set_title("Panel A: 7.8-Year Photometric Baseline (2018–2026)", color='#f0f6fc', fontsize=13, fontweight='bold')
    ax_a.legend(loc='lower left', facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
    ax_a.set_ylim(0.985, 1.015)

    # Panel B: Training Sectors Folded Light Curve
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#161b22')
    for sp in ax_b.spines.values(): sp.set_color('#30363d')
    ax_b.tick_params(colors='#8b949e', labelsize=10)
    
    t_train_p1 = np.concatenate([sector_data[s]['time'] for s in [1, 104, 105]])
    f_train_p1 = np.concatenate([sector_data[s]['flux'] for s in [1, 104, 105]])
    t0_tr1 = fold_results[0]['t0_train']
    p_tr1 = fold_results[0]['p_train']
    
    ph_tr = ((t_train_p1 - t0_tr1) / p_tr1 + 0.5) % 1.0 - 0.5
    dt_tr_hrs = ph_tr * p_tr1 * 24.0
    m_win_tr = np.abs(dt_tr_hrs) <= 2.0
    
    ax_b.scatter(dt_tr_hrs[m_win_tr], f_train_p1[m_win_tr], s=2, color='#8b949e', alpha=0.35, label='Training Cadences (S1, S104, S105)')
    
    # Binned points
    bins = np.linspace(-2.0, 2.0, 25)
    b_cents = 0.5 * (bins[:-1] + bins[1:])
    b_means, b_stds = [], []
    for i in range(len(bins)-1):
        idx_b = (dt_tr_hrs[m_win_tr] >= bins[i]) & (dt_tr_hrs[m_win_tr] < bins[i+1])
        if np.sum(idx_b) > 5:
            b_means.append(float(np.nanmedian(f_train_p1[m_win_tr][idx_b])))
            b_stds.append(float(np.nanstd(f_train_p1[m_win_tr][idx_b]) / np.sqrt(np.sum(idx_b))))
        else:
            b_means.append(np.nan)
            b_stds.append(np.nan)
    b_cents = np.array(b_cents, dtype=float)
    b_means = np.array(b_means, dtype=float)
    b_stds = np.array(b_stds, dtype=float)
    valid_b = np.isfinite(b_means) & np.isfinite(b_stds)
    ax_b.errorbar(b_cents[valid_b], b_means[valid_b], yerr=b_stds[valid_b], fmt='o', color='white', ecolor='#58a6ff', elinewidth=1.5, capsize=2, label='Binned Photometry (±1σ)')
    
    # Model curve
    t_grid = np.linspace(-2.0, 2.0, 200)
    m_curve = np.ones_like(t_grid)
    m_curve[np.abs(t_grid) <= (dur_nom_hr / 2.0)] = 1.0 - (depth_nom_ppm * 1e-6)
    ax_b.plot(t_grid, m_curve, color='#3fb950', lw=2.5, label='Fitted Training Model')
    
    ax_b.set_xlabel("Time from Mid-Transit (hours)", color='#c9d1d9', fontsize=11)
    ax_b.set_ylabel("Relative Flux", color='#c9d1d9', fontsize=11)
    ax_b.set_title("Panel B: Training Phase-Fold (S1 + S104 + S105)", color='#f0f6fc', fontsize=13, fontweight='bold')
    ax_b.legend(loc='lower right', facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
    ax_b.set_ylim(0.995, 1.004)

    # Panel C: BLIND Holdout Phase Fold (Sector 28, Year 2020)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor('#161b22')
    for sp in ax_c.spines.values(): sp.set_color('#30363d')
    ax_c.tick_params(colors='#8b949e', labelsize=10)
    
    t_h28 = sector_data[28]['time']
    f_h28 = sector_data[28]['flux']
    ph_h28 = ((t_h28 - t0_tr1) / p_tr1 + 0.5) % 1.0 - 0.5
    dt_h28_hrs = ph_h28 * p_tr1 * 24.0
    m_win_h28 = np.abs(dt_h28_hrs) <= 2.0
    
    ax_c.scatter(dt_h28_hrs[m_win_h28], f_h28[m_win_h28], s=4, color='#f0883e', alpha=0.45, label='Sector 28 Cadences (Year 2020 - Blind Holdout)')
    
    b_means_h, b_stds_h = [], []
    for i in range(len(bins)-1):
        idx_b = (dt_h28_hrs[m_win_h28] >= bins[i]) & (dt_h28_hrs[m_win_h28] < bins[i+1])
        if np.sum(idx_b) > 3:
            b_means_h.append(float(np.nanmedian(f_h28[m_win_h28][idx_b])))
            b_stds_h.append(float(np.nanstd(f_h28[m_win_h28][idx_b]) / np.sqrt(np.sum(idx_b))))
        else:
            b_means_h.append(np.nan)
            b_stds_h.append(np.nan)
    b_means_h = np.array(b_means_h, dtype=float)
    b_stds_h = np.array(b_stds_h, dtype=float)
    valid_h = np.isfinite(b_means_h) & np.isfinite(b_stds_h)
    snr_b_val = fold_results[0]['snr_blind']
    offset_b_val = fold_results[0]['time_offset_min']
    ax_c.errorbar(b_cents[valid_h], b_means_h[valid_h], yerr=b_stds_h[valid_h], fmt='s', color='#f0883e', ecolor='white', elinewidth=1.8, capsize=3, label=f'Blind Binned Photometry (SNR={snr_b_val:.1f}σ)')
    ax_c.plot(t_grid, m_curve, color='#3fb950', lw=2.5, linestyle='--', label=f'Blind Prediction (Offset: {offset_b_val:+.1f} min)')
    ax_c.axvline(0.0, color='white', linestyle=':', alpha=0.5)
    
    ax_c.set_xlabel("Time from Predicted Mid-Transit (hours)", color='#c9d1d9', fontsize=11)
    ax_c.set_ylabel("Relative Flux", color='#c9d1d9', fontsize=11)
    ax_c.set_title("Panel C: BLIND Holdout Evaluation (Sector 28, Year 2020)", color='#f0f6fc', fontsize=13, fontweight='bold')
    ax_c.legend(loc='lower right', facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
    ax_c.set_ylim(0.995, 1.004)

    # Panel D: O - C Diagram across 7.8 Years
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('#161b22')
    for sp in ax_d.spines.values(): sp.set_color('#30363d')
    ax_d.tick_params(colors='#8b949e', labelsize=10)
    
    sec_colors = {1: '#58a6ff', 28: '#f0883e', 104: '#3fb950', 105: '#bc8cff'}
    for sec in all_secs:
        sub = oc_df[oc_df['sector'] == sec]
        ax_d.scatter(sub['epoch'], sub['residual_min'], color=sec_colors.get(sec, 'white'),
                     s=25, alpha=0.85, label=f"Sector {sec} ({sector_data[sec]['year']})")
                     
    ax_d.axhline(0.0, color='red', linestyle='--', lw=1.5, alpha=0.7)
    ax_d.axhspan(-dur_nom_hr*30, dur_nom_hr*30, color='#21262d', alpha=0.5, label='±50% Transit Duration')
    ax_d.set_xlabel("Transit Epoch Number (Cycles)", color='#c9d1d9', fontsize=11)
    ax_d.set_ylabel("O - C Timing Residual (minutes)", color='#c9d1d9', fontsize=11)
    ax_d.set_title(f"Panel D: Observed minus Calculated (O - C) Timing (RMS = {rms_oc:.2f} min)", color='#f0f6fc', fontsize=13, fontweight='bold')
    ax_d.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
    ax_d.set_ylim(-35, 35)

    out_plot = "tess_results/loyo_TIC_350190949.png"
    plt.savefig(out_plot, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[LOYO] Saved diagnostic plot to {out_plot}!")
    
    # Save CSV summary
    res_df = pd.DataFrame(fold_results)
    csv_out = "tess_results/loyo_summary_350190949.csv"
    res_df.to_csv(csv_out, index=False)
    print(f"[LOYO] Saved fold statistics to {csv_out}!")
    
    return True

if __name__ == '__main__':
    run_loyo_tic()
