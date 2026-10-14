import os
import sys
import ssl
import glob
import shutil
import argparse
import warnings
from datetime import datetime, timedelta

# Global SSL bypass for MAST / Caltech TAP compatibility
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import lightkurve as lk
from astroquery.mast import Catalogs

RESULTS_DIR = "tess_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Author priority for deduplicating overlapping sectors in the Master Stack
AUTHOR_PRIORITY = {"SPOC": 1, "TESS-SPOC": 2, "TGLC": 3, "QLP": 4}
AUTHOR_COLORS = {"SPOC": "crimson", "TESS-SPOC": "darkorange", "TGLC": "royalblue", "QLP": "seagreen"}


def to_numpy(col) -> np.ndarray:
    mask = None
    if hasattr(col, "mask") and col.mask is not None:
        mask = np.asarray(col.mask, dtype=bool)
    for _ in range(3):
        if hasattr(col, "unmasked"):
            col = col.unmasked
        if hasattr(col, "value"):
            col = col.value
    arr = np.array(col, dtype=np.float64, copy=True)
    if mask is not None and mask.shape == arr.shape:
        arr[mask] = np.nan
    return arr


def btjd_to_year(btjd_val: float) -> int:
    """Converts TESS Barycentric Julian Date (BTJD = BJD - 2457000.0) to calendar year."""
    base_dt = datetime(2014, 12, 8, 12, 0, 0)
    target_dt = base_dt + timedelta(days=float(btjd_val))
    return int(target_dt.year)


def get_stellar_limb_darkening(tic_id: str) -> dict:
    """Queries TIC v8.2 for R_star, M_star, Teff, Tmag and estimates quadratic limb darkening."""
    info = {"r_star": 0.61, "m_star": 0.60, "teff": 3958.0, "tmag": 13.8, "u1": 0.57, "u2": 0.20}
    try:
        tic_data = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
        if len(tic_data) > 0:
            row = tic_data[0]
            r_val = float(to_numpy(row["rad"]))
            t_val = float(to_numpy(row["Teff"]))
            m_val = float(to_numpy(row["mass"])) if "mass" in row.colnames else np.nan
            tmag_val = float(to_numpy(row["Tmag"])) if "Tmag" in row.colnames else np.nan
            if np.isfinite(r_val) and r_val > 0:
                info["r_star"] = r_val
                info["m_star"] = m_val if (np.isfinite(m_val) and m_val > 0) else (r_val ** 1.1)
            if np.isfinite(t_val) and t_val > 0:
                info["teff"] = t_val
                info["u1"] = float(np.clip(0.45 - 0.00008 * (t_val - 5500.0), 0.15, 0.70))
                info["u2"] = float(np.clip(0.25 + 0.00003 * (t_val - 5500.0), 0.05, 0.45))
            if np.isfinite(tmag_val):
                info["tmag"] = tmag_val
    except Exception:
        pass
    return info


def extract_multi_author_lc(lc_obj, author: str, edge_trim_days: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Extracts time, normalized flux, flux error, and cadence (days) from SPOC, TESS-SPOC, QLP, or TGLC
    light curves. Prioritizes PSF-deblended columns in TGLC and KSP detrended columns in QLP.
    """
    try:
        t = to_numpy(lc_obj.time)
        
        # Select optimal flux column based on pipeline author
        f = None
        if author == "TGLC":
            for col_name in ["cal_psf_flux", "psf_flux", "cal_aper_flux", "flux"]:
                if hasattr(lc_obj, col_name) and getattr(lc_obj, col_name) is not None:
                    f_try = to_numpy(getattr(lc_obj, col_name))
                    if np.sum(np.isfinite(f_try)) > 100:
                        f = f_try
                        break
        elif author == "QLP":
            for col_name in ["kspsap_flux", "sap_flux", "flux"]:
                if hasattr(lc_obj, col_name) and getattr(lc_obj, col_name) is not None:
                    f_try = to_numpy(getattr(lc_obj, col_name))
                    if np.sum(np.isfinite(f_try)) > 100:
                        f = f_try
                        break
        if f is None:
            f = to_numpy(lc_obj.flux)

        # Quality flag filtering
        if hasattr(lc_obj, "quality") and lc_obj.quality is not None:
            qual = np.asarray(lc_obj.quality, dtype=int)
            q_valid = (qual == 0)
        else:
            q_valid = np.ones_like(t, dtype=bool)

        valid = np.isfinite(t) & np.isfinite(f) & q_valid
        if np.sum(valid) < 80:
            return None
        t, f = t[valid], f[valid]

        # Handle negative zero-point offsets in raw TGLC PSF flux if present
        med_f = float(np.nanmedian(f))
        if med_f <= 0:
            f = f - med_f + 1.0
            med_f = 1.0
        f = f / med_f

        # Remove extreme positive flares (> 3.5 MAD) and corrupted negative drops (< 0.5)
        mad_f = float(1.4826 * np.nanmedian(np.abs(f - 1.0)) + 1e-12)
        keep = (f > 0.5) & (f < (1.0 + 3.5 * mad_f))
        if np.sum(keep) < 80:
            return None
        t, f = t[keep], f[keep]
        sort_idx = np.argsort(t)
        t, f = t[sort_idx], f[sort_idx]

        # Estimate cadence in days
        dt_arr = np.diff(t)
        cadence_days = float(np.nanmedian(dt_arr[dt_arr > 0])) if len(dt_arr) > 0 else (120.0 / 86400.0)

        # Trim scattered-light ramps around major downlink gaps (dt > 0.08 days)
        if edge_trim_days > 0 and (t[-1] - t[0]) > 5.0:
            gap_idx = np.where(np.diff(t) > 0.08)[0]
            edge_keep = (t >= (t[0] + edge_trim_days)) & (t <= (t[-1] - edge_trim_days))
            for g_i in gap_idx:
                edge_keep &= ~((t >= (t[g_i] - edge_trim_days)) & (t <= (t[g_i + 1] + edge_trim_days)))
            if np.sum(edge_keep) >= 80:
                t, f = t[edge_keep], f[edge_keep]

        # Robust point-to-point white noise estimation via first-difference MAD
        diff_mad = float(1.4826 * np.nanmedian(np.abs(np.diff(f))) / np.sqrt(2.0))
        sigma_pt = max(diff_mad, 1e-5)
        ferr = np.full_like(f, sigma_pt)

        return t, f, ferr, cadence_days
    except Exception:
        return None


def hybrid_tukey_savgol_detrend(
    t: np.ndarray,
    f: np.ndarray,
    cadence_days: float,
    window_days: float = 0.65,
    transit_mask: np.ndarray = None
) -> np.ndarray:
    """
    Two-stage detrender: applies piecewise Savitzky-Golay across continuous segments followed by
    a Wotan-style time-windowed Tukey biweight slider, strictly masking in-transit cadences.
    """
    if transit_mask is None:
        transit_mask = np.zeros_like(t, dtype=bool)
    t_unmasked = t[~transit_mask]
    f_unmasked = f[~transit_mask]
    if len(t_unmasked) < 20:
        return f / np.nanmedian(f)

    # Stage A: Piecewise Savitzky-Golay on interpolated continuous segments
    f_interp = np.interp(t, t_unmasked, f_unmasked)
    trend_sg = np.ones_like(f)
    split_idx = np.where(np.diff(t) > 0.08)[0] + 1
    segments = np.split(np.arange(len(t)), split_idx)
    target_win_pts = max(15, int(round((window_days * 1.3) / max(cadence_days, 0.0005))))

    for seg in segments:
        if len(seg) < 15:
            trend_sg[seg] = np.nanmedian(f_interp[seg]) if len(seg) > 0 else 1.0
            continue
        win = min(target_win_pts, (len(seg) // 2) * 2 - 1)
        if win >= 7:
            trend_sg[seg] = savgol_filter(f_interp[seg], window_length=win, polyorder=2)
        else:
            trend_sg[seg] = np.nanmedian(f_interp[seg])

    f_sg = f / np.where(np.isfinite(trend_sg) & (trend_sg > 0.1), trend_sg, 1.0)
    f_sg_unmasked = f_sg[~transit_mask]

    # Stage B: Local Tukey Biweight time-windowed slider to remove residual slow ripples
    half_win = 0.5 * window_days
    c_tukey = 6.0
    step = max(1, len(t) // 350)
    t_grid = t[::step]
    trend_grid = np.ones_like(t_grid)

    idx_l = np.searchsorted(t_unmasked, t_grid - half_win, side="left")
    idx_r = np.searchsorted(t_unmasked, t_grid + half_win, side="right")

    for i, (l_i, r_i) in enumerate(zip(idx_l, idx_r)):
        if r_i - l_i < 5:
            trend_grid[i] = 1.0
            continue
        seg = f_sg_unmasked[l_i:r_i]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        if mad < 1e-12:
            trend_grid[i] = med
            continue
        for _ in range(2):
            u = (seg - med) / (c_tukey * mad)
            v = np.abs(u) < 1.0
            if not np.any(v):
                break
            w = (1.0 - (u[v] ** 2)) ** 2
            med = np.sum(w * seg[v]) / np.sum(w)
        trend_grid[i] = med

    trend_tb = np.interp(t, t_grid, trend_grid)
    f_clean = f_sg / np.where(np.isfinite(trend_tb) & (trend_tb > 0.1), trend_tb, 1.0)
    return f_clean / np.nanmedian(f_clean)


def annulus_300_transit_model(
    phase_days: np.ndarray,
    rp_rs: float,
    b: float,
    dur_days: float,
    u1: float,
    u2: float,
    exp_time_days: float = 0.0
) -> np.ndarray:
    """
    300-ring limb-darkened transit model with Kipping (2010) 7-point numerical supersampling
    to account for finite exposure time smearing in 30-min / 10-min FFI light curves.
    """
    rp_rs = float(np.clip(rp_rs, 0.001, 0.35))
    b = float(np.clip(b, 0.0, 0.995))
    dur_days = max(float(dur_days), 0.005)

    n_rings = 300
    r_edges = np.linspace(0.0, 1.0, n_rings + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    ring_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    mu = np.sqrt(np.clip(1.0 - r_centers ** 2, 0.0, 1.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    ring_flux = ring_areas * intensity
    total_flux = np.sum(ring_flux)
    r_c = r_centers[None, :]
    half_chord = np.sqrt(max(1e-6, (1.0 + rp_rs) ** 2 - b ** 2))

    def eval_instantaneous(ph_arr: np.ndarray) -> np.ndarray:
        x_proj = (ph_arr / (0.5 * dur_days)) * half_chord
        z = np.hypot(b, x_proj)
        fl = np.ones_like(ph_arr, dtype=np.float64)
        in_d = z < (1.0 + rp_rs)
        if not np.any(in_d):
            return fl
        z_in = z[in_d][:, None]
        blocked = np.zeros((z_in.shape[0], n_rings), dtype=np.float64)
        full_c = (z_in + r_c) <= rp_rs
        blocked[full_c] = 1.0
        part_c = (~full_c) & (np.abs(z_in - r_c) < rp_rs) & (z_in > 0)
        if np.any(part_c):
            zp = np.broadcast_to(z_in, blocked.shape)[part_c]
            rp = np.broadcast_to(r_c, blocked.shape)[part_c]
            cos_a = (zp ** 2 + rp ** 2 - rp_rs ** 2) / (2.0 * zp * rp)
            blocked[part_c] = np.arccos(np.clip(cos_a, -1.0, 1.0)) / np.pi
        fl[in_d] = 1.0 - (np.sum(blocked * ring_flux[None, :], axis=1) / total_flux)
        return fl

    if exp_time_days <= 0.002:
        return eval_instantaneous(phase_days)

    # 7-point Kipping (2010) supersampling across [-0.5 * exp_time, +0.5 * exp_time]
    offsets = np.linspace(-0.5 * exp_time_days, 0.5 * exp_time_days, 7)
    acc = np.zeros_like(phase_days, dtype=np.float64)
    for off in offsets:
        acc += eval_instantaneous(phase_days + off)
    return acc / 7.0


def refine_multi_year_ephemeris(
    t_all: np.ndarray,
    f_all: np.ndarray,
    ferr_all: np.ndarray,
    p_init: float,
    t0_init: float,
    dur_days: float
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Executes an ultra-fine coherent matched-filter search around p_init across the full
    multi-year baseline to eliminate multi-hour phase drift over thousands of orbits.
    """
    baseline_days = max(27.0, float(np.max(t_all) - np.min(t_all)))
    n_cycles = max(10.0, baseline_days / p_init)

    # Search +/- 0.05% around p_init with step dP <= 0.025 * dur / n_cycles
    half_span_p = max(0.00015 * p_init, min(0.0012 * p_init, 3.5 * dur_days / max(n_cycles, 50.0)))
    max_dp = (0.025 * dur_days) / n_cycles
    n_grid = int(np.clip((2.0 * half_span_p) / max_dp, 1200, 8000))
    trial_periods = np.linspace(p_init - half_span_p, p_init + half_span_p, n_grid)

    weights = 1.0 / np.maximum(ferr_all ** 2, 1e-10)
    deficit = 1.0 - f_all
    half_tr = 0.45 * dur_days

    snr_grid = np.zeros(n_grid, dtype=np.float64)
    for i, p_try in enumerate(trial_periods):
        ph = ((t_all - t0_init + 0.5 * p_try) % p_try) - 0.5 * p_try
        in_m = np.abs(ph) < half_tr
        n_in = np.sum(in_m)
        if n_in < 8:
            continue
        w_in = weights[in_m]
        w_sum = np.sum(w_in)
        mean_def = np.sum(w_in * deficit[in_m]) / w_sum
        snr_grid[i] = mean_def * np.sqrt(w_sum)

    best_idx = int(np.nanargmax(snr_grid))
    p_refined = float(trial_periods[best_idx])

    # Fine-tune T0 within +/- 0.35 * dur_days at p_refined
    t0_grid = np.linspace(t0_init - 0.35 * dur_days, t0_init + 0.35 * dur_days, 300)
    t0_snr = np.zeros_like(t0_grid)
    for j, t0_try in enumerate(t0_grid):
        ph = ((t_all - t0_try + 0.5 * p_refined) % p_refined) - 0.5 * p_refined
        in_m = np.abs(ph) < half_tr
        if np.sum(in_m) >= 8:
            w_in = weights[in_m]
            w_sum = np.sum(w_in)
            t0_snr[j] = (np.sum(w_in * deficit[in_m]) / w_sum) * np.sqrt(w_sum)

    t0_refined = float(t0_grid[int(np.nanargmax(t0_snr))])
    return p_refined, t0_refined, trial_periods, snr_grid


def compute_transit_metrics(
    t: np.ndarray,
    f: np.ndarray,
    ferr: np.ndarray,
    p_val: float,
    t0_val: float,
    dur_days: float,
    cadence_days: float = 0.0014
) -> dict:
    """Computes depth, white-noise SNR, red-noise corrected SNR, and unique transit count."""
    eff_dur = np.hypot(dur_days, cadence_days)
    ph = ((t - t0_val + 0.5 * p_val) % p_val) - 0.5 * p_val
    in_tr = np.abs(ph) < (0.45 * eff_dur)
    out_tr = (np.abs(ph) > (0.70 * (dur_days + cadence_days))) & (np.abs(ph) < min(0.25 * p_val, 3.5 * dur_days))

    n_pts_in = int(np.sum(in_tr))
    if n_pts_in < 3 or np.sum(out_tr) < 10:
        return {"n_transits": 0, "n_cadences_in": n_pts_in, "depth_ppm": np.nan, "err_white_ppm": np.nan, "snr_white": 0.0, "snr_red": 0.0, "beta_red": 1.0}

    orbit_nums = np.round((t - t0_val) / p_val).astype(int)
    n_unique_transits = int(len(np.unique(orbit_nums[in_tr])))

    w_in = 1.0 / np.maximum(ferr[in_tr] ** 2, 1e-10)
    w_out = 1.0 / np.maximum(ferr[out_tr] ** 2, 1e-10)
    base_val = float(np.sum(w_out * f[out_tr]) / np.sum(w_out))
    in_val = float(np.sum(w_in * f[in_tr]) / np.sum(w_in))

    depth_ppm = (base_val - in_val) * 1e6
    err_white_ppm = float((1.0 / np.sqrt(np.sum(w_in))) * 1e6)
    snr_white = depth_ppm / max(err_white_ppm, 1.0)

    # Measure empirical out-of-transit bin scatter (beta factor) on transit-duration scale
    win_ph = min(0.20 * p_val, 3.0 * dur_days)
    vis = np.abs(ph) <= win_ph
    bins = np.linspace(-win_ph, win_ph, 35)
    b_means, b_edges, _ = stats.binned_statistic(ph[vis], f[vis], statistic="mean", bins=bins)
    b_cnt, _, _ = stats.binned_statistic(ph[vis], f[vis], statistic="count", bins=bins)
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    oot_b = (np.abs(b_centers) > (0.70 * (dur_days + cadence_days))) & (b_cnt >= 3)

    if np.sum(oot_b) >= 6:
        oot_bin_std_ppm = float(np.nanstd(b_means[oot_b]) * 1e6)
        bin_w = (2.0 * win_ph) / 34.0
        n_bins_tr = max(1.0, eff_dur / bin_w)
        err_red_ppm = oot_bin_std_ppm / np.sqrt(n_bins_tr)
        beta_red = max(1.0, err_red_ppm / max(err_white_ppm, 1.0))
        snr_red = depth_ppm / max(err_red_ppm, err_white_ppm, 1.0)
    else:
        beta_red = 1.0
        snr_red = snr_white

    return {
        "n_transits": n_unique_transits,
        "n_cadences_in": n_pts_in,
        "depth_ppm": round(depth_ppm, 1),
        "err_white_ppm": round(err_white_ppm, 1),
        "snr_white": round(snr_white, 2),
        "snr_red": round(snr_red, 2),
        "beta_red": round(beta_red, 2)
    }


def run_antares_detective(tic_id: str, p_init: float, t0_init: float, dur_hr: float):
    dur_days = dur_hr / 24.0
    print("=" * 76)
    print(f"ANTARES HISTORICAL SPECTRO-PHOTOMETRIC DETECTIVE: TIC {tic_id}")
    print(f"Input Ephemeris: P = {p_init:.6f} d | T0 = {t0_init:.5f} BTJD | Dur = {dur_hr:.2f} h")
    print("=" * 76)

    star_info = get_stellar_limb_darkening(tic_id)
    print(
        f"[1/6] Host Star Parameters: R* = {star_info['r_star']:.2f} R_Sun | M* = {star_info['m_star']:.2f} M_Sun | "
        f"Teff = {star_info['teff']:.0f} K | Tmag = {star_info['tmag']:.2f} | LD = ({star_info['u1']:.2f}, {star_info['u2']:.2f})"
    )

    # Step 2: Query MAST for all available TESS light curves across all pipelines
    print(f"[2/6] Querying MAST archive for all SPOC, TESS-SPOC, QLP, and TGLC light curves (2018-2026)...")
    search_all = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    if len(search_all) == 0:
        raise RuntimeError(f"No TESS light curves found on MAST for TIC {tic_id}.")

    records = []
    seen_sec_author = set()

    for item in search_all:
        try:
            auth = str(item.author[0]).upper()
            if auth not in AUTHOR_PRIORITY:
                continue
            mission_str = str(item.mission[0])
            sec_num = int(mission_str.split()[-1])
            exptime_s = float(item.exptime[0].value) if hasattr(item.exptime[0], "value") else float(item.exptime[0])
            key = (sec_num, auth)
            # If multiple cadences exist for same sector & author (e.g. 20s vs 120s SPOC), prefer 120s-1800s standard
            if key in seen_sec_author and exptime_s < 100:
                continue
            seen_sec_author.add(key)

            lc_dl = item.download()
            extracted = extract_multi_author_lc(lc_dl, auth, edge_trim_days=0.08)
            if extracted is None:
                continue
            t_s, f_s, ferr_s, cad_days = extracted
            obs_year = btjd_to_year(float(np.nanmedian(t_s)))

            # Pass 1 Detrending (unmasked robust Tukey slider to prepare for fine ephemeris lock)
            f_pass1 = hybrid_tukey_savgol_detrend(t_s, f_s, cad_days, window_days=0.65, transit_mask=None)

            records.append({
                "sector": sec_num,
                "year": obs_year,
                "author": auth,
                "priority": AUTHOR_PRIORITY[auth],
                "exptime_s": int(round(exptime_s)),
                "cadence_days": cad_days,
                "t": t_s,
                "f_raw": f_s,
                "f_pass1": f_pass1,
                "ferr": ferr_s
            })
            print(f"      -> Loaded Sector {sec_num:3d} ({obs_year}) | Author: {auth:9s} | Cadence: {int(round(exptime_s)):4d}s | Points: {len(t_s)}")
        except Exception as e:
            continue

    if not records:
        raise RuntimeError(f"Could not extract any valid SPOC/QLP/TGLC light curves for TIC {tic_id}.")

    # Sort records by sector and author priority
    records.sort(key=lambda r: (r["sector"], r["priority"]))

    # Select 1 best light curve per unique sector for the Deduplicated Master Stack
    master_by_sector = {}
    for rec in records:
        sec = rec["sector"]
        if sec not in master_by_sector or rec["priority"] < master_by_sector[sec]["priority"]:
            master_by_sector[sec] = rec
    master_records = [master_by_sector[s] for s in sorted(master_by_sector.keys())]

    # Step 3: Multi-Year Ephemeris Refinement (Pass 1 -> Ephemeris Lock -> Pass 2 Masked Detrend)
    t_pass1 = np.concatenate([r["t"] for r in master_records])
    f_pass1 = np.concatenate([r["f_pass1"] for r in master_records])
    ferr_pass1 = np.concatenate([r["ferr"] for r in master_records])
    s_idx = np.argsort(t_pass1)
    t_pass1, f_pass1, ferr_pass1 = t_pass1[s_idx], f_pass1[s_idx], ferr_pass1[s_idx]

    print(f"[3/6] Locking coherent ephemeris across {len(master_records)} unique sectors ({t_pass1[-1]-t_pass1[0]:.1f} d baseline)...")
    p_ref, t0_ref, p_grid, snr_grid = refine_multi_year_ephemeris(t_pass1, f_pass1, ferr_pass1, p_init, t0_init, dur_days)
    dp_sec = (p_ref - p_init) * 86400.0
    print(f"      -> Refined Period: P = {p_ref:.7f} d (shift = {dp_sec:+.2f} s/orbit) | Refined T0 = {t0_ref:.5f} BTJD")

    # Step 4: Pass 2 Cadence-Aware Transit-Masked Detrending for all records
    print(f"[4/6] Running Pass-2 transit-masked Tukey/SavGol detrending across all {len(records)} sector datasets...")
    summary_rows = []
    for rec in records:
        t_s = rec["t"]
        cad_d = rec["cadence_days"]
        ph_s = ((t_s - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref
        # Expand mask by cadence_days to protect finite-integration smeared ingress/egress wings
        tr_mask = np.abs(ph_s) < (0.65 * (dur_days + cad_d))
        rec["f_clean"] = hybrid_tukey_savgol_detrend(t_s, rec["f_raw"], cad_d, window_days=0.65, transit_mask=tr_mask)

        m_sec = compute_transit_metrics(t_s, rec["f_clean"], rec["ferr"], p_ref, t0_ref, dur_days, cad_d)
        rec["metrics"] = m_sec
        summary_rows.append({
            "tic_id": tic_id,
            "group_type": "SECTOR_AUTHOR",
            "label": f"S{rec['sector']}_{rec['author']}",
            "sector": rec["sector"],
            "year": rec["year"],
            "author": rec["author"],
            "exptime_s": rec["exptime_s"],
            "in_master_stack": (master_by_sector[rec["sector"]]["author"] == rec["author"]),
            "refined_period_days": round(p_ref, 7),
            "refined_t0_btjd": round(t0_ref, 5),
            **m_sec
        })

    # Re-assemble Master Deduplicated Multi-Year Time Series
    t_master = np.concatenate([r["t"] for r in master_records])
    f_master = np.concatenate([r["f_clean"] for r in master_records])
    ferr_master = np.concatenate([r["ferr"] for r in master_records])
    cad_master = float(np.median([r["cadence_days"] for r in master_records]))
    m_idx = np.argsort(t_master)
    t_master, f_master, ferr_master = t_master[m_idx], f_master[m_idx], ferr_master[m_idx]

    master_metrics = compute_transit_metrics(t_master, f_master, ferr_master, p_ref, t0_ref, dur_days, cad_master)
    summary_rows.insert(0, {
        "tic_id": tic_id,
        "group_type": "MASTER_DEDUPLICATED_ALL_YEARS",
        "label": "ALL_SECTORS_MASTER",
        "sector": -1,
        "year": -1,
        "author": "COMBINED",
        "exptime_s": int(round(cad_master * 86400.0)),
        "in_master_stack": True,
        "refined_period_days": round(p_ref, 7),
        "refined_t0_btjd": round(t0_ref, 5),
        **master_metrics
    })

    # Fit 300-ring physical transit model to Master Stack
    print(f"[5/6] Fitting 300-ring limb-darkened model & evaluating secular year-by-year stability...")
    ph_master = ((t_master - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref
    fit_win = np.abs(ph_master) <= (2.5 * dur_days)
    rp_rs_guess = np.sqrt(max(1e-5, master_metrics["depth_ppm"] * 1e-6))

    def master_model(ph, rp_rs, b_val, d_val):
        return annulus_300_transit_model(ph, rp_rs, b_val, d_val, star_info["u1"], star_info["u2"], exp_time_days=cad_master)

    try:
        popt, _ = curve_fit(
            master_model,
            ph_master[fit_win],
            f_master[fit_win],
            sigma=ferr_master[fit_win],
            p0=[rp_rs_guess, 0.35, dur_days],
            bounds=([0.005, 0.0, dur_days * 0.6], [0.25, 0.95, dur_days * 1.4]),
            maxfev=5000
        )
        rp_rs_fit, b_fit, dur_fit = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        rp_rs_fit, b_fit, dur_fit = rp_rs_guess, 0.35, dur_days

    rp_earth_fit = star_info["r_star"] * rp_rs_fit * 109.076
    sigma_rp_earth = rp_earth_fit * np.hypot(0.03 / star_info["r_star"], 1.0 / (2.0 * max(master_metrics["snr_red"], 1.0)))

    # Group by Calendar Year (from deduplicated master records)
    years_unique = sorted(list(set(r["year"] for r in master_records)))
    year_stats = {}
    for yr in years_unique:
        yr_recs = [r for r in master_records if r["year"] == yr]
        t_y = np.concatenate([r["t"] for r in yr_recs])
        f_y = np.concatenate([r["f_clean"] for r in yr_recs])
        ferr_y = np.concatenate([r["ferr"] for r in yr_recs])
        cad_y = float(np.median([r["cadence_days"] for r in yr_recs]))
        m_y = compute_transit_metrics(t_y, f_y, ferr_y, p_ref, t0_ref, dur_fit, cad_y)
        year_stats[yr] = {"t": t_y, "f": f_y, "ferr": ferr_y, "cad": cad_y, "sectors": [r["sector"] for r in yr_recs], **m_y}
        summary_rows.append({
            "tic_id": tic_id,
            "group_type": "YEAR_STACK",
            "label": f"YEAR_{yr}",
            "sector": -1,
            "year": yr,
            "author": ",".join(sorted(set(r["author"] for r in yr_recs))),
            "exptime_s": int(round(cad_y * 86400.0)),
            "in_master_stack": True,
            "refined_period_days": round(p_ref, 7),
            "refined_t0_btjd": round(t0_ref, 5),
            **m_y
        })

    # Group by Pipeline Author (across all downloaded records)
    authors_unique = sorted(list(set(r["author"] for r in records)), key=lambda a: AUTHOR_PRIORITY.get(a, 99))
    author_stats = {}
    for auth in authors_unique:
        a_recs = [r for r in records if r["author"] == auth]
        t_a = np.concatenate([r["t"] for r in a_recs])
        f_a = np.concatenate([r["f_clean"] for r in a_recs])
        ferr_a = np.concatenate([r["ferr"] for r in a_recs])
        cad_a = float(np.median([r["cadence_days"] for r in a_recs]))
        m_a = compute_transit_metrics(t_a, f_a, ferr_a, p_ref, t0_ref, dur_fit, cad_a)
        author_stats[auth] = {"t": t_a, "f": f_a, "ferr": ferr_a, "cad": cad_a, "sectors": [r["sector"] for r in a_recs], **m_a}
        summary_rows.append({
            "tic_id": tic_id,
            "group_type": "AUTHOR_STACK",
            "label": f"AUTHOR_{auth}",
            "sector": -1,
            "year": -1,
            "author": auth,
            "exptime_s": int(round(cad_a * 86400.0)),
            "in_master_stack": False,
            "refined_period_days": round(p_ref, 7),
            "refined_t0_btjd": round(t0_ref, 5),
            **m_a
        })

    # Step 6: Generate 6-Panel Publication Poster & Save CSV
    print(f"[6/6] Rendering 6-panel diagnostic poster & saving CSV report...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    win_hrs = min(0.18 * p_ref, 2.5 * dur_fit) * 24.0
    bins_hrs = np.linspace(-win_hrs, win_hrs, 35)
    b_centers = 0.5 * (bins_hrs[:-1] + bins_hrs[1:])

    # Panel 1 (0, 0): Master Multi-Year Coherent Phase-Folded Transit
    ph_m_hrs = ph_master * 24.0
    vis_m = np.abs(ph_m_hrs) <= win_hrs
    axes[0, 0].scatter(ph_m_hrs[vis_m], f_master[vis_m], s=2, color="silver", alpha=0.30, label=f"Cadences ({len(master_records)} Sectors)")
    bm, _, _ = stats.binned_statistic(ph_m_hrs[vis_m], f_master[vis_m], statistic="mean", bins=bins_hrs)
    bs, _, _ = stats.binned_statistic(ph_m_hrs[vis_m], f_master[vis_m], statistic="std", bins=bins_hrs)
    bc, _, _ = stats.binned_statistic(ph_m_hrs[vis_m], f_master[vis_m], statistic="count", bins=bins_hrs)
    vb = bc >= 3
    berr = bs[vb] / np.sqrt(bc[vb])
    axes[0, 0].errorbar(b_centers[vb], bm[vb], yerr=berr, fmt="o", color="black", ms=5, capsize=2.5, lw=1.2, zorder=4, label="Master Binned (+/-1 sigma)")

    ph_grid_d = np.linspace(-win_hrs / 24.0, win_hrs / 24.0, 500)
    axes[0, 0].plot(
        ph_grid_d * 24.0,
        master_model(ph_grid_d, rp_rs_fit, b_fit, dur_fit),
        color="crimson", lw=2.6, zorder=5,
        label=f"300-Ring Fit: Rp={rp_earth_fit:.2f}+/-{sigma_rp_earth:.2f} R_E"
    )
    depth_frac = rp_rs_fit ** 2
    axes[0, 0].set_ylim(
        min(1.0 - max(0.003, 3.5 * depth_frac), float(np.nanmin(bm[vb] - berr)) - 0.0004),
        max(1.0 + max(0.002, 2.0 * depth_frac), float(np.nanmax(bm[vb] + berr)) + 0.0004)
    )
    axes[0, 0].set_xlim(-win_hrs, win_hrs)
    axes[0, 0].set_xlabel("Time from Mid-Transit (hours)")
    axes[0, 0].set_ylabel("Relative Flux")
    axes[0, 0].set_title(
        f"Master Multi-Year Stack: N={master_metrics['n_transits']} Transits | "
        f"SNR_white={master_metrics['snr_white']:.1f} sigma (SNR_red={master_metrics['snr_red']:.1f} sigma)"
    )
    axes[0, 0].legend(loc="lower right", fontsize=9)

    # Panel 2 (0, 1): Secular Stability by Calendar Year
    yr_colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(1, len(years_unique))))
    offset_step = max(0.0022, 2.2 * depth_frac)
    for idx_y, yr in enumerate(years_unique):
        yst = year_stats[yr]
        ph_y_hrs = (((yst["t"] - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref) * 24.0
        vis_y = np.abs(ph_y_hrs) <= win_hrs
        if np.sum(vis_y) < 15:
            continue
        ym, _, _ = stats.binned_statistic(ph_y_hrs[vis_y], yst["f"][vis_y], statistic="mean", bins=bins_hrs)
        yc, _, _ = stats.binned_statistic(ph_y_hrs[vis_y], yst["f"][vis_y], statistic="count", bins=bins_hrs)
        vy = yc >= 2
        v_off = idx_y * offset_step
        axes[0, 1].plot(
            b_centers[vy], ym[vy] - v_off, "o-", color=yr_colors[idx_y], ms=4, lw=1.6,
            label=f"{yr} (S{','.join(str(s) for s in yst['sectors'])}): N={yst['n_transits']}, d={yst['depth_ppm']:.0f}ppm ({yst['snr_white']:.1f} sigma)"
        )
        axes[0, 1].plot(ph_grid_d * 24.0, master_model(ph_grid_d, rp_rs_fit, b_fit, dur_fit) - v_off, color="black", linestyle="--", lw=1.0, alpha=0.5)
    axes[0, 1].axvline(0.0, color="crimson", linestyle=":", alpha=0.6)
    axes[0, 1].set_xlim(-win_hrs, win_hrs)
    axes[0, 1].set_xlabel("Time from Mid-Transit (hours)")
    axes[0, 1].set_ylabel("Relative Flux - Vertical Offset")
    axes[0, 1].set_title(f"Secular Stability Across Mission Years ({min(years_unique)}-{max(years_unique)})")
    axes[0, 1].legend(loc="lower right", fontsize=8.5)

    # Panel 3 (1, 0): Cross-Pipeline De-Blending Comparison (SPOC vs TGLC vs QLP)
    for idx_a, auth in enumerate(authors_unique):
        ast = author_stats[auth]
        ph_a_hrs = (((ast["t"] - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref) * 24.0
        vis_a = np.abs(ph_a_hrs) <= win_hrs
        if np.sum(vis_a) < 15:
            continue
        am, _, _ = stats.binned_statistic(ph_a_hrs[vis_a], ast["f"][vis_a], statistic="mean", bins=bins_hrs)
        ac, _, _ = stats.binned_statistic(ph_a_hrs[vis_a], ast["f"][vis_a], statistic="count", bins=bins_hrs)
        va = ac >= 2
        v_off = idx_a * offset_step
        c_auth = AUTHOR_COLORS.get(auth, "navy")
        axes[1, 0].plot(
            b_centers[va], am[va] - v_off, "s-", color=c_auth, ms=4, lw=1.7,
            label=f"{auth} (S{','.join(str(s) for s in ast['sectors'])}): d={ast['depth_ppm']:.0f}+/-{ast['err_white_ppm']:.0f}ppm ({ast['snr_white']:.1f} sigma)"
        )
        axes[1, 0].plot(ph_grid_d * 24.0, master_model(ph_grid_d, rp_rs_fit, b_fit, dur_fit) - v_off, color="gray", linestyle="--", lw=1.0, alpha=0.6)
    axes[1, 0].axvline(0.0, color="crimson", linestyle=":", alpha=0.6)
    axes[1, 0].set_xlim(-win_hrs, win_hrs)
    axes[1, 0].set_xlabel("Time from Mid-Transit (hours)")
    axes[1, 0].set_ylabel("Relative Flux - Vertical Offset")
    axes[1, 0].set_title("Cross-Author De-Blending Test (SPOC Aperture vs QLP vs TGLC PSF)")
    axes[1, 0].legend(loc="lower right", fontsize=8.5)

    # Panel 4 (1, 1): Ultra-Fine Multi-Year Ephemeris Periodogram
    dp_ppm_grid = ((p_grid - p_ref) / p_ref) * 1e6
    axes[1, 1].plot(dp_ppm_grid, snr_grid, color="black", lw=1.0)
    axes[1, 1].axvline(0.0, color="crimson", linestyle="--", lw=1.8, label=f"Locked P = {p_ref:.7f} d")
    axes[1, 1].axvline(((p_init - p_ref) / p_ref) * 1e6, color="royalblue", linestyle=":", lw=1.5, label=f"Input P = {p_init:.6f} d")
    axes[1, 1].set_xlabel(f"Relative Period Offset (P - P_ref) / P_ref [ppm] around {p_ref:.7f} d")
    axes[1, 1].set_ylabel("Coherent Multi-Year Matched-Filter SNR (sigma)")
    axes[1, 1].set_title("Multi-Year Ephemeris Locking Periodogram")
    axes[1, 1].legend(loc="upper right", fontsize=9)

    # Panel 5 (2, 0): Sector-by-Sector Transit Depth Consistency
    sec_labels = [f"S{r['sector']}\n({r['author']}\n{r['year']})" for r in records]
    sec_depths = [r["metrics"]["depth_ppm"] for r in records]
    sec_errs = [r["metrics"]["err_white_ppm"] for r in records]
    x_pos = np.arange(len(records))
    bar_colors = [AUTHOR_COLORS.get(r["author"], "steelblue") for r in records]
    axes[2, 0].errorbar(x_pos, sec_depths, yerr=sec_errs, fmt="o", color="black", capsize=4, lw=1.5, zorder=4)
    axes[2, 0].bar(x_pos, sec_depths, width=0.55, color=bar_colors, alpha=0.65, edgecolor="black", zorder=3)
    axes[2, 0].axhline(master_metrics["depth_ppm"], color="crimson", linestyle="--", lw=2.0, label=f"Master Depth: {master_metrics['depth_ppm']:.0f} ppm")
    axes[2, 0].axhline(0.0, color="black", linestyle=":", lw=1.0)
    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].set_xticklabels(sec_labels, fontsize=8.5)
    axes[2, 0].set_ylabel("Measured Transit Depth (ppm)")
    axes[2, 0].set_title("Sector-by-Sector & Pipeline Depth Stability")
    axes[2, 0].legend(loc="upper right", fontsize=9)

    # Panel 6 (2, 1): Antares Quantitative Summary Dossier
    axes[2, 1].axis("off")
    n_pos_years = sum(1 for y in years_unique if year_stats[y]["snr_white"] >= 2.0)
    verdict = (
        "CONFIRMED MULTI-EPOCH PLANET CANDIDATE"
        if (master_metrics["snr_red"] >= 7.1 and n_pos_years >= 2)
        else ("PERSISTENT MULTI-SECTOR CANDIDATE (Grey Zone 5-7 sigma)" if master_metrics["snr_red"] >= 5.0 else "SUSPECT / MARGINAL SNR")
    )
    tglc_str = (
        f"{author_stats['TGLC']['depth_ppm']:.0f} +/- {author_stats['TGLC']['err_white_ppm']:.0f} ppm ({author_stats['TGLC']['snr_white']:.1f} sigma)"
        if "TGLC" in author_stats else "Not available on MAST"
    )
    qlp_str = (
        f"{author_stats['QLP']['depth_ppm']:.0f} +/- {author_stats['QLP']['err_white_ppm']:.0f} ppm ({author_stats['QLP']['snr_white']:.1f} sigma)"
        if "QLP" in author_stats else "Not available on MAST"
    )
    spoc_str = (
        f"{author_stats['SPOC']['depth_ppm']:.0f} +/- {author_stats['SPOC']['err_white_ppm']:.0f} ppm ({author_stats['SPOC']['snr_white']:.1f} sigma)"
        if "SPOC" in author_stats else "Not available on MAST"
    )

    card_text = (
        f"ANTARES MULTI-YEAR DOSSIER: TIC {tic_id}\n"
        f"{'-'*52}\n"
        f"Verdict:             {verdict}\n"
        f"Unique Sectors:      {len(master_records)} ({','.join(str(r['sector']) for r in master_records)})\n"
        f"Mission Span:        {min(years_unique)} - {max(years_unique)} ({t_master[-1]-t_master[0]:.1f} days)\n"
        f"Total Transits:      N = {master_metrics['n_transits']} unique events ({master_metrics['n_cadences_in']} in-transit pts)\n\n"
        f"COHERENT EPHEMERIS & PHYSICAL FIT (Conditional on Planet):\n"
        f"  Refined Period:    {p_ref:.7f} d (dP = {dp_sec:+.2f} s)\n"
        f"  Refined T0:        {t0_ref:.5f} BTJD\n"
        f"  Master Depth:      {master_metrics['depth_ppm']:.1f} +/- {master_metrics['err_white_ppm']:.1f} ppm\n"
        f"  Integrated SNR:    {master_metrics['snr_white']:.2f} sigma (white) | {master_metrics['snr_red']:.2f} sigma (red-corrected, beta={master_metrics['beta_red']:.2f})\n"
        f"  Fitted Radius:     Rp = {rp_earth_fit:.2f} +/- {sigma_rp_earth:.2f} R_Earth (b = {b_fit:.2f})\n\n"
        f"CROSS-PIPELINE DE-BLENDING & SECULAR CHECK:\n"
        f"  SPOC (2-min):      {spoc_str}\n"
        f"  QLP  (FFI):        {qlp_str}\n"
        f"  TGLC (PSF Gaia):   {tglc_str}\n"
        f"  Years w/ SNR>=2 sigma:  {n_pos_years} / {len(years_unique)} observed years"
    )
    axes[2, 1].text(
        0.02, 0.96, card_text, transform=axes[2, 1].transAxes, fontsize=10.0,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f6f9", edgecolor="gray")
    )

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"antares_TIC_{tic_id}.png")
    out_csv = os.path.join(RESULTS_DIR, f"antares_summary_{tic_id}.csv")
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    print("\n" + "=" * 76)
    print(f"ANTARES SUMMARY FOR TIC {tic_id}:")
    print(f"  -> Unique Sectors Stitched: {len(master_records)} | Total Unique Transits: {master_metrics['n_transits']}")
    print(f"  -> Master Depth: {master_metrics['depth_ppm']:.1f} +/- {master_metrics['err_white_ppm']:.1f} ppm")
    print(f"  -> Master SNR (White): {master_metrics['snr_white']:.2f} sigma | Master SNR (Red-Noise Corrected): {master_metrics['snr_red']:.2f} sigma")
    print(f"  -> Fitted Radius (with photometric error): Rp = {rp_earth_fit:.2f} +/- {sigma_rp_earth:.2f} R_Earth")
    print(f"  -> Saved Diagnostic Poster: {out_png}")
    print(f"  -> Saved Telemetry Table:   {out_csv}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antares: Multi-Year Archival SPOC/QLP/TGLC Exoplanet Vetter")
    parser.add_argument("--tic", type=str, default="46314221", help="Target TIC ID (default: 46314221)")
    parser.add_argument("--period", type=float, default=0.925309, help="Orbital period in days (default: 0.925309)")
    parser.add_argument("--t0", type=float, default=3988.66325, help="Reference mid-transit epoch in BTJD (default: 3988.66325)")
    parser.add_argument("--dur", type=float, default=1.16, help="Transit duration in hours (default: 1.16)")
    args = parser.parse_args()

    run_antares_detective(tic_id=args.tic.strip(), p_init=args.period, t0_init=args.t0, dur_hr=args.dur)
