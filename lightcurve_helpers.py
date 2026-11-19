"""
lightcurve_helpers.py - light-curve helpers shared by Antares (Tier 4) and the
Antares packager (Stage 3B), and to_numpy and the Keplerian duration bounds
shared with Aldebaran (Tier 1) and Pleiades (Tier 2), moved unchanged from
aldebaran.py in RC2-FREEZE-01.

Roadmap C9 (S3.5): each function had equivalent copies in several modules. The
bodies below were moved unchanged from antares.py at 0ca167abe3; the importing
modules re-export the same names, so their module attributes are unchanged.
"""
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

from ephemeris_contract import fold_phase_centered, compute_orbit_number, days_to_hours


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


def extract_multi_author_lc(lc_obj, author: str, edge_trim_days: float = 0.08,
                            raise_errors: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Extracts time, normalized flux, flux error, and cadence (days) from SPOC, TESS-SPOC, QLP, or TGLC
    light curves. Prioritizes PSF-deblended columns in TGLC and KSP detrended columns in QLP.

    Returns None when fewer than 80 valid points remain. With raise_errors=True an unreadable product
    raises instead of returning None, so the caller can record why it failed (C1g).
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
        if raise_errors:
            raise
        return None


def light_curve_product_failure(sector, author, stage: str, reason) -> dict:
    """One archival light-curve product that could not be used, recorded instead of dropped (C1g).

    stage is PRODUCT_METADATA, DOWNLOAD, EXTRACTION or DETRENDING; reason is an exception or a text.
    """
    if isinstance(reason, BaseException):
        reason = f"{type(reason).__name__}: {reason}"
    return {"sector": int(sector), "author": str(author), "stage": stage, "reason": str(reason)[:150]}


EXTRACTION_TOO_FEW_POINTS = "fewer than 80 valid points after quality and flux cuts"


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


# Pre-computed 5-point Gauss-Legendre quadrature on [-0.5, +0.5] for 300-ring sub-sampling
_GL5_NODES, _GL5_WEIGHTS = np.polynomial.legendre.leggauss(5)
_GL5_NODES = 0.5 * _GL5_NODES
_GL5_WEIGHTS = 0.5 * _GL5_WEIGHTS


def annulus_300_transit_model(
    phase_days: np.ndarray,
    rp_rs: float,
    b: float,
    dur_days: float,
    u1: float,
    u2: float,
    exp_time_days: float = 0.0,
    n_rings: int = 300
) -> np.ndarray:
    """
    300-ring limb-darkened transit model using 5-point Gauss-Legendre radial sub-quadrature
    (< 3.1 ppm max error vs analytical Mandel & Agol 2002) with Kipping (2010) 7-point
    numerical supersampling to account for finite exposure time smearing in 30-min / 10-min FFI light curves.
    """
    rp_rs = float(np.clip(rp_rs, 0.001, 0.35))
    b = float(np.clip(b, 0.0, 0.995))
    dur_days = max(float(dur_days), 0.005)

    r_edges = np.linspace(0.0, 1.0, n_rings + 1)
    dr = 1.0 / n_rings
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    r_sub = (r_mid[:, None] + _GL5_NODES[None, :] * dr).ravel()
    w_sub = (2.0 * np.pi * r_sub * dr) * np.tile(_GL5_WEIGHTS, n_rings)
    mu = np.sqrt(np.clip(1.0 - r_sub ** 2, 0.0, 1.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    sub_flux = w_sub * intensity
    total_flux = np.sum(sub_flux)

    r_c = r_sub[None, :]
    half_chord = np.sqrt(max(1e-6, (1.0 + rp_rs) ** 2 - b ** 2))

    def eval_instantaneous(ph_arr: np.ndarray) -> np.ndarray:
        x_proj = (ph_arr / (0.5 * dur_days)) * half_chord
        z = np.hypot(b, x_proj)
        fl = np.ones_like(ph_arr, dtype=np.float64)
        in_d = z < (1.0 + rp_rs)
        if not np.any(in_d):
            return fl
        z_in = z[in_d][:, None]
        blocked = np.zeros((z_in.shape[0], len(r_sub)), dtype=np.float64)
        full_c = (z_in + r_c) <= rp_rs
        blocked[full_c] = 1.0
        part_c = (~full_c) & (np.abs(z_in - r_c) < rp_rs) & (z_in > 0)
        if np.any(part_c):
            zp = np.broadcast_to(z_in, blocked.shape)[part_c]
            rp = np.broadcast_to(r_c, blocked.shape)[part_c]
            cos_a = (zp ** 2 + rp ** 2 - rp_rs ** 2) / (2.0 * zp * rp)
            blocked[part_c] = np.arccos(np.clip(cos_a, -1.0, 1.0)) / np.pi
        fl[in_d] = 1.0 - (np.sum(blocked * sub_flux[None, :], axis=1) / total_flux)
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
        ph = fold_phase_centered(t_all, t0_init, p_try, in_days=True)
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
        ph = fold_phase_centered(t_all, t0_try, p_refined, in_days=True)
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
    ph = fold_phase_centered(t, t0_val, p_val, in_days=True)
    in_tr = np.abs(ph) < (0.45 * eff_dur)
    out_tr = (np.abs(ph) > (0.70 * (dur_days + cadence_days))) & (np.abs(ph) < min(0.25 * p_val, 3.5 * dur_days))

    n_pts_in = int(np.sum(in_tr))
    if n_pts_in < 3 or np.sum(out_tr) < 10:
        return {"n_transits": 0, "n_cadences_in": n_pts_in, "depth_ppm": np.nan, "err_white_ppm": np.nan, "snr_white": 0.0, "snr_red": 0.0, "beta_red": 1.0}

    orbit_nums = compute_orbit_number(t, t0_val, p_val)
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


def calc_max_keplerian_duration_hours(p_days: float, r_star_sun: float, m_star_sun: float, rp_rs: float) -> float:
    """Calculates theoretical maximum central transit duration (b=0, circular orbit) from Kepler's 3rd Law."""
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    
    p_sec = p_days * 86400.0
    m_kg = max(0.08, m_star_sun) * M_sun
    r_m = max(0.10, r_star_sun) * R_sun
    
    a_m = (G * m_kg * (p_sec ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    ratio = (r_m * (1.0 + rp_rs)) / a_m
    if ratio >= 1.0:
        return days_to_hours(p_days) * 0.25
    t14_sec = (p_sec / np.pi) * np.arcsin(ratio)
    return float(t14_sec / 3600.0)


def compute_keplerian_duration_bounds_hours(
    p_days: float, r_star: float, e_r_star: float, m_star: float, e_m_star: float, rp_rs: float
) -> tuple[float, float, float]:
    """
    Returns (nominal_circular_max_hr, strict_lower_bound_hr, eccentric_upper_bound_hr)
    accounting for 1-sigma TIC stellar uncertainties (e_rad, e_mass) and e <= 0.30 eccentricity.
    """
    nom_max_hr = calc_max_keplerian_duration_hours(p_days, r_star, m_star, rp_rs)
    
    r_upper = r_star + max(e_r_star, 0.08 * r_star)
    m_lower = max(0.08, m_star - max(e_m_star, 0.10 * m_star))
    upper_circ_hr = calc_max_keplerian_duration_hours(p_days, r_upper, m_lower, rp_rs)
    
    r_lower = max(0.08, r_star - max(e_r_star, 0.08 * r_star))
    m_upper = m_star + max(e_m_star, 0.10 * m_star)
    lower_circ_hr = calc_max_keplerian_duration_hours(p_days, r_lower, m_upper, rp_rs)
    
    # 0.35x lower bound (grazing b <= 0.93 + periastron e <= 0.30)
    # 1.38x upper bound on 1-sigma upper stellar radius (apastron e <= 0.30)
    return nom_max_hr, 0.35 * lower_circ_hr, 1.38 * upper_circ_hr
