import os
import io
import ssl
import urllib.parse
import urllib.request
import warnings
import numpy as np
import pandas as pd
import requests
import urllib3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from astroquery.mast import Catalogs

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_CSV = os.path.join("tess_results", "candidates_report.csv")
VET_DIR = os.path.join("tess_results", "deep_vetting")
os.makedirs(VET_DIR, exist_ok=True)


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


def safe_float(val, default: float) -> float:
    """Safely parses float values from pandas rows, handling NaN, None, and 'N/A' strings."""
    try:
        if pd.isna(val) or str(val).strip().upper() in ("", "N/A", "NAN", "NONE"):
            return default
        f_val = float(val)
        return f_val if np.isfinite(f_val) else default
    except Exception:
        return default


def fetch_csv_no_ssl(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, verify=False, timeout=45)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), comment="#", on_bad_lines="skip")


def calc_max_keplerian_duration_hours(p_days: float, r_star_sun: float, m_star_sun: float, rp_rs: float) -> float:
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    
    p_sec = p_days * 86400.0
    m_kg = max(0.08, m_star_sun) * M_sun
    r_m = max(0.10, r_star_sun) * R_sun
    
    a_m = (G * m_kg * (p_sec ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    ratio = (r_m * (1.0 + rp_rs)) / a_m
    if ratio >= 1.0:
        return p_days * 24.0 * 0.25
    t14_sec = (p_sec / np.pi) * np.arcsin(ratio)
    return float(t14_sec / 3600.0)


def tukey_biweight_detrend(t: np.ndarray, f: np.ndarray, window_days: float = 0.75, mask: np.ndarray = None) -> np.ndarray:
    """Wotan-style Tukey's biweight time-windowed slider."""
    if mask is None:
        mask = np.zeros_like(t, dtype=bool)
    t_unmasked = t[~mask]
    f_unmasked = f[~mask]
    
    half_win = 0.5 * window_days
    c = 6.0
    
    step = max(1, len(t) // 400)
    t_grid = t[::step]
    trend_grid = np.ones_like(t_grid)
    
    idx_left = np.searchsorted(t_unmasked, t_grid - half_win, side="left")
    idx_right = np.searchsorted(t_unmasked, t_grid + half_win, side="right")
    
    for i, (l_idx, r_idx) in enumerate(zip(idx_left, idx_right)):
        if r_idx - l_idx < 5:
            trend_grid[i] = np.nanmedian(f_unmasked)
            continue
        seg = f_unmasked[l_idx:r_idx]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        if mad < 1e-12:
            trend_grid[i] = med
            continue
        for _ in range(2):
            u = (seg - med) / (c * mad)
            valid_u = np.abs(u) < 1.0
            if not np.any(valid_u):
                break
            w = (1.0 - u[valid_u] ** 2) ** 2
            med = np.sum(w * seg[valid_u]) / np.sum(w)
        trend_grid[i] = med
        
    trend = np.interp(t, t_grid, trend_grid)
    return f / trend


def exact_annulus_transit_model(phase: np.ndarray, rp_rs: float, b: float, dur_total: float, u1: float, u2: float) -> np.ndarray:
    """
    High-precision Mandel-Agol transit model using 60-ring radial annulus integration
    across the quadratic limb-darkened stellar disk. Accurate for both small and giant planets.
    """
    rp_rs = float(np.clip(rp_rs, 0.001, 0.35))
    b = float(np.clip(b, 0.0, 0.999))
    dur_total = max(float(dur_total), 0.005)
    
    half_chord = np.sqrt(max(1e-6, (1.0 + rp_rs) ** 2 - b ** 2))
    x_proj = (phase / (0.5 * dur_total)) * half_chord
    z = np.hypot(b, x_proj)
    
    flux = np.ones_like(phase, dtype=np.float64)
    in_disk = z < (1.0 + rp_rs)
    if not np.any(in_disk):
        return flux
        
    n_rings = 60
    r_edges = np.linspace(0.0, 1.0, n_rings + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    ring_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    
    mu = np.sqrt(np.clip(1.0 - r_centers ** 2, 0.0, 1.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    ring_flux = ring_areas * intensity
    total_stellar_flux = np.sum(ring_flux)
    
    z_in = z[in_disk][:, None]
    r_c = r_centers[None, :]
    
    blocked_frac = np.zeros((z_in.shape[0], n_rings), dtype=np.float64)
    full_cov = (z_in + r_c) <= rp_rs
    blocked_frac[full_cov] = 1.0
    
    part_cov = (~full_cov) & (np.abs(z_in - r_c) < rp_rs) & (z_in > 0)
    if np.any(part_cov):
        z_p = np.broadcast_to(z_in, blocked_frac.shape)[part_cov]
        r_p = np.broadcast_to(r_c, blocked_frac.shape)[part_cov]
        cos_alpha = (z_p ** 2 + r_p ** 2 - rp_rs ** 2) / (2.0 * z_p * r_p)
        alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
        blocked_frac[part_cov] = alpha / np.pi
        
    occulted_flux = np.sum(blocked_frac * ring_flux[None, :], axis=1) / total_stellar_flux
    flux[in_disk] = 1.0 - occulted_flux
    return flux


def query_gaia_contamination(tic_id: str, depth_ppm: float) -> dict:
    info = {"n_neighbors": 0, "contamination_ratio": 0.0, "neb_capable_stars": 0, "target_gmag": np.nan, "ruwe": np.nan}
    try:
        tic_data = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
        if len(tic_data) == 0:
            return info
        row = tic_data[0]
        ra = float(to_numpy(row["ra"]))
        dec = float(to_numpy(row["dec"]))
        
        adql = (
            f"SELECT TOP 100 ra, dec, phot_g_mean_mag, ruwe, "
            f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec}))*3600.0 AS sep_arcsec "
            f"FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.03333)) "
            f"ORDER BY sep_arcsec ASC"
        )
        gaia_url = (
            "https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery"
            f"&LANG=ADQL&FORMAT=csv&QUERY={urllib.parse.quote(adql)}"
        )
        df_g = fetch_csv_no_ssl(gaia_url)
        if len(df_g) == 0:
            return info
            
        target_row = df_g.iloc[0]
        g_target = float(target_row["phot_g_mean_mag"])
        info["target_gmag"] = g_target
        info["ruwe"] = safe_float(target_row.get("ruwe"), np.nan)
        
        if len(df_g) == 1:
            return info
            
        neighbors = df_g.iloc[1:].dropna(subset=["phot_g_mean_mag", "sep_arcsec"])
        info["n_neighbors"] = len(neighbors)
        
        sigma_psf = 21.0
        rel_flux_total = 0.0
        neb_capable = 0
        obs_frac_depth = depth_ppm * 1e-6
        
        for _, nrow in neighbors.iterrows():
            dg = float(nrow["phot_g_mean_mag"]) - g_target
            sep = float(nrow["sep_arcsec"])
            flux_ratio = (10.0 ** (-0.4 * dg)) * np.exp(-0.5 * (sep / sigma_psf) ** 2)
            rel_flux_total += flux_ratio
            if 0.50 * (flux_ratio / (1.0 + flux_ratio)) >= obs_frac_depth:
                neb_capable += 1
                
        info["contamination_ratio"] = rel_flux_total
        info["neb_capable_stars"] = neb_capable
    except Exception:
        pass
    return info


def deep_vet_target(cand_row: dict) -> dict | None:
    tic_id = str(int(cand_row["tic_id"]))
    p_init = safe_float(cand_row.get("period_days"), 5.0)
    t0_init = safe_float(cand_row.get("t0_btjd"), 3500.0)
    dur_init = safe_float(cand_row.get("duration_hours"), 2.5) / 24.0
    r_star = safe_float(cand_row.get("r_star_sun"), 1.0)
    m_star = safe_float(cand_row.get("m_star_sun"), r_star ** 1.1)
    teff = safe_float(cand_row.get("teff_k"), 5500.0)
    depth_init_ppm = safe_float(cand_row.get("depth_ppm"), 1000.0)
    
    print(f"\n======================================================================")
    print(f"DEEP SCIENTIFIC VETTING v2.0: TIC {tic_id} (Initial P = {p_init:.5f} d)")
    print(f"======================================================================")
    
    search_res = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
    if len(search_res) == 0:
        print(f"[!] No SPOC lightcurves found for TIC {tic_id}")
        return None
        
    use_search = search_res[-4:]
    sectors_used = [int(str(s).split()[-1]) for s in use_search.mission]
    print(f"[1/4] Downloading & stitching Sectors {sectors_used} with Tukey Biweight detrender...")
    
    t_all, f_all = [], []
    for s_item in use_search:
        lc = s_item.download()
        t_sec = to_numpy(lc.time)
        f_sec = to_numpy(lc.flux)
        qual = np.asarray(lc.quality, dtype=int) if (hasattr(lc, "quality") and lc.quality is not None) else np.zeros_like(t_sec, dtype=int)
        valid = np.isfinite(t_sec) & np.isfinite(f_sec) & (f_sec > 0) & (qual == 0)
        t_sec, f_sec = t_sec[valid], f_sec[valid]
        f_sec = f_sec / np.nanmedian(f_sec)
        
        phase_sec = ((t_sec - t0_init + 0.5 * p_init) % p_init) - 0.5 * p_init
        tr_mask = np.abs(phase_sec) < (0.65 * dur_init)
        f_flat = tukey_biweight_detrend(t_sec, f_sec, window_days=0.75, mask=tr_mask)
        
        t_all.append(t_sec)
        f_all.append(f_flat)
        
    t = np.concatenate(t_all)
    f = np.concatenate(f_all)
    sort_idx = np.argsort(t)
    t, f = t[sort_idx], f[sort_idx]
    
    fine_p_grid = np.linspace(p_init * 0.995, p_init * 1.005, 6000)
    bls = BoxLeastSquares(t, f)
    bls_res = bls.power(fine_p_grid, np.linspace(max(0.025, dur_init * 0.7), min(0.25, dur_init * 1.3), 10))
    best_idx = int(np.nanargmax(bls_res.power))
    p_ref = float(bls_res.period[best_idx])
    t0_ref = float(bls_res.transit_time[best_idx])
    dur_ref = float(bls_res.duration[best_idx])
    
    phase = ((t - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref
    
    print(f"[2/4] Fitting 60-Ring Annular Limb-Darkened Transit Model (Teff={teff:.0f} K, R*={r_star:.2f} R_Sun)...")
    u1_est = float(np.clip(0.45 - 0.00008 * (teff - 5500.0), 0.15, 0.70))
    u2_est = float(np.clip(0.25 + 0.00003 * (teff - 5500.0), 0.05, 0.45))
    
    fit_win = np.abs(phase) <= (2.5 * dur_ref)
    ph_fit = phase[fit_win]
    f_fit = f[fit_win]
    
    rp_rs_guess = float(np.sqrt(max(1e-5, depth_init_ppm * 1e-6)))
    max_kep_dur_days = calc_max_keplerian_duration_hours(p_ref, r_star, m_star, rp_rs_guess) / 24.0
    upper_dur_bound = max(dur_ref * 0.7, min(dur_ref * 1.5, 1.25 * max_kep_dur_days))
    
    def model_to_fit(ph, rp_rs, b, dur_t):
        return exact_annulus_transit_model(ph, rp_rs, b, dur_t, u1_est, u2_est)
        
    try:
        p0_dur = np.clip(dur_ref, dur_ref * 0.6, upper_dur_bound * 0.98)
        popt, _ = curve_fit(
            model_to_fit,
            ph_fit,
            f_fit,
            p0=[rp_rs_guess, 0.3, p0_dur],
            bounds=([0.002, 0.0, dur_ref * 0.5], [0.30, 0.98, upper_dur_bound]),
            maxfev=6000
        )
        rp_rs_fit, b_fit, dur_fit = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        rp_rs_fit, b_fit, dur_fit = rp_rs_guess, 0.5, dur_ref
        
    true_rp_earth = r_star * rp_rs_fit * 109.076
    dur_ratio_kepler = (dur_fit * 24.0) / max(max_kep_dur_days * 24.0, 0.1)
    
    print(f"[3/4] Running Secondary Eclipse & Phase Curve test at Phase = 0.5...")
    phase_0to1 = ((t - t0_ref) % p_ref) / p_ref
    sec_mask = np.abs(phase_0to1 - 0.5) < (0.5 * dur_fit / p_ref)
    oot_sec_mask = (np.abs(phase_0to1 - 0.5) > (0.8 * dur_fit / p_ref)) & (np.abs(phase_0to1 - 0.5) < (2.5 * dur_fit / p_ref))
    
    if np.sum(sec_mask) > 5 and np.sum(oot_sec_mask) > 10:
        sec_depth_ppm = float((np.nanmedian(f[oot_sec_mask]) - np.nanmean(f[sec_mask])) * 1e6)
        sec_err_ppm = float(np.nanstd(f[oot_sec_mask]) / np.sqrt(np.sum(sec_mask)) * 1e6)
        sec_snr = sec_depth_ppm / max(sec_err_ppm, 1.0)
    else:
        sec_depth_ppm, sec_snr = 0.0, 0.0
        
    print(f"[4/4] Querying Gaia DR3 neighborhood (r = 120\") & computing Heuristic FP Risk Score...")
    gaia_contam = query_gaia_contamination(tic_id, depth_init_ppm)
    
    # Composite Heuristic False-Positive Risk Score (0.0% = clean, 100% = definite FP)
    risk_score = 0.2
    if b_fit > 0.75:
        risk_score += 25.0 * ((b_fit - 0.75) / 0.25) ** 2
    if sec_snr > 2.2:
        risk_score += 45.0 * (sec_snr / 3.0)
    if gaia_contam["neb_capable_stars"] > 0:
        risk_score += 8.0 * gaia_contam["neb_capable_stars"]
    if gaia_contam["contamination_ratio"] > 0.25:
        risk_score += 35.0 * gaia_contam["contamination_ratio"]
    if dur_ratio_kepler > 1.20:
        risk_score += 40.0 * (dur_ratio_kepler - 1.0)
    if np.isfinite(gaia_contam["ruwe"]) and gaia_contam["ruwe"] > 1.25:
        risk_score += 15.0
    risk_pct = float(np.clip(risk_score, 0.1, 99.9))
    
    is_clean = (risk_pct < 5.0 and b_fit < 0.82 and sec_snr < 2.5 and dur_ratio_kepler <= 1.25)
    disposition = "TIER-2 VETTED CANDIDATE (Low FP Risk)" if is_clean else "SUSPECT / GRAZING OR CONTAMINATED"
    
    print(f"      -> Fitted Radius: Rp = {true_rp_earth:.2f} R_Earth (Rp/R* = {rp_rs_fit:.4f})")
    print(f"      -> Impact Parameter: b = {b_fit:.3f} | Duration / Max_Kepler = {dur_ratio_kepler:.2f}")
    print(f"      -> Secondary Eclipse at Phase 0.5: Depth = {sec_depth_ppm:.0f} ppm (SNR = {sec_snr:.2f} sigma)")
    print(f"      -> Gaia DR3 Neighbors (120\"): {gaia_contam['n_neighbors']} stars | Flux Contamination = {gaia_contam['contamination_ratio']*100:.2f}%")
    print(f"      -> Heuristic FP Risk Score: {risk_pct:.1f}% -> {disposition}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    ph_grid = np.linspace(-2.2 * dur_fit, 2.2 * dur_fit, 500)
    axes[0, 0].scatter(ph_fit * 24.0, f_fit, s=3, color="silver", alpha=0.4, label="Tukey Biweight Cadences")
    bins = np.linspace(-2.2 * dur_fit, 2.2 * dur_fit, 35)
    b_means, b_edges, _ = stats.binned_statistic(ph_fit, f_fit, statistic="mean", bins=bins)
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    axes[0, 0].scatter(b_centers * 24.0, b_means, s=28, color="black", zorder=4, label="Binned Photometry")
    axes[0, 0].plot(
        ph_grid * 24.0,
        model_to_fit(ph_grid, rp_rs_fit, b_fit, dur_fit),
        color="crimson",
        lw=2.5,
        zorder=5,
        label=f"60-Ring Fit (b={b_fit:.2f}, Rp={true_rp_earth:.2f} R_E)"
    )
    axes[0, 0].set_xlabel("Time from Mid-Transit (hours)")
    axes[0, 0].set_ylabel("Relative Flux")
    axes[0, 0].set_title(f"TIC {tic_id}: Annular Limb-Darkened Transit Model (Sectors {sectors_used})")
    axes[0, 0].legend(loc="lower right")
    
    sec_win = np.abs(phase_0to1 - 0.5) <= min(0.12, 3.0 * dur_fit / p_ref)
    ph_sec_hrs = (phase_0to1[sec_win] - 0.5) * p_ref * 24.0
    axes[0, 1].scatter(ph_sec_hrs, f[sec_win], s=3, color="silver", alpha=0.4)
    if len(ph_sec_hrs) > 20:
        s_bins = np.linspace(np.min(ph_sec_hrs), np.max(ph_sec_hrs), 25)
        s_means, s_edges, _ = stats.binned_statistic(ph_sec_hrs, f[sec_win], statistic="mean", bins=s_bins)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        axes[0, 1].plot(s_centers, s_means, color="royalblue", lw=2, label=f"Phase 0.5 Binned (SNR={sec_snr:.1f} sigma)")
    axes[0, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[0, 1].set_xlabel("Time from Secondary Phase 0.5 (hours)")
    axes[0, 1].set_ylabel("Relative Flux")
    axes[0, 1].set_title(f"Secondary Eclipse Test (Self-Luminous EB Check: {sec_depth_ppm:.0f} ppm)")
    axes[0, 1].legend(loc="lower right")
    
    orbit_nums = np.round((t - t0_ref) / p_ref).astype(int)
    in_tr_orbits = np.unique(orbit_nums[np.abs(phase) < (0.4 * dur_fit)])
    show_orbits = in_tr_orbits[:8]
    offset_step = max(0.003, 2.2 * (rp_rs_fit ** 2))
    for k, orb in enumerate(show_orbits):
        m_orb = (orbit_nums == orb) & (np.abs(phase) < (1.8 * dur_fit))
        if np.sum(m_orb) > 4:
            axes[1, 0].plot(phase[m_orb] * 24.0, f[m_orb] - k * offset_step, "o-", ms=2.5, lw=1, alpha=0.8, label=f"Epoch #{orb}")
    axes[1, 0].axvline(0.0, color="red", linestyle=":", alpha=0.6)
    axes[1, 0].set_xlabel("Time from Mid-Transit (hours)")
    axes[1, 0].set_ylabel("Flux + Vertical Offset")
    axes[1, 0].set_title(f"Individual Transit Consistency ({len(in_tr_orbits)} Total Transits)")
    
    axes[1, 1].axis("off")
    ruwe_str = f"{gaia_contam['ruwe']:.3f}" if np.isfinite(gaia_contam["ruwe"]) else "N/A"
    summary_text = (
        f"SCIENTIFIC VETTING DOSSIER: TIC {tic_id}\n"
        f"{'-'*48}\n"
        f"Disposition:       {disposition}\n"
        f"Heuristic FP Risk: {risk_pct:.1f}% (Clean Threshold < 5.0%)\n\n"
        f"ORBITAL & PHYSICAL PARAMETERS (60-Ring Fit):\n"
        f"  Refined Period:  {p_ref:.6f} days\n"
        f"  Mid-Transit T0:  {t0_ref:.5f} BTJD\n"
        f"  Transit Dur:     {dur_fit*24.0:.2f} h (Max Keplerian: {max_kep_dur_days*24.0:.2f} h)\n"
        f"  Radius Ratio:    Rp/R* = {rp_rs_fit:.4f}\n"
        f"  Planet Radius:   Rp = {true_rp_earth:.2f} R_Earth\n"
        f"  Impact Param:    b = {b_fit:.3f} (0=central, >0.82=grazing)\n"
        f"  Limb Darkening:  u1 = {u1_est:.2f}, u2 = {u2_est:.2f} (Teff={teff:.0f} K)\n\n"
        f"FALSE POSITIVE FALSIFICATION TESTS:\n"
        f"  Secondary Ecl:   {sec_depth_ppm:.0f} ppm ({sec_snr:.2f} sigma at Phase 0.5)\n"
        f"  Odd/Even Diff:   {cand_row.get('odd_even_sigma', 'N/A')} sigma\n"
        f"  Centroid Shift:  {cand_row.get('centroid_shift_arcsec', 'N/A')} arcsec\n"
        f"  Gaia DR3 RUWE:   {ruwe_str}\n"
        f"  Gaia Neighbors:  {gaia_contam['n_neighbors']} within 120\" (Contam: {gaia_contam['contamination_ratio']*100:.2f}%)\n"
        f"  NEB-Capable:     {gaia_contam['neb_capable_stars']} nearby stars"
    )
    axes[1, 1].text(0.03, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=10.2,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f6f9", edgecolor="gray"))
    
    plt.tight_layout()
    out_img = os.path.join(VET_DIR, f"dossier_TIC_{tic_id}.png")
    plt.savefig(out_img, dpi=160)
    plt.close(fig)
    
    return {
        "tic_id": tic_id,
        "disposition": disposition,
        "fp_risk_percent": round(risk_pct, 1),
        "refined_period_days": round(p_ref, 6),
        "fitted_rp_earth": round(true_rp_earth, 2),
        "impact_param_b": round(b_fit, 3),
        "dur_to_kepler_max": round(dur_ratio_kepler, 2),
        "secondary_snr_sigma": round(sec_snr, 2),
        "flux_contam_percent": round(gaia_contam["contamination_ratio"] * 100, 2),
        "neb_capable_neighbors": gaia_contam["neb_capable_stars"],
        "sectors_stitched": ",".join(str(s) for s in sectors_used)
    }


if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"File '{INPUT_CSV}' not found. Run aldebaran.py first.")
    else:
        df_cands = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df_cands)} candidates from '{INPUT_CSV}' for Tier-2 Deep Vetting v2.0.")
        vet_results = []
        for _, row in df_cands.iterrows():
            try:
                res = deep_vet_target(row.to_dict())
                if res is not None:
                    vet_results.append(res)
            except Exception as e:
                print(f"[!] Error vetting TIC {row['tic_id']}: {e}")
        if vet_results:
            out_csv = os.path.join(VET_DIR, "scientific_dossier_summary.csv")
            pd.DataFrame(vet_results).to_csv(out_csv, index=False)
            print(f"\nDeep vetting complete! Dossiers saved to '{VET_DIR}'.")