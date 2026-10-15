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
from scipy.ndimage import median_filter
from astropy.timeseries import BoxLeastSquares
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import lightkurve as lk
from astroquery.mast import Catalogs

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

PIPELINE_VERSION = "v2.5"
if os.environ.get("LIGHTKURVE_CACHE_DIR"):
    os.makedirs(os.environ["LIGHTKURVE_CACHE_DIR"], exist_ok=True)
    lk.conf.cache_dir = os.environ["LIGHTKURVE_CACHE_DIR"]
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
    
    return nom_max_hr, 0.35 * lower_circ_hr, 1.38 * upper_circ_hr


def scan_full_orbit_secondary_eclipse(
    t: np.ndarray,
    flat_f: np.ndarray,
    period: float,
    t0: float,
    dur: float,
    primary_depth_ppm: float
) -> tuple[bool, float, float, float, float]:
    """
    Scans all out-of-transit orbital phases phi in [min_phi, 1 - min_phi] using a sliding
    box of width 0.85 * dur to detect eccentric secondary eclipses (e > 0.1).
    Applies Sidak-Bonferroni trials-factor correction for alpha = 0.001 false-alarm rate.
    Returns: (is_eb_flag, max_sec_snr, max_sec_depth_ppm, best_sec_phase, z_crit_threshold)
    """
    phase_0to1 = ((t - t0) % period) / period
    dur_phase = dur / period
    min_phi = max(0.06, 1.6 * dur_phase)
    
    if min_phi >= 0.45:
        return False, 0.0, 0.0, 0.5, 4.0
        
    oot_mask = (phase_0to1 >= min_phi) & (phase_0to1 <= (1.0 - min_phi))
    if np.sum(oot_mask) < 50:
        return False, 0.0, 0.0, 0.5, 4.0
        
    oot_baseline = float(np.nanmedian(flat_f[oot_mask]))
    oot_noise = float(1.4826 * np.nanmedian(np.abs(flat_f[oot_mask] - oot_baseline)) + 1e-12)
    
    n_indep_win = max(5, int((1.0 - 2.0 * min_phi) / max(dur_phase, 0.005)))
    z_crit = float(stats.norm.isf(0.001 / n_indep_win))
    
    n_steps = max(20, n_indep_win * 2)
    trial_phases = np.linspace(min_phi + 0.5 * dur_phase, 1.0 - min_phi - 0.5 * dur_phase, n_steps)
    half_box = 0.425 * dur_phase
    
    ph_oot = phase_0to1[oot_mask]
    f_oot = flat_f[oot_mask]
    sort_idx = np.argsort(ph_oot)
    ph_sorted = ph_oot[sort_idx]
    f_sorted = f_oot[sort_idx]
    
    left_idx = np.searchsorted(ph_sorted, trial_phases - half_box, side="left")
    right_idx = np.searchsorted(ph_sorted, trial_phases + half_box, side="right")
    
    max_snr = 0.0
    max_depth_ppm = 0.0
    best_phi = 0.5
    
    for phi_c, l_i, r_i in zip(trial_phases, left_idx, right_idx):
        n_pts = r_i - l_i
        if n_pts < 5:
            continue
        dip_ppm = float((oot_baseline - np.nanmean(f_sorted[l_i:r_i])) * 1e6)
        err_ppm = (oot_noise * 1e6) / np.sqrt(n_pts)
        snr = dip_ppm / max(err_ppm, 1.0)
        if snr > max_snr:
            max_snr = snr
            max_depth_ppm = dip_ppm
            best_phi = float(phi_c)
            
    is_eb = (max_snr > z_crit) and (max_depth_ppm > 220.0) and (max_depth_ppm >= 0.12 * primary_depth_ppm)
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit


def check_binary_catalogs(ra: float, dec: float) -> dict:
    """
    Cross-matches target against VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022)
    and Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Table.
    """
    res = {
        "is_tess_eb": False,
        "tess_eb_tic": None,
        "tess_eb_period": None,
        "is_gaia_nss": False,
        "gaia_nss_type": None,
        "gaia_nss_period": None
    }
    # 1. VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022, ApJS, 258, 16)
    try:
        v = Vizier(columns=["TIC", "Per", "Morph"], row_limit=1)
        coord = SkyCoord(ra, dec, unit="deg", frame="icrs")
        v_res = v.query_region(coord, radius=8.0 * u.arcsec, catalog="J/ApJS/258/16/tess-ebs")
        if len(v_res) > 0 and len(v_res[0]) > 0:
            res["is_tess_eb"] = True
            res["tess_eb_tic"] = str(v_res[0]["TIC"][0])
            res["tess_eb_period"] = safe_float(v_res[0]["Per"][0], None)
    except Exception:
        pass

    # 2. Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Table
    try:
        adql_nss = (
            f"SELECT TOP 1 n.nss_solution_type, n.period FROM gaiadr3.nss_two_body_orbit AS n "
            f"JOIN gaiadr3.gaia_source AS g ON n.source_id=g.source_id "
            f"WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra}, {dec}, 0.001))"
        )
        url_nss = (
            "https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery"
            f"&LANG=ADQL&FORMAT=csv&QUERY={urllib.parse.quote(adql_nss)}"
        )
        df_nss = fetch_csv_no_ssl(url_nss)
        if len(df_nss) > 0 and "nss_solution_type" in df_nss.columns:
            sol = str(df_nss["nss_solution_type"].iloc[0]).strip()
            if sol and sol.upper() != "NAN":
                res["is_gaia_nss"] = True
                res["gaia_nss_type"] = sol
                res["gaia_nss_period"] = safe_float(df_nss["period"].iloc[0], None)
    except Exception:
        pass
    return res


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
    High-precision Mandel-Agol transit model using 300-ring radial annulus integration
    across the quadratic limb-darkened stellar disk. Sub-ppm accurate without staircase artifacts.
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
        
    n_rings = 300
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
    info = {
        "n_neighbors": 0, "contamination_ratio": 0.0, "neb_capable_stars": 0, "target_gmag": np.nan, "ruwe": np.nan,
        "is_tess_eb": False, "tess_eb_tic": None, "tess_eb_period": None,
        "is_gaia_nss": False, "gaia_nss_type": None, "gaia_nss_period": None
    }
    try:
        tic_data = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
        if len(tic_data) == 0:
            return info
        row = tic_data[0]
        ra = float(to_numpy(row["ra"]))
        dec = float(to_numpy(row["dec"]))
        
        # Cross-match against VizieR TESS EB & Gaia DR3 NSS binary tables
        bin_cats = check_binary_catalogs(ra, dec)
        info.update(bin_cats)
        
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
    print(f"PLEIADES DEEP SCIENTIFIC VETTING {PIPELINE_VERSION}: TIC {tic_id} (Initial P = {p_init:.5f} d)")
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
    
    # Pre-clip >4-sigma stellar flares and extreme single-cadence downward glitches
    med_raw = float(np.nanmedian(f))
    mad_raw = float(1.4826 * np.nanmedian(np.abs(f - med_raw)) + 1e-12)
    clean_clip = (f < (med_raw + 4.0 * mad_raw)) & (f > (med_raw - 8.0 * mad_raw))
    t, f = t[clean_clip], f[clean_clip]
    
    fine_p_grid = np.linspace(p_init * 0.995, p_init * 1.005, 6000)
    bls = BoxLeastSquares(t, f)
    bls_res = bls.power(fine_p_grid, np.linspace(max(0.025, dur_init * 0.7), min(0.25, dur_init * 1.3), 10))
    best_idx = int(np.nanargmax(bls_res.power))
    p_ref = float(bls_res.period[best_idx])
    t0_ref = float(bls_res.transit_time[best_idx])
    dur_ref = float(bls_res.duration[best_idx])
    
    phase = ((t - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref
    
    print(f"[2/4] Fitting 300-Ring Annular Limb-Darkened Transit Model (Teff={teff:.0f} K, R*={r_star:.2f} R_Sun)...")
    u1_est = float(np.clip(0.45 - 0.00008 * (teff - 5500.0), 0.15, 0.70))
    u2_est = float(np.clip(0.25 + 0.00003 * (teff - 5500.0), 0.05, 0.45))
    
    fit_win = np.abs(phase) <= (2.5 * dur_ref)
    ph_fit = phase[fit_win]
    f_fit = f[fit_win]
    
    rp_rs_guess = float(np.sqrt(max(1e-5, depth_init_ppm * 1e-6)))
    e_r_star = safe_float(cand_row.get("e_r_star_sun"), 0.10 * r_star)
    e_m_star = safe_float(cand_row.get("e_m_star_sun"), 0.12 * m_star)
    nom_max_dur_hr, min_kepler_dur_hr, max_kepler_dur_hr = compute_keplerian_duration_bounds_hours(
        p_ref, r_star, e_r_star, m_star, e_m_star, rp_rs_guess
    )
    upper_dur_bound = max(dur_ref * 0.7, min(dur_ref * 1.6, max_kepler_dur_hr / 24.0))
    lower_dur_bound = max(0.005, min(dur_ref * 0.5, min_kepler_dur_hr / 24.0))
    
    def model_to_fit(ph, rp_rs, b, dur_t):
        return exact_annulus_transit_model(ph, rp_rs, b, dur_t, u1_est, u2_est)
        
    try:
        p0_dur = np.clip(dur_ref, lower_dur_bound * 1.05, upper_dur_bound * 0.98)
        popt, _ = curve_fit(
            model_to_fit,
            ph_fit,
            f_fit,
            p0=[rp_rs_guess, 0.3, p0_dur],
            bounds=([0.002, 0.0, lower_dur_bound], [0.30, 0.98, upper_dur_bound]),
            maxfev=6000
        )
        rp_rs_fit, b_fit, dur_fit = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        rp_rs_fit, b_fit, dur_fit = rp_rs_guess, 0.5, dur_ref
        
    true_rp_earth = r_star * rp_rs_fit * 109.076
    dur_ratio_kepler = (dur_fit * 24.0) / max(nom_max_dur_hr, 0.1)
    
    print(f"[3/4] Scanning Full-Orbit Secondary Eclipse across all out-of-transit phases...")
    is_sec_eb, sec_snr, sec_depth_ppm, sec_phi, z_crit = scan_full_orbit_secondary_eclipse(
        t, f, p_ref, t0_ref, dur_fit, depth_init_ppm
    )
    sec_center_phi = sec_phi if np.isfinite(sec_phi) else 0.5
    
    print(f"[4/4] Querying Gaia DR3 neighborhood, NSS two-body & VizieR TESS EB catalogs...")
    gaia_contam = query_gaia_contamination(tic_id, depth_init_ppm)
    
    # Composite Heuristic False-Positive Risk Score (0.0% = clean, 100% = definite FP)
    risk_score = 0.2
    if b_fit > 0.75:
        risk_score += 25.0 * ((b_fit - 0.75) / 0.25) ** 2
    
    # Secondary eclipse significance calibration (Šidák-Bonferroni correction across ~30 phase bins):
    # Pure Gaussian white noise across ~30 independent windows has E[max(z)] ~ 2.1-2.5 sigma.
    # Penalize only when exceeding the true statistical significance threshold:
    # Circular zone (|phase - 0.50| <= 0.04): SNR > 2.8 sigma
    # Eccentric zone (|phase - 0.50| > 0.04): SNR > 3.4 sigma
    is_circular_sec = abs(sec_center_phi - 0.50) <= 0.04
    sec_snr_threshold = 2.8 if is_circular_sec else 3.4
    if sec_snr > sec_snr_threshold:
        risk_score += 45.0 * (sec_snr / 3.0)
    if is_sec_eb:
        risk_score += 65.0
    if gaia_contam.get("is_tess_eb"):
        risk_score += 90.0
    if gaia_contam.get("is_gaia_nss"):
        risk_score += 80.0
    if gaia_contam["neb_capable_stars"] > 0:
        risk_score += 8.0 * gaia_contam["neb_capable_stars"]
    if gaia_contam["contamination_ratio"] > 0.25:
        risk_score += 35.0 * gaia_contam["contamination_ratio"]
    if dur_ratio_kepler > 1.38:
        risk_score += 40.0 * (dur_ratio_kepler - 1.0)
    elif dur_ratio_kepler < 0.35:
        risk_score += 85.0 * ((0.35 - dur_ratio_kepler) / 0.35)
    if np.isfinite(gaia_contam["ruwe"]) and gaia_contam["ruwe"] > 1.25:
        risk_score += 15.0
        
    # Enforce Stitched Primary Transit SNR Gate (prevents marginal white noise from masquerading as confirmed candidate)
    in_tr_mask = np.abs(ph_fit) < (0.45 * dur_fit)
    oot_mask_local = (np.abs(ph_fit) > (0.65 * dur_fit)) & (np.abs(ph_fit) <= (2.2 * dur_fit))
    oot_std_pt = float(1.4826 * np.nanmedian(np.abs(f_fit[oot_mask_local] - 1.0)) + 1e-12)
    n_in = int(np.sum(in_tr_mask))
    stitched_primary_snr = float((1.0 - np.nanmean(f_fit[in_tr_mask])) / (oot_std_pt / np.sqrt(max(1, n_in)))) if n_in >= 3 else 0.0
    if stitched_primary_snr < 5.5:
        risk_score += min(65.0, max(0.0, (5.5 - stitched_primary_snr) * 22.0))
        
    risk_pct = float(np.clip(risk_score, 0.1, 99.9))
    
    is_clean = (
        risk_pct < 5.0
        and b_fit < 0.82
        and sec_snr <= sec_snr_threshold
        and not is_sec_eb
        and not gaia_contam.get("is_tess_eb")
        and not gaia_contam.get("is_gaia_nss")
        and 0.35 <= dur_ratio_kepler <= 1.38
        and stitched_primary_snr >= 5.5
    )
    if gaia_contam.get("is_tess_eb"):
        disposition = f"CONFIRMED FP / CATALOGED TESS EB (TIC {gaia_contam.get('tess_eb_tic')})"
    elif gaia_contam.get("is_gaia_nss"):
        disposition = f"CONFIRMED FP / GAIA NSS BINARY ({gaia_contam.get('gaia_nss_type')})"
    elif is_sec_eb:
        disposition = f"CONFIRMED FP / ECCENTRIC EB (Phase {sec_center_phi:.2f}, {sec_snr:.1f} sigma)"
    elif dur_ratio_kepler < 0.35:
        disposition = "SUSPECT / UNPHYSICALLY NARROW TRANSIT"
    elif dur_ratio_kepler > 1.38:
        disposition = "SUSPECT / UNPHYSICALLY LONG TRANSIT (Subgiant/Blend)"
    elif is_clean:
        disposition = "TIER-2 VETTED CANDIDATE (Low FP Risk)"
    else:
        disposition = "SUSPECT / GRAZING OR CONTAMINATED"
    
    print(f"      -> Fitted Radius: Rp = {true_rp_earth:.2f} R_Earth (Rp/R* = {rp_rs_fit:.4f})")
    print(f"      -> Impact Parameter: b = {b_fit:.3f} | Duration / Nom_Kepler = {dur_ratio_kepler:.2f}")
    print(f"      -> Secondary Eclipse: Depth = {sec_depth_ppm:.0f} ppm at Phase {sec_center_phi:.2f} (SNR = {sec_snr:.2f} sigma)")
    print(f"      -> Gaia DR3 Neighbors (120\"): {gaia_contam['n_neighbors']} stars | Contamination = {gaia_contam['contamination_ratio']*100:.2f}%")
    if gaia_contam.get("is_tess_eb"):
        print(f"      -> [!] VizieR TESS EB Match: TIC {gaia_contam.get('tess_eb_tic')} (P={gaia_contam.get('tess_eb_period')} d)")
    if gaia_contam.get("is_gaia_nss"):
        print(f"      -> [!] Gaia NSS Two-Body Orbit: {gaia_contam.get('gaia_nss_type')} (P={gaia_contam.get('gaia_nss_period')} d)")
    print(f"      -> Heuristic FP Risk Score: {risk_pct:.1f}% -> {disposition}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    ph_grid = np.linspace(-2.2 * dur_fit, 2.2 * dur_fit, 500)
    axes[0, 0].scatter(ph_fit * 24.0, f_fit, s=3, color="silver", alpha=0.35, label="Tukey Biweight Cadences")
    bins = np.linspace(-2.2 * dur_fit, 2.2 * dur_fit, 35)
    b_means, b_edges, _ = stats.binned_statistic(ph_fit, f_fit, statistic="mean", bins=bins)
    b_std, _, _ = stats.binned_statistic(ph_fit, f_fit, statistic="std", bins=bins)
    b_cnt, _, _ = stats.binned_statistic(ph_fit, f_fit, statistic="count", bins=bins)
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    valid_b = b_cnt >= 3
    b_err = b_std[valid_b] / np.sqrt(b_cnt[valid_b])
    axes[0, 0].errorbar(
        b_centers[valid_b] * 24.0,
        b_means[valid_b],
        yerr=b_err,
        fmt="o",
        color="black",
        ms=5,
        capsize=2.5,
        elinewidth=1.2,
        zorder=4,
        label="Binned Photometry (±1σ)"
    )
    axes[0, 0].plot(
        ph_grid * 24.0,
        model_to_fit(ph_grid, rp_rs_fit, b_fit, dur_fit),
        color="crimson",
        lw=2.5,
        zorder=5,
        label=f"300-Ring Fit (b={b_fit:.2f}, Rp={true_rp_earth:.2f} R_E)"
    )
    # Focused Y-axis zoom on the transit feature to prevent noisy outliers from squishing the curve
    transit_depth = rp_rs_fit ** 2
    y_min_target = 1.0 - max(0.003, 3.5 * transit_depth)
    y_max_target = 1.0 + max(0.002, 2.0 * transit_depth)
    if np.sum(valid_b) > 0:
        y_min = min(y_min_target, np.nanmin(b_means[valid_b] - b_err) - 0.0005)
        y_max = max(y_max_target, np.nanmax(b_means[valid_b] + b_err) + 0.0005)
    else:
        y_min, y_max = y_min_target, y_max_target
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 0].set_xlabel("Time from Mid-Transit (hours)")
    axes[0, 0].set_ylabel("Relative Flux")
    axes[0, 0].set_title(f"TIC {tic_id}: Annular Limb-Darkened Transit Model (Sectors {sectors_used})")
    axes[0, 0].legend(loc="lower right")
    
    phase_0to1 = ((t - t0_ref) % p_ref) / p_ref
    sec_phase_diff = ((phase_0to1 - sec_center_phi + 0.5) % 1.0) - 0.5
    sec_win = np.abs(sec_phase_diff) <= min(0.12, 3.0 * dur_fit / p_ref)
    ph_sec_hrs = sec_phase_diff[sec_win] * p_ref * 24.0
    axes[0, 1].scatter(ph_sec_hrs, f[sec_win], s=3, color="silver", alpha=0.4)
    if len(ph_sec_hrs) > 20:
        s_bins = np.linspace(np.min(ph_sec_hrs), np.max(ph_sec_hrs), 25)
        s_means, s_edges, _ = stats.binned_statistic(ph_sec_hrs, f[sec_win], statistic="mean", bins=s_bins)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        axes[0, 1].plot(s_centers, s_means, color="royalblue", lw=2, label=f"Phase {sec_center_phi:.2f} Binned (SNR={sec_snr:.1f} sigma)")
    axes[0, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[0, 1].set_xlabel(f"Time from Secondary Phase {sec_center_phi:.2f} (hours)")
    axes[0, 1].set_ylabel("Relative Flux")
    axes[0, 1].set_title(f"Secondary Eclipse Scan (Max Dip at Phase {sec_center_phi:.2f}: {sec_depth_ppm:.0f} ppm)")
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
    eb_match_str = f"TIC {gaia_contam.get('tess_eb_tic')} (P={gaia_contam.get('tess_eb_period', 'N/A')}d)" if gaia_contam.get("is_tess_eb") else "None"
    nss_match_str = f"{gaia_contam.get('gaia_nss_type')} (P={gaia_contam.get('gaia_nss_period', 'N/A')}d)" if gaia_contam.get("is_gaia_nss") else "None"
    summary_text = (
        f"SCIENTIFIC VETTING DOSSIER: TIC {tic_id}\n"
        f"{'-'*48}\n"
        f"Disposition:       {disposition}\n"
        f"Heuristic FP Risk: {risk_pct:.1f}% (Clean Threshold < 5.0%)\n\n"
        f"ORBITAL & PHYSICAL PARAMETERS (300-Ring Fit):\n"
        f"  Refined Period:  {p_ref:.6f} days\n"
        f"  Mid-Transit T0:  {t0_ref:.5f} BTJD\n"
        f"  Transit Dur:     {dur_fit*24.0:.2f} h (Kepler Bounds: {min_kepler_dur_hr:.2f}h - {max_kepler_dur_hr:.2f}h)\n"
        f"  Radius Ratio:    Rp/R* = {rp_rs_fit:.4f}\n"
        f"  Planet Radius:   Rp = {true_rp_earth:.2f} R_Earth\n"
        f"  Impact Param:    b = {b_fit:.3f} (0=central, >0.82=grazing)\n"
        f"  Limb Darkening:  u1 = {u1_est:.2f}, u2 = {u2_est:.2f} (Teff={teff:.0f} K)\n\n"
        f"FALSE POSITIVE FALSIFICATION TESTS:\n"
        f"  Secondary Ecl:   {sec_depth_ppm:.0f} ppm ({sec_snr:.2f} sigma at Phase {sec_center_phi:.2f})\n"
        f"  TESS EB Match:   {eb_match_str}\n"
        f"  Gaia NSS Orbit:  {nss_match_str}\n"
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
        "refined_t0_btjd": round(t0_ref, 5),
        "fitted_depth_ppm": round((rp_rs_fit ** 2) * 1e6, 1),
        "fitted_duration_hours": round(dur_fit * 24.0, 2),
        "fitted_rp_earth": round(true_rp_earth, 2),
        "impact_param_b": round(b_fit, 3),
        "dur_to_kepler_max": round(dur_ratio_kepler, 2),
        "stitched_primary_snr": round(stitched_primary_snr, 2),
        "secondary_snr_sigma": round(sec_snr, 2),
        "sec_eclipse_phase": round(sec_center_phi, 3),
        "is_known_tess_eb": gaia_contam.get("is_tess_eb", False),
        "gaia_nss_type": gaia_contam.get("gaia_nss_type"),
        "flux_contam_percent": round(gaia_contam["contamination_ratio"] * 100, 2),
        "neb_capable_neighbors": gaia_contam["neb_capable_stars"],
        "sectors_stitched": ",".join(str(s) for s in sectors_used)
    }


if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"File '{INPUT_CSV}' not found. Run aldebaran.py first.")
    else:
        df_cands = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df_cands)} candidates from '{INPUT_CSV}' for Pleiades Tier-2 Deep Vetting {PIPELINE_VERSION}.")
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
            print(f"\nPleiades deep vetting complete! Dossiers saved to '{VET_DIR}'.")