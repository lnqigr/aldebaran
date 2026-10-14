import os
import io
import ssl
import glob
import warnings
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import requests
import urllib3
from astropy.io import fits
from astroquery.mast import Catalogs

# Global SSL bypass for Caltech/MAST TAP compatibility
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

# Physical & Astronomical Constants (SI / IAU 2015 Nominal)
G_SI = 6.67430e-11
M_SUN = 1.98847e30
R_SUN = 6.957e8
T_SUN = 5772.0
AU_M = 1.495978707e11
M_EARTH = 5.97217e24
R_EARTH = 6.371e6
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23
BTJD_OFFSET = 2457000.0

# File Paths
RESULTS_DIR = "tess_results"
TIER1_CSV = os.path.join(RESULTS_DIR, "candidates_report.csv")
TIER2_CSV = os.path.join(RESULTS_DIR, "deep_vetting", "scientific_dossier_summary.csv")
RANKING_OUT_CSV = os.path.join(RESULTS_DIR, "prioritized_targets_ranking.csv")
EXOFOP_OUT_CSV = os.path.join(RESULTS_DIR, "exofop_ctoi_upload.csv")


def to_clean_float(val, default: float = np.nan) -> float:
    try:
        if pd.isna(val) or str(val).strip().upper() in ("", "N/A", "NAN", "NONE", "NULL"):
            return default
        f_val = float(val)
        return f_val if np.isfinite(f_val) else default
    except Exception:
        return default


def estimate_missing_photometry(tmag: float, vmag: float, gmag: float, jmag: float, hmag: float, kmag: float, teff: float) -> dict:
    """
    Reconstructs missing optical/NIR magnitudes using empirical main-sequence dwarf
    color-temperature relations (Pecaut & Mamajek 2013; Stassun et al. 2019).
    """
    teff_clamped = float(np.clip(teff if np.isfinite(teff) else 5500.0, 2800.0, 10000.0))
    
    # Empirical grid: Teff -> (V - G, G - T, T - J, J - H, H - Ks)
    teff_grid = np.array([3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5770.0, 6500.0, 7500.0, 9500.0])
    vg_grid   = np.array([1.45,   0.95,   0.55,   0.35,   0.22,   0.12,   0.06,   0.02,   0.00])
    gt_grid   = np.array([1.35,   1.05,   0.78,   0.60,   0.46,   0.33,   0.22,   0.10,   0.00])
    tj_grid   = np.array([1.55,   1.32,   1.12,   0.92,   0.75,   0.58,   0.42,   0.25,   0.05])
    jh_grid   = np.array([0.60,   0.62,   0.64,   0.56,   0.46,   0.33,   0.23,   0.12,   0.02])
    hk_grid   = np.array([0.26,   0.23,   0.19,   0.14,   0.10,   0.07,   0.05,   0.03,   0.01])
    
    vg = float(np.interp(teff_clamped, teff_grid, vg_grid))
    gt = float(np.interp(teff_clamped, teff_grid, gt_grid))
    tj = float(np.interp(teff_clamped, teff_grid, tj_grid))
    jh = float(np.interp(teff_clamped, teff_grid, jh_grid))
    hk = float(np.interp(teff_clamped, teff_grid, hk_grid))
    
    # Establish optical/TESS anchor magnitude
    if not np.isfinite(tmag):
        if np.isfinite(gmag):
            tmag = gmag - gt
        elif np.isfinite(vmag):
            tmag = vmag - vg - gt
        elif np.isfinite(jmag):
            tmag = jmag + tj
        else:
            tmag = 12.5  # Conservative fallback if all magnitudes are missing
            
    if not np.isfinite(gmag):
        gmag = tmag + gt
    if not np.isfinite(vmag):
        vmag = gmag + vg
        
    # Reconstruct 2MASS NIR magnitudes (J, H, Ks)
    j_fallback = False
    if not np.isfinite(jmag):
        jmag = tmag - tj
        j_fallback = True
    if not np.isfinite(hmag):
        hmag = jmag - jh
    if not np.isfinite(kmag):
        kmag = hmag - hk
        
    return {
        "Tmag": round(float(tmag), 3),
        "Vmag": round(float(vmag), 3),
        "Gmag": round(float(gmag), 3),
        "Jmag": round(float(jmag), 3),
        "Hmag": round(float(hmag), 3),
        "Kmag": round(float(kmag), 3),
        "nir_estimated": j_fallback
    }


def batch_query_tic_photometry(tic_ids: list[str]) -> dict:
    """
    Performs a single fast batch query to the MAST TIC v8.2 catalog for all surviving TIC IDs.
    Extracts Tmag, Vmag, GAIAmag, Jmag, Hmag, Kmag, RA, Dec, R_star, M_star, Teff.
    """
    catalog_map = {}
    int_ids = [int(t) for t in tic_ids if str(t).isdigit()]
    if not int_ids:
        return catalog_map
        
    print(f"[1/4] Querying MAST TIC v8.2 catalog in batch mode for {len(int_ids)} targets...")
    try:
        tic_table = Catalogs.query_criteria(catalog="Tic", ID=int_ids)
        for row in tic_table:
            tid = str(int(row["ID"]))
            catalog_map[tid] = {
                "ra": to_clean_float(row["ra"]),
                "dec": to_clean_float(row["dec"]),
                "Tmag": to_clean_float(row["Tmag"]),
                "Vmag": to_clean_float(row["Vmag"]),
                "Gmag": to_clean_float(row["GAIAmag"]),
                "Jmag": to_clean_float(row["Jmag"]),
                "Hmag": to_clean_float(row["Hmag"]),
                "Kmag": to_clean_float(row["Kmag"]),
                "r_star": to_clean_float(row["rad"]),
                "e_r_star": to_clean_float(row["e_rad"]) if "e_rad" in row.colnames else np.nan,
                "m_star": to_clean_float(row["mass"]),
                "e_m_star": to_clean_float(row["e_mass"]) if "e_mass" in row.colnames else np.nan,
                "teff": to_clean_float(row["Teff"]),
                "e_teff": to_clean_float(row["e_Teff"]) if "e_Teff" in row.colnames else np.nan,
            }
    except Exception as e:
        print(f"      [!] Batch TIC query warning ({e}), falling back to individual lookups...")
        for tid in int_ids:
            try:
                sub = Catalogs.query_criteria(catalog="Tic", ID=tid)
                if len(sub) > 0:
                    row = sub[0]
                    catalog_map[str(tid)] = {
                        "ra": to_clean_float(row["ra"]),
                        "dec": to_clean_float(row["dec"]),
                        "Tmag": to_clean_float(row["Tmag"]),
                        "Vmag": to_clean_float(row["Vmag"]),
                        "Gmag": to_clean_float(row["GAIAmag"]),
                        "Jmag": to_clean_float(row["Jmag"]),
                        "Hmag": to_clean_float(row["Hmag"]),
                        "Kmag": to_clean_float(row["Kmag"]),
                        "r_star": to_clean_float(row["rad"]),
                        "e_r_star": to_clean_float(row["e_rad"]) if "e_rad" in row.colnames else np.nan,
                        "m_star": to_clean_float(row["mass"]),
                        "e_m_star": to_clean_float(row["e_mass"]) if "e_mass" in row.colnames else np.nan,
                        "teff": to_clean_float(row["Teff"]),
                        "e_teff": to_clean_float(row["e_Teff"]) if "e_Teff" in row.colnames else np.nan,
                    }
            except Exception:
                continue
    return catalog_map


def extract_camera_ccd_from_cache(tic_id: str) -> tuple[str, str]:
    """Scans local Lightkurve cache for FITS headers of tic_id to extract CAMERA and CCD numbers."""
    cache_dirs = [
        os.environ.get("LIGHTKURVE_CACHE_DIR", ""),
        os.path.join(os.path.expanduser("~"), ".lightkurve", "cache"),
        os.path.join(os.path.expanduser("~"), ".cache", "lightkurve")
    ]
    padded = str(tic_id).zfill(16)
    for cdir in cache_dirs:
        if not cdir or not os.path.exists(cdir):
            continue
        matches = glob.glob(os.path.join(cdir, "**", f"*{padded}*lc.fits"), recursive=True)
        if matches:
            try:
                with fits.open(matches[-1], memmap=False) as hdul:
                    hdr = hdul[0].header
                    cam = str(hdr.get("CAMERA", "N/A"))
                    ccd = str(hdr.get("CCD", "N/A"))
                    return cam, ccd
            except Exception:
                pass
    return "N/A", "N/A"


def planck_spectral_radiance(wavelength_m: float, temp_k: float) -> float:
    """Evaluates Planck blackbody spectral radiance B_lambda(T) at wavelength_m."""
    if temp_k <= 20.0:
        return 0.0
    exponent = (H_PLANCK * C_LIGHT) / (wavelength_m * K_BOLTZ * temp_k)
    exponent = min(float(exponent), 500.0)
    return float((2.0 * H_PLANCK * (C_LIGHT ** 2) / (wavelength_m ** 5)) / np.expm1(exponent))


def compute_astrophysical_metrics(
    rp_earth: float,
    p_days: float,
    b_impact: float,
    r_star_sun: float,
    m_star_sun: float,
    teff_k: float,
    jmag: float,
    kmag: float,
    v_or_tmag: float
) -> dict:
    """
    Computes planet mass (Chen & Kipping 2017; Wolfgang et al. 2016), bulk density,
    surface gravity, semi-major axis, insolation, Teq, RV semi-amplitude K,
    and Kempton et al. (2018) TSM & ESM atmospheric characterization metrics.
    """
    rp_e = max(0.3, float(rp_earth))
    p_d = max(0.1, float(p_days))
    r_s = max(0.08, float(r_star_sun))
    m_s = max(0.08, float(m_star_sun)) if (np.isfinite(m_star_sun) and m_star_sun > 0) else (r_s ** 1.1)
    t_eff = max(2500.0, float(teff_k)) if (np.isfinite(teff_k) and teff_k > 0) else 5500.0
    b_val = float(np.clip(b_impact if np.isfinite(b_impact) else 0.3, 0.0, 0.98))
    
    # 1. Planet Mass M_p (M_Earth)
    # Chen & Kipping (2017, ApJ, 834, 17) piecewise power-law:
    if rp_e < 1.23:
        mp_ck17 = (rp_e / 1.008) ** (1.0 / 0.279)
    elif rp_e <= 14.26:
        mp_ck17 = (rp_e / 0.8081) ** (1.0 / 0.589)
    else:
        mp_ck17 = 131.6 * ((rp_e / 14.26) ** 2.0)
        
    # Wolfgang et al. (2016, ApJ, 825, 19) sub-Neptune relation:
    mp_w16 = 2.7 * (rp_e ** 1.3)
    
    # 2. Bulk Density (g/cm^3) & Surface Gravity (m/s^2) using Chen & Kipping mass
    rp_m = rp_e * R_EARTH
    vol_cm3 = (4.0 / 3.0) * np.pi * ((rp_m * 100.0) ** 3)
    rho_p_cgs = (mp_ck17 * M_EARTH * 1000.0) / vol_cm3
    g_p_ms2 = (G_SI * mp_ck17 * M_EARTH) / (rp_m ** 2)
    
    # 3. Orbital Dynamics, Insolation & Equilibrium Temperature
    p_sec = p_d * 86400.0
    m_star_kg = m_s * M_SUN
    r_star_m = r_s * R_SUN
    a_m = (G_SI * m_star_kg * (p_sec ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    a_au = a_m / AU_M
    a_over_rs = a_m / r_star_m
    
    l_star_sun = (r_s ** 2) * ((t_eff / T_SUN) ** 4)
    insolation_earth = l_star_sun / (a_au ** 2)
    
    # Kempton et al. (2018) standard zero-albedo, full-redistribution Teq & dayside T_day
    teq_k = t_eff * np.sqrt(r_star_m / (2.0 * a_m))
    t_day_k = 1.10 * teq_k
    
    # 4. Radial Velocity Semi-Amplitude K (m/s)
    sin_i = np.sqrt(max(0.01, 1.0 - (b_val / max(a_over_rs, 1.1)) ** 2))
    rv_prefactor = ((2.0 * np.pi * G_SI / p_sec) ** (1.0 / 3.0)) * sin_i / (m_star_kg ** (2.0 / 3.0))
    k_rv_ck17 = float(rv_prefactor * (mp_ck17 * M_EARTH))
    k_rv_w16 = float(rv_prefactor * (mp_w16 * M_EARTH))
    
    # 5. Transmission Spectroscopy Metric (TSM, Kempton et al. 2018 Eq. 1 & Table 1)
    if rp_e < 1.50:
        tsm_scale = 0.190
        tsm_benchmark = 10.0
    elif rp_e < 2.75:
        tsm_scale = 1.26
        tsm_benchmark = 90.0
    elif rp_e < 4.00:
        tsm_scale = 1.28
        tsm_benchmark = 90.0
    elif rp_e <= 10.00:
        tsm_scale = 1.15
        tsm_benchmark = 90.0
    else:
        tsm_scale = 1.00
        tsm_benchmark = 90.0
        
    tsm = float(tsm_scale * ((rp_e ** 3) * teq_k / (mp_ck17 * (r_s ** 2))) * (10.0 ** (-0.2 * jmag)))
    
    # 6. Emission Spectroscopy Metric (ESM, Kempton et al. 2018 Eq. 4 at lambda = 7.5 um)
    lam_75um = 7.5e-6
    b_planet = planck_spectral_radiance(lam_75um, t_day_k)
    b_star = max(1e-30, planck_spectral_radiance(lam_75um, t_eff))
    rp_over_rs_phys = rp_m / r_star_m
    esm = float(4.29e6 * (b_planet / b_star) * (rp_over_rs_phys ** 2) * (10.0 ** (-0.2 * kmag)))
    
    # 7. Spectroscopic Follow-Up Tiering (ESPRESSO / HARPS-N / MAROON-X & JWST)
    if k_rv_ck17 >= 3.0 and v_or_tmag <= 13.5:
        rv_tier = "HIGH (Tier-1 ESPRESSO/HARPS-N/MAROON-X)"
    elif k_rv_ck17 >= 1.5 and v_or_tmag <= 15.0:
        rv_tier = "MEDIUM (Tier-2 MAROON-X/ESPRESSO)"
    else:
        rv_tier = "LOW (Faint host or K < 1.5 m/s)"
        
    if tsm >= tsm_benchmark or esm >= 7.5:
        jwst_tier = "TOP-QUARTILE JWST TARGET (TSM/ESM Pass)"
    elif tsm >= 0.5 * tsm_benchmark or esm >= 4.0:
        jwst_tier = "MODERATE JWST TARGET"
    else:
        jwst_tier = "BASELINE PHOTOMETRIC TARGET"
        
    return {
        "mp_ck17_earth": round(mp_ck17, 2),
        "mp_w16_earth": round(mp_w16, 2),
        "rho_p_gcm3": round(rho_p_cgs, 2),
        "g_p_ms2": round(g_p_ms2, 2),
        "a_au": round(a_au, 5),
        "a_over_rstar": round(a_over_rs, 2),
        "insolation_earth": round(insolation_earth, 1),
        "teq_k": round(teq_k, 0),
        "t_day_k": round(t_day_k, 0),
        "rv_k_ck17_ms": round(k_rv_ck17, 2),
        "rv_k_w16_ms": round(k_rv_w16, 2),
        "tsm": round(tsm, 1),
        "esm": round(esm, 2),
        "rv_followup_priority": rv_tier,
        "jwst_followup_priority": jwst_tier
    }


def run_ctoi_packager():
    print("=" * 72)
    print("ALDEBARAN CTOI PACKAGER & FOLLOW-UP PRIORITIZATION ENGINE")
    print("=" * 72)
    
    if not os.path.exists(TIER1_CSV):
        raise FileNotFoundError(f"Tier-1 report '{TIER1_CSV}' not found. Run aldebaran.py / merge_shards.py first.")
        
    df_t1 = pd.read_csv(TIER1_CSV)
    df_t1["tic_id"] = df_t1["tic_id"].astype(str).str.strip()
    df_t1 = df_t1.drop_duplicates(subset=["tic_id"], keep="last")
    
    if os.path.exists(TIER2_CSV):
        df_t2 = pd.read_csv(TIER2_CSV)
        df_t2["tic_id"] = df_t2["tic_id"].astype(str).str.strip()
        df_t2 = df_t2.drop_duplicates(subset=["tic_id"], keep="last")
        df_merged = pd.merge(df_t1, df_t2, on="tic_id", how="left", suffixes=("_t1", ""))
        print(f"Loaded {len(df_t1)} Tier-1 candidates and matched {len(df_t2)} Tier-2 Pleiades dossiers.")
    else:
        print(f"[!] Warning: Tier-2 summary '{TIER2_CSV}' not found. Packaging directly from Tier-1 report.")
        df_merged = df_t1.copy()
        for col, default_v in [
            ("disposition", "TIER-1 CANDIDATE"),
            ("fp_risk_percent", 25.0),
            ("refined_period_days", np.nan),
            ("fitted_rp_earth", np.nan),
            ("impact_param_b", 0.30),
            ("dur_to_kepler_max", 1.00),
            ("secondary_snr_sigma", 0.0),
            ("sec_eclipse_phase", 0.50),
            ("is_known_tess_eb", False),
            ("gaia_nss_type", np.nan),
            ("flux_contam_percent", 0.0),
            ("neb_capable_neighbors", 0),
            ("sectors_stitched", "")
        ]:
            if col not in df_merged.columns:
                df_merged[col] = default_v

    # Step 2: Filter out confirmed False Positives
    n_initial = len(df_merged)
    has_gaia_nss = df_merged["gaia_nss_type"].notna() & (
        ~df_merged["gaia_nss_type"].astype(str).str.strip().str.upper().isin(["", "NAN", "NONE", "N/A", "FALSE"])
    )
    has_sec_eb = pd.to_numeric(df_merged["secondary_snr_sigma"], errors="coerce").fillna(0.0) > 3.5
    has_tess_eb = df_merged["is_known_tess_eb"].astype(str).str.strip().str.upper() == "TRUE"
    
    fp_mask = has_gaia_nss | has_sec_eb | has_tess_eb
    df_rejected = df_merged[fp_mask]
    df_clean = df_merged[~fp_mask].copy()
    
    print(f"Filtered out {len(df_rejected)} confirmed FPs (Gaia NSS / Secondary SNR > 3.5 / Known TESS EB).")
    if len(df_rejected) > 0:
        for _, r_row in df_rejected.iterrows():
            print(f"      -> Excluded TIC {r_row['tic_id']}: {r_row.get('disposition', 'FP')}")
            
    if len(df_clean) == 0:
        print("[!] No surviving candidates remain after FP filtering.")
        return

    # Step 3: Batch query TIC v8.2 photometry
    surviving_tics = df_clean["tic_id"].tolist()
    tic_phot_map = batch_query_tic_photometry(surviving_tics)
    
    # Step 4: Calculate astrophysical parameters & build ranking + ExoFOP rows
    print(f"[2/4] Computing M_p, RV semi-amplitude K, TSM, and ESM for {len(df_clean)} candidates...")
    ranking_rows = []
    exofop_rows = []
    
    for _, row in df_clean.iterrows():
        tid = str(row["tic_id"])
        t_meta = tic_phot_map.get(tid, {})
        
        p_days = to_clean_float(row.get("refined_period_days"), to_clean_float(row.get("period_days"), 1.0))
        t0_btjd = to_clean_float(row.get("t0_btjd"), 3980.0)
        epoch_bjd = t0_btjd + BTJD_OFFSET
        dur_hr = to_clean_float(row.get("duration_hours"), 2.0)
        depth_ppm = to_clean_float(row.get("depth_ppm"), 1000.0)
        rp_earth = to_clean_float(row.get("fitted_rp_earth"), to_clean_float(row.get("r_planet_earth"), 2.0))
        b_imp = to_clean_float(row.get("impact_param_b"), 0.30)
        
        r_star = to_clean_float(row.get("r_star_sun"), t_meta.get("r_star", 1.0))
        m_star = to_clean_float(row.get("m_star_sun"), t_meta.get("m_star", r_star ** 1.1))
        teff = to_clean_float(row.get("teff_k"), t_meta.get("teff", 5500.0))
        
        phot = estimate_missing_photometry(
            t_meta.get("Tmag", np.nan),
            t_meta.get("Vmag", np.nan),
            t_meta.get("Gmag", np.nan),
            t_meta.get("Jmag", np.nan),
            t_meta.get("Hmag", np.nan),
            t_meta.get("Kmag", np.nan),
            teff
        )
        
        astro = compute_astrophysical_metrics(
            rp_earth=rp_earth,
            p_days=p_days,
            b_impact=b_imp,
            r_star_sun=r_star,
            m_star_sun=m_star,
            teff_k=teff,
            jmag=phot["Jmag"],
            kmag=phot["Kmag"],
            v_or_tmag=phot["Tmag"]
        )
        
        fp_risk = to_clean_float(row.get("fp_risk_percent"), 30.0)
        sde_val = to_clean_float(row.get("sde"), 6.5)
        contam_pct = to_clean_float(row.get("flux_contam_percent"), 0.0)
        neb_cnt = int(to_clean_float(row.get("neb_capable_neighbors"), 0))
        secs_str = str(row.get("sectors_stitched") if pd.notna(row.get("sectors_stitched")) and str(row.get("sectors_stitched")) != "" else row.get("sector", "N/A"))
        
        # Composite Follow-Up Score: rewards low FP risk, high TSM, and high RV K
        clean_weight = max(0.1, (100.0 - fp_risk) / 100.0)
        priority_score = round(clean_weight * (0.65 * astro["tsm"] + 4.5 * astro["rv_k_ck17_ms"] + 1.5 * astro["esm"]), 2)
        
        cam_str, ccd_str = extract_camera_ccd_from_cache(tid)
        
        ranking_rows.append({
            "tic_id": tid,
            "priority_score": priority_score,
            "disposition": row.get("disposition", "CANDIDATE"),
            "fp_risk_percent": round(fp_risk, 1),
            "sectors": secs_str,
            "period_days": round(p_days, 6),
            "t0_btjd": round(t0_btjd, 5),
            "epoch_bjd": round(epoch_bjd, 5),
            "duration_hours": round(dur_hr, 2),
            "depth_ppm": round(depth_ppm, 1),
            "rp_earth": round(rp_earth, 2),
            "mp_ck17_earth": astro["mp_ck17_earth"],
            "mp_w16_earth": astro["mp_w16_earth"],
            "rho_p_gcm3": astro["rho_p_gcm3"],
            "g_p_ms2": astro["g_p_ms2"],
            "rv_k_ck17_ms": astro["rv_k_ck17_ms"],
            "rv_k_w16_ms": astro["rv_k_w16_ms"],
            "tsm": astro["tsm"],
            "esm": astro["esm"],
            "teq_k": int(astro["teq_k"]),
            "insolation_earth": astro["insolation_earth"],
            "a_au": astro["a_au"],
            "a_over_rstar": astro["a_over_rstar"],
            "impact_param_b": round(b_imp, 3),
            "r_star_sun": round(r_star, 2),
            "m_star_sun": round(m_star, 2),
            "teff_k": int(round(teff, 0)),
            "Tmag": phot["Tmag"],
            "Vmag": phot["Vmag"],
            "Gmag": phot["Gmag"],
            "Jmag": phot["Jmag"],
            "Hmag": phot["Hmag"],
            "Kmag": phot["Kmag"],
            "nir_phot_estimated": phot["nir_estimated"],
            "flux_contam_percent": round(contam_pct, 2),
            "neb_capable_neighbors": neb_cnt,
            "rv_followup_priority": astro["rv_followup_priority"],
            "jwst_followup_priority": astro["jwst_followup_priority"],
            "exofop_url": f"https://exofop.ipac.caltech.edu/tess/target.php?id={tid}"
        })
        
        comment_str = (
            f"Aldebaran v8.0 / Pleiades v2.2 Candidate (Sectors {secs_str}, SDE={sde_val:.1f}, b={b_imp:.2f}). "
            f"Teq={int(astro['teq_k'])}K, Mp_est={astro['mp_ck17_earth']:.1f} M_E, RV K={astro['rv_k_ck17_ms']:.1f} m/s, "
            f"TSM={astro['tsm']:.1f}, ESM={astro['esm']:.1f}. "
            f"Gaia 120arcsec Contam={contam_pct:.1f}% ({neb_cnt} NEB-capable neighbors). "
            f"Disposition: {row.get('disposition', 'PC')} (FP Risk={fp_risk:.1f}%)."
        )
        
        exofop_rows.append({
            "TIC ID": tid,
            "Camera": cam_str,
            "CCD": ccd_str,
            "Epoch BJD": round(epoch_bjd, 5),
            "Period": round(p_days, 6),
            "Depth": round(depth_ppm, 1),
            "Duration": round(dur_hr, 2),
            "Rp": round(rp_earth, 2),
            "User Comments / Disposition": comment_str
        })

    # Step 5: Sort by priority_score, TSM, and RV K descending
    df_rank = pd.DataFrame(ranking_rows)
    df_rank = df_rank.sort_values(
        by=["priority_score", "tsm", "rv_k_ck17_ms"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    df_rank.insert(0, "rank", np.arange(1, len(df_rank) + 1))
    
    # Sort ExoFOP upload table in the exact same ranked order
    ordered_tics = df_rank["tic_id"].tolist()
    df_exofop = pd.DataFrame(exofop_rows)
    df_exofop["sort_key"] = df_exofop["TIC ID"].apply(lambda x: ordered_tics.index(str(x)))
    df_exofop = df_exofop.sort_values(by="sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    
    # Step 6: Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_rank.to_csv(RANKING_OUT_CSV, index=False)
    df_exofop.to_csv(EXOFOP_OUT_CSV, index=False, sep="|")
    # Also save comma-separated copy for spreadsheet inspection
    df_exofop.to_csv(EXOFOP_OUT_CSV.replace(".csv", "_comma.csv"), index=False)
    
    print(f"[3/4] Saved ranked astrophysical catalog to: '{RANKING_OUT_CSV}'")
    print(f"[4/4] Saved ExoFOP CTOI upload table to:     '{EXOFOP_OUT_CSV}' (pipe-delimited) & '_comma.csv'\n")
    
    display_cols = [
        "rank", "tic_id", "period_days", "rp_earth", "mp_ck17_earth",
        "rv_k_ck17_ms", "tsm", "esm", "Tmag", "Jmag", "fp_risk_percent"
    ]
    print("TOP RANKED EXOPLANET CANDIDATES:")
    print(df_rank[display_cols].head(15).to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    run_ctoi_packager()
