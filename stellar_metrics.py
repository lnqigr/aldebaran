"""
stellar_metrics.py - stellar photometry and planet characterization helpers shared
by the Antares packager (Stage 3) and the CTOI packager.

Roadmap C9 (S3.5): each function had equivalent copies in antares_packager.py and
ctoi_packager.py. The bodies below were moved unchanged from antares_packager.py at
0ca167abe3, except that batch_query_tic_photometry takes the progress line prefix as
a keyword, so each packager prints the same line as before. The importing modules
re-export the same names, so their module attributes are unchanged.
"""
import os
import glob

import numpy as np
import pandas as pd
from astropy.io import fits
from astroquery.mast import Catalogs


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
    
    if not np.isfinite(tmag):
        if np.isfinite(gmag):
            tmag = gmag - gt
        elif np.isfinite(vmag):
            tmag = vmag - vg - gt
        elif np.isfinite(jmag):
            tmag = jmag + tj
        else:
            tmag = 12.5
            
    if not np.isfinite(gmag):
        gmag = tmag + gt
    if not np.isfinite(vmag):
        vmag = gmag + vg
        
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


def batch_query_tic_photometry(tic_ids: list[str], progress_prefix: str = "[TIC Query] Querying MAST TIC v8.2 catalog") -> dict:
    catalog_map = {}
    int_ids = [int(t) for t in tic_ids if str(t).isdigit()]
    if not int_ids:
        return catalog_map
        
    print(f"{progress_prefix} for {len(int_ids)} targets...")
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
    rp_e = max(0.3, float(rp_earth))
    p_d = max(0.1, float(p_days))
    r_s = max(0.08, float(r_star_sun))
    m_s = max(0.08, float(m_star_sun)) if (np.isfinite(m_star_sun) and m_star_sun > 0) else (r_s ** 1.1)
    t_eff = max(2500.0, float(teff_k)) if (np.isfinite(teff_k) and teff_k > 0) else 5500.0
    b_val = float(np.clip(b_impact if np.isfinite(b_impact) else 0.3, 0.0, 0.98))
    
    # 1. Planet Mass M_p (Chen & Kipping 2017 & Wolfgang et al. 2016)
    if rp_e < 1.23:
        mp_ck17 = (rp_e / 1.008) ** (1.0 / 0.279)
    elif rp_e <= 14.26:
        mp_ck17 = (rp_e / 0.8081) ** (1.0 / 0.589)
    else:
        mp_ck17 = 131.6 * ((rp_e / 14.26) ** 2.0)
    mp_w16 = 2.7 * (rp_e ** 1.3)
    
    # 2. Bulk Density & Surface Gravity
    rp_m = rp_e * R_EARTH
    vol_cm3 = (4.0 / 3.0) * np.pi * ((rp_m * 100.0) ** 3)
    rho_p_cgs = (mp_ck17 * M_EARTH * 1000.0) / vol_cm3
    g_p_ms2 = (G_SI * mp_ck17 * M_EARTH) / (rp_m ** 2)
    
    # 3. Orbital Dynamics & Temperature
    p_sec = p_d * 86400.0
    m_star_kg = m_s * M_SUN
    r_star_m = r_s * R_SUN
    a_m = (G_SI * m_star_kg * (p_sec ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    a_au = a_m / AU_M
    a_over_rs = a_m / r_star_m
    
    l_star_sun = (r_s ** 2) * ((t_eff / T_SUN) ** 4)
    insolation_earth = l_star_sun / (a_au ** 2)
    teq_k = t_eff * np.sqrt(r_star_m / (2.0 * a_m))
    t_day_k = 1.10 * teq_k
    
    # 4. Radial Velocity Semi-Amplitude K (m/s)
    sin_i = np.sqrt(max(0.01, 1.0 - (b_val / max(a_over_rs, 1.1)) ** 2))
    rv_prefactor = ((2.0 * np.pi * G_SI / p_sec) ** (1.0 / 3.0)) * sin_i / (m_star_kg ** (2.0 / 3.0))
    k_rv_ck17 = float(rv_prefactor * (mp_ck17 * M_EARTH))
    k_rv_w16 = float(rv_prefactor * (mp_w16 * M_EARTH))
    
    # 5. Transmission Spectroscopy Metric (TSM, Kempton et al. 2018)
    if rp_e < 1.50:
        tsm_scale, tsm_benchmark = 0.190, 10.0
    elif rp_e < 2.75:
        tsm_scale, tsm_benchmark = 1.26, 90.0
    elif rp_e < 4.00:
        tsm_scale, tsm_benchmark = 1.28, 90.0
    elif rp_e <= 10.00:
        tsm_scale, tsm_benchmark = 1.15, 90.0
    else:
        tsm_scale, tsm_benchmark = 1.00, 90.0
        
    tsm = float(tsm_scale * ((rp_e ** 3) * teq_k / (mp_ck17 * (r_s ** 2))) * (10.0 ** (-0.2 * jmag)))
    
    # 6. Emission Spectroscopy Metric (ESM, Kempton et al. 2018)
    lam_75um = 7.5e-6
    b_planet = planck_spectral_radiance(lam_75um, t_day_k)
    b_star = max(1e-30, planck_spectral_radiance(lam_75um, t_eff))
    rp_over_rs_phys = rp_m / r_star_m
    esm = float(4.29e6 * (b_planet / b_star) * (rp_over_rs_phys ** 2) * (10.0 ** (-0.2 * kmag)))
    
    # 7. Follow-Up Prioritization Tiers
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
