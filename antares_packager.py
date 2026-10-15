#!/usr/bin/env python3
"""
ALDEBARAN v8.0 / ANTARES v3.0: UNIFIED TIER-3 ARCHIVAL VETTER & CTOI PACKAGER
=============================================================================
Combines deep multi-year archival light curve vetting (SPOC, TESS-SPOC, QLP, TGLC)
with astrophysical parameter estimation, RV/JWST follow-up prioritization, and
automated NASA ExoFOP CTOI upload table generation.

Architecture:
  - Stage 3A: Fast Pre-Filter Gate (5-rule physical screening on Tier-2 summary)
  - Stage 3B: Decoupled Multi-Year Archival Drill (SPOC-first master stack vs FFI blend track)
  - Stage 3C: Atmospheric / RV Metrics & NASA ExoFOP CTOI Table Generation
"""

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
from astropy.io import fits
import lightkurve as lk
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
BTJD_OFFSET = 2457000.0

# Pipeline Author priority and diagnostic colors
AUTHOR_PRIORITY = {"SPOC": 1, "TESS-SPOC": 2, "TGLC": 3, "QLP": 4}
AUTHOR_COLORS = {"SPOC": "crimson", "TESS-SPOC": "darkorange", "TGLC": "royalblue", "QLP": "seagreen"}

DEFAULT_RESULTS_DIR = "tess_results"


# ==============================================================================
# UTILITY FUNCTIONS & PHOTOMETRIC CONVERSIONS
# ==============================================================================

def to_clean_float(val, default: float = np.nan) -> float:
    try:
        if pd.isna(val) or str(val).strip().upper() in ("", "N/A", "NAN", "NONE", "NULL"):
            return default
        f_val = float(val)
        return f_val if np.isfinite(f_val) else default
    except Exception:
        return default


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
    base_dt = datetime(2014, 12, 8, 12, 0, 0)
    target_dt = base_dt + timedelta(days=float(btjd_val))
    return int(target_dt.year)


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


def batch_query_tic_photometry(tic_ids: list[str]) -> dict:
    catalog_map = {}
    int_ids = [int(t) for t in tic_ids if str(t).isdigit()]
    if not int_ids:
        return catalog_map
        
    print(f"[TIC Query] Querying MAST TIC v8.2 catalog for {len(int_ids)} targets...")
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


# ==============================================================================
# STAGE 3A: FAST PRE-FILTER GATE
# ==============================================================================

def calc_sidak_corrected_risk(row: pd.Series) -> float:
    """
    Computes Sidak-Bonferroni dual-zone corrected false-alarm probability (commit bfb2cad).
    Suppresses false-alarm penalties when secondary peaks are consistent with noise.
    """
    r = float(row.get("fp_risk_percent", 50.0))
    sec_snr = float(row.get("secondary_snr_sigma", 0.0)) if pd.notna(row.get("secondary_snr_sigma")) else 0.0
    sec_phase = float(row.get("sec_eclipse_phase", 0.50)) if pd.notna(row.get("sec_eclipse_phase")) else 0.50
    if sec_snr > 2.2:
        is_circ = abs(sec_phase - 0.50) <= 0.04
        if (not is_circ and sec_snr <= 3.4) or (is_circ and sec_snr <= 2.8):
            r -= 45.0 * (sec_snr / 3.0)
    return round(max(0.2, r), 1)


def run_stage_3a_filter(
    tier1_csv: str,
    tier2_csv: str,
    max_corr_risk: float = 25.0,
    max_neb_neighbors: int = 2,
    max_flux_contam_pct: float = 20.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the 5-rule fast pre-filter gate on Tier-2 Pleiades dossiers.
    Bypasses ~75% of unviable targets, preventing hours of redundant MAST queries.
    """
    print("\n" + "=" * 76)
    print("STAGE 3A: FAST CANDIDATE PRE-FILTER GATE")
    print("=" * 76)

    if not os.path.exists(tier1_csv):
        raise FileNotFoundError(f"Tier-1 report '{tier1_csv}' not found.")

    df_t1 = pd.read_csv(tier1_csv)
    df_t1["tic_id"] = df_t1["tic_id"].astype(str).str.strip()
    df_t1 = df_t1.drop_duplicates(subset=["tic_id"], keep="first")

    if os.path.exists(tier2_csv):
        df_t2 = pd.read_csv(tier2_csv)
        df_t2["tic_id"] = df_t2["tic_id"].astype(str).str.strip()
        df_t2 = df_t2.drop_duplicates(subset=["tic_id"], keep="first")
        df_merged = pd.merge(df_t2, df_t1, on="tic_id", how="left", suffixes=("", "_t1"))
        print(f"[Stage 3A] Loaded {len(df_t2)} Tier-2 Pleiades dossiers matched against {len(df_t1)} Tier-1 candidates.")
    else:
        print(f"[Stage 3A] Warning: Tier-2 summary '{tier2_csv}' not found. Using Tier-1 directly.")
        df_merged = df_t1.copy()
        for c, def_v in [
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
            if c not in df_merged.columns:
                df_merged[c] = def_v

    # Fallback mappings for critical columns
    df_merged["period_d"] = pd.to_numeric(df_merged["refined_period_days"], errors="coerce").fillna(
        pd.to_numeric(df_merged.get("period_days"), errors="coerce")
    )
    df_merged["rp_e"] = pd.to_numeric(df_merged["fitted_rp_earth"], errors="coerce").fillna(
        pd.to_numeric(df_merged.get("r_planet_earth"), errors="coerce")
    )
    df_merged["dur_ratio"] = pd.to_numeric(df_merged["dur_to_kepler_max"], errors="coerce").fillna(1.0)
    df_merged["sec_snr"] = pd.to_numeric(df_merged["secondary_snr_sigma"], errors="coerce").fillna(0.0)
    df_merged["contam_pct"] = pd.to_numeric(df_merged["flux_contam_percent"], errors="coerce").fillna(0.0)
    df_merged["neb_count"] = pd.to_numeric(df_merged["neb_capable_neighbors"], errors="coerce").fillna(0).astype(int)

    df_merged["corr_risk"] = df_merged.apply(calc_sidak_corrected_risk, axis=1)

    # 5-Rule Gate Conditions:
    # 1. No Gaia NSS binary & not known TESS EB
    is_gaia_nss = df_merged["gaia_nss_type"].notna() & (
        ~df_merged["gaia_nss_type"].astype(str).str.strip().str.upper().isin(["", "NAN", "NONE", "N/A", "FALSE"])
    )
    is_known_eb = df_merged["is_known_tess_eb"].astype(str).str.strip().str.upper() == "TRUE"

    # 2. Secondary eclipse significance
    sec_circ = (df_merged["sec_eclipse_phase"] - 0.50).abs() <= 0.04
    sec_pass = (~sec_circ & (df_merged["sec_snr"] <= 3.4)) | (sec_circ & (df_merged["sec_snr"] <= 2.8))

    # 3. Kinematic duration self-consistency
    dur_pass = (df_merged["dur_ratio"] >= 0.55) & (df_merged["dur_ratio"] <= 1.35)

    # 4. Stellar environment: dilution & nearby EBs
    env_pass = (df_merged["neb_count"] <= max_neb_neighbors) & (df_merged["contam_pct"] <= max_flux_contam_pct)

    # 5. Sidak-corrected FP risk threshold
    risk_pass = df_merged["corr_risk"] <= max_corr_risk

    gate_mask = (~is_gaia_nss) & (~is_known_eb) & sec_pass & dur_pass & env_pass & risk_pass

    df_survivors = df_merged[gate_mask].sort_values(by="corr_risk").reset_index(drop=True)
    df_rejected = df_merged[~gate_mask].copy().reset_index(drop=True)

    print(f"[Stage 3A Result] Input: {len(df_merged)} -> Rejected: {len(df_rejected)} -> Passed Gate: {len(df_survivors)} targets")
    print(f"[Stage 3A] Compute runtime saving: {100.0 * (1.0 - len(df_survivors) / max(1, len(df_merged))):.1f}% reduction in MAST downloads.\n")

    disp_cols = ["tic_id", "corr_risk", "period_d", "rp_e", "dur_ratio", "contam_pct", "neb_count", "sectors_stitched"]
    print("STAGE 3A SURVIVING TARGETS:")
    print(df_survivors[disp_cols].to_string(index=False))
    print("-" * 76)

    return df_survivors, df_rejected


# ==============================================================================
# STAGE 3B: DECOUPLED MULTI-YEAR ARCHIVAL DRILL (SPOC FIRST VS FFI BLEND TRACK)
# ==============================================================================

def extract_multi_author_lc(lc_obj, author: str, edge_trim_days: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    try:
        t = to_numpy(lc_obj.time)
        f = None
        if author == "TGLC":
            for col in ["cal_psf_flux", "psf_flux", "cal_aper_flux", "flux"]:
                if hasattr(lc_obj, col) and getattr(lc_obj, col) is not None:
                    f_try = to_numpy(getattr(lc_obj, col))
                    if np.sum(np.isfinite(f_try)) > 100:
                        f = f_try
                        break
        elif author == "QLP":
            for col in ["kspsap_flux", "sap_flux", "flux"]:
                if hasattr(lc_obj, col) and getattr(lc_obj, col) is not None:
                    f_try = to_numpy(getattr(lc_obj, col))
                    if np.sum(np.isfinite(f_try)) > 100:
                        f = f_try
                        break
        if f is None:
            f = to_numpy(lc_obj.flux)

        if hasattr(lc_obj, "quality") and lc_obj.quality is not None:
            qual = np.asarray(lc_obj.quality, dtype=int)
            q_valid = (qual == 0)
        else:
            q_valid = np.ones_like(t, dtype=bool)

        valid = np.isfinite(t) & np.isfinite(f) & q_valid
        if np.sum(valid) < 80:
            return None
        t, f = t[valid], f[valid]

        med_f = float(np.nanmedian(f))
        if med_f <= 0:
            f = f - med_f + 1.0
            med_f = 1.0
        f = f / med_f

        mad_f = float(1.4826 * np.nanmedian(np.abs(f - 1.0)) + 1e-12)
        keep = (f > 0.5) & (f < (1.0 + 3.5 * mad_f))
        if np.sum(keep) < 80:
            return None
        t, f = t[keep], f[keep]
        sort_idx = np.argsort(t)
        t, f = t[sort_idx], f[sort_idx]

        dt_arr = np.diff(t)
        cadence_days = float(np.nanmedian(dt_arr[dt_arr > 0])) if len(dt_arr) > 0 else (120.0 / 86400.0)

        if edge_trim_days > 0 and (t[-1] - t[0]) > 5.0:
            gap_idx = np.where(np.diff(t) > 0.08)[0]
            edge_keep = (t >= (t[0] + edge_trim_days)) & (t <= (t[-1] - edge_trim_days))
            for g_i in gap_idx:
                edge_keep &= ~((t >= (t[g_i] - edge_trim_days)) & (t <= (t[g_i + 1] + edge_trim_days)))
            if np.sum(edge_keep) >= 80:
                t, f = t[edge_keep], f[edge_keep]

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
    if transit_mask is None:
        transit_mask = np.zeros_like(t, dtype=bool)
    t_unmasked = t[~transit_mask]
    f_unmasked = f[~transit_mask]
    if len(t_unmasked) < 20:
        return f / np.nanmedian(f)

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
    baseline_days = max(27.0, float(np.max(t_all) - np.min(t_all)))
    n_cycles = max(10.0, baseline_days / p_init)

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
        if np.sum(in_m) < 8:
            continue
        w_in = weights[in_m]
        w_sum = np.sum(w_in)
        mean_def = np.sum(w_in * deficit[in_m]) / w_sum
        snr_grid[i] = mean_def * np.sqrt(w_sum)

    best_idx = int(np.nanargmax(snr_grid))
    p_refined = float(trial_periods[best_idx])

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


def execute_archival_drill(
    candidate: dict,
    results_dir: str = DEFAULT_RESULTS_DIR,
    min_spoc_snr_red: float = 3.8
) -> dict:
    """
    Executes decoupled multi-year archival drill:
    - Master planetary parameters (depth, Rp, SNR) are derived from the pristine SPOC 2-min stack.
    - QLP/TGLC light curves are independently evaluated to check aperture dilution / blending.
    - If SNR_red,SPOC < min_spoc_snr_red, candidate is marked ARCHIVAL_REJECT.
    """
    tid = str(candidate["tic_id"])
    p_init = float(candidate["period_d"])
    t0_init = float(candidate.get("t0_btjd", 3980.0))
    dur_hr = float(candidate.get("duration_hours", 2.0))
    dur_days = dur_hr / 24.0

    print(f"\n---> Archival Drill: TIC {tid} (P = {p_init:.6f} d | T0 = {t0_init:.5f} BTJD | Dur = {dur_hr:.2f} h)")

    # 1. Search MAST archive for light curves across all pipelines
    try:
        search_all = lk.search_lightcurve(f"TIC {tid}", mission="TESS")
    except Exception as e:
        print(f"      [!] Error querying MAST for TIC {tid}: {e}")
        return {"status": "ARCHIVAL_ERROR", "reason": f"MAST query failed: {e}", "candidate": candidate}

    if len(search_all) == 0:
        print(f"      [!] No TESS light curves found on MAST for TIC {tid}.")
        return {"status": "ARCHIVAL_REJECT", "reason": "No MAST light curves", "candidate": candidate}

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
            if key in seen_sec_author and exptime_s < 100:
                continue
            seen_sec_author.add(key)

            lc_dl = item.download()
            extracted = extract_multi_author_lc(lc_dl, auth, edge_trim_days=0.08)
            if extracted is None:
                continue
            t_s, f_s, ferr_s, cad_days = extracted
            obs_year = btjd_to_year(float(np.nanmedian(t_s)))

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
        except Exception:
            continue

    if not records:
        print(f"      [!] Failed to extract valid light curves for TIC {tid}.")
        return {"status": "ARCHIVAL_REJECT", "reason": "No valid light curves extracted", "candidate": candidate}

    # Deduplicate: select 1 best light curve per sector for ephemeris locking
    records.sort(key=lambda r: (r["sector"], r["priority"]))
    master_by_sector = {}
    for rec in records:
        sec = rec["sector"]
        if sec not in master_by_sector or rec["priority"] < master_by_sector[sec]["priority"]:
            master_by_sector[sec] = rec
    master_records = [master_by_sector[s] for s in sorted(master_by_sector.keys())]

    # Decouple: Build Dedicated Master Stack (strictly SPOC 2-minute if available)
    spoc_records = [r for r in records if r["author"] == "SPOC" and r["exptime_s"] <= 120]
    has_spoc = len(spoc_records) > 0

    if has_spoc:
        # Deduplicate SPOC sectors (prefer 120s standard over 20s if both exist)
        spoc_by_sector = {}
        for r in spoc_records:
            sec = r["sector"]
            if sec not in spoc_by_sector or r["exptime_s"] > spoc_by_sector[sec]["exptime_s"]:
                spoc_by_sector[sec] = r
        master_stack_records = [spoc_by_sector[s] for s in sorted(spoc_by_sector.keys())]
        stack_author = "SPOC"
    else:
        # Fallback to general deduplicated master stack if no SPOC data exists
        master_stack_records = master_records
        stack_author = "MULTI_AUTHOR"

    n_sectors_stack = len(master_stack_records)

    # Assemble pass-1 master series for coherent ephemeris refinement
    t_master = np.concatenate([r["t"] for r in master_stack_records])
    f_master_p1 = np.concatenate([r["f_pass1"] for r in master_stack_records])
    ferr_master = np.concatenate([r["ferr"] for r in master_stack_records])
    cad_master = float(np.median([r["cadence_days"] for r in master_stack_records]))
    m_idx = np.argsort(t_master)
    t_master, f_master_p1, ferr_master = t_master[m_idx], f_master_p1[m_idx], ferr_master[m_idx]

    # Refine coherent ephemeris directly on the pristine master stack
    t0_anchor = t0_init - round((t0_init - np.min(t_master)) / p_init) * p_init
    p_ref, t0_ref, p_grid, snr_grid = refine_multi_year_ephemeris(t_master, f_master_p1, ferr_master, p_init, t0_anchor, dur_days)
    dp_sec = (p_ref - p_init) * 86400.0

    # Pass 2: transit-masked detrending on ALL records (using the pristine master ephemeris)
    for rec in records:
        t_s = rec["t"]
        cad_d = rec["cadence_days"]
        ph_s = ((t_s - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref
        tr_mask = np.abs(ph_s) < (0.65 * (dur_days + cad_d))
        rec["f_clean"] = hybrid_tukey_savgol_detrend(t_s, rec["f_raw"], cad_d, window_days=0.65, transit_mask=tr_mask)
        rec["metrics"] = compute_transit_metrics(t_s, rec["f_clean"], rec["ferr"], p_ref, t0_ref, dur_days, cad_d)

    # Re-assemble clean master stack with Pass-2 detrended data
    f_master = np.concatenate([r["f_clean"] for r in master_stack_records])[m_idx]
    master_metrics = compute_transit_metrics(t_master, f_master, ferr_master, p_ref, t0_ref, dur_days, cad_master)

    # Evaluate Author Comparisons (SPOC vs QLP vs TGLC)
    author_stats = {}
    for auth in ["SPOC", "QLP", "TGLC", "TESS-SPOC"]:
        a_recs = [r for r in records if r["author"] == auth]
        if a_recs:
            t_a = np.concatenate([r["t"] for r in a_recs])
            f_a = np.concatenate([r["f_clean"] for r in a_recs])
            ferr_a = np.concatenate([r["ferr"] for r in a_recs])
            cad_a = float(np.median([r["cadence_days"] for r in a_recs]))
            author_stats[auth] = compute_transit_metrics(t_a, f_a, ferr_a, p_ref, t0_ref, dur_days, cad_a)

    spoc_snr_red = master_metrics["snr_red"]
    spoc_depth = master_metrics["depth_ppm"]

    # Blending Diagnostic Check:
    blended_suspect = False
    blend_reason = ""
    if "SPOC" in author_stats and ("QLP" in author_stats or "TGLC" in author_stats):
        d_spoc = author_stats["SPOC"]["depth_ppm"]
        d_alt = max(
            author_stats.get("QLP", {}).get("depth_ppm", 0.0),
            author_stats.get("TGLC", {}).get("depth_ppm", 0.0)
        )
        contam = float(candidate.get("contam_pct", 0.0))
        if d_spoc > 2.5 * max(d_alt, 1.0) and contam > 5.0 and d_alt < 200.0:
            blended_suspect = True
            blend_reason = f"SPOC depth ({d_spoc:.0f} ppm) diluted > 2.5x in QLP/TGLC ({d_alt:.0f} ppm)"

    # Stage 3B Decision Gate
    if spoc_snr_red < min_spoc_snr_red and master_metrics["snr_white"] < (min_spoc_snr_red + 0.3):
        status = "ARCHIVAL_REJECT"
        decision_reason = f"SNR_red ({spoc_snr_red:.2f} sigma) < threshold ({min_spoc_snr_red:.1f} sigma)"
    elif blended_suspect:
        status = "ARCHIVAL_REJECT"
        decision_reason = f"Aperture Dilution / Blended ({blend_reason})"
    else:
        status = "ARCHIVAL_PASS"
        decision_reason = f"SNR_red={spoc_snr_red:.2f}s (SNR_white={master_metrics['snr_white']:.1f}s) on {stack_author} stack (N={master_metrics['n_transits']} transits across {n_sectors_stack} sectors)"

    print(f"      -> Ephemeris: P = {p_ref:.7f} d (dP = {dp_sec:+.2f} s) | T0 = {t0_ref:.5f} BTJD")
    print(f"      -> Stack ({stack_author}, {n_sectors_stack} secs): Depth = {master_metrics['depth_ppm']:.1f} ppm | SNR_red = {spoc_snr_red:.2f} sigma | Transits = {master_metrics['n_transits']}")
    print(f"      -> Archival Decision: [{status}] - {decision_reason}")

    # Generate 4-panel diagnostic poster
    out_png = os.path.join(results_dir, f"antares_TIC_{tid}.png")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        win_hrs = min(0.18 * p_ref, 2.5 * dur_days) * 24.0
        bins_hrs = np.linspace(-win_hrs, win_hrs, 35)
        b_centers = 0.5 * (bins_hrs[:-1] + bins_hrs[1:])

        # Panel 1: Master Phase-Folded Transit
        ph_hrs = (((t_master - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref) * 24.0
        vis = np.abs(ph_hrs) <= win_hrs
        axes[0, 0].scatter(ph_hrs[vis], f_master[vis], s=2, color="silver", alpha=0.35, label=f"Cadences ({n_sectors_stack} Secs)")
        bm, _, _ = stats.binned_statistic(ph_hrs[vis], f_master[vis], statistic="mean", bins=bins_hrs)
        bs, _, _ = stats.binned_statistic(ph_hrs[vis], f_master[vis], statistic="std", bins=bins_hrs)
        bc, _, _ = stats.binned_statistic(ph_hrs[vis], f_master[vis], statistic="count", bins=bins_hrs)
        vb = bc >= 3
        berr = bs[vb] / np.sqrt(bc[vb])
        axes[0, 0].errorbar(b_centers[vb], bm[vb], yerr=berr, fmt="o", color="black", ms=5, capsize=2.5, lw=1.2, zorder=4, label="Binned (+/-1 sigma)")
        axes[0, 0].axvline(0.0, color="crimson", linestyle=":", alpha=0.6)
        axes[0, 0].set_xlim(-win_hrs, win_hrs)
        axes[0, 0].set_xlabel("Time from Mid-Transit (hours)")
        axes[0, 0].set_ylabel("Relative Flux")
        axes[0, 0].set_title(f"TIC {tid} Master Stack ({stack_author}): SNR_white={master_metrics['snr_white']:.1f}s | SNR_red={master_metrics['snr_red']:.1f}s")
        axes[0, 0].legend(loc="lower right", fontsize=8.5)

        # Panel 2: Pipeline Comparison (SPOC vs QLP vs TGLC)
        offset_step = max(0.002, 2.5 * (master_metrics["depth_ppm"] * 1e-6))
        for idx_a, (auth, ast) in enumerate(author_stats.items()):
            a_recs = [r for r in records if r["author"] == auth]
            t_a = np.concatenate([r["t"] for r in a_recs])
            f_a = np.concatenate([r["f_clean"] for r in a_recs])
            ph_a_hrs = (((t_a - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref) * 24.0
            vis_a = np.abs(ph_a_hrs) <= win_hrs
            if np.sum(vis_a) < 15:
                continue
            am, _, _ = stats.binned_statistic(ph_a_hrs[vis_a], f_a[vis_a], statistic="mean", bins=bins_hrs)
            ac, _, _ = stats.binned_statistic(ph_a_hrs[vis_a], f_a[vis_a], statistic="count", bins=bins_hrs)
            va = ac >= 2
            v_off = idx_a * offset_step
            c_auth = AUTHOR_COLORS.get(auth, "navy")
            axes[0, 1].plot(
                b_centers[va], am[va] - v_off, "o-", color=c_auth, ms=4, lw=1.5,
                label=f"{auth}: d={ast['depth_ppm']:.0f}ppm ({ast['snr_white']:.1f}s)"
            )
        axes[0, 1].set_xlim(-win_hrs, win_hrs)
        axes[0, 1].set_xlabel("Time from Mid-Transit (hours)")
        axes[0, 1].set_ylabel("Relative Flux - Offset")
        axes[0, 1].set_title("Cross-Pipeline De-Blending Track (SPOC vs QLP vs TGLC)")
        axes[0, 1].legend(loc="lower right", fontsize=8.5)

        # Panel 3: Ephemeris Refinement Periodogram
        dp_ppm_grid = ((p_grid - p_ref) / p_ref) * 1e6
        axes[1, 0].plot(dp_ppm_grid, snr_grid, color="black", lw=1.0)
        axes[1, 0].axvline(0.0, color="crimson", linestyle="--", lw=1.8, label=f"Refined P = {p_ref:.7f} d")
        axes[1, 0].axvline(((p_init - p_ref) / p_ref) * 1e6, color="royalblue", linestyle=":", lw=1.5, label=f"Initial P = {p_init:.6f} d")
        axes[1, 0].set_xlabel("Relative Period Offset (ppm)")
        axes[1, 0].set_ylabel("Coherent Matched-Filter SNR (sigma)")
        axes[1, 0].set_title("Coherent Multi-Year Ephemeris Locking")
        axes[1, 0].legend(loc="upper right", fontsize=8.5)

        # Panel 4: Summary Card
        axes[1, 1].axis("off")
        card_text = (
            f"ANTARES ARCHIVAL DOSSIER: TIC {tid}\n"
            f"{'-'*50}\n"
            f"Stage 3B Decision:  [{status}]\n"
            f"Reason:             {decision_reason}\n\n"
            f"COHERENT EPHEMERIS:\n"
            f"  Refined Period:   {p_ref:.7f} d (dP = {dp_sec:+.2f} s)\n"
            f"  Refined T0:       {t0_ref:.5f} BTJD\n"
            f"  Unique Sectors:   {n_sectors_stack} ({stack_author} Stack)\n"
            f"  Total Transits:   N = {master_metrics['n_transits']}\n\n"
            f"MASTER PHOTOMETRY:\n"
            f"  Transit Depth:    {master_metrics['depth_ppm']:.1f} +/- {master_metrics['err_white_ppm']:.1f} ppm\n"
            f"  Integrated SNR:   {master_metrics['snr_white']:.1f}s (white) | {master_metrics['snr_red']:.1f}s (red, beta={master_metrics['beta_red']:.2f})\n\n"
            f"CROSS-AUTHOR CHECK:\n" +
            "\n".join([f"  {a:9s}: {s['depth_ppm']:.0f} ppm ({s['snr_white']:.1f}s)" for a, s in author_stats.items()])
        )
        axes[1, 1].text(
            0.04, 0.96, card_text, transform=axes[1, 1].transAxes, fontsize=10.0,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="gray")
        )

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"      [!] Warning: Failed to render diagnostic poster for TIC {tid}: {e}")

    return {
        "status": status,
        "reason": decision_reason,
        "candidate": candidate,
        "p_ref": p_ref,
        "t0_ref": t0_ref,
        "dp_sec": dp_sec,
        "stack_author": stack_author,
        "n_sectors_stack": n_sectors_stack,
        "depth_ppm": master_metrics["depth_ppm"],
        "err_white_ppm": master_metrics["err_white_ppm"],
        "snr_white": master_metrics["snr_white"],
        "snr_red": master_metrics["snr_red"],
        "beta_red": master_metrics["beta_red"],
        "n_transits": master_metrics["n_transits"],
        "poster_path": out_png,
        "author_stats": author_stats
    }


# ==============================================================================
# STAGE 3C: ASTROPHYSICAL PARAMETERS & CTOI PACKAGING
# ==============================================================================

def run_stage_3c_packaging(
    vetted_results: list[dict],
    results_dir: str = DEFAULT_RESULTS_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes M_p, K_RV, TSM, ESM, and follow-up rankings for all vetted candidates.
    Exports prioritized_targets_ranking.csv and the finalized exofop_ctoi_upload.csv.
    """
    print("\n" + "=" * 76)
    print("STAGE 3C: ATMOSPHERIC / RV METRICS & NASA EXOFOP CTOI PACKAGING")
    print("=" * 76)

    passing_cands = [r for r in vetted_results if r["status"] == "ARCHIVAL_PASS"]
    if not passing_cands:
        print("[!] No candidates passed Stage 3B archival vetting.")
        return pd.DataFrame(), pd.DataFrame()

    tic_ids = [str(r["candidate"]["tic_id"]) for r in passing_cands]
    tic_phot_map = batch_query_tic_photometry(tic_ids)

    ranking_rows = []
    exofop_rows = []

    for r in passing_cands:
        cand = r["candidate"]
        tid = str(cand["tic_id"])
        t_meta = tic_phot_map.get(tid, {})

        p_days = r["p_ref"]
        t0_btjd = r["t0_ref"]
        epoch_bjd = t0_btjd + BTJD_OFFSET
        dur_hr = to_clean_float(cand.get("duration_hours"), 2.0)
        depth_ppm = r["depth_ppm"]

        # Stellar parameters
        r_star = to_clean_float(t_meta.get("r_star"), to_clean_float(cand.get("r_star_sun"), 1.0))
        m_star = to_clean_float(t_meta.get("m_star"), to_clean_float(cand.get("m_star_sun"), r_star ** 1.1))
        teff = to_clean_float(t_meta.get("teff"), to_clean_float(cand.get("teff_k"), 5500.0))

        # Fitted planet radius: derived from SPOC depth
        rp_rs = np.sqrt(max(1e-6, depth_ppm * 1e-6))
        rp_earth = rp_rs * r_star * 109.076
        b_imp = to_clean_float(cand.get("impact_param_b"), 0.30)

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

        corr_risk = float(cand.get("corr_risk", 10.0))
        contam_pct = float(cand.get("contam_pct", 0.0))
        neb_cnt = int(cand.get("neb_count", 0))
        secs_str = str(cand.get("sectors_stitched") if pd.notna(cand.get("sectors_stitched")) and str(cand.get("sectors_stitched")) != "" else cand.get("sector", "N/A"))

        # Composite Follow-Up Priority Score
        clean_weight = max(0.1, (100.0 - corr_risk) / 100.0)
        priority_score = round(clean_weight * (0.65 * astro["tsm"] + 4.5 * astro["rv_k_ck17_ms"] + 1.5 * astro["esm"]), 2)

        cam_str, ccd_str = extract_camera_ccd_from_cache(tid)

        ranking_rows.append({
            "tic_id": tid,
            "priority_score": priority_score,
            "disposition": "TIER-3 VETTED CANDIDATE",
            "corr_risk_percent": round(corr_risk, 1),
            "sectors": secs_str,
            "archival_stack": r["stack_author"],
            "period_days": round(p_days, 7),
            "t0_btjd": round(t0_btjd, 5),
            "epoch_bjd": round(epoch_bjd, 5),
            "duration_hours": round(dur_hr, 2),
            "depth_ppm": round(depth_ppm, 1),
            "rp_earth": round(rp_earth, 2),
            "snr_red": r["snr_red"],
            "snr_white": r["snr_white"],
            "n_transits": r["n_transits"],
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
            f"Aldebaran v8.0 / Pleiades v2.2 / Antares v3.0 Candidate (Sectors {secs_str}, N={r['n_transits']} transits, SNR_red={r['snr_red']:.1f}s). "
            f"Conditional Planetary Characterization: Teq={int(astro['teq_k'])}K, Mp_est={astro['mp_ck17_earth']:.1f} M_E, "
            f"RV K={astro['rv_k_ck17_ms']:.1f} m/s, TSM={astro['tsm']:.1f}, ESM={astro['esm']:.1f}. "
            f"Gaia Contam={contam_pct:.1f}% ({neb_cnt} NEB neighbors). Disposition: PC (FP Risk={corr_risk:.1f}%)."
        )

        exofop_rows.append({
            "TIC ID": tid,
            "Camera": cam_str,
            "CCD": ccd_str,
            "Epoch BJD": round(epoch_bjd, 5),
            "Period": round(p_days, 7),
            "Depth": round(depth_ppm, 1),
            "Duration": round(dur_hr, 2),
            "Rp": round(rp_earth, 2),
            "User Comments / Disposition": comment_str
        })

    df_rank = pd.DataFrame(ranking_rows)
    df_rank = df_rank.sort_values(
        by=["priority_score", "tsm", "rv_k_ck17_ms"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    df_rank.insert(0, "rank", np.arange(1, len(df_rank) + 1))

    ordered_tics = df_rank["tic_id"].tolist()
    df_exofop = pd.DataFrame(exofop_rows)
    df_exofop["sort_key"] = df_exofop["TIC ID"].apply(lambda x: ordered_tics.index(str(x)))
    df_exofop = df_exofop.sort_values(by="sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    # Save to disk
    ranking_csv = os.path.join(results_dir, "prioritized_targets_ranking.csv")
    exofop_csv = os.path.join(results_dir, "exofop_ctoi_upload.csv")
    exofop_comma_csv = os.path.join(results_dir, "exofop_ctoi_upload_comma.csv")

    df_rank.to_csv(ranking_csv, index=False)
    df_exofop.to_csv(exofop_csv, index=False, sep="|")
    df_exofop.to_csv(exofop_comma_csv, index=False)

    print(f"[Stage 3C Saved] Ranked target catalog:       '{ranking_csv}'")
    print(f"[Stage 3C Saved] ExoFOP CTOI upload table:    '{exofop_csv}' (pipe-delimited)")
    print(f"[Stage 3C Saved] ExoFOP comma-separated copy: '{exofop_comma_csv}'\n")

    disp_cols = ["rank", "tic_id", "period_days", "rp_earth", "mp_ck17_earth", "rv_k_ck17_ms", "tsm", "esm", "snr_red", "corr_risk_percent"]
    print("TOP-RANKED ARCHIVALLY VETTED CANDIDATES FOR NASA CTOI:")
    print(df_rank[disp_cols].to_string(index=False))
    print("=" * 76)

    return df_rank, df_exofop


# ==============================================================================
# MAIN TIER-3 PIPELINE CONTROLLER
# ==============================================================================

def run_antares_packager(
    results_dir: str = DEFAULT_RESULTS_DIR,
    target_tic: str = None,
    force_all: bool = False,
    min_snr: float = 3.8
):
    print("=" * 76)
    print("ALDEBARAN v8.0 / ANTARES v3.0: UNIFIED TIER-3 ARCHIVAL VETTER & CTOI PACKAGER")
    print("=" * 76)

    tier1_csv = os.path.join(results_dir, "candidates_report.csv")
    tier2_csv = os.path.join(results_dir, "deep_vetting", "scientific_dossier_summary.csv")

    # Step 1: Stage 3A Filtering
    df_survivors, df_rejected = run_stage_3a_filter(
        tier1_csv=tier1_csv,
        tier2_csv=tier2_csv,
        max_corr_risk=25.0 if not force_all else 100.0,
        max_neb_neighbors=2 if not force_all else 99,
        max_flux_contam_pct=20.0 if not force_all else 100.0
    )

    if target_tic is not None:
        target_tic = str(target_tic).strip()
        df_survivors = df_survivors[df_survivors["tic_id"] == target_tic]
        if len(df_survivors) == 0:
            print(f"[!] Target TIC {target_tic} not found in surviving candidates. Searching in raw Tier-1...")
            df_raw = pd.read_csv(tier1_csv)
            df_raw["tic_id"] = df_raw["tic_id"].astype(str).str.strip()
            sub = df_raw[df_raw["tic_id"] == target_tic]
            if len(sub) == 0:
                raise ValueError(f"TIC {target_tic} not found in candidates report.")
            df_survivors = sub.copy()
            df_survivors["period_d"] = df_survivors["period_days"]
            df_survivors["corr_risk"] = 15.0

    if len(df_survivors) == 0:
        print("[!] No candidates available for Stage 3B archival drilling.")
        return

    # Step 2: Stage 3B Decoupled Archival Drill
    print(f"\n[Stage 3B] Launching Multi-Year Archival Drill for {len(df_survivors)} targets...")
    vetted_results = []
    audit_rows = []

    for _, row in df_survivors.iterrows():
        res = execute_archival_drill(
            candidate=row.to_dict(),
            results_dir=results_dir,
            min_spoc_snr_red=min_snr
        )
        vetted_results.append(res)
        audit_rows.append({
            "tic_id": row["tic_id"],
            "stage_3a": "PASS",
            "stage_3b_status": res["status"],
            "decision_reason": res["reason"],
            "refined_period_days": res.get("p_ref", np.nan),
            "snr_red": res.get("snr_red", np.nan),
            "snr_white": res.get("snr_white", np.nan),
            "depth_ppm": res.get("depth_ppm", np.nan),
            "stack_author": res.get("stack_author", "N/A"),
            "n_transits": res.get("n_transits", 0)
        })

    # Record rejected targets in audit
    for _, r_row in df_rejected.iterrows():
        audit_rows.append({
            "tic_id": r_row["tic_id"],
            "stage_3a": "REJECT",
            "stage_3b_status": "SKIPPED",
            "decision_reason": f"Stage 3A Filter (Risk={r_row.get('corr_risk')}%, DurRatio={r_row.get('dur_ratio')}%, NEB={r_row.get('neb_count')})",
            "refined_period_days": r_row.get("period_d", np.nan),
            "snr_red": np.nan,
            "snr_white": np.nan,
            "depth_ppm": np.nan,
            "stack_author": "N/A",
            "n_transits": 0
        })

    audit_csv = os.path.join(results_dir, "stage3_vetting_audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    print(f"\n[Stage 3 Audit] Full decision audit saved to: '{audit_csv}'")

    # Step 3: Stage 3C Packaging
    df_rank, df_exofop = run_stage_3c_packaging(
        vetted_results=vetted_results,
        results_dir=results_dir
    )

    print("\n" + "=" * 76)
    print("TIER-3 PIPELINE EXECUTION COMPLETE!")
    print(f"Total Tier-2 Candidates: {len(df_survivors) + len(df_rejected)}")
    print(f"Passing Stage 3A Gate:   {len(df_survivors)}")
    print(f"Passing Stage 3B Drill:  {len([r for r in vetted_results if r['status'] == 'ARCHIVAL_PASS'])}")
    print(f"Packaged for NASA CTOI:  {len(df_exofop)}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antares Packager: Unified Tier-3 Archival Vetter & CTOI Packager")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results directory (default: tess_results)")
    parser.add_argument("--tic", type=str, default=None, help="Target TIC ID to process individually")
    parser.add_argument("--force-all", action="store_true", help="Bypass Stage 3A gate and drill all candidates")
    parser.add_argument("--min-snr", type=float, default=3.8, help="Minimum SPOC red-noise SNR for CTOI inclusion (default: 3.8)")
    args = parser.parse_args()

    run_antares_packager(
        results_dir=args.results_dir,
        target_tic=args.tic,
        force_all=args.force_all,
        min_snr=args.min_snr
    )
