import os
import json
from depth_contract import fit_depth_fields
import sys
import ssl
import glob
import shutil
import argparse
import warnings
from datetime import datetime, timedelta

import logging
import certifi

logger = logging.getLogger(__name__)

# Priority P1: Enforce verified TLS context anchored to certifi CA bundle
CA_BUNDLE_PATH = certifi.where()
os.environ["SSL_CERT_FILE"] = CA_BUNDLE_PATH
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE_PATH
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=CA_BUNDLE_PATH)
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
from ephemeris_contract import BTJD_OFFSET, btjd_to_year as contract_btjd_to_year, fold_phase_centered, compute_orbit_number, days_to_hours, hours_to_days
from lightcurve_helpers import (  # C9: shared copies, see lightcurve_helpers.py
    to_numpy, extract_multi_author_lc, hybrid_tukey_savgol_detrend, _GL5_NODES, _GL5_WEIGHTS,
    light_curve_product_failure, EXTRACTION_TOO_FEW_POINTS,
    annulus_300_transit_model, refine_multi_year_ephemeris, compute_transit_metrics,
)

RESULTS_DIR = "tess_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Author priority for deduplicating overlapping sectors in the Master Stack
AUTHOR_PRIORITY = {"SPOC": 1, "TESS-SPOC": 2, "TGLC": 3, "QLP": 4}
AUTHOR_COLORS = {"SPOC": "crimson", "TESS-SPOC": "darkorange", "TGLC": "royalblue", "QLP": "seagreen"}


def btjd_to_year(btjd_val: float) -> int:
    """Converts TESS Barycentric Julian Date (BTJD = BJD - 2457000.0) to calendar year."""
    return contract_btjd_to_year(btjd_val)


def get_stellar_limb_darkening(tic_id: str) -> dict:
    """Queries TIC v8.2 for R_star, M_star, Teff, Tmag and estimates quadratic limb darkening."""
    info = {"r_star": 0.61, "m_star": 0.60, "teff": 3958.0, "tmag": 13.8, "u1": 0.57, "u2": 0.20}
    info["stellar_parameter_sources"] = dict.fromkeys(("r_star", "m_star", "teff", "tmag"), "fixed_initial_assumption")
    info["catalog_error"] = ""
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
                info["stellar_parameter_sources"]["r_star"] = "TIC_catalog"
                info["m_star"] = m_val if (np.isfinite(m_val) and m_val > 0) else (r_val ** 1.1)
                info["stellar_parameter_sources"]["m_star"] = "TIC_catalog" if (np.isfinite(m_val) and m_val > 0) else "radius_power_law_assumption"
            if np.isfinite(t_val) and t_val > 0:
                info["teff"] = t_val
                info["stellar_parameter_sources"]["teff"] = "TIC_catalog"
                info["u1"] = float(np.clip(0.45 - 0.00008 * (t_val - 5500.0), 0.15, 0.70))
                info["u2"] = float(np.clip(0.25 + 0.00003 * (t_val - 5500.0), 0.05, 0.45))
            if np.isfinite(tmag_val):
                info["tmag"] = tmag_val
                info["stellar_parameter_sources"]["tmag"] = "TIC_catalog"
    except Exception as exc:
        info["catalog_error"] = type(exc).__name__ + ": " + str(exc)
    return info


def run_antares_detective(tic_id: str, p_init: float, t0_init: float, dur_hr: float):
    dur_days = hours_to_days(dur_hr)
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

    products_failed = []  # C1g: every product that could not be used, with its stage and reason
    for item in search_all:
        sec_num, auth, stage = -1, "", "PRODUCT_METADATA"
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

            stage = "DOWNLOAD"
            lc_dl = item.download()
            stage = "EXTRACTION"
            extracted = extract_multi_author_lc(lc_dl, auth, edge_trim_days=0.08, raise_errors=True)
            if extracted is None:
                products_failed.append(light_curve_product_failure(sec_num, auth, stage, EXTRACTION_TOO_FEW_POINTS))
                continue
            t_s, f_s, ferr_s, cad_days = extracted
            obs_year = btjd_to_year(float(np.nanmedian(t_s)))

            # Pass 1 Detrending (unmasked robust Tukey slider to prepare for fine ephemeris lock)
            stage = "DETRENDING"
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
            products_failed.append(light_curve_product_failure(sec_num, auth, stage, e))
            continue

    for failed in products_failed:
        print(f"      -> [!] Product not used: Sector {failed['sector']} | Author: {failed['author'] or '?'} | "
              f"{failed['stage']}: {failed['reason']}")
    if not records:
        failed_note = f" ({len(products_failed)} product(s) failed: {products_failed})" if products_failed else ""
        raise RuntimeError(f"Could not extract any valid SPOC/QLP/TGLC light curves for TIC {tic_id}{failed_note}.")

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

    fit_converged, fit_error = False, ""
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
        if not np.all(np.isfinite(popt)) or rp_rs_fit <= 0 or dur_fit <= 0:
            raise ValueError("Nonfinite or nonphysical fit parameters")
        fit_converged = True
    except Exception as exc:
        fit_error = type(exc).__name__ + ": " + str(exc)
        rp_rs_fit, b_fit, dur_fit = rp_rs_guess, 0.35, dur_days
    depth_fields = fit_depth_fields(
        rp_rs_fit, master_model(np.array([0.0]), rp_rs_fit, b_fit, dur_fit)[0],
        master_metrics["depth_ppm"], fit_converged, fit_error,
        measured_source="antares_master_inverse_variance_box")

    host_sources = star_info.get("stellar_parameter_sources", {"r_star": "UNKNOWN"})
    depth_fields["stellar_parameter_provenance"] = json.dumps(host_sources, sort_keys=True)
    depth_fields["stellar_catalog_error"] = star_info.get("catalog_error", "")
    host_radius_measured = host_sources.get("r_star") == "TIC_catalog"
    if fit_converged and not host_radius_measured:
        depth_fields["fitted_radius_kind"] = "geometric_fit_with_assumed_or_unknown_host"
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
    win_hrs = days_to_hours(min(0.18 * p_ref, 2.5 * dur_fit))
    bins_hrs = np.linspace(-win_hrs, win_hrs, 35)
    b_centers = 0.5 * (bins_hrs[:-1] + bins_hrs[1:])

    # Panel 1 (0, 0): Master Multi-Year Coherent Phase-Folded Transit
    ph_m_hrs = days_to_hours(ph_master)
    vis_m = np.abs(ph_m_hrs) <= win_hrs
    axes[0, 0].scatter(ph_m_hrs[vis_m], f_master[vis_m], s=2, color="silver", alpha=0.30, label=f"Cadences ({len(master_records)} Sectors)")
    bm, _, _ = stats.binned_statistic(ph_m_hrs[vis_m], f_master[vis_m], statistic="mean", bins=bins_hrs)
    bs, _, _ = stats.binned_statistic(ph_m_hrs[vis_m], f_master[vis_m], statistic="std", bins=bins_hrs)
    bc, _, _ = stats.binned_statistic(ph_m_hrs[vis_m], f_master[vis_m], statistic="count", bins=bins_hrs)
    vb = bc >= 3
    berr = bs[vb] / np.sqrt(bc[vb])
    axes[0, 0].errorbar(b_centers[vb], bm[vb], yerr=berr, fmt="o", color="black", ms=5, capsize=2.5, lw=1.2, zorder=4, label="Master Binned (+/-1 sigma)")

    ph_grid_d = np.linspace(hours_to_days(-win_hrs), hours_to_days(win_hrs), 500)
    axes[0, 0].plot(
        days_to_hours(ph_grid_d),
        master_model(ph_grid_d, rp_rs_fit, b_fit, dur_fit),
        color="crimson", lw=2.6, zorder=5,
        label=f"300-Ring Model (fit={fit_converged}): Rp={rp_earth_fit:.2f}+/-{sigma_rp_earth:.2f} R_E"
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
        ph_y_hrs = days_to_hours(((yst["t"] - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref)
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
        axes[0, 1].plot(days_to_hours(ph_grid_d), master_model(ph_grid_d, rp_rs_fit, b_fit, dur_fit) - v_off, color="black", linestyle="--", lw=1.0, alpha=0.5)
    axes[0, 1].axvline(0.0, color="crimson", linestyle=":", alpha=0.6)
    axes[0, 1].set_xlim(-win_hrs, win_hrs)
    axes[0, 1].set_xlabel("Time from Mid-Transit (hours)")
    axes[0, 1].set_ylabel("Relative Flux - Vertical Offset")
    axes[0, 1].set_title(f"Secular Stability Across Mission Years ({min(years_unique)}-{max(years_unique)})")
    axes[0, 1].legend(loc="lower right", fontsize=8.5)

    # Panel 3 (1, 0): Cross-Pipeline De-Blending Comparison (SPOC vs TGLC vs QLP)
    for idx_a, auth in enumerate(authors_unique):
        ast = author_stats[auth]
        ph_a_hrs = days_to_hours(((ast["t"] - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref)
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
        axes[1, 0].plot(days_to_hours(ph_grid_d), master_model(ph_grid_d, rp_rs_fit, b_fit, dur_fit) - v_off, color="gray", linestyle="--", lw=1.0, alpha=0.6)
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
        "MULTI-EPOCH CANDIDATE SIGNAL CRITERIA MET (not independent confirmation)"
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
        f"  Radius (fit={fit_converged}, host_measured={host_radius_measured}):     Rp = {rp_earth_fit:.2f} +/- {sigma_rp_earth:.2f} R_Earth (b = {b_fit:.2f})\n\n"
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
    for summary in summary_rows:
        if summary.get("group_type") == "MASTER_DEDUPLICATED_ALL_YEARS":
            summary.update(depth_fields)
            summary["fitted_rp_earth"] = rp_earth_fit if (fit_converged and host_radius_measured) else np.nan
            summary["assumed_host_radius_estimate_earth"] = rp_earth_fit if (fit_converged and not host_radius_measured) else np.nan
            summary["legacy_initial_radius_earth"] = rp_earth_fit if not fit_converged else np.nan
            summary["legacy_radius_uncertainty_earth"] = sigma_rp_earth
            summary["radius_uncertainty_method"] = "legacy_fixed_host_error_and_snr_proxy_NOT_CALIBRATED"
            if products_failed:
                # C1g: the multi-year verdict was computed without these products; say so.
                summary["light_curve_products_failed"] = json.dumps(products_failed)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    print("\n" + "=" * 76)
    print(f"ANTARES SUMMARY FOR TIC {tic_id}:")
    print(f"  -> Unique Sectors Stitched: {len(master_records)} | Total Unique Transits: {master_metrics['n_transits']}")
    print(f"  -> Master Depth: {master_metrics['depth_ppm']:.1f} +/- {master_metrics['err_white_ppm']:.1f} ppm")
    print(f"  -> Master SNR (White): {master_metrics['snr_white']:.2f} sigma | Master SNR (Red-Noise Corrected): {master_metrics['snr_red']:.2f} sigma")
    print(f"  -> Radius (fit={fit_converged}, host_measured={host_radius_measured}, uncertainty heuristic uncalibrated): Rp = {rp_earth_fit:.2f} +/- {sigma_rp_earth:.2f} R_Earth")
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
