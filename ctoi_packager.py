import os
from depth_contract import photometric_depth, geometric_radius, validate_observables, check_output_schema, DEPTH_SCHEMA
import io
import ssl
import glob
import warnings
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import logging
import certifi
import requests
from astropy.io import fits
from astroquery.mast import Catalogs
from vps_contract import resolve_secondary_evidence
from ephemeris_contract import BTJD_OFFSET, btjd_to_bjd
from stellar_metrics import (  # C9: shared copies, see stellar_metrics.py
    G_SI, M_SUN, R_SUN, T_SUN, AU_M, M_EARTH, R_EARTH, H_PLANCK, C_LIGHT, K_BOLTZ,
    to_clean_float, estimate_missing_photometry, batch_query_tic_photometry,
    extract_camera_ccd_from_cache, planck_spectral_radiance, compute_astrophysical_metrics,
)

# Progress line printed by batch_query_tic_photometry for this packager (text unchanged by C9)
CTOI_TIC_QUERY_PROGRESS_PREFIX = "[1/4] Querying MAST TIC v8.2 catalog in batch mode"

logger = logging.getLogger(__name__)

# Priority P1: Enforce verified TLS context anchored to certifi CA bundle
CA_BUNDLE_PATH = certifi.where()
os.environ["SSL_CERT_FILE"] = CA_BUNDLE_PATH
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE_PATH
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=CA_BUNDLE_PATH)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

# BTJD_OFFSET imported from ephemeris_contract

# File Paths
RESULTS_DIR = "tess_results"
TIER1_CSV = os.path.join(RESULTS_DIR, "candidates_report.csv")
TIER2_CSV = os.path.join(RESULTS_DIR, "deep_vetting", "scientific_dossier_summary.csv")
RANKING_OUT_CSV = os.path.join(RESULTS_DIR, "prioritized_targets_ranking.csv")
EXOFOP_OUT_CSV = os.path.join(RESULTS_DIR, "exofop_ctoi_upload.csv")

# S3.13 (Codex S3.10 review): only a candidate row may be exported. The Pleiades pass disposition, its
# pre-ALD wording found in saved summaries, and the Tier-1-only default when no Tier-2 summary exists.
CTOI_CANDIDATE_DISPOSITION_PREFIXES = (
    "TIER-2 HEURISTIC SCREEN PASS",
    "TIER-2 VETTED CANDIDATE",
)
TIER1_ONLY_DISPOSITION = "TIER-1 CANDIDATE"


def ctoi_export_block_reason(row, tier2_present: bool = True):
    """Why a merged row is not a candidate for export, or None (S3.13).

    A failed Pleiades fit (fit_status FAILED, item 39) and an ERROR record are never exported, whatever the
    disposition. Otherwise the disposition must be a candidate one; a Tier-1 row with no Tier-2 record while a
    Tier-2 summary exists has none and is not exported.
    """
    def _text(key):
        value = row.get(key)
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()

    if _text("fit_status").upper() == "FAILED":
        return "fit_status FAILED"
    if _text("operational_status").upper() == "ERROR":
        return "operational_status ERROR"
    disposition = _text("disposition")
    if not tier2_present and disposition == TIER1_ONLY_DISPOSITION:
        return None
    if disposition.upper().startswith(CTOI_CANDIDATE_DISPOSITION_PREFIXES):
        return None
    return f"non-candidate disposition ({disposition or 'no Tier-2 record'})"


def run_ctoi_packager():
    print("=" * 72)
    print("ALDEBARAN CTOI PACKAGER & FOLLOW-UP PRIORITIZATION ENGINE")
    print("=" * 72)
    
    if not os.path.exists(TIER1_CSV):
        raise FileNotFoundError(f"Tier-1 report '{TIER1_CSV}' not found. Run aldebaran.py / merge_shards.py first.")
        
    df_t1 = pd.read_csv(TIER1_CSV)
    df_t1["tic_id"] = df_t1["tic_id"].astype(str).str.strip()
    df_t1 = df_t1.drop_duplicates(subset=["tic_id"], keep="last")
    
    tier2_present = os.path.exists(TIER2_CSV)
    if tier2_present:
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
            ("impact_param_b", np.nan),
            ("dur_to_kepler_max", 1.00),
            ("secondary_snr_sigma", np.nan),
            ("sec_eclipse_phase", 0.50),
            ("is_known_tess_eb", False),
            ("gaia_nss_type", np.nan),
            ("flux_contam_percent", 0.0),
            ("neb_capable_neighbors", 0),
            ("sectors_stitched", "")
        ]:
            if col not in df_merged.columns:
                df_merged[col] = default_v

    # Step 2: Exclude candidates flagged by catalog and diagnostic criteria
    n_initial = len(df_merged)
    has_gaia_nss = df_merged["gaia_nss_type"].notna() & (
        ~df_merged["gaia_nss_type"].astype(str).str.strip().str.upper().isin(["", "NAN", "NONE", "N/A", "FALSE"])
    )
    # Unknown legacy evidence remains unknown in the resolver; compatibility
    # packaging excludes only an established positive CTOI-policy predicate.
    has_sec_eb = pd.Series(
        [resolve_secondary_evidence(row)["ctoi_reject"] is True for _, row in df_merged.iterrows()],
        index=df_merged.index, dtype=bool)
    has_tess_eb = df_merged["is_known_tess_eb"].astype(str).str.strip().str.upper() == "TRUE"
    
    fp_mask = has_gaia_nss | has_sec_eb | has_tess_eb
    df_rejected = df_merged[fp_mask]
    df_clean = df_merged[~fp_mask].copy()
    
    print(f"Excluded {len(df_rejected)} candidates flagged by Gaia NSS, secondary-SNR, or known-TESS-EB criteria.")
    if len(df_rejected) > 0:
        for _, r_row in df_rejected.iterrows():
            print(f"      -> Excluded TIC {r_row['tic_id']}: {r_row.get('disposition', 'FP')}")

    # S3.13: disposition and fit_status gate; an INCOMPLETE_FIT_FAILED or other non-candidate row is never exported.
    block_reasons = [ctoi_export_block_reason(r_row, tier2_present) for _, r_row in df_clean.iterrows()]
    not_candidate = pd.Series([b is not None for b in block_reasons], index=df_clean.index, dtype=bool)
    if not_candidate.any():
        print(f"Excluded {int(not_candidate.sum())} non-candidate row(s) (disposition / fit_status gate).")
        for (_, r_row), reason in zip(df_clean.iterrows(), block_reasons):
            if reason is not None:
                print(f"      -> Not exported TIC {r_row['tic_id']}: {reason}")
    df_clean = df_clean[~not_candidate].copy()

    if len(df_clean) == 0:
        print("[!] No surviving candidates remain after FP filtering.")
        return

    # Step 3: Batch query TIC v8.2 photometry
    surviving_tics = df_clean["tic_id"].tolist()
    tic_phot_map = batch_query_tic_photometry(surviving_tics, progress_prefix=CTOI_TIC_QUERY_PROGRESS_PREFIX)
    
    # Step 4: Calculate astrophysical parameters & build ranking + ExoFOP rows
    print(f"[2/4] Computing M_p, RV semi-amplitude K, TSM, and ESM for {len(df_clean)} candidates...")
    ranking_rows = []
    exofop_rows = []
    
    for _, row in df_clean.iterrows():
        tid = str(row["tic_id"])
        t_meta = tic_phot_map.get(tid, {})
        
        p_days = to_clean_float(row.get("refined_period_days"), to_clean_float(row.get("period_days"), np.nan))
        t0_btjd = to_clean_float(row.get("refined_t0_btjd"), to_clean_float(row.get("t0_btjd"), np.nan))
        epoch_bjd = btjd_to_bjd(t0_btjd, strict=True)
        dur_hr = to_clean_float(row.get("fitted_duration_hours"), to_clean_float(row.get("duration_hours"), np.nan))
        try:
            depth_ppm, depth_source = photometric_depth(row)
            rp_earth, radius_source = geometric_radius(row)
            validate_observables(p_days, t0_btjd, dur_hr, depth_ppm, rp_earth)
        except ValueError as exc:
            logger.error("TIC %s not characterizable: %s", tid, exc)
            continue
        b_imp = to_clean_float(row.get("impact_param_b"), np.nan)
        
        r_star = to_clean_float(row.get("r_star_sun"), t_meta.get("r_star", np.nan))
        m_star = to_clean_float(row.get("m_star_sun"), t_meta.get("m_star", np.nan))
        teff = to_clean_float(row.get("teff_k"), t_meta.get("teff", np.nan))
        if not all(np.isfinite(v) and v > 0 for v in (r_star, m_star, teff)):
            logger.error("TIC %s lacks host measurements; no nominal host exported", tid)
            continue
        
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
        
        fp_risk = to_clean_float(row.get("vps_score", row.get("fp_risk_percent")), 30.0)
        vps_score = fp_risk
        sde_val = to_clean_float(row.get("sde"), 6.5)
        contam_pct = to_clean_float(row.get("flux_contam_percent"), 0.0)
        neb_cnt = int(to_clean_float(row.get("neb_capable_neighbors"), 0))
        secs_str = str(row.get("sectors_stitched") if pd.notna(row.get("sectors_stitched")) and str(row.get("sectors_stitched")) != "" else row.get("sector", "N/A"))
        
        # Composite Follow-Up Score: rewards low VPS risk, high TSM, and high RV K
        clean_weight = max(0.1, (100.0 - vps_score) / 100.0)
        priority_score = round(clean_weight * (0.65 * astro["tsm"] + 4.5 * astro["rv_k_ck17_ms"] + 1.5 * astro["esm"]), 2)
        
        cam_str, ccd_str = extract_camera_ccd_from_cache(tid)
        
        ranking_rows.append({
            "tic_id": tid,
            "depth_schema_version": DEPTH_SCHEMA,
            "depth_kind": "measured_box_photometric",
            "depth_source": depth_source,
            "radius_source": radius_source,
            "impact_parameter_source": "input_fit_or_legacy" if np.isfinite(b_imp) else "missing",
            "characterization_assumptions": "legacy_model_b_default_0.3" if not np.isfinite(b_imp) else "legacy_characterization_model",
            "output_producer": "ctoi_packager",
            "priority_score": priority_score,
            "disposition": row.get("disposition", "CANDIDATE"),
            "vps_score": round(vps_score, 1),
            "fp_risk_percent": round(vps_score, 1),
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
            f"Depth=measured box ({depth_source}); radius={radius_source}. "
            f"Impact source={'input_fit_or_legacy' if np.isfinite(b_imp) else 'missing; characterization assumes b=0.3'}. "
            f"Disposition: {row.get('disposition', 'PC')} (VPS heuristic points={fp_risk:.1f}/100; uncalibrated, not a probability)."
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
    if not ranking_rows:
        raise ValueError("No candidates with required physical observables; existing outputs untouched")
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
    check_output_schema(RANKING_OUT_CSV, df_rank.columns)
    check_output_schema(EXOFOP_OUT_CSV, df_exofop.columns, "|")
    check_output_schema(EXOFOP_OUT_CSV.replace(".csv", "_comma.csv"), df_exofop.columns)
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
