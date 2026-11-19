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
from depth_contract import photometric_depth, geometric_radius, validate_observables, check_output_schema, DEPTH_SCHEMA
import sys
import ssl
import glob
import shutil
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Optional

import logging
import json
import math
import certifi
from ephemeris_contract import BTJD_OFFSET, btjd_to_bjd, btjd_to_year, fold_phase_centered, compute_orbit_number, days_to_hours, hours_to_days

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
from vps_contract import (
    validate_and_extract_vps_score,
    VPSContractError,
    ANTARES_DEFAULT_MAX_CORR_RISK,
    VPS_SCHEMA_VERSION
)
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from astropy.io import fits
import lightkurve as lk
from astroquery.mast import Catalogs
from lightcurve_helpers import (  # C9: shared copies, see lightcurve_helpers.py
    to_numpy, extract_multi_author_lc, hybrid_tukey_savgol_detrend, _GL5_NODES, _GL5_WEIGHTS,
    light_curve_product_failure, EXTRACTION_TOO_FEW_POINTS,
    annulus_300_transit_model, refine_multi_year_ephemeris, compute_transit_metrics,
)
from stellar_metrics import (  # C9: shared copies, see stellar_metrics.py
    G_SI, M_SUN, R_SUN, T_SUN, AU_M, M_EARTH, R_EARTH, H_PLANCK, C_LIGHT, K_BOLTZ,
    to_clean_float, estimate_missing_photometry, batch_query_tic_photometry,
    extract_camera_ccd_from_cache, planck_spectral_radiance, compute_astrophysical_metrics,
)

# BTJD_OFFSET imported from ephemeris_contract

# Pipeline Author priority and diagnostic colors
AUTHOR_PRIORITY = {"SPOC": 1, "TESS-SPOC": 2, "TGLC": 3, "QLP": 4}
AUTHOR_COLORS = {"SPOC": "crimson", "TESS-SPOC": "darkorange", "TGLC": "royalblue", "QLP": "seagreen"}

DEFAULT_RESULTS_DIR = "tess_results"


# ==============================================================================
# UTILITY FUNCTIONS & PHOTOMETRIC CONVERSIONS
# ==============================================================================
# Moved by C9 (S3.5) to lightcurve_helpers.py and stellar_metrics.py; imported above.

# ==============================================================================
# STAGE 3A: FAST PRE-FILTER GATE
# ==============================================================================

def calc_sidak_corrected_risk(row: pd.Series) -> float:
    """
    Validates versioned VPS component accounting and extracts authoritative score.
    Legacy unsupported secondary-penalty subtraction has been removed per ALD-012 / SCI-16.
    """
    return validate_and_extract_vps_score(row)


def pleiades_fit_failed(df: pd.DataFrame) -> pd.Series:
    """SUCCESS Tier-2 rows whose Pleiades transit fit failed (S3.10, item 39.1). ERROR rows stay with the contract."""
    if "fit_status" not in df.columns or "operational_status" not in df.columns:
        return pd.Series(False, index=df.index)
    fit = df["fit_status"].astype(str).str.strip().str.upper() == "FAILED"
    ok = df["operational_status"].astype(str).str.strip().str.upper() == "SUCCESS"
    return fit & ok


def fit_failed_stage_3a_status(disposition) -> str:
    """INCOMPLETE_FIT_FAILED stays incomplete; a negative flag kept after the failed fit is a reject."""
    return "INCOMPLETE_FIT_FAILED" if str(disposition).strip() == "INCOMPLETE_FIT_FAILED" else "REJECT"


def run_stage_3a_filter(
    tier1_csv: str,
    tier2_csv: str,
    max_corr_risk: float = ANTARES_DEFAULT_MAX_CORR_RISK,
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

    if len(df_t1) == 0:
        print("[Stage 3A] Tier-1 candidate report is empty (0 candidates). Skipping Stage 3A filtering.")
        empty_survivors = pd.DataFrame(columns=[
            "tic_id", "corr_risk", "period_d", "rp_e", "dur_ratio", "contam_pct", "neb_count", "sectors_stitched"
        ])
        return empty_survivors, pd.DataFrame()

    if not os.path.exists(tier2_csv):
        raise FileNotFoundError(
            f"Tier-2 Pleiades summary '{tier2_csv}' not found. "
            f"Stage 3A requires verified Tier-2 score provenance."
        )

    df_t2 = pd.read_csv(tier2_csv)
    df_t2["tic_id"] = df_t2["tic_id"].astype(str).str.strip()
    df_t2 = df_t2.drop_duplicates(subset=["tic_id"], keep="first")
    df_merged = pd.merge(df_t2, df_t1, on="tic_id", how="left", suffixes=("", "_t1"))
    print(f"[Stage 3A] Loaded {len(df_t2)} Tier-2 Pleiades dossiers matched against {len(df_t1)} Tier-1 candidates.")

    # Fallback mappings for critical columns
    def _get_num_series(col: str, fallback: Optional[str] = None) -> pd.Series:
        s = pd.to_numeric(df_merged[col], errors="coerce") if col in df_merged.columns else pd.Series(np.nan, index=df_merged.index)
        if fallback and fallback in df_merged.columns:
            s_fb = pd.to_numeric(df_merged[fallback], errors="coerce")
            s = s.fillna(s_fb)
        return s

    df_merged["period_d"] = _get_num_series("refined_period_days", "period_days")
    df_merged["t0_btjd"] = _get_num_series("refined_t0_btjd", "t0_btjd")
    df_merged["duration_hours"] = _get_num_series("fitted_duration_hours", "duration_hours")
    # Depth is descriptive here, not one of the five gate predicates below.
    # Keep missing evidence missing; only packaging requires a positive observable.
    def _box_value(row):
        try:
            return photometric_depth(row)[0]
        except ValueError:
            return np.nan
    df_merged["depth_ppm"] = df_merged.apply(_box_value, axis=1)
    df_merged["rp_e"] = _get_num_series("fitted_rp_earth", "r_planet_earth")
    df_merged["dur_ratio"] = _get_num_series("dur_to_kepler_max")
    df_merged["sec_snr"] = _get_num_series("secondary_snr_sigma")
    df_merged["contam_pct"] = _get_num_series("flux_contam_percent")
    df_merged["neb_count"] = _get_num_series("neb_capable_neighbors")
    if "sec_eclipse_phase" not in df_merged.columns:
        df_merged["sec_eclipse_phase"] = np.nan
    if "sectors_stitched" not in df_merged.columns:
        df_merged["sectors_stitched"] = ""

    # S3.10 (item 39.1): a SUCCESS record whose Pleiades fit failed has no VPS score. It leaves Stage 3A
    # here, before the score contract, is never drilled, and gets its own Stage 3 audit row.
    fit_failed = pleiades_fit_failed(df_merged)
    df_fit_failed = df_merged[fit_failed].copy().reset_index(drop=True)
    df_fit_failed["stage_3a_status"] = [fit_failed_stage_3a_status(d) for d in df_fit_failed.get("disposition", [])]
    df_fit_failed["stage_3a_reason"] = [f"Pleiades transit fit FAILED: {d}" for d in df_fit_failed.get("disposition", [])]
    df_merged = df_merged[~fit_failed].copy()
    if len(df_fit_failed):
        print(f"[Stage 3A] {len(df_fit_failed)} Tier-2 record(s) with a failed Pleiades fit are not drilled: "
              f"{', '.join(df_fit_failed['tic_id'])}")
        if len(df_merged) == 0:
            empty_survivors = pd.DataFrame(columns=[
                "tic_id", "corr_risk", "period_d", "rp_e", "dur_ratio", "contam_pct", "neb_count", "sectors_stitched"
            ])
            return empty_survivors, df_fit_failed

    # Validates versioned VPS component accounting and extracts authoritative score (no subtraction)
    df_merged["corr_risk"] = df_merged.apply(calc_sidak_corrected_risk, axis=1)

    if len(df_merged) == 0:
        empty_survivors = pd.DataFrame(columns=[
            "tic_id", "corr_risk", "period_d", "rp_e", "dur_ratio", "contam_pct", "neb_count", "sectors_stitched"
        ])
        return empty_survivors, pd.DataFrame()

    # B2 / C1 Gate Contract Validation: Validate mandatory physical measurements, stellar environment, catalog flags, and duplicate fields
    def _validate_gate_contract_and_sec(row) -> dict:
        tic_id = str(row.get("tic_id", "UNKNOWN")).strip()

        # 1. dur_to_kepler_max: mandatory positive finite float
        raw_dur = row.get("dur_to_kepler_max")
        if "dur_to_kepler_max" not in row or raw_dur is None or pd.isna(raw_dur):
            raise VPSContractError(f"Target TIC {tic_id} missing mandatory gate measurement 'dur_to_kepler_max'.")
        try:
            dur_val = float(raw_dur)
            if not math.isfinite(dur_val) or dur_val <= 0.0:
                raise ValueError(f"non-positive or non-finite: {dur_val}")
        except (TypeError, ValueError) as exc:
            raise VPSContractError(f"Target TIC {tic_id} invalid mandatory gate measurement 'dur_to_kepler_max': {exc}")

        # 2. flux_contam_percent: mandatory non-negative finite float (ratio percentage, can exceed 100%)
        raw_contam = row.get("flux_contam_percent")
        if "flux_contam_percent" not in row or raw_contam is None or pd.isna(raw_contam):
            raise VPSContractError(f"Target TIC {tic_id} missing mandatory gate measurement 'flux_contam_percent'.")
        try:
            contam_val = float(raw_contam)
            if not math.isfinite(contam_val) or contam_val < 0.0:
                raise ValueError(f"negative or non-finite: {contam_val}")
        except (TypeError, ValueError) as exc:
            raise VPSContractError(f"Target TIC {tic_id} invalid mandatory gate measurement 'flux_contam_percent': {exc}")

        # 3. neb_capable_neighbors: mandatory non-negative integer
        raw_neb = row.get("neb_capable_neighbors")
        if "neb_capable_neighbors" not in row or raw_neb is None or pd.isna(raw_neb):
            raise VPSContractError(f"Target TIC {tic_id} missing mandatory gate measurement 'neb_capable_neighbors'.")
        try:
            neb_float = float(raw_neb)
            if not math.isfinite(neb_float) or neb_float < 0 or int(neb_float) != neb_float:
                raise ValueError(f"non-integer, negative, or non-finite: {raw_neb}")
        except (TypeError, ValueError) as exc:
            raise VPSContractError(f"Target TIC {tic_id} invalid mandatory gate measurement 'neb_capable_neighbors': {exc}")

        # 4. is_known_tess_eb: mandatory boolean flag
        if "is_known_tess_eb" not in row or pd.isna(row.get("is_known_tess_eb")) or str(row.get("is_known_tess_eb")).strip() == "":
            raise VPSContractError(f"Target TIC {tic_id} missing mandatory gate flag 'is_known_tess_eb'.")
        raw_eb = row.get("is_known_tess_eb")
        if isinstance(raw_eb, (bool, np.bool_)):
            is_known_eb_val = bool(raw_eb)
        else:
            s_eb = str(raw_eb).strip().upper()
            if s_eb in ("TRUE", "1", "1.0", "YES"):
                is_known_eb_val = True
            elif s_eb in ("FALSE", "0", "0.0", "NO"):
                is_known_eb_val = False
            else:
                raise VPSContractError(f"Target TIC {tic_id} invalid non-boolean 'is_known_tess_eb': '{raw_eb}'.")

        # 5. Gaia NSS catalog context: gaia_nss_type and is_gaia_nss
        has_nss_type_col = "gaia_nss_type" in row
        has_nss_flag_col = "is_gaia_nss" in row
        if not has_nss_type_col and not has_nss_flag_col:
            raise VPSContractError(f"Target TIC {tic_id} missing mandatory Gaia NSS catalog context.")

        raw_nss_type = row.get("gaia_nss_type") if has_nss_type_col else None
        raw_nss_flag = row.get("is_gaia_nss") if has_nss_flag_col else None

        # Parse gaia_nss_type if present
        type_str = "" if (raw_nss_type is None or pd.isna(raw_nss_type)) else str(raw_nss_type).strip()
        if type_str.upper() in ("", "NAN", "NONE", "N/A", "NULL", "FALSE", "0"):
            type_is_match = False
            type_is_explicit_negative = type_str.upper() in ("FALSE", "0")
        else:
            type_is_match = True
            type_is_explicit_negative = False

        # Parse is_gaia_nss if present
        if not has_nss_flag_col or raw_nss_flag is None or pd.isna(raw_nss_flag) or str(raw_nss_flag).strip() == "":
            flag_is_specified = False
            flag_val = None
        elif isinstance(raw_nss_flag, (bool, np.bool_)):
            flag_is_specified = True
            flag_val = bool(raw_nss_flag)
        else:
            s_flag = str(raw_nss_flag).strip().upper()
            if s_flag in ("TRUE", "1", "1.0", "YES"):
                flag_is_specified = True
                flag_val = True
            elif s_flag in ("FALSE", "0", "0.0", "NO"):
                flag_is_specified = True
                flag_val = False
            else:
                raise VPSContractError(f"Target TIC {tic_id} invalid non-boolean 'is_gaia_nss': '{raw_nss_flag}'.")

        # Context validation across flag and type
        if has_nss_flag_col and not flag_is_specified and not has_nss_type_col:
            raise VPSContractError(f"Target TIC {tic_id} missing mandatory Gaia NSS catalog context.")

        if flag_is_specified and has_nss_type_col:
            # Check for contradiction between explicit boolean flag and explicit solution type
            if flag_val is False and type_is_match is True:
                raise VPSContractError(
                    f"Target TIC {tic_id} contradictory Gaia NSS context: is_gaia_nss=False but gaia_nss_type='{type_str}'."
                )
            if flag_val is True and type_is_explicit_negative is True:
                raise VPSContractError(
                    f"Target TIC {tic_id} contradictory Gaia NSS context: is_gaia_nss=True but gaia_nss_type='{type_str}'."
                )

        if flag_is_specified and flag_val is True:
            is_gaia_nss_val = True
        elif type_is_match:
            is_gaia_nss_val = True
        else:
            is_gaia_nss_val = False

        # 6. Secondary eclipse significance from canonical validated ledger & duplicate field validation (B1)
        raw_ledger = row.get("vps_ledger_json")
        ledger = raw_ledger if isinstance(raw_ledger, dict) else json.loads(str(raw_ledger))
        sec_meta = ledger["secondary_metadata"]

        if "secondary_snr_sigma" in row and pd.notna(row["secondary_snr_sigma"]):
            top_snr = float(row["secondary_snr_sigma"])
            canonical_snr = float(sec_meta["snr"])
            rounded_expected_snr = round(canonical_snr, 2)
            if abs(top_snr - rounded_expected_snr) > 1e-4 and abs(top_snr - canonical_snr) > 1e-4:
                raise VPSContractError(
                    f"Target TIC {tic_id} contradictory secondary SNR: "
                    f"top-level secondary_snr_sigma={top_snr} vs ledger snr={canonical_snr} (expected {rounded_expected_snr})."
                )
        if "sec_eclipse_phase" in row and pd.notna(row["sec_eclipse_phase"]):
            top_phase = float(row["sec_eclipse_phase"])
            canonical_phase = float(sec_meta["phase"])
            rounded_expected_phase = round(canonical_phase, 3)
            if abs(top_phase - rounded_expected_phase) > 1e-4 and abs(top_phase - canonical_phase) > 1e-4:
                raise VPSContractError(
                    f"Target TIC {tic_id} contradictory secondary phase: "
                    f"top-level sec_eclipse_phase={top_phase} vs ledger phase={canonical_phase} (expected {rounded_expected_phase})."
                )
        sec_pass_val = not bool(sec_meta["penalty_applied"])

        return {
            "sec_pass": sec_pass_val,
            "is_known_eb": is_known_eb_val,
            "is_gaia_nss": is_gaia_nss_val,
        }

    gate_contract_res = df_merged.apply(_validate_gate_contract_and_sec, axis=1)
    sec_pass = gate_contract_res.apply(lambda r: r["sec_pass"])
    is_known_eb = gate_contract_res.apply(lambda r: r["is_known_eb"])
    is_gaia_nss = gate_contract_res.apply(lambda r: r["is_gaia_nss"])

    # Update df_merged with normalized catalog columns
    df_merged["is_known_tess_eb"] = is_known_eb
    df_merged["is_gaia_nss"] = is_gaia_nss

    # 5-Rule Gate Conditions on validated physical and catalog quantities:
    # 1. No Gaia NSS binary & not known TESS EB
    # 2. Secondary eclipse significance from canonical validated ledger (sec_pass computed above)
    # 3. Kinematic duration self-consistency
    dur_pass = (df_merged["dur_ratio"] >= 0.55) & (df_merged["dur_ratio"] <= 1.35)

    # 4. Stellar environment: dilution & nearby EBs
    env_pass = (df_merged["neb_count"] <= max_neb_neighbors) & (df_merged["contam_pct"] <= max_flux_contam_pct)

    # 5. Uncalibrated heuristic VPS threshold (Gate <= max_corr_risk, default <= 25.0; not a probability)
    risk_pass = df_merged["corr_risk"] <= max_corr_risk

    gate_mask = (~is_gaia_nss) & (~is_known_eb) & sec_pass & dur_pass & env_pass & risk_pass

    df_survivors = df_merged[gate_mask].sort_values(by="corr_risk").reset_index(drop=True)
    df_rejected = df_merged[~gate_mask].copy().reset_index(drop=True)
    if len(df_fit_failed):
        df_rejected = pd.concat([df_rejected, df_fit_failed], ignore_index=True)

    print(f"[Stage 3A Result] Input: {len(df_merged)} -> Rejected: {len(df_rejected)} -> Passed Gate: {len(df_survivors)} targets")
    print(f"[Stage 3A] Gate condition 5: VPS heuristic points <= {max_corr_risk:.1f} (default <= {ANTARES_DEFAULT_MAX_CORR_RISK:.1f}; uncalibrated, not a probability)")
    print(f"[Stage 3A] Compute runtime saving: {100.0 * (1.0 - len(df_survivors) / max(1, len(df_merged))):.1f}% reduction in MAST downloads.\n")

    disp_cols = ["tic_id", "corr_risk", "period_d", "rp_e", "dur_ratio", "contam_pct", "neb_count", "sectors_stitched"]
    print("STAGE 3A SURVIVING TARGETS:")
    if len(df_survivors) > 0:
        print(df_survivors[disp_cols].to_string(index=False))
    else:
        print("(0 surviving targets)")
    print("-" * 76)

    return df_survivors, df_rejected


# ==============================================================================
# STAGE 3B: DECOUPLED MULTI-YEAR ARCHIVAL DRILL (SPOC FIRST VS FFI BLEND TRACK)
# ==============================================================================

def assess_blend(author_stats: dict, contam_pct: float) -> tuple:
    """Stage 3B blend check on finite depths only.

    Returns (blended_suspect, blend_reason, blend_check) where blend_check is
    "EVALUATED" or "NOT_EVALUATED". A check without a finite SPOC depth or
    without at least one finite QLP/TGLC depth never rejects.
    """
    if "SPOC" not in author_stats or not ("QLP" in author_stats or "TGLC" in author_stats):
        return False, "", "NOT_EVALUATED"
    d_spoc = float(author_stats["SPOC"].get("depth_ppm", np.nan))
    alt = [float(author_stats[a].get("depth_ppm", np.nan)) for a in ("QLP", "TGLC") if a in author_stats]
    alt = [d for d in alt if np.isfinite(d)]
    if not np.isfinite(d_spoc) or not alt:
        return False, "", "NOT_EVALUATED"
    d_alt = max(alt)
    if d_spoc > 2.5 * max(d_alt, 1.0) and contam_pct > 5.0 and d_alt < 200.0:
        return True, f"SPOC depth ({d_spoc:.0f} ppm) diluted > 2.5x in QLP/TGLC ({d_alt:.0f} ppm)", "EVALUATED"
    return False, "", "EVALUATED"


# S3.13, ARGON_DECISIONS item 34.1 (Stage 3B part). The Filter 4 state travels in the candidate record
# (`centroid_state`, written by aldebaran.py since S3.8). A CENTROID_NOT_EVALUATED peak may stay a candidate only
# if a pixel-level spatial check evaluated it and passed: the TPF check, read only from the explicit record field
# `tpf_spatial_pass` (written by aldebaran.py from S3.12 on), or the Stage 3B blend check. Otherwise it is capped
# at REVIEW and not packaged. A legacy record without `centroid_state` is NOT_RECORDED and is not capped.
CENTROID_EVALUATED = "EVALUATED"
CENTROID_NOT_EVALUATED = "CENTROID_NOT_EVALUATED"
CENTROID_NOT_RECORDED = "NOT_RECORDED"
ARCHIVAL_REVIEW = "ARCHIVAL_REVIEW"
SPATIAL_TPF_PASS = "TPF_PASS"
SPATIAL_BLEND_PASS = "STAGE_3B_BLEND_PASS"
SPATIAL_NONE = "NONE"


def _record_text(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return str(value).strip()


def candidate_centroid_state(candidate: dict) -> str:
    """Filter 4 state carried by the candidate record; NOT_RECORDED when the record has none (legacy)."""
    return _record_text(candidate.get("centroid_state")).upper() or CENTROID_NOT_RECORDED


def tpf_spatial_pass_recorded(candidate: dict) -> bool:
    """True only when the record states explicitly that the TPF spatial check evaluated the peak and passed."""
    value = candidate.get("tpf_spatial_pass")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _record_text(value).upper() in ("TRUE", "1", "1.0")


def stage_3b_centroid_cap(candidate: dict, blend_check: str, blended_suspect: bool) -> tuple:
    """Item 34.1 at Stage 3B. Returns (capped, centroid_state, spatial_check)."""
    state = candidate_centroid_state(candidate)
    if tpf_spatial_pass_recorded(candidate):
        spatial = SPATIAL_TPF_PASS
    elif blend_check == "EVALUATED" and not blended_suspect:
        spatial = SPATIAL_BLEND_PASS
    else:
        spatial = SPATIAL_NONE
    # Any recorded state other than EVALUATED (CENTROID_NOT_EVALUATED or an unknown value) needs a spatial pass.
    capped = state not in (CENTROID_EVALUATED, CENTROID_NOT_RECORDED) and spatial == SPATIAL_NONE
    return capped, state, spatial


def archival_decision_reason(res: dict) -> str:
    """Audit text for one Stage 3B result; failed products are named, never dropped (C1g)."""
    failed = res.get("products_failed") or []
    if not failed or res.get("status") == "ARCHIVAL_ERROR":
        reason = res["reason"]
    else:
        reason = f"{res['reason']}; {len(failed)} light curve product(s) not used"
    if failed:
        reason += ": " + "; ".join(f"S{f['sector']} {f['author'] or '?'} {f['stage']} ({f['reason']})" for f in failed)
    return reason


def stage_3b_audit_row(tic_id, res: dict) -> dict:
    """Stage 3 audit row of one drilled target (C1g: failed products named in decision_reason)."""
    return {
        "tic_id": tic_id,
        "stage_3a": "PASS",
        "stage_3b_status": res["status"],
        "decision_reason": archival_decision_reason(res),
        "refined_period_days": res.get("p_ref", np.nan),
        "snr_red": res.get("snr_red", np.nan),
        "snr_white": res.get("snr_white", np.nan),
        "depth_ppm": res.get("depth_ppm", np.nan),
        "stack_author": res.get("stack_author", "N/A"),
        "n_transits": res.get("n_transits", 0),
        "blend_check": res.get("blend_check", "NOT_EVALUATED"),
        "centroid_state": candidate_centroid_state(res.get("candidate") or {}),
        "spatial_check": res.get("spatial_check", "NOT_EVALUATED"),
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
    dur_days = hours_to_days(dur_hr)

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
        except Exception as exc:
            products_failed.append(light_curve_product_failure(sec_num, auth, stage, exc))
            continue

    for failed in products_failed:
        print(f"      [!] Product not used: Sector {failed['sector']} | Author: {failed['author'] or '?'} | "
              f"{failed['stage']}: {failed['reason']}")
    if not records:
        print(f"      [!] Failed to extract valid light curves for TIC {tid}.")
        if products_failed:
            # C1g (items 29 and 31.6): products exist but none could be read; nothing was evaluated.
            return {"status": "ARCHIVAL_ERROR",
                    "reason": f"All {len(products_failed)} light curve product(s) failed to download or prepare",
                    "candidate": candidate, "products_failed": products_failed}
        return {"status": "ARCHIVAL_REJECT", "reason": "No valid light curves extracted", "candidate": candidate,
                "products_failed": products_failed}

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

    # Blending Diagnostic Check (finite depths only; see assess_blend):
    blended_suspect, blend_reason, blend_check = assess_blend(
        author_stats, float(candidate.get("contam_pct", 0.0))
    )
    centroid_capped, centroid_state, spatial_check = stage_3b_centroid_cap(candidate, blend_check, blended_suspect)

    # Stage 3B Decision Gate
    if spoc_snr_red < min_spoc_snr_red and master_metrics["snr_white"] < (min_spoc_snr_red + 0.3):
        status = "ARCHIVAL_REJECT"
        decision_reason = f"SNR_red ({spoc_snr_red:.2f} sigma) < threshold ({min_spoc_snr_red:.1f} sigma)"
    elif blended_suspect:
        status = "ARCHIVAL_REJECT"
        decision_reason = f"Aperture Dilution / Blended ({blend_reason})"
    elif centroid_capped:
        status = ARCHIVAL_REVIEW
        decision_reason = (f"{CENTROID_NOT_EVALUATED}: no pixel-level spatial check passed (TPF pass not recorded; "
                           f"Stage 3B blend check {blend_check}); capped at REVIEW, not packaged (item 34.1)")
    else:
        status = "ARCHIVAL_PASS"
        decision_reason = f"SNR_red={spoc_snr_red:.2f}s (SNR_white={master_metrics['snr_white']:.1f}s) on {stack_author} stack (N={master_metrics['n_transits']} transits across {n_sectors_stack} sectors)"

    print(f"      -> Ephemeris: P = {p_ref:.7f} d (dP = {dp_sec:+.2f} s) | T0 = {t0_ref:.5f} BTJD")
    print(f"      -> Stack ({stack_author}, {n_sectors_stack} secs): Depth = {master_metrics['depth_ppm']:.1f} ppm | SNR_red = {spoc_snr_red:.2f} sigma | Transits = {master_metrics['n_transits']}")
    print(f"      -> Archival Decision: [{status}] - {decision_reason}")

    # Generate 4-panel diagnostic poster in archival_vetting subfolder
    archival_dir = os.path.join(results_dir, "archival_vetting")
    os.makedirs(archival_dir, exist_ok=True)
    out_png = os.path.join(archival_dir, f"antares_TIC_{tid}.png")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        win_hrs = days_to_hours(min(0.18 * p_ref, 2.5 * dur_days))
        bins_hrs = np.linspace(-win_hrs, win_hrs, 35)
        b_centers = 0.5 * (bins_hrs[:-1] + bins_hrs[1:])

        # Panel 1: Master Phase-Folded Transit
        ph_hrs = days_to_hours(((t_master - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref)
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
            ph_a_hrs = days_to_hours(((t_a - t0_ref + 0.5 * p_ref) % p_ref) - 0.5 * p_ref)
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
        "author_stats": author_stats,
        "blend_check": blend_check,
        "centroid_state": centroid_state,
        "spatial_check": spatial_check,
        "products_failed": products_failed,
    }


# ==============================================================================
# STAGE 3C: ASTROPHYSICAL PARAMETERS & CTOI PACKAGING
# ==============================================================================

# S2.5 (IMPROVE-QUICK 001): one header per Stage 3C table, whether or not a candidate is packaged.
# The order is that of the ranking row built in run_stage_3c_packaging, with "rank" inserted first.
ANTARES_RANKING_COLS = (
    "rank", "tic_id", "depth_schema_version", "depth_kind", "depth_source", "radius_source",
    "impact_parameter_source", "characterization_assumptions", "output_producer", "priority_score",
    "disposition", "corr_risk_percent", "sectors", "archival_stack", "period_days", "t0_btjd",
    "epoch_bjd", "duration_hours", "depth_ppm", "rp_earth", "snr_red", "snr_white", "n_transits",
    "mp_ck17_earth", "mp_w16_earth", "rho_p_gcm3", "g_p_ms2", "rv_k_ck17_ms", "rv_k_w16_ms",
    "tsm", "esm", "teq_k", "insolation_earth", "a_au", "a_over_rstar", "impact_param_b",
    "r_star_sun", "m_star_sun", "teff_k", "Tmag", "Vmag", "Gmag", "Jmag", "Hmag", "Kmag",
    "nir_phot_estimated", "flux_contam_percent", "neb_capable_neighbors", "rv_followup_priority",
    "jwst_followup_priority", "exofop_url",
)
ANTARES_EXOFOP_COLS = (
    "TIC ID", "Camera", "CCD", "Epoch BJD", "Period",
    "Depth", "Duration", "Rp", "User Comments / Disposition",
)


def write_empty_tier3_outputs(results_dir: str) -> None:
    """Write the four Tier 3 tables with headers and no rows (no candidate to package)."""
    ranking_cols = list(ANTARES_RANKING_COLS)
    exofop_cols = list(ANTARES_EXOFOP_COLS)
    ranking_csv = os.path.join(results_dir, "antares_prioritized_targets_ranking.csv")
    exofop_csv = os.path.join(results_dir, "antares_exofop_ctoi_upload.csv")
    exofop_comma_csv = os.path.join(results_dir, "antares_exofop_ctoi_upload_comma.csv")
    top3_csv = os.path.join(results_dir, "antares_exofop_ctoi_upload_top3.csv")
    # Precheck every path before writing any zero-survivor table.
    check_output_schema(ranking_csv, ranking_cols)
    check_output_schema(exofop_csv, exofop_cols, "|")
    check_output_schema(exofop_comma_csv, exofop_cols)
    check_output_schema(top3_csv, exofop_cols)
    pd.DataFrame(columns=ranking_cols).to_csv(ranking_csv, index=False)
    pd.DataFrame(columns=exofop_cols).to_csv(exofop_csv, index=False, sep="|")
    pd.DataFrame(columns=exofop_cols).to_csv(exofop_comma_csv, index=False)
    pd.DataFrame(columns=exofop_cols).to_csv(top3_csv, index=False)


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
    # S3.13 (item 34.1): a result whose centroid cap holds is never packaged, whatever its status says.
    capped = [r for r in passing_cands if stage_3b_centroid_cap(r["candidate"], r.get("blend_check"), False)[0]]
    for r in capped:
        print(f"[!] TIC {r['candidate']['tic_id']}: {CENTROID_NOT_EVALUATED} without a passed spatial check; not packaged.")
    passing_cands = [r for r in passing_cands if not any(r is c for c in capped)]
    if not passing_cands:
        print("[!] No candidates passed Stage 3B archival vetting.")
        write_empty_tier3_outputs(results_dir)
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
        epoch_bjd = btjd_to_bjd(t0_btjd, strict=True)
        dur_hr = to_clean_float(cand.get("duration_hours"), np.nan)
        depth_ppm = r["depth_ppm"]

        # Stellar parameters
        r_star = to_clean_float(t_meta.get("r_star"), to_clean_float(cand.get("r_star_sun"), np.nan))
        m_star = to_clean_float(t_meta.get("m_star"), to_clean_float(cand.get("m_star_sun"), np.nan))
        teff = to_clean_float(t_meta.get("teff"), to_clean_float(cand.get("teff_k"), np.nan))

        # Radius and box depth have different meanings; never derive k from the box.
        try:
            rp_earth, radius_source = geometric_radius(cand)
            validate_observables(p_days, t0_btjd, dur_hr, depth_ppm, rp_earth)
            if not all(np.isfinite(v) and v > 0 for v in (r_star, m_star, teff)):
                raise ValueError("Missing measured host parameters")
        except ValueError as exc:
            logger.error("TIC %s not characterizable: %s", tid, exc)
            continue
        b_imp = to_clean_float(cand.get("impact_param_b"), np.nan)

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

        # Composite Follow-Up Priority Score (uses uncalibrated heuristic VPS points, not a probability)
        clean_weight = max(0.1, (100.0 - corr_risk) / 100.0)
        priority_score = round(clean_weight * (0.65 * astro["tsm"] + 4.5 * astro["rv_k_ck17_ms"] + 1.5 * astro["esm"]), 2)

        cam_str, ccd_str = extract_camera_ccd_from_cache(tid)

        ranking_rows.append({
            "tic_id": tid,
            "depth_schema_version": DEPTH_SCHEMA,
            "depth_kind": "measured_box_photometric",
            "depth_source": "antares_archive_inverse_variance_box",
            "radius_source": radius_source,
            "impact_parameter_source": "input_fit_or_legacy" if np.isfinite(b_imp) else "missing",
            "characterization_assumptions": "legacy_model_b_default_0.3" if not np.isfinite(b_imp) else "legacy_characterization_model",
            "output_producer": "antares_packager",
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
            f"Depth=archival measured box; radius={radius_source}. "
            f"Impact source={'input_fit_or_legacy' if np.isfinite(b_imp) else 'missing; characterization assumes b=0.3'}. "
            f"Gaia Contam={contam_pct:.1f}% ({neb_cnt} NEB neighbors). Disposition: PC (VPS heuristic points={corr_risk:.1f}/100; uncalibrated, not a probability)."
        )
        # Item 34.1: a candidate without a usable centroid test keeps the label, with the spatial check that passed.
        _cstate, _spatial = stage_3b_centroid_cap(cand, r.get("blend_check"), False)[1:]
        if _cstate == CENTROID_NOT_EVALUATED:
            comment_str += f" Centroid test: {CENTROID_NOT_EVALUATED} (spatial check passed: {_spatial})."

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

    if not ranking_rows:
        raise ValueError("No characterizable archival candidates; existing outputs untouched")
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
    ranking_csv = os.path.join(results_dir, "antares_prioritized_targets_ranking.csv")
    exofop_csv = os.path.join(results_dir, "antares_exofop_ctoi_upload.csv")
    exofop_comma_csv = os.path.join(results_dir, "antares_exofop_ctoi_upload_comma.csv")

    if list(df_rank.columns) != list(ANTARES_RANKING_COLS) or list(df_exofop.columns) != list(ANTARES_EXOFOP_COLS):
        raise ValueError("Stage 3C table columns differ from ANTARES_RANKING_COLS / ANTARES_EXOFOP_COLS; "
                         "existing outputs untouched")
    check_output_schema(ranking_csv, df_rank.columns)
    check_output_schema(exofop_csv, df_exofop.columns, "|")
    check_output_schema(exofop_comma_csv, df_exofop.columns)
    # S2.5: a non-empty run writes no top-3 table, so Tier-4 keeps reading the full ranking. A header-only
    # top-3 table left by an earlier zero-survivor run would make merge_shards skip Tier-4; refuse it.
    top3_csv = os.path.join(results_dir, "antares_exofop_ctoi_upload_top3.csv")
    if os.path.exists(top3_csv):
        raise ValueError("Refusing stale zero-survivor table at " + top3_csv + "; existing outputs untouched")
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
    min_snr: float = 3.8,
    tier1_csv: str = None,
    tier2_csv: str = None
):
    print("=" * 76)
    print("ALDEBARAN v8.0 / ANTARES v3.0: UNIFIED TIER-3 ARCHIVAL VETTER & CTOI PACKAGER")
    print("=" * 76)

    if tier1_csv is None:
        tier1_csv = os.path.join(results_dir, "candidates_report.csv")
    if tier2_csv is None:
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
        fit_failed_tics = set(df_rejected["tic_id"][df_rejected["stage_3a_status"].notna()]) \
            if "stage_3a_status" in df_rejected.columns else set()
        if len(df_survivors) == 0 and target_tic in fit_failed_tics:
            # S3.10 (item 39.1): a failed Pleiades fit is never drilled, not even on an explicit --tic.
            print(f"[!] Target TIC {target_tic} has a failed Pleiades transit fit; it is not drilled.")
        elif len(df_survivors) == 0:
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
        print("[!] No candidates available for Stage 3B archival drilling (0 survivors).")
        write_empty_tier3_outputs(results_dir)
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
        audit_rows.append(stage_3b_audit_row(row["tic_id"], res))

    # Record rejected targets in audit
    for _, r_row in df_rejected.iterrows():
        fit_failed_status = r_row.get("stage_3a_status")
        fit_failed_row = isinstance(fit_failed_status, str) and bool(fit_failed_status)
        audit_rows.append({
            "tic_id": r_row["tic_id"],
            "stage_3a": fit_failed_status if fit_failed_row else "REJECT",
            "stage_3b_status": "SKIPPED",
            "decision_reason": r_row["stage_3a_reason"] if fit_failed_row else
            f"Stage 3A Filter (VPS heuristic points={r_row.get('corr_risk')}/100, DurRatio={r_row.get('dur_ratio')}%, NEB={r_row.get('neb_count')})",
            "refined_period_days": r_row.get("period_d", np.nan),
            "snr_red": np.nan,
            "snr_white": np.nan,
            "depth_ppm": np.nan,
            "stack_author": "N/A",
            "n_transits": 0,
            "blend_check": "SKIPPED",
            "centroid_state": candidate_centroid_state(r_row.to_dict()),
            "spatial_check": "SKIPPED",
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
    n_review = len([r for r in vetted_results if r["status"] == ARCHIVAL_REVIEW])
    if n_review:
        print(f"Stage 3B REVIEW (centroid cap): {n_review}")
    print(f"Packaged for NASA CTOI:  {len(df_exofop)}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antares Packager: Unified Tier-3 Archival Vetter & CTOI Packager")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results directory (default: tess_results)")
    parser.add_argument("--tier1-csv", type=str, default=None, help="Explicit path to Tier-1 candidates report CSV")
    parser.add_argument("--tier2-csv", type=str, default=None, help="Explicit path to Tier-2 scientific dossier summary CSV")
    parser.add_argument("--tic", type=str, default=None, help="Target TIC ID to process individually")
    parser.add_argument("--force-all", action="store_true", help="Bypass Stage 3A gate and drill all candidates")
    parser.add_argument("--min-snr", type=float, default=3.8, help="Minimum SPOC red-noise SNR for CTOI inclusion (default: 3.8)")
    args = parser.parse_args()

    run_antares_packager(
        results_dir=args.results_dir,
        target_tic=args.tic,
        force_all=args.force_all,
        min_snr=args.min_snr,
        tier1_csv=args.tier1_csv,
        tier2_csv=args.tier2_csv
    )
