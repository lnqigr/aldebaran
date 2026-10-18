#!/usr/bin/env python3
"""
search_toi_companions.py (Aldebaran v9.0)
========================================
Dedicated High-Yield Sibling Exoplanet Search Engine on Known NASA TOI Hosts.

Features:
- Ingests official TOI ephemerides (P, T0, T_dur, Depth) from local cache or NASA TAP.
- Cross-matches TOI hosts against SPOC 2-minute cadence light curves across specified sectors.
- In-transit masking ([-1.0*T_dur, +1.0*T_dur]) with linear flux interpolation prior to
  piecewise Savitzky-Golay detrending to prevent Gibbs edge ringing.
- Residual Box Least Squares (BLS) periodic search for shallow sub-Earth / super-Earth siblings.
- Protects against harmonic/alias false positives (m/n in {1, 1/2, 2, 1/3, 3, 2/3, 3/2, 1/4, 4}).
- Saves publication-grade discovery plots and candidate summaries.
- Auditable per-target outcomes (CANDIDATE, REJECT, EMPTY, SKIP, ERROR) with failure tracking.
- Verified output persistence, truthful zero-candidate artifacts, and run identity.
- Portable relative artifact references and non-zero exit propagation on partial failures.
"""

import os
import sys
import shutil
import argparse
import json
import logging
import traceback
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aldebaran as ald

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TOI_Companions")

CANDIDATE_SCHEMA_VERSION = "v1.1"

CANDIDATE_COLUMNS = [
    "schema_version",
    "pipeline_version",
    "run_id",
    "tic_id",
    "sector",
    "is_toi_host",
    "known_toi_context",
    "candidate_type",
    "ephemeris_source",
    "ephemeris_status",
    "ephemeris_fallback_reason",
    "time_system",
    "time_standard",
    "epoch_offset_btjd",
    "period_units",
    "duration_units",
    "depth_units",
    "transit_measurement_source",
    "timing_uncertainty_source",
    "period_days",
    "t0_btjd",
    "discovery_period_days",
    "discovery_t0_btjd",
    "refined_period_days",
    "refined_t0_btjd",
    "duration_hours",
    "min_kepler_dur_hours",
    "max_kepler_dur_hours",
    "n_transits",
    "depth_ppm",
    "crowdsap",
    "sde",
    "r_planet_earth",
    "r_star_sun",
    "m_star_sun",
    "teff_k",
    "gaia_ruwe",
    "odd_even_sigma",
    "centroid_shift_arcsec",
    "archival_sectors",
    "n_archival_sectors",
    "archival_sectors_requested",
    "archival_sectors_attempted",
    "archival_sectors_used",
    "archival_products_requested",
    "archival_products_attempted",
    "archival_products_used",
    "archival_products_failed",
    "archival_snr_sigma",
    "archival_trials",
    "archival_snr_crit",
    "est_multi_sde_proxy",
    "multi_sde",
    "cramer_rao_sigma_p_sec",
    "cramer_rao_sigma_t0_sec",
    "covariance_p_t0",
    "plot_path",
    "exofop_url"
]

AUDIT_COLUMNS = [
    "run_id",
    "timestamp",
    "sector",
    "tic_id",
    "status",
    "reason",
    "message",
    "error_detail"
]


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Aldebaran TOI Sibling Exoplanet Search Engine")
    parser.add_argument("--sectors", type=int, nargs="+", default=[106, 105, 104, 103, 102, 101, 100, 99],
                        help="TESS sectors to search for TOI host companion planets")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--limit-per-sector", type=int, default=500, help="Max TOI hosts to screen per sector")
    parser.add_argument("--output-dir", type=str, default="tess_results/toi_companions",
                        help="Directory to save companion candidate reports and plots")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run identifier (default: auto-generated timestamp)")
    parser.add_argument("--append", action="store_true", default=False,
                        help="Append to existing summary report instead of starting a fresh run")
    return parser.parse_args(args_list)


def _init_worker(output_dir: str, toi_ephemerides: dict, log_file: str = None, run_id: str = None):
    """Initializer for child processes in ProcessPoolExecutor (supports both fork and spawn)."""
    ald.configure_environment(output_dir=output_dir, toi_ephemerides=toi_ephemerides, log_file=log_file, run_id=run_id)


def query_sector_toi_hosts(sector: int, toi_hosts_set: set) -> tuple[list, str | None]:
    """
    Finds all TOI hosts with SPOC 2-minute light curves in the specified sector.
    Returns (matched_tics, error_message).
    """
    try:
        from astroquery.mast import Observations
        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries"
        )
        if len(obs) == 0:
            return [], None
        df_obs = obs.to_pandas()
        spoc_mask = df_obs["target_name"].astype(str).str.isdigit() & (df_obs["provenance_name"] == "SPOC")
        spoc_tics = set(df_obs.loc[spoc_mask, "target_name"].astype(str).str.lstrip("0"))
        cleaned_host_set = {str(t).lstrip("0") for t in toi_hosts_set}
        matched = sorted(list(spoc_tics & cleaned_host_set))
        return matched, None
    except Exception as e:
        err_msg = f"Error querying MAST for sector {sector}: {e}"
        logger.warning(err_msg)
        return [], err_msg


def get_toi_hosts_in_sector(sector: int, toi_hosts_set: set) -> list:
    """Legacy compatibility wrapper."""
    matched, _ = query_sector_toi_hosts(sector, toi_hosts_set)
    return matched


def process_toi_host(tic_id: str, sector: int, run_id: str = "run_default", attempt_id: str = None) -> dict:
    """
    Executes masking and residual BLS search on a single TOI host star.
    Returns a structured audit record with status, reason, message, and error details.
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    tid_str = str(tic_id).lstrip("0")
    if attempt_id is None:
        import uuid
        attempt_id = f"toi_{str(tic_id).lstrip('0')}_s{sector}_{uuid.uuid4().hex[:8]}"

    # Verify that the worker has access to the required TOI ephemeris context
    known_ephem = ald.get_known_transits_for_tic(tid_str)
    if not known_ephem:
        err_msg = f"TIC {tic_id} queued as TOI host but has no known TOI ephemeris in catalog context"
        logger.error(err_msg)
        return {
            "run_id": run_id,
            "timestamp": ts,
            "sector": int(sector),
            "tic_id": str(tic_id),
            "status": "ERROR",
            "reason": "MISSING_EPHEMERIS_CONTEXT",
            "message": err_msg,
            "error_detail": "Empty ephemeris list returned by get_known_transits_for_tic",
            "candidate": None,
            "plot_path": ""
        }

    try:
        import inspect
        supports_ids = False
        try:
            sig = inspect.signature(ald.analyze_single_target)
            supports_ids = ("run_id" in sig.parameters and "attempt_id" in sig.parameters)
        except (ValueError, TypeError):
            supports_ids = False

        if supports_ids:
            res = ald.analyze_single_target(tic_id, sector, run_id=run_id, attempt_id=attempt_id)
        else:
            res = ald.analyze_single_target(tic_id, sector)

        tic_out, disp, msg, cand_data = res[0], res[1], res[2], res[3]
        disp_clean = str(disp).strip()

        if disp_clean == "CANDIDATE" and cand_data is not None:
            cand_copy = dict(cand_data)
            cand_copy["run_id"] = run_id
            cand_copy["sector"] = int(sector)
            cand_copy["tic_id"] = str(tic_id)

            # Ensure plot_path contains run_id for unique per-run namespace
            target_plot_filename = f"candidate_TIC_{tic_id}_s{sector}_{run_id}.png"
            raw_plot_path = cand_copy.get("plot_path", "")
            target_full_path = os.path.join(ald.OUTPUT_DIR, target_plot_filename)

            # If raw_plot_path was an existing file (e.g. from mock or legacy name), copy to target filename
            if raw_plot_path:
                src_cand = raw_plot_path if os.path.isabs(raw_plot_path) else os.path.join(ald.OUTPUT_DIR, raw_plot_path)
                if not os.path.exists(src_cand) and os.path.exists(raw_plot_path):
                    src_cand = raw_plot_path
                if os.path.exists(src_cand) and os.path.abspath(src_cand) != os.path.abspath(target_full_path):
                    shutil.copy2(src_cand, target_full_path)

            cand_copy["plot_path"] = target_plot_filename
            logger.info(">>> SIBLING CANDIDATE DISCOVERED! TIC %s (Sector %d, SDE=%.2f, P=%.4f d)",
                        tic_id, sector, cand_copy.get("sde", 0.0), cand_copy.get("period_days", 0.0))
            return {
                "run_id": run_id,
                "timestamp": ts,
                "sector": int(sector),
                "tic_id": str(tic_id),
                "status": "CANDIDATE",
                "reason": "CANDIDATE_FOUND",
                "message": msg,
                "error_detail": "",
                "candidate": cand_copy,
                "plot_path": target_plot_filename
            }
        elif disp_clean == "ERROR":
            logger.warning("[Sector %d] [TIC %s] Core analysis ERROR: %s", sector, tic_id, msg)
            arch_diag = getattr(res, "archival_diagnostic", None)
            if arch_diag is None and "archival" in msg.lower():
                arch_diag = ald.get_latest_archival_diagnostic(
                    tic_id, sector, output_dir=ald.OUTPUT_DIR, run_id=run_id, attempt_id=attempt_id
                )
            err_detail = f"{msg} | ArchivalDiag: {json.dumps(arch_diag)}" if arch_diag else msg
            return {
                "run_id": run_id,
                "timestamp": ts,
                "sector": int(sector),
                "tic_id": str(tic_id),
                "status": "ERROR",
                "reason": "PIPELINE_ERROR",
                "message": msg,
                "error_detail": err_detail,
                "candidate": None,
                "plot_path": ""
            }
        else:
            reason = msg.split(":")[0].strip() if ":" in msg else disp_clean
            arch_diag = getattr(res, "archival_diagnostic", None)
            if arch_diag is None and disp_clean == "REJECT_ARCHIVAL":
                arch_diag = ald.get_latest_archival_diagnostic(
                    tic_id, sector, output_dir=ald.OUTPUT_DIR, run_id=run_id, attempt_id=attempt_id
                )
            elif disp_clean != "REJECT_ARCHIVAL":
                arch_diag = None

            err_detail = json.dumps(arch_diag) if arch_diag else ""
            return {
                "run_id": run_id,
                "timestamp": ts,
                "sector": int(sector),
                "tic_id": str(tic_id),
                "status": disp_clean,
                "reason": reason,
                "message": msg,
                "error_detail": err_detail,
                "candidate": None,
                "plot_path": ""
            }
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"Unhandled exception processing TIC {tic_id} in sector {sector}: {e}"
        logger.error(err_msg)
        return {
            "run_id": run_id,
            "timestamp": ts,
            "sector": int(sector),
            "tic_id": str(tic_id),
            "status": "ERROR",
            "reason": type(e).__name__,
            "message": str(e),
            "error_detail": tb,
            "candidate": None,
            "plot_path": ""
        }


def atomic_write_json(file_path: str, data: dict):
    """Atomically writes JSON to file_path using a temporary file in the same directory."""
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = os.path.join(dir_name, f".tmp_{os.path.basename(file_path)}.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def atomic_write_csv(csv_path: str, df: pd.DataFrame, index: bool = False, header: bool = True):
    """
    Atomically writes a DataFrame to csv_path using a temporary file in the same directory.
    Guarantees that on failure, the existing file is preserved intact.
    """
    dir_name = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = os.path.join(dir_name, f".tmp_{os.path.basename(csv_path)}.{os.getpid()}")
    try:
        df.to_csv(tmp_path, index=index, header=header)
        with open(tmp_path, "a", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def initialize_output_artifacts(output_dir: str, run_id: str, append_mode: bool = False):
    """
    Initializes output directory, per-run audit and candidate reports, and cumulative reports.
    Preserves historical run evidence by maintaining dedicated per-run files
    (toi_companions_summary_{run_id}.csv and toi_companions_audit_{run_id}.csv)
    while updating cumulative working files.
    Rejects collisions if run_id already exists and append_mode=False to preserve
    completed run metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "toi_companions_summary.csv")
    audit_file = os.path.join(output_dir, "toi_companions_audit.csv")
    run_report_file = os.path.join(output_dir, f"toi_companions_summary_{run_id}.csv")
    run_audit_file = os.path.join(output_dir, f"toi_companions_audit_{run_id}.csv")
    per_run_manifest = os.path.join(output_dir, f"run_summary_{run_id}.json")

    # Reject run ID collision on fresh run (not append_mode) to protect completed run metadata
    if not append_mode:
        id_exists = os.path.exists(per_run_manifest) or os.path.exists(run_audit_file) or os.path.exists(run_report_file)
        if not id_exists and os.path.exists(audit_file):
            try:
                df_aud = pd.read_csv(audit_file)
                if "run_id" in df_aud.columns and (df_aud["run_id"].astype(str) == str(run_id)).any():
                    id_exists = True
            except Exception as e:
                raise RuntimeError(f"Cannot verify run collision: failed to read audit file '{audit_file}': {e}")
        if id_exists:
            raise FileExistsError(
                f"Run ID '{run_id}' already exists in output directory '{output_dir}'. "
                f"Cannot overwrite completed run metadata without --append. Specify a new run ID or pass --append to resume."
            )

        # Fresh run: initialize dedicated per-run files
        df_empty_report = pd.DataFrame(columns=CANDIDATE_COLUMNS)
        atomic_write_csv(run_report_file, df_empty_report, index=False)

        df_empty_audit = pd.DataFrame(columns=AUDIT_COLUMNS)
        atomic_write_csv(run_audit_file, df_empty_audit, index=False)

        # Reset cumulative/working files for this fresh run
        atomic_write_csv(report_file, df_empty_report, index=False)
        atomic_write_csv(audit_file, df_empty_audit, index=False)
    else:
        # Append mode: ensure per-run and cumulative files exist
        for target_csv in (report_file, run_report_file):
            if os.path.exists(target_csv) and os.path.getsize(target_csv) > 0:
                try:
                    df_check = pd.read_csv(target_csv, nrows=0)
                    if list(df_check.columns) != CANDIDATE_COLUMNS:
                        raise RuntimeError(
                            f"Incompatible candidate CSV schema in '{target_csv}'. "
                            f"Existing columns do not match schema {CANDIDATE_SCHEMA_VERSION} ({len(CANDIDATE_COLUMNS)} cols). "
                            f"Refusing to append to incompatible historical file to prevent corrupting historical artifacts."
                        )
                except RuntimeError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to inspect candidate CSV '{target_csv}' for append mode: {e}")

        if not os.path.exists(run_report_file):
            df_empty_report = pd.DataFrame(columns=CANDIDATE_COLUMNS)
            atomic_write_csv(run_report_file, df_empty_report, index=False)

        if not os.path.exists(run_audit_file):
            df_empty_audit = pd.DataFrame(columns=AUDIT_COLUMNS)
            atomic_write_csv(run_audit_file, df_empty_audit, index=False)

        if not os.path.exists(report_file):
            df_empty_report = pd.DataFrame(columns=CANDIDATE_COLUMNS)
            atomic_write_csv(report_file, df_empty_report, index=False)

        if not os.path.exists(audit_file):
            df_empty_audit = pd.DataFrame(columns=AUDIT_COLUMNS)
            atomic_write_csv(audit_file, df_empty_audit, index=False)

    return report_file, audit_file, run_report_file, run_audit_file


def _append_candidate_dedup(csv_path: str, df_cand: pd.DataFrame, tic, sector, run_id: str):
    """
    Appends a candidate row to csv_path, replacing any prior incomplete/interrupted
    record for the same (run_id, tic_id, sector) to ensure idempotency without duplicates.
    Uses atomic file replacement when replacing an existing record.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        atomic_write_csv(csv_path, df_cand, index=False)
        return

    try:
        df_exist = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read candidate CSV '{csv_path}' for append dedup: {e}")

    if list(df_exist.columns) != list(df_cand.columns):
        raise RuntimeError(
            f"Incompatible candidate CSV schema in '{csv_path}'. "
            f"Existing columns ({len(df_exist.columns)}) do not match candidate schema {CANDIDATE_SCHEMA_VERSION} ({len(df_cand.columns)} cols). "
            f"Refusing to append to incompatible historical file."
        )

    required_cols = {"run_id", "tic_id", "sector"}
    if not required_cols.issubset(df_exist.columns):
        raise RuntimeError(
            f"Candidate CSV '{csv_path}' missing required columns for append dedup: {required_cols - set(df_exist.columns)}"
        )

    mask = (df_exist["run_id"].astype(str) == str(run_id)) & \
           (df_exist["tic_id"].astype(str).str.lstrip("0") == str(tic).lstrip("0")) & \
           (df_exist["sector"].astype(int) == int(sector))
    if mask.any():
        df_exist = df_exist[~mask]
        if len(df_exist) == 0:
            atomic_write_csv(csv_path, df_cand, index=False)
        else:
            df_updated = pd.concat([df_exist, df_cand], ignore_index=True)
            atomic_write_csv(csv_path, df_updated, index=False)
        return

    # If row was not already present, append using mode='a'
    df_cand.to_csv(csv_path, mode="a", index=False, header=False)


def _remove_candidate_row(csv_path: str, tic, sector, run_id: str):
    """
    Removes any active candidate row for (run_id, tic_id, sector) from csv_path,
    ensuring that if a previously saved or incomplete candidate is re-analyzed
    and yields a non-candidate outcome (REJECT, EMPTY, SKIP, ERROR), active candidate
    tables do not retain stale candidate records.
    Uses atomic file replacement to preserve the existing file on failure.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read candidate CSV '{csv_path}' for row removal: {e}")

    required_cols = {"run_id", "tic_id", "sector"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(
            f"Candidate CSV '{csv_path}' missing required columns for row removal: {required_cols - set(df.columns)}"
        )

    mask = (df["run_id"].astype(str) == str(run_id)) & \
           (df["tic_id"].astype(str).str.lstrip("0") == str(tic).lstrip("0")) & \
           (df["sector"].astype(int) == int(sector))
    if mask.any():
        df_cleaned = df[~mask]
        atomic_write_csv(csv_path, df_cleaned, index=False)


def run_search(args) -> tuple[int, dict]:
    """
    Core execution logic for TOI companion search.
    Returns (exit_code, run_summary_dict).
    """
    start_time = datetime.now(timezone.utc)
    run_id = args.run_id or f"toi_run_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"

    # Explicitly configure environment in parent process
    ald.configure_environment(output_dir=args.output_dir, run_id=run_id)

    report_file, audit_file, run_report_file, run_audit_file = initialize_output_artifacts(
        args.output_dir, run_id, append_mode=args.append
    )

    summary_json_path = os.path.join(args.output_dir, "run_summary.json")
    per_run_manifest_path = os.path.join(args.output_dir, f"run_summary_{run_id}.json")

    # Write initial startup manifest immediately to eliminate stale summary files
    initial_summary = {
        "run_id": run_id,
        "run_status": "STARTING",
        "exit_code": None,
        "start_time_utc": start_time.isoformat(),
        "sectors": args.sectors,
        "workers": args.workers,
        "limit_per_sector": args.limit_per_sector,
        "total_queued": 0,
        "total_processed": 0,
        "counts": {
            "CANDIDATE": 0,
            "REJECT": 0,
            "EMPTY": 0,
            "SKIP": 0,
            "ERROR": 0
        },
        "sector_query_errors": 0,
        "candidates_count": 0,
        "cumulative_candidates_count": 0,
        "summary_csv": os.path.basename(run_report_file),
        "audit_csv": os.path.basename(run_audit_file),
        "cumulative_summary_csv": os.path.basename(report_file),
        "cumulative_audit_csv": os.path.basename(audit_file),
        "per_run_manifest": os.path.basename(per_run_manifest_path)
    }
    atomic_write_json(summary_json_path, initial_summary)
    atomic_write_json(per_run_manifest_path, initial_summary)

    logger.info("======================================================================")
    logger.info("ALDEBARAN v9.0 - AUDITABLE TOI SIBLING EXOPLANET SEARCH ENGINE")
    logger.info("Run ID: %s", run_id)
    logger.info("Target Sectors: %s | Workers: %d", args.sectors, args.workers)
    logger.info("Output Directory: %s", args.output_dir)
    logger.info("======================================================================")

    try:
        # 1. Load TOI catalog ephemerides
        known_tics, eb_tics, toi_ephem_dict = ald.load_known_catalogs_and_toi_ephemerides()
        toi_host_set = set(toi_ephem_dict.keys())
        # Update global in-memory ephemeris dictionary and configure run
        ald.configure_environment(output_dir=args.output_dir, toi_ephemerides=toi_ephem_dict, run_id=run_id)
        logger.info("Loaded %d TOI host systems with cached orbital ephemerides.", len(toi_host_set))
    except Exception as exc:
        end_time = datetime.now(timezone.utc)
        logger.error("Startup failure loading TOI catalog ephemerides: %s", exc)
        startup_err_summary = dict(initial_summary)
        startup_err_summary.update({
            "run_status": "STARTUP_ERROR",
            "exit_code": 1,
            "end_time_utc": end_time.isoformat(),
            "error": str(exc),
            "error_detail": traceback.format_exc()
        })
        atomic_write_json(summary_json_path, startup_err_summary)
        atomic_write_json(per_run_manifest_path, startup_err_summary)
        raise

    total_candidates = []
    counts = {
        "CANDIDATE": 0,
        "REJECT": 0,
        "EMPTY": 0,
        "SKIP": 0,
        "ERROR": 0
    }
    total_queued = 0
    total_processed = 0
    sector_query_errors = 0

    audit_records = []

    # R2a/R2b: Check for already completed targets when in append mode (idempotent resume for this run_id)
    completed_in_this_run = {}
    if args.append:
        # 1. Load and validate audit records for this run_id
        target_audit_file = run_audit_file if os.path.exists(run_audit_file) else audit_file
        latest_audits = {}
        if os.path.exists(target_audit_file) and os.path.getsize(target_audit_file) > 0:
            try:
                df_aud = pd.read_csv(target_audit_file)
            except Exception as e:
                raise RuntimeError(f"Cannot resume run '{run_id}': failed to read audit file '{target_audit_file}': {e}")
            required_audit_cols = {"run_id", "tic_id", "sector", "status"}
            if not required_audit_cols.issubset(df_aud.columns):
                raise RuntimeError(
                    f"Cannot resume run '{run_id}': audit file '{target_audit_file}' missing required columns {required_audit_cols - set(df_aud.columns)}"
                )
            df_this = df_aud[df_aud["run_id"].astype(str) == str(run_id)]
            for _, r in df_this.iterrows():
                key = (str(r.get("tic_id")).lstrip("0"), int(r.get("sector")))
                latest_audits[key] = (str(r.get("status", "")).strip(), r)

        # 2. Load and validate candidate records for this run_id from both per-run and cumulative reports
        per_run_cands = {}
        if os.path.exists(run_report_file) and os.path.getsize(run_report_file) > 0:
            try:
                df_rep = pd.read_csv(run_report_file)
            except Exception as e:
                raise RuntimeError(f"Cannot resume run '{run_id}': failed to read candidate report '{run_report_file}': {e}")
            required_cand_cols = {"run_id", "tic_id", "sector"}
            if not required_cand_cols.issubset(df_rep.columns):
                raise RuntimeError(
                    f"Cannot resume run '{run_id}': candidate report '{run_report_file}' missing required columns {required_cand_cols - set(df_rep.columns)}"
                )
            df_cands_this = df_rep[df_rep["run_id"].astype(str) == str(run_id)]
            for _, r in df_cands_this.iterrows():
                key = (str(r.get("tic_id")).lstrip("0"), int(r.get("sector")))
                per_run_cands[key] = r.to_dict()

        cum_cands = {}
        if os.path.exists(report_file) and os.path.getsize(report_file) > 0:
            try:
                df_cum = pd.read_csv(report_file)
            except Exception as e:
                raise RuntimeError(f"Cannot resume run '{run_id}': failed to read cumulative candidate report '{report_file}': {e}")
            required_cand_cols = {"run_id", "tic_id", "sector"}
            if not required_cand_cols.issubset(df_cum.columns):
                raise RuntimeError(
                    f"Cannot resume run '{run_id}': cumulative candidate report '{report_file}' missing required columns {required_cand_cols - set(df_cum.columns)}"
                )
            df_cands_cum = df_cum[df_cum["run_id"].astype(str) == str(run_id)]
            for _, r in df_cands_cum.iterrows():
                key = (str(r.get("tic_id")).lstrip("0"), int(r.get("sector")))
                cum_cands[key] = r.to_dict()

        # 3. Reconcile active candidate tables against latest audit state
        all_active_cand_keys = set(per_run_cands.keys()) | set(cum_cands.keys())
        for key in list(all_active_cand_keys):
            latest_st, _ = latest_audits.get(key, (None, None))
            # If the latest terminal audit is a non-candidate (REJECT, EMPTY, SKIP, ERROR)
            # or if there is no audit record at all, any active candidate row is stale
            # (e.g. from an interrupted invalidation on retry) and must be purged.
            if latest_st in ("REJECT", "EMPTY", "SKIP", "ERROR") or latest_st is None:
                logger.warning(
                    "Target TIC %s (Sector %d) has active candidate row in run %s, but latest audit status is '%s'. Purging stale candidate row.",
                    key[0], key[1], run_id, latest_st
                )
                _remove_candidate_row(run_report_file, key[0], key[1], run_id)
                _remove_candidate_row(report_file, key[0], key[1], run_id)
                per_run_cands.pop(key, None)
                cum_cands.pop(key, None)

        # 4. Populate completed_in_this_run and total_candidates from reconciled state
        for key, (st_val, r_aud) in latest_audits.items():
            if st_val in ("REJECT", "EMPTY", "SKIP"):
                if key in per_run_cands or key in cum_cands:
                    raise RuntimeError(
                        f"Cannot resume run '{run_id}': target TIC {key[0]} (Sector {key[1]}) has terminal audit '{st_val}' "
                        f"but candidate row still present in candidate tables."
                    )
                completed_in_this_run[key] = st_val
            elif st_val == "CANDIDATE":
                cand_dict = per_run_cands.get(key) or cum_cands.get(key)
                rel_plot = str(cand_dict.get("plot_path", "")).strip() if cand_dict else ""
                plot_full = os.path.join(args.output_dir, rel_plot) if not os.path.isabs(rel_plot) else rel_plot
                has_valid_plot = bool(rel_plot and os.path.exists(plot_full) and os.path.getsize(plot_full) > 0)
                is_in_both = (key in per_run_cands) and (key in cum_cands)
                if has_valid_plot and is_in_both:
                    completed_in_this_run[key] = "CANDIDATE"
                    total_candidates.append(cand_dict)
                else:
                    if not has_valid_plot:
                        logger.warning(
                            "Target TIC %s (Sector %d) audited as CANDIDATE in run %s, but plot file '%s' is missing or empty. Re-queueing for reprocessing.",
                            key[0], key[1], run_id, rel_plot
                        )
                    else:
                        logger.warning(
                            "Target TIC %s (Sector %d) audited as CANDIDATE in run %s, but candidate row is missing from cumulative report (interrupted commit). Re-queueing for reprocessing.",
                            key[0], key[1], run_id
                        )
            elif st_val == "ERROR":
                logger.info(
                    "Target TIC %s (Sector %d) previously had ERROR in run %s. Re-queueing for retry.",
                    key[0], key[1], run_id
                )

    for sec in args.sectors:
        logger.info("----------------------------------------------------------------------")
        logger.info("Scanning Sector %d for known TOI planetary hosts...", sec)
        toi_candidates, q_err = query_sector_toi_hosts(sec, toi_host_set)

        if q_err is not None:
            sector_query_errors += 1
            audit_record = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sector": int(sec),
                "tic_id": "ALL",
                "status": "ERROR",
                "reason": "SECTOR_QUERY_FAILED",
                "message": q_err,
                "error_detail": q_err
            }
            audit_records.append(audit_record)
            df_q_err = pd.DataFrame([audit_record], columns=AUDIT_COLUMNS)
            df_q_err.to_csv(run_audit_file, mode="a", index=False, header=False)
            df_q_err.to_csv(audit_file, mode="a", index=False, header=False)
            logger.error("Sector %d query failed: %s", sec, q_err)
            continue

        logger.info("Sector %d: Found %d TOI host systems with SPOC light curves.", sec, len(toi_candidates))
        if not toi_candidates:
            continue

        selected_tics = toi_candidates[:args.limit_per_sector]
        total_queued += len(selected_tics)
        logger.info("Queueing %d TOI hosts for companion search across %d workers...", len(selected_tics), args.workers)

        # Filter out targets already completed by this run_id for idempotent resume
        pending_tics = []
        for tic in selected_tics:
            key = (str(tic).lstrip("0"), int(sec))
            if key in completed_in_this_run:
                st = completed_in_this_run[key]
                if st not in counts:
                    counts[st] = 0
                counts[st] += 1
                total_processed += 1
                logger.info(
                    "[%d/%d] [Sector %d] [TIC %s] %s (ALREADY_COMPLETED - Idempotent resume)",
                    total_processed, total_queued, sec, tic, st
                )
            else:
                pending_tics.append(tic)

        def record_result(res, tic, sector):
            nonlocal total_processed
            status = res.get("status", "ERROR")
            if status not in counts:
                counts[status] = 0
            counts[status] += 1
            total_processed += 1

            # Commit audit row to both run-specific audit and cumulative audit FIRST
            audit_row = {
                "run_id": run_id,
                "timestamp": res.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "sector": int(res.get("sector", sector)),
                "tic_id": str(res.get("tic_id", tic)),
                "status": status,
                "reason": res.get("reason", ""),
                "message": res.get("message", ""),
                "error_detail": res.get("error_detail", "")
            }
            audit_records.append(audit_row)
            df_audit_row = pd.DataFrame([audit_row], columns=AUDIT_COLUMNS)
            df_audit_row.to_csv(run_audit_file, mode="a", index=False, header=False)
            df_audit_row.to_csv(audit_file, mode="a", index=False, header=False)

            # If CANDIDATE, commit candidate row to CSV
            if status == "CANDIDATE" and res.get("candidate"):
                cand = res["candidate"]
                total_candidates.append(cand)
                if "schema_version" not in cand or not cand["schema_version"]:
                    cand["schema_version"] = CANDIDATE_SCHEMA_VERSION
                if "pipeline_version" not in cand or not cand["pipeline_version"]:
                    cand["pipeline_version"] = ald.PIPELINE_VERSION
                if "time_system" not in cand or not cand["time_system"]:
                    cand["time_system"] = "BTJD"
                if "period_units" not in cand or not cand["period_units"]:
                    cand["period_units"] = "days"
                if "duration_units" not in cand or not cand["duration_units"]:
                    cand["duration_units"] = "hours"
                if "depth_units" not in cand or not cand["depth_units"]:
                    cand["depth_units"] = "ppm"
                if "covariance_p_t0" not in cand:
                    cand["covariance_p_t0"] = np.nan
                row_dict = {col: cand.get(col, "") for col in CANDIDATE_COLUMNS}
                df_cand = pd.DataFrame([row_dict], columns=CANDIDATE_COLUMNS)

                # Commit to cumulative report_file (mode="a")
                _append_candidate_dedup(report_file, df_cand, tic, sector, run_id)
                # Commit to run_report_file (mode="a")
                _append_candidate_dedup(run_report_file, df_cand, tic, sector, run_id)
            else:
                # If retry/re-analysis yields a non-candidate outcome (REJECT, EMPTY, SKIP, ERROR),
                # reconcile active candidate tables by removing any stale candidate row for this target
                _remove_candidate_row(report_file, tic, sector, run_id)
                _remove_candidate_row(run_report_file, tic, sector, run_id)

            logger.info(
                "[%d/%d] [Sector %d] [TIC %s] %s (%s)",
                total_processed, total_queued, sector, tic, status, res.get("reason", "")
            )

        if args.workers <= 1:
            for tic in pending_tics:
                try:
                    res = process_toi_host(tic, sec, run_id)
                except Exception as exc:
                    tb = traceback.format_exc()
                    res = {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "sector": int(sec),
                        "tic_id": str(tic),
                        "status": "ERROR",
                        "reason": type(exc).__name__,
                        "message": str(exc),
                        "error_detail": tb,
                        "candidate": None,
                        "plot_path": ""
                    }
                record_result(res, tic, sec)
        else:
            if pending_tics:
                with ProcessPoolExecutor(
                    max_workers=min(args.workers, len(pending_tics)),
                    initializer=_init_worker,
                    initargs=(args.output_dir, toi_ephem_dict, None, run_id)
                ) as executor:
                    futures = {
                        executor.submit(process_toi_host, tic, sec, run_id): (tic, sec)
                        for tic in pending_tics
                    }
                    for fut in as_completed(futures):
                        tic, sector = futures[fut]
                        try:
                            res = fut.result()
                        except Exception as exc:
                            tb = traceback.format_exc()
                            res = {
                                "run_id": run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "sector": int(sector),
                                "tic_id": str(tic),
                                "status": "ERROR",
                                "reason": type(exc).__name__,
                                "message": str(exc),
                                "error_detail": tb,
                                "candidate": None,
                                "plot_path": ""
                            }
                        record_result(res, tic, sector)

    end_time = datetime.now(timezone.utc)

    # Determine overall run outcome status and exit code according to R1 rules
    has_candidates = counts["CANDIDATE"] > 0
    has_worker_errors = counts["ERROR"] > 0
    has_query_errors = sector_query_errors > 0
    has_any_errors = has_worker_errors or has_query_errors

    if total_queued == 0:
        if has_query_errors:
            run_status = "FAILED_QUERY"
            exit_code = 1
        else:
            run_status = "ZERO_HOSTS"
            exit_code = 0
    elif has_any_errors:
        if has_candidates:
            # Candidates were discovered but some targets or queries failed
            run_status = "CANDIDATES_FOUND_WITH_ERRORS"
            exit_code = 1
        elif total_processed > 0 and counts["ERROR"] == total_processed and not has_query_errors:
            run_status = "ALL_ERRORS"
            exit_code = 1
        elif total_processed == 0 and has_query_errors:
            run_status = "FAILED_QUERY"
            exit_code = 1
        else:
            # Partial failure (e.g. some targets rejected, but query error or worker error occurred)
            run_status = "PARTIAL_ERROR"
            exit_code = 1
    else:
        # Fully error-free run
        if has_candidates:
            run_status = "CANDIDATES_FOUND"
            exit_code = 0
        else:
            run_status = "ALL_REJECTED"
            exit_code = 0

    run_candidates_count = len(total_candidates)

    cumulative_candidates_count = run_candidates_count
    if os.path.exists(report_file):
        try:
            df_all_cand = pd.read_csv(report_file)
            cumulative_candidates_count = len(df_all_cand)
        except Exception:
            pass

    run_summary = {
        "run_id": run_id,
        "run_status": run_status,
        "exit_code": exit_code,
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "sectors": args.sectors,
        "workers": args.workers,
        "limit_per_sector": args.limit_per_sector,
        "total_queued": total_queued,
        "total_processed": total_processed,
        "counts": counts,
        "sector_query_errors": sector_query_errors,
        "candidates_count": run_candidates_count,
        "cumulative_candidates_count": cumulative_candidates_count,
        "summary_csv": os.path.basename(run_report_file),
        "audit_csv": os.path.basename(run_audit_file),
        "cumulative_summary_csv": os.path.basename(report_file),
        "cumulative_audit_csv": os.path.basename(audit_file),
        "per_run_manifest": os.path.basename(per_run_manifest_path)
    }

    # Save final run summary atomically to both latest JSON and per-run manifest
    atomic_write_json(summary_json_path, run_summary)
    atomic_write_json(per_run_manifest_path, run_summary)

    logger.info("======================================================================")
    logger.info("TOI COMPANION SEARCH FINISHED")
    logger.info("Run ID: %s | Status: %s (Exit code: %d)", run_id, run_status, exit_code)
    logger.info("Targets: Queued=%d, Processed=%d", total_queued, total_processed)
    logger.info("Breakdown: %s", json.dumps(counts))
    logger.info("Run Candidates Discovered: %d (Cumulative: %d)", run_candidates_count, cumulative_candidates_count)
    logger.info("Artifacts:")
    logger.info("  Candidate Summary CSV: %s", report_file)
    logger.info("  Per-Target Audit CSV:  %s", audit_file)
    logger.info("  Latest Summary JSON:   %s", summary_json_path)
    logger.info("  Per-Run Manifest JSON: %s", per_run_manifest_path)
    logger.info("======================================================================")

    return exit_code, run_summary


def main():
    args = parse_args()
    exit_code, _ = run_search(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
