#!/usr/bin/env python3
"""
ALDEBARAN v9.1: Explicit-Input, Lossless Shard Reducer & Safe Downstream Orchestrator
====================================================================================
Consolidates candidate reports and scanned target histories from explicitly declared
shard artifacts with full source provenance, strict cryptographic verification,
lossless candidate preservation (no TIC-only dedup or SDE-drop), and guarded
downstream orchestration.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Exit codes
EXIT_SUCCESS = 0
EXIT_OUTPUT_DIR_EXISTS = 1
EXIT_USAGE_OR_ARG_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_EXECUTION_FAILURE = 4
EXIT_CLEANUP_FAILURE = 5
EXIT_BLOCKED_DOWNSTREAM = 6
EXIT_STAGE_FAILURE = 7

# Closed status enums (B2)
VALID_SHARD_STATUSES = {"COMPLETED", "FAILED", "MISSING", "PARTIAL", "SKIPPED", "EMPTY_NO_CANDIDATES"}
VALID_ARTIFACT_STATUSES = {"COMPLETED", "FAILED", "MISSING", "PARTIAL", "SKIPPED", "EMPTY_NO_CANDIDATES"}
NON_COMPLETE_STATUSES = {"FAILED", "MISSING", "PARTIAL", "SKIPPED"}

# Canonical measurement columns
CANONICAL_OUTPUT_COLUMNS = {
    "tic_id",
    "period_days",
    "t0_btjd",
    "duration_hours",
    "depth_ppm"
}

# Reserved provenance columns that must be namespaced if present in raw inputs
RESERVED_PROVENANCE_COLUMNS = {
    "source_run_id",
    "source_sector",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_row_ordinal",
    "observation_record_id"
}


class ShardReductionError(Exception):
    """Base exception for shard reduction errors."""
    def __init__(self, message: str, stage: str = "general", exit_code: int = EXIT_EXECUTION_FAILURE):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.exit_code = exit_code


class ReconciliationError(ShardReductionError):
    """Raised when loaded candidate rows do not reconcile with output records."""
    def __init__(self, message: str):
        super().__init__(message, stage="candidate_reconciliation", exit_code=EXIT_VALIDATION_ERROR)


class BlockedDownstreamIncompatibilityError(ShardReductionError):
    """Raised when duplicate TICs exist and unsafe downstream consumers are invoked."""
    def __init__(self, message: str):
        super().__init__(message, stage="downstream_compatibility_check", exit_code=EXIT_BLOCKED_DOWNSTREAM)


class StageExecutionError(ShardReductionError):
    """Raised when an orchestrated downstream stage fails or produces incomplete output."""
    def __init__(self, stage: str, returncode: int, stderr_text: str):
        msg = f"Stage '{stage}' failed with returncode {returncode}. Stderr: {stderr_text[:300]}"
        super().__init__(msg, stage=stage, exit_code=EXIT_STAGE_FAILURE)
        self.returncode = returncode
        self.stderr_text = stderr_text


def compute_file_sha256(filepath: str) -> str:
    """Computes the SHA256 hexadecimal hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def write_run_status(output_dir: str, status_dict: Dict[str, Any]) -> bool:
    """
    Safely writes run_status.json to output_dir.
    If writing fails, falls back to printing both errors to sys.stderr.
    """
    status_path = os.path.join(output_dir, "run_status.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2)
        return True
    except Exception as exc:
        sys.stderr.write(
            f"ERROR: Failed to write run_status.json to '{status_path}': {exc}\n"
            f"Original Status Payload: {json.dumps(status_dict)}\n"
        )
        sys.stderr.flush()
        return False


def validate_and_load_manifest(
    manifest_path: str,
    allow_partial: bool = False
) -> Tuple[Dict[str, Any], str, str, bool, List[Dict[str, Any]]]:
    """
    Validates manifest structure, resolves paths relative to manifest location,
    verifies SHA256 hashes, enforces artifact uniqueness, and detects conflicting shards.
    Returns: (manifest_data, manifest_dir, effective_survey_mode, is_run_partial, partial_caveats)
    """
    if not os.path.exists(manifest_path):
        raise ShardReductionError(
            f"Manifest file not found: '{manifest_path}'",
            stage="manifest_validation",
            exit_code=EXIT_VALIDATION_ERROR
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as exc:
        raise ShardReductionError(
            f"Failed to parse manifest JSON '{manifest_path}': {exc}",
            stage="manifest_validation",
            exit_code=EXIT_VALIDATION_ERROR
        )

    if not isinstance(manifest_data, dict):
        raise ShardReductionError(
            "Manifest root must be a JSON object",
            stage="manifest_validation",
            exit_code=EXIT_VALIDATION_ERROR
        )

    manifest_dir = os.path.abspath(os.path.dirname(manifest_path))

    # Required top-level fields
    run_id = str(manifest_data.get("run_id", "")).strip()
    if not run_id:
        raise ShardReductionError(
            "Manifest missing required non-empty 'run_id'",
            stage="manifest_validation",
            exit_code=EXIT_VALIDATION_ERROR
        )

    shards = manifest_data.get("shards")
    if not isinstance(shards, list) or len(shards) == 0:
        raise ShardReductionError(
            "Manifest 'shards' must be a non-empty list of shard specifications",
            stage="manifest_validation",
            exit_code=EXIT_VALIDATION_ERROR
        )

    declared_survey_mode = str(manifest_data.get("survey_mode", "complete")).strip().lower()
    partial_policy = manifest_data.get("partial_policy")
    is_partial_allowed = allow_partial or (declared_survey_mode == "partial") or (partial_policy in ("explicitly_allowed", "allow_partial", True))

    seen_candidate_paths: Set[str] = set()
    seen_history_paths: Set[str] = set()
    seen_sector_runs: Dict[str, str] = {}
    partial_caveats: List[Dict[str, Any]] = []
    is_run_partial = (declared_survey_mode == "partial")

    for idx, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ShardReductionError(
                f"Shard at index {idx} is not a valid JSON object",
                stage="manifest_validation",
                exit_code=EXIT_VALIDATION_ERROR
            )

        sector = shard.get("sector")
        sec_str = str(sector) if sector is not None else "N/A"
        source_run_id = str(shard.get("source_run_id", "")).strip()
        if not source_run_id:
            raise ShardReductionError(
                f"Shard {idx} (sector {sec_str}) missing required 'source_run_id'",
                stage="manifest_validation",
                exit_code=EXIT_VALIDATION_ERROR
            )

        shard_status = str(shard.get("status", "COMPLETED")).upper()
        if shard_status not in VALID_SHARD_STATUSES:
            raise ShardReductionError(
                f"Shard {idx} (sector {sec_str}) has invalid status '{shard_status}'. Must be one of {sorted(VALID_SHARD_STATUSES)}",
                stage="manifest_validation",
                exit_code=EXIT_VALIDATION_ERROR
            )

        cand_art = shard.get("candidate_artifact")
        hist_art = shard.get("history_artifact")

        # Shard must declare at least one artifact or explicit non-completed status (R1)
        if cand_art is None and hist_art is None:
            if shard_status not in ("FAILED", "MISSING", "SKIPPED"):
                raise ShardReductionError(
                    f"Shard {idx} (sector {sec_str}) has no artifacts declared and no FAILED/MISSING status",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )

        if shard_status in NON_COMPLETE_STATUSES:
            if not is_partial_allowed:
                raise ShardReductionError(
                    f"Shard {idx} (sector {sec_str}) has status '{shard_status}', but survey_mode is 'complete'. "
                    f"Explicit opt-in required for partial shards.",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
            is_run_partial = True
            partial_caveats.append({
                "shard_index": idx,
                "sector": sec_str,
                "source_run_id": source_run_id,
                "status": shard_status,
                "reason": "Shard declared with non-complete status"
            })

        # Detect conflicting runs for the same sector
        if sec_str != "N/A":
            if sec_str in seen_sector_runs and seen_sector_runs[sec_str] != source_run_id:
                raise ShardReductionError(
                    f"Conflicting source runs declared for sector {sec_str}: '{seen_sector_runs[sec_str]}' vs '{source_run_id}'",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
            seen_sector_runs[sec_str] = source_run_id

        # Validate candidate artifact
        if cand_art is not None:
            if not isinstance(cand_art, dict):
                raise ShardReductionError(
                    f"Shard {idx} candidate_artifact must be a JSON object",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
            c_status = str(cand_art.get("status", shard_status)).upper()
            if c_status not in VALID_ARTIFACT_STATUSES:
                raise ShardReductionError(
                    f"Shard {idx} (sector {sec_str}) candidate_artifact has invalid status '{c_status}'. Must be one of {sorted(VALID_ARTIFACT_STATUSES)}",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
            cand_art["status"] = c_status

            if c_status in NON_COMPLETE_STATUSES:
                if not is_partial_allowed:
                    raise ShardReductionError(
                        f"Shard {idx} (sector {sec_str}) candidate_artifact has status '{c_status}', but survey_mode is 'complete'. "
                        f"Explicit opt-in required for partial shards.",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                is_run_partial = True
                partial_caveats.append({
                    "shard_index": idx,
                    "sector": sec_str,
                    "source_run_id": source_run_id,
                    "artifact": "candidate_artifact",
                    "status": c_status
                })

            c_path = cand_art.get("path")
            if c_status not in ("FAILED", "MISSING", "SKIPPED"):
                if not c_path or not str(c_path).strip():
                    raise ShardReductionError(
                        f"Shard {idx} (sector {sec_str}) candidate_artifact missing required 'path'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                # SHA256 is strictly mandatory (R2)
                exp_hash = cand_art.get("sha256")
                if not exp_hash or not str(exp_hash).strip():
                    raise ShardReductionError(
                        f"Shard {idx} (sector {sec_str}) candidate_artifact missing required 'sha256'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                exp_hash = str(exp_hash).strip().upper()
                cand_art["sha256"] = exp_hash

                resolved_c_path = os.path.realpath(os.path.abspath(os.path.join(manifest_dir, c_path)))
                cand_art["resolved_path"] = resolved_c_path

                if resolved_c_path in seen_candidate_paths:
                    raise ShardReductionError(
                        f"Duplicate candidate artifact path declared in manifest: '{resolved_c_path}'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                seen_candidate_paths.add(resolved_c_path)

                if not os.path.exists(resolved_c_path):
                    raise ShardReductionError(
                        f"Declared candidate artifact not found on disk: '{resolved_c_path}'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                if os.path.getsize(resolved_c_path) == 0:
                    raise ShardReductionError(
                        f"Candidate artifact '{resolved_c_path}' is empty (0 bytes). A valid candidate CSV must define a schema.",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                actual_hash = compute_file_sha256(resolved_c_path)
                if actual_hash != exp_hash:
                    raise ShardReductionError(
                        f"SHA256 mismatch for candidate artifact '{resolved_c_path}': "
                        f"expected {exp_hash}, computed {actual_hash}",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )

        # Validate history artifact
        if hist_art is not None:
            if not isinstance(hist_art, dict):
                raise ShardReductionError(
                    f"Shard {idx} history_artifact must be a JSON object",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
            h_status = str(hist_art.get("status", shard_status)).upper()
            if h_status not in VALID_ARTIFACT_STATUSES:
                raise ShardReductionError(
                    f"Shard {idx} (sector {sec_str}) history_artifact has invalid status '{h_status}'. Must be one of {sorted(VALID_ARTIFACT_STATUSES)}",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
            hist_art["status"] = h_status

            if h_status in NON_COMPLETE_STATUSES:
                if not is_partial_allowed:
                    raise ShardReductionError(
                        f"Shard {idx} (sector {sec_str}) history_artifact has status '{h_status}', but survey_mode is 'complete'. "
                        f"Explicit opt-in required for partial shards.",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                is_run_partial = True
                partial_caveats.append({
                    "shard_index": idx,
                    "sector": sec_str,
                    "source_run_id": source_run_id,
                    "artifact": "history_artifact",
                    "status": h_status
                })

            h_path = hist_art.get("path")
            if h_status not in ("FAILED", "MISSING", "SKIPPED"):
                if not h_path or not str(h_path).strip():
                    raise ShardReductionError(
                        f"Shard {idx} (sector {sec_str}) history_artifact missing required 'path'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                # SHA256 is strictly mandatory (R2)
                exp_h_hash = hist_art.get("sha256")
                if not exp_h_hash or not str(exp_h_hash).strip():
                    raise ShardReductionError(
                        f"Shard {idx} (sector {sec_str}) history_artifact missing required 'sha256'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                exp_h_hash = str(exp_h_hash).strip().upper()
                hist_art["sha256"] = exp_h_hash

                resolved_h_path = os.path.realpath(os.path.abspath(os.path.join(manifest_dir, h_path)))
                hist_art["resolved_path"] = resolved_h_path

                if resolved_h_path in seen_history_paths:
                    raise ShardReductionError(
                        f"Duplicate history artifact path declared in manifest: '{resolved_h_path}'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                seen_history_paths.add(resolved_h_path)

                if not os.path.exists(resolved_h_path):
                    raise ShardReductionError(
                        f"Declared history artifact not found on disk: '{resolved_h_path}'",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                actual_h_hash = compute_file_sha256(resolved_h_path)
                if actual_h_hash != exp_h_hash:
                    raise ShardReductionError(
                        f"SHA256 mismatch for history artifact '{resolved_h_path}': "
                        f"expected {exp_h_hash}, computed {actual_h_hash}",
                        stage="manifest_validation",
                        exit_code=EXIT_VALIDATION_ERROR
                    )

    effective_survey_mode = "partial" if is_run_partial else declared_survey_mode
    return manifest_data, manifest_dir, effective_survey_mode, is_run_partial, partial_caveats


def check_and_prepare_output_dir(
    output_dir: str,
    manifest_path: Optional[str] = None,
    source_paths: Optional[List[str]] = None
) -> None:
    """
    Refuses execution if destination directory already exists and is non-empty,
    or if it aliases manifest directory or source artifact paths.
    Leaves existing populated directories completely untouched.
    """
    out_real = os.path.realpath(os.path.abspath(output_dir))

    if manifest_path is not None:
        man_real = os.path.realpath(os.path.abspath(manifest_path))
        man_dir_real = os.path.dirname(man_real) if os.path.isfile(man_real) else man_real
        if out_real == man_real or out_real == man_dir_real:
            raise ShardReductionError(
                f"Output directory '{output_dir}' aliases manifest location '{manifest_path}'. Refusing execution to prevent data loss.",
                stage="preflight_check",
                exit_code=EXIT_OUTPUT_DIR_EXISTS
            )
        if man_real.startswith(out_real + os.sep):
            raise ShardReductionError(
                f"Output directory '{output_dir}' is an ancestor of manifest path '{manifest_path}'. Refusing execution to prevent data loss.",
                stage="preflight_check",
                exit_code=EXIT_OUTPUT_DIR_EXISTS
            )

    if source_paths:
        for sp in source_paths:
            sp_real = os.path.realpath(os.path.abspath(sp))
            if out_real == sp_real or out_real == os.path.dirname(sp_real) or sp_real.startswith(out_real + os.sep):
                raise ShardReductionError(
                    f"Output directory '{output_dir}' aliases or contains source artifact path '{sp}'. Refusing execution to prevent data loss.",
                    stage="preflight_check",
                    exit_code=EXIT_OUTPUT_DIR_EXISTS
                )

    if os.path.exists(output_dir):
        existing_items = os.listdir(output_dir)
        if len(existing_items) > 0:
            msg = (
                f"Output directory '{output_dir}' already exists and is non-empty "
                f"({len(existing_items)} items present). Refusing to overwrite existing diagnostic run."
            )
            raise ShardReductionError(msg, stage="preflight_check", exit_code=EXIT_OUTPUT_DIR_EXISTS)
    os.makedirs(output_dir, exist_ok=True)



def extract_field_name(columns: List[str], candidate_names: List[str]) -> Optional[str]:
    """Finds matching column name from a list of case-insensitive candidate names."""
    col_map = {c.strip().lower(): c for c in columns}
    for cand in candidate_names:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def load_and_validate_candidate_shards(
    manifest_data: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, pd.DataFrame, pd.DataFrame]:
    """
    Losslessly loads, validates, and stages candidate records across all declared shards in memory.
    Enforces minimum measurement contract, strict TIC strings, and distinct observation IDs.
    """
    shards = manifest_data.get("shards", [])
    all_rows: List[Dict[str, Any]] = []
    reconciliation_rows: List[Dict[str, Any]] = []

    total_declared_candidate_shards = 0
    total_loaded_candidate_rows = 0

    for idx, shard in enumerate(shards):
        sector = shard.get("sector")
        sec_str = str(sector) if sector is not None else "N/A"
        source_run_id = shard.get("source_run_id", "")
        shard_status = shard.get("status", "COMPLETED")
        cand_art = shard.get("candidate_artifact")

        if cand_art is None:
            reconciliation_rows.append({
                "sector": sec_str,
                "source_run_id": source_run_id,
                "artifact_path": "NONE",
                "artifact_sha256": "NONE",
                "declared_status": shard_status if shard_status in ("FAILED", "MISSING", "SKIPPED") else "NO_CANDIDATE_ARTIFACT",
                "loaded_rows": 0,
                "reconciled_rows": 0
            })
            continue

        c_status = cand_art.get("status", shard_status)
        resolved_path = cand_art.get("resolved_path")
        exp_hash = cand_art.get("sha256", "UNKNOWN")

        if c_status in ("FAILED", "MISSING", "SKIPPED") or shard_status in ("FAILED", "MISSING", "SKIPPED"):
            reconciliation_rows.append({
                "sector": sec_str,
                "source_run_id": source_run_id,
                "artifact_path": cand_art.get("path", "N/A"),
                "artifact_sha256": exp_hash,
                "declared_status": c_status if c_status in ("FAILED", "MISSING", "SKIPPED") else shard_status,
                "loaded_rows": 0,
                "reconciled_rows": 0
            })
            continue

        if resolved_path is None or not os.path.exists(resolved_path):
            raise ShardReductionError(
                f"Candidate artifact path '{resolved_path}' missing during load",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        total_declared_candidate_shards += 1

        if os.path.getsize(resolved_path) == 0:
            raise ShardReductionError(
                f"Candidate artifact '{resolved_path}' is empty (0 bytes). A valid candidate CSV must define a schema.",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        # Validate raw CSV header before pandas (B2)
        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            raw_header = next(reader, None)

        if not raw_header or len(raw_header) == 0:
            raise ShardReductionError(
                f"Candidate artifact '{resolved_path}' is empty or has no header.",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        cleaned_headers = [h.strip() for h in raw_header]
        headers_lower = [h.lower() for h in cleaned_headers]
        if len(headers_lower) != len(set(headers_lower)):
            raise ShardReductionError(
                f"Duplicate or case-colliding column names in raw CSV header of '{resolved_path}': {raw_header}",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        try:
            df_shard = pd.read_csv(resolved_path, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise ShardReductionError(
                f"Failed to read CSV '{resolved_path}': {exc}",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        cols = list(df_shard.columns)
        tic_col = extract_field_name(cols, ["tic_id", "TIC ID", "tic"])
        p_col = extract_field_name(cols, ["period_days"])
        # Scientific assumption integrity: require explicit t0_btjd, duration_hours, depth_ppm (R3 & B2)
        t0_col = extract_field_name(cols, ["t0_btjd"])
        dur_col = extract_field_name(cols, ["duration_hours"])
        depth_col = extract_field_name(cols, ["depth_ppm"])

        if not tic_col:
            raise ShardReductionError(
                f"Required column 'tic_id' not found in candidate artifact '{resolved_path}'. Available: {cols}",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )
        if not p_col or not t0_col or not dur_col or not depth_col:
            raise ShardReductionError(
                f"Missing minimum explicit measurement contract (period_days/t0_btjd/duration_hours/depth_ppm) in '{resolved_path}'. Available: {cols}",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        csv_sector_col = extract_field_name(cols, ["sector"])
        csv_run_col = extract_field_name(cols, ["run_id"])

        shard_row_count = len(df_shard)
        if shard_row_count == 0:
            # Valid header-only CSV is a legitimate zero-candidate shard
            reconciliation_rows.append({
                "sector": sec_str,
                "source_run_id": source_run_id,
                "artifact_path": cand_art.get("path", resolved_path),
                "artifact_sha256": exp_hash,
                "declared_status": "EMPTY_NO_CANDIDATES" if c_status == "EMPTY_NO_CANDIDATES" else "HEADER_ONLY_ZERO_ROWS",
                "loaded_rows": 0,
                "reconciled_rows": 0
            })
            continue

        if c_status == "EMPTY_NO_CANDIDATES":
            raise ShardReductionError(
                f"Shard {idx} declared EMPTY_NO_CANDIDATES but artifact '{resolved_path}' contains {shard_row_count} rows.",
                stage="candidate_merging",
                exit_code=EXIT_VALIDATION_ERROR
            )

        for ordinal, (_, row) in enumerate(df_shard.iterrows()):
            raw_tic = str(row[tic_col]).strip()
            # Validate TIC ID string: must be unsigned integer digits without decimal or float coercion
            if not re.match(r"^\d+$", raw_tic):
                raise ShardReductionError(
                    f"Malformed TIC ID '{raw_tic}' in artifact '{resolved_path}' at row {ordinal}: "
                    f"must be a valid integer string without float coercion.",
                    stage="candidate_merging",
                    exit_code=EXIT_VALIDATION_ERROR
                )

            # Validate source identity agreement between declared manifest and CSV contents (B2)
            if csv_sector_col and sec_str != "N/A":
                row_sec = str(row[csv_sector_col]).strip()
                if row_sec != sec_str:
                    raise ShardReductionError(
                        f"Row {ordinal} in candidate artifact '{resolved_path}' has sector '{row_sec}' "
                        f"which contradicts declared manifest sector '{sec_str}'.",
                        stage="candidate_merging",
                        exit_code=EXIT_VALIDATION_ERROR
                    )

            if csv_run_col:
                row_run = str(row[csv_run_col]).strip()
                if row_run != source_run_id:
                    raise ShardReductionError(
                        f"Row {ordinal} in candidate artifact '{resolved_path}' has run_id '{row_run}' "
                        f"which contradicts declared manifest source_run_id '{source_run_id}'.",
                        stage="candidate_merging",
                        exit_code=EXIT_VALIDATION_ERROR
                    )

            # Validate numerical measurement contracts
            try:
                p_val = float(row[p_col])
                if not np.isfinite(p_val) or p_val <= 0:
                    raise ValueError(f"Non-positive period: {p_val}")
            except Exception as exc:
                raise ShardReductionError(
                    f"Invalid period '{row[p_col]}' at row {ordinal} in '{resolved_path}': {exc}",
                    stage="candidate_merging",
                    exit_code=EXIT_VALIDATION_ERROR
                )

            try:
                t0_val = float(row[t0_col])
                if not np.isfinite(t0_val):
                    raise ValueError(f"Non-finite epoch: {t0_val}")
            except Exception as exc:
                raise ShardReductionError(
                    f"Invalid epoch '{row[t0_col]}' at row {ordinal} in '{resolved_path}': {exc}",
                    stage="candidate_merging",
                    exit_code=EXIT_VALIDATION_ERROR
                )

            try:
                dur_val = float(row[dur_col])
                if not np.isfinite(dur_val) or dur_val <= 0:
                    raise ValueError(f"Non-positive duration: {dur_val}")
            except Exception as exc:
                raise ShardReductionError(
                    f"Invalid duration '{row[dur_col]}' at row {ordinal} in '{resolved_path}': {exc}",
                    stage="candidate_merging",
                    exit_code=EXIT_VALIDATION_ERROR
                )

            try:
                depth_val = float(row[depth_col])
                if not np.isfinite(depth_val):
                    raise ValueError(f"Non-finite depth: {depth_val}")
            except Exception as exc:
                raise ShardReductionError(
                    f"Invalid depth '{row[depth_col]}' at row {ordinal} in '{resolved_path}': {exc}",
                    stage="candidate_merging",
                    exit_code=EXIT_VALIDATION_ERROR
                )

            # Generate stable deterministic observation record ID independent of manifest ordering (B2)
            canon_path = os.path.realpath(resolved_path)
            id_seed = f"{source_run_id}:{sec_str}:{canon_path}:{exp_hash}:{ordinal}"
            id_digest = hashlib.sha256(id_seed.encode("utf-8")).hexdigest()[:12].upper()
            obs_id = f"obs_{id_digest}_s{sec_str}_r{ordinal:04d}"

            # Preserve raw input columns and recursively namespace collisions against reserved provenance fields (B2)
            row_dict = {}
            for k, v in row.items():
                if k in RESERVED_PROVENANCE_COLUMNS or k in CANONICAL_OUTPUT_COLUMNS:
                    escaped_k = f"orig_{k}"
                    while escaped_k in row or escaped_k in row_dict:
                        escaped_k = f"orig_{escaped_k}"
                    row_dict[escaped_k] = v
                else:
                    row_dict[k] = v

            # Populate standardized columns
            row_dict["tic_id"] = raw_tic
            row_dict["period_days"] = row[p_col]
            row_dict["t0_btjd"] = row[t0_col]
            row_dict["duration_hours"] = row[dur_col]
            row_dict["depth_ppm"] = row[depth_col]

            # Populate canonical reduction provenance
            row_dict["source_run_id"] = source_run_id
            row_dict["source_sector"] = sec_str
            row_dict["source_artifact_path"] = cand_art.get("path", resolved_path)
            row_dict["source_artifact_sha256"] = exp_hash
            row_dict["source_row_ordinal"] = ordinal
            row_dict["observation_record_id"] = obs_id

            all_rows.append(row_dict)

        total_loaded_candidate_rows += shard_row_count
        reconciliation_rows.append({
            "sector": sec_str,
            "source_run_id": source_run_id,
            "artifact_path": cand_art.get("path", resolved_path),
            "artifact_sha256": exp_hash,
            "declared_status": c_status,
            "loaded_rows": shard_row_count,
            "reconciled_rows": shard_row_count
        })

    # Build consolidated dataframe
    if all_rows:
        df_consolidated = pd.DataFrame(all_rows)
    else:
        df_consolidated = pd.DataFrame(columns=[
            "observation_record_id", "source_run_id", "source_sector",
            "source_artifact_path", "source_artifact_sha256", "source_row_ordinal",
            "tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm"
        ])

    df_recon = pd.DataFrame(reconciliation_rows)

    # Recompute and verify exact row accounting
    reconciled_total = int(df_recon["reconciled_rows"].sum())
    actual_consolidated_total = len(df_consolidated)

    if reconciled_total != actual_consolidated_total or actual_consolidated_total != total_loaded_candidate_rows:
        raise ReconciliationError(
            f"Row reconciliation failure: loaded {total_loaded_candidate_rows} rows, "
            f"reconciliation accounts for {reconciled_total}, "
            f"consolidated table has {actual_consolidated_total}."
        )

    # Verify uniqueness of observation_record_ids (R3)
    obs_ids = [r["observation_record_id"] for r in all_rows]
    if len(obs_ids) != len(set(obs_ids)):
        raise ReconciliationError(
            f"Duplicate observation_record_ids generated: {len(obs_ids)} rows but only {len(set(obs_ids))} unique IDs."
        )

    cand_summary = {
        "shards_with_candidate_artifacts": total_declared_candidate_shards,
        "total_candidate_observations": actual_consolidated_total,
        "unique_host_tics": int(df_consolidated["tic_id"].nunique()) if actual_consolidated_total > 0 else 0
    }

    return all_rows, reconciliation_rows, total_declared_candidate_shards, df_consolidated, df_recon


def load_and_validate_history_shards(
    manifest_data: Dict[str, Any]
) -> Tuple[Set[str], List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    """
    Losslessly loads and consolidates target scanning history across all declared shards in memory.
    Enforces strict numeric TIC format without silently skipping malformed entries (R2).
    """
    shards = manifest_data.get("shards", [])
    history_recon_rows: List[Dict[str, Any]] = []
    union_scanned_tics: Set[str] = set()
    total_raw_scanned_entries = 0

    for shard in shards:
        sector = shard.get("sector")
        sec_str = str(sector) if sector is not None else "N/A"
        source_run_id = shard.get("source_run_id", "")
        shard_status = shard.get("status", "COMPLETED")
        hist_art = shard.get("history_artifact")

        if hist_art is None:
            history_recon_rows.append({
                "sector": sec_str,
                "source_run_id": source_run_id,
                "artifact_path": "NONE",
                "artifact_sha256": "NONE",
                "declared_status": shard_status if shard_status in ("FAILED", "MISSING", "SKIPPED") else "NO_HISTORY_ARTIFACT",
                "scanned_count": 0
            })
            continue

        h_status = hist_art.get("status", shard_status)
        resolved_path = hist_art.get("resolved_path")
        exp_hash = hist_art.get("sha256", "UNKNOWN")

        if h_status in ("FAILED", "MISSING", "SKIPPED") or shard_status in ("FAILED", "MISSING", "SKIPPED") or resolved_path is None or not os.path.exists(resolved_path):
            history_recon_rows.append({
                "sector": sec_str,
                "source_run_id": source_run_id,
                "artifact_path": hist_art.get("path", "N/A"),
                "artifact_sha256": exp_hash,
                "declared_status": h_status if h_status in ("FAILED", "MISSING", "SKIPPED") else shard_status,
                "scanned_count": 0
            })
            continue

        shard_tics: Set[str] = set()
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s_line = line.strip()
                if not s_line:
                    continue
                # Strictly validate TIC format; do NOT silently drop corrupt/malformed strings (R2)
                if not re.match(r"^\d+$", s_line):
                    raise ShardReductionError(
                        f"Malformed TIC ID '{s_line}' in history artifact '{resolved_path}' at line {line_no}: "
                        f"must be numeric digits only without float coercion or alphabetic characters.",
                        stage="history_merging",
                        exit_code=EXIT_VALIDATION_ERROR
                    )
                shard_tics.add(s_line)
                total_raw_scanned_entries += 1

        union_scanned_tics.update(shard_tics)
        history_recon_rows.append({
            "sector": sec_str,
            "source_run_id": source_run_id,
            "artifact_path": hist_art.get("path", resolved_path),
            "artifact_sha256": exp_hash,
            "declared_status": h_status,
            "scanned_count": len(shard_tics)
        })

    df_hist_recon = pd.DataFrame(history_recon_rows)
    hist_summary = {
        "total_raw_entries": total_raw_scanned_entries,
        "total_unique_scanned_targets": len(union_scanned_tics)
    }

    return union_scanned_tics, history_recon_rows, hist_summary, df_hist_recon


def merge_candidate_shards(
    manifest_data: Dict[str, Any],
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Public entry point for candidate merging: validates and writes candidate tables.
    """
    _, _, _, df_consolidated, df_recon = load_and_validate_candidate_shards(manifest_data)
    os.makedirs(output_dir, exist_ok=True)

    out_consolidated_csv = os.path.join(output_dir, "candidates_consolidated.csv")
    out_legacy_csv = os.path.join(output_dir, "candidates_report.csv")
    out_recon_csv = os.path.join(output_dir, "candidate_reconciliation.csv")

    df_consolidated.to_csv(out_consolidated_csv, index=False)
    df_consolidated.to_csv(out_legacy_csv, index=False)
    df_recon.to_csv(out_recon_csv, index=False)

    cand_summary = {
        "total_candidate_observations": len(df_consolidated),
        "unique_host_tics": int(df_consolidated["tic_id"].nunique()) if len(df_consolidated) > 0 else 0,
        "consolidated_csv": out_consolidated_csv,
        "reconciliation_csv": out_recon_csv
    }
    return df_consolidated, df_recon, cand_summary


def merge_history_shards(
    manifest_data: Dict[str, Any],
    output_dir: str
) -> Tuple[Set[str], pd.DataFrame, Dict[str, Any]]:
    """
    Public entry point for history merging: validates and writes history files.
    """
    union_scanned_tics, _, hist_summary, df_hist_recon = load_and_validate_history_shards(manifest_data)
    os.makedirs(output_dir, exist_ok=True)

    out_history_file = os.path.join(output_dir, "scanned_tics.txt")
    out_history_recon_csv = os.path.join(output_dir, "history_reconciliation.csv")

    with open(out_history_file, "w", encoding="utf-8") as f:
        for tic in sorted(union_scanned_tics, key=lambda x: int(x)):
            f.write(f"{tic}\n")

    df_hist_recon.to_csv(out_history_recon_csv, index=False)
    hist_summary["history_file"] = out_history_file
    hist_summary["reconciliation_csv"] = out_history_recon_csv

    return union_scanned_tics, df_hist_recon, hist_summary


def save_stage_log(
    stage_name: str,
    stage_log: str,
    cmd: List[str],
    proc: subprocess.CompletedProcess,
    stages_executed: List[Dict[str, Any]]
) -> None:
    """
    Saves stage diagnostic log to disk.
    If saving fails, logs cause to sys.stderr and raises StageExecutionError (C3).
    """
    try:
        with open(stage_log, "w", encoding="utf-8") as f:
            f.write(f"COMMAND: {' '.join(cmd)}\nRETURNCODE: {proc.returncode}\n")
            f.write(f"--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n")
    except OSError as exc:
        err_msg = (
            f"Failed to write required diagnostic log '{stage_log}' for stage '{stage_name}': {exc}"
        )
        if proc.returncode != 0:
            err_msg = f"Stage command '{stage_name}' failed with returncode {proc.returncode} AND {err_msg}"
        sys.stderr.write(f"ERROR: {err_msg}\n")
        sys.stderr.flush()
        stages_executed.append({
            "stage": stage_name,
            "status": "FAILED",
            "returncode": proc.returncode if proc.returncode != 0 else EXIT_EXECUTION_FAILURE,
            "log_file": None,
            "log_write_error": str(exc),
            "stderr_tail": err_msg
        })
        raise StageExecutionError(
            stage=stage_name,
            returncode=proc.returncode if proc.returncode != 0 else EXIT_EXECUTION_FAILURE,
            stderr_text=err_msg
        ) from exc


def orchestrate_downstream_stages(
    df_consolidated: pd.DataFrame,
    output_dir: str,
    run_pleiades: bool = False,
    run_antares: bool = False,
    run_triceratops: bool = False,
    python_exe: str = sys.executable,
    stages_executed: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Safely orchestrates downstream vetting and validation stages:
    Pleiades (Tier-2) -> Antares (Tier-3) -> TRICERATOPS (Tier-4).
    Enforces strict dependency chains, exact target reconciliation, and incremental logging.
    """
    if stages_executed is None:
        stages_executed = []

    if not (run_pleiades or run_antares or run_triceratops):
        return stages_executed

    # Check for empty catalog
    if len(df_consolidated) == 0:
        print("[+] Consolidated candidates catalog is empty (0 rows).")
        print("[+] Legitimately skipping downstream vetting stages without consuming stale artifacts.")
        stages_executed.append({
            "stage": "downstream_orchestration",
            "status": "SKIPPED_EMPTY_CANDIDATES",
            "reason": "No candidates present in consolidated catalog."
        })
        return stages_executed

    # Check for duplicate TICs (multi-signal / multiple observations on same host)
    if "tic_id" in df_consolidated.columns:
        tic_counts = df_consolidated["tic_id"].value_counts()
        dup_tics = tic_counts[tic_counts > 1].index.tolist()
        if len(dup_tics) > 0:
            msg = (
                f"BLOCKED: Consolidated candidate catalog contains multiple observations for the same TIC ID(s): {dup_tics}. "
                f"Downstream consumers (Pleiades, Antares, TRICERATOPS) currently deduplicate or join on TIC ID alone (FLOW-01), "
                f"which would silently drop distinct candidate signals on the same host. "
                f"Downstream execution is blocked until multi-signal routing is migrated. "
                f"All candidate rows remain safely preserved in candidates_consolidated.csv."
            )
            raise BlockedDownstreamIncompatibilityError(msg)

    cand_csv = os.path.join(output_dir, "candidates_report.csv")
    vet_dir = os.path.join(output_dir, "deep_vetting")

    # 1. Pleiades Tier-2 Deep Vetting
    if run_pleiades:
        print(f"\n[+] Launching Pleiades Tier-2 Deep Vetting on '{cand_csv}' -> '{vet_dir}'...")
        os.makedirs(vet_dir, exist_ok=True)
        stage_log = os.path.join(output_dir, "stage_pleiades.log")
        cmd = [python_exe, "pleiades.py", "--input-csv", cand_csv, "--output-dir", vet_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        save_stage_log("pleiades", stage_log, cmd, proc, stages_executed)

        if proc.returncode != 0:
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": proc.returncode,
                "log_file": stage_log,
                "stderr_tail": proc.stderr[-500:] if proc.stderr else ""
            })
            raise StageExecutionError("pleiades", proc.returncode, proc.stderr)

        expected_dossier_csv = os.path.join(vet_dir, "scientific_dossier_summary.csv")
        if not os.path.exists(expected_dossier_csv):
            err_msg = f"Expected artifact '{expected_dossier_csv}' not generated."
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": proc.returncode,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("pleiades", proc.returncode, err_msg)

        # Per-input operational status / target reconciliation (B3 & C1)
        expected_tics = [str(x).strip() for x in df_consolidated["tic_id"]] if "tic_id" in df_consolidated.columns else []
        try:
            df_dossier = pd.read_csv(expected_dossier_csv)
        except Exception as exc:
            err_msg = f"Failed to read Pleiades dossier summary '{expected_dossier_csv}': {exc}"
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("pleiades", EXIT_EXECUTION_FAILURE, err_msg)

        dossier_tics = [str(x).strip() for x in df_dossier["tic_id"]] if "tic_id" in df_dossier.columns else []
        
        # Check multiset agreement
        missing_dossier_tics = set(expected_tics) - set(dossier_tics)
        if missing_dossier_tics:
            err_msg = (
                f"Pleiades summary dossier failed to account for input candidate target(s): {sorted(missing_dossier_tics)}. "
                f"Every candidate target must have an operational status/reconciliation record in scientific_dossier_summary.csv."
            )
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("pleiades", EXIT_EXECUTION_FAILURE, err_msg)

        extra_dossier_tics = set(dossier_tics) - set(expected_tics)
        if extra_dossier_tics:
            err_msg = f"Pleiades summary dossier contains unexpected/unrequested target(s): {sorted(extra_dossier_tics)}."
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("pleiades", EXIT_EXECUTION_FAILURE, err_msg)

        if len(dossier_tics) != len(set(dossier_tics)) and len(expected_tics) == len(set(expected_tics)):
            err_msg = "Pleiades summary dossier contains duplicate records for target(s)."
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("pleiades", EXIT_EXECUTION_FAILURE, err_msg)

        # Check for target operational errors (C1 & D2)
        failed_targets = []
        for _, r in df_dossier.iterrows():
            t_id = str(r.get("tic_id", "")).strip()
            op_status = str(r.get("operational_status", "")).strip().upper() if "operational_status" in df_dossier.columns and pd.notna(r.get("operational_status")) else None
            disp = str(r.get("disposition", "")).strip().upper() if "disposition" in df_dossier.columns and pd.notna(r.get("disposition")) else ""
            raw_err = str(r.get("error_message", "")).strip() if "error_message" in df_dossier.columns and pd.notna(r.get("error_message")) else ""
            has_error_message = bool(raw_err and raw_err.upper() not in ("", "NAN", "NONE"))

            is_failed = False
            failure_reason = ""

            if has_error_message:
                is_failed = True
                failure_reason = f"error_message: {raw_err}"
            elif not op_status:
                is_failed = True
                failure_reason = "missing operational_status"
            elif op_status not in ("SUCCESS", "COMPLETED", "PASS", "PASSED"):
                is_failed = True
                failure_reason = f"operational_status: {op_status}"
            elif disp in ("ERROR", "FAILED", "NO_LIGHTCURVE"):
                is_failed = True
                failure_reason = f"disposition: {disp}"

            if is_failed:
                failed_targets.append((t_id, op_status or "MISSING_STATUS", failure_reason))

        if failed_targets:
            err_msg = (
                f"Pleiades deep vetting failed for target(s): {failed_targets}. "
                f"Target operational errors cannot be converted to physical false positives or zero-survivor science. "
                f"Downstream chain continuation is stopped."
            )
            stages_executed.append({
                "stage": "pleiades",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "failed_targets": [ft[0] for ft in failed_targets],
                "stderr_tail": err_msg
            })
            raise StageExecutionError("pleiades", EXIT_EXECUTION_FAILURE, err_msg)

        stages_executed.append({
            "stage": "pleiades",
            "status": "COMPLETED",
            "output_csv": expected_dossier_csv,
            "returncode": proc.returncode,
            "log_file": stage_log,
            "reconciled_targets": len(dossier_tics),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else ""
        })

    # 2. Antares Tier-3 Packager & Archival Follow-Up
    if run_antares:
        print(f"\n[+] Launching Antares Tier-3 Packager on '{output_dir}'...")
        stage_log = os.path.join(output_dir, "stage_antares.log")
        cmd = [python_exe, "antares_packager.py", "--results-dir", output_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        save_stage_log("antares", stage_log, cmd, proc, stages_executed)

        if proc.returncode != 0:
            stages_executed.append({
                "stage": "antares",
                "status": "FAILED",
                "returncode": proc.returncode,
                "log_file": stage_log,
                "stderr_tail": proc.stderr[-500:] if proc.stderr else ""
            })
            raise StageExecutionError("antares", proc.returncode, proc.stderr)

        expected_exofop_csv = os.path.join(output_dir, "antares_exofop_ctoi_upload.csv")
        expected_ranking_csv = os.path.join(output_dir, "antares_prioritized_targets_ranking.csv")
        if not os.path.exists(expected_exofop_csv):
            err_msg = f"Expected artifact '{expected_exofop_csv}' not generated."
            stages_executed.append({
                "stage": "antares",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("antares", EXIT_EXECUTION_FAILURE, err_msg)

        stages_executed.append({
            "stage": "antares",
            "status": "COMPLETED",
            "output_csv": expected_exofop_csv,
            "ranking_csv": expected_ranking_csv if os.path.exists(expected_ranking_csv) else None,
            "returncode": proc.returncode,
            "log_file": stage_log,
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else ""
        })

    # 3. TRICERATOPS Tier-4 Bayesian Statistical Validation (Strict Opt-In)
    if run_triceratops:
        # A requested Antares stage must never consume stale CTOI outputs.
        producer_prefix = "antares_" if run_antares else ""
        top3_csv = os.path.join(output_dir, producer_prefix + "exofop_ctoi_upload_top3.csv")
        if not os.path.exists(top3_csv):
            ranking_csv = os.path.join(output_dir, producer_prefix + "prioritized_targets_ranking.csv")
            if os.path.exists(ranking_csv):
                top3_csv = ranking_csv
            else:
                # If neither exists: check if consolidated candidates was empty
                if len(df_consolidated) == 0:
                    print("[+] Candidate catalog has 0 candidates. Skipping TRICERATOPS validation.")
                    stages_executed.append({
                        "stage": "triceratops",
                        "status": "SKIPPED_ZERO_TARGETS",
                        "reason": "Consolidated catalog has zero candidates."
                    })
                    return stages_executed

                # Non-empty candidates, but missing Tier-3 prerequisite artifacts -> Stage error (B3)
                err_msg = (
                    "Missing Tier-3 dependency: neither 'exofop_ctoi_upload_top3.csv' nor 'prioritized_targets_ranking.csv' "
                    "found for TRICERATOPS validation. Prior stage did not produce required target table."
                )
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

        # Check delimiter and read top3 table
        with open(top3_csv, "r", encoding="utf-8") as f:
            first_line = f.readline()
        top_sep = "|" if "|" in first_line else ","
        try:
            df_top = pd.read_csv(top3_csv, sep=top_sep)
        except Exception as exc:
            err_msg = f"Failed to read top candidates table '{top3_csv}': {exc}"
            stages_executed.append({
                "stage": "triceratops",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

        if len(df_top) == 0:
            print("[+] Target table for TRICERATOPS is empty (0 rows). Skipping Bayesian validation.")
            stages_executed.append({
                "stage": "triceratops",
                "status": "SKIPPED_ZERO_TARGETS",
                "reason": "Top candidates table is empty (0 rows)."
            })
            return stages_executed

        # Producer schema normalization: handle TIC ID and tic_id (C2)
        tic_cols = [c for c in df_top.columns if c.strip().lower() in ("tic_id", "tic id", "tic", "tid")]
        if not tic_cols:
            tic_cols = [c for c in df_top.columns if "TIC" in c.upper()]
        if not tic_cols:
            err_msg = f"Top candidates table '{top3_csv}' contains {len(df_top)} rows but no recognizable TIC ID column (found: {list(df_top.columns)})."
            stages_executed.append({
                "stage": "triceratops",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

        tic_col_name = tic_cols[0]

        # Ephemeris column validation before expensive invocation (C2)
        p_cols = [c for c in df_top.columns if "PERIOD" in c.upper()]
        epoch_cols = [c for c in df_top.columns if "EPOCH" in c.upper() or "T0" in c.upper()]
        depth_cols = [c for c in ("Depth", "measured_box_depth_ppm", "depth_ppm") if c in df_top.columns]
        dur_cols = [c for c in df_top.columns if "DUR" in c.upper()]
        if not (p_cols and epoch_cols and depth_cols and dur_cols):
            missing_cols = []
            if not p_cols: missing_cols.append("Period")
            if not epoch_cols: missing_cols.append("Epoch/T0")
            if not depth_cols: missing_cols.append("Depth")
            if not dur_cols: missing_cols.append("Duration")
            err_msg = (
                f"Top candidates table '{top3_csv}' is missing required ephemeris columns for TRICERATOPS: {missing_cols}. "
                f"Available columns: {list(df_top.columns)}."
            )
            stages_executed.append({
                "stage": "triceratops",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

        expected_top_tics = [str(x).strip() for x in df_top[tic_col_name].dropna().unique() if str(x).strip()]
        if len(expected_top_tics) == 0:
            stages_executed.append({
                "stage": "triceratops",
                "status": "SKIPPED_ZERO_TARGETS",
                "reason": "Top candidates table has no non-null TIC IDs."
            })
            return stages_executed

        print(f"\n[+] Launching Tier-4 TRICERATOPS Bayesian Validation on '{top3_csv}'...")
        stage_log = os.path.join(output_dir, "stage_triceratops.log")
        cmd = [python_exe, "validate_triceratops.py", "--auto", "--top-csv", top3_csv, "--results-dir", output_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        save_stage_log("triceratops", stage_log, cmd, proc, stages_executed)

        if proc.returncode != 0:
            stages_executed.append({
                "stage": "triceratops",
                "status": "FAILED",
                "returncode": proc.returncode,
                "log_file": stage_log,
                "stderr_tail": proc.stderr[-500:] if proc.stderr else ""
            })
            raise StageExecutionError("triceratops", proc.returncode, proc.stderr)

        # Validate that actual per-target probability artifacts were created and satisfy contract (B3 & C2)
        tric_dir = os.path.join(output_dir, "triceratops")
        if not os.path.exists(tric_dir):
            err_msg = f"TRICERATOPS output directory '{tric_dir}' was not created."
            stages_executed.append({
                "stage": "triceratops",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

        prob_files = []
        target_hashes = {}
        for tic in expected_top_tics:
            target_prob_file = os.path.join(tric_dir, f"triceratops_probs_TIC_{tic}.csv")
            if not os.path.exists(target_prob_file):
                err_msg = f"Missing TRICERATOPS probability table for expected target TIC {tic} ('{target_prob_file}')."
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)
            if os.path.getsize(target_prob_file) == 0:
                err_msg = f"TRICERATOPS probability table for TIC {tic} is empty (0 bytes)."
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            # Structural artifact contract validation: parse CSV and verify columns and finite numbers (C2)
            try:
                df_prob = pd.read_csv(target_prob_file)
            except Exception as exc:
                err_msg = f"TRICERATOPS probability table for TIC {tic} ('{target_prob_file}') is not a parseable CSV: {exc}"
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            if len(df_prob) == 0:
                err_msg = f"TRICERATOPS probability table for TIC {tic} contains 0 rows."
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            # Scenario column requirement (D1)
            scen_cols = [c for c in df_prob.columns if c.strip().lower() in ("scenario", "scenarios")]
            if not scen_cols:
                err_msg = (
                    f"TRICERATOPS probability table for TIC {tic} ('{target_prob_file}') is missing required "
                    f"'scenario' column. Found columns: {list(df_prob.columns)}."
                )
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            scen_col = scen_cols[0]
            if df_prob[scen_col].dropna().astype(str).str.strip().eq("").all():
                err_msg = f"TRICERATOPS probability table for TIC {tic} contains only empty scenario values."
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            # Fractional probability column requirement: 'prob' or 'probability' (D1)
            prob_cols = [c for c in df_prob.columns if c.strip().lower() in ("prob", "probability")]
            if not prob_cols:
                err_msg = (
                    f"TRICERATOPS probability table for TIC {tic} ('{target_prob_file}') is missing required "
                    f"fractional probability column ('prob' or 'probability'). Found columns: {list(df_prob.columns)}."
                )
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            prob_col = prob_cols[0]
            numeric_vals = pd.to_numeric(df_prob[prob_col], errors="coerce")
            if numeric_vals.isna().any() or (~np.isfinite(numeric_vals)).any():
                err_msg = f"TRICERATOPS probability table for TIC {tic} contains non-numeric or non-finite values in column '{prob_col}'."
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            # Fractional bounds [0.0, 1.0] (D1)
            if (numeric_vals < 0.0).any() or (numeric_vals > 1.0).any():
                err_msg = f"TRICERATOPS probability table for TIC {tic} contains values outside [0, 1] in column '{prob_col}'."
                stages_executed.append({
                    "stage": "triceratops",
                    "status": "FAILED",
                    "returncode": EXIT_EXECUTION_FAILURE,
                    "log_file": stage_log,
                    "stderr_tail": err_msg
                })
                raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            # Target ID validation: target TIC must be present (nearby stars may also be present) (D1)
            id_cols = [c for c in df_prob.columns if c.strip().lower() in ("id", "tic_id", "tic")]
            if id_cols:
                id_col = id_cols[0]
                unique_ids = [str(x).strip() for x in df_prob[id_col].dropna().unique()]
                if unique_ids and str(tic) not in unique_ids:
                    err_msg = (
                        f"TRICERATOPS probability table '{target_prob_file}' contains ID(s) {unique_ids} "
                        f"which do not include expected target TIC {tic}."
                    )
                    stages_executed.append({
                        "stage": "triceratops",
                        "status": "FAILED",
                        "returncode": EXIT_EXECUTION_FAILURE,
                        "log_file": stage_log,
                        "stderr_tail": err_msg
                    })
                    raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

            prob_files.append(f"triceratops_probs_TIC_{tic}.csv")
            target_hashes[tic] = compute_file_sha256(target_prob_file)

        # Check for unrequested extra probability files
        all_prob_files = [f for f in os.listdir(tric_dir) if f.startswith("triceratops_probs_TIC_") and f.endswith(".csv")]
        expected_prob_filenames = {f"triceratops_probs_TIC_{t}.csv" for t in expected_top_tics}
        extra_prob_files = set(all_prob_files) - expected_prob_filenames
        if extra_prob_files:
            err_msg = f"TRICERATOPS directory contains unexpected probability files for unrequested targets: {sorted(extra_prob_files)}."
            stages_executed.append({
                "stage": "triceratops",
                "status": "FAILED",
                "returncode": EXIT_EXECUTION_FAILURE,
                "log_file": stage_log,
                "stderr_tail": err_msg
            })
            raise StageExecutionError("triceratops", EXIT_EXECUTION_FAILURE, err_msg)

        stages_executed.append({
            "stage": "triceratops",
            "status": "COMPLETED",
            "returncode": proc.returncode,
            "log_file": stage_log,
            "artifacts_dir": tric_dir,
            "probability_tables": prob_files,
            "target_hashes": target_hashes,
            "validated_targets": len(expected_top_tics),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else ""
        })

    return stages_executed


def run_merge_shards(
    manifest_path: str,
    output_dir: str,
    run_id: Optional[str] = None,
    run_pleiades: bool = False,
    run_antares: bool = False,
    run_triceratops: bool = False,
    allow_partial: bool = False,
    _inject_write_failure: bool = False
) -> int:
    """
    Main entry point for explicit-input shard reduction and safe orchestration.
    Wrapped in run-level guarded error boundaries.
    """
    start_time = datetime.now(timezone.utc).isoformat()
    current_stage = "preflight_check"
    stages_executed: List[Dict[str, Any]] = []
    acquired_fresh_dir = False

    try:
        # Preflight check: refuse occupied output or manifest aliasing before any writes (B1)
        check_and_prepare_output_dir(output_dir, manifest_path=manifest_path)
        acquired_fresh_dir = True

        current_stage = "manifest_validation"
        manifest_data, manifest_dir, effective_survey_mode, is_run_partial, partial_caveats = validate_and_load_manifest(
            manifest_path, allow_partial=allow_partial
        )

        # Check source artifact aliasing against output_dir (B1)
        out_real = os.path.realpath(os.path.abspath(output_dir))
        for s in manifest_data.get("shards", []):
            for art_key in ("candidate_artifact", "history_artifact"):
                art = s.get(art_key)
                if isinstance(art, dict) and "resolved_path" in art:
                    art_real = os.path.realpath(os.path.abspath(art["resolved_path"]))
                    if out_real == art_real or out_real == os.path.dirname(art_real) or art_real.startswith(out_real + os.sep):
                        raise ShardReductionError(
                            f"Output directory '{output_dir}' aliases or contains source artifact '{art_real}'. Refusing execution to prevent data loss.",
                            stage="preflight_check",
                            exit_code=EXIT_OUTPUT_DIR_EXISTS
                        )

        manifest_run_id = str(manifest_data.get("run_id", "")).strip()
        if run_id:
            # Enforce consistency between CLI argument and manifest content (R1)
            if str(run_id).strip() != manifest_run_id:
                raise ShardReductionError(
                    f"CLI run_id '{run_id}' contradicts manifest run_id '{manifest_run_id}'. Explicit matching required.",
                    stage="manifest_validation",
                    exit_code=EXIT_VALIDATION_ERROR
                )
        else:
            run_id = manifest_run_id

        # Phase 1: Load and validate all candidate shards in memory (R2)
        current_stage = "candidate_validation"
        all_cand_rows, cand_recon_rows, total_cand_shards, df_consolidated, df_recon = load_and_validate_candidate_shards(manifest_data)

        # Phase 2: Load and validate all history shards in memory (R2)
        current_stage = "history_validation"
        union_scanned_tics, hist_recon_rows, hist_summary, df_hist_recon = load_and_validate_history_shards(manifest_data)

        # Phase 3: ALL validations passed; now publish reduction outputs to permanent storage (R2)
        current_stage = "publishing_outputs"
        out_consolidated_csv = os.path.join(output_dir, "candidates_consolidated.csv")
        out_legacy_csv = os.path.join(output_dir, "candidates_report.csv")
        out_recon_csv = os.path.join(output_dir, "candidate_reconciliation.csv")
        out_history_file = os.path.join(output_dir, "scanned_tics.txt")
        out_history_recon_csv = os.path.join(output_dir, "history_reconciliation.csv")

        df_consolidated.to_csv(out_consolidated_csv, index=False)
        df_consolidated.to_csv(out_legacy_csv, index=False)
        df_recon.to_csv(out_recon_csv, index=False)

        with open(out_history_file, "w", encoding="utf-8") as f:
            for tic in sorted(union_scanned_tics, key=lambda x: int(x)):
                f.write(f"{tic}\n")
        df_hist_recon.to_csv(out_history_recon_csv, index=False)

        # Phase 4: Downstream Orchestration (Opt-In)
        current_stage = "downstream_orchestration"
        stages_executed = orchestrate_downstream_stages(
            df_consolidated=df_consolidated,
            output_dir=output_dir,
            run_pleiades=run_pleiades,
            run_antares=run_antares,
            run_triceratops=run_triceratops,
            stages_executed=stages_executed
        )

        current_stage = "finalizing_manifest"
        if _inject_write_failure:
            raise OSError("Injected deliberate disk/write failure during finalization")

        # Copy input manifest into output directory
        shutil.copy2(manifest_path, os.path.join(output_dir, "input_manifest_copy.json"))

        # Inventory generated artifacts
        output_artifacts: Dict[str, Dict[str, Any]] = {}
        for root, _, files in os.walk(output_dir):
            for fname in sorted(files):
                if fname in ("manifest.json", "run_status.json"):
                    continue
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, output_dir).replace("\\", "/")
                output_artifacts[rel_path] = {
                    "size_bytes": os.path.getsize(fpath),
                    "sha256": compute_file_sha256(fpath)
                }

        out_manifest = {
            "run_id": run_id,
            "manifest_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "survey_mode": effective_survey_mode,
            "is_partial": is_run_partial,
            "partial_caveats": partial_caveats,
            "candidate_observations": len(df_consolidated),
            "unique_tics": int(df_consolidated["tic_id"].nunique()) if len(df_consolidated) > 0 else 0,
            "scanned_targets": len(union_scanned_tics),
            "stages_executed": [s["stage"] for s in stages_executed],
            "artifacts": output_artifacts
        }

        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(out_manifest, f, indent=2)

        # Phase 5: Write final run status (R4)
        current_stage = "finalizing_status"
        final_status = "SUCCESS" if not is_run_partial else "SUCCESS_PARTIAL_SURVEY"
        run_status_dict = {
            "run_id": run_id,
            "status": final_status,
            "survey_mode": effective_survey_mode,
            "is_partial": is_run_partial,
            "partial_caveats": partial_caveats,
            "candidate_observations": len(df_consolidated),
            "unique_tics": int(df_consolidated["tic_id"].nunique()) if len(df_consolidated) > 0 else 0,
            "scanned_targets": len(union_scanned_tics),
            "stages_executed": stages_executed,
            "start_time": start_time,
            "completion_time": datetime.now(timezone.utc).isoformat()
        }

        # Gate success on verified status write (R4)
        status_saved = write_run_status(output_dir, run_status_dict)
        if not status_saved:
            manifest_p = os.path.join(output_dir, "manifest.json")
            if os.path.exists(manifest_p):
                try:
                    os.remove(manifest_p)
                except OSError:
                    pass
            raise ShardReductionError(
                f"Failed to write required run_status.json to '{output_dir}'. Success manifest unpublished.",
                stage="finalizing_status",
                exit_code=EXIT_EXECUTION_FAILURE
            )

        print("\n" + "=" * 70)
        print("ALDEBARAN v9.1: Shard Reduction & Orchestration Completed Successfully!")
        print(f"Run ID:                    {run_id}")
        print(f"Survey Mode:               {effective_survey_mode}")
        print(f"Is Partial:                {is_run_partial}")
        print(f"Total Candidate Rows:      {len(df_consolidated)}")
        print(f"Unique Candidate Hosts:    {int(df_consolidated['tic_id'].nunique()) if len(df_consolidated) > 0 else 0}")
        print(f"Unique Scanned Targets:    {len(union_scanned_tics)}")
        print(f"Downstream Stages Run:     {[s['stage'] for s in stages_executed]}")
        print(f"Output Directory:          {output_dir}")
        print("=" * 70)

        return EXIT_SUCCESS

    except Exception as exc:
        exit_code = getattr(exc, "exit_code", EXIT_EXECUTION_FAILURE)
        stage_name = getattr(exc, "stage", current_stage)

        sys.stderr.write(f"\nERROR: Shard reduction failed during stage '{stage_name}': {exc}\n")
        sys.stderr.flush()

        # Suppress any partial success manifest ONLY in freshly acquired directory (B1)
        if acquired_fresh_dir and os.path.exists(output_dir):
            partial_manifest = os.path.join(output_dir, "manifest.json")
            if os.path.exists(partial_manifest):
                try:
                    os.remove(partial_manifest)
                except OSError:
                    pass

            failure_status = {
                "run_id": run_id or "unknown_run",
                "status": "FAILED",
                "failed_stage": stage_name,
                "error_message": str(exc),
                "exit_code": exit_code,
                "stages_executed": stages_executed,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_run_status(output_dir, failure_status)

        return exit_code



def main():
    parser = argparse.ArgumentParser(
        description="Aldebaran v9.1: Shard Reduction, Lossless Consolidation & Safe Downstream Orchestration",
        usage="python merge_shards.py --manifest <path> --output-dir <path> [--run-id <id>] [downstream options]"
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to input manifest JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Fresh destination directory for consolidated run")
    parser.add_argument("--run-id", type=str, default=None, help="Identifier for this reduction run")
    parser.add_argument("--run-pleiades", action="store_true", help="Opt-in: execute Pleiades Tier-2 deep vetting")
    parser.add_argument("--run-antares", action="store_true", help="Opt-in: execute Antares Tier-3 packaging")
    parser.add_argument("--run-triceratops", action="store_true", help="Opt-in: execute TRICERATOPS Tier-4 validation")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly allow reduction over partial/missing shards")

    # Fast validation for no-arg legacy invocation
    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(EXIT_USAGE_OR_ARG_ERROR)

    args = parser.parse_args()

    exit_code = run_merge_shards(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        run_pleiades=args.run_pleiades,
        run_antares=args.run_antares,
        run_triceratops=args.run_triceratops,
        allow_partial=args.allow_partial
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
