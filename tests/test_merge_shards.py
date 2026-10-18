#!/usr/bin/env python3
"""
tests/test_merge_shards.py
==========================
Comprehensive offline regression and falsification test suite for ALD-010:
Explicit-input, lossless shard reduction and safe downstream orchestration.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd

from merge_shards import (
    EXIT_BLOCKED_DOWNSTREAM,
    EXIT_EXECUTION_FAILURE,
    EXIT_OUTPUT_DIR_EXISTS,
    EXIT_STAGE_FAILURE,
    EXIT_SUCCESS,
    EXIT_USAGE_OR_ARG_ERROR,
    EXIT_VALIDATION_ERROR,
    BlockedDownstreamIncompatibilityError,
    ReconciliationError,
    ShardReductionError,
    StageExecutionError,
    compute_file_sha256,
    load_and_validate_candidate_shards,
    load_and_validate_history_shards,
    merge_candidate_shards,
    merge_history_shards,
    orchestrate_downstream_stages,
    run_merge_shards,
    validate_and_load_manifest,
    write_run_status,
)


def make_dummy_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    """Helper to write CSV and return its SHA256 hash."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return compute_file_sha256(path)


def make_dummy_txt(path: str, lines: List[str]) -> str:
    """Helper to write text file and return its SHA256 hash."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return compute_file_sha256(path)


class TestManifestValidationAndIntegrity(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_manifest_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_valid_manifest_loads_and_resolves_relative_paths(self):
        """Manifest with relative paths resolves relative to manifest location."""
        shard1_dir = os.path.join(self.test_dir, "shards", "s100")
        cand1_path = os.path.join(shard1_dir, "candidates_report.csv")
        hash1 = make_dummy_csv(cand1_path, [
            {"tic_id": "301152169", "period_days": "3.103757", "t0_btjd": "4074.63460", "duration_hours": "2.59", "depth_ppm": "1200.0"}
        ])

        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "manifest_version": "1.0",
            "run_id": "run_test_valid",
            "shards": [
                {
                    "sector": 100,
                    "source_run_id": "run_s100",
                    "candidate_artifact": {
                        "path": "shards/s100/candidates_report.csv",
                        "sha256": hash1,
                        "status": "COMPLETED"
                    }
                }
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        loaded_data, manifest_dir, survey_mode, is_partial, caveats = validate_and_load_manifest(manifest_file)
        self.assertEqual(manifest_dir, os.path.abspath(self.test_dir))
        self.assertEqual(survey_mode, "complete")
        self.assertFalse(is_partial)
        resolved_path = loaded_data["shards"][0]["candidate_artifact"]["resolved_path"]
        self.assertEqual(resolved_path, os.path.abspath(cand1_path))

    def test_sha256_mismatch_fails_fast(self):
        """Negative control NC-1: SHA256 mutation in candidate artifact is caught."""
        cand_path = os.path.join(self.test_dir, "candidates_report.csv")
        make_dummy_csv(cand_path, [
            {"tic_id": "301152169", "period_days": "3.103757", "t0_btjd": "4074.63460", "duration_hours": "2.59", "depth_ppm": "1200.0"}
        ])
        mutated_hash = "E" * 64

        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_bad_hash",
            "shards": [
                {
                    "sector": 100,
                    "source_run_id": "run_s100",
                    "candidate_artifact": {
                        "path": "candidates_report.csv",
                        "sha256": mutated_hash,
                        "status": "COMPLETED"
                    }
                }
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("SHA256 mismatch", str(cm.exception))

    def test_missing_candidate_artifact_fails_fast(self):
        """Negative control NC-2: Missing declared candidate artifact fails fast."""
        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_missing_file",
            "shards": [
                {
                    "sector": 100,
                    "source_run_id": "run_s100",
                    "candidate_artifact": {
                        "path": "nonexistent_candidates.csv",
                        "sha256": "A" * 64,
                        "status": "COMPLETED"
                    }
                }
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("Declared candidate artifact not found", str(cm.exception))

    def test_duplicate_candidate_artifact_rejected(self):
        """Negative control NC-3: Repeated candidate artifact path across shards is rejected."""
        cand_path = os.path.join(self.test_dir, "cand.csv")
        actual_hash = make_dummy_csv(cand_path, [
            {"tic_id": "111", "period_days": "1.0", "t0_btjd": "100.0", "duration_hours": "1.0", "depth_ppm": "500.0"}
        ])

        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_dup_artifact",
            "shards": [
                {"sector": 1, "source_run_id": "run1", "candidate_artifact": {"path": "cand.csv", "sha256": actual_hash}},
                {"sector": 2, "source_run_id": "run2", "candidate_artifact": {"path": "cand.csv", "sha256": actual_hash}}
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("Duplicate candidate artifact path", str(cm.exception))

    def test_conflicting_sector_runs_rejected(self):
        """Negative control NC-4: Multiple shards declaring the same sector with different run IDs rejected."""
        c1 = os.path.join(self.test_dir, "c1.csv")
        c2 = os.path.join(self.test_dir, "c2.csv")
        h1 = make_dummy_csv(c1, [{"tic_id": "1", "period_days": "1.0", "t0_btjd": "10.0", "duration_hours": "1.0", "depth_ppm": "100.0"}])
        h2 = make_dummy_csv(c2, [{"tic_id": "2", "period_days": "2.0", "t0_btjd": "20.0", "duration_hours": "2.0", "depth_ppm": "200.0"}])

        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_conflict_sector",
            "shards": [
                {"sector": 100, "source_run_id": "run_A", "candidate_artifact": {"path": "c1.csv", "sha256": h1}},
                {"sector": 100, "source_run_id": "run_B", "candidate_artifact": {"path": "c2.csv", "sha256": h2}}
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("Conflicting source runs declared for sector 100", str(cm.exception))

    def test_failed_shard_rejected_in_complete_mode_but_accepted_with_allow_partial(self):
        """Negative control NC-5: Failed shard requires explicit partial opt-in."""
        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_partial_optin",
            "survey_mode": "complete",
            "shards": [
                {"sector": 108, "source_run_id": "run_failed_108", "candidate_artifact": {"status": "FAILED"}}
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        # In complete mode without opt-in -> must fail
        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file, allow_partial=False)
        self.assertIn("Explicit opt-in required for partial shards", str(cm.exception))

        # With allow_partial=True -> passes and correctly reports survey_mode='partial' (R1)
        _, _, survey_mode, is_partial, caveats = validate_and_load_manifest(manifest_file, allow_partial=True)
        self.assertEqual(survey_mode, "partial")
        self.assertTrue(is_partial)
        self.assertEqual(len(caveats), 1)

    def test_missing_sha256_in_artifact_rejected(self):
        """R2 / NC: Missing sha256 in declared candidate artifact is strictly rejected."""
        cand_path = os.path.join(self.test_dir, "c.csv")
        make_dummy_csv(cand_path, [{"tic_id": "1", "period_days": "1.0", "t0_btjd": "10.0", "duration_hours": "1.0", "depth_ppm": "100.0"}])

        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_no_sha",
            "shards": [
                {"sector": 100, "source_run_id": "r100", "candidate_artifact": {"path": "c.csv", "status": "COMPLETED"}}
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("missing required 'sha256'", str(cm.exception))

    def test_no_artifacts_declared_without_failed_status_rejected(self):
        """R1 / NC: Shard with no artifacts and no FAILED status is rejected."""
        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_no_art",
            "shards": [
                {"sector": 100, "source_run_id": "r100"}
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("no artifacts declared and no FAILED/MISSING status", str(cm.exception))

    def test_empty_zero_byte_csv_rejected(self):
        """R2 / NC: A 0-byte file declared as candidate artifact is rejected."""
        cand_path = os.path.join(self.test_dir, "empty.csv")
        open(cand_path, "wb").close()
        h = compute_file_sha256(cand_path)

        manifest_file = os.path.join(self.test_dir, "manifest.json")
        manifest_data = {
            "run_id": "run_zero_byte",
            "shards": [
                {"sector": 100, "source_run_id": "r100", "candidate_artifact": {"path": "empty.csv", "sha256": h, "status": "COMPLETED"}}
            ]
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(manifest_file)
        self.assertIn("empty (0 bytes)", str(cm.exception))


class TestLosslessCandidateMergingAndReconciliation(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_cand_")
        self.out_dir = os.path.join(self.test_dir, "output")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_lossless_merge_preserves_multiple_signals_on_same_tic(self):
        """R1/R2: Distinct signals on the same host TIC are preserved losslessly."""
        c1 = os.path.join(self.test_dir, "s100_cands.csv")
        h1 = make_dummy_csv(c1, [
            {"tic_id": "33692729", "period_days": "16.538", "t0_btjd": "4001.23", "duration_hours": "3.5", "depth_ppm": "1500.0"},
            {"tic_id": "33692729", "period_days": "4.052", "t0_btjd": "4005.67", "duration_hours": "2.1", "depth_ppm": "800.0"}
        ])

        manifest_data = {
            "run_id": "run_multi_signal",
            "shards": [
                {"sector": 100, "source_run_id": "run_s100", "candidate_artifact": {"path": "s100_cands.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}
            ]
        }

        df_merged, df_recon, summary = merge_candidate_shards(manifest_data, self.out_dir)
        self.assertEqual(len(df_merged), 2)
        self.assertEqual(summary["total_candidate_observations"], 2)
        self.assertEqual(summary["unique_host_tics"], 1)
        self.assertEqual(len(set(df_merged["observation_record_id"])), 2)
        self.assertEqual(int(df_recon["loaded_rows"].sum()), 2)
        self.assertEqual(int(df_recon["reconciled_rows"].sum()), 2)

    def test_header_only_valid_csv_is_legitimate_zero_rows(self):
        """Header-only candidate CSV is legitimate zero-candidate shard, not an error."""
        c_empty = os.path.join(self.test_dir, "s_empty.csv")
        df_empty_schema = pd.DataFrame(columns=["tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm"])
        df_empty_schema.to_csv(c_empty, index=False)
        h_empty = compute_file_sha256(c_empty)

        manifest_data = {
            "run_id": "run_empty_shard",
            "shards": [
                {"sector": 105, "source_run_id": "run_s105", "candidate_artifact": {"path": "s_empty.csv", "resolved_path": c_empty, "sha256": h_empty, "status": "EMPTY_NO_CANDIDATES"}}
            ]
        }

        df_merged, df_recon, summary = merge_candidate_shards(manifest_data, self.out_dir)
        self.assertEqual(len(df_merged), 0)
        self.assertEqual(summary["total_candidate_observations"], 0)
        self.assertEqual(int(df_recon["loaded_rows"].sum()), 0)
        self.assertEqual(int(df_recon["reconciled_rows"].sum()), 0)

    def test_malformed_tic_id_rejected_without_float_coercion(self):
        """Negative control NC-6: Malformed TIC ID string (e.g. '12345.67' or 'abc') is rejected."""
        c_bad = os.path.join(self.test_dir, "s_bad_tic.csv")
        h_bad = make_dummy_csv(c_bad, [
            {"tic_id": "12345.67", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
        ])

        manifest_data = {
            "run_id": "run_bad_tic",
            "shards": [
                {"sector": 100, "source_run_id": "run_s100", "candidate_artifact": {"path": "s_bad_tic.csv", "resolved_path": c_bad, "sha256": h_bad, "status": "COMPLETED"}}
            ]
        }

        with self.assertRaises(ShardReductionError) as cm:
            merge_candidate_shards(manifest_data, self.out_dir)
        self.assertIn("Malformed TIC ID", str(cm.exception))

    def test_deliberate_tic_drop_mutation_fails_reconciliation(self):
        """
        Negative control NC-7: A deliberate row drop or row count mismatch
        is caught by the production reconciliation check and raises ReconciliationError.
        Exercises the actual production reconciliation branch with intermediate mutation.
        """
        c1 = os.path.join(self.test_dir, "c1.csv")
        h1 = make_dummy_csv(c1, [
            {"tic_id": "1001", "period_days": "1.0", "t0_btjd": "10.0", "duration_hours": "1.0", "depth_ppm": "500.0"},
            {"tic_id": "1002", "period_days": "2.0", "t0_btjd": "20.0", "duration_hours": "2.0", "depth_ppm": "600.0"}
        ])

        manifest_data = {
            "run_id": "run_recon_mutation",
            "shards": [
                {"sector": 1, "source_run_id": "r1", "candidate_artifact": {"path": "c1.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}
            ]
        }

        # 1. Correct control passes: full production code reconciles without error
        df_merged, df_recon, _ = merge_candidate_shards(manifest_data, self.out_dir)
        self.assertEqual(len(df_merged), 2)
        self.assertEqual(int(df_recon["reconciled_rows"].sum()), 2)

        # 2. Mutate intermediate candidate records during production execution
        # Intercept DataFrame construction in merge_shards and drop the last row
        real_df = pd.DataFrame
        def faulty_dataframe(data=None, *args, **kwargs):
            df = real_df(data, *args, **kwargs)
            if isinstance(data, list) and len(data) > 1 and "observation_record_id" in data[0]:
                return df.iloc[:-1].copy()
            return df

        with patch("merge_shards.pd.DataFrame", side_effect=faulty_dataframe):
            with self.assertRaises(ReconciliationError) as cm:
                merge_candidate_shards(manifest_data, self.out_dir)
            self.assertIn("Row reconciliation failure", str(cm.exception))


    def test_duplicate_identical_files_produce_unique_observation_ids(self):
        """R3 / NC: Two byte-identical files in the same sector produce distinct observation IDs."""
        c1 = os.path.join(self.test_dir, "c1.csv")
        c2 = os.path.join(self.test_dir, "c2.csv")
        row = [{"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}]
        h1 = make_dummy_csv(c1, row)
        shutil.copy2(c1, c2)
        h2 = compute_file_sha256(c2)

        manifest_data = {
            "run_id": "run_identical_files",
            "shards": [
                {"sector": 100, "source_run_id": "r1", "candidate_artifact": {"path": "c1.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}},
                {"sector": 100, "source_run_id": "r2", "candidate_artifact": {"path": "c2.csv", "resolved_path": c2, "sha256": h2, "status": "COMPLETED"}}
            ]
        }

        df_merged, _, _ = merge_candidate_shards(manifest_data, self.out_dir)
        self.assertEqual(len(df_merged), 2)
        unique_ids = set(df_merged["observation_record_id"])
        self.assertEqual(len(unique_ids), 2, "Duplicate observation_record_ids produced for identical input files!")

    def test_reserved_columns_namespaced_and_preserved(self):
        """R3 / NC: Input rows containing reserved provenance columns are namespaced with orig_ and preserved."""
        c1 = os.path.join(self.test_dir, "c_reserved.csv")
        rows = [{
            "tic_id": "12345",
            "period_days": "2.5",
            "t0_btjd": "4000.0",
            "duration_hours": "1.5",
            "depth_ppm": "500.0",
            "source_run_id": "original-run",
            "source_sector": "42",
            "observation_record_id": "original-id"
        }]
        h1 = make_dummy_csv(c1, rows)

        manifest_data = {
            "run_id": "run_reserved",
            "shards": [
                {"sector": 100, "source_run_id": "shard-run", "candidate_artifact": {"path": "c_reserved.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}
            ]
        }

        df_merged, _, _ = merge_candidate_shards(manifest_data, self.out_dir)
        r0 = df_merged.iloc[0]
        self.assertEqual(r0["source_run_id"], "shard-run")
        self.assertEqual(r0["source_sector"], "100")
        self.assertTrue(r0["observation_record_id"].startswith("obs_"))
        # Verify original raw input values preserved under orig_ namespace
        self.assertEqual(r0["orig_source_run_id"], "original-run")
        self.assertEqual(r0["orig_source_sector"], "42")
        self.assertEqual(r0["orig_observation_record_id"], "original-id")


class TestHistoryMerging(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_hist_")
        self.out_dir = os.path.join(self.test_dir, "output")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_history_merging_dedupes_and_reconciles_counts(self):
        """History merging only ingests declared artifacts and produces exact union count."""
        h1_path = os.path.join(self.test_dir, "h1.txt")
        h2_path = os.path.join(self.test_dir, "h2.txt")

        hash1 = make_dummy_txt(h1_path, ["10001", "10002", "10003"])
        hash2 = make_dummy_txt(h2_path, ["10002", "10003", "10004"])

        manifest_data = {
            "run_id": "run_hist_dedup",
            "shards": [
                {"sector": 1, "source_run_id": "r1", "history_artifact": {"path": "h1.txt", "resolved_path": h1_path, "sha256": hash1, "status": "COMPLETED"}},
                {"sector": 2, "source_run_id": "r2", "history_artifact": {"path": "h2.txt", "resolved_path": h2_path, "sha256": hash2, "status": "COMPLETED"}}
            ]
        }

        union_tics, df_hist_recon, summary = merge_history_shards(manifest_data, self.out_dir)
        self.assertEqual(len(union_tics), 4)
        self.assertEqual(summary["total_raw_entries"], 6)
        self.assertEqual(summary["total_unique_scanned_targets"], 4)

        hist_file = os.path.join(self.out_dir, "scanned_tics.txt")
        with open(hist_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(lines, ["10001", "10002", "10003", "10004"])

    def test_malformed_history_tic_line_rejected(self):
        """R2 / NC: Corrupt or non-numeric line in history artifact raises error instead of silent drop."""
        h_bad_path = os.path.join(self.test_dir, "h_bad.txt")
        hash_bad = make_dummy_txt(h_bad_path, ["10001", "BAD-TIC-ID", "10003"])

        manifest_data = {
            "run_id": "run_hist_bad",
            "shards": [
                {"sector": 1, "source_run_id": "r1", "history_artifact": {"path": "h_bad.txt", "resolved_path": h_bad_path, "sha256": hash_bad, "status": "COMPLETED"}}
            ]
        }

        with self.assertRaises(ShardReductionError) as cm:
            merge_history_shards(manifest_data, self.out_dir)
        self.assertIn("Malformed TIC ID", str(cm.exception))


class TestFreshOutputDirectoryAndGuardedErrorBoundary(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_guard_")
        self.manifest_path = os.path.join(self.test_dir, "manifest.json")
        c1 = os.path.join(self.test_dir, "c1.csv")
        h1 = make_dummy_csv(c1, [{"tic_id": "123", "period_days": "1.0", "t0_btjd": "10.0", "duration_hours": "1.0", "depth_ppm": "100.0"}])
        with open(self.manifest_path, "w") as f:
            json.dump({
                "manifest_version": "1.0",
                "run_id": "guard_run",
                "shards": [{"sector": 1, "source_run_id": "r1", "candidate_artifact": {"path": "c1.csv", "sha256": h1, "status": "COMPLETED"}}]
            }, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_populated_output_directory_refuses_execution(self):
        """Negative control NC-8: Existing non-empty output directory refuses execution."""
        populated_dir = os.path.join(self.test_dir, "populated_output")
        os.makedirs(populated_dir, exist_ok=True)
        sentinel_file = os.path.join(populated_dir, "prior_data.txt")
        with open(sentinel_file, "w") as f:
            f.write("DO_NOT_OVERWRITE\n")

        rc = run_merge_shards(self.manifest_path, populated_dir)
        self.assertEqual(rc, EXIT_OUTPUT_DIR_EXISTS)

        with open(sentinel_file, "r") as f:
            content = f.read()
        self.assertEqual(content, "DO_NOT_OVERWRITE\n")

    def test_injected_write_error_triggers_failure_status_and_suppresses_manifest(self):
        """Negative control NC-9: Injected disk write fault triggers failure diagnostic and suppresses manifest."""
        out_dir = os.path.join(self.test_dir, "crash_output")
        rc = run_merge_shards(self.manifest_path, out_dir, _inject_write_failure=True)
        self.assertEqual(rc, EXIT_EXECUTION_FAILURE)

        self.assertFalse(os.path.exists(os.path.join(out_dir, "manifest.json")))
        status_file = os.path.join(out_dir, "run_status.json")
        self.assertTrue(os.path.exists(status_file))
        with open(status_file, "r") as f:
            status = json.load(f)
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["failed_stage"], "finalizing_manifest")

    def test_status_write_failure_unpublishes_manifest_and_fails_nonzero(self):
        """R4 / NC: If writing run_status.json fails, success manifest is unpublished and run fails nonzero."""
        out_dir = os.path.join(self.test_dir, "status_fail_output")

        with patch("merge_shards.write_run_status", return_value=False):
            rc = run_merge_shards(self.manifest_path, out_dir)

        self.assertNotEqual(rc, EXIT_SUCCESS)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "manifest.json")))

    def test_contradictory_cli_run_id_rejected(self):
        """R1 / NC: Contradiction between CLI --run-id and manifest run_id is rejected."""
        out_dir = os.path.join(self.test_dir, "contradictory_out")
        rc = run_merge_shards(self.manifest_path, out_dir, run_id="different_run_id")
        self.assertEqual(rc, EXIT_VALIDATION_ERROR)


class TestDownstreamOrchestrationSafety(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_downstream_")
        self.out_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_duplicate_tics_explicitly_blocks_downstream_consumers(self):
        """Requirement 6 & Negative control NC-11: Multiple signals on same TIC blocks downstream consumers."""
        df_dups = pd.DataFrame([
            {"tic_id": "33692729", "period_days": "16.538", "t0_btjd": "4001.23", "duration_hours": "3.5", "depth_ppm": "1500.0"},
            {"tic_id": "33692729", "period_days": "4.052", "t0_btjd": "4005.67", "duration_hours": "2.1", "depth_ppm": "800.0"}
        ])

        with self.assertRaises(BlockedDownstreamIncompatibilityError) as cm:
            orchestrate_downstream_stages(
                df_consolidated=df_dups,
                output_dir=self.out_dir,
                run_pleiades=True
            )
        self.assertIn("BLOCKED: Consolidated candidate catalog contains multiple observations", str(cm.exception))

    def test_zero_candidates_legitimately_skips_downstream_without_stale_root_access(self):
        """Requirement 5: Empty candidate catalog skips downstream stages without error."""
        df_empty = pd.DataFrame(columns=["tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm"])

        stages = orchestrate_downstream_stages(
            df_consolidated=df_empty,
            output_dir=self.out_dir,
            run_pleiades=True,
            run_antares=True
        )

        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["status"], "SKIPPED_EMPTY_CANDIDATES")

    def test_downstream_stage_failure_propagates_and_stops_chain(self):
        """
        Negative control NC-12: Downstream stage failure (rc=7) raises StageExecutionError,
        records failure status, and stops chain execution.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_stage_fail_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "1001", "period_days": "2.0", "t0_btjd": "100.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Create an isolated stub pleiades.py that exits 7
            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write("import sys\nsys.stderr.write('Injected Pleiades Crash\\n')\nsys.exit(7)\n")

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_pleiades=True,
                    run_antares=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "pleiades")
            self.assertEqual(cm.exception.returncode, 7)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_downstream_stage_rc0_but_missing_output_fails(self):
        """
        Negative control NC-13: Downstream stage exits 0 but does not create expected artifact.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_missing_out_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "2002", "period_days": "3.0", "t0_btjd": "200.0", "duration_hours": "2.0", "depth_ppm": "800.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Create stub pleiades.py that exits 0 but creates NO output file
            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write("import sys\nsys.exit(0)\n")

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_pleiades=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "pleiades")
            self.assertEqual(cm.exception.returncode, 0)
            self.assertIn("not generated", cm.exception.stderr_text)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_triceratops_rc0_missing_output_fails(self):
        """
        R4 / R5: TRICERATOPS stage exiting 0 without generating probability artifacts raises StageExecutionError.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_tric_missing_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "3003", "period_days": "4.0", "t0_btjd": "300.0", "duration_hours": "2.5", "depth_ppm": "1000.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            # Stub validate_triceratops.py exits 0 without creating probability outputs
            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write("import sys\nsys.exit(0)\n")

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertTrue(
                "not created" in cm.exception.stderr_text or "Missing TRICERATOPS probability table" in cm.exception.stderr_text
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_triceratops_zero_survivors_skips_cleanly(self):
        """
        R5: When earlier stages produce 0 candidate survivors, TRICERATOPS cleanly skips.
        """
        df_cand = pd.DataFrame([
            {"tic_id": "4004", "period_days": "1.0", "t0_btjd": "400.0", "duration_hours": "1.0", "depth_ppm": "500.0"}
        ])
        top3_empty = os.path.join(self.out_dir, "exofop_ctoi_upload_top3.csv")
        pd.DataFrame(columns=["tic_id"]).to_csv(top3_empty, index=False)
        stages = orchestrate_downstream_stages(
            df_consolidated=df_cand,
            output_dir=self.out_dir,
            run_triceratops=True
        )
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["status"], "SKIPPED_ZERO_TARGETS")

    def test_populated_output_dir_refusal_preserves_historical_files(self):
        """
        B1 / NC: Refusing an occupied output directory must never mutate or delete
        historical metadata or files (manifest.json, run_status.json, canary.txt).
        """
        pop_dir = os.path.join(self.test_dir, "populated_historical")
        os.makedirs(pop_dir, exist_ok=True)
        man_file = os.path.join(pop_dir, "manifest.json")
        stat_file = os.path.join(pop_dir, "run_status.json")
        canary_file = os.path.join(pop_dir, "canary.txt")

        with open(man_file, "w") as f:
            f.write('{"historical": "manifest_data"}')
        with open(stat_file, "w") as f:
            f.write('{"status": "SUCCESS", "run_id": "hist_01"}')
        with open(canary_file, "w") as f:
            f.write("canary_content_keep")

        before_hashes = {
            fname: compute_file_sha256(os.path.join(pop_dir, fname))
            for fname in os.listdir(pop_dir)
        }

        # Create valid input manifest
        c1 = os.path.join(self.test_dir, "c_pop.csv")
        h1 = make_dummy_csv(c1, [{"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}])
        input_man = os.path.join(self.test_dir, "input_pop_manifest.json")
        with open(input_man, "w") as f:
            json.dump({
                "manifest_version": "1.0",
                "run_id": "test_pop_refusal",
                "shards": [{"sector": 100, "source_run_id": "s1", "candidate_artifact": {"path": "c_pop.csv", "sha256": h1, "status": "COMPLETED"}}]
            }, f)

        # Execute reduction targeting populated directory
        rc = run_merge_shards(input_man, pop_dir)
        self.assertEqual(rc, EXIT_OUTPUT_DIR_EXISTS)

        # Verify populated directory was left 100% untouched
        after_hashes = {
            fname: compute_file_sha256(os.path.join(pop_dir, fname))
            for fname in os.listdir(pop_dir)
        }
        self.assertEqual(before_hashes, after_hashes, "Populated output directory files were mutated or deleted!")

    def test_recursive_collision_namespacing_preserves_all_sources(self):
        """
        B2 / NC: Recursive collision namespacing preserves both source_run_id and orig_source_run_id
        without silently clobbering existing metadata.
        """
        c1 = os.path.join(self.test_dir, "c_collision.csv")
        row = {
            "tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0",
            "source_run_id": "original-source", "orig_source_run_id": "older-source"
        }
        h1 = make_dummy_csv(c1, [row])
        manifest_data = {
            "run_id": "run_collision_test",
            "shards": [{"sector": 100, "source_run_id": "current-source", "candidate_artifact": {"path": "c_collision.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}]
        }

        df_merged, _, _ = merge_candidate_shards(manifest_data, self.out_dir)
        self.assertEqual(len(df_merged), 1)
        res_row = df_merged.iloc[0].to_dict()

        self.assertEqual(res_row["source_run_id"], "current-source")
        self.assertIn("original-source", res_row.values(), "original-source was lost during namespacing!")
        self.assertIn("older-source", res_row.values(), "older-source was lost during namespacing!")

    def test_duplicate_raw_csv_headers_rejected(self):
        """
        B2 / NC: Raw CSV containing duplicate column headers is rejected before pandas parsing.
        """
        c1 = os.path.join(self.test_dir, "c_dup_header.csv")
        with open(c1, "w", encoding="utf-8") as f:
            f.write("tic_id,period_days,t0_btjd,duration_hours,depth_ppm,note,note\n12345,2.5,4000.0,1.5,500.0,a,b\n")
        h1 = compute_file_sha256(c1)

        manifest_data = {
            "run_id": "run_dup_header",
            "shards": [{"sector": 100, "source_run_id": "s1", "candidate_artifact": {"path": "c_dup_header.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}]
        }

        with self.assertRaises(ShardReductionError) as cm:
            merge_candidate_shards(manifest_data, self.out_dir)
        self.assertIn("Duplicate or case-colliding column names", str(cm.exception))

    def test_skipped_artifact_requires_allow_partial(self):
        """
        B2 / NC: A shard with candidate_artifact status SKIPPED requires allow_partial=True.
        """
        input_man = os.path.join(self.test_dir, "input_skipped.json")
        with open(input_man, "w") as f:
            json.dump({
                "manifest_version": "1.0",
                "run_id": "test_skipped",
                "shards": [{"sector": 100, "source_run_id": "s1", "candidate_artifact": {"status": "SKIPPED"}}]
            }, f)

        # Disallowed by default (survey_mode complete)
        with self.assertRaises(ShardReductionError) as cm:
            validate_and_load_manifest(input_man, allow_partial=False)
        self.assertIn("Explicit opt-in required for partial shards", str(cm.exception))

        # Allowed with allow_partial=True
        data, _, eff_mode, is_part, caveats = validate_and_load_manifest(input_man, allow_partial=True)
        self.assertEqual(eff_mode, "partial")
        self.assertTrue(is_part)
        self.assertEqual(len(caveats), 1)

    def test_manifest_source_identity_mismatch_rejected(self):
        """
        B2 / NC: CSV containing contradictory sector or run_id is rejected.
        """
        c1 = os.path.join(self.test_dir, "c_mismatch.csv")
        row = {
            "tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0",
            "sector": "42", "run_id": "unrelated"
        }
        h1 = make_dummy_csv(c1, [row])
        manifest_data = {
            "run_id": "run_mismatch_test",
            "shards": [{"sector": 100, "source_run_id": "declared_source", "candidate_artifact": {"path": "c_mismatch.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}]
        }

        with self.assertRaises(ShardReductionError) as cm:
            merge_candidate_shards(manifest_data, self.out_dir)
        self.assertIn("contradicts declared manifest", str(cm.exception))

    def test_manifest_order_invariance_for_observation_ids(self):
        """
        B2 / NC: Observation record IDs must be invariant under manifest reordering.
        """
        c1 = os.path.join(self.test_dir, "c_order1.csv")
        c2 = os.path.join(self.test_dir, "c_order2.csv")
        h1 = make_dummy_csv(c1, [{"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}])
        h2 = make_dummy_csv(c2, [{"tic_id": "67890", "period_days": "3.5", "t0_btjd": "5000.0", "duration_hours": "2.0", "depth_ppm": "800.0"}])

        s1 = {"sector": 100, "source_run_id": "s1", "candidate_artifact": {"path": "c_order1.csv", "resolved_path": c1, "sha256": h1, "status": "COMPLETED"}}
        s2 = {"sector": 101, "source_run_id": "s2", "candidate_artifact": {"path": "c_order2.csv", "resolved_path": c2, "sha256": h2, "status": "COMPLETED"}}

        out1 = os.path.join(self.test_dir, "out_order1")
        out2 = os.path.join(self.test_dir, "out_order2")

        df1, _, _ = merge_candidate_shards({"run_id": "r", "shards": [s1, s2]}, out1)
        df2, _, _ = merge_candidate_shards({"run_id": "r", "shards": [s2, s1]}, out2)

        ids1 = {row["tic_id"]: row["observation_record_id"] for _, row in df1.iterrows()}
        ids2 = {row["tic_id"]: row["observation_record_id"] for _, row in df2.iterrows()}

        self.assertEqual(ids1, ids2, "Observation record IDs changed when manifest shards were reordered!")

    def test_pleiades_partial_targets_rejected(self):
        """
        B3 / NC: Pleiades stage generating summary for only a subset of candidate targets is rejected.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_pleiades_part_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"},
                {"tic_id": "67890", "period_days": "3.5", "t0_btjd": "5000.0", "duration_hours": "2.0", "depth_ppm": "800.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Stub pleiades only writes 1 target into summary
            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'scientific_dossier_summary.csv').write_text('tic_id\\n12345\\n')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_pleiades=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "pleiades")
            self.assertIn("failed to account for input candidate target", cm.exception.stderr_text)
            self.assertTrue(os.path.exists(os.path.join(stage_out, "stage_pleiades.log")))
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_triceratops_unrelated_target_rejected(self):
        """
        B3 / NC: TRICERATOPS stage generating probability file for unrelated target TIC 99999 is rejected.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_tric_unrelated_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"},
                {"tic_id": "67890", "period_days": "3.5", "t0_btjd": "5000.0", "duration_hours": "2.0", "depth_ppm": "800.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            # Stub validate_triceratops.py only writes unrelated target 99999
            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_99999.csv').write_text('unrelated\\n1\\n')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertIn("Missing TRICERATOPS probability table for expected target", cm.exception.stderr_text)
            self.assertTrue(os.path.exists(os.path.join(stage_out, "stage_triceratops.log")))
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_c1_pleiades_failed_target_stops_downstream_chain(self):
        """
        C1 / NC: Target operational error in Pleiades deep vetting stops downstream chain and prevents calling Antares.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_c1_error_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"},
                {"tic_id": "67890", "period_days": "3.5", "t0_btjd": "5000.0", "duration_hours": "2.0", "depth_ppm": "800.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Mock pleiades writing error rows
            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'scientific_dossier_summary.csv').write_text("
                    "'tic_id,operational_status,disposition,error_message\\n"
                    "12345,ERROR,,fit failure\\n"
                    "67890,ERROR,,fit failure\\n')\n"
                )

            # Mock antares writing a canary
            with open(os.path.join(iso_dir, "antares_packager.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1])\n"
                    "(p / 'antares_called.txt').write_text('yes')\n"
                )

            stages_hist = []
            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_pleiades=True,
                    run_antares=True,
                    python_exe=sys.executable,
                    stages_executed=stages_hist
                )
            self.assertEqual(cm.exception.stage, "pleiades")
            self.assertIn("Pleiades deep vetting failed for target(s)", cm.exception.stderr_text)
            self.assertFalse(os.path.exists(os.path.join(stage_out, "antares_called.txt")), "Antares was invoked despite Pleiades errors!")
            self.assertEqual(stages_hist[-1]["status"], "FAILED")
            self.assertEqual(stages_hist[-1]["failed_targets"], ["12345", "67890"])
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_c1_pleiades_cli_error_row_no_fabricated_science(self):
        """
        C1: Injected fit failure in Pleiades CLI block records operational ERROR with NaNs, not fabricated 0/100 science.
        """
        import ast
        from pleiades import __file__ as pleiades_source_path
        with open(pleiades_source_path, "r", encoding="utf-8-sig") as f:
            tree = ast.parse(f.read())
        block = next(x for x in tree.body if isinstance(x, ast.If) and isinstance(x.test, ast.Compare) and isinstance(x.test.left, ast.Name) and x.test.left.id == "__name__")

        iso_dir = tempfile.mkdtemp(prefix="ald_iso_c1_cli_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            cliout = os.path.join(iso_dir, "cliout")
            inp = os.path.join(iso_dir, "cliinput.csv")
            pd.DataFrame([
                {"tic_id": "12345", "period_days": 2.5, "t0_btjd": 4000.0, "duration_hours": 1.5, "depth_ppm": 500.0}
            ]).to_csv(inp, index=False)

            oldargv = sys.argv
            sys.argv = ["pleiades.py", "--input-csv", inp, "--output-dir", cliout]
            def injected_fail(*args, **kwargs):
                raise RuntimeError("deliberate fit failure")

            try:
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(
                        compile(ast.Module(body=[block], type_ignores=[]), "pleiades.py:main", "exec"),
                        {
                            "__name__": "__main__",
                            "os": os,
                            "sys": sys,
                            "pd": pd,
                            "INPUT_CSV": inp,
                            "VET_DIR": cliout,
                            "PIPELINE_VERSION": "test_c1",
                            "deep_vet_target": injected_fail
                        }
                    )
            finally:
                sys.argv = oldargv

            summary_csv = os.path.join(cliout, "scientific_dossier_summary.csv")
            df_summary = pd.read_csv(summary_csv, dtype={"tic_id": str})
            row = df_summary.iloc[0]
            self.assertEqual(str(row["tic_id"]), "12345")
            self.assertEqual(str(row["operational_status"]), "ERROR")
            self.assertTrue(pd.isna(row["disposition"]), f"Expected null disposition, got {row['disposition']}")
            self.assertTrue(pd.isna(row["vps_score"]), f"Expected NaN vps_score, got {row['vps_score']}")
            self.assertTrue(pd.isna(row["fp_risk_percent"]), f"Expected NaN fp_risk_percent, got {row['fp_risk_percent']}")
            self.assertTrue(pd.isna(row["fitted_rp_earth"]), f"Expected NaN fitted_rp_earth, got {row['fitted_rp_earth']}")
            self.assertTrue(pd.isna(row["dur_to_kepler_max"]), f"Expected NaN dur_to_kepler_max, got {row['dur_to_kepler_max']}")
            self.assertEqual(row["input_period_days"], 2.5)
            self.assertEqual(row["input_t0_btjd"], 4000.0)
            self.assertIn("deliberate fit failure", str(row["error_message"]))
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_c2_native_exofop_column_normalization_accepted(self):
        """
        C2: Native ExoFOP candidate table format ('TIC ID', 'Period', 'Epoch BJD', 'Duration', 'Depth')
        is normalized and accepted for TRICERATOPS validation.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_c2_exofop_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Native ExoFOP CTOI upload top3 table
            df_exofop_top3 = pd.DataFrame([
                {"TIC ID": "12345", "Period": 2.5, "Epoch BJD": 2459000.0, "Duration": 1.5, "Depth": 500.0}
            ])
            df_exofop_top3.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            # Mock validate_triceratops.py producing valid scenario probability table
            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('ID,scenario,prob\\n12345,TP,0.95\\n12345,EB,0.05\\n')\n"
                )

            stages = orchestrate_downstream_stages(
                df_consolidated=df_cand,
                output_dir=stage_out,
                run_triceratops=True,
                python_exe=sys.executable
            )
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]["stage"], "triceratops")
            self.assertEqual(stages[0]["status"], "COMPLETED")
            self.assertEqual(stages[0]["validated_targets"], 1)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_c2_triceratops_garbage_probability_rejected(self):
        """
        C2 / NC: Corrupt or unparseable TRICERATOPS probability CSV is rejected, preventing false completion.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_c2_garbage_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            # Mock writing literal garbage
            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('garbage')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertTrue(
                "missing required probability column" in cm.exception.stderr_text or "0 rows" in cm.exception.stderr_text
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_c2_triceratops_extra_unrequested_target_rejected(self):
        """
        C2 / NC: Extra unrequested probability files in triceratops directory are rejected.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_c2_extra_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('ID,scenario,prob\\n12345,TP,0.95\\n')\n"
                    "(p / 'triceratops_probs_TIC_99999.csv').write_text('ID,scenario,prob\\n99999,TP,0.90\\n')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertIn("unexpected probability files for unrequested targets", cm.exception.stderr_text)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_c3_stage_diagnostic_log_write_failure_is_fatal(self):
        """
        C3 / NC: Failure to persist required stage diagnostic log raises StageExecutionError and halts pipeline.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_c3_log_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Stub pleiades that creates valid output
            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'scientific_dossier_summary.csv').write_text('tic_id,operational_status\\n12345,SUCCESS\\n')\n"
                )

            # Block writing stage_pleiades.log by creating a directory with that name
            log_block = os.path.join(stage_out, "stage_pleiades.log")
            os.makedirs(log_block, exist_ok=True)

            stages_hist = []
            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_pleiades=True,
                    python_exe=sys.executable,
                    stages_executed=stages_hist
                )
            self.assertEqual(cm.exception.stage, "pleiades")
            self.assertIn("Failed to write required diagnostic log", cm.exception.stderr_text)
            self.assertIsNone(stages_hist[-1]["log_file"])
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_triceratops_missing_tier3_dependency_rejected(self):
        """
        B3 / NC: TRICERATOPS stage requested with non-empty candidates but missing Tier-3 tables raises StageExecutionError.
        """
        stage_out = os.path.join(self.test_dir, "stages_no_tier3")
        os.makedirs(stage_out, exist_ok=True)
        df_cand = pd.DataFrame([
            {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
        ])
        df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

        with self.assertRaises(StageExecutionError) as cm:
            orchestrate_downstream_stages(
                df_consolidated=df_cand,
                output_dir=stage_out,
                run_triceratops=True
            )
        self.assertEqual(cm.exception.stage, "triceratops")
        self.assertIn("Missing Tier-3 dependency", cm.exception.stderr_text)


    def test_d2_success_with_nonempty_error_message_halts_chain(self):
        """
        D2 / NC: Target reporting operational_status=SUCCESS but non-empty error_message halts downstream chain.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_d2_err_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            # Pleiades stub reporting contradictory metadata: SUCCESS + non-empty error_message
            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'scientific_dossier_summary.csv').write_text('tic_id,operational_status,error_message\\n12345,SUCCESS,injected fit failure\\n')\n"
                )
            with open(os.path.join(iso_dir, "antares_packager.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1])\n"
                    "(p / 'antares_ran.sentinel').write_text('yes')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_pleiades=True,
                    run_antares=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "pleiades")
            self.assertIn("error_message: injected fit failure", cm.exception.stderr_text)
            self.assertFalse(os.path.exists(os.path.join(stage_out, "antares_ran.sentinel")), "Antares must NOT be called when Pleiades target has error")
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_d2_legitimate_success_empty_error_message_passes(self):
        """
        D2: Target reporting operational_status=SUCCESS and empty error_message passes verification.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_d2_pass_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)

            with open(os.path.join(iso_dir, "pleiades.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'scientific_dossier_summary.csv').write_text('tic_id,operational_status,error_message\\n12345,SUCCESS,\\n')\n"
                )

            stages = orchestrate_downstream_stages(
                df_consolidated=df_cand,
                output_dir=stage_out,
                run_pleiades=True,
                python_exe=sys.executable
            )
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]["status"], "COMPLETED")
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_d1_triceratops_probability_greater_than_one_rejected(self):
        """
        D1 / NC: TRICERATOPS probability > 1.0 (e.g. 1.5 or 1.000001) is rejected with StageExecutionError.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_d1_gt1_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            # Mock stub producing prob 1.5 (fraction 150%)
            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('ID,scenario,prob\\n12345,TP,1.5\\n')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertIn("values outside [0, 1]", cm.exception.stderr_text)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_d1_triceratops_probability_negative_or_nan_rejected(self):
        """
        D1 / NC: Negative probability values or NaNs in probability column are rejected.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_d1_neg_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('ID,scenario,prob\\n12345,TP,-0.05\\n')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertIn("values outside [0, 1]", cm.exception.stderr_text)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_d1_triceratops_missing_scenario_column_rejected(self):
        """
        D1 / NC: TRICERATOPS probability table missing required 'scenario' column is rejected.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_d1_noscen_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            # Table without scenario column (e.g. ID,prob or ID,FPP,NFPP)
            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('ID,prob\\n12345,0.95\\n')\n"
                )

            with self.assertRaises(StageExecutionError) as cm:
                orchestrate_downstream_stages(
                    df_consolidated=df_cand,
                    output_dir=stage_out,
                    run_triceratops=True,
                    python_exe=sys.executable
                )
            self.assertEqual(cm.exception.stage, "triceratops")
            self.assertIn("missing required 'scenario' column", cm.exception.stderr_text)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_d1_triceratops_valid_scenarios_with_nearby_stars_accepted(self):
        """
        D1: TRICERATOPS scenario probability table with target TIC and nearby stars in [0, 1] is accepted.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_d1_blend_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            stage_out = os.path.join(iso_dir, "stages")
            os.makedirs(stage_out, exist_ok=True)
            df_cand = pd.DataFrame([
                {"tic_id": "12345", "period_days": "2.5", "t0_btjd": "4000.0", "duration_hours": "1.5", "depth_ppm": "500.0"}
            ])
            df_cand.to_csv(os.path.join(stage_out, "candidates_report.csv"), index=False)
            df_cand.to_csv(os.path.join(stage_out, "exofop_ctoi_upload_top3.csv"), index=False)

            with open(os.path.join(iso_dir, "validate_triceratops.py"), "w") as f:
                f.write(
                    "import sys, pathlib\n"
                    "p = pathlib.Path(sys.argv[sys.argv.index('--results-dir')+1]) / 'triceratops'\n"
                    "p.mkdir(exist_ok=True)\n"
                    "(p / 'triceratops_probs_TIC_12345.csv').write_text('ID,scenario,prob\\n12345,TP,0.90\\n99999,EB,0.10\\n')\n"
                )

            stages = orchestrate_downstream_stages(
                df_consolidated=df_cand,
                output_dir=stage_out,
                run_triceratops=True,
                python_exe=sys.executable
            )
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]["status"], "COMPLETED")
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)


    def test_e1_skipped_candidate_alone_full_pipeline(self):
        """
        E1 / NC: Shard with SKIPPED candidate artifact fails in complete mode (exit 3)
        and succeeds under --allow-partial (exit 0, status SUCCESS_PARTIAL_SURVEY).
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_e1_alone_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            man_path = os.path.join(iso_dir, "manifest.json")
            man_data = {
                "run_id": "test_e1_alone",
                "manifest_version": "1.0",
                "survey_mode": "complete",
                "shards": [{
                    "sector": 108,
                    "source_run_id": "s108_skipped_run",
                    "candidate_artifact": {"status": "SKIPPED"}
                }]
            }
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(man_data, f)

            # Complete mode: refusal with exit 3, success manifest suppressed, status FAILED
            out_complete = os.path.join(iso_dir, "out_complete")
            rc_comp = run_merge_shards(man_path, out_complete, allow_partial=False)
            self.assertEqual(rc_comp, EXIT_VALIDATION_ERROR)
            self.assertFalse(os.path.exists(os.path.join(out_complete, "manifest.json")))
            status_fail = os.path.join(out_complete, "run_status.json")
            with open(status_fail, "r", encoding="utf-8") as f:
                st_fail = json.load(f)
            self.assertEqual(st_fail["status"], "FAILED")
            self.assertEqual(st_fail["failed_stage"], "manifest_validation")

            # Partial mode: success with exit 0, SUCCESS_PARTIAL_SURVEY
            out_partial = os.path.join(iso_dir, "out_partial")
            rc_part = run_merge_shards(man_path, out_partial, allow_partial=True)
            self.assertEqual(rc_part, EXIT_SUCCESS)

            status_p = os.path.join(out_partial, "run_status.json")
            with open(status_p, "r", encoding="utf-8") as f:
                st = json.load(f)
            self.assertEqual(st["status"], "SUCCESS_PARTIAL_SURVEY")
            self.assertTrue(st["is_partial"])
            self.assertEqual(st["candidate_observations"], 0)
            self.assertEqual(st["scanned_targets"], 0)

            # Verify reconciliation row
            recon_p = os.path.join(out_partial, "candidate_reconciliation.csv")
            df_recon = pd.read_csv(recon_p)
            self.assertEqual(len(df_recon), 1)
            self.assertEqual(df_recon.iloc[0]["declared_status"], "SKIPPED")
            self.assertEqual(df_recon.iloc[0]["loaded_rows"], 0)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)

    def test_e1_valid_completed_plus_skipped_candidate_and_history(self):
        """
        E1: Compound run with a valid completed shard and a SKIPPED shard succeeds under --allow-partial.
        Verifies partial flag, caveats, counts, reconciliation tables, and artifact hashes.
        """
        iso_dir = tempfile.mkdtemp(prefix="ald_iso_e1_compound_")
        old_cwd = os.getcwd()
        try:
            os.chdir(iso_dir)
            s1_csv = os.path.join(iso_dir, "sector_1.csv")
            with open(s1_csv, "w", encoding="utf-8") as f:
                f.write("tic_id,period_days,t0_btjd,duration_hours,depth_ppm\n12345,2.5,4000.0,1.5,500.0\n")
            s1_hash = compute_file_sha256(s1_csv)

            s1_hist = os.path.join(iso_dir, "s1_hist.txt")
            with open(s1_hist, "w", encoding="utf-8") as f:
                f.write("12345\n67890\n")
            s1_hist_hash = compute_file_sha256(s1_hist)

            man_path = os.path.join(iso_dir, "manifest.json")
            man_data = {
                "run_id": "test_e1_compound",
                "manifest_version": "1.0",
                "survey_mode": "complete",
                "shards": [
                    {
                        "sector": 1,
                        "source_run_id": "s1_run",
                        "candidate_artifact": {"path": "sector_1.csv", "sha256": s1_hash, "status": "COMPLETED"},
                        "history_artifact": {"path": "s1_hist.txt", "sha256": s1_hist_hash, "status": "COMPLETED"}
                    },
                    {
                        "sector": 2,
                        "source_run_id": "s2_run",
                        "status": "SKIPPED",
                        "candidate_artifact": {"status": "SKIPPED"},
                        "history_artifact": {"status": "SKIPPED"}
                    }
                ]
            }
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(man_data, f)

            # Complete mode refusal
            out_complete = os.path.join(iso_dir, "out_complete")
            rc_comp = run_merge_shards(man_path, out_complete, allow_partial=False)
            self.assertEqual(rc_comp, EXIT_VALIDATION_ERROR)

            # Partial mode opt-in
            out_partial = os.path.join(iso_dir, "out_partial")
            rc_part = run_merge_shards(man_path, out_partial, allow_partial=True)
            self.assertEqual(rc_part, EXIT_SUCCESS)

            status_p = os.path.join(out_partial, "run_status.json")
            with open(status_p, "r", encoding="utf-8") as f:
                st = json.load(f)
            self.assertEqual(st["status"], "SUCCESS_PARTIAL_SURVEY")
            self.assertTrue(st["is_partial"])
            self.assertEqual(st["candidate_observations"], 1)
            self.assertEqual(st["scanned_targets"], 2)

            manifest_p = os.path.join(out_partial, "manifest.json")
            with open(manifest_p, "r", encoding="utf-8") as f:
                man_out = json.load(f)
            self.assertTrue(man_out["is_partial"])
            self.assertGreater(len(man_out["partial_caveats"]), 0)

            # Check that all artifacts on disk match declared SHA256 hashes
            for rel_p, art_info in man_out["artifacts"].items():
                full_p = os.path.join(out_partial, rel_p)
                self.assertTrue(os.path.exists(full_p))
                actual_h = compute_file_sha256(full_p)
                self.assertEqual(actual_h.upper(), art_info["sha256"].upper())

            # Check candidate reconciliation table
            recon_df = pd.read_csv(os.path.join(out_partial, "candidate_reconciliation.csv"))
            self.assertEqual(len(recon_df), 2)
            self.assertEqual(set(recon_df["declared_status"]), {"COMPLETED", "SKIPPED"})

            # Check history reconciliation table
            hist_recon_df = pd.read_csv(os.path.join(out_partial, "history_reconciliation.csv"))
            self.assertEqual(len(hist_recon_df), 2)
            self.assertEqual(set(hist_recon_df["declared_status"]), {"COMPLETED", "SKIPPED"})
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(iso_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()



