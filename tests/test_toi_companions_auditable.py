#!/usr/bin/env python3
"""
tests/test_toi_companions_auditable.py
======================================
Comprehensive offline regression test suite for ALD-002A (R1-R4 revisions):
Auditable TOI companion search outcomes, persistence, and run isolation.

Verifies:
1. Status, reason, and error capture for CANDIDATE, REJECT, EMPTY, SKIP, ERROR, and raised exceptions.
2. Exact totals reconciliation between queued targets and per-target audit records.
3. R1: Distinct recorded run outcomes and strict non-zero exit codes for ALL_ERRORS, FAILED_QUERY,
   PARTIAL_ERROR, and CANDIDATES_FOUND_WITH_ERRORS (preserving discovered candidates).
4. Truthful zero-candidate artifacts (headers preserved, 0 rows) and prevention of stale report masquerading.
5. Explicit worker initialization under both fork and spawn contexts.
6. R2: Run ID plot isolation and startup/final failure manifest lifecycle.
7. R2: Append mode resume and deduplication semantics.
8. R3: Portable relative artifact paths verified across scratch directory relocation and cleanup.
9. R4: Real Bash execution of cluster/run_toi_companions.pbs logic with shell stubs.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import unittest
import multiprocessing as mp
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np

import aldebaran as ald
import search_toi_companions as stc


def _probe_worker_context():
    """Top-level helper to probe spawned child process state."""
    import aldebaran as a
    return {
        "output_dir": a.OUTPUT_DIR,
        "log_file": a.LOG_FILE,
        "report_file": a.REPORT_FILE,
        "history_file": a.HISTORY_FILE,
        "run_id": a.CURRENT_RUN_ID,
        "ephems": dict(a.GLOBAL_TOI_EPHEMERIDES)
    }


class TestTOICompanionsAuditable(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_toi_")
        self.ephem_context = {
            "12345": [{"toi": "123.01", "period": 3.14, "t0_btjd": 1000.0, "duration_hours": 2.5, "depth_ppm": 500.0}],
            "67890": [{"toi": "456.01", "period": 5.20, "t0_btjd": 1005.0, "duration_hours": 3.0, "depth_ppm": 800.0}],
            "99999": [{"toi": "789.01", "period": 1.50, "t0_btjd": 1010.0, "duration_hours": 1.2, "depth_ppm": 300.0}]
        }
        ald.configure_environment(output_dir=self.test_dir, toi_ephemerides=self.ephem_context, run_id="setup_run")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        ald.configure_environment(output_dir="output", run_id="run_default")

    def test_process_toi_host_outcomes(self):
        """Test process_toi_host on CANDIDATE, REJECT, EMPTY, ERROR, and unhandled exceptions."""
        tic_id = "12345"
        sector = 106

        # 1. CANDIDATE outcome
        mock_cand = {
            "tic_id": tic_id,
            "sector": sector,
            "period_days": 2.5,
            "sde": 9.5,
            "plot_path": os.path.join(self.test_dir, f"candidate_TIC_{tic_id}_s{sector}_run_test.png")
        }
        with patch.object(ald, "analyze_single_target", return_value=(tic_id, "CANDIDATE", "Candidate detected!", mock_cand)):
            res = stc.process_toi_host(tic_id, sector, run_id="run_test")
            self.assertEqual(res["status"], "CANDIDATE")
            self.assertEqual(res["reason"], "CANDIDATE_FOUND")
            self.assertIsNotNone(res["candidate"])
            self.assertEqual(res["candidate"]["run_id"], "run_test")
            self.assertEqual(res["plot_path"], f"candidate_TIC_{tic_id}_s{sector}_run_test.png")

        # 2. REJECT outcome
        with patch.object(ald, "analyze_single_target", return_value=(tic_id, "REJECT", "REJECT: Severe aperture crowding (CROWDSAP=0.50)", None)):
            res = stc.process_toi_host(tic_id, sector, run_id="run_test")
            self.assertEqual(res["status"], "REJECT")
            self.assertEqual(res["reason"], "REJECT")
            self.assertIn("Severe aperture crowding", res["message"])
            self.assertIsNone(res["candidate"])

        # 3. EMPTY outcome
        with patch.object(ald, "analyze_single_target", return_value=(tic_id, "EMPTY", "Empty (SDE=5.2)", None)):
            res = stc.process_toi_host(tic_id, sector, run_id="run_test")
            self.assertEqual(res["status"], "EMPTY")
            self.assertIn("Empty", res["message"])
            self.assertIsNone(res["candidate"])

        # 4. Core ERROR outcome
        with patch.object(ald, "analyze_single_target", return_value=(tic_id, "ERROR", "ERROR: FITS corrupt or empty flux array", None)):
            res = stc.process_toi_host(tic_id, sector, run_id="run_test")
            self.assertEqual(res["status"], "ERROR")
            self.assertEqual(res["reason"], "PIPELINE_ERROR")
            self.assertIn("FITS corrupt", res["message"])
            self.assertIn("FITS corrupt", res["error_detail"])
            self.assertIsNone(res["candidate"])

        # 5. Unhandled exception in analyze_single_target
        with patch.object(ald, "analyze_single_target", side_effect=ZeroDivisionError("Division by zero in detrending")):
            res = stc.process_toi_host(tic_id, sector, run_id="run_test")
            self.assertEqual(res["status"], "ERROR")
            self.assertEqual(res["reason"], "ZeroDivisionError")
            self.assertIn("Division by zero", res["message"])
            self.assertTrue(len(res["error_detail"]) > 0)
            self.assertIn("ZeroDivisionError", res["error_detail"])

        # 6. Missing ephemeris context
        missing_tic = "999999999"
        res = stc.process_toi_host(missing_tic, sector, run_id="run_test")
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["reason"], "MISSING_EPHEMERIS_CONTEXT")
        self.assertIn("has no known TOI ephemeris", res["message"])

    def test_totals_reconciliation_and_audit_csv(self):
        """Verify that total queued targets reconcile exactly with audit records and counts."""
        run_output_dir = os.path.join(self.test_dir, "run_reconciliation")
        args = stc.parse_args([
            "--sectors", "106",
            "--workers", "1",
            "--output-dir", run_output_dir,
            "--run-id", "test_recon_001"
        ])

        targets = ["12345", "67890", "99999"]
        statuses = [
            ("12345", "CANDIDATE", "Candidate detected!", {"tic_id": "12345", "period_days": 3.0, "sde": 8.5}),
            ("67890", "REJECT", "REJECT: Duty cycle too wide", None),
            ("99999", "EMPTY", "Empty (SDE=4.9)", None)
        ]

        def mock_analyze(tid, sec):
            for t, disp, msg, cand in statuses:
                if str(t) == str(tid):
                    return tid, disp, msg, cand
            return tid, "ERROR", "Not found", None

        with patch.object(stc, "query_sector_toi_hosts", return_value=(targets, None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", side_effect=mock_analyze):

            exit_code, summary = stc.run_search(args)
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["run_status"], "CANDIDATES_FOUND")
            self.assertEqual(summary["total_queued"], 3)
            self.assertEqual(summary["total_processed"], 3)
            self.assertEqual(summary["counts"]["CANDIDATE"], 1)
            self.assertEqual(summary["counts"]["REJECT"], 1)
            self.assertEqual(summary["counts"]["EMPTY"], 1)
            self.assertEqual(summary["counts"]["ERROR"], 0)

            # Check audit CSV
            audit_path = os.path.join(run_output_dir, "toi_companions_audit.csv")
            self.assertTrue(os.path.exists(audit_path))
            df_audit = pd.read_csv(audit_path)
            self.assertEqual(len(df_audit), 3)
            self.assertEqual(set(df_audit["tic_id"].astype(str)), {"12345", "67890", "99999"})
            self.assertEqual(set(df_audit["status"]), {"CANDIDATE", "REJECT", "EMPTY"})

            # Check candidate summary CSV
            report_path = os.path.join(run_output_dir, "toi_companions_summary.csv")
            self.assertTrue(os.path.exists(report_path))
            df_report = pd.read_csv(report_path)
            self.assertEqual(len(df_report), 1)
            self.assertEqual(str(df_report.iloc[0]["tic_id"]), "12345")
            self.assertEqual(df_report.iloc[0]["run_id"], "test_recon_001")

    def test_r1_partial_failure_outcomes(self):
        """
        R1: Verify explicit non-zero exit codes and partial-failure statuses for mixed outcomes.
        Must preserve discovered candidates while propagating exit code 1.
        """
        # Scenario 1: Candidate discovered, but a worker ERROR occurred
        dir_cand_err = os.path.join(self.test_dir, "run_cand_err")
        args_ce = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", dir_cand_err, "--run-id", "run_ce"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345", "67890"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", side_effect=[
                 ("12345", "CANDIDATE", "Candidate", {"tic_id": "12345", "sde": 8.0, "period_days": 2.0}),
                 ("67890", "ERROR", "FITS corrupt", None)
             ]):
            exit_code, summary = stc.run_search(args_ce)
            self.assertEqual(exit_code, 1, "Must exit with code 1 when target error occurs despite candidates")
            self.assertEqual(summary["run_status"], "CANDIDATES_FOUND_WITH_ERRORS")
            self.assertEqual(summary["candidates_count"], 1)
            # Candidate must still be written to CSV
            df_cand = pd.read_csv(os.path.join(dir_cand_err, "toi_companions_summary.csv"))
            self.assertEqual(len(df_cand), 1)
            self.assertEqual(str(df_cand.iloc[0]["tic_id"]), "12345")

        # Scenario 2: Target rejected, but a sector query failed
        dir_part_q = os.path.join(self.test_dir, "run_part_q")
        args_pq = stc.parse_args(["--sectors", "106", "105", "--workers", "1", "--output-dir", dir_part_q, "--run-id", "run_pq"])
        with patch.object(stc, "query_sector_toi_hosts", side_effect=[(["12345"], None), ([], "MAST Timeout")]), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "REJECT: Low SDE", None)):
            exit_code, summary = stc.run_search(args_pq)
            self.assertEqual(exit_code, 1, "Must exit with code 1 when a sector query fails")
            self.assertEqual(summary["run_status"], "PARTIAL_ERROR")
            self.assertEqual(summary["sector_query_errors"], 1)

    def test_distinct_outcomes_zero_host_and_all_rejected(self):
        """Verify distinct outcomes for ZERO_HOSTS and ALL_REJECTED (exit code 0)."""
        # Case A: ZERO_HOSTS
        dir_zh = os.path.join(self.test_dir, "run_zh")
        args_zh = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", dir_zh, "--run-id", "run_zh"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=([], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)):
            exit_code, summary = stc.run_search(args_zh)
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["run_status"], "ZERO_HOSTS")
            self.assertEqual(summary["total_queued"], 0)
            self.assertEqual(summary["candidates_count"], 0)
            self.assertTrue(os.path.exists(os.path.join(dir_zh, "run_summary.json")))

        # Case B: ALL_REJECTED
        dir_ar = os.path.join(self.test_dir, "run_ar")
        args_ar = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", dir_ar, "--run-id", "run_ar"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345", "67890"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "REJECT: Low SDE", None)):
            exit_code, summary = stc.run_search(args_ar)
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["run_status"], "ALL_REJECTED")
            self.assertEqual(summary["total_queued"], 2)
            self.assertEqual(summary["counts"]["REJECT"], 2)
            self.assertEqual(summary["candidates_count"], 0)

    def test_distinct_outcomes_all_error_and_failed_query(self):
        """Verify non-zero exit and distinct recorded outcomes for ALL_ERRORS and FAILED_QUERY."""
        # Case A: ALL_ERRORS
        dir_ae = os.path.join(self.test_dir, "run_ae")
        args_ae = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", dir_ae, "--run-id", "run_ae"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345", "67890"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=("12345", "ERROR", "ERROR: Corrupt data", None)):
            exit_code, summary = stc.run_search(args_ae)
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["run_status"], "ALL_ERRORS")
            self.assertEqual(summary["total_processed"], 2)
            self.assertEqual(summary["counts"]["ERROR"], 2)

        # Case B: FAILED_QUERY
        dir_fq = os.path.join(self.test_dir, "run_fq")
        args_fq = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", dir_fq, "--run-id", "run_fq"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=([], "MAST Connection Timeout")), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)):
            exit_code, summary = stc.run_search(args_fq)
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["run_status"], "FAILED_QUERY")
            self.assertEqual(summary["sector_query_errors"], 1)

            # Audit CSV must record query failure
            df_audit = pd.read_csv(os.path.join(dir_fq, "toi_companions_audit.csv"))
            self.assertEqual(len(df_audit), 1)
            self.assertEqual(df_audit.iloc[0]["reason"], "SECTOR_QUERY_FAILED")

    def test_zero_candidate_truthful_output_no_stale_mixing(self):
        """Verify that zero-candidate runs produce valid empty CSV and do not keep old reports."""
        run_dir = os.path.join(self.test_dir, "run_stale_test")
        os.makedirs(run_dir, exist_ok=True)
        report_file = os.path.join(run_dir, "toi_companions_summary.csv")

        # Create a stale report with a previous fake candidate
        stale_df = pd.DataFrame([{
            "run_id": "old_stale_run",
            "tic_id": "999999",
            "sector": 100,
            "period_days": 10.0,
            "sde": 15.0
        }])
        stale_df.to_csv(report_file, index=False)
        self.assertTrue(os.path.exists(report_file))

        # Run a fresh zero-candidate search (default append=False)
        args = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", run_dir, "--run-id", "fresh_run"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "REJECT: Low SDE", None)):
            exit_code, summary = stc.run_search(args)
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["candidates_count"], 0)

            # Report file must exist, contain standard headers, and have EXACTLY 0 data rows
            self.assertTrue(os.path.exists(report_file))
            df_fresh = pd.read_csv(report_file)
            self.assertEqual(len(df_fresh), 0)
            self.assertIn("tic_id", df_fresh.columns)
            self.assertIn("period_days", df_fresh.columns)
            # Old candidate must NOT be present
            self.assertNotIn("old_stale_run", df_fresh["run_id"].values if len(df_fresh) > 0 else [])

    def test_worker_initialization_and_spawn_environment(self):
        """Test _init_worker sets paths and TOI ephemeris dictionary in current process."""
        worker_dir = os.path.join(self.test_dir, "worker_env_test")
        ephems = {"55555": [{"toi": "555.01", "period": 4.0, "t0_btjd": 100.0, "duration_hours": 2.0, "depth_ppm": 400.0}]}

        stc._init_worker(worker_dir, ephems, run_id="w_run")
        self.assertEqual(ald.OUTPUT_DIR, worker_dir)
        self.assertEqual(ald.LOG_FILE, os.path.join(worker_dir, "pipeline_run.log"))
        self.assertEqual(ald.REPORT_FILE, os.path.join(worker_dir, "candidates_report.csv"))
        self.assertEqual(ald.HISTORY_FILE, os.path.join(worker_dir, "scanned_tics.txt"))
        self.assertEqual(ald.CURRENT_RUN_ID, "w_run")
        self.assertIn("55555", ald.GLOBAL_TOI_EPHEMERIDES)
        self.assertEqual(ald.get_known_transits_for_tic("55555")[0]["toi"], "555.01")

    def test_spawn_worker_context(self):
        """Verify spawned worker process initializes output paths and catalog context under spawn."""
        spawn_ctx = mp.get_context("spawn")
        test_out = os.path.join(self.test_dir, "spawn_test_out")
        os.makedirs(test_out, exist_ok=True)
        test_ephems = {"88888": [{"toi": "888.01", "period": 7.0, "t0_btjd": 200.0, "duration_hours": 3.0, "depth_ppm": 600.0}]}

        with spawn_ctx.Pool(processes=1, initializer=stc._init_worker, initargs=(test_out, test_ephems, None, "spawn_run_01")) as pool:
            res = pool.apply(_probe_worker_context)

        self.assertEqual(res["output_dir"], test_out)
        self.assertEqual(res["log_file"], os.path.join(test_out, "pipeline_run.log"))
        self.assertEqual(res["report_file"], os.path.join(test_out, "candidates_report.csv"))
        self.assertEqual(res["run_id"], "spawn_run_01")
        self.assertIn("88888", res["ephems"])
        self.assertEqual(res["ephems"]["88888"][0]["toi"], "888.01")

    def test_r2_run_id_plot_isolation_and_startup_manifest_lifecycle(self):
        """
        R2: Verify run ID plot isolation (no overwrite on repeated runs with append)
        and startup failure manifest lifecycle (no stale summary left behind).
        """
        root = os.path.join(self.test_dir, "run_r2_root")
        os.makedirs(root, exist_ok=True)

        # 1. Run A produces candidate
        dummy_plot_a = os.path.join(root, "dummy_a.png")
        Path(dummy_plot_a).write_text("runA_content", encoding="utf-8")
        args_a = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "runA"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=(
                 "12345", "CANDIDATE", "Candidate A", {"tic_id": "12345", "sector": 106, "plot_path": dummy_plot_a, "sde": 8.0, "period_days": 2.0}
             )):
            stc.run_search(args_a)

        plot_a = Path(root) / "candidate_TIC_12345_s106_runA.png"
        self.assertTrue(plot_a.exists())
        self.assertEqual(plot_a.read_text(encoding="utf-8"), "runA_content")

        # 2. Run B (with --append) processes same target
        dummy_plot_b = os.path.join(root, "dummy_b.png")
        Path(dummy_plot_b).write_text("runB_content", encoding="utf-8")
        args_b = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "runB", "--append"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=(
                 "12345", "CANDIDATE", "Candidate B", {"tic_id": "12345", "sector": 106, "plot_path": dummy_plot_b, "sde": 8.5, "period_days": 2.5}
             )):
            stc.run_search(args_b)

        # Plot A must NOT be overwritten by Run B
        plot_b = Path(root) / "candidate_TIC_12345_s106_runB.png"
        self.assertTrue(plot_a.exists())
        self.assertTrue(plot_b.exists())
        self.assertEqual(plot_a.read_text(encoding="utf-8"), "runA_content")
        self.assertEqual(plot_b.read_text(encoding="utf-8"), "runB_content")

        df_summary = pd.read_csv(os.path.join(root, "toi_companions_summary.csv"))
        # Summary CSV must contain both distinct plot paths
        self.assertEqual(df_summary["plot_path"].nunique(), len(df_summary))

        # 3. Startup failure manifest lifecycle
        args_fail = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "run_startup_fail"])
        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", side_effect=RuntimeError("Catalog offline")):
            with self.assertRaises(RuntimeError):
                stc.run_search(args_fail)

        # The run_summary.json must NOT describe runB anymore; it must describe run_startup_fail with STARTUP_ERROR!
        with open(os.path.join(root, "run_summary.json"), "r", encoding="utf-8") as f:
            summary_fail = json.load(f)
        self.assertEqual(summary_fail["run_id"], "run_startup_fail")
        self.assertEqual(summary_fail["run_status"], "STARTUP_ERROR")
        self.assertEqual(summary_fail["exit_code"], 1)

    def test_r3_portable_relative_paths_and_relocation(self):
        """
        R3: Verify that saved plot_path, summary_csv, and audit_csv use portable relative paths
        that resolve correctly after copying to a new root and deleting the original scratch directory.
        """
        scratch_dir = os.path.join(self.test_dir, "scratch_source")
        repo_dir = os.path.join(self.test_dir, "repo_destination")
        os.makedirs(scratch_dir, exist_ok=True)

        args = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", scratch_dir, "--run-id", "portable_run"])

        dummy_plot = os.path.join(scratch_dir, "plot_source.png")
        Path(dummy_plot).write_text("portable_plot_data", encoding="utf-8")

        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=(
                 "12345", "CANDIDATE", "Candidate", {"tic_id": "12345", "sector": 106, "plot_path": dummy_plot, "sde": 9.0, "period_days": 3.0}
             )):
            exit_code, summary = stc.run_search(args)
            self.assertEqual(exit_code, 0)

        # Verify summary CSV has relative plot_path
        df_scratch = pd.read_csv(os.path.join(scratch_dir, "toi_companions_summary.csv"))
        saved_rel_plot = df_scratch.iloc[0]["plot_path"]
        self.assertFalse(os.path.isabs(saved_rel_plot), f"plot_path must be relative, got {saved_rel_plot}")

        # Simulate PBS copy to permanent repository and scratch cleanup
        shutil.copytree(scratch_dir, repo_dir)
        shutil.rmtree(scratch_dir)  # Original scratch is completely deleted!

        # At the new repository destination:
        df_repo = pd.read_csv(os.path.join(repo_dir, "toi_companions_summary.csv"))
        dest_plot_file = os.path.join(repo_dir, df_repo.iloc[0]["plot_path"])
        self.assertTrue(os.path.exists(dest_plot_file), f"Plot must resolve at destination: {dest_plot_file}")
        self.assertEqual(Path(dest_plot_file).read_text(encoding="utf-8"), "portable_plot_data")

        with open(os.path.join(repo_dir, "run_summary.json"), "r", encoding="utf-8") as f:
            dest_summary = json.load(f)
        self.assertTrue(os.path.exists(os.path.join(repo_dir, dest_summary["summary_csv"])))
        self.assertTrue(os.path.exists(os.path.join(repo_dir, dest_summary["audit_csv"])))

    def _get_bash_path(self):
        for candidate in ["C:\\Program Files\\Git\\bin\\bash.exe", "bash"]:
            if shutil.which(candidate):
                return candidate
        return None

    def test_r4_pbs_shell_script_execution(self):
        """
        R4: Executes the actual PBS bash script logic under Bash with local shell stubs.
        Verifies:
        1. Success: exit 0, results copied, clean_scratch invoked.
        2. Python failure: exit 1, diagnostics copied, clean_scratch NOT invoked.
        3. Copy failure: exit 1, clean_scratch NOT invoked.
        4. Cleanup failure: non-zero cleanup exit code propagated, results preserved in repo storage.
        5. Workload failure preservation: original error code preserved without overwrite.
        """
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
set -e
# Source sync_and_finalize from actual PBS file
eval "$(sed -n '/^sync_and_finalize()/,/^}}/p' '{pbs_path}')"

TEST_ROOT="$(mktemp -d)"
SCRATCH="$TEST_ROOT/scratch"
REPO="$TEST_ROOT/repo"
mkdir -p "$SCRATCH/tess_results/toi_companions" "$REPO"

# Stub clean_scratch
CLEAN_COUNT=0
clean_scratch() {{
    CLEAN_COUNT=$((CLEAN_COUNT + 1))
    return 0
}}

# Case 1: Normal success (py_exit=0, copy succeeds)
echo "test_artifact" > "$SCRATCH/tess_results/toi_companions/result.txt"
sync_and_finalize 0 "$SCRATCH" "$REPO"
[ "$?" -eq 0 ] || exit 11
[ -f "$REPO/tess_results/toi_companions/result.txt" ] || exit 12
[ "$CLEAN_COUNT" -eq 1 ] || exit 13

# Case 2: Python failure (py_exit=1, copy succeeds)
CLEAN_COUNT=0
echo "error_diag" > "$SCRATCH/tess_results/toi_companions/diag.txt"
set +e
sync_and_finalize 1 "$SCRATCH" "$REPO"
RC_FAIL=$?
set -e
[ "$RC_FAIL" -eq 1 ] || exit 21
[ -f "$REPO/tess_results/toi_companions/diag.txt" ] || exit 22
[ "$CLEAN_COUNT" -eq 0 ] || exit 23

# Case 3: Copy failure (destination is blocked by a non-directory file)
CLEAN_COUNT=0
touch "$TEST_ROOT/block_file"
INVALID_REPO="$TEST_ROOT/block_file/invalid_dest"
set +e
sync_and_finalize 0 "$SCRATCH" "$INVALID_REPO"
RC_CP_FAIL=$?
set -e
[ "$RC_CP_FAIL" -eq 1 ] || exit 31
[ "$CLEAN_COUNT" -eq 0 ] || exit 32

# Case 4: Cleanup failure (py_exit=0, copy succeeds, clean_scratch fails with exit code 5)
CLEAN_COUNT=0
echo "test_artifact_c4" > "$SCRATCH/tess_results/toi_companions/result_c4.txt"
clean_scratch() {{
    CLEAN_COUNT=$((CLEAN_COUNT + 1))
    echo "Simulated clean_scratch failure" >&2
    return 5
}}
set +e
sync_and_finalize 0 "$SCRATCH" "$REPO"
RC_CLEAN_FAIL=$?
set -e
[ "$RC_CLEAN_FAIL" -eq 5 ] || exit 41
[ -f "$REPO/tess_results/toi_companions/result_c4.txt" ] || exit 42
[ "$CLEAN_COUNT" -eq 1 ] || exit 43

# Case 5: Original workload error preserved without overwrite (py_exit=42)
CLEAN_COUNT=0
echo "test_artifact_c5" > "$SCRATCH/tess_results/toi_companions/result_c5.txt"
set +e
sync_and_finalize 42 "$SCRATCH" "$REPO"
RC_ORIG_FAIL=$?
set -e
[ "$RC_ORIG_FAIL" -eq 42 ] || exit 51
[ -f "$REPO/tess_results/toi_companions/result_c5.txt" ] || exit 52
[ "$CLEAN_COUNT" -eq 0 ] || exit 53

echo "ALL_BASH_STUB_TESTS_PASSED"
rm -rf "$TEST_ROOT"
"""
        proc = subprocess.run(
            [bash_path, "-c", test_script],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, f"Bash stub failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("ALL_BASH_STUB_TESTS_PASSED", proc.stdout)

    def test_r5_pbs_setup_venv_creation_failure_fails_closed(self):
        """Verify that venv creation failure fails closed with exit code 3 and diagnostic."""
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
PBS_PATH='{pbs_path}'
run_setup() {{
    eval "$(sed -n '/# Setup Python virtual environment/,/^sync_and_finalize()/ {{ /^sync_and_finalize()/!p }}' "$PBS_PATH")"
}}

(
    TEST_ROOT="$(mktemp -d)"
    SCRATCHDIR="$TEST_ROOT/scratch"
    mkdir -p "$SCRATCHDIR"
    cd "$SCRATCHDIR"

    python3() {{
        echo "[STUB] python3 -m venv failed" >&2
        return 127
    }}

    run_setup
)
RC=$?
[ "$RC" -eq 3 ] || exit 10
echo "VENV_CREATION_FAILURE_HANDLED"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("VENV_CREATION_FAILURE_HANDLED", proc.stdout)
        self.assertIn("ERROR: Failed to create virtual environment", proc.stderr)

    def test_r5_pbs_setup_activation_missing_fails_closed(self):
        """Verify that missing activation script fails closed with exit code 3 and diagnostic."""
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
PBS_PATH='{pbs_path}'
run_setup() {{
    eval "$(sed -n '/# Setup Python virtual environment/,/^sync_and_finalize()/ {{ /^sync_and_finalize()/!p }}' "$PBS_PATH")"
}}

(
    TEST_ROOT="$(mktemp -d)"
    SCRATCHDIR="$TEST_ROOT/scratch"
    mkdir -p "$SCRATCHDIR"
    cd "$SCRATCHDIR"

    python3() {{
        mkdir -p "$SCRATCHDIR/venv"
        return 0
    }}

    run_setup
)
RC=$?
[ "$RC" -eq 3 ] || exit 20
echo "ACTIVATION_MISSING_HANDLED"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("ACTIVATION_MISSING_HANDLED", proc.stdout)
        self.assertIn("ERROR: Virtual environment activation script missing", proc.stderr)

    def test_r5_pbs_setup_activation_source_failure_fails_closed(self):
        """Verify that failed activation script source fails closed with exit code 3."""
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
PBS_PATH='{pbs_path}'
run_setup() {{
    eval "$(sed -n '/# Setup Python virtual environment/,/^sync_and_finalize()/ {{ /^sync_and_finalize()/!p }}' "$PBS_PATH")"
}}

(
    TEST_ROOT="$(mktemp -d)"
    SCRATCHDIR="$TEST_ROOT/scratch"
    mkdir -p "$SCRATCHDIR/venv/bin"
    cd "$SCRATCHDIR"
    echo "return 1" > "$SCRATCHDIR/venv/bin/activate"

    python3() {{ return 0; }}

    run_setup
)
RC=$?
[ "$RC" -eq 3 ] || exit 30
echo "ACTIVATION_SOURCE_FAILURE_HANDLED"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("ACTIVATION_SOURCE_FAILURE_HANDLED", proc.stdout)
        self.assertIn("ERROR: Failed to activate virtual environment", proc.stderr)

    def test_r5_pbs_setup_interpreter_mismatch_fails_closed(self):
        """
        Verify that an actual active Python interpreter mismatch (probe runs and fails)
        fails closed with exit code 3, logs diagnostic, and prevents workload execution.
        """
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")
        real_python = sys.executable.replace("\\", "/")

        test_script = f"""
TEST_ROOT="$(mktemp -d)"
FAKE_REPO="$TEST_ROOT/repo"
FAKE_SCRATCH="$TEST_ROOT/scratch"
SENTINEL="$TEST_ROOT/workload_ran.txt"

mkdir -p "$FAKE_REPO" "$FAKE_SCRATCH"
touch "$FAKE_REPO/search_toi_companions.py"

(
    export PBS_O_WORKDIR="$FAKE_REPO"
    export SCRATCHDIR="$FAKE_SCRATCH"
    export PBS_JOBID="test_job_123"

    # Stub venv creation and activation to succeed without a real venv directory
    python3() {{
        mkdir -p "$SCRATCHDIR/venv/bin"
        echo "# dummy activate" > "$SCRATCHDIR/venv/bin/activate"
        return 0
    }}

    # Active python resolves to the host/system Python outside SCRATCHDIR/venv,
    # allowing the actual production probe command to run and evaluate sys.prefix.
    # If the workload were executed, it would touch SENTINEL.
    python() {{
        if [ "$1" = "search_toi_companions.py" ]; then
            touch "$SENTINEL"
            return 0
        fi
        '{real_python}' "$@"
    }}

    export -f python3 python

    bash '{pbs_path}'
)
RC=$?

[ "$RC" -eq 3 ] || exit 40
[ ! -f "$SENTINEL" ] || exit 41
echo "INTERPRETER_MISMATCH_HANDLED_AND_WORKLOAD_PREVENTED"
rm -rf "$TEST_ROOT"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("INTERPRETER_MISMATCH_HANDLED_AND_WORKLOAD_PREVENTED", proc.stdout)
        self.assertIn("Active Python sys.prefix", proc.stderr)
        self.assertIn("ERROR: Active Python interpreter is not from intended venv", proc.stderr)

    def test_r5_pbs_setup_pip_upgrade_failure_fails_closed(self):
        """Verify that pip upgrade failure fails closed with exit code 3 and diagnostic."""
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
PBS_PATH='{pbs_path}'
run_setup() {{
    eval "$(sed -n '/# Setup Python virtual environment/,/^sync_and_finalize()/ {{ /^sync_and_finalize()/!p }}' "$PBS_PATH")"
}}

(
    TEST_ROOT="$(mktemp -d)"
    SCRATCHDIR="$TEST_ROOT/scratch"
    mkdir -p "$SCRATCHDIR/venv/bin"
    cd "$SCRATCHDIR"
    echo "# activate" > "$SCRATCHDIR/venv/bin/activate"

    python3() {{ return 0; }}
    python() {{
        if [ "$1" = "-c" ]; then
            return 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ] && [ "$4" = "--upgrade" ]; then
            echo "[STUB] pip upgrade failed" >&2
            return 1
        fi
        return 0
    }}

    run_setup
)
RC=$?
[ "$RC" -eq 3 ] || exit 50
echo "PIP_UPGRADE_FAILURE_HANDLED"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("PIP_UPGRADE_FAILURE_HANDLED", proc.stdout)
        self.assertIn("ERROR: Failed to upgrade pip in virtual environment", proc.stderr)

    def test_r5_pbs_setup_dependency_install_failure_fails_closed(self):
        """Verify that dependency installation failure fails closed with exit code 3 and diagnostic."""
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
PBS_PATH='{pbs_path}'
run_setup() {{
    eval "$(sed -n '/# Setup Python virtual environment/,/^sync_and_finalize()/ {{ /^sync_and_finalize()/!p }}' "$PBS_PATH")"
}}

(
    TEST_ROOT="$(mktemp -d)"
    SCRATCHDIR="$TEST_ROOT/scratch"
    mkdir -p "$SCRATCHDIR/venv/bin"
    cd "$SCRATCHDIR"
    echo "# activate" > "$SCRATCHDIR/venv/bin/activate"
    touch "$SCRATCHDIR/requirements.txt"

    python3() {{ return 0; }}
    python() {{
        if [ "$1" = "-c" ]; then
            return 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
            if [ "$4" = "--upgrade" ]; then
                # pip upgrade succeeds
                return 0
            fi
            # dependency install fails
            echo "[STUB] pip dependency install failed" >&2
            return 1
        fi
        return 0
    }}

    run_setup
)
RC=$?
[ "$RC" -eq 3 ] || exit 51
echo "PIP_DEP_INSTALL_FAILURE_HANDLED"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("PIP_DEP_INSTALL_FAILURE_HANDLED", proc.stdout)
        self.assertIn("ERROR: Failed to install dependencies", proc.stderr)

    def test_r5_pbs_setup_bare_pip_shadowing_not_invoked(self):
        """
        Verify that bare pip on PATH is not invoked by the corrected setup path,
        ensuring pip is executed exclusively through the verified python interpreter.
        """
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
PBS_PATH='{pbs_path}'
run_setup() {{
    eval "$(sed -n '/# Setup Python virtual environment/,/^sync_and_finalize()/ {{ /^sync_and_finalize()/!p }}' "$PBS_PATH")"
}}

TEST_ROOT="$(mktemp -d)"
SCRATCHDIR="$TEST_ROOT/scratch"
mkdir -p "$SCRATCHDIR/venv/bin"
cd "$SCRATCHDIR"
echo "# activate" > "$SCRATCHDIR/venv/bin/activate"
touch "$SCRATCHDIR/requirements.txt"

SHADOW_SENTINEL="$TEST_ROOT/shadowed_bare_pip_invoked.txt"

# Create a rogue/shadowed bare pip executable in a dedicated directory
SHADOW_BIN="$TEST_ROOT/shadow_bin"
mkdir -p "$SHADOW_BIN"
cat << SHADOW_EOF > "$SHADOW_BIN/pip"
#!/bin/sh
touch "$SHADOW_SENTINEL"
echo "ROGUE_BARE_PIP_INVOKED" >&2
exit 99
SHADOW_EOF
chmod +x "$SHADOW_BIN/pip"

(
    # Prepend rogue bare pip directory to PATH
    export PATH="$SHADOW_BIN:$PATH"

    python3() {{ return 0; }}
    python() {{
        if [ "$1" = "-c" ]; then
            return 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
            # Verified venv python pip succeeds
            return 0
        fi
        return 0
    }}

    run_setup
)
RC=$?
[ "$RC" -eq 0 ] || exit 60
[ ! -f "$SHADOW_SENTINEL" ] || exit 61
echo "BARE_PIP_SHADOWING_AVOIDED_SUCCESSFULLY"
rm -rf "$TEST_ROOT"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("BARE_PIP_SHADOWING_AVOIDED_SUCCESSFULLY", proc.stdout)
        self.assertNotIn("ROGUE_BARE_PIP_INVOKED", proc.stderr)

    def test_r5_pbs_full_script_setup_failure_prevents_workload_execution(self):
        """Verify that in the full PBS script, a setup failure immediately aborts before workload execution."""
        bash_path = self._get_bash_path()
        if not bash_path:
            self.skipTest("Bash executable not found in environment")

        pbs_path = os.path.abspath("cluster/run_toi_companions.pbs").replace("\\", "/")

        test_script = f"""
TEST_ROOT="$(mktemp -d)"
FAKE_REPO="$TEST_ROOT/repo"
FAKE_SCRATCH="$TEST_ROOT/scratch"
SENTINEL="$TEST_ROOT/workload_ran.txt"

mkdir -p "$FAKE_REPO" "$FAKE_SCRATCH"
touch "$FAKE_REPO/search_toi_companions.py"

(
    export PBS_O_WORKDIR="$FAKE_REPO"
    export SCRATCHDIR="$FAKE_SCRATCH"
    export PBS_JOBID="test_job_123"

    python3() {{
        echo "[STUB] python3 -m venv failed" >&2
        return 127
    }}

    python() {{
        touch "$SENTINEL"
        return 0
    }}

    export -f python3 python

    bash '{pbs_path}'
)
RC=$?

[ "$RC" -eq 3 ] || exit 60
[ ! -f "$SENTINEL" ] || exit 61
echo "FULL_PBS_SETUP_FAILURE_ABORTED_SAFELY"
rm -rf "$TEST_ROOT"
"""
        proc = subprocess.run([bash_path, "-c", test_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("FULL_PBS_SETUP_FAILURE_ABORTED_SAFELY", proc.stdout)

    def test_r2_same_id_append_idempotent_resume(self):
        """
        R2: Verify that repeating --append with the same run-id on already completed targets
        is strictly idempotent: skips redundant execution, preserves candidate plot without
        overwriting, avoids duplicate CSV rows, and keeps candidates_count synchronized with CSV.
        """
        root = os.path.join(self.test_dir, "run_idempotent_resume")
        os.makedirs(root, exist_ok=True)

        seq = [0]
        def mock_analyze(tic, sec):
            seq[0] += 1
            plot_file = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot_file.write_text(f"call_{seq[0]}", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic,
                "sector": sec,
                "period_days": 2.0,
                "sde": 8.0,
                "plot_path": str(plot_file)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "analyze_single_target", side_effect=mock_analyze):

            # Call 1: initial run
            args1 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "same", "--append"])
            rc1, summary1 = stc.run_search(args1)
            self.assertEqual(rc1, 0)
            self.assertEqual(seq[0], 1)
            self.assertEqual(summary1["candidates_count"], 1)
            self.assertEqual(summary1["cumulative_candidates_count"], 1)

            # Call 2: repeated run with identical run-id and --append (retry/resume)
            args2 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "same", "--append"])
            rc2, summary2 = stc.run_search(args2)
            self.assertEqual(rc2, 0)

            # Execution must be idempotent: analyze was NOT called a second time
            self.assertEqual(seq[0], 1)

            df_summary = pd.read_csv(os.path.join(root, "toi_companions_summary.csv"))
            # CSV must contain exactly 1 row (no duplicate row appended)
            self.assertEqual(len(df_summary), 1)
            self.assertEqual(df_summary["plot_path"].nunique(), 1)

            # Plot must NOT have been overwritten
            plot_path = Path(root) / df_summary["plot_path"].iloc[0]
            self.assertTrue(plot_path.exists())
            self.assertEqual(plot_path.read_text(encoding="utf-8"), "call_1")

            # Candidate counts must remain synchronized between JSON manifest and CSV
            self.assertEqual(summary2["candidates_count"], 1)
            self.assertEqual(summary2["cumulative_candidates_count"], 1)

            # Per-run manifest must exist and match
            per_run_manifest = Path(root) / "run_summary_same.json"
            self.assertTrue(per_run_manifest.exists())
            with open(per_run_manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            self.assertEqual(manifest_data["candidates_count"], 1)
            self.assertEqual(manifest_data["cumulative_candidates_count"], 1)

    def test_r2_same_id_fresh_run_collision_rejection(self):
        """
        R2: Verify that starting a fresh run (without --append) with an existing run-id
        is rejected before resetting outputs, preserving completed run metadata and artifacts.
        """
        root = os.path.join(self.test_dir, "run_collision_rejection")
        os.makedirs(root, exist_ok=True)

        args1 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_collision"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "REJECT", None)):
            rc1, summary1 = stc.run_search(args1)
            self.assertEqual(rc1, 0)

        manifest_path = Path(root) / "run_summary_test_collision.json"
        self.assertTrue(manifest_path.exists())
        audit_path = Path(root) / "toi_companions_audit.csv"
        df_aud_before = pd.read_csv(audit_path)
        self.assertEqual(len(df_aud_before), 1)

        # Attempting second run with same run-id WITHOUT --append must raise FileExistsError
        args2 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_collision"])
        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)):
            with self.assertRaises(FileExistsError):
                stc.run_search(args2)

        # Completed run metadata and audit records must be preserved
        self.assertTrue(manifest_path.exists())
        df_aud_after = pd.read_csv(audit_path)
        self.assertEqual(len(df_aud_after), 1)

    def test_r2_default_run_id_microsecond_resolution_and_manifests(self):
        """
        R2: Verify default run-id includes microsecond resolution so rapid back-to-back
        runs receive distinct IDs, preserving independent per-run manifests.
        """
        root = os.path.join(self.test_dir, "run_default_ids")
        os.makedirs(root, exist_ok=True)

        args1 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root])
        args2 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root])

        with patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), self.ephem_context)), \
             patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "REJECT", None)):
            rc1, sum1 = stc.run_search(args1)
            rc2, sum2 = stc.run_search(args2)

        id1 = sum1["run_id"]
        id2 = sum2["run_id"]
        self.assertNotEqual(id1, id2)

        # Both independent per-run manifests must exist on disk
        manifest1 = Path(root) / f"run_summary_{id1}.json"
        manifest2 = Path(root) / f"run_summary_{id2}.json"
        self.assertTrue(manifest1.exists())
        self.assertTrue(manifest2.exists())

    def test_r2a_fresh_run_preserves_prior_manifest_and_evidence(self):
        """
        R2a: Verify that starting a new distinct-ID fresh run preserves prior run's
        referenced candidate and audit evidence via durable per-run files.
        """
        root = os.path.join(self.test_dir, "run_r2a_preservation")
        os.makedirs(root, exist_ok=True)

        def cand_a(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot_A", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            # Run A: candidate found
            args_a = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "run_A"])
            with patch.object(ald, "analyze_single_target", side_effect=cand_a):
                rc_a, sum_a = stc.run_search(args_a)
            self.assertEqual(rc_a, 0)
            self.assertEqual(sum_a["candidates_count"], 1)

            # Read Run A manifest
            manifest_a_path = Path(root) / "run_summary_run_A.json"
            self.assertTrue(manifest_a_path.exists())
            with open(manifest_a_path, "r", encoding="utf-8") as f:
                manifest_a = json.load(f)

            # Run B: fresh run in same directory (append=False) with different ID
            args_b = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "run_B"])
            with patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "REJECT", None)):
                rc_b, sum_b = stc.run_search(args_b)
            self.assertEqual(rc_b, 0)
            self.assertEqual(sum_b["candidates_count"], 0)

            # Inspect Run A's referenced evidence: MUST BE FULLY PRESERVED
            report_a = pd.read_csv(Path(root) / manifest_a["summary_csv"])
            audit_a = pd.read_csv(Path(root) / manifest_a["audit_csv"])
            self.assertEqual(manifest_a["candidates_count"], 1)
            self.assertEqual(len(report_a), 1)
            self.assertEqual(set(audit_a["run_id"]), {"run_A"})

            # Run B's manifest references its own files
            manifest_b_path = Path(root) / "run_summary_run_B.json"
            with open(manifest_b_path, "r", encoding="utf-8") as f:
                manifest_b = json.load(f)
            report_b = pd.read_csv(Path(root) / manifest_b["summary_csv"])
            audit_b = pd.read_csv(Path(root) / manifest_b["audit_csv"])
            self.assertEqual(manifest_b["candidates_count"], 0)
            self.assertEqual(len(report_b), 0)
            self.assertEqual(set(audit_b["run_id"]), {"run_B"})

            # Ordinary append behavior: Run C with --append accumulates
            args_c = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "run_C", "--append"])
            with patch.object(ald, "analyze_single_target", side_effect=cand_a):
                rc_c, sum_c = stc.run_search(args_c)
            self.assertEqual(rc_c, 0)
            self.assertEqual(sum_c["candidates_count"], 1)
            # Cumulative summary file contains accumulated candidates
            df_cum = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertGreaterEqual(len(df_cum), 1)

    def test_r2b_interrupted_candidate_persistence_recovery(self):
        """
        R2b: Verify that when candidate CSV persistence fails after audit row write,
        resuming with --append does NOT silently skip the candidate; it reprocesses
        the target, commits the candidate to CSV and manifest, and finishes exit 0.
        """
        root = os.path.join(self.test_dir, "run_r2b_interrupted")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot_data", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "resume_test"])
        args_resume = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "resume_test", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            # Attempt 1: simulate failure during candidate CSV append
            original_to_csv = pd.DataFrame.to_csv
            def fail_candidate_save(df_self, path, *a, **kw):
                if Path(path).name == "toi_companions_summary.csv" and kw.get("mode") == "a":
                    raise OSError("simulated interrupted candidate persistence")
                return original_to_csv(df_self, path, *a, **kw)

            with patch.object(ald, "analyze_single_target", side_effect=mock_cand), \
                 patch.object(pd.DataFrame, "to_csv", new=fail_candidate_save):
                with self.assertRaises(OSError):
                    stc.run_search(args_run)

            # Audit has candidate recorded, but candidate CSV has NO candidate
            audit_file = Path(root) / "toi_companions_audit_resume_test.csv"
            self.assertTrue(audit_file.exists())
            df_aud = pd.read_csv(audit_file)
            self.assertEqual(len(df_aud[df_aud["status"] == "CANDIDATE"]), 1)
            cand_file = Path(root) / "toi_companions_summary_resume_test.csv"
            self.assertTrue(cand_file.exists())
            df_cand_empty = pd.read_csv(cand_file)
            self.assertEqual(len(df_cand_empty), 0)

            # Attempt 2: resume with --append; MUST re-execute worker and recover candidate
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand) as mock_worker:
                rc2, sum2 = stc.run_search(args_resume)

            self.assertEqual(mock_worker.call_count, 1)
            self.assertEqual(rc2, 0)
            self.assertEqual(sum2["counts"]["CANDIDATE"], 1)
            self.assertEqual(sum2["candidates_count"], 1)
            df_cand_recovered = pd.read_csv(cand_file)
            self.assertEqual(len(df_cand_recovered), 1)

    def test_r2b_resume_reprocesses_when_candidate_record_or_plot_missing(self):
        """
        R2b: Verify that on resume, if an audited CANDIDATE is missing its candidate CSV
        record or its plot file is missing/empty, it is re-queued and re-processed.
        """
        root = os.path.join(self.test_dir, "run_r2b_missing_plot")
        os.makedirs(root, exist_ok=True)

        call_count = [0]
        def mock_cand(tic, sec):
            call_count[0] += 1
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("valid_plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args1 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "plot_test", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             patch.object(ald, "analyze_single_target", side_effect=mock_cand):

            rc1, sum1 = stc.run_search(args1)
            self.assertEqual(rc1, 0)
            self.assertEqual(call_count[0], 1)

            # Corrupt/delete the plot file
            plot_file = Path(root) / f"candidate_TIC_12345_s106_plot_test.png"
            self.assertTrue(plot_file.exists())
            plot_file.unlink()  # delete plot file

            # Second call with --append: since plot is missing, must re-execute worker and regenerate
            rc2, sum2 = stc.run_search(args1)
            self.assertEqual(rc2, 0)
            self.assertEqual(call_count[0], 2)
            self.assertTrue(plot_file.exists())
            self.assertEqual(sum2["candidates_count"], 1)

    def test_r2b_resume_rejects_unreadable_resume_evidence(self):
        """
        R2b: Verify that unreadable/corrupted resume files raise RuntimeError instead of
        being silently swallowed into a false successful run.
        """
        root = os.path.join(self.test_dir, "run_r2b_unreadable")
        os.makedirs(root, exist_ok=True)

        audit_file = Path(root) / "toi_companions_audit.csv"
        audit_file.write_text("corrupted,binary,\x00\xff\xfe", encoding="latin-1")

        args = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "unreadable", "--append"])
        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):
            with patch("pandas.read_csv", side_effect=Exception("Disk read error")):
                with self.assertRaises(RuntimeError) as cm:
                    stc.run_search(args)
                self.assertIn("failed to read", str(cm.exception))

    def test_r2b_retry_noncandidate_reject_cleans_candidate_tables(self):
        """
        R2b: Verify that when a formerly candidate target is re-analyzed (e.g. after plot deletion)
        and yields REJECT, stale candidate records are removed from both per-run and cumulative
        candidate reports, while audit history, counts and exit 0 are preserved.
        """
        root = os.path.join(self.test_dir, "run_r2b_retry_reject")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_retry_rej", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            # Step 1: candidate discovered and saved
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand):
                rc1, sum1 = stc.run_search(args_run)
            self.assertEqual(rc1, 0)
            self.assertEqual(sum1["candidates_count"], 1)
            self.assertEqual(len(pd.read_csv(Path(root) / sum1["summary_csv"])), 1)
            self.assertEqual(len(pd.read_csv(Path(root) / "toi_companions_summary.csv")), 1)

            # Step 2: plot deleted -> retry re-analyzes and returns REJECT
            plot_file = Path(root) / f"candidate_TIC_12345_s106_test_retry_rej.png"
            plot_file.unlink()

            with patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "Transit shape rejected", None)):
                rc2, sum2 = stc.run_search(args_run)

            self.assertEqual(rc2, 0)
            self.assertEqual(sum2["run_status"], "ALL_REJECTED")
            self.assertEqual(sum2["candidates_count"], 0)
            self.assertEqual(sum2["cumulative_candidates_count"], 0)
            self.assertEqual(sum2["counts"]["CANDIDATE"], 0)
            self.assertEqual(sum2["counts"]["REJECT"], 1)

            # Candidate CSVs must have 0 rows (stale candidate removed)
            df_per_run = pd.read_csv(Path(root) / sum2["summary_csv"])
            self.assertEqual(len(df_per_run), 0)
            df_cum = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum), 0)

            # Audit history preserved: contains both attempts
            df_audit = pd.read_csv(Path(root) / sum2["audit_csv"])
            self.assertEqual(list(df_audit["status"]), ["CANDIDATE", "REJECT"])

    def test_r2b_retry_noncandidate_empty_cleans_candidate_tables(self):
        """
        R2b: Verify that when a formerly candidate target is re-analyzed and yields EMPTY,
        stale candidate records are removed from both candidate reports and counts agree.
        """
        root = os.path.join(self.test_dir, "run_r2b_retry_empty")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_retry_emp", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            with patch.object(ald, "analyze_single_target", side_effect=mock_cand):
                rc1, sum1 = stc.run_search(args_run)
            self.assertEqual(rc1, 0)

            plot_file = Path(root) / f"candidate_TIC_12345_s106_test_retry_emp.png"
            plot_file.unlink()

            with patch.object(ald, "analyze_single_target", return_value=("12345", "EMPTY", "No BLS peak", None)):
                rc2, sum2 = stc.run_search(args_run)

            self.assertEqual(rc2, 0)
            self.assertEqual(sum2["run_status"], "ALL_REJECTED")
            self.assertEqual(sum2["candidates_count"], 0)
            self.assertEqual(sum2["counts"]["EMPTY"], 1)

            df_per_run = pd.read_csv(Path(root) / sum2["summary_csv"])
            self.assertEqual(len(df_per_run), 0)
            df_cum = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum), 0)
            df_audit = pd.read_csv(Path(root) / sum2["audit_csv"])
            self.assertEqual(list(df_audit["status"]), ["CANDIDATE", "EMPTY"])

    def test_r2b_retry_error_cleans_candidate_tables_and_fails_explicitly(self):
        """
        R2b: Verify that when a formerly candidate target is re-analyzed and fails with ERROR,
        the unverified candidate is NOT presented as valid in candidate tables, and the run
        fails explicitly with exit code 1 (ALL_ERRORS).
        """
        root = os.path.join(self.test_dir, "run_r2b_retry_error")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_retry_err", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            with patch.object(ald, "analyze_single_target", side_effect=mock_cand):
                rc1, sum1 = stc.run_search(args_run)
            self.assertEqual(rc1, 0)

            plot_file = Path(root) / f"candidate_TIC_12345_s106_test_retry_err.png"
            plot_file.unlink()

            with patch.object(ald, "analyze_single_target", return_value=("12345", "ERROR", "Pipeline crash on target", None)):
                rc2, sum2 = stc.run_search(args_run)

            # Must fail explicitly with exit code 1 and ALL_ERRORS status
            self.assertEqual(rc2, 1)
            self.assertEqual(sum2["exit_code"], 1)
            self.assertEqual(sum2["run_status"], "ALL_ERRORS")
            self.assertEqual(sum2["candidates_count"], 0)
            self.assertEqual(sum2["counts"]["ERROR"], 1)

            # Active candidate tables must NOT contain the unresolved candidate
            df_per_run = pd.read_csv(Path(root) / sum2["summary_csv"])
            self.assertEqual(len(df_per_run), 0)
            df_cum = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum), 0)

            # Audit log records the error attempt
            df_audit = pd.read_csv(Path(root) / sum2["audit_csv"])
            self.assertEqual(list(df_audit["status"]), ["CANDIDATE", "ERROR"])

    def test_r2b_interrupted_noncandidate_invalidation_recovery_reject(self):
        """
        R2b: Verify that when non-candidate invalidation is interrupted (e.g. OSError
        removing per-run candidate row after cumulative cleanup during REJECT retry),
        the subsequent resume startup detects the inconsistency, purges the stale candidate row,
        synchronizes candidate counts/manifests to 0, and remains idempotent on second restart.
        """
        root = os.path.join(self.test_dir, "run_r2b_interrupted_reject")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_interr_rej", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            # Step 1: initial run commits candidate to both tables
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand):
                rc1, sum1 = stc.run_search(args_run)
            self.assertEqual(rc1, 0)
            self.assertEqual(sum1["candidates_count"], 1)

            # Step 2: delete plot and re-run with REJECT, injecting failure removing per-run candidate row
            plot_file = Path(root) / f"candidate_TIC_12345_s106_test_interr_rej.png"
            plot_file.unlink()

            original_remove = stc._remove_candidate_row
            def fail_remove(path, *a, **kw):
                if Path(path).name == "toi_companions_summary_test_interr_rej.csv":
                    raise OSError("injected interrupted per-run invalidation")
                return original_remove(path, *a, **kw)

            with patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "Transit shape rejected", None)), \
                 patch.object(stc, "_remove_candidate_row", side_effect=fail_remove):
                with self.assertRaises(OSError):
                    stc.run_search(args_run)

            # Fault verification: cumulative report was cleaned, but per-run CSV still holds stale candidate
            df_cum_interr = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum_interr), 0)
            df_per_run_interr = pd.read_csv(Path(root) / "toi_companions_summary_test_interr_rej.csv")
            self.assertEqual(len(df_per_run_interr), 1)

            # Step 3 (Restart 1): resume must reconcile tables, purge stale per-run row, and not re-run worker
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand) as worker:
                rc3, sum3 = stc.run_search(args_run)

            self.assertEqual(worker.call_count, 0)
            self.assertEqual(rc3, 0)
            self.assertEqual(sum3["run_status"], "ALL_REJECTED")
            self.assertEqual(sum3["candidates_count"], 0)
            self.assertEqual(sum3["cumulative_candidates_count"], 0)
            self.assertEqual(sum3["counts"]["REJECT"], 1)

            # Per-run CSV now has 0 rows (reconciliation repaired state)
            df_per_run_repaired = pd.read_csv(Path(root) / sum3["summary_csv"])
            self.assertEqual(len(df_per_run_repaired), 0)
            df_cum_repaired = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum_repaired), 0)

            # Step 4 (Restart 2): second resume remains clean and idempotent
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand) as worker2:
                rc4, sum4 = stc.run_search(args_run)

            self.assertEqual(worker2.call_count, 0)
            self.assertEqual(rc4, 0)
            self.assertEqual(sum4["candidates_count"], 0)
            df_per_run_4 = pd.read_csv(Path(root) / sum4["summary_csv"])
            self.assertEqual(len(df_per_run_4), 0)

    def test_r2b_interrupted_noncandidate_invalidation_recovery_empty(self):
        """
        R2b: Verify that interrupted invalidation on retry returning EMPTY is recovered
        on resume: stale candidate row is purged, counts agree, and worker is not re-run.
        """
        root = os.path.join(self.test_dir, "run_r2b_interrupted_empty")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_interr_emp", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            with patch.object(ald, "analyze_single_target", side_effect=mock_cand):
                stc.run_search(args_run)

            plot_file = Path(root) / f"candidate_TIC_12345_s106_test_interr_emp.png"
            plot_file.unlink()

            original_remove = stc._remove_candidate_row
            def fail_remove(path, *a, **kw):
                if Path(path).name == "toi_companions_summary_test_interr_emp.csv":
                    raise OSError("injected interrupted per-run invalidation")
                return original_remove(path, *a, **kw)

            with patch.object(ald, "analyze_single_target", return_value=("12345", "EMPTY", "No peak", None)), \
                 patch.object(stc, "_remove_candidate_row", side_effect=fail_remove):
                with self.assertRaises(OSError):
                    stc.run_search(args_run)

            # On resume: stale row cleaned, counts agree, worker not called
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand) as worker:
                rc, sum_res = stc.run_search(args_run)

            self.assertEqual(worker.call_count, 0)
            self.assertEqual(rc, 0)
            self.assertEqual(sum_res["run_status"], "ALL_REJECTED")
            self.assertEqual(sum_res["candidates_count"], 0)
            self.assertEqual(sum_res["counts"]["EMPTY"], 1)

            df_per_run = pd.read_csv(Path(root) / sum_res["summary_csv"])
            self.assertEqual(len(df_per_run), 0)
            df_cum = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum), 0)

    def test_r2b_interrupted_noncandidate_invalidation_recovery_error(self):
        """
        R2b: Verify that interrupted invalidation on retry returning ERROR is recovered on resume:
        stale candidate row is purged from candidate tables, and target is re-queued rather than
        silently treated as completed, failing explicitly if error recurs.
        """
        root = os.path.join(self.test_dir, "run_r2b_interrupted_error")
        os.makedirs(root, exist_ok=True)

        def mock_cand(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {"12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}]}
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_interr_err", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)):

            with patch.object(ald, "analyze_single_target", side_effect=mock_cand):
                stc.run_search(args_run)

            plot_file = Path(root) / f"candidate_TIC_12345_s106_test_interr_err.png"
            plot_file.unlink()

            original_remove = stc._remove_candidate_row
            def fail_remove(path, *a, **kw):
                if Path(path).name == "toi_companions_summary_test_interr_err.csv":
                    raise OSError("injected interrupted per-run invalidation")
                return original_remove(path, *a, **kw)

            with patch.object(ald, "analyze_single_target", return_value=("12345", "ERROR", "Crash", None)), \
                 patch.object(stc, "_remove_candidate_row", side_effect=fail_remove):
                with self.assertRaises(OSError):
                    stc.run_search(args_run)

            # On resume: target has ERROR in audit, so it must be re-queued, NOT marked completed!
            # If retry errors again, it must fail explicitly with exit_code=1 and ALL_ERRORS
            with patch.object(ald, "analyze_single_target", return_value=("12345", "ERROR", "Crash again", None)) as worker:
                rc, sum_res = stc.run_search(args_run)

            self.assertEqual(worker.call_count, 1)
            self.assertEqual(rc, 1)
            self.assertEqual(sum_res["exit_code"], 1)
            self.assertEqual(sum_res["run_status"], "ALL_ERRORS")
            self.assertEqual(sum_res["candidates_count"], 0)
            self.assertEqual(sum_res["counts"]["ERROR"], 1)

            # Both candidate tables must have 0 rows (no unverified candidate advertised)
            df_per_run = pd.read_csv(Path(root) / sum_res["summary_csv"])
            self.assertEqual(len(df_per_run), 0)
            df_cum = pd.read_csv(Path(root) / "toi_companions_summary.csv")
            self.assertEqual(len(df_cum), 0)

    def test_r2b_interrupted_invalidation_preserves_unrelated_candidates_and_runs(self):
        """
        R2b: Verify that candidate table reconciliation on resume purges only the invalid key,
        preserving other valid candidates in the same run and prior runs in cumulative files.
        """
        root = os.path.join(self.test_dir, "run_r2b_preserve_unrelated")
        os.makedirs(root, exist_ok=True)

        def mock_cand_multi(tic, sec):
            plot = Path(ald.OUTPUT_DIR) / f"candidate_TIC_{tic}_s{sec}_{ald.CURRENT_RUN_ID}.png"
            plot.write_text("plot", encoding="utf-8")
            return tic, "CANDIDATE", "Candidate detected", {
                "tic_id": tic, "sector": sec, "period_days": 2.0, "sde": 8.0, "plot_path": str(plot)
            }

        ctx = {
            "12345": [{"toi": "123.01", "period": 2.0, "t0_btjd": 100.0, "duration_hours": 1.0, "depth_ppm": 500.0}],
            "99999": [{"toi": "999.01", "period": 3.0, "t0_btjd": 105.0, "duration_hours": 1.5, "depth_ppm": 600.0}]
        }
        args_run = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", root, "--run-id", "test_preserve", "--append"])

        with patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
             patch.object(stc, "query_sector_toi_hosts", return_value=(["12345", "99999"], None)):

            # Initial run: discovers candidates for both 12345 and 99999
            with patch.object(ald, "analyze_single_target", side_effect=mock_cand_multi):
                rc1, sum1 = stc.run_search(args_run)
            self.assertEqual(rc1, 0)
            self.assertEqual(sum1["candidates_count"], 2)

            # Inject an unrelated prior run's candidate into cumulative report
            cum_path = Path(root) / "toi_companions_summary.csv"
            df_cum = pd.read_csv(cum_path)
            prior_cand = {col: "" for col in stc.CANDIDATE_COLUMNS}
            prior_cand.update({"run_id": "run_prior", "tic_id": "88888", "sector": 106, "period_days": 4.0, "sde": 9.0})
            df_cum = pd.concat([df_cum, pd.DataFrame([prior_cand])], ignore_index=True)
            df_cum.to_csv(cum_path, index=False)

            # Delete plot ONLY for 12345 (99999 remains intact)
            plot_12345 = Path(root) / "candidate_TIC_12345_s106_test_preserve.png"
            plot_12345.unlink()

            # Retry: 12345 re-analyzed and returns REJECT; inject failure on per-run CSV invalidation
            original_remove = stc._remove_candidate_row
            def fail_remove(path, *a, **kw):
                if Path(path).name == "toi_companions_summary_test_preserve.csv":
                    raise OSError("injected interrupted per-run invalidation")
                return original_remove(path, *a, **kw)

            with patch.object(ald, "analyze_single_target", return_value=("12345", "REJECT", "Rejected", None)), \
                 patch.object(stc, "_remove_candidate_row", side_effect=fail_remove):
                with self.assertRaises(OSError):
                    stc.run_search(args_run)

            # On resume: 12345 is purged, but 99999 and run_prior (88888) MUST be preserved!
            rc_res, sum_res = stc.run_search(args_run)
            self.assertEqual(rc_res, 0)
            self.assertEqual(sum_res["candidates_count"], 1)

            # Per-run CSV contains only 99999
            df_per_run = pd.read_csv(Path(root) / sum_res["summary_csv"])
            self.assertEqual(len(df_per_run), 1)
            self.assertEqual(str(df_per_run["tic_id"].iloc[0]), "99999")

            # Cumulative CSV contains 99999 AND run_prior's 88888
            df_cum_final = pd.read_csv(cum_path)
            self.assertEqual(len(df_cum_final), 2)
            self.assertSetEqual(set(df_cum_final["tic_id"].astype(str)), {"99999", "88888"})


if __name__ == "__main__":
    unittest.main()
