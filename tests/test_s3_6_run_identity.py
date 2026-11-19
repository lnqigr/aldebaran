#!/usr/bin/env python3
"""
Targeted offline tests for S3.6 (OPS-13, OPS-14, OPS-33, interruption-safe bound resume).
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import aldebaran as ald
import merge_shards as ms
import run_identity as ri
import search_toi_companions as stc
import validate_triceratops as vt


class TestRunIdentityCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ald_s3_6_")
        self.code_file = os.path.join(self.tmp, "dummy_code.py")
        self.cfg_file = os.path.join(self.tmp, "prereg.md")
        self.in_file = os.path.join(self.tmp, "inputs.csv")
        with open(self.code_file, "w", encoding="utf-8") as f:
            f.write("VALUE = 1\n")
        with open(self.cfg_file, "w", encoding="utf-8") as f:
            f.write("# Preregistration v1\n")
        with open(self.in_file, "w", encoding="utf-8") as f:
            f.write("tic_id\n12345\n")
        self.record_path = os.path.join(self.tmp, "run_identity_r1.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _open(self, resume=True, config=None, input_spec=None):
        return ri.BoundRun.open_or_resume(
            self.record_path,
            run_id="r1",
            runner="test_runner",
            code_files=[self.code_file],
            config=config or {"k": 1},
            preregistration_path=self.cfg_file,
            input_paths=[self.in_file],
            input_spec=input_spec or {"n": 1},
            repo_dir=self.tmp,
            resume=resume,
        )

    def test_ops13_frozen_environment_captured(self):
        br = self._open(resume=False)
        env = br.environment
        self.assertIn("python_version", env)
        self.assertIn("platform", env)
        self.assertIsInstance(env["pip_freeze"], list)
        self.assertGreater(len(env["pip_freeze"]), 0)
        self.assertIn("git_revision", env)
        self.assertIn("git_dirty", env)

    def test_ops14_bound_resume_and_three_hash_mismatches(self):
        br = self._open(resume=False)
        br.record_interruption("SIGTERM", completed_count=2)
        self.assertEqual(br.data["status"], "INTERRUPTED")
        self.assertEqual(len(br.data["interruptions"]), 1)

        # Valid resume succeeds and logs resume
        br2 = self._open(resume=True)
        self.assertEqual(br2.data["status"], "IN_PROGRESS")
        self.assertEqual(len(br2.data["resumes"]), 1)

        # Fresh start (resume=False) refuses to overwrite existing record
        with self.assertRaisesRegex(ri.RunIdentityError, "RUN_RECORD_EXISTS"):
            self._open(resume=False)

        # 1. Code SHA256 mismatch stops
        with open(self.code_file, "w", encoding="utf-8") as f:
            f.write("VALUE = 2\n")
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_CODE_SHA256"):
            self._open(resume=True)
        with open(self.code_file, "w", encoding="utf-8") as f:
            f.write("VALUE = 1\n")

        # 2. Config / preregistration SHA256 mismatch stops
        with open(self.cfg_file, "w", encoding="utf-8") as f:
            f.write("# Preregistration v2\n")
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_CONFIG_SHA256"):
            self._open(resume=True)
        with open(self.cfg_file, "w", encoding="utf-8") as f:
            f.write("# Preregistration v1\n")

        # 3. Input SHA256 mismatch stops
        with open(self.in_file, "w", encoding="utf-8") as f:
            f.write("tic_id\n99999\n")
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_INPUT_SHA256"):
            self._open(resume=True)

    def test_truncated_or_partial_record_rejected(self):
        self._open(resume=False)
        raw = Path(self.record_path).read_text(encoding="utf-8")
        # Truncate JSON mid-file
        Path(self.record_path).write_text(raw[: len(raw) // 2], encoding="utf-8")
        with self.assertRaisesRegex(ri.RunIdentityError, "PARTIAL_OR_CORRUPT_RECORD"):
            self._open(resume=True)

        # Empty file
        Path(self.record_path).write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ri.RunIdentityError, "PARTIAL_OR_CORRUPT_RECORD"):
            self._open(resume=True)

        # Valid JSON but missing required keys
        Path(self.record_path).write_text('{"run_id": "r1"}', encoding="utf-8")
        with self.assertRaisesRegex(ri.RunIdentityError, "PARTIAL_OR_CORRUPT_RECORD"):
            self._open(resume=True)

    def test_interrupted_atomic_write_leaves_no_partial_record(self):
        target = os.path.join(self.tmp, "state.json")
        ri.atomic_write_json(target, {"ok": 1})
        before = Path(target).read_bytes()

        with mock.patch("run_identity.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                ri.atomic_write_json(target, {"ok": 2})

        self.assertEqual(Path(target).read_bytes(), before)
        leftovers = [n for n in os.listdir(self.tmp) if n.startswith(".tmp_")]
        self.assertEqual(leftovers, [])

    def test_ops33_mast_product_change_detected_on_resume(self):
        mast_fits = os.path.join(self.tmp, "tess_lc.fits")
        Path(mast_fits).write_bytes(b"FITS_HEADER_V1")
        br = self._open(resume=False)
        br.record_mast_product(mast_fits, tic_id="12345", sector=10, product_type="LC")
        br.record_interruption("SIGTERM", completed_count=0)

        # Valid resume with unchanged MAST file passes
        br2 = self._open(resume=True)
        self.assertIn(os.path.abspath(mast_fits), br2.data["mast_products"])

        # Mutate MAST product on disk -> detected on resume
        Path(mast_fits).write_bytes(b"FITS_HEADER_V2_MODIFIED")
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_MAST_PRODUCT"):
            self._open(resume=True)


class TestRunnerIntegrations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ald_s3_6_runners_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        ald.configure_environment(output_dir="output", run_id="run_default")

    def test_aldebaran_mast_tracking_and_bound_resume(self):
        out_dir = os.path.join(self.tmp, "ald_out")
        os.makedirs(out_dir, exist_ok=True)
        ald.configure_environment(output_dir=out_dir, run_id="ald_r1")
        fits_path = os.path.join(self.tmp, "spoc_s10.fits")
        Path(fits_path).write_bytes(b"SPOC_DATA_1")

        fake_lc = types.SimpleNamespace(filename=fits_path)
        fake_search = types.SimpleNamespace(download=lambda: fake_lc)
        res = ald.safe_download(fake_search, "12345", download_all=False)
        self.assertIs(res, fake_lc)

        br = ald.init_aldebaran_bound_run(out_dir, "ald_r1", 10, ["12345", "67890"])
        ald.sync_aldebaran_mast_products(br, out_dir, "ald_r1")
        self.assertIn(os.path.abspath(fits_path), br.data["mast_products"])

        # Resume with identical inputs works
        br2 = ald.init_aldebaran_bound_run(out_dir, "ald_r1", 10, ["12345", "67890"])
        self.assertEqual(len(br2.data["resumes"]), 1)

        # Resume with changed sector stops
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_INPUT_SHA256"):
            ald.init_aldebaran_bound_run(out_dir, "ald_r1", 11, ["12345", "67890"])

    def test_search_toi_companions_bound_resume_and_environment(self):
        out_dir = os.path.join(self.tmp, "toi_out")
        ephem = {"12345": [{"toi": "1.01", "period": 3.0, "t0_btjd": 1000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]}
        args1 = stc.parse_args(["--sectors", "106", "--workers", "1", "--output-dir", out_dir, "--run-id", "r_toi", "--append"])
        with mock.patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ephem)), \
             mock.patch.object(stc, "query_sector_toi_hosts", return_value=(["12345"], None)), \
             mock.patch.object(stc, "process_toi_host", return_value={
                 "run_id": "r_toi", "timestamp": "2026-11-13T00:00:00Z", "sector": 106,
                 "tic_id": "12345", "status": "REJECT", "reason": "REJECT",
                 "message": "rejected", "error_detail": "", "candidate": None, "plot_path": "",
             }):
            rc, summary = stc.run_search(args1)
        self.assertEqual(rc, 0)
        self.assertIn("environment", summary)
        self.assertIn("identity", summary)

        # Mismatched config on resume stops
        args_bad = stc.parse_args(["--sectors", "105", "--workers", "1", "--output-dir", out_dir, "--run-id", "r_toi", "--append"])
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_CONFIG_SHA256|STALE_INPUT_SHA256"):
            stc.run_search(args_bad)

    def test_validate_triceratops_interrupt_and_bound_resume(self):
        res_dir = os.path.join(self.tmp, "tri_res")
        os.makedirs(res_dir, exist_ok=True)
        top_csv = os.path.join(res_dir, "exofop_ctoi_upload_top3.csv")
        pd.DataFrame([
            {"TIC": 111, "Period": 2.5, "Epoch": 2459000.0, "Depth": 500.0, "Duration": 2.0, "sectors": "[10]"},
            {"TIC": 222, "Period": 4.5, "Epoch": 2459001.0, "Depth": 800.0, "Duration": 3.0, "sectors": "[10]"},
        ]).to_csv(top_csv, index=False)

        calls = []

        def fake_val(**kwargs):
            tid = kwargs["tic_id"]
            calls.append(tid)
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated kill")
            out_sub = os.path.join(res_dir, "triceratops")
            os.makedirs(out_sub, exist_ok=True)
            pd.DataFrame([{"scenario": "TP", "prob": 0.99}]).to_csv(
                os.path.join(out_sub, f"triceratops_probs_TIC_{tid}.csv")
            )
            return {"tic_id": tid, "fpp_mean": 0.01, "nfpp_mean": 0.0, "disposition": "PASS"}

        with mock.patch.object(vt, "run_triceratops_validation", side_effect=fake_val):
            with self.assertRaises(KeyboardInterrupt):
                vt.validate_auto(n_mc=2, n_bins=20, top_csv=top_csv, results_dir=res_dir, resume=False)

        # Resume skips TIC 111 and runs TIC 222
        calls.clear()
        with mock.patch.object(vt, "run_triceratops_validation", side_effect=fake_val):
            vt.validate_auto(n_mc=2, n_bins=20, top_csv=top_csv, results_dir=res_dir, resume=True)
        self.assertEqual(calls, [222])

        # Changing config on resume stops
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_CONFIG_SHA256"):
            vt.validate_auto(n_mc=9, n_bins=20, top_csv=top_csv, results_dir=res_dir, resume=True)

    def test_merge_shards_identity_environment_and_resume(self):
        in_dir = os.path.join(self.tmp, "shards")
        out_dir = os.path.join(self.tmp, "merged")
        os.makedirs(in_dir, exist_ok=True)
        cand_csv = os.path.join(in_dir, "cands_s10.csv")
        hist_txt = os.path.join(in_dir, "scanned_s10.txt")
        pd.DataFrame(columns=["tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm", "sde"]).to_csv(cand_csv, index=False)
        Path(hist_txt).write_text("12345\n", encoding="utf-8")

        manifest_path = os.path.join(in_dir, "manifest.json")
        manifest = {
            "run_id": "m_run1",
            "survey_mode": "complete",
            "shards": [{
                "shard_id": 0,
                "sector": 10,
                "source_run_id": "m_run1_s10",
                "status": "EMPTY_NO_CANDIDATES",
                "candidate_artifact": {
                    "path": "cands_s10.csv",
                    "status": "EMPTY_NO_CANDIDATES",
                    "sha256": ms.compute_file_sha256(cand_csv),
                },
                "history_artifact": {
                    "path": "scanned_s10.txt",
                    "status": "COMPLETED",
                    "sha256": ms.compute_file_sha256(hist_txt),
                },
            }],
        }
        ri.atomic_write_json(manifest_path, manifest)

        rc = ms.run_merge_shards(manifest_path=manifest_path, output_dir=out_dir, run_id="m_run1")
        self.assertEqual(rc, ms.EXIT_SUCCESS)
        out_man = json.loads(Path(os.path.join(out_dir, "manifest.json")).read_text(encoding="utf-8"))
        self.assertIn("environment", out_man)
        self.assertIn("identity", out_man)

        # Resume with matching identity succeeds
        rc2 = ms.run_merge_shards(manifest_path=manifest_path, output_dir=out_dir, run_id="m_run1", resume=True)
        self.assertEqual(rc2, ms.EXIT_SUCCESS)

        # Resume with changed config stops
        rc3 = ms.run_merge_shards(
            manifest_path=manifest_path, output_dir=out_dir, run_id="m_run1", allow_partial=True, resume=True
        )
        self.assertEqual(rc3, ms.EXIT_VALIDATION_ERROR)

    def test_item_record_rejects_missing_payload_and_missing_artifact(self):
        code_file = os.path.join(self.tmp, "runner.py")
        Path(code_file).write_text("x = 1\n", encoding="utf-8")
        rec_path = os.path.join(self.tmp, "run_identity_item.json")
        run = ri.BoundRun.open_or_resume(
            rec_path,
            run_id="item_r1",
            runner="validate_triceratops",
            code_files=[code_file],
            config={"n_mc": 5},
        )
        csv_path = os.path.join(self.tmp, "probs.csv")
        Path(csv_path).write_text("scenario,prob\nTP,0.9\n", encoding="utf-8")
        item_path = os.path.join(self.tmp, "item_TIC_1.json")
        run.save_item_record(item_path, {})
        with self.assertRaises(ri.RunIdentityError):
            run.load_verified_item_record(
                item_path,
                required_keys=("run_id", "identity", "tic_id", "result", "csv_path", "csv_sha256"),
                artifact_fields={"csv_path": "csv_sha256"},
            )

        run.save_item_record(
            item_path,
            {
                "tic_id": 1,
                "result": {"fpp": 0.01},
                "csv_path": csv_path,
                "csv_sha256": ri.sha256_file(csv_path),
            },
        )
        loaded = run.load_verified_item_record(
            item_path,
            required_keys=("run_id", "identity", "tic_id", "result", "csv_path", "csv_sha256"),
            artifact_fields={"csv_path": "csv_sha256"},
        )
        self.assertEqual(loaded["tic_id"], 1)

        # Tampering artifact bytes must be rejected
        Path(csv_path).write_text("scenario,prob\nTP,0.1\n", encoding="utf-8")
        with self.assertRaises(ri.RunIdentityError):
            run.load_verified_item_record(
                item_path,
                required_keys=("run_id", "identity", "tic_id", "result", "csv_path", "csv_sha256"),
                artifact_fields={"csv_path": "csv_sha256"},
            )

        os.remove(csv_path)
        with self.assertRaises(ri.RunIdentityError):
            run.load_verified_item_record(
                item_path,
                required_keys=("run_id", "identity", "tic_id", "result", "csv_path", "csv_sha256"),
                artifact_fields={"csv_path": "csv_sha256"},
            )

    def test_input_sha256_distinguishes_same_named_files_and_rejects_missing(self):
        dir_a = os.path.join(self.tmp, "a")
        dir_b = os.path.join(self.tmp, "b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        fa = os.path.join(dir_a, "input.csv")
        fb = os.path.join(dir_b, "input.csv")
        Path(fa).write_text("tic_id\n100\n", encoding="utf-8")
        Path(fb).write_text("tic_id\n200\n", encoding="utf-8")

        h1 = ri.compute_input_sha256([fa, fb])
        Path(fa).write_text("tic_id\n999\n", encoding="utf-8")
        h2 = ri.compute_input_sha256([fa, fb])
        self.assertNotEqual(h1, h2)

        with self.assertRaises(ri.RunIdentityError):
            ri.compute_input_sha256([os.path.join(self.tmp, "does_not_exist.csv")])

    def test_aldebaran_sync_rejects_tampered_mast_product_after_sidecar(self):
        import aldebaran as ald
        out_dir = os.path.join(self.tmp, "ald_out")
        os.makedirs(out_dir, exist_ok=True)
        fits_path = os.path.join(self.tmp, "cached_lc.fits")
        Path(fits_path).write_bytes(b"ORIGINAL_FITS")

        old_out, old_run_id = ald.OUTPUT_DIR, ald.CURRENT_RUN_ID
        try:
            ald.OUTPUT_DIR = out_dir
            ald.CURRENT_RUN_ID = "ald_mast_test"
            bound_run = ald.init_aldebaran_bound_run(out_dir, "ald_mast_test", 14, [12345])

            class FakeLK:
                filename = fits_path

            ald._record_mast_product_evidence(FakeLK(), "12345")
            Path(fits_path).write_bytes(b"TAMPERED_FITS")
            with self.assertRaises(ri.RunIdentityError):
                ald.sync_aldebaran_mast_products(bound_run, out_dir, "ald_mast_test")
        finally:
            ald.OUTPUT_DIR = old_out
            ald.CURRENT_RUN_ID = old_run_id

    def test_aldebaran_rejects_unbound_historical_reuse(self):
        import aldebaran as ald
        out_dir = os.path.join(self.tmp, "ald_hist")
        os.makedirs(out_dir, exist_ok=True)
        with self.assertRaises(ri.RunIdentityError):
            ald.init_aldebaran_bound_run(out_dir, "missing_id", 14, [12345], has_prior_outputs=True)

    def test_search_toi_companions_rejects_unbound_historical_reuse(self):
        import types
        import search_toi_companions as stc
        out_dir = os.path.join(self.tmp, "toi_out")
        os.makedirs(out_dir, exist_ok=True)
        audit_csv = os.path.join(out_dir, "toi_companions_audit_toi_r1.csv")
        Path(audit_csv).write_text("run_id,tic_id,sector,status\ntoi_r1,123,14,REJECT\n", encoding="utf-8")
        args = types.SimpleNamespace(
            run_id="toi_r1",
            output_dir=out_dir,
            append=True,
            sectors=[14],
            workers=1,
            limit_per_sector=1,
            target_file=None,
        )
        with self.assertRaises(ri.RunIdentityError):
            stc.run_search(args)

    def test_merge_shards_propagates_resume_to_downstream_stages(self):
        import pandas as pd
        import subprocess as sp
        from unittest import mock
        import merge_shards as ms

        out_dir = os.path.join(self.tmp, "merge_out")
        vet_dir = os.path.join(out_dir, "deep_vetting")
        os.makedirs(vet_dir, exist_ok=True)
        Path(os.path.join(vet_dir, "run_identity_pleiades.json")).write_text("{}", encoding="utf-8")
        df_cons = pd.DataFrame([{"tic_id": "12345"}])
        Path(os.path.join(vet_dir, "scientific_dossier_summary.csv")).write_text(
            "tic_id,operational_status,disposition,error_message\n12345,SUCCESS,PASS,\n",
            encoding="utf-8",
        )

        seen_cmds = []

        def fake_run(cmd, capture_output=True, text=True):
            seen_cmds.append(list(cmd))
            return sp.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with mock.patch.object(ms.subprocess, "run", side_effect=fake_run):
            ms.orchestrate_downstream_stages(
                df_consolidated=df_cons,
                output_dir=out_dir,
                run_pleiades=True,
                resume=True,
            )
        self.assertEqual(len(seen_cmds), 1)
        self.assertIn("--resume", seen_cmds[0])

    def test_pleiades_binds_scientific_dependencies_and_mast_products(self):
        import pleiades as pl
        vet_dir = os.path.join(self.tmp, "pl_vet")
        os.makedirs(vet_dir, exist_ok=True)
        fits_path = os.path.join(self.tmp, "pl_spoc.fits")
        Path(fits_path).write_bytes(b"SPOC_LC_BYTES")

        class FakeLK:
            filename = fits_path

        here_dir = os.path.dirname(os.path.abspath(ri.__file__))
        in_csv = os.path.join(self.tmp, "cands.csv")
        Path(in_csv).write_text("tic_id\n12345\n", encoding="utf-8")
        bound_run = ri.BoundRun.open_or_resume(
            os.path.join(vet_dir, "run_identity_pleiades.json"),
            run_id="pleiades_cli",
            runner="pleiades",
            code_files=ri.resolve_code_files(here_dir, ("pleiades.py", "vps_contract.py", "ephemeris_contract.py")),
            config={"pipeline_version": pl.PIPELINE_VERSION},
            input_paths=[in_csv],
            repo_dir=here_dir,
        )
        self.assertIn("vps_contract.py", bound_run.data["identity"]["code_sha256"])
        pl._record_pleiades_mast_product(FakeLK(), "12345", vet_dir)
        pl.sync_pleiades_mast_products(bound_run, vet_dir)
        self.assertIn(os.path.abspath(fits_path), bound_run.data["mast_products"])

        Path(fits_path).write_bytes(b"TAMPERED_SPOC")
        with self.assertRaises(ri.RunIdentityError):
            pl.sync_pleiades_mast_products(bound_run, vet_dir)

    def test_pleiades_rejects_missing_dossier_metadata_on_success_resume(self):
        import subprocess as sp
        import pleiades as pl

        vet_dir = os.path.join(self.tmp, "pl_missing_meta")
        os.makedirs(vet_dir, exist_ok=True)
        in_csv = os.path.join(self.tmp, "pl_in.csv")
        pd.DataFrame([{"tic_id": "12345", "period_days": 3.0, "t0_btjd": 1500.0, "depth_ppm": 1000.0, "duration_hours": 2.0}]).to_csv(in_csv, index=False)
        here_dir = os.path.dirname(os.path.abspath(ri.__file__))
        bound_run = ri.BoundRun.open_or_resume(
            os.path.join(vet_dir, "run_identity_pleiades.json"),
            run_id="pleiades_cli",
            runner="pleiades",
            code_files=ri.discover_local_code_dependencies(here_dir, ("pleiades.py",)),
            config={"pipeline_version": pl.PIPELINE_VERSION},
            input_paths=[in_csv],
            repo_dir=here_dir,
        )
        rec = dict.fromkeys(pl.PLEIADES_SUMMARY_COLS, 0.0)
        rec.update({"tic_id": "12345", "operational_status": "SUCCESS", "disposition": "CANDIDATE", "error_message": ""})
        item_path = os.path.join(vet_dir, "pleiades_record_TIC_12345.json")
        # Deliberately omit has_dossier / dossier_path / dossier_sha256
        bound_run.save_item_record(item_path, {"tic_id": "12345", "record": rec})

        proc = sp.run(
            [sys.executable, "-B", os.path.join(here_dir, "pleiades.py"), "--input-csv", in_csv, "--output-dir", vet_dir, "--resume"],
            cwd=here_dir,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PARTIAL_OR_CORRUPT_RECORD", proc.stderr + proc.stdout)

    def test_merge_shards_binds_antares_stellar_metrics_dependency(self):
        here_dir = os.path.dirname(os.path.abspath(ri.__file__))
        deps = ri.discover_local_code_dependencies(here_dir, ("merge_shards.py", "antares_packager.py"))
        basenames = {os.path.basename(p) for p in deps}
        self.assertIn("stellar_metrics.py", basenames)
        self.assertIn("depth_contract.py", basenames)
        self.assertIn("vps_contract.py", basenames)

    def test_validate_triceratops_persists_mast_before_item_on_hard_kill(self):
        res_dir = os.path.join(self.tmp, "tri_hard_kill")
        os.makedirs(res_dir, exist_ok=True)
        top_csv = os.path.join(res_dir, "exofop_ctoi_upload_top3.csv")
        pd.DataFrame([
            {"TIC": 111, "Period": 2.5, "Epoch": 2459000.0, "Depth": 500.0, "Duration": 2.0, "sectors": "[10]"},
        ]).to_csv(top_csv, index=False)

        fits_path = os.path.join(self.tmp, "tri_spoc.fits")
        Path(fits_path).write_bytes(b"ORIGINAL_TRI_FITS")

        class FakeLK:
            filename = fits_path

        def fake_val(**kwargs):
            tid = kwargs["tic_id"]
            if vt._ACTIVE_BOUND_RUN is not None:
                vt._ACTIVE_BOUND_RUN.record_mast_products_from_lk(
                    [FakeLK()], tic_id=tid, sector=10, product_type="SPOC_SAP_TPF", save=False
                )
            out_sub = os.path.join(res_dir, "triceratops")
            os.makedirs(out_sub, exist_ok=True)
            pd.DataFrame([{"scenario": "TP", "prob": 0.99}]).to_csv(
                os.path.join(out_sub, f"triceratops_probs_TIC_{tid}.csv")
            )
            return {"tic_id": tid, "fpp": 0.01}

        class HardKill(BaseException):
            pass

        orig_save_item = ri.BoundRun.save_item_record

        def kill_after_item(self_run, item_path, payload):
            orig_save_item(self_run, item_path, payload)
            raise HardKill("simulated SIGKILL after item publish")

        with mock.patch.object(vt, "run_triceratops_validation", side_effect=fake_val):
            with mock.patch.object(ri.BoundRun, "save_item_record", side_effect=kill_after_item, autospec=True):
                with self.assertRaises(HardKill):
                    vt.validate_auto(n_mc=2, n_bins=20, top_csv=top_csv, results_dir=res_dir, resume=False)

        # Persisted run record MUST already contain mast_products before item was published
        id_data = json.loads(Path(os.path.join(res_dir, "triceratops", "run_identity_triceratops.json")).read_text(encoding="utf-8"))
        self.assertIn(os.path.abspath(fits_path), id_data["mast_products"])

        # Tampering product bytes must fail resume
        Path(fits_path).write_bytes(b"TAMPERED_TRI_FITS")
        with mock.patch.object(vt, "run_triceratops_validation", side_effect=fake_val):
            with self.assertRaisesRegex(ri.RunIdentityError, "STALE_MAST_PRODUCT"):
                vt.validate_auto(n_mc=2, n_bins=20, top_csv=top_csv, results_dir=res_dir, resume=True)


if __name__ == "__main__":
    unittest.main()
