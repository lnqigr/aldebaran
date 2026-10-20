"""
Unit and integration tests for decision_ledger.py.

Covers:
- Context pickling and identity immutability across spawn boundary
- Strict JSON serialization with nonfinite handling
- Full lifecycle of SubmissionLedger and RunLedger
- Validator negative controls (A5): deleted gate, deleted final, duplicate ID,
  wrong ID, wrong TIC/sector, wrong unit, stale threshold, stale source hash,
  skipped->PASS mutation, path escape, session collision.
- Fault isolation controls (A6): observer write failure preserves science,
  original archival writer failure propagates ERROR, both fail simultaneously.
- Real ProcessPoolExecutor with spawn boundary controls (A4):
  multiple distinct submissions with repeated TIC/sector, worker exit after partial
  evidence, worker exit before first record.
"""

import os
import sys
import ast
import json
import math
import uuid
import time
import pickle
import shutil
import tempfile
import unittest
import multiprocessing
from contextlib import ExitStack
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import decision_ledger as dl
from decision_ledger import (
    SCHEMA_VERSION,
    COVERAGE_MANIFEST,
    ObserverContext,
    SubmissionLedger,
    RunLedger,
    ValidationResult,
    validate_run_ledger,
    sanitize_for_strict_json,
    strict_json_dumps,
    get_module_source_hash,
)
import aldebaran


class TestContextAndSerialization(unittest.TestCase):
    def test_context_pickling_and_identity(self):
        ctx = ObserverContext(
            observer_dir="C:/tmp/obs",
            observer_run_id="obs_run_1234",
            submission_id="sub_tic123_s1_001"
        )
        # Verify immutability
        with self.assertRaises(Exception):
            ctx.submission_id = "other"

        # Verify pickling across process boundaries
        pickled = pickle.dumps(ctx)
        unpickled = pickle.loads(pickled)
        self.assertEqual(ctx, unpickled)
        self.assertEqual(unpickled.observer_dir, "C:/tmp/obs")
        self.assertEqual(unpickled.observer_run_id, "obs_run_1234")
        self.assertEqual(unpickled.submission_id, "sub_tic123_s1_001")

        # from_any conversions
        self.assertIsNone(ObserverContext.from_any(None))
        from_tup = ObserverContext.from_any(("d", "r", "s"))
        self.assertEqual(from_tup.submission_id, "s")
        from_dict = ObserverContext.from_any({"observer_dir": "d", "observer_run_id": "r", "submission_id": "s"})
        self.assertEqual(from_dict.observer_run_id, "r")

    def test_strict_json_serialization_nonfinites(self):
        sample = {
            "normal_float": 1.2345,
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "neginf_val": float("-inf"),
            "nested": {
                "nan_in_nested": float("nan"),
                "clean_val": 42
            },
            "list_with_nan": [1.0, float("nan"), 3.0]
        }
        serialized = strict_json_dumps(sample)
        # Strict JSON must not contain raw NaN or Infinity literals
        self.assertNotIn("NaN", serialized)
        self.assertNotIn("Infinity", serialized)

        parsed = json.loads(serialized)
        self.assertIsNone(parsed["nan_val"])
        self.assertEqual(parsed["nan_val_availability"], "NONFINITE")
        self.assertIsNone(parsed["inf_val"])
        self.assertIsNone(parsed["nested"]["nan_in_nested"])
        self.assertEqual(parsed["normal_float"], 1.2345)


class TestLedgerLifecycleAndValidation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_ledger_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_clean_run(self) -> tuple[RunLedger, str, str]:
        run_ledger = RunLedger(observer_dir=self.test_dir, legacy_run_id="test_run")
        ctx = run_ledger.create_submission_context(tic_id="12345678", sector=1)
        sub_ledger = SubmissionLedger(ctx, tic_id="12345678", sector=1)

        sub_ledger.record_submission_begin()
        attempt_id = "att_12345678_s1_test01"
        sub_ledger.bind_attempt_id(attempt_id)
        sub_ledger.record_spoc_search(num_products=1, selected_product_info={"target": "TIC 12345678"})
        sub_ledger.record_crowding_gate(crowdsap=0.92, passed=True)
        sub_ledger.record_baseline_gate(span=27.4, min_period=0.8, max_period=13.0, max_p=13.0, passed=True)
        sub_ledger.record_bls_start(
            n_freq=9000, min_freq=0.0769, max_freq=1.25,
            min_dur=0.045, max_dur=0.18, n_dur=12
        )
        sub_ledger.record_bls_search(
            n_freq=9000, min_freq=0.0769, max_freq=1.25,
            min_dur=0.045, max_dur=0.18, n_dur=12,
            window_pts=501, top_peak_indices=[100, 200, 300]
        )
        sub_ledger.record_low_sde_gate(peak_idx=100, rank_idx=1, sde=8.5, passed=True)
        sub_ledger.record_target_result(
            status_code="CANDIDATE",
            message="Candidate found",
            has_candidate=True,
            cand_data={"period_days": 3.0, "t0_btjd": 1500.0},
        )
        sub_ledger.record_submission_final(status="COMPLETE")

        run_ledger.record_submission_outcome(
            ctx.submission_id,
            future_result=("12345678", "CANDIDATE", "Candidate found", {"cand": 1})
        )
        run_ledger.finalize_run(completed_count=1, error_count=0, candidate_count=1)

        return run_ledger, ctx.submission_id, sub_ledger.events_path

    def test_clean_run_passes_validation(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        res = validate_run_ledger(run_ledger.run_dir, expected_source_hash=run_ledger.source_hash)
        self.assertTrue(res.valid, f"Validation failed with errors: {res.errors}")
        self.assertEqual(res.summary["submissions_audited"], 1)
        self.assertGreater(res.summary["events_audited"], 5)
        self.assertEqual(len(res.errors), 0)

    def test_worker_terminal_result_matches_parent_candidate_outcome(self):
        run_ledger, _, _ = self._create_clean_run()
        result = validate_run_ledger(run_ledger.run_dir)
        self.assertTrue(result.valid, f"Validation failed with errors: {result.errors}")

    def test_negative_control_worker_verdict_must_match_parent_and_manifest(self):
        run_ledger, _, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        target_results = [
            event for event in events
            if event.get("record_kind") == "target_result"
        ]
        self.assertEqual(len(target_results), 1)
        target_results[0]["status_code"] = "REJECT_CROWDSAP"
        target_results[0]["has_candidate"] = False
        with open(events_path, "w", encoding="utf-8") as f:
            for event in events:
                f.write(strict_json_dumps(event) + "\n")

        result = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("does not match parent terminal_code 'CANDIDATE'" in error for error in result.errors),
            result.errors,
        )

    def test_negative_control_run_completed_count_must_match_outcomes(self):
        run_ledger, _, _ = self._create_clean_run()
        with open(run_ledger.run_log_path, "r", encoding="utf-8") as f:
            run_events = [json.loads(line) for line in f if line.strip()]
        for event in run_events:
            if event.get("record_kind") == "run" and event.get("event_name") == "FINAL":
                event["completed_count"] = 99
        with open(run_ledger.run_log_path, "w", encoding="utf-8") as f:
            for event in run_events:
                f.write(strict_json_dumps(event) + "\n")

        result = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(result.valid)
        self.assertTrue(any("completed_count mismatch" in error for error in result.errors))

    def test_negative_control_bls_start_requires_terminal_event(self):
        run_ledger, _, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        events = [
            event for event in events
            if not (event.get("record_kind") == "search" and event.get("event_name") == "END")
        ]
        with open(events_path, "w", encoding="utf-8") as f:
            for event in events:
                f.write(strict_json_dumps(event) + "\n")

        result = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(result.valid)
        self.assertTrue(any("exactly one END or ERROR terminal event" in error for error in result.errors))

    def test_negative_control_expected_source_hash_cannot_be_omitted(self):
        run_ledger, _, events_path = self._create_clean_run()
        with open(run_ledger.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.pop("source_hash", None)
        with open(run_ledger.manifest_path, "w", encoding="utf-8") as f:
            f.write(strict_json_dumps(manifest) + "\n")

        for path in (run_ledger.run_log_path, events_path):
            with open(path, "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            for event in events:
                event.pop("source_hash", None)
                source_ref = event.get("source_ref")
                if isinstance(source_ref, dict):
                    source_ref.pop("source_hash", None)
            with open(path, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(strict_json_dumps(event) + "\n")

        result = validate_run_ledger(
            run_ledger.run_dir,
            expected_source_hash=run_ledger.source_hash,
        )
        self.assertFalse(result.valid)
        self.assertTrue(
            any("missing required source hash" in error.lower() for error in result.errors),
            result.errors,
        )

    # A5 Negative Controls
    def test_negative_control_deleted_gate(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Drop the crowding gate event (index 3)
        corrupted_lines = [l for l in lines if "gate_crowding" not in l]
        with open(events_path, "w", encoding="utf-8") as f:
            f.writelines(corrupted_lines)

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("sequence mismatch" in e for e in res.errors))

    def test_negative_control_deleted_submission_final(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Drop last line (submission FINAL)
        corrupted_lines = lines[:-1]
        with open(events_path, "w", encoding="utf-8") as f:
            f.writelines(corrupted_lines)

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("unfinalized submission" in e for e in res.errors))

    def test_negative_control_deleted_run_final(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        run_log = run_ledger.run_log_path
        with open(run_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        corrupted_lines = [l for l in lines if '"event_name":"FINAL"' not in l]
        with open(run_log, "w", encoding="utf-8") as f:
            f.writelines(corrupted_lines)

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("run.jsonl missing run FINAL" in e for e in res.errors))

    def test_negative_control_duplicate_event_id(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Duplicate event_id on event 2
        lines[2]["event_id"] = lines[1]["event_id"]
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("Duplicate event_id" in e for e in res.errors))

    def test_negative_control_wrong_submission_or_attempt_id(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Tamper submission_id on gate_baseline
        for d in lines:
            if d.get("gate_id") == "gate_baseline":
                d["submission_id"] = "sub_hacked"
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("submission_id mismatch" in e for e in res.errors))

    def test_negative_control_wrong_tic_or_sector(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Tamper sector on gate_crowding
        for d in lines:
            if d.get("gate_id") == "gate_crowding":
                d["sector"] = 99
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("sector mismatch" in e for e in res.errors))

    def test_negative_control_wrong_unit(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Tamper unit on gate_crowding
        for d in lines:
            if d.get("gate_id") == "gate_crowding":
                d["thresholds"][0]["unit"] = "hours"
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("gate_crowding unit mismatch" in e for e in res.errors))

    def test_negative_control_stale_threshold(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Tamper threshold value on gate_crowding from 0.70 to 0.65
        for d in lines:
            if d.get("gate_id") == "gate_crowding":
                d["thresholds"][0]["value"] = 0.65
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("gate_crowding threshold altered" in e for e in res.errors))

    def test_negative_control_stale_source_hash(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        # Validate against mismatched source hash
        res = validate_run_ledger(run_ledger.run_dir, expected_source_hash="bad_hash_000000000000")
        self.assertFalse(res.valid)
        self.assertTrue(any("Stale source hash" in e for e in res.errors))

    def test_negative_control_skipped_to_pass_mutation(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Invert gate_crowding: set crowdsap to 0.50 but claim diagnostic_outcome="PASS"
        for d in lines:
            if d.get("gate_id") == "gate_crowding":
                d["evidence"]["crowdsap"]["value"] = 0.50
                d["diagnostic_outcome"] = "PASS"
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("falsely marked PASS" in e for e in res.errors))

    def test_negative_control_path_escape(self):
        run_ledger, sub_id, events_path = self._create_clean_run()
        # Add rogue submission with path traversal
        run_ledger.submissions["../../escape_attempt"] = {
            "submission_id": "../../escape_attempt",
            "tic_id": "999",
            "sector": 1,
            "events_path": "../../escape_attempt/events.jsonl",
            "status": "DISPATCHED"
        }
        run_ledger._update_manifest()

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("Path escape detected" in e for e in res.errors))

    def test_negative_control_deleted_gate_with_renumbered_sequence(self):
        """Counterexample 1: gate_crowding deleted and sequence renumbered."""
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # Remove gate_crowding
        filtered = [d for d in lines if d.get("gate_id") != "gate_crowding"]
        # Renumber sequence and event_id so there is no gap in sequence numbers
        for idx, d in enumerate(filtered, 1):
            d["sequence"] = idx
            d["event_id"] = f"evt_{sub_id}_{idx}"
        with open(events_path, "w", encoding="utf-8") as f:
            for d in filtered:
                f.write(strict_json_dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("Branch required gate 'gate_crowding' is missing" in e for e in res.errors))

    def test_negative_control_stripped_manifest_submissions(self):
        """Counterexample 2: manifest submissions cleared while run FINAL claims completed count."""
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(run_ledger.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["submissions"] = {}
        with open(run_ledger.manifest_path, "w", encoding="utf-8") as f:
            f.write(strict_json_dumps(manifest) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("Manifest claims 0 submissions but run.jsonl" in e for e in res.errors))

    def test_negative_control_missing_provenance(self):
        """Negative control: missing provenance or source_ref fails closed."""
        run_ledger, sub_id, events_path = self._create_clean_run()
        with open(events_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for d in lines:
            if d.get("gate_id") == "gate_crowding":
                del d["provenance"]
                del d["source_ref"]
        with open(events_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(strict_json_dumps(d) + "\n")

        res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(res.valid)
        self.assertTrue(any("missing required source_ref provenance" in e or "missing valid provenance" in e for e in res.errors))

    def test_negative_control_sibling_prefix_path_escape(self):
        """Negative control: sibling prefix matches (/tmp/obs vs /tmp/obs_sibling) fail closed."""
        parent = os.path.join(self.test_dir, "obs")
        sibling = os.path.join(self.test_dir, "obs_sibling")
        os.makedirs(parent, exist_ok=True)
        os.makedirs(sibling, exist_ok=True)

        self.assertFalse(dl.is_safely_contained(parent, sibling))
        self.assertFalse(dl.is_safely_contained(parent, os.path.join(sibling, "sub1")))

        ctx_escape = ObserverContext(
            observer_dir=parent,
            observer_run_id="../obs_sibling",
            submission_id="sub_escape"
        )
        ledger = SubmissionLedger(ctx_escape, "12345", 1)
        self.assertTrue(ledger.has_error)
        self.assertIn("Path escape detected", ledger.error_message)

    def test_negative_control_repeated_default_run_isolation(self):
        # Two consecutive default_run sessions must create distinct observer_run_ids
        run1 = RunLedger(observer_dir=self.test_dir, legacy_run_id="default_run")
        run2 = RunLedger(observer_dir=self.test_dir, legacy_run_id="default_run")
        self.assertNotEqual(run1.observer_run_id, run2.observer_run_id)
        self.assertNotEqual(run1.run_dir, run2.run_dir)
        self.assertTrue(os.path.exists(run1.manifest_path))
        self.assertTrue(os.path.exists(run2.manifest_path))


class TestFaultIsolationAndFailClosed(unittest.TestCase):
    """
    Acceptance criteria A6:
    - Observer-only serialization/write failure preserves baseline production result and invalidates audit.
    - ORIGINAL archival-writer failure produces original ERROR with observer off/on.
    - Both fail -> preserve original error plus incomplete observer.
    """
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_faults_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_observer_write_failure_preserves_science_result(self):
        # Real analyze_single_target call: observer write failure must not break science
        ctx = ObserverContext(
            observer_dir=self.test_dir,
            observer_run_id="obs_run_fault",
            submission_id="sub_fault_01"
        )
        orig_open = open
        def faulty_open(file, *args, **kwargs):
            if isinstance(file, str) and file.endswith("events.jsonl"):
                raise PermissionError("Observer disk full / permission denied")
            return orig_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=faulty_open), \
             patch("aldebaran.lk.search_lightcurve", return_value=[]):
            res = aldebaran.analyze_single_target("88888888", 1, observer_context=ctx)

        # Baseline science result returned without exception
        self.assertEqual(res[1], "SKIP_NO_DATA")
        self.assertEqual(res[0], "88888888")
        # Observer events file was never written
        events_path = os.path.join(self.test_dir, "obs_run_fault", "sub_fault_01", "events.jsonl")
        self.assertFalse(os.path.exists(events_path))

    def test_original_archival_writer_failure_propagates_error(self):
        # Real analyze_single_target call: archival diagnostic persistence failure bubbles to ERROR
        import numpy as np
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 1000)) * 0.0001
        ferr_arr = np.full(1000, 0.001)
        dx_arr = np.zeros(1000)
        dy_arr = np.zeros(1000)

        # Inject transit dips so depth, point-count, and shape gates pass
        transit_times = [1.0, 6.0, 11.0, 16.0, 21.0, 26.0]
        for tt in transit_times:
            in_tr = np.abs(t_arr - tt) < 0.03
            f_arr[in_tr] = 0.995

        fake_bls = MagicMock()
        fake_bls.power = np.full(100, 1.0)
        fake_bls.period = np.full(100, 5.0)
        fake_bls.transit_time = np.full(100, 1.0)
        fake_bls.duration = np.full(100, 0.1)
        sde_arr = np.full(100, 10.0)

        mock_star = {
            "r_star": 1.0,
            "e_r_star": 0.05,
            "m_star": 1.0,
            "e_m_star": 0.05,
            "teff": 5800,
            "ruwe": 1.0,
            "has_params": True
        }
        mock_arch = (True, "Archival_match_in_Sectors_1", {"archival_snr": 5.0, "multi_sde": 12.0})

        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]), \
             patch("aldebaran.BoxLeastSquares.power", return_value=fake_bls), \
             patch("aldebaran.compute_whitened_kovacs_sde", return_value=(sde_arr, None)), \
             patch("aldebaran.get_top_distinct_bls_peaks", return_value=[0]), \
             patch("aldebaran.get_stellar_and_gaia_params", return_value=mock_star), \
             patch("aldebaran.verify_and_filter_tpf", return_value=(True, "OK", None, None)), \
             patch("aldebaran.verify_multi_epoch", return_value=mock_arch), \
             patch("aldebaran.record_archival_diagnostic", side_effect=RuntimeError("Archival persistence failed")):

            res = aldebaran.analyze_single_target("88888888", 1, observer_context=None)

        self.assertEqual(res[1], "ERROR")
        self.assertIn("Archival persistence failed", res[2])

    def test_both_observer_and_archival_fail_preserves_original_error(self):
        # Real analyze_single_target call: both observer and archival fail simultaneously
        ctx = ObserverContext(
            observer_dir=self.test_dir,
            observer_run_id="obs_both_faults",
            submission_id="sub_both_01"
        )
        import numpy as np
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 1000)) * 0.0001
        ferr_arr = np.full(1000, 0.001)
        dx_arr = np.zeros(1000)
        dy_arr = np.zeros(1000)

        # Inject transit dips so depth, point-count, and shape gates pass
        transit_times = [1.0, 6.0, 11.0, 16.0, 21.0, 26.0]
        for tt in transit_times:
            in_tr = np.abs(t_arr - tt) < 0.03
            f_arr[in_tr] = 0.995

        fake_bls = MagicMock()
        fake_bls.power = np.full(100, 1.0)
        fake_bls.period = np.full(100, 5.0)
        fake_bls.transit_time = np.full(100, 1.0)
        fake_bls.duration = np.full(100, 0.1)
        sde_arr = np.full(100, 10.0)

        mock_star = {
            "r_star": 1.0,
            "e_r_star": 0.05,
            "m_star": 1.0,
            "e_m_star": 0.05,
            "teff": 5800,
            "ruwe": 1.0,
            "has_params": True
        }
        mock_arch = (True, "Archival_match_in_Sectors_1", {"archival_snr": 5.0, "multi_sde": 12.0})

        orig_open = open
        def faulty_open(file, *args, **kwargs):
            if isinstance(file, str) and file.endswith("events.jsonl"):
                raise PermissionError("Observer write failed")
            return orig_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=faulty_open), \
             patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]), \
             patch("aldebaran.BoxLeastSquares.power", return_value=fake_bls), \
             patch("aldebaran.compute_whitened_kovacs_sde", return_value=(sde_arr, None)), \
             patch("aldebaran.get_top_distinct_bls_peaks", return_value=[0]), \
             patch("aldebaran.get_stellar_and_gaia_params", return_value=mock_star), \
             patch("aldebaran.verify_and_filter_tpf", return_value=(True, "OK", None, None)), \
             patch("aldebaran.verify_multi_epoch", return_value=mock_arch), \
             patch("aldebaran.record_archival_diagnostic", side_effect=RuntimeError("Archival persistence failed")):

            res = aldebaran.analyze_single_target("99999999", 1, observer_context=ctx)

        # Original archival error is preserved and returned
        self.assertEqual(res[1], "ERROR")
        self.assertIn("Archival persistence failed", res[2])


class TestParentObserverFaultIsolation(unittest.TestCase):
    """Execute the actual __main__ parent orchestration with controlled collaborators."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_parent_faults_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _execute_parent(
        self,
        result=None,
        worker_error=None,
        context_error=None,
        outcome_error=None,
        prequeue_error=None,
    ):
        with open(aldebaran.__file__, "r", encoding="utf-8") as source_file:
            tree = ast.parse(source_file.read(), filename=aldebaran.__file__)
        main_guard = next(
            node for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and any(
                isinstance(comparator, ast.Constant) and comparator.value == "__main__"
                for comparator in node.test.comparators
            )
        )
        parent_code = compile(
            ast.fix_missing_locations(ast.Module(body=[main_guard], type_ignores=[])),
            aldebaran.__file__,
            "exec",
        )
        namespace = dict(aldebaran.__dict__)
        namespace["__name__"] = "__main__"
        report_path = os.path.join(self.test_dir, "candidates.csv")
        submitted = []
        outcome_calls = []
        executors = []
        self.submitted = submitted
        self.outcome_calls = outcome_calls

        def controlled_worker(*args, **kwargs):
            submitted.append((args, kwargs))
            if worker_error is not None:
                raise worker_error
            return result

        class ControlledExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.futures = []
                self.shutdown_called = False
                executors.append(self)

            def submit(self, function, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(function(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
                self.futures.append(future)
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_called = True

        def record_outcome(submission_id, future_result=None, exc=None):
            outcome_calls.append((submission_id, future_result, exc))
            raise outcome_error

        def select_targets(**kwargs):
            if prequeue_error is not None:
                raise prequeue_error
            return 1, ["12345"]

        overrides = {
            "ProcessPoolExecutor": ControlledExecutor,
            "as_completed": lambda futures: list(futures),
            "analyze_single_target": controlled_worker,
            "REPORT_FILE": report_path,
            "OUTPUT_DIR": self.test_dir,
            "MAX_WORKERS": 1,
            "MAX_TARGETS": 1,
            "PREFERRED_SECTORS": (1,),
            "CURRENT_RUN_ID": "parent_test",
            "select_latest_sector_and_targets": select_targets,
            "write_log": lambda *args, **kwargs: None,
            "write_event": lambda *args, **kwargs: None,
            "append_scanned_history": lambda *args, **kwargs: None,
        }

        with patch.dict(os.environ, {"ALDEBARAN_OBSERVER_DIR": self.test_dir}, clear=False):
            with patch.dict(namespace, overrides), patch("signal.signal"):
                with ExitStack() as stack:
                    if context_error is not None:
                        stack.enter_context(
                            patch.object(RunLedger, "create_submission_context", side_effect=context_error)
                        )
                    if outcome_error is not None:
                        stack.enter_context(
                            patch.object(RunLedger, "record_submission_outcome", side_effect=record_outcome)
                        )
                    exec(parent_code, namespace)

        return report_path, submitted, outcome_calls, executors

    def test_actual_parent_dispatch_and_candidate_persistence_survive_outcome_fault(self):
        candidate = {"tic_id": "12345", "period_days": 3.14}
        result = ("12345", "CANDIDATE", "Candidate found", candidate)
        observer_error = RuntimeError("Outcome ledger unavailable")
        report_path, submitted, outcome_calls, executors = self._execute_parent(
            result=result,
            outcome_error=observer_error,
        )

        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0][0], ("12345", 1))
        self.assertIsInstance(submitted[0][1]["observer_context"], ObserverContext)
        self.assertIs(outcome_calls[0][1], result)
        self.assertIsNone(outcome_calls[0][2])
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as report_file:
            self.assertIn("12345", report_file.read())
        self.assertTrue(executors[0].shutdown_called)

    def test_actual_parent_fallback_dispatch_survives_context_creation_fault(self):
        result = ("12345", "REJECT_BASELINE", "Short baseline", None)
        _, submitted, outcome_calls, executors = self._execute_parent(
            result=result,
            context_error=OSError("Observer context unavailable"),
        )
        self.assertEqual(len(submitted), 1)
        self.assertIsNone(submitted[0][1]["observer_context"])
        self.assertEqual(outcome_calls, [])
        self.assertTrue(executors[0].shutdown_called)

    def test_actual_parent_preserves_worker_exception_identity_when_outcome_faults(self):
        worker_error = RuntimeError("Worker computation failed")
        outcome_error = OSError("Observer outcome write failed")
        with self.assertRaises(RuntimeError) as raised:
            self._execute_parent(worker_error=worker_error, outcome_error=outcome_error)

        self.assertIs(raised.exception, worker_error)
        self.assertEqual(len(self.outcome_calls), 1)
        self.assertIs(self.outcome_calls[0][2], worker_error)

    def test_actual_parent_prequeue_failure_is_finalized_before_reraise(self):
        prequeue_error = RuntimeError("Catalog pre-queue failure")
        with self.assertRaises(RuntimeError) as raised:
            self._execute_parent(prequeue_error=prequeue_error)
        self.assertIs(raised.exception, prequeue_error)

        run_dirs = [
            os.path.join(self.test_dir, entry)
            for entry in os.listdir(self.test_dir)
            if os.path.isdir(os.path.join(self.test_dir, entry))
        ]
        self.assertEqual(len(run_dirs), 1)
        result = validate_run_ledger(run_dirs[0], expected_source_hash=get_module_source_hash())
        self.assertTrue(result.valid, result.errors)
        with open(os.path.join(run_dirs[0], "run.jsonl"), "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(any(event.get("event_name") == "PREQUEUE_FAILURE" for event in events))
        self.assertEqual(events[-1].get("event_name"), "FINAL")


# Helper function at module top-level for ProcessPoolExecutor spawn compatibility
def _run_mocked_worker(args):
    tic_id, sector, ctx, fail_mode = args
    if fail_mode == "exit_before_id":
        # Simulate worker dying before any records are written
        os._exit(99)

    # In clean or partial mode, call analyze_single_target with mocks
    with patch("aldebaran.lk.search_lightcurve") as mock_search, \
         patch("aldebaran.safe_download") as mock_dl, \
         patch("aldebaran.extract_and_stitch_arrays") as mock_stitch, \
         patch("aldebaran.get_known_transits_for_tic", return_value=[]):

        mock_search.return_value = ["dummy_product"]
        if fail_mode == "exit_after_partial":
            # Worker executes pipeline through SPOC search, then dies in safe_download naturally
            mock_dl.side_effect = lambda *a, **kw: os._exit(42)
        else:
            mock_dl.return_value = MagicMock()

        import numpy as np
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 1000)) * 0.001
        ferr_arr = np.full(1000, 0.001)
        dx_arr = np.zeros(1000)
        dy_arr = np.zeros(1000)
        crowdsap = 0.95
        mock_stitch.return_value = (t_arr, f_arr, ferr_arr, dx_arr, dy_arr, crowdsap)

        return aldebaran.analyze_single_target(
            tic_id, sector, min_period=0.8, max_period=13.0, observer_context=ctx
        )


class TestProcessPoolExecutorSpawnIntegration(unittest.TestCase):
    """
    Acceptance criteria A4:
    - REAL bounded ProcessPoolExecutor with spawn
    - >= 2 distinct submissions including repeated TIC/sector
    - Actual analyze_single_target entrypoint with mocks installed INSIDE spawned worker
    - Partial evidence worker self-exits; parent maps declared ledger, marks incomplete
    - Before-ID worker failure control
    - Finite deadlines, clean cleanup, no arbitrary PID signaling.
    """
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_ppe_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _assert_processes_stopped(self, processes):
        for process in processes:
            process.join(timeout=1)
        self.assertFalse(
            [process.pid for process in processes if process.is_alive()],
            "Owned ProcessPoolExecutor workers must be stopped after shutdown",
        )

    def test_real_process_pool_spawn_multi_submission_and_faults(self):
        ctx_spawn = multiprocessing.get_context("spawn")
        run_ledger = RunLedger(observer_dir=self.test_dir, legacy_run_id="ppe_run")

        # >= 2 distinct submissions including repeated TIC/sector:
        # 1. TIC 1001, Sector 1 (clean attempt 1)
        # 2. TIC 1001, Sector 1 (repeated TIC and sector! clean attempt 2)
        # 3. TIC 1002, Sector 1 (partial exit fault)
        # 4. TIC 1003, Sector 1 (before-ID exit fault)
        ctx1 = run_ledger.create_submission_context(tic_id="1001", sector=1)
        ctx2 = run_ledger.create_submission_context(tic_id="1001", sector=1)
        ctx3 = run_ledger.create_submission_context(tic_id="1002", sector=1)
        ctx4 = run_ledger.create_submission_context(tic_id="1003", sector=1)

        self.assertNotEqual(ctx1.submission_id, ctx2.submission_id)

        tasks = [
            ("1001", 1, ctx1, "clean"),
            ("1001", 1, ctx2, "clean"),
        ]

        completed_results = []
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx_spawn) as executor:
            future_to_sub = {
                executor.submit(_run_mocked_worker, task): task[2].submission_id
                for task in tasks
            }
            for fut in as_completed(future_to_sub, timeout=30):
                sub_id = future_to_sub[fut]
                try:
                    res = fut.result()
                    run_ledger.record_submission_outcome(sub_id, future_result=res)
                    completed_results.append(res)
                except Exception as exc:
                    run_ledger.record_submission_outcome(sub_id, exc=exc)
            clean_workers = tuple(executor._processes.values())

        self._assert_processes_stopped(clean_workers)
        self.assertEqual(len(completed_results), 2)
        # Both clean submissions executed and have valid results
        for res in completed_results:
            self.assertTrue(res[1] == "CANDIDATE" or res[1] == "EMPTY_LOW_SDE" or res[1].startswith("REJECT_"))

        # Test worker dying after partial evidence naturally inside the pipeline
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx_spawn) as executor:
            fut = executor.submit(_run_mocked_worker, ("1002", 1, ctx3, "exit_after_partial"))
            with self.assertRaises(Exception):
                # Worker called os._exit(42), raises BrokenProcessPool or similar
                fut.result(timeout=10)
            run_ledger.record_submission_outcome(ctx3.submission_id, exc=RuntimeError("Worker exited after partial"))
            partial_workers = tuple(executor._processes.values())

        self._assert_processes_stopped(partial_workers)

        # Check declared submission path for ctx3: has BEGIN event and gate_spoc_search written by real pipeline
        sub3_events = os.path.join(self.test_dir, run_ledger.observer_run_id, ctx3.submission_id, "events.jsonl")
        self.assertTrue(os.path.exists(sub3_events))
        with open(sub3_events, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertGreaterEqual(len(lines), 1)
        self.assertEqual(lines[0]["record_kind"], "submission")
        self.assertEqual(lines[0]["event_name"], "BEGIN")

        # Test worker dying before any records
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx_spawn) as executor:
            fut = executor.submit(_run_mocked_worker, ("1003", 1, ctx4, "exit_before_id"))
            with self.assertRaises(Exception):
                fut.result(timeout=10)
            run_ledger.record_submission_outcome(ctx4.submission_id, exc=RuntimeError("Worker exited before ID"))
            failed_workers = tuple(executor._processes.values())

        self._assert_processes_stopped(failed_workers)

        sub4_events = os.path.join(self.test_dir, run_ledger.observer_run_id, ctx4.submission_id, "events.jsonl")
        self.assertFalse(os.path.exists(sub4_events))

        # Finalize run
        run_ledger.finalize_run(completed_count=2, error_count=2, candidate_count=0)

        # Validate ledger: must fail because submissions 3 and 4 were unfinalized/missing
        v_res = validate_run_ledger(run_ledger.run_dir)
        self.assertFalse(v_res.valid)
        self.assertTrue(any("unfinalized submission" in e for e in v_res.errors))
        self.assertTrue(any("Missing events file" in e for e in v_res.errors))


if __name__ == "__main__":
    unittest.main()
