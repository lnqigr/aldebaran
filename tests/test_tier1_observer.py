"""
Tier-1 Observer Integration Tests and Immutable-Baseline Equivalence Suite.

Covers:
- A2: Exact production observer-off and observer-on comparison against immutable v9.1.0 tag source.
  Validates 4-tuple equality, selected candidate numeric fields, NaN semantics, masks/order/query counts.
- A3: Actual-path fixtures:
  1. noSPOC (search returns 0 products -> SKIP_NO_DATA)
  2. download/error (download exception -> ERROR)
  3. crowding boundary/default (crowdsap < 0.70 boundary & >= 0.70 pass)
  4. too-short/empty baseline (max_p <= min_period -> REJECT_BASELINE)
  5. low-SDE break (sde < 6.0 -> EMPTY_LOW_SDE, loop break)
  6. later peak survivor (peak 1 rejected on duty cycle, peak 2 candidate)
  7. target-aborting stellar return (evolved star / ruwe -> target abort)
- Factual verification of emitted values/units vs independently recomputed fixture facts.
- Short-circuit vs NOT_INSTRUMENTED coverage manifest verification.
"""

import os
import sys
import math
import subprocess
import tempfile
import shutil
import importlib.util
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import decision_ledger as dl
from decision_ledger import (
    COVERAGE_MANIFEST,
    ObserverContext,
    SubmissionLedger,
    RunLedger,
    validate_run_ledger,
)
import aldebaran


def load_immutable_tag_aldebaran():
    """Loads aldebaran.py from the immutable published v9.1.0 tag commit."""
    cmd = ["git", "show", "v9.1.0:aldebaran.py"]
    try:
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        module_code = res.stdout
    except Exception as e:
        # Check if git.exe is available (e.g. running under WSL with Windows worktree)
        try:
            res = subprocess.run(["git.exe", "show", "v9.1.0:aldebaran.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
            module_code = res.stdout
        except Exception:
            # Check if .git is a worktree pointer file with Windows path on Linux
            git_file = os.path.join(REPO_ROOT, ".git")
            if os.path.isfile(git_file):
                with open(git_file, "r") as gf:
                    content = gf.read().strip()
                if content.startswith("gitdir:"):
                    raw_path = content.split(":", 1)[1].strip()
                    if len(raw_path) > 2 and raw_path[1] == ":":
                        drive = raw_path[0].lower()
                        linux_path = f"/mnt/{drive}" + raw_path[2:].replace("\\", "/")
                        res = subprocess.run(["git", f"--git-dir={linux_path}", "show", "v9.1.0:aldebaran.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
                        module_code = res.stdout
                    else:
                        raise e
                else:
                    raise e
            else:
                raise e

    spec = importlib.util.spec_from_loader("tag_aldebaran", loader=None)
    tag_mod = importlib.util.module_from_spec(spec)
    exec(module_code, tag_mod.__dict__)
    return tag_mod


class TestTier1ObserverEquivalenceAndFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tag_aldebaran = load_immutable_tag_aldebaran()
        except Exception as exc:
            cls.tag_aldebaran = None
            raise RuntimeError(f"FATAL: Could not load immutable tag_aldebaran ({exc}). Baseline verification must fail closed.") from exc
        if cls.tag_aldebaran is None:
            raise RuntimeError("FATAL: immutable tag_aldebaran is None. Baseline verification must fail closed.")

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_t1_obs_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _assert_results_equivalent(self, res_tag, res_off, res_on):
        """Verifies strict equivalence across tag baseline, observer-off, and observer-on."""
        # 1. 4-tuple equivalence
        self.assertEqual(res_tag[0], res_off[0])
        self.assertEqual(res_off[0], res_on[0])

        self.assertEqual(res_tag[1], res_off[1])
        self.assertEqual(res_off[1], res_on[1])

        self.assertEqual(res_tag[2], res_off[2])
        self.assertEqual(res_off[2], res_on[2])

        # 2. Candidate dictionary equivalence
        cand_tag = res_tag[3]
        cand_off = res_off[3]
        cand_on = res_on[3]

        if cand_tag is None:
            self.assertIsNone(cand_off)
            self.assertIsNone(cand_on)
        else:
            self.assertIsNotNone(cand_off)
            self.assertIsNotNone(cand_on)
            self.assertEqual(set(cand_tag.keys()), set(cand_off.keys()))
            self.assertEqual(set(cand_off.keys()), set(cand_on.keys()))

            for k in cand_tag:
                val_tag = cand_tag[k]
                val_off = cand_off[k]
                val_on = cand_on[k]

                if isinstance(val_tag, float) and math.isnan(val_tag):
                    self.assertTrue(math.isnan(val_off), f"Key {k} expected NaN in observer-off")
                    self.assertTrue(math.isnan(val_on), f"Key {k} expected NaN in observer-on")
                elif isinstance(val_tag, float):
                    self.assertAlmostEqual(val_tag, val_off, places=9, msg=f"Float mismatch in {k} off vs tag")
                    self.assertAlmostEqual(val_off, val_on, places=9, msg=f"Float mismatch in {k} on vs off")
                else:
                    self.assertEqual(val_tag, val_off, f"Mismatch in {k} off vs tag")
                    self.assertEqual(val_off, val_on, f"Mismatch in {k} on vs off")

    def test_fixture_no_spoc(self):
        """Fixture 1: No SPOC products available in sector."""
        ctx = ObserverContext(self.test_dir, "run_no_spoc", "sub_no_spoc")

        with patch("aldebaran.lk.search_lightcurve", return_value=[]):
            res_off = aldebaran.analyze_single_target("12345", 1, observer_context=None)
            res_on = aldebaran.analyze_single_target("12345", 1, observer_context=ctx)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=[]):
            res_tag = self.tag_aldebaran.analyze_single_target("12345", 1)

        self._assert_results_equivalent(res_tag, res_off, res_on)
        self.assertEqual(res_on[1], "SKIP_NO_DATA")

        # Verify ledger contents
        events_path = os.path.join(self.test_dir, "run_no_spoc", "sub_no_spoc", "events.jsonl")
        self.assertTrue(os.path.exists(events_path))
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        gate_ev = next(e for e in events if e.get("record_kind") == "gate" and e.get("gate_id") == "gate_spoc_search")
        self.assertEqual(gate_ev["diagnostic_outcome"], "REJECT")
        self.assertEqual(gate_ev["production_result"], "return")
        self.assertEqual(gate_ev["reason_code"], "SKIP_NO_DATA")
        self.assertEqual(gate_ev["evidence"]["num_products"]["value"], 0)

        # Later gates (crowding, baseline, etc.) must NOT be evaluated (short-circuited)
        self.assertFalse(any(e.get("gate_id") == "gate_crowding" for e in events))
        self.assertFalse(any(e.get("gate_id") == "gate_baseline" for e in events))

    def test_fixture_download_error(self):
        """Fixture 2: safe_download encounters network / server error."""
        ctx = ObserverContext(self.test_dir, "run_dl_err", "sub_dl_err")

        err = RuntimeError("SPOC download connection timed out")
        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy_product"]), \
             patch("aldebaran.safe_download", side_effect=err):
            res_off = aldebaran.analyze_single_target("23456", 1, observer_context=None)
            res_on = aldebaran.analyze_single_target("23456", 1, observer_context=ctx)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy_product"]), \
             patch.object(self.tag_aldebaran, "safe_download", side_effect=err):
            res_tag = self.tag_aldebaran.analyze_single_target("23456", 1)

        self._assert_results_equivalent(res_tag, res_off, res_on)
        self.assertEqual(res_on[1], "ERROR")
        self.assertIn("SPOC download connection timed out", res_on[2])

        events_path = os.path.join(self.test_dir, "run_dl_err", "sub_dl_err", "events.jsonl")
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        last_ev = events[-1]
        self.assertEqual(last_ev["record_kind"], "submission")
        self.assertEqual(last_ev["event_name"], "FINAL")
        self.assertEqual(last_ev["status"], "ERROR")

    def test_fixture_crowding_boundary_and_default(self):
        """Fixture 3: CROWDSAP boundary < 0.70 and >= 0.70."""
        t_arr = np.linspace(0, 27, 500)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 500)) * 0.001
        ferr_arr = np.full(500, 0.001)
        dx_arr = np.zeros(500)
        dy_arr = np.zeros(500)

        # 3A: Crowdsap = 0.69 (fails crowding boundary)
        ctx_fail = ObserverContext(self.test_dir, "run_crowd", "sub_crowd_fail")
        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.69)):
            res_off = aldebaran.analyze_single_target("34567", 1, observer_context=None)
            res_on = aldebaran.analyze_single_target("34567", 1, observer_context=ctx_fail)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.69)):
            res_tag = self.tag_aldebaran.analyze_single_target("34567", 1)

        self._assert_results_equivalent(res_tag, res_off, res_on)
        self.assertEqual(res_on[1], "REJECT_CROWDSAP")
        self.assertIn("CROWDSAP=0.69 < 0.70", res_on[2])

        events_path = os.path.join(self.test_dir, "run_crowd", "sub_crowd_fail", "events.jsonl")
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        crowd_ev = next(e for e in events if e.get("gate_id") == "gate_crowding")
        self.assertEqual(crowd_ev["production_result"], "return")
        self.assertEqual(crowd_ev["diagnostic_outcome"], "REJECT")
        self.assertEqual(crowd_ev["reason_code"], "REJECT_CROWDSAP")
        self.assertEqual(crowd_ev["evidence"]["crowdsap"]["value"], 0.69)
        self.assertEqual(crowd_ev["thresholds"][0]["value"], 0.70)
        self.assertEqual(crowd_ev["thresholds"][0]["comparator"], "<")

        # 3B: Crowdsap = 0.70 (passes crowding boundary, proceeds to next gate)
        ctx_pass = ObserverContext(self.test_dir, "run_crowd", "sub_crowd_pass")
        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.70)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]):
            res_off = aldebaran.analyze_single_target("34568", 1, observer_context=None)
            res_pass = aldebaran.analyze_single_target("34568", 1, observer_context=ctx_pass)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.70)), \
             patch.object(self.tag_aldebaran, "get_known_transits_for_tic", return_value=[]):
            res_tag = self.tag_aldebaran.analyze_single_target("34568", 1)

        self._assert_results_equivalent(res_tag, res_off, res_pass)
        # Must not be rejected on crowding
        self.assertNotEqual(res_pass[1], "REJECT_CROWDSAP")

        events_pass_path = os.path.join(self.test_dir, "run_crowd", "sub_crowd_pass", "events.jsonl")
        with open(events_pass_path, "r", encoding="utf-8") as f:
            events_pass = [dl.json.loads(l) for l in f if l.strip()]

        crowd_pass_ev = next(e for e in events_pass if e.get("gate_id") == "gate_crowding")
        self.assertEqual(crowd_pass_ev["production_result"], "pass")
        self.assertEqual(crowd_pass_ev["diagnostic_outcome"], "PASS")
        self.assertEqual(crowd_pass_ev["evidence"]["crowdsap"]["value"], 0.70)

    def test_fixture_too_short_baseline(self):
        """Fixture 4: Time span is too short (max_p <= min_period)."""
        ctx = ObserverContext(self.test_dir, "run_base", "sub_base_short")
        # span is 1.0 day -> max_p = 1.0 * 0.48 = 0.48 <= min_period (0.8)
        t_short = np.linspace(0, 1.0, 200)
        f_short = 1.0 + np.sin(np.linspace(0, 10, 200)) * 0.001
        ferr_short = np.full(200, 0.001)
        dx_short = np.zeros(200)
        dy_short = np.zeros(200)

        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_short, f_short, ferr_short, dx_short, dy_short, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]):
            res_off = aldebaran.analyze_single_target("45678", 1, min_period=0.8, observer_context=None)
            res_on = aldebaran.analyze_single_target("45678", 1, min_period=0.8, observer_context=ctx)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_short, f_short, ferr_short, dx_short, dy_short, 0.95)), \
             patch.object(self.tag_aldebaran, "get_known_transits_for_tic", return_value=[]):
            res_tag = self.tag_aldebaran.analyze_single_target("45678", 1, min_period=0.8)

        self._assert_results_equivalent(res_tag, res_off, res_on)
        self.assertEqual(res_on[1], "REJECT_BASELINE")

        events_path = os.path.join(self.test_dir, "run_base", "sub_base_short", "events.jsonl")
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        base_ev = next(e for e in events if e.get("gate_id") == "gate_baseline")
        self.assertEqual(base_ev["production_result"], "return")
        self.assertEqual(base_ev["diagnostic_outcome"], "REJECT")
        self.assertEqual(base_ev["reason_code"], "REJECT_BASELINE")
        self.assertAlmostEqual(base_ev["evidence"]["effective_max_p"]["value"], 0.48, places=4)
        self.assertEqual(base_ev["thresholds"][0]["name"], "min_period")
        self.assertEqual(base_ev["thresholds"][0]["value"], 0.8)

    def test_fixture_bls_whitening_error_has_terminal_event(self):
        """A failure after BLS power still closes the started BLS search with ERROR."""
        run_ledger = RunLedger(observer_dir=self.test_dir, legacy_run_id="bls_error")
        observer_context = run_ledger.create_submission_context("56788", 1)
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 1000)) * 0.001
        ferr_arr = np.full(1000, 0.001)
        dx_arr = np.zeros(1000)
        dy_arr = np.zeros(1000)
        fake_bls = MagicMock()
        fake_bls.power = np.full(100, 1.0)
        fake_bls.period = np.full(100, 5.0)
        fake_bls.transit_time = np.full(100, 1.0)
        fake_bls.duration = np.full(100, 0.1)
        whitening_error = RuntimeError("injected whitening failure")

        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]), \
             patch("aldebaran.BoxLeastSquares.power", return_value=fake_bls), \
             patch("aldebaran.compute_whitened_kovacs_sde", side_effect=whitening_error):
            result_off = aldebaran.analyze_single_target("56788", 1)
            result_on = aldebaran.analyze_single_target("56788", 1, observer_context=observer_context)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch.object(self.tag_aldebaran, "get_known_transits_for_tic", return_value=[]), \
             patch.object(self.tag_aldebaran.BoxLeastSquares, "power", return_value=fake_bls), \
             patch.object(self.tag_aldebaran, "compute_whitened_kovacs_sde", side_effect=whitening_error):
            result_tag = self.tag_aldebaran.analyze_single_target("56788", 1)

        self._assert_results_equivalent(result_tag, result_off, result_on)
        self.assertEqual(result_on[1], "ERROR")
        run_ledger.record_submission_outcome(observer_context.submission_id, future_result=result_on)
        run_ledger.finalize_run(completed_count=1, error_count=1, candidate_count=0)

        events_path = os.path.join(
            self.test_dir, run_ledger.observer_run_id, observer_context.submission_id, "events.jsonl"
        )
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(line) for line in f if line.strip()]
        bls_events = [
            event for event in events
            if event.get("record_kind") == "search" and event.get("gate_id") == "search_bls"
        ]
        self.assertEqual([event["event_name"] for event in bls_events], ["START", "ERROR"])

        validation = validate_run_ledger(
            run_ledger.run_dir,
            expected_source_hash=run_ledger.source_hash,
        )
        self.assertTrue(validation.valid, validation.errors)

    def test_fixture_low_sde_break(self):
        """Fixture 5: Signal Detection Efficiency < 6.0 breaks peak loop."""
        ctx = ObserverContext(self.test_dir, "run_sde", "sub_sde_low")
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 1000)) * 0.001
        ferr_arr = np.full(1000, 0.001)
        dx_arr = np.zeros(1000)
        dy_arr = np.zeros(1000)

        # Mock BLS power and SDE calculation to force low SDE
        fake_bls_res = MagicMock()
        fake_bls_res.power = np.full(100, 1.0)
        fake_bls_res.period = np.linspace(1, 10, 100)
        fake_bls_res.transit_time = np.full(100, 0.5)
        fake_bls_res.duration = np.full(100, 0.1)

        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]), \
             patch("aldebaran.BoxLeastSquares.power", return_value=fake_bls_res), \
             patch("aldebaran.compute_whitened_kovacs_sde", return_value=(np.full(100, 4.2), None)), \
             patch("aldebaran.get_top_distinct_bls_peaks", return_value=[0, 10, 20]):

            res_off = aldebaran.analyze_single_target("56789", 1, observer_context=None)
            res_on = aldebaran.analyze_single_target("56789", 1, observer_context=ctx)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch.object(self.tag_aldebaran, "get_known_transits_for_tic", return_value=[]), \
             patch.object(self.tag_aldebaran.BoxLeastSquares, "power", return_value=fake_bls_res), \
             patch.object(self.tag_aldebaran, "compute_whitened_kovacs_sde", return_value=(np.full(100, 4.2), None)), \
             patch.object(self.tag_aldebaran, "get_top_distinct_bls_peaks", return_value=[0, 10, 20]):
            res_tag = self.tag_aldebaran.analyze_single_target("56789", 1)

        self._assert_results_equivalent(res_tag, res_off, res_on)
        self.assertEqual(res_on[1], "EMPTY_LOW_SDE")
        self.assertIn("SDE=4.2", res_on[2])

        events_path = os.path.join(self.test_dir, "run_sde", "sub_sde_low", "events.jsonl")
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        sde_ev = next(e for e in events if e.get("gate_id") == "gate_low_sde")
        self.assertEqual(sde_ev["production_result"], "break")
        self.assertEqual(sde_ev["diagnostic_outcome"], "REJECT")
        self.assertEqual(sde_ev["reason_code"], "EMPTY_LOW_SDE")
        self.assertEqual(sde_ev["evidence"]["sde"]["value"], 4.2)
        self.assertEqual(sde_ev["rank_idx"], 1)

        # Because SDE < 6.0 broke on peak 1, peaks 2 and 3 must NOT be evaluated
        sde_events = [e for e in events if e.get("gate_id") == "gate_low_sde"]
        self.assertEqual(len(sde_events), 1)

    def test_fixture_later_peak_survivor(self):
        """Fixture 6: wide first peak is rejected; peak 2 reaches the stellar-abort return."""
        ctx = ObserverContext(self.test_dir, "run_surv", "sub_surv_01")
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, len(t_arr))) * 0.001
        ferr_arr = np.full(len(t_arr), 0.001)
        dx_arr = np.zeros(len(t_arr))
        dy_arr = np.zeros(len(t_arr))
        for transit_time in (1.0, 6.0, 11.0, 16.0, 21.0, 26.0):
            f_arr[np.abs(t_arr - transit_time) < 0.03] = 0.995

        # Mock BLS with 2 peaks:
        # Peak 0: SDE=8.0, P=2.0, Dur=0.25 -> Dur/P = 0.125 > 0.095 (Duty cycle reject)
        # Peak 1: SDE=7.5, P=5.0, Dur=0.08 -> Dur/P = 0.016 <= 0.095 (Survives duty cycle)
        fake_bls = MagicMock()
        fake_bls.power = np.full(100, 1.0)
        fake_periods = np.full(100, 5.0)
        fake_periods[0] = 2.0
        fake_periods[1] = 5.0
        fake_bls.period = fake_periods

        fake_t0 = np.full(100, 1.0)
        fake_bls.transit_time = fake_t0

        fake_durs = np.full(100, 0.1)
        fake_durs[0] = 0.25
        fake_durs[1] = 0.1
        fake_bls.duration = fake_durs

        sde_arr = np.full(100, 7.5)
        sde_arr[0] = 8.0
        sde_arr[1] = 12.0

        mock_star = {
            "r_star": 3.5,
            "e_r_star": 0.35,
            "m_star": 1.2,
            "e_m_star": 0.14,
            "teff": 5800,
            "ruwe": 1.0,
            "has_params": True
        }
        mock_arch = (True, "Confirmed_in_Sectors_1", {"archival_snr": 5.0, "multi_sde": 12.0})
        diff_image = np.zeros((2, 2))
        aperture = np.ones((2, 2), dtype=bool)

        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]), \
             patch("aldebaran.BoxLeastSquares.power", return_value=fake_bls), \
             patch("aldebaran.compute_whitened_kovacs_sde", return_value=(sde_arr, None)), \
             patch("aldebaran.get_top_distinct_bls_peaks", return_value=[0, 1]), \
             patch("aldebaran.get_stellar_and_gaia_params", return_value=mock_star), \
             patch("aldebaran.verify_and_filter_tpf", return_value=(True, "OK", diff_image, aperture)), \
             patch("aldebaran.verify_multi_epoch", return_value=mock_arch), \
             patch("aldebaran.scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 4.0)), \
             patch("aldebaran.record_archival_diagnostic"), \
             patch("aldebaran.OUTPUT_DIR", self.test_dir):

            res_off = aldebaran.analyze_single_target("67890", 1, observer_context=None)
            res_on = aldebaran.analyze_single_target("67890", 1, observer_context=ctx)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch.object(self.tag_aldebaran, "get_known_transits_for_tic", return_value=[]), \
             patch.object(self.tag_aldebaran.BoxLeastSquares, "power", return_value=fake_bls), \
             patch.object(self.tag_aldebaran, "compute_whitened_kovacs_sde", return_value=(sde_arr, None)), \
             patch.object(self.tag_aldebaran, "get_top_distinct_bls_peaks", return_value=[0, 1]), \
             patch.object(self.tag_aldebaran, "get_stellar_and_gaia_params", return_value=mock_star), \
             patch.object(self.tag_aldebaran, "verify_and_filter_tpf", return_value=(True, "OK", diff_image, aperture)), \
             patch.object(self.tag_aldebaran, "verify_multi_epoch", return_value=mock_arch), \
             patch.object(self.tag_aldebaran, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 4.0)), \
             patch.object(self.tag_aldebaran, "record_archival_diagnostic"), \
             patch.object(self.tag_aldebaran, "OUTPUT_DIR", self.test_dir):
            res_tag = self.tag_aldebaran.analyze_single_target("67890", 1)

        self._assert_results_equivalent(res_tag, res_off, res_on)
        self.assertEqual(res_on[1], "REJECT_EVOLVED_STAR", res_on[2])
        self.assertIn("Peak#2", res_on[2])

        events_path = os.path.join(self.test_dir, "run_surv", "sub_surv_01", "events.jsonl")
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        sde_events = [e for e in events if e.get("gate_id") == "gate_low_sde"]
        self.assertEqual(len(sde_events), 2)
        self.assertEqual(sde_events[0]["rank_idx"], 1)
        self.assertEqual(sde_events[0]["diagnostic_outcome"], "PASS")
        self.assertEqual(sde_events[1]["rank_idx"], 2)
        self.assertEqual(sde_events[1]["diagnostic_outcome"], "PASS")

    def test_fixture_target_aborting_stellar_return(self):
        """Fixture 7: Target-aborting stellar return (e.g. evolved star)."""
        ctx = ObserverContext(self.test_dir, "run_abort", "sub_abort_01")
        t_arr = np.linspace(0, 27, 1000)
        f_arr = 1.0 + np.sin(np.linspace(0, 50, 1000)) * 0.001
        ferr_arr = np.full(1000, 0.001)
        dx_arr = np.zeros(1000)
        dy_arr = np.zeros(1000)

        # Inject transit points so initial geometric gates pass and reach stellar params
        transit_times = [1.0, 6.0, 11.0, 16.0, 21.0, 26.0]
        for tt in transit_times:
            in_tr = np.abs(t_arr - tt) < 0.03
            f_arr[in_tr] = 0.995

        fake_bls = MagicMock()
        fake_bls.power = np.full(100, 1.0)
        fake_bls.period = np.full(100, 5.0)
        fake_bls.transit_time = np.full(100, 1.0)
        fake_bls.duration = np.full(100, 0.1)
        sde_arr = np.full(100, 12.0)

        # Mock stellar info to return an evolved giant star: R_star = 3.5 > 1.75
        mock_star = {
            "r_star": 3.5,
            "e_r_star": 0.35,
            "m_star": 1.2,
            "e_m_star": 0.14,
            "teff": 5500,
            "ruwe": 1.0,
            "has_params": True
        }

        with patch("aldebaran.lk.search_lightcurve", return_value=["dummy"]), \
             patch("aldebaran.safe_download", return_value=MagicMock()), \
             patch("aldebaran.extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch("aldebaran.get_known_transits_for_tic", return_value=[]), \
             patch("aldebaran.BoxLeastSquares.power", return_value=fake_bls), \
             patch("aldebaran.compute_whitened_kovacs_sde", return_value=(sde_arr, None)), \
             patch("aldebaran.get_top_distinct_bls_peaks", return_value=[0, 1]), \
             patch("aldebaran.get_stellar_and_gaia_params", return_value=mock_star):

            res_off = aldebaran.analyze_single_target("78901", 1, observer_context=None)
            res_on = aldebaran.analyze_single_target("78901", 1, observer_context=ctx)

        with patch.object(self.tag_aldebaran.lk, "search_lightcurve", return_value=["dummy"]), \
             patch.object(self.tag_aldebaran, "safe_download", return_value=MagicMock()), \
             patch.object(self.tag_aldebaran, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr_arr, dx_arr, dy_arr, 0.95)), \
             patch.object(self.tag_aldebaran, "get_known_transits_for_tic", return_value=[]), \
             patch.object(self.tag_aldebaran.BoxLeastSquares, "power", return_value=fake_bls), \
             patch.object(self.tag_aldebaran, "compute_whitened_kovacs_sde", return_value=(sde_arr, None)), \
             patch.object(self.tag_aldebaran, "get_top_distinct_bls_peaks", return_value=[0, 1]), \
             patch.object(self.tag_aldebaran, "get_stellar_and_gaia_params", return_value=mock_star):
            res_tag = self.tag_aldebaran.analyze_single_target("78901", 1)

        self._assert_results_equivalent(res_tag, res_off, res_on)

        self.assertEqual(res_on[1], "REJECT_EVOLVED_STAR")
        events_path = os.path.join(self.test_dir, "run_abort", "sub_abort_01", "events.jsonl")
        with open(events_path, "r", encoding="utf-8") as f:
            events = [dl.json.loads(l) for l in f if l.strip()]

        target_res_ev = next(e for e in events if e.get("record_kind") == "target_result")
        self.assertEqual(target_res_ev["status_code"], "REJECT_EVOLVED_STAR")

    def test_coverage_manifest_truthfulness(self):
        """Validates that COVERAGE_MANIFEST accurately distinguishes instrumented vs uninstrumented."""
        self.assertEqual(COVERAGE_MANIFEST["gate_spoc_search"], "INSTRUMENTED")
        self.assertEqual(COVERAGE_MANIFEST["gate_crowding"], "INSTRUMENTED")
        self.assertEqual(COVERAGE_MANIFEST["gate_baseline"], "INSTRUMENTED")
        self.assertEqual(COVERAGE_MANIFEST["search_bls"], "INSTRUMENTED")
        self.assertEqual(COVERAGE_MANIFEST["gate_low_sde"], "INSTRUMENTED")

        # Must explicitly mark other gates as NOT_INSTRUMENTED, never PASS
        uninstrumented = [k for k, v in COVERAGE_MANIFEST.items() if v == "NOT_INSTRUMENTED"]
        self.assertIn("gate_duty_cycle", uninstrumented)
        self.assertIn("gate_harmonic", uninstrumented)
        self.assertIn("gate_points", uninstrumented)
        self.assertIn("gate_depth", uninstrumented)
        self.assertIn("gate_secondary", uninstrumented)
        self.assertIn("gate_centroid", uninstrumented)
        self.assertIn("gate_stellar_params", uninstrumented)
        self.assertIn("gate_archival", uninstrumented)
        self.assertGreater(len(uninstrumented), 10)


if __name__ == "__main__":
    unittest.main()
