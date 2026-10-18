"""
tests/test_known_signal_diagnostics.py - Comprehensive Verification and Falsification Suite
for Defensive Known-Signal Geometry Diagnostics (Task ALD-007B1).

Strictly offline, reproducible, non-active test coverage verifying:
  - Acceptance Criteria A1 through A8
  - Findings R1 through R6 from independent review round 1
  - All 25 consultation status controls
  - Deterministic hand calculations with covariance transport and Cauchy-Schwarz bounds
  - Real-path scoped mutation testing demonstrating mutant killing (R5)
  - Direct JSON serialization without caller conversion (R2)
"""

from __future__ import annotations

import json
import math
import unittest
from unittest import mock

import numpy as np

import known_signal_diagnostics as ksd


class TestOfflineModuleAndBoundaries(unittest.TestCase):
    """
    Acceptance Criterion A1 & Finding R2:
    Public module operates offline, import has no side effects;
    entry point produces versioned diagnostic-only outputs and never authorizes production.
    Outputs are directly JSON-serializable (mask is list[bool]).
    Caller inputs are never mutated.
    """

    def test_module_constants_and_declarations(self):
        self.assertEqual(ksd.SCHEMA_VERSION, "known_signal_diagnostics_v1")
        self.assertEqual(ksd.MODE, "DIAGNOSTIC_ONLY")
        self.assertIs(ksd.PRODUCTION_MASK_ALLOWED, False)

    def test_offline_execution_no_network(self):
        with mock.patch("socket.socket", side_effect=RuntimeError("Network access forbidden")):
            t = np.array([2000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(
                t,
                period_days=10.0,
                epoch_btjd=2000.0,
                duration_days=0.1,
                raw_disposition="CP",
                stage_policy_action="ALLOW_ACTIVE_MASK",
                allowed_dispositions=("CP",),
                normalized_time_format="BTJD",
                normalized_time_scale="TDB",
                original_time_format="BTJD",
                original_time_scale="TDB",
                ephemeris_coherence_status="COHERENT",
                solution_id="FIT_001",
                marginal_errors_solution_id="FIT_001",
                sigma_t0_days=0.01,
                sigma_p_days=0.0001,
                sigma_dur_days=0.0,
                published_epoch_btjd=2000.0,
                marginal_errors_at_published_epoch=True,
                duration_frame_type="INTRINSIC_DECONVOLVED",
                exp_time_days=120.0 / 86400.0,
                k_sigma=3.0,
                k_dur=1.0,
                alpha_stage=1.0,
            )
            self.assertEqual(res["mode"], "DIAGNOSTIC_ONLY")
            self.assertIs(res["production_mask_allowed"], False)

    def test_caller_array_immutability(self):
        t_orig = np.array([2000.0, 2010.0, 2020.0], dtype=float)
        t_copy = t_orig.copy()
        _ = ksd.evaluate_known_signal_geometry_diagnostics(
            t_orig,
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        np.testing.assert_array_equal(t_orig, t_copy)

    def test_direct_json_serialization_no_caller_repair(self):
        # Strict R2 check: json.dumps(res, allow_nan=False) must succeed directly on returned dict
        t = np.array([2000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(
            t,
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        self.assertIsInstance(res["mask"], list)
        self.assertTrue(all(isinstance(x, bool) for x in res["mask"]))
        dumped = json.dumps(res, allow_nan=False)
        self.assertIsInstance(dumped, str)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)


class TestCoreStatusControls(unittest.TestCase):
    """
    Acceptance Criterion A2 & Findings R3/R4:
    Adapts all 25 status controls from the consultation to the real public API.
    Rejects fractional and boolean shifts.
    Tests missing, boolean, and non-finite policy/time/provenance inputs.
    Verifies actual mathematical/logical invariants, not strings alone.
    """

    def setUp(self):
        self.base_kwargs = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )

    def test_all_25_consultation_status_controls_matrix(self):
        cases = [
            ("valid_positive", "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND", dict(t=[2000.0])),
            ("zero_diagonal_nonPSD", "INVALID_COVARIANCE_MATRIX", dict(t=[2000.0], var_t0_days2=0.0, cov_t0_p_days2=-5e-13, var_p_days2=0.0)),
            ("ordinary_nonPSD", "INVALID_COVARIANCE_MATRIX", dict(t=[2000.0], var_t0_days2=1e-4, cov_t0_p_days2=2e-6, var_p_days2=1e-8)),
            ("k_sigma_nan", "INVALID_POLICY_PARAMETERS", dict(t=[2000.0], k_sigma=float("nan"))),
            ("k_sigma_negative", "INVALID_POLICY_PARAMETERS", dict(t=[2000.0], k_sigma=-3.0)),
            ("k_sigma_inf", "INVALID_POLICY_PARAMETERS", dict(t=[2000.0], k_sigma=float("inf"))),
            ("k_sigma_bool", "INVALID_POLICY_PARAMETERS", dict(t=[2000.0], k_sigma=True)),
            ("time_empty", "INVALID_TIME_ARRAY", dict(t=[])),
            ("time_nan", "INVALID_TIME_ARRAY", dict(t=[float("nan")])),
            ("time_2d", "INVALID_TIME_ARRAY", dict(t=[[2000.0]])),
            ("unknown_duration", "INVALID_DURATION_FRAME_TYPE", dict(t=[2000.0], duration_frame_type="UNKNOWN")),
            ("none_scale", "UNSAFE_OR_UNKNOWN_TIME_SCALE", dict(t=[2000.0], original_time_scale=None)),
            ("nan_epoch_uncertainty", "INSUFFICIENT_UNCERTAINTY_METADATA", dict(t=[2000.0], sigma_t0_days=float("nan"))),
            ("rebased_missing_original", "INSUFFICIENT_UNCERTAINTY_PROVENANCE", dict(t=[7000.0], epoch_btjd=7000.0, published_epoch_btjd=None)),
            ("rebased_false_provenance", "INSUFFICIENT_UNCERTAINTY_PROVENANCE", dict(t=[7000.0], epoch_btjd=7000.0, marginal_errors_at_published_epoch=False)),
            ("rebased_valid_original", "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND", dict(t=[7000.0], epoch_btjd=7000.0)),
            ("unknown_disposition", "DISPOSITION_NOT_PERMITTED_FOR_ACTIVE_MASK", dict(t=[2000.0], raw_disposition="UNKNOWN")),
            ("no_stage_authorization", "DISPOSITION_NOT_PERMITTED_FOR_ACTIVE_MASK", dict(t=[2000.0], stage_policy_action=None)),
            ("incomplete_covariance", "INSUFFICIENT_UNCERTAINTY_METADATA", dict(t=[2000.0], var_t0_days2=1e-4)),
            ("uncapped_2sigma", "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND", dict(t=[2000.2], sigma_t0_days=0.1, sigma_p_days=0.0)),
            ("budget", "INSUFFICIENT_TIMING_PRECISION_FOR_STAGE", dict(t=[2000.2], sigma_t0_days=0.1, sigma_p_days=0.0, max_stage_half_width_days=0.15)),
            ("cov_before", "ACTIVE_MASK_VALID_EXACT_COVARIANCE", dict(t=[3000.0], var_t0_days2=1e-4, cov_t0_p_days2=8e-7, var_p_days2=1e-8, covariance_epoch_btjd=2000.0, duration_frame_type="SMEARED_OBSERVED")),
            ("cov_after", "ACTIVE_MASK_VALID_EXACT_COVARIANCE", dict(t=[3000.0], duration_frame_type="SMEARED_OBSERVED", **dict(zip(('epoch_btjd', 'var_t0_days2', 'cov_t0_p_days2', 'var_p_days2'), ksd.transform_ephemeris_epoch_covariance(2000.0, 10.0, 1e-4, 8e-7, 1e-8, 500))), covariance_epoch_btjd=7000.0)),
            ("noninteger_original_epoch_relation", "INSUFFICIENT_UNCERTAINTY_PROVENANCE", dict(t=[2004.0], epoch_btjd=2004.0, sigma_t0_days=0.0, sigma_p_days=0.0)),
            ("string_false_provenance", "INSUFFICIENT_UNCERTAINTY_PROVENANCE", dict(t=[2000.0], marginal_errors_at_published_epoch="False")),
        ]
        self.assertEqual(len(cases), 25)
        for name, expected, kw_override in cases:
            with self.subTest(case_name=name):
                call_kw = self.base_kwargs.copy()
                t_val = kw_override.get("t", [2000.0])
                for k, v in kw_override.items():
                    if k != "t":
                        call_kw[k] = v
                t_arr = np.asarray(t_val, dtype=float) if (isinstance(t_val, list) and t_val and isinstance(t_val[0], list)) else np.array(t_val, dtype=float)
                res = ksd.evaluate_known_signal_geometry_diagnostics(t_arr, **call_kw)
                self.assertEqual(res["status"], expected, f"Case {name} failed: got {res['status']!r}, expected {expected!r}")

    def test_valid_positive_baseline(self):
        t = np.array([2000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **self.base_kwargs)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertTrue(res["mask_valid"])
        self.assertTrue(res["mask"][0])
        self.assertAlmostEqual(res["required_half_width_median_minutes"], 116.2, places=7)

    def test_zero_diagonal_non_psd(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw.update(var_t0_days2=0.0, cov_t0_p_days2=-5e-13, var_p_days2=0.0)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INVALID_COVARIANCE_MATRIX")
        self.assertEqual(res["geometry_status"], "NON_PSD_COVARIANCE")
        self.assertFalse(res["mask_valid"])

    def test_ordinary_non_psd_covariance(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw.update(var_t0_days2=1e-4, cov_t0_p_days2=2e-6, var_p_days2=1e-8)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INVALID_COVARIANCE_MATRIX")
        self.assertEqual(res["geometry_status"], "NON_PSD_COVARIANCE")
        self.assertFalse(res["mask_valid"])

    def test_policy_parameter_invalid_types(self):
        t = np.array([2000.0])
        for val in (float("nan"), -3.0, float("inf"), True, False, "invalid_str"):
            kw = self.base_kwargs.copy()
            kw["k_sigma"] = val
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INVALID_POLICY_PARAMETERS")
            self.assertFalse(res["mask_valid"])

    def test_time_array_invalid_inputs(self):
        for bad_t in (np.array([]), np.array([float("nan")]), np.array([[2000.0]]), np.array([True])):
            res = ksd.evaluate_known_signal_geometry_diagnostics(bad_t, **self.base_kwargs)
            self.assertEqual(res["status"], "INVALID_TIME_ARRAY")
            self.assertFalse(res["mask_valid"])

    def test_unknown_duration_frame(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["duration_frame_type"] = "UNKNOWN"
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INVALID_DURATION_FRAME_TYPE")
        self.assertFalse(res["mask_valid"])

    def test_unsafe_time_scale(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["original_time_scale"] = None
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
        self.assertEqual(res["geometry_status"], "UNSAFE_TIME_SCALE")
        self.assertFalse(res["mask_valid"])

    def test_nan_epoch_uncertainty(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["sigma_t0_days"] = float("nan")
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_METADATA")
        self.assertEqual(res["geometry_status"], "MISSING_EPOCH_ERR")
        self.assertFalse(res["mask_valid"])

    def test_rebased_missing_original_epoch(self):
        t = np.array([7000.0])
        kw = self.base_kwargs.copy()
        kw.update(epoch_btjd=7000.0, published_epoch_btjd=None)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_PUBLISHED_EPOCH_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_rebased_false_provenance(self):
        t = np.array([7000.0])
        kw = self.base_kwargs.copy()
        kw.update(epoch_btjd=7000.0, marginal_errors_at_published_epoch=False)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_PUBLISHED_EPOCH_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_rebased_valid_original_epoch(self):
        t = np.array([7000.0])
        kw = self.base_kwargs.copy()
        kw.update(epoch_btjd=7000.0)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertTrue(res["mask_valid"])
        self.assertAlmostEqual(res["required_half_width_median_minutes"], 332.2, places=7)

    def test_unknown_disposition_not_permitted(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["raw_disposition"] = "UNKNOWN"
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "DISPOSITION_NOT_PERMITTED_FOR_ACTIVE_MASK")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertFalse(res["mask_valid"])
        self.assertAlmostEqual(res["diagnostic_half_width_median_minutes"], 116.2, places=7)

    def test_no_stage_authorization(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["stage_policy_action"] = None
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "DISPOSITION_NOT_PERMITTED_FOR_ACTIVE_MASK")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertFalse(res["mask_valid"])

    def test_incomplete_covariance_metadata(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw.update(var_t0_days2=1e-4)  # missing cov_t0_p and var_p
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_METADATA")
        self.assertEqual(res["geometry_status"], "INCOMPLETE_COVARIANCE")
        self.assertFalse(res["mask_valid"])

    def test_uncapped_large_sigma_valid(self):
        t = np.array([2000.2])
        kw = self.base_kwargs.copy()
        kw.update(sigma_t0_days=0.1, sigma_p_days=0.0)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertTrue(res["mask_valid"])
        self.assertTrue(res["mask"][0])
        self.assertAlmostEqual(res["required_half_width_median_minutes"], 505.0, places=7)

    def test_stage_budget_exceeded(self):
        t = np.array([2000.2])
        kw = self.base_kwargs.copy()
        kw.update(sigma_t0_days=0.1, sigma_p_days=0.0, max_stage_half_width_days=0.15)
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_TIMING_PRECISION_FOR_STAGE")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertFalse(res["mask_valid"])

    def test_boolean_time_array_rejected(self):
        # Strict R3 check: boolean time array must be rejected
        res_bool_arr = ksd.evaluate_known_signal_geometry_diagnostics(
            np.array([True]),
            **self.base_kwargs,
        )
        self.assertEqual(res_bool_arr["status"], "INVALID_TIME_ARRAY")
        self.assertFalse(res_bool_arr["mask_valid"])

    def test_bad_policy_parameter_type_rejected(self):
        # Strict R3 check: non-numeric string must return INVALID_POLICY_PARAMETERS without uncaught TypeError
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["k_sigma"] = "three"
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INVALID_POLICY_PARAMETERS")
        self.assertFalse(res["mask_valid"])

    def test_covariance_epoch_transform_shift_rejection(self):
        with self.assertRaises(ValueError):
            ksd.transform_ephemeris_epoch_covariance(2000.0, 10.0, 1e-4, 8e-7, 1e-8, 1.9)
        with self.assertRaises(ValueError):
            ksd.transform_ephemeris_epoch_covariance(2000.0, 10.0, 1e-4, 8e-7, 1e-8, True)
        with self.assertRaises(ValueError):
            ksd.transform_ephemeris_epoch_covariance(2000.0, 10.0, 1e-4, 8e-7, 1e-8, False)


class TestHandCalculationsAndEpochTransport(unittest.TestCase):
    """
    Acceptance Criterion A3 & Review Finding R1:
    Independent hand calculation:
      sigmaT0 = 0.01 d, sigmaP = 0.0001 d, rho = 0.8, N = 100 => Var = 0.00036 d^2.
      With alpha = 1, D = 0.1 d, k_time = 3, sigmaD = 0, SMEARED => width = 153.9662369515644 min.
      Unchanged after m = 500 with covariance/provenance transport.
      Covariance bound at 2000, evaluated at 7000:
        Var(T0 + 500P) = 1e-4 + 1000*8e-7 + 250000*1e-8 = 0.0034 d^2
        Width = (0.05 + 3*sqrt(0.0034))*1440 = 323.897121857317 min.
      CS publishedT0 = 2000, currentT0 = 7000, P = 10, t = 7000 => 332.2 min with 120s intrinsic exposure.
      Missing original metadata is unavailable; no silent 116.2 min fallback.
    """

    def test_covariance_epoch_bound_transport_hand_calculation(self):
        # Review Finding R1:
        # currentT0 = 7000, covariance_epoch = 2000, P = 10, t = 7000
        # Predictive timing variance relative to bound covariance epoch 2000:
        # n_cov = (7000 - 2000)/10 = 500
        # Var = 1e-4 + 2*500*(8e-7) + 500^2*(1e-8) = 1e-4 + 8e-4 + 2.5e-3 = 0.0034 d^2
        # Width with SMEARED_OBSERVED: (0.05 + 3 * sqrt(0.0034)) * 1440 = 323.897121857317 min.
        t_eval = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(
            t_eval,
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            var_t0_days2=1e-4,
            cov_t0_p_days2=8e-7,
            var_p_days2=1e-8,
            covariance_epoch_btjd=2000.0,
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
        )
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_EXACT_COVARIANCE")
        expected_width = (0.05 + 3.0 * math.sqrt(0.0034)) * 1440.0
        self.assertAlmostEqual(res["required_half_width_median_minutes"], expected_width, places=10)
        self.assertAlmostEqual(res["required_half_width_median_minutes"], 323.897121857317, places=7)

    def test_missing_covariance_epoch_rejected(self):
        # Review Finding R1:
        # Missing covariance_epoch_btjd must return INSUFFICIENT_UNCERTAINTY_PROVENANCE
        t_eval = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(
            t_eval,
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            var_t0_days2=1e-4,
            cov_t0_p_days2=8e-7,
            var_p_days2=1e-8,
            covariance_epoch_btjd=None,  # missing!
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
        )
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_COVARIANCE_EPOCH_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_exact_covariance_transport_hand_calculation(self):
        var_t0 = 1e-4
        var_p = 1e-8
        cov_t0_p = 8e-7
        p = 10.0
        t0_pub = 2000.0

        # Hand derivation for N = 100:
        # Var(T0 + 100*P) = 0.00036 d^2
        # Width with SMEARED_OBSERVED: 153.9662369515644 min.
        t_eval = np.array([3000.0])
        res_before = ksd.evaluate_known_signal_geometry_diagnostics(
            t_eval,
            period_days=p,
            epoch_btjd=t0_pub,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            var_t0_days2=var_t0,
            cov_t0_p_days2=cov_t0_p,
            var_p_days2=var_p,
            covariance_epoch_btjd=t0_pub,
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
        )
        self.assertEqual(res_before["status"], "ACTIVE_MASK_VALID_EXACT_COVARIANCE")
        self.assertAlmostEqual(res_before["required_half_width_median_minutes"], 153.9662369515644, places=10)

        # Transport covariance by m = 500 cycles (T0' = 2000 + 500*10 = 7000)
        tc = ksd.transform_ephemeris_epoch_covariance(
            t0_pub, p, var_t0, cov_t0_p, var_p, 500, epoch_identity="SOL_EPOCH_2000"
        )
        self.assertAlmostEqual(tc.epoch_btjd, 7000.0)
        self.assertEqual(tc.epoch_identity, "SOL_EPOCH_2000")
        self.assertEqual(tc.cycle_shift_m, 500)

        t0_new, var_t0_new, cov_new, var_p_new = tc

        # Evaluate after shift at same evaluation time t = 3000
        res_after = ksd.evaluate_known_signal_geometry_diagnostics(
            t_eval,
            period_days=p,
            epoch_btjd=t0_new,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            var_t0_days2=var_t0_new,
            cov_t0_p_days2=cov_new,
            var_p_days2=var_p_new,
            covariance_epoch_btjd=t0_new,
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
        )
        self.assertEqual(res_after["status"], "ACTIVE_MASK_VALID_EXACT_COVARIANCE")
        self.assertAlmostEqual(res_after["required_half_width_median_minutes"], 153.9662369515644, places=10)

    def test_cauchy_schwarz_hand_calculation_and_no_silent_fallback(self):
        t_eval = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(
            t_eval,
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertAlmostEqual(res["required_half_width_median_minutes"], 332.2, places=8)

        # Missing published epoch metadata must NOT silently fall back to current T0
        res_missing = ksd.evaluate_known_signal_geometry_diagnostics(
            t_eval,
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=None,  # missing
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        self.assertEqual(res_missing["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertFalse(res_missing["mask_valid"])


class TestSurvivingControlsAndProvenance(unittest.TestCase):
    """
    Acceptance Criterion A4:
    New surviving controls rejected:
      - publishedT0 = 2000, currentT0 = 2004, P = 10 with copied original errors (non-integer rebase)
      - provenance string 'False'
      - conflicting solution/error epoch identity and incomplete covariance.
    Correct explicit coherent controls succeed.
    """

    def setUp(self):
        self.base_kwargs = dict(
            period_days=10.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )

    def test_non_integer_rebase_rejected(self):
        t = np.array([2004.0])
        kw = self.base_kwargs.copy()
        kw.update(
            epoch_btjd=2004.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
        )
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_provenance_string_false_rejected(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw.update(
            epoch_btjd=2000.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch="False",
        )
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_covariance_epoch_binding_mismatch_rejected(self):
        t = np.array([2004.0])
        kw = self.base_kwargs.copy()
        kw.update(
            epoch_btjd=2004.0,
            covariance_epoch_btjd=2000.0,
            var_t0_days2=1e-4,
            cov_t0_p_days2=0.0,
            var_p_days2=1e-8,
        )
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "COVARIANCE_EPOCH_MISMATCH")
        self.assertFalse(res["mask_valid"])

    def test_covariance_solution_mismatch_rejected(self):
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw.update(
            epoch_btjd=2000.0,
            covariance_epoch_btjd=2000.0,
            var_t0_days2=1e-4,
            cov_t0_p_days2=0.0,
            var_p_days2=1e-8,
            solution_id="SOL_ORIGINAL",
            covariance_solution_id="SOL_DIFFERENT",
        )
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "COVARIANCE_SOLUTION_MISMATCH")
        self.assertFalse(res["mask_valid"])


class TestCovariancePSDAndNumericalStability(unittest.TestCase):
    """
    Acceptance Criterion A5:
    2x2 PSD covariance matrix validation:
      - Non-PSD zero-diagonal [[0, -5e-13], [-5e-13, 0]] rejected.
      - Ordinary rho > 1 rejected.
      - Extreme underflow var = 1e-200, cov = 2e-200 rejected.
      - Rank-deficient valid matrices (rho = 1.0) accepted.
      - Negative-correlation cancellation near pivot verified.
      - Derived overflow failures documented.
    """

    def test_zero_diagonal_covariance_rejection(self):
        self.assertFalse(ksd.is_psd_covariance_2x2(0.0, -5e-13, 0.0))
        self.assertFalse(ksd.is_psd_covariance_2x2(0.0, 1e-5, 1e-4))
        self.assertTrue(ksd.is_psd_covariance_2x2(0.0, 0.0, 1e-4))
        self.assertTrue(ksd.is_psd_covariance_2x2(0.0, 0.0, 0.0))

    def test_ordinary_correlation_greater_than_one(self):
        self.assertFalse(ksd.is_psd_covariance_2x2(1e-4, 2e-6, 1e-8))

    def test_extreme_underflow_scale_rejection(self):
        self.assertFalse(ksd.is_psd_covariance_2x2(1e-200, 2e-200, 1e-200))
        self.assertFalse(ksd.is_psd_covariance_2x2(1e-200, 0.0, 1e-200))

    def test_rank_deficient_valid_matrix(self):
        v0 = 1.0
        vp = 0.01
        cov = math.sqrt(v0 * vp)  # 0.1
        self.assertTrue(ksd.is_psd_covariance_2x2(v0, cov, vp))

    def test_negative_correlation_pivot_cancellation(self):
        tc = ksd.transform_ephemeris_epoch_covariance(2000.0, 10.0, 1.0, -0.1, 0.01, 10)
        self.assertAlmostEqual(tc.epoch_btjd, 2100.0)
        self.assertAlmostEqual(tc.var_t0_days2, 0.0)
        self.assertAlmostEqual(tc.cov_t0_p_days2, 0.0)
        self.assertAlmostEqual(tc.var_p_days2, 0.01)

    def test_overflow_protection(self):
        with self.assertRaises(ValueError):
            ksd.transform_ephemeris_epoch_covariance(2000.0, 10.0, 1e160, 0.0, 1e160, 10)


class TestWidthPolicyAndBudgetConstraints(unittest.TestCase):
    """
    Acceptance Criterion A6 & Review Finding R4:
      - P = 10, D = 0.1, sigma_T0 = 0.1, t = 2000.2 => width 505 min with 120s exposure.
      - Explicit 0.15 d budget returns unavailable rather than clipped valid.
      - No scientific default whitelist: allowed_dispositions must be supplied explicitly.
      - No arbitrary default 0.25P / 0.5P cutoffs: applied only when caller requests them.
    """

    def setUp(self):
        self.base_kwargs = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.1,
            sigma_p_days=0.0,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )

    def test_505_min_half_width_and_containment(self):
        t = np.array([2000.2])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **self.base_kwargs)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertTrue(res["mask_valid"])
        self.assertTrue(res["mask"][0])
        self.assertAlmostEqual(res["required_half_width_median_minutes"], 505.0, places=7)

    def test_budget_exceeded_returns_unavailable_not_clipped(self):
        t = np.array([2000.2])
        kw = self.base_kwargs.copy()
        kw["max_stage_half_width_days"] = 0.15
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_TIMING_PRECISION_FOR_STAGE")
        self.assertFalse(res["mask_valid"])
        self.assertFalse(res["mask"][0])

    def test_caller_supplied_cycle_ambiguity_gate(self):
        # Finding R4: ambiguity gate applied only when caller supplies max_cycle_ambiguity_fraction
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        # k_sigma * sig_pred = 3.0 * 1.0 = 3.0 = 0.3 * P (P=10)
        kw.update(sigma_t0_days=1.0, covariance_epoch_btjd=2000.0)

        # Without caller gate: evaluates without arbitrary failure
        res_no_gate = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res_no_gate["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertAlmostEqual(res_no_gate["cycle_ambiguity_fraction"], 0.3)

        # With explicit caller gate of 0.25: triggers structured unavailable
        kw["max_cycle_ambiguity_fraction"] = 0.25
        res_with_gate = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res_with_gate["status"], "AMBIGUOUS_CYCLE_COUNT_DEGENERATE")
        self.assertFalse(res_with_gate["mask_valid"])

    def test_no_default_whitelist(self):
        # Finding R4: without caller-supplied allowed_dispositions, no disposition is granted
        t = np.array([2000.0])
        kw = self.base_kwargs.copy()
        kw["allowed_dispositions"] = None  # no explicit whitelist
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "DISPOSITION_NOT_PERMITTED_FOR_ACTIVE_MASK")
        self.assertFalse(res["mask_valid"])


class TestIntervalUnionAndCoverage(unittest.TestCase):
    """
    Acceptance Criterion A7 & Review Finding R3:
      - Window [0, 1], exposures [-0.05, 0.15], [0.85, 1.05] => G = 0.3.
      - Duplicate [0, 1] => 1.
      - Union clean C <= G <= 1 on valid inputs.
      - Non-finite exposure bounds return structured unavailable (never silently dropped).
      - clean_mask=['False'] rejected as INVALID_CLEAN_MASK.
      - Excluded observation time fraction and counts reported.
    """

    def test_lebesgue_window_coverage_g_0_3(self):
        window = (0.0, 1.0)
        exposures = [(-0.05, 0.15), (0.85, 1.05)]
        res = ksd.compute_window_interval_coverage(exposures, window)
        self.assertTrue(res["coverage_valid"])
        self.assertAlmostEqual(res["G"], 0.3, places=10)
        self.assertAlmostEqual(res["C"], 0.3, places=10)

    def test_duplicate_exposures_deduplication_g_1_0(self):
        window = (0.0, 1.0)
        exposures = [(0.0, 1.0), (0.0, 1.0)]
        res = ksd.compute_window_interval_coverage(exposures, window)
        self.assertTrue(res["coverage_valid"])
        self.assertAlmostEqual(res["G"], 1.0, places=10)
        self.assertAlmostEqual(res["C"], 1.0, places=10)
        self.assertEqual(res["raw_exposure_count"], 2)
        self.assertEqual(res["deduplicated_exposure_count"], 1)

    def test_non_finite_exposure_not_silently_skipped(self):
        # Finding R3: non-finite exposure must NOT be silently dropped
        window = (0.0, 1.0)
        exposures = [(0.0, 1.0), (float("nan"), 2.0)]
        res = ksd.compute_window_interval_coverage(exposures, window)
        self.assertEqual(res["status"], "NON_FINITE_EXPOSURE_BOUNDS")
        self.assertFalse(res["coverage_valid"])
        self.assertIsNone(res["G"])
        self.assertIsNone(res["C"])
        self.assertEqual(res["raw_exposure_count"], 2)

    def test_clean_mask_string_false_rejected(self):
        # Finding R3: clean_mask=['False'] must be rejected as INVALID_CLEAN_MASK
        window = (0.0, 1.0)
        exposures = [(0.0, 1.0)]
        res = ksd.compute_window_interval_coverage(exposures, window, clean_mask=["False"])
        self.assertEqual(res["status"], "INVALID_CLEAN_MASK")
        self.assertFalse(res["coverage_valid"])

    def test_excluded_time_metrics(self):
        # Finding R3: verify excluded observation time fraction
        window = (0.0, 1.0)
        exposures = [(0.0, 1.0)]
        res = ksd.compute_window_interval_coverage(exposures, window, clean_mask=[False])
        self.assertTrue(res["coverage_valid"])
        self.assertEqual(res["G"], 1.0)
        self.assertEqual(res["C"], 0.0)
        self.assertEqual(res["excluded_observation_time_days"], 1.0)
        self.assertEqual(res["excluded_observation_time_fraction"], 1.0)

    def test_zero_or_negative_window_length(self):
        res_zero = ksd.compute_window_interval_coverage([(0.0, 1.0)], (1.0, 1.0))
        self.assertEqual(res_zero["status"], "ZERO_OR_NEGATIVE_WINDOW_DENOMINATOR")
        self.assertFalse(res_zero["coverage_valid"])


class TestDeliberateFaultInjections(unittest.TestCase):
    """
    Acceptance Criterion A8 & Review Finding R5:
    Real-path mutation testing using scoped patches of actual called helpers/code.
    Demonstrates that acceptance checks detect and kill relevant introduced faults:
      - Fault 1: Covariance zero-fallback mutation (underestimating variance when rho > 0).
      - Fault 2: Provenance truthiness mutation (accepting string 'False' as True).
      - Fault 3: Interval union summed lengths mutation (overcounting overlapping coverage).
      - Fault 4: Production non-activation permission mutation (violating DIAGNOSTIC_ONLY invariant).
    """

    def test_fault_1_covariance_zero_fallback(self):
        kw = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            covariance_solution_id="FIT_001",
            var_t0_days2=1e-4,
            cov_t0_p_days2=None,  # missing!
            var_p_days2=1e-8,
            covariance_epoch_btjd=2000.0,
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
        )
        t = np.array([3000.0])

        # Positive control (unmutated production path):
        res_clean = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res_clean["status"], "INSUFFICIENT_UNCERTAINTY_METADATA")
        self.assertFalse(res_clean["mask_valid"])

        # Injected Fault: mutant silently falls back to cov_t0_p_days2 = 0.0 when missing
        real_evaluate = ksd.evaluate_signal_mask_contract
        def mutant_evaluate(*args, **call_kw):
            if call_kw.get("cov_t0_p_days2") is None and call_kw.get("var_t0_days2") is not None:
                call_kw = call_kw.copy()
                call_kw["cov_t0_p_days2"] = 0.0  # BUG: silent zero fallback
            return real_evaluate(*args, **call_kw)

        with mock.patch("known_signal_diagnostics.evaluate_signal_mask_contract", side_effect=mutant_evaluate):
            res_mutant = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            with self.assertRaises(AssertionError):
                self.assertEqual(res_mutant["status"], "INSUFFICIENT_UNCERTAINTY_METADATA")

    def test_fault_2_provenance_truthiness_mutation(self):
        kw = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch="False",  # string!
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        t = np.array([2000.0])

        # Positive control (unmutated production path):
        res_clean = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res_clean["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertFalse(res_clean["mask_valid"])

        # Injected Fault: mutant coerces provenance flag via truthiness bool(prov)
        real_evaluate = ksd.evaluate_signal_mask_contract
        def mutant_evaluate(*args, **call_kw):
            call_kw = call_kw.copy()
            prov = call_kw.get("marginal_errors_at_published_epoch")
            if isinstance(prov, str):
                call_kw["marginal_errors_at_published_epoch"] = bool(prov)  # BUG: 'False' becomes True!
            return real_evaluate(*args, **call_kw)

        with mock.patch("known_signal_diagnostics.evaluate_signal_mask_contract", side_effect=mutant_evaluate):
            res_mutant = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            with self.assertRaises(AssertionError):
                self.assertEqual(res_mutant["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")

    def test_fault_3_summed_exposure_lengths_mutation(self):
        exposures = [(0.0, 1.0), (0.0, 1.0)]
        window = (0.0, 1.0)

        # Positive control:
        res_clean = ksd.compute_window_interval_coverage(exposures, window)
        self.assertEqual(res_clean["G"], 1.0)

        # Injected Fault: mutant replaces measure_interval_union with sum of interval lengths
        def mutant_measure_interval_union(intervals):
            return sum(b - a for a, b in intervals)

        with mock.patch("known_signal_diagnostics.measure_interval_union", side_effect=mutant_measure_interval_union):
            res_mutant = ksd.compute_window_interval_coverage(exposures, window)
            with self.assertRaises(AssertionError):
                self.assertEqual(res_mutant["G"], 1.0)

    def test_fault_4_production_mask_authorization_mutation(self):
        kw = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            solution_id="FIT_001",
            marginal_errors_solution_id="FIT_001",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        t = np.array([2000.0])

        # Positive control:
        res_clean = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertIs(res_clean["production_mask_allowed"], False)
        self.assertEqual(res_clean["mode"], "DIAGNOSTIC_ONLY")

        # Injected Fault: mutant sets production_mask_allowed = True
        real_evaluate = ksd.evaluate_signal_mask_contract
        def mutant_evaluate(*args, **call_kw):
            r = real_evaluate(*args, **call_kw)
            r["production_mask_allowed"] = True  # BUG
            return r

        with mock.patch("known_signal_diagnostics.evaluate_signal_mask_contract", side_effect=mutant_evaluate):
            res_mutant = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            with self.assertRaises(AssertionError):
                self.assertIs(res_mutant["production_mask_allowed"], False)


class TestRound3ReviewFindingsS1ToS4(unittest.TestCase):
    """
    Dedicated test suite for Round 2 findings S1–S4:
      - S1: Operation/ULP-aware integer rebase tolerance rejects unmodeled phase errors
            (e.g. 7.776s shift), while safely accepting exact integer shifts on large BJDs,
            zero, and negative epochs.
      - S2: Explicit provenance fields, covariance/marginal error solution ID bindings,
            normalized time format/scale declarations, and roundtrip recalculation from saved record.
      - S3: Input hardening (malformed lists, non-scalars), derived finite-overflow guards
            (e.g. k_sigma=1e308), and strict JSON compliance without Inf/NaN.
      - S4: Interval coverage zero observed-time denominator handling, separate exposure
            deduplication count vs merged connected components, and touching frames.
    """

    def setUp(self):
        self.base_cov_kw = dict(
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            has_unmodeled_ttv=False,
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
            var_t0_days2=1e-4,
            cov_t0_p_days2=8e-7,
            var_p_days2=1e-8,
            covariance_epoch_btjd=2000.0,
            signal_id="s1",
            solution_id="fit1",
            source_snapshot_id="snap1",
            covariance_solution_id="fit1",
        )

    # --- S1: Integer Rebase Tolerance ---

    def test_s1_tiny_rebase_false_permission_rejected(self):
        # Epochs differ by 7.776 seconds on P=10d => delta/P = 9e-6 > 1e-11
        kw = self.base_cov_kw.copy()
        kw.update(
            epoch_btjd=2000.00009,
            covariance_epoch_btjd=2000.0,
            duration_days=0.00001,
            var_t0_days2=0.0,
            cov_t0_p_days2=0.0,
            var_p_days2=0.0,
        )
        t = np.array([2000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "COVARIANCE_EPOCH_MISMATCH")
        self.assertFalse(res["mask_valid"])
        self.assertTrue(any("Non-integer epoch rebase" in r for r in res["reasons"]))

    def test_s1_integer_rebase_large_bjd_accepted(self):
        # Large BJD shift: 2000 cycles shift with P=3.5d
        kw = self.base_cov_kw.copy()
        kw.update(
            period_days=3.5,
            epoch_btjd=2457000.0,
            covariance_epoch_btjd=2450000.0,
            covariance_solution_id="fit1",
            solution_id="fit1",
        )
        t = np.array([2457000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["geometry_status"], "VALID_EXACT_COVARIANCE")
        self.assertTrue(res["mask_valid"])

    def test_s1_integer_rebase_epoch_zero_and_negative_accepted(self):
        # Epoch 0.0 and negative epoch -20.0
        kw = self.base_cov_kw.copy()
        kw.update(
            period_days=10.0,
            epoch_btjd=-20.0,
            covariance_epoch_btjd=0.0,
        )
        t = np.array([-20.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["geometry_status"], "VALID_EXACT_COVARIANCE")
        self.assertTrue(res["mask_valid"])

    def test_s1_check_integer_rebase_ulp_bounds(self):
        # ULP boundary tests directly on helper
        ok, m, diff, tol = ksd.check_integer_rebase(2000.00009, 2000.0, 10.0)
        self.assertFalse(ok)

        ok, m, diff, tol = ksd.check_integer_rebase(2000.0, 2000.0, 10.0)
        self.assertTrue(ok)
        self.assertEqual(m, 0)
        self.assertEqual(diff, 0.0)

        ok, m, diff, tol = ksd.check_integer_rebase(2450000.0 + 500 * 2.5, 2450000.0, 2.5)
        self.assertTrue(ok)
        self.assertEqual(m, 500)

    # --- S2: Provenance and Solution Binding ---

    def test_s2_missing_covariance_solution_provenance_rejected(self):
        kw = self.base_cov_kw.copy()
        kw["covariance_solution_id"] = None
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_COVARIANCE_SOLUTION_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_s2_missing_marginal_errors_solution_provenance_rejected(self):
        kw = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            solution_id="fit1",
            marginal_errors_solution_id=None,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        t = np.array([2000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_MARGINAL_ERRORS_SOLUTION_PROVENANCE")
        self.assertFalse(res["mask_valid"])

    def test_s2_marginal_errors_solution_mismatch_rejected(self):
        kw = dict(
            period_days=10.0,
            epoch_btjd=2000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            sigma_dur_days=0.0,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            solution_id="fit1",
            marginal_errors_solution_id="fit2",
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
        )
        t = np.array([2000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MARGINAL_ERRORS_SOLUTION_MISMATCH")
        self.assertFalse(res["mask_valid"])

    def test_s2_normalized_time_declaration_required(self):
        kw = self.base_cov_kw.copy()
        kw["normalized_time_format"] = "BJD"  # Not normalized BTJD
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
        self.assertFalse(res["mask_valid"])

    def test_s2_roundtrip_recalculation_from_saved_record(self):
        t = np.array([7000.0, 7000.05, 7005.0])
        res1 = ksd.evaluate_known_signal_geometry_diagnostics(t, **self.base_cov_kw)
        self.assertTrue(res1["mask_valid"])

        # Reconstruct call strictly from saved fields in res1
        reconstructed_kw = dict(
            period_days=res1["period_days"],
            epoch_btjd=res1["epoch_btjd"],
            duration_days=res1["duration_days"],
            raw_disposition=res1["raw_disposition"],
            stage_policy_action=res1["stage_policy_action"],
            allowed_dispositions=res1["allowed_dispositions"],
            normalized_time_format=res1["normalized_time_format"],
            normalized_time_scale=res1["normalized_time_scale"],
            original_time_format=res1["original_time_format"],
            original_time_scale=res1["original_time_scale"],
            ephemeris_coherence_status=res1["ephemeris_coherence_status"],
            has_unmodeled_ttv=res1["has_unmodeled_ttv"],
            var_t0_days2=res1["var_t0_days2"],
            cov_t0_p_days2=res1["cov_t0_p_days2"],
            var_p_days2=res1["var_p_days2"],
            covariance_epoch_btjd=res1["covariance_epoch_btjd"],
            covariance_solution_id=res1["covariance_solution_id"],
            sigma_t0_days=res1["sigma_t0_days"],
            sigma_p_days=res1["sigma_p_days"],
            sigma_dur_days=res1["sigma_dur_days"],
            published_epoch_btjd=res1["published_epoch_btjd"],
            marginal_errors_at_published_epoch=res1["marginal_errors_at_published_epoch"],
            marginal_errors_solution_id=res1["marginal_errors_solution_id"],
            duration_frame_type=res1["duration_frame_type"],
            exp_time_days=res1["exp_time_days"],
            k_sigma=res1["k_sigma"],
            k_dur=res1["k_dur"],
            alpha_stage=res1["alpha_stage"],
            max_stage_half_width_days=res1["max_stage_half_width_days"],
            max_cycle_ambiguity_fraction=res1["max_cycle_ambiguity_fraction"],
            max_half_orbit_fraction=res1["max_half_orbit_fraction"],
            signal_id=res1["signal_id"],
            solution_id=res1["solution_id"],
            source_snapshot_id=res1["source_snapshot_id"],
        )
        res2 = ksd.evaluate_known_signal_geometry_diagnostics(t, **reconstructed_kw)
        self.assertEqual(res1["status"], res2["status"])
        self.assertEqual(res1["mask"], res2["mask"])
        self.assertEqual(res1["required_half_width_median_minutes"], res2["required_half_width_median_minutes"])
        self.assertEqual(res1["time_mask_fraction"], res2["time_mask_fraction"])

    # --- S3: Input Hardening and Derived Finite Overflow ---

    def test_s3_malformed_policy_list_handled_safely(self):
        kw = self.base_cov_kw.copy()
        kw["k_sigma"] = []  # list instead of numeric scalar
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INVALID_POLICY_PARAMETERS")
        self.assertFalse(res["mask_valid"])
        # Strict JSON dump must succeed without exception
        dumped = json.dumps(res, allow_nan=False)
        self.assertIsInstance(dumped, str)

    def test_s3_overflow_finite_inputs_strict_json(self):
        kw = self.base_cov_kw.copy()
        kw["k_sigma"] = 1e308  # Finite float that causes scaled width to overflow
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "DERIVED_QUANTITY_OVERFLOW")
        self.assertEqual(res["geometry_status"], "DERIVED_NUMERICAL_OVERFLOW")
        self.assertFalse(res["mask_valid"])
        # Guaranteed strict JSON compliance (no Inf or NaN)
        dumped = json.dumps(res, allow_nan=False)
        self.assertIsInstance(dumped, str)

    # --- S4: Interval Denominator & Exposure Deduplication ---

    def test_s4_zero_observed_time_fraction_undefined(self):
        res = ksd.compute_window_interval_coverage([], (0.0, 1.0))
        self.assertTrue(res["coverage_valid"])
        self.assertEqual(res["G"], 0.0)
        self.assertEqual(res["C"], 0.0)
        self.assertIsNone(res["excluded_observation_time_fraction"])
        self.assertEqual(res["raw_exposure_count"], 0)
        self.assertEqual(res["deduplicated_exposure_count"], 0)
        self.assertEqual(res["merged_interval_count"], 0)
        self.assertTrue(any("Total observed time in window is zero" in r for r in res["reasons"]))

    def test_s4_touching_frames_vs_duplicate_exposures(self):
        # Touching exposures: two separate exposures sharing an endpoint
        touching = [(0.0, 0.5), (0.5, 1.0)]
        res_touching = ksd.compute_window_interval_coverage(touching, (0.0, 1.0))
        self.assertEqual(res_touching["raw_exposure_count"], 2)
        self.assertEqual(res_touching["deduplicated_exposure_count"], 2)
        self.assertEqual(res_touching["merged_interval_count"], 1)  # 1 contiguous component

        # Duplicate exposures: identical intervals
        duplicate = [(0.0, 0.5), (0.0, 0.5)]
        res_dup = ksd.compute_window_interval_coverage(duplicate, (0.0, 1.0))
        self.assertEqual(res_dup["raw_exposure_count"], 2)
        self.assertEqual(res_dup["deduplicated_exposure_count"], 1)  # 1 unique exposure
        self.assertEqual(res_dup["merged_interval_count"], 1)

    def test_s4_zero_duration_exposures_in_coverage(self):
        res = ksd.compute_window_interval_coverage([(0.5, 0.5)], (0.0, 1.0))
        self.assertTrue(res["coverage_valid"])
        self.assertEqual(res["G"], 0.0)
        self.assertIsNone(res["excluded_observation_time_fraction"])


class TestRound4ReviewFindingsT1AndT2(unittest.TestCase):
    """
    Dedicated test suite for Round 3 findings T1 and T2:
      - T1: Explicit normalized-time declaration gate (absence of normalized_time_format
            or normalized_time_scale defaults to None and strictly fails with
            UNSAFE_OR_UNKNOWN_TIME_SCALE / UNSAFE_TIME_SCALE / mask_valid=False).
            Caller must explicitly provide BTJD and TDB.
      - T2: Explicit solution identity and provenance binding in both uncertainty branches
            (Exact Covariance and Cauchy-Schwarz Bound). Missing, blank, whitespace-only,
            or boolean solution IDs must fail with INSUFFICIENT_UNCERTAINTY_PROVENANCE.
            Unidentified inputs cannot prove fitted errors by absence of metadata.
    """

    def setUp(self):
        # Base valid parameters for exact covariance branch
        self.base_cov_kw = dict(
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            has_unmodeled_ttv=False,
            sigma_dur_days=0.0,
            duration_frame_type="SMEARED_OBSERVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=0.0,
            alpha_stage=1.0,
            var_t0_days2=1e-4,
            cov_t0_p_days2=8e-7,
            var_p_days2=1e-8,
            covariance_epoch_btjd=2000.0,
            signal_id="s1",
            solution_id="fit1",
            source_snapshot_id="snap1",
            covariance_solution_id="fit1",
        )

        # Base valid parameters for Cauchy-Schwarz bound branch
        self.base_cs_kw = dict(
            period_days=10.0,
            epoch_btjd=7000.0,
            duration_days=0.1,
            raw_disposition="CP",
            stage_policy_action="ALLOW_ACTIVE_MASK",
            allowed_dispositions=("CP",),
            normalized_time_format="BTJD",
            normalized_time_scale="TDB",
            original_time_format="BTJD",
            original_time_scale="TDB",
            ephemeris_coherence_status="COHERENT",
            has_unmodeled_ttv=False,
            sigma_dur_days=0.0,
            duration_frame_type="INTRINSIC_DECONVOLVED",
            exp_time_days=120.0 / 86400.0,
            k_sigma=3.0,
            k_dur=1.0,
            alpha_stage=1.0,
            sigma_t0_days=0.01,
            sigma_p_days=0.0001,
            published_epoch_btjd=2000.0,
            marginal_errors_at_published_epoch=True,
            signal_id="s1",
            solution_id="fit1",
            source_snapshot_id="snap1",
            marginal_errors_solution_id="fit1",
        )

    # --- T1: Explicit Normalized Time Declaration Gate ---

    def test_t1_omitted_both_normalized_declarations_rejected(self):
        kw = self.base_cov_kw.copy()
        kw.pop("normalized_time_format")
        kw.pop("normalized_time_scale")
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
        self.assertEqual(res["geometry_status"], "UNSAFE_TIME_SCALE")
        self.assertFalse(res["mask_valid"])
        self.assertIsNone(res["normalized_time_format"])
        self.assertIsNone(res["normalized_time_scale"])
        self.assertTrue(any("Requires explicit normalized BTJD format with TDB scale" in r for r in res["reasons"]))
        # Direct JSON compliance
        dumped = json.dumps(res, allow_nan=False)
        self.assertIsInstance(dumped, str)

    def test_t1_omitted_normalized_format_alone_rejected(self):
        kw = self.base_cov_kw.copy()
        kw.pop("normalized_time_format")
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
        self.assertEqual(res["geometry_status"], "UNSAFE_TIME_SCALE")
        self.assertFalse(res["mask_valid"])
        self.assertIsNone(res["normalized_time_format"])
        self.assertEqual(res["normalized_time_scale"], "TDB")

    def test_t1_omitted_normalized_scale_alone_rejected(self):
        kw = self.base_cov_kw.copy()
        kw.pop("normalized_time_scale")
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
        self.assertEqual(res["geometry_status"], "UNSAFE_TIME_SCALE")
        self.assertFalse(res["mask_valid"])
        self.assertEqual(res["normalized_time_format"], "BTJD")
        self.assertIsNone(res["normalized_time_scale"])

    def test_t1_explicit_none_normalized_declarations_rejected(self):
        kw = self.base_cov_kw.copy()
        kw["normalized_time_format"] = None
        kw["normalized_time_scale"] = None
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
        self.assertEqual(res["geometry_status"], "UNSAFE_TIME_SCALE")
        self.assertFalse(res["mask_valid"])

    def test_t1_invalid_strings_normalized_declarations_rejected(self):
        t = np.array([7000.0])
        for fmt, scale in [("BJD", "TDB"), ("BTJD", "UTC"), ("JD", "TT"), ("", ""), ("   ", "   ")]:
            kw = self.base_cov_kw.copy()
            kw["normalized_time_format"] = fmt
            kw["normalized_time_scale"] = scale
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "UNSAFE_OR_UNKNOWN_TIME_SCALE")
            self.assertEqual(res["geometry_status"], "UNSAFE_TIME_SCALE")
            self.assertFalse(res["mask_valid"])

    def test_t1_explicit_valid_normalized_declarations_accepted(self):
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **self.base_cov_kw)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_EXACT_COVARIANCE")
        self.assertEqual(res["geometry_status"], "VALID_EXACT_COVARIANCE")
        self.assertTrue(res["mask_valid"])
        self.assertEqual(res["normalized_time_format"], "BTJD")
        self.assertEqual(res["normalized_time_scale"], "TDB")

    # --- T2: Solution Identity and Association Gate: Covariance Branch ---

    def test_t2_covariance_both_solution_ids_none_rejected(self):
        kw = self.base_cov_kw.copy()
        kw["solution_id"] = None
        kw["covariance_solution_id"] = None
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
        self.assertFalse(res["mask_valid"])
        self.assertTrue(any("Missing or blank ephemeris solution_id" in r for r in res["reasons"]))
        dumped = json.dumps(res, allow_nan=False)
        self.assertIsInstance(dumped, str)

    def test_t2_covariance_both_solution_ids_blank_string_rejected(self):
        for s_id, cov_id in [("", ""), ("   ", "   "), ("\t", "\n")]:
            kw = self.base_cov_kw.copy()
            kw["solution_id"] = s_id
            kw["covariance_solution_id"] = cov_id
            t = np.array([7000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
            self.assertFalse(res["mask_valid"])

    def test_t2_covariance_missing_covariance_solution_id_rejected(self):
        for cov_id in [None, "", "   "]:
            kw = self.base_cov_kw.copy()
            kw["solution_id"] = "fit1"
            kw["covariance_solution_id"] = cov_id
            t = np.array([7000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res["geometry_status"], "MISSING_COVARIANCE_SOLUTION_PROVENANCE")
            self.assertFalse(res["mask_valid"])

    def test_t2_covariance_missing_ephemeris_solution_id_rejected(self):
        for s_id in [None, "", "   "]:
            kw = self.base_cov_kw.copy()
            kw["solution_id"] = s_id
            kw["covariance_solution_id"] = "fit1"
            t = np.array([7000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
            self.assertFalse(res["mask_valid"])

    def test_t2_covariance_mismatched_solution_ids_rejected(self):
        kw = self.base_cov_kw.copy()
        kw["solution_id"] = "fit1"
        kw["covariance_solution_id"] = "fit2"
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "COVARIANCE_SOLUTION_MISMATCH")
        self.assertFalse(res["mask_valid"])

    def test_t2_covariance_matching_solution_ids_accepted(self):
        kw = self.base_cov_kw.copy()
        kw["solution_id"] = "fit_alpha"
        kw["covariance_solution_id"] = "fit_alpha"
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_EXACT_COVARIANCE")
        self.assertEqual(res["geometry_status"], "VALID_EXACT_COVARIANCE")
        self.assertTrue(res["mask_valid"])

    # --- T2: Solution Identity and Association Gate: Cauchy-Schwarz Branch ---

    def test_t2_cs_both_solution_ids_none_rejected(self):
        kw = self.base_cs_kw.copy()
        kw["solution_id"] = None
        kw["marginal_errors_solution_id"] = None
        kw["published_solution_id"] = None
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
        self.assertFalse(res["mask_valid"])
        self.assertTrue(any("Missing or blank ephemeris solution_id" in r for r in res["reasons"]))
        dumped = json.dumps(res, allow_nan=False)
        self.assertIsInstance(dumped, str)

    def test_t2_cs_both_solution_ids_blank_string_rejected(self):
        for s_id, marg_id in [("", ""), ("   ", "   "), ("\t", "\n")]:
            kw = self.base_cs_kw.copy()
            kw["solution_id"] = s_id
            kw["marginal_errors_solution_id"] = marg_id
            kw["published_solution_id"] = None
            t = np.array([7000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
            self.assertFalse(res["mask_valid"])

    def test_t2_cs_missing_marginal_errors_solution_id_rejected(self):
        for marg_id in [None, "", "   "]:
            kw = self.base_cs_kw.copy()
            kw["solution_id"] = "fit1"
            kw["marginal_errors_solution_id"] = marg_id
            kw["published_solution_id"] = None
            t = np.array([7000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res["geometry_status"], "MISSING_MARGINAL_ERRORS_SOLUTION_PROVENANCE")
            self.assertFalse(res["mask_valid"])

    def test_t2_cs_missing_ephemeris_solution_id_rejected(self):
        for s_id in [None, "", "   "]:
            kw = self.base_cs_kw.copy()
            kw["solution_id"] = s_id
            kw["marginal_errors_solution_id"] = "fit1"
            kw["published_solution_id"] = None
            t = np.array([7000.0])
            res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
            self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
            self.assertFalse(res["mask_valid"])

    def test_t2_cs_mismatched_solution_ids_rejected(self):
        kw = self.base_cs_kw.copy()
        kw["solution_id"] = "fit1"
        kw["marginal_errors_solution_id"] = "fit2"
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
        self.assertEqual(res["geometry_status"], "MARGINAL_ERRORS_SOLUTION_MISMATCH")
        self.assertFalse(res["mask_valid"])

    def test_t2_cs_matching_solution_ids_accepted(self):
        kw = self.base_cs_kw.copy()
        kw["solution_id"] = "fit_cs"
        kw["marginal_errors_solution_id"] = "fit_cs"
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertTrue(res["mask_valid"])

    def test_t2_cs_published_solution_id_fallback_when_marginal_is_none(self):
        kw = self.base_cs_kw.copy()
        kw["solution_id"] = "pub_sol_1"
        kw["marginal_errors_solution_id"] = None
        kw["published_solution_id"] = "pub_sol_1"
        t = np.array([7000.0])
        res = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw)
        self.assertEqual(res["status"], "ACTIVE_MASK_VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertEqual(res["geometry_status"], "VALID_CAUCHY_SCHWARZ_UPPER_BOUND")
        self.assertTrue(res["mask_valid"])

    def test_t2_boolean_solution_id_rejected(self):
        for bad_bool in [True, False]:
            kw_cov = self.base_cov_kw.copy()
            kw_cov["solution_id"] = bad_bool
            kw_cov["covariance_solution_id"] = bad_bool
            t = np.array([7000.0])
            res_cov = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw_cov)
            self.assertEqual(res_cov["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res_cov["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
            self.assertFalse(res_cov["mask_valid"])

            kw_cs = self.base_cs_kw.copy()
            kw_cs["solution_id"] = bad_bool
            kw_cs["marginal_errors_solution_id"] = bad_bool
            res_cs = ksd.evaluate_known_signal_geometry_diagnostics(t, **kw_cs)
            self.assertEqual(res_cs["status"], "INSUFFICIENT_UNCERTAINTY_PROVENANCE")
            self.assertEqual(res_cs["geometry_status"], "MISSING_EPHEMERIS_SOLUTION_PROVENANCE")
            self.assertFalse(res_cs["mask_valid"])


if __name__ == "__main__":
    unittest.main()
