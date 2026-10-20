#!/usr/bin/env python3
"""
tests/test_ephemeris_and_provenance.py
======================================
Comprehensive regression and falsification test suite for ALD-006:
Coherent Ephemeris Export, Truthful Archival-Sector Provenance, and Candidate Schema v1.1.

Verifies:
1. Production archive/candidate persistence exercised with distinct discovery and refined P/T0;
   stored pair is strictly coherent from the same solution, and discovery pair is preserved.
2. Independent recomputation of predicted transit times at cycle numbers N in {0, 1, 5, 50, 200, 500}
   demonstrating machine-precision agreement and revealing catastrophic phase drift in mixed-pair fixtures.
3. Negative control / falsification: deliberate mixed-pair mutation (refined_p with discovery_t0)
   demonstrably fails the coherence check, while the corrected control passes.
4. Strict nearby-sector projection (delta_t_arch <= 60 d) and single-sector fallback export
   the coherent discovery pair with explicit fallback reasons.
5. Missing/NaN/invalid refined values cannot produce a mixed pair or fabricated refinement.
6. Truthful archival provenance in verify_multi_epoch: distinguishes requested, attempted, used,
   and failed products/sectors with bounded failure stages (DOWNLOAD, EXTRACTION, CLIPPING, DETRENDING).
7. Duplicate-sector products counted once for unique sector count and proxy SDE calculation.
8. Empty or invalid arrays are never incorporated as used.
9. All-failed / no-prior / query-error outcomes retain distinct context and zero usage.
10. Incompatible historical CSV headers are rejected with RuntimeError without in-place corruption.
11. Primary (aldebaran.py) and companion (search_toi_companions.py) writers both retain schema v1.1.
12. All tests offline with bounded execution and zero external network calls.
"""

import os
import sys
import ast
import json
import signal
import shutil
import tempfile
import unittest
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

import aldebaran as ald
import search_toi_companions as stc


class TestEphemerisSerializationAndCoherence(unittest.TestCase):
    """Test coherent ephemeris serialization, discovery preservation, and timing accuracy."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_ephem_")
        ald.configure_environment(output_dir=self.test_dir, run_id="test_ephem_run")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        ald.configure_environment(output_dir="output", run_id="run_default")

    def test_coherent_refined_export_and_timing_predictions(self):
        """
        Verify that when a valid refined solution is selected, primary period_days and t0_btjd
        come strictly from the refined solution, discovery values are preserved separately,
        and predicted transit times match the refined ephemeris across 500 cycles at machine precision.
        """
        tic_id = "12345678"
        sector = 106
        discovery_p = 3.141592653589793
        discovery_t0 = 2459000.123456789
        refined_p = 3.141890123456789
        refined_t0 = 2458000.654321098

        mock_arch_info = {
            "n_archival_sectors": 2,
            "archival_sectors_str": "80,85",
            "refined_p": refined_p,
            "refined_t0": refined_t0,
            "archival_snr": 8.5,
            "est_multi_sde_proxy": 11.3,
            "multi_sde": 11.3,
            "arch_bin_centers": np.linspace(-0.1, 0.1, 10),
            "arch_bin_means": np.ones(10),
            "n_trials": 12,
            "snr_crit": 3.2,
            "is_projected_strict": False,
            "delta_t_arch_days": 700.0,
            "sigma_p_sec": 1.25,
            "sigma_t0_sec": 45.0,
            "archival_sectors_requested": [80, 85],
            "archival_sectors_attempted": [80, 85],
            "archival_sectors_used": [80, 85],
            "archival_products_requested": ["TESS_s0080_idx0", "TESS_s0085_idx1"],
            "archival_products_attempted": ["TESS_s0080_idx0", "TESS_s0085_idx1"],
            "archival_products_used": ["TESS_s0080_idx0", "TESS_s0085_idx1"],
            "archival_products_failed": [],
            "ephemeris_source": "REFINED_MULTI_SECTOR",
            "ephemeris_status": "REFINED_VALID",
            "ephemeris_fallback_reason": "NONE"
        }

        n_pts = 19440
        rng = np.random.default_rng(42)
        t_arr = np.linspace(2459000, 2459027, n_pts)
        f_arr = 1.0 + 1e-4 * rng.standard_normal(n_pts)
        ph = ((t_arr - discovery_t0 + 0.5 * discovery_p) % discovery_p) - 0.5 * discovery_p
        in_tr = np.abs(ph) < 0.04
        f_arr[in_tr] -= 0.0010

        with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
             patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
             patch.object(ald, "verify_multi_epoch", return_value=(True, "Confirmed_in_Sectors_80,85", mock_arch_info)), \
             patch.object(ald, "BoxLeastSquares") as mock_bls_cls, \
             patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)), \
             patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

            mock_bls = MagicMock()
            mock_bls_cls.return_value = mock_bls
            mock_bls.power.return_value = MagicMock(
                period=np.full(9000, discovery_p),
                power=np.full(9000, 100.0),
                duration=np.full(9000, 0.08),
                transit_time=np.full(9000, discovery_t0),
                depth=np.full(9000, 0.001)
            )

            res = ald.analyze_single_target(tic_id, sector)
            ret_tic, status, msg, cand_data = res

            self.assertEqual(status, "CANDIDATE")
            self.assertIsNotNone(cand_data)

            # 1. Coherent export check
            self.assertEqual(cand_data["period_days"], refined_p)
            self.assertEqual(cand_data["t0_btjd"], refined_t0)
            self.assertEqual(cand_data["discovery_period_days"], discovery_p)
            self.assertEqual(cand_data["discovery_t0_btjd"], discovery_t0)
            self.assertEqual(cand_data["refined_period_days"], refined_p)
            self.assertEqual(cand_data["refined_t0_btjd"], refined_t0)
            self.assertEqual(cand_data["ephemeris_source"], "REFINED_MULTI_SECTOR")
            self.assertEqual(cand_data["ephemeris_status"], "REFINED_VALID")
            self.assertEqual(cand_data["ephemeris_fallback_reason"], "NONE")
            self.assertEqual(cand_data["time_system"], "BTJD")
            self.assertEqual(cand_data["time_standard"], "TDB")
            self.assertEqual(cand_data["epoch_offset_btjd"], 2457000.0)
            self.assertEqual(cand_data["period_units"], "days")
            self.assertEqual(cand_data["duration_units"], "hours")
            self.assertEqual(cand_data["depth_units"], "ppm")
            self.assertEqual(cand_data["transit_measurement_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertEqual(cand_data["timing_uncertainty_source"], "CARTER_2008_DISCOVERY_HEURISTIC")
            self.assertTrue(np.isnan(cand_data["covariance_p_t0"]))

            # 2. Independent derivation and transit time matching at multiple cycle numbers
            cycle_numbers = [0, 1, 5, 50, 200, 500]
            for n in cycle_numbers:
                t_pred_cand = cand_data["t0_btjd"] + n * cand_data["period_days"]
                t_pred_refined = refined_t0 + n * refined_p

                # Stored pair must match the refined solution within float64 ULP precision (~1e-9 d on ~2.46e6 BTJD)
                diff_refined = abs(t_pred_cand - t_pred_refined)
                self.assertLess(diff_refined, 1e-9, f"Transit time at cycle {n} deviated from refined solution by {diff_refined} d")

                # Physical phase check: compare to nearest discovery cycle to evaluate true physical ephemeris divergence
                # (matching cycle origin rather than unaligned cycle index across ~1000 days)
                n_disc = round((t_pred_cand - discovery_t0) / discovery_p)
                t_pred_discovery_aligned = discovery_t0 + n_disc * discovery_p
                drift_from_discovery = abs(t_pred_cand - t_pred_discovery_aligned)
                if n >= 50:
                    self.assertGreater(drift_from_discovery, 0.001, f"Expected physical ephemeris divergence at cycle {n}")

    def test_deliberate_mixed_pair_fault_detected_and_control_passes(self):
        """
        Negative control / falsification:
        Demonstrate that a deliberately mixed-pair candidate (refined_p paired with discovery_t0)
        violates solution coherence and produces large timing errors, whereas the corrected control passes.
        """
        refined_p = 3.141890123456789
        refined_t0 = 2458000.654321098
        discovery_p = 3.141592653589793
        discovery_t0 = 2459000.123456789

        # Helper function acting as an independent validator
        def validate_ephemeris_coherence(row: dict) -> tuple[bool, str]:
            src = row.get("ephemeris_source")
            p = row.get("period_days")
            t0 = row.get("t0_btjd")
            disc_p = row.get("discovery_period_days")
            disc_t0 = row.get("discovery_t0_btjd")
            ref_p = row.get("refined_period_days")
            ref_t0 = row.get("refined_t0_btjd")

            if src == "REFINED_MULTI_SECTOR":
                if p == ref_p and t0 == disc_t0 and ref_t0 != disc_t0:
                    return False, "Fatal mixed-pair defect: refined period paired with discovery epoch!"
                if p != ref_p or t0 != ref_t0:
                    return False, f"Incoherent export: period_days ({p}) or t0_btjd ({t0}) does not match refined pair ({ref_p}, {ref_t0})"
                return True, "Valid coherent refined ephemeris"
            elif src == "DISCOVERY_SINGLE_SECTOR":
                if p != disc_p or t0 != disc_t0:
                    return False, f"Incoherent export: period_days ({p}) or t0_btjd ({t0}) does not match discovery pair ({disc_p}, {disc_t0})"
                return True, "Valid coherent discovery ephemeris"
            return False, f"Unknown ephemeris_source: {src}"

        # 1. Deliberate fault: inject the old bug (refined_p with discovery_t0)
        faulty_mixed_row = {
            "ephemeris_source": "REFINED_MULTI_SECTOR",
            "period_days": refined_p,
            "t0_btjd": discovery_t0,  # BUG: discovery epoch mixed with refined period!
            "discovery_period_days": discovery_p,
            "discovery_t0_btjd": discovery_t0,
            "refined_period_days": refined_p,
            "refined_t0_btjd": refined_t0,
        }
        is_valid_fault, fault_msg = validate_ephemeris_coherence(faulty_mixed_row)
        self.assertFalse(is_valid_fault, "Validator must reject deliberately mixed ephemeris pair")
        self.assertIn("mixed-pair defect", fault_msg)

        # Show that transit predictions from the faulty row drift catastrophically
        t_fault_n500 = faulty_mixed_row["t0_btjd"] + 500 * faulty_mixed_row["period_days"]
        t_true_n500 = refined_t0 + 500 * refined_p
        raw_offset_days = abs(t_fault_n500 - t_true_n500)
        self.assertGreater(raw_offset_days, 999.0, "Mixed pair raw offset must reflect mismatched ~1000-day epoch origin")

        # And after cycle-aligning to the nearest integer transit cycle:
        n_nearest = round((t_fault_n500 - refined_t0) / refined_p)
        t_true_aligned = refined_t0 + n_nearest * refined_p
        phase_drift_days = abs(t_fault_n500 - t_true_aligned)
        self.assertGreater(phase_drift_days, 0.05, "Mixed pair must demonstrate substantial cycle-aligned transit phase error")

        # 2. Corrected control: coherent pair
        correct_control_row = {
            "ephemeris_source": "REFINED_MULTI_SECTOR",
            "period_days": refined_p,
            "t0_btjd": refined_t0,
            "discovery_period_days": discovery_p,
            "discovery_t0_btjd": discovery_t0,
            "refined_period_days": refined_p,
            "refined_t0_btjd": refined_t0,
        }
        is_valid_ctl, ctl_msg = validate_ephemeris_coherence(correct_control_row)
        self.assertTrue(is_valid_ctl, "Validator must pass coherent ephemeris pair")
        self.assertEqual(ctl_msg, "Valid coherent refined ephemeris")

    def test_strict_projection_and_single_sector_fallbacks(self):
        """
        Verify that strict nearby-sector projection (delta_t_arch <= 60 d), single-sector only,
        and archival preparation failures all export the complete, coherent discovery pair.
        """
        tic_id = "23456789"
        sector = 106
        discovery_p = 4.56789123
        discovery_t0 = 2459100.234567

        # Case A: Strict projection (delta_t_arch <= 60 d)
        mock_strict_info = {
            "n_archival_sectors": 1,
            "archival_sectors_str": "105",
            "refined_p": discovery_p,
            "refined_t0": discovery_t0,
            "archival_snr": 6.2,
            "est_multi_sde_proxy": 8.5,
            "multi_sde": 8.5,
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 27.5,
            "sigma_p_sec": 2.1,
            "sigma_t0_sec": 30.0,
            "archival_sectors_requested": [105],
            "archival_sectors_attempted": [105],
            "archival_sectors_used": [105],
            "archival_products_requested": ["TESS_s0105_idx0"],
            "archival_products_attempted": ["TESS_s0105_idx0"],
            "archival_products_used": ["TESS_s0105_idx0"],
            "archival_products_failed": [],
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "STRICT_PROJECTION_NEARBY_SECTOR"
        }

        n_pts = 19440
        rng = np.random.default_rng(42)
        t_arr = np.linspace(2459000, 2459027, n_pts)
        f_arr = 1.0 + 1e-4 * rng.standard_normal(n_pts)
        ph = ((t_arr - discovery_t0 + 0.5 * discovery_p) % discovery_p) - 0.5 * discovery_p
        in_tr = np.abs(ph) < 0.04
        f_arr[in_tr] -= 0.0010

        with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
             patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
             patch.object(ald, "verify_multi_epoch", return_value=(True, "Confirmed_in_Sectors_105", mock_strict_info)), \
             patch.object(ald, "BoxLeastSquares") as mock_bls_cls, \
             patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)), \
             patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

            mock_bls = MagicMock()
            mock_bls_cls.return_value = mock_bls
            mock_bls.power.return_value = MagicMock(
                period=np.full(9000, discovery_p),
                power=np.full(9000, 100.0),
                duration=np.full(9000, 0.08),
                transit_time=np.full(9000, discovery_t0),
                depth=np.full(9000, 0.001)
            )

            _, _, _, cand_strict = ald.analyze_single_target(tic_id, sector)
            self.assertEqual(cand_strict["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertEqual(cand_strict["ephemeris_status"], "FALLBACK_DISCOVERY")
            self.assertEqual(cand_strict["ephemeris_fallback_reason"], "STRICT_PROJECTION_NEARBY_SECTOR")
            self.assertEqual(cand_strict["period_days"], discovery_p)
            self.assertEqual(cand_strict["t0_btjd"], discovery_t0)
            self.assertTrue(np.isnan(cand_strict["refined_period_days"]))
            self.assertTrue(np.isnan(cand_strict["refined_t0_btjd"]))

        # Case B: Missing/NaN refined solution must not export mixed or fabricated refined values
        mock_nan_info = dict(mock_strict_info)
        mock_nan_info.update({
            "is_projected_strict": False,
            "refined_p": np.nan,  # BLS fit failed to refine
            "refined_t0": np.nan,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "REFINED_EPHEMERIS_INVALID"
        })
        with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
             patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
             patch.object(ald, "verify_multi_epoch", return_value=(True, "Confirmed_in_Sectors_105", mock_nan_info)), \
             patch.object(ald, "BoxLeastSquares") as mock_bls_cls, \
             patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)), \
             patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

            mock_bls = MagicMock()
            mock_bls_cls.return_value = mock_bls
            mock_bls.power.return_value = MagicMock(
                period=np.full(9000, discovery_p),
                power=np.full(9000, 100.0),
                duration=np.full(9000, 0.08),
                transit_time=np.full(9000, discovery_t0),
                depth=np.full(9000, 0.001)
            )

            _, _, _, cand_nan = ald.analyze_single_target(tic_id, sector)
            self.assertEqual(cand_nan["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertEqual(cand_nan["ephemeris_status"], "FALLBACK_DISCOVERY")
            self.assertEqual(cand_nan["ephemeris_fallback_reason"], "REFINED_EPHEMERIS_INVALID")
            self.assertEqual(cand_nan["period_days"], discovery_p)
            self.assertEqual(cand_nan["t0_btjd"], discovery_t0)
            self.assertTrue(np.isnan(cand_nan["refined_period_days"]))
            self.assertTrue(np.isnan(cand_nan["refined_t0_btjd"]))

        # Case C: Incomplete / asymmetric refined member (e.g. valid P but NaN T0) falls back to discovery pair
        mock_incomplete_info = dict(mock_strict_info)
        mock_incomplete_info.update({
            "is_projected_strict": False,
            "refined_p": 4.56789123,  # P was refined, but t0 was NaN
            "refined_t0": np.nan,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "REFINED_EPHEMERIS_INVALID"
        })
        with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
             patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
             patch.object(ald, "verify_multi_epoch", return_value=(True, "Confirmed_in_Sectors_105", mock_incomplete_info)), \
             patch.object(ald, "BoxLeastSquares") as mock_bls_cls, \
             patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)), \
             patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

            mock_bls = MagicMock()
            mock_bls_cls.return_value = mock_bls
            mock_bls.power.return_value = MagicMock(
                period=np.full(9000, discovery_p),
                power=np.full(9000, 100.0),
                duration=np.full(9000, 0.08),
                transit_time=np.full(9000, discovery_t0),
                depth=np.full(9000, 0.001)
            )

            _, _, _, cand_inc = ald.analyze_single_target(tic_id, sector)
            self.assertEqual(cand_inc["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertEqual(cand_inc["ephemeris_status"], "FALLBACK_DISCOVERY")
            self.assertEqual(cand_inc["ephemeris_fallback_reason"], "REFINED_EPHEMERIS_INVALID")
            self.assertEqual(cand_inc["period_days"], discovery_p)
            self.assertEqual(cand_inc["t0_btjd"], discovery_t0)
            self.assertTrue(np.isnan(cand_inc["refined_period_days"]))
            self.assertTrue(np.isnan(cand_inc["refined_t0_btjd"]))


class TestTruthfulArchivalProvenance(unittest.TestCase):
    """Test archival provenance accounting in verify_multi_epoch across various failure modes."""

    def _run_control_phase_fixture(self, control_phase: float, shift_periods: int = 0):
        """Run the production archival gate on deterministic transit/control dips."""
        period = 4.0
        t0 = 2000.0 + shift_periods * period
        best_dur = 0.15
        t_arch = np.arange(1600.0, 1628.0, 0.01) + shift_periods * period
        phase = ((t_arch - t0 + 0.5 * period) % period) - 0.5 * period
        transit = np.abs(phase) < 0.045
        control_distance = np.abs(
            ((phase - control_phase * period + 0.5 * period) % period) - 0.5 * period
        )
        control = control_distance < 0.045
        f_arch = np.ones_like(t_arch)
        f_arch[transit] -= 0.001
        f_arch[control] -= 0.002

        t_curr = np.arange(2000.0, 2028.0, 0.01) + shift_periods * period
        phase_curr = ((t_curr - t0 + 0.5 * period) % period) - 0.5 * period
        f_curr = np.ones_like(t_curr)
        f_curr[np.abs(phase_curr) < 0.045] -= 0.001
        ferr_curr = np.full_like(t_curr, 0.0001)

        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "TESS_s0080_control_phase"

        class FixedEphemerisBLS:
            def __init__(self, *args, **kwargs):
                pass

            def power(self, periods, durations):
                return MagicMock(
                    power=np.array([1.0]),
                    period=np.array([period]),
                    transit_time=np.array([t0]),
                )

        with patch("lightkurve.search_lightcurve", return_value=[mock_row]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(
                 t_arch, f_arch, np.full_like(t_arch, 0.0001), np.zeros_like(t_arch),
                 np.zeros_like(t_arch), 0.95
             )), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "BoxLeastSquares", FixedEphemerisBLS), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)):
            result = ald.verify_multi_epoch(
                "34567890", 106, t_curr, f_curr, ferr_curr,
                best_p=period, best_t0=t0, best_dur=best_dur,
                depth_ppm_curr=1000.0, sde_curr=20.0,
            )

        # Independent recomputation from the exact frozen input arrays and selected ephemeris.
        in_transit = np.abs(phase) < (0.45 * best_dur)
        out_transit = np.abs(phase) > (0.65 * best_dur)
        baseline = float(np.nanmedian(f_arch[out_transit]))
        err_ppm = float(np.nanstd(f_arch[out_transit]) / np.sqrt(np.sum(in_transit)) * 1e6)
        transit_snr = float((baseline - np.nanmean(f_arch[in_transit])) * 1e6 / max(err_ppm, 1.0))
        control_mean = float(np.nanmean(f_arch[control]))
        control_snr = float((baseline - control_mean) * 1e6 / max(err_ppm, 1.0))
        return result, {
            "n_control": int(np.sum(control)),
            "n_transit": int(np.sum(in_transit)),
            "transit_snr": transit_snr,
            "control_snr": control_snr,
            "t0": t0,
        }

    def test_all_eight_control_centers_reject_control_dominant_signal(self):
        """Each declared control center must be reachable through verify_multi_epoch."""
        for center in (0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85):
            with self.subTest(control_center=center):
                (ok, status, diag), expected = self._run_control_phase_fixture(center)
                self.assertGreater(expected["n_control"], 0)
                self.assertGreater(expected["n_transit"], 0)
                self.assertGreater(expected["control_snr"], expected["transit_snr"] + 1.5)
                self.assertAlmostEqual(diag["archival_snr"], expected["transit_snr"], places=10)
                self.assertFalse(diag["is_projected_strict"])
                self.assertFalse(ok)
                self.assertIn("control phase null test", status)

    def test_wrapped_control_gate_is_invariant_under_integer_period_shifts(self):
        """A control at phase 0.70 remains the same decision after common kP shifts."""
        outcomes = []
        for shift in (-250, 1, 500):
            (ok, status, diag), expected = self._run_control_phase_fixture(0.70, shift)
            self.assertGreater(expected["n_control"], 0)
            self.assertFalse(diag["is_projected_strict"])
            self.assertFalse(ok)
            self.assertIn("control phase null test", status)
            outcomes.append((ok, diag["archival_snr"], expected["control_snr"]))

        self.assertEqual([item[0] for item in outcomes], [False, False, False])
        for outcome in outcomes[1:]:
            self.assertAlmostEqual(outcome[1], outcomes[0][1], places=10)
            self.assertAlmostEqual(outcome[2], outcomes[0][2], places=10)

    def test_partial_success_and_failure_stages(self):
        """
        Verify that when 3 products are requested across prior sectors:
        - Product 1 (Sector 80): succeeds
        - Product 2 (Sector 85): fails at DOWNLOAD
        - Product 3 (Sector 90): fails at EXTRACTION
        the used count is exactly 1, used sector is [80], failure stages are recorded,
        and proxy SDE correctly uses N_used=1 (not N_req=3).
        """
        tic_id = "34567890"
        curr_sec = 106
        t_curr = np.linspace(2459000, 2459027, 2000)
        f_curr = np.ones(2000)
        ferr_curr = np.full(2000, 0.0001)

        best_p = 3.5
        best_t0 = 2459001.0
        best_dur = 0.15

        # Lightcurve for Sector 80 with injected transit dip
        t_arch_80 = np.linspace(2458000, 2458027, 2000)
        f_arch_80 = np.ones(2000)
        ph_80 = ((t_arch_80 - best_t0 + 0.5 * best_p) % best_p) - 0.5 * best_p
        in_tr_80 = np.abs(ph_80) < (0.45 * best_dur)
        f_arch_80[in_tr_80] -= 0.0010  # 1000 ppm transit dip

        # Mock lightkurve search result returning 3 prior rows
        mock_row1 = MagicMock()
        mock_row1.mission = ["TESS Sector 80"]
        mock_row1.obsid = "TESS_s0080_prod_001"

        mock_row2 = MagicMock()
        mock_row2.mission = ["TESS Sector 85"]
        mock_row2.obsid = "TESS_s0085_prod_002"

        mock_row3 = MagicMock()
        mock_row3.mission = ["TESS Sector 90"]
        mock_row3.obsid = "TESS_s0090_prod_003"

        mock_search = [mock_row1, mock_row2, mock_row3]

        def mock_safe_download(row, target_tic, download_all=False):
            if "85" in str(row.mission):
                raise OSError("Connection reset by peer (HPC mirror timeout)")
            return MagicMock()

        call_count = {"extract": 0}
        def dynamic_extract(lc_list, **kw):
            call_count["extract"] += 1
            if call_count["extract"] == 1:
                # First attempt (Sector 80): succeeds
                return (t_arch_80, f_arch_80, np.full(2000, 0.0001), np.zeros(2000), np.zeros(2000), 0.95)
            else:
                # Second attempt (Sector 90): returns empty array
                return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0.9)

        with patch("lightkurve.search_lightcurve", return_value=mock_search), \
             patch.object(ald, "safe_download", side_effect=mock_safe_download), \
             patch.object(ald, "extract_and_stitch_arrays", side_effect=dynamic_extract), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)):

            ok, status_msg, arch_info = ald.verify_multi_epoch(
                tic_id, curr_sec, t_curr, f_curr, ferr_curr,
                best_p=best_p, best_t0=best_t0, best_dur=best_dur, depth_ppm_curr=1000.0, sde_curr=8.0
            )

            self.assertTrue(ok)
            self.assertEqual(arch_info["n_archival_sectors"], 1)
            self.assertEqual(arch_info["archival_sectors_str"], "80")
            self.assertIn("Confirmed_in_Sectors_80", status_msg)
            self.assertNotIn("85", status_msg)
            self.assertNotIn("90", status_msg)

            # Check requested vs attempted vs used vs failed
            self.assertEqual(arch_info["archival_sectors_requested"], [80, 85, 90])
            self.assertEqual(arch_info["archival_sectors_attempted"], [80, 85, 90])
            self.assertEqual(arch_info["archival_sectors_used"], [80])

            self.assertEqual(arch_info["archival_products_used"], ["TESS_s0080_prod_001"])
            self.assertEqual(len(arch_info["archival_products_failed"]), 2)

            f1 = arch_info["archival_products_failed"][0]
            self.assertEqual(f1["product_id"], "TESS_s0085_prod_002")
            self.assertEqual(f1["sector"], 85)
            self.assertEqual(f1["stage"], "DOWNLOAD")
            self.assertIn("Connection reset", f1["reason"])

            f2 = arch_info["archival_products_failed"][1]
            self.assertEqual(f2["product_id"], "TESS_s0090_prod_003")
            self.assertEqual(f2["sector"], 90)
            self.assertEqual(f2["stage"], "EXTRACTION")

            # Check proxy SDE uses N_used = 1
            # Formula: sde_curr * sqrt(1.0 + n_used * 0.5) = 8.0 * sqrt(1.0 + 1 * 0.5) = 8.0 * sqrt(1.5)
            expected_proxy_sde = 8.0 * np.sqrt(1.0 + 1 * 0.5)
            self.assertAlmostEqual(arch_info["est_multi_sde_proxy"], expected_proxy_sde, places=5)
            self.assertAlmostEqual(arch_info["multi_sde"], expected_proxy_sde, places=5)

    def test_duplicate_sector_products_counted_once_as_sector(self):
        """
        Verify that multiple products in the same sector count as distinct product attempts
        and distinct product uses, but count only once in n_archival_sectors and proxy SDE.
        """
        tic_id = "45678901"
        curr_sec = 106
        t_curr = np.linspace(2459000, 2459027, 2000)
        f_curr = np.ones(2000)
        ferr_curr = np.full(2000, 0.0001)

        best_p = 3.5
        best_t0 = 2459001.0
        best_dur = 0.15

        # 2 products both from Sector 80
        t_arch_80a = np.linspace(2458000, 2458013, 1000)
        f_arch_80a = np.ones(1000)
        ph_a = ((t_arch_80a - best_t0 + 0.5 * best_p) % best_p) - 0.5 * best_p
        f_arch_80a[np.abs(ph_a) < (0.45 * best_dur)] -= 0.0010

        t_arch_80b = np.linspace(2458014, 2458027, 1000)
        f_arch_80b = np.ones(1000)
        ph_b = ((t_arch_80b - best_t0 + 0.5 * best_p) % best_p) - 0.5 * best_p
        f_arch_80b[np.abs(ph_b) < (0.45 * best_dur)] -= 0.0010

        mock_row1 = MagicMock()
        mock_row1.mission = ["TESS Sector 80"]
        mock_row1.obsid = "TESS_s0080_prod_A"

        mock_row2 = MagicMock()
        mock_row2.mission = ["TESS Sector 80"]
        mock_row2.obsid = "TESS_s0080_prod_B"

        call_cnt = {"val": 0}
        def dynamic_extract_2(lc_list, **kw):
            call_cnt["val"] += 1
            if call_cnt["val"] == 1:
                return (t_arch_80a, f_arch_80a, np.full(1000, 0.0001), np.zeros(1000), np.zeros(1000), 0.95)
            else:
                return (t_arch_80b, f_arch_80b, np.full(1000, 0.0001), np.zeros(1000), np.zeros(1000), 0.95)

        with patch("lightkurve.search_lightcurve", return_value=[mock_row1, mock_row2]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", side_effect=dynamic_extract_2), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)):

            ok, status_msg, arch_info = ald.verify_multi_epoch(
                tic_id, curr_sec, t_curr, f_curr, ferr_curr,
                best_p=best_p, best_t0=best_t0, best_dur=best_dur, depth_ppm_curr=1000.0, sde_curr=8.0
            )

            self.assertTrue(ok)
            # Both products used
            self.assertEqual(len(arch_info["archival_products_used"]), 2)
            self.assertEqual(arch_info["archival_products_used"], ["TESS_s0080_prod_A", "TESS_s0080_prod_B"])

            # Unique sector count must be 1!
            self.assertEqual(arch_info["n_archival_sectors"], 1)
            self.assertEqual(arch_info["archival_sectors_used"], [80])
            self.assertEqual(arch_info["archival_sectors_str"], "80")

            # Proxy SDE must be computed with 1 sector, NOT 2!
            expected_proxy = 8.0 * np.sqrt(1.0 + 1 * 0.5)
            self.assertAlmostEqual(arch_info["est_multi_sde_proxy"], expected_proxy, places=5)

    def test_rejection_labels_cite_actual_used_sectors(self):
        """
        Verify that when an archival test fails (e.g. non-detection), the rejection reason
        specifically and truthfully cites the actual used sectors, NOT the requested sectors.
        """
        tic_id = "67890123"
        curr_sec = 106
        t_curr = np.linspace(2459000, 2459027, 2000)
        f_curr = np.ones(2000)
        ferr_curr = np.full(2000, 0.0001)

        # Sector 80 has flat light curve (no transit dip -> non-detection)
        t_arch_80 = np.linspace(2458000, 2458027, 2000)
        f_arch_80 = np.ones(2000)  # No dip!

        mock_row1 = MagicMock()
        mock_row1.mission = ["TESS Sector 80"]
        mock_row1.obsid = "TESS_s0080_prod_1"

        mock_row2 = MagicMock()
        mock_row2.mission = ["TESS Sector 85"]
        mock_row2.obsid = "TESS_s0085_prod_2"

        def mock_download(row, *a, **kw):
            if "85" in str(row.mission):
                raise OSError("Download timeout")
            return MagicMock()

        with patch("lightkurve.search_lightcurve", return_value=[mock_row1, mock_row2]), \
             patch.object(ald, "safe_download", side_effect=mock_download), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arch_80, f_arch_80, np.full(2000, 0.0001), np.zeros(2000), np.zeros(2000), 0.95)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)):

            ok, status_msg, arch_info = ald.verify_multi_epoch(
                tic_id, curr_sec, t_curr, f_curr, ferr_curr,
                best_p=3.5, best_t0=2459001.0, best_dur=0.15, depth_ppm_curr=1000.0, sde_curr=8.0
            )

            self.assertFalse(ok)
            # Rejection message must cite used sector 80, NOT requested 80, 85!
            self.assertIn("Archival non-detection in sectors 80", status_msg)
            self.assertNotIn("85", status_msg)

    def test_all_failed_and_query_error_outcomes(self):
        """
        Verify distinct recorded reasons and zero usage for all-failed downloads and query errors.
        """
        tic_id = "56789012"
        curr_sec = 106
        t_curr = np.linspace(2459000, 2459027, 1000)
        f_curr = np.ones(1000)
        ferr_curr = np.full(1000, 0.001)

        # 1. All downloads fail
        mock_row1 = MagicMock()
        mock_row1.mission = ["TESS Sector 80"]
        mock_row1.obsid = "TESS_s0080_fail"

        with patch("lightkurve.search_lightcurve", return_value=[mock_row1]), \
             patch.object(ald, "safe_download", side_effect=RuntimeError("MAST 503 Service Unavailable")):

            ok, status_msg, arch_info = ald.verify_multi_epoch(
                tic_id, curr_sec, t_curr, f_curr, ferr_curr,
                best_p=3.5, best_t0=2459001.0, best_dur=0.1, depth_ppm_curr=500.0, sde_curr=7.0
            )
            self.assertTrue(ok)  # Passes via single-sector fallback since sde >= 6.5
            self.assertIn("Archival Download Failed", status_msg)
            self.assertEqual(arch_info["n_archival_sectors"], 0)
            self.assertEqual(arch_info["archival_sectors_used"], [])
            self.assertEqual(arch_info["archival_products_used"], [])
            self.assertEqual(len(arch_info["archival_products_failed"]), 1)
            self.assertEqual(arch_info["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertEqual(arch_info["ephemeris_fallback_reason"], "ARCHIVAL_DOWNLOAD_OR_PREPARATION_FAILED")

        # 2. Search query error
        with patch("lightkurve.search_lightcurve", side_effect=ConnectionError("DNS lookup failed")):
            ok_q, status_msg_q, arch_info_q = ald.verify_multi_epoch(
                tic_id, curr_sec, t_curr, f_curr, ferr_curr,
                best_p=3.5, best_t0=2459001.0, best_dur=0.1, depth_ppm_curr=500.0, sde_curr=7.0
            )
            self.assertTrue(ok_q)
            self.assertIn("Archival Query Error", status_msg_q)
            self.assertEqual(arch_info_q["n_archival_sectors"], 0)
            self.assertEqual(arch_info_q["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertIn("ARCHIVAL_QUERY_FAILED: ConnectionError", arch_info_q["ephemeris_fallback_reason"])

    def test_deliberate_requested_as_used_fault_detected_and_control_passes(self):
        """
        Falsification / Negative control:
        Demonstrate that a validator checking provenance consistency catches a corrupted row
        where requested sectors were reported as used despite failed products.
        """
        def validate_archival_provenance_truthfulness(row: dict) -> tuple[bool, str]:
            n_used = int(row.get("n_archival_sectors", 0))
            sectors_used = json.loads(row.get("archival_sectors_used", "[]"))
            arch_str = str(row.get("archival_sectors", "None"))

            if n_used != len(sectors_used):
                return False, f"Inconsistent sector count: n_archival_sectors ({n_used}) != len(archival_sectors_used) ({len(sectors_used)})"

            expected_str = ",".join(str(s) for s in sorted(sectors_used)) if sectors_used else "None"
            if arch_str != expected_str:
                return False, f"Inconsistent display string: archival_sectors ('{arch_str}') != actual used sectors ('{expected_str}')"

            return True, "Truthful provenance"

        # 1. Deliberate fault: requested sectors [80, 85] masquerading as used when only 80 was used
        faulty_row = {
            "n_archival_sectors": 2,  # BUG: claimed 2 sectors used
            "archival_sectors": "80,85",  # BUG: claimed both sectors used
            "archival_sectors_used": json.dumps([80]),  # Reality: only 80 was incorporated
            "archival_products_used": json.dumps(["TESS_s0080_01"]),
            "archival_products_failed": json.dumps([{"product_id": "TESS_s0085_01", "stage": "DOWNLOAD"}])
        }
        is_valid_f, msg_f = validate_archival_provenance_truthfulness(faulty_row)
        self.assertFalse(is_valid_f, "Validator must catch requested-as-used fault")
        self.assertIn("Inconsistent", msg_f)

        # 2. Corrected control
        correct_row = {
            "n_archival_sectors": 1,
            "archival_sectors": "80",
            "archival_sectors_used": json.dumps([80]),
            "archival_products_used": json.dumps(["TESS_s0080_01"]),
            "archival_products_failed": json.dumps([{"product_id": "TESS_s0085_01", "stage": "DOWNLOAD"}])
        }
        is_valid_c, msg_c = validate_archival_provenance_truthfulness(correct_row)
        self.assertTrue(is_valid_c, "Validator must pass truthful provenance")

    def test_post_preparation_refinement_error_preserves_provenance(self):
        """
        R1 Acceptance check:
        When products are successfully downloaded and prepared, but BoxLeastSquares raises
        RuntimeError during multi-sector refinement, verify_multi_epoch must:
        1. Preserve attempted and used product/sector lists (not empty!).
        2. Record failure stage as REFINEMENT (not QUERY!).
        3. Fall back to discovery ephemeris with reason ARCHIVAL_REFINEMENT_FAILED.
        4. Return populated diagnostic dictionary (not None).
        """
        tic_id = "12345"
        curr_sec = 106
        t = np.linspace(1000.0, 1027.0, 2000)
        f = np.ones(2000)
        e = np.full(2000, 0.001)
        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "archive-product-80"

        with patch.object(ald.lk, "search_lightcurve", return_value=[mock_row]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t, f, e, None, None, 0.99)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "BoxLeastSquares", side_effect=RuntimeError("injected refinement failure")):

            ok, msg, info = ald.verify_multi_epoch(
                tic_id, curr_sec, t + 800, f, e,
                best_p=3.0, best_t0=1801.0, best_dur=0.1, depth_ppm_curr=1000.0, sde_curr=8.0
            )

            self.assertTrue(ok)
            self.assertIn("Archival Refinement Error", msg)
            self.assertIsNotNone(info)
            self.assertEqual(info["archival_products_attempted"], ["archive-product-80"])
            self.assertEqual(info["archival_products_used"], ["archive-product-80"])
            self.assertEqual(info["archival_sectors_used"], [80])
            self.assertEqual(info["n_archival_sectors"], 1)
            self.assertEqual(info["archival_sectors_str"], "80")
            self.assertEqual(info["n_archival_sectors"], len(info["archival_sectors_used"]))
            self.assertEqual(info["est_multi_sde_proxy"], 8.0)
            self.assertEqual(info["multi_sde"], 8.0)
            self.assertEqual(len(info["archival_products_failed"]), 1)
            self.assertEqual(info["archival_products_failed"][0]["stage"], "REFINEMENT")
            self.assertIn("injected refinement failure", info["archival_products_failed"][0]["reason"])
            self.assertEqual(info["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
            self.assertEqual(info["ephemeris_status"], "FALLBACK_DISCOVERY")
            self.assertIn("ARCHIVAL_REFINEMENT_FAILED", info["ephemeris_fallback_reason"])
            self.assertTrue(np.isnan(info["refined_p"]))
            self.assertTrue(np.isnan(info["refined_t0"]))

        # And when SDE < 6.5, verify it returns (False, msg, info) with full diagnostic metadata, not None
        with patch.object(ald.lk, "search_lightcurve", return_value=[mock_row]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t, f, e, None, None, 0.99)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "BoxLeastSquares", side_effect=RuntimeError("injected refinement failure")):

            ok_low, msg_low, info_low = ald.verify_multi_epoch(
                tic_id, curr_sec, t + 800, f, e,
                best_p=3.0, best_t0=1801.0, best_dur=0.1, depth_ppm_curr=1000.0, sde_curr=6.0
            )
            self.assertFalse(ok_low)
            self.assertIsNotNone(info_low)
            self.assertEqual(info_low["archival_products_attempted"], ["archive-product-80"])
            self.assertEqual(info_low["archival_products_used"], ["archive-product-80"])
            self.assertEqual(info_low["archival_sectors_used"], [80])
            self.assertEqual(info_low["n_archival_sectors"], 1)
            self.assertEqual(info_low["archival_sectors_str"], "80")
            self.assertEqual(info_low["n_archival_sectors"], len(info_low["archival_sectors_used"]))
            self.assertEqual(info_low["est_multi_sde_proxy"], 6.0)
            self.assertEqual(info_low["multi_sde"], 6.0)
            self.assertEqual(info_low["archival_products_failed"][0]["stage"], "REFINEMENT")

    def test_post_preparation_fallback_candidate_persistence(self):
        """
        R1a Acceptance check through actual candidate persistence:
        When refinement fails after product preparation, verify that the actual serialized
        candidate record preserves the discovery ephemeris, records fallback status,
        and maintains 100% consistent n_archival_sectors == len(archival_sectors_used) == 1.
        """
        tic_id = "88990011"
        sector = 106
        discovery_p = 3.0
        discovery_t0 = 2459001.0
        n_pts = 19440
        rng = np.random.default_rng(42)
        t_arr = np.linspace(2459000.0, 2459027.0, n_pts)
        f_arr = 1.0 + 1e-4 * rng.standard_normal(n_pts)
        ph = ((t_arr - discovery_t0 + 0.5 * discovery_p) % discovery_p) - 0.5 * discovery_p
        f_arr[np.abs(ph) < 0.04] -= 0.0010

        t_arch = np.linspace(2458000.0, 2458027.0, n_pts)
        f_arch = np.ones(n_pts)
        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "archive-product-80"

        with tempfile.TemporaryDirectory(prefix="ald_test_r1a_persist_") as tmp_dir:
            ald.configure_environment(output_dir=tmp_dir, run_id="r1a_persist_run")
            try:
                with patch("lightkurve.search_lightcurve", side_effect=[
                    [MagicMock()],  # Sector 106 discovery search
                    [mock_row]      # Sector 80 archival search
                ]), \
                     patch.object(ald, "safe_download", return_value=MagicMock()), \
                     patch.object(ald, "extract_and_stitch_arrays", side_effect=[
                         (t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95),
                         (t_arch, f_arch, np.full(n_pts, 0.0001), None, None, 0.99)
                     ]), \
                     patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
                     patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
                     patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
                     patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
                     patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
                     patch.object(ald, "scan_full_orbit_secondary_eclipse", return_value=(False, 0.0, 0.0, 0.5, 3.0)), \
                     patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

                    # First BoxLeastSquares is discovery BLS; second BoxLeastSquares is refinement BLS which raises RuntimeError
                    call_count = {"cnt": 0}
                    def mock_bls_factory(*args, **kwargs):
                        call_count["cnt"] += 1
                        if call_count["cnt"] == 1:
                            m = MagicMock()
                            m.power.return_value = MagicMock(
                                period=np.full(9000, discovery_p),
                                power=np.full(9000, 100.0),
                                duration=np.full(9000, 0.08),
                                transit_time=np.full(9000, discovery_t0),
                                depth=np.full(9000, 0.001)
                            )
                            return m
                        else:
                            raise RuntimeError("injected refinement failure")

                    with patch.object(ald, "BoxLeastSquares", side_effect=mock_bls_factory):
                        res = ald.analyze_single_target(tic_id, sector)
                        ret_tic, status, msg, cand_data = res

                        self.assertEqual(status, "CANDIDATE")
                        self.assertIsNotNone(cand_data)

                        # Coherent discovery fallback verification
                        self.assertEqual(cand_data["period_days"], discovery_p)
                        self.assertEqual(cand_data["t0_btjd"], discovery_t0)
                        self.assertEqual(cand_data["discovery_period_days"], discovery_p)
                        self.assertEqual(cand_data["discovery_t0_btjd"], discovery_t0)
                        self.assertTrue(np.isnan(float(cand_data["refined_period_days"])))
                        self.assertTrue(np.isnan(float(cand_data["refined_t0_btjd"])))
                        self.assertEqual(cand_data["ephemeris_source"], "DISCOVERY_SINGLE_SECTOR")
                        self.assertEqual(cand_data["ephemeris_status"], "FALLBACK_DISCOVERY")
                        self.assertEqual(cand_data["ephemeris_fallback_reason"], "ARCHIVAL_REFINEMENT_FAILED: RuntimeError")

                        # Reconciled sector count and provenance
                        self.assertEqual(cand_data["n_archival_sectors"], 1)
                        self.assertEqual(cand_data["archival_sectors"], "80")
                        self.assertEqual(json.loads(cand_data["archival_sectors_used"]), [80])
                        self.assertEqual(json.loads(cand_data["archival_products_used"]), ["archive-product-80"])
                        self.assertEqual(cand_data["est_multi_sde_proxy"], 8.5)

                        # Save to report and verify CSV row matches
                        rep_file = os.path.join(tmp_dir, "candidates_report.csv")
                        ald.save_candidates_to_report(rep_file, [cand_data])
                        df_saved = pd.read_csv(rep_file)
                        self.assertEqual(len(df_saved), 1)
                        saved_row = df_saved.iloc[0].to_dict()
                        self.assertEqual(saved_row["period_days"], discovery_p)
                        self.assertEqual(saved_row["t0_btjd"], discovery_t0)
                        self.assertEqual(saved_row["n_archival_sectors"], 1)
                        self.assertEqual(str(saved_row["archival_sectors"]), "80")
                        self.assertEqual(saved_row["ephemeris_status"], "FALLBACK_DISCOVERY")
            finally:
                ald.configure_environment(output_dir="output", run_id="run_default")

    def test_dynamic_stage_naming_in_combined_pipeline(self):
        """
        Verify that failures during multi-sector combined evaluation dynamically and accurately
        report the specific processing stage (CONCATENATION, DETRENDING, EVALUATION, REFINEMENT),
        rather than hardcoding all failures as REFINEMENT.
        """
        tic_id = "55443322"
        curr_sec = 106
        t = np.linspace(1000.0, 1027.0, 2000)
        f = np.ones(2000)
        e = np.full(2000, 0.001)
        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "archive-product-80"

        # 1. Failure during DETRENDING
        with patch.object(ald.lk, "search_lightcurve", return_value=[mock_row]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t, f, e, None, None, 0.99)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=[
                 np.ones_like(f),  # initial archival detrend
                 RuntimeError("injected savgol detrending crash")  # combined prior detrend
             ]):

            ok, msg, info = ald.verify_multi_epoch(
                tic_id, curr_sec, t + 800, f, e,
                best_p=3.0, best_t0=1801.0, best_dur=0.1, depth_ppm_curr=1000.0, sde_curr=8.0
            )
            self.assertTrue(ok)
            self.assertIn("Archival Detrending Error", msg)
            self.assertEqual(info["archival_products_failed"][0]["stage"], "DETRENDING")
            self.assertEqual(info["ephemeris_fallback_reason"], "ARCHIVAL_DETRENDING_FAILED: RuntimeError")

        # 2. Failure during EVALUATION (e.g. secondary eclipse scan error)
        # Prior array with prominent transit dip so archival_snr passes non-detection and null test
        f_eval_arch = np.ones(2000)
        ph_eval = ((t - 1801.0 + 0.5 * 3.0) % 3.0) - 0.5 * 3.0
        f_eval_arch[np.abs(ph_eval) < (0.45 * 0.1)] -= 0.0020

        with patch.object(ald.lk, "search_lightcurve", return_value=[mock_row]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t, f_eval_arch, e, None, None, 0.99)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "scan_full_orbit_secondary_eclipse", side_effect=ValueError("injected secondary eclipse scan error")):

            ok_eval, msg_eval, info_eval = ald.verify_multi_epoch(
                tic_id, curr_sec, t + 800, f, e,
                best_p=3.0, best_t0=1801.0, best_dur=0.1, depth_ppm_curr=1000.0, sde_curr=8.0
            )
            self.assertTrue(ok_eval)
            self.assertIn("Archival Evaluation Error", msg_eval)
            self.assertEqual(info_eval["archival_products_failed"][0]["stage"], "EVALUATION")
            self.assertEqual(info_eval["ephemeris_fallback_reason"], "ARCHIVAL_EVALUATION_FAILED: ValueError")

    def test_r1b_rejected_target_provenance_preservation_and_caller_transport(self):
        """
        R1b Acceptance check:
        Verifies that when an inspected target is rejected (by archival veto or low-SDE failure),
        archival diagnostics are durably persisted to archival_vetting_diagnostics.jsonl in OUTPUT_DIR,
        and companion search transports the evidence in audit records (error_detail).
        Includes negative controls:
        1. Deliberate dropped-diagnostics fault is caught and fails.
        2. Persistence write failures raise explicit errors rather than silently swallowing.
        """
        tic_id = "99887766"
        sector = 106
        discovery_p = 3.0
        discovery_t0 = 2459001.0
        n_pts = 19440
        rng = np.random.default_rng(42)
        t_arr = np.linspace(2459000.0, 2459027.0, n_pts)
        f_arr = 1.0 + 1e-4 * rng.standard_normal(n_pts)
        ph = ((t_arr - discovery_t0 + 0.5 * discovery_p) % discovery_p) - 0.5 * discovery_p
        f_arr[np.abs(ph) < 0.04] -= 0.0010

        t_arch = np.linspace(2458000.0, 2458027.0, n_pts)
        f_arch = np.ones(n_pts)  # Flat prior -> non-detection veto
        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "archive-product-80"

        with tempfile.TemporaryDirectory(prefix="ald_test_r1b_") as tmp_dir:
            ald.configure_environment(output_dir=tmp_dir, run_id="r1b_test_run")
            try:
                # 1. Primary caller test (analyze_single_target on archival veto rejection)
                with patch("lightkurve.search_lightcurve", side_effect=[
                    [MagicMock()],
                    [mock_row]
                ]), \
                     patch.object(ald, "safe_download", return_value=MagicMock()), \
                     patch.object(ald, "extract_and_stitch_arrays", side_effect=[
                         (t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95),
                         (t_arch, f_arch, np.full(n_pts, 0.0001), None, None, 0.99)
                     ]), \
                     patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
                     patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
                     patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
                     patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
                     patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
                     patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

                    mock_bls = MagicMock()
                    mock_bls.power.return_value = MagicMock(
                        period=np.full(9000, discovery_p),
                        power=np.full(9000, 100.0),
                        duration=np.full(9000, 0.08),
                        transit_time=np.full(9000, discovery_t0),
                        depth=np.full(9000, 0.001)
                    )
                    with patch.object(ald, "BoxLeastSquares", return_value=mock_bls):
                        target_res = ald.analyze_single_target(tic_id, sector)
                        ret_tic, status, msg, cand_data = target_res

                self.assertEqual(status, "REJECT_ARCHIVAL")
                self.assertIsNone(cand_data)

                # Verify durable diagnostic artifact exists in configured output_dir
                diag_file = os.path.join(tmp_dir, "archival_vetting_diagnostics.jsonl")
                self.assertTrue(os.path.exists(diag_file), "Diagnostic file must be persisted to OUTPUT_DIR")
                latest_diag = ald.get_latest_archival_diagnostic(tic_id, sector, output_dir=tmp_dir)
                self.assertIsNotNone(latest_diag)
                self.assertEqual(latest_diag["tic_id"], tic_id)
                self.assertEqual(latest_diag["sector"], sector)
                self.assertFalse(latest_diag["arch_ok"])
                self.assertEqual(latest_diag["archival_products_attempted"], ["archive-product-80"])
                self.assertEqual(latest_diag["archival_sectors_used"], [80])

                # 2. Companion caller test: process_toi_host transports diagnostic evidence into audit CSV
                with patch.object(ald, "analyze_single_target", return_value=target_res), \
                     patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "100.01", "period": 10.0}]):
                    audit_res = stc.process_toi_host(tic_id, sector, run_id="r1b_test_run")

                self.assertEqual(audit_res["status"], "REJECT_ARCHIVAL")
                self.assertTrue(audit_res["error_detail"], "Audit record must transport archival diagnostics in error_detail")
                transported_diag = json.loads(audit_res["error_detail"])
                self.assertEqual(transported_diag["tic_id"], tic_id)
                self.assertEqual(transported_diag["archival_products_attempted"], ["archive-product-80"])

                # 3. Negative control: Deliberate dropped-diagnostics fault
                def audit_verification_check(out_dir, t_id, sec_num):
                    rec = ald.get_latest_archival_diagnostic(t_id, sec_num, output_dir=out_dir)
                    if rec is None or not rec.get("archival_products_attempted"):
                        raise AssertionError(f"Provenance fault: archival diagnostics dropped for TIC {t_id}")

                # Passing control
                audit_verification_check(tmp_dir, tic_id, sector)

                # Fault injection: dropped diagnostics must trigger assertion failure
                with tempfile.TemporaryDirectory(prefix="ald_empty_dir_") as empty_dir:
                    with self.assertRaises(AssertionError) as ctx:
                        audit_verification_check(empty_dir, tic_id, sector)
                    self.assertIn("archival diagnostics dropped", str(ctx.exception))

                # 4. Write failure detection: Verify write failures are NOT swallowed
                with patch("builtins.open", side_effect=OSError("Read-only file system")):
                    with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
                         patch.object(ald, "safe_download", return_value=MagicMock()), \
                         patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95)), \
                         patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
                         patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
                         patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
                         patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
                         patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
                         patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))), \
                         patch.object(ald, "BoxLeastSquares", return_value=mock_bls), \
                         patch.object(ald, "verify_multi_epoch", return_value=(False, "Archival non-detection", {"archival_products_attempted": ["p1"]})):

                        err_tic, err_status, err_msg, err_cand = ald.analyze_single_target(tic_id, sector)
                        self.assertEqual(err_status, "ERROR")
                        self.assertIn("Failed to persist archival vetting diagnostic", err_msg)
            finally:
                ald.configure_environment(output_dir="output", run_id="run_default")

    def test_vetting_rejections_return_diagnostic_dict(self):
        """
        Verify that when multi-epoch verification fails a vetting check
        (insufficient points, non-detection, red noise null test, secondary eclipse),
        it returns (False, reject_msg, base_diag) with a populated diagnostic dictionary,
        retaining all attempted, used, and failed products rather than discarding evidence.
        """
        tic_id = "77788899"
        curr_sec = 106
        t_curr = np.linspace(2459000, 2459027, 2000)
        f_curr = np.ones(2000)
        ferr_curr = np.full(2000, 0.0001)

        t_arch = np.linspace(2458000, 2458027, 2000)
        f_arch = np.ones(2000)  # Flat -> non-detection

        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "prod_vetting_test_80"

        with patch("lightkurve.search_lightcurve", return_value=[mock_row]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arch, f_arch, np.full(2000, 0.0001), None, None, 0.95)), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)):

            ok, msg, diag = ald.verify_multi_epoch(
                tic_id, curr_sec, t_curr, f_curr, ferr_curr,
                best_p=3.5, best_t0=2459001.0, best_dur=0.1, depth_ppm_curr=1000.0, sde_curr=8.0
            )

            self.assertFalse(ok)
            self.assertIsNotNone(diag)
            self.assertEqual(diag["archival_sectors_used"], [80])
            self.assertEqual(diag["archival_products_used"], ["prod_vetting_test_80"])
            self.assertIn("Archival non-detection in sectors 80", msg)


class TestPersistenceAndSchemaCompatibility(unittest.TestCase):
    """Test candidate schema v1.1 persistence, header compatibility protection, and companion persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_persist_")
        ald.configure_environment(output_dir=self.test_dir, run_id="persist_run")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        ald.configure_environment(output_dir="output", run_id="run_default")

    def test_schema_v1_1_columns_complete_and_order_consistent(self):
        """Verify that ald.CANDIDATE_COLUMNS and stc.CANDIDATE_COLUMNS are identical and contain 58 columns."""
        self.assertEqual(ald.CANDIDATE_COLUMNS, stc.CANDIDATE_COLUMNS)
        self.assertEqual(len(ald.CANDIDATE_COLUMNS), 58)
        self.assertEqual(ald.CANDIDATE_SCHEMA_VERSION, "v1.1")
        self.assertEqual(stc.CANDIDATE_SCHEMA_VERSION, "v1.1")

        for col in [
            "time_standard", "epoch_offset_btjd",
            "transit_measurement_source", "timing_uncertainty_source"
        ]:
            self.assertIn(col, ald.CANDIDATE_COLUMNS)
            self.assertIn(col, stc.CANDIDATE_COLUMNS)

    def test_companion_persistence_retains_all_schema_fields(self):
        """
        Verify that search_toi_companions persists all 58 columns of schema v1.1
        with proper units, metadata, and JSON serialized lists.
        """
        report_file, audit_file, run_report, run_audit = stc.initialize_output_artifacts(
            self.test_dir, run_id="test_persist_run"
        )

        mock_candidate = {
            "schema_version": "v1.1",
            "pipeline_version": "v9.0",
            "run_id": "test_persist_run",
            "tic_id": "99990001",
            "sector": 106,
            "is_toi_host": True,
            "known_toi_context": "TOI 999.01(P=2.500d)",
            "candidate_type": "TOI_SIBLING_COMPANION",
            "ephemeris_source": "REFINED_MULTI_SECTOR",
            "ephemeris_status": "REFINED_VALID",
            "ephemeris_fallback_reason": "NONE",
            "time_system": "BTJD",
            "time_standard": "TDB",
            "epoch_offset_btjd": 2457000.0,
            "period_units": "days",
            "duration_units": "hours",
            "depth_units": "ppm",
            "transit_measurement_source": "DISCOVERY_SINGLE_SECTOR",
            "timing_uncertainty_source": "CARTER_2008_DISCOVERY_HEURISTIC",
            "period_days": 1.234567890123,
            "t0_btjd": 2459001.123456789,
            "discovery_period_days": 1.234500000000,
            "discovery_t0_btjd": 2459001.000000000,
            "refined_period_days": 1.234567890123,
            "refined_t0_btjd": 2459001.123456789,
            "duration_hours": 2.5,
            "min_kepler_dur_hours": 1.0,
            "max_kepler_dur_hours": 4.0,
            "n_transits": 10,
            "depth_ppm": 350.5,
            "crowdsap": 0.98,
            "sde": 9.2,
            "r_planet_earth": 1.45,
            "r_star_sun": 0.95,
            "m_star_sun": 0.92,
            "teff_k": 5400,
            "gaia_ruwe": 1.02,
            "odd_even_sigma": 0.35,
            "centroid_shift_arcsec": 0.42,
            "archival_sectors": "80,85",
            "n_archival_sectors": 2,
            "archival_sectors_requested": json.dumps([80, 85]),
            "archival_sectors_attempted": json.dumps([80, 85]),
            "archival_sectors_used": json.dumps([80, 85]),
            "archival_products_requested": json.dumps(["p1", "p2"]),
            "archival_products_attempted": json.dumps(["p1", "p2"]),
            "archival_products_used": json.dumps(["p1", "p2"]),
            "archival_products_failed": json.dumps([]),
            "archival_snr_sigma": 7.8,
            "archival_trials": 10,
            "archival_snr_crit": 3.0,
            "est_multi_sde_proxy": 13.0,
            "multi_sde": 13.0,
            "cramer_rao_sigma_p_sec": 0.85,
            "cramer_rao_sigma_t0_sec": 35.0,
            "covariance_p_t0": np.nan,
            "plot_path": "candidate_TIC_99990001_s106.png",
            "exofop_url": "https://exofop.ipac.caltech.edu/tess/target.php?id=99990001"
        }

        df_cand = pd.DataFrame([mock_candidate], columns=stc.CANDIDATE_COLUMNS)
        stc._append_candidate_dedup(report_file, df_cand, tic="99990001", sector=106, run_id="test_persist_run")

        df_saved = pd.read_csv(report_file)
        self.assertEqual(list(df_saved.columns), stc.CANDIDATE_COLUMNS)
        self.assertEqual(len(df_saved), 1)

        row = df_saved.iloc[0]
        self.assertEqual(row["schema_version"], "v1.1")
        self.assertEqual(row["time_system"], "BTJD")
        self.assertEqual(row["time_standard"], "TDB")
        self.assertEqual(float(row["epoch_offset_btjd"]), 2457000.0)
        self.assertEqual(row["transit_measurement_source"], "DISCOVERY_SINGLE_SECTOR")
        self.assertEqual(row["timing_uncertainty_source"], "CARTER_2008_DISCOVERY_HEURISTIC")
        self.assertEqual(float(row["period_days"]), 1.234567890123)
        self.assertEqual(float(row["t0_btjd"]), 2459001.123456789)
        self.assertTrue(pd.isna(row["covariance_p_t0"]))

        # Verify JSON list deserialization
        self.assertEqual(json.loads(row["archival_sectors_used"]), [80, 85])
        self.assertEqual(json.loads(row["archival_products_used"]), ["p1", "p2"])

    def test_primary_writer_end_to_end_roundtrip_with_schema_v1_1(self):
        """
        R2 Acceptance check:
        Execute the actual production main AST of aldebaran.py offline:
        1. On fresh start: discovers and persists a candidate in REPORT_FILE with all 58 columns.
        2. On second execution: loads existing report, appends new candidate without destroying history.
        """
        report_file = os.path.join(self.test_dir, "primary_candidates_report.csv")
        tree = ast.parse(Path(ald.__file__).read_text(encoding="utf-8"))
        main_node = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")

        cand1 = {c: "val1" for c in ald.CANDIDATE_COLUMNS}
        cand1.update({"tic_id": "11111111", "sector": 106, "period_days": 2.5, "t0_btjd": 2459001.0})

        class MockFuture:
            def __init__(self, cand):
                self._cand = cand
            def result(self):
                return (self._cand["tic_id"], "CANDIDATE", "Candidate Found", self._cand)

        class MockExecutor1:
            def __init__(self, **kw): pass
            def submit(self, *args):
                return MockFuture(cand1)
            def shutdown(self, **kw): pass

        ns = dict(ald.__dict__)
        ns.update({
            "__name__": "__main__",
            "REPORT_FILE": report_file,
            "MAX_WORKERS": 1,
            "select_latest_sector_and_targets": lambda **kw: (106, ["11111111"]),
            "ProcessPoolExecutor": MockExecutor1,
            "as_completed": lambda fs: list(fs),
            "append_scanned_history": lambda tic: None,
            "write_log": lambda *a, **k: None,
            "write_event": lambda *a, **k: None
        })

        # Run 1: file does not exist initially
        with patch.object(signal, "signal"):
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main>", "exec"), ns)

        self.assertTrue(os.path.exists(report_file))
        df_after1 = pd.read_csv(report_file)
        self.assertEqual(list(df_after1.columns), ald.CANDIDATE_COLUMNS)
        self.assertEqual(len(df_after1), 1)
        self.assertEqual(str(df_after1.iloc[0]["tic_id"]), "11111111")

        # Run 2: loads candidate 1 and appends candidate 2
        cand2 = {c: "val2" for c in ald.CANDIDATE_COLUMNS}
        cand2.update({"tic_id": "22222222", "sector": 106, "period_days": 4.0, "t0_btjd": 2459005.0})

        class MockExecutor2:
            def __init__(self, **kw): pass
            def submit(self, *args):
                return MockFuture(cand2)
            def shutdown(self, **kw): pass

        ns["select_latest_sector_and_targets"] = lambda **kw: (106, ["22222222"])
        ns["ProcessPoolExecutor"] = MockExecutor2
        with patch.object(signal, "signal"):
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main>", "exec"), ns)

        df_after2 = pd.read_csv(report_file)
        self.assertEqual(len(df_after2), 2)
        self.assertEqual(list(df_after2["tic_id"].astype(str)), ["11111111", "22222222"])

    def test_primary_writer_fail_fast_on_incompatible_or_unreadable_file(self):
        """
        R2 Acceptance check:
        Verify that if REPORT_FILE has an incompatible header or corrupted/unparseable content:
        1. aldebaran.py main AST raises RuntimeError immediately.
        2. Historical file bytes and SHA256 are 100% unchanged.
        3. Target selection and worker executor are never invoked.
        """
        report_file = os.path.join(self.test_dir, "historical_corrupt_report.csv")
        corrupt_content = b"header1,header2,header3\nincomplete_row\x00\xff_corrupt\n"
        Path(report_file).write_bytes(corrupt_content)
        hash_before = hashlib.sha256(corrupt_content).hexdigest()

        tree = ast.parse(Path(ald.__file__).read_text(encoding="utf-8"))
        main_node = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")

        target_selector_called = []
        def mock_selector(**kw):
            target_selector_called.append(True)
            return (106, ["12345678"])

        ns = dict(ald.__dict__)
        ns.update({
            "__name__": "__main__",
            "REPORT_FILE": report_file,
            "MAX_WORKERS": 1,
            "select_latest_sector_and_targets": mock_selector,
            "write_log": lambda *a, **k: None,
            "write_event": lambda *a, **k: None
        })

        with patch.object(signal, "signal"), self.assertRaises(RuntimeError) as ctx:
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main>", "exec"), ns)

        # File must be preserved bit-for-bit
        hash_after = hashlib.sha256(Path(report_file).read_bytes()).hexdigest()
        self.assertEqual(hash_before, hash_after, "Corrupted historical file must NOT be modified or overwritten")
        self.assertFalse(target_selector_called, "Science target selection must NOT be invoked when historical report is unreadable")
        self.assertTrue("Failed to read" in str(ctx.exception) or "Incompatible" in str(ctx.exception))

        # Also test incompatible header (e.g. 4 columns instead of 58)
        old_headers = "col1,col2,col3,col4\nval1,val2,val3,val4\n"
        Path(report_file).write_text(old_headers, encoding="utf-8")
        hash_hdr_before = hashlib.sha256(Path(report_file).read_bytes()).hexdigest()
        target_selector_called.clear()

        with patch.object(signal, "signal"), self.assertRaises(RuntimeError) as ctx2:
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main>", "exec"), ns)

        hash_hdr_after = hashlib.sha256(Path(report_file).read_bytes()).hexdigest()
        self.assertEqual(hash_hdr_before, hash_hdr_after, "Incompatible header file must remain bit-for-bit unchanged")
        self.assertFalse(target_selector_called, "Science target selection must NOT run when header is incompatible")
        self.assertIn("Incompatible candidate CSV schema", str(ctx2.exception))

    def test_primary_writer_valid_empty_and_header_only_files_handled_cleanly(self):
        """
        Verify that a zero-byte file or a 58-column header-only file both load safely without error
        and permit candidate persistence without corruption.
        """
        report_file = os.path.join(self.test_dir, "empty_report.csv")
        Path(report_file).write_bytes(b"")  # 0 bytes

        tree = ast.parse(Path(ald.__file__).read_text(encoding="utf-8"))
        main_node = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")

        cand = {c: "val0" for c in ald.CANDIDATE_COLUMNS}
        cand.update({"tic_id": "33333333", "sector": 106, "period_days": 1.5, "t0_btjd": 2459002.0})

        class MockFuture:
            def result(self):
                return (cand["tic_id"], "CANDIDATE", "Candidate Found", cand)

        class MockExecutor:
            def __init__(self, **kw): pass
            def submit(self, *args): return MockFuture()
            def shutdown(self, **kw): pass

        ns = dict(ald.__dict__)
        ns.update({
            "__name__": "__main__",
            "REPORT_FILE": report_file,
            "MAX_WORKERS": 1,
            "select_latest_sector_and_targets": lambda **kw: (106, ["33333333"]),
            "ProcessPoolExecutor": MockExecutor,
            "as_completed": lambda fs: list(fs),
            "append_scanned_history": lambda tic: None,
            "write_log": lambda *a, **k: None,
            "write_event": lambda *a, **k: None
        })

        with patch.object(signal, "signal"):
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main>", "exec"), ns)

        df_out = pd.read_csv(report_file)
        self.assertEqual(len(df_out), 1)
        self.assertEqual(str(df_out.iloc[0]["tic_id"]), "33333333")

    def test_incompatible_historical_header_rejected_without_modifying_file(self):
        """
        Verify that if an existing candidate CSV has an incompatible historical header (e.g. 28 columns),
        search_toi_companions and aldebaran both refuse to append and raise RuntimeError,
        preserving the historical file untouched.
        """
        old_headers = [
            "run_id", "tic_id", "sector", "is_toi_host", "known_toi_context",
            "candidate_type", "period_days", "t0_btjd", "duration_hours",
            "min_kepler_dur_hours", "max_kepler_dur_hours", "n_transits",
            "depth_ppm", "crowdsap", "sde", "r_planet_earth", "r_star_sun",
            "m_star_sun", "teff_k", "gaia_ruwe", "odd_even_sigma",
            "centroid_shift_arcsec", "archival_sectors", "archival_snr_sigma",
            "archival_trials", "archival_snr_crit", "cramer_rao_sigma_p_sec",
            "plot_path", "exofop_url"
        ]

        # Create a historical CSV with old schema
        hist_file = os.path.join(self.test_dir, "toi_companions_summary.csv")
        df_hist = pd.DataFrame(columns=old_headers)
        # Add a dummy historical candidate row
        dummy_row = {col: "hist_val" for col in old_headers}
        dummy_row["tic_id"] = "11111111"
        dummy_row["sector"] = 99
        dummy_row["run_id"] = "run_historical"
        df_hist = pd.concat([df_hist, pd.DataFrame([dummy_row])], ignore_index=True)
        df_hist.to_csv(hist_file, index=False)

        hist_hash_before = hashlib.sha256(Path(hist_file).read_bytes()).hexdigest()

        # 1. Attempt to initialize in append mode over incompatible file must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx1:
            stc.initialize_output_artifacts(self.test_dir, run_id="run_new", append_mode=True)
        self.assertIn("Incompatible candidate CSV schema", str(ctx1.exception))

        # 2. Attempt to call _append_candidate_dedup over incompatible file must raise RuntimeError
        mock_new_cand = {col: "new_val" for col in stc.CANDIDATE_COLUMNS}
        df_new = pd.DataFrame([mock_new_cand], columns=stc.CANDIDATE_COLUMNS)
        with self.assertRaises(RuntimeError) as ctx2:
            stc._append_candidate_dedup(hist_file, df_new, tic="22222222", sector=106, run_id="run_new")
        self.assertIn("Incompatible candidate CSV schema", str(ctx2.exception))

        # Verify historical file was NOT altered in any way
        hist_hash_after = hashlib.sha256(Path(hist_file).read_bytes()).hexdigest()
        self.assertEqual(hist_hash_before, hist_hash_after, "Historical file must remain bit-for-bit unchanged on rejection")


class TestDiagnosticAttemptIsolationAndStaleAttributionPrevention(unittest.TestCase):
    """
    R4 Acceptance checks:
    Verifies that archival vetting diagnostics are bound strictly to current run_id
    and attempt_id, preventing stale evidence from being attributed across runs,
    across same-run retries, or to early rejections / pipeline errors before archive.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ald_test_r4_")
        ald.configure_environment(output_dir=self.test_dir, run_id="r4_setup_run")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        ald.configure_environment(output_dir="output", run_id="run_default")

    def test_two_runs_same_output_dir_early_rejection_no_stale_evidence(self):
        """
        Verify that when an old run records an archival diagnostic in the shared output dir,
        a new run with early rejection (REJECT_CROWDSAP) on the same TIC/sector does NOT
        inherit or transport the old run's archival diagnostic.
        """
        tic_id = "11223344"
        sector = 106

        # Run 1 writes diagnostic for TIC 11223344
        ald.configure_environment(output_dir=self.test_dir, run_id="run_alpha")
        ald.record_archival_diagnostic({
            "run_id": "run_alpha",
            "attempt_id": "att_alpha_01",
            "tic_id": tic_id,
            "sector": sector,
            "arch_ok": False,
            "arch_status": "Archival non-detection in Sector 80",
            "archival_products_attempted": ["old-alpha-product-80"],
            "archival_products_failed": [{"product_id": "old-alpha-product-80", "stage": "EVALUATION", "reason": "non-detection"}],
        })

        # Run 2 starts in same output directory
        ald.configure_environment(output_dir=self.test_dir, run_id="run_beta")
        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "112.01", "period": 5.0}]), \
             patch.object(ald, "analyze_single_target", return_value=ald.AnalysisResult(
                 tic_id, "REJECT_CROWDSAP", "REJECT: Severe aperture crowding", None, run_id="run_beta", attempt_id="att_beta_01"
             )):
            res_beta = stc.process_toi_host(tic_id, sector, run_id="run_beta", attempt_id="att_beta_01")

        self.assertEqual(res_beta["run_id"], "run_beta")
        self.assertEqual(res_beta["status"], "REJECT_CROWDSAP")
        self.assertEqual(res_beta["error_detail"], "", "Early rejection in new run must NOT have error_detail containing old archival diagnostics")
        self.assertNotIn("old-alpha-product-80", res_beta["error_detail"])

        # Also test deliberate stale-lookup negative control
        def deliberate_faulty_lookup(t_id, sec_num):
            # Faulty lookup ignoring run_id/attempt_id
            rec = ald.get_latest_archival_diagnostic(t_id, sec_num, output_dir=self.test_dir)
            if rec and rec.get("run_id") != "run_beta":
                raise AssertionError(f"Stale attribution fault: record from '{rec.get('run_id')}' attached to 'run_beta'")

        with self.assertRaises(AssertionError) as ctx:
            deliberate_faulty_lookup(tic_id, sector)
        self.assertIn("Stale attribution fault", str(ctx.exception))

    def test_same_run_retry_isolation(self):
        """
        Verify that retries within the SAME run maintain strict attempt isolation:
        Attempt 1 (fails archive) -> Attempt 2 (fails early) does NOT inherit Attempt 1's archive data.
        Attempt 3 (fails archive) -> transports Attempt 3's archive data, NOT Attempt 1's.
        All records in archival_vetting_diagnostics.jsonl are preserved.
        """
        tic_id = "55667788"
        sector = 106
        run_id = "run_retry_test"
        ald.configure_environment(output_dir=self.test_dir, run_id=run_id)

        # Attempt 1: Reaches Filter 7 and fails archival check
        att1_diag = {
            "run_id": run_id,
            "attempt_id": "att_001",
            "tic_id": tic_id,
            "sector": sector,
            "arch_ok": False,
            "arch_status": "Non-detection in s80",
            "archival_products_attempted": ["product_att_1"],
            "archival_products_failed": [{"product_id": "product_att_1", "stage": "EVALUATION", "reason": "SNR < crit"}],
        }
        ald.record_archival_diagnostic(att1_diag)

        # Attempt 2: Retried in same run, but fails early at RUWE before archive
        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "556.01", "period": 3.0}]), \
             patch.object(ald, "analyze_single_target", return_value=ald.AnalysisResult(
                 tic_id, "REJECT_RUWE", "REJECT: RUWE > 1.4", None, run_id=run_id, attempt_id="att_002"
             )):
            res_att2 = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_002")

        self.assertEqual(res_att2["status"], "REJECT_RUWE")
        self.assertEqual(res_att2["error_detail"], "", "Same-run early retry must NOT inherit previous attempt's archival diagnostic")
        self.assertNotIn("product_att_1", res_att2["error_detail"])

        # Attempt 3: Retried again in same run, reaches Filter 7 and fails archive with different product
        att3_diag = {
            "run_id": run_id,
            "attempt_id": "att_003",
            "tic_id": tic_id,
            "sector": sector,
            "arch_ok": False,
            "arch_status": "Non-detection in s85",
            "archival_products_attempted": ["product_att_3"],
            "archival_products_failed": [{"product_id": "product_att_3", "stage": "EVALUATION", "reason": "SNR < crit"}],
        }
        ald.record_archival_diagnostic(att3_diag)

        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "556.01", "period": 3.0}]), \
             patch.object(ald, "analyze_single_target", return_value=ald.AnalysisResult(
                 tic_id, "REJECT_ARCHIVAL", "REJECT: Non-detection in s85", None,
                 run_id=run_id, attempt_id="att_003", archival_diagnostics=[att3_diag]
             )):
            res_att3 = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_003")

        self.assertEqual(res_att3["status"], "REJECT_ARCHIVAL")
        self.assertIn("product_att_3", res_att3["error_detail"])
        self.assertNotIn("product_att_1", res_att3["error_detail"])

        # Verify both diagnostic records survive on disk (no destructive global delete)
        all_recs = ald.load_archival_diagnostics(output_dir=self.test_dir)
        matching_recs = [r for r in all_recs if r.get("tic_id") == tic_id]
        self.assertEqual(len(matching_recs), 2, "Both Attempt 1 and Attempt 3 records must be preserved on disk")
        self.assertEqual(matching_recs[0]["attempt_id"], "att_001")
        self.assertEqual(matching_recs[1]["attempt_id"], "att_003")

    def test_production_path_early_rejection_and_error_isolation(self):
        """
        Verify through the actual production analyze_single_target implementation that early
        rejections (REJECT_CROWDSAP) and errors return AnalysisResult with archival_diagnostic=None,
        and that process_toi_host produces clean audit rows without archival diagnostics.
        """
        tic_id = "99001122"
        sector = 106
        n_pts = 1000
        t_arr = np.linspace(2459000.0, 2459027.0, n_pts)
        f_arr = np.ones(n_pts)
        ferr = np.full(n_pts, 0.001)

        # 1. Early rejection: CROWDSAP < 0.70
        with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", return_value=(t_arr, f_arr, ferr, np.zeros(n_pts), np.zeros(n_pts), 0.50)):  # CROWDSAP=0.50 -> REJECT_CROWDSAP

            res_crowd = ald.analyze_single_target(tic_id, sector, run_id="prod_early_run", attempt_id="att_crowd")
            self.assertEqual(res_crowd[1], "REJECT_CROWDSAP")
            self.assertIsNone(res_crowd.archival_diagnostic, "Early rejection must have archival_diagnostic is None")
            self.assertEqual(res_crowd.archival_diagnostics, [])

            # Companion caller integration
            with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "990.01", "period": 4.0}]):
                audit_crowd = stc.process_toi_host(tic_id, sector, run_id="prod_early_run", attempt_id="att_crowd")
            self.assertEqual(audit_crowd["status"], "REJECT_CROWDSAP")
            self.assertEqual(audit_crowd["error_detail"], "")

        # 2. Pipeline ERROR before archival verification (e.g. corrupted FITS extraction)
        with patch("lightkurve.search_lightcurve", return_value=[MagicMock()]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", side_effect=OSError("corrupted FITS table")):

            res_err = ald.analyze_single_target(tic_id, sector, run_id="prod_err_run", attempt_id="att_err")
            self.assertEqual(res_err[1], "ERROR")
            self.assertIsNone(res_err.archival_diagnostic)

            with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "990.01", "period": 4.0}]):
                audit_err = stc.process_toi_host(tic_id, sector, run_id="prod_err_run", attempt_id="att_err")
            self.assertEqual(audit_err["status"], "ERROR")
            self.assertIn("corrupted FITS table", audit_err["error_detail"])
            self.assertNotIn("ArchivalDiag", audit_err["error_detail"])

    def test_production_path_archival_rejection_attaches_current_attempt_evidence(self):
        """
        Verify through production analyze_single_target that when Filter 7 rejects,
        the AnalysisResult carries the exact current-attempt diagnostic, and process_toi_host
        transports it into error_detail.
        """
        tic_id = "77889900"
        sector = 106
        discovery_p = 3.0
        discovery_t0 = 2459001.0
        n_pts = 19440
        rng = np.random.default_rng(42)
        t_arr = np.linspace(2459000.0, 2459027.0, n_pts)
        f_arr = 1.0 + 1e-4 * rng.standard_normal(n_pts)
        ph = ((t_arr - discovery_t0 + 0.5 * discovery_p) % discovery_p) - 0.5 * discovery_p
        f_arr[np.abs(ph) < 0.04] -= 0.0010

        t_arch = np.linspace(2458000.0, 2458027.0, n_pts)
        f_arch = np.ones(n_pts)  # Flat prior -> non-detection veto
        mock_row = MagicMock()
        mock_row.mission = ["TESS Sector 80"]
        mock_row.obsid = "current-attempt-product-80"

        with patch("lightkurve.search_lightcurve", side_effect=[
            [MagicMock()],
            [mock_row]
        ]), \
             patch.object(ald, "safe_download", return_value=MagicMock()), \
             patch.object(ald, "extract_and_stitch_arrays", side_effect=[
                 (t_arr, f_arr, np.full(n_pts, 0.0001), np.zeros(n_pts), np.zeros(n_pts), 0.95),
                 (t_arch, f_arch, np.full(n_pts, 0.0001), None, None, 0.99)
             ]), \
             patch.object(ald, "clip_upper_outliers", side_effect=lambda t, f, e: (t, f, e)), \
             patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda t, f, **kw: np.ones_like(f)), \
             patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)), \
             patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]), \
             patch.object(ald, "get_stellar_and_gaia_params", return_value={"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05, "teff": 5800, "ruwe": 1.05}), \
             patch.object(ald, "verify_and_filter_tpf", return_value=(True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))):

            mock_bls = MagicMock()
            mock_bls.power.return_value = MagicMock(
                period=np.full(9000, discovery_p),
                power=np.full(9000, 100.0),
                duration=np.full(9000, 0.08),
                transit_time=np.full(9000, discovery_t0),
                depth=np.full(9000, 0.001)
            )
            with patch.object(ald, "BoxLeastSquares", return_value=mock_bls):
                res = ald.analyze_single_target(tic_id, sector, run_id="arch_run_42", attempt_id="att_arch_42")

        self.assertEqual(res[1], "REJECT_ARCHIVAL")
        self.assertIsNotNone(res.archival_diagnostic)
        self.assertEqual(res.archival_diagnostic["run_id"], "arch_run_42")
        self.assertEqual(res.archival_diagnostic["attempt_id"], "att_arch_42")
        self.assertEqual(res.archival_diagnostic["archival_products_attempted"], ["current-attempt-product-80"])

        with patch.object(ald, "analyze_single_target", return_value=res), \
             patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "778.01", "period": 3.0}]):
            audit_res = stc.process_toi_host(tic_id, sector, run_id="arch_run_42", attempt_id="att_arch_42")

        self.assertEqual(audit_res["status"], "REJECT_ARCHIVAL")
        self.assertIn("current-attempt-product-80", audit_res["error_detail"])
        transported = json.loads(audit_res["error_detail"])
        self.assertEqual(transported["attempt_id"], "att_arch_42")
        self.assertEqual(transported["run_id"], "arch_run_42")

    def test_primary_logger_uses_current_run_and_result_diagnostic(self):
        """
        Verify that in primary main execution, logging of archival diagnostics only occurs
        for REJECT_ARCHIVAL outcomes, correctly pulling the result-owned diagnostic.
        """
        import ast
        tree = ast.parse(Path(ald.__file__).read_text(encoding="utf-8"))
        main_node = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")

        report_file = os.path.join(self.test_dir, "logger_test_report.csv")
        logged_messages = []

        class MockFutureReject:
            def result(self):
                diag = {
                    "run_id": "test_logger_run",
                    "attempt_id": "att_log_01",
                    "archival_products_attempted": ["prod_logged_80"],
                    "archival_products_failed": [{"stage": "EVALUATION", "reason": "veto"}]
                }
                return ald.AnalysisResult("44556677", "REJECT_ARCHIVAL", "REJECT: Archival non-detection", None,
                                          run_id="test_logger_run", attempt_id="att_log_01", archival_diagnostics=[diag])

        class MockExecutor:
            def __init__(self, **kw): pass
            def submit(self, *args): return MockFutureReject()
            def shutdown(self, **kw): pass

        ns = dict(ald.__dict__)
        ns.update({
            "__name__": "__main__",
            "CURRENT_RUN_ID": "test_logger_run",
            "REPORT_FILE": report_file,
            "MAX_WORKERS": 1,
            "PREFERRED_SECTORS": [106],
            "MAX_TARGETS": 1,
            "select_latest_sector_and_targets": lambda **kw: (106, ["44556677"]),
            "ProcessPoolExecutor": MockExecutor,
            "as_completed": lambda fs: list(fs),
            "append_scanned_history": lambda tic: None,
            "write_log": lambda msg, *a, **k: logged_messages.append(msg),
            "write_event": lambda *a, **k: None
        })

        with patch.object(signal, "signal"):
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main>", "exec"), ns)

        diag_logs = [m for m in logged_messages if "Archival diagnostics" in m]
        self.assertEqual(len(diag_logs), 1, "Must log archival diagnostics for REJECT_ARCHIVAL")
        self.assertIn("prod_logged_80", diag_logs[0])

    def test_r4a_missing_own_diagnostic_does_not_borrow_historical_evidence(self):
        """
        R4a Acceptance Check:
        Verifies that when an attempt yields REJECT_ARCHIVAL but has NO diagnostic of its own
        (both for AnalysisResult with empty diagnostics and legacy 4-tuple with no identity),
        it does NOT borrow a diagnostic from an earlier attempt in the same run.
        Also tests that primary main logger does NOT log archival diagnostics when current attempt
        has no diagnostic, while normal current attempt transport and historical records on disk
        remain completely preserved.
        """
        tic_id = "88990011"
        sector = 106
        run_id = "same_run_r4a"
        ald.configure_environment(output_dir=self.test_dir, run_id=run_id)

        # 1. Historical record from an earlier attempt in the same run
        old_diag = {
            "run_id": run_id,
            "attempt_id": "old_attempt_r4a",
            "tic_id": tic_id,
            "sector": sector,
            "arch_ok": False,
            "arch_status": "Old failure",
            "archival_products_attempted": ["old-product-r4a"],
            "archival_products_failed": [{"product_id": "old-product-r4a", "stage": "EVALUATION", "reason": "veto"}]
        }
        ald.record_archival_diagnostic(old_diag)

        # Part A: AnalysisResult with current attempt_id but NO diagnostic
        empty_res = ald.AnalysisResult(
            tic_id, "REJECT_ARCHIVAL", "REJECT: current archive non-detection", None,
            run_id=run_id, attempt_id="new_attempt_a"
        )
        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "889.01", "period": 4.0}]), \
             patch.object(ald, "analyze_single_target", return_value=empty_res):
            res_a = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="new_attempt_a")

        self.assertEqual(res_a["status"], "REJECT_ARCHIVAL")
        self.assertEqual(res_a["error_detail"], "", "Must NOT borrow old attempt diagnostic when current AnalysisResult has none")
        self.assertNotIn("old-product-r4a", res_a["error_detail"])

        # Part B: Legacy 4-tuple without attempt identity
        legacy_res = (tic_id, "REJECT_ARCHIVAL", "REJECT: legacy target archive", None)
        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "889.01", "period": 4.0}]), \
             patch.object(ald, "analyze_single_target", return_value=legacy_res):
            res_b = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="new_attempt_b")

        self.assertEqual(res_b["status"], "REJECT_ARCHIVAL")
        self.assertEqual(res_b["error_detail"], "", "Legacy tuple without identity must NOT borrow historical diagnostic from previous attempt")
        self.assertNotIn("old-product-r4a", res_b["error_detail"])

        # Part C: Primary logger path with missing own diagnostic
        import ast
        tree = ast.parse(Path(ald.__file__).read_text(encoding="utf-8"))
        main_node = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")
        report_file = os.path.join(self.test_dir, "r4a_logger_report.csv")
        logged_msgs = []

        class MockFutureEmpty:
            def result(self):
                return ald.AnalysisResult(tic_id, "REJECT_ARCHIVAL", "REJECT: no diag", None,
                                          run_id=run_id, attempt_id="new_attempt_c")

        class MockExecutorEmpty:
            def __init__(self, **kw): pass
            def submit(self, *args): return MockFutureEmpty()
            def shutdown(self, **kw): pass

        ns = dict(ald.__dict__)
        ns.update({
            "__name__": "__main__",
            "CURRENT_RUN_ID": run_id,
            "REPORT_FILE": report_file,
            "MAX_WORKERS": 1,
            "PREFERRED_SECTORS": [sector],
            "MAX_TARGETS": 1,
            "select_latest_sector_and_targets": lambda **kw: (sector, [tic_id]),
            "ProcessPoolExecutor": MockExecutorEmpty,
            "as_completed": lambda fs: list(fs),
            "append_scanned_history": lambda tic: None,
            "write_log": lambda msg, *a, **k: logged_msgs.append(msg),
            "write_event": lambda *a, **k: None
        })
        with patch.object(signal, "signal"):
            exec(compile(ast.Module(body=main_node.body, type_ignores=[]), "<production-main-r4a>", "exec"), ns)

        diag_logs = [m for m in logged_msgs if "Archival diagnostics" in m]
        self.assertEqual(len(diag_logs), 0, "Primary logger must NOT log archival diagnostics when current attempt has no diagnostic")

        # Part D: Positive control - current attempt WITH own diagnostic transports properly
        curr_diag = {
            "run_id": run_id,
            "attempt_id": "good_attempt",
            "tic_id": tic_id,
            "sector": sector,
            "arch_ok": False,
            "arch_status": "Current failure",
            "archival_products_attempted": ["current-product-r4a"],
            "archival_products_failed": [{"product_id": "current-product-r4a", "stage": "EVALUATION", "reason": "veto"}]
        }
        good_res = ald.AnalysisResult(
            tic_id, "REJECT_ARCHIVAL", "REJECT: current archive veto", None,
            run_id=run_id, attempt_id="good_attempt", archival_diagnostics=[curr_diag]
        )
        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "889.01", "period": 4.0}]), \
             patch.object(ald, "analyze_single_target", return_value=good_res):
            res_good = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="good_attempt")

        self.assertEqual(res_good["status"], "REJECT_ARCHIVAL")
        self.assertIn("current-product-r4a", res_good["error_detail"])
        self.assertNotIn("old-product-r4a", res_good["error_detail"])

        # Part E: Verify disk preservation - old_diag survives on disk
        disk_recs = ald.load_archival_diagnostics(output_dir=self.test_dir)
        old_matches = [r for r in disk_recs if r.get("attempt_id") == "old_attempt_r4a"]
        self.assertEqual(len(old_matches), 1, "Historical diagnostic record must remain preserved on disk")

    def test_r4b_single_call_execution_and_error_propagation(self):
        """
        R4b Acceptance Check:
        Verifies that process_toi_host inspects callable signature separately from execution,
        invokes analyze_single_target exactly once, and propagates exceptions (RuntimeError,
        OSError, internal TypeError) directly to structured ERROR audit rows without retrying.
        Also verifies single-call execution for compatible legacy callables and ID-bearing callables.
        """
        tic_id = "99881122"
        sector = 106
        run_id = "single_call_run"
        ald.configure_environment(output_dir=self.test_dir, run_id=run_id)

        # Case 1: Callable raises RuntimeError (injected real analysis failure)
        calls_rt = []
        def failing_runtime(t_id, sec_num, run_id=None, attempt_id=None):
            calls_rt.append((run_id, attempt_id))
            raise RuntimeError("injected real analysis failure")

        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "998.01", "period": 5.0}]), \
             patch.object(ald, "analyze_single_target", failing_runtime):
            res_rt = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_rt_01")

        self.assertEqual(len(calls_rt), 1, "RuntimeError must result in exactly 1 call with zero retries")
        self.assertEqual(calls_rt[0], (run_id, "att_rt_01"))
        self.assertEqual(res_rt["status"], "ERROR")
        self.assertEqual(res_rt["reason"], "RuntimeError")
        self.assertIn("injected real analysis failure", res_rt["message"])
        self.assertIn("injected real analysis failure", res_rt["error_detail"])

        # Case 2: Callable raises OSError (e.g. disk/I/O failure during analysis)
        calls_os = []
        def failing_os(t_id, sec_num, run_id=None, attempt_id=None):
            calls_os.append((run_id, attempt_id))
            raise OSError("disk read failure during FITS extraction")

        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "998.01", "period": 5.0}]), \
             patch.object(ald, "analyze_single_target", failing_os):
            res_os = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_os_01")

        self.assertEqual(len(calls_os), 1, "OSError must result in exactly 1 call with zero retries")
        self.assertEqual(calls_os[0], (run_id, "att_os_01"))
        self.assertEqual(res_os["status"], "ERROR")
        self.assertEqual(res_os["reason"], "OSError")
        self.assertIn("disk read failure during FITS extraction", res_os["message"])

        # Case 3: Callable raises internal TypeError (e.g. bug inside analyze_single_target)
        calls_te = []
        def failing_internal_type_error(t_id, sec_num, run_id=None, attempt_id=None):
            calls_te.append((run_id, attempt_id))
            raise TypeError("unsupported operand type inside analysis algorithm")

        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "998.01", "period": 5.0}]), \
             patch.object(ald, "analyze_single_target", failing_internal_type_error):
            res_te = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_te_01")

        self.assertEqual(len(calls_te), 1, "Internal TypeError must NOT trigger retry; exactly 1 call made")
        self.assertEqual(calls_te[0], (run_id, "att_te_01"))
        self.assertEqual(res_te["status"], "ERROR")
        self.assertEqual(res_te["reason"], "TypeError")
        self.assertIn("unsupported operand type inside analysis algorithm", res_te["message"])

        # Case 4: Compatible legacy callable taking only (tic_id, sector)
        calls_leg = []
        def legacy_callable(t_id, sec_num):
            calls_leg.append((t_id, sec_num))
            return (t_id, "REJECT_CROWDSAP", "REJECT: Crowding", None)

        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "998.01", "period": 5.0}]), \
             patch.object(ald, "analyze_single_target", legacy_callable):
            res_leg = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_leg_01")

        self.assertEqual(len(calls_leg), 1, "Legacy callable must be called exactly once")
        self.assertEqual(calls_leg[0], (tic_id, sector))
        self.assertEqual(res_leg["status"], "REJECT_CROWDSAP")

        # Case 5: Correct ID-bearing callable
        calls_id = []
        def id_callable(t_id, sec_num, run_id=None, attempt_id=None):
            calls_id.append((run_id, attempt_id))
            return ald.AnalysisResult(t_id, "REJECT_RUWE", "REJECT: RUWE > 1.4", None, run_id=run_id, attempt_id=attempt_id)

        with patch.object(ald, "get_known_transits_for_tic", return_value=[{"toi": "998.01", "period": 5.0}]), \
             patch.object(ald, "analyze_single_target", id_callable):
            res_id = stc.process_toi_host(tic_id, sector, run_id=run_id, attempt_id="att_id_01")

        self.assertEqual(len(calls_id), 1, "ID-bearing callable must be called exactly once")
        self.assertEqual(calls_id[0], (run_id, "att_id_01"))
        self.assertEqual(res_id["status"], "REJECT_RUWE")


if __name__ == "__main__":
    unittest.main()
