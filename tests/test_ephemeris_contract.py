"""
tests/test_ephemeris_contract.py — Differential and Metamorphic Ephemeris Test Suite
===================================================================================
Verifies cross-tier ephemeris, time coordinate, and phase wrapping consistency
under ALD-MATH-03-S2 requirements:
  1. BTJD <-> BJD_TDB coordinate offsets with bounded float64 roundoff.
  2. Phase wrapping boundary conditions in [0, 1) and centered [-P/2, P/2).
  3. Numerically bounded integer-period shift equivalence away from boundaries.
  4. Cyclic orbital distance metric resolving branch-cut discontinuities (SCI01).
  5. Cross-tier ephemeris propagation from Tier 1 to Tier 4.
  6. Retained SDE-based timing approximation and input negative controls.
"""

import math
import unittest
from unittest import mock
from types import SimpleNamespace
import contextlib
import io
import sys
import types
import numpy as np

from ephemeris_contract import (
    EPHEMERIS_SCHEMA_VERSION,
    BTJD_OFFSET,
    BJD_BTJD_OFFSET,
    TIME_SYSTEM_BTJD,
    TIME_STANDARD_TDB,
    bjd_to_btjd,
    btjd_to_bjd,
    btjd_to_year,
    fold_phase_0to1,
    fold_phase_centered,
    cyclic_orbital_distance,
    compute_orbit_number,
    predict_transit_time,
    normalize_epoch_to_baseline,
    calculate_cyclic_timing_residual,
    days_to_hours,
    hours_to_days,
    cramer_rao_timing_uncertainty,
)

import aldebaran as tier1
import pleiades as tier2
import ctoi_packager as tier3_ctoi
import antares_packager as tier3_antares
import antares as tier4_antares
import validate_triceratops as tier4_triceratops
import known_signals
import loyo_validator


class TestTimescaleConversions(unittest.TestCase):
    """Verifies BTJD <-> BJD_TDB timescale arithmetic, roundtrips, and boundaries."""

    def test_constants_and_metadata(self):
        self.assertEqual(EPHEMERIS_SCHEMA_VERSION, "ephemeris_v1")
        self.assertEqual(BTJD_OFFSET, 2457000.0)
        self.assertEqual(BJD_BTJD_OFFSET, 2457000.0)
        self.assertEqual(TIME_SYSTEM_BTJD, "BTJD")
        self.assertEqual(TIME_STANDARD_TDB, "TDB")

    def test_scalar_roundtrip_precision(self):
        """
        Tests that BJD <-> BTJD roundtrips within float64 ULP spacing at magnitude ~2.46e6.
        At BJD ~2,458,000, 1 ULP = 2^(21) * 2^(-52) ~ 4.65e-10 days (~40 microseconds).
        """
        test_epochs_btjd = [
            0.0,
            1325.12345678,
            2000.0,
            2450.98765432,
            3500.5,
            4999.11122233,
        ]
        for t_btjd in test_epochs_btjd:
            bjd_val = btjd_to_bjd(t_btjd, strict=True)
            self.assertAlmostEqual(bjd_val, t_btjd + 2457000.0, delta=1e-9)
            t_back = bjd_to_btjd(bjd_val, strict=True)
            self.assertAlmostEqual(t_back, t_btjd, delta=5e-10)

    def test_array_vectorization(self):
        """Tests vectorized NumPy conversions on multi-sector observation grids."""
        t_grid_btjd = np.linspace(1325.0, 3500.0, 10000)
        bjd_arr = btjd_to_bjd(t_grid_btjd, strict=True)
        self.assertTrue(np.allclose(bjd_arr - t_grid_btjd, 2457000.0, atol=1e-9, rtol=0.0))

        btjd_rec = bjd_to_btjd(bjd_arr, strict=True)
        self.assertTrue(np.allclose(btjd_rec, t_grid_btjd, atol=5e-10, rtol=0.0))

    def test_auto_detection_boundaries(self):
        """Tests non-strict auto-detection of already-converted vs uncoverted epochs."""
        # Value < 2,400,000 is recognized as BTJD:
        self.assertEqual(bjd_to_btjd(2000.0, strict=False), 2000.0)
        # Value > 2,400,000 is converted to BTJD:
        self.assertEqual(bjd_to_btjd(2459000.0, strict=False), 2000.0)

        # Value > 2,400,000 is recognized as BJD:
        self.assertEqual(btjd_to_bjd(2459000.0, strict=False), 2459000.0)
        # Value < 2,400,000 is converted to BJD:
        self.assertEqual(btjd_to_bjd(2000.0, strict=False), 2459000.0)

    def test_btjd_to_year(self):
        """Verifies calendar year mapping anchored to 2014-12-08."""
        self.assertEqual(btjd_to_year(0.0), 2014)
        self.assertEqual(btjd_to_year(1325.0), 2018)  # TESS launch / Sector 1
        self.assertEqual(btjd_to_year(2000.0), 2020)
        self.assertEqual(btjd_to_year(3500.0), 2024)

    def test_unit_conversions(self):
        self.assertEqual(days_to_hours(1.0), 24.0)
        self.assertAlmostEqual(days_to_hours(0.05), 1.2, places=12)
        self.assertEqual(hours_to_days(24.0), 1.0)
        self.assertAlmostEqual(hours_to_days(1.2), 0.05, places=12)


class TestPhaseFoldingAndWrapping(unittest.TestCase):
    """Verifies phase folding mathematical properties and boundary conditions."""

    def test_fold_phase_0to1_bounds(self):
        """Tests that fold_phase_0to1 produces values strictly in [0.0, 1.0)."""
        period = 3.14159
        t0 = 2000.0
        # Times spanning multiple orbits before and after t0
        t_vals = np.linspace(1500.0, 2500.0, 50000)
        phases = fold_phase_0to1(t_vals, t0, period)

        self.assertTrue(np.all(phases >= 0.0))
        self.assertTrue(np.all(phases < 1.0))

    def test_fold_phase_0to1_exact_boundaries(self):
        """Tests exact mid-transit and orbital edges."""
        p = 4.0
        t0 = 2000.0

        # Exact transit midpoints must evaluate to phase 0.0
        for n in [-100, -1, 0, 1, 100]:
            t_mid = t0 + n * p
            self.assertEqual(fold_phase_0to1(t_mid, t0, p), 0.0)

        # Secondary eclipse on circular orbit (half-period)
        self.assertAlmostEqual(fold_phase_0to1(t0 + 0.5 * p, t0, p), 0.5, places=12)
        self.assertAlmostEqual(fold_phase_0to1(t0 - 0.5 * p, t0, p), 0.5, places=12)

    def test_fold_phase_centered_bounds_and_symmetry(self):
        """Tests centered phase interval [-P/2, P/2) in days and [-0.5, 0.5) in cycles."""
        period = 5.0
        t0 = 2000.0
        half_p = 0.5 * period

        t_vals = np.linspace(1900.0, 2100.0, 20000)
        ph_days = fold_phase_centered(t_vals, t0, period, in_days=True)
        ph_cycles = fold_phase_centered(t_vals, t0, period, in_days=False)

        # Bounds check
        self.assertTrue(np.all(ph_days >= -half_p))
        self.assertTrue(np.all(ph_days < half_p))
        self.assertTrue(np.all(ph_cycles >= -0.5))
        self.assertTrue(np.all(ph_cycles < 0.5))

        # Mid-transit center must be 0.0
        self.assertEqual(fold_phase_centered(t0, t0, period, in_days=True), 0.0)
        self.assertEqual(fold_phase_centered(t0, t0, period, in_days=False), 0.0)

        # Antisymmetry around mid-transit: tau(t0 - dt) == -tau(t0 + dt)
        dt = 0.35 * period
        tau_pos = fold_phase_centered(t0 + dt, t0, period, in_days=True)
        tau_neg = fold_phase_centered(t0 - dt, t0, period, in_days=True)
        self.assertAlmostEqual(tau_pos, -tau_neg, places=12)

    def test_invalid_period_fails_closed(self):
        with self.assertRaises(ValueError):
            fold_phase_0to1(2000.0, 2000.0, 0.0)
        with self.assertRaises(ValueError):
            fold_phase_0to1(2000.0, 2000.0, -2.5)
        with self.assertRaises(ValueError):
            fold_phase_centered(2000.0, 2000.0, -1.0)


class TestMetamorphicIntegerPeriodInvariance(unittest.TestCase):
    """
    Metamorphic Invariance: Physical observables and phase coordinates must be
    equivalent within float64 tolerances away from boundaries under integer-cycle shifts.
    """

    def test_integer_period_shift_invariance(self):
        p = 3.456789
        t0 = 2000.12345
        t_arr = np.linspace(1950.0, 2050.0, 1000)

        base_0to1 = fold_phase_0to1(t_arr, t0, p)
        base_days = fold_phase_centered(t_arr, t0, p, in_days=True)
        base_cycles = fold_phase_centered(t_arr, t0, p, in_days=False)

        shifts = [-500, -50, -1, 1, 50, 500]
        for k in shifts:
            shifted_t0 = t0 + k * p
            shifted_0to1 = fold_phase_0to1(t_arr, shifted_t0, p)
            shifted_days = fold_phase_centered(t_arr, shifted_t0, p, in_days=True)
            shifted_cycles = fold_phase_centered(t_arr, shifted_t0, p, in_days=False)

            self.assertTrue(
                np.allclose(shifted_0to1, base_0to1, atol=1e-12, rtol=0.0),
                f"fold_phase_0to1 violated invariance at k={k}"
            )
            self.assertTrue(
                np.allclose(shifted_days, base_days, atol=1e-12, rtol=0.0),
                f"fold_phase_centered(in_days=True) violated invariance at k={k}"
            )
            self.assertTrue(
                np.allclose(shifted_cycles, base_cycles, atol=1e-12, rtol=0.0),
                f"fold_phase_centered(in_days=False) violated invariance at k={k}"
            )

    def test_orbit_number_shift_consistency(self):
        """Orbit numbers must shift exactly by -k when t0 shifts by +k*P."""
        p = 2.5
        t0 = 2000.0
        t_sample = np.array([1997.5, 2000.0, 2002.5, 2005.0])

        n_base = compute_orbit_number(t_sample, t0, p)
        np.testing.assert_array_equal(n_base, [-1, 0, 1, 2])

        for k in [-10, -3, 2, 7]:
            shifted_t0 = t0 + k * p
            n_shifted = compute_orbit_number(t_sample, shifted_t0, p)
            np.testing.assert_array_equal(n_shifted, n_base - k)

            # Reconstructed transit time must be strictly identical:
            t_pred_base = predict_transit_time(t0, p, n_base)
            t_pred_shifted = predict_transit_time(shifted_t0, p, n_shifted)
            np.testing.assert_allclose(t_pred_shifted, t_pred_base, atol=1e-12, rtol=0.0)

    def test_epoch_normalization_to_baseline(self):
        """Tests normalizing t0 by integer multiples of P into the observation window."""
        p = 4.0
        t0 = 1000.0  # Far away
        ref_time = 2000.0  # Current sector midpoint

        t0_norm = normalize_epoch_to_baseline(t0, p, ref_time)
        self.assertLessEqual(abs(t0_norm - ref_time), 0.5 * p)
        # Must be integer shift:
        k = (t0_norm - t0) / p
        self.assertAlmostEqual(k, round(k), places=12)

        # Projected transits from t0_norm and t0 must be identical
        self.assertEqual(predict_transit_time(t0, p, 250), predict_transit_time(t0_norm, p, 0))


class TestCyclicOrbitalDistance(unittest.TestCase):
    """Verifies cyclic distance across periodic orbital boundaries (SCI01 resolution)."""

    def test_control_phase_null_test_distances(self):
        """
        Tests the 8 archival control phases: [0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85].
        Verifies that phases 0.70-0.85 correctly map to distance 0.15-0.30 cycles from mid-transit.
        """
        p = 4.0
        ctrl_phases = [0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85]
        expected_distances_cycles = [0.15, 0.20, 0.25, 0.30, 0.30, 0.25, 0.20, 0.15]

        for cp, exp_d in zip(ctrl_phases, expected_distances_cycles):
            # Phase 0.0 (in-transit) vs control phase
            dist_cycles = abs(cyclic_orbital_distance(0.0, cp))
            self.assertAlmostEqual(dist_cycles, exp_d, places=12)

            dist_days = abs(cyclic_orbital_distance(0.0, cp * p, period=p))
            self.assertAlmostEqual(dist_days, exp_d * p, places=12)

    def test_signed_vs_unsigned_equivalence(self):
        """Verifies that signed phase -0.20 and positive phase 0.80 yield zero distance."""
        dist = cyclic_orbital_distance(-0.20, 0.80)
        self.assertAlmostEqual(dist, 0.0, places=12)

        dist_days = cyclic_orbital_distance(-0.20 * 5.0, 0.80 * 5.0, period=5.0)
        self.assertAlmostEqual(dist_days, 0.0, places=12)

    def test_naive_subtraction_negative_control(self):
        """Demonstrates that naive subtraction fails at 0.70-0.85 while cyclic distance succeeds."""
        p = 4.0
        cp = 0.80
        # For a cadence near phase -0.20:
        cadence_phase = -0.20
        # Naive subtraction gives abs(-0.20 - 0.80) = 1.00 cycles
        naive_dist = abs(cadence_phase - cp)
        self.assertAlmostEqual(naive_dist, 1.00, places=12)

        # Cyclic distance correctly gives 0.00 cycles
        cyclic_dist = abs(cyclic_orbital_distance(cadence_phase, cp))
        self.assertAlmostEqual(cyclic_dist, 0.00, places=12)


class TestCramerRaoTimingUncertainty(unittest.TestCase):
    """Verifies the retained timing approximation arithmetic, not uncertainty calibration."""

    def test_scaling_laws(self):
        dur = 0.08  # ~2 hours
        p = 3.0
        baseline = 27.0  # 1 sector
        sde = 10.0

        sigma_t0, sigma_p = cramer_rao_timing_uncertainty(dur, sde, baseline, p)
        # sigma_t0 = 0.5 * 0.08 / 10.0 = 0.004 d (~5.76 min)
        self.assertAlmostEqual(sigma_t0, 0.004, places=6)

        # n_cyc = 27.0 / 3.0 = 9.0
        # sigma_p = sqrt(12) * 0.004 / 9.0 = 0.0015396 d
        exp_sigma_p = math.sqrt(12.0) * 0.004 / 9.0
        self.assertAlmostEqual(sigma_p, exp_sigma_p, places=8)

    def test_sde_floor_at_5(self):
        """Verifies SDE is clamped to at least 5.0 to prevent division by zero / exploding bounds."""
        dur = 0.05
        p = 2.0
        baseline = 25.0
        sigma_t0_low, _ = cramer_rao_timing_uncertainty(dur, 2.0, baseline, p)
        sigma_t0_five, _ = cramer_rao_timing_uncertainty(dur, 5.0, baseline, p)
        self.assertEqual(sigma_t0_low, sigma_t0_five)

    def test_negative_controls(self):
        with self.assertRaises(ValueError):
            cramer_rao_timing_uncertainty(-0.05, 10.0, 27.0, 3.0)
        with self.assertRaises(ValueError):
            cramer_rao_timing_uncertainty(0.05, 10.0, -27.0, 3.0)
        with self.assertRaises(ValueError):
            cramer_rao_timing_uncertainty(0.05, 10.0, 27.0, -3.0)


class TestCrossTierCompatibility(unittest.TestCase):
    """
    Verifies that all 4 production tiers share identical mathematical semantics
    for time coordinates, phase folding, and epoch projection.
    """

    def test_tier1_tier2_phase_folding_equivalence(self):
        """Tier 1 and Tier 2 secondary scan phase fold must be bitwise/numerically identical."""
        t = np.linspace(2000.0, 2027.0, 1000)
        p = 3.5
        t0 = 2001.2

        ph_tier1 = tier1.fold_phase_0to1(t, t0, p)
        ph_tier2 = tier2.fold_phase_0to1(t, t0, p)
        ph_direct = fold_phase_0to1(t, t0, p)

        np.testing.assert_array_equal(ph_tier1, ph_direct)
        np.testing.assert_array_equal(ph_tier2, ph_direct)

    def test_tier3_packager_bjd_export_compatibility(self):
        """CTOI and Antares packagers must produce identical BJD epochs from BTJD."""
        t0_btjd = 2450.123456
        bjd_ctoi = tier3_ctoi.btjd_to_bjd(t0_btjd, strict=True)
        bjd_antares = tier3_antares.btjd_to_bjd(t0_btjd, strict=True)
        bjd_direct = btjd_to_bjd(t0_btjd, strict=True)

        self.assertEqual(bjd_ctoi, bjd_direct)
        self.assertEqual(bjd_antares, bjd_direct)
        self.assertAlmostEqual(bjd_ctoi, t0_btjd + 2457000.0, places=12)

    def test_tier4_antares_and_packager_refinement_math_parity(self):
        """Antares standalone and Antares packager must use identical centered phase and orbit logic."""
        t = np.linspace(1500.0, 3000.0, 5000)
        p = 4.2
        t0 = 1502.1

        # Centered phase
        ph_antares = tier4_antares.fold_phase_centered(t, t0, p, in_days=True)
        ph_packager = tier3_antares.fold_phase_centered(t, t0, p, in_days=True)
        ph_direct = fold_phase_centered(t, t0, p, in_days=True)

        np.testing.assert_array_equal(ph_antares, ph_direct)
        np.testing.assert_array_equal(ph_packager, ph_direct)

        # Orbit numbering
        orb_antares = tier4_antares.compute_orbit_number(t, t0, p)
        orb_packager = tier3_antares.compute_orbit_number(t, t0, p)
        orb_direct = compute_orbit_number(t, t0, p)

        np.testing.assert_array_equal(orb_antares, orb_direct)
        np.testing.assert_array_equal(orb_packager, orb_direct)

    def test_tier4_triceratops_phase_and_epoch_compatibility(self):
        """TRICERATOPS helper integration must match canonical helpers."""
        epoch_bjd = 2459450.5
        t0_btjd = tier4_triceratops.bjd_to_btjd(epoch_bjd)
        self.assertEqual(t0_btjd, 2450.5)

        t_sample = np.linspace(2450.0, 2455.0, 100)
        p = 2.0
        ph_days = tier4_triceratops.fold_phase_centered(t_sample, t0_btjd, p, in_days=True)
        ph_cycles = tier4_triceratops.fold_phase_centered(t_sample, t0_btjd, p, in_days=False)

        np.testing.assert_array_equal(ph_days, fold_phase_centered(t_sample, t0_btjd, p, in_days=True))
        np.testing.assert_array_equal(ph_cycles, fold_phase_centered(t_sample, t0_btjd, p, in_days=False))

    def test_known_signals_ephemeris_matching_parity(self):
        """known_signals.calculate_cyclic_timing_residual must match ephemeris_contract directly."""
        cand_p = 3.5001
        cand_t0 = 2000.10
        cat_p = 3.5000
        cat_t0 = 1500.10  # Shuffled by ~142 orbits

        res_ks = known_signals.calculate_cyclic_timing_residual(cand_p, cand_t0, cat_p, cat_t0)
        res_direct = calculate_cyclic_timing_residual(cand_p, cand_t0, cat_p, cat_t0)

        self.assertEqual(res_ks, res_direct)
        # Shift cat_t0 by arbitrary 100 orbits, result must remain invariant:
        res_shifted = known_signals.calculate_cyclic_timing_residual(cand_p, cand_t0, cat_p, cat_t0 + 100 * cat_p)
        self.assertAlmostEqual(res_shifted[0], res_ks[0], places=10)
        self.assertAlmostEqual(res_shifted[1], res_ks[1], places=10)
        self.assertAlmostEqual(res_shifted[2], res_ks[2], places=10)


class TestProductionPhaseConsumers(unittest.TestCase):
    """Exercise actual mask/binning consumers, not just shared imported aliases."""

    @staticmethod
    def legacy_cycles(t, t0, period):
        return ((t - t0) / period + 0.5) % 1.0 - 0.5

    def test_loyo_contact_membership_preserves_predecessor(self):
        t0, period, depth, duration = 2000.1, 1.3, 0.001, 2.4
        contact = 1350.1499999999999
        t = np.array([np.nextafter(contact, -np.inf), contact,
                      np.nextafter(contact, np.inf), t0, t0 + 0.2])
        phase = self.legacy_cycles(t, t0, period)
        expected = np.ones_like(t)
        expected[np.abs(phase * period * 24.0) <= duration / 2.0] = 1.0 - depth
        np.testing.assert_array_equal(
            loyo_validator.box_transit_model(t, t0, period, depth, duration), expected)
        self.assertEqual(expected[1], 1.0)
        # Rejected day-first mutation changes actual production mask membership.
        def day_first(t, t0, period, in_days=False):
            days = ((t - t0 + 0.5 * period) % period) - 0.5 * period
            return days if in_days else days / period
        with mock.patch.object(loyo_validator, "fold_phase_centered", day_first):
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(
                    loyo_validator.box_transit_model(t, t0, period, depth, duration), expected)

    def test_loyo_holdout_counts_preserve_predecessor(self):
        t0, period, duration = 2000.1, 1.3, 2.4
        t = np.array([1350.1499999999999] * 3 + [t0, t0 + 0.01, t0 - 0.01]
                     + list(t0 + np.linspace(0.06, 0.2, 20)))
        f = 1.0 + np.linspace(-1e-4, 1e-4, len(t))
        hours = self.legacy_cycles(t, t0, period) * period * 24.0
        result = loyo_validator.evaluate_holdout_sector(t, f, np.full_like(t, 1e-4),
                                                       t0, period, duration, 0.001)
        self.assertEqual(result["n_in"], int(np.sum(np.abs(hours) <= duration / 2.0)))
        self.assertEqual(result["n_out"], int(np.sum(
            (np.abs(hours) > duration / 2.0) & (np.abs(hours) <= 2.5 * duration))))

    def test_triceratops_consumes_legacy_dimensionless_days(self):
        # Synthetic local photometry; stop at calc_probs, before simulation/output.
        t0, period = 2000.1, 1.3
        t = np.array([1350.1499999999999] * 3 + list(t0 + np.linspace(-0.19, 0.19, 121)))
        flux = 1.0 + np.arange(len(t)) * 1e-6
        lc = mock.Mock()
        lc.remove_nans.return_value.remove_outliers.return_value = lc
        lc.time = SimpleNamespace(btjd=t)
        lc.flux = SimpleNamespace(value=flux)
        lc.flux_err = SimpleNamespace(value=np.full_like(t, 1e-4))
        target = mock.Mock()
        target.sectors = [1]
        target.stars = tier3_ctoi.pd.DataFrame([{"rad": 1.0, "Teff": 5700.0}])
        class Captured(BaseException):
            pass
        target.calc_probs.side_effect = Captured
        package = types.ModuleType("triceratops")
        module = types.ModuleType("triceratops.triceratops")
        module.target = mock.Mock(return_value=target)
        package.triceratops = module
        days = self.legacy_cycles(t, t0, period) * period
        duration_hours, n_bins = 1.2, 4
        dur_days = duration_hours / 24.0
        keep = np.abs(days) <= 2.0 * dur_days
        bins = np.linspace(-2.0 * dur_days, 2.0 * dur_days, n_bins + 1)
        centers, medians = [], []
        for i in range(n_bins):
            m = keep & (days >= bins[i]) & (days < bins[i + 1])
            if m.sum() >= 3:
                centers.append(0.5 * (bins[i] + bins[i + 1]))
                medians.append(float(np.nanmedian(flux[m])))
        baseline = np.median(flux[(np.abs(days) > 0.6 * dur_days) & keep])
        normalized_medians = []
        for i in range(n_bins):
            m = keep & (days >= bins[i]) & (days < bins[i + 1])
            if m.sum() >= 3:
                normalized_medians.append(float(np.nanmedian((flux / baseline)[m])))
        expected_flux = np.array(normalized_medians) / np.nanmedian(normalized_medians)
        with mock.patch.dict(sys.modules, {"triceratops": package,
                                            "triceratops.triceratops": module}), \
             mock.patch.object(tier4_triceratops, "version", return_value="1.0.21"), \
             mock.patch.object(tier4_triceratops, "_download_spoc_sector",
                               return_value=(t, flux, np.full_like(t, 1e-4), np.array([[1, 1]]))), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(Captured):
                tier4_triceratops.run_triceratops_validation(
                    "123", period, t0, duration_hours, 1000.0, [1], n_mc=1, n_bins=n_bins)
        supplied = target.calc_probs.call_args.kwargs
        np.testing.assert_array_equal(supplied["time"], centers)
        np.testing.assert_array_equal(supplied["flux_0"], expected_flux)

    def test_vector_test_detects_86_second_conversion_mutation(self):
        test = TestTimescaleConversions("test_array_vectorization")
        # Proves the actual precision regression test rejects the previous survivor.
        module = sys.modules[__name__]
        original = module.btjd_to_bjd
        with mock.patch.object(module, "btjd_to_bjd",
                               side_effect=lambda *a, **kw: original(*a, **kw) + 0.001):
            with self.assertRaises(AssertionError):
                test.test_array_vectorization()


class TestProductionDurationUnits(unittest.TestCase):
    """Unit extraction preserves numeric operation order and real consumers."""

    def test_numeric_dtype_and_nonfinite_parity(self):
        for dtype in (np.float32, np.float64):
            array = np.array([-np.inf, -2.4, -0.0, 0.0, 0.1, 2.4, np.inf, np.nan], dtype=dtype)
            for value in (array, array[::2], dtype(2.4)):
                for convert, expected in ((days_to_hours, value * 24.0),
                                           (hours_to_days, value / 24.0)):
                    actual = convert(value)
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        for value in (0, 1, -0.0, 0.1, 2.4):
            self.assertEqual(days_to_hours(value), value * 24.0)
            self.assertEqual(hours_to_days(value), value / 24.0)

    def test_tier1_tier2_duration_fallback_uses_contract(self):
        for producer in (tier1, tier2):
            with self.subTest(producer=producer.__name__):
                with mock.patch.object(producer, "days_to_hours", wraps=days_to_hours) as convert:
                    actual = producer.calc_max_keplerian_duration_hours(1.3, 1000.0, 1.0, 0.1)
                self.assertEqual(actual, 1.3 * 24.0 * 0.25)
                convert.assert_called_once_with(1.3)
                # Wrong unit factor changes the actual production fallback result.
                with mock.patch.object(producer, "days_to_hours", side_effect=lambda x: x * 25.0):
                    self.assertNotEqual(producer.calc_max_keplerian_duration_hours(1.3, 1000.0, 1.0, 0.1), actual)

    def test_antares_entrypoints_route_hour_duration(self):
        class StopBeforeCatalog(BaseException):
            pass
        converted = []
        def capture(value):
            result = hours_to_days(value)
            converted.append(result)
            return result
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(tier4_antares, "hours_to_days", side_effect=capture) as convert, \
             mock.patch.object(tier4_antares, "get_stellar_limb_darkening", side_effect=StopBeforeCatalog):
            with self.assertRaises(StopBeforeCatalog):
                tier4_antares.run_antares_detective("123", 1.3, 2000.1, 2.4)
            convert.assert_called_once_with(2.4)
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(tier3_antares, "hours_to_days", side_effect=capture) as convert, \
             mock.patch.object(tier3_antares.lk, "search_lightcurve", return_value=[]):
            result = tier3_antares.execute_archival_drill(
                {"tic_id": "123", "period_d": 1.3, "t0_btjd": 2000.1, "duration_hours": 2.4})
            convert.assert_called_once_with(2.4)
            self.assertEqual(result["status"], "ARCHIVAL_REJECT")
        self.assertEqual(converted, [2.4 / 24.0, 2.4 / 24.0])

    def test_loyo_model_and_fit_preserve_conversion_order(self):
        t0, period, depth, duration = 2000.1, 1.3, 0.001, 2.4
        t = np.concatenate(([1350.1499999999999], t0 + np.linspace(-0.5, 0.5, 200)))
        phase = ((t - t0) / period + 0.5) % 1.0 - 0.5
        mask = np.abs(phase * period * 24.0) <= duration / 2.0
        expected = np.where(mask, 1.0 - depth, 1.0)
        with mock.patch.object(loyo_validator, "days_to_hours", wraps=days_to_hours) as convert:
            np.testing.assert_array_equal(
                loyo_validator.box_transit_model(t, t0, period, depth, duration), expected)
            np.testing.assert_array_equal(convert.call_args.args[0], phase * period)
        flux = expected + np.linspace(-1e-5, 1e-5, len(t))
        def optimizer(loss, initial, **kwargs):
            expected_loss = np.sum((flux[mask] - (1.0 - depth)) ** 2) + np.sum((flux[~mask] - 1.0) ** 2)
            self.assertEqual(loss(initial), expected_loss)
            return SimpleNamespace(x=initial)
        with mock.patch.object(loyo_validator, "minimize", side_effect=optimizer), \
             mock.patch.object(loyo_validator, "hours_to_days", wraps=hours_to_days) as hours:
            self.assertEqual(loyo_validator.fit_ephemeris(t, flux, period, t0, duration, depth),
                             (t0, period, depth))
            hours.assert_called_once_with(duration)


if __name__ == "__main__":
    unittest.main()
