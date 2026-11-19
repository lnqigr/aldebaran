"""S3.8-CENTROID-01: Filter 4 without usable centroid data (roadmap S3.8, ARGON_DECISIONS items 34 and 35).

Item 34: no median or 0.0 filling of the Filter 4 input; only cadences with finite flux, x and y enter the means, the
shift and the Welch tests; CENTROID_NOT_EVALUATED when f_in < 0.50, N_in < 5, f_out < 0.50, N_out < 20 or zero
variance. The state is carried in the candidate record, and such a peak is CANDIDATE only if the downstream pixel check
evaluated it and passed, otherwise REVIEW. Item 35 rule A: the high-pass runs on the original cadence grid with local
support f_win >= 0.75 (else NaN), unchanged where the whole window is finite. Item 35 rule B: an in-transit cadence needs
local out-of-transit support (M_loc,out > 0, N_loc,out >= 20, N_loc,out / M_loc,out >= 0.50); the fraction
denominators are the full geometric masks. Every rule has a valid control. Offline: every network call is patched.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy import stats
from scipy.signal import medfilt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import aldebaran as ald  # noqa: E402

GOOD_STAR = {"has_params": True, "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05,
             "teff": 5800, "ruwe": 1.05, "tic_query_error": None, "gaia_query_error": None}
TPF_OK = (True, "OK", np.ones((5, 5)), np.ones((5, 5), dtype=bool))
TPF_HOLD = (True, "REVIEW_TPF_EDGE_OR_MARGINAL: border pixel (col=0, row=0) z=5.1, z_diff=3.2", np.ones((5, 5)),
            np.ones((5, 5), dtype=bool))
P_FIX, T0_FIX = 3.141592653589793, 2459000.123456789
N = 19440


def arch_pass():
    return {"n_archival_sectors": 2, "archival_sectors_str": "80,85", "refined_p": P_FIX, "refined_t0": T0_FIX,
            "archival_snr": 8.5, "est_multi_sde_proxy": 11.3, "multi_sde": 11.3,
            "arch_bin_centers": np.linspace(-0.1, 0.1, 10), "arch_bin_means": np.ones(10), "n_trials": 12,
            "snr_crit": 3.2, "is_projected_strict": False, "delta_t_arch_days": 700.0, "sigma_p_sec": 1.25,
            "sigma_t0_sec": 45.0, "archival_sectors_requested": [80, 85], "archival_sectors_attempted": [80, 85],
            "archival_sectors_used": [80, 85], "archival_products_requested": ["a", "b"],
            "archival_products_attempted": ["a", "b"], "archival_products_used": ["a", "b"],
            "archival_products_failed": [], "ephemeris_source": "REFINED_MULTI_SECTOR",
            "ephemeris_status": "REFINED_VALID", "ephemeris_fallback_reason": "NONE"}


def masks(n_in=200, n_out=2000):
    """Out-of-transit half, a 50-cadence buffer, the in-transit block, a buffer, the other half (rule B support)."""
    a = n_out // 2
    in_tr = np.zeros(n_in + n_out + 100, dtype=bool)
    out_tr = np.zeros_like(in_tr)
    out_tr[:a] = True
    in_tr[a + 50:a + 50 + n_in] = True
    out_tr[a + 100 + n_in:] = True
    return in_tr, out_tr


def far_out(in_tr, out_tr):
    """Out-of-transit indices, farthest from the in-transit block first."""
    idx = np.flatnonzero(out_tr)
    centre = np.mean(np.flatnonzero(in_tr))
    return idx[np.argsort(-np.abs(idx - centre), kind="stable")]


def series(in_tr, shift=0.0, seed=1, sigma=0.01):
    rng = np.random.default_rng(seed)
    dx = sigma * rng.standard_normal(in_tr.size)
    dy = sigma * rng.standard_normal(in_tr.size)
    dx[in_tr] += shift
    return dx, dy


def local_layout():
    """Ten in-transit cadences at 1000..1009, a local out-of-transit block 1100..1199 (within 200 of all of them) and
    far out-of-transit cadences 0..699 and 1300..1999 (more than 200 away)."""
    n = 2000
    in_tr = np.zeros(n, dtype=bool)
    out_tr = np.zeros(n, dtype=bool)
    in_tr[1000:1010] = True
    out_tr[1100:1200] = True
    out_tr[:700] = True
    out_tr[1300:] = True
    return in_tr, out_tr


def reference_highpass(v, ok):
    """Independent brute-force item 35 rule A for one axis (pair validity ok)."""
    n = v.size
    k = min(401, (n // 2) * 2 - 1)
    filled = np.where(ok, v, 0.0)
    zero_padded = filled - medfilt(filled, kernel_size=k)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - 200), min(n - 1, i + 200)
        f_win = np.sum(ok[lo:hi + 1]) / (hi - lo + 1)
        if not ok[i] or f_win < 0.75:
            continue
        out[i] = zero_padded[i] if f_win == 1.0 else v[i] - np.median(v[lo:hi + 1][ok[lo:hi + 1]])
    return out


class GateUnitTests(unittest.TestCase):
    def gate(self, dx, dy, in_tr, out_tr, flux=None, time=None):
        flux = np.ones(in_tr.size) if flux is None else flux
        return ald.evaluate_centroid_gate(flux, dx, dy, in_tr, out_tr, time=time)

    def test_valid_control_full_coverage_equals_old_formula(self):
        in_tr, out_tr = masks()
        dx, dy = series(in_tr, shift=0.02)
        g = self.gate(dx, dy, in_tr, out_tr)
        self.assertEqual(g["state"], ald.CENTROID_EVALUATED)
        _, px = stats.ttest_ind(dx[in_tr], dx[out_tr], equal_var=False)
        _, py = stats.ttest_ind(dy[in_tr], dy[out_tr], equal_var=False)
        shift = float(np.hypot(np.mean(dx[in_tr]) - np.mean(dx[out_tr]), np.mean(dy[in_tr]) - np.mean(dy[out_tr])) * 21.0)
        self.assertEqual((g["p_x"], g["p_y"], g["shift_arcsec"]), (float(px), float(py), shift))
        self.assertEqual((g["f_in"], g["n_in"], g["f_out"], g["n_out"]), (1.0, 200, 1.0, 2000))

    def test_all_nan_in_transit_not_evaluated(self):
        in_tr, out_tr = masks()
        dx, dy = series(in_tr)
        dx[in_tr] = np.nan
        g = self.gate(dx, dy, in_tr, out_tr)
        self.assertEqual((g["state"], g["reason"], g["f_in"]), (ald.CENTROID_NOT_EVALUATED, "IN_TRANSIT_COVERAGE", 0.0))
        self.assertTrue(np.isnan(g["shift_arcsec"]))

    def test_in_transit_fraction_boundary(self):
        in_tr, out_tr = masks(n_in=100)
        idx = np.flatnonzero(in_tr)
        for n_nan, want in ((51, ald.CENTROID_NOT_EVALUATED), (50, ald.CENTROID_EVALUATED), (55, ald.CENTROID_NOT_EVALUATED)):
            dx, dy = series(in_tr)
            dy[idx[:n_nan]] = np.nan  # a missing y alone makes the cadence unusable
            self.assertEqual(self.gate(dx, dy, in_tr, out_tr)["state"], want, n_nan)

    def test_in_transit_count_boundary(self):
        for n_in, want in ((4, ald.CENTROID_NOT_EVALUATED), (5, ald.CENTROID_EVALUATED)):
            in_tr, out_tr = masks(n_in=n_in)
            dx, dy = series(in_tr)
            g = self.gate(dx, dy, in_tr, out_tr)
            self.assertEqual(g["state"], want, n_in)
            self.assertEqual(g["f_in"], 1.0)

    def test_out_of_transit_fraction_and_count_boundaries(self):
        in_tr, out_tr = masks(n_out=1000)
        far = far_out(in_tr, out_tr)  # removing the far half leaves every local window intact
        for n_nan, want in ((501, ald.CENTROID_NOT_EVALUATED), (500, ald.CENTROID_EVALUATED), (550, ald.CENTROID_NOT_EVALUATED)):
            dx, dy = series(in_tr)
            dx[far[:n_nan]] = np.nan
            g = self.gate(dx, dy, in_tr, out_tr)
            self.assertEqual(g["state"], want, n_nan)
            if want == ald.CENTROID_NOT_EVALUATED:
                self.assertEqual(g["reason"], "OUT_OF_TRANSIT_COVERAGE")
        for n_out, want in ((19, ald.CENTROID_NOT_EVALUATED), (20, ald.CENTROID_EVALUATED)):
            in_tr, out_tr = masks(n_in=20, n_out=n_out)
            dx, dy = series(in_tr)
            self.assertEqual(self.gate(dx, dy, in_tr, out_tr)["state"], want, n_out)

    def test_zero_variance_not_evaluated(self):
        in_tr, out_tr = masks()
        for axis in (0, 1):
            dx, dy = series(in_tr)
            (dx if axis == 0 else dy)[out_tr] = 0.0  # a product pre-filled with a constant
            g = self.gate(dx, dy, in_tr, out_tr)
            self.assertEqual((g["state"], g["reason"]), (ald.CENTROID_NOT_EVALUATED, "ZERO_VARIANCE"))
        g = self.gate(np.zeros(in_tr.size), np.zeros(in_tr.size), in_tr, out_tr)
        self.assertEqual((g["state"], g["reason"]), (ald.CENTROID_NOT_EVALUATED, "ZERO_VARIANCE"))

    def test_fraction_denominator_is_full_geometric_mask(self):
        in_tr, out_tr = masks(n_in=100)
        idx = np.flatnonzero(in_tr)
        dx, dy = series(in_tr)
        flux = np.ones(in_tr.size)
        flux[idx[:40]] = np.nan  # no flux there: unusable, but still in the denominator M_in (item 35.3)
        dx[idx[40:50]] = np.nan
        g = self.gate(dx, dy, in_tr, out_tr, flux=flux)
        self.assertEqual((g["m_in"], g["n_in"], g["f_in"]), (100, 50, 0.5))
        self.assertEqual(g["state"], ald.CENTROID_EVALUATED)
        dx[idx[50]] = np.nan
        self.assertEqual(self.gate(dx, dy, in_tr, out_tr, flux=flux)["state"], ald.CENTROID_NOT_EVALUATED)

    def test_out_of_transit_denominator_counts_cadences_without_flux(self):
        in_tr, out_tr = masks(n_out=1000)
        far = far_out(in_tr, out_tr)
        for n_nan, want in ((500, ald.CENTROID_EVALUATED), (501, ald.CENTROID_NOT_EVALUATED)):
            dx, dy = series(in_tr)
            flux = np.ones(in_tr.size)
            flux[far[:n_nan]] = np.nan  # no flux: unusable, still in M_out (item 35.3)
            g = self.gate(dx, dy, in_tr, out_tr, flux=flux)
            self.assertEqual((g["m_out"], g["state"]), (1000, want), n_nan)

    def test_finite_masking_keeps_the_shift_that_median_filling_attenuated(self):
        in_tr, out_tr = masks(n_in=200)
        dx, dy = series(in_tr, shift=0.15, sigma=0.005)  # 0.15 px = 3.15 arcsec in transit
        gaps = np.flatnonzero(in_tr)[::5].tolist() + np.flatnonzero(in_tr)[1::5].tolist()  # 40 % gaps
        dx[gaps] = np.nan
        g = self.gate(dx, dy, in_tr, out_tr)
        self.assertEqual(g["state"], ald.CENTROID_EVALUATED)
        self.assertGreater(g["shift_arcsec"], 2.0)
        self.assertLess(g["p_x"], 0.001)
        filled = np.where(np.isfinite(dx), dx, np.nanmedian(dx))  # the removed sector-median filling
        old = float(np.hypot(np.mean(filled[in_tr]) - np.mean(filled[out_tr]), np.mean(dy[in_tr]) - np.mean(dy[out_tr])) * 21.0)
        self.assertLess(old, 2.0)

    def test_missing_series_not_evaluated(self):
        in_tr, out_tr = masks()
        g = self.gate(None, None, in_tr, out_tr)
        self.assertEqual((g["state"], g["reason"]), (ald.CENTROID_NOT_EVALUATED, "NO_CENTROID_SERIES"))

    def test_candidate_allowed_truth_table(self):
        self.assertTrue(ald.centroid_candidate_allowed(ald.CENTROID_EVALUATED, False))
        self.assertTrue(ald.centroid_candidate_allowed(ald.CENTROID_EVALUATED, True))
        self.assertTrue(ald.centroid_candidate_allowed(ald.CENTROID_NOT_EVALUATED, True))
        self.assertFalse(ald.centroid_candidate_allowed(ald.CENTROID_NOT_EVALUATED, False))


class LocalSupportTests(unittest.TestCase):
    """Item 35 rule B on the production gate function; the far out-of-transit cadences keep f_out above 0.50."""

    def gate(self, dx, dy, in_tr, out_tr, time=None):
        return ald.evaluate_centroid_gate(np.ones(in_tr.size), dx, dy, in_tr, out_tr, time=time)

    def test_valid_control_local_support(self):
        in_tr, out_tr = local_layout()
        dx, dy = series(in_tr)
        g = self.gate(dx, dy, in_tr, out_tr)
        self.assertEqual((g["state"], g["n_in"], g["n_in_usable"]), (ald.CENTROID_EVALUATED, 10, 10))

    def test_local_fraction_boundary(self):
        in_tr, out_tr = local_layout()
        loc = np.arange(1100, 1200)
        for n_nan, want_n_in in ((50, 10), (51, 0)):  # N_loc,out / M_loc,out = 50/100 kept, 49/100 dropped
            dx, dy = series(in_tr)
            dx[loc[:n_nan]] = np.nan
            g = self.gate(dx, dy, in_tr, out_tr)
            self.assertEqual((g["n_in"], g["n_in_usable"]), (want_n_in, 10), n_nan)
            self.assertGreaterEqual(g["f_out"], 0.5)
        self.assertEqual((g["state"], g["reason"]), (ald.CENTROID_NOT_EVALUATED, "IN_TRANSIT_COVERAGE"))

    def test_local_count_boundary(self):
        in_tr, out_tr = local_layout()
        out_tr[1130:1200] = False  # 30 local out-of-transit cadences remain
        loc = np.arange(1100, 1130)
        for n_nan, want_n_in in ((10, 10), (11, 0)):  # N_loc,out 20 kept, 19 dropped (fraction above 0.50 both)
            dx, dy = series(in_tr)
            dx[loc[:n_nan]] = np.nan
            self.assertEqual(self.gate(dx, dy, in_tr, out_tr)["n_in"], want_n_in, n_nan)

    def test_no_local_out_of_transit_mask_drops_cadence(self):
        in_tr, out_tr = local_layout()
        out_tr[1100:1200] = False  # M_loc,out = 0
        dx, dy = series(in_tr)
        g = self.gate(dx, dy, in_tr, out_tr)
        self.assertEqual((g["n_in"], g["state"], g["reason"]), (0, ald.CENTROID_NOT_EVALUATED, "IN_TRANSIT_COVERAGE"))

    def test_time_window_excludes_cadences_across_a_gap(self):
        in_tr, out_tr = local_layout()
        dx, dy = series(in_tr)
        t = np.arange(in_tr.size, dtype=float)
        self.assertEqual(self.gate(dx, dy, in_tr, out_tr, time=t)["n_in"], 10)  # valid control: uniform time
        t_gap = t.copy()
        t_gap[1050:] += 1000.0  # the local block is now more than 205 dt_med away in time
        self.assertEqual(self.gate(dx, dy, in_tr, out_tr, time=t_gap)["n_in"], 0)

    def test_denominators_use_geometric_masks(self):
        in_tr, out_tr = local_layout()
        dx, dy = series(in_tr)
        g = self.gate(dx, dy, in_tr, out_tr)
        self.assertEqual((g["m_in"], g["m_out"]), (int(in_tr.sum()), int(out_tr.sum())))


class InputPreparationTests(unittest.TestCase):
    def test_highpass_unchanged_when_all_finite(self):
        rng = np.random.default_rng(3)
        v, w = rng.standard_normal(5000), rng.standard_normal(5000)
        hx, hy = ald._centroid_highpass_pair(v, w)
        np.testing.assert_array_equal(hx, v - medfilt(v, kernel_size=401))
        np.testing.assert_array_equal(hy, w - medfilt(w, kernel_size=401))

    def test_highpass_matches_bruteforce_rule_a_with_gaps(self):
        rng = np.random.default_rng(4)
        v, w = rng.standard_normal(3000), rng.standard_normal(3000)
        v[100:250] = np.nan  # long gap: neighbours fall below f_win 0.75
        v[1000:1400:3] = np.nan  # scattered gaps: f_win between 0.75 and 1.0
        w[2500] = np.nan  # a missing y alone invalidates the pair
        ok = np.isfinite(v) & np.isfinite(w)
        hx, hy = ald._centroid_highpass_pair(v, w)
        np.testing.assert_allclose(hx, reference_highpass(v, ok), rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(hy, reference_highpass(w, ok), rtol=0, atol=0, equal_nan=True)
        self.assertTrue(np.isnan(hx[2500]) and np.isnan(hy[2500]))
        self.assertTrue(np.all(np.isnan(hx[100:250])))
        self.assertTrue(np.any(np.isfinite(hx[1000:1400])))  # partial support kept, not discarded

    def test_no_compaction_keeps_periodic_shift(self):
        # The S3.8-R1-P1-1 mechanism: compacting the finite subsequence made distant in-transit epochs contiguous.
        n, per = 15000, 1250
        i = np.arange(n)
        tr = np.abs(((i - 560 + per // 2) % per) - per // 2) < 60
        rng = np.random.default_rng(9)
        x = 0.001 * rng.standard_normal(n)
        x[tr] += 0.13
        keep = np.ones(n, dtype=bool)
        for k in range(6, 12):
            keep[k * per:(k + 1) * per] = False
            keep[k * per + 500:k * per + 720] = True
        x[~keep] = np.nan
        hx, _ = ald._centroid_highpass_pair(x, 0.001 * rng.standard_normal(n))
        sel = tr & np.isfinite(hx)
        self.assertGreater(np.mean(hx[sel]), 0.12)  # the shift survives; compaction left about 0.09
        self.assertTrue(np.all(np.isnan(hx[tr & (i >= 6 * per)])))  # unsupported epochs are NaN, not attenuated

    def lc(self, mom1):
        n = mom1.size
        t = np.linspace(1800.0, 1827.0, n)
        rng = np.random.default_rng(5)
        return types.SimpleNamespace(meta={"CROWDSAP": 0.99}, time=t, flux=1000.0 + rng.standard_normal(n),
                                     flux_err=np.full(n, 1.0), mom_centr1=mom1, mom_centr2=500.0 + 0.01 * rng.standard_normal(n),
                                     pos_corr1=np.zeros(n), pos_corr2=np.zeros(n), quality=np.zeros(n, dtype=int))

    def test_extract_does_not_fill_missing_centroids(self):
        n = 4000
        rng = np.random.default_rng(6)
        mom1 = 300.0 + 0.01 * rng.standard_normal(n)
        mom1[1500:1700] = np.nan
        lc = self.lc(mom1)
        t, f, ferr, dx, dy, _ = ald.extract_and_stitch_arrays([lc], edge_trim_days=0.0)
        ok = np.isfinite(mom1)
        np.testing.assert_allclose(dx, reference_highpass(mom1, ok), rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(dy, reference_highpass(np.asarray(lc.mom_centr2), ok), rtol=0, atol=0, equal_nan=True)
        self.assertTrue(np.all(np.isnan(dx[1500:1700])) and np.all(np.isnan(dy[1500:1700])))
        self.assertTrue(np.all(np.isfinite(f)))

    def test_extract_valid_control_all_finite(self):
        n = 4000
        rng = np.random.default_rng(7)
        mom1 = 300.0 + 0.01 * rng.standard_normal(n)
        _, _, _, dx, _, _ = ald.extract_and_stitch_arrays([self.lc(mom1)], edge_trim_days=0.0)
        np.testing.assert_array_equal(dx, mom1 - medfilt(mom1, kernel_size=401))


class Item35ControlTests(unittest.TestCase):
    """Item 35.5 controls on the Codex fixture geometry, through extract_and_stitch_arrays and the gate."""
    NN, PER, C = 15000, 1250, 560

    def run_fixture(self, keep, shift_px=0.13):
        n = self.NN
        rng = np.random.default_rng(111)
        i = np.arange(n)
        phase = ((i - self.C + self.PER // 2) % self.PER) - self.PER // 2
        dur = 120 / 0.90
        tr, oot = np.abs(phase) < 0.45 * dur, np.abs(phase) > 0.65 * dur
        x = 300 + rng.normal(0, .001, n)
        y = 500 + rng.normal(0, .001, n)
        x[tr] += shift_px
        x, y = np.where(keep, x, np.nan), np.where(keep, y, np.nan)
        lc = types.SimpleNamespace(meta={"CROWDSAP": .99}, time=i / 720 + 1800, flux=np.ones(n), flux_err=np.ones(n) * .001,
                                   mom_centr1=x, mom_centr2=y, pos_corr1=np.zeros(n), pos_corr2=np.zeros(n),
                                   quality=np.zeros(n, int))
        t, f, _, dx, dy, _ = ald.extract_and_stitch_arrays([lc], edge_trim_days=0)
        g = ald.evaluate_centroid_gate(f, dx, dy, tr, oot, time=t)
        if g["state"] != ald.CENTROID_EVALUATED:
            return g, "NOT_EVALUATED"
        return g, "REJECT" if min(g["p_x"], g["p_y"]) < 0.001 and g["shift_arcsec"] > 2.0 else "PASS"

    def keep_b(self, first_gap_orbit=6):
        keep = np.ones(self.NN, bool)
        for k in range(first_gap_orbit, 12):
            keep[k * self.PER:(k + 1) * self.PER] = False
            keep[k * self.PER + 500:k * self.PER + 720] = True
        return keep

    def keep_d(self):
        return np.random.default_rng(2026).random(self.NN) >= 0.20

    def test_a_full_telemetry_identity(self):
        g, d = self.run_fixture(np.ones(self.NN, bool))
        self.assertEqual(round(g["shift_arcsec"], 9), 2.719585413)
        self.assertEqual(d, "REJECT")

    def test_b_codex_fixture_not_evaluated(self):
        # Item 40 (amends item 35.4): the unchanged Codex 6-of-12 fixture ends CENTROID_NOT_EVALUATED, REVIEW, never PASS.
        g, d = self.run_fixture(self.keep_b())
        self.assertEqual((d, g["reason"]), ("NOT_EVALUATED", "OUT_OF_TRANSIT_COVERAGE"))
        self.assertEqual((g["f_in"], g["n_out"], g["m_out"]), (0.5, 6362, 12924))

    def test_c_seven_of_twelve_not_evaluated(self):
        n = self.NN
        i = np.arange(n)
        tr = np.abs(((i - self.C + self.PER // 2) % self.PER) - self.PER // 2) < 0.45 * 120 / 0.90
        keep = np.ones(n, bool)
        for k in range(5, 12):
            keep[k * self.PER + self.C - 400:k * self.PER + self.C + 400] = False
        keep |= tr
        g, d = self.run_fixture(keep)
        self.assertEqual(d, "NOT_EVALUATED")
        self.assertAlmostEqual(g["f_in"], 5 / 12, places=12)

    def test_d_random_gaps_reject(self):
        g, d = self.run_fixture(self.keep_d())
        self.assertEqual(d, "REJECT")
        self.assertGreater(g["shift_arcsec"], 2.5)

    def test_e_zero_shift_no_false_reject(self):
        self.assertEqual(self.run_fixture(self.keep_d(), 0.0)[1], "PASS")
        g, d = self.run_fixture(self.keep_b(), 0.0)  # pattern b: identical to control b (item 40)
        self.assertEqual((d, g["reason"], g["n_out"]), ("NOT_EVALUATED", "OUT_OF_TRANSIT_COVERAGE", 6362))

    def test_optional_five_of_twelve_pair(self):
        g, d = self.run_fixture(self.keep_b(7))
        self.assertEqual(d, "REJECT")
        self.assertAlmostEqual(g["f_in"], 7 / 12, places=12)
        self.assertEqual(g["n_out"], 7439)
        self.assertAlmostEqual(g["shift_arcsec"], 2.72, delta=0.01)
        self.assertEqual(self.run_fixture(self.keep_b(7), 0.0)[1], "PASS")


class TargetRun:
    """End to end through analyze_single_target (Cp, the only production path), with a temporary OUTPUT_DIR."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s38-out-")
        self.out_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        n = N
        rng = np.random.default_rng(42)
        self.t = np.linspace(2459000, 2459027, n)
        self.f = 1.0 + 1e-4 * rng.standard_normal(n)
        ph = ((self.t - T0_FIX + 0.5 * P_FIX) % P_FIX) - 0.5 * P_FIX
        self.in_window = np.abs(ph) < 0.03
        self.f[np.abs(ph) < 0.04] -= 0.0010
        self.dx = 0.005 * rng.standard_normal(n)
        self.dy = 0.005 * rng.standard_normal(n)

    def run_target(self, dx, dy, tpf=TPF_OK):
        n = N
        patches = [
            patch("lightkurve.search_lightcurve", side_effect=[[MagicMock()]]),
            patch.object(ald, "safe_download", return_value=MagicMock()),
            patch.object(ald, "extract_and_stitch_arrays", return_value=(self.t, self.f, np.full(n, 1e-4), dx, dy, 0.95)),
            patch.object(ald, "clip_upper_outliers", side_effect=lambda a, b, c: (a, b, c)),
            patch.object(ald, "piecewise_savgol_detrend", side_effect=lambda a, b, **kw: np.ones_like(b)),
            patch.object(ald, "compute_whitened_kovacs_sde", return_value=(np.full(9000, 8.5), None)),
            patch.object(ald, "get_top_distinct_bls_peaks", return_value=[0]),
            patch.object(ald, "OUTPUT_DIR", self.out_dir),
            patch.object(ald, "ARCHIVAL_DIAGNOSTICS_FILE", os.path.join(self.out_dir, "archival_vetting_diagnostics.jsonl")),
            patch.object(ald, "get_stellar_and_gaia_params", return_value=dict(GOOD_STAR)),
            patch.object(ald, "_scan_full_orbit_secondary_eclipse_cp", return_value=(False, 0.0, 0.0, 0.5, 3.0)),
            patch.object(ald, "_verify_and_filter_tpf_cp", return_value=tpf),
            patch.object(ald, "verify_multi_epoch", side_effect=[(True, "Confirmed_in_Sectors_80,85", arch_pass())]),
        ]
        for p in patches:
            p.start()
        try:
            with patch.object(ald, "BoxLeastSquares") as bls_cls:
                bls = MagicMock()
                bls_cls.return_value = bls
                bls.power.return_value = MagicMock(period=np.full(9000, P_FIX), power=np.full(9000, 100.0),
                                                   duration=np.full(9000, 0.08), transit_time=np.full(9000, T0_FIX),
                                                   depth=np.full(9000, 0.001))
                _, status, msg, cand = ald.analyze_single_target("12345678", 106)
        finally:
            for p in reversed(patches):
                p.stop()
        return status, msg, cand

    def peak(self):
        with open(os.path.join(self.out_dir, "experimental_v2_vetting.jsonl"), encoding="utf-8") as fh:
            rec = [json.loads(x) for x in fh if x.strip()][-1]
        return rec["peaks"][0]

    def centroid_gate(self):
        return next(g for g in self.peak()["gates"] if g["gate"] == "CENTROID")


class ProductionPathTests(TargetRun, unittest.TestCase):
    def test_valid_control_candidate_evaluated(self):
        status, _, cand = self.run_target(self.dx, self.dy)
        self.assertEqual(status, "CANDIDATE")
        self.assertEqual(cand["centroid_state"], ald.CENTROID_EVALUATED)
        g = self.centroid_gate()
        self.assertEqual(g["status"], "PASS")
        self.assertEqual(g["metrics"]["f_in"], 1.0)
        self.assertTrue(np.isfinite(cand["centroid_shift_arcsec"]))

    def test_all_nan_centroids_candidate_only_through_tpf_pass_and_carried(self):
        nan = np.full(N, np.nan)
        status, _, cand = self.run_target(nan, nan, tpf=TPF_OK)
        self.assertEqual(status, "CANDIDATE")  # the TPF pixel check evaluated the peak and passed (item 34.1)
        self.assertEqual(cand["centroid_state"], ald.CENTROID_NOT_EVALUATED)
        self.assertTrue(np.isnan(cand["centroid_shift_arcsec"]))
        g = self.centroid_gate()
        self.assertEqual((g["status"], g["metrics"]["reason"]), ("NOT_EVALUATED", "IN_TRANSIT_COVERAGE"))
        self.assertEqual(self.peak()["disposition"], "CANDIDATE")

    def test_constant_centroids_not_evaluated_not_incomplete(self):
        # The old path gave CENTROID MISSING_INPUT (zero-variance t-test) and an INCOMPLETE peak under a hold.
        zeros = np.zeros(N)
        status, _, _ = self.run_target(zeros, zeros, tpf=TPF_HOLD)
        self.assertEqual(status, "REVIEW_VETTING_HOLD")
        self.assertEqual(self.centroid_gate()["status"], "NOT_EVALUATED")
        self.assertEqual(self.peak()["disposition"], "REVIEW")

    def test_hold_with_evaluated_centroid_is_review_control(self):
        status, _, _ = self.run_target(self.dx, self.dy, tpf=TPF_HOLD)
        self.assertEqual(status, "REVIEW_VETTING_HOLD")
        self.assertEqual(self.centroid_gate()["status"], "PASS")

    def test_partial_gap_shift_is_rejected(self):
        dx = self.dx.copy()
        idx = np.flatnonzero(self.in_window)
        dx[idx] += 0.15
        dx[np.concatenate([idx[::5], idx[1::5]])] = np.nan
        status, msg, cand = self.run_target(dx, self.dy)
        self.assertEqual(status, "REJECT_CENTROID")
        self.assertIsNone(cand)
        self.assertIn("Centroid offset", msg)

    def test_review_cap_when_no_spatial_check_passed(self):
        # Within the detector the TPF check always precedes this point, so the cap is exercised by forcing
        # centroid_candidate_allowed to report that no spatial check passed; this checks the wiring and the host code.
        nan = np.full(N, np.nan)
        seen = []

        def allowed(state, tpf_pass):
            seen.append((state, tpf_pass))
            return False
        with patch.object(ald, "centroid_candidate_allowed", side_effect=allowed):
            status, _, cand = self.run_target(nan, nan)
        self.assertEqual(seen, [(ald.CENTROID_NOT_EVALUATED, True)])
        self.assertEqual(status, "REVIEW_CENTROID_NOT_EVALUATED")
        self.assertIsNone(cand)
        self.assertEqual((self.peak()["disposition"], self.peak()["cause"]), ("REVIEW", "CENTROID_NOT_EVALUATED_NO_SPATIAL_CHECK"))

    def test_production_gate_receives_the_evaluation_time(self):
        seen = []
        real = ald.evaluate_centroid_gate

        def spy(*args, **kwargs):
            seen.append((len(args[0]), kwargs.get("time")))
            return real(*args, **kwargs)
        with patch.object(ald, "evaluate_centroid_gate", side_effect=spy):
            self.run_target(self.dx, self.dy)
        self.assertEqual(len(seen), 1)
        n_flux, t = seen[0]
        self.assertIsNotNone(t)  # rule B time window uses t_curr_eval, not the cadence index
        self.assertEqual(len(t), n_flux)
        self.assertTrue(np.all(np.isin(t, self.t)) and np.all(np.diff(t) > 0))

    def test_candidate_report_has_centroid_state_column(self):
        _, _, cand = self.run_target(self.dx, self.dy)
        path = os.path.join(self.out_dir, "report.csv")
        ald.save_candidates_to_report(path, [cand])
        import pandas as pd
        self.assertEqual(pd.read_csv(path)["centroid_state"].tolist(), [ald.CENTROID_EVALUATED])


if __name__ == "__main__":
    unittest.main()
