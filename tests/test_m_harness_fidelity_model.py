"""S4-M-PACKAGE-01, parts B2 and B3 (S4.4): the development fidelity gate (ARGON_DECISIONS item 31.7 with the
item 32 definitions) and the transit model check with the item 31.5 tolerance max(1 s, 1e-3 x T14)."""
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ALDEBARAN_OFFLINE", "1")
from m_harness import fidelity, fixtures, prodpath  # noqa: E402
from m_harness import transit_model as TM  # noqa: E402


def row(star, cdpp=(100.0, 100.0), mad=(1000.0, 1000.0), acf1=(0.002, 0.002), acf10=(0.001, 0.001)):
    return dict(star=star, U=dict(cdpp_1h=cdpp[0], mad=mad[0], acf1=acf1[0], acf10=acf10[0]),
                N1=dict(cdpp_1h=cdpp[1], mad=mad[1], acf1=acf1[1], acf10=acf10[1]))


STARS = [f"s{i:02d}" for i in range(45)]


def cohort(**kw):
    return [row(s, **kw) for s in STARS]


class NoiseMetricTests(unittest.TestCase):
    def test_white_noise_metrics(self):
        rng = np.random.default_rng(1)
        t = 2000.0 + fixtures.CADENCE_D * np.arange(20000)
        x = rng.normal(0.0, 1e-3, t.size)
        m = fidelity.noise_metrics(t, x)
        self.assertAlmostEqual(m["mad"], 1e-3, delta=3e-5)
        self.assertAlmostEqual(m["cdpp_1h"], 1e-3 / math.sqrt(30), delta=1.5e-5)    # 30 cadences per hour
        self.assertLess(abs(m["acf1"]), 0.03)
        self.assertLess(abs(m["acf10"]), 0.03)
        self.assertEqual(m["n_pairs_1"], t.size - 1)

    def test_autocorrelation_is_measured_and_pairs_respect_gaps(self):
        rng = np.random.default_rng(2)
        e = rng.normal(size=20000)
        x = np.empty_like(e)
        x[0] = e[0]
        for i in range(1, e.size):
            x[i] = 0.5 * x[i - 1] + e[i]                                    # AR(1): r_1 = 0.5, r_10 = 0.5 ** 10
        t = 2000.0 + fixtures.CADENCE_D * np.arange(e.size)
        m = fidelity.noise_metrics(t, x)
        self.assertAlmostEqual(m["acf1"], 0.5, delta=0.03)
        self.assertAlmostEqual(m["acf10"], 0.5 ** 10, delta=0.03)
        t2 = t.copy()
        t2[10000:] += 1.0                                                   # a one-day gap: pairs across it are dropped
        self.assertEqual(fidelity.noise_metrics(t2, x)["n_pairs_10"], t.size - 10 - 10)

    def test_arrays_come_from_the_production_functions(self):
        ald = prodpath.load_production()
        rng = np.random.default_rng(3)
        t = 2000.0 + fixtures.CADENCE_D * np.arange(3000)
        lc = prodpath.ArrayLightCurve(t, 1.0e4 * (1 + rng.normal(0, 1e-3, t.size)), np.zeros(t.size, int))
        with patch.object(ald, "extract_and_stitch_arrays", wraps=ald.extract_and_stitch_arrays) as ex, \
                patch.object(ald, "clip_upper_outliers", wraps=ald.clip_upper_outliers) as clip, \
                patch.object(ald, "piecewise_savgol_detrend", wraps=ald.piecewise_savgol_detrend) as det:
            tt, x = fidelity.production_arrays(ald, lc)
        self.assertEqual((ex.call_count, clip.call_count, det.call_count), (1, 1, 1))
        self.assertEqual(det.call_args.kwargs, dict(mask=None, max_win=601))
        self.assertEqual(ex.call_args.kwargs, dict(edge_trim_days=0.10))
        self.assertLess(abs(float(np.median(x))), 1e-3)
        self.assertLess(tt.size, t.size)                                    # edge trim and upper clip applied


class NoiseGateTests(unittest.TestCase):
    def test_identical_arms_pass(self):
        g = fidelity.noise_gate(cohort(), STARS)
        self.assertEqual(g["status"], "PASS")
        self.assertEqual({k: v["status"] for k, v in g["criteria"].items()}, dict(C1="PASS", C2="PASS", C3="PASS"))
        self.assertEqual(g["criteria"]["C1"]["n"], 45)

    def test_c1_needs_both_scatter_measures(self):
        self.assertEqual(fidelity.noise_gate(cohort(cdpp=(100.0, 104.9)), STARS)["status"], "PASS")
        g = fidelity.noise_gate(cohort(cdpp=(100.0, 105.2)), STARS)
        self.assertEqual((g["status"], g["criteria"]["C1"]["status"]), ("FAIL", "FAIL"))
        g = fidelity.noise_gate(cohort(mad=(1000.0, 940.0)), STARS)                                      # MAD alone fails it
        self.assertEqual(g["criteria"]["C1"]["status"], "FAIL")
        self.assertTrue(g["criteria"]["C1"]["cdpp_1h"]["passed"])

    def test_autocorrelation_change_fails_with_unchanged_scatter(self):
        # The control for "a fidelity gate that ignores autocorrelation": scatter identical, lag-1 shifted.
        g = fidelity.noise_gate(cohort(acf1=(0.002, 0.010)), STARS)
        self.assertEqual(g["criteria"]["C1"]["status"], "PASS")
        self.assertEqual(g["criteria"]["C2"]["status"], "FAIL")
        self.assertEqual(g["status"], "FAIL")
        g = fidelity.noise_gate(cohort(acf10=(0.001, -0.02)), STARS)
        self.assertEqual((g["criteria"]["C2"]["status"], g["criteria"]["C3"]["status"]), ("PASS", "FAIL"))

    def test_median_rule_is_absolute_near_zero_and_relative_for_large_values(self):
        ok = fidelity.noise_gate(cohort(acf1=(0.002, 0.007)), STARS)        # |diff| = 0.005 = max(0.005, 0.0002)
        self.assertTrue(ok["criteria"]["C2"]["median"]["passed"])
        big = fidelity.noise_gate(cohort(acf1=(0.30, 0.329)), STARS)        # limit 0.03
        self.assertTrue(big["criteria"]["C2"]["median"]["passed"])
        self.assertAlmostEqual(big["criteria"]["C2"]["median"]["limit"], 0.03)
        self.assertFalse(fidelity.noise_gate(cohort(acf1=(0.30, 0.331)), STARS)["criteria"]["C2"]["median"]["passed"])

    def test_p90_and_max_conditions_bite_separately(self):
        rows = cohort()
        for r in rows[:5]:                                                  # 5 of 45 stars differ by 0.016
            r["N1"]["acf1"] = r["U"]["acf1"] + 0.016
        c2 = fidelity.noise_gate(rows, STARS)["criteria"]["C2"]
        # Type 7 at n = 45: 0.4 x[39] + 0.6 x[40] = 0.6 x 0.016 = 0.0096, still inside 0.015.
        self.assertAlmostEqual(c2["p90"]["value"], 0.0096, places=12)
        self.assertTrue(c2["p90"]["passed"])
        rows[5]["N1"]["acf1"] = rows[5]["U"]["acf1"] + 0.016                # a sixth star: x[39] = x[40] = 0.016
        g = fidelity.noise_gate(rows, STARS)
        c2 = g["criteria"]["C2"]
        self.assertEqual(g["status"], "FAIL")
        self.assertTrue(c2["median"]["passed"])
        self.assertFalse(c2["p90"]["passed"])
        self.assertTrue(c2["max"]["passed"])
        rows = cohort()
        rows[7]["N1"]["acf10"] = rows[7]["U"]["acf10"] + 0.21               # one star: only the maximum fails
        c3 = fidelity.noise_gate(rows, STARS)["criteria"]["C3"]
        self.assertEqual((c3["median"]["passed"], c3["p90"]["passed"], c3["max"]["passed"]), (True, True, False))
        rows[7]["U"]["acf10"], rows[7]["N1"]["acf10"] = 0.0, 0.20           # exactly at the limit: equality passes
        self.assertTrue(fidelity.noise_gate(rows, STARS)["criteria"]["C3"]["max"]["passed"])

    def test_partial_or_invalid_cohort_is_a_failed_evaluation(self):
        self.assertEqual(fidelity.noise_gate(cohort()[:-1], STARS)["status"], "FAILED_EVALUATION")
        self.assertEqual(fidelity.noise_gate(cohort() + [row("s00")], STARS)["status"], "FAILED_EVALUATION")
        self.assertEqual(fidelity.noise_gate(cohort() + [row("other")], STARS)["status"], "FAILED_EVALUATION")
        for bad in (float("nan"), None, True, "0.1", float("inf")):
            rows = cohort()
            rows[3]["N1"]["acf1"] = bad
            with self.subTest(bad=bad):
                g = fidelity.noise_gate(rows, STARS)
                self.assertEqual(g["status"], "FAILED_EVALUATION")
                self.assertTrue(g["problems"])
        self.assertEqual(fidelity.noise_gate(cohort(cdpp=(0.0, 1.0)), STARS)["status"], "FAILED_EVALUATION")
        self.assertEqual(fidelity.noise_gate([], [])["status"], "FAILED_EVALUATION")


class RecoveryGateTests(unittest.TestCase):
    def rows(self, n, n_short, recovered=(), errors=()):
        out = []
        for i in range(n):
            out.append(dict(planet=f"p{i}", period_days=1.5 if i < n_short else 5.0, recovered=i in recovered, error=i in errors))
        return out, [f"p{i}" for i in range(n)]

    def test_limits_of_item_31_7(self):
        self.assertEqual((fidelity.recovery_limit(50), fidelity.recovery_limit(22), fidelity.recovery_limit(500)), (2, 1, 25))
        rows, exp = self.rows(50, 22, recovered=(30, 31))                   # 2 of 50, none short
        g = fidelity.recovery_gate(rows, exp)
        self.assertEqual((g["status"], g["criteria"]["C4"]["recovered"], g["criteria"]["C5"]["n"]), ("PASS", 2, 22))
        rows, exp = self.rows(50, 22, recovered=(30, 31, 32))               # 3 of 50
        self.assertEqual(fidelity.recovery_gate(rows, exp)["criteria"]["C4"]["status"], "FAIL")
        rows, exp = self.rows(50, 22, recovered=(0, 1))                     # 2 of 22 short: C5 fails, C4 passes
        g = fidelity.recovery_gate(rows, exp)
        self.assertEqual((g["criteria"]["C4"]["status"], g["criteria"]["C5"]["status"], g["status"]), ("PASS", "FAIL", "FAIL"))

    def test_error_counts_as_recovered_and_missing_planet_fails_evaluation(self):
        rows, exp = self.rows(50, 22, recovered=(30, 31), errors=(40,))
        g = fidelity.recovery_gate(rows, exp)
        self.assertEqual((g["criteria"]["C4"]["recovered"], g["criteria"]["C4"]["status"]), (3, "FAIL"))
        rows, exp = self.rows(50, 22)
        self.assertEqual(fidelity.recovery_gate(rows[:-1], exp)["status"], "FAILED_EVALUATION")
        rows, exp = self.rows(10, 0)                                        # no short-period planet: C5 cannot be evaluated
        self.assertEqual(fidelity.recovery_gate(rows, exp)["status"], "FAILED_EVALUATION")

    def test_recovered_after_n1_is_period_only_and_candidate_only(self):
        rec = dict(status="REVIEW_VETTING_HOLD", peaks=[dict(disposition="CANDIDATE", period_days=3.05),
                                                         dict(disposition="REJECT", period_days=3.0)])
        self.assertEqual(fidelity.recovered_after_n1(rec, 3.0), (True, False))          # 1.7 % off: counted
        self.assertEqual(fidelity.recovered_after_n1(rec, 2.9), (False, False))         # 5 % off
        rec["peaks"][0]["disposition"] = "REVIEW"
        self.assertEqual(fidelity.recovered_after_n1(rec, 3.0), (False, False))
        self.assertEqual(fidelity.recovered_after_n1(dict(status="ERROR", peaks=[]), 3.0), (True, True))

    def test_whole_gate_scope_and_blocking(self):
        rows, exp = self.rows(50, 22)
        full = fidelity.fidelity_gate(cohort(), STARS, rows, exp)
        self.assertEqual((full["status"], full["scope"], full["blocks_m4"]), ("PASS", "FULL_DEVELOPMENT_GATE", False))
        pre = fidelity.fidelity_gate(cohort(), STARS)
        self.assertEqual((pre["status"], pre["scope"]), ("PASS", "NOISE_ONLY_PRE_GATE"))
        bad = fidelity.fidelity_gate(cohort(acf1=(0.002, 0.05)), STARS, rows, exp)
        self.assertEqual((bad["status"], bad["blocks_m4"]), ("FAIL", True))
        self.assertTrue(fidelity.fidelity_gate(cohort()[:-1], STARS)["blocks_m4"])


class ReferenceModelTests(unittest.TestCase):
    def test_uniform_disk_closed_forms(self):
        np.testing.assert_allclose(TM.occulted_fraction([0.0, 0.3, 0.89], 0.1, 0.0, 0.0), 0.01, rtol=1e-11)

        def lens(z, k):
            a = k * k * math.acos((z * z + k * k - 1) / (2 * z * k)) + math.acos((z * z + 1 - k * k) / (2 * z)) \
                - 0.5 * math.sqrt((-z + k + 1) * (z + k - 1) * (z - k + 1) * (z + k + 1))
            return a / math.pi
        for z in (0.95, 1.0, 1.05):
            self.assertAlmostEqual(float(TM.occulted_fraction([z], 0.1, 0.0, 0.0)[0]) / lens(z, 0.1), 1.0, places=9)
        self.assertEqual(float(TM.occulted_fraction([1.2], 0.1, 0.4, 0.2)[0]), 0.0)

    def test_limb_darkened_disk_against_two_dimensional_quadrature(self):
        u1, u2 = 0.4, 0.2

        def brute(z, k):
            def inten(r):
                mu = math.sqrt(max(0.0, 1 - r * r))
                return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
            f = lambda y, x: inten(math.hypot(x + z, y)) if (x + z) ** 2 + y * y < 1 else 0.0
            v, _ = integrate.dblquad(f, -k, k, lambda x: -math.sqrt(k * k - x * x), lambda x: math.sqrt(k * k - x * x),
                                     epsabs=1e-10, epsrel=1e-10)
            return v / (math.pi * (1 - u1 / 3 - u2 / 6))
        for z, k in ((0.0, 0.1), (0.5, 0.1), (0.2, 0.3)):
            with self.subTest(z=z, k=k):
                self.assertAlmostEqual(float(TM.occulted_fraction([z], k, u1, u2)[0]) / brute(z, k), 1.0, places=8)

    def test_exposure_integration_and_out_of_transit(self):
        ars = TM.a_over_rs(3.0, 1.0, 1.0)
        inc = TM.inclination(0.3, ars)
        t = np.array([0.0, 0.5, 1.5])                                       # mid-transit, out of transit, behind the star
        f = TM.transit_flux(t, 0.1, (0.4, 0.2), 0.0, 3.0, ars, inc, "reference")
        self.assertLess(f[0], 0.99)
        self.assertEqual((f[1], f[2]), (1.0, 1.0))
        self.assertEqual(TM.transit_flux(np.array([np.nan]), 0.1, (0.4, 0.2), 0.0, 3.0, ars, inc, "reference")[0], 1.0)
        with self.assertRaises(ValueError):
            TM.transit_flux(t, 0.1, (0.4, 0.2), 0.0, 3.0, ars, inc, "batman")


class ModelCheckTests(unittest.TestCase):
    def truth(self, p=3.0, k=0.05, b=0.3, star=(1.0, 1.0)):
        ars = TM.a_over_rs(p, *star)
        d = TM.t14(p, k, b, ars)
        tr = dict(period=p, k=k, b=b, a_rs=ars, inc=TM.inclination(b, ars), t0=2000.3, t14=d, ld=[0.4, 0.2])
        tr["depth"] = TM.mean_deficit(k, tr["ld"], p, b, ars, "reference")[0]
        return tr

    def test_item_31_5_tolerance(self):
        self.assertAlmostEqual(TM.t14_tolerance_days(0.01) * 86400.0, 1.0, places=9)       # short transit: the 1 s floor
        self.assertAlmostEqual(TM.t14_tolerance_days(0.2) * 86400.0, 17.28, places=9)      # long transit: 1e-3 x T14
        self.assertAlmostEqual(TM.t14_tolerance_days(1000.0 / 86400.0) * 86400.0, 1.0, places=9)   # equal at T14 = 1000 s

    def test_reference_backend_passes_over_a_small_geometry_grid(self):
        grid = TM.geometry_grid(periods=(1.0, 7.0), impact=(0.02, 0.79), ratios=(0.02, 0.30), stars=((0.2, 0.22), (1.2, 1.75)))
        s = TM.model_check_grid("reference", grid)
        self.assertEqual((s["n"], s["n_failed"]), (len(grid), 0))
        self.assertTrue(s["passed"])
        self.assertLess(s["max_t14_diff_s"], 1e-3)
        self.assertLess(s["min_a_rs"], 3.0)                                 # the small a / R* corner is covered

    def test_wrong_duration_is_caught_with_the_tolerance_scaled_by_t14(self):
        real = TM.transit_flux

        def stretched(stretch_s):
            def flux(times, k, ld, t0, p, ars, inc, backend, **kw):
                # A model whose transit is longer than the geometry by stretch_s seconds.
                d = TM.t14(p, k, math.cos(inc) * ars, ars)
                scale = d / (d + stretch_s / 86400.0)
                return real(t0 + (np.asarray(times, float) - t0) * scale, k, ld, t0, p, ars, inc, backend, **kw)
            return flux
        short = self.truth(p=1.0, k=0.02, b=0.6, star=(0.2, 0.22))          # T14 about 0.6 h: tolerance 2.2 s
        long_ = self.truth(p=11.0, k=0.05, b=0.1, star=(1.2, 1.75))         # T14 about 7.6 h: tolerance about 27 s
        self.assertLess(short["t14"] * 86400.0 * 1e-3, 3.0)
        self.assertGreater(long_["t14"] * 86400.0 * 1e-3, 20.0)
        with patch.object(TM, "transit_flux", stretched(5.0)):
            self.assertFalse(TM.model_check(short, "reference")["t14_passed"])           # 5 s is too much for a short transit
            r = TM.model_check(long_, "reference")
            self.assertTrue(r["t14_passed"])                                             # and inside 1e-3 x T14 for a long one
            self.assertAlmostEqual(r["t14_contacts_minus_formula_s"], 5.0, delta=0.05)
        with patch.object(TM, "transit_flux", stretched(40.0)):
            self.assertFalse(TM.model_check(long_, "reference")["t14_passed"])
        ok = TM.model_check(short, "reference")
        self.assertTrue(ok["passed"])

    def test_wrong_depth_is_caught(self):
        tr = self.truth()
        tr["depth"] *= 1.001
        r = TM.model_check(tr, "reference")
        self.assertFalse(r["depth_passed"])
        self.assertFalse(r["passed"])

    @unittest.skipUnless(TM.pytransit_available(), "pytransit needs numpy below 2 (injection environment only)")
    def test_pytransit_injector_against_the_reference(self):
        grid = TM.geometry_grid(periods=(1.0, 7.0), impact=(0.02, 0.79), ratios=(0.02, 0.30), stars=((0.2, 0.22), (1.2, 1.75)))
        s = TM.model_check_grid("pytransit", grid)
        self.assertEqual(s["n_failed"], 0)
        tr = self.truth()
        times = tr["t0"] + np.arange(-0.6 * tr["t14"], 0.6 * tr["t14"], TM.CAD_D)
        self.assertLess(TM.compare_backends(tr, times)["max_abs_diff_over_depth"], 1e-3)


if __name__ == "__main__":
    unittest.main()
