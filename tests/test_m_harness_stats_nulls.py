"""S4-M-PACKAGE-01, parts B1 and B4 (intervals): exact binomial bounds and the N1 / N1-Archival null constructions
(ARGON_DECISIONS items 31.7 and 32). Synthetic fixtures only; the file-level cases read the transformed products
back through lightkurve and the production reader, because the nulls must act before production detrending."""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ALDEBARAN_OFFLINE", "1")
from m_harness import fixtures, nulls, prodpath, stats  # noqa: E402


class ClopperPearsonTests(unittest.TestCase):
    def test_item_31_7_precision_examples(self):
        # ARGON_DECISIONS item 31.7 and the Claude recomputation: 500 stars, 0, 1 and 2 alarms.
        self.assertAlmostEqual(stats.cp_upper(0, 500), 0.00597, places=5)
        self.assertAlmostEqual(stats.cp_upper(1, 500), 0.00945, places=5)
        self.assertAlmostEqual(stats.cp_upper(2, 500), 0.01254, places=5)

    def test_zero_events_closed_form(self):
        self.assertAlmostEqual(stats.cp_upper(0, 500), 1.0 - 0.05 ** (1.0 / 500), places=12)
        self.assertAlmostEqual(stats.cp_lower(500, 500), 0.05 ** (1.0 / 500), places=12)

    def test_one_sided_and_two_sided_are_different_bounds(self):
        one = stats.cp_upper(2, 500)
        lo, two = stats.cp_two_sided(2, 500)
        self.assertGreater(two, one)                       # 2.5 % in the upper tail against 5 %
        self.assertAlmostEqual(two, 0.01436, places=4)
        self.assertEqual(stats.rate(2, 500, "upper")["hi"], one)
        self.assertEqual(stats.rate(2, 500, "upper")["lo"], 0.0)
        self.assertEqual(stats.rate(2, 500, "two-sided")["hi"], two)
        self.assertEqual(stats.rate(2, 500, "lower")["hi"], 1.0)

    def test_edges_and_refusals(self):
        self.assertEqual(stats.cp_upper(5, 5), 1.0)
        self.assertEqual(stats.cp_lower(0, 5), 0.0)
        self.assertEqual(stats.cp_two_sided(0, 5)[0], 0.0)
        self.assertEqual(stats.rate(0, 0, "upper")["status"], "NOT_AVAILABLE_EMPTY_DENOMINATOR")
        self.assertIsNone(stats.rate(0, 0, "upper")["rate"])
        for bad in ((-1, 5), (6, 5), (1, 0), (1.5, 5), (True, 5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stats.cp_upper(*bad)
        with self.assertRaises(ValueError):
            stats.rate(1, 5, "both")

    def test_sample_sizes(self):
        n = stats.n_for_upper_bound(0, 0.006)
        self.assertLessEqual(stats.cp_upper(0, n), 0.006)
        self.assertGreater(stats.cp_upper(0, n - 1), 0.006)
        self.assertEqual(n, 498)
        n2 = stats.n_for_upper_bound(2, 0.015)
        self.assertLessEqual(stats.cp_upper(2, n2), 0.015)
        self.assertGreater(stats.cp_upper(2, n2 - 1), 0.015)
        nh = stats.n_for_halfwidth(0.10)
        self.assertLessEqual(stats.halfwidth_two_sided(nh, 0.5), 0.10)
        self.assertTrue(90 <= nh <= 110)


class OrderStatisticTests(unittest.TestCase):
    def test_median_convention(self):
        self.assertEqual(stats.median_fixed([3, 1, 2]), 2.0)
        self.assertEqual(stats.median_fixed([4, 1, 2, 3]), 2.5)

    def test_type7_p90_for_45_values_interpolates_indices_39_and_40(self):
        x = list(range(45))
        self.assertAlmostEqual(stats.quantile_type7(x, 0.9), 0.4 * 39 + 0.6 * 40, places=12)
        self.assertEqual(stats.quantile_type7([1, 2, 3, 4, 5], 0.5), 3.0)   # g == 0: x[j] alone
        self.assertAlmostEqual(stats.quantile_type7(x, 0.9), float(np.percentile(x, 90)), places=12)

    def test_host_cluster_bootstrap(self):
        # 20 hosts with 5 perfectly correlated cases each: the clustered interval is wider than the binomial one.
        rows = [dict(host=h, hit=(h % 2 == 0)) for h in range(20) for _ in range(5)]
        stat = lambda s: sum(r["hit"] for r in s) / len(s) if s else None
        a = stats.host_cluster_bootstrap(rows, "host", stat, n_boot=800, seed=3)
        b = stats.host_cluster_bootstrap(rows, "host", stat, n_boot=800, seed=3)
        self.assertEqual((a["lo"], a["hi"]), (b["lo"], b["hi"]))           # seeded, reproducible
        lo, hi = stats.cp_two_sided(50, 100)
        self.assertGreater(a["hi"] - a["lo"], hi - lo)
        self.assertEqual((a["n_hosts"], a["n_rows"], a["point"]), (20, 100, 0.5))
        one = stats.host_cluster_bootstrap(rows[:5], "host", stat)
        self.assertEqual(one["status"], "NOT_AVAILABLE_FEWER_THAN_TWO_HOSTS")


def seg_times(lengths, gap=1.0):
    t, start = [], 1000.0
    for n in lengths:
        t.append(start + fixtures.CADENCE_D * np.arange(n))
        start = t[-1][-1] + gap
    return np.concatenate(t)


class N1ArrayTests(unittest.TestCase):
    def test_two_segments_reverse_orbit_one_and_keep_orbit_two(self):
        t = seg_times([6, 4])
        v = np.arange(10.0)
        out, info = nulls.n1_values(t, v)
        self.assertEqual(info["n_segments"], 2)
        np.testing.assert_array_equal(out, [5, 4, 3, 2, 1, 0, 6, 7, 8, 9])
        np.testing.assert_array_equal(nulls.n1_values(t, out)[0], v)       # involution
        self.assertFalse(np.array_equal(out[:6], v[:6]))                   # a reversal that keeps the order fails

    def test_three_segments_swap_pair_and_reverse_the_odd_last(self):
        out, info = nulls.n1_values(seg_times([3, 2, 4]), np.arange(9.0))
        self.assertEqual(info["n_segments"], 3)
        # pair (1, 2): later segment forward, then earlier reversed; odd last segment reversed.
        np.testing.assert_array_equal(out, [3, 4, 2, 1, 0, 8, 7, 6, 5])

    def test_four_and_five_segments(self):
        out4, _ = nulls.n1_values(seg_times([2, 2, 2, 2]), np.arange(8.0))
        np.testing.assert_array_equal(out4, [2, 3, 1, 0, 6, 7, 5, 4])
        out5, info = nulls.n1_values(seg_times([2, 1, 2, 1, 2]), np.arange(8.0))
        self.assertEqual(info["n_segments"], 5)
        np.testing.assert_array_equal(out5, [2, 1, 0, 5, 4, 3, 7, 6])

    def test_single_segment_is_split_at_its_largest_gap(self):
        t = np.array([0.0, 0.01, 0.02, 0.33, 0.34])                        # largest gap 0.31 d, below 0.5 d
        out, info = nulls.n1_values(t, np.arange(5.0))
        self.assertEqual(info["n_segments"], 2)
        np.testing.assert_array_equal(out, [2, 1, 0, 3, 4])

    def test_values_are_a_permutation(self):
        rng = np.random.default_rng(0)
        t = seg_times([40, 55, 13])
        v = rng.normal(size=t.size)
        for sector in (None, 7, 12):
            with self.subTest(sector=sector):
                out, _ = nulls.n1_values(t, v, archival_sector=sector)
                np.testing.assert_array_equal(np.sort(out), np.sort(v))

    def test_archival_fraction_and_roll(self):
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        for s in (1, 12, 13, 55, 80):
            f = nulls.archival_fraction(s)
            self.assertAlmostEqual(f, 0.25 + 0.5 * ((s * phi) % 1.0), places=12)
            self.assertTrue(0.25 <= f <= 0.75)
        self.assertNotAlmostEqual(nulls.archival_fraction(12), nulls.archival_fraction(13), places=3)
        t = seg_times([60, 40])
        v = np.arange(100.0)
        base, _ = nulls.n1_values(t, v)
        arch, _ = nulls.n1_values(t, v, archival_sector=12)
        k = int(round(nulls.archival_fraction(12) * 100))
        np.testing.assert_array_equal(arch, np.roll(base, k))
        self.assertTrue(25 <= k <= 75)
        self.assertFalse(np.array_equal(arch, base))                       # the shift is not optional

    def test_corridor_diagnostic(self):
        d = nulls.shift_outside_corridor(shift_days=6.3, sigma_p_days=1e-4, n_cycles=300, period_days=3.0)
        self.assertAlmostEqual(d["corridor_reach_days"], 3.5 * 1e-4 * 300)
        self.assertTrue(d["outside"])
        self.assertFalse(nulls.shift_outside_corridor(6.0, 1e-4, 300, 3.0)["outside"])   # a multiple of the period
        self.assertIsNone(nulls.shift_outside_corridor(None, 1e-4, 300, 3.0))


class N1FileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mh-nulls-")
        cls.host = fixtures.synthetic_host(cls.tmp + "/hosts", 900101, 40, archival_sectors=(12,), seed=5, sigma=1e-3)
        cls.ald = prodpath.load_production()
        # Flag some cadences and blank one flux so the invalid-row rules are exercised on a real-format file.
        p = os.path.join(cls.tmp, "hosts", cls.host["source_fits"])
        with fits.open(p, mode="update", memmap=False) as h:
            q = np.asarray(h[1].data["QUALITY"]).copy()
            q[100:110] = 8
            h[1].data["QUALITY"] = q
            f = np.asarray(h[1].data["PDCSAP_FLUX"]).copy()
            f[2000] = np.nan
            h[1].data["PDCSAP_FLUX"] = f
            h.flush()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def copy(self, rel, name):
        dst = os.path.join(self.tmp, name)
        shutil.copyfile(os.path.join(self.tmp, "hosts", rel), dst)
        return dst

    def test_discovery_file_transform_and_invariants(self):
        orig = os.path.join(self.tmp, "hosts", self.host["source_fits"])
        new = self.copy(self.host["source_fits"], "n1_lc.fits")
        info, cmap, vcad = nulls.transform_lc(new)
        self.assertEqual(info["n_segments"], 2)
        inv = nulls.invariants(orig, new)
        self.assertTrue(inv["passed"], inv)
        self.assertTrue(inv["involution"] and inv["values_moved"] and inv["invalid_rows_equal"])
        with fits.open(orig) as a, fits.open(new) as b:
            fa, fb = np.asarray(a[1].data["PDCSAP_FLUX"], float), np.asarray(b[1].data["PDCSAP_FLUX"], float)
            valid = nulls.valid_mask(a[1].data)
            n1 = info["segment_lengths"][0]
            np.testing.assert_array_equal(fb[valid][:n1], fa[valid][:n1][::-1])      # orbit 1 reversed
            np.testing.assert_array_equal(fb[valid][n1:], fa[valid][n1:])            # orbit 2 forward
            np.testing.assert_array_equal(np.asarray(b[1].data["MOM_CENTR1"])[valid][:n1],
                                          np.asarray(a[1].data["MOM_CENTR1"])[valid][:n1][::-1])   # columns move together
            self.assertTrue(np.isnan(fb[2000]) and np.all(fb[100:110] == fa[100:110]))             # invalid rows stay
        self.assertEqual(len(vcad), int(valid.sum()))

    def test_production_reader_sees_the_null_before_detrending(self):
        import lightkurve as lk
        orig = os.path.join(self.tmp, "hosts", self.host["source_fits"])
        new = self.copy(self.host["source_fits"], "n1_prod.fits")
        nulls.transform_lc(new)
        t_u, f_u = self.ald.extract_and_stitch_arrays([lk.read(orig)], edge_trim_days=0.10)[:2]
        t_n, f_n = self.ald.extract_and_stitch_arrays([lk.read(new)], edge_trim_days=0.10)[:2]
        np.testing.assert_array_equal(t_u, t_n)                             # same cadences survive extraction
        self.assertFalse(np.array_equal(f_u, f_n))                          # with different flux on them
        # After the production upper clip the same values remain (the clip acts on values, wherever they now sit).
        c_u = prodpath.main_search_arrays(self.ald, lk.read(orig))[1]
        c_n = prodpath.main_search_arrays(self.ald, lk.read(new))[1]
        self.assertEqual(c_u.size, c_n.size)
        self.assertLess(c_u.size, f_u.size)
        self.assertFalse(np.array_equal(c_u, c_n))

    def test_identity_transform_fails_the_invariants(self):
        orig = os.path.join(self.tmp, "hosts", self.host["source_fits"])
        same = self.copy(self.host["source_fits"], "identity.fits")        # "a reversal that keeps the original order"
        inv = nulls.invariants(orig, same)
        self.assertFalse(inv["passed"])
        self.assertFalse(inv["values_placed_by_permutation"])

    def test_archival_file_gets_reversal_plus_shift(self):
        rel = next(r["path"] for r in self.host["lightcurves"] if r["sector"] == 12)
        orig = os.path.join(self.tmp, "hosts", rel)
        new = self.copy(rel, "arch.fits")
        info, _, _ = nulls.transform_lc(new, archival_sector=12)
        self.assertEqual(info["roll_cadences"], int(round(nulls.archival_fraction(12) * info["n_valid"])))
        self.assertGreater(info["shift_days_nominal"], 0.25 * info["baseline_days"] - 1e-9)
        self.assertTrue(nulls.invariants(orig, new, archival_sector=12)["passed"])
        self.assertFalse(nulls.invariants(orig, new)["passed"])            # it is not the plain N1 permutation

    def test_transform_case_updates_hashes_and_permutes_the_pixels(self):
        root = os.path.join(self.tmp, "case")
        shutil.copytree(os.path.join(self.tmp, "hosts"), root)
        case = dict(sector=self.host["sector"], source_fits=self.host["source_fits"], source_sha256=self.host["source_sha256"],
                    tpf_fits=self.host["tpf_fits"], tpf_sha256=self.host["tpf_sha256"],
                    lightcurves=[dict(r) for r in self.host["lightcurves"]])
        before = {r["path"]: fixtures.sha256_file(os.path.join(root, r["path"])) for r in case["lightcurves"]}
        info = nulls.transform_case(case, root, fixtures.sha256_file)
        self.assertTrue(info["ok"], info)
        self.assertEqual(len(info["invariants"]), 3)                        # discovery LC, discovery TPF, one archival LC
        self.assertEqual(len(info["archival"]), 1)
        for r in case["lightcurves"]:
            self.assertEqual(r["sha256"], fixtures.sha256_file(os.path.join(root, r["path"])))
            self.assertNotEqual(r["sha256"], before[r["path"]])
        self.assertNotEqual(case["tpf_sha256"], self.host["tpf_sha256"])
        self.assertFalse([f for _, _, fs in os.walk(root) for f in fs if f.endswith(".orig")])
        with fits.open(os.path.join(self.tmp, "hosts", self.host["tpf_fits"])) as a, fits.open(os.path.join(root, case["tpf_fits"])) as b:
            n1 = info["discovery"]["segment_lengths"][0]
            self.assertFalse(np.array_equal(a[1].data["FLUX"][:50], b[1].data["FLUX"][:50]))
            # A reversal of n1 cadences moves all of them except the middle one when n1 is odd.
            self.assertEqual(info["tpf"]["n_moved"], n1 - n1 % 2)

    def test_target_pixel_file_without_a_valid_cadence_is_not_run(self):
        root = os.path.join(self.tmp, "case_short_tpf")
        shutil.copytree(os.path.join(self.tmp, "hosts"), root)
        tp = os.path.join(root, self.host["tpf_fits"])
        with fits.open(tp, memmap=False) as h:
            h[1].data = h[1].data[:-500]
            h.writeto(tp, overwrite=True)
        case = dict(sector=self.host["sector"], source_fits=self.host["source_fits"], source_sha256="x", tpf_fits=self.host["tpf_fits"],
                    tpf_sha256="x", lightcurves=[dict(r) for r in self.host["lightcurves"]])
        info = nulls.transform_case(case, root, fixtures.sha256_file)
        self.assertFalse(info["ok"])
        self.assertEqual(info["reason"], "LC_VALID_CADENCE_MISSING_IN_TPF")


if __name__ == "__main__":
    unittest.main()
