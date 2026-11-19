"""S4-M-PACKAGE-01, parts B3 and C: the M1 injection driver and the product routing controls of roadmap 2.I (S4.3).

The routing cases run the production entry point aldebaran.analyze_single_target on synthetic SPOC-format files
through the router. They show that the injected signal reaches the included stages, that an uninjected file cannot
replace it unnoticed, that truth is not exposed to detection, and which stages stay outside the boundary."""
import copy
import dataclasses
import math
import os
import shutil
import socket
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ALDEBARAN_OFFLINE", "1")
from m_harness import fixtures, injection, matching, prodpath, routing  # noqa: E402
from m_harness import transit_model as TM  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DESIGN = dataclasses.replace(injection.DRAFT_DESIGN, backend="reference")


def fixed_truth(ctx, ld, period=2.7, b=0.3, snr=30.0, phase=0.4):
    """A planet at a chosen period through the production joint solve (no random draw)."""
    ars = TM.a_over_rs(period, ctx["m_star"], ctx["r_star"])
    t0 = float(np.nanmin(ctx["main_t"])) + phase * period
    sol = injection.joint_solve(snr, b, period, t0, ld, ars, ctx["snr_ctx"], "reference")
    assert sol["valid"], sol
    return dict(b=b, period=period, t0=t0, a_rs=ars, inc=TM.inclination(b, ars), ld=list(ld), backend="reference", **sol)


class Shared(unittest.TestCase):
    """One synthetic host, one strong injected planet, and one detector run shared by the cases below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mh-route-")
        cls.ald = prodpath.load_production()
        cls.hosts = os.path.join(cls.tmp, "hosts")
        cls.inj = os.path.join(cls.tmp, "inj")
        cls.host = fixtures.synthetic_host(cls.hosts, 900201, 40, archival_sectors=(12, 13), seed=11, sigma=1.5e-3)
        cls.ctx = injection.host_context(cls.ald, cls.host, cls.hosts)
        cls.truth = fixed_truth(cls.ctx, cls.host["ld"])
        cls.case, cls.tinfo = injection.write_injection(cls.host, cls.hosts, "case_a", cls.truth, cls.inj, fixtures.sha256_file)
        cls.originals = [r["sha256"] for r in cls.host["lightcurves"]] + [cls.host["tpf_sha256"]]
        cls.record = routing.run_case(cls.ald, cls.case, cls.inj, os.path.join(cls.tmp, "out", "case_a"), "route_test")
        cls.times = np.load(os.path.join(cls.tmp, "out", "case_a", cls.record["search"]["times_file"]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def out(self, name):
        return os.path.join(self.tmp, "out", name)


class SnrAndSolveTests(Shared):
    def test_bins_of_item_31_8(self):
        b = injection.M1_LOW_SNR_BINS
        self.assertEqual(b, ((4.0, 6.0), (6.0, 8.0), (8.0, 10.0)))
        self.assertEqual([injection.snr_bin(x, b) for x in (3.99, 4.0, 5.999, 6.0, 8.0, 10.0, 10.01)], [None, 0, 0, 1, 2, 2, None])
        self.assertEqual([injection.bin_label(lo, hi) for lo, hi in b], ["[4, 6)", "[6, 8)", "[8, 10]"])
        full = injection.DRAFT_DESIGN.snr_bins
        self.assertEqual(injection.snr_bin(10.0, full), 2)                  # 10 belongs to [8, 10], not to (10, 15]
        self.assertEqual(injection.bin_label(*full[3]), "(10, 15)")
        self.assertIsNone(injection.snr_bin(float("nan"), b))
        self.assertEqual(injection.GREY_SUPPORT_SNR, (5.0, 9.0))
        self.assertEqual(injection.DRAFT_DESIGN.status, "DRAFT")

    def test_snr_definition_on_the_production_arrays(self):
        t, flat = self.ctx["snr_ctx"]
        p, t0, d = 2.7, float(t[0]) + 1.0, 0.1
        r = injection.snr_from_context(self.ctx["snr_ctx"], p, t0, d, 1000.0)
        self.assertEqual(r["status"], "OK")
        # Independent recomputation of the definition with other array operations.
        mids = t0 + p * np.arange(-3, 14)
        mids = mids[(mids >= t[0]) & (mids <= t[-1])]
        n_ev = sum(1 for m in mids if np.sum(np.abs(t - m) <= d / 2) >= 0.5 * d / fixtures.CADENCE_D)
        oot = np.array([np.min(np.abs(x - mids)) > d for x in t])
        to, fo = t[oot], flat[oot]
        idx = np.searchsorted(np.arange(to.min(), to.max() + d, d), to, side="right")
        cnt = np.bincount(idx)
        means = (np.bincount(idx, weights=fo)[cnt > 0] / cnt[cnt > 0])[cnt[cnt > 0] >= 0.5 * d / fixtures.CADENCE_D]
        sigma = 1.4826 * np.median(np.abs(means - np.median(means)))
        self.assertEqual(r["N_ev"], n_ev)
        self.assertEqual(r["n_bins"], means.size)
        self.assertAlmostEqual(r["sigma_T14_ppm"] / (sigma * 1e6), 1.0, places=9)
        self.assertAlmostEqual(r["SNR"], 1000.0 / (sigma * 1e6) * math.sqrt(n_ev), places=6)
        # White noise of 1.5e-3 per cadence, 72 cadences per bin; the production detrending removes part of it.
        white = 1.5e-3 / math.sqrt(72) * 1e6
        self.assertTrue(0.7 * white < r["sigma_T14_ppm"] < 1.05 * white, r["sigma_T14_ppm"])
        self.assertEqual(injection.snr_from_context(None, p, t0, d, 1000.0)["status"], "INSUFFICIENT_CADENCES")
        short = injection.snr_inj(self.ald, t[:300], np.ones(300), np.zeros(300, int), p, t0, d, 1000.0)
        self.assertEqual(short["status"], "INSUFFICIENT_CADENCES")          # the production reader keeps too few cadences

    def test_joint_solve_reaches_the_target_within_five_percent(self):
        tr = self.truth
        self.assertEqual(tr["stop_reason"], "CONVERGED")
        self.assertLessEqual(abs(tr["snr_achieved"] / 30.0 - 1.0), 0.05)
        again = injection.snr_from_context(self.ctx["snr_ctx"], tr["period"], tr["t0"], tr["t14"], tr["depth"] * 1e6)
        self.assertAlmostEqual(again["SNR"], tr["snr_achieved"], places=6)  # the SNR of the solved depth, recomputed
        self.assertEqual(set(injection.SOLVE_FIELDS) - set(tr), set())
        self.assertTrue(TM.model_check(tr, "reference")["passed"])

    def test_stop_rule_iterates_until_five_percent(self):
        # A near-grazing transit: the starting guess k = sqrt(depth) is far from the target, so the item 25 stop rule
        # (|achieved / target - 1| <= 0.05) must send the solve through at least one more iterate.
        p, b = 4.3, 0.79
        ars = TM.a_over_rs(p, self.ctx["m_star"], self.ctx["r_star"])
        t0 = float(np.nanmin(self.ctx["main_t"])) + 0.3 * p
        sol = injection.joint_solve(20.0, b, p, t0, self.host["ld"], ars, self.ctx["snr_ctx"], "reference")
        self.assertEqual((sol["valid"], sol["stop_reason"]), (True, "CONVERGED"))
        self.assertGreaterEqual(sol["n_iter"], 1)
        self.assertLessEqual(abs(sol["snr_residual"]), 0.05)
        self.assertEqual(injection.SNR_TOL, 0.05)

    def test_model_error_is_recorded_not_raised(self):
        from unittest.mock import patch
        with patch.object(TM, "mean_deficit", side_effect=ZeroDivisionError("z = 1 - k")):
            sol = injection.joint_solve(30.0, 0.3, 2.7, self.truth["t0"], self.host["ld"], self.truth["a_rs"],
                                        self.ctx["snr_ctx"], "reference")
        self.assertEqual((sol["valid"], sol["stop_reason"]), (False, "MODEL_ERROR"))

    def test_draws_fall_in_their_bin_and_in_the_declared_domain(self):
        design = dataclasses.replace(DESIGN, snr_bins=((8.0, 10.0), (15.0, 25.0)))
        slots = injection.draw_m1_host(self.ald, self.host, self.hosts, design, 3)
        self.assertEqual([s["snr_bin"] for s in slots], ["[8, 10]", "[15, 25)"])
        again = injection.draw_m1_host(self.ald, self.host, self.hosts, design, 3)
        for s, s2, (lo, hi) in zip(slots, again, design.snr_bins):
            tr = s["truth"]
            self.assertEqual(s["status"], "DRAWN")
            self.assertEqual(tr, s2["truth"])                               # seeded: the same draw again
            self.assertEqual(s["seed"], injection.slot_seed(design, 3, s["snr_bin_index"], 0))
            self.assertTrue(lo <= tr["snr_target"] < hi)
            self.assertLessEqual(abs(tr["snr_achieved"] / tr["snr_target"] - 1.0), 0.05)
            self.assertTrue(180.0 <= tr["depth"] * 1e6 <= 30000.0)
            self.assertAlmostEqual(tr["rp_earth"], tr["k"] * 1.0 * 109.076, places=9)
            self.assertLessEqual(tr["rp_earth"], 15.0)
            self.assertTrue(1.0 <= tr["period"] <= 13.0 and 0.0 <= tr["b"] < 0.8)
            self.assertGreaterEqual(tr["n_discovery_events"], 3)
        other = injection.draw_m1_host(self.ald, self.host, self.hosts, design, 4)
        self.assertNotEqual(other[0]["truth"]["period"], slots[0]["truth"]["period"])     # another host index, another seed

    def test_failed_draw_is_reported(self):
        design = dataclasses.replace(DESIGN, snr_bins=((4.0, 6.0),), depth_domain_ppm=(1.0e5, 2.0e5), max_draws=2, max_t0_tries=2)
        slot = injection.draw_m1_host(self.ald, self.host, self.hosts, design, 0)[0]
        self.assertEqual((slot["status"], slot["truth"]), ("DRAW_FAILED", None))
        self.assertEqual(len(slot["attempts"]), 2)
        outcomes = {t["try_outcome"] for a in slot["attempts"] for t in a["tries"]}
        self.assertTrue(outcomes <= {"DEPTH_OUTSIDE_DOMAIN", "SOLVE_FAILED"}, outcomes)
        # The radius bound of the declared domain (Rp <= 15 R_E) is enforced the same way.
        small = dataclasses.replace(DESIGN, snr_bins=((25.0, 35.0),), rp_max_earth=0.5, max_draws=2, max_t0_tries=2)
        slot = injection.draw_m1_host(self.ald, self.host, self.hosts, small, 0)[0]
        self.assertEqual(slot["status"], "DRAW_FAILED")
        self.assertIn("RADIUS_OUTSIDE_DOMAIN", {t["try_outcome"] for a in slot["attempts"] for t in a["tries"]})
        self.assertEqual((injection.DRAFT_DESIGN.rp_max_earth, injection.DRAFT_DESIGN.depth_domain_ppm), (15.0, (180.0, 30000.0)))

    def test_limb_darkening_interpolation(self):
        grid = dict(teff=[5000.0, 6000.0], logg=[4.0, 5.0], a=[[0.4, 0.5], [0.2, 0.3]], b=[[0.1, 0.1], [0.3, 0.3]])
        a, b, clamped = injection.ld_coefficients(5500.0, 4.5, grid)
        self.assertEqual((round(a, 12), round(b, 12), clamped), (0.35, 0.2, False))
        self.assertTrue(injection.ld_coefficients(7000.0, 4.5, grid)[2])


class InjectionFileTests(Shared):
    def test_exact_identities_of_the_injected_copies(self):
        fid = injection.fidelity(self.host, self.hosts, self.case, self.inj, self.tinfo, self.truth, fixtures.sha256_file)
        self.assertTrue(fid["passed"], fid)
        self.assertLessEqual(fid["lc_max_abs_ratio_minus_m"], 1e-9)
        self.assertTrue(fid["originals_unchanged"] and fid["outside_pixels_unchanged"])
        self.assertGreater(fid["aperture_sum_n_resolvable"], 100)

    def test_mutated_copies_fail_the_identities(self):
        bad_w = dict(self.tinfo, w=np.where(self.tinfo["w"] > 0, self.tinfo["w"] * 1.001, 0.0))
        self.assertFalse(injection.fidelity(self.host, self.hosts, self.case, self.inj, bad_w, self.truth, fixtures.sha256_file)["passed"])
        case_m, tinfo_m = injection.write_injection(self.host, self.hosts, "mutant_lc", self.truth, self.inj, fixtures.sha256_file)
        with fits.open(os.path.join(self.inj, case_m["source_fits"]), mode="update", memmap=False) as h:
            k = int(np.nonzero(np.isfinite(h[1].data["PDCSAP_FLUX"]))[0][100])
            h[1].data["PDCSAP_FLUX"][k] *= (1 + 1e-6)
            h.flush()
        self.assertFalse(injection.fidelity(self.host, self.hosts, case_m, self.inj, tinfo_m, self.truth, fixtures.sha256_file)["passed"])

    def test_case_carries_no_truth(self):
        self.assertEqual(set(self.case), set(routing.CASE_KEYS))
        text = repr(self.case)
        for value in (self.truth["period"], self.truth["t0"], self.truth["depth"], self.truth["k"]):
            self.assertNotIn(repr(value), text)
        self.assertNotIn("truth", text)


class RoutingControlTests(Shared):
    def test_freeze_check_blocks_live_catalog_and_registry_bindings(self):
        from unittest.mock import patch
        with routing.OfflineRouter(self.ald, self.case, self.inj, self.out('freeze_check'), 'freeze_check') as router:
            self.assertTrue(router.verify_offline_binding()['passed'])
            self.assertEqual(self.ald.get_stellar_and_gaia_params(self.case['tic_id']), self.case['stellar'])
            self.assertEqual(self.ald.get_known_transits_for_tic(self.case['tic_id']), [])
            with patch.object(self.ald, 'get_stellar_and_gaia_params', lambda _: {}):
                with self.assertRaisesRegex(routing.RoutingError, 'LIVE_QUERY_BINDING'):
                    router.verify_offline_binding()
            with patch.object(self.ald, 'get_known_transits_for_tic', lambda _: []):
                with self.assertRaisesRegex(routing.RoutingError, 'LIVE_QUERY_BINDING'):
                    router.verify_offline_binding()
            with self.assertRaisesRegex(OSError, 'NETWORK_BLOCKED'):
                socket.getaddrinfo('catalog.invalid', 443)

    def test_missing_production_pixel_provenance_is_refused(self):
        import copy
        rec = copy.deepcopy(self.record)
        rec["routing"]["production_sha256_records"] = [
            r for r in rec["routing"]["production_sha256_records"] if r["sha256"] != self.case["tpf_sha256"]]
        self.assertFalse(routing.verify_routing(rec, self.case)["passed"])
        self.assertTrue(routing.verify_routing(self.record, self.case)["passed"])

    def test_r1_every_consumed_product_is_the_injected_copy(self):
        v = routing.verify_routing(self.record, self.case, originals=self.originals)
        self.assertTrue(v["passed"], v["problems"])
        kinds = sorted((p["kind"], p["sector"]) for p in v["products"])
        self.assertEqual(kinds, [("archival_lc", 12), ("archival_lc", 13), ("discovery_lc", 40), ("discovery_tpf", 40)])
        case_hashes = {r["sha256"] for r in self.case["lightcurves"]} | {self.case["tpf_sha256"]}
        self.assertEqual({p["sha256"] for p in v["products"]}, case_hashes)
        self.assertFalse(case_hashes & set(self.originals))                 # every stage read an injected file
        # The production code's own SHA256 records (S3.6) name the same light curves and pixel file: an independent witness.
        prod = {m["sha256"] for m in self.record["routing"]["production_sha256_records"]}
        self.assertEqual(prod, case_hashes)
        self.assertEqual(self.record["routing"]["network_attempts"], [])
        # If the production records named another set of files than the router's ledger, the run is not valid.
        fewer = dict(self.record, routing=dict(self.record["routing"],
                                               production_sha256_records=self.record["routing"]["production_sha256_records"][:-1]))
        v2 = routing.verify_routing(fewer, self.case, originals=self.originals)
        self.assertFalse(v2["passed"])
        self.assertEqual(len(v2["problems"]), 1)
        self.assertIn("production SHA256 records", v2["problems"][0])

    def test_r2_the_signal_survives_the_included_stages(self):
        s = routing.signal_in_production_arrays(self.ald, self.case, self.inj, self.host, self.hosts, injection.model_flux, self.truth)
        self.assertEqual(s["n_common"], s["n_original"])
        self.assertGreater(s["n_in_transit"], 100)
        self.assertLessEqual(s["max_abs_ratio_minus_model"], 1e-9)          # discovery arrays carry exactly the model
        rec = matching.recover(self.record["search"]["top_peaks"], self.truth, self.times)
        self.assertEqual((rec["outcome"], rec["rank"]), ("MATCH_FUNDAMENTAL", 1))        # search stage
        peak = next(p for p in self.record["peaks"] if p["peak_rank"] == 1)
        gates = {g["gate"]: g["status"] for g in peak["gates"]}
        self.assertEqual(gates["TPF"], "PASS")                              # the injected pixels reach the pixel gate
        self.assertEqual(gates["ARCHIVAL"], "PASS")                         # the injected archival curves reach Filter 7
        arch = self.record["archival_diagnostics"][0]
        self.assertEqual(sorted(arch["archival_sectors_used"]), [12, 13])
        self.assertTrue(arch["arch_status"].startswith("Confirmed_in_Sectors_12,13"))
        self.assertEqual(self.record["status"], "CANDIDATE", self.record["message"])
        cand = self.record["candidate"]
        self.assertAlmostEqual(cand["discovery_period_days"], self.truth["period"], delta=0.01)
        self.assertAlmostEqual(cand["depth_ppm"] / (self.truth["depth"] * 1e6), 1.0, delta=0.35)

    def test_r2_uninjected_host_does_not_show_the_signal(self):
        base = dict(self.case, case_id="uninjected", source_fits=self.host["source_fits"], source_sha256=self.host["source_sha256"],
                    tpf_fits=self.host["tpf_fits"], tpf_sha256=self.host["tpf_sha256"], lightcurves=[dict(r) for r in self.host["lightcurves"]])
        rec = routing.run_case(self.ald, base, self.hosts, self.out("uninjected"), "route_test")
        self.assertNotEqual(rec["status"], "CANDIDATE")
        times = np.load(os.path.join(self.out("uninjected"), rec["search"]["times_file"]))
        self.assertNotEqual(matching.recover(rec["search"]["top_peaks"], self.truth, times)["outcome"], "MATCH_FUNDAMENTAL")
        # The ledger knows these are originals: the control that would catch a silent replacement.
        v = routing.verify_routing(rec, base, originals=self.originals)
        self.assertFalse(v["passed"])
        self.assertTrue(any("uninjected original" in p for p in v["problems"]))

    def test_r3_swapped_file_is_refused_before_the_detector_runs(self):
        root = os.path.join(self.tmp, "swap")
        shutil.copytree(self.inj, root)
        shutil.copyfile(os.path.join(self.hosts, self.host["source_fits"]), os.path.join(root, self.case["source_fits"]))
        with self.assertRaisesRegex(routing.RoutingError, "INPUT_HASH_MISMATCH"):
            routing.run_case(self.ald, self.case, root, self.out("swap"), "route_test")
        self.assertFalse(os.path.exists(os.path.join(self.out("swap"), "experimental_v2_vetting.jsonl")))
        root2 = os.path.join(self.tmp, "swap_tpf")
        shutil.copytree(self.inj, root2)
        shutil.copyfile(os.path.join(self.hosts, self.host["tpf_fits"]), os.path.join(root2, self.case["tpf_fits"]))
        with self.assertRaisesRegex(routing.RoutingError, "INPUT_HASH_MISMATCH"):
            routing.run_case(self.ald, self.case, root2, self.out("swap_tpf"), "route_test")

    def test_r3_file_changed_during_the_run_is_caught_at_download(self):
        root = os.path.join(self.tmp, "late_swap")
        shutil.copytree(self.inj, root)
        arch = next(r for r in self.case["lightcurves"] if r["sector"] == 12)
        orig = next(r for r in self.host["lightcurves"] if r["sector"] == 12)
        real_check = routing.check_case

        def check_then_swap(case, r):
            real_check(case, r)                                             # the case is valid when the run starts ...
            shutil.copyfile(os.path.join(self.hosts, orig["path"]), os.path.join(root, arch["path"]))   # ... then replaced
        from unittest.mock import patch
        with patch.object(routing, "check_case", check_then_swap):
            rec = routing.run_case(self.ald, self.case, root, self.out("late_swap"), "route_test")
        # The production archival stage records the refused product as failed and goes on with sector 13 alone, so
        # the uninjected file is never read; the ledger marks the run as not valid for a measurement.
        arch_diag = rec["archival_diagnostics"][0]
        self.assertEqual(arch_diag["archival_sectors_used"], [13])
        self.assertEqual([f["stage"] for f in arch_diag["archival_products_failed"]], ["DOWNLOAD"])
        bad = [c for c in rec["routing"]["consumed"] if not c["match"]]
        self.assertTrue(bad)
        self.assertEqual({(c["kind"], c["sector"]) for c in bad}, {("archival_lc", 12)})
        self.assertEqual({m["sha256"] for m in rec["routing"]["production_sha256_records"]},
                         {r["sha256"] for r in self.case["lightcurves"] if r["sector"] != 12} | {self.case["tpf_sha256"]})
        v = routing.verify_routing(rec, self.case, originals=self.originals)
        self.assertFalse(v["passed"])
        self.assertTrue(any("changed after the case was written" in x for x in v["problems"]))

    def test_r3_unrouted_fetches_are_blocked_and_recorded(self):
        with routing.OfflineRouter(self.ald, self.case, self.inj, self.out("net"), "route_test") as router:
            with self.assertRaises(OSError):
                socket.create_connection(("mast.stsci.edu", 443), timeout=1)
            with self.assertRaises(OSError):
                socket.getaddrinfo("archive.stsci.edu", 443)
            for bad in (("TIC 1", dict(mission="TESS", author="SPOC", exptime=120)),
                        (f"TIC {self.case['tic_id']}", dict(mission="TESS", author="QLP", exptime=120)),
                        (f"TIC {self.case['tic_id']}", dict(mission="TESS", author="SPOC", exptime=20))):
                with self.subTest(bad=bad), self.assertRaisesRegex(routing.RoutingError, "ROUTING_UNEXPECTED_QUERY"):
                    router.search_lightcurve(bad[0], **bad[1])
            with self.assertRaisesRegex(routing.RoutingError, "ROUTING_UNEXPECTED_QUERY"):
                router.stellar("999")
            ok = dict(mission="TESS", author="SPOC", exptime=120)
            self.assertEqual(len(router.search_targetpixelfile(f"TIC {self.case['tic_id']}", sector=40, **ok)), 1)
            self.assertEqual(len(router.search_targetpixelfile(f"TIC {self.case['tic_id']}", sector=12, **ok)), 0)   # no substitute
            self.assertEqual(len(router.search_lightcurve(f"TIC {self.case['tic_id']}", sector=40, **ok)), 1)
            self.assertEqual(len(router.search_lightcurve(f"TIC {self.case['tic_id']}", **ok)), 3)
            self.assertEqual(len(router.network_attempts), 2)
            fake = dict(self.record, routing=dict(self.record["routing"], network_attempts=list(router.network_attempts)))
        self.assertFalse(routing.verify_routing(fake, self.case)["passed"])
        self.assertIsNot(socket.create_connection, None)

    def test_router_restores_the_production_module(self):
        import lightkurve as lk
        before = (lk.search_lightcurve, lk.search_targetpixelfile, self.ald.get_stellar_and_gaia_params,
                  self.ald.get_known_transits_for_tic, self.ald.BoxLeastSquares, self.ald.get_top_distinct_bls_peaks,
                  self.ald.OUTPUT_DIR, self.ald.AUTO_PURGE_CACHE, socket.create_connection, socket.getaddrinfo)
        with routing.OfflineRouter(self.ald, self.case, self.inj, self.out("restore"), "route_test"):
            self.assertIsNot(lk.search_lightcurve, before[0])
            self.assertFalse(self.ald.AUTO_PURGE_CACHE)
            self.assertEqual(self.ald.OUTPUT_DIR, self.out("restore"))
        after = (lk.search_lightcurve, lk.search_targetpixelfile, self.ald.get_stellar_and_gaia_params,
                 self.ald.get_known_transits_for_tic, self.ald.BoxLeastSquares, self.ald.get_top_distinct_bls_peaks,
                 self.ald.OUTPUT_DIR, self.ald.AUTO_PURGE_CACHE, socket.create_connection, socket.getaddrinfo)
        self.assertEqual(before, after)

    def test_r4_truth_is_not_exposed_to_detection(self):
        for leak in (dict(truth=self.truth), dict(period=2.7), dict(injection_id="x")):
            with self.subTest(leak=list(leak)), self.assertRaisesRegex(routing.RoutingError, "DETECTOR_TRUTH_LEAK"):
                routing.run_case(self.ald, dict(self.case, **leak), self.inj, self.out("leak_" + list(leak)[0]), "route_test")
        bad = copy.deepcopy(self.case)
        bad["lightcurves"][0]["snr"] = 30.0
        with self.assertRaisesRegex(routing.RoutingError, "DETECTOR_TRUTH_LEAK"):
            routing.check_case(bad, self.inj)
        bad = copy.deepcopy(self.case)
        bad["stellar"]["k_true"] = 0.05
        with self.assertRaisesRegex(routing.RoutingError, "DETECTOR_TRUTH_LEAK"):
            routing.check_case(bad, self.inj)
        with self.assertRaisesRegex(routing.RoutingError, "CASE_INCOMPLETE"):
            routing.check_case({k: v for k, v in self.case.items() if k != "known_transits"}, self.inj)
        # The detector's known-signal context comes from the case alone (empty here: a catalog-blinded run).
        self.assertIn(dict(kind="known_transits", target=self.case["tic_id"], sector=None, problems=[]), self.record["routing"]["queries"])

    def test_observation_hooks_do_not_change_the_detector_records(self):
        plain = routing.run_case(self.ald, self.case, self.inj, self.out("plain"), "route_test", observe=False)
        self.assertIsNone(plain["search"]["top_peaks"])
        for key in ("status", "message", "candidate", "peaks", "journal", "host_end"):
            with self.subTest(key=key):
                a, b = copy.deepcopy(self.record[key]), copy.deepcopy(plain[key])
                self.assertEqual(a, b)
        strip = lambda d: [{k: v for k, v in x.items() if k not in ("timestamp", "attempt_id")} for x in d]
        self.assertEqual(strip(self.record["archival_diagnostics"]), strip(plain["archival_diagnostics"]))
        self.assertEqual([c["sha256"] for c in self.record["routing"]["consumed"]], [c["sha256"] for c in plain["routing"]["consumed"]])

    def test_missing_target_pixel_file_and_missing_archive_follow_the_production_states(self):
        no_tpf = dict(self.case, case_id="no_tpf", tpf_fits=None, tpf_sha256=None)
        rec = routing.run_case(self.ald, no_tpf, self.inj, self.out("no_tpf"), "route_test")
        self.assertEqual(rec["status"], "INCOMPLETE_TPF_UNAVAILABLE")       # item 37, through the routed query
        self.assertTrue(routing.verify_routing(rec, no_tpf, originals=self.originals)["passed"])
        disc = [r for r in self.case["lightcurves"] if r["sector"] == self.case["sector"]]
        single = dict(self.case, case_id="single", lightcurves=disc)
        rec = routing.run_case(self.ald, single, self.inj, self.out("single"), "route_test")
        self.assertEqual(rec["status"], "CANDIDATE", rec["message"])
        self.assertEqual(rec["candidate"]["n_archival_sectors"], 0)         # item 31.6: a confirmed zero-sector archive
        self.assertEqual([p["kind"] for p in routing.verify_routing(rec, single)["products"]], ["discovery_lc", "discovery_tpf"])

    def test_run_directory_must_be_new(self):
        with self.assertRaisesRegex(routing.RoutingError, "OUTPUT_NOT_EMPTY"):
            routing.run_case(self.ald, self.case, self.inj, self.out("case_a"), "route_test")

    def test_r5_boundary_inventory(self):
        inv = routing.boundary_inventory(REPO)
        self.assertTrue(inv["passed"], inv["unrouted_detector_sites"])
        sites = {(s["function"], s["call"]) for s in inv["detector_sites"]}
        self.assertIn(("_load_discovery_light_curve", "lk.search_lightcurve"), sites)  # C7 discovery-sector stage
        self.assertIn(("verify_multi_epoch", "lk.search_lightcurve"), sites)
        self.assertIn(("_verify_and_filter_tpf_cp", "lk.search_targetpixelfile"), sites)
        self.assertEqual(sorted(inv["outside_boundary"]), sorted(routing.UNROUTED_STAGES))
        for name, st in inv["outside_boundary"].items():
            self.assertTrue(st["present"] and st["fetch_sites"], name)      # these stages do fetch on their own
            self.assertFalse(st["inside_boundary"])
        # A new fetch site in a detector function, or in a detector stage (C7, DETECTOR_STAGES), fails the inventory.
        alt = os.path.join(self.tmp, "alt_repo")
        os.makedirs(alt)
        for name in ["aldebaran.py"] + list(routing.UNROUTED_STAGES):
            shutil.copyfile(os.path.join(REPO, name), os.path.join(alt, name))
        with open(os.path.join(alt, "aldebaran.py"), encoding="utf-8") as fh:
            src = fh.read()
        marker = "    products_requested = []\n"
        self.assertEqual(src.count(marker), 1)
        with open(os.path.join(alt, "aldebaran.py"), "w", encoding="utf-8") as fh:
            stage = '    """Filter 6: pixel-level TPF localization."""\n'
            self.assertEqual(src.count(stage), 1)
            fh.write(src.replace(marker, marker + '    _extra = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))\n')
                     .replace(stage, stage + '    _extra = Catalogs.query_criteria(catalog="Tic", ID=int(host.tic_id))\n'))
        bad = routing.boundary_inventory(alt)
        self.assertFalse(bad["passed"])
        self.assertEqual([(s["function"], s["call"]) for s in bad["unrouted_detector_sites"]],
                         [("verify_multi_epoch", "Catalogs.query_criteria"), ("_peak_pixel_gate", "Catalogs.query_criteria")])


if __name__ == "__main__":
    unittest.main()
