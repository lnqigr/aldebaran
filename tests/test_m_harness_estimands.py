"""S4-M-PACKAGE-01, part B4: estimand calculators (ARGON_DECISIONS items 31.3, 31.4, 31.7 to 31.9, 34.3, 36.2, 37.1,
38.4). Records are synthetic and have the shape routing.run_case returns; matching uses the frozen scorer."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ALDEBARAN_OFFLINE", "1")
from m_harness import estimands as E  # noqa: E402
from m_harness import matching, prodpath, stats  # noqa: E402

TIMES = 2000.0 + (2.0 / 1440.0) * np.arange(18000)          # 25 days of 2-minute cadences
TRUTH = dict(period=3.0, t0=2001.0, t14=0.1)


def peak(rank, period=3.0, t0=2001.0, sde=9.0, dur=0.1):
    return dict(rank=rank, period_days=period, t0_btjd=t0, duration_days=dur, sde=sde)


def evidence(rank, disposition, gates=()):
    return dict(peak_rank=rank, period_days=3.0, t0_btjd=2001.0, duration_days=0.1, disposition=disposition,
                gates=[dict(gate=g, status=s, metrics={}) for g, s in gates])


def jrow(rank, disposition, exit_code=None, cause=None, gate=None, sde=9.0, period=3.0):
    return dict(record="rank", rank=rank, period_days=period, sde=sde, disposition=disposition, exit_code=exit_code,
                cause=cause, first_reject_gate=gate)


def record(case_id, status, top=None, journal=(), peaks=(), first_candidate=None, candidate=None, arch=()):
    return dict(case_id=case_id, status=status, message="", candidate=candidate, peaks=list(peaks), journal=list(journal),
                host_end=dict(first_candidate_rank=first_candidate) if journal else None,
                archival_diagnostics=list(arch), search=dict(top_peaks=top))


def cand(sde=9.0, n_arch=2, period=3.0):
    return dict(sde=sde, n_archival_sectors=n_arch, period_days=period, discovery_period_days=period, t0_btjd=2001.0,
                depth_ppm=1000.0, r_planet_earth=3.0, cramer_rao_sigma_p_sec=86.4, cramer_rao_sigma_t0_sec=86.4,
                ephemeris_source="DISCOVERY_SINGLE_SECTOR")


def matched_candidate(case_id="c", sde=9.0):
    return record(case_id, "CANDIDATE", top=[peak(1, sde=sde)], journal=[jrow(1, "CANDIDATE", sde=sde)],
                  peaks=[evidence(1, "CANDIDATE", [("TPF", "PASS"), ("ARCHIVAL", "PASS")])], first_candidate=1, candidate=cand(sde))


class StatusTests(unittest.TestCase):
    def test_every_production_status_has_a_class(self):
        cases = dict(CANDIDATE="CANDIDATE", REVIEW_VETTING_HOLD="REVIEW", REVIEW_SECONDARY_NOT_EVALUATED="REVIEW",
                     INCOMPLETE_TPF_UNAVAILABLE="INCOMPLETE", INCOMPLETE_ARCHIVAL_UNAVAILABLE="INCOMPLETE", ERROR="FAILED",
                     SKIP_NO_DATA="UNAVAILABLE", REJECT_CROWDSAP="DOMAIN_STOP", REJECT_RUWE="DOMAIN_STOP",
                     REJECT_EVOLVED_STAR="DOMAIN_STOP", REJECT_BASELINE="HOST_STOP", REJECT_NO_STELLAR_PARAMS="HOST_STOP",
                     EMPTY_LOW_SDE="NO_DETECTION", REJECT_SECONDARY_ECLIPSE="GATE_REJECT", REJECT_DEPTH="GATE_REJECT")
        for status, klass in cases.items():
            self.assertEqual(E.classify_status(status), klass, status)

    def test_unknown_status_raises(self):
        for s in ("DONE", "", "candidate", "PASS"):
            with self.subTest(s=s), self.assertRaisesRegex(ValueError, "UNCLASSIFIED_STATUS"):
                E.classify_status(s)

    def test_statuses_in_the_detector_source_are_all_classified(self):
        import re
        ald = prodpath.load_production()
        with open(ald.__file__, encoding="utf-8") as fh:
            src = fh.read()
        found = set(re.findall(r'"((?:REJECT|SKIP|EMPTY|INCOMPLETE|REVIEW)_[A-Z_]+)"', src))
        found |= set(re.findall(r"'((?:REJECT|SKIP|EMPTY|INCOMPLETE|REVIEW)_[A-Z_]+)'", src))
        self.assertGreater(len(found), 25)
        for s in sorted(found):
            E.classify_status(s)                                            # raises if a status is unknown

    def test_crowding_strata_of_item_31_9(self):
        self.assertEqual(E.crowd_stratum(0.90), "CROWDSAP >= 0.90")
        self.assertEqual(E.crowd_stratum(0.8999), "CROWDSAP [0.70, 0.90)")
        self.assertEqual(E.crowd_stratum(0.70), "CROWDSAP [0.70, 0.90)")
        self.assertEqual(E.crowd_stratum(0.6999), "CROWDSAP < 0.70 (outside the domain)")
        self.assertEqual(E.crowd_stratum(None), "CROWDSAP unknown")

    def test_domain_boundaries(self):
        edge = dict(crowdsap=0.70, ruwe=1.4, r_star=1.75, teff=9500.0, depth_ppm=180.0, rp_earth=15.0)
        self.assertEqual(E.domain_check(edge), dict(in_domain=True, reasons=[], unknown=[]))
        self.assertTrue(E.domain_check(dict(edge, depth_ppm=30000.0))["in_domain"])
        for key, value, why in (("crowdsap", 0.699, "CROWDSAP < 0.70"), ("ruwe", 1.41, "RUWE > 1.4"),
                                ("r_star", 1.76, "R* > 1.75 R_sun"), ("teff", 9501.0, "Teff > 9500 K"),
                                ("depth_ppm", 179.9, "depth < 180 ppm"), ("depth_ppm", 30001.0, "depth > 30,000 ppm"),
                                ("rp_earth", 15.1, "Rp > 15 R_E")):
            d = E.domain_check(dict(edge, **{key: value}))
            self.assertEqual((d["in_domain"], d["reasons"]), (False, [why]))
        unknown = E.domain_check(dict(edge, ruwe=None))
        self.assertEqual((unknown["in_domain"], unknown["unknown"]), (None, ["ruwe"]))   # unknown is not inside
        self.assertEqual(E.RP_EARTH_PER_RSUN, 109.076)
        with open(prodpath.load_production().__file__, encoding="utf-8") as fh:
            self.assertIn("r_star * rp_rs * 109.076", fh.read())


class CaseOutcomeTests(unittest.TestCase):
    def test_matched_candidate_survives(self):
        o = E.case_outcome(matched_candidate(), TRUTH, TIMES)
        self.assertEqual((o["stage"], o["line"], o["matched_rank"]), ("NONE", "MATCHED_CANDIDATE", 1))
        self.assertTrue(o["matched_candidate"] and o["search_recovered"] and o["matched_candidate_or_review"])
        self.assertTrue(o["reached_archival"])

    def test_search_losses(self):
        low = record("a", "EMPTY_LOW_SDE", top=[peak(1, sde=5.2)], journal=[jrow(1, "PRE_EVIDENCE_EXIT", "EMPTY_LOW_SDE", sde=5.2)])
        self.assertEqual(E.case_outcome(low, TRUTH, TIMES)["line"], "NO_PEAK_ABOVE_6")
        alias = record("b", "REJECT_ODD_EVEN", top=[peak(1, period=1.5)], journal=[jrow(1, "REJECT", "REJECT_ODD_EVEN", period=1.5)])
        o = E.case_outcome(alias, TRUTH, TIMES)
        self.assertEqual((o["stage"], o["line"], o["search_recovered"]), ("SEARCH", "ALIAS_ONLY", False))
        other = record("c", "CANDIDATE", top=[peak(1, period=4.1, t0=2000.5)], journal=[jrow(1, "CANDIDATE", period=4.1)],
                       first_candidate=1, candidate=cand(period=4.1))
        o = E.case_outcome(other, TRUTH, TIMES)                             # a CANDIDATE at an unrelated period is not a recovery
        self.assertEqual((o["line"], o["matched_candidate"], o["is_candidate"]), ("NOT_MATCHED", False, True))
        sparse = dict(period=11.0, t0=2001.0, t14=0.1)                      # 3 events would need 22 d after t0 ... only 2 fit
        short_times = TIMES[:9000]
        unavailable = record("d", "EMPTY_LOW_SDE", top=[peak(1, period=11.0)], journal=[jrow(1, "REJECT", "REJECT_FEW_TRANSITS")])
        self.assertEqual(E.case_outcome(unavailable, sparse, short_times)["line"], "MATCH_UNAVAILABLE")

    def test_vetting_losses_follow_the_matched_rank(self):
        rej = record("a", "REJECT_SECONDARY_ECLIPSE", top=[peak(1)], journal=[jrow(1, "REJECT", "REJECT_SECONDARY_ECLIPSE", "SECONDARY", "SECONDARY")],
                     peaks=[evidence(1, "REJECT", [("SECONDARY", "REJECT")])])
        self.assertEqual(E.case_outcome(rej, TRUTH, TIMES)["line"], "GATE:SECONDARY")
        pre = record("b", "REJECT_DEPTH", top=[peak(1)], journal=[jrow(1, "PRE_EVIDENCE_EXIT", "REJECT_MORPHOLOGY", "REJECT_MORPHOLOGY")])
        self.assertEqual(E.case_outcome(pre, TRUTH, TIMES)["line"], "GATE:REJECT_MORPHOLOGY")
        rev = record("c", "REVIEW_SECONDARY_NOT_EVALUATED", top=[peak(1)], journal=[jrow(1, "REVIEW", cause="SECONDARY_NOT_EVALUATED")],
                     peaks=[evidence(1, "REVIEW", [("CENTROID", "NOT_EVALUATED")])])
        o = E.case_outcome(rev, TRUTH, TIMES)
        self.assertEqual((o["stage"], o["line"]), ("VETTING", "REVIEW:SECONDARY_NOT_EVALUATED"))
        self.assertEqual((o["matched_candidate"], o["matched_candidate_or_review"]), (False, True))
        self.assertEqual(o["review_subrow"], "REVIEW_SECONDARY_NOT_EVALUATED")       # item 38.4 sub-row
        self.assertTrue(o["centroid_not_evaluated"])                                # item 34.3 stratum
        # The matched peak is rank 2 and it was rejected, while rank 1 (an unrelated peak) became the CANDIDATE.
        two = record("d", "CANDIDATE", top=[peak(1, period=4.1, t0=2000.5), peak(2)],
                     journal=[jrow(1, "CANDIDATE", period=4.1), jrow(2, "REJECT", "REJECT_CENTROID", gate="CENTROID")],
                     first_candidate=1, candidate=cand(period=4.1))
        o = E.case_outcome(two, TRUTH, TIMES)
        self.assertEqual((o["matched_rank"], o["line"], o["matched_candidate"]), (2, "GATE:CENTROID", False))
        # The matched peak is CANDIDATE evidence at rank 2, but the reported candidate is rank 1.
        shadow = record("e", "CANDIDATE", top=[peak(1, period=4.1, t0=2000.5), peak(2)],
                        journal=[jrow(1, "CANDIDATE", period=4.1), jrow(2, "CANDIDATE")], first_candidate=1, candidate=cand(period=4.1))
        o = E.case_outcome(shadow, TRUTH, TIMES)
        self.assertEqual((o["stage"], o["matched_candidate"]), ("FINAL", False))
        # A host-level return ended the host before the matched rank was evaluated.
        cut = record("f", "REJECT_RUWE", top=[peak(1, period=4.1, t0=2000.5), peak(2)], journal=[jrow(1, "REJECT", "REJECT_RUWE", gate="RUWE", period=4.1)])
        self.assertEqual(E.case_outcome(cut, TRUTH, TIMES)["line"], "NOT_EVALUATED:REJECT_RUWE")

    def test_host_and_run_level_losses(self):
        self.assertEqual(E.case_outcome(record("a", "REJECT_CROWDSAP"), TRUTH, TIMES)["stage"], "HOST")
        self.assertEqual(E.case_outcome(record("b", "ERROR"), TRUTH, TIMES)["stage"], "RUN")
        self.assertEqual(E.case_outcome(record("c", "SKIP_NO_DATA"), TRUTH, TIMES)["line"], "SKIP_NO_DATA")
        with self.assertRaisesRegex(ValueError, "NO_SEARCH_RECORD"):
            E.case_outcome(record("d", "EMPTY_LOW_SDE"), TRUTH, TIMES)      # a searched host must have its peaks
        with self.assertRaisesRegex(ValueError, "NO_SEARCH_TIMES"):
            E.case_outcome(matched_candidate(), TRUTH, None)


def m1_row(i, snr_bin, snr, host=None, crowdsap=0.95, grey=False, lost=None, sde=9.0):
    r = dict(case_id=f"i{i}", host=host if host is not None else f"h{i}", crowdsap=crowdsap, snr_bin=snr_bin, snr_target=snr,
             search_recovered=lost is None, matched_candidate=lost is None, matched_candidate_or_review=lost is None,
             matched_sde=None if lost else (6.8 if grey else sde), candidate_sde=None if lost else (6.8 if grey else sde),
             stage="NONE" if lost is None else lost[0], line="MATCHED_CANDIDATE" if lost is None else lost[1],
             review_subrow=None, centroid_not_evaluated=False)
    return r


class LedgerAndM1Tests(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(10):                                                 # bin [4, 6): 2 grey recoveries of 10
            rows.append(m1_row(i, "[4, 6)", 4.2 + 0.18 * i, grey=i < 2, lost=None if i < 2 else ("SEARCH", "NO_PEAK_ABOVE_6")))
        for i in range(10, 20):                                             # bin [6, 8): 3 grey, 3 strong, 4 lost
            rows.append(m1_row(i, "[6, 8)", 6.1 + 0.18 * (i - 10), grey=i < 13, lost=None if i < 16 else ("VETTING", "GATE:SECONDARY")))
        for i in range(20, 30):                                             # bin [8, 10]: 1 grey, 7 strong, 2 lost
            rows.append(m1_row(i, "[8, 10]", 8.1 + 0.18 * (i - 20), grey=i == 20, crowdsap=0.80,
                               lost=None if i < 28 else ("VETTING", "REVIEW:ODD_EVEN_INCONCLUSIVE")))
        return rows

    def test_ledger_balances_and_refuses_gaps(self):
        rows = self.rows() + [dict(case_id="x", stage="PLANNING", line="DRAW_FAILED")]
        t = E.attrition_table(rows, 31)
        self.assertEqual((t["n_planned"], t["n_survivors"]), (31, 16))
        self.assertEqual([(s["stage"], s["lost"], s["remaining"]) for s in t["stages"]],
                         [("PLANNING", 1, 30), ("RUN", 0, 30), ("HOST", 0, 30), ("SEARCH", 8, 22), ("VETTING", 6, 16), ("FINAL", 0, 16)])
        self.assertEqual(t["stages"][4]["lines"], [dict(line="GATE:SECONDARY", n=4), dict(line="REVIEW:ODD_EVEN_INCONCLUSIVE", n=2)])
        with self.assertRaisesRegex(ValueError, "LEDGER_UNBALANCED"):
            E.attrition_table(rows, 32)                                     # a planned case without a row
        with self.assertRaisesRegex(ValueError, "LEDGER_UNBALANCED"):
            E.attrition_table(rows[:-1] + [dict(case_id="y", stage="LOST", line="?")], 31)

    def test_completeness_and_grey_gain_per_bin(self):
        m = E.m1_completeness(self.rows(), ["[4, 6)", "[6, 8)", "[8, 10]"])
        b = m["by_snr_bin"]
        self.assertEqual([(b[k]["c_sde_6"]["k"], b[k]["c_sde_7_5"]["k"], b[k]["dc_grey"]["k"], b[k]["n"]) for k in b],
                         [(2, 0, 2, 10), (6, 3, 3, 10), (8, 7, 1, 10)])
        self.assertAlmostEqual(b["[6, 8)"]["dc_grey"]["rate"], b["[6, 8)"]["c_sde_6"]["rate"] - b["[6, 8)"]["c_sde_7_5"]["rate"])
        self.assertEqual(m["pooled"]["matched_candidate"]["k"], 16)
        self.assertEqual(m["pooled"]["matched_candidate"]["interval"], "Clopper-Pearson 95 % two-sided")
        self.assertEqual(sorted(m["by_crowding"]), ["CROWDSAP >= 0.90", "CROWDSAP [0.70, 0.90)"])
        self.assertEqual(m["by_crowding"]["CROWDSAP [0.70, 0.90)"]["n"], 10)
        self.assertTrue(m["pooled"]["supported"])                           # item 43.4: 30 distinct hosts
        marked = E.m1_completeness(self.rows(), ["[4, 6)", "[6, 8)", "[8, 10]"], min_cell=30)
        self.assertEqual((marked["pooled"]["supported"], marked["by_snr_bin"]["[4, 6)"]["supported"]), (True, False))
        with self.assertRaisesRegex(ValueError, "M1_UNBINNED_CASES"):
            E.m1_completeness(self.rows() + [m1_row(99, "[10, 15)", 12.0)], ["[4, 6)", "[6, 8)", "[8, 10]"])

    def test_completeness_is_conditioned_on_injected_snr_never_on_recovered_sde(self):
        rows = self.rows()
        # Lost injections stay in the denominator of their SNR_inj bin although they have no SDE at all.
        self.assertEqual(E.m1_completeness(rows, ["[4, 6)", "[6, 8)", "[8, 10]"])["by_snr_bin"]["[4, 6)"]["n"], 10)
        self.assertEqual(sum(1 for r in rows if r["snr_bin"] == "[4, 6)" and r["candidate_sde"] is None), 8)

    def test_levels_differ_when_a_matched_peak_is_not_the_candidate(self):
        rows = self.rows()
        rows[16]["search_recovered"], rows[16]["matched_sde"] = True, 6.5   # found by the search, rejected by vetting
        cand_level = E.completeness_cell(rows, "candidate")
        search_level = E.completeness_cell(rows, "search")
        self.assertEqual(search_level["dc_grey"]["k"], cand_level["dc_grey"]["k"] + 1)
        with self.assertRaises(ValueError):
            E.completeness_cell(rows, "peak")

    def test_grey_gain_over_snr_5_to_9(self):
        rows = self.rows()
        g = E.grey_zone_gain(rows, "snr_target", bin_edges=[(4.0, 6.0), (6.0, 8.0), (8.0, 10.0)])
        inside = [r for r in rows if 5.0 <= r["snr_target"] <= 9.0]
        self.assertEqual(g["pooled"]["n"], len(inside))
        self.assertEqual(g["pooled"]["dc_grey"]["k"], sum(1 for r in inside if r["candidate_sde"] == 6.8))
        self.assertTrue(all(r["snr_target"] >= 5.0 for r in inside) and len(inside) < len(rows))
        self.assertEqual([p["range"] for p in g["bin_weighted"]["parts"]], [[5.0, 6.0], [6.0, 8.0], [8.0, 9.0]])
        parts = g["bin_weighted"]["parts"]
        self.assertAlmostEqual(g["bin_weighted"]["value"], (1 * parts[0]["dc_grey"] + 2 * parts[1]["dc_grey"] + 1 * parts[2]["dc_grey"]) / 4)
        self.assertEqual(sum(p["n"] for p in parts), len(inside))
        edge = [m1_row(1, "[4, 6)", 5.0, grey=True), m1_row(2, "[8, 10]", 9.0, grey=True), m1_row(3, "[4, 6)", 4.999, grey=True),
                m1_row(4, "[8, 10]", 9.001, grey=True)]
        self.assertEqual(E.grey_zone_gain(edge, "snr_target")["pooled"]["n"], 2)        # both ends of the support count

    def test_repeated_hosts_get_a_cluster_interval(self):
        rows = [m1_row(i, "[6, 8)", 7.0, host=f"h{i // 5}", grey=(i // 5) % 2 == 0) for i in range(100)]
        c = E.completeness_cell(rows, "candidate", n_boot=400, seed=1)
        self.assertEqual(c["dc_grey"]["n_hosts"], 20)
        hc = c["dc_grey"]["host_cluster"]
        self.assertGreater(hc["hi"] - hc["lo"], c["dc_grey"]["hi"] - c["dc_grey"]["lo"])
        single = E.completeness_cell([m1_row(i, "[6, 8)", 7.0) for i in range(20)], "candidate")
        self.assertNotIn("host_cluster", single["dc_grey"])

    def test_grey_zone_decision_of_item_31_8(self):
        self.assertEqual(E.grey_zone_decision(0.05, 0.015)["decision"], "KEEP")          # equality satisfies both
        self.assertEqual(E.grey_zone_decision(0.0499, 0.010)["decision"], "WITHDRAW")
        self.assertEqual(E.grey_zone_decision(0.20, 0.0151)["decision"], "WITHDRAW")
        d = E.grey_zone_decision(0.20, 0.0151)
        self.assertEqual((d["gain_condition"], d["far_condition"], d["keep_claim"]), (True, False, False))
        self.assertEqual(E.grey_zone_decision(None, 0.01), dict(decision="NOT_EVALUABLE", keep_claim=False, dc_grey=None, m4c_upper_at_6=0.01))
        # At N = 500 the claim survives 2 CANDIDATEs (upper bound 1.25 %) but not 3 (1.54 %).
        self.assertTrue(E.grey_zone_decision(0.06, stats.cp_upper(2, 500))["keep_claim"])
        self.assertFalse(E.grey_zone_decision(0.06, stats.cp_upper(3, 500))["keep_claim"])


def m2_row(i, accepted, in_domain=True, status=None, crowdsap=0.95, snr=12.0, reasons=()):
    status = status or ("CANDIDATE" if accepted else "REJECT_ODD_EVEN")
    return dict(case_id=f"k{i}", status=status, status_class=E.classify_status(status), matched_candidate=accepted,
                in_domain=in_domain, domain_reasons=list(reasons), crowdsap=crowdsap, snr=snr, review_subrow=None,
                centroid_not_evaluated=False)


class M2M3Tests(unittest.TestCase):
    def test_in_domain_and_end_to_end_false_rejection(self):
        rows = [m2_row(i, i < 12) for i in range(20)]                       # 20 in domain, 12 accepted
        rows += [m2_row(20 + i, False, in_domain=False, status="REJECT_CROWDSAP", crowdsap=0.5, reasons=["CROWDSAP < 0.70"]) for i in range(3)]
        rows += [m2_row(23, False, in_domain=False, status="REJECT_DEPTH", reasons=["depth < 180 ppm"]),
                 m2_row(24, True, in_domain=False, reasons=["Rp > 15 R_E"]), m2_row(25, False, in_domain=None)]
        m = E.m2_false_rejection(rows, snr_bins=((0.0, 10.0), (10.0, 25.0)))
        self.assertEqual((m["in_domain"]["k"], m["in_domain"]["n"], m["in_domain"]["accepted"]), (8, 20, 12))
        self.assertEqual((m["end_to_end"]["k"], m["end_to_end"]["n"]), (13, 26))        # every cohort planet is in the denominator
        self.assertEqual(m["in_domain"]["interval"], "Clopper-Pearson 95 % two-sided")
        d = m["design_exclusion"]
        self.assertEqual((d["out_of_domain_planets"], d["domain_unknown"]), (5, 1))
        self.assertEqual(d["by_reason"], {"CROWDSAP < 0.70": 3, "depth < 180 ppm": 1, "Rp > 15 R_E": 1})
        self.assertEqual(d["pre_search_domain_stops"]["REJECT_CROWDSAP"], 3)
        self.assertEqual(m["by_snr"]["[10, 25)"]["n"], 20)
        self.assertEqual(sum(x["n"] for x in m["lines"]), 26)

    def test_m3_rate_uses_the_one_sided_upper_bound(self):
        def rows(k, n):
            return [dict(case_id=f"e{i}", status="CANDIDATE" if i < k else "REJECT_SECONDARY_ECLIPSE", is_candidate=i < k,
                         status_class="CANDIDATE" if i < k else "GATE_REJECT", crowdsap=0.95, review_subrow=None,
                         candidate_relation="1:1" if i == 0 else ("1/2" if i == 1 else None)) for i in range(n)]
        m = E.m3_false_positive(rows(8, 100))
        self.assertEqual(m["raw_candidate"]["interval"], "Clopper-Pearson 95 % upper")
        self.assertAlmostEqual(m["raw_candidate"]["hi"], stats.cp_upper(8, 100))
        self.assertLess(m["raw_candidate"]["hi"], 0.15)
        self.assertGreater(stats.cp_two_sided(8, 100)[1], 0.15)             # a two-sided bound would trigger the label
        self.assertFalse(m["label_rule"]["preliminary_label"])
        self.assertEqual(m["candidate_period_relation"], {"1:1": 1, "1/2": 1, "unrelated period": 6})
        self.assertEqual(m["alias_promotions"], 1)
        nine = E.m3_false_positive(rows(9, 100))["label_rule"]              # upper bound 15.2 %: label by the bound alone
        self.assertEqual((nine["point_above_limit"], nine["upper_above_limit"], nine["preliminary_label"]), (False, True, True))
        ten = E.m3_false_positive(rows(100, 1000))["label_rule"]            # exactly 10.0 % is not above 10.0 %
        self.assertFalse(ten["point_above_limit"])
        self.assertTrue(E.m3_false_positive(rows(101, 1000))["label_rule"]["point_above_limit"])
        self.assertEqual(E.m3_false_positive([])["label_rule"]["status"], "NOT_EVALUABLE")

    def test_m3_failed_rows_block_label_even_with_optimistic_raw_rate(self):
        valid = [dict(case_id=f"e{i}", status="CANDIDATE" if i < 8 else "REJECT_SECONDARY_ECLIPSE",
                      is_candidate=i < 8, status_class="CANDIDATE" if i < 8 else "GATE_REJECT",
                      crowdsap=0.95, review_subrow=None) for i in range(60)]
        for status, status_class in (("ERROR", "FAILED"), ("SKIP_NO_DATA", "UNAVAILABLE")):
            with self.subTest(status=status):
                bad = [dict(case_id=f"bad{i}", status=status, status_class=status_class,
                            is_candidate=False, crowdsap=0.95, review_subrow=None) for i in range(40)]
                result = E.m3_false_positive(valid + bad)
                self.assertEqual(result["label_rule"], dict(status="NOT_EVALUABLE", preliminary_label=None))
                self.assertEqual(result["failed_or_unavailable"],
                                 [dict(case_id=r["case_id"], status=status) for r in bad])
                self.assertEqual((result["worst_case"]["k"], result["worst_case"]["n"]), (48, 100))
                self.assertAlmostEqual(result["worst_case"]["hi"], stats.cp_upper(48, 100))
                self.assertTrue(E.m3_false_positive(valid)["label_rule"]["preliminary_label"])
                self.assertNotIn("worst_case", E.m3_false_positive(valid))
                self.assertNotIn("failed_or_unavailable", E.m3_false_positive(valid))

    def test_period_relation_labels(self):
        self.assertEqual(matching.period_relation(3.0, 3.1), "1:1")          # inside 4 %
        self.assertIsNone(matching.period_relation(3.0, 3.2))
        self.assertEqual(matching.period_relation(1.5, 3.0), "1/2")
        self.assertEqual(matching.period_relation(6.05, 3.0), "2/1")
        self.assertEqual(matching.period_relation(2.0, 3.0), "2/3")
        self.assertIsNone(matching.period_relation(1.5 * 1.03, 3.0))         # 3 % off a 1/2 alias: outside 2 %


def arch_diag(rank, mode, ok=True, status="Confirmed_in_Sectors_12,13", sci74=None):
    vr = dict(mode=mode)
    if sci74 is not None:
        vr["sci74"] = sci74
    return dict(peak_rank=rank, arch_ok=ok, arch_status=status, vr=vr)


class M4Tests(unittest.TestCase):
    def null(self, i, status="EMPTY_LOW_SDE", top_sde=5.0, **kw):
        return record(f"n{i}", status, top=[peak(1, period=1.7 + 0.01 * i, sde=top_sde)],
                      journal=[jrow(1, "PRE_EVIDENCE_EXIT", status, sde=top_sde)], **kw)

    def cohort(self):
        recs = [self.null(i) for i in range(480)]
        recs += [self.null(480 + i, "REJECT_ODD_EVEN", top_sde=6.4) for i in range(6)]                 # top peak 6.0 to 7.5
        recs += [self.null(486 + i, "REJECT_ARCHIVAL", top_sde=8.0, peaks=[evidence(1, "REJECT", [("TPF", "PASS"), ("ARCHIVAL", "REJECT")])],
                           arch=[arch_diag(1, 2, ok=False, status="Archival signal failed control phase null test")]) for i in range(4)]
        recs += [record("n490", "CANDIDATE", top=[peak(1, sde=6.9)], journal=[jrow(1, "CANDIDATE", sde=6.9)], first_candidate=1,
                        candidate=cand(6.9, n_arch=0), peaks=[evidence(1, "CANDIDATE", [("ARCHIVAL", "PASS")])]),
                 record("n491", "CANDIDATE", top=[peak(1, sde=8.2)], journal=[jrow(1, "CANDIDATE", sde=8.2)], first_candidate=1,
                        candidate=cand(8.2, n_arch=2), peaks=[evidence(1, "CANDIDATE", [("ARCHIVAL", "PASS")])], arch=[arch_diag(1, 2)])]
        recs += [self.null(492, "INCOMPLETE_TPF_UNAVAILABLE", top_sde=7.0), self.null(493, "REVIEW_SECONDARY_NOT_EVALUATED", top_sde=7.0),
                 record("n494", "REJECT_CROWDSAP"), record("n495", "ERROR"), record("n496", "SKIP_NO_DATA"),
                 self.null(497, "REVIEW_VETTING_HOLD", top_sde=6.2, peaks=[evidence(1, "REVIEW", [("ARCHIVAL", "HOLD")])],
                           arch=[arch_diag(1, 2, status="REVIEW_ARCHIVAL_DEPTH_BAND: x")]),
                 self.null(498), self.null(499)]
        return recs

    def test_rates_a_to_d_and_denominators(self):
        recs = self.cohort()
        self.assertEqual(len(recs), 500)
        m = E.m4_false_alarm(recs, {r["case_id"]: 0.95 for r in recs}, "N1")
        self.assertEqual((m["n_planned"], m["n_evaluated"], m["n_failed_or_unavailable"]), (500, 498, 2))
        p = m["pooled"]
        # (a) top peak SDE >= 6.0: 6 + 4 + 2 candidates + 2 (incomplete, capped review) + 1 hold = 15; >= 7.5: 4 + 1.
        self.assertEqual((p["a_top_peak_sde_6"]["k"], p["a_top_peak_sde_7_5"]["k"], p["n"]), (15, 5, 498))
        self.assertEqual(p["b_reached_archival"]["k"], 7)                   # 4 archival rejects, 2 candidates, 1 hold
        self.assertEqual((p["c_candidate"]["k"], p["c_candidate_multi_epoch"]["k"], p["c_candidate_single_sector"]["k"]), (2, 1, 1))
        self.assertEqual((p["c_candidate_sde_6"]["k"], p["c_candidate_sde_7_5"]["k"]), (2, 1))
        self.assertEqual(p["dfar_grey"]["candidate_level"]["k"], 1)         # the CANDIDATE with SDE 6.9
        self.assertEqual(p["dfar_grey"]["search_level"]["k"], 10)
        d = p["d_mode2"]
        self.assertEqual((d["per_trigger"]["k"], d["per_trigger"]["n"], d["holds"]), (1, 6, 1))    # 4 rejects, 1 pass, 1 hold
        self.assertEqual(p["c_candidate"]["interval"], "Clopper-Pearson 95 % upper")
        self.assertAlmostEqual(p["c_candidate"]["hi"], stats.cp_upper(2, 498))
        self.assertAlmostEqual(p["c_candidate"]["worst_case"]["hi"], stats.cp_upper(4, 500))       # failed runs as alarms
        # Items 37.1 and 38.4: incomplete and capped-review hosts stay in the denominator, on their own lines.
        self.assertEqual(m["incomplete_lines"], {"INCOMPLETE_TPF_UNAVAILABLE": 1})
        self.assertEqual(m["review_subrows"]["REVIEW_SECONDARY_NOT_EVALUATED"], 1)
        self.assertEqual(sum(x["n"] for x in m["lines"]), 500)
        self.assertEqual([c["kind"] for c in m["candidates"]], ["Single-Sector_Only", "Multi-Epoch"])

    def test_failed_runs_enter_each_stratum_worst_case(self):
        recs = [self.null(0), record("n1", "ERROR"), record("n2", "SKIP_NO_DATA"), self.null(3)]
        result = E.m4_false_alarm(recs, {"n0": 0.8, "n1": 0.8, "n2": 0.95, "n3": 0.6}, "N1")
        strata = result["by_crowding"]
        for crowd, k, n in ((0.8, 1, 2), (0.95, 1, 1), (0.6, 0, 1)):
            rate = strata[E.crowd_stratum(crowd)]["c_candidate"]["worst_case"]
            self.assertEqual((rate["k"], rate["n"]), (k, n))
            self.assertAlmostEqual(rate["hi"], stats.cp_upper(k, n))
        self.assertEqual(result["pooled"]["c_candidate"]["worst_case"]["n"], 4)

    def test_zero_and_two_alarms_at_500(self):
        recs = [self.null(i) for i in range(500)]
        m = E.m4_false_alarm(recs, {}, "N1")["pooled"]
        self.assertAlmostEqual(m["c_candidate"]["hi"], 0.00597, places=5)   # item 31.7: 0.60 %
        self.assertEqual(m["d_mode2"]["per_trigger"]["status"], "NOT_AVAILABLE_EMPTY_DENOMINATOR")
        self.assertEqual(E.m4_false_alarm(recs, {}, "N1")["by_crowding"]["CROWDSAP unknown"]["n"], 500)


class M5AndDiagnosticsTests(unittest.TestCase):
    def test_parameter_errors_and_coverage(self):
        truth = dict(period=3.0, t0=2001.0, depth=0.001, k=0.03)
        cases = []
        for i, (dp_sigma, dt_sigma) in enumerate([(0.5, 0.2), (1.5, 0.9), (2.5, 1.1), (3.5, 2.5)]):
            c = cand()
            c["period_days"] = 3.0 + dp_sigma * 0.001                       # sigma_P = 86.4 s = 0.001 d
            c["t0_btjd"] = 2001.0 + 3.0 * 7 + dt_sigma * 0.001              # seven cycles later: wrapped to the truth epoch
            cases.append(dict(case_id=f"m{i}", candidate=c, truth=truth, r_star=1.0))
        nosig = cand()
        nosig["cramer_rao_sigma_p_sec"] = float("nan")
        nosig["cramer_rao_sigma_t0_sec"] = None
        cases.append(dict(case_id="m4", candidate=nosig, truth=truth, r_star=1.0))
        m = E.m5_parameters(cases)
        cov = m["period"]["coverage"]
        self.assertEqual((cov["n"], cov["n_with_sigma"], cov["not_available"]), (5, 4, 1))           # a missing sigma is not invented
        self.assertEqual([cov["levels"][f"{k} sigma"]["k"] for k in (1, 2, 3)], [1, 2, 3])
        self.assertEqual([m["t0"]["coverage"]["levels"][f"{k} sigma"]["k"] for k in (1, 2, 3)], [2, 3, 4])
        self.assertEqual(cov["levels"]["1 sigma"]["nominal"], 0.6827)
        self.assertAlmostEqual(m["rows"][1]["t0_error_d"], 0.0009, places=9)
        self.assertEqual(m["depth"]["coverage"], "NOT_AVAILABLE (no production interval)")
        self.assertAlmostEqual(m["depth"]["ratio"]["median"], 1.0)
        self.assertAlmostEqual(m["radius"]["ratio"]["median"], 3.0 / (0.03 * 109.076))
        self.assertEqual(list(m["by_ephemeris_source"]), ["DISCOVERY_SINGLE_SECTOR"])

    def test_item_31_4_diagnostics(self):
        ald = prodpath.load_production()

        def s74(flip, near, bls_above, p_ref=3.0005, status="COMPLETE"):
            return dict(status=status, ref_sigma_flip=flip, step_flip=flip, near_threshold=near, p_ref_d=p_ref,
                        sigma_p_disc_projected_d=1e-4, sigma_p_ref=dict(status="DEFINED", sigma_p_ref_d=1e-4),
                        sci78=dict(bls_above_n_sigma=bls_above))
        recs = [record("d0", "CANDIDATE", journal=[jrow(1, "CANDIDATE", period=3.0)], arch=[arch_diag(1, 2, sci74=s74(True, True, True))]),
                record("d1", "CANDIDATE", journal=[jrow(1, "CANDIDATE", period=3.0)], arch=[arch_diag(1, 2, sci74=s74(False, False, False, p_ref=3.0001))]),
                record("d2", "REJECT_ARCHIVAL", journal=[jrow(1, "REJECT", period=3.0)],
                       arch=[arch_diag(1, 2, sci74=s74(None, True, None, status="PARTIAL_SIGMA_P_REF_NOT_DEFINED")), arch_diag(2, 1)])]
        d = E.archival_secondary_diagnostics(ald, recs, p_cat_by_case=dict(d0=3.0, d1=3.0), sigma_p_cat_by_case=dict(d0=0.0, d1=0.0))
        self.assertEqual(d["n_evaluations"], 3)                             # the Mode 1 row without sci74 is not an evaluation
        f = d["archival_sec_flip_rate"]
        self.assertEqual((f["k"], f["n"], f["not_available"]), (1, 2, 1))   # the undefined flip is listed, not counted
        self.assertEqual((d["near_threshold"]["k"], d["near_threshold"]["n"]), (2, 3))
        self.assertEqual((d["refinement_above_3_5_sigma_from_p_bls"]["k"], d["refinement_above_3_5_sigma_from_p_bls"]["n"]), (1, 2))
        cat = d["refinement_above_3_5_sigma_from_p_cat"]
        self.assertEqual((cat["k"], cat["n"]), (1, 2))                      # 5 sigma and 1 sigma from the catalog period
        self.assertAlmostEqual(d["rows"][0]["cat_offset_ref_sigma"], 5.0, places=6)
        self.assertAlmostEqual(d["rows"][1]["cat_offset_disc_sigma"], 1.0, places=6)

    def test_mode2_triggers(self):
        rec = record("t", "REVIEW_VETTING_HOLD", arch=[arch_diag(1, 2), arch_diag(2, 2, status="REVIEW_ARCHIVAL_NONDETECTION: x"),
                                                         arch_diag(3, 1), arch_diag(4, 2, ok=None, status="archival evaluation failed")])
        t = E.mode2_triggers(rec)
        self.assertEqual([(x["peak_rank"], x["passed"], x["hold"], x["not_evaluated"]) for x in t],
                         [(1, True, False, False), (2, False, True, False), (4, False, False, True)])


if __name__ == "__main__":
    unittest.main()
