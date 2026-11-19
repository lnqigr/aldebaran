"""S3.2 (roadmap C8, SCI-69, ARGON_DECISIONS item 31.1): Cp is the only Pleiades EB decision.

The old Pleiades single-maximum scan stays as a diagnostic: its raw depth, depth ratio and
flag are persisted but decide nothing. Every successful record carries
sec_stage_disagreement (discovery-stage decision against the Pleiades Cp decision).

The negative control is a circular secondary of 170 ppm under a 1200 ppm primary. The old
rule needs more than 220 ppm and calls it not EB; the Cp rule (above 100 ppm, at least 0.10 of
the primary, Sidak z_crit_circ) calls it EB. The two rules are nested (an old-rule EB is
always a Cp EB on the same inputs, SEC-DIVERGE-01: 0 of 1315 peaks the other way), so the
opposite direction is not constructible; a shared planet case pins agreement instead.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import pleiades
import aldebaran
import test_char_pleiades_deep_vet as char

PRIMARY_PPM = 1200.0
SECONDARY_PPM = 170.0


def disagreement_sectors():
    t1, f1 = char.synth_sector(21, 2000.0, depth=PRIMARY_PPM * 1e-6, sec_depth=SECONDARY_PPM * 1e-6)
    t2, f2 = char.synth_sector(22, 2014.0, depth=PRIMARY_PPM * 1e-6, sec_depth=SECONDARY_PPM * 1e-6)
    return char.FakeSearch([char.FakeItem(5, char.FakeLC(t1, f1)), char.FakeItem(6, char.FakeLC(t2, f2))])


def stitched_arrays(sec_ppm):
    t1, f1 = char.synth_sector(21, 2000.0, depth=PRIMARY_PPM * 1e-6, sec_depth=sec_ppm * 1e-6)
    t2, f2 = char.synth_sector(22, 2014.0, depth=PRIMARY_PPM * 1e-6, sec_depth=sec_ppm * 1e-6)
    t = np.concatenate([t1, t2])
    f = np.concatenate([f1, f2]) / 1000.0
    return t, f


class TestCpIsTheOnlyRule(unittest.TestCase):
    def test_pleiades_calls_the_detector_cp_function(self):
        self.assertIs(pleiades.cp_secondary_scan(), aldebaran._scan_full_orbit_secondary_eclipse_cp)

    def test_rules_disagree_on_the_fixture_arrays(self):
        t, f = stitched_arrays(SECONDARY_PPM)
        args = (t, f, char.PERIOD, char.T0, char.DUR_D, PRIMARY_PPM)
        raw = pleiades.scan_full_orbit_secondary_eclipse(*args)
        cp = pleiades.cp_secondary_scan()(*args)
        self.assertFalse(raw[0])
        self.assertTrue(cp[0])
        self.assertLess(raw[2], 220.0)
        self.assertGreater(cp[2], 100.0)
        self.assertEqual(cp[3], 0.50)

    def test_rules_agree_without_secondary(self):
        t, f = stitched_arrays(0.0)
        args = (t, f, char.PERIOD, char.T0, char.DUR_D, PRIMARY_PPM)
        self.assertFalse(pleiades.scan_full_orbit_secondary_eclipse(*args)[0])
        self.assertFalse(pleiades.cp_secondary_scan()(*args)[0])


class TestDiscoveryDecision(unittest.TestCase):
    def test_explicit_field(self):
        for value, expected in ((True, True), (np.bool_(False), False), ("True", True), (" false ", False)):
            with self.subTest(value=value):
                self.assertEqual(pleiades.discovery_secondary_decision({"discovery_sec_cp_eb": value}),
                                 (expected, "row_field_discovery_sec_cp_eb"))

    def test_cp_path_row_proves_not_eb(self):
        self.assertEqual(pleiades.discovery_secondary_decision({"odd_even_method": "ODD_EVEN_V2_EXPERIMENTAL"}),
                         (False, "cp_path_report_candidate"))

    def test_explicit_field_wins_over_marker(self):
        row = {"discovery_sec_cp_eb": "True", "odd_even_method": "ODD_EVEN_V2_EXPERIMENTAL"}
        self.assertEqual(pleiades.discovery_secondary_decision(row), (True, "row_field_discovery_sec_cp_eb"))

    def test_unknown_for_legacy_or_unproven_rows(self):
        # Codex review round 1 (P2): passing the legacy gate does not prove passing Cp.
        for row in ({}, {"discovery_sec_cp_eb": float("nan")}, {"discovery_sec_cp_eb": ""},
                    {"discovery_sec_cp_eb": None}, {"odd_even_method": float("nan")},
                    {"odd_even_method": "legacy"}):
            with self.subTest(row=row):
                self.assertEqual(pleiades.discovery_secondary_decision(row),
                                 (None, "unknown_discovery_cp_decision"))


class TestDeepVetDecision(unittest.TestCase):
    def test_cp_flags_eb_the_old_rule_missed(self):
        cand = char.candidate(depth_ppm=PRIMARY_PPM, odd_even_method="ODD_EVEN_V2_EXPERIMENTAL")
        result, _, _ = char.run_vet(disagreement_sectors(), cand)
        self.assertTrue(result["sec_cp_is_eb"])
        self.assertFalse(result["sec_raw_rule_is_eb"])
        self.assertLess(result["sec_raw_depth_ppm"], 220.0)
        self.assertAlmostEqual(result["sec_raw_depth_ratio"], result["sec_raw_depth_ppm"] / PRIMARY_PPM, places=3)
        self.assertEqual(result["sec_eb_decision_rule"], "cp_full_orbit_item_31.1")
        self.assertTrue(result["disposition"].startswith("SECONDARY-ECLIPSE FLAG: possible circular EB (Phase 0.50,"))
        ledger = char.json.loads(result["vps_ledger_json"])
        self.assertEqual(ledger["components"]["secondary_eb"], 65.0)
        self.assertFalse(result["sec_discovery_is_eb"])
        self.assertEqual(result["sec_discovery_decision_source"], "cp_path_report_candidate")
        self.assertTrue(result["sec_stage_disagreement"])

    def test_legacy_row_gives_unknown_disagreement(self):
        # Same arrays, legacy provenance: the EB decision is still Cp, but the stage
        # disagreement is unknown, not a rule change reported as a stage difference.
        result, _, _ = char.run_vet(disagreement_sectors(), char.candidate(depth_ppm=PRIMARY_PPM))
        self.assertTrue(result["sec_cp_is_eb"])
        self.assertIsNone(result["sec_discovery_is_eb"])
        self.assertEqual(result["sec_discovery_decision_source"], "unknown_discovery_cp_decision")
        self.assertIsNone(result["sec_stage_disagreement"])

    def test_explicit_discovery_eb_agrees(self):
        cand = char.candidate(depth_ppm=PRIMARY_PPM, discovery_sec_cp_eb=True)
        result, _, _ = char.run_vet(disagreement_sectors(), cand)
        self.assertTrue(result["sec_cp_is_eb"])
        self.assertFalse(result["sec_stage_disagreement"])

    def test_planet_has_no_disagreement_and_all_fields(self):
        result, _, _ = char.run_vet(char.two_sectors(), char.candidate(odd_even_method="ODD_EVEN_V2_EXPERIMENTAL"))
        self.assertFalse(result["sec_cp_is_eb"])
        self.assertFalse(result["sec_raw_rule_is_eb"])
        self.assertFalse(result["sec_stage_disagreement"])
        for key in pleiades.SEC_DIAGNOSTIC_FIELDS:
            self.assertIn(key, result)


class TestCliRows(unittest.TestCase):
    def test_error_row_carries_empty_secondary_fields(self):
        import ast, contextlib, io, tempfile
        import pandas as pd
        with open(pleiades.__file__, "r", encoding="utf-8-sig") as fh:
            tree = ast.parse(fh.read())
        block = next(x for x in tree.body if isinstance(x, ast.If) and isinstance(x.test, ast.Compare)
                     and isinstance(x.test.left, ast.Name) and x.test.left.id == "__name__")

        def fail(*args, **kwargs):
            raise RuntimeError("deliberate failure")

        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "in.csv")
            out = os.path.join(tmp, "out")
            pd.DataFrame([{"tic_id": "12345", "period_days": 2.5, "t0_btjd": 4000.0,
                           "duration_hours": 1.5, "depth_ppm": 500.0}]).to_csv(inp, index=False)
            old_argv = sys.argv
            sys.argv = ["pleiades.py", "--input-csv", inp, "--output-dir", out]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.Module(body=[block], type_ignores=[]), "pleiades.py:main", "exec"),
                         {"__name__": "__main__", "os": os, "sys": sys, "pd": pd, "INPUT_CSV": inp,
                          "VET_DIR": out, "PIPELINE_VERSION": "s3_2", "deep_vet_target": fail})
            finally:
                sys.argv = old_argv
            row = pd.read_csv(os.path.join(out, "scientific_dossier_summary.csv"), dtype={"tic_id": str}).iloc[0]
        self.assertEqual(row["operational_status"], "ERROR")
        for key in pleiades.SEC_DIAGNOSTIC_FIELDS:
            with self.subTest(field=key):
                self.assertTrue(pd.isna(row[key]))


if __name__ == "__main__":
    unittest.main()
