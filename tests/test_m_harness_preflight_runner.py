"""S4-M-PACKAGE-01, parts B5, B6 and D: the C14 preflight over cohort metadata (AGENTS.md "Preflight before long
runs"), the bound resume of the long runner (S3.6 run_identity), and the end-to-end dry run on synthetic fixtures."""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ALDEBARAN_OFFLINE", "1")
import run_identity as ri  # noqa: E402
from m_harness import fixtures, preflight, prodpath, runner  # noqa: E402
from m_harness import preflight_core as core  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(REPO, "aldebaran.py")
ACCEPTED_CORE_SHA256 = "496620ec512c027cb52c7eff39184cdb80593731dc52bab29d2f2f00f8e03e75"


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mh-pre-")
        cls.root = os.path.join(cls.tmp, "hosts")
        cls.dev = [fixtures.synthetic_host(cls.root, 900300 + i, 40, archival_sectors=(12,), seed=30 + i, sigma=1e-3) for i in range(2)]
        cls.held = [fixtures.synthetic_host(cls.root, 900310 + i, 41, archival_sectors=(12,), seed=40 + i, sigma=1e-3) for i in range(2)]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cohorts(self, dev=None, held=None):
        return dict(development=dict(blinded=False, root=self.root, hosts=copy.deepcopy(self.dev if dev is None else dev)),
                    heldout=dict(blinded=True, root=self.root, hosts=copy.deepcopy(self.held if held is None else held)))

    def run_pre(self, cohorts, name, **kw):
        return preflight.run(source=SOURCE, cohorts=cohorts, out=os.path.join(self.tmp, name + ".json"), **kw)

    def test_blinded_inputs_are_not_hashed(self):
        held_paths = {str(preflight.Path(self.root) / r["path"]) for h in self.held for r in h["lightcurves"]}
        held_paths.update(str(preflight.Path(self.root) / h["tpf_fits"]) for h in self.held)
        real_hash = core.sha256
        def guarded_hash(path):
            self.assertNotIn(str(path), held_paths)
            return real_hash(path)
        with patch.object(core, "sha256", guarded_hash):
            result = self.run_pre(self.cohorts(), "no_blinded_hash")
        self.assertTrue(result["passed"])
        for host in result["cohorts"]["heldout"]["hosts"]:
            for entry in host["inputs"]:
                self.assertNotIn("sha256", entry)
                self.assertNotIn("match", entry)
                self.assertIn("NOT_RUN", entry["verification"])

    def test_gates_are_read_from_the_detector_source(self):
        inv = core.gate_inventory(SOURCE)
        self.assertEqual([g["status"] for g in inv["gates"]],
                         ["SKIP_NO_DATA", "REJECT_CROWDSAP", "REJECT_TOI_MASK_TOO_LARGE", "REJECT_BASELINE",
                          "REJECT_NO_STELLAR_PARAMS", "REJECT_EVOLVED_STAR", "REJECT_RUWE"])
        cond = {g["status"]: g["condition"] for g in inv["gates"]}
        self.assertEqual(cond["REJECT_CROWDSAP"], "(crowdsap < 0.7)")
        self.assertEqual(cond["REJECT_EVOLVED_STAR"], "(r_star > 1.75 or teff > 9500)")
        self.assertEqual(cond["REJECT_RUWE"], "(np.isfinite(ruwe) and ruwe > 1.4)")
        self.assertEqual((inv["defaults"]["min_period"], inv["defaults"]["max_period"]), (0.8, 13.0))
        with open(SOURCE, encoding="utf-8") as fh:                          # the baseline factor the manifest check uses
            self.assertIn("max_p = min(max_period, (t_bls[-1] - t_bls[0]) * 0.48)", fh.read())
        self.assertEqual(preflight.SEARCH_BASELINE_FACTOR, 0.48)

    def test_c7_gates_are_read_from_every_detector_stage(self):
        """RC2-FREEZE-01 C7: every function that takes the host state is a listed stage, and a new host-level gate in
        a per-peak stage appears in the inventory, in the loop (negative control for the stage reading)."""
        import ast
        with open(SOURCE, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        stages = core.detector_stages(tree)
        takes_host = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.args.args
                      and n.args.args[0].arg == "host"}
        self.assertTrue(takes_host)
        self.assertLessEqual(takes_host, set(stages))
        marker = '    """Filter 6: pixel-level TPF localization."""\n'
        self.assertEqual(src.count(marker), 1)
        alt = os.path.join(self.tmp, "c7_alt_aldebaran.py")
        with open(alt, "w", encoding="utf-8") as fh:
            fh.write(src.replace(marker, marker + '    if peak.best_p > 99:\n        return host.end_host(peak, "REJECT_NEW_GATE", "new")\n'))
        gates = {g["status"]: g for g in core.gate_inventory(alt)["gates"]}
        self.assertEqual(len(gates), 8)
        self.assertEqual(gates["REJECT_NEW_GATE"]["condition"], "(peak.best_p > 99)")
        self.assertTrue(gates["REJECT_NEW_GATE"]["in_loop"])

    def test_valid_cohorts_pass_and_only_the_result_file_is_written(self):
        before = sorted(os.listdir(self.tmp))
        tree_before = sorted(f for _, _, fs in os.walk(self.root) for f in fs)
        res = self.run_pre(self.cohorts(), "valid")
        self.assertTrue(res["passed"], [h["failures"] for c in res["cohorts"].values() for h in c["hosts"]])
        self.assertEqual(sorted(os.listdir(self.tmp)), sorted(before + ["valid.json"]))
        self.assertEqual(sorted(f for _, _, fs in os.walk(self.root) for f in fs), tree_before)
        self.assertEqual(preflight.passing_hosts(res, "development"), [h["tic_id"] for h in self.dev])
        host = res["cohorts"]["development"]["hosts"][0]
        self.assertEqual({g: v["result"] for g, v in host["gates"].items()}, {s: "PASS" for s in host["gates"]})
        self.assertEqual(len(host["inputs"]), 3)                            # two light curves and the target pixel file
        self.assertTrue(all(i["present"] and i["match"] and len(i["sha256"]) == 64 for i in host["inputs"]))
        self.assertEqual(host["counts"]["tpf_covers_valid_lc_cadences"]["ok"], True)
        self.assertLess(res["m_preflight"]["elapsed_s"], 60.0)
        with self.assertRaises(FileExistsError):                            # a result file is never overwritten
            self.run_pre(self.cohorts(), "valid")

    def test_blinded_cohort_is_checked_from_metadata_only(self):
        opened = []
        real_open = fits.open

        def spy(path, *a, **k):
            opened.append(os.path.basename(str(path)))
            return real_open(path, *a, **k)
        with patch.object(preflight.fits, "open", spy):
            res = self.run_pre(self.cohorts(), "blinded", draw=lambda c, h: {"F": dict(drawn=True, attempts=1)})
        self.assertTrue(res["passed"])
        held_files = {os.path.basename(r["path"]) for h in self.held for r in h["lightcurves"]} | \
                     {os.path.basename(h["tpf_fits"]) for h in self.held}
        self.assertTrue(opened)                                             # the development files were opened ...
        self.assertFalse(held_files & set(opened))                          # ... no held-out file was
        self.assertEqual(res["m_preflight"]["blinded_files_opened"], [])
        for h in res["cohorts"]["heldout"]["hosts"]:
            self.assertEqual((h["data_checks"], h["draw"]), ("NOT_RUN (blinded)", "NOT_RUN (blinded)"))
            self.assertEqual(len(h["inputs"]), 3)                           # presence and declared SHA256 are checked
            self.assertIn("search_period_range_days", h["ranges"])          # and so are the metadata ranges
            self.assertNotIn("valid_cadences_in_file", h["counts"])
        self.assertEqual(res["cohorts"]["development"]["hosts"][0]["data_checks"], "RUN")

    def test_reader_refuses_a_blinded_cohort_by_itself(self):
        mp = preflight.MPreflight(self.cohorts())
        with self.assertRaises(preflight.BlindedCohortRead):
            mp.data_checks("heldout", self.held[0]["tic_id"])
        self.assertEqual(mp.opened, [])
        self.assertIn("valid_cadences_in_file", mp.data_checks("development", self.dev[0]["tic_id"])["counts"])

    def test_gate_rejects_and_missing_metadata_stop_the_preflight(self):
        for field, value, gate in (("crowdsap", 0.65, "REJECT_CROWDSAP"), ("r_star", 1.9, "REJECT_EVOLVED_STAR"),
                                   ("teff", 9600.0, "REJECT_EVOLVED_STAR"), ("ruwe", 1.5, "REJECT_RUWE"),
                                   ("has_params", False, "REJECT_NO_STELLAR_PARAMS")):
            held = copy.deepcopy(self.held)
            if field == "crowdsap":
                held[1][field] = value
            else:
                held[1]["stellar"][field] = value
            with self.subTest(gate=gate, field=field):
                res = self.run_pre(self.cohorts(held=held), f"reject_{field}")
                self.assertFalse(res["passed"])
                self.assertEqual(res["cohorts"]["heldout"]["gate_rejects"][gate], [held[1]["tic_id"]])
                self.assertEqual(preflight.passing_hosts(res, "heldout"), [held[0]["tic_id"]])     # control hosts come from these
        allowed = copy.deepcopy(self.held)
        allowed[1]["crowdsap"] = 0.65
        self.assertTrue(self.run_pre(self.cohorts(held=allowed), "reject_allowed", allow_gate_rejects=("heldout",))["passed"])
        for drop in ("ruwe", "teff"):
            held = copy.deepcopy(self.held)
            del held[0]["stellar"][drop]
            with self.subTest(missing=drop):
                res = self.run_pre(self.cohorts(held=held), f"missing_{drop}")
                self.assertFalse(res["passed"])                             # a missing value is never read as a pass
                gate = "REJECT_RUWE" if drop == "ruwe" else "REJECT_EVOLVED_STAR"
                self.assertEqual(res["cohorts"]["heldout"]["hosts"][0]["gates"][gate]["result"], "NOT_EVALUATED")
        held = copy.deepcopy(self.held)
        held[0]["known_transits"] = [dict(period=3.0, t0=2000.0, duration=0.1)]            # a known-signal host without its mask fraction
        self.assertFalse(self.run_pre(self.cohorts(held=held), "missing_mask")["passed"])

    def test_ranges_counts_and_inputs_stop_the_preflight(self):
        held = copy.deepcopy(self.held)
        held[0]["discovery_baseline_days"] = 1.5                            # max_p = 0.72 d: below the 0.8 d minimum period
        res = self.run_pre(self.cohorts(held=held), "short_baseline")
        self.assertFalse(res["passed"])
        self.assertEqual(res["cohorts"]["heldout"]["gate_rejects"]["REJECT_BASELINE"], [held[0]["tic_id"]])
        self.assertFalse(res["cohorts"]["heldout"]["hosts"][0]["ranges"]["search_period_range_days"]["ok"])
        held = copy.deepcopy(self.held)
        held[0]["n_valid_cadences"] = 499
        self.assertFalse(self.run_pre(self.cohorts(held=held), "few_cadences")["passed"])
        held = copy.deepcopy(self.held)
        del held[0]["n_valid_cadences"]
        self.assertFalse(self.run_pre(self.cohorts(held=held), "no_cadence_count")["passed"])
        held = copy.deepcopy(self.held)
        held[0]["lightcurves"][0]["sha256"] = "invalid"
        res = self.run_pre(self.cohorts(held=held), "bad_sha")
        self.assertFalse(res["passed"])
        self.assertTrue(any("SHA256 declaration missing or invalid" in f for f in res["cohorts"]["heldout"]["hosts"][0]["failures"]))
        held = copy.deepcopy(self.held)
        held[0]["tpf_fits"] = "missing_tp.fits"
        self.assertFalse(self.run_pre(self.cohorts(held=held), "missing_input")["passed"])
        dev = copy.deepcopy(self.dev)
        dev[0]["n_valid_cadences"] += 7                                     # the manifest disagrees with the file
        res = self.run_pre(self.cohorts(dev=dev), "manifest_mismatch")
        self.assertFalse(res["passed"])
        self.assertFalse(res["cohorts"]["development"]["hosts"][0]["counts"]["manifest_cadence_count_matches_file"]["ok"])
        self.assertFalse(self.run_pre(self.cohorts(), "bad_source_sha", expected_source_sha256="0" * 64)["passed"])
        res = self.run_pre(self.cohorts(), "undrawn", draw=lambda c, h: {"[4, 6)": dict(drawn=False, attempts=50)}, max_attempts=50)
        self.assertFalse(res["passed"])                                     # a planned family that cannot be drawn


def null_jobs(ald, root, hosts):
    jobs = []
    for i, h in enumerate(hosts):
        case = dict(tic_id=h["tic_id"], case_id=f"job{i}", sector=h["sector"], source_fits=h["source_fits"],
                    source_sha256=h["source_sha256"], tpf_fits=h["tpf_fits"], tpf_sha256=h["tpf_sha256"], stellar=h["stellar"],
                    known_transits=[], lightcurves=[dict(r) for r in h["lightcurves"]])
        jobs.append(dict(case=case, root=root, arm="N2"))
    return jobs


class BoundResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mh-run-")
        cls.ald = prodpath.load_production()
        cls.root = os.path.join(cls.tmp, "hosts")
        cls.hosts = [fixtures.synthetic_host(cls.root, 900400 + i, 40, seed=50 + i, sigma=1e-3) for i in range(3)]
        cls.prereg = os.path.join(cls.tmp, "PREREG.md")
        with open(cls.prereg, "w", encoding="utf-8") as fh:
            fh.write("frozen preregistration text\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def go(self, name, jobs=None, config=None, on_case=None, prereg=None):
        return runner.run_cases(self.ald, run_dir=os.path.join(self.tmp, name), run_id=name, jobs=jobs or null_jobs(self.ald, self.root, self.hosts),
                                config=config or dict(purpose="test"), repo_dir=REPO, preregistration_path=prereg or self.prereg, on_case=on_case)

    def interrupt_after(self, n):
        def on_case(cid, done):
            if done == n:
                raise KeyboardInterrupt("simulated kill")
        return on_case

    def test_interrupted_run_resumes_without_rerunning_finished_cases(self):
        with self.assertRaises(KeyboardInterrupt):
            self.go("resume", on_case=self.interrupt_after(1))
        run = json.load(open(os.path.join(self.tmp, "resume", "run_identity.json"), encoding="utf-8"))
        self.assertEqual((run["status"], len(run["interruptions"]), run["interruptions"][0]["completed_count"]), ("INTERRUPTED", 1, 1))
        first = os.path.join(self.tmp, "resume", "records", "job0.json")
        stamp = os.path.getmtime(first)
        res = self.go("resume")
        self.assertEqual((res["reused"], res["ran"]), (["job0"], ["job1", "job2"]))
        self.assertEqual(os.path.getmtime(first), stamp)                    # the finished record was not rewritten
        self.assertEqual((res["run"]["status"], len(res["run"]["resumes"]), res["run"]["resumes"][0]["previous_status"]), ("COMPLETE", 1, "INTERRUPTED"))
        self.assertEqual(set(res["run"]["identity"]), {"code_sha256", "config_sha256", "input_sha256"})
        self.assertIn("m_harness/routing.py", res["run"]["identity"]["code_sha256"])
        self.assertIn("aldebaran.py", res["run"]["identity"]["code_sha256"])
        for cid, item in res["records"].items():
            self.assertEqual(item["identity"], res["run"]["identity"])      # every case record carries the run identity
            self.assertTrue(item["routing_check"]["passed"])
        again = self.go("resume")                                           # a third start reuses everything
        self.assertEqual((len(again["reused"]), again["ran"]), (3, []))

    def test_partial_record_is_never_reused(self):
        self.go("partial")
        rec = os.path.join(self.tmp, "partial", "records", "job1.json")
        text = open(rec, encoding="utf-8").read()
        with open(rec, "w", encoding="utf-8") as fh:
            fh.write(text[: len(text) // 2])                                # a truncated record
        with self.assertRaisesRegex(ri.RunIdentityError, "PARTIAL_OR_CORRUPT_RECORD"):
            self.go("partial")
        with open(rec, "w", encoding="utf-8") as fh:                        # complete JSON, but a required field is gone
            d = json.loads(text)
            del d["record"]
            json.dump(d, fh)
        with self.assertRaisesRegex(ri.RunIdentityError, "PARTIAL_OR_CORRUPT_RECORD"):
            self.go("partial")

    def test_changed_artifact_of_a_finished_case_is_refused(self):
        self.go("artifact")
        item = json.load(open(os.path.join(self.tmp, "artifact", "records", "job0.json"), encoding="utf-8"))
        np.save(item["times_path"], np.arange(5.0))                         # the saved search timestamps were altered
        with self.assertRaisesRegex(ri.RunIdentityError, "SHA256 mismatch"):
            self.go("artifact")

    def test_resume_is_bound_to_configuration_preregistration_and_inputs(self):
        root = os.path.join(self.tmp, "hosts_bound")
        shutil.copytree(self.root, root)
        jobs = lambda: null_jobs(self.ald, root, self.hosts)
        self.go("bound", jobs=jobs())
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_CONFIG_SHA256"):
            self.go("bound", jobs=jobs(), config=dict(purpose="another design"))
        other = os.path.join(self.tmp, "PREREG_changed.md")
        with open(other, "w", encoding="utf-8") as fh:
            fh.write("frozen preregistration text, edited after the freeze\n")
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_CONFIG_SHA256"):
            self.go("bound", jobs=jobs(), prereg=other)
        target = os.path.join(root, self.hosts[2]["source_fits"])           # an input file changes on disk after the run
        saved = open(target, "rb").read()
        with open(target, "wb") as fh:
            fh.write(saved[:-1] + bytes([saved[-1] ^ 1]))
        with self.assertRaisesRegex(ri.RunIdentityError, "STALE_INPUT_SHA256"):
            self.go("bound", jobs=jobs())
        with open(target, "wb") as fh:
            fh.write(saved)
        self.assertEqual(len(self.go("bound", jobs=jobs())["reused"]), 3)   # the unchanged run still resumes

    def test_output_of_an_interrupted_case_is_cleared_and_logged(self):
        with self.assertRaises(KeyboardInterrupt):
            self.go("cleared", on_case=self.interrupt_after(1))
        det = os.path.join(self.tmp, "cleared", "detector", "job1")
        os.makedirs(det)
        with open(os.path.join(det, "experimental_v2_vetting.jsonl"), "w", encoding="utf-8") as fh:
            fh.write('{"partial": true}\n')                                 # what a killed detector run leaves behind
        res = self.go("cleared")
        self.assertEqual(res["run"]["partial_outputs_cleared"], ["job1"])
        lines = open(os.path.join(det, "experimental_v2_vetting.jsonl"), encoding="utf-8").read().splitlines()
        self.assertNotIn('{"partial": true}', lines)
        self.assertTrue(res["records"]["job1"]["routing_check"]["passed"])

    def test_missing_preregistration_is_refused_before_run_creation(self):
        for prereg in (None, "", os.path.join(self.tmp, "absent.md")):
            target = os.path.join(self.tmp, "no_prereg")
            with self.assertRaisesRegex(ValueError, "preregistration file is required"):
                runner.run_cases(self.ald, run_dir=target, run_id="missing", jobs=[], config={},
                                 repo_dir=REPO, preregistration_path=prereg)
            self.assertFalse(os.path.exists(target))

    def test_duplicate_case_ids_are_refused(self):
        jobs = null_jobs(self.ald, self.root, self.hosts)
        jobs[1]["case"]["case_id"] = "job0"
        with self.assertRaises(ValueError):
            self.go("dup", jobs=jobs)


class DryRunTests(unittest.TestCase):
    def test_end_to_end_on_a_small_synthetic_cohort(self):
        tmp = tempfile.mkdtemp(prefix="mh-dry-")
        self.addCleanup(shutil.rmtree, tmp, True)
        out = os.path.join(tmp, "dry")
        s = runner.dry_run(out, n_hosts=2, snr_bins=((8.0, 10.0), (25.0, 35.0)), repo_dir=REPO)
        self.assertEqual(s["status"], "COMPLETE")
        self.assertIn("ENGINEERING CHECK", s["label"])
        self.assertIn("Not a measurement", open(os.path.join(out, "DRY_RUN_REPORT.md"), encoding="utf-8").read())
        st, t = s["steps"], s["tables"]
        self.assertTrue(st["preflight"]["passed"])
        self.assertEqual(st["preflight"]["m_preflight"]["blinded_files_opened"], [])
        self.assertEqual(st["preflight"]["cohorts"]["m2_known_planets"]["gate_rejects"], {"REJECT_CROWDSAP": ["920003"]})
        self.assertEqual((st["m1_injection"]["n_planned"], st["m1_injection"]["fidelity_failed"], st["m1_injection"]["model_check_failed"]), (4, [], []))
        self.assertEqual(st['m1_injection']['realized_grey_counts']['counts'], [0, 0, 1])
        self.assertFalse(st['m1_injection']['realized_grey_counts']['passed'])
        self.assertFalse(t['grey_zone_gain_candidate_level']['aggregation_equivalence_verified'])
        det = st["detector_runs"]
        self.assertEqual((det["routing_failed"], det["interruptions"], det["resumes"], det["n_reused"]), ([], 0, 0, 0))
        self.assertEqual((det["n_jobs"], det["n_ran"], det["run_status"]), (17, 17, "COMPLETE"))
        self.assertEqual(det["by_arm"], {"M1": 4, "M2": 4, "M3": 3, "N1": 2, "N1_AFTER_INJECTION": 2, "N2": 2})
        self.assertEqual(st["m4_noise_pre_gate"]["status"], "PASS")
        # Every table has its real denominator.
        self.assertEqual(t["m1_attrition"]["n_planned"], 4)
        self.assertEqual(sum(s_["lost"] for s_ in t["m1_attrition"]["stages"]) + t["m1_attrition"]["n_survivors"], 4)
        self.assertEqual([c["n"] for c in t["m1_completeness_candidate_level"]["by_snr_bin"].values()], [2, 2])
        self.assertEqual((t["m4_n1"]["n_planned"], t["m4_n2"]["n_planned"]), (2, 2))
        self.assertEqual(t["m2_false_rejection"]["n_cohort"], 4)
        self.assertEqual(t["m2_false_rejection"]["design_exclusion"]["out_of_domain_planets"], 1)
        self.assertEqual(t["m2_false_rejection"]["in_domain"]["n"], 3)
        self.assertEqual(t["m3_false_positive"]["n"], 3)
        self.assertEqual(t["development_signal_destruction"]["n_expected"], 2)
        # With only two planned nulls, even zero observed alarms cannot meet the 1.5 % upper-bound rule.
        self.assertEqual(t["grey_zone_decision"]["decision"], "FAIL")
        self.assertEqual(t["m5_parameters"]["n"], t["m1_attrition"]["n_survivors"])
        self.assertTrue(os.path.isfile(os.path.join(out, "preflight.json")))
        self.assertTrue(os.path.isfile(os.path.join(out, "run", "run_identity.json")))


if __name__ == "__main__":
    unittest.main()
