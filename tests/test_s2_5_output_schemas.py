"""S2.5 (IMPROVE-QUICK 001 and 002, plan 006 tests): one output schema per Pleiades and Antares table.

Pleiades: every scientific_dossier_summary.csv row (SUCCESS, deep ERROR, CLI NO_LIGHTCURVE and CLI ERROR)
and the empty file share PLEIADES_SUMMARY_COLS; CLI failure rows keep blank fit fields (S3.9 semantics);
a missing input file exits 1 instead of raising NameError.
Antares: the zero-survivor tables and the packaged tables share ANTARES_RANKING_COLS and
ANTARES_EXOFOP_COLS; a non-empty run writes no top-3 table and refuses a stale zero-survivor one.
Offline: the network is mocked at its boundary with the CHAR-TESTS-01 fakes.
"""
import ast
import contextlib
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import antares_packager as pack
import pleiades
import test_char_pleiades_deep_vet as vet

LEGACY_EMPTY_RANKING = ["rank", "tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm", "rp_earth",
                        "mp_ck17_earth", "rv_k_ck17_ms", "tsm", "esm", "snr_red", "corr_risk_percent",
                        "priority_score"]


def blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def cli_block():
    with open(pleiades.__file__, "r", encoding="utf-8-sig") as fh:
        tree = ast.parse(fh.read())
    block = next(x for x in tree.body if isinstance(x, ast.If) and isinstance(x.test, ast.Compare)
                 and isinstance(x.test.left, ast.Name) and x.test.left.id == "__name__")
    return compile(ast.Module(body=[block], type_ignores=[]), "pleiades.py:main", "exec")


def run_cli(rows, searches, catalog=None):
    """Run the pleiades.py CLI block over the real deep_vet_target; return the summary CSV (str cells)."""
    catalog = catalog or vet.Catalog()

    def search(query, **kwargs):
        return searches[query.split()[-1]]

    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in.csv"), os.path.join(tmp, "out")
        pd.DataFrame(rows).to_csv(inp, index=False)
        namespace = dict(vars(pleiades), __name__="__main__")
        old_argv = sys.argv
        sys.argv = ["pleiades.py", "--input-csv", inp, "--output-dir", out]
        try:
            with mock.patch.object(pleiades.lk, "search_lightcurve", side_effect=search), \
                 mock.patch.object(pleiades.Catalogs, "query_criteria", side_effect=catalog.tic), \
                 mock.patch.object(pleiades, "Vizier", side_effect=catalog.vizier), \
                 mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=catalog.fetch), \
                 contextlib.redirect_stdout(io.StringIO()):
                exec(cli_block(), namespace)
        finally:
            sys.argv = old_argv
        path = os.path.join(out, "scientific_dossier_summary.csv")
        return pd.read_csv(path, dtype=str, keep_default_na=False)


class TestPleiadesSummarySchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.success, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate())
        cls.fit_failed, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate(depth_ppm=150000.0))
        cls.deep_error, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate(),
                                           vet.Catalog(tic_error=ConnectionError("MAST down")))

    def test_success_and_deep_error_records_follow_summary_cols(self):
        self.assertEqual(self.success["operational_status"], "SUCCESS")
        self.assertEqual(self.fit_failed["fit_status"], "FAILED")  # S3.10: SUCCESS, INCOMPLETE_FIT_FAILED
        self.assertEqual(self.deep_error["operational_status"], "ERROR")
        for record in (self.success, self.fit_failed, self.deep_error):
            self.assertEqual(list(record), list(pleiades.PLEIADES_SUMMARY_COLS))

    def test_cli_failure_record_has_every_column_and_blank_fit_fields(self):
        row = pd.Series({"tic_id": "7", "period_days": 2.5, "t0_btjd": 4000.0,
                         "duration_hours": 1.5, "depth_ppm": 500.0, "r_star_sun": 1.0})
        for status, message in (("NO_LIGHTCURVE", "No light curve found"), ("ERROR", "boom")):
            with self.subTest(status=status):
                rec = pleiades.cli_failure_record(row, "7", status, message)
                self.assertEqual(list(rec), list(pleiades.PLEIADES_SUMMARY_COLS))
                self.assertEqual(rec["operational_status"], status)
                self.assertEqual(rec["error_message"], message)
                self.assertIsNone(rec["disposition"])
                self.assertEqual(rec["input_depth_ppm"], 500.0)
                # S3.9: nothing that was not evaluated is filled in.
                for key in (*pleiades.DEPTH_RECORD_FIELDS, *pleiades.SEC_DIAGNOSTIC_FIELDS,
                            "vps_score", "fp_risk_percent", "fitted_depth_ppm", "is_known_tess_eb"):
                    self.assertTrue(blank(rec[key]), key)

    def test_cli_mixed_batch_keeps_success_values(self):
        cand = vet.candidate()
        rows = [dict(cand, tic_id=222), dict(cand, tic_id=333, depth_ppm=float("nan")), cand]
        searches = {"222": vet.FakeSearch([]), "123456789": vet.two_sectors()}
        df = run_cli(rows, searches)
        self.assertEqual(list(df.columns), list(pleiades.PLEIADES_SUMMARY_COLS))
        self.assertEqual(df["operational_status"].tolist(), ["NO_LIGHTCURVE", "ERROR", "SUCCESS"])
        got = df.iloc[2]
        for key in pleiades.PLEIADES_SUMMARY_COLS:
            with self.subTest(column=key):
                want = "" if blank(self.success[key]) else str(self.success[key])
                if key == "vps_ledger_json":
                    self.assertEqual(got[key], want)
                    continue
                try:
                    self.assertAlmostEqual(float(got[key]), float(want), places=9)
                except ValueError:
                    self.assertEqual(got[key], want)
        for i in (0, 1):
            for key in pleiades.DEPTH_RECORD_FIELDS:
                self.assertEqual(df.iloc[i][key], "", key)

    def test_cli_all_failure_batch_has_full_header(self):
        cand = vet.candidate()
        for name, rows, searches in (
                ("no_lightcurve", [dict(cand, tic_id=222)], {"222": vet.FakeSearch([])}),
                ("error", [dict(cand, tic_id=333, depth_ppm=float("nan"))], {})):
            with self.subTest(case=name):
                df = run_cli(rows, searches)
                self.assertEqual(list(df.columns), list(pleiades.PLEIADES_SUMMARY_COLS))
                self.assertEqual(len(df), 1)
                for key in pleiades.DEPTH_RECORD_FIELDS:
                    self.assertEqual(df.iloc[0][key], "", key)

    def test_cli_empty_input_has_full_header(self):
        df = run_cli(pd.DataFrame(columns=list(vet.candidate())), {})
        self.assertEqual(list(df.columns), list(pleiades.PLEIADES_SUMMARY_COLS))
        self.assertEqual(len(df), 0)

    def test_missing_input_exits_1_without_name_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-B", os.path.join(ROOT, "pleiades.py"), "--input-csv",
                 os.path.join(tmp, "absent.csv"), "--output-dir", os.path.join(tmp, "out")],
                capture_output=True, text=True, cwd=tmp, env=dict(os.environ, ALDEBARAN_OFFLINE="1", MPLBACKEND="Agg"))
        self.assertEqual(proc.returncode, 1, proc.stderr[-2000:])
        self.assertNotIn("NameError", proc.stderr)
        self.assertIn("not found. Run aldebaran.py first.", proc.stdout)


class TestAntaresStage3CSchema(unittest.TestCase):
    NAMES = ("antares_prioritized_targets_ranking.csv", "antares_exofop_ctoi_upload.csv",
             "antares_exofop_ctoi_upload_comma.csv", "antares_exofop_ctoi_upload_top3.csv")

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    @staticmethod
    def archive(tic, depth):
        cand = {"tic_id": tic, "duration_hours": 2.4, "r_star_sun": 1.0, "m_star_sun": 1.0, "teff_k": 5700.0,
                "fit_converged": True, "fitted_rp_earth": 5.45, "impact_param_b": 0.3,
                "depth_schema_version": "depth_contract_v1", "measured_box_depth_ppm": depth,
                "corr_risk": 0.0, "contam_pct": 0.0, "neb_count": 0, "sectors_stitched": "5,6"}
        return dict(status="ARCHIVAL_PASS", candidate=cand, p_ref=4.0, t0_ref=2000.0, depth_ppm=depth,
                    stack_author="SPOC", snr_red=10.0, snr_white=12.0, n_transits=5)

    def packaged(self):
        phot = {tic: {"r_star": 1.0, "m_star": 1.0, "teff": 5700.0, "Tmag": 10.0, "Jmag": 9.0, "Kmag": 8.8}
                for tic in ("123", "456")}
        with mock.patch.object(pack, "batch_query_tic_photometry", return_value=phot), \
             mock.patch.object(pack, "extract_camera_ccd_from_cache", return_value=("1", "2")), \
             contextlib.redirect_stdout(io.StringIO()):
            return pack.run_stage_3c_packaging([self.archive("123", 800.0), self.archive("456", 1500.0)],
                                               results_dir=self.d)

    def empty(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return pack.run_stage_3c_packaging([{"status": "ARCHIVAL_REJECT", "candidate": {"tic_id": "9"}}],
                                               results_dir=self.d)

    def header(self, name, sep=","):
        with open(os.path.join(self.d, name), encoding="utf-8") as fh:
            return fh.readline().rstrip("\r\n").split(sep)

    def snapshot(self):
        return {p.name: p.read_bytes() for p in Path(self.d).iterdir()}

    def test_packaged_tables_use_the_constants(self):
        rank, upload = self.packaged()
        self.assertEqual(list(rank.columns), list(pack.ANTARES_RANKING_COLS))
        self.assertEqual(list(upload.columns), list(pack.ANTARES_EXOFOP_COLS))
        self.assertEqual(self.header(self.NAMES[0]), list(pack.ANTARES_RANKING_COLS))
        self.assertFalse(os.path.exists(os.path.join(self.d, self.NAMES[3])))

    def test_empty_tables_match_the_packaged_headers(self):
        self.empty()
        self.assertEqual(self.header(self.NAMES[0]), list(pack.ANTARES_RANKING_COLS))
        self.assertEqual(self.header(self.NAMES[1], "|"), list(pack.ANTARES_EXOFOP_COLS))
        self.assertEqual(self.header(self.NAMES[2]), list(pack.ANTARES_EXOFOP_COLS))
        self.assertEqual(self.header(self.NAMES[3]), list(pack.ANTARES_EXOFOP_COLS))

    def test_stage_3a_zero_survivors_match_the_packaged_headers(self):
        with mock.patch.object(pack, "run_stage_3a_filter",
                               return_value=(pd.DataFrame(columns=["tic_id"]), pd.DataFrame())), \
             contextlib.redirect_stdout(io.StringIO()):
            pack.run_antares_packager(results_dir=self.d, tier1_csv="unused", tier2_csv="unused")
        self.assertEqual(self.header(self.NAMES[0]), list(pack.ANTARES_RANKING_COLS))

    def test_packaged_then_empty_run_in_the_same_directory(self):
        self.packaged()
        self.empty()
        for name in self.NAMES:
            self.assertEqual(len(pd.read_csv(os.path.join(self.d, name), sep="|" if name == self.NAMES[1] else ",")), 0)

    def test_packaged_twice_in_the_same_directory(self):
        self.packaged()
        first = self.snapshot()
        self.packaged()
        self.assertEqual(self.snapshot(), first)

    def test_empty_then_packaged_refuses_the_stale_top3_table(self):
        self.empty()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Refusing stale zero-survivor table"):
            self.packaged()
        self.assertEqual(self.snapshot(), before)

    def test_legacy_empty_ranking_still_refuses(self):
        path = Path(self.d, self.NAMES[0])
        path.write_text(",".join(LEGACY_EMPTY_RANKING) + "\n")
        before = path.read_bytes()
        for run in (self.empty, self.packaged):
            with self.subTest(run=run.__name__):
                with self.assertRaisesRegex(ValueError, "Refusing conflicting output schema"):
                    run()
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
