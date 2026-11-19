"""S3.9 (roadmap C1f, C1g and the Pleiades SCI-14 fit-failure part).

Item 29: a check that could not run is an explicit state, never a pass or a reject.
- C1f: a raised Pleiades catalog query (TIC, VizieR TESS EB, Gaia DR3 NSS, Gaia neighbours) is
  recorded as not evaluated and the vetting record ends ERROR without disposition or VPS score.
- SCI-14: a failed Pleiades transit fit does the same.
- C1g: every Antares / Antares packager light-curve product that cannot be used is recorded;
  an all-failed packager drill ends ARCHIVAL_ERROR.
Valid controls (successful queries with no match, products skipped by design) keep today's result.
"""
import contextlib
import io
import json
import os
import ssl
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antares
import antares_packager
import lightcurve_helpers
import pleiades
import test_char_antares_archival_drill as drill
import test_char_pleiades_deep_vet as vet


class RaisingVizier:
    def __init__(self, *args, **kwargs):
        pass

    def query_region(self, *args, **kwargs):
        raise TimeoutError("VizieR timeout")


class EmptyVizier:
    def __init__(self, *args, **kwargs):
        pass

    def query_region(self, *args, **kwargs):
        return []


def empty_nss(url):
    return pd.DataFrame(columns=["nss_solution_type", "period"])


class FailingCatalog(vet.Catalog):
    """Catalog answers with one named query raising."""

    def __init__(self, fail, **kwargs):
        super().__init__(**kwargs)
        self.fail = fail

    def vizier(self, *args, **kwargs):
        if self.fail == "vizier":
            return RaisingVizier()
        return super().vizier(*args, **kwargs)

    def fetch(self, url):
        query = vet.unquote(url)
        if self.fail == "nss" and "nss_two_body_orbit" in query:
            raise OSError("Gaia TAP 500")
        if self.fail == "cone" and "nss_two_body_orbit" not in query:
            raise OSError("Gaia TAP 502")
        return super().fetch(url)


class TestPleiadesCatalogChecks(unittest.TestCase):
    def test_vizier_failure_recorded_nss_still_evaluated(self):
        with mock.patch.object(pleiades, "Vizier", RaisingVizier), \
             mock.patch.object(pleiades, "fetch_csv_no_ssl",
                               return_value=pd.DataFrame({"nss_solution_type": ["SB1"], "period": [4.0]})):
            res = pleiades.check_binary_catalogs(150.0, 20.0)
        self.assertEqual(res["catalog_check_failures"], ["TESS_EB_QUERY: TimeoutError: VizieR timeout"])
        self.assertTrue(res["is_gaia_nss"])

    def test_nss_failure_recorded(self):
        with mock.patch.object(pleiades, "Vizier", EmptyVizier), \
             mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=OSError("Gaia TAP 500")):
            res = pleiades.check_binary_catalogs(150.0, 20.0)
        self.assertEqual(res["catalog_check_failures"], ["GAIA_NSS_QUERY: OSError: Gaia TAP 500"])
        self.assertFalse(res["is_tess_eb"])

    def test_successful_queries_without_match_are_clean(self):
        # Valid control: a query that succeeds with no row is evaluated, not a failure.
        with mock.patch.object(pleiades, "Vizier", EmptyVizier), \
             mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=empty_nss):
            res = pleiades.check_binary_catalogs(150.0, 20.0)
        self.assertEqual(res["catalog_check_failures"], [])
        self.assertFalse(res["is_tess_eb"] or res["is_gaia_nss"])

    def test_ssl_error_still_propagates(self):
        with mock.patch.object(pleiades, "Vizier", EmptyVizier), \
             mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=ssl.SSLError("bad cert")):
            with self.assertRaises(ssl.SSLError):
                pleiades.check_binary_catalogs(150.0, 20.0)

    def test_tic_query_failure_recorded_and_later_checks_named(self):
        with mock.patch.object(pleiades.Catalogs, "query_criteria", side_effect=ConnectionError("MAST down")):
            info = pleiades.query_gaia_contamination("123", 2000.0)
        self.assertEqual(len(info["catalog_check_failures"]), 1)
        self.assertTrue(info["catalog_check_failures"][0].startswith("TIC_QUERY: ConnectionError: MAST down"))

    def test_empty_tic_result_is_not_a_failure(self):
        # Item 29.1 control: a successful query with no row keeps the defaults and no failure.
        with mock.patch.object(pleiades.Catalogs, "query_criteria", return_value=Table({"ra": [], "dec": []})):
            info = pleiades.query_gaia_contamination("123", 2000.0)
        self.assertEqual(info["catalog_check_failures"], [])

    def test_neighbour_query_failure_recorded(self):
        catalog = FailingCatalog("cone")
        with mock.patch.object(pleiades.Catalogs, "query_criteria", side_effect=catalog.tic), \
             mock.patch.object(pleiades, "Vizier", side_effect=catalog.vizier), \
             mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=catalog.fetch):
            info = pleiades.query_gaia_contamination("123", 2000.0)
        self.assertEqual(info["catalog_check_failures"], ["GAIA_NEIGHBOUR_QUERY: OSError: Gaia TAP 502"])


class TestPleiadesDeepVetRecord(unittest.TestCase):
    def assert_error_record(self, result, message):
        self.assertEqual(result["operational_status"], "ERROR")
        self.assertIsNone(result["disposition"])
        self.assertEqual(result["error_message"], message)
        self.assertTrue(np.isnan(result["vps_score"]))
        self.assertTrue(np.isnan(result["fp_risk_percent"]))
        self.assertEqual(result["vps_ledger_json"], "")
        self.assertEqual(result["vps_version"], "")

    def test_clean_run_is_success(self):
        result, written, _ = vet.run_vet(vet.two_sectors(), vet.candidate())
        self.assertTrue(written)
        self.assertEqual(result["operational_status"], "SUCCESS")
        self.assertEqual(result["error_message"], "")
        self.assertTrue(result["disposition"].startswith("TIER-2 HEURISTIC SCREEN PASS"))

    def test_vizier_failure_ends_error(self):
        result, written, _ = vet.run_vet(vet.two_sectors(), vet.candidate(), FailingCatalog("vizier"))
        self.assertTrue(written)
        self.assert_error_record(result, "NOT_EVALUATED: TESS_EB_QUERY: TimeoutError: VizieR timeout")
        self.assertIsNone(result["is_known_tess_eb"])
        self.assertEqual(result["flux_contam_percent"], 0.0)  # neighbour check ran

    def test_nss_failure_ends_error(self):
        result, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate(), FailingCatalog("nss"))
        self.assert_error_record(result, "NOT_EVALUATED: GAIA_NSS_QUERY: OSError: Gaia TAP 500")
        self.assertFalse(result["is_known_tess_eb"])  # EB check ran

    def test_neighbour_failure_ends_error(self):
        result, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate(), FailingCatalog("cone"))
        self.assert_error_record(result, "NOT_EVALUATED: GAIA_NEIGHBOUR_QUERY: OSError: Gaia TAP 502")
        self.assertTrue(np.isnan(result["flux_contam_percent"]))
        self.assertTrue(np.isnan(result["neb_capable_neighbors"]))

    def test_failure_with_secondary_flag_still_error(self):
        # Fail closed, as item 29.1 discards a CANDIDATE: no partial disposition is published.
        result, _, _ = vet.run_vet(vet.two_sectors(sec_depth=900e-6), vet.candidate(), FailingCatalog("vizier"))
        self.assertEqual(result["operational_status"], "ERROR")
        self.assertIsNone(result["disposition"])
        self.assertTrue(result["sec_cp_is_eb"])  # measured field kept

    def test_fit_failure_and_catalog_failure_is_catalog_error(self):
        # S3.10 (item 39.3): ERROR names the catalog failure only; fit_status FAILED stays in the record.
        result, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate(depth_ppm=150000.0), FailingCatalog("nss"))
        self.assert_error_record(result, "NOT_EVALUATED: GAIA_NSS_QUERY: OSError: Gaia TAP 500")
        self.assertEqual(result["fit_status"], "FAILED")
        self.assertRegex(result["fit_error"], r"^ValueError: ")

    def test_error_record_refused_by_vps_contract(self):
        from vps_contract import VPSContractError, validate_and_extract_vps_score
        result, _, _ = vet.run_vet(vet.two_sectors(), vet.candidate(), FailingCatalog("vizier"))
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(result)


def run_vet_rendered(search, cand, catalog=None):
    """Run deep_vet_target and capture the console text and the text given to the dossier renderer."""
    import matplotlib.axes
    catalog = catalog or vet.Catalog()
    texts = []
    real_text = matplotlib.axes.Axes.text

    def capture(self, x, y, s, *args, **kwargs):
        texts.append(s)
        return real_text(self, x, y, s, *args, **kwargs)
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp,          mock.patch.object(pleiades.lk, "search_lightcurve", return_value=search),          mock.patch.object(pleiades.Catalogs, "query_criteria", side_effect=catalog.tic),          mock.patch.object(pleiades, "Vizier", side_effect=catalog.vizier),          mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=catalog.fetch),          mock.patch.object(matplotlib.axes.Axes, "text", capture),          contextlib.redirect_stdout(out):
        result = pleiades.deep_vet_target(cand, vet_dir=tmp)
    dossier = [t for t in texts if t.startswith("SCIENTIFIC VETTING DOSSIER")]
    assert len(dossier) == 1, texts
    return result, dossier[0], out.getvalue()


class TestPleiadesDisplayMatchesRecord(unittest.TestCase):
    """S3.9 round 2 (Codex S3.9-R1-P2-1): the dossier and console show what the record says."""

    def test_clean_control_display_unchanged(self):
        result, dossier, console = run_vet_rendered(vet.two_sectors(), vet.candidate())
        self.assertEqual(result["operational_status"], "SUCCESS")
        self.assertIn("Vetting Score VPS: 0.2 / 100", dossier)
        self.assertIn("TESS EB Match:   None", dossier)
        self.assertIn("Gaia NSS Orbit:  None", dossier)
        self.assertIn("Gaia DR3 RUWE:   1.020", dossier)
        self.assertIn('Gaia Neighbors:  0 within 120" (Contam: 0.00%)', dossier)
        self.assertIn("NEB-Capable:     0 nearby stars", dossier)
        self.assertRegex(dossier, r"Impact Param:    b = 0\.3\d\d \(0=central")
        self.assertNotIn("NOT_EVALUATED", dossier + console)
        self.assertIn("Vetting Penalty Score (VPS): 0.2 (Pleiades", console)
        self.assertIn('Gaia DR3 Neighbors (120"): 0 stars | Contamination = 0.00%', console)

    def test_tic_failure_display_not_evaluated(self):
        result, dossier, console = run_vet_rendered(
            vet.two_sectors(), vet.candidate(), vet.Catalog(tic_error=ConnectionError("MAST timeout")))
        self.assertEqual(result["operational_status"], "ERROR")
        for line in ("Vetting Score VPS: NOT_EVALUATED / 100", "TESS EB Match:   NOT_EVALUATED",
                     "Gaia NSS Orbit:  NOT_EVALUATED", "Gaia DR3 RUWE:   NOT_EVALUATED",
                     "Gaia Neighbors:  NOT_EVALUATED", "NEB-Capable:     NOT_EVALUATED"):
            self.assertIn(line, dossier)
        for clean in ("0.2 / 100", "Match:   None", "Orbit:  None", "within 120", "nearby stars"):
            self.assertNotIn(clean, dossier)
        self.assertIn("Disposition:       ERROR (NOT_EVALUATED: TIC_QUERY", dossier)
        self.assertIn("Vetting Penalty Score (VPS): NOT_EVALUATED (Pleiades", console)
        self.assertIn('Gaia DR3 Neighbors (120"): NOT_EVALUATED', console)
        self.assertIn("VizieR TESS EB check: NOT_EVALUATED", console)
        self.assertIn("Gaia NSS check: NOT_EVALUATED", console)
        self.assertNotIn("Contamination = 0.00%", console)

    def test_partial_catalog_failure_keeps_measured_checks(self):
        # VizieR failed; the Gaia NSS match and the neighbour check ran and stay visible.
        result, dossier, console = run_vet_rendered(
            vet.two_sectors(), vet.candidate(), FailingCatalog("vizier", nss="SB1"))
        self.assertEqual(result["operational_status"], "ERROR")
        self.assertIn("TESS EB Match:   NOT_EVALUATED", dossier)
        self.assertIn("Gaia NSS Orbit:  SB1 (P=12.5d)", dossier)
        self.assertIn('Gaia Neighbors:  0 within 120" (Contam: 0.00%)', dossier)
        self.assertIn("Vetting Score VPS: NOT_EVALUATED / 100", dossier)
        self.assertIn("Gaia NSS Two-Body Orbit: SB1", console)
        self.assertNotIn("Gaia NSS check: NOT_EVALUATED", console)

    def test_neighbour_failure_display(self):
        _, dossier, console = run_vet_rendered(vet.two_sectors(), vet.candidate(), FailingCatalog("cone"))
        self.assertIn("Gaia Neighbors:  NOT_EVALUATED", dossier)
        self.assertIn("NEB-Capable:     NOT_EVALUATED", dossier)
        self.assertIn("TESS EB Match:   None", dossier)
        self.assertIn('Gaia DR3 Neighbors (120"): NOT_EVALUATED', console)

    def test_fit_failure_display(self):
        result, dossier, console = run_vet_rendered(vet.two_sectors(), vet.candidate(depth_ppm=150000.0))
        self.assertEqual(result["fit_status"], "FAILED")
        self.assertIn("Vetting Score VPS: NOT_EVALUATED / 100", dossier)
        self.assertIn("Radius Ratio:    Rp/R* = NOT_EVALUATED (initial guess 0.3873)", dossier)
        self.assertIn("Planet Radius:   Rp = NOT_EVALUATED (initial guess 42.24 R_Earth)", dossier)
        self.assertIn("Impact Param:    b = NOT_EVALUATED (initial guess 0.500)", dossier)
        self.assertIn("TESS EB Match:   None", dossier)  # catalog checks ran
        self.assertIn("Rp = NOT_EVALUATED (initial guess 42.24 R_Earth)", console)
        self.assertIn("b = NOT_EVALUATED (initial guess 0.500)", console)
        self.assertIn("Vetting Penalty Score (VPS): NOT_EVALUATED", console)


class TestLightCurveHelpers(unittest.TestCase):
    def test_extract_default_returns_none_on_exception(self):
        self.assertIsNone(lightcurve_helpers.extract_multi_author_lc(object(), "SPOC"))

    def test_extract_raise_errors(self):
        with self.assertRaises(AttributeError):
            lightcurve_helpers.extract_multi_author_lc(object(), "SPOC", raise_errors=True)

    def test_too_few_points_still_returns_none_with_raise_errors(self):
        lc = drill.synth_lc(3, 5, 0.0, n_points=60)
        self.assertIsNone(lightcurve_helpers.extract_multi_author_lc(lc, "SPOC", raise_errors=True))

    def test_product_failure_record(self):
        rec = lightcurve_helpers.light_curve_product_failure(7, "QLP", "DOWNLOAD", OSError("x" * 300))
        self.assertEqual(rec["stage"], "DOWNLOAD")
        self.assertEqual(rec["sector"], 7)
        self.assertEqual(len(rec["reason"]), 150)
        self.assertTrue(rec["reason"].startswith("OSError: x"))


class BrokenLC:
    """A downloaded product whose arrays cannot be read."""

    @property
    def time(self):
        raise ValueError("corrupt FITS table")


class BadMissionItem(drill.FakeItem):
    def __init__(self):
        super().__init__("SPOC", 5, 120)
        self.mission = ["TESS Sector ??"]


class TestPackagerProductFailures(unittest.TestCase):
    def test_extraction_exception_recorded_partial_verdict_kept(self):
        broken = drill.FakeItem("QLP", 5, 600, lc=BrokenLC())
        search = [drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22), broken]
        result, _ = drill.run_drill(search)
        self.assertEqual(result["status"], "ARCHIVAL_PASS")
        self.assertEqual(result["products_failed"], [
            {"sector": 5, "author": "QLP", "stage": "EXTRACTION", "reason": "ValueError: corrupt FITS table"}])

    def test_metadata_failure_recorded(self):
        search = [BadMissionItem(), drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22)]
        result, _ = drill.run_drill(search)
        self.assertEqual(result["status"], "ARCHIVAL_PASS")
        self.assertEqual([(f["sector"], f["stage"]) for f in result["products_failed"]], [(-1, "PRODUCT_METADATA")])

    def test_detrending_exception_recorded(self):
        search = [drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22)]
        real = antares_packager.hybrid_tukey_savgol_detrend
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise FloatingPointError("trend diverged")
            return real(*args, **kwargs)
        with mock.patch.object(antares_packager, "hybrid_tukey_savgol_detrend", side_effect=flaky):
            result, _ = drill.run_drill(search)
        self.assertEqual(result["products_failed"][0],
                         {"sector": 5, "author": "SPOC", "stage": "DETRENDING",
                          "reason": "FloatingPointError: trend diverged"})

    def test_all_products_failed_is_error(self):
        search = [drill.FakeItem("SPOC", 5, 120, error=OSError("HTTP 503"))]
        result, written = drill.run_drill(search)
        self.assertEqual(result["status"], "ARCHIVAL_ERROR")
        self.assertEqual(result["reason"], "All 1 light curve product(s) failed to download or prepare")
        self.assertFalse(written)
        self.assertNotIn("p_ref", result)

    def test_only_unsupported_authors_stays_reject(self):
        # Valid control: products skipped by design are not failures; the reject is unchanged.
        result, _ = drill.run_drill([drill.item("K2SFF", 5, 2000e-6, seed=1)])
        self.assertEqual(result["status"], "ARCHIVAL_REJECT")
        self.assertEqual(result["reason"], "No valid light curves extracted")
        self.assertEqual(result["products_failed"], [])

    def test_no_failure_pass_has_empty_record(self):
        search = [drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22)]
        result, _ = drill.run_drill(search)
        self.assertEqual(result["products_failed"], [])

    def test_audit_row_names_failed_products(self):
        search = [drill.FakeItem("SPOC", 5, 120, error=OSError("HTTP 503")),
                  drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22)]
        result, _ = drill.run_drill(search)
        row = antares_packager.stage_3b_audit_row("123456789", result)
        self.assertEqual(row["stage_3b_status"], result["status"])
        self.assertTrue(row["decision_reason"].startswith(result["reason"] + "; 1 light curve product(s) not used: "))
        clean = {"status": "ARCHIVAL_REJECT", "reason": "No MAST light curves"}
        self.assertEqual(antares_packager.stage_3b_audit_row("1", clean)["decision_reason"], "No MAST light curves")

    def test_decision_reason_text(self):
        base = {"status": "ARCHIVAL_PASS", "reason": "SNR ok"}
        self.assertEqual(antares_packager.archival_decision_reason(base), "SNR ok")
        self.assertEqual(antares_packager.archival_decision_reason(dict(base, products_failed=[])), "SNR ok")
        failed = [{"sector": 5, "author": "QLP", "stage": "DOWNLOAD", "reason": "OSError: 503"}]
        self.assertEqual(antares_packager.archival_decision_reason(dict(base, products_failed=failed)),
                         "SNR ok; 1 light curve product(s) not used: S5 QLP DOWNLOAD (OSError: 503)")
        err = {"status": "ARCHIVAL_ERROR", "reason": "All 1 light curve product(s) failed to download or prepare",
               "products_failed": failed}
        self.assertEqual(antares_packager.archival_decision_reason(err),
                         "All 1 light curve product(s) failed to download or prepare: S5 QLP DOWNLOAD (OSError: 503)")


STAR = {"r_star": 1.0, "m_star": 1.0, "teff": 5700.0, "tmag": 10.0, "u1": 0.4, "u2": 0.25,
        "stellar_parameter_sources": {}, "catalog_error": ""}


def run_detective(search):
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(antares, "RESULTS_DIR", tmp), \
         mock.patch.object(antares, "get_stellar_limb_darkening", return_value=dict(STAR)), \
         mock.patch.object(antares.lk, "search_lightcurve", return_value=search), \
         contextlib.redirect_stdout(io.StringIO()) as out:
        antares.run_antares_detective("123456789", drill.PERIOD, drill.T0, 2.4)
        summary = pd.read_csv(os.path.join(tmp, "antares_summary_123456789.csv"))
    return summary, out.getvalue()


class TestAntaresDetectiveProductFailures(unittest.TestCase):
    def test_all_failed_error_names_failures(self):
        with mock.patch.object(antares, "get_stellar_limb_darkening", return_value=dict(STAR)), \
             mock.patch.object(antares.lk, "search_lightcurve",
                               return_value=[drill.FakeItem("SPOC", 5, 120, error=OSError("HTTP 503"))]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, r"1 product\(s\) failed: .*'stage': 'DOWNLOAD'"):
                antares.run_antares_detective("123456789", drill.PERIOD, drill.T0, 2.4)

    def test_nothing_usable_without_failure_keeps_message(self):
        with mock.patch.object(antares, "get_stellar_limb_darkening", return_value=dict(STAR)), \
             mock.patch.object(antares.lk, "search_lightcurve",
                               return_value=[drill.item("K2SFF", 5, 2000e-6, seed=1)]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as ctx:
                antares.run_antares_detective("123456789", drill.PERIOD, drill.T0, 2.4)
        self.assertEqual(str(ctx.exception),
                         "Could not extract any valid SPOC/QLP/TGLC light curves for TIC 123456789.")

    def test_partial_failure_recorded_in_summary(self):
        search = [drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22),
                  drill.FakeItem("QLP", 6, 600, error=OSError("HTTP 503"))]
        summary, out = run_detective(search)
        master = summary[summary["group_type"] == "MASTER_DEDUPLICATED_ALL_YEARS"].iloc[0]
        self.assertEqual(json.loads(master["light_curve_products_failed"]),
                         [{"sector": 6, "author": "QLP", "stage": "DOWNLOAD", "reason": "OSError: HTTP 503"}])
        self.assertIn("Product not used: Sector 6 | Author: QLP | DOWNLOAD: OSError: HTTP 503", out)

    def test_no_failure_summary_has_no_new_column(self):
        summary, out = run_detective([drill.item("SPOC", 5, 2000e-6, 21), drill.item("SPOC", 6, 2000e-6, 22)])
        self.assertNotIn("light_curve_products_failed", summary.columns)
        self.assertNotIn("Product not used", out)


if __name__ == "__main__":
    unittest.main()
