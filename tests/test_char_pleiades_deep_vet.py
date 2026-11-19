"""Characterization tests for pleiades.deep_vet_target (CHAR-TESTS-01, plan 008).

They pin the current Tier 2 outputs on seeded synthetic light curves. Every
external call is mocked at its network boundary: lightkurve search and
download, the MAST TIC query, the VizieR TESS EB query and the two Gaia TAP
requests (NSS two-body orbit and the 120 arcsec cone). The vetting code in
between runs unchanged. Plots go to a temporary directory.

These tests describe present behaviour, including behaviour the roadmap
intended to change (C1f and SCI-14, changed by S3.9: a failed catalog query
or transit fit now ends the record ERROR). A task that changes such behaviour on purpose updates the matching
assertion and names it in its equivalence evidence.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import unquote

import numpy as np
import pandas as pd
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pleiades

PERIOD = 3.0
T0 = 2001.3
DUR_D = 0.1
CADENCE_D = 10.0 / 1440.0
NOISE = 300e-6


def synth_sector(seed, start, span=12.0, depth=2000e-6, sec_depth=0.0):
    rng = np.random.default_rng(seed)
    t = start + np.arange(0.0, span, CADENCE_D)
    f = 1.0 + rng.normal(0.0, NOISE, t.size)
    ph = ((t - T0 + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD
    f[np.abs(ph) < 0.5 * DUR_D] -= depth
    if sec_depth:
        ph2 = ((t - T0) % PERIOD) / PERIOD
        f[np.abs(ph2 - 0.5) * PERIOD < 0.5 * DUR_D] -= sec_depth
    return t, f * 1000.0


class FakeLC:
    def __init__(self, t, f):
        self.time = t
        self.flux = f
        self.quality = np.zeros(t.size, dtype=int)


class FakeItem:
    def __init__(self, sector, lc):
        self.sector = sector
        self.lc = lc

    def download(self):
        return self.lc


class FakeSearch:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FakeSearch(self.items[key])
        return self.items[key]

    def __iter__(self):
        return iter(self.items)

    @property
    def mission(self):
        return [f"TESS Sector {item.sector:02d}" for item in self.items]


def two_sectors(**kwargs):
    t1, f1 = synth_sector(11, 2000.0, **kwargs)
    t2, f2 = synth_sector(12, 2014.0, **kwargs)
    return FakeSearch([FakeItem(5, FakeLC(t1, f1)), FakeItem(6, FakeLC(t2, f2))])


def candidate(**overrides):
    row = {
        "tic_id": 123456789, "period_days": PERIOD, "t0_btjd": T0, "duration_hours": 2.4,
        "depth_ppm": 2000.0, "r_star_sun": 1.0, "m_star_sun": 1.0, "teff_k": 5700.0,
        "e_r_star_sun": 0.05, "e_m_star_sun": 0.05,
        "odd_even_sigma": 0.4, "centroid_shift_arcsec": 0.2,
    }
    row.update(overrides)
    return row


def gaia_cone(neighbours=()):
    rows = [{"ra": 150.0, "dec": 20.0, "phot_g_mean_mag": 11.0, "ruwe": 1.02, "sep_arcsec": 0.0}]
    for g_mag, sep in neighbours:
        rows.append({"ra": 150.0, "dec": 20.0, "phot_g_mean_mag": g_mag, "ruwe": 1.0, "sep_arcsec": sep})
    return pd.DataFrame(rows)


class Catalog:
    """Mocked external catalog answers for one run."""

    def __init__(self, tess_eb=False, nss=None, neighbours=(), tic_error=None):
        self.tess_eb = tess_eb
        self.nss = nss
        self.neighbours = neighbours
        self.tic_error = tic_error
        self.urls = []

    def tic(self, catalog, ID):
        if self.tic_error is not None:
            raise self.tic_error
        return Table({"ra": [150.0], "dec": [20.0]})

    def vizier(self, *args, **kwargs):
        viz = mock.Mock()
        if self.tess_eb:
            viz.query_region.return_value = [Table({"TIC": [987654321], "Per": [6.0], "Morph": [0.7]})]
        else:
            viz.query_region.return_value = []
        return viz

    def fetch(self, url):
        query = unquote(url)
        self.urls.append(query)
        if "nss_two_body_orbit" in query:
            if self.nss is None:
                return pd.DataFrame(columns=["nss_solution_type", "period"])
            return pd.DataFrame({"nss_solution_type": [self.nss], "period": [12.5]})
        return gaia_cone(self.neighbours)


def run_vet(search, cand, catalog=None):
    catalog = catalog or Catalog()
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(pleiades.lk, "search_lightcurve", return_value=search) as search_mock, \
         mock.patch.object(pleiades.Catalogs, "query_criteria", side_effect=catalog.tic), \
         mock.patch.object(pleiades, "Vizier", side_effect=catalog.vizier), \
         mock.patch.object(pleiades, "fetch_csv_no_ssl", side_effect=catalog.fetch), \
         contextlib.redirect_stdout(io.StringIO()):
        result = pleiades.deep_vet_target(cand, vet_dir=tmp)
        dossier = os.path.join(tmp, f"dossier_TIC_{int(cand['tic_id'])}.png")
        written = os.path.exists(dossier)
    search_mock.assert_called_once_with(
        f"TIC {int(cand['tic_id'])}", mission="TESS", author="SPOC", exptime=120)
    return result, written, catalog


# Pinned outputs. Categorical fields are exact. Numeric fields carry a tolerance
# met by both the Linux venv (Python 3.11, numpy 1.23.5, scipy 1.13.1) and
# Windows (Python 3.14, numpy 2.5.4, scipy 1.18.1).
PINNED = {
    "pass": {
        "operational_status": 'SUCCESS',
        "disposition": 'TIER-2 HEURISTIC SCREEN PASS (candidate; no calibrated FP-risk estimate)',
        "fit_status": 'CONVERGED',
        "vps_score": 0.2,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1823.65,
        "fitted_duration_hours": 2.695,
        "fitted_rp_earth": 4.66,
        "impact_param_b": 0.324,
        "dur_to_kepler_max": 0.98,
        "stitched_primary_snr": 77.325,
        "secondary_snr_sigma": 1.795,
        "sec_eclipse_phase": 0.135,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "tess_eb": {
        "operational_status": 'SUCCESS',
        "disposition": 'CATALOG MATCH: TESS EB (possible false-positive flag; TIC 987654321)',
        "fit_status": 'CONVERGED',
        "vps_score": 99.9,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1823.65,
        "fitted_duration_hours": 2.695,
        "fitted_rp_earth": 4.66,
        "impact_param_b": 0.324,
        "dur_to_kepler_max": 0.98,
        "stitched_primary_snr": 77.325,
        "secondary_snr_sigma": 1.795,
        "sec_eclipse_phase": 0.135,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": True,
        "gaia_nss_type": 'SB1',
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "secondary": {
        "operational_status": 'SUCCESS',
        # S3.2 (C8, item 31.1): the flag text now comes from the Cp decision, which names the
        # circular zone and reports phase 0.50 (was "eccentric EB (Phase 0.49, ...)", SCI-69 message defect).
        "disposition": 'SECONDARY-ECLIPSE FLAG: possible circular EB (Phase 0.50, 18.6 sigma)',
        "fit_status": 'CONVERGED',
        "vps_score": 99.9,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1843.4,
        "fitted_duration_hours": 2.69,
        "fitted_rp_earth": 4.68,
        "impact_param_b": 0.34,
        "dur_to_kepler_max": 0.98,
        "stitched_primary_snr": 78.84,
        "secondary_snr_sigma": 18.55,
        "sec_eclipse_phase": 0.491,
        "sec_eclipse_zone": 'circular',
        "sec_penalty_applied": True,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "nss": {
        "operational_status": 'SUCCESS',
        "disposition": 'GAIA NSS MATCH / DUAL-STAR SYSTEM FLAG (SB1)',
        "fit_status": 'CONVERGED',
        "vps_score": 80.2,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1823.65,
        "fitted_duration_hours": 2.695,
        "fitted_rp_earth": 4.66,
        "impact_param_b": 0.324,
        "dur_to_kepler_max": 0.98,
        "stitched_primary_snr": 77.325,
        "secondary_snr_sigma": 1.795,
        "sec_eclipse_phase": 0.135,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": 'SB1',
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "narrow": {
        "operational_status": 'SUCCESS',
        "disposition": 'SUSPECT / UNPHYSICALLY NARROW TRANSIT',
        "fit_status": 'CONVERGED',
        "vps_score": 46.65,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1827.25,
        "fitted_duration_hours": 2.69,
        "fitted_rp_earth": 27.975,
        "impact_param_b": 0.33,
        "dur_to_kepler_max": 0.16,
        "stitched_primary_snr": 77.26,
        "secondary_snr_sigma": 1.795,
        "sec_eclipse_phase": 0.135,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "long": {
        "operational_status": 'SUCCESS',
        "disposition": 'SUSPECT / UNPHYSICALLY LONG TRANSIT (Subgiant/Blend)',
        "fit_status": 'CONVERGED',
        "vps_score": 51.6,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1817.45,
        "fitted_duration_hours": 1.76,
        "fitted_rp_earth": 0.7,
        "impact_param_b": 0.2825,
        "dur_to_kepler_max": 2.28,
        "stitched_primary_snr": 66.32,
        "secondary_snr_sigma": 2.1,
        "sec_eclipse_phase": 0.705,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "contaminated": {
        "operational_status": 'SUCCESS',
        "disposition": 'SUSPECT / GRAZING OR CONTAMINATED',
        "fit_status": 'CONVERGED',
        "vps_score": 36.1,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 1823.65,
        "fitted_duration_hours": 2.695,
        "fitted_rp_earth": 4.66,
        "impact_param_b": 0.324,
        "dur_to_kepler_max": 0.98,
        "stitched_primary_snr": 77.325,
        "secondary_snr_sigma": 1.795,
        "sec_eclipse_phase": 0.135,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 79.85,
        "neb_capable_neighbors": 1,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 2000.0,
    },
    "weak": {
        "operational_status": 'SUCCESS',
        "disposition": 'SUSPECT / GRAZING OR CONTAMINATED',
        "fit_status": 'CONVERGED',
        "vps_score": 7.85,
        "refined_period_days": 3.001533,
        "refined_t0_btjd": 2001.323,
        "fitted_depth_ppm": 122.35,
        "fitted_duration_hours": 2.09,
        "fitted_rp_earth": 1.205,
        "impact_param_b": 0.3015,
        "dur_to_kepler_max": 0.785,
        "stitched_primary_snr": 5.15,
        "secondary_snr_sigma": 1.945,
        "sec_eclipse_phase": 0.118,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 60.0,
    },
    "fallback": {
        "operational_status": 'SUCCESS',
        "disposition": 'TIER-2 HEURISTIC SCREEN PASS (candidate; no calibrated FP-risk estimate)',
        "fit_status": 'INITIAL_GUESS_FALLBACK',
        "vps_score": 0.2,
        "refined_period_days": 2.999632,
        "refined_t0_btjd": 2001.2985,
        "fitted_depth_ppm": 150000.0,
        "fitted_duration_hours": 2.52,
        "fitted_rp_earth": 42.24,
        "impact_param_b": 0.5,
        "dur_to_kepler_max": 0.69,
        "stitched_primary_snr": 73.44,
        "secondary_snr_sigma": 1.87,
        "sec_eclipse_phase": 0.698,
        "sec_eclipse_zone": 'eccentric',
        "sec_penalty_applied": False,
        "is_known_tess_eb": False,
        "gaia_nss_type": None,
        "flux_contam_percent": 0.0,
        "neb_capable_neighbors": 0,
        "sectors_stitched": '5,6',
        "stellar_parameter_provenance": 'input_host_parameters',
        "error_message": '',
        "input_depth_ppm": 150000.0,
    },
}

# Absolute tolerances. The two interpreters agree on every categorical field
# and on the refined ephemeris; the transit fit (scipy curve_fit) differs by up
# to 7.1 ppm in depth, 0.17 sigma in secondary SNR and 0.14 in stitched SNR.
NUM_TOL = {
    "refined_period_days": 1e-6, "refined_t0_btjd": 1e-4, "fitted_depth_ppm": 15.0,
    "fitted_duration_hours": 0.05, "fitted_rp_earth": 0.1, "impact_param_b": 0.02,
    "dur_to_kepler_max": 0.02, "stitched_primary_snr": 0.3, "secondary_snr_sigma": 0.2,
    "sec_eclipse_phase": 0.002, "vps_score": 0.3, "flux_contam_percent": 0.05,
}
# The weak case sits near the stitched SNR gate of 5.5, where the VPS SNR penalty
# is steep: 9.3 on Windows and 6.4 on Linux. Both stay above the clean
# criterion of 5.0, which the categorical disposition already pins.
CASE_TOL = {("weak", "vps_score"): 2.0}


class TestDeepVetTargetCharacterization(unittest.TestCase):
    def assert_pinned(self, name, result):
        expected = PINNED[name]
        for key, value in expected.items():
            with self.subTest(case=name, field=key):
                if key in NUM_TOL:
                    tol = CASE_TOL.get((name, key), NUM_TOL[key])
                    self.assertAlmostEqual(result[key], value, delta=tol)
                else:
                    self.assertEqual(result[key], value)

    def test_missing_depth_raises_before_any_query(self):
        for depth in (None, float("nan"), 0.0, -5.0, "N/A"):
            with self.subTest(depth=depth):
                with mock.patch.object(pleiades.lk, "search_lightcurve") as search_mock:
                    with self.assertRaisesRegex(ValueError, "Missing positive Tier-1 photometric depth"):
                        pleiades.deep_vet_target(candidate(depth_ppm=depth), vet_dir=tempfile.gettempdir())
                search_mock.assert_not_called()

    def test_no_spoc_light_curve_returns_none(self):
        result, written, _ = run_vet(FakeSearch([]), candidate())
        self.assertIsNone(result)
        self.assertFalse(written)

    def test_heuristic_pass(self):
        result, written, catalog = run_vet(two_sectors(), candidate())
        self.assertTrue(written)
        self.assert_pinned("pass", result)
        self.assertEqual(len(catalog.urls), 2)
        ledger = json.loads(result["vps_ledger_json"])
        self.assertEqual(ledger["final_score"], result["vps_score"])

    def test_catalog_failure_is_error_not_clean(self):
        # S3.9 (C1f, item 29): a failed TIC query skips VizieR and Gaia; the record
        # ends ERROR with no disposition or VPS score (was the clean-run PASS).
        result, _, catalog = run_vet(two_sectors(), candidate(),
                                     Catalog(tic_error=ConnectionError("MAST timeout")))
        self.assertEqual(catalog.urls, [])
        self.assertEqual(result["operational_status"], "ERROR")
        self.assertIsNone(result["disposition"])
        self.assertEqual(result["error_message"],
                         "NOT_EVALUATED: TIC_QUERY: ConnectionError: MAST timeout "
                         "(TESS EB, Gaia NSS and Gaia neighbour checks not run)")
        self.assertTrue(np.isnan(result["vps_score"]))
        self.assertEqual(result["vps_ledger_json"], "")
        self.assertIsNone(result["is_known_tess_eb"])
        self.assertTrue(np.isnan(result["flux_contam_percent"]))
        # Measured fields are kept.
        self.assertAlmostEqual(result["refined_period_days"], PINNED["pass"]["refined_period_days"], delta=1e-6)
        self.assertEqual(result["fit_status"], "CONVERGED")

    def test_tess_eb_catalog_match(self):
        result, _, _ = run_vet(two_sectors(), candidate(), Catalog(tess_eb=True, nss="SB1"))
        self.assert_pinned("tess_eb", result)

    def test_secondary_eclipse_flag(self):
        result, _, _ = run_vet(two_sectors(sec_depth=900e-6), candidate())
        self.assert_pinned("secondary", result)

    def test_gaia_nss_match(self):
        result, _, _ = run_vet(two_sectors(), candidate(), Catalog(nss="SB1"))
        self.assert_pinned("nss", result)

    def test_unphysically_narrow_transit(self):
        result, _, _ = run_vet(two_sectors(), candidate(r_star_sun=6.0, m_star_sun=1.2))
        self.assert_pinned("narrow", result)

    def test_unphysically_long_transit(self):
        result, _, _ = run_vet(two_sectors(), candidate(r_star_sun=0.15, m_star_sun=0.15))
        self.assert_pinned("long", result)

    def test_contaminated_by_neb_capable_neighbour(self):
        result, _, _ = run_vet(two_sectors(), candidate(), Catalog(neighbours=[(11.2, 6.0)]))
        self.assert_pinned("contaminated", result)

    def test_weak_transit_fails_snr_gate(self):
        result, _, _ = run_vet(two_sectors(depth=60e-6), candidate(depth_ppm=60.0))
        self.assert_pinned("weak", result)
        self.assertLess(result["stitched_primary_snr"], 5.5)

    def test_fit_failure_is_incomplete(self):
        # A Tier 1 depth of 150000 ppm puts the radius-ratio guess (0.387) above the
        # fit bound of 0.30, so curve_fit refuses the start point. S3.10 (item 39.1):
        # SUCCESS, fit_status FAILED, disposition INCOMPLETE_FIT_FAILED, no VPS score
        # and no guess-based fit column (S3.9 gave ERROR; before S3.9 "TIER-2 HEURISTIC
        # SCREEN PASS" built on b = 0.5).
        result, _, _ = run_vet(two_sectors(), candidate(depth_ppm=150000.0))
        expected = dict(PINNED["fallback"], disposition="INCOMPLETE_FIT_FAILED", fit_status="FAILED")
        for key in ("vps_score", *pleiades.FIT_DERIVED_FIELDS):
            expected.pop(key)
        PINNED["_fallback_incomplete"] = expected
        self.assert_pinned("_fallback_incomplete", result)
        self.assertRegex(result["fit_error"], r"^ValueError: ")
        self.assertIsNone(result["vps_score"])
        for key in pleiades.FIT_DERIVED_FIELDS:
            self.assertTrue(np.isnan(result[key]), key)

if __name__ == "__main__":
    unittest.main()
