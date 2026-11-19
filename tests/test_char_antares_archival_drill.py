"""Characterization tests for antares_packager.execute_archival_drill (CHAR-TESTS-01, plan 008).

They pin the current Stage 3B outputs on seeded synthetic light curves. The
only external calls, the lightkurve MAST search and each item's download, are
mocked; extraction, detrending, ephemeris refinement, transit metrics, the
blend check and the decision gate run unchanged. Posters go to a temporary
directory.

These tests describe present behaviour, including behaviour the roadmap
intended to change (C1g, changed by S3.9: a failed product is now recorded, and
an all-failed drill ends ARCHIVAL_ERROR instead of ARCHIVAL_REJECT). A
task that changes such behaviour on purpose updates the matching assertion and
names it in its equivalence evidence.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antares_packager

PERIOD = 3.0
T0 = 2001.3
DUR_D = 0.1
CADENCE_D = 10.0 / 1440.0
NOISE = 300e-6
SECTOR_START = {5: 2000.0, 6: 2014.0}


def synth_lc(seed, sector, depth, n_points=None):
    rng = np.random.default_rng(seed)
    t = SECTOR_START[sector] + np.arange(0.0, 12.0, CADENCE_D)
    if n_points is not None:
        t = t[:n_points]
    f = 1.0 + rng.normal(0.0, NOISE, t.size)
    ph = ((t - T0 + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD
    f[np.abs(ph) < 0.5 * DUR_D] -= depth
    return FakeLC(t, f * 5000.0)


class FakeLC:
    def __init__(self, t, f):
        self.time = t
        self.flux = f
        self.quality = np.zeros(t.size, dtype=int)


class FakeItem:
    def __init__(self, author, sector, exptime, lc=None, error=None):
        self.author = [author]
        self.mission = [f"TESS Sector {sector:02d}"]
        self.exptime = [float(exptime)]
        self.lc = lc
        self.error = error
        self.downloads = 0

    def download(self):
        self.downloads += 1
        if self.error is not None:
            raise self.error
        return self.lc


def item(author, sector, depth, seed, exptime=None, **kwargs):
    if exptime is None:
        exptime = 120 if author == "SPOC" else 600
    return FakeItem(author, sector, exptime, lc=synth_lc(seed, sector, depth, **kwargs))


def candidate(**overrides):
    cand = {"tic_id": "123456789", "period_d": PERIOD, "t0_btjd": T0, "duration_hours": 2.4, "contam_pct": 10.0}
    cand.update(overrides)
    return cand


def run_drill(search, cand=None, search_error=None):
    cand = cand or candidate()
    kwargs = {"side_effect": search_error} if search_error is not None else {"return_value": search}
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(antares_packager.lk, "search_lightcurve", **kwargs) as search_mock, \
         contextlib.redirect_stdout(io.StringIO()):
        result = antares_packager.execute_archival_drill(cand, results_dir=tmp)
        poster = result.get("poster_path")
        poster_written = bool(poster) and os.path.exists(poster)
        if poster:
            result["poster_path"] = os.path.relpath(poster, tmp).replace(os.sep, "/")
    search_mock.assert_called_once_with(f"TIC {cand['tic_id']}", mission="TESS")
    return result, poster_written


# Pinned outputs, identical on the Linux venv (Python 3.11, numpy 1.23.5,
# scipy 1.13.1) and Windows (Python 3.14, numpy 2.5.4, scipy 1.18.1). Exact for
# strings and integers; floats carry rounding-level tolerances below.
PINNED = {
    "spoc_pass": {
        "status": 'ARCHIVAL_PASS',
        "reason": 'SNR_red=70.18s (SNR_white=70.2s) on SPOC stack (N=8 transits across 2 sectors)',
        "stack_author": 'SPOC',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 3.0003212677231024,
        "t0_ref": 2001.3022240802675,
        "dp_sec": 27.757531276044745,
        "depth_ppm": 1958.7,
        "err_white_ppm": 27.9,
        "snr_white": 70.18,
        "snr_red": 70.18,
        "beta_red": 1.0,
        "author_stats": {
            "SPOC": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 1958.7, 'err_white_ppm': 27.9, 'snr_white': 70.18, 'snr_red': 70.18, 'beta_red': 1.0},
            "QLP": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 1859.7, 'err_white_ppm': 29.6, 'snr_white': 62.82, 'snr_red': 55.51, 'beta_red': 1.13},
        },
    },
    "spoc_only_pass": {
        "status": 'ARCHIVAL_PASS',
        "reason": 'SNR_red=70.18s (SNR_white=70.2s) on SPOC stack (N=8 transits across 2 sectors)',
        "stack_author": 'SPOC',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'NOT_EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 3.0003212677231024,
        "t0_ref": 2001.3022240802675,
        "dp_sec": 27.757531276044745,
        "depth_ppm": 1958.7,
        "err_white_ppm": 27.9,
        "snr_white": 70.18,
        "snr_red": 70.18,
        "beta_red": 1.0,
        "author_stats": {
            "SPOC": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 1958.7, 'err_white_ppm': 27.9, 'snr_white': 70.18, 'snr_red': 70.18, 'beta_red': 1.0},
        },
    },
    "blended": {
        "status": 'ARCHIVAL_REJECT',
        "reason": 'Aperture Dilution / Blended (SPOC depth (2072 ppm) diluted > 2.5x in QLP/TGLC (54 ppm))',
        "stack_author": 'SPOC',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 3.0003212677231024,
        "t0_ref": 2001.2884113712373,
        "dp_sec": 27.757531276044745,
        "depth_ppm": 2072.5,
        "err_white_ppm": 29.9,
        "snr_white": 69.39,
        "snr_red": 69.39,
        "beta_red": 1.0,
        "author_stats": {
            "SPOC": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 2072.5, 'err_white_ppm': 29.9, 'snr_white': 69.39, 'snr_red': 69.39, 'beta_red': 1.0},
            "QLP": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 53.7, 'err_white_ppm': 29.3, 'snr_white': 1.83, 'snr_red': 1.74, 'beta_red': 1.06},
        },
    },
    "diluted_low_contam": {
        "status": 'ARCHIVAL_PASS',
        "reason": 'SNR_red=69.39s (SNR_white=69.4s) on SPOC stack (N=8 transits across 2 sectors)',
        "stack_author": 'SPOC',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 3.0003212677231024,
        "t0_ref": 2001.2884113712373,
        "dp_sec": 27.757531276044745,
        "depth_ppm": 2072.5,
        "err_white_ppm": 29.9,
        "snr_white": 69.39,
        "snr_red": 69.39,
        "beta_red": 1.0,
        "author_stats": {
            "SPOC": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 2072.5, 'err_white_ppm': 29.9, 'snr_white': 69.39, 'snr_red': 69.39, 'beta_red': 1.0},
            "QLP": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 53.7, 'err_white_ppm': 29.3, 'snr_white': 1.83, 'snr_red': 1.74, 'beta_red': 1.06},
        },
    },
    "low_snr": {
        "status": 'ARCHIVAL_REJECT',
        "reason": 'SNR_red (0.68 sigma) < threshold (3.8 sigma)',
        "stack_author": 'SPOC',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'NOT_EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 2.999000166805671,
        "t0_ref": 2001.3001170568562,
        "dp_sec": -86.38558799001146,
        "depth_ppm": 20.0,
        "err_white_ppm": 29.5,
        "snr_white": 0.68,
        "snr_red": 0.68,
        "beta_red": 1.0,
        "author_stats": {
            "SPOC": {'n_transits': 8, 'n_cadences_in': 103, 'depth_ppm': 20.0, 'err_white_ppm': 29.5, 'snr_white': 0.68, 'snr_red': 0.68, 'beta_red': 1.0},
        },
    },
    "multi_author": {
        "status": 'ARCHIVAL_PASS',
        "reason": 'SNR_red=70.26s (SNR_white=70.3s) on MULTI_AUTHOR stack (N=8 transits across 2 sectors)',
        "stack_author": 'MULTI_AUTHOR',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'NOT_EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 3.0007416180150126,
        "t0_ref": 2001.3022240802675,
        "dp_sec": 64.07579649708879,
        "depth_ppm": 2068.9,
        "err_white_ppm": 29.4,
        "snr_white": 70.26,
        "snr_red": 70.26,
        "beta_red": 1.0,
        "author_stats": {
            "QLP": {'n_transits': 4, 'n_cadences_in': 52, 'depth_ppm': 2077.1, 'err_white_ppm': 41.9, 'snr_white': 49.54, 'snr_red': 49.54, 'beta_red': 1.0},
            "TGLC": {'n_transits': 4, 'n_cadences_in': 52, 'depth_ppm': 2060.9, 'err_white_ppm': 41.4, 'snr_white': 49.82, 'snr_red': 49.82, 'beta_red': 1.0},
        },
    },
    "spoc_download_failed": {
        "status": 'ARCHIVAL_PASS',
        "reason": 'SNR_red=66.97s (SNR_white=67.0s) on MULTI_AUTHOR stack (N=8 transits across 2 sectors)',
        "stack_author": 'MULTI_AUTHOR',
        "n_sectors_stack": 2,
        "n_transits": 8,
        "blend_check": 'NOT_EVALUATED',
        "poster_path": 'archival_vetting/antares_TIC_123456789.png',
        "p_ref": 3.0007416180150126,
        "t0_ref": 2001.2909866220734,
        "dp_sec": 64.07579649708879,
        "depth_ppm": 2007.4,
        "err_white_ppm": 30.0,
        "snr_white": 66.97,
        "snr_red": 66.97,
        "beta_red": 1.0,
        "author_stats": {
            "QLP": {'n_transits': 8, 'n_cadences_in': 104, 'depth_ppm': 2007.4, 'err_white_ppm': 30.0, 'snr_white': 66.97, 'snr_red': 66.97, 'beta_red': 1.0},
        },
    },
}

NUM_TOL = {
    "p_ref": 1e-7, "t0_ref": 1e-5, "dp_sec": 0.01, "depth_ppm": 0.1, "err_white_ppm": 0.1,
    "snr_white": 0.01, "snr_red": 0.01, "beta_red": 0.01,
    "n_transits": 0, "n_cadences_in": 0,
}


class TestExecuteArchivalDrillCharacterization(unittest.TestCase):
    def assert_pinned(self, name, result):
        expected = PINNED[name]
        for key, value in expected.items():
            with self.subTest(case=name, field=key):
                if key == "author_stats":
                    self.assertEqual(sorted(result[key]), sorted(value))
                    for auth, stats in value.items():
                        for stat_key, stat_val in stats.items():
                            self.assertAlmostEqual(result[key][auth][stat_key], stat_val,
                                                   delta=NUM_TOL.get(stat_key, 0.0))
                elif key in NUM_TOL:
                    self.assertAlmostEqual(result[key], value, delta=NUM_TOL[key])
                else:
                    self.assertEqual(result[key], value)

    def test_mast_search_error(self):
        cand = candidate()
        result, written = run_drill(None, cand, search_error=ConnectionError("MAST unreachable"))
        self.assertEqual(result, {"status": "ARCHIVAL_ERROR", "reason": "MAST query failed: MAST unreachable",
                                  "candidate": cand})
        self.assertFalse(written)

    def test_no_light_curves_on_mast(self):
        cand = candidate()
        result, written = run_drill([], cand)
        self.assertEqual(result, {"status": "ARCHIVAL_REJECT", "reason": "No MAST light curves", "candidate": cand})
        self.assertFalse(written)

    def test_nothing_extractable_is_error(self):
        # S3.9 (C1g): unknown author (never downloaded, not a failure), failing download
        # and too few cadences are both recorded; with every product failed nothing was
        # evaluated, so the drill ends ARCHIVAL_ERROR (was ARCHIVAL_REJECT "No valid light curves extracted").
        unknown = item("K2SFF", 5, 2000e-6, seed=1)
        failing = FakeItem("SPOC", 5, 120, error=OSError("HTTP 503"))
        short = item("QLP", 6, 2000e-6, seed=2, n_points=60)
        cand = candidate()
        result, written = run_drill([unknown, failing, short], cand)
        self.assertEqual(result, {
            "status": "ARCHIVAL_ERROR",
            "reason": "All 2 light curve product(s) failed to download or prepare",
            "candidate": cand,
            "products_failed": [
                {"sector": 5, "author": "SPOC", "stage": "DOWNLOAD", "reason": "OSError: HTTP 503"},
                {"sector": 6, "author": "QLP", "stage": "EXTRACTION",
                 "reason": "fewer than 80 valid points after quality and flux cuts"},
            ]})
        self.assertEqual((unknown.downloads, failing.downloads, short.downloads), (0, 1, 1))
        self.assertFalse(written)

    def test_spoc_stack_pass_with_consistent_qlp(self):
        search = [item("SPOC", 5, 2000e-6, 21), item("SPOC", 6, 2000e-6, 22),
                  item("QLP", 5, 1900e-6, 23), item("QLP", 6, 1900e-6, 24)]
        result, written = run_drill(search)
        self.assertTrue(written)
        self.assert_pinned("spoc_pass", result)

    def test_duplicate_short_cadence_item_is_not_downloaded(self):
        dup = item("SPOC", 5, 2000e-6, 25, exptime=20)
        search = [item("SPOC", 5, 2000e-6, 21), dup, item("SPOC", 6, 2000e-6, 22)]
        result, _ = run_drill(search)
        self.assertEqual(dup.downloads, 0)
        self.assert_pinned("spoc_only_pass", result)

    def test_blended_reject(self):
        search = [item("SPOC", 5, 2000e-6, 31), item("SPOC", 6, 2000e-6, 32),
                  item("QLP", 5, 100e-6, 33), item("QLP", 6, 100e-6, 34)]
        result, written = run_drill(search)
        self.assertTrue(written)
        self.assert_pinned("blended", result)

    def test_same_dilution_with_low_contamination_passes(self):
        search = [item("SPOC", 5, 2000e-6, 31), item("SPOC", 6, 2000e-6, 32),
                  item("QLP", 5, 100e-6, 33), item("QLP", 6, 100e-6, 34)]
        result, _ = run_drill(search, candidate(contam_pct=2.0))
        self.assert_pinned("diluted_low_contam", result)

    def test_low_snr_reject(self):
        search = [item("SPOC", 5, 0.0, 41), item("SPOC", 6, 0.0, 42)]
        result, _ = run_drill(search)
        self.assert_pinned("low_snr", result)

    def test_multi_author_stack_without_spoc(self):
        search = [item("QLP", 5, 2000e-6, 51), item("TGLC", 6, 2000e-6, 52)]
        result, _ = run_drill(search)
        self.assert_pinned("multi_author", result)

    def test_failed_spoc_download_falls_back_to_other_authors(self):
        # S3.9 (C1g): the drill still passes on the QLP stack and the blend check is not
        # evaluated (verdict unchanged), but both SPOC download failures are now recorded.
        failing = [FakeItem("SPOC", 5, 120, error=OSError("HTTP 503")),
                   FakeItem("SPOC", 6, 120, error=OSError("HTTP 503"))]
        search = failing + [item("QLP", 5, 2000e-6, 61), item("QLP", 6, 2000e-6, 62)]
        result, _ = run_drill(search)
        self.assertEqual([f.downloads for f in failing], [1, 1])
        self.assert_pinned("spoc_download_failed", result)
        self.assertEqual(result["products_failed"], [
            {"sector": s, "author": "SPOC", "stage": "DOWNLOAD", "reason": "OSError: HTTP 503"} for s in (5, 6)])
        self.assertIn("2 light curve product(s) not used: S5 SPOC DOWNLOAD (OSError: HTTP 503)",
                      antares_packager.archival_decision_reason(result))


if __name__ == "__main__":
    unittest.main()
