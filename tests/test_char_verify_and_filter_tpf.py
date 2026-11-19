"""Characterization tests for aldebaran.verify_and_filter_tpf (CHAR-TESTS-01, plan 008).

They pin the current return values of every branch on small noise-free
synthetic pixel cubes. MAST access (lightkurve search and download) and the
cache purge are mocked; nothing touches the network or the real cache.
These tests describe present behaviour, not desired behaviour: a later task
that changes a branch on purpose updates the matching assertion.
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aldebaran

P_VAL = 2.0
T0_VAL = 1000.5
DUR_VAL = 0.1
SHAPE = (5, 5)


class FakeTPF:
    def __init__(self, time, flux, pipeline_mask):
        self.time = time
        self.flux = flux
        self.pipeline_mask = pipeline_mask


class FakeItem:
    def __init__(self, tpf=None, error=None):
        self.tpf = tpf
        self.error = error
        self.calls = 0

    def download(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.tpf


def centre_aperture():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1:4, 1:4] = True
    return mask


def cube(time, in_transit_delta):
    """Flux 100 e-/s everywhere; in-transit cadences get in_transit_delta added."""
    flux = np.full((len(time),) + SHAPE, 100.0)
    phase = ((time - T0_VAL + 0.5 * P_VAL) % P_VAL) - 0.5 * P_VAL
    in_tr = np.abs(phase) < 0.45 * DUR_VAL
    flux[in_tr] += in_transit_delta
    return flux


def default_time():
    return 1000.0 + np.arange(0.0, 10.0, 2.0 / 1440.0)


class TestVerifyAndFilterTpfCharacterization(unittest.TestCase):
    def run_check(self, search_result, purge=None):
        with mock.patch.object(aldebaran.lk, "search_targetpixelfile", return_value=search_result) as search, \
             mock.patch.object(aldebaran, "purge_corrupted_cache") as purge_mock:
            result = aldebaran.verify_and_filter_tpf("12345", 7, P_VAL, T0_VAL, DUR_VAL)
        search.assert_called_once_with("TIC 12345", mission="TESS", author="SPOC", sector=7, exptime=120)
        if purge is not None:
            self.assertEqual(purge_mock.call_count, purge)
        return result

    def run_tpf(self, flux, mask, time=None):
        time = default_time() if time is None else time
        return self.run_check([FakeItem(FakeTPF(time, flux, mask))], purge=0)

    def test_no_tpf_on_mast(self):
        self.assertEqual(self.run_check([]), (False, "TPF unavailable on MAST", None, None))

    def test_empty_pipeline_aperture(self):
        delta = np.zeros(SHAPE)
        result = self.run_tpf(cube(default_time(), delta), np.zeros(SHAPE, dtype=bool))
        self.assertEqual(result, (False, "Empty SPOC pipeline aperture mask", None, None))

    def test_insufficient_cadences_in_transit_window(self):
        # Two days of data that never reach phase |0.45 * dur| around T0.
        time = 1001.2 + np.arange(0.0, 0.5, 2.0 / 1440.0)
        delta = np.zeros(SHAPE)
        result = self.run_tpf(cube(time, delta), centre_aperture(), time=time)
        self.assertEqual(result, (False, "Insufficient TPF cadences in transit window", None, None))

    def test_off_target_drop_outside_aperture(self):
        delta = np.zeros(SHAPE)
        delta[0, 4] = -30.0
        ok, reason, diff, aperture = self.run_tpf(cube(default_time(), delta), centre_aperture())
        self.assertFalse(ok)
        self.assertEqual(
            reason,
            "Off-target source! Max flux drop at pixel (col=4, row=0) outside aperture -> BEB",
        )
        np.testing.assert_allclose(diff, -delta, atol=1e-9)
        np.testing.assert_array_equal(aperture, centre_aperture())

    def test_dipole_inside_aperture(self):
        delta = np.zeros(SHAPE)
        delta[2, 2] = -20.0
        delta[1, 2] = 10.0
        ok, reason, diff, _ = self.run_tpf(cube(default_time(), delta), centre_aperture())
        self.assertFalse(ok)
        self.assertEqual(
            reason,
            "Dipole jitter inside aperture (max=20.0, min=-10.0 e-/s) -> Pointing drift / Variable",
        )
        np.testing.assert_allclose(diff, -delta, atol=1e-9)

    def test_flat_image_with_aperture_at_argmax_is_labelled_dipole(self):
        # All differences are zero: argmax is pixel (0, 0); with that pixel in the
        # aperture the max_ap <= 0 clause of the dipole branch fires.
        mask = centre_aperture()
        mask[0, 0] = True
        ok, reason, diff, _ = self.run_tpf(cube(default_time(), np.zeros(SHAPE)), mask)
        self.assertFalse(ok)
        self.assertEqual(
            reason,
            "Dipole jitter inside aperture (max=0.0, min=0.0 e-/s) -> Pointing drift / Variable",
        )
        self.assertTrue(np.all(diff == 0.0))

    def test_weak_localization_against_background_scatter(self):
        delta = np.zeros(SHAPE)
        delta[2, 2] = -2.0
        mask = centre_aperture()
        # Background pixels alternate +1.9 / -1.9 e-/s; their std (1.9) times 2.8 exceeds 2.0,
        # while every background drop stays below the aperture peak.
        sign = 1.0
        for idx in zip(*np.where(~mask)):
            delta[idx] = sign * 1.9
            sign = -sign
        ok, reason, _, _ = self.run_tpf(cube(default_time(), delta), mask)
        self.assertFalse(ok)
        self.assertEqual(
            reason,
            "Weak pixel localization (peak=2.0 vs bg_std=1.9 e-/s) -> Background scatter",
        )

    def test_clean_on_target_drop_passes(self):
        delta = np.zeros(SHAPE)
        delta[2, 2] = -50.0
        delta[2, 3] = -5.0
        ok, reason, diff, aperture = self.run_tpf(cube(default_time(), delta), centre_aperture())
        self.assertEqual((ok, reason), (True, "OK"))
        np.testing.assert_allclose(diff, -delta, atol=1e-9)
        np.testing.assert_array_equal(aperture, centre_aperture())

    def test_nan_pixels_become_zero_difference(self):
        flux = cube(default_time(), np.where(np.arange(25).reshape(SHAPE) == 12, -50.0, 0.0))
        flux[:, 0, 0] = np.nan
        ok, reason, diff, _ = self.run_tpf(flux, centre_aperture())
        self.assertEqual((ok, reason), (True, "OK"))
        self.assertEqual(diff[0, 0], 0.0)

    def test_search_exception_is_read_error(self):
        with mock.patch.object(aldebaran.lk, "search_targetpixelfile",
                               side_effect=RuntimeError("MAST is down for maintenance today, retry later")):
            result = aldebaran.verify_and_filter_tpf("12345", 7, P_VAL, T0_VAL, DUR_VAL)
        # The message is cut to its first 40 characters.
        # S3.1 (ARGON_DECISIONS item 29.2): a read error is "not evaluated" (None), not a pixel reject (False).
        self.assertEqual(result, (None, "TPF read error: MAST is down for maintenance today, retr", None, None))

    def test_download_failure_purges_cache_once_then_is_read_error(self):
        item = FakeItem(error=OSError("corrupt FITS"))
        result = self.run_check([item], purge=1)
        self.assertEqual(item.calls, 2)
        self.assertEqual(result, (None, "TPF read error: corrupt FITS", None, None))  # S3.1, item 29.2

    def test_download_retry_after_purge_succeeds(self):
        delta = np.zeros(SHAPE)
        delta[2, 2] = -50.0
        tpf = FakeTPF(default_time(), cube(default_time(), delta), centre_aperture())

        class FlakyItem(FakeItem):
            def download(self):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("first read truncated")
                return tpf

        item = FlakyItem()
        ok, reason, _, _ = self.run_check([item], purge=1)
        self.assertEqual((ok, reason, item.calls), (True, "OK", 2))


if __name__ == "__main__":
    unittest.main()
