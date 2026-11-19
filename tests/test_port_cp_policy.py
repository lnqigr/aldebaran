"""Boundary controls for the materialized, accepted Cp production functions.

No network, catalog, replay import, or real photometry is used here. Full-case
differential evidence is recorded separately by PORT-CP-01.
"""
import unittest
from unittest import mock

import numpy as np
from scipy import stats

import aldebaran as ald


class AcceptedCpPolicy(unittest.TestCase):
    def test_public_entrypoint_forwards_explicit_v2(self):
        expected = ('1', 'REJECT_SDE', 'fixture', None)
        with mock.patch.object(ald, '_analyze_single_target_impl', return_value=expected) as run, \
                mock.patch.object(ald, 'AUTO_PURGE_CACHE', False):
            self.assertEqual(ald.analyze_single_target('1', 1, odd_even_method='v2'), expected)
        self.assertEqual(run.call_args.kwargs['odd_even_method'], 'v2')

    def test_review_cannot_be_written_to_candidate_report(self):
        with self.assertRaisesRegex(ValueError, 'Non-candidate disposition'):
            ald.save_candidates_to_report('/tmp/port-cp-must-not-be-created.csv', [{'disposition': 'REVIEW'}])

    def test_circular_threshold_has_requested_familywise_tail(self):
        for period, duration in ((1., .1), (9., .05), (3., .15)):
            z, n = ald.vr_z_crit_circ(period, duration)
            self.assertAlmostEqual(1. - (1. - stats.norm.sf(z)) ** n, .001, places=13)

    def test_timing_uses_pivot_and_per_sector_extrapolation(self):
        time = np.linspace(100., 127., 1000)
        archival = [(1, np.linspace(1., 27., 1000)), (2, np.linspace(40., 67., 1000))]
        got = ald.vr_timing(101., 3., .1, 10., time, archival)
        shifted = ald.vr_timing(101. + 30., 3., .1, 10., time, archival)
        self.assertEqual(got, shifted)
        self.assertGreater(got['sigma_pred'][1], got['sigma_pred'][2])
        self.assertFalse(got['mode1'])

    def test_full_transit_count_excludes_under_supported_events(self):
        t = np.concatenate([np.arange(c - .04, c + .04, 2. / 1440.) for c in (1., 4.)] + [np.array([7.])])
        self.assertEqual(ald.vr_count_full_transits(t, 3., 1., .1), 2)

    @staticmethod
    def pixel_case(row, col, significance):
        ap = np.zeros((5, 5), dtype=bool)
        ap[2, 2] = True
        d = np.zeros_like(ap, dtype=float)
        sig = np.ones_like(d)
        d[2, 2] = 5.
        d[row, col] = significance
        return d, sig, d / sig, ap

    def test_interior_off_target_rejects_but_border_holds(self):
        interior = ald.vr_tpf_decide(*self.pixel_case(1, 1, 12.))
        border = ald.vr_tpf_decide(*self.pixel_case(0, 1, 12.))
        self.assertEqual(interior[0], 'REJECT')
        self.assertIn('Off-target source!', interior[1])
        self.assertEqual(border[0], 'REVIEW')
        self.assertTrue(border[1].startswith('REVIEW_TPF_EDGE_OR_MARGINAL'))

    def test_weak_localization_is_not_a_dipole_rejection(self):
        ap = np.ones((2, 2), dtype=bool)
        d = np.array([[2., -5.], [0., 0.]])
        sig = np.ones_like(d)
        result = ald.vr_tpf_decide(d, sig, d / sig, ap)
        self.assertEqual(result[:2], ('PASS', 'OK_WEAK_LOCALIZATION'))

    def test_significant_aperture_dipole_holds(self):
        ap = np.ones((2, 2), dtype=bool)
        d = np.array([[8., -5.], [0., 0.]])
        sig = np.ones_like(d)
        result = ald.vr_tpf_decide(d, sig, d / sig, ap)
        self.assertEqual(result[0], 'REVIEW')
        self.assertTrue(result[1].startswith('REVIEW_TPF_DIPOLE_JITTER'))

    def test_constrained_search_obeys_epoch_window(self):
        t = np.arange(0., 12., .01)
        flux = np.ones(t.size)
        flux[np.abs((t - .7 + 1.5) % 3. - 1.5) < .05] -= .002
        result = ald.vr_constrained_box_search(t, flux, np.full(t.size, .001), [3.], [.1], .7, .02, .01)
        self.assertLessEqual(abs(result['t0'] - .7), .02 + 1e-12)
        self.assertEqual(result['period'], 3.)
        self.assertGreater(result['power'], 0.)


if __name__ == '__main__':
    unittest.main()
