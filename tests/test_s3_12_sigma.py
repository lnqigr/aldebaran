"""S3.12-SIGMA-P-REF-01: sigma_P,ref (ARGON_DECISIONS items 36.1 and 42) and the SCI-78 ratios (items 36.2 and 42.3).

Diagnostic only: the archival decision must not change. The hand-checked fixture has known in- and out-of-transit
counts, so the box SNR, sigma_T0,ref and sigma_P,ref are worked out by hand below, not by calling the formula.
"""
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import aldebaran as ald

try:  # S3.8 end-to-end fixture (analyze_single_target with a temporary OUTPUT_DIR)
    from tests.test_s3_8_centroid import TargetRun, TPF_OK, TPF_HOLD
except ImportError:  # run with -s tests
    from test_s3_8_centroid import TargetRun, TPF_OK, TPF_HOLD

P, T0, T14 = 4.0, 0.0, 0.2
EPOCHS = (0, 1, 2, 250, 251)          # three discovery transits, two archival transits about 1000 d later


def hand_fixture(depth=0.002, sigma=0.001):
    """10 in-transit cadences (|phase| <= 0.045 d < T14/2) and 5 out-of-transit cadences (|phase| >= 1 d) per epoch."""
    t, y = [], []
    for k in EPOCHS:
        for off in np.linspace(-0.045, 0.045, 10):
            t.append(k * P + off); y.append(1.0 - depth)
        for off in (1.0, 1.5, 2.0, 2.5, 3.0):
            t.append(k * P + off); y.append(1.0)
    t, y = np.array(t), np.array(y)
    return t, y, np.full(t.size, sigma)


def box(t, period, t0, half_width, depth):
    tau = (t - t0 + 0.5 * period) % period - 0.5 * period
    out = np.zeros_like(t)
    out[np.abs(tau) < half_width] = depth
    return out


class SigmaPRefHandChecked(unittest.TestCase):
    def test_hand_checked_value(self):
        t, y, dy = hand_fixture()
        out = ald.sigma_p_ref_combined(t, y, dy, P, T0, T14)
        # n_in = 5 x 10 = 50, n_out = 5 x 5 = 25, equal weights: depth 0.002, error 0.001 sqrt(1/50 + 1/25)
        snr = 0.002 / (0.001 * math.sqrt(0.06))                    # 8.16496580927726, above the floor of 5
        self.assertAlmostEqual(snr, 8.16496580927726, places=12)
        # sigma_T0^2 = (0.1 / SNR)^2 + 0.0045^2 / 12 + 0.016^2 = 1.5e-4 + 1.6875e-6 + 2.56e-4
        var_t0 = 1.5e-4 + 1.6875e-6 + 2.56e-4                       # 4.076875e-4
        baseline = 251 * 4.0 + 3.0 + 0.045                          # 1007.045 d, from the first to the last cadence
        n_cyc = baseline / 4.0                                      # 251.76125
        self.assertEqual(out['status'], 'DEFINED')
        self.assertAlmostEqual(out['snr_comb'], snr, places=9)
        self.assertAlmostEqual(out['sigma_t0_ref_d'], math.sqrt(var_t0), places=12)
        self.assertAlmostEqual(out['baseline_comb_d'], baseline, places=9)
        self.assertAlmostEqual(out['n_cycles_comb'], n_cyc, places=9)
        self.assertAlmostEqual(out['sigma_p_ref_d'], 2.778212332165877e-4, places=15)
        self.assertAlmostEqual(out['sigma_p_ref_d'], math.sqrt(12.0 * var_t0) / n_cyc, places=15)
        self.assertEqual(out['n_transits_observed'], 5)
        self.assertEqual((out['p_ref_d'], out['t0_ref_btjd'], out['t14_ref_d']), (P, T0, T14))

    def test_snr_floor_of_five(self):
        t, y, dy = hand_fixture(depth=0.0005)           # box SNR = 0.0005 / 2.449e-4 = 2.04 < 5
        out = ald.sigma_p_ref_combined(t, y, dy, P, T0, T14)
        self.assertLess(out['snr_comb'], 5.0)
        var_t0 = (0.1 / 5.0) ** 2 + 0.0045 ** 2 / 12.0 + 0.016 ** 2
        self.assertAlmostEqual(out['sigma_t0_ref_d'], math.sqrt(var_t0), places=12)

    def test_longer_baseline_narrows_sigma(self):
        t, y, dy = hand_fixture()
        disc = t < 100.0
        short = ald.sigma_p_ref_combined(t[disc], y[disc], dy[disc], P, T0, T14)
        full = ald.sigma_p_ref_combined(t, y, dy, P, T0, T14)
        self.assertEqual(short['n_transits_observed'], 3)
        self.assertGreater(short['sigma_p_ref_d'], 50 * full['sigma_p_ref_d'])

    def test_not_defined_states(self):
        t, y, dy = hand_fixture()
        for p, t0, t14 in ((float('nan'), T0, T14), (P, T0, None), (P, T0, 0.0), (-1.0, T0, T14), (P, None, T14)):
            self.assertEqual(ald.sigma_p_ref_combined(t, y, dy, p, t0, t14)['status'], 'NOT_DEFINED_INVALID_EPHEMERIS')
        # every cadence in transit: no out-of-transit weight, so no box SNR
        out = ald.sigma_p_ref_combined(t, y, dy, P, T0, 100.0)
        self.assertEqual(out['status'], 'NOT_DEFINED_NO_BOX_SNR')
        self.assertIsNone(out['sigma_p_ref_d'])

    def test_exception_is_recorded_not_raised(self):
        t, y, dy = hand_fixture()
        out = ald.sigma_p_ref_combined(t, y[:-3], dy, P, T0, T14)   # mismatched lengths
        self.assertEqual(out['status'], 'NOT_DEFINED_ERROR')
        self.assertIn('error', out)


class Sci78Ratios(unittest.TestCase):
    def test_bls_ratio_uses_disc_sigma_and_strict_boundary(self):
        # exactly 3.5 sigma (binary-exact values: 0.875 / 0.25) is not "more than 3.5 sigma"
        out = ald.sci78_period_offsets(4.0, 4.875, 0.25, 1e-5)
        self.assertEqual(out['bls_offset_disc_sigma'], 3.5)
        self.assertFalse(out['bls_above_n_sigma'])
        out = ald.sci78_period_offsets(4.0, 3.99929, 2e-4, 1e-5)
        self.assertAlmostEqual(out['bls_offset_disc_sigma'], 3.55, places=9)
        self.assertTrue(out['bls_above_n_sigma'])
        self.assertNotIn('cat_offset_d', out)

    def test_cat_ratios(self):
        out = ald.sci78_period_offsets(4.0, 4.0, 2e-4, 3e-4, p_cat=4.002, sigma_p_cat_d=4e-4)
        self.assertAlmostEqual(out['cat_offset_d'], 2e-3, places=12)
        self.assertAlmostEqual(out['cat_offset_combined_sigma'], 2e-3 / 5e-4, places=9)   # sqrt(3^2 + 4^2) = 5
        self.assertAlmostEqual(out['cat_offset_ref_sigma'], 2e-3 / 3e-4, places=9)
        self.assertAlmostEqual(out['cat_offset_disc_sigma'], 10.0, places=9)
        self.assertTrue(out['cat_above_n_sigma'])
        self.assertEqual(out['bls_offset_disc_sigma'], 0.0)
        self.assertFalse(out['bls_above_n_sigma'])

    def test_injected_truth_and_missing_sigmas(self):
        out = ald.sci78_period_offsets(4.0, 4.0, 2e-4, 3e-4, p_cat=4.0009, sigma_p_cat_d=0.0)
        self.assertAlmostEqual(out['cat_offset_combined_sigma'], 3.0, places=9)
        self.assertFalse(out['cat_above_n_sigma'])
        out = ald.sci78_period_offsets(4.0, 4.0, None, None, p_cat=4.001, sigma_p_cat_d=None)
        self.assertIsNone(out['bls_offset_disc_sigma'])
        self.assertIsNone(out['cat_offset_combined_sigma'])
        self.assertIsNone(out['cat_offset_ref_sigma'])
        self.assertIsNone(out['cat_above_n_sigma'])


class Sci74WithSigmaPRef(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(1000., 1027., 19440)
        rng = np.random.default_rng(3)
        self.f = (1. - box(self.t, 4.0, 1.0, .06, .002) - box(self.t, 4.0, 3.0, .06, .0003)
                  + rng.normal(0., .0005, self.t.size))
        self.sec = {}
        self.decision = ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, 4.0, 1.0, .15, 2000.,
                                                                  details=self.sec)

    def snr(self, period):
        d = {}
        ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, period, 1.0, .15, 2000., details=d)
        return d['circ_snr']

    def test_ref_points_flip_and_complete(self):
        info = dict(status='DEFINED', sigma_p_ref_d=3e-4)
        out = ald.sci74_secondary_stability(self.t, self.f, 4.0, 1.0, .15, 2000., self.sec, 1e-4, 2e-4, 4.0 - 8e-4,
                                            grid_source='FINE', sigma_p_ref_info=info)
        self.assertEqual(out['status'], 'COMPLETE')
        self.assertEqual(out['sigma_p_ref_status'], 'DEFINED')
        self.assertIs(out['sigma_p_ref'], info)
        for key, period in (('P_ref-ref_sigma', 4.0 - 3e-4), ('P_ref+ref_sigma', 4.0 + 3e-4),
                            ('P_ref-disc_sigma', 4.0 - 2e-4), ('P_ref+disc_sigma', 4.0 + 2e-4)):
            self.assertEqual(out['points'][key]['period_d'], period, key)
            self.assertEqual(out['points'][key]['circ_snr'], self.snr(period), key)
        zc = self.sec['z_crit_circ']
        ref = [out['points'][k]['circ_snr'] for k in ('P_ref', 'P_ref-step', 'P_ref+step', 'P_ref-ref_sigma',
                                                      'P_ref+ref_sigma')]
        self.assertEqual(out['ref_sigma_flip'], max(ref) >= zc and min(ref) < zc)
        self.assertEqual(out['snr_sec_at_sigma_p_ref'], dict(minus=self.snr(4.0 - 3e-4), plus=self.snr(4.0 + 3e-4)))
        self.assertEqual(out['snr_sec_at_sigma_p_disc'], dict(minus=self.snr(4.0 - 2e-4), plus=self.snr(4.0 + 2e-4)))
        self.assertAlmostEqual(out['sci78']['bls_offset_disc_sigma'], 4.0, places=9)
        self.assertTrue(out['sci78']['bls_above_n_sigma'])
        self.assertAlmostEqual(out['refinement_offset_disc_sigma'], 4.0, places=9)   # S3.3 signed field kept
        self.assertEqual(ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, 4.0, 1.0, .15, 2000.),
                         self.decision)

    def test_ref_flip_detects_crossing_only_at_ref_points(self):
        # Controlled crossing: the step points stay above z_crit, a sigma_P,ref point falls below it.
        zc = self.sec['z_crit_circ']
        values = {4.0 - 1e-4: zc + 1, 4.0 + 1e-4: zc + 1, 4.0 - 3e-4: zc - 1, 4.0 + 3e-4: zc + 1}

        def fake(t, f, period, t0, dur, depth, details=None):
            details['circ_snr'] = values[period]
        sec = dict(self.sec, circ_snr=zc + 1)
        with mock.patch.object(ald, '_scan_full_orbit_secondary_eclipse_cp', side_effect=fake):
            out = ald.sci74_secondary_stability(self.t, self.f, 4.0, 1.0, .15, 2000., sec, 1e-4, None, 4.0,
                                                sigma_p_ref_info=dict(status='DEFINED', sigma_p_ref_d=3e-4))
        self.assertFalse(out['step_flip'])
        self.assertTrue(out['ref_sigma_flip'])
        self.assertIsNone(out['snr_sec_at_sigma_p_disc'])

    def test_not_defined_ref_stays_partial(self):
        for info in (None, dict(status='NOT_DEFINED_NO_BOX_SNR', sigma_p_ref_d=None),
                     dict(status='NOT_DEFINED_INVALID_EPHEMERIS', sigma_p_ref_d=3e-4),
                     dict(status='NOT_DEFINED_ERROR', error='ValueError')):
            out = ald.sci74_secondary_stability(self.t, self.f, 4.0, 1.0, .15, 2000., self.sec, 1e-4, 2e-4, 4.0,
                                                sigma_p_ref_info=info)
            self.assertEqual(out['status'], 'PARTIAL_SIGMA_P_REF_NOT_DEFINED')
            self.assertNotIn('P_ref+ref_sigma', out['points'])
            self.assertIsNone(out['ref_sigma_flip'])
            self.assertIsNone(out['snr_sec_at_sigma_p_ref'])
        self.assertEqual(out['sigma_p_ref_status'], 'NOT_DEFINED_ERROR')


class ProductionArchivalPath(unittest.TestCase):
    """verify_multi_epoch, Mode 2 and Mode 1, with the S3.3 Mode2Label fixture (refiner returns T14_ref = 0.16 d)."""

    def archival(self, arch_start=2458000, arch_span=27, refiner_duration=0.16):
        t_curr = np.linspace(2459000, 2459027, 2000)
        f_curr = 1. - box(t_curr, 3.5, 2459001., .45 * .15, .001)
        t_arch = np.linspace(arch_start, arch_start + arch_span, 2000)
        f_arch = 1. - box(t_arch, 3.5, 2459001., .45 * .15, .001)
        row = mock.MagicMock()
        row.mission = ['TESS Sector 80']
        row.obsid = 'TESS_s0080_label'
        ref = dict(period=3.5, t0=2459001., duration=refiner_duration, n_req=1, n_grid=60, stages=1,
                   grid_step_d=2e-4, grid_source='SINGLE')
        with mock.patch('lightkurve.search_lightcurve', return_value=[row]), \
                mock.patch.object(ald, 'safe_download', return_value=mock.MagicMock()), \
                mock.patch.object(ald, 'extract_and_stitch_arrays', return_value=(
                    t_arch, f_arch, np.full(2000, .0001), np.zeros(2000), np.zeros(2000), .95)), \
                mock.patch.object(ald, 'clip_upper_outliers', side_effect=lambda t, f, e: (t, f, e)), \
                mock.patch.object(ald, 'piecewise_savgol_detrend', side_effect=lambda t, f, **kw: np.ones_like(f)), \
                mock.patch.object(ald, 'vr_refine_p3', return_value=ref):
            return ald.verify_multi_epoch('34567890', 106, t_curr, f_curr, np.full(2000, .0001),
                                          best_p=3.5, best_t0=2459001., best_dur=.15, depth_ppm_curr=1000., sde_curr=8.)

    @staticmethod
    def without_sci74(info):
        out = dict(info)
        out['vr'] = {k: v for k, v in info['vr'].items() if k != 'sci74'}
        return out

    def test_mode2_defines_sigma_p_ref_at_refined_ephemeris(self):
        ok, status, info = self.archival()
        self.assertTrue(ok, status)
        s74 = info['vr']['sci74']
        ref = s74['sigma_p_ref']
        self.assertEqual(ref['status'], 'DEFINED')
        self.assertEqual((ref['p_ref_d'], ref['t0_ref_btjd'], ref['t14_ref_d']), (3.5, 2459001., 0.16))
        self.assertAlmostEqual(ref['baseline_comb_d'], 2459027. - 2458000., places=6)
        self.assertEqual(s74['status'], 'COMPLETE')
        self.assertEqual(s74['points']['P_ref+ref_sigma']['period_d'], 3.5 + ref['sigma_p_ref_d'])
        # the combined baseline is far longer than the discovery sector, so sigma_P,ref < the corridor sigma
        self.assertLess(ref['sigma_p_ref_d'], s74['sigma_p_disc_projected_d'])

    def test_decision_and_other_fields_unchanged(self):
        base = self.archival()
        with mock.patch.object(ald, '_sigma_p_ref_combined', side_effect=RuntimeError('injected')):
            faulted = self.archival()
        self.assertEqual(faulted[1], base[1])
        self.assertEqual(faulted[0], base[0])
        self.assertEqual(faulted[2]['vr']['sci74']['sigma_p_ref_status'], 'NOT_DEFINED_ERROR')
        self.assertEqual(faulted[2]['vr']['sci74']['status'], 'PARTIAL_SIGMA_P_REF_NOT_DEFINED')
        a, b = self.without_sci74(base[2]), self.without_sci74(faulted[2])
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_equal(a[key], b[key], err_msg=key)

    def test_mode1_projection_is_not_applicable(self):
        ok, status, info = self.archival(arch_start=2459028, arch_span=4)
        self.assertTrue(ok, status)
        self.assertTrue(info['is_projected_strict'])
        self.assertEqual(info['vr']['sci74']['sigma_p_ref_status'], 'NOT_APPLICABLE_NO_REFINEMENT')


class TpfSpatialPassField(TargetRun, unittest.TestCase):
    """Record-only field for the S3.13 Stage 3B cap (item 34.1): Filter 6 spatial check passed (True) or held (False)."""

    def test_passed_tpf_check_is_true_and_written_to_report(self):
        status, _, cand = self.run_target(self.dx, self.dy, tpf=TPF_OK)
        self.assertEqual(status, 'CANDIDATE')
        self.assertIs(cand['tpf_spatial_pass'], True)
        import pandas as pd
        path = os.path.join(self.out_dir, 'report.csv')
        ald.save_candidates_to_report(path, [cand])
        df = pd.read_csv(path)
        self.assertEqual(df['tpf_spatial_pass'].tolist(), [True])
        self.assertEqual(list(df.columns[-2:]), ['centroid_state', 'tpf_spatial_pass'])

    def test_tpf_review_hold_writes_no_candidate_record(self):
        # A Filter 6 REVIEW_ hold sets tpf_spatial_pass False on the internal record, but the host ends REVIEW and no
        # candidate record is returned or written, so every candidates_report row carries True today.
        status, _, cand = self.run_target(self.dx, self.dy, tpf=TPF_HOLD)
        self.assertEqual(status, 'REVIEW_VETTING_HOLD')
        self.assertIsNone(cand)

    def test_column_absent_without_field(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'c.csv')
            rec = {c: 1 for c in ald.CANDIDATE_COLUMNS}
            ald.save_candidates_to_report(path, [rec])
            import pandas as pd
            self.assertNotIn('tpf_spatial_pass', pd.read_csv(path).columns)


if __name__ == '__main__':
    unittest.main()
