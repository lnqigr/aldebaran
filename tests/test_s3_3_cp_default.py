"""S3.3-CP-DEFAULT-01: Cp-AR default, all-ranks journal, fail-closed status, Mode 2 label and SCI-74 diagnostics.

Offline fixtures only. Production functions run unpatched except for I/O and the expensive search steps; the
identity of the full detector on saved inputs is recorded separately (agent_exchange/s3_3_cp_default).
"""
import contextlib
import glob
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import aldebaran as ald

T = np.linspace(1000., 1027., 19440)


def box(t, period, t0, half_width, depth):
    tau = (t - t0 + 0.5 * period) % period - 0.5 * period
    out = np.zeros_like(t)
    out[np.abs(tau) < half_width] = depth
    return out


class DetectorFixture:
    """Three BLS ranks on one light curve with a 3.0 d transit: rank 1 P = 3.0 d (the planet), rank 2 P = 1.5 d
    (contains every event of rank 1, so the frozen review_event_link links it) and rank 3 P = 4.7 d (no transit)."""

    periods = (3.0, 1.5, 4.7)

    def run(self, extra_patches=None, method=None):
        rng = np.random.default_rng(7)
        flux = 1. + rng.normal(0., .0001, T.size) - box(T, 3.0, 1001., .04, .001)
        centroid = rng.normal(0., .001, T.size)
        n = 9000
        period = np.full(n, 3.0)
        period[:3] = self.periods
        bls = SimpleNamespace(power=lambda *a, **k: SimpleNamespace(
            period=period, power=np.full(n, 100.), duration=np.full(n, .08),
            transit_time=np.full(n, 1001.), depth=np.full(n, .001)))
        arch = dict(n_archival_sectors=0, archival_sectors_str='', refined_p=np.nan, refined_t0=np.nan,
                    archival_snr=np.nan, multi_sde=8.5, est_multi_sde_proxy=8.5, n_trials=1, snr_crit=2.4,
                    ephemeris_source='DISCOVERY_SINGLE_SECTOR', ephemeris_status='FALLBACK_DISCOVERY',
                    ephemeris_fallback_reason='NO_PRIOR_SECTORS')
        oe = ald.OddEvenV2Result('ASSESSED', 'fixture', 0., 1000., 1000., 10., 10., {})
        td = tempfile.mkdtemp(prefix='s33-ar-')
        patches = dict(AUTO_PURGE_CACHE=False, OUTPUT_DIR=td, ARCHIVAL_DIAGNOSTICS_FILE=td + '/archival.jsonl',
            get_known_transits_for_tic=lambda _: [], safe_download=lambda *a, **k: object(),
            extract_and_stitch_arrays=lambda *a, **k: (T, flux, np.full(T.size, .0001), centroid, centroid, .95),
            clip_upper_outliers=lambda t, f, e: (t, f, e),
            piecewise_savgol_detrend=lambda t, f, **k: np.ones_like(f),
            compute_whitened_kovacs_sde=lambda *a, **k: (np.full(n, 8.5), None),
            get_top_distinct_bls_peaks=lambda *a, **k: [0, 1, 2],
            get_stellar_and_gaia_params=lambda _: dict(has_params=True, r_star=1., e_r_star=.05, m_star=1.,
                                                      e_m_star=.05, teff=5800., ruwe=1.05),
            BoxLeastSquares=lambda *a, **k: bls,
            _scan_full_orbit_secondary_eclipse_cp=lambda *a, **k: (False, 0., 0., .5, 3.),
            _verify_and_filter_tpf_cp=lambda *a, **k: (True, 'OK', np.ones((5, 5)), np.ones((5, 5), dtype=bool)),
            verify_multi_epoch=lambda *a, **k: (True, 'Single-Sector_Only', dict(arch)),
            compute_odd_even_v2=lambda *a, **k: oe)
        patches.update(extra_patches or {})
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(ald, name, value))
            stack.enter_context(mock.patch.object(ald.lk, 'search_lightcurve', return_value=[object()]))
            if method is None:
                result = ald.analyze_single_target('12345678', 106)
            else:
                result = ald.analyze_single_target('12345678', 106, odd_even_method=method)
        journal_path = os.path.join(td, ald.AR_JOURNAL_FILENAME)
        def lines(path):
            with open(path) as handle:
                return [json.loads(x) for x in handle]
        journal = lines(journal_path) if os.path.exists(journal_path) else []
        evidence = lines(os.path.join(td, 'experimental_v2_vetting.jsonl'))
        return result, journal, evidence, td


class AllRanksJournal(unittest.TestCase):
    def test_evaluation_continues_after_first_candidate(self):
        result, journal, evidence, td = DetectorFixture().run()
        self.assertEqual(result[1], 'CANDIDATE', result[2])
        self.assertIn('Peak #1', result[2])
        self.assertEqual(result[3]['period_days'], 3.0)
        ranks = {r['rank']: r for r in journal if r['record'] == 'rank'}
        self.assertEqual(sorted(ranks), [1, 2, 3])
        self.assertEqual(ranks[1]['disposition'], 'CANDIDATE')
        # rank 2 shares all its events with the rank 1 CANDIDATE: REVIEW, never a second CANDIDATE (item 26.1)
        self.assertEqual(ranks[2]['disposition'], 'REVIEW')
        self.assertEqual(ranks[2]['cause'], 'SHARED_EVENTS_WITH_CANDIDATE')
        self.assertEqual(ranks[2]['linked_candidate_rank'], 1)
        end = [r for r in journal if r['record'] == 'host_end']
        self.assertEqual(len(end), 1)
        self.assertEqual(end[0]['first_candidate_rank'], 1)
        self.assertEqual(end[0]['evaluated_ranks'], [1, 2, 3])
        self.assertEqual(end[0]['not_evaluated_ranks'], [])
        # vetting evidence covers every evaluated rank with evidence; only one candidate plot is written
        peaks = {p['peak_rank']: p for p in evidence[-1]['peaks']}
        self.assertEqual(peaks[1]['disposition'], 'CANDIDATE')
        self.assertEqual(peaks[2]['disposition'], 'REVIEW')
        self.assertEqual(len(glob.glob(os.path.join(td, 'candidate_TIC_*.png'))), 1)

    def test_independent_later_candidate_keeps_first_record_and_plot(self):
        # Negative control for the plot/record guard: a later unlinked CANDIDATE peak (link check disabled) must not
        # replace the first CANDIDATE's record or write a second plot.
        result, journal, evidence, td = DetectorFixture().run(dict(review_event_link=lambda *a, **k: None))
        self.assertEqual(result[1], 'CANDIDATE')
        self.assertEqual(result[3]['period_days'], 3.0)
        ranks = {r['rank']: r for r in journal if r['record'] == 'rank'}
        self.assertEqual(ranks[2]['disposition'], 'CANDIDATE')
        self.assertEqual(len(glob.glob(os.path.join(td, 'candidate_TIC_*.png'))), 1)
        end = [r for r in journal if r['record'] == 'host_end'][0]
        self.assertEqual(end['first_candidate_rank'], 1)

    def test_exception_after_candidate_ends_error(self):
        # Fail closed (Cp-AR fault control): an exception while evaluating a later rank ends ERROR, not CANDIDATE,
        # and the journal then has no host_end.
        calls = {'n': 0}
        real = ald.review_event_link

        def link(*args, **kwargs):
            calls['n'] += 1
            if args[0].get('disposition') == 'CANDIDATE':
                raise RuntimeError('injected fault after CANDIDATE')
            return real(*args, **kwargs)

        result, journal, evidence, td = DetectorFixture().run(dict(review_event_link=link))
        self.assertEqual(result[1], 'ERROR', result[2])
        self.assertIn('injected fault after CANDIDATE', result[2])
        self.assertIsNone(result[3])
        self.assertEqual([r['rank'] for r in journal if r['record'] == 'rank'], [1])
        self.assertFalse([r for r in journal if r['record'] == 'host_end'])

    def test_low_sde_break_lists_unevaluated_ranks(self):
        sde = np.full(9000, 8.5)
        sde[1] = 5.0
        result, journal, evidence, td = DetectorFixture().run(dict(
            compute_whitened_kovacs_sde=lambda *a, **k: (sde, None)))
        self.assertEqual(result[1], 'CANDIDATE')
        ranks = [r for r in journal if r['record'] == 'rank']
        self.assertEqual(ranks[-1]['rank'], 2)
        self.assertEqual(ranks[-1]['exit_code'], 'EMPTY_LOW_SDE')
        self.assertEqual(ranks[-1]['stop'], 'break')
        end = [r for r in journal if r['record'] == 'host_end'][0]
        self.assertEqual(end['not_evaluated_ranks'], [3])


class Mode2Label(unittest.TestCase):
    def archival(self, arch_start=2458000, arch_span=27):
        t_curr = np.linspace(2459000, 2459027, 2000)
        f_curr = 1. - box(t_curr, 3.5, 2459001., .45 * .15, .001)
        t_arch = np.linspace(arch_start, arch_start + arch_span, 2000)
        f_arch = 1. - box(t_arch, 3.5, 2459001., .45 * .15, .001)
        row = mock.MagicMock()
        row.mission = ['TESS Sector 80']
        row.obsid = 'TESS_s0080_label'
        with mock.patch('lightkurve.search_lightcurve', return_value=[row]), \
                mock.patch.object(ald, 'safe_download', return_value=mock.MagicMock()), \
                mock.patch.object(ald, 'extract_and_stitch_arrays', return_value=(
                    t_arch, f_arch, np.full(2000, .0001), np.zeros(2000), np.zeros(2000), .95)), \
                mock.patch.object(ald, 'clip_upper_outliers', side_effect=lambda t, f, e: (t, f, e)), \
                mock.patch.object(ald, 'piecewise_savgol_detrend', side_effect=lambda t, f, **kw: np.ones_like(f)), \
                mock.patch.object(ald, 'vr_refine_p3', return_value=dict(period=3.5, t0=2459001., n_req=1, n_grid=60, stages=1,
                                                                         grid_step_d=2e-4, grid_source='SINGLE')):
            return ald.verify_multi_epoch('34567890', 106, t_curr, f_curr, np.full(2000, .0001),
                                          best_p=3.5, best_t0=2459001., best_dur=.15, depth_ppm_curr=1000., sde_curr=8.)

    def test_mode2_pass_carries_label_and_sci74(self):
        ok, status, info = self.archival()
        self.assertTrue(ok, status)
        self.assertTrue(status.startswith('Confirmed_in_Sectors_80'))
        self.assertFalse(info['is_projected_strict'])
        self.assertEqual(info['vr']['mode'], 2)
        self.assertEqual(info['archival_confirmation_label'],
                         'Multi-Epoch_Refined (Conditional on Archival Period Search)')
        s74 = info['vr']['sci74']
        self.assertEqual(s74['p_ref_d'], 3.5)
        self.assertEqual(set(s74['points']),
                         {'P_ref', 'P_ref-step', 'P_ref+step', 'P_ref-disc_sigma', 'P_ref+disc_sigma'})
        # the recorded step is the actual winning-grid spacing returned by the refiner, passed through unchanged
        self.assertEqual(info['vr']['p3_grid_step_d'], 2e-4)
        self.assertEqual(s74['grid_step_d'], 2e-4)
        self.assertEqual(s74['grid_step_source'], 'SINGLE')
        # S3.12: this refiner mock returns no duration, so T14_ref and sigma_P,ref are not defined
        self.assertEqual(s74['sigma_p_ref_status'], 'NOT_DEFINED_INVALID_EPHEMERIS')
        self.assertEqual(s74['status'], 'PARTIAL_SIGMA_P_REF_NOT_DEFINED')

    def test_mode1_projection_pass_has_no_label(self):
        # Item 12 Mode 1 (projected sigma_pred within a quarter duration, checked with the production vr_timing for this
        # window): strict forward projection, no conditional label.
        ok, status, info = self.archival(arch_start=2459028, arch_span=4)
        self.assertTrue(ok, status)
        self.assertTrue(info['is_projected_strict'])
        self.assertNotIn('archival_confirmation_label', info)

    def test_label_column_in_candidate_report(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'c.csv')
            rec = {c: 1 for c in ald.CANDIDATE_COLUMNS}
            rec.update(odd_even_method='ODD_EVEN_V2_EXPERIMENTAL', odd_even_status='ASSESSED',
                       odd_even_diagnostics_json='{}', archival_confirmation_label=ald.ARCHIVAL_MODE2_LABEL)
            ald.save_candidates_to_report(path, [rec])
            df = pd.read_csv(path)
            self.assertEqual(list(df.columns[-1:]), ['archival_confirmation_label'])
            self.assertEqual(df['archival_confirmation_label'][0], ald.ARCHIVAL_MODE2_LABEL)


class Sci74Diagnostic(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0., 80., 40000)
        # primary 2000 ppm at phase 0, circular secondary 300 ppm at phase 0.5, white noise 500 ppm
        rng = np.random.default_rng(3)
        self.f = (1. - box(self.t, 4.0, 1.0, .06, .002) - box(self.t, 4.0, 3.0, .06, .0003)
                  + rng.normal(0., .0005, self.t.size))

    def snr(self, period):
        d = {}
        ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, period, 1.0, .15, 2000., details=d)
        return d['circ_snr']

    def test_points_rule_unchanged_and_sigma_p_ref_not_claimed(self):
        sec = {}
        decision = ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, 4.0, 1.0, .15, 2000., details=sec)
        out = ald.sci74_secondary_stability(self.t, self.f, 4.0, 1.0, .15, 2000., sec, 1e-4, 2e-4, 4.0 - 4e-4,
                                            grid_source='FINE')
        # S3.12: without sigma_p_ref_info (no refined ephemeris) the result is never reported complete
        self.assertEqual(out['status'], 'PARTIAL_SIGMA_P_REF_NOT_DEFINED')
        self.assertEqual(out['sigma_p_ref_status'], 'NOT_APPLICABLE_NO_REFINEMENT')
        self.assertNotIn('P_ref+sigma', out['points'])
        self.assertEqual(out['grid_step_source'], 'FINE')
        self.assertEqual(out['points']['P_ref']['circ_snr'], sec['circ_snr'])
        zc = sec['z_crit_circ']
        # independent recomputation of every shifted point with the production scan
        expect = {'P_ref-step': 4.0 - 1e-4, 'P_ref+step': 4.0 + 1e-4,
                  'P_ref-disc_sigma': 4.0 - 2e-4, 'P_ref+disc_sigma': 4.0 + 2e-4}
        for key, period in expect.items():
            self.assertEqual(out['points'][key]['period_d'], period, key)
            self.assertEqual(out['points'][key]['circ_snr'], self.snr(period), key)
        step = [out['points'][k]['circ_snr'] for k in ('P_ref', 'P_ref-step', 'P_ref+step')]
        allp = [p['circ_snr'] for p in out['points'].values()]
        self.assertEqual(out['step_flip'], max(step) >= zc and min(step) < zc)
        self.assertEqual(out['disc_sigma_flip'], max(allp) >= zc and min(allp) < zc)
        self.assertEqual(out['near_threshold'], abs(sec['circ_snr'] - zc) <= .5)
        self.assertAlmostEqual(out['refinement_offset_disc_sigma'], 2.0)
        self.assertEqual(out['sigma_p_disc_projected_d'], 2e-4)
        # the diagnostic does not alter the decision of the scan it reports on
        self.assertEqual(ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, 4.0, 1.0, .15, 2000.), decision)

    def test_missing_inputs_are_not_available(self):
        out = ald.sci74_secondary_stability(self.t, self.f, 4.0, 1.0, .15, 2000., {}, None, float('nan'), 4.0)
        self.assertEqual(out['status'], 'NOT_AVAILABLE')
        self.assertIsNone(out['step_flip'])
        self.assertIsNone(out['disc_sigma_flip'])
        self.assertEqual(list(out['points']), ['P_ref'])

    def test_without_grid_step_no_flip_is_reported(self):
        sec = {}
        ald._scan_full_orbit_secondary_eclipse_cp(self.t, self.f, 4.0, 1.0, .15, 2000., details=sec)
        out = ald.sci74_secondary_stability(self.t, self.f, 4.0, 1.0, .15, 2000., sec, None, 2e-4, 4.0)
        self.assertEqual(out['status'], 'NOT_AVAILABLE')
        self.assertEqual(set(out['points']), {'P_ref', 'P_ref-disc_sigma', 'P_ref+disc_sigma'})
        self.assertIsNone(out['step_flip'])


class RefinerGridStep(unittest.TestCase):
    """P2-2 (Codex S3.3 round 1): the recorded step is the actual spacing of the grid that supplied the solution."""

    def run_refiner(self, fine_power):
        grids = []

        def scan(t, y, dy, periods, *a):
            return [(float(p), 1.0, .1, 1.0) for p in periods]  # equal power everywhere: argmax keeps index 0

        def search(t, y, dy, periods, *a):
            grids.append(np.asarray(periods))
            return dict(period=float(periods[len(periods) // 2]), t0=1.0, duration=.1, power=fine_power)
        with mock.patch.object(ald, 'vr_period_scan', side_effect=scan), \
                mock.patch.object(ald, 'vr_constrained_box_search', side_effect=search):
            # W = 0.2 d, n_max = 2000, dur = 0.1 d: n_req = 16001 > 4000, two stages
            r = ald.vr_refine_p3(None, None, None, 1.0, 1.2, 1, None, 1.0, .01, .001, 2000, .1)
        return r, grids

    def test_coarse_retained(self):
        r, grids = self.run_refiner(fine_power=1.0)  # not strictly greater: the coarse maximum is kept
        self.assertEqual(r['stages'], 2)
        self.assertEqual(r['period'], 1.0)
        self.assertEqual(r['grid_source'], 'COARSE')
        self.assertAlmostEqual(r['grid_step_d'], float(np.diff(np.linspace(1.0, 1.2, 4000))[0]), places=15)
        self.assertNotAlmostEqual(r['grid_step_d'], .1 / (8 * 2000), places=9)

    def test_fine_winner(self):
        r, grids = self.run_refiner(fine_power=2.0)  # the first fine grid wins; later ones only tie
        self.assertEqual(r['grid_source'], 'FINE')
        self.assertEqual(r['period'], float(grids[0][len(grids[0]) // 2]))
        self.assertAlmostEqual(r['grid_step_d'], float(np.diff(grids[0])[0]), places=15)

    def test_single_stage(self):
        with mock.patch.object(ald, 'vr_constrained_box_search',
                               side_effect=lambda t, y, dy, periods, *a: dict(period=1.0, t0=1.0, duration=.1, power=1.)):
            r = ald.vr_refine_p3(None, None, None, 1.0, 1.001, 1, None, 1.0, .01, .001, 10, .1)
        self.assertEqual((r['stages'], r['grid_source']), (1, 'SINGLE'))
        self.assertAlmostEqual(r['grid_step_d'], float(np.diff(np.linspace(1.0, 1.001, r['n_grid']))[0]), places=15)


if __name__ == '__main__':
    unittest.main()
