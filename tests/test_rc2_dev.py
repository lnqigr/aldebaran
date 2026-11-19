"""RC2 rules on the production cascade; switches stay off for existing callers."""
import json
import os
import contextlib
import unittest
from unittest.mock import patch

import numpy as np

import aldebaran as ald
from test_s3_3_cp_default import DetectorFixture, T, box
from m_harness import rc2_reporting as reporting


def photometry(depth, triangular=False, half_width=.04):
    phase = (T - 1001. + 1.5) % 3. - 1.5
    shape = np.maximum(0., 1. - np.abs(phase) / half_width) if triangular else box(T, 3., 1001., half_width, 1.)
    flux = 1. - depth * 1e-6 * shape + np.random.default_rng(7).normal(0., .0001, T.size)
    return lambda *a, **kw: (T, flux, np.full(T.size, .0001), np.zeros(T.size), np.zeros(T.size), .95)


def run(enabled=(), extra=None, trace=False):
    return DetectorFixture().run(dict(RC2_SWITCHES=frozenset(enabled), RC2_RECORD_ALL_GATES=trace, **(extra or {})))


class Rules(unittest.TestCase):
    def test_release_defaults_and_unknown_switch_is_refused(self):
        # RC2-FREEZE-01: the item 50.4 combination is the default; DEPTH_FULL_T14 stays off.
        self.assertEqual(ald.RC2_SWITCHES, frozenset({'A2', 'ARCHIVAL_COVERAGE', 'V_SHAPE_FLAG', 'DEPTH_UPPER'}))
        self.assertFalse(ald.RC2_RECORD_ALL_GATES)
        with patch.object(ald, 'RC2_SWITCHES', {'typo'}), self.assertRaises(ValueError):
            ald.rc2_switches()

    def test_a2_contract_and_other_failures_stay_review(self):
        diag = dict(n_odd=2, n_even=1, supported_orbits=[0, 1, 2], reason='UNREPLICATED_PARITY_NO_VALIDATED_PRIOR')
        oe = ald.OddEvenV2Result('INCONCLUSIVE', diag['reason'], np.nan, 1000., 990., np.nan, np.nan, diag)
        extra = dict(compute_odd_even_v2=lambda *a, **kw: oe)
        before = run(extra=extra)
        self.assertEqual(before[0][1], 'REVIEW_ODD_EVEN_INCONCLUSIVE')
        result, _, ev, _ = run(('A2',), extra)
        self.assertEqual(result[1], 'CANDIDATE', result[2])
        self.assertEqual(result[3]['odd_even_status'], 'ODD_EVEN_NOT_TESTABLE')
        self.assertTrue(np.isnan(result[3]['odd_even_sigma']))
        peak = ev[-1]['peaks'][0]
        self.assertEqual(peak['odd_even_metrics']['sigma'], None)
        self.assertEqual(peak['odd_even_metrics']['odd_err'], None)
        gate = next(g for g in peak['gates'] if g['gate'] == 'ODD_EVEN')
        self.assertEqual(gate['status'], 'NOT_TESTABLE')
        self.assertEqual(gate['metrics']['n_even'], 1)
        other = ald.OddEvenV2Result('INCONCLUSIVE', 'NO_CADENCE_SCALE', np.nan, 1000., 990., np.nan, np.nan, dict(n_odd=2, n_even=2))
        self.assertEqual(run(('A2',), dict(compute_odd_even_v2=lambda *a, **kw: other))[0][1], before[0][1])

    def test_depth_upper_and_full_t14_fallback(self):
        stellar = lambda _: dict(has_params=True, r_star=.5, m_star=1., e_r_star=.01, e_m_star=.01, teff=5800., ruwe=1.)
        extra = dict(extract_and_stitch_arrays=photometry(34000), get_stellar_and_gaia_params=stellar,
                     get_top_distinct_bls_peaks=lambda *a, **kw: [0],
                     compute_keplerian_duration_bounds_hours=lambda *a: (2., .1, 10.))
        self.assertEqual(run(extra=extra)[0][1], 'REJECT_DEPTH')
        self.assertEqual(run(('DEPTH_UPPER',), extra)[0][1], 'CANDIDATE')
        self.assertEqual(run(('DEPTH_FULL_T14',), extra)[0][1], 'REJECT_DEPTH')
        full_t14_rescue = dict(extra, extract_and_stitch_arrays=photometry(34000, half_width=.036))
        self.assertEqual(run(('DEPTH_FULL_T14',), full_t14_rescue)[0][1], 'CANDIDATE')
        low = dict(extra, extract_and_stitch_arrays=photometry(170))
        self.assertEqual(run(('DEPTH_UPPER',), low)[0][1], 'REJECT_DEPTH')

    def test_v_shape_warn_continues_to_other_gates(self):
        extra = dict(extract_and_stitch_arrays=photometry(12000, triangular=True),
                     get_top_distinct_bls_peaks=lambda *a, **kw: [0])
        self.assertEqual(run(extra=extra)[0][1], 'REJECT_V_SHAPE')
        result, _, ev, _ = run(('V_SHAPE_FLAG',), extra)
        self.assertEqual(result[1], 'CANDIDATE', result[2])
        self.assertIn('V_SHAPE_GRAZING_WARN', ev[-1]['peaks'][0]['warnings'])
        self.assertEqual(run(('V_SHAPE_FLAG',), dict(extra,
            _verify_and_filter_tpf_cp=lambda *a, **kw: (False, 'BEB', np.ones((5, 5)), np.ones((5, 5)))))[0][1], 'REJECT_TPF_PIXEL')

    def test_all_gate_trace_restores_first_rejection(self):
        extra = dict(extract_and_stitch_arrays=photometry(34000))
        off, j0, ev0, _ = run(extra=extra)
        traced, j1, ev1, td = run(extra=extra, trace=True)
        self.assertEqual(off[:3], traced[:3])
        self.assertEqual(off[3], traced[3])
        self.assertEqual(ev0[-1]['peaks'], ev1[-1]['peaks'])
        self.assertEqual([(j['rank'], j.get('exit_code')) for j in j0 if j['record'] == 'rank'],
                         [(j['rank'], j.get('exit_code')) for j in j1 if j['record'] == 'rank'])
        with open(os.path.join(td, 'rc2_all_gates.jsonl')) as handle:
            traces = [json.loads(line) for line in handle]
        self.assertTrue(any(g['gate'] == 'ARCHIVAL' for g in traces[0]['gates']))
        self.assertIn('1', traces[1]['secondary_variants'])
        self.assertIn('1', traces[1]['links'])

    def test_archival_coverage_keeps_sde_and_secondary_cap_contract(self):
        t = np.linspace(1000., 1027., 2000)
        prior = np.linspace(970., 997., 2000)
        prior = prior[np.abs((prior - 1001. + 1.5) % 3. - 1.5) > .06]
        flux = 1. - box(t, 3., 1001., .045, .001)
        row = unittest.mock.MagicMock()
        row.mission = ['TESS Sector 80']
        row.obsid = 's0080_coverage'
        patches = dict(safe_download=lambda *a, **kw: object(),
            extract_and_stitch_arrays=lambda *a, **kw: (prior, np.ones(prior.size), np.full(prior.size, .0001),
                np.zeros(prior.size), np.zeros(prior.size), .95),
            clip_upper_outliers=lambda t, f, e: (t, f, e),
            piecewise_savgol_detrend=lambda t, f, **kw: np.ones_like(f),
            vr_refine_p3=lambda *a, **kw: dict(period=3., t0=1001., n_req=1, n_grid=60, stages=1,
                                            grid_step_d=.0002, grid_source='SINGLE'))
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(patch.object(ald, name, value))
            stack.enter_context(patch.object(ald.lk, 'search_lightcurve', return_value=[row]))
            def archival(sde):
                return ald.verify_multi_epoch('12345678', 106, t, flux, np.full(t.size, .0001),
                                              3., 1001., .1, 1000., sde)
            with patch.object(ald, 'RC2_SWITCHES', frozenset()):
                self.assertFalse(archival(8.)[0])
            with patch.object(ald, 'RC2_SWITCHES', {'ARCHIVAL_COVERAGE'}):
                good = archival(6.5)
                self.assertEqual(good[:2], (True, 'Single-Sector_Only'))
                self.assertEqual(good[2]['vr'], {})
                self.assertEqual(good[2]['n_archival_sectors'], 0)
                self.assertFalse(archival(6.49)[0])

    def test_counterfactual_trace_matches_live_switches(self):
        extra = dict(extract_and_stitch_arrays=photometry(34000),
            get_stellar_and_gaia_params=lambda _: dict(has_params=True, r_star=.5, m_star=1., e_r_star=.01, e_m_star=.01, teff=5800., ruwe=1.),
            compute_keplerian_duration_bounds_hours=lambda *a: (2., .1, 10.))
        _, _, _, td = run(('DEPTH_UPPER',), extra, trace=True)
        with open(os.path.join(td, 'rc2_all_gates.jsonl')) as handle:
            traces = [json.loads(line) for line in handle]
        for switches in ((), ('DEPTH_UPPER',), ('DEPTH_FULL_T14',)):
            scored = reporting.score_trace(traces, switches)
            live = run(switches, extra)[0]
            self.assertEqual(scored['first_candidate_rank'] is not None, live[1] == 'CANDIDATE')


class Reporting(unittest.TestCase):
    def test_marginals_keep_denominators_and_conditional_population(self):
        rows = [dict(snr_target=10, n_discovery_events=3, t14_days=.18, truth_period_days=3,
                     truth_depth_ppm=300, search_recovered=True, matched_candidate=True),
                dict(snr_target=10, n_discovery_events=4, t14_days=.2, truth_period_days=3,
                     truth_depth_ppm=299, search_recovered=True, matched_candidate=False),
                dict(snr_target=5, n_discovery_events=10, t14_days=.1, truth_period_days=3,
                     truth_depth_ppm=300, search_recovered=False, matched_candidate=False)]
        report = reporting.m1_diagnostics(rows)
        self.assertEqual(report['snr_ge_10']['pooled']['conditional_vetting']['n'], 2)
        self.assertEqual(report['all_snr']['by_t14']['<=0.18d']['n'], 2)
        self.assertEqual(report['comparison_domain']['snr_ge_10']['n'], 1)

    def test_scorer_skips_gates_a_rank_never_reached(self):
        gates = [dict(gate='SDE', status='PASS', metrics=dict(sde=9.)),
                 dict(gate='DUTY_CYCLE', status='REJECT', metrics=dict(duty_cycle=.2))]
        gates += [dict(gate=g, status='NOT_EVALUATED', metrics=dict(reason='Not reached'))
                  for g in ('DEPTH', 'V_SHAPE', 'ARCHIVAL', 'SECONDARY')]
        for enabled in ((), reporting.SWITCHES):
            ranks = reporting.score_trace([dict(rank=1, sde=9., gates=gates)], enabled)['ranks']
            self.assertEqual(ranks, [dict(rank=1, disposition='REJECT', gate='DUTY_CYCLE',
                                          archival_secondary_rejection=False)])

    def test_m2_strata_keep_the_in_domain_headline_denominator(self):
        gates = [dict(planet=p, snr_support=dict(N_ev=3), reference_depth_ppm=500.) for p in 'abc']
        rows = [dict(case_id=f'm2_1_p{i}', in_domain=d, snr=12., matched_candidate=m)
                for i, (d, m) in enumerate(((True, True), (False, True), (None, False)))]
        design = {r['case_id']: dict(host='1', target_planet='abc'[i], truth=dict(period=3.)) for i, r in enumerate(rows)}
        report = reporting.from_run(dict(m1_rows=[], m2_rows=rows), [], {}, dict(M2_reference_gates={'1': gates}), design)
        self.assertEqual(report['m2']['headline']['all_snr']['n'], 1)
        self.assertEqual(report['m2']['headline']['all_snr']['k'], 1)


if __name__ == '__main__':
    unittest.main()
