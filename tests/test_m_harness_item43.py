"""Item 43 policy regressions on synthetic inputs; no real host or network access."""
import dataclasses
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('ALDEBARAN_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
from m_harness import estimands as E, injection as I, runner, stats
from m_harness import transit_model as TM
import test_m_harness_estimands as T


class Item43Tests(unittest.TestCase):
    def test_unresolved_two_three_and_observed_fail_at_500(self):
        for alarms, bad, expected in ((1, 1, 'KEEP'), (1, 2, 'NOT_EVALUABLE'), (3, 1, 'FAIL'), (0, 0, 'KEEP')):
            rows = [T.matched_candidate(f'a{i}', 6.8) for i in range(alarms)]
            rows += [T.record(f'e{i}', 'ERROR') for i in range(bad)]
            rows += [T.record(f'n{i}', 'EMPTY_LOW_SDE') for i in range(500 - len(rows))]
            m4 = E.m4_false_alarm(rows, {}, 'N1')
            result = E.grey_zone_m4_decision(0.06, m4)
            self.assertEqual(result['decision'], expected)
            self.assertAlmostEqual(result['m4c_upper_at_6'], stats.cp_upper(alarms + bad, 500))
            self.assertEqual(result['observed_planned']['n'], 500)
        self.assertEqual(E.grey_zone_m4_decision(None, m4)['decision'], 'NOT_EVALUABLE')

    def test_random_edge_assignment_and_written_realized_counts(self):
        hosts = list(range(300))
        plan = I.m1_target_plan(hosts, I.DRAFT_DESIGN)
        self.assertEqual(plan, I.m1_target_plan(hosts, I.DRAFT_DESIGN))
        for bi, split in ((0, 5), (2, 9)):
            ids = {h for h in hosts if plan[h][bi, 0] >= split}
            self.assertEqual(len(ids), 150)
            self.assertNotEqual(ids, set(range(150)))
            self.assertNotEqual(ids, set(range(150, 300)))
        ctx = dict(main_t=np.array([0., 30.]), m_star=1., r_star=1., snr_ctx=None, archival=[])
        def solve(target, b, p, t0, ld, ars, context, backend):
            return dict(valid=True, snr_target=target, snr_achieved=target, k=.02, depth=.001, t14=.1)
        with patch.object(I, 'host_context', return_value=ctx), patch.object(I, 'joint_solve', side_effect=solve), \
             patch.object(I, 'discovery_supported_events', return_value=3):
            draws = [s for h in hosts for s in I.draw_m1_host(None, dict(ld=(.3, .2)), '.', I.DRAFT_DESIGN, h, plan[h])]
        rows = [dict(truth=s['truth'], draw_status=s['status']) for s in draws]
        self.assertEqual(I.realized_grey_counts(rows), dict(counts=[150, 300, 150], expected=[150, 300, 150], passed=True))
        lost = next(r for r in rows if 5 <= r['truth']['snr_target'] < 6)
        lost['draw_status'] = 'INJECTION_FAILED'
        self.assertFalse(I.realized_grey_counts(rows)['passed'])
        self.assertEqual(I.realized_grey_counts(rows)['counts'], [149, 300, 150])

    def test_geometry_floor_and_empty_range(self):
        for mass, radius in ((1, 1), (0.1, 1.75)):
            pop = I.conditional_period_range(mass, radius, I.DRAFT_DESIGN)
            floor = math.sqrt(4 * math.pi**2 * (2.6 * radius * 6.957e8)**3 / (6.67430e-11 * mass * 1.98847e30)) / 86400
            self.assertAlmostEqual(pop['lo'], max(1, floor))
            self.assertEqual(pop['hi'], 13)
            self.assertGreaterEqual(TM.a_over_rs(pop['lo'], mass, radius), 2.6 - 1e-12)
        pop = I.conditional_period_range(0.1, 20, I.DRAFT_DESIGN)
        self.assertEqual(pop['status'], 'EMPTY_PERIOD_RANGE')
        tr, att = I.draw_planet(np.random.default_rng(1), 6, 8,
                               dict(main_t=np.array([0., 30.]), m_star=0.1, r_star=20), (.3, .2), I.DRAFT_DESIGN)
        self.assertIsNone(tr)
        self.assertEqual(att[0]['draw_outcome'], 'EMPTY_PERIOD_RANGE')

    def test_targets_survive_retries_and_geometry_is_checked(self):
        ctx = dict(main_t=np.array([0., 30.]), m_star=.1, r_star=1.75, snr_ctx=None, archival=[])
        design = dataclasses.replace(I.DRAFT_DESIGN, max_t0_tries=1)
        seen = []
        def solve(target, b, p, t0, ld, ars, context, backend):
            seen.append((target, b, p, t0, ars))
            return dict(valid=len(seen) > 2, snr_target=target, snr_achieved=target, k=.02, depth=.001, t14=.1)
        with patch.object(I, 'joint_solve', side_effect=solve), patch.object(I, 'discovery_supported_events', return_value=3):
            tr, att = I.draw_planet(np.random.default_rng(2), 6, 8, ctx, (.3, .2), design)
        self.assertIsNotNone(tr)
        self.assertGreaterEqual(len(att), 3)
        self.assertEqual(len({x[0] for x in seen}), 1)
        self.assertTrue(all(x[-1] >= 2.6 for x in seen))
        # A checked value below the floor must not reach the model, even if a future geometry change produces it.
        with patch.object(TM, 'a_over_rs', return_value=2.59), \
             patch.object(I, 'joint_solve', side_effect=AssertionError('below-floor model call')) as model:
            tr, att = I.draw_planet(np.random.default_rng(2), 6, 8, ctx, (.3, .2), I.DRAFT_DESIGN)
        self.assertIsNone(tr)
        model.assert_not_called()
        self.assertTrue(all(a['draw_outcome'] == 'BELOW_AR_FLOOR' for a in att))

    def test_host_bin_copies_draw_fresh_geometry(self):
        design = dataclasses.replace(I.DRAFT_DESIGN, max_t0_tries=1)
        host = dict(tic_id='h', ld=(.3, .2))
        ctx = dict(main_t=np.array([0., 30.]), m_star=1., r_star=1., snr_ctx=None, archival=[])
        def solve(target, b, p, t0, ld, ars, context, backend):
            return dict(valid=True, snr_target=target, snr_achieved=target, k=.02, depth=.001, t14=.1)
        targets = I.m1_target_plan(['h'], design)['h']
        with patch.object(I, 'host_context', return_value=ctx), patch.object(I, 'joint_solve', side_effect=solve), \
             patch.object(I, 'discovery_supported_events', return_value=3):
            rows = I.draw_m1_host(None, host, '.', design, 0, targets)
        for field in ('period', 't0', 'b'):
            self.assertEqual(len({r['truth'][field] for r in rows}), 6)
        self.assertEqual([r['truth']['snr_target'] for r in rows], list(targets.values()))

    def test_support_counts_hosts_and_achieved_snr_uses_target_bin(self):
        rows = [T.m1_row(i, '[6, 8)', 7, host=f'h{i // 2}') for i in range(40)]
        for i, r in enumerate(rows):
            r.update(snr_achieved=8.1 if i == 0 else 7., rp_true_earth=.7 if i < 10 else 1.)
        m = E.m1_completeness(rows, ['[6, 8)'], n_boot=20)
        self.assertFalse(m['pooled']['supported'])
        self.assertEqual(m['pooled']['n_hosts'], 20)
        self.assertEqual(m['headline_rp_ge_0_8']['n'], 30)
        self.assertEqual(m['radius_below_2']['n'], 40)
        a = m['achieved_snr']['[6, 8)']
        self.assertEqual((a['n'], a['median'], a['iqr'], a['out_of_bin_fraction']), (40, 7, 0, .025))

    def test_radius_and_period_edges(self):
        self.assertEqual([runner._label(runner.DRAFT_RADIUS_BINS, x) for x in (.799, .8, 2, 4, 8, 15)],
                         ['[0, 0.8)', '[0.8, 2)', '[2, 4)', '[4, 8)', '[8, 15]', '[8, 15]'])
        self.assertEqual(runner.DRAFT_PERIOD_BINS[-1], (5, 13))

    def test_m2_sibling_is_loss_and_host_cluster_support(self):
        sibling = dict(period=4.1, t0=2001., t14=.1)
        rec = T.record('sibling', 'CANDIDATE', top=[T.peak(1, period=4.1)],
                       journal=[T.jrow(1, 'CANDIDATE', period=4.1)], first_candidate=1,
                       candidate=dict(T.cand(period=4.1), duration_hours=2.4))
        o = E.m2_case_outcome(rec, T.TRUTH, T.TIMES, 'host', [T.TRUTH, sibling], sibling_references=[sibling])
        self.assertFalse(o['matched_candidate'])
        self.assertTrue(o['sibling_win'])
        rows = [dict(T.m2_row(i, i % 3 == 0, snr=9 if i < 40 else 25), host=f'h{i // 2}',
                     n_reference_planets=2 if i < 40 else 1, sibling_win=i == 1) for i in range(100)]
        m = E.m2_false_rejection(rows, ((0, 10), (10, 25), (25, math.inf)), n_boot=40)
        self.assertEqual(m['sibling_wins']['n'], 1)
        self.assertEqual(m['by_multiplicity']['multi-planet']['end_to_end']['n'], 40)
        self.assertEqual(m['by_multiplicity']['single-planet']['end_to_end']['n'], 60)
        self.assertFalse(m['by_snr']['[0, 10)']['supported'])
        self.assertTrue(m['by_snr']['[25, inf)']['supported'])
        self.assertIn('n_hosts', m['end_to_end'])
        self.assertEqual(m['end_to_end']['n_hosts'], 50)
        self.assertIn('host_cluster', m['end_to_end'])

    def test_m5_bias_and_95_percent_quantiles(self):
        cases = [dict(case_id=str(i), candidate=dict(T.cand(), period_days=3 + i / 100,
                      depth_ppm=1000 * (1 + i / 100)), truth=dict(period=3., t0=2001., depth=.001, k=.03), r_star=1.)
                 for i in range(5)]
        m = E.m5_parameters(cases)
        p = m['period']['error_d']
        self.assertAlmostEqual(p['median_bias'], .02)
        self.assertAlmostEqual(p['p2_5'], .001)
        self.assertAlmostEqual(p['p97_5'], .039)
        self.assertAlmostEqual(m['depth']['ratio']['median_bias'], .02)

    def test_m_context_guard_runs_before_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            declaration = Path(tmp) / 'design.json'
            declaration.write_text('{}')
            with patch.object(runner, 'harness_code_files', return_value=[]), \
                 patch('run_identity.BoundRun.open_or_resume', side_effect=AssertionError('binding reached')), \
                 self.assertRaisesRegex(ValueError, 'EMPTY_KNOWN_SIGNAL_CONTEXT'):
                runner.run_cases(None, run_dir=tmp, run_id='guard', repo_dir=tmp, config={},
                                 preregistration_path=declaration,
                                 jobs=[dict(arm='M2', root=tmp, case=dict(case_id='p', lightcurves=[], tpf_fits=None,
                                                                 known_transits=[dict(period=3)]))])


if __name__ == '__main__':
    unittest.main()
