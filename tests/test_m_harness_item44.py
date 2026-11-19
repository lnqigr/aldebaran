"""Prospective item 44 controls; Item32 remains the default development policy."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from m_harness import fidelity


def sample(n=500, count=25, lag=1, shift=0.30):
    rows = []
    for i in range(n):
        u = dict(cdpp_1h=1.0, mad=1.0, acf1=0.0, acf10=0.0)
        n1 = dict(u)
        if i < count:
            n1[f"acf{lag}"] = shift
        rows.append(dict(star=str(i), sector=51, U=u, N1=n1))
    return rows, [r['star'] for r in rows]


class Item44Tests(unittest.TestCase):
    def test_runner_selects_m4_and_writes_joined_listing(self):
        from m_harness import runner
        with tempfile.TemporaryDirectory(prefix='item44-') as tmp:
            with patch.object(fidelity, 'fidelity_gate', wraps=fidelity.fidelity_gate) as gate, \
                    patch.object(fidelity, 'm4_tail_report', wraps=fidelity.m4_tail_report) as report:
                summary = runner.dry_run(Path(tmp) / 'dry', n_hosts=2,
                                         snr_bins=((8.0, 10.0), (25.0, 35.0)),
                                         repo_dir=Path(__file__).resolve().parents[1])
            self.assertEqual(gate.call_args.kwargs, dict(m4=True))
            tail = summary['steps']['m4_acf_tail']
            self.assertEqual(tail['status'], 'COMPLETE')
            self.assertEqual(len(tail['top_10']), 2)
            outcomes = report.call_args.args[2]
            for r in tail['top_10']:
                self.assertEqual(r['max_tier1_sde'], outcomes[r['tic_id']]['top_sde'])
                self.assertEqual(set(r['acf']), {'1', '10'})
            text = (Path(tmp) / 'dry/DRY_RUN_REPORT.md').read_text()
            for heading in ('M4 ACF tail (item 44)', 'All stars above 0.05', 'Top 10 stars',
                            'Cohort maximum', 'Max Tier 1 SDE', 'False alarm SDE >= 6.0'):
                self.assertIn(heading, text)

    def test_25_pass_26_fail_separately_at_either_lag(self):
        for lag in (1, 10):
            for count, status in ((25, 'PASS'), (26, 'FAIL')):
                with self.subTest(lag=lag, count=count):
                    rows, stars = sample(count=count, lag=lag)
                    g = fidelity.fidelity_gate(rows, stars, m4=True)
                    self.assertEqual(g['status'], status)
                    criterion = g['noise']['criteria'][f'C{2 if lag == 1 else 3}']
                    self.assertIn('tail', criterion)
                    self.assertEqual(criterion['tail']['count'], count)
                    self.assertEqual(g['blocks_m4'], status != 'PASS')

    def test_lags_are_separate_and_equality_is_not_exceedance(self):
        rows, stars = sample(count=25)
        for r in rows[25:50]:
            r['N1']['acf10'] = -0.30
        self.assertEqual(fidelity.noise_gate(rows, stars, m4=True)['status'], 'PASS')
        rows, stars = sample(count=26, shift=0.05)
        self.assertEqual(fidelity.noise_gate(rows, stars, m4=True)['status'], 'PASS')

    def test_development_default_retains_maximum_and_full_gate_cannot_select_m4(self):
        rows, stars = sample(n=45, count=1)
        self.assertEqual(fidelity.noise_gate(rows, stars)['status'], 'FAIL')
        self.assertEqual(fidelity.noise_gate(rows, stars, m4=True)['status'], 'PASS')
        with self.assertRaises(ValueError):
            fidelity.fidelity_gate(rows, stars, [], [], m4=True)

    def test_median_p90_and_missingness_still_block_m4(self):
        for count, shift in ((500, 0.006), (60, 0.016)):
            rows, stars = sample(count=count, shift=shift)
            self.assertEqual(fidelity.noise_gate(rows, stars, m4=True)['status'], 'FAIL')
        rows, stars = sample()
        self.assertEqual(fidelity.noise_gate(rows[:-1], stars, m4=True)['status'], 'FAILED_EVALUATION')
        rows[0]['N1']['acf10'] = float('nan')
        self.assertEqual(fidelity.noise_gate(rows, stars, m4=True)['status'], 'FAILED_EVALUATION')

    def test_report_lists_union_top10_maximum_and_unknown_outcomes(self):
        rows, stars = sample(count=12)
        rows[12]['N1']['acf10'] = -0.4
        outcomes = {'0': dict(top_sde=6.0, status_class='NO_DETECTION', status='EMPTY'),
                    '12': dict(top_sde=5.9, status_class='NO_DETECTION', status='EMPTY'),
                    '1': dict(top_sde=7.0, status_class='FAILED', status='ERROR')}
        r = fidelity.m4_tail_report(rows, stars, outcomes)
        self.assertEqual(len(r['above_005']), 13)
        self.assertEqual(len(r['top_10']), 10)
        self.assertEqual(r['cohort_maximum']['tic_id'], '12')
        self.assertEqual(r['cohort_maximum']['acf']['10']['signed_difference'], -0.4)
        by_tic = {v['tic_id']: v for v in r['above_005']}
        self.assertTrue(by_tic['0']['false_alarm_sde_6'])
        self.assertFalse(by_tic['12']['false_alarm_sde_6'])
        self.assertIsNone(by_tic['1']['false_alarm_sde_6'])
        self.assertIsNone(by_tic['2']['false_alarm_sde_6'])
        self.assertEqual(by_tic['0']['sector'], 51)
        rows[0].pop('sector')
        self.assertEqual(fidelity.m4_tail_report(rows, stars, outcomes)['status'], 'FAILED_EVALUATION')
