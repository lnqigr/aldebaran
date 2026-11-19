"""M4-only routing, all-host transform preflight and failed-gate isolation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_identity as ri
from m_harness import cohort_run as cr, fixtures, m4_run

REPO = Path(__file__).resolve().parents[1]


class M4OnlyTests(unittest.TestCase):
    def exercise(self, blocks):
        with tempfile.TemporaryDirectory(prefix='m4-only-test-') as tmp:
            base = Path(tmp)
            host = fixtures.synthetic_host(base / 'hosts', 990100, 40, archival_sectors=(12,), seed=31, sigma=1e-3)
            manifest_path, config_path = (base / n for n in ('manifest.json', 'config.json'))
            pre = base / 'checks/preflight.json'
            cr.save(manifest_path, dict(cohorts=dict(M4=[host])))
            prior = base / 'prior.json'
            cr.save(prior, dict(run_id='accepted_fixture', tables=dict(grey_zone_gain_candidate=dict(
                pooled=dict(dc_grey=dict(k=6, n=100, rate=.06))))))
            config = dict(manifest_sha256=ri.sha256_file(manifest_path), n_hosts=1, injection_python=sys.executable,
                          accepted_m1_tables_sha256=ri.sha256_file(prior), accepted_m1_run_id='accepted_fixture',
                          run_id='m4_fixture')
            cr.save(config_path, config)
            with patch.object(m4_run.fidelity, 'production_arrays', side_effect=AssertionError('noise during preflight')):
                checked = m4_run.transform_preflight(manifest_path=manifest_path, config_path=config_path,
                    root=base / 'hosts', scratch=base / 'scratch', out=pre, repo=REPO)
            self.assertTrue(checked['passed'])
            self.assertEqual(checked['detector_runs'], 0)
            config['preflight_sha256'] = ri.sha256_file(pre)
            cr.save(config_path, config)
            prereg = base / 'prereg.md'
            prereg.write_text('Synthetic fixture only', encoding='utf-8')
            events = []
            def gate(*args):
                events.append('gate')
                return dict(gate=dict(blocks_m4=blocks), rows=[])
            def case(ald, row, *args):
                self.assertEqual(events[0], 'gate')
                self.assertIn(row['arm'], ('N1', 'N2'))
                events.append(row['arm'])
                return dict(record=cr.error_record(row, 'EMPTY_LOW_SDE', 'fixture')), []
            with patch.object(cr, 'noise_pregate', side_effect=gate), patch.object(cr, 'case_result', side_effect=case), \
                 patch.object(m4_run.fidelity, 'm4_tail_report', return_value={'status': 'fixture'}):
                m4_run.run(None, manifest_path=manifest_path, config_path=config_path, prereg=prereg,
                    preflight_path=pre, prior_tables=prior, root=base / 'hosts', scratch=base / 'scratch',
                    out=base / 'run', repo=REPO)
            result = cr.read(base / 'run/tables.json')
            self.assertEqual(len(result['ledger']), 2)
            self.assertEqual(events, ['gate'] if blocks else ['gate', 'N1', 'N2'])
            self.assertEqual(ri.sha256_file(prior), config['accepted_m1_tables_sha256'])
            if blocks:
                self.assertTrue(all(r['status'] == 'NOT_RUN' for r in result['ledger']))

    def test_only_null_arms_run_after_passing_gate(self):
        self.exercise(False)

    def test_failed_gate_runs_no_detector_and_keeps_both_ledger_lines(self):
        self.exercise(True)


if __name__ == '__main__':
    unittest.main()
