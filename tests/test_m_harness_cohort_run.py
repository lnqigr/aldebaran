"""Real manifest-driver fixture integration and bounded negative/resume controls."""
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('ALDEBARAN_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import run_identity as ri
from m_harness import cohort_run as cr, fixtures, injection, preflight, prodpath, runner

REPO = Path(__file__).resolve().parents[1]


def make_fixture(base):
    base = Path(base)
    root = base / 'hosts'
    ald = prodpath.load_production()
    hosts = [fixtures.synthetic_host(root, 970000 + i, 40, archival_sectors=(12,), seed=73 + i, sigma=1.5e-3)
             for i in range(7)]
    for h in hosts:
        h['M3_source'] = 'Prsa_EB'
    hosts[3]['M3_source'] = 'TFOPWG_FP'
    manifest = dict(cohorts=dict(M1=[hosts[0]], M2=[hosts[1]], M3=hosts[2:4], M4=[hosts[6]]),
                    diagnostic_strata=dict(M3_DIAGNOSTIC_out_of_signal_domain=[hosts[4]],
                                           M3_DIAGNOSTIC_unknown_signal_domain=[hosts[5]]),
                    M2_reference_gates={hosts[1]['tic_id']: []}, scoring_sidecars={})
    catalog = dict(hosts={hosts[1]['tic_id']: dict(reference_planets=[])})
    for name, period in [('fixture target b', 2.3), ('fixture sibling c', 5.1), ('fixture missing d', None)]:
        ref = None if period is None else dict(period_days=period, epoch_btjd=2100., T14_hours=2.)
        catalog['hosts'][hosts[1]['tic_id']]['reference_planets'].append(dict(catalog=dict(pl_name=name),
            reference_status='NO_TRANSIT_REFERENCE' if ref is None else 'VALID_SYNTHETIC', selected_reference=ref))
        manifest['M2_reference_gates'][hosts[1]['tic_id']].append(dict(planet=name,
            domain=dict(in_domain=None if ref is None else True, reasons=[]), SNR_ref=12., snr_support=dict(N_ev=4)))
    m3 = dict(hosts={h['tic_id']: dict(selected_signal=dict(catalog=dict(Per='2.3', pl_orbper='2.3')))
                    for h in hosts[2:6]})
    for arm, data in [('M2', catalog), ('M3', m3)]:
        path = base / (arm + '_catalog.json')
        cr.save(path, data)
        manifest['scoring_sidecars'][arm] = dict(path=str(path), sha256=ri.sha256_file(path))
    manifest_path = base / 'manifest.json'
    cr.save(manifest_path, manifest)
    design = dataclasses.replace(injection.DRAFT_DESIGN, backend='reference')
    target = injection.m1_target_plan([hosts[0]['tic_id']], design)
    rows = injection.draw_m1_host(ald, hosts[0], root, design, 0, target[hosts[0]['tic_id']])
    draws = base / 'draws'
    path = draws / (hosts[0]['tic_id'] + '.json')
    cr.save(path, dict(tic_id=hosts[0]['tic_id'], host_index=0, rows=rows))
    cohorts = {k: dict(root=str(root), hosts=hh, blinded=k != 'M1')
               for k, hh in dict(manifest['cohorts'], **manifest['diagnostic_strata']).items()}
    pre = base / 'preflight.json'
    result = preflight.run(source=REPO / 'aldebaran.py', cohorts=cohorts, out=pre)
    assert result['passed']
    config = dict(manifest_sha256=ri.sha256_file(manifest_path), injection_design=design.as_dict(),
                  draw_files_sha256={path.name: ri.sha256_file(path)}, preflight_sha256=ri.sha256_file(pre),
                  injection_python=sys.executable, run_id='s5_fixture', max_live_injection_bytes=20_000_000_000)
    config_path = base / 'config.json'
    cr.save(config_path, config)
    prereg = base / 'SYNTHETIC_PREREGISTRATION.md'
    prereg.write_text(runner.ENGINEERING_LABEL, encoding='utf-8')
    return ald, dict(manifest_path=manifest_path, draws_dir=draws, root=root, scratch_root=base / 'scratch',
                     out=base / 'run', config_path=config_path, prereg=prereg, preflight_path=pre, repo=REPO)


class CohortDriverTests(unittest.TestCase):
    def test_fixture_through_real_driver(self):
        evidence = os.environ.get('S5_FIXTURE_EVIDENCE')
        with tempfile.TemporaryDirectory(prefix='s5-driver-integration-') as tmp:
            ald, args = make_fixture(Path(evidence) if evidence else Path(tmp))
            result = cr.run(ald, **args)
            self.assertEqual(len(result['ledger']), 13)
            self.assertEqual(result['tables']['m2_false_rejection']['n_cohort'], 3)
            self.assertEqual(result['tables']['m3']['M3']['n'], 2)
            self.assertEqual(result['tables']['m3']['M3_DIAGNOSTIC_out_of_signal_domain']['n'], 1)
            self.assertEqual(result['tables']['m3']['M3_DIAGNOSTIC_unknown_signal_domain']['n'], 1)
            self.assertTrue(all('attempts' in r for r in result['ledger']))
            with patch.object(runner, 'run_cases', side_effect=AssertionError('completed cases rerun')):
                resumed = cr.run(ald, **args)
            self.assertEqual(result['tables'], resumed['tables'])

    def test_retry_all_technical_failures_with_unique_ids_and_keep_final_failure(self):
        self._attempt_control(['ERROR', 'EMPTY_LOW_SDE'], ['ERROR', 'EMPTY_LOW_SDE'])
        self._attempt_control(['SKIP_NO_DATA', 'SKIP_NO_DATA'], ['SKIP_NO_DATA', 'SKIP_NO_DATA'])

    def test_successful_case_is_never_retried(self):
        self._attempt_control(['EMPTY_LOW_SDE'], ['EMPTY_LOW_SDE'])

    def test_interrupted_attempt_retained_and_resumed_with_new_attempt_id(self):
        self._attempt_control(['EMPTY_LOW_SDE'], ['INTERRUPTED', 'EMPTY_LOW_SDE'], interrupted=True)

    def _attempt_control(self, statuses, expected, interrupted=False):
        with tempfile.TemporaryDirectory(prefix='s5-attempt-control-') as tmp:
            base = Path(tmp)
            prereg = base / 'prereg.md'
            prereg.write_text('synthetic', encoding='utf-8')
            bound = ri.BoundRun.open_or_resume(base / 'identity.json', run_id='attempt_control', runner='test',
                                               code_files=[__file__], preregistration_path=prereg, repo_dir=REPO)
            row = dict(case_id='case', arm='M3', host=dict(tic_id='case', sector=40, source_fits=None,
                       source_sha256=None, tpf_fits=None, tpf_sha256=None, stellar={}, lightcurves=[]))
            if interrupted:
                directory = base / 'out/attempts/case/0001'
                directory.mkdir(parents=True)
                bound.save_item_record(directory / 'started.json', dict(case_id='case', attempt_id='case_attempt_0001'))
            calls = []
            def fake_run(ald, **kw):
                calls.append(kw['jobs'][0]['attempt_id'])
                status = statuses[len(calls) - 1]
                return dict(records=dict(case=dict(record=cr.error_record(row, status, 'fixture'),
                                                    routing_check=dict(passed=True), times_path=None)))
            with patch.object(cr, 'prepare_case', return_value=(dict(case=dict(case_id='case'), root=tmp, arm='M3'), {})), \
                 patch.object(runner, 'run_cases', side_effect=fake_run):
                item, history = cr.case_result(None, row, base, base, sys.executable, REPO, base / 'out',
                                               bound, {}, prereg)
                self.assertEqual([r['status'] for r in history], expected)
                self.assertEqual(len(set(calls)), len(calls))
                self.assertEqual(item['record']['status'], statuses[-1])
                self.assertEqual(len(list((base / 'out/attempts/case').glob('*/started.json'))), len(expected))
                resumed, again = cr.case_result(None, row, base, base, sys.executable, REPO, base / 'out',
                                               bound, {}, prereg)
                self.assertEqual(history, again)
                self.assertEqual(item, resumed)

    def test_missing_m2_reference_is_counted_and_target_name_is_frozen(self):
        with tempfile.TemporaryDirectory(prefix='s5-metadata-control-') as tmp:
            _, args = make_fixture(Path(tmp))
            config = cr.read(args['config_path'])
            manifest, catalogs, draws, design, _ = cr.load_inputs(args['manifest_path'], args['draws_dir'], config)
            plan = cr.plan_cases(manifest, draws)
            items = {row['case_id']: dict(record=cr.error_record(row, 'ERROR', 'fixture'), times_path=None)
                     for row in plan}
            # Other tables are irrelevant to this identity check; inspect the production helper call.
            captured = {}
            actual = runner.m2_table_rows
            def inspect(items, design, *a, **kw):
                captured.update(design)
                return actual(items, design, *a, **kw)
            with patch.object(runner, 'm2_table_rows', side_effect=inspect):
                tables = cr.build_tables(None, manifest, catalogs, draws, design, plan, items,
                                        dict(gate=dict(blocks_m4=True)))
            self.assertEqual({r['target_planet'] for r in captured.values()},
                             {'fixture target b', 'fixture sibling c', 'fixture missing d'})
            self.assertEqual(tables['m2_false_rejection']['n_cohort'], 3)
            self.assertEqual(tables['m2_rows'][-1]['line'], 'MATCH_UNAVAILABLE')

    def test_failed_noise_gate_blocks_both_null_arms_before_result_reads(self):
        with tempfile.TemporaryDirectory(prefix='s5-gate-control-') as tmp:
            ald, args = make_fixture(Path(tmp))
            events = []
            def gate(*a):
                events.append('gate')
                return dict(gate=dict(blocks_m4=True), rows=[])
            def result(ald, row, *a):
                self.assertEqual(events[0], 'gate')
                self.assertNotIn(row['arm'], ('N1', 'N2'))
                return dict(record=cr.error_record(row, 'ERROR', 'fixture'), times_path=None), []
            with patch.object(cr, 'noise_pregate', side_effect=gate), patch.object(cr, 'case_result', side_effect=result):
                output = cr.run(ald, **args)
            blocked = [r for r in output['ledger'] if r['arm'] in ('N1', 'N2')]
            self.assertEqual([r['status'] for r in blocked], ['NOT_RUN', 'NOT_RUN'])
            self.assertEqual(output['tables']['m4_n1']['status'], 'NOT_RUN')

    def test_changed_saved_draw_is_refused_without_detector_call(self):
        with tempfile.TemporaryDirectory(prefix='s5-hash-control-') as tmp:
            _, args = make_fixture(Path(tmp))
            path = next(args['draws_dir'].glob('*.json'))
            path.write_text(path.read_text() + '\n', encoding='utf-8')
            with self.assertRaisesRegex(cr.ControlFailure, 'SAVED_DRAW_CHANGED'):
                cr.load_inputs(args['manifest_path'], args['draws_dir'], cr.read(args['config_path']))

    def test_noise_transform_failure_blocks_m4_and_is_recorded(self):
        with tempfile.TemporaryDirectory(prefix='s5-noise-control-') as tmp:
            base = Path(tmp)
            bound = ri.BoundRun.open_or_resume(base / 'identity.json', run_id='noise_control', runner='test',
                                               code_files=[__file__], repo_dir=REPO)
            manifest = dict(cohorts=dict(M4=[dict(tic_id='failed', sector=40)]))
            with patch.object(cr, 'prepare_case', side_effect=cr.ControlFailure('NULL_TRANSFORM_FAILED')):
                result = cr.noise_pregate(None, manifest, base, base, bound, base)
            self.assertTrue(result['gate']['blocks_m4'])
            self.assertEqual(result['evaluation_failures'][0]['tic_id'], 'failed')

    def test_injection_preparation_distinguishes_technical_and_control_failure(self):
        with tempfile.TemporaryDirectory(prefix='s5-prepare-control-') as tmp:
            base = Path(tmp)
            request = base / 'request.json'
            result = base / 'result.json'
            cr.save(request, dict(host={}, root=tmp, case_id='fixture', truth={}, scratch=tmp, result=str(result)))
            for exception, status in [(OSError('fixture write failure'), 'ERROR'),
                                      (cr.ControlFailure('fixture control failure'), 'CONTROL_FAILED')]:
                with patch.object(injection, 'write_injection', side_effect=exception):
                    cr.prepare_injection(request)
                self.assertEqual(cr.read(result)['status'], status)


if __name__ == '__main__':
    unittest.main()
