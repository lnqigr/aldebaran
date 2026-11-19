"""M4-only completion of the sealed RC1 baseline; never executes M1/M2/M3."""
import argparse
import json
from pathlib import Path
import tempfile

import run_identity as ri
from . import cohort_run as cr, estimands, fidelity, preflight, prodpath, runner


def inputs(manifest_path, config):
    if ri.sha256_file(manifest_path) != config['manifest_sha256']:
        raise cr.ControlFailure('M4_MANIFEST_CHANGED')
    manifest = cr.read(manifest_path)
    hosts = manifest['cohorts']['M4']
    if len(hosts) != config['n_hosts'] or len({h['tic_id'] for h in hosts}) != len(hosts):
        raise cr.ControlFailure('M4_COHORT_CHANGED')
    for h in hosts:
        if h['known_transits'] or int(h['sector']) == 108 or any(int(r['sector']) == 108 for r in h['lightcurves']):
            raise cr.ControlFailure('M4_BLINDING_OR_SECTOR_108')
    return manifest


def transform_preflight(*, manifest_path, config_path, root, scratch, out, repo=None):
    """Host gates and actual transforms on every host; no production import or noise metric."""
    repo = Path(repo or Path(__file__).resolve().parents[1])
    config = cr.read(config_path)
    manifest = inputs(manifest_path, config)
    root, scratch, out = Path(root), Path(scratch), Path(out)
    if out.exists():
        raise FileExistsError(out)
    _scratch(scratch, root, out.parent, repo)
    rows = []
    with tempfile.TemporaryDirectory(prefix='m4-preflight-', dir=scratch) as tmp:
        gates = preflight.run(source=repo / 'aldebaran.py',
            cohorts=dict(M4=dict(root=str(root), blinded=True, hosts=manifest['cohorts']['M4'])),
            out=Path(tmp) / 'host_gates.json', note='M4-FIX-01 host gates only; transforms checked separately')
    for h in manifest['cohorts']['M4']:
        row = dict(host=h, case_id='n1_' + h['tic_id'], arm='N1')
        try:
            with tempfile.TemporaryDirectory(prefix='m4-preflight-', dir=scratch) as tmp:
                job, prepared = cr.prepare_case(row, root, Path(tmp), config['injection_python'], repo)
                rows.append(dict(tic_id=h['tic_id'], passed=True, invariants=prepared['transform'],
                                 transformed_case=job['case']))
        except Exception as exc:
            rows.append(dict(tic_id=h['tic_id'], passed=False, error=f'{type(exc).__name__}: {exc}'))
    result = dict(task='M4-FIX-01', manifest_sha256=config['manifest_sha256'],
        code_sha256=ri.compute_code_sha256(runner.harness_code_files(repo), base_dir=repo),
        host_gates=gates, transforms=rows, n_hosts=len(rows),
        passed=gates['passed'] and all(r['passed'] for r in rows),
        detector_runs=0, noise_metrics_evaluated=False)
    cr.save(out, result)
    return result


def _scratch(scratch, root, out, repo):
    for protected in (root.resolve(), out.resolve(), repo.resolve()):
        if scratch.resolve().is_relative_to(protected) or protected.is_relative_to(scratch.resolve()):
            raise cr.ControlFailure('M4_SCRATCH_MUST_BE_SEPARATE')
    scratch.mkdir(parents=True, exist_ok=True)


def run(ald, *, manifest_path, config_path, prereg, preflight_path, prior_tables, root, scratch, out,
        repo=None, on_case=None):
    repo = Path(repo or Path(__file__).resolve().parents[1])
    root, scratch, out = Path(root), Path(scratch), Path(out)
    config = cr.read(config_path)
    if not prereg or not Path(prereg).is_file():
        raise cr.ControlFailure('M4_PREREGISTRATION_REQUIRED')
    manifest = inputs(manifest_path, config)
    check = cr.read(preflight_path)
    code = ri.compute_code_sha256(runner.harness_code_files(repo), base_dir=repo)
    if not check['passed'] or check['code_sha256'] != code or check['manifest_sha256'] != config['manifest_sha256']:
        raise cr.ControlFailure('M4_TRANSFORM_PREFLIGHT_REQUIRED')
    if (check['n_hosts'] != len(manifest['cohorts']['M4']) or
        {r['tic_id'] for r in check['transforms']} != {h['tic_id'] for h in manifest['cohorts']['M4']} or
        not all(r['passed'] for r in check['transforms'])):
        raise cr.ControlFailure('M4_PREFLIGHT_COHORT_MISMATCH')
    if ri.sha256_file(preflight_path) != config['preflight_sha256']:
        raise cr.ControlFailure('M4_PREFLIGHT_CHANGED')
    if ri.sha256_file(prior_tables) != config['accepted_m1_tables_sha256']:
        raise cr.ControlFailure('M4_ACCEPTED_M1_TABLES_CHANGED')
    previous = cr.read(prior_tables)
    if previous['run_id'] != config['accepted_m1_run_id']:
        raise cr.ControlFailure('M4_ACCEPTED_M1_RUN_CHANGED')
    _scratch(scratch, root, out, repo)
    plan = [dict(case_id=f'{arm.lower()}_{h["tic_id"]}', arm=arm, host=h)
            for arm in ('N1', 'N2') for h in manifest['cohorts']['M4']]
    bound = ri.BoundRun.open_or_resume(out / 'run_identity.json', run_id=config['run_id'],
        runner='m_harness.m4_run', code_files=runner.harness_code_files(repo), repo_dir=repo,
        config=dict(config, root=str(root.resolve()), scratch_root=str(scratch.resolve())),
        preregistration_path=prereg, input_paths=[str(manifest_path), str(config_path), str(preflight_path), str(prior_tables)],
        input_spec=plan)
    ledger, items = [], {}
    try:
        gate = cr.noise_pregate(ald, manifest, root, scratch, bound, out)
        for row in plan:
            if gate['gate']['blocks_m4']:
                ledger.append(dict(case_id=row['case_id'], arm=row['arm'], status='NOT_RUN',
                                   reason='M4 noise-only pre-gate failed', attempts=[]))
                continue
            item, attempts = cr.case_result(ald, row, root, scratch, config['injection_python'], repo,
                                            out, bound, config, prereg)
            items[row['case_id']] = item
            ledger.append(dict(case_id=row['case_id'], arm=row['arm'], status=item['record']['status'], attempts=attempts))
            if on_case:
                on_case(row['case_id'], len(ledger))
        gain = previous['tables']['grey_zone_gain_candidate']['pooled']['dc_grey']
        if gate['gate']['blocks_m4']:
            tables = dict(m4_n1=dict(status='NOT_RUN'), m4_n2=dict(status='NOT_RUN'),
                          grey_zone_decision=estimands.grey_zone_decision(None, None))
        else:
            crowd = {r['case_id']: r['host']['crowdsap'] for r in plan}
            tables = {'m4_' + arm.lower(): estimands.m4_false_alarm(
                [items[r['case_id']]['record'] for r in plan if r['arm'] == arm], crowd, arm) for arm in ('N1', 'N2')}
            tables['grey_zone_decision'] = estimands.grey_zone_m4_decision(gain['rate'], tables['m4_n1'])
            tables['m4_tail_report'] = fidelity.m4_tail_report(gate['rows'],
                [h['tic_id'] for h in manifest['cohorts']['M4']],
                {r['host']['tic_id']: estimands.case_outcome(items[r['case_id']]['record']) for r in plan if r['arm'] == 'N1'})
        tables['accepted_m1_gain'] = gain
        bound.save_item_record(out / 'tables.json', dict(tables=tables, ledger=ledger, noise_gate=gate['gate'],
                                                        accepted_m1_tables_sha256=config['accepted_m1_tables_sha256']))
        bound.record_completion(extra=dict(n_planned=len(plan), n_final=len(ledger)))
        return tables
    except (KeyboardInterrupt, SystemExit):
        bound.record_interruption('M4-only driver interrupted', completed_count=len(ledger))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('preflight', 'run'))
    for name in ('manifest', 'config', 'root', 'scratch', 'out'):
        parser.add_argument('--' + name, required=True)
    for name in ('prereg', 'preflight', 'prior-tables'):
        parser.add_argument('--' + name)
    a = parser.parse_args()
    if a.step == 'preflight':
        check = transform_preflight(manifest_path=a.manifest, config_path=a.config, root=a.root, scratch=a.scratch, out=a.out)
        print(json.dumps(dict(passed=check['passed'], n_hosts=check['n_hosts'],
                             failed=[r['tic_id'] for r in check['transforms'] if not r['passed']])))
        return 0 if check['passed'] else 1
    run(prodpath.load_production(), manifest_path=a.manifest, config_path=a.config, prereg=a.prereg,
        preflight_path=a.preflight, prior_tables=a.prior_tables, root=a.root, scratch=a.scratch, out=a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
