"""Frozen-manifest M1--M5 driver. No sampling, acquisition or scientific rule changes.

Run with ``python -B -m m_harness.cohort_run run --help``. Injection preparation
runs in the pinned injection interpreter; detection runs in the suite interpreter.
Scratch contains one generated case at a time. All attempt records survive retries.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


import run_identity as ri
from . import estimands, fidelity, injection, matching, nulls, prodpath, runner, routing, transit_model


class ControlFailure(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    ri.atomic_write_json(path, value)


def load_inputs(manifest_path, draws_dir, config):
    """Only metadata and saved draw records are read here, never photometry."""
    if ri.sha256_file(manifest_path) != config['manifest_sha256']:
        raise ControlFailure('MANIFEST_CHANGED')
    manifest = read(manifest_path)
    catalogs = {}
    inputs = [str(manifest_path)]
    for arm, binding in manifest['scoring_sidecars'].items():
        if ri.sha256_file(binding['path']) != binding['sha256']:
            raise ControlFailure('SCORING_SIDECAR_CHANGED: ' + arm)
        catalogs[arm] = read(binding['path'])
        inputs.append(binding['path'])
    draws = {}
    design = injection.InjectionDesign(**{**config['injection_design'],
                                          'snr_bins': tuple(tuple(b) for b in config['injection_design']['snr_bins'])})
    hosts = manifest['cohorts']['M1']
    targets = injection.m1_target_plan([h['tic_id'] for h in hosts], design)
    for i, h in enumerate(hosts):
        path = Path(draws_dir) / (h['tic_id'] + '.json')
        if ri.sha256_file(path) != config['draw_files_sha256'][path.name]:
            raise ControlFailure('SAVED_DRAW_CHANGED: ' + str(path))
        item = read(path)
        if item['tic_id'] != h['tic_id'] or item['host_index'] != i:
            raise ControlFailure('SAVED_DRAW_HOST_CHANGED')
        rows = item['rows']
        if [(r['snr_bin_index'], r['slot']) for r in rows] != [(b, s) for b in range(len(design.snr_bins))
                                                                         for s in range(design.n_per_bin)]:
            raise ControlFailure('SAVED_DRAW_SLOTS_CHANGED')
        for r in rows:
            if (r['seed'] != injection.slot_seed(design, i, r['snr_bin_index'], r['slot']) or
                    r['snr_target'] != targets[h['tic_id']][r['snr_bin_index'], r['slot']]):
                raise ControlFailure('SAVED_DRAW_SEED_OR_TARGET_CHANGED')
            if not estimands.m1_draw_assessment(r, h['stellar']['r_star'], design.max_draws,
                                              design.max_t0_tries)['accepted']:
                raise ControlFailure('UNVERIFIED_DRAW_FAMILY')
        draws[h['tic_id']] = rows
        inputs.append(str(path))
    all_hosts = [h for group in (manifest['cohorts'], manifest['diagnostic_strata']) for hh in group.values() for h in hh]
    if len({h['tic_id'] for h in all_hosts}) != len(all_hosts):
        raise ControlFailure('COHORT_OVERLAP')
    for h in all_hosts:
        if h['known_transits'] or int(h['sector']) == 108 or any(int(r['sector']) == 108 for r in h['lightcurves']):
            raise ControlFailure('BLINDING_OR_SECTOR_108')
    return manifest, catalogs, draws, design, inputs


def plan_cases(manifest, draws):
    rows = []
    for h in manifest['cohorts']['M1']:
        for d in draws[h['tic_id']]:
            rows.append(dict(case_id=f"m1_{h['tic_id']}_b{d['snr_bin_index']}_s{d['slot']}",
                             arm='M1', host=h, draw=d))
    for arm in ('M2', 'M3'):
        rows.extend(dict(case_id=f"{arm.lower()}_{h['tic_id']}", arm=arm, host=h) for h in manifest['cohorts'][arm])
    for arm, hosts in manifest['diagnostic_strata'].items():
        rows.extend(dict(case_id=f"{arm.lower()}_{h['tic_id']}", arm=arm, host=h) for h in hosts)
    for arm in ('N1', 'N2'):
        rows.extend(dict(case_id=f"{arm.lower()}_{h['tic_id']}", arm=arm, host=h) for h in manifest['cohorts']['M4'])
    return rows


def prepare_injection(request_path):
    """Subcommand in the injection environment, using saved truth without a draw."""
    req = read(request_path)
    try:
        case, info = injection.write_injection(req['host'], Path(req['root']), req['case_id'], req['truth'],
                                               Path(req['scratch']), ri.sha256_file)
        if case is None:
            raise RuntimeError('INJECTION_FAILED: ' + str(info))
        check = injection.fidelity(req['host'], Path(req['root']), case, Path(req['scratch']), info,
                                   req['truth'], ri.sha256_file)
        model = transit_model.model_check(req['truth'], req['truth']['backend'])
        if not check['passed'] or not model['passed']:
            raise ControlFailure('INJECTION_FIDELITY_OR_MODEL_FAILED')
        save(req['result'], dict(status='OK', case=case, fidelity=check, model_check=model))
    except Exception as exc:
        save(req['result'], dict(status='CONTROL_FAILED' if isinstance(exc, ControlFailure) else 'ERROR',
                                error=f'{type(exc).__name__}: {exc}'))


def prepare_case(row, root, scratch, injection_python, repo):
    h, cid, arm = row['host'], row['case_id'], row['arm']
    base = runner._case_from_host(h, cid)
    # Verify originals before deriving copies; stale hashes are a control failure.
    for path in runner.job_inputs(dict(case=base, root=str(root))):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    routing.check_case(base, root)
    if arm == 'M1':
        req = dict(host=h, case_id=cid, truth=row['draw']['truth'], root=str(root), scratch=str(scratch),
                   result=str(scratch / 'injection_result.json'))
        request = scratch / 'injection_request.json'
        save(request, req)
        proc = subprocess.run([str(injection_python), '-B', '-m', 'm_harness.cohort_run', 'prepare-injection',
                               '--request', str(request)], cwd=repo, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError('INJECTION_ENVIRONMENT_FAILED: ' + proc.stderr)
        prepared = read(req['result'])
        if prepared['status'] == 'CONTROL_FAILED':
            raise ControlFailure(prepared['error'])
        if prepared['status'] != 'OK':
            raise RuntimeError(prepared['error'])
        return dict(case=prepared['case'], root=str(scratch), arm=arm,
                    originals=[r['sha256'] for r in h['lightcurves']] + ([h['tpf_sha256']] if h['tpf_fits'] else [])), prepared
    if arm == 'N1':
        case = runner._copy_case(base, root, scratch, cid)
        transform = nulls.transform_case(case, scratch, ri.sha256_file)
        if not transform['ok']:
            raise ControlFailure('NULL_TRANSFORM_FAILED: ' + str(transform))
        return dict(case=case, root=str(scratch), arm=arm), dict(transform=transform)
    return dict(case=base, root=str(root), arm=arm), {}


def noise_pregate(ald, manifest, root, scratch_root, bound, out):
    """Must finish before looking at any saved or new null detector result."""
    path = out / 'noise_gate.json'
    if path.exists():
        return bound.load_verified_item_record(path, required_keys=('gate', 'rows'))
    import lightkurve as lk
    rows, failures = [], []
    for h in manifest['cohorts']['M4']:
        row = dict(host=h, case_id='n1_' + h['tic_id'], arm='N1')
        with tempfile.TemporaryDirectory(prefix='s5-null-', dir=scratch_root) as tmp:
            try:
                job, _ = prepare_case(row, root, Path(tmp), sys.executable, Path(__file__).resolve().parents[1])
            except (ControlFailure, FileNotFoundError) as exc:
                failures.append(dict(tic_id=h['tic_id'], error=f'{type(exc).__name__}: {exc}'))
                continue
            tu, xu = fidelity.production_arrays(ald, lk.read(str(root / h['source_fits'])))
            tn, xn = fidelity.production_arrays(ald, lk.read(str(Path(tmp) / job['case']['source_fits'])))
            rows.append(dict(fidelity.star_row(h['tic_id'], tu, xu, tn, xn), sector=h['sector']))
    gate = fidelity.fidelity_gate(rows, [h['tic_id'] for h in manifest['cohorts']['M4']], m4=True)
    payload = dict(gate=gate, rows=rows, evaluation_failures=failures)
    bound.save_item_record(path, payload)
    return payload


def error_record(row, status, message):
    return dict(case_id=row['case_id'], status=status, message=message, candidate=None, peaks=[], journal=[],
                host_end=None, archival_diagnostics={}, search=dict(top_peaks=[], times_file=None, times_sha256=None),
                runtime_s=0)


def run_attempt(ald, row, root, scratch_root, injection_python, repo, out, bound, number, config, prereg):
    cid = row['case_id']
    attempt_id = f'{cid}_attempt_{number:04d}'
    dest = out / 'attempts' / cid / f'{number:04d}'
    dest.mkdir(parents=True, exist_ok=False)
    bound.save_item_record(dest / 'started.json', dict(case_id=cid, arm=row['arm'], attempt_id=attempt_id))
    prepared_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix='s5-case-', dir=scratch_root) as tmp:
            source_bytes = sum(Path(p).stat().st_size for p in runner.job_inputs(
                dict(case=runner._case_from_host(row['host'], cid), root=str(root))) if Path(p).is_file())
            if row['arm'] in ('M1', 'N1') and 3 * source_bytes > config.get('max_live_injection_bytes', 20_000_000_000):
                raise ControlFailure('SCRATCH_BUDGET_EXCEEDED_BEFORE_WRITE')
            job, prep = prepare_case(row, root, Path(tmp), injection_python, repo)
            prepared_ok = True
            live_bytes = sum(p.stat().st_size for p in Path(tmp).rglob('*') if p.is_file())
            if live_bytes > config.get('max_live_injection_bytes', 20_000_000_000):
                raise ControlFailure('SCRATCH_BUDGET_EXCEEDED')
            bound.save_item_record(dest / 'prepared.json', dict(case=job['case'], preparation=prep,
                                                               live_generated_bytes=live_bytes))
            job['attempt_id'] = attempt_id
            result = runner.run_cases(ald, run_dir=dest / 'run', run_id=bound.run_id + '_' + attempt_id,
                                      jobs=[job], config=config, repo_dir=repo, preregistration_path=prereg)
            item = result['records'][cid]
            if not item['routing_check']['passed']:
                raise ControlFailure('ROUTING_LEDGER_FAILED: ' + cid)
    except (ControlFailure, routing.RoutingError, ri.RunIdentityError):
        raise
    except (KeyboardInterrupt, SystemExit):
        bound.save_item_record(dest / 'interrupted.json', dict(case_id=cid, attempt_id=attempt_id, status='INTERRUPTED'))
        raise
    except Exception as exc:
        item = dict(record=error_record(row, 'SKIP_NO_DATA' if isinstance(exc, FileNotFoundError) else 'ERROR',
                                       f'{type(exc).__name__}: {exc}'), times_path=None,
                    preparation_failed=not prepared_ok)
    # Keep the nested detector record and search timestamps, with independent hashes.
    detector_record = dest / 'run' / 'records' / (cid + '.json')
    payload = dict(case_id=cid, arm=row['arm'], attempt_id=attempt_id, item=item,
                   detector_record_path=str(detector_record) if detector_record.exists() else None,
                   detector_record_sha256=ri.sha256_file(detector_record) if detector_record.exists() else None)
    bound.save_item_record(dest / 'result.json', payload)
    return payload


def case_result(ald, row, root, scratch_root, injection_python, repo, out, bound, config, prereg):
    """One retry for every completed ERROR/SKIP; interrupted attempts always get a new ID."""
    parent = out / 'attempts' / row['case_id']
    completed, history = [], []
    numbers = sorted(parent.iterdir()) if parent.exists() else []
    for directory in numbers:
        start = bound.load_verified_item_record(directory / 'started.json', required_keys=('case_id', 'attempt_id'))
        path = directory / 'result.json'
        if not path.exists():
            # A recorded start without a completed result is an interrupted attempt, retained as such.
            if not (directory / 'interrupted.json').exists():
                bound.save_item_record(directory / 'interrupted.json',
                                       dict(case_id=row['case_id'], status='INTERRUPTED', reason='resume after incomplete attempt'))
            history.append(dict(attempt_id=start['attempt_id'], status='INTERRUPTED'))
            continue
        payload = bound.load_verified_item_record(path, required_keys=('item', 'attempt_id'))
        if payload['detector_record_path'] and ri.sha256_file(payload['detector_record_path']) != payload['detector_record_sha256']:
            raise ri.RunIdentityError('DETECTOR_RECORD_CHANGED')
        item = payload['item']
        if item.get('times_path') and ri.sha256_file(item['times_path']) != item['times_sha256']:
            raise ri.RunIdentityError('SEARCH_TIMES_CHANGED')
        completed.append(payload)
        history.append(dict(attempt_id=payload['attempt_id'], status=item['record']['status'],
                            message=item['record'].get('message')))
    if any(p['item']['record']['status'] not in ('ERROR', 'SKIP_NO_DATA') for p in completed[:-1]):
        raise ri.RunIdentityError('ATTEMPT_AFTER_SUCCESS')
    while not completed or (completed[-1]['item']['record']['status'] in ('ERROR', 'SKIP_NO_DATA') and len(completed) < 2):
        number = len(numbers) + 1
        payload = run_attempt(ald, row, root, scratch_root, injection_python, repo, out, bound, number, config, prereg)
        numbers.append(parent / f'{number:04d}')
        completed.append(payload)
        history.append(dict(attempt_id=payload['attempt_id'], status=payload['item']['record']['status'],
                            message=payload['item']['record'].get('message')))
    return completed[-1]['item'], history


def build_tables(ald, manifest, catalogs, draws, design, plan, items, gate):
    m1, m3 = [], {k: [] for k in ['M3', *manifest['diagnostic_strata']]}
    m2_items, m2_design, matched = {}, {}, []
    crowd = {}
    for row in plan:
        cid, h, arm = row['case_id'], row['host'], row['arm']
        if arm == 'M1':
            d, truth = row['draw'], row['draw']['truth']
            if truth is None or items[cid].get('preparation_failed'):
                o = dict(case_id=cid, stage='PLANNING', line='DRAW_FAILED' if truth is None else 'INJECTION_FAILED', search_recovered=False,
                         matched_candidate=False, matched_candidate_or_review=False, candidate_sde=None, matched_sde=None)
            else:
                it = items[cid]
                o = estimands.case_outcome(it['record'], truth, runner.load_times(it))
                if o['matched_candidate']:
                    matched.append(dict(case_id=cid, candidate=it['record']['candidate'], truth=truth,
                                        r_star=h['stellar']['r_star']))
            rp = None if truth is None else truth['k'] * h['stellar']['r_star'] * estimands.RP_EARTH_PER_RSUN
            o.update(host=h['tic_id'], crowdsap=h['crowdsap'], snr_bin=d['snr_bin'], snr_target=d['snr_target'],
                     snr_achieved=None if truth is None else truth['snr_achieved'], rp_true_earth=rp,
                     period_bin='not drawn' if truth is None else runner._label(runner.DRAFT_PERIOD_BINS, truth['period']),
                     radius_bin='not drawn' if rp is None else runner._label(runner.DRAFT_RADIUS_BINS, rp))
            m1.append(o)
        elif arm == 'M2':
            refs = {r['catalog']['pl_name']: r for r in catalogs['M2']['hosts'][h['tic_id']]['reference_planets']}
            for i, g in enumerate(manifest['M2_reference_gates'][h['tic_id']]):
                key = cid + '_p' + str(i)
                r = refs[g['planet']]
                ref = r.get('selected_reference') if r['reference_status'].startswith('VALID_') else None
                truth = None if ref is None else dict(period=ref['period_days'], t0=ref['epoch_btjd'], t14=ref['T14_hours'] / 24)
                m2_design[key] = dict(host=h['tic_id'], target_planet=g['planet'], truth=truth, crowdsap=h['crowdsap'],
                                      in_domain=g['domain']['in_domain'], domain_reasons=g['domain']['reasons'],
                                      snr=g.get('SNR_ref'), sigma_p_cat=None if ref is None else ref.get('sigma_P_days'))
                m2_items[key] = dict(items[cid], record=dict(items[cid]['record'], case_id=key))
        elif arm.startswith('M3'):
            it = items[cid]
            o = estimands.case_outcome(it['record'])
            selected = catalogs['M3']['hosts'][h['tic_id']].get('selected_signal')
            cat = {} if selected is None else selected['catalog']
            period = cat.get('Per') if h.get('M3_source') == 'Prsa_EB' else cat.get('pl_orbper')
            candidate = it['record'].get('candidate')
            o.update(host=h['tic_id'], crowdsap=h['crowdsap'], source=h.get('M3_source'),
                     candidate_relation=None if not period or candidate is None else
                     matching.period_relation(candidate['discovery_period_days'], float(period)))
            m3[arm].append(o)
        elif cid in items:
            crowd[cid] = h['crowdsap']
    labels = [injection.bin_label(*b) for b in design.snr_bins]
    tables = dict(m1_rows=m1, m1_attrition=estimands.attrition_table(m1, len(m1)),
                  m1_draw_failure_hosts=estimands.m1_draw_failure_hosts(draws, manifest['cohorts']['M1']))
    for level in ('candidate', 'search'):
        tables['m1_completeness_' + level] = estimands.m1_completeness(m1, labels, level=level,
            cohort_sha256=manifest.get('M1_population_cohort_sha256'), draw_records=draws)
        gain = estimands.grey_zone_gain(m1, 'snr_target', level=level, bin_edges=[b for b in design.snr_bins if b[0] < 10])
        gain['aggregation_equivalence_verified'] = injection.realized_grey_counts(
            [r['draw'] for r in plan if r['arm'] == 'M1'])['passed']
        tables['grey_zone_gain_' + level] = gain
    m2_rows = runner.m2_table_rows(m2_items, m2_design, manifest, catalogs['M2'], include_missing_truth=True)
    tables.update(m2_rows=m2_rows, m2_false_rejection=estimands.m2_false_rejection(m2_rows,
                  snr_bins=((0., 10.), (10., 25.), (25., float('inf')))),
                  m2_attrition=estimands.attrition_table(m2_rows, len(m2_rows)),
                  m3={arm: estimands.m3_false_positive(rows) for arm, rows in m3.items()}, m3_rows=m3,
                  m3_by_source={source: estimands.m3_false_positive([r for r in m3['M3'] if r['source'] == source])
                                for source in sorted({r['source'] for r in m3['M3']})},
                  m5_parameters=estimands.m5_parameters(matched))
    if gate['gate']['blocks_m4']:
        tables['m4_n1'] = tables['m4_n2'] = dict(status='NOT_RUN', reason='noise-only pre-gate failed')
        tables['grey_zone_decision'] = estimands.grey_zone_decision(None, None)
    else:
        for arm in ('N1', 'N2'):
            tables['m4_' + arm.lower()] = estimands.m4_false_alarm(
                [items[r['case_id']]['record'] for r in plan if r['arm'] == arm], crowd, arm)
        tables['grey_zone_decision'] = estimands.grey_zone_m4_decision(
            tables['grey_zone_gain_candidate']['pooled']['dc_grey']['rate'], tables['m4_n1'])
        tables['m4_tail_report'] = fidelity.m4_tail_report(gate['rows'],
            [h['tic_id'] for h in manifest['cohorts']['M4']],
            {r['host']['tic_id']: estimands.case_outcome(items[r['case_id']]['record']) for r in plan if r['arm'] == 'N1'})
    tables['m2_secondary_diagnostics'] = estimands.archival_secondary_diagnostics(ald,
        [item['record'] for item in m2_items.values()],
        p_cat_by_case={cid: d['truth']['period'] for cid, d in m2_design.items() if d['truth'] is not None},
        sigma_p_cat_by_case={cid: d['sigma_p_cat'] for cid, d in m2_design.items()})
    tables['m3_secondary_diagnostics'] = {arm: estimands.archival_secondary_diagnostics(ald,
        [items[r['case_id']]['record'] for r in plan if r['arm'] == arm]) for arm in m3}
    if getattr(ald, 'RC2_RECORD_ALL_GATES', False) or getattr(ald, 'RC2_SWITCHES', ()):
        from . import rc2_reporting
        tables['rc2_diagnostics'] = rc2_reporting.from_run(tables, plan, items, manifest, m2_design)
    return tables


def run(ald, *, manifest_path, draws_dir, root, scratch_root, out, config_path, prereg, preflight_path,
        repo=None, on_case=None):
    repo = Path(repo or Path(__file__).resolve().parents[1])
    root, scratch_root, out = Path(root), Path(scratch_root), Path(out)
    config = read(config_path)
    pre = read(preflight_path)
    if not pre['passed'] or pre['inventory']['sha256'] != ri.sha256_file(repo / 'aldebaran.py'):
        raise ControlFailure('FROZEN_PREFLIGHT_REQUIRED')
    if ri.sha256_file(preflight_path) != config['preflight_sha256']:
        raise ControlFailure('PREFLIGHT_CHANGED')
    if not Path(prereg).is_file():
        raise ControlFailure('PREREGISTRATION_REQUIRED')
    manifest, catalogs, draws, design, inputs = load_inputs(manifest_path, draws_dir, config)
    expected = {arm: {h['tic_id'] for h in hosts} for arm, hosts in
                dict(manifest['cohorts'], **manifest['diagnostic_strata']).items()}
    actual = {arm: {str(h['host']) for h in group['hosts']} for arm, group in pre['cohorts'].items()}
    if actual != expected or any(group['n_passed'] != group['n_hosts'] for group in pre['cohorts'].values()):
        raise ControlFailure('PREFLIGHT_COHORT_MISMATCH')
    plan = plan_cases(manifest, draws)
    # A separate scratch root prevents generated FITS from ever replacing original inputs or saved records.
    for protected in (root.resolve(), out.resolve(), repo.resolve()):
        if scratch_root.resolve().is_relative_to(protected) or protected.is_relative_to(scratch_root.resolve()):
            raise ControlFailure('SCRATCH_MUST_BE_SEPARATE')
    scratch_root.mkdir(parents=True, exist_ok=True)
    bound = ri.BoundRun.open_or_resume(out / 'run_identity.json', run_id=config['run_id'],
        runner='m_harness.cohort_run', code_files=runner.harness_code_files(repo), repo_dir=repo,
        config=dict(config, root=str(root.resolve()), scratch_root=str(scratch_root.resolve())),
        preregistration_path=prereg, input_paths=inputs + [str(preflight_path), str(config_path)], input_spec=plan)
    items, ledger = {}, []
    try:
        gate = noise_pregate(ald, manifest, root, scratch_root, bound, out)
        for row in plan:
            cid, arm = row['case_id'], row['arm']
            if arm == 'M1' and row['draw']['truth'] is None:
                ledger.append(dict(case_id=cid, arm=arm, status='DRAW_FAILED', attempts=[]))
                continue
            if arm in ('N1', 'N2') and gate['gate']['blocks_m4']:
                ledger.append(dict(case_id=cid, arm=arm, status='NOT_RUN', reason='M4 noise-only pre-gate failed', attempts=[]))
                continue
            item, attempts = case_result(ald, row, root, scratch_root, config['injection_python'], repo,
                                         out, bound, config, prereg)
            items[cid] = item
            ledger.append(dict(case_id=cid, arm=arm, status=item['record']['status'], attempts=attempts))
            if on_case:
                on_case(cid, len(ledger))
        tables = build_tables(ald, manifest, catalogs, draws, design, plan, items, gate)
        bound.save_item_record(out / 'tables.json', dict(tables=tables, ledger=ledger, noise_gate=gate['gate']))
        bound.record_completion(extra=dict(n_planned=len(plan), n_final=len(ledger)))
        return dict(tables=tables, ledger=ledger, noise_gate=gate['gate'])
    except (KeyboardInterrupt, SystemExit):
        bound.record_interruption('cohort driver interrupted', completed_count=len(ledger))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare-injection')
    prep.add_argument('--request', required=True)
    run_parser = commands.add_parser('run')
    for name in ('manifest', 'draws', 'root', 'scratch', 'out', 'config', 'prereg', 'preflight'):
        run_parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-injection':
        prepare_injection(args.request)
    else:
        run(prodpath.load_production(), manifest_path=args.manifest, draws_dir=args.draws, root=args.root,
            scratch_root=args.scratch, out=args.out, config_path=args.config, prereg=args.prereg,
            preflight_path=args.preflight)


if __name__ == '__main__':
    main()
