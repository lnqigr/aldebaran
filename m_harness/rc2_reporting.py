"""Item 48/49 development diagnostics; frozen headline estimands are unchanged."""
import itertools
import math

from . import stats


SWITCHES = ('A2', 'ARCHIVAL_COVERAGE', 'V_SHAPE_FLAG', 'DEPTH_UPPER', 'DEPTH_FULL_T14')


def _rate(rows, flag='matched_candidate'):
    return stats.rate(sum(bool(r[flag]) for r in rows), len(rows), 'two-sided')


def candidate_strata(records):
    """Candidate host counts; unknown legacy status remains explicit."""
    candidates = [r for r in records if r['status'] == 'CANDIDATE']
    strata = {'PASS': 0, 'ODD_EVEN_NOT_TESTABLE': 0, 'UNKNOWN': 0}
    for r in candidates:
        status = (r.get('candidate') or {}).get('odd_even_status')
        key = 'PASS' if status in ('PASS', 'ASSESSED') else status
        strata[key if key in strata else 'UNKNOWN'] += 1
    return dict(n_records=len(records), pooled=len(candidates), by_odd_even_status=strata)


def m1_diagnostics(rows):
    """Rows join frozen outcome fields to draw metadata; no population filtering."""
    def table(group):
        matched = [r for r in group if r['search_recovered']]
        return dict(n=len(group), final=_rate(group), search=_rate(group, 'search_recovered'),
                    conditional_vetting=_rate(matched))

    def event_bin(row):
        n = row.get('n_discovery_events')
        if n is None:
            return 'UNKNOWN'
        return '3' if n == 3 else ('4-5' if 4 <= n <= 5 else ('6-9' if 6 <= n <= 9 else ('10+' if n >= 10 else '<3')))

    def duration_bin(row):
        t = row.get('t14_days')
        return 'UNKNOWN' if t is None or not math.isfinite(t) else ('<=0.18d' if t <= .18 else '>0.18d')

    out = {}
    for name, sel in (('all_snr', rows), ('snr_ge_10', [r for r in rows if r['snr_target'] >= 10])):
        out[name] = dict(pooled=table(sel),
            by_discovery_events={k: table([r for r in sel if event_bin(r) == k])
                                 for k in ('3', '4-5', '6-9', '10+', '<3', 'UNKNOWN')},
            by_t14={k: table([r for r in sel if duration_bin(r) == k]) for k in ('<=0.18d', '>0.18d', 'UNKNOWN')})
    external = [r for r in rows if r.get('truth_period_days') is not None and r.get('truth_depth_ppm') is not None
                and .5 <= r['truth_period_days'] <= 16 and r['truth_depth_ppm'] >= 300]
    out['comparison_domain'] = dict(all_snr=table(external),
        snr_ge_10=table([r for r in external if r['snr_target'] >= 10]),
        metadata_unavailable=sum(r.get('truth_period_days') is None or r.get('truth_depth_ppm') is None for r in rows))
    return out


def m2_diagnostics(rows):
    """Item 49.7: the headline keeps every in-domain reference planet (the preregistered M2 denominator, RC1 189);
    strata use observed counts. The caller passes in-domain rows only."""
    def table(sel):
        groups = {'all_snr': sel, 'snr_unknown': [r for r in sel if r.get('snr') is None]}
        for name, low, high in (('snr_lt_10', -math.inf, 10), ('snr_10_25', 10, 25),
                                ('snr_ge_25', 25, math.inf), ('snr_ge_10', 10, math.inf)):
            groups[name] = [r for r in sel if r.get('snr') is not None and low <= r['snr'] < high]
        return {k: _rate(v) for k, v in groups.items()}
    out = dict(headline=table(rows), observed_count_unknown=sum(r.get('n_observed_transits') is None for r in rows))
    for n in (1, 3):
        out[f'observed_ge_{n}'] = table([r for r in rows if r.get('n_observed_transits') is not None and r['n_observed_transits'] >= n])
    out['comparison_domain'] = table([r for r in rows if r.get('reference_period_days') is not None
        and r.get('reference_depth_ppm') is not None and .5 <= r['reference_period_days'] <= 16
        and r['reference_depth_ppm'] >= 300])
    return out


def from_run(tables, plan, items, manifest, m2_design):
    """Join reporting metadata without changing any frozen table or denominator."""
    by_case = {r['case_id']: r for r in plan}
    m1 = []
    for row in tables['m1_rows']:
        truth = by_case[row['case_id']]['draw']['truth']
        if truth is not None:
            m1.append(dict(row, n_discovery_events=truth['n_discovery_events'], t14_days=truth['t14'],
                           truth_period_days=truth['period'], truth_depth_ppm=truth['depth'] * 1e6))
    m2 = []
    for row in tables['m2_rows']:
        if row.get('in_domain') is not True:
            continue
        d = m2_design[row['case_id']]
        gate = next(g for g in manifest['M2_reference_gates'][d['host']] if g['planet'] == d['target_planet'])
        m2.append(dict(row, n_observed_transits=(gate.get('snr_support') or {}).get('N_ev'),
                       reference_period_days=None if d['truth'] is None else d['truth']['period'],
                       reference_depth_ppm=gate.get('reference_depth_ppm')))
    candidates = {}
    for arm in sorted({r['arm'] for r in plan}):
        candidates[arm] = candidate_strata([items[r['case_id']]['record'] for r in plan
            if r['arm'] == arm and r['case_id'] in items])
    return dict(m1=m1_diagnostics(m1), m2=m2_diagnostics(m2), candidate_strata=candidates)


def score_trace(traces, enabled):
    """Full rank cascade from production metrics, including secondary mask contexts.

    Missing downstream measurements are never treated as passes. The caller keeps
    NOT_EVALUABLE rows in its ledger and must not report their rate as measured.
    """
    enabled = frozenset(enabled)
    if enabled - set(SWITCHES):
        raise ValueError('UNKNOWN_RC2_SWITCH')
    ranked, candidates, reviews = [], [], []
    eb_gates = {'SECONDARY', 'ODD_EVEN', 'TPF', 'RADIUS', 'DEPTH'}
    for trace in sorted(traces, key=lambda r: r['rank']):
        rank = trace['rank']
        gates = {g['gate']: g for g in trace['gates']}
        rejects = []
        for name, gate in gates.items():
            if gate['status'] == 'NOT_EVALUATED' and name != 'SECONDARY':
                continue  # not reached: an earlier gate ended the rank, or the gate does not apply
            rejected = gate['status'] == 'REJECT'
            m = gate['metrics']
            if name == 'DEPTH':
                upper = m['full_t14_depth_ppm'] > 33000 if 'DEPTH_FULL_T14' in enabled else m['core_depth_ppm'] > (35000 if 'DEPTH_UPPER' in enabled else 30000)
                rejected = upper or m['core_depth_ppm'] < 180
            elif name == 'V_SHAPE':
                rejected = bool(m.get('warning')) and 'V_SHAPE_FLAG' not in enabled
            elif name == 'ARCHIVAL' and trace.get('archival_coverage_shortfall'):
                rejected = 'ARCHIVAL_COVERAGE' not in enabled or trace['sde'] < 6.5
            elif name == 'SECONDARY':
                if gate['status'] == 'NOT_EVALUATED' and not trace.get('secondary_variants'):
                    continue
                key = ','.join(str(r['rank']) for r in candidates)
                variant = trace.get('secondary_variants', {}).get(key)
                if variant is None:
                    raise ValueError('NOT_EVALUABLE: missing secondary mask context')
                rejected = variant['rejected']
            if rejected:
                rejects.append(name)
        if rejects:
            ranked.append(dict(rank=rank, disposition='REJECT', gate=rejects[0],
                               archival_secondary_rejection=trace.get('archival_secondary_rejection', False)))
            continue
        if not trace.get('complete') or 'DOWNSTREAM_UNAVAILABLE' in gates:
            raise ValueError('NOT_EVALUABLE: incomplete gate trace')
        a2 = ('A2' in enabled and trace['odd_even_reason'] == 'UNREPLICATED_PARITY_NO_VALIDATED_PRIOR'
              and (trace['odd_even_diagnostics']['n_odd'] < 2 or trace['odd_even_diagnostics']['n_even'] < 2))
        # Secondary NOT_EVALUATED is context-dependent, as is its review cap.
        key = ','.join(str(r['rank']) for r in candidates)
        variant = trace['secondary_variants'][key]
        caps = [c for c in trace['review_caps'] if c != 'SECONDARY_NOT_EVALUATED']
        if variant['details'].get('not_evaluated_reason'):
            caps.append('SECONDARY_NOT_EVALUATED')
        hold = bool(caps or trace['vetting_holds']) or (trace['odd_even_status'] in ('INCONCLUSIVE', 'ODD_EVEN_NOT_TESTABLE') and not a2)
        links = trace['links']
        linked_rejection = any(links.get(str(p['rank'])) and (p.get('gate') in eb_gates or
            (p.get('gate') == 'ARCHIVAL' and p.get('archival_secondary_rejection'))) for p in ranked if p['disposition'] == 'REJECT')
        linked_review = any(links.get(str(p['rank'])) for p in reviews if p.get('cause') != 'SHARED_EVENTS_WITH_STRONGER_REVIEW')
        linked_candidate = any(links.get(str(p['rank'])) for p in candidates)
        cause = ('HOLD' if hold else ('SHARED_EVENTS_WITH_STRONGER_REJECTION' if linked_rejection else
                 ('SHARED_EVENTS_WITH_STRONGER_REVIEW' if linked_review else ('SHARED_EVENTS_WITH_CANDIDATE' if linked_candidate else None))))
        row = dict(rank=rank, disposition='REVIEW' if cause else 'CANDIDATE', cause=cause)
        ranked.append(row)
        if cause != 'SHARED_EVENTS_WITH_CANDIDATE':
            (reviews if cause else candidates).append(row)
    return dict(first_candidate_rank=candidates[0]['rank'] if candidates else None, ranks=ranked)


def score_combinations(traces):
    return {','.join(enabled) or 'RC1': score_trace(traces, enabled)
            for bits in itertools.product((False, True), repeat=len(SWITCHES))
            for enabled in [tuple(k for k, bit in zip(SWITCHES, bits) if bit)]}
