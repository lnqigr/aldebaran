import copy
import unittest
from m_harness import estimands as e


def failed():
    sol=dict(valid=True,stop_reason='CONVERGED',snr_achieved=20.,depth=.04,k=.2,try_outcome='DEPTH_OUTSIDE_DOMAIN')
    return dict(truth=None,status='DRAW_FAILED',snr_target=20.,attempts=[dict(draw=i, snr_target=20., draw_outcome='INNER_LOOP_TRIES_FAILED', tries=[dict(sol) for _ in range(20)]) for i in range(50)])


def outcome(host='a',lost=False):
    return dict(case_id=host,host=host,stage='PLANNING' if lost else 'NONE',line='DRAW_FAILED' if lost else 'MATCHED_CANDIDATE',
                snr_bin='[4, 6)',snr_target=5.5,crowdsap=.95,matched_candidate=not lost,matched_candidate_or_review=not lost,
                search_recovered=not lost,candidate_sde=None if lost else 7.,matched_sde=None if lost else 7.,
                period_bin='short',radius_bin='small',rp_true_earth=1.)


class Item46(unittest.TestCase):
    def test_every_denominator_and_ledger(self):
        rows=[outcome(),outcome('b'),outcome('a',True)]
        for level in ('candidate','search'):
            table=e.m1_completeness(rows,['[4, 6)'],level=level,n_boot=30,min_cell=2)
            cells=[table['pooled'],table['by_snr_bin']['[4, 6)'],*table['by_crowding'].values(),
                   *table['by_period_bin'].values(),*table['by_radius_bin'].values(),table['headline_rp_ge_0_8'],table['radius_below_2']]
            cells+= [c for strata in table['by_snr_bin_and_crowding'].values() for c in strata.values()]
            for cell in cells:
                self.assertEqual(cell['n_evaluated'],2)
                self.assertEqual(cell['primary_completeness']['rate'],1.)
                self.assertEqual(cell['unconditioned_completeness']['rate'],2/3)
                self.assertEqual(cell['n_draw_failed'],1)
            grey=e.grey_zone_gain(rows,'snr_target',n_boot=30,bin_edges=[(4,6)])
            self.assertEqual(grey['pooled']['n'],2)
            self.assertEqual(grey['bin_weighted']['value'],1.)
        self.assertEqual(e.attrition_table(rows,3)['stages'][0]['lost'],1)

    def test_other_planning_losses_stay(self):
        row=outcome('b',True); row['line']='INJECTION_FAILED'
        self.assertEqual(e.completeness_cell([outcome(),row],'candidate')['n'],2)

    def test_bootstrap_excludes_failed_host_rows(self):
        rows=[outcome(),outcome(),outcome('b'),outcome('c',True)]
        c=e.completeness_cell(rows,'candidate',n_boot=30)
        self.assertEqual(c['matched_candidate']['n_hosts'],2)
        self.assertEqual(c['matched_candidate']['host_cluster']['lo'],1.)

    def test_verified_domain_exhaustion(self):
        row=failed()
        self.assertTrue(e.m1_draw_assessment(row,.57)['accepted'])
        row['attempts'][0]['tries'][0]=dict(valid=False,stop_reason='NOISE_INVALID_START',try_outcome='SOLVE_FAILED')
        self.assertTrue(e.m1_draw_assessment(row,.57)['accepted'])

    def test_noise_only_family_rejected(self):
        row=failed()
        for a in row['attempts']:
            for t in a['tries']: t.update(valid=False,stop_reason='NOISE_INVALID_START',try_outcome='SOLVE_FAILED')
        self.assertFalse(e.m1_draw_assessment(row,.57)['accepted'])

    def test_k_range_only_family_accepted(self):
        for status in ('DEPTH_OUTSIDE_K_BRACKET','OUTSIDE_VALID_RANGE'):
            row=failed()
            for a in row['attempts']:
                for t in a['tries']: t.update(valid=False,stop_reason=status,try_outcome='SOLVE_FAILED')
            check=e.m1_draw_assessment(row,.57)
            self.assertTrue(check['accepted'])
            self.assertEqual(check['n_k_range_stops'],1000)
            self.assertEqual(check['binding_caps']['k_outside_model_range'],1000)

    def test_geometry_rejections_and_no_evidence(self):
        row=failed()
        for i,status in enumerate(('BELOW_AR_FLOOR','OUTSIDE_SEARCH_DOMAIN','NON_TRANSITING_GEOMETRY')):
            row['attempts'][i].update(tries=[],draw_outcome=status)
        self.assertTrue(e.m1_draw_assessment(row,.57)['accepted'])
        for a in row['attempts']: a.update(tries=[],draw_outcome='OUTSIDE_SEARCH_DOMAIN')
        self.assertFalse(e.m1_draw_assessment(row,.57)['accepted'])
        row=failed()
        for a in row['attempts']:
            for t in a['tries']: t.update(valid=False,stop_reason='NOT_CONVERGED_50',try_outcome='SOLVE_FAILED')
        self.assertFalse(e.m1_draw_assessment(row,.57)['accepted'])

    def test_in_domain_model_error_environment_and_incomplete_rejected(self):
        for patch in (dict(depth=.001,k=.03),dict(stop_reason='MODEL_ERROR'),dict(stop_reason='ENVIRONMENT_ERROR'),
                      dict(snr_achieved=18.),dict(try_outcome='T0_CONDITIONS')):
            row=failed(); row['attempts'][0]['tries'][0].update(patch)
            self.assertFalse(e.m1_draw_assessment(row,.57)['accepted'])
        row=failed(); row['attempts'].pop()
        self.assertFalse(e.m1_draw_assessment(row,.57)['accepted'])

    def test_radius_cap_and_host_table(self):
        row=failed()
        for a in row['attempts']:
            for t in a['tries']: t.update(depth=.02,k=.3,try_outcome='RADIUS_OUTSIDE_DOMAIN')
        self.assertTrue(e.m1_draw_assessment(row,.57)['accepted'])
        row.update(snr_bin='[15, 25)')
        h=dict(tic_id='a',stellar=dict(r_star=.57,teff=3500),lightcurves=[{}],discovery_baseline_days=27.)
        table=e.m1_draw_failure_hosts({'a':[row]},[h])
        self.assertEqual(table[0]['baseline'],27.)
        self.assertEqual(table[0]['failed_families'][0]['binding_caps']['rp_above_15_earth'],1000)

if __name__=='__main__': unittest.main()

