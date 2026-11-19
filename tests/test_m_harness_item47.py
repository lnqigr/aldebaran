import copy
import unittest
from m_harness import estimands as est,runner
from test_m_harness_item46 import failed,outcome

class Item47(unittest.TestCase):
    def test_fewer_than_three_events_and_full_population_guard(self):
        for events in (0,1,2,3,4,None,True,float('nan')):
            row=failed(); t=row['attempts'][0]['tries'][0]
            t.update(depth=.001,k=.03,try_outcome='T0_CONDITIONS',n_discovery_events=events)
            self.assertEqual(est.m1_draw_assessment(row,.57)['accepted'],type(events) is int and events<3)

    def test_population_exhaustion_still_needs_domain_witness(self):
        row=failed()
        for a in row['attempts']:
            for t in a['tries']: t.update(depth=.001,k=.03,try_outcome='T0_CONDITIONS',n_discovery_events=2)
        self.assertFalse(est.m1_draw_assessment(row,.57)['accepted'])
        row['attempts'][0]['tries'][0].update(valid=False,stop_reason='DEPTH_OUTSIDE_K_BRACKET',try_outcome='SOLVE_FAILED')
        self.assertTrue(est.m1_draw_assessment(row,.57)['accepted'])

    def test_scoped_limit_31_and_32_rejection(self):
        scope=est.ITEM47_COHORT_SHA256
        self.assertEqual(est.m1_planning_limit(0,scope),31)
        self.assertEqual(est.m1_planning_limit(0),30)
        self.assertEqual(est.m1_planning_limit(0,'different cohort'),30)
        self.assertTrue(31<=est.m1_planning_limit(0,scope))
        self.assertFalse(32<=est.m1_planning_limit(0,scope))
        self.assertEqual([est.m1_planning_limit(i,scope) for i in range(6)],[31,30,30,30,45,45])

    def test_annotation_on_tables_and_report_generator(self):
        labels=['[4, 6)','[6, 8)','[8, 10)','[10, 15)','[15, 25)','[25, 35]']
        rows=[]; draws={str(i):[] for i in range(300)}
        for label, failures in zip(labels,[31,11,6,3,6,7]):
            for i in range(300):
                rows.append(dict(outcome(str(i),i<failures),snr_bin=label,case_id=f'{i}:{label}'))
                draws[str(i)].append(dict(snr_bin=label,status='DRAW_FAILED' if i<failures else 'DRAWN'))
        for level in ('candidate','search'):
            t=est.m1_completeness(rows,labels,level=level,n_boot=10,cohort_sha256=est.ITEM47_COHORT_SHA256,draw_records=draws)
            self.assertEqual(t['population']['annotation'],est.ITEM47_ANNOTATION)
            for c in [t['pooled'],*t['by_crowding'].values(),*t['by_period_bin'].values(),*t['by_radius_bin'].values()]:
                self.assertEqual(c['population']['status'],'PASS_WITH_AMENDMENT')
            text=' '.join(runner.m1_report_notes(t))
            self.assertIn(est.ITEM47_ANNOTATION,text)
            for word in ('Low-SNR floor','High-SNR ceiling','141/150','289/300','147/150','96.4 %'):
                self.assertIn(word,text)
        normal=est.m1_completeness(rows,labels,n_boot=10)
        self.assertNotIn('annotation',normal['population'])
        g=est.grey_zone_gain(rows,'snr_target',n_boot=10,cohort_sha256=est.ITEM47_COHORT_SHA256)
        self.assertFalse(g['population']['aggregation_equivalence_claimed'])

if __name__=='__main__': unittest.main()
