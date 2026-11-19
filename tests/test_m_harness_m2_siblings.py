import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from m_harness import estimands as E, runner
import test_m_harness_estimands as T
from test_m_harness_item46 import outcome


def inputs(multi=True):
    sibling = dict(period=4.1, t0=2001., t14=.1)
    truths = [T.TRUTH, sibling] if multi else [T.TRUTH]
    catalog = dict(hosts={'42': dict(reference_planets=[dict(
        catalog=dict(pl_name=str(i)), reference_status='VALID_CONSENSUS', selected_reference=dict(
            period_days=tr['period'], epoch_btjd=tr['t0'], T14_hours=tr['t14'] * 24))
        for i, tr in enumerate(truths)])})
    manifest = dict(M2_reference_gates={'42': [dict(planet=str(i), domain=dict(in_domain=True),
        SNR_status='OK', snr_support=dict(N_ev=3)) for i in range(len(truths))]})
    design = dict(target=dict(truth=dict(T.TRUTH), target_planet='0', host=42, crowdsap=.95,
                             in_domain=True, domain_reasons=[], snr=20.))
    p = 4.1 if multi else 3.
    rec = T.record('target', 'CANDIDATE', top=[T.peak(1, period=p)],
        journal=[T.jrow(1, 'CANDIDATE', period=p)], first_candidate=1,
        candidate=dict(T.cand(period=p), duration_hours=2.4))
    return dict(target=dict(record=rec)), design, manifest, catalog


class M2SiblingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.times_path = Path(self.tmp.name) / 'times.npy'
        np.save(self.times_path, T.TIMES)

    def inputs(self, multi=True):
        items, design, manifest, catalog = inputs(multi)
        items['target']['times_path'] = str(self.times_path)
        return items, design, manifest, catalog
    def rows(self, multi=True):
        return runner.m2_table_rows(*self.inputs(multi))

    def test_multi_planet_sibling_win_is_target_loss_on_runner_path(self):
        rows = self.rows()
        self.assertEqual(rows[0]['n_reference_planets'], 2)
        self.assertTrue(rows[0]['sibling_win'])
        self.assertFalse(rows[0]['matched_candidate'])
        table = E.m2_false_rejection(rows, n_boot=10)
        self.assertEqual(table['by_multiplicity']['multi-planet']['in_domain']['n'], 1)
        self.assertEqual(table['by_multiplicity']['single-planet']['in_domain']['n'], 0)
        self.assertEqual(table['in_domain']['rate'], 1.)
        self.assertEqual(table['sibling_wins']['n'], 1)

    def test_single_planet_recovers(self):
        row = self.rows(False)[0]
        self.assertEqual(row['n_reference_planets'], 1)
        self.assertTrue(row['matched_candidate'])
        self.assertFalse(row['sibling_win'])

    def test_target_at_reported_rank_is_never_its_own_sibling(self):
        for multi in (False, True):
            with self.subTest(multi=multi):
                items, design, manifest, catalog = self.inputs(multi)
                design['target']['truth']['snr_achieved'] = 20.
                items['target']['record'] = T.record('target', 'CANDIDATE',
                    top=[T.peak(1), T.peak(2)], journal=[T.jrow(1, 'REVIEW'), T.jrow(2, 'CANDIDATE')],
                    first_candidate=2, candidate=dict(T.cand(), duration_hours=2.4))
                row = runner.m2_table_rows(items, design, manifest, catalog)[0]
                self.assertFalse(row['matched_candidate'])
                self.assertFalse(row['sibling_win'])
                self.assertEqual(row['n_reference_planets'], 2 if multi else 1)
                self.assertEqual(E.m2_false_rejection([row], n_boot=10)['sibling_wins']['n'], 0)

    def test_distinct_sibling_with_same_ephemeris_is_still_a_sibling(self):
        items, design, manifest, catalog = self.inputs()
        refs = catalog['hosts']['42']['reference_planets']
        refs[1]['selected_reference'] = dict(refs[0]['selected_reference'])
        items['target']['record'] = T.record('target', 'CANDIDATE',
            top=[T.peak(1), T.peak(2)], journal=[T.jrow(1, 'REVIEW'), T.jrow(2, 'CANDIDATE')],
            first_candidate=2, candidate=dict(T.cand(), duration_hours=2.4))
        row = runner.m2_table_rows(items, design, manifest, catalog)[0]
        self.assertFalse(row['matched_candidate'])
        self.assertTrue(row['sibling_win'])

    def test_missing_target_identity_is_not_available(self):
        for identity in (None, 'absent'):
            items, design, manifest, catalog = self.inputs()
            design['target']['target_planet'] = identity
            row = runner.m2_table_rows(items, design, manifest, catalog)[0]
            self.assertIsNone(row['sibling_win'])
            self.assertEqual(row['sibling_win_status'], 'NOT_AVAILABLE')

    def test_unreadable_or_incomplete_catalog_is_not_available(self):
        items, design, manifest, catalog = self.inputs()
        for broken in (None, {}, dict(hosts={}), dict(hosts={'42': dict(reference_planets=[])})):
            with self.subTest(catalog=broken):
                rows = runner.m2_table_rows(items, design, manifest, broken)
                self.assertIsNone(rows[0]['n_reference_planets'])
                self.assertIsNone(rows[0]['sibling_win'])
                self.assertEqual(rows[0]['multiplicity_status'], 'NOT_AVAILABLE')
                table = E.m2_false_rejection(rows, n_boot=10)
                self.assertEqual(table['multiplicity_unavailable'], 1)
                self.assertEqual(table['sibling_wins']['unavailable'], 1)
        manifest['M2_reference_gates'] = {}
        self.assertIsNone(runner.m2_table_rows(items, design, manifest, catalog)[0]['n_reference_planets'])

    def test_bound_sidecar_loading_on_runner_path(self):
        items, design, manifest, catalog = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'references.json'
            data = json.dumps(catalog).encode()
            path.write_bytes(data)
            manifest['scoring_sidecars'] = dict(M2=dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
            row = runner.m2_table_rows(items, design, manifest)[0]
            self.assertEqual(row['n_reference_planets'], 2)
            self.assertTrue(row['sibling_win'])
            path.write_bytes(data + b' ')
            self.assertIsNone(runner.m2_table_rows(items, design, manifest)[0]['n_reference_planets'])
            self.assertIsNone(runner.load_m2_reference_catalog(manifest, Path(tmp) / 'absent'))
            path.write_bytes(b'{')
            manifest['scoring_sidecars']['M2']['sha256'] = hashlib.sha256(b'{').hexdigest()
            self.assertIsNone(runner.load_m2_reference_catalog(manifest))

    def test_missing_sibling_ephemeris_preserves_catalog_multiplicity(self):
        items, design, manifest, catalog = self.inputs()
        catalog['hosts']['42']['reference_planets'][1].update(reference_status='NO_TRANSIT_REFERENCE', selected_reference=None)
        row = runner.m2_table_rows(items, design, manifest, catalog)[0]
        self.assertEqual(row['n_reference_planets'], 2)
        self.assertIsNone(row['sibling_win'])
        self.assertEqual(row['sibling_win_status'], 'NOT_AVAILABLE')

    def test_no_discovery_event_is_derived_without_manifest_mutation(self):
        items, design, manifest, catalog = self.inputs()
        manifest['M2_reference_gates']['42'][1]['snr_support']['N_ev'] = 0
        before = copy.deepcopy(manifest)
        row = runner.m2_table_rows(items, design, manifest, catalog)[0]
        self.assertEqual(row['reference_gates'][0]['SNR_status'], 'OK')
        self.assertEqual(row['reference_gates'][1]['SNR_status'], 'NO_DISCOVERY_EVENT')
        self.assertEqual(manifest, before)

    def test_m1_draw_counts_must_agree_with_rows(self):
        rows = [outcome(), outcome('b', True)]
        draws = {'a': [dict(snr_bin='[4, 6)', status='DRAWN')],
                 'b': [dict(snr_bin='[4, 6)', status='DRAW_FAILED')]}
        kwargs = dict(n_boot=10, draw_records=draws)
        self.assertEqual(E.m1_completeness(rows, ['[4, 6)'], **kwargs)['pooled']['n_draw_failed'], 1)
        for bad in ([outcome(), outcome('b')], rows[:1], rows + [outcome('c')],
                    [rows[0], dict(rows[1], snr_bin='other')]):
            with self.subTest(rows=bad), self.assertRaisesRegex(ValueError, 'M1_DRAW_COUNT_MISMATCH'):
                E.m1_completeness(bad, ['[4, 6)'], **kwargs)

    def test_amended_population_requires_matching_draw_counts(self):
        with self.assertRaisesRegex(ValueError, 'M1_DRAW_RECORDS_REQUIRED'):
            E.m1_completeness([outcome()], ['[4, 6)'], cohort_sha256=E.ITEM47_COHORT_SHA256, n_boot=10)
        with self.assertRaisesRegex(ValueError, 'M1_DRAW_COUNT_MISMATCH'):
            E.m1_completeness([outcome()], ['[4, 6)'], cohort_sha256=E.ITEM47_COHORT_SHA256,
                draw_records={'a': [dict(snr_bin='[4, 6)', status='DRAWN')]}, n_boot=10)


if __name__ == '__main__':
    unittest.main()
