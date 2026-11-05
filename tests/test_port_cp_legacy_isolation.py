"""Real-cascade policy isolation and hold regressions for PORT-CP-01."""
import contextlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
import aldebaran as ald


class LegacyIsolation(unittest.TestCase):
    def run_fixture(self, method, hold):
        t = np.linspace(1000., 1027., 19440)
        rng = np.random.default_rng(42)
        flux = 1. + rng.normal(0., .0001, t.size)
        tau = (t - 1001. + 1.5) % 3. - 1.5
        flux[np.abs(tau) < .04] -= .001
        centroid = rng.normal(0., .001, t.size)
        arch = dict(n_archival_sectors=0, archival_sectors_str='', refined_p=np.nan,
                    refined_t0=np.nan, archival_snr=np.nan, multi_sde=8.5,
                    est_multi_sde_proxy=8.5, n_trials=1, snr_crit=2.4,
                    ephemeris_source='DISCOVERY_SINGLE_SECTOR', ephemeris_status='FALLBACK_DISCOVERY',
                    ephemeris_fallback_reason='NO_PRIOR_SECTORS')
        bls = SimpleNamespace(power=lambda *a, **k: SimpleNamespace(
            period=np.full(9000, 3.), power=np.full(9000, 100.),
            duration=np.full(9000, .08), transit_time=np.full(9000, 1001.),
            depth=np.full(9000, .001)))
        tpf = (True, 'REVIEW_TPF_EDGE_OR_MARGINAL: fixture' if hold == 'TPF' else 'OK',
               np.ones((5, 5)), np.ones((5, 5), dtype=bool))
        cp_arch = (True, 'REVIEW_ARCHIVAL_NONDETECTION: fixture' if hold == 'ARCHIVAL' else 'No prior sectors', arch)
        legacy_arch = (True, 'No prior sectors', arch)
        patches = dict(AUTO_PURGE_CACHE=False, OUTPUT_DIR=None, ARCHIVAL_DIAGNOSTICS_FILE=None,
            get_known_transits_for_tic=lambda _: [], safe_download=lambda *a, **k: object(),
            extract_and_stitch_arrays=lambda *a, **k: (t, flux, np.full(t.size, .0001), centroid, centroid, .95),
            clip_upper_outliers=lambda t, f, e: (t, f, e),
            piecewise_savgol_detrend=lambda t, f, **k: np.ones_like(f),
            compute_whitened_kovacs_sde=lambda *a, **k: (np.full(9000, 8.5), None),
            get_top_distinct_bls_peaks=lambda *a, **k: [0],
            get_stellar_and_gaia_params=lambda _: dict(has_params=True, r_star=1., e_r_star=.05,
                m_star=1., e_m_star=.05, teff=5800., ruwe=1.05),
            BoxLeastSquares=lambda *a, **k: bls)
        with tempfile.TemporaryDirectory(prefix='port-cp-hold-') as td, contextlib.ExitStack() as stack:
            patches.update(OUTPUT_DIR=td, ARCHIVAL_DIAGNOSTICS_FILE=td + '/archival.jsonl')
            for name, value in patches.items(): stack.enter_context(mock.patch.object(ald, name, value))
            stack.enter_context(mock.patch.object(ald.lk, 'search_lightcurve', return_value=[object()]))
            secondary = (False, 0., 0., .5, 3.)
            stack.enter_context(mock.patch.object(ald, 'scan_full_orbit_secondary_eclipse', return_value=secondary))
            cp_sec = stack.enter_context(mock.patch.object(ald, '_scan_full_orbit_secondary_eclipse_cp', return_value=secondary))
            # A patched legacy helper may itself return a REVIEW_-prefixed
            # message: the old cascade must still not index v2-only evidence.
            stack.enter_context(mock.patch.object(ald, 'verify_and_filter_tpf', return_value=tpf))
            cp_tpf = stack.enter_context(mock.patch.object(ald, '_verify_and_filter_tpf_cp', return_value=tpf))
            stack.enter_context(mock.patch.object(ald, 'verify_multi_epoch',
                return_value=cp_arch if hold == 'ARCHIVAL' else legacy_arch))
            cp_archive = stack.enter_context(mock.patch.object(ald, '_verify_multi_epoch_cp', return_value=cp_arch))
            oe = ald.OddEvenV2Result('ASSESSED', 'fixture', 0., 1000., 1000., 10., 10., {})
            stack.enter_context(mock.patch.object(ald, 'compute_odd_even_v2', return_value=oe))
            if method == 'legacy':
                result = ald.analyze_single_target('12345678', 106)
            else:
                result = ald.analyze_single_target('12345678', 106, odd_even_method='v2')
            if method == 'legacy':
                cp_sec.assert_not_called()
                cp_tpf.assert_not_called()
                cp_archive.assert_not_called()
            else:
                cp_sec.assert_called_once()
                cp_tpf.assert_called_once()
                cp_archive.assert_called_once()
            return result

    def test_legacy_tpf_hold_cannot_crash(self):
        result = self.run_fixture('legacy', 'TPF')
        self.assertEqual(result[1], 'CANDIDATE', result[2])

    def test_legacy_archival_hold_cannot_crash(self):
        result = self.run_fixture('legacy', 'ARCHIVAL')
        self.assertEqual(result[1], 'CANDIDATE', result[2])

    def test_explicit_v2_holds_remain_review(self):
        for hold in ('TPF', 'ARCHIVAL'):
            with self.subTest(hold=hold):
                result = self.run_fixture('v2', hold)
                self.assertEqual(result[1], 'REVIEW_VETTING_HOLD', result[2])
                self.assertIsNone(result[3])


if __name__ == '__main__':
    unittest.main()

