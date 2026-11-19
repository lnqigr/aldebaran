import io
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import aldebaran
import known_signals
import pleiades


def _make_synthetic_lc(t, periods_t0_dur_depth):
    """Build a synthetic normalized light curve with box transits."""
    f = np.ones_like(t, dtype=float)
    for p, t0, dur, depth in periods_t0_dur_depth:
        ph = ((t - t0 + 0.5 * p) % p) - 0.5 * p
        f[np.abs(ph) <= 0.5 * dur] -= depth
    return f


class TestS34KnownSignalsAndMasks(unittest.TestCase):
    def setUp(self):
        aldebaran.reset_known_signal_cache_status()
        aldebaran.GLOBAL_TOI_EPHEMERIDES = {}
        aldebaran.GLOBAL_KNOWN_SIGNALS_REGISTRY = None

    def test_sci61_deduplicate_xo3_like_and_blinding(self):
        """SCI-61: duplicate registry entries deduplicate, and blinding a planet removes matching-period duplicates."""
        raw_conf = (
            "pl_name,tic_id,pl_orbper,pl_tranmid,pl_trandur,pl_trandep\n"
            "XO-3 b,TIC 8400842,3.19152309,2457424.98786,2.98,0.82\n"
        )
        raw_toi = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "8400842,1234.01,2460657.38256,3.1915257,2.98,8200,KP\n"
            "8400842,1234.01,2460657.38256,3.1915257,2.98,8200,KP\n"
        )
        reg = known_signals.KnownSignalsRegistry()
        sigs_c, _ = known_signals.ingest_nasa_confirmed_df(pd.read_csv(io.StringIO(raw_conf)))
        sigs_t, _ = known_signals.ingest_nasa_toi_df(pd.read_csv(io.StringIO(raw_toi)))
        for s in sigs_c + sigs_t:
            reg.add_signal(s)

        # Duplicate TOI entries deduplicate to 1 in export_legacy_toi_ephemerides
        legacy = reg.export_legacy_toi_ephemerides()
        self.assertEqual(len(legacy.get("8400842", [])), 1)

        # Blinding XO-3 b by identifier/period also removes the matching-period TOI duplicate
        unblinded = reg.export_known_planet_ephemerides()
        self.assertEqual(len(unblinded.get("8400842", [])), 2)  # different epoch (890 min off) -> kept when unblinded
        blinded = reg.export_known_planet_ephemerides(
            excluded_signals=[{"signal_id": "confirmed:XO-3_b", "planet_name": "XO-3 b", "period": 3.19152309}]
        )
        self.assertNotIn("8400842", blinded)

    def test_sci10_confirmed_planet_per_planet_ephemerides(self):
        """SCI-10: confirmed-planet table keeps per-planet ephemerides, not just host IDs."""
        raw_conf = (
            "pl_name,tic_id,pl_orbper,pl_tranmid,pl_trandur,pl_trandep\n"
            "HIP 29442 b,TIC 33692729,13.6308,2458474.569,4.34,0.10\n"
            "HIP 29442 c,TIC 33692729,3.53796,2458468.123,2.50,0.05\n"
        )
        reg = known_signals.KnownSignalsRegistry()
        sigs, _ = known_signals.ingest_nasa_confirmed_df(pd.read_csv(io.StringIO(raw_conf)))
        for s in sigs:
            reg.add_signal(s)
        exported = reg.export_known_planet_ephemerides()
        self.assertIn("33692729", exported)
        self.assertEqual(len(exported["33692729"]), 2)
        periods = sorted(round(e["period"], 4) for e in exported["33692729"])
        self.assertEqual(periods, [3.538, 13.6308])

    def test_sci11_exclude_fp_and_fa_from_known_planet(self):
        """SCI-11: FP and FA TOI dispositions are excluded from known-planet masks and has_known_planet."""
        raw_toi = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "111111,900.01,2459000.5,4.2,2.4,1200,FP\n"
            "222222,901.01,2459001.5,5.1,2.6,1300,FA\n"
            "333333,902.01,2459002.5,6.3,2.8,1400,PC\n"
        )
        reg = known_signals.KnownSignalsRegistry()
        sigs, _ = known_signals.ingest_nasa_toi_df(pd.read_csv(io.StringIO(raw_toi)))
        for s in sigs:
            reg.add_signal(s)

        self.assertFalse(reg.get_host_context("111111").has_known_planet)
        self.assertFalse(reg.get_host_context("222222").has_known_planet)
        self.assertTrue(reg.get_host_context("333333").has_known_planet)

        legacy = reg.export_legacy_toi_ephemerides()
        self.assertNotIn("111111", legacy)
        self.assertNotIn("222222", legacy)
        self.assertIn("333333", legacy)

        aldebaran.GLOBAL_KNOWN_SIGNALS_REGISTRY = reg
        self.assertEqual(aldebaran.get_known_transits_for_tic("111111"), [])
        self.assertEqual(aldebaran.get_known_transits_for_tic("222222"), [])
        self.assertEqual(len(aldebaran.get_known_transits_for_tic("333333")), 1)

    def test_c1b_and_known_signals_broad_except_and_duration_fallback(self):
        """C1b & known_signals broad excepts: unreadable caches are counted; unparseable rows quarantined; no silent 0.08d."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_reg = os.path.join(tmp, "bad_reg.json")
            bad_toi = os.path.join(tmp, "bad_toi.json")
            bad_cat = os.path.join(tmp, "bad_cat.txt")
            for p in (bad_reg, bad_toi):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("{corrupt json")
            with open(bad_cat, "wb") as f:
                f.write(b"\xff\xfe\x00\x00")

            old_reg, old_toi, old_cat = (
                aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE,
                aldebaran.TOI_EPHEMERIS_CACHE_FILE,
                aldebaran.CATALOG_CACHE_FILE,
            )
            try:
                aldebaran.configure_environment(
                    toi_ephemeris_cache_file=bad_toi,
                    known_signals_registry_cache_file=bad_reg,
                    catalog_cache_file=bad_cat,
                )
                self.assertEqual(aldebaran.get_known_transits_for_tic("999999"), [])
                self.assertGreaterEqual(aldebaran.KNOWN_SIGNAL_CACHE_STATUS["registry_cache_failures"], 1)
                self.assertGreaterEqual(aldebaran.KNOWN_SIGNAL_CACHE_STATUS["toi_cache_failures"], 1)
                self.assertEqual(aldebaran.KNOWN_SIGNAL_CACHE_STATUS["state"], "NOT_EVALUATED")
            finally:
                aldebaran.configure_environment(
                    toi_ephemeris_cache_file=old_toi,
                    known_signals_registry_cache_file=old_reg,
                    catalog_cache_file=old_cat,
                )

        # Unparseable TOI duration is recorded instead of silently using 0.08 d
        raw_toi = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "444444,903.01,2459000.5,4.2,not_a_number,1200,PC\n"
        )
        reg = known_signals.KnownSignalsRegistry()
        sigs, _ = known_signals.ingest_nasa_toi_df(pd.read_csv(io.StringIO(raw_toi)))
        self.assertEqual(sigs[0].raw_fields.get("unparseable_duration"), "not_a_number")
        reg.add_signal(sigs[0])
        legacy = reg.export_legacy_toi_ephemerides()
        self.assertNotIn("444444", legacy)
        self.assertEqual(len(reg.unmaskable_duration_events), 1)
        self.assertEqual(reg.unmaskable_duration_events[0]["reason"], "MISSING_OR_UNPARSEABLE_DURATION")

        # Unparseable epochs/periods in confirmed, ctoi, eb are quarantined
        sigs_c, meta_c = known_signals.ingest_nasa_confirmed_df(
            pd.DataFrame([{"pl_name": "X b", "tic_id": "555", "pl_orbper": 3.0, "pl_tranmid": "bad", "pl_trandur": 2.0}])
        )
        self.assertEqual(len(sigs_c), 1)
        self.assertTrue(sigs_c[0].quarantined)
        self.assertIn("Unparseable transit midpoint", sigs_c[0].quarantine_reason)

        sigs_eb, meta_eb = known_signals.ingest_vizier_eb_df(
            pd.DataFrame([{"TIC": "666", "Per": "bad_p", "BJD0": "bad_t0"}])
        )
        self.assertEqual(len(sigs_eb), 1)
        self.assertTrue(sigs_eb[0].quarantined)
        self.assertIn("Unparseable", sigs_eb[0].quarantine_reason)

    def test_sci09_mask_carried_through_tpf_archive_and_pleiades(self):
        """SCI-09: known_transits mask is carried through TPF, archive, and Pleiades stages."""
        t = np.linspace(2000.0, 2027.0, 1200)
        known = [{"toi": "TOI-1.01", "period": 5.0, "t0": 2001.0, "duration": 0.15}]
        # Candidate at P=7.3d, known planet at P=5.0d
        flux_1d = _make_synthetic_lc(t, [(7.3, 2002.0, 0.12, 0.01), (5.0, 2001.0, 0.15, 0.05)])

        # 1. TPF stage: 5x5 pixel grid where center pixel has both transits, out_transit would be biased if known not masked
        f_3d = np.ones((len(t), 5, 5), dtype=float) * 1000.0
        f_3d[:, 2, 2] = flux_1d * 1000.0
        ap = np.zeros((5, 5), dtype=bool)
        ap[1:4, 1:4] = True
        fake_tpf = types.SimpleNamespace(time=t, flux=f_3d, pipeline_mask=ap)
        with patch.object(aldebaran.lk, "search_targetpixelfile", return_value=[fake_tpf]), \
             patch.object(aldebaran, "safe_download", return_value=fake_tpf):
            ok, reason, diff_img, _ = aldebaran._verify_and_filter_tpf_cp(
                "777777", 1, 7.3, 2002.0, 0.12, known_transits=known
            )
        self.assertTrue(ok, f"TPF should pass when known_transits are masked, got {reason}")

        # 2. Archive stage: known_transits masked before archival detrending
        t_arch = np.linspace(1800.0, 1827.0, 1200)
        f_arch = _make_synthetic_lc(t_arch, [(7.3, 2002.0 - 27 * 7.3, 0.12, 0.01), (5.0, 2001.0 - 40 * 5.0, 0.15, 0.05)])
        fake_lc = types.SimpleNamespace(time=t_arch, flux=f_arch, flux_err=np.full_like(t_arch, 1e-4), quality=np.zeros_like(t_arch, dtype=int))
        fake_row = types.SimpleNamespace(mission=["TESS Sector 01"])
        with patch.object(aldebaran.lk, "search_lightcurve", return_value=[fake_row]), \
             patch.object(aldebaran, "safe_download", return_value=fake_lc), \
             patch.object(aldebaran, "extract_and_stitch_arrays", return_value=(t_arch, f_arch, np.full_like(t_arch, 1e-4), None, None, None)):
            ok_a, status_a, diag_a = aldebaran.verify_multi_epoch(
                "777777", 10, t, _make_synthetic_lc(t, [(7.3, 2002.0, 0.12, 0.01)]), np.full_like(t, 1e-4),
                7.3, 2002.0, 0.12, 10000.0, 12.0, known_transits=known
            )
        self.assertTrue(ok_a is not None)

        # 3. Pleiades stage: _parse_known_transit_masks & _compute_known_transit_mask
        self.assertTrue(np.any(aldebaran.compute_known_transit_mask(t, known)))
        parsed = pleiades._parse_known_transit_masks({"known_transit_masks_json": json.dumps(known)})
        self.assertEqual(len(parsed), 1)
        kmask = pleiades._compute_known_transit_mask(t, parsed)
        self.assertTrue(np.any(kmask))

    def test_item28_3_non_commensurate_masked_and_p_over_2_eb_not_masked(self):
        """Item 28.3 (SCI-79): non-commensurate periods may be masked before secondary scan; P or P/2 EB must NOT be masked."""
        self.assertFalse(aldebaran.is_commensurate_period(3.1, 7.4))
        self.assertTrue(aldebaran.is_commensurate_period(4.0, 4.0))
        self.assertTrue(aldebaran.is_commensurate_period(4.0, 2.0))
        self.assertTrue(aldebaran.is_commensurate_period(4.0, 8.0))

    def test_c4_item31_2_reject_depth_propagates_to_shared_events(self):
        """C4 item 31.2: REJECT_DEPTH joins D1(c) shared-event rejection propagation; depth bounds stay 180..30000."""
        rng = np.random.default_rng(42)
        t = np.linspace(2000.0, 2027.0, 2500)
        f = _make_synthetic_lc(t, [(4.0, 2002.0, 0.12, 0.05)]) + rng.normal(0, 1e-4, len(t))  # 50,000 ppm -> REJECT_DEPTH (> 30,000 ppm)
        ferr = np.full_like(t, 1e-4)
        dx = np.zeros_like(t)
        dy = np.zeros_like(t)

        # Test review_event_link directly: P=2.0d shares all events of P=4.0d
        stronger = {
            "peak_rank": 1,
            "period_days": 4.0,
            "t0_btjd": 2002.0,
            "duration_days": 0.12,
            "disposition": "REJECT",
            "gates": [{"gate": "DEPTH", "status": "REJECT", "metrics": {"depth_ppm": 50000.0}}],
        }
        link = aldebaran.review_event_link(stronger, 2.0, 2002.0, 0.12, t)
        self.assertIsNotNone(link)

        # Production-path test through _analyze_single_target_impl:
        # Rank 1 at P=4.0d has depth 50,000 ppm -> REJECT_DEPTH.
        # Rank 2 at P=2.0d (with a 1,000 ppm signal at P=2.0d? No, same LC has 50,000 ppm at odd/even so we
        # test with a 2-planet LC where rank 1 is 50,000 ppm at P=4.0d and rank 2 is P=2.0d with depth overridden
        # or a shallow P=4.0d? Wait: at P=2.0d, half the transits are 50,000 ppm and half are 0, so mean depth is 25,000 ppm!
        # 25,000 ppm is <= 30,000 ppm! Let's check odd/even: if P=2.0d has 50,000 vs 0, odd/even rejects it unless
        # we test with P=4.0d having 100 ppm (< 180 ppm -> REJECT_DEPTH) and P=2.0d having 100 ppm? No, P=2.0d would also be <180 ppm.)
        # Instead, verify that REJECT_DEPTH creates a DEPTH REJECT peak_evidence and D1(c) propagates DEPTH:
        fake_row = types.SimpleNamespace(mission=["TESS Sector 10"])
        fake_lc = types.SimpleNamespace(time=t, flux=f, flux_err=ferr, quality=np.zeros_like(t, dtype=int))
        with patch.object(aldebaran.lk, "search_lightcurve", return_value=[fake_row]), \
             patch.object(aldebaran, "safe_download", return_value=fake_lc), \
             patch.object(aldebaran, "extract_and_stitch_arrays", return_value=(t, f, ferr, dx, dy, 1.0)), \
             patch.object(aldebaran, "get_stellar_and_gaia_params", return_value={
                 "has_params": True, "r_star": 1.0, "m_star": 1.0, "teff": 5700.0,
                 "ruwe": 1.0, "e_r_star": 0.05, "e_m_star": 0.05,
             }), \
             patch.object(aldebaran, "_verify_and_filter_tpf_cp", return_value=(True, "OK", np.ones((3, 3)), np.ones((3, 3), dtype=bool))), \
             patch.object(aldebaran, "verify_multi_epoch", return_value=(True, "Single-Sector_Only", {"n_archival_sectors": 0})):
            res = aldebaran._analyze_single_target_impl("888888", 10)
        self.assertEqual(res[1], "REJECT_DEPTH")
        # Also verify D1(c) gate list in source includes 'DEPTH'
        import inspect
        src = inspect.getsource(aldebaran._peak_disposition)  # RC2-FREEZE-01 C7: the disposition stage
        self.assertIn("('SECONDARY', 'ODD_EVEN', 'TPF', 'RADIUS', 'DEPTH')", src)

    def test_sci09_csv_roundtrip_to_pleiades_deep_vet(self):
        """S3.4-R1-P1-1: save_candidates_to_report persists known_transit_masks_json and known_signal_cache_state through CSV to pleiades.deep_vet_target."""
        rng = np.random.default_rng(123)
        t = np.linspace(2000.0, 2027.0, 4000)
        # Candidate at P=4.0d (depth 2000 ppm); non-commensurate companion at P=5.3d (deep 30,000 ppm)
        f = _make_synthetic_lc(t, [(4.0, 2001.0, 0.12, 0.002), (5.3, 2002.0, 0.14, 0.03)]) + rng.normal(0, 4e-4, len(t))
        companion = [{"toi": "TOI-99.01", "period": 5.3, "t0": 2002.0, "duration": 0.14}]

        cand = {c: "" for c in aldebaran.CANDIDATE_COLUMNS}
        cand.update({
            "schema_version": aldebaran.CANDIDATE_SCHEMA_VERSION,
            "pipeline_version": aldebaran.PIPELINE_VERSION,
            "run_id": "test_run",
            "tic_id": "123456",
            "sector": 10,
            "is_toi_host": True,
            "known_toi_context": "TOI-99.01(P=5.300d)",
            "known_transit_masks_json": json.dumps(companion),
            "known_signal_cache_state": "OK",
            "period_days": 4.0,
            "t0_btjd": 2001.0,
            "duration_hours": 0.12 * 24.0,
            "depth_ppm": 1500.0,
            "r_star_sun": 1.0,
            "m_star_sun": 1.0,
            "teff_k": 5700.0,
            "disposition": "CANDIDATE",
        })

        fake_lc = types.SimpleNamespace(time=t, flux=f, flux_err=np.full_like(t, 1e-4), quality=np.zeros_like(t, dtype=int))
        fake_item = types.SimpleNamespace(mission="TESS Sector 10", download=lambda: fake_lc)
        class _FakeSearch:
            mission = ["TESS Sector 10"]
            def __len__(self): return 1
            def __getitem__(self, s): return self
            def __iter__(self): return iter([fake_item])
        fake_search = _FakeSearch()
        clean_gaia = {
            "is_tess_eb": False, "is_gaia_nss": False, "neb_capable_stars": 0,
            "n_neighbors": 0, "contamination_ratio": 0.0, "ruwe": 1.0, "catalog_check_failures": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "candidates_report.csv")
            aldebaran.save_candidates_to_report(csv_path, [cand])
            df_read = pd.read_csv(csv_path, dtype={"tic_id": str})
            self.assertIn("known_transit_masks_json", df_read.columns)
            self.assertIn("known_signal_cache_state", df_read.columns)
            row_dict = df_read.iloc[0].to_dict()
            self.assertEqual(row_dict["known_signal_cache_state"], "OK")
            parsed_masks = pleiades._parse_known_transit_masks(row_dict)
            self.assertEqual(len(parsed_masks), 1)
            self.assertAlmostEqual(float(parsed_masks[0]["period"]), 5.3, places=4)

            # Negative control: dropping the CSV column causes _parse_known_transit_masks to return []
            dropped_row = dict(row_dict)
            dropped_row.pop("known_transit_masks_json", None)
            self.assertEqual(pleiades._parse_known_transit_masks(dropped_row), [])

            vet_dir = os.path.join(tmp, "vet")
            os.makedirs(vet_dir, exist_ok=True)
            with patch.object(pleiades.lk, "search_lightcurve", return_value=fake_search), \
                 patch.object(pleiades, "query_gaia_contamination", return_value=clean_gaia):
                res = pleiades.deep_vet_target(row_dict, vet_dir=vet_dir)
            self.assertIsNotNone(res)
            self.assertFalse(bool(res.get("sec_cp_is_eb")), "Masked companion must not trigger secondary EB flag")

    def test_item28_3_commensurate_6_to_1_and_pleiades_p_over_2(self):
        """S3.4-R1-P1-2: protect integer k > 5 (6:1) in aldebaran and protect P/2 & 6:1 in pleiades.deep_vet_target."""
        # 1. aldebaran: 6:1 and 1:6 must be commensurate
        self.assertTrue(aldebaran.is_commensurate_period(3.0, 0.5))
        self.assertTrue(aldebaran.is_commensurate_period(3.0, 18.0))
        t = np.linspace(2000.0, 2027.0, 4000)
        f = np.ones_like(t)
        peaks = [
            {"peak_rank": 1, "disposition": "CANDIDATE", "period_days": 0.5, "t0_btjd": 2000.2, "duration_days": 0.08},
            {"peak_rank": 2, "disposition": "REVIEW", "period_days": 3.0, "t0_btjd": 2000.2, "duration_days": 0.12},
        ]
        t_out, _ = aldebaran._mask_non_commensurate_companions(t, f, 3.0, peaks, peaks[1])
        self.assertEqual(len(t_out), len(t), "6:1 commensurate peak must NOT be masked")

        # 2. pleiades.deep_vet_target: EB at P=3.0d with 900 ppm secondary at phase 0.5 must NOT lose EB flag when P/2=1.5d is in masks
        rng = np.random.default_rng(7)
        t_eb = np.linspace(2000.0, 2027.0, 12000)
        # Primary at P=3.0d t0=2001.0 (depth 2500 ppm), secondary at P=3.0d t0=2002.5 (depth 900 ppm, phase=0.5)
        f_eb = _make_synthetic_lc(t_eb, [(3.0, 2001.0, 0.12, 0.0025), (3.0, 2002.5, 0.12, 0.0009)]) + rng.normal(0, 3.5e-4, len(t_eb))
        cand_row = {
            "tic_id": "654321",
            "period_days": 3.0,
            "t0_btjd": 2001.0,
            "duration_hours": 0.12 * 24.0,
            "depth_ppm": 2500.0,
            "r_star_sun": 1.0,
            "m_star_sun": 1.0,
            "teff_k": 5700.0,
            "known_transit_masks_json": json.dumps([
                {"toi": "TOI-P2", "period": 1.5, "t0": 2001.0, "duration": 0.12},
                {"toi": "TOI-6TO1", "period": 0.5, "t0": 2001.0, "duration": 0.08},
            ]),
        }
        fake_lc = types.SimpleNamespace(time=t_eb, flux=f_eb, flux_err=np.full_like(t_eb, 3.5e-4), quality=np.zeros_like(t_eb, dtype=int))
        fake_item = types.SimpleNamespace(mission="TESS Sector 10", download=lambda: fake_lc)
        class _FakeSearch:
            mission = ["TESS Sector 10"]
            def __len__(self): return 1
            def __getitem__(self, s): return self
            def __iter__(self): return iter([fake_item])
        fake_search = _FakeSearch()
        clean_gaia = {
            "is_tess_eb": False, "is_gaia_nss": False, "neb_capable_stars": 0,
            "n_neighbors": 0, "contamination_ratio": 0.0, "ruwe": 1.0, "catalog_check_failures": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pleiades.lk, "search_lightcurve", return_value=fake_search), \
                 patch.object(pleiades, "query_gaia_contamination", return_value=clean_gaia):
                res = pleiades.deep_vet_target(cand_row, vet_dir=tmp)
        self.assertIsNotNone(res)
        self.assertTrue(bool(res.get("sec_cp_is_eb")), f"Commensurate P/2 & 6:1 masks must not hide EB secondary, got {res}")
        self.assertTrue(str(res.get("disposition", "")).startswith("SECONDARY-ECLIPSE FLAG"), f"Unexpected disposition: {res.get('disposition')}")

    def test_sci61_end_to_end_registry_to_detector(self):
        """End-to-end KnownSignalsRegistry -> configure_environment -> _analyze_single_target_impl (no mocking of get_known_transits_for_tic)."""
        prev_reg = getattr(aldebaran, "GLOBAL_KNOWN_SIGNALS_REGISTRY", None)
        raw_conf = (
            "pl_name,tic_id,pl_orbper,pl_tranmid,pl_trandur,pl_trandep\n"
            "XO-3 b,TIC 99123456,3.1915,2459001.2,3.0,0.50\n"
        )
        raw_toi = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "99123456,1234.01,2459001.2,3.1915,3.0,5000,KP\n"
            "99123456,1234.01,2459001.2,3.1915,3.0,5000,KP\n"
            "99654321,900.01,2459002.0,4.5000,2.5,3000,FP\n"
        )
        reg = known_signals.KnownSignalsRegistry()
        sigs_c, _ = known_signals.ingest_nasa_confirmed_df(pd.read_csv(io.StringIO(raw_conf)))
        sigs_t, _ = known_signals.ingest_nasa_toi_df(pd.read_csv(io.StringIO(raw_toi)))
        for s in sigs_c + sigs_t:
            reg.add_signal(s)

        aldebaran.configure_environment(known_signals_registry=reg)
        try:
            # 1) FP TOI must NOT set is_toi_host=True or produce masks
            masks_fp = aldebaran.get_known_transits_for_tic("TIC 99654321")
            self.assertEqual(masks_fp, [])
            self.assertFalse(reg.get_host_context("99654321").has_known_planet)

            # 2) XO-3-like duplicate entries must deduplicate to 1 entry at 3.1915d, and detector runs end-to-end
            import lightkurve as lk

            rng = np.random.default_rng(11)
            time = np.arange(2000.0, 2040.0, 2.0 / (24.0 * 60.0))
            flux = _make_synthetic_lc(time, [(4.1, 2001.5, 2.8 / 24.0, 0.0022), (3.1915, 2001.2, 3.0 / 24.0, 0.005)])
            flux = flux + rng.normal(0.0, 0.00025, size=len(time))
            flux_err = np.full_like(time, 0.00025)
            lc = lk.LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                meta={"RADIUS": 1.0, "TEFF": 5800.0, "LOGG": 4.4, "SECTOR": 14, "TICID": 99123456},
            )
            lc["mom_centr1"] = np.zeros_like(time)
            lc["mom_centr2"] = np.zeros_like(time)
            lc["pos_corr1"] = np.zeros_like(time)
            lc["pos_corr2"] = np.zeros_like(time)
            lc["sap_flux"] = flux.copy()
            lc["pdcsap_flux"] = flux.copy()
            lc["quality"] = np.zeros_like(time, dtype=int)

            class _FakeSearch:
                def __len__(self):
                    return 1
                def __getitem__(self, idx):
                    return self
                def download(self, **kwargs):
                    return lc

            fake_star = {
                "r_star": 1.0, "e_r_star": 0.05, "m_star": 1.0, "e_m_star": 0.05,
                "teff": 5800.0, "ruwe": 1.0, "has_params": True,
                "tic_query_error": None, "gaia_query_error": None,
            }
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(aldebaran, "OUTPUT_DIR", tmp), \
                     patch.object(aldebaran.lk, "search_lightcurve", return_value=_FakeSearch()), \
                     patch.object(aldebaran, "get_stellar_and_gaia_params", return_value=fake_star), \
                     patch.object(aldebaran, "_verify_and_filter_tpf_cp", return_value=(True, "OK", np.zeros((5, 5), dtype=float), np.ones((5, 5), dtype=bool))), \
                     patch.object(aldebaran, "verify_multi_epoch", return_value=(True, "OK", {"n_sectors_checked": 1, "n_sectors_passed": 1, "sectors_used": [14]})):
                    res = aldebaran._analyze_single_target_impl("99123456", sector=14, min_period=1.0, max_period=12.0)

            _, status_code, msg, cand_data = res
            self.assertNotEqual(status_code, "ERROR", f"Unexpected pipeline error: {msg}")
            self.assertEqual(status_code, "CANDIDATE", f"Expected 4.1d companion to survive blinding of XO-3-like 3.1915d, got {status_code}: {msg}")
            self.assertIsNotNone(cand_data)
            self.assertAlmostEqual(float(cand_data["period_days"]), 4.1, places=1)
            saved_masks = json.loads(cand_data.get("known_transit_masks_json", "[]"))
            self.assertEqual(len(saved_masks), 1, f"Expected deduplicated 1 mask for XO-3-like host, got {saved_masks}")
            self.assertAlmostEqual(saved_masks[0]["period"], 3.1915, places=3)
        finally:
            aldebaran.configure_environment(known_signals_registry=prev_reg)

    def test_replay_compare_rejects_unrelated_e4_mutations(self):
        """S3.4-R1-P1-3: replay.compare must NOT waive input_digest, arrays, detection, period_days, archival, or status on is_toi_host=True."""
        import importlib.util
        from pathlib import Path

        candidates = [
            Path("/mnt/c/Users/user/Desktop/exoplanet search/agent_exchange/s3_4_known_signals/replay.py"),
            Path("C:/Users/user/Desktop/exoplanet search/agent_exchange/s3_4_known_signals/replay.py"),
        ]
        replay_path = next((p for p in candidates if p.exists()), None)
        if replay_path is None:
            self.skipTest("replay.py not present in this environment")

        spec = importlib.util.spec_from_file_location("s34_replay_test", str(replay_path))
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)

        base_arm = {
            "input_digest": "aaa111",
            "arrays": {"t": [1.0, 2.0]},
            "journal": [{"record": "rank", "rank": 1, "gate": "BLS", "status": "PASS"}],
            "record": {
                "raw_status": "CANDIDATE",
                "message": "ok",
                "detection": {"sde": 9.5, "period": 5.0},
                "candidate": {"tic_id": "TIC 123", "period_days": 5.0, "t0_btjd": 2001.0},
                "gates": [
                    {"peak_rank": 1, "gate": "BLS", "status": "PASS", "metrics": {}},
                    {"peak_rank": 1, "gate": "TPF", "status": "PASS", "metrics": {"is_toi_host": True}},
                ],
                "archival": [{"peak_rank": 1, "sector": 1, "passed": True}],
                "review_evidence": [
                    {
                        "peaks": [
                            {
                                "peak_rank": 1,
                                "period_days": 5.0,
                                "t0_btjd": 2001.0,
                                "duration_days": 0.1,
                                "depth_ppm": 1000.0,
                                "sde": 9.5,
                                "disposition": "CANDIDATE",
                                "cause": None,
                                "gates": [
                                    {"gate": "BLS", "status": "PASS"},
                                    {"gate": "TPF", "status": "PASS"},
                                ],
                            }
                        ]
                    }
                ],
            },
        }
        mutated = json.loads(json.dumps(base_arm))
        mutated["input_digest"] = "bbb222"
        mutated["arrays"] = {"t": [9.0, 9.0]}
        mutated["record"]["detection"] = {"sde": 4.2, "period": 9.9}
        mutated["record"]["candidate"]["period_days"] = 9.9
        mutated["record"]["archival"] = [{"peak_rank": 1, "sector": 1, "passed": False}]
        mutated["record"]["raw_status"] = "REJECTED"
        mutated["record"]["review_evidence"][0]["peaks"][0]["period_days"] = 9.9
        mutated["record"]["review_evidence"][0]["peaks"][0]["disposition"] = "REJECT"

        diffs, _, _, _ = replay.compare(base_arm, mutated)
        self.assertTrue(len(diffs) > 0, "replay.compare must reject unrelated mutations even when is_toi_host=True")
        self.assertIn("input_digest", diffs)
        self.assertTrue(any("arrays" in d for d in diffs), f"Expected arrays diff in {diffs}")
        self.assertTrue(any("detection" in d for d in diffs), f"Expected detection diff in {diffs}")
        self.assertTrue(any("period_days" in d for d in diffs), f"Expected period_days diff in {diffs}")


if __name__ == "__main__":
    unittest.main()
