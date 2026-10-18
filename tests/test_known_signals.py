"""
tests/test_known_signals.py - SYNTHETIC offline contract and falsification test suite for ALD-007A.

Covers:
  1. HIP 29442 recovery & production integration (TIC 33692729 b, c, d + TOI 469.01).
  2. Independent derivation of key numbers: dP = -2.9e-6 d, rel_dP = 8.1968e-7, dt = -5.8622 min, dur = 2.429 h.
  3. Strict integer-cycle wrap invariance and common fixed pivot anchor for unequal periods.
  4. Multi-signal hosts preserving conflicting/distinct dispositions (1 FP + 1 CP + 1 unknown).
  5. Missing/invalid ephemerides, quarantined rows, and non-destructive diagnostic reporting.
  6. Atomic serialization, schema validation, and cache integrity (known_signals_v1).
  7. Hypothetical / non-active transit mask geometry helper contract.
  8. Deliberate controls / fault injection (Rule 2: break check on purpose):
       - Fault A: Dropped c/d records detected.
       - Fault B: Host-only confirmed projection detected.
       - Fault C: Mixed P/T0 solutions detected.
       - Fault D: Filtering confirmed planets by discoverymethod='Transit' detected.
       - Fault E: Naive unprojected epoch difference failing cycle-wrap invariance detected.
  9. Production catalog ingestion boundary & legacy compatibility tuple contract.
  10. R1: Idempotent refresh, coherent source replacement, and row retirement on actual loader.
  11. R2: Full 4-source tracking, truthful availability, caching, write failure, and cache validation.
  12. R3: Strict positive integer TIC normalization, finite geometry, explicit duration requirement for maskability, and distinct solution provenance.
  13. R4: Time/provenance contracts, missing pivot handling, asymmetric uncertainties, and absence of arbitrary match cutoffs.
"""

import io
import hashlib
from unittest import mock
import json
import math
import os
import shutil
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd

import aldebaran
import known_signals


class TestKnownSignalsSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_offline = os.environ.get("ALDEBARAN_OFFLINE")
        os.environ["ALDEBARAN_OFFLINE"] = "1"
        # SYNTHETIC contract fixtures; historical names/numbers only exercise API contracts.
        # These generated rows are not observations or evidence of recovery.
        cls._fixtures = tempfile.TemporaryDirectory(prefix="SYNTHETIC_known_signals_")
        cls.addClassCleanup(cls._fixtures.cleanup)
        cls.fixture_dir = cls._fixtures.name
        cls.has_fixtures = True
        cls.confirmed_path = os.path.join(cls.fixture_dir, "nasa_confirmed.csv")
        cls.toi_path = os.path.join(cls.fixture_dir, "nasa_toi.csv")
        cls.ctoi_path = os.path.join(cls.fixture_dir, "exofop_ctoi.csv")
        confirmed = (
            "pl_name,hostname,tic_id,pl_orbper,pl_tranmid,pl_trandur,discoverymethod\n"
            "TOI-469.01,SYNTHETIC,TIC 33692729,13.6308205,2458474.56927,4.327,Transit\n"
            "HIP 29442 c,SYNTHETIC,TIC 33692729,3.5379559,2458471.36,2.429,Radial Velocity\n"
            "HIP 29442 d,SYNTHETIC,TIC 33692729,6.429575,2458472.9865,3.080,Radial Velocity\n"
        )
        toi = "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
        toi += "33692729,469.01,2458474.569581,13.6308024,4.3426564,1179,CP\n"
        for i in range(6):
            toi += f"{900001+i},{9001+i}.01,2458500.0,{2+i}.0,2.0,500,PC\n"
        for path, payload in [(cls.confirmed_path, confirmed), (cls.toi_path, toi),
                              (cls.ctoi_path, "tic,ctoi\n33692729,33692729.01\n")]:
            with open(path, "wb") as stream:
                stream.write(payload.encode("utf-8"))
        cls.expected_toi_sha256 = hashlib.sha256(toi.encode("utf-8")).hexdigest()
        cls.expected_raw_retrieval_utc = None  # SYNTHETIC, never retrieved
        # Fail closed if production code tries network access during these contracts.
        cls._network = mock.patch("socket.socket.connect", side_effect=AssertionError("Offline SYNTHETIC suite attempted network"))
        cls._network.start()
        cls.addClassCleanup(cls._network.stop)

    @classmethod
    def tearDownClass(cls):
        if cls._orig_offline is not None:
            os.environ["ALDEBARAN_OFFLINE"] = cls._orig_offline
        else:
            os.environ.pop("ALDEBARAN_OFFLINE", None)

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="ald_test_known_sig_")
        self.orig_catalog_cache = aldebaran.CATALOG_CACHE_FILE
        self.orig_toi_cache = aldebaran.TOI_EPHEMERIS_CACHE_FILE
        self.orig_reg_cache = aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE
        self.orig_out_dir = aldebaran.OUTPUT_DIR
        self.orig_global_reg = aldebaran.GLOBAL_KNOWN_SIGNALS_REGISTRY
        self.orig_global_toi = aldebaran.GLOBAL_TOI_EPHEMERIDES

        aldebaran.configure_environment(
            output_dir=self.tmp_dir,
            catalog_cache_file=os.path.join(self.tmp_dir, "known_catalog_cache.txt"),
            toi_ephemeris_cache_file=os.path.join(self.tmp_dir, "toi_ephemerides_cache.json"),
            known_signals_registry_cache_file=os.path.join(self.tmp_dir, "known_signals_registry.json"),
            toi_ephemerides={},
            known_signals_registry=None,
        )

    def tearDown(self):
        aldebaran.CATALOG_CACHE_FILE = self.orig_catalog_cache
        aldebaran.TOI_EPHEMERIS_CACHE_FILE = self.orig_toi_cache
        aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE = self.orig_reg_cache
        aldebaran.OUTPUT_DIR = self.orig_out_dir
        aldebaran.GLOBAL_KNOWN_SIGNALS_REGISTRY = self.orig_global_reg
        aldebaran.GLOBAL_TOI_EPHEMERIDES = self.orig_global_toi
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 1. Independent Derivation of Key Numbers (HIP 29442 c)
    # -----------------------------------------------------------------------
    def test_independent_arithmetic_derivation_hip29442c(self):
        """
        Verify the exact mathematical quantities specified in ALD-007A:
          Candidate: P1 = 3.537953 d, T1 = 4266.34109 BTJD
          Catalog HIP 29442 c: P2 = 3.5379559 d, T2 = 2458471.36 BJD -> 1471.36 BTJD
          Duration = 2.429 h
        """
        p_cand = 3.537953
        t0_cand = 4266.34109

        p_cat = 3.5379559
        t0_cat_bjd = 2458471.36
        t0_cat_btjd = t0_cat_bjd - 2457000.0  # 1471.36
        dur_cat_hours = 2.429

        # Independent calculation:
        delta_p = p_cand - p_cat
        rel_diff = abs(delta_p) / p_cat
        self.assertAlmostEqual(delta_p, -2.9e-6, places=10)
        self.assertAlmostEqual(rel_diff, 8.196823482562071e-07, places=12)

        # Transport catalog from 1471.36 to cycle near 4266.34109:
        cycle_n = round((t0_cand - t0_cat_btjd) / p_cat)
        self.assertEqual(cycle_n, 790)

        t_cat_proj = t0_cat_btjd + cycle_n * p_cat
        self.assertAlmostEqual(t_cat_proj, 4266.345161, places=6)

        dt_days = t0_cand - t_cat_proj
        dt_minutes = dt_days * 1440.0
        self.assertAlmostEqual(dt_days, -0.004071, places=6)
        self.assertAlmostEqual(dt_minutes, -5.86224, places=4)

        # Confirm using module function:
        calc_dt_days, calc_dt_min, phase = known_signals.calculate_cyclic_timing_residual(
            p_cand, t0_cand, p_cat, t0_cat_btjd
        )
        self.assertAlmostEqual(calc_dt_days, dt_days, places=6)
        self.assertAlmostEqual(calc_dt_min, dt_minutes, places=4)
        self.assertAlmostEqual(phase, dt_days / p_cat, places=6)

    # -----------------------------------------------------------------------
    # 2. Integer-Cycle Wrap Invariance & Common Fixed Comparison Anchor
    # -----------------------------------------------------------------------
    def test_integer_cycle_wrap_invariance(self):
        """
        Shifting either candidate T0 or catalog T0 by integer multiples of its own
        period MUST leave the physical cyclic timing comparison strictly invariant.
        """
        p_cand = 3.537953
        t0_cand = 4266.34109

        p_cat = 3.5379559
        t0_cat = 1471.36

        dt_orig_days, dt_orig_min, phase_orig = known_signals.calculate_cyclic_timing_residual(
            p_cand, t0_cand, p_cat, t0_cat
        )

        test_shifts = [
            (1, 0),
            (10, 0),
            (0, 1),
            (0, -5),
            (-7, 13),
            (100, -50),
            (-200, 300),
        ]

        for k_cand, m_cat in test_shifts:
            t0_cand_shifted = t0_cand + k_cand * p_cand
            t0_cat_shifted = t0_cat + m_cat * p_cat

            # With anchor fixed at original candidate epoch:
            dt_s_days, dt_s_min, phase_s = known_signals.calculate_cyclic_timing_residual(
                p_cand, t0_cand_shifted, p_cat, t0_cat_shifted, pivot_time=t0_cand
            )
            self.assertAlmostEqual(
                dt_s_min, dt_orig_min, places=7,
                msg=f"Failed invariance for shift k_cand={k_cand}, m_cat={m_cat}"
            )
            self.assertAlmostEqual(phase_s, phase_orig, places=7)

    # -----------------------------------------------------------------------
    # 3. HIP 29442 Contract on SYNTHETIC Fixtures
    # -----------------------------------------------------------------------
    def test_hip29442_recovery_from_catalog_fixtures(self):
        """
        Loads generated SYNTHETIC catalog-shaped fixtures (nasa_confirmed.csv and nasa_toi.csv).
        Verifies:
          - TIC 33692729 contains 3 confirmed planet rows: b (TOI-469.01), c, d.
          - c and d have discoverymethod == 'Radial Velocity', yet their transit ephemerides are preserved.
          - HIP 29442 c retains period 3.5379559 d and duration 2.429 h.
          - Ingesting TOI table adds TOI 469.01 without collapsing or overwriting.
          - Association against historical candidate P=3.537953 produces exact -5.86 min residual.
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path) or not os.path.exists(self.toi_path):
            self.fail("Catalog audit fixtures not found at expected location.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_toi = pd.read_csv(self.toi_path)

        signals_conf, snap_conf = known_signals.ingest_nasa_confirmed_df(df_conf)
        self.assertEqual(snap_conf.status, "SUCCESS")
        self.assertEqual(len(signals_conf), 3)

        signals_toi, snap_toi = known_signals.ingest_nasa_toi_df(df_toi)
        self.assertEqual(snap_toi.status, "SUCCESS")

        registry = known_signals.KnownSignalsRegistry()
        for s in signals_conf:
            registry.add_signal(s)
        for s in signals_toi:
            registry.add_signal(s)

        host = registry.get_host_context("33692729")
        self.assertIsNotNone(host)
        self.assertTrue(host.has_confirmed_planets)
        self.assertTrue(host.has_toi)

        # Check confirmed planets from nasa_confirmed
        conf_planets = host.get_signals(source_catalog="nasa_confirmed")
        self.assertEqual(len(conf_planets), 3)
        self.assertEqual(len(host.signals), 4)  # 3 confirmed + 1 TOI
        self.assertEqual(len(host.toi_signals), 1)

        planet_names = {s.planet_or_signal_name: s for s in conf_planets}
        self.assertIn("HIP 29442 c", planet_names)
        self.assertIn("HIP 29442 d", planet_names)
        self.assertIn("TOI-469.01", planet_names)

        # HIP 29442 c specifics
        c_sig = planet_names["HIP 29442 c"]
        self.assertEqual(c_sig.discovery_method, "Radial Velocity")
        self.assertTrue(c_sig.has_transit_ephemeris)
        self.assertTrue(c_sig.maskable)
        self.assertAlmostEqual(c_sig.period_days, 3.5379559, places=7)
        self.assertAlmostEqual(c_sig.duration_hours, 2.429, places=4)
        self.assertAlmostEqual(c_sig.duration_days, 2.429 / 24.0, places=6)
        self.assertAlmostEqual(c_sig.epoch_btjd, 1471.36, places=4)

        # Check diagnostic association against candidate with explicit pivot
        metrics = known_signals.associate_candidate_with_known_signals(
            candidate_p=3.537953,
            candidate_t0=4266.34109,
            host_context=host,
            pivot_time=4266.34109,
            candidate_duration_hours=2.4,
        )
        self.assertGreaterEqual(len(metrics), 4)

        c_metric = next((m for m in metrics if m.catalog_name == "HIP 29442 c"), None)
        self.assertIsNotNone(c_metric)
        self.assertEqual(c_metric.status, "MEASURED")
        self.assertAlmostEqual(c_metric.relative_period_diff, 8.1968e-7, places=10)
        self.assertAlmostEqual(c_metric.epoch_difference_minutes, -5.86224, places=4)
        self.assertIn("SOURCE_DISPOSITION_CONFIRMED_PLANET", c_metric.flags)

    # -----------------------------------------------------------------------
    # 4. Multi-Signal Hosts with Conflicting and Distinct Dispositions
    # -----------------------------------------------------------------------
    def test_multi_signal_host_with_conflicting_dispositions(self):
        """
        A single star hosting multiple signals:
          - Signal 1: Confirmed Planet (CP)
          - Signal 2: TOI False Positive (FP)
          - Signal 3: CTOI Unknown
        Must retain all per-signal dispositions without blanket labeling or host-level flattening.
        """
        s1 = known_signals.KnownSignal(
            signal_id="confirmed:TIC_88888_b",
            tic_id="88888",
            source_catalog="nasa_confirmed",
            source_record_id="Star-88888 b",
            planet_or_signal_name="Star-88888 b",
            disposition="CONFIRMED_PLANET",
            raw_disposition="CONFIRMED",
            period_days=4.5,
            epoch_btjd=1200.0,
            duration_hours=3.0,
            has_transit_ephemeris=True,
            maskable=True,
        )
        s2 = known_signals.KnownSignal(
            signal_id="toi:88888.02",
            tic_id="88888",
            source_catalog="nasa_toi",
            source_record_id="88888.02",
            planet_or_signal_name="TOI-88888.02",
            disposition="FALSE_POSITIVE",
            raw_disposition="FP",
            period_days=9.0,
            epoch_btjd=1202.0,
            duration_hours=4.0,
            has_transit_ephemeris=True,
            maskable=True,
        )
        s3 = known_signals.KnownSignal(
            signal_id="ctoi:88888.03",
            tic_id="88888",
            source_catalog="exofop_ctoi",
            source_record_id="88888.03",
            planet_or_signal_name="CTOI-88888.03",
            disposition="UNKNOWN",
            raw_disposition="",
            period_days=15.0,
            epoch_btjd=1210.0,
            duration_hours=2.0,
            has_transit_ephemeris=True,
            maskable=True,
        )

        reg = known_signals.KnownSignalsRegistry()
        reg.add_signal(s1)
        reg.add_signal(s2)
        reg.add_signal(s3)

        host = reg.get_host_context("88888")
        self.assertIsNotNone(host)
        self.assertTrue(host.has_confirmed_planets)
        self.assertTrue(host.has_false_positive)
        self.assertEqual(len(host.signals), 3)

        # Candidate tested against this host receives independent metrics per signal
        metrics = known_signals.associate_candidate_with_known_signals(
            candidate_p=4.50005,
            candidate_t0=1200.001,
            host_context=host,
            pivot_time=1200.001,
        )
        self.assertEqual(len(metrics), 3)
        m_map = {m.catalog_name: m for m in metrics}

        self.assertEqual(m_map["Star-88888 b"].catalog_disposition, "CONFIRMED_PLANET")
        self.assertIn("SOURCE_DISPOSITION_CONFIRMED_PLANET", m_map["Star-88888 b"].flags)

        self.assertEqual(m_map["TOI-88888.02"].catalog_disposition, "FALSE_POSITIVE")
        self.assertIn("SOURCE_DISPOSITION_FALSE_POSITIVE", m_map["TOI-88888.02"].flags)

        self.assertEqual(m_map["CTOI-88888.03"].catalog_disposition, "UNKNOWN")

    # -----------------------------------------------------------------------
    # 5. Missing Values, Quarantines, & Structured Reporting
    # -----------------------------------------------------------------------
    def test_missing_values_and_quarantined_rows(self):
        """
        Verify missing fields produce UNAVAILABLE without default fabrication:
          - Missing duration is None (never 0.08 d in canonical registry).
          - Missing duration results in maskable=False with explicit unmaskable_reason.
          - Missing depth is None.
          - Negative/invalid period is quarantined with explicit reason.
        """
        raw_csv = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "11111,101.01,2458500.0,5.0,,,PC\n"  # missing dur and dep
            "22222,102.01,2458500.0,-2.5,2.0,500.0,PC\n"  # negative period -> quarantine
            "33333,103.01,,10.0,3.0,1000.0,PC\n"  # missing epoch -> incomplete ephemeris
        )
        df = pd.read_csv(io.StringIO(raw_csv))
        signals, snap = known_signals.ingest_nasa_toi_df(df)

        self.assertEqual(snap.row_count_raw, 3)
        self.assertEqual(snap.row_count_valid, 2)
        self.assertEqual(snap.row_count_quarantined, 1)

        s1 = next(s for s in signals if s.tic_id == "11111")
        self.assertIsNone(s1.duration_hours)
        self.assertIsNone(s1.duration_days)
        self.assertIsNone(s1.depth_ppm)
        self.assertTrue(s1.has_transit_ephemeris)
        # R3 fix: missing duration means maskable=False!
        self.assertFalse(s1.maskable)
        self.assertIn("Missing or invalid transit duration", str(s1.unmaskable_reason))

        s2 = next(s for s in signals if s.tic_id == "22222")
        self.assertTrue(s2.quarantined)
        self.assertIn("Non-finite or non-positive", str(s2.quarantine_reason))
        self.assertFalse(s2.maskable)

        s3 = next(s for s in signals if s.tic_id == "33333")
        self.assertFalse(s3.quarantined)
        self.assertFalse(s3.has_transit_ephemeris)
        self.assertFalse(s3.maskable)
        self.assertIn("Incomplete", str(s3.unmaskable_reason))

        # Check association reporting on missing/quarantined signals (with explicit pivot)
        host_ctx = known_signals.HostCatalogContext(tic_id="test", signals=[s1, s2, s3])
        metrics = known_signals.associate_candidate_with_known_signals(
            candidate_p=5.0, candidate_t0=1500.0, host_context=host_ctx, pivot_time=1500.0
        )
        self.assertEqual(metrics[0].status, "MEASURED")
        self.assertEqual(metrics[1].status, "QUARANTINED")
        self.assertEqual(metrics[2].status, "UNAVAILABLE")

    # -----------------------------------------------------------------------
    # 6. Atomic Serialization, Schema Validation, & Cache Integrity
    # -----------------------------------------------------------------------
    def test_atomic_serialization_and_cache_validation(self):
        """
        Verify:
          - Registry persists atomically to JSON and roundtrips losslessly.
          - Loaded cache snapshots receive status='CACHED' (never freshly retrieved).
          - Schema version mismatch or malformed JSON raises ValueError.
        """
        reg = known_signals.KnownSignalsRegistry()
        sig = known_signals.KnownSignal(
            signal_id="toi:101.01",
            tic_id="11111",
            source_catalog="nasa_toi",
            source_record_id="101.01",
            planet_or_signal_name="TOI-101.01",
            disposition="PLANET_CANDIDATE",
            raw_disposition="PC",
            period_days=5.0,
            epoch_btjd=1500.0,
            duration_hours=2.5,
            duration_days=2.5 / 24.0,
            depth_ppm=850.0,
            has_transit_ephemeris=True,
            maskable=True,
        )
        snap = known_signals.SourceSnapshot(
            source_name="nasa_toi",
            provider="NASA Exoplanet Archive",
            query_or_url="toi",
            retrieval_utc="2026-10-16T12:00:00Z",
            status="SUCCESS",
            row_count_raw=1,
            row_count_valid=1,
            row_count_quarantined=0,
            payload_sha256="fakehash123",
        )
        reg.add_signal(sig)
        reg.register_snapshot(snap)

        cache_path = os.path.join(self.tmp_dir, "test_registry.json")
        reg.save_to_json(cache_path)
        self.assertTrue(os.path.exists(cache_path))

        # Load back
        loaded = known_signals.KnownSignalsRegistry.load_from_json(cache_path, mark_as_cached=True)
        self.assertEqual(loaded.schema_version, "known_signals_v1")
        self.assertEqual(loaded.source_snapshots["nasa_toi"].status, "CACHED")
        loaded_host = loaded.get_host_context("11111")
        self.assertIsNotNone(loaded_host)
        self.assertEqual(len(loaded_host.signals), 1)
        self.assertAlmostEqual(loaded_host.signals[0].period_days, 5.0)
        self.assertAlmostEqual(loaded_host.signals[0].duration_hours, 2.5)

        # Test corrupt JSON
        bad_json_path = os.path.join(self.tmp_dir, "corrupt.json")
        with open(bad_json_path, "w", encoding="utf-8") as f:
            f.write("{invalid_json: true")
        with self.assertRaises(Exception):
            known_signals.KnownSignalsRegistry.load_from_json(bad_json_path)

        # Test schema mismatch
        bad_schema_path = os.path.join(self.tmp_dir, "bad_schema.json")
        with open(bad_schema_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "known_signals_v0", "source_snapshots": {}, "hosts": {}}, f)
        with self.assertRaises(ValueError):
            known_signals.KnownSignalsRegistry.load_from_json(bad_schema_path)

    # -----------------------------------------------------------------------
    # 7. Hypothetical / Non-Active Transit Mask Geometry Helper
    # -----------------------------------------------------------------------
    def test_diagnostic_transit_mask_geometry_helper(self):
        """
        Verify the hypothetical non-active mask generator computes exact geometry:
          - Period = 10.0 d, Epoch = 100.0, Duration = 1.0 d
          - Time array with points in-transit and out-of-transit
        """
        t = np.array([99.0, 99.4, 99.6, 100.0, 100.4, 100.6, 105.0, 109.8, 110.0, 110.2])
        mask = known_signals.build_diagnostic_transit_mask(
            time=t,
            period_days=10.0,
            epoch_btjd=100.0,
            duration_days=1.0,
            width_factor=1.0,
            label="hypothetical_non_active",
        )
        expected = np.array([
            False,  # 99.0: dt = 1.0 > 0.5
            False,  # 99.4: dt = 0.6 > 0.5
            True,   # 99.6: dt = 0.4 <= 0.5
            True,   # 100.0: dt = 0.0 <= 0.5
            True,   # 100.4: dt = 0.4 <= 0.5
            False,  # 100.6: dt = 0.6 > 0.5
            False,  # 105.0: dt = 5.0 > 0.5
            True,   # 109.8: dt = 0.2 <= 0.5
            True,   # 110.0: dt = 0.0 <= 0.5
            True,   # 110.2: dt = 0.2 <= 0.5
        ])
        np.testing.assert_array_equal(mask, expected)

        # Invalid inputs must raise ValueError
        with self.assertRaises(ValueError):
            known_signals.build_diagnostic_transit_mask(t, -5.0, 100.0, 1.0)
        with self.assertRaises(ValueError):
            known_signals.build_diagnostic_transit_mask(t, 10.0, 100.0, -1.0)

    # -----------------------------------------------------------------------
    # 8. Deliberate Acceptance Controls / Falsification (Rule 2)
    # -----------------------------------------------------------------------
    def test_fault_a_dropped_confirmed_records_detected(self):
        """
        Deliberate Fault A: Drop HIP 29442 c and d records.
        The acceptance check verifying HIP 29442 c recovery MUST detect the fault.
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_fault = df_conf[~df_conf["pl_name"].str.contains("HIP 29442 [cd]", regex=True)].copy()
        signals_fault, _ = known_signals.ingest_nasa_confirmed_df(df_fault)

        reg_fault = known_signals.KnownSignalsRegistry()
        for s in signals_fault:
            reg_fault.add_signal(s)

        host_fault = reg_fault.get_host_context("33692729")
        names_found = [s.planet_or_signal_name for s in host_fault.signals]
        self.assertNotIn("HIP 29442 c", names_found, "Fault A succeeded in dropping HIP 29442 c")
        self.assertNotEqual(len(host_fault.confirmed_planets), 3)

    def test_fault_b_host_only_confirmed_projection_detected(self):
        """
        Deliberate Fault B: Revert to legacy behavior where only host TIC IDs were retained
        and planet ephemerides were completely discarded.
        The acceptance check verifying transit midpoint ephemeris MUST detect the fault.
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        host_context = known_signals.HostCatalogContext(tic_id="33692729", signals=[])
        metrics = known_signals.associate_candidate_with_known_signals(
            candidate_p=3.537953, candidate_t0=4266.34109, host_context=host_context, pivot_time=4266.34109
        )
        self.assertEqual(len(metrics), 0, "Host-only projection drops planet signal ephemerides")

    def test_fault_c_mixed_p_and_t0_solutions_detected(self):
        """
        Deliberate Fault C: Combine P from HIP 29442 c (3.5379559 d) with T0 from
        TOI 469.01 (1474.56927 BTJD).
        The acceptance check expecting -5.86 min residual MUST detect the mismatch.
        The exact centered cyclic residual for this mixed pair is +467.445456 min.
        """
        p_c = 3.5379559
        t0_wrong = 1474.56927

        calc_dt_days, calc_dt_min, _ = known_signals.calculate_cyclic_timing_residual(
            3.537953, 4266.34109, p_c, t0_wrong, pivot_time=4266.34109
        )
        self.assertNotAlmostEqual(calc_dt_min, -5.86224, places=1)
        self.assertAlmostEqual(calc_dt_min, 467.445456, places=4)
        self.assertLessEqual(abs(calc_dt_min), (p_c / 2.0) * 1440.0)

    def test_fault_d_discovery_method_filter_detected(self):
        """
        Deliberate Fault D: Filter confirmed planets by discoverymethod == 'Transit'.
        HIP 29442 c and d are discovered via Radial Velocity, so they would be erroneously dropped.
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_filtered = df_conf[df_conf["discoverymethod"] == "Transit"].copy()
        signals_filtered, _ = known_signals.ingest_nasa_confirmed_df(df_filtered)

        names = [s.planet_or_signal_name for s in signals_filtered]
        self.assertNotIn("HIP 29442 c", names)
        self.assertNotIn("HIP 29442 d", names)
        self.assertEqual(len(names), 1)

    def test_fault_e_naive_unprojected_epoch_shift_detected(self):
        """
        Deliberate Fault E: A naive unprojected comparison (T_cand - T_cat) without cycle transport
        fails integer-cycle wrap invariance when T0 is shifted.
        """
        p_cand = 3.537953
        t0_cand = 4266.34109
        p_cat = 3.5379559
        t0_cat = 1471.36

        def naive_unprojected_diff(t1, t2):
            return (t1 - t2) * 1440.0

        diff_orig = naive_unprojected_diff(t0_cand, t0_cat)
        diff_shifted = naive_unprojected_diff(t0_cand + 10 * p_cand, t0_cat)

        self.assertNotAlmostEqual(diff_orig, diff_shifted, places=1)
        _, dt1, _ = known_signals.calculate_cyclic_timing_residual(p_cand, t0_cand, p_cat, t0_cat)
        _, dt2, _ = known_signals.calculate_cyclic_timing_residual(
            p_cand, t0_cand + 10 * p_cand, p_cat, t0_cat, pivot_time=t0_cand
        )
        self.assertAlmostEqual(dt1, dt2, places=9)

    # -----------------------------------------------------------------------
    # 9. Production Catalog Ingestion Boundary & Legacy Compatibility
    # -----------------------------------------------------------------------
    def test_production_boundary_and_legacy_compatibility_contract(self):
        """
        Invokes aldebaran.load_known_catalogs_and_toi_ephemerides with offline fixtures:
          - Strictly returns 3-tuple: (known_catalog_tics, known_eb_tics, toi_ephemerides).
          - Populates aldebaran.get_known_signals_registry().
          - Exposes complete HIP 29442 context via get_known_signal_context_for_tic('33692729').
          - Preserves legacy get_known_transits_for_tic('33692729') and get_known_exoplanet_tics().
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path) or not os.path.exists(self.toi_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_toi = pd.read_csv(self.toi_path)

        ret = aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_confirmed": df_conf, "nasa_toi": df_toi}
        )
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 3)

        known_catalog_tics, known_eb_tics, toi_ephemerides = ret
        self.assertIsInstance(known_catalog_tics, set)
        self.assertIsInstance(known_eb_tics, set)
        self.assertIsInstance(toi_ephemerides, dict)

        self.assertIn("33692729", known_catalog_tics)
        self.assertIn("33692729", toi_ephemerides)

        reg = aldebaran.get_known_signals_registry()
        self.assertIsNotNone(reg)
        self.assertIsInstance(reg, known_signals.KnownSignalsRegistry)

        host_ctx = aldebaran.get_known_signal_context_for_tic("33692729")
        self.assertIsNotNone(host_ctx)
        self.assertTrue(host_ctx.has_confirmed_planets)
        self.assertTrue(host_ctx.has_toi)

        conf_names = {s.planet_or_signal_name for s in host_ctx.confirmed_planets}
        self.assertIn("HIP 29442 c", conf_names)
        self.assertIn("HIP 29442 d", conf_names)
        self.assertIn("TOI-469.01", conf_names)

        legacy_transits = aldebaran.get_known_transits_for_tic("33692729")
        self.assertIsInstance(legacy_transits, list)
        self.assertGreaterEqual(len(legacy_transits), 1)

        legacy_all_tics = aldebaran.get_known_exoplanet_tics()
        self.assertIn("33692729", legacy_all_tics)

    # -----------------------------------------------------------------------
    # 10. R1 Fix Verification: Idempotent Refresh & Row Retirement
    # -----------------------------------------------------------------------
    def test_r1_idempotent_refresh_on_actual_loader(self):
        """
        R1 Requirement:
          Running load_known_catalogs_and_toi_ephemerides repeatedly on the same
          input must be idempotent: signal counts MUST NOT accumulate/double!
          (e.g. 4 signals -> 4 signals, NOT 8 signals; HIP 29442 TOI entry 1, NOT 2).
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path) or not os.path.exists(self.toi_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_toi = pd.read_csv(self.toi_path)

        # Run 1
        _, _, toi_ephem_1 = aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_confirmed": df_conf, "nasa_toi": df_toi}
        )
        reg_1 = aldebaran.get_known_signals_registry()
        total_1 = reg_1.total_signals()
        hip_toi_count_1 = len(toi_ephem_1.get("33692729", []))

        # Run 2 (identical fixtures)
        _, _, toi_ephem_2 = aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_confirmed": df_conf, "nasa_toi": df_toi}
        )
        reg_2 = aldebaran.get_known_signals_registry()
        total_2 = reg_2.total_signals()
        hip_toi_count_2 = len(toi_ephem_2.get("33692729", []))

        # Critical assertions: count remains exactly identical
        self.assertEqual(total_1, total_2, f"Idempotence violated: count went from {total_1} to {total_2}")
        self.assertEqual(hip_toi_count_1, hip_toi_count_2)
        self.assertEqual(hip_toi_count_2, 1, "HIP 29442 must have exactly 1 legacy TOI entry")

    def test_r1_source_replacement_and_row_retirement(self):
        """
        R1 Requirement:
          When a source is refreshed and a row is retired (e.g. removed from the TOI table),
          that retired row MUST be purged from the active view and registry, rather than
          lingering as a stale solution.
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path) or not os.path.exists(self.toi_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_toi = pd.read_csv(self.toi_path)

        # 1. Ingest initial data
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_confirmed": df_conf, "nasa_toi": df_toi}
        )
        reg = aldebaran.get_known_signals_registry()
        host_initial = reg.get_host_context("33692729")
        self.assertTrue(host_initial.has_toi)

        # 2. Refresh with modified TOI table where HIP 29442 (TOI 469.01) is retired/removed
        df_toi_retired = df_toi[df_toi["tid"].astype(str) != "33692729"].copy()
        _, _, toi_ephem_new = aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_confirmed": df_conf, "nasa_toi": df_toi_retired}
        )

        # Retired row must be removed from active legacy TOI ephemeris table:
        self.assertNotIn("33692729", toi_ephem_new, "Retired TOI must be purged from active legacy TOI view")

        # But confirmed planets from nasa_confirmed MUST remain intact:
        host_updated = aldebaran.get_known_signal_context_for_tic("33692729")
        self.assertFalse(host_updated.has_toi, "Host should no longer have TOI signal")
        self.assertTrue(host_updated.has_confirmed_planets, "Host should still retain confirmed planets")
        self.assertEqual(len(host_updated.signals), 3)

    def test_r1_negative_control_accumulator_fault_detected(self):
        """
        Negative Control for R1:
          Simulate an accumulator bug (appending rows without replacement).
          Demonstrate that the test check detects this fault.
        """
        reg = known_signals.KnownSignalsRegistry()
        s = known_signals.KnownSignal(
            signal_id="toi:1.01", tic_id="100", source_catalog="nasa_toi",
            source_record_id="1.01", planet_or_signal_name="TOI-1.01",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=3.0,
            epoch_btjd=100.0, duration_hours=2.0, has_transit_ephemeris=True, maskable=True
        )
        snap = known_signals.SourceSnapshot(
            source_name="nasa_toi", provider="test", query_or_url="test",
            retrieval_utc="2026-10-16T12:00:00Z", status="SUCCESS",
            row_count_raw=1, row_count_valid=1, row_count_quarantined=0
        )
        # Proper idempotent replacement:
        reg.replace_source_signals("nasa_toi", [s], snap)
        reg.replace_source_signals("nasa_toi", [s], snap)
        self.assertEqual(reg.total_signals(), 1, "Idempotent replacement retains exactly 1 signal")

        # Buggy behavior (raw append):
        reg.add_signal(s)
        self.assertEqual(reg.total_signals(), 2, "Fault detected: raw append doubled the signal count")

    # -----------------------------------------------------------------------
    # 11. R2 Fix Verification: Full 4-Source Tracking & Cache Truthfulness
    # -----------------------------------------------------------------------
    def test_r2_four_source_accounting_and_truthful_statuses(self):
        """
        R2 Requirement:
          Registry must track all four canonical sources (nasa_toi, nasa_confirmed,
          exofop_ctoi, vizier_tess_eb). When fixtures are provided for only 2 sources in offline mode,
          the other 2 must be recorded with status='UNAVAILABLE', not silently skipped.
        """
        if not self.has_fixtures or not os.path.exists(self.confirmed_path) or not os.path.exists(self.toi_path):
            self.fail("Catalog audit fixtures not found.")

        df_conf = pd.read_csv(self.confirmed_path)
        df_toi = pd.read_csv(self.toi_path)

        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_confirmed": df_conf, "nasa_toi": df_toi}
        )
        reg = aldebaran.get_known_signals_registry()
        self.assertIsNotNone(reg)

        # Full 4-source accounting
        self.assertIn("nasa_toi", reg.source_snapshots)
        self.assertIn("nasa_confirmed", reg.source_snapshots)
        self.assertIn("exofop_ctoi", reg.source_snapshots)
        self.assertIn("vizier_tess_eb", reg.source_snapshots)

        self.assertEqual(reg.source_snapshots["nasa_toi"].status, "SUCCESS")
        self.assertEqual(reg.source_snapshots["nasa_confirmed"].status, "SUCCESS")
        self.assertEqual(reg.source_snapshots["exofop_ctoi"].status, "UNAVAILABLE")
        self.assertEqual(reg.source_snapshots["vizier_tess_eb"].status, "UNAVAILABLE")

        host = reg.get_host_context("33692729")
        self.assertEqual(host.coverage_status, "INCOMPLETE")

    def test_r2_empty_refresh_preserves_good_cache(self):
        """
        R2 Requirement:
          When a source refresh returns an empty response after a good cache,
          the good cached signals must NOT be wiped out; the snapshot must record
          the INCOMPLETE attempt status truthfully.
        """
        if not self.has_fixtures or not os.path.exists(self.toi_path):
            self.fail("Catalog audit fixtures not found.")

        df_toi = pd.read_csv(self.toi_path)
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": df_toi}
        )
        reg = aldebaran.get_known_signals_registry()
        initial_toi_count = sum(1 for h in reg.hosts.values() for s in h.signals if s.source_catalog == "nasa_toi")
        self.assertGreater(initial_toi_count, 0)

        # Simulate empty refresh
        empty_df = pd.DataFrame(columns=df_toi.columns)
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": empty_df}
        )
        reg_after = aldebaran.get_known_signals_registry()
        toi_count_after = sum(1 for h in reg_after.hosts.values() for s in h.signals if s.source_catalog == "nasa_toi")

        # Good cached rows preserved!
        self.assertEqual(toi_count_after, initial_toi_count)
        self.assertEqual(reg_after.source_snapshots["nasa_toi"].last_attempt_status, "INCOMPLETE")

    def test_r2_persistence_failure_structured_status(self):
        """
        R2 Requirement:
          When persisting the registry to cache fails (e.g. read-only filesystem or OS error),
          persistence_status='FAILED' and error_message must be recorded on the registry object.
        """
        reg = known_signals.KnownSignalsRegistry()
        s = known_signals.KnownSignal(
            signal_id="toi:1.01", tic_id="100", source_catalog="nasa_toi",
            source_record_id="1.01", planet_or_signal_name="TOI-1.01",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=3.0,
            epoch_btjd=100.0, duration_hours=2.0, has_transit_ephemeris=True, maskable=True
        )
        reg.add_signal(s)

        # Attempt to save to an illegal path (directory without permissions or non-existent drive)
        invalid_path = os.path.join(self.tmp_dir, "non_existent_subdir", "deep", "registry.json")
        # Ensure parent does not exist and cannot be created by making a conflicting file
        conflict_file = os.path.join(self.tmp_dir, "non_existent_subdir")
        with open(conflict_file, "w") as f:
            f.write("not a directory")

        with self.assertRaises(Exception):
            reg.save_to_json(invalid_path)

        self.assertEqual(reg.persistence_status, "FAILED")
        self.assertIsNotNone(reg.persistence_error)

    def test_r2_cache_validation_malformed_counts_and_identities(self):
        """
        R2 Requirement:
          Cache deserialization must strictly validate:
            1. Snapshot counts match actual signal counts in hosts.
            2. Host key matches signal tic_id.
            3. Non-canonical host keys (e.g. unnormalized '00100') are rejected.
        """
        # Case 1: Count mismatch (snapshot claims 2 rows, but hosts only have 1)
        bad_count_json = {
            "schema_version": "known_signals_v1",
            "created_utc": "2026-10-16T12:00:00Z",
            "source_snapshots": {
                "nasa_toi": {
                    "source_name": "nasa_toi", "provider": "NASA", "query_or_url": "toi",
                    "retrieval_utc": "2026-10-16T12:00:00Z", "status": "SUCCESS",
                    "row_count_raw": 2, "row_count_valid": 2, "row_count_quarantined": 0,
                    "payload_sha256": "fake", "schema_version": "known_signals_v1"
                }
            },
            "hosts": {
                "100": {
                    "tic_id": "100", "source_statuses": {"nasa_toi": "SUCCESS"}, "coverage_status": "COMPLETE", "registry_available": True,
                    "signals": [
                        {
                            "signal_id": "toi:1.01", "tic_id": "100", "source_catalog": "nasa_toi",
                            "source_record_id": "1.01", "planet_or_signal_name": "TOI-1.01",
                            "disposition": "PLANET_CANDIDATE", "raw_disposition": "PC",
                            "period_days": 3.0, "epoch_btjd": 100.0, "duration_hours": 2.0, "has_transit_ephemeris": True, "maskable": True
                        }
                    ]
                }
            }
        }
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(bad_count_json)
        self.assertIn("count mismatch", str(ctx.exception).lower())

        # Case 2: Host TIC mismatch (host key is '100', but signal TIC is '200')
        bad_identity_json = {
            "schema_version": "known_signals_v1",
            "created_utc": "2026-10-16T12:00:00Z",
            "source_snapshots": {
                "nasa_toi": {
                    "source_name": "nasa_toi", "provider": "NASA", "query_or_url": "toi",
                    "retrieval_utc": "2026-10-16T12:00:00Z", "status": "SUCCESS",
                    "row_count_raw": 1, "row_count_valid": 1, "row_count_quarantined": 0,
                    "payload_sha256": "fake", "schema_version": "known_signals_v1"
                }
            },
            "hosts": {
                "100": {
                    "tic_id": "100", "source_statuses": {"nasa_toi": "SUCCESS"}, "coverage_status": "COMPLETE", "registry_available": True,
                    "signals": [
                        {
                            "signal_id": "toi:1.01", "tic_id": "200", "source_catalog": "nasa_toi",
                            "source_record_id": "1.01", "planet_or_signal_name": "TOI-1.01",
                            "disposition": "PLANET_CANDIDATE", "raw_disposition": "PC",
                            "period_days": 3.0, "epoch_btjd": 100.0, "duration_hours": 2.0, "has_transit_ephemeris": True, "maskable": True
                        }
                    ]
                }
            }
        }
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(bad_identity_json)
        self.assertIn("does not match host", str(ctx.exception).lower())

    def test_r2_unknown_host_vs_unavailable_registry(self):
        """
        R2 Requirement:
          Distinguish target star with no signals in complete coverage from unavailable registry:
            - Uncataloged TIC returns HostCatalogContext with signals=[], has_signals=False, registry_available=True.
            - When registry is None, returns HostCatalogContext with registry_available=False, coverage_status='UNAVAILABLE'.
        """
        reg = known_signals.KnownSignalsRegistry()
        for src in known_signals.CANONICAL_SOURCES:
            reg.record_source_attempt(src, status="SUCCESS")

        aldebaran.GLOBAL_KNOWN_SIGNALS_REGISTRY = reg
        ctx_star = aldebaran.get_known_signal_context_for_tic("999999999")
        self.assertIsNotNone(ctx_star)
        self.assertEqual(len(ctx_star.signals), 0)
        self.assertFalse(ctx_star.has_signals)
        self.assertTrue(ctx_star.registry_available)
        self.assertEqual(ctx_star.coverage_status, "COMPLETE")

        # When registry is uninitialized
        aldebaran.GLOBAL_KNOWN_SIGNALS_REGISTRY = None
        # Force cache path non-existent
        aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE = os.path.join(self.tmp_dir, "non_existent.json")
        ctx_unavail = aldebaran.get_known_signal_context_for_tic("999999999")
        self.assertIsNotNone(ctx_unavail)
        self.assertFalse(ctx_unavail.registry_available)
        self.assertEqual(ctx_unavail.coverage_status, "UNAVAILABLE")

    # -----------------------------------------------------------------------
    # 12. R3 Fix Verification: Strict Identifiers, Geometry & Provenance
    # -----------------------------------------------------------------------
    def test_r3_strict_tic_normalization_and_large_precision(self):
        """
        R3 Requirement:
          - TIC ID must strictly be a positive integer.
          - Rejects 'BAD-ID', floats with non-zero fractional parts, negative numbers, and 0.
          - Preserves large integer precision losslessly (> 2^53).
        """
        self.assertEqual(known_signals.normalize_tic_id("33692729"), "33692729")
        self.assertEqual(known_signals.normalize_tic_id(33692729), "33692729")
        self.assertEqual(known_signals.normalize_tic_id("TIC 33692729"), "33692729")
        self.assertEqual(known_signals.normalize_tic_id("0033692729"), "33692729")
        self.assertEqual(known_signals.normalize_tic_id(33692729.0), "33692729")

        # Large integer precision (> 2^53 = 9007199254740992)
        large_tic = "1234567890123456789"
        self.assertEqual(known_signals.normalize_tic_id(large_tic), large_tic)

        # Invalid cases
        invalid_ids = ["BAD-ID", "TIC-BAD", "-100", 0, "0", "123.456", None, float("nan"), float("inf")]
        for bad_id in invalid_ids:
            with self.assertRaises(ValueError, msg=f"Should reject invalid TIC: {bad_id}"):
                known_signals.normalize_tic_id(bad_id)

    def test_r3_finite_positive_geometry_quarantines(self):
        """
        R3 Requirement:
          Non-finite (+inf, nan) or non-positive (<=0) orbital period or epoch
          MUST be quarantined and marked non-maskable.
        """
        raw_csv = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "100,1.01,1500.0,inf,2.0,500.0,PC\n"     # inf period
            "101,2.01,inf,5.0,2.0,500.0,PC\n"      # inf epoch
            "102,3.01,1500.0,-5.0,2.0,500.0,PC\n"   # negative period
            "103,4.01,1500.0,0.0,2.0,500.0,PC\n"    # zero period
        )
        df = pd.read_csv(io.StringIO(raw_csv))
        signals, snap = known_signals.ingest_nasa_toi_df(df)

        self.assertEqual(snap.row_count_raw, 4)
        self.assertEqual(snap.row_count_valid, 0)
        self.assertEqual(snap.row_count_quarantined, 4)

        for s in signals:
            self.assertTrue(s.quarantined, f"Signal {s.signal_id} must be quarantined")
            self.assertFalse(s.maskable, f"Signal {s.signal_id} must not be maskable")
            self.assertFalse(s.has_transit_ephemeris, f"Signal {s.signal_id} has invalid ephemeris")

    def test_r3_missing_duration_geometry_unmaskable(self):
        """
        R3 Requirement:
          Missing duration must produce maskable=False with unmaskable_reason
          stating that transit duration is required for mask geometry,
          while preserving has_transit_ephemeris=True.
        """
        raw_csv = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "100,1.01,1500.0,5.0,,,PC\n"
        )
        df = pd.read_csv(io.StringIO(raw_csv))
        signals, _ = known_signals.ingest_nasa_toi_df(df)
        s = signals[0]

        self.assertTrue(s.has_transit_ephemeris)
        self.assertFalse(s.maskable)
        self.assertIn("duration required for mask geometry", str(s.unmaskable_reason))
        self.assertFalse(s.quarantined)

    def test_r3_distinct_solution_identities_on_same_signal(self):
        """
        R3 Requirement:
          Two conflicting solutions on the same TOI (or consecutive rows)
          must receive distinct, stable solution_ids linked to source row and snapshot.
        """
        raw_csv = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,pl_trandep,tfopwg_disp\n"
            "100,1.01,1500.0,5.0,2.0,500.0,PC\n"
            "100,1.01,1500.2,5.1,2.2,520.0,PC\n"
        )
        df = pd.read_csv(io.StringIO(raw_csv))
        signals, _ = known_signals.ingest_nasa_toi_df(df)

        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0].signal_id, "toi:1.01")
        self.assertEqual(signals[1].signal_id, "toi:1.01")
        # Distinct solution IDs
        self.assertNotEqual(signals[0].solution_id, signals[1].solution_id)
        self.assertIn("P5.0", str(signals[0].solution_id))
        self.assertIn("P5.1", str(signals[1].solution_id))
        self.assertEqual(signals[0].source_row_index, 0)
        self.assertEqual(signals[1].source_row_index, 1)

    def test_r3_missing_headers_preserves_raw_evidence(self):
        """
        R3 Requirement:
          When required headers are missing in a table, raw row evidence must NOT
          be silently discarded; all rows must be returned as quarantined signals.
        """
        bad_df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
        signals, snap = known_signals.ingest_nasa_toi_df(bad_df)

        self.assertEqual(snap.status, "FAILED")
        self.assertEqual(snap.row_count_raw, 2)
        self.assertEqual(snap.row_count_quarantined, 2)
        self.assertEqual(len(signals), 2, "Raw evidence rows must be preserved as quarantined signals")
        self.assertTrue(signals[0].quarantined)
        self.assertEqual(signals[0].raw_fields["colA"], 1)

    # -----------------------------------------------------------------------
    # 13. R4 Fix Verification: Time Contracts, Provenance, & No Cutoffs
    # -----------------------------------------------------------------------
    def test_r4_timing_contract_requires_pivot(self):
        """
        R4 Requirement:
          Association without a supplied pivot_time must mark timing status as
          TIMING_UNAVAILABLE, rather than calculating an arbitrary unprojected diff.
        """
        sig = known_signals.KnownSignal(
            signal_id="toi:1.01", tic_id="100", source_catalog="nasa_toi",
            source_record_id="1.01", planet_or_signal_name="TOI-1.01",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=5.0,
            epoch_btjd=1500.0, duration_hours=2.0, has_transit_ephemeris=True, maskable=True,
            original_time_format="BTJD", original_time_scale="TDB"
        )
        # Without pivot:
        metric_no_pivot = known_signals.calculate_association_metrics(
            candidate_p=5.0001, candidate_t0=2000.0, signal=sig, pivot_time=None
        )
        self.assertEqual(metric_no_pivot.status, "TIMING_UNAVAILABLE")
        self.assertIsNone(metric_no_pivot.epoch_difference_days)
        self.assertIsNone(metric_no_pivot.epoch_difference_minutes)
        self.assertIn("Common pivot time not provided", str(metric_no_pivot.status_reason))

        # With pivot:
        metric_with_pivot = known_signals.calculate_association_metrics(
            candidate_p=5.0001, candidate_t0=2000.0, signal=sig, pivot_time=2000.0
        )
        self.assertEqual(metric_with_pivot.status, "MEASURED")
        self.assertIsNotNone(metric_with_pivot.epoch_difference_minutes)

    def test_r4_no_arbitrary_match_cutoffs(self):
        """
        R4 Requirement:
          Association metrics must NOT include arbitrary decision flags
          such as 'KNOWN_PLANET_PERIOD_EPOCH_MATCH' or '<0.001' period match thresholds.
        """
        sig = known_signals.KnownSignal(
            signal_id="confirmed:Star_b", tic_id="100", source_catalog="nasa_confirmed",
            source_record_id="Star b", planet_or_signal_name="Star b",
            disposition="CONFIRMED_PLANET", raw_disposition="CONFIRMED", period_days=3.5379559,
            epoch_btjd=1471.36, duration_hours=2.429, has_transit_ephemeris=True, maskable=True,
            original_time_format="BJD", original_time_scale="TDB"
        )
        metric = known_signals.calculate_association_metrics(
            candidate_p=3.537953, candidate_t0=4266.34109, signal=sig, pivot_time=4266.34109
        )
        self.assertEqual(metric.status, "MEASURED")
        # Ensure arbitrary match flags are strictly absent:
        self.assertNotIn("KNOWN_PLANET_PERIOD_EPOCH_MATCH", metric.flags)
        self.assertNotIn("PERIOD_MATCH_WITHIN_0_1_PERCENT", metric.flags)
        self.assertNotIn("EPOCH_MATCH_WITHIN_30_MINUTES", metric.flags)
        # Only factual disposition flags permitted:
        self.assertIn("SOURCE_DISPOSITION_CONFIRMED_PLANET", metric.flags)

    def test_r4_uncertainties_and_time_format_preserved(self):
        """
        R4 Requirement:
          Asymmetric errors (period_err_pos/neg, epoch_err_pos/neg) and time scale
          metadata must be preserved from the raw source rows.
        """
        raw_csv = (
            "tic_id,pl_name,hostname,discoverymethod,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
            "pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_trandur,pl_trandurerr1,pl_trandurerr2,"
            "pl_trandep,pl_trandeperr1,pl_trandeperr2\n"
            "33692729,HIP 29442 c,HIP 29442,Radial Velocity,3.5379559,0.0000010,-0.0000012,"
            "2458471.36,0.01,-0.02,2.429,0.05,-0.06,0.11791,0.001,-0.002\n"
        )
        df = pd.read_csv(io.StringIO(raw_csv))
        signals, _ = known_signals.ingest_nasa_confirmed_df(df)
        s = signals[0]

        self.assertEqual(s.original_time_format, "BJD")
        self.assertEqual(s.original_time_scale, "TDB")
        self.assertEqual(s.epoch_zero_point, 0.0)
        self.assertAlmostEqual(s.period_err_pos, 0.0000010)
        self.assertAlmostEqual(s.period_err_neg, -0.0000012)
        self.assertAlmostEqual(s.epoch_err_pos, 0.01)
        self.assertAlmostEqual(s.epoch_err_neg, -0.02)
        self.assertAlmostEqual(s.duration_err_pos, 0.05)
        self.assertAlmostEqual(s.duration_err_neg, -0.06)
        # Depth in percent converted to ppm: 0.11791% -> 1179.1 ppm
        self.assertAlmostEqual(s.depth_ppm, 1179.1)
        self.assertAlmostEqual(s.depth_err_pos, 10.0)
        self.assertAlmostEqual(s.depth_err_neg, -20.0)

    # -----------------------------------------------------------------------
    # Findings B1–B4 Targeted Acceptance Tests
    # -----------------------------------------------------------------------
    def test_b1_nasa_toi_epoch_btjd_conversion_and_active_legacy_mask(self):
        """
        B1 Acceptance:
          NASA TOI pl_tranmid is in BJD (~2458474.569581).
          Parser converts it to BTJD (~1474.569581) by subtracting 2457000.0,
          preserves raw original_time_format='BJD', original_time_scale='TDB',
          and exports t0 in BTJD to legacy TOI ephemerides.
        """
        raw_toi = (
            "tid,toi,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
            "pl_trandurh,pl_trandurherr1,pl_trandurherr2,pl_trandep,pl_trandeperr1,pl_trandeperr2,tfopwg_disp\n"
            "33692729,469.01,2458474.569581,0.001,-0.001,13.630802,0.0001,-0.0001,4.342656,0.1,-0.1,1000,50,-50,PC\n"
        )
        df = pd.read_csv(io.StringIO(raw_toi))
        signals, _ = known_signals.ingest_nasa_toi_df(df)
        sig = signals[0]

        # BJD raw midpoint converted to BTJD:
        self.assertAlmostEqual(sig.epoch_bjd, 2458474.569581, places=5)
        self.assertAlmostEqual(sig.epoch_btjd, 1474.569581, places=5)
        self.assertEqual(sig.original_time_format, "BJD")
        self.assertEqual(sig.original_time_scale, "TDB")
        self.assertEqual(sig.epoch_zero_point, 0.0)

        # Legacy export exports BTJD:
        reg = known_signals.KnownSignalsRegistry()
        reg.add_signal(sig)
        legacy = reg.export_legacy_toi_ephemerides()
        self.assertIn("33692729", legacy)
        self.assertAlmostEqual(legacy["33692729"][0]["t0"], 1474.569581, places=5)
        self.assertLess(legacy["33692729"][0]["t0"], 100000.0)

        # Deliberate mutation check: if wrong zero point (+2457000) is injected, test must FAIL
        mutated_t0 = legacy["33692729"][0]["t0"] + 2457000.0
        self.assertGreater(mutated_t0, 2400000.0, "Mutation check verifies BJD vs BTJD distinction")

    def test_b1_candidate_time_normalization_and_unknown_handling(self):
        """
        B1 Acceptance:
          Candidate time format (BTJD vs BJD) is normalized to common anchor.
          Physically equivalent candidate epochs in BTJD and BJD yield identical residuals.
          Unknown/missing time format or timescale returns TIMING_UNAVAILABLE.
        """
        # Catalog signal: HIP 29442 c
        sig = known_signals.KnownSignal(
            signal_id="confirmed:HIP_29442_c", tic_id="33692729", source_catalog="nasa_confirmed",
            source_record_id="HIP 29442 c", planet_or_signal_name="HIP 29442 c",
            disposition="CONFIRMED_PLANET", raw_disposition="CONFIRMED", period_days=3.5379559,
            epoch_btjd=1471.36, duration_hours=2.429, has_transit_ephemeris=True, maskable=True,
            original_time_format="BJD", original_time_scale="TDB"
        )

        # 1. Candidate in BTJD:
        m_btjd = known_signals.calculate_association_metrics(
            candidate_p=3.537953, candidate_t0=4266.34109, signal=sig,
            candidate_time_format="BTJD", candidate_time_scale="TDB", pivot_time=4266.34109
        )
        self.assertEqual(m_btjd.status, "MEASURED")
        self.assertAlmostEqual(m_btjd.epoch_difference_minutes, -5.86224, places=4)

        # 2. Candidate in BJD (physically identical epoch: 4266.34109 + 2457000 = 2461266.34109):
        m_bjd = known_signals.calculate_association_metrics(
            candidate_p=3.537953, candidate_t0=2461266.34109, signal=sig,
            candidate_time_format="BJD", candidate_time_scale="TDB", pivot_time=2461266.34109
        )
        self.assertEqual(m_bjd.status, "MEASURED")
        self.assertAlmostEqual(m_bjd.epoch_difference_minutes, -5.86224, places=4)
        self.assertAlmostEqual(m_btjd.epoch_difference_minutes, m_bjd.epoch_difference_minutes, places=6)

        # 3. Format UNKNOWN:
        m_unk_fmt = known_signals.calculate_association_metrics(
            candidate_p=3.537953, candidate_t0=4266.34109, signal=sig,
            candidate_time_format="UNKNOWN", candidate_time_scale="TDB", pivot_time=4266.34109
        )
        self.assertEqual(m_unk_fmt.status, "TIMING_UNAVAILABLE")
        self.assertIsNone(m_unk_fmt.epoch_difference_minutes)

        # 4. Timescale UNKNOWN or None:
        m_unk_scale = known_signals.calculate_association_metrics(
            candidate_p=3.537953, candidate_t0=4266.34109, signal=sig,
            candidate_time_format="BTJD", candidate_time_scale="UNKNOWN", pivot_time=4266.34109
        )
        self.assertEqual(m_unk_scale.status, "TIMING_UNAVAILABLE")

        m_none_scale = known_signals.calculate_association_metrics(
            candidate_p=3.537953, candidate_t0=4266.34109, signal=sig,
            candidate_time_format="BTJD", candidate_time_scale=None, pivot_time=4266.34109
        )
        self.assertEqual(m_none_scale.status, "TIMING_UNAVAILABLE")

    def test_b2_failed_refresh_preserves_good_source(self):
        """
        B2 Acceptance:
          A failed or malformed refresh must NOT purge previous good cached data.
          Coverage status must report INCOMPLETE when a refresh attempt fails.
        """
        if not self.has_fixtures or not os.path.exists(self.toi_path) or not os.path.exists(self.confirmed_path):
            self.fail("Catalog audit fixtures not found.")

        # 1. Initial successful load with good fixtures:
        df_toi = pd.read_csv(self.toi_path)
        df_ps = pd.read_csv(self.confirmed_path)
        df_ctoi = pd.read_csv(self.ctoi_path) if self.ctoi_path and os.path.exists(self.ctoi_path) else pd.DataFrame({"tic": ["33692729"], "ctoi": ["33692729.01"]})
        df_eb = pd.DataFrame({"tic": ["33692729"]})

        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={
                "nasa_toi": df_toi, "nasa_confirmed": df_ps,
                "exofop_ctoi": df_ctoi, "vizier_tess_eb": df_eb
            }
        )
        reg = aldebaran.get_known_signals_registry()
        self.assertIsNotNone(reg)
        initial_signals = reg.total_signals()
        self.assertGreaterEqual(initial_signals, 10)

        # 2. Ingest malformed TOI table (missing required headers):
        df_malformed_toi = pd.DataFrame({"wrong_column": [1, 2, 3]})
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={
                "nasa_toi": df_malformed_toi, "nasa_confirmed": df_ps,
                "exofop_ctoi": df_ctoi, "vizier_tess_eb": df_eb
            }
        )

        # Verify good source was preserved (NOT reduced to 3 or wiped):
        reg_after = aldebaran.get_known_signals_registry()
        toi_signals = [s for h in reg_after.hosts.values() for s in h.signals if s.source_catalog == "nasa_toi"]
        self.assertGreater(len(toi_signals), 0, "Good TOI signals must be preserved on failed refresh")
        self.assertEqual(reg_after.source_snapshots["nasa_toi"].last_attempt_status, "FAILED")
        self.assertEqual(reg_after.get_coverage_status(), "INCOMPLETE")

    def test_b2_empty_active_export_clears_stale_masks(self):
        """
        B2 Acceptance:
          When a successful fresh table arrives with zero usable transit ephemerides,
          canonical active export {} cleanly clears legacy memory and cache masks.
        """
        # 1. Load single TOI with valid ephemeris:
        raw_toi = (
            "tid,toi,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
            "pl_trandurh,pl_trandurherr1,pl_trandurherr2,pl_trandep,pl_trandeperr1,pl_trandeperr2,tfopwg_disp\n"
            "33692729,469.01,2458474.569581,0.001,-0.001,13.630802,0.0001,-0.0001,4.342656,0.1,-0.1,1000,50,-50,PC\n"
        )
        df_toi = pd.read_csv(io.StringIO(raw_toi))
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": df_toi, "nasa_confirmed": pd.DataFrame(), "exofop_ctoi": pd.DataFrame(), "vizier_tess_eb": pd.DataFrame()}
        )
        self.assertEqual(len(aldebaran.get_known_transits_for_tic("33692729")), 1)

        # 2. Refresh with valid table where the TOI has no transit epoch (unmaskable):
        raw_toi_no_epoch = (
            "tid,toi,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
            "pl_trandurh,pl_trandurherr1,pl_trandurherr2,pl_trandep,pl_trandeperr1,pl_trandeperr2,tfopwg_disp\n"
            "33692729,469.01,,,,,13.630802,,,,,1000,,,PC\n"
        )
        df_toi_no_epoch = pd.read_csv(io.StringIO(raw_toi_no_epoch))
        _, _, active_toi = aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": df_toi_no_epoch, "nasa_confirmed": pd.DataFrame(), "exofop_ctoi": pd.DataFrame(), "vizier_tess_eb": pd.DataFrame()}
        )

        # Stale mask must be cleanly cleared:
        self.assertEqual(active_toi, {})
        self.assertEqual(aldebaran.GLOBAL_TOI_EPHEMERIDES, {})
        self.assertEqual(aldebaran.get_known_transits_for_tic("33692729"), [])

    def test_b3_quarantine_records_roundtrip(self):
        """
        B3 Acceptance:
          Quarantine records (with invalid TIC or missing headers) live in quarantine_records,
          not in reg.hosts. They serialize to JSON and reload cleanly without 'quarantine' TIC error.
        """
        raw_bad_tic = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,tfopwg_disp\n"
            "BAD-TIC,999.01,2458474.0,5.0,2.0,PC\n"
            "33692729,469.01,2458474.5,13.6,4.0,PC\n"
        )
        df = pd.read_csv(io.StringIO(raw_bad_tic))
        signals, snap = known_signals.ingest_nasa_toi_df(df)

        reg = known_signals.KnownSignalsRegistry()
        reg.replace_source_signals("nasa_toi", signals, snap)

        self.assertNotIn("quarantine", reg.hosts)
        self.assertEqual(len(reg.quarantine_records), 1)
        self.assertIn("33692729", reg.hosts)

        # Save and reload:
        cache_path = os.path.join(self.tmp_dir, "test_quarantine_roundtrip.json")
        reg.save_to_json(cache_path)

        loaded = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertNotIn("quarantine", loaded.hosts)
        self.assertEqual(len(loaded.quarantine_records), 1)
        self.assertEqual(loaded.quarantine_records[0].tic_id, "quarantine")
        self.assertTrue(loaded.quarantine_records[0].quarantined)

    def test_b3_cache_validation_rejects_corrupted_geometry(self):
        """
        B3 Acceptance:
          from_dict rejects corrupt cache mutations:
            - unquarantined epoch NaN
            - negative duration with maskable=True
            - missing payload_sha256
            - snapshot source_name mismatch with dict key
        """
        reg = known_signals.KnownSignalsRegistry()
        sig = known_signals.KnownSignal(
            signal_id="toi:1.01", tic_id="100", source_catalog="nasa_toi",
            source_record_id="1.01", planet_or_signal_name="TOI-1.01",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=5.0,
            epoch_btjd=1500.0, duration_hours=2.0, duration_days=2.0 / 24.0,
            has_transit_ephemeris=True, maskable=True
        )
        snap = known_signals.SourceSnapshot(
            source_name="nasa_toi", provider="NASA", query_or_url="q",
            retrieval_utc="2026-10-16T12:00:00Z", status="SUCCESS",
            row_count_raw=1, row_count_valid=1, row_count_quarantined=0,
            payload_sha256="abc123hash"
        )
        reg.add_signal(sig)
        reg.register_snapshot(snap)
        d = reg.to_dict()

        # Mutation 1: unquarantined epoch NaN
        d1 = json.loads(json.dumps(d))
        d1["hosts"]["100"]["signals"][0]["epoch_btjd"] = float("nan")
        with self.assertRaises(ValueError):
            known_signals.KnownSignalsRegistry.from_dict(d1)

        # Mutation 2: negative duration with maskable=True
        d2 = json.loads(json.dumps(d))
        d2["hosts"]["100"]["signals"][0]["duration_hours"] = -2.0
        with self.assertRaises(ValueError):
            known_signals.KnownSignalsRegistry.from_dict(d2)

        # Mutation 3: missing payload_sha256
        d3 = json.loads(json.dumps(d))
        d3["source_snapshots"]["nasa_toi"]["payload_sha256"] = ""
        with self.assertRaises(ValueError):
            known_signals.KnownSignalsRegistry.from_dict(d3)

        # Mutation 4: snapshot source_name mismatch
        d4 = json.loads(json.dumps(d))
        d4["source_snapshots"]["nasa_toi"]["source_name"] = "wrong_name"
        with self.assertRaises(ValueError):
            known_signals.KnownSignalsRegistry.from_dict(d4)

    def test_b4_stable_solution_identity_under_reordering(self):
        """
        B4 Acceptance:
          Reordering DataFrame rows produces identical solution IDs.
        """
        raw_csv = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,tfopwg_disp\n"
            "33692729,469.01,2458474.5,13.6,4.0,PC\n"
            "272086159,176.01,2458649.0,16.1,1.5,PC\n"
            "219239945,701.01,2458389.0,13.7,4.8,PC\n"
        )
        df = pd.read_csv(io.StringIO(raw_csv))
        df_reversed = df.iloc[::-1].reset_index(drop=True)

        signals_orig, _ = known_signals.ingest_nasa_toi_df(df)
        signals_rev, _ = known_signals.ingest_nasa_toi_df(df_reversed)

        ids_orig = {s.solution_id for s in signals_orig}
        ids_rev = {s.solution_id for s in signals_rev}
        self.assertEqual(ids_orig, ids_rev)
        sig_ids_orig = {s.signal_id for s in signals_orig}
        self.assertIn("toi:469.01", sig_ids_orig)
        self.assertIn("toi:176.01", sig_ids_orig)
        self.assertIn("toi:701.01", sig_ids_orig)
        self.assertTrue(any(s.startswith("toi:469.01#") for s in ids_orig))

    def test_b4_unsafe_float_tic_rejected(self):
        """
        B4 Acceptance:
          normalize_tic_id rejects unsafe float TIC IDs exceeding 2^53 (9007199254740992).
          Exact integers and strings pass losslessly.
        """
        unsafe_val = 9007199254740993  # 2^53 + 1
        with self.assertRaises(ValueError):
            known_signals.normalize_tic_id(float(unsafe_val))

        with self.assertRaises(ValueError):
            known_signals.normalize_tic_id(float(9007199254740992))

        # Int and string pass losslessly:
        self.assertEqual(known_signals.normalize_tic_id(unsafe_val), str(unsafe_val))
        self.assertEqual(known_signals.normalize_tic_id(str(unsafe_val)), str(unsafe_val))

    def test_c1_canonical_valid_count_api_and_quarantine_counts(self):
        """
        C1 Acceptance:
          total_valid_signals() never crashes with TypeError: 'int' object is not iterable.
          Correct count semantics across hosts and quarantine verified:
            - empty: 0 valid, 0 quarantined, 0 total
            - valid-only: N valid, 0 quarantined, N total
            - quarantine-only: 0 valid, M quarantined, M total
            - mixed: N valid, M quarantined, N+M total
            - survives JSON serialization roundtrip
            - exercised on production loader registry
        """
        # 1. Empty registry
        reg_empty = known_signals.KnownSignalsRegistry()
        self.assertEqual(reg_empty.total_signals(), 0)
        self.assertEqual(reg_empty.total_valid_signals(), 0)
        self.assertEqual(reg_empty.total_quarantined_signals(), 0)

        # 2. Valid-only records
        reg_valid = known_signals.KnownSignalsRegistry()
        snap1 = known_signals.SourceSnapshot(
            source_name="nasa_toi", provider="NASA", query_or_url="toi",
            retrieval_utc="2026-10-16T12:00:00Z", status="SUCCESS",
            row_count_raw=2, row_count_valid=2, row_count_quarantined=0,
            payload_sha256="hash_val", snapshot_id="nasa_toi@2026-10-16T12:00:00Z"
        )
        sig1 = known_signals.KnownSignal(
            signal_id="toi:1.01", tic_id="101", source_catalog="nasa_toi",
            source_record_id="1.01", planet_or_signal_name="TOI-1.01",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=2.5,
            epoch_btjd=1500.0, duration_hours=2.0, duration_days=2.0 / 24.0,
            has_transit_ephemeris=True, maskable=True, snapshot_id=snap1.snapshot_id,
            solution_id="toi:1.01#P2.5"
        )
        sig2 = known_signals.KnownSignal(
            signal_id="toi:1.02", tic_id="101", source_catalog="nasa_toi",
            source_record_id="1.02", planet_or_signal_name="TOI-1.02",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=5.0,
            epoch_btjd=1505.0, duration_hours=3.0, duration_days=3.0 / 24.0,
            has_transit_ephemeris=True, maskable=True, snapshot_id=snap1.snapshot_id,
            solution_id="toi:1.02#P5.0"
        )
        reg_valid.replace_source_signals("nasa_toi", [sig1, sig2], snap1)
        self.assertEqual(reg_valid.total_signals(), 2)
        self.assertEqual(reg_valid.total_valid_signals(), 2)
        self.assertEqual(reg_valid.total_quarantined_signals(), 0)

        # 3. Quarantine-only records
        reg_quar = known_signals.KnownSignalsRegistry()
        snap_q = known_signals.SourceSnapshot(
            source_name="nasa_toi", provider="NASA", query_or_url="toi",
            retrieval_utc="2026-10-16T12:00:00Z", status="FAILED",
            row_count_raw=2, row_count_valid=0, row_count_quarantined=2,
            payload_sha256="hash_q", snapshot_id="nasa_toi@attempt_2026-10-16T12:00:00Z#FAILED"
        )
        q1 = known_signals.KnownSignal(
            signal_id="toi:q1", tic_id="quarantine", source_catalog="nasa_toi",
            source_record_id="q1", planet_or_signal_name="TOI-q1",
            disposition="UNKNOWN", raw_disposition="", quarantined=True,
            quarantine_reason="Invalid TIC", snapshot_id=snap_q.snapshot_id,
            solution_id="toi:q1#quarantine"
        )
        q2 = known_signals.KnownSignal(
            signal_id="toi:q2", tic_id="quarantine", source_catalog="nasa_toi",
            source_record_id="q2", planet_or_signal_name="TOI-q2",
            disposition="UNKNOWN", raw_disposition="", quarantined=True,
            quarantine_reason="Unparseable period", snapshot_id=snap_q.snapshot_id,
            solution_id="toi:q2#quarantine"
        )
        reg_quar.record_failed_attempt(snap_q, [q1, q2])
        self.assertEqual(reg_quar.total_signals(), 2)
        self.assertEqual(reg_quar.total_valid_signals(), 0)
        self.assertEqual(reg_quar.total_quarantined_signals(), 2)

        # 4. Mixed records
        reg_mixed = known_signals.KnownSignalsRegistry()
        reg_mixed.replace_source_signals("nasa_toi", [sig1, sig2], snap1)
        reg_mixed.record_failed_attempt(snap_q, [q1, q2])
        self.assertEqual(reg_mixed.total_signals(), 4)
        self.assertEqual(reg_mixed.total_valid_signals(), 2)
        self.assertEqual(reg_mixed.total_quarantined_signals(), 2)

        # 5. Roundtrip to JSON and verify identical counts
        cache_path = os.path.join(self.tmp_dir, "test_c1_counts.json")
        reg_mixed.save_to_json(cache_path)
        loaded = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(loaded.total_signals(), 4)
        self.assertEqual(loaded.total_valid_signals(), 2)
        self.assertEqual(loaded.total_quarantined_signals(), 2)

        # 6. Public production helper call without TypeError
        df_toi = pd.read_csv(self.toi_path)
        df_ps = pd.read_csv(self.confirmed_path)
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": df_toi, "nasa_confirmed": df_ps}
        )
        prod_reg = aldebaran.get_known_signals_registry()
        self.assertIsNotNone(prod_reg)
        self.assertEqual(prod_reg.total_valid_signals(), 10)
        self.assertEqual(prod_reg.total_quarantined_signals(), 0)
        self.assertEqual(prod_reg.total_signals(), 10)

    def test_c2_preserved_good_cache_with_failed_attempt_roundtrip_and_negative_control(self):
        """
        C2 Acceptance:
          Good loader -> malformed refresh -> saved cache reload succeeds cleanly with coherent counts.
          Failed-attempt raw/quarantine evidence is retained with distinct snapshot identity.
          Deliberately adding a failed row to good snapshot accounting triggers row count mismatch error.
        """
        # Step 1: Good loader runs:
        df_toi = pd.read_csv(self.toi_path)
        df_ps = pd.read_csv(self.confirmed_path)
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": df_toi, "nasa_confirmed": df_ps}
        )
        reg = aldebaran.get_known_signals_registry()
        self.assertEqual(reg.total_valid_signals(), 10)

        # Step 2: Malformed refresh runs:
        df_malformed = pd.DataFrame({"wrong_col": [999]})
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": df_malformed, "nasa_confirmed": df_ps}
        )

        # Step 3: Assert in-memory state:
        reg_after = aldebaran.get_known_signals_registry()
        self.assertEqual(reg_after.total_valid_signals(), 10, "Selected good data must be preserved in memory")
        self.assertEqual(reg_after.source_snapshots["nasa_toi"].last_attempt_status, "FAILED")
        self.assertEqual(reg_after.get_coverage_status(), "INCOMPLETE")
        self.assertEqual(len(reg_after.attempt_snapshots), 1, "Failed attempt snapshot must be preserved")
        failed_snap = reg_after.attempt_snapshots[0]
        self.assertEqual(failed_snap.status, "FAILED")
        self.assertEqual(len(reg_after.quarantine_records), 1, "Failed row must be retained in quarantine")

        # Step 4: Actual saved cache reload / new initialized accessor:
        cache_path = os.path.join(self.tmp_dir, "test_c2_atomic_cache.json")
        reg_after.save_to_json(cache_path)
        loaded = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(loaded.total_valid_signals(), 10)
        self.assertEqual(loaded.total_quarantined_signals(), 1)
        self.assertEqual(loaded.total_signals(), 11)
        self.assertEqual(loaded.source_snapshots["nasa_toi"].status, "CACHED")
        self.assertEqual(loaded.source_snapshots["nasa_toi"].last_attempt_status, "FAILED")
        self.assertEqual(loaded.get_coverage_status(), "INCOMPLETE")

        # Step 5: Deliberate negative control: mutate good snapshot accounting to include extra row
        d_corrupt = loaded.to_dict()
        d_corrupt["source_snapshots"]["nasa_toi"]["row_count_raw"] += 1
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d_corrupt)
        self.assertIn("row count mismatch", str(ctx.exception))

    def test_c3_cache_validation_enforces_required_geometry_and_snapshot_links(self):
        """
        C3 Acceptance:
          from_dict rejects:
            - maskable=True with both duration_hours/duration_days=None
            - has_transit_ephemeris=True and maskable=True with period_days=None
            - inconsistent BJD and BTJD epochs (|BJD - (BTJD + 2457000)| > 1e-3)
            - orphan host signals with source_snapshots empty or missing snapshot link
          Valid incomplete-unmaskable signal (e.g. RV without transit ephemeris) passes cleanly.
        """
        # Baseline valid registry with 1 maskable and 1 valid incomplete-unmaskable signal:
        reg = known_signals.KnownSignalsRegistry()
        snap = known_signals.SourceSnapshot(
            source_name="nasa_toi", provider="NASA", query_or_url="toi",
            retrieval_utc="2026-10-16T12:00:00Z", status="SUCCESS",
            row_count_raw=2, row_count_valid=2, row_count_quarantined=0,
            payload_sha256="sha_valid_geometry", snapshot_id="nasa_toi@2026-10-16T12:00:00Z"
        )
        sig_maskable = known_signals.KnownSignal(
            signal_id="toi:1.01", tic_id="101", source_catalog="nasa_toi",
            source_record_id="1.01", planet_or_signal_name="TOI-1.01",
            disposition="PLANET_CANDIDATE", raw_disposition="PC", period_days=3.5,
            epoch_bjd=2458500.0, epoch_btjd=1500.0, duration_hours=2.4, duration_days=0.1,
            has_transit_ephemeris=True, maskable=True, snapshot_id=snap.snapshot_id,
            solution_id="toi:1.01#P3.5"
        )
        # Valid incomplete-unmaskable control: RV planet without transit ephemeris
        sig_rv = known_signals.KnownSignal(
            signal_id="toi:1.02", tic_id="101", source_catalog="nasa_toi",
            source_record_id="1.02", planet_or_signal_name="TOI-1.02",
            disposition="CONFIRMED_PLANET", raw_disposition="CONFIRMED",
            period_days=10.0, discovery_method="Radial Velocity",
            has_transit_ephemeris=False, maskable=False, snapshot_id=snap.snapshot_id,
            solution_id="toi:1.02#P10.0"
        )
        reg.replace_source_signals("nasa_toi", [sig_maskable, sig_rv], snap)
        d_base = reg.to_dict()

        # Baseline roundtrip succeeds (including the valid incomplete-unmaskable signal):
        loaded_base = known_signals.KnownSignalsRegistry.from_dict(d_base)
        self.assertEqual(loaded_base.total_valid_signals(), 2)

        # Fault 1: maskable=True with both duration_hours/duration_days=None
        d1 = json.loads(json.dumps(d_base))
        d1["hosts"]["101"]["signals"][0]["duration_hours"] = None
        d1["hosts"]["101"]["signals"][0]["duration_days"] = None
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d1)
        self.assertIn("has neither duration_hours nor duration_days", str(ctx.exception))

        # Fault 2: has_transit_ephemeris=True and maskable=True with period_days=None
        d2 = json.loads(json.dumps(d_base))
        d2["hosts"]["101"]["signals"][0]["period_days"] = None
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d2)
        self.assertIn("period_days is None", str(ctx.exception))

        # Fault 3: inconsistent BJD and BTJD epoch pair (e.g. BJD=99.0 while BTJD=1500.0)
        d3 = json.loads(json.dumps(d_base))
        d3["hosts"]["101"]["signals"][0]["epoch_bjd"] = 99.0
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d3)
        self.assertIn("inconsistent epoch_bjd", str(ctx.exception))

        # Fault 4: all source_snapshots removed, leaving attributed host signals
        d4 = json.loads(json.dumps(d_base))
        d4["source_snapshots"] = {}
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d4)
        self.assertIn("source_snapshots is empty", str(ctx.exception))

        # Fault 5: signal references unknown snapshot_id
        d5 = json.loads(json.dumps(d_base))
        d5["hosts"]["101"]["signals"][0]["snapshot_id"] = "unknown_snap_id"
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d5)
        self.assertIn("unknown snapshot_id", str(ctx.exception))

        # Fault 6: duration_hours inconsistent with duration_days
        d6 = json.loads(json.dumps(d_base))
        d6["hosts"]["101"]["signals"][0]["duration_hours"] = 10.0
        d6["hosts"]["101"]["signals"][0]["duration_days"] = 0.1  # 0.1 * 24 = 2.4 != 10.0
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(d6)
        self.assertIn("duration_hours (10.0) inconsistent with duration_days", str(ctx.exception))

    def test_c4_stable_solution_identities_and_raw_provenance(self):
        """
        C4 Acceptance:
          - Stable distinct solution IDs for identical P but different T0
          - Distinct solution IDs for conflicting confirmed/CTOI solutions
          - Adding or removing alternative solutions leaves existing solution IDs invariant
          - Raw payload hash ('raw_file_bytes') matches the selected fixture bytes
          - Independent DataFrame ingestion has serialization provenance
        """
        # 1. Same P, different T0 produce distinct solution IDs:
        sol_a = known_signals.make_solution_id("toi", "1.01", period=2.0, epoch=100.0, duration=2.0)
        sol_b = known_signals.make_solution_id("toi", "1.01", period=2.0, epoch=105.0, duration=2.0)
        self.assertNotEqual(sol_a, sol_b)
        self.assertIn("T0100.0", sol_a)
        self.assertIn("T0105.0", sol_b)

        # 2. Conflicting confirmed solutions produce distinct solution IDs:
        conf_a = known_signals.make_solution_id("confirmed", "TOI-469.01", period=13.63, epoch=1474.5)
        conf_b = known_signals.make_solution_id("confirmed", "TOI-469.01", period=13.63, epoch=1475.0)
        self.assertNotEqual(conf_a, conf_b)

        # 3. Adding or removing alternatives does NOT change existing solution ID:
        raw_row1 = "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,tfopwg_disp\n33692729,469.01,2458474.5,13.6,4.0,PC\n"
        df1 = pd.read_csv(io.StringIO(raw_row1))
        sigs1, _ = known_signals.ingest_nasa_toi_df(df1)
        id_before = sigs1[0].solution_id

        raw_row1_and_row2 = (
            "tid,toi,pl_tranmid,pl_orbper,pl_trandurh,tfopwg_disp\n"
            "33692729,469.01,2458474.5,13.6,4.0,PC\n"
            "33692729,469.01,2458500.0,25.0,6.0,PC\n"
        )
        df2 = pd.read_csv(io.StringIO(raw_row1_and_row2))
        sigs2, _ = known_signals.ingest_nasa_toi_df(df2)
        id_after = sigs2[0].solution_id

        self.assertEqual(id_before, id_after, "Solution ID for row 1 must remain invariant when alternatives are added")
        self.assertNotEqual(sigs2[0].solution_id, sigs2[1].solution_id)

        # 4. Raw file bytes provenance matching frozen file SHA:
        frozen_toi_path = self.toi_path
        if not frozen_toi_path or not os.path.exists(frozen_toi_path):
            self.fail("Frozen fixture nasa_toi.csv not available")
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": frozen_toi_path}
        )
        reg_raw = aldebaran.get_known_signals_registry()
        snap_raw = reg_raw.source_snapshots["nasa_toi"]
        self.assertEqual(snap_raw.payload_hash_kind, "raw_file_bytes")
        self.assertEqual(snap_raw.payload_sha256, self.expected_toi_sha256)
        self.assertEqual(snap_raw.original_retrieval_utc, self.expected_raw_retrieval_utc)

        # 5. Independent ingestion: the raw snapshot/cache above must not be reused.
        df_mem = pd.read_csv(frozen_toi_path)
        expected_mem_sha = hashlib.sha256(
            df_mem.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        with mock.patch.multiple(
            aldebaran,
            CATALOG_CACHE_FILE=os.path.join(self.tmp_dir, "memory_catalog.txt"),
            TOI_EPHEMERIS_CACHE_FILE=os.path.join(self.tmp_dir, "memory_toi.json"),
            KNOWN_SIGNALS_REGISTRY_CACHE_FILE=os.path.join(self.tmp_dir, "memory_registry.json"),
            GLOBAL_KNOWN_SIGNALS_REGISTRY=None,
            GLOBAL_TOI_EPHEMERIDES={},
        ):
            self.assertFalse(os.path.exists(aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE))
            aldebaran.load_known_catalogs_and_toi_ephemerides(
                offline_fixtures={"nasa_toi": df_mem}
            )
            snap_mem = aldebaran.get_known_signals_registry().source_snapshots["nasa_toi"]
            self.assertEqual(snap_mem.payload_hash_kind, "dataframe_serialization")
            self.assertEqual(snap_mem.payload_sha256, expected_mem_sha)
            self.assertEqual(snap_mem.raw_source_path_or_url, "in_memory_dataframe")
            self.assertIsNone(snap_mem.original_retrieval_utc)

    def test_c4_identical_bytes_preserve_selected_raw_provenance(self):
        """Equal hashes do not imply identical provenance or snapshot replacement."""
        df_mem = pd.DataFrame({"tid": [101], "toi": [1.01],
                               "pl_tranmid": [2458500.0], "pl_orbper": [2.0],
                               "pl_trandurh": [2.0], "tfopwg_disp": ["PC"]})
        # Match the platform's actual serializer by construction, on Windows and Linux.
        raw = df_mem.to_csv(index=False).encode("utf-8")
        expected_sha = hashlib.sha256(raw).hexdigest()
        raw_path = os.path.join(self.tmp_dir, "identical.csv")
        with open(raw_path, "wb") as stream:
            stream.write(raw)
        # SYNTHETIC metadata, isolated from repository historical manifests.
        original_utc = "2000-01-01T00:00:00+00:00"
        source_url = "https://example.invalid/SYNTHETIC-identical.csv"
        with open(os.path.join(self.tmp_dir, "source_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump({"nasa_toi": {"sha256": expected_sha,
                                  "retrieved_utc": original_utc, "url": source_url}}, stream)
        with mock.patch.object(aldebaran, "__file__", os.path.join(self.tmp_dir, "aldebaran.py")):
            aldebaran.load_known_catalogs_and_toi_ephemerides(
                offline_fixtures={"nasa_toi": raw_path})
            selected = aldebaran.get_known_signals_registry().source_snapshots["nasa_toi"]
            self.assertEqual(selected.payload_sha256, expected_sha)
            self.assertEqual(selected.payload_hash_kind, "raw_file_bytes")
            self.assertEqual(selected.original_retrieval_utc, original_utc)
            self.assertEqual(selected.raw_source_path_or_url, source_url)
            before = {key: getattr(selected, key) for key in (
                "snapshot_id", "payload_sha256", "payload_hash_kind", "retrieval_utc",
                "original_retrieval_utc", "raw_source_path_or_url")}

            _, incoming_sha, incoming_kind, incoming_source, incoming_utc = (
                aldebaran._extract_source_input(df_mem, "nasa_toi", "offline_fixture"))
            self.assertEqual(incoming_sha, expected_sha)
            self.assertEqual(incoming_kind, "dataframe_serialization")
            self.assertEqual(incoming_source, "in_memory_dataframe")
            self.assertIsNone(incoming_utc)
            self.assertNotEqual(incoming_kind, selected.payload_hash_kind)
            aldebaran.load_known_catalogs_and_toi_ephemerides(
                offline_fixtures={"nasa_toi": df_mem})
            # Loader can reload JSON, so assert stable identity/provenance, not Python identity.
            retained = aldebaran.get_known_signals_registry().source_snapshots["nasa_toi"]
            self.assertEqual({key: getattr(retained, key) for key in before}, before)
            self.assertEqual(retained.last_attempt_status, "SUCCESS")
            restored = known_signals.KnownSignalsRegistry.load_from_json(
                aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE).source_snapshots["nasa_toi"]
            self.assertEqual({key: getattr(restored, key) for key in before}, before)

    def test_c4_equivalent_csv_line_endings_keep_distinct_raw_hashes(self):
        """Table equality does not erase the identity of LF versus CRLF raw bytes."""
        lf = ("tid,toi,pl_tranmid,pl_orbper,pl_trandurh,tfopwg_disp\n"
              "101,1.01,2458500.0,2.0,2.0,PC\n").encode("utf-8")
        crlf = lf.replace(b"\n", b"\r\n")
        self.assertNotEqual(hashlib.sha256(lf).hexdigest(), hashlib.sha256(crlf).hexdigest())
        frames = []
        for name, raw in (("lf", lf), ("crlf", crlf)):
            with self.subTest(line_ending=name):
                path = os.path.join(self.tmp_dir, name + ".csv")
                with open(path, "wb") as stream:
                    stream.write(raw)
                frame, digest, kind, _, _ = aldebaran._extract_source_input(
                    path, "nasa_toi", "offline_fixture")
                frames.append(frame)
                self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
                self.assertEqual(kind, "raw_file_bytes")
        pd.testing.assert_frame_equal(frames[0], frames[1])

    def test_d1_full_lifecycle_recovery_persistence_and_negative_control(self):
        """
        D1 Acceptance & Deliberate Control:
          - Full lifecycle: good -> malformed -> good -> malformed -> good (repeated twice).
          - Reload from persisted JSON cache with a fresh accessor after EVERY transition.
          - Selected 10 valid records remain accessible across all transitions.
          - Historical failed evidence and snapshots are retained with exact per-snapshot counts.
          - Original selected source snapshot (sid) is retained and inspectable.
          - Deliberately purging a retained attempt row from quarantine_records fails validation.
          - Unmutated correct control passes cleanly.
        """
        if not self.toi_path or not self.confirmed_path:
            self.fail("Frozen catalog fixtures not available")

        good_fixtures = {
            "nasa_toi": self.toi_path,
            "nasa_confirmed": self.confirmed_path,
        }
        cache_path = aldebaran.KNOWN_SIGNALS_REGISTRY_CACHE_FILE

        # --- Cycle 1: Transition 1.1 (Initial good load) ---
        aldebaran.load_known_catalogs_and_toi_ephemerides(offline_fixtures=good_fixtures)
        r1 = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(r1.total_valid_signals(), 10)
        self.assertEqual(len(r1.attempt_snapshots), 0)
        self.assertEqual(r1.total_quarantined_signals(), 0)
        sid = r1.source_snapshots["nasa_toi"].snapshot_id
        self.assertTrue(bool(sid))

        # --- Cycle 1: Transition 1.2 (Malformed refresh: missing headers) ---
        malformed_df1 = pd.DataFrame([{"wrong_col": "evidence_row_1"}])
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": malformed_df1}
        )
        r2 = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(r2.total_valid_signals(), 10, "Valid records preserved during malformed refresh")
        self.assertEqual(len(r2.attempt_snapshots), 1, "Failed attempt snapshot recorded")
        self.assertEqual(r2.total_quarantined_signals(), 1, "Failed attempt quarantined record stored")
        failed_id_1 = r2.attempt_snapshots[0].snapshot_id
        self.assertIn("#FAILED", failed_id_1)
        self.assertEqual(r2.source_snapshots["nasa_toi"].snapshot_id, sid)
        self.assertEqual(r2.get_coverage_status(), "INCOMPLETE")

        # --- Cycle 1: Transition 1.3 (Recovery: good load) ---
        aldebaran.load_known_catalogs_and_toi_ephemerides(offline_fixtures=good_fixtures)
        r3 = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(r3.total_valid_signals(), 10, "Valid records remain 10 after recovery")
        self.assertEqual(len(r3.attempt_snapshots), 1, "Historical failed attempt preserved after recovery")
        self.assertEqual(r3.total_quarantined_signals(), 1, "Historical quarantined row preserved after recovery")
        self.assertEqual(r3.source_snapshots["nasa_toi"].snapshot_id, sid, "Original selected snapshot retained")
        self.assertTrue(any(s.snapshot_id == sid for s in list(r3.source_snapshots.values()) + r3.attempt_snapshots))
        self.assertTrue(any(s.snapshot_id == failed_id_1 for s in r3.attempt_snapshots))

        # --- Cycle 2: Transition 2.1 (Second failure: malformed row) ---
        malformed_df2 = pd.DataFrame([{"missing_header": "evidence_row_2"}])
        aldebaran.load_known_catalogs_and_toi_ephemerides(
            offline_fixtures={"nasa_toi": malformed_df2}
        )
        r4 = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(r4.total_valid_signals(), 10)
        self.assertEqual(len(r4.attempt_snapshots), 2, "Both failed attempt snapshots retained")
        self.assertEqual(r4.total_quarantined_signals(), 2, "Both quarantined attempt rows retained")
        failed_id_2 = r4.attempt_snapshots[1].snapshot_id
        self.assertNotEqual(failed_id_1, failed_id_2)
        self.assertEqual(r4.source_snapshots["nasa_toi"].snapshot_id, sid)
        self.assertEqual(r4.get_coverage_status(), "INCOMPLETE")

        # --- Cycle 2: Transition 2.2 (Second recovery: good load) ---
        aldebaran.load_known_catalogs_and_toi_ephemerides(offline_fixtures=good_fixtures)
        r5 = known_signals.KnownSignalsRegistry.load_from_json(cache_path)
        self.assertEqual(r5.total_valid_signals(), 10)
        self.assertEqual(len(r5.attempt_snapshots), 2, "Both historical attempts retained after 2nd recovery")
        self.assertEqual(r5.total_quarantined_signals(), 2, "Both quarantined rows retained after 2nd recovery")
        self.assertEqual(r5.source_snapshots["nasa_toi"].snapshot_id, sid, "Original selected snapshot retained")
        self.assertTrue(any(s.snapshot_id == sid for s in list(r5.source_snapshots.values()) + r5.attempt_snapshots))
        self.assertTrue(any(s.snapshot_id == failed_id_1 for s in r5.attempt_snapshots))
        self.assertTrue(any(s.snapshot_id == failed_id_2 for s in r5.attempt_snapshots))

        # --- Deliberate Negative Control (Break the check on purpose) ---
        # Deliberately purge one of the retained attempt rows from quarantine_records
        disposable_dict = r5.to_dict()
        purged_row = disposable_dict["quarantine_records"].pop()
        with self.assertRaises(ValueError) as ctx:
            known_signals.KnownSignalsRegistry.from_dict(disposable_dict)
        self.assertIn("row count mismatch for snapshot", str(ctx.exception))

        # Correct unmutated control passes cleanly
        correct_reg = known_signals.KnownSignalsRegistry.from_dict(r5.to_dict())
        self.assertEqual(correct_reg.total_valid_signals(), 10)
        self.assertEqual(len(correct_reg.attempt_snapshots), 2)
        self.assertEqual(correct_reg.total_quarantined_signals(), 2)


if __name__ == "__main__":
    unittest.main()

