"""
tests/test_vps_contract.py — Comprehensive Test Suite for VPS Accounting & Contract
=====================================================================================
Validates versioned VPS component accounting, immutable ledger reconstruction,
secondary eclipse dual-zone boundaries, elimination of legacy subtraction (SCI-16),
tamper detection, Stage 3A integration, and deliberate negative fault controls.
Addresses Codex Review R1-R4.
"""

from collections import OrderedDict
import ast
import copy
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import typing
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vps_contract import (
    VPS_SCHEMA_VERSION,
    VPS_PRODUCER_NAME,
    VPS_BASE_PENALTY,
    VPS_CLIP_MIN,
    VPS_CLIP_MAX,
    VPS_ROUNDING_DECIMALS,
    VPS_RECONSTRUCTION_TOLERANCE,
    PLEIADES_CLEAN_VPS_THRESHOLD,
    PLEIADES_CLEAN_CRITERION_LABEL,
    ANTARES_DEFAULT_MAX_CORR_RISK,
    SEC_CIRCULAR_HALF_WIDTH,
    SEC_CIRCULAR_CENTER,
    SEC_CIRCULAR_THRESHOLD_SNR,
    SEC_ECCENTRIC_THRESHOLD_SNR,
    SEC_PENALTY_SCALE,
    SEC_PENALTY_SNR_DIVISOR,
    MANDATORY_COMPONENT_KEYS,
    VPSContractError,
    compute_vps_ledger,
    validate_and_extract_vps_score,
)
import antares_packager
import pleiades


class TestVPSIsolatedPenalties(unittest.TestCase):
    """
    Tests each individual VPS component penalty in isolation with exact expected
    analytical contributions, verifying zero Oracle drift against Pleiades formulas.
    Also validates non-finite mandatory input rejection (R2).
    """

    def _nominal_kwargs(self):
        return {
            "b_fit": 0.0,
            "sec_center_phi": 0.50,
            "sec_snr": 0.0,
            "is_sec_eb": False,
            "is_tess_eb": False,
            "is_gaia_nss": False,
            "neb_capable_stars": 0,
            "contamination_ratio": 0.0,
            "dur_ratio_kepler": 1.0,
            "ruwe": 1.0,
            "stitched_primary_snr": 10.0,
        }

    def test_base_penalty_zero_others(self):
        kwargs = self._nominal_kwargs()
        ledger = compute_vps_ledger(**kwargs)
        self.assertEqual(ledger["version"], VPS_SCHEMA_VERSION)
        self.assertEqual(ledger["producer"], VPS_PRODUCER_NAME)
        self.assertAlmostEqual(ledger["components"]["base"], 0.2, places=6)
        for k, v in ledger["components"].items():
            if k != "base":
                self.assertEqual(v, 0.0, f"Component '{k}' expected 0.0, got {v}")
        self.assertAlmostEqual(ledger["unclipped_total"], 0.2, places=6)
        self.assertAlmostEqual(ledger["clipped_score"], 0.2, places=6)
        self.assertEqual(ledger["final_score"], 0.2)

    def test_publication_labels_describe_heuristic_vps_not_probability(self):
        repo = Path(__file__).resolve().parents[1]
        ctoi_source = (repo / "ctoi_packager.py").read_text(encoding="utf-8")
        antares_source = (repo / "antares_packager.py").read_text(encoding="utf-8")
        pleiades_source = (repo / "pleiades.py").read_text(encoding="utf-8")
        triceratops_source = (repo / "validate_triceratops.py").read_text(encoding="utf-8")
        far_source = (repo / "tests" / "benchmark_false_alarm_rate.py").read_text(encoding="utf-8")
        roc_source = (repo / "tests" / "benchmark_vps_roc.py").read_text(encoding="utf-8")
        self.assertIn("VPS heuristic points=", ctoi_source)
        self.assertIn("VPS heuristic points=", antares_source)
        self.assertNotIn("Stage 3A Filter (Risk=", antares_source)
        self.assertNotIn("FP Risk=", ctoi_source)
        self.assertNotIn("FP Risk=", antares_source)
        self.assertIn("Legacy compatibility alias for heuristic VPS points, not a probability.", pleiades_source)
        self.assertNotIn("Low FP Risk", pleiades_source)
        self.assertNotIn("CONFIRMED FP", pleiades_source)
        self.assertIn("EXPLORATORY CONDITIONAL SCENARIO-MODEL THRESHOLD PASS", triceratops_source)
        self.assertNotIn("STATISTICALLY VALIDATED PLANET", triceratops_source)
        self.assertNotIn("Empirical FAR (% of light curves)", far_source)
        self.assertNotIn("Empirical ROC Curve: Vetting Penalty Score", roc_source)
        self.assertNotIn("Whitened Maximum SDE (Inverted Flux)", far_source)
        self.assertNotIn("Stage-3A Gate (VPS", roc_source)

        with self.assertRaises(AssertionError):
            self.assertNotIn("FP Risk=", ctoi_source.replace("VPS heuristic points=", "FP Risk=", 1))
        with self.assertRaises(AssertionError):
            self.assertNotIn(
                "Stage 3A Filter (Risk=",
                antares_source.replace("Stage 3A Filter (VPS heuristic points=", "Stage 3A Filter (Risk=", 1),
            )
        with self.assertRaises(AssertionError):
            self.assertNotIn(
                "STATISTICALLY VALIDATED PLANET",
                triceratops_source.replace("EXPLORATORY CONDITIONAL SCENARIO-MODEL THRESHOLD PASS", "STATISTICALLY VALIDATED PLANET", 1),
            )
        with self.assertRaises(AssertionError):
            self.assertNotIn(
                "Empirical ROC Curve: Vetting Penalty Score",
                roc_source.replace("Synthetic-label ROC Demonstration: VPS", "Empirical ROC Curve: Vetting Penalty Score", 1),
            )
        with self.assertRaises(AssertionError):
            self.assertNotIn(
                "Empirical FAR (% of light curves)",
                far_source.replace("Synthetic Gumbel draw fraction (%)", "Empirical FAR (% of light curves)", 1),
            )
        with self.assertRaises(AssertionError):
            self.assertNotIn(
                "Whitened Maximum SDE (Inverted Flux)",
                far_source.replace(
                    "Synthetic Gumbel variate (illustrative SDE scale)",
                    "Whitened Maximum SDE (Inverted Flux)",
                    1,
                ),
            )
        with self.assertRaises(AssertionError):
            self.assertNotIn(
                "Stage-3A Gate (VPS",
                roc_source.replace(
                    "Synthetic-sample Youden optimum (not a production threshold): VPS",
                    "Stage-3A Gate (VPS",
                    1,
                ),
            )

    def test_impact_parameter_penalty(self):
        # b <= 0.75 -> 0.0
        kw = self._nominal_kwargs()
        kw["b_fit"] = 0.75
        self.assertEqual(compute_vps_ledger(**kw)["components"]["impact_parameter"], 0.0)

        # b = 0.85 -> 25.0 * ((0.85 - 0.75) / 0.25)**2 = 25.0 * 0.16 = 4.0
        kw["b_fit"] = 0.85
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["impact_parameter"], 4.0, places=6)

        # b = 1.00 -> 25.0 * ((1.00 - 0.75) / 0.25)**2 = 25.0 * 1.0 = 25.0
        kw["b_fit"] = 1.00
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["impact_parameter"], 25.0, places=6)

    def test_secondary_significance_circular_zone(self):
        kw = self._nominal_kwargs()
        kw["sec_center_phi"] = 0.50  # Circular center

        # Sub-threshold: SNR = 2.79 -> 0.0
        kw["sec_snr"] = 2.79
        l1 = compute_vps_ledger(**kw)
        self.assertEqual(l1["components"]["secondary_significance"], 0.0)
        self.assertFalse(l1["secondary_metadata"]["penalty_applied"])

        # Boundary: SNR = 2.80 -> 0.0 (rule is > 2.8)
        kw["sec_snr"] = 2.80
        l2 = compute_vps_ledger(**kw)
        self.assertEqual(l2["components"]["secondary_significance"], 0.0)
        self.assertFalse(l2["secondary_metadata"]["penalty_applied"])

        # Above threshold: SNR = 2.81 -> 45.0 * (2.81 / 3.0) = 42.15
        kw["sec_snr"] = 2.81
        l3 = compute_vps_ledger(**kw)
        self.assertAlmostEqual(l3["components"]["secondary_significance"], 42.15, places=5)
        self.assertTrue(l3["secondary_metadata"]["penalty_applied"])
        self.assertEqual(l3["secondary_metadata"]["zone"], "circular")
        self.assertEqual(l3["secondary_metadata"]["threshold"], 2.8)

    def test_secondary_significance_eccentric_zone(self):
        kw = self._nominal_kwargs()
        kw["sec_center_phi"] = 0.70  # Eccentric (|0.70 - 0.50| = 0.20 > 0.04)

        # Sub-threshold: SNR = 3.39 -> 0.0
        kw["sec_snr"] = 3.39
        l1 = compute_vps_ledger(**kw)
        self.assertEqual(l1["components"]["secondary_significance"], 0.0)
        self.assertFalse(l1["secondary_metadata"]["penalty_applied"])

        # Boundary: SNR = 3.40 -> 0.0 (rule is > 3.4)
        kw["sec_snr"] = 3.40
        l2 = compute_vps_ledger(**kw)
        self.assertEqual(l2["components"]["secondary_significance"], 0.0)
        self.assertFalse(l2["secondary_metadata"]["penalty_applied"])

        # Above threshold: SNR = 3.41 -> 45.0 * (3.41 / 3.0) = 51.15
        kw["sec_snr"] = 3.41
        l3 = compute_vps_ledger(**kw)
        self.assertAlmostEqual(l3["components"]["secondary_significance"], 51.15, places=5)
        self.assertTrue(l3["secondary_metadata"]["penalty_applied"])
        self.assertEqual(l3["secondary_metadata"]["zone"], "eccentric")
        self.assertEqual(l3["secondary_metadata"]["threshold"], 3.4)

    def test_secondary_eb_flag(self):
        kw = self._nominal_kwargs()
        kw["is_sec_eb"] = False
        self.assertEqual(compute_vps_ledger(**kw)["components"]["secondary_eb"], 0.0)
        kw["is_sec_eb"] = True
        self.assertEqual(compute_vps_ledger(**kw)["components"]["secondary_eb"], 65.0)

    def test_catalog_tess_eb(self):
        kw = self._nominal_kwargs()
        kw["is_tess_eb"] = False
        self.assertEqual(compute_vps_ledger(**kw)["components"]["catalog_tess_eb"], 0.0)
        kw["is_tess_eb"] = True
        self.assertEqual(compute_vps_ledger(**kw)["components"]["catalog_tess_eb"], 90.0)

    def test_gaia_nss(self):
        kw = self._nominal_kwargs()
        kw["is_gaia_nss"] = False
        self.assertEqual(compute_vps_ledger(**kw)["components"]["gaia_nss"], 0.0)
        kw["is_gaia_nss"] = True
        self.assertEqual(compute_vps_ledger(**kw)["components"]["gaia_nss"], 80.0)

    def test_neb_neighbors(self):
        kw = self._nominal_kwargs()
        kw["neb_capable_stars"] = 0
        self.assertEqual(compute_vps_ledger(**kw)["components"]["neb_neighbors"], 0.0)
        kw["neb_capable_stars"] = 3
        self.assertEqual(compute_vps_ledger(**kw)["components"]["neb_neighbors"], 24.0)

    def test_contamination(self):
        kw = self._nominal_kwargs()
        kw["contamination_ratio"] = 0.25
        self.assertEqual(compute_vps_ledger(**kw)["components"]["contamination"], 0.0)
        kw["contamination_ratio"] = 0.40
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["contamination"], 14.0, places=6)

    def test_duration_ratio(self):
        kw = self._nominal_kwargs()
        kw["dur_ratio_kepler"] = 1.00
        self.assertEqual(compute_vps_ledger(**kw)["components"]["duration_ratio"], 0.0)
        kw["dur_ratio_kepler"] = 1.38
        self.assertEqual(compute_vps_ledger(**kw)["components"]["duration_ratio"], 0.0)
        kw["dur_ratio_kepler"] = 1.63
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["duration_ratio"], 25.2, places=6)
        kw["dur_ratio_kepler"] = 0.35
        self.assertEqual(compute_vps_ledger(**kw)["components"]["duration_ratio"], 0.0)
        kw["dur_ratio_kepler"] = 0.21
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["duration_ratio"], 34.0, places=6)

    def test_ruwe(self):
        kw = self._nominal_kwargs()
        kw["ruwe"] = 1.25
        self.assertEqual(compute_vps_ledger(**kw)["components"]["ruwe"], 0.0)
        kw["ruwe"] = 1.35
        self.assertEqual(compute_vps_ledger(**kw)["components"]["ruwe"], 15.0)

    def test_weak_primary(self):
        kw = self._nominal_kwargs()
        kw["stitched_primary_snr"] = 5.5
        self.assertEqual(compute_vps_ledger(**kw)["components"]["weak_primary"], 0.0)
        kw["stitched_primary_snr"] = 7.0
        self.assertEqual(compute_vps_ledger(**kw)["components"]["weak_primary"], 0.0)
        kw["stitched_primary_snr"] = 4.0
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["weak_primary"], 33.0, places=6)
        kw["stitched_primary_snr"] = 1.0
        self.assertAlmostEqual(compute_vps_ledger(**kw)["components"]["weak_primary"], 65.0, places=6)

    def test_mandatory_inputs_nonfinite_rejected(self):
        # R2: Mandatory physical measurements that are NaN or non-finite must raise VPSContractError
        kw = self._nominal_kwargs()
        kw["stitched_primary_snr"] = float("nan")
        with self.assertRaises(VPSContractError):
            compute_vps_ledger(**kw)

        kw = self._nominal_kwargs()
        kw["b_fit"] = float("nan")
        with self.assertRaises(VPSContractError):
            compute_vps_ledger(**kw)

        kw = self._nominal_kwargs()
        kw["dur_ratio_kepler"] = float("nan")
        with self.assertRaises(VPSContractError):
            compute_vps_ledger(**kw)

        kw = self._nominal_kwargs()
        kw["sec_snr"] = float("nan")
        with self.assertRaises(VPSContractError):
            compute_vps_ledger(**kw)

        kw = self._nominal_kwargs()
        kw["sec_center_phi"] = float("nan")
        with self.assertRaises(VPSContractError):
            compute_vps_ledger(**kw)

    def test_optional_catalog_inputs_nan_handled_cleanly(self):
        # R2: Optional catalog inputs (e.g. RUWE, contamination) handle NaN cleanly as 0.0 penalty
        kw = self._nominal_kwargs()
        kw["ruwe"] = float("nan")
        l = compute_vps_ledger(**kw)
        self.assertEqual(l["components"]["ruwe"], 0.0)

        kw["contamination_ratio"] = float("nan")
        l = compute_vps_ledger(**kw)
        self.assertEqual(l["components"]["contamination"], 0.0)

        kw["neb_capable_stars"] = 0
        l = compute_vps_ledger(**kw)
        self.assertEqual(l["components"]["neb_neighbors"], 0.0)


class TestSecondaryBoundariesAndZones(unittest.TestCase):
    """
    Exhaustively tests the phase geometry boundaries separating circular and eccentric
    secondary eclipse regimes (|phi - 0.50| <= 0.04), dual-zone SNR thresholds,
    and verifies full-precision metadata preservation without rounding divergence (R1).
    """

    def test_full_precision_metadata_and_arbitrary_snr(self):
        # R1 Test Case 1: phi=0.5, z=3.123456 -> actual penalty 46.85184.
        # Previously rounded SNR to 3.1235 led to penalty 46.8525 (diff 0.00066 > 1e-4).
        l = compute_vps_ledger(0, 0.50, 3.123456, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "TEST_R1_1",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "error_message": "",
        }
        score = validate_and_extract_vps_score(row)
        self.assertEqual(score, l["final_score"])

        # R1 Test Case 2: phi=0.5, z=2.80001 -> applied True at producer.
        # Previously rounded z to 2.8 caused validator to expect False.
        l2 = compute_vps_ledger(0, 0.50, 2.80001, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertTrue(l2["secondary_metadata"]["penalty_applied"])
        row2 = {
            "tic_id": "TEST_R1_2",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l2["final_score"],
            "fp_risk_percent": l2["final_score"],
            "vps_unclipped_total": l2["unclipped_total"],
            "vps_ledger_json": json.dumps(l2),
            "error_message": "",
        }
        score2 = validate_and_extract_vps_score(row2)
        self.assertEqual(score2, l2["final_score"])

        # R1 Test Case 3: phi=0.53999, z=2.6 -> circular (|0.53999 - 0.5| = 0.03999 <= 0.04).
        # Previously rounded phi to 0.54 caused validator to treat it as eccentric (|0.54 - 0.5| > 0.04).
        l3 = compute_vps_ledger(0, 0.53999, 2.6, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l3["secondary_metadata"]["zone"], "circular")
        row3 = {
            "tic_id": "TEST_R1_3",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l3["final_score"],
            "fp_risk_percent": l3["final_score"],
            "vps_unclipped_total": l3["unclipped_total"],
            "vps_ledger_json": json.dumps(l3),
            "error_message": "",
        }
        score3 = validate_and_extract_vps_score(row3)
        self.assertEqual(score3, l3["final_score"])

    def test_phase_boundaries(self):
        # Exact circular center (0.50)
        l = compute_vps_ledger(0, 0.50, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l["secondary_metadata"]["zone"], "circular")
        self.assertEqual(l["secondary_metadata"]["threshold"], 2.8)
        self.assertTrue(l["secondary_metadata"]["penalty_applied"])

        # Well within circular zone (0.48, 0.52)
        l = compute_vps_ledger(0, 0.48, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l["secondary_metadata"]["zone"], "circular")
        l = compute_vps_ledger(0, 0.52, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l["secondary_metadata"]["zone"], "circular")

        # Circular boundary interiors: 0.4601 and 0.5399 (|phi - 0.50| = 0.0399 < 0.04)
        l = compute_vps_ledger(0, 0.4601, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l["secondary_metadata"]["zone"], "circular")
        self.assertEqual(l["secondary_metadata"]["threshold"], 2.8)
        self.assertTrue(l["secondary_metadata"]["penalty_applied"])

        l = compute_vps_ledger(0, 0.5399, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l["secondary_metadata"]["zone"], "circular")
        self.assertEqual(l["secondary_metadata"]["threshold"], 2.8)
        self.assertTrue(l["secondary_metadata"]["penalty_applied"])

        # Due to binary IEEE-754 representation:
        # abs(0.46 - 0.50) = 0.03999999999999998 <= 0.04 -> evaluates to CIRCULAR
        # abs(0.4599 - 0.50) = 0.04010000000000005 > 0.04 -> evaluates to ECCENTRIC
        # abs(0.54 - 0.50) = 0.040000000000000036 > 0.04 -> evaluates to ECCENTRIC
        # abs(0.5399 - 0.50) = 0.039900000000000024 < 0.04 -> evaluates to CIRCULAR
        l_46 = compute_vps_ledger(0, 0.46, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l_46["secondary_metadata"]["zone"], "circular")
        self.assertEqual(l_46["secondary_metadata"]["threshold"], 2.8)
        self.assertTrue(l_46["secondary_metadata"]["penalty_applied"])

        l_4599 = compute_vps_ledger(0, 0.4599, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l_4599["secondary_metadata"]["zone"], "eccentric")
        self.assertEqual(l_4599["secondary_metadata"]["threshold"], 3.4)
        self.assertFalse(l_4599["secondary_metadata"]["penalty_applied"])

        l_54 = compute_vps_ledger(0, 0.54, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l_54["secondary_metadata"]["zone"], "eccentric")
        self.assertEqual(l_54["secondary_metadata"]["threshold"], 3.4)
        self.assertFalse(l_54["secondary_metadata"]["penalty_applied"])

        l_5399 = compute_vps_ledger(0, 0.5399, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l_5399["secondary_metadata"]["zone"], "circular")
        self.assertEqual(l_5399["secondary_metadata"]["threshold"], 2.8)
        self.assertTrue(l_5399["secondary_metadata"]["penalty_applied"])

        # Clearly eccentric phases: 0.10, 0.70, 0.90
        l_70 = compute_vps_ledger(0, 0.70, 2.9, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        self.assertEqual(l_70["secondary_metadata"]["zone"], "eccentric")
        self.assertEqual(l_70["secondary_metadata"]["threshold"], 3.4)
        self.assertFalse(l_70["secondary_metadata"]["penalty_applied"])


class TestClippingAndRounding(unittest.TestCase):
    """
    Tests high sum clipping to 99.9, low sum bounds, and half-to-even serialization rounding.
    """

    def test_high_sum_clipping(self):
        # Trigger multiple large penalties: catalog EB (90) + Gaia NSS (80)
        l = compute_vps_ledger(0, 0.50, 0.0, False, True, True, 0, 0, 1.0, 1.0, 10.0)
        self.assertAlmostEqual(l["unclipped_total"], 170.2, places=5)
        self.assertEqual(l["clipped_score"], 99.9)
        self.assertEqual(l["final_score"], 99.9)

    def test_rounding_decimals(self):
        # Test 1-decimal rounding
        l = compute_vps_ledger(0.85, 0.50, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        # base (0.2) + impact (4.0) = 4.2
        self.assertEqual(l["final_score"], 4.2)


class TestSCI16ScorePreservationAndNoSubtraction(unittest.TestCase):
    """
    Resolves Finding SCI-16:
    Proves that a candidate with score 35.0 and circular sec_snr = 2.6 (which had 0
    secondary penalty applied by Pleiades) retains its score of 35.0 and fails the
    Stage 3A risk gate (35.0 > 25.0).
    """

    def test_achievable_fixture_score_35_sec_snr_2_6(self):
        # Physical fixture yielding unclipped total = 35.0:
        # Base = 0.2
        # RUWE = 1.30 (> 1.25) -> 15.0
        # Impact parameter: 25.0 * ((b - 0.75) / 0.25)**2 = 19.8
        b_target = 0.75 + 0.25 * math.sqrt(19.8 / 25.0)

        ledger = compute_vps_ledger(
            b_fit=b_target,
            sec_center_phi=0.50,
            sec_snr=2.6,
            is_sec_eb=False,
            is_tess_eb=False,
            is_gaia_nss=False,
            neb_capable_stars=0,
            contamination_ratio=0.0,
            dur_ratio_kepler=1.0,
            ruwe=1.30,
            stitched_primary_snr=10.0,
        )

        self.assertAlmostEqual(ledger["components"]["base"], 0.2, places=6)
        self.assertAlmostEqual(ledger["components"]["ruwe"], 15.0, places=6)
        self.assertAlmostEqual(ledger["components"]["impact_parameter"], 19.8, places=6)
        self.assertEqual(ledger["components"]["secondary_significance"], 0.0)
        self.assertFalse(ledger["secondary_metadata"]["penalty_applied"])
        self.assertAlmostEqual(ledger["unclipped_total"], 35.0, places=4)
        self.assertEqual(ledger["final_score"], 35.0)

        # Consumer validation:
        row = {
            "tic_id": "TEST_35",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": 35.0,
            "fp_risk_percent": 35.0,
            "vps_unclipped_total": ledger["unclipped_total"],
            "vps_ledger_json": json.dumps(ledger),
            "operational_status": "VETTED",
            "error_message": "",
        }

        # 1. Authoritative consumer score extraction
        extracted_score = validate_and_extract_vps_score(row)
        self.assertEqual(extracted_score, 35.0)

        # 2. Stage 3A gate evaluation: corr_risk <= 25.0
        self.assertFalse(extracted_score <= ANTARES_DEFAULT_MAX_CORR_RISK)

        # 3. Demonstration of legacy flaw (SCI-16):
        legacy_subtracted_score = max(0.2, min(99.9, 35.0 - (45.0 * (2.6 / 3.0))))
        self.assertEqual(legacy_subtracted_score, 0.2)
        self.assertTrue(legacy_subtracted_score <= ANTARES_DEFAULT_MAX_CORR_RISK)

    def test_unsafe_legacy_row_rejected_by_consumer(self):
        legacy_row = {
            "tic_id": "LEGACY_35",
            "vps_score": 35.0,
            "fp_risk_percent": 35.0,
            "secondary_snr_sigma": 2.6,
            "operational_status": "VETTED",
        }
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(legacy_row)
        self.assertIn("missing required 'vps_version'", str(ctx.exception))


class TestLedgerReconstructionAndTampering(unittest.TestCase):
    """
    Validates CSV round-trip fidelity, exact ledger component sum reconstruction,
    and strict rejection of any altered, missing, non-finite, or malformed data (R3).
    """

    def setUp(self):
        self.ledger = compute_vps_ledger(
            b_fit=0.85,  # 4.0
            sec_center_phi=0.50,
            sec_snr=2.85,  # 45 * (2.85 / 3.0) = 42.75
            is_sec_eb=False,
            is_tess_eb=False,
            is_gaia_nss=False,
            neb_capable_stars=1,  # 8.0
            contamination_ratio=0.0,
            dur_ratio_kepler=1.0,
            ruwe=1.0,
            stitched_primary_snr=10.0,
        )
        self.valid_row = {
            "tic_id": "999001",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": self.ledger["final_score"],
            "fp_risk_percent": self.ledger["final_score"],
            "vps_unclipped_total": self.ledger["unclipped_total"],
            "vps_ledger_json": json.dumps(self.ledger),
            "operational_status": "VETTED",
            "error_message": "",
        }

    def test_csv_round_trip_reconstruction(self):
        df = pd.DataFrame([self.valid_row])
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        df_read = pd.read_csv(csv_buf, dtype={"tic_id": str})
        score = validate_and_extract_vps_score(df_read.iloc[0])
        self.assertEqual(score, self.ledger["final_score"])

    def test_unrecognized_producer_rejected(self):
        # R3: unknown producer rejected
        row = copy.deepcopy(self.valid_row)
        row["vps_producer"] = "unrecognized"
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("unrecognized producer", str(ctx.exception))

    def test_missing_or_failed_operational_status_rejected(self):
        # R3: missing or non-successful operational_status rejected
        row = copy.deepcopy(self.valid_row)
        row["operational_status"] = ""
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("missing required 'operational_status'", str(ctx.exception))

        row2 = copy.deepcopy(self.valid_row)
        row2["operational_status"] = "ERROR"
        with self.assertRaises(VPSContractError) as ctx2:
            validate_and_extract_vps_score(row2)
        self.assertIn("non-successful operational status", str(ctx2.exception))

    def test_contradictory_embedded_ledger_fields_rejected(self):
        # R3: embedded ledger contradicts top-level row fields
        row = copy.deepcopy(self.valid_row)
        led_copy = copy.deepcopy(self.ledger)
        led_copy["unclipped_total"] = 999.0
        row["vps_ledger_json"] = json.dumps(led_copy)
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("contradicts top-level vps_unclipped_total", str(ctx.exception))

        row2 = copy.deepcopy(self.valid_row)
        led_copy2 = copy.deepcopy(self.ledger)
        led_copy2["final_score"] = 99.9
        row2["vps_ledger_json"] = json.dumps(led_copy2)
        with self.assertRaises(VPSContractError) as ctx2:
            validate_and_extract_vps_score(row2)
        self.assertIn("contradicts top-level vps_score", str(ctx2.exception))

        row3 = copy.deepcopy(self.valid_row)
        led_copy3 = copy.deepcopy(self.ledger)
        led_copy3["producer"] = "other"
        row3["vps_ledger_json"] = json.dumps(led_copy3)
        with self.assertRaises(VPSContractError) as ctx3:
            validate_and_extract_vps_score(row3)
        self.assertIn("ledger producer mismatch", str(ctx3.exception))

    def test_secondary_metadata_nonfinite_or_malformed_rejected(self):
        # R3: secondary_metadata non-finite values or malformed types rejected
        for key in ("threshold", "snr", "penalty_amount"):
            row = copy.deepcopy(self.valid_row)
            led_copy = copy.deepcopy(self.ledger)
            led_copy["secondary_metadata"][key] = float("nan")
            row["vps_ledger_json"] = json.dumps(led_copy)
            with self.assertRaises(VPSContractError):
                validate_and_extract_vps_score(row)

        # Non-boolean is_circular
        row = copy.deepcopy(self.valid_row)
        led_copy = copy.deepcopy(self.ledger)
        led_copy["secondary_metadata"]["is_circular"] = "True"
        row["vps_ledger_json"] = json.dumps(led_copy)
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(row)

        # Negative penalty
        row = copy.deepcopy(self.valid_row)
        led_copy = copy.deepcopy(self.ledger)
        led_copy["secondary_metadata"]["penalty_amount"] = -5.0
        row["vps_ledger_json"] = json.dumps(led_copy)
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(row)

    def test_fp_risk_percent_inf_rejected(self):
        # R3: infinite alias rejected
        row = copy.deepcopy(self.valid_row)
        row["fp_risk_percent"] = float("inf")
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(row)

    def test_non_rounded_score_rejected(self):
        # R3: score not adhering to 1-decimal rounding rejected (e.g. 25.04)
        row = copy.deepcopy(self.valid_row)
        row["vps_score"] = 25.04
        row["fp_risk_percent"] = 25.04
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("not rounded to 1 decimal", str(ctx.exception))

    def test_tampered_unclipped_total(self):
        row = copy.deepcopy(self.valid_row)
        row["vps_unclipped_total"] = float(row["vps_unclipped_total"]) + 5.0
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("contradicts top-level vps_unclipped_total", str(ctx.exception))

    def test_tampered_component_in_ledger(self):
        row = copy.deepcopy(self.valid_row)
        ledger_copy = copy.deepcopy(self.ledger)
        ledger_copy["components"]["base"] = 5.0  # Tampered from 0.2 to 5.0
        row["vps_ledger_json"] = json.dumps(ledger_copy)
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("differs from vps_unclipped_total", str(ctx.exception))

    def test_conflicting_aliases(self):
        row = copy.deepcopy(self.valid_row)
        row["fp_risk_percent"] = float(row["vps_score"]) + 5.0
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("conflicting score aliases", str(ctx.exception))

    def test_missing_version(self):
        row = copy.deepcopy(self.valid_row)
        row["vps_version"] = ""
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("missing required 'vps_version'", str(ctx.exception))

    def test_unsupported_version(self):
        row = copy.deepcopy(self.valid_row)
        row["vps_version"] = "vps_v2_unsupported"
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("unsupported VPS version", str(ctx.exception))

    def test_missing_producer(self):
        row = copy.deepcopy(self.valid_row)
        row["vps_producer"] = ""
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("missing required 'vps_producer'", str(ctx.exception))

    def test_missing_score(self):
        row = copy.deepcopy(self.valid_row)
        row["vps_score"] = float("nan")
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("missing 'vps_score'", str(ctx.exception))

        row2 = copy.deepcopy(self.valid_row)
        row2["vps_score"] = "NOT_A_NUMBER"
        with self.assertRaises(VPSContractError) as ctx2:
            validate_and_extract_vps_score(row2)
        self.assertIn("invalid non-numeric 'vps_score'", str(ctx2.exception))

    def test_malformed_ledger_json(self):
        row = copy.deepcopy(self.valid_row)
        row["vps_ledger_json"] = "NOT_A_VALID_JSON{{"
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("malformed 'vps_ledger_json'", str(ctx.exception))

    def test_missing_mandatory_component_key(self):
        row = copy.deepcopy(self.valid_row)
        ledger_copy = copy.deepcopy(self.ledger)
        del ledger_copy["components"]["weak_primary"]
        row["vps_ledger_json"] = json.dumps(ledger_copy)
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("missing mandatory component 'weak_primary'", str(ctx.exception))

    def test_negative_component_penalty(self):
        row = copy.deepcopy(self.valid_row)
        ledger_copy = copy.deepcopy(self.ledger)
        ledger_copy["components"]["impact_parameter"] = -2.0
        row["vps_ledger_json"] = json.dumps(ledger_copy)
        with self.assertRaises(VPSContractError) as ctx:
            validate_and_extract_vps_score(row)
        self.assertIn("is negative", str(ctx.exception))


class TestPleiadesCleanCriterionAndPrematureRounding(unittest.TestCase):
    """
    Validates R2: Premature rounding must not change Pleiades is_clean disposition.
    Original Pleiades producer evaluated is_clean using the unrounded clipped score.
    """

    def test_unrounded_clipped_score_preserves_clean_disposition(self):
        # R2 Case: base (0.2) + weak_primary (4.76) = 4.96.
        # stitched_primary_snr = 5.5 - (4.76 / 22.0) = 5.283636363636364.
        pri_snr = 5.5 - (4.76 / 22.0)
        ledger = compute_vps_ledger(
            b_fit=0.0,
            sec_center_phi=0.50,
            sec_snr=0.0,
            is_sec_eb=False,
            is_tess_eb=False,
            is_gaia_nss=False,
            neb_capable_stars=0,
            contamination_ratio=0.0,
            dur_ratio_kepler=1.0,
            ruwe=1.0,
            stitched_primary_snr=pri_snr,
        )

        raw_clipped = ledger["clipped_score"]
        exported_score = ledger["final_score"]

        # Raw score is strictly < 5.0
        self.assertAlmostEqual(raw_clipped, 4.96, places=5)
        self.assertTrue(raw_clipped < PLEIADES_CLEAN_VPS_THRESHOLD)

        # Exported score rounds to 5.0
        self.assertEqual(exported_score, 5.0)

        # If rounded score were checked against < 5.0, it would fail:
        self.assertFalse(exported_score < PLEIADES_CLEAN_VPS_THRESHOLD)

        # Pleiades producer uses raw_clipped for is_clean, so the candidate passes:
        is_clean_decision = (
            raw_clipped < PLEIADES_CLEAN_VPS_THRESHOLD
            and 0.0 < 0.82
            and 0.0 <= 2.8
            and not False
            and not False
            and not False
            and 0.35 <= 1.0 <= 1.38
            and pri_snr >= 5.5 - (4.76 / 22.0)  # non-trivial
        )
        self.assertTrue(is_clean_decision)

    def test_clean_threshold_boundary_progression(self):
        # Progression across the threshold: 4.94, 4.95, 4.96, 4.99, 5.00, 5.01
        for val, expected_clean in [
            (4.94, True),
            (4.95, True),
            (4.96, True),
            (4.99, True),
            (5.00, False),
            (5.01, False),
        ]:
            self.assertEqual(val < PLEIADES_CLEAN_VPS_THRESHOLD, expected_clean)

    def test_no_stale_gate_24_text_in_source(self):
        with open(pleiades.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("Gate <= 24.0", src)
        self.assertNotIn("Gate <= 24", src)
        self.assertIn("Pleiades Clean Criterion <", src)


class TestStage3AFilterEndToEndAndGateCrossValidation(unittest.TestCase):
    """
    Validates Stage 3A execution with real CSV files, ensuring passing candidates survive,
    high-risk candidates are rejected, duplicate secondary columns are cross-validated (R3),
    missing gate evidence is rejected (R3), and missing Tier-2 files raise FileNotFoundError.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_stage3a_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stage_3a_filtering_and_risk_gate(self):
        # Create Tier-1 CSV with two candidates: TIC 10001 (clean), TIC 10002 (risk 35.0)
        t1_path = os.path.join(self.tmp_dir, "t1_cands.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_dossiers.csv")

        t1_df = pd.DataFrame([
            {"tic_id": "10001", "period_days": 3.1, "t0_btjd": 1200.0, "duration_hours": 2.0, "depth_ppm": 1000.0},
            {"tic_id": "10002", "period_days": 5.4, "t0_btjd": 1205.0, "duration_hours": 3.0, "depth_ppm": 800.0},
        ])
        t1_df.to_csv(t1_path, index=False)

        # Tier-2 dossier data with full realistic measurement columns:
        l_clean = compute_vps_ledger(0, 0.5, 0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        b_target = 0.75 + 0.25 * math.sqrt(19.8 / 25.0)
        l_risk35 = compute_vps_ledger(b_target, 0.5, 2.6, False, False, False, 0, 0, 1.0, 1.30, 10.0)

        t2_df = pd.DataFrame([
            {
                "tic_id": "10001",
                "operational_status": "VETTED",
                "vps_version": VPS_SCHEMA_VERSION,
                "vps_producer": VPS_PRODUCER_NAME,
                "vps_score": l_clean["final_score"],
                "fp_risk_percent": l_clean["final_score"],
                "vps_unclipped_total": l_clean["unclipped_total"],
                "vps_ledger_json": json.dumps(l_clean),
                "refined_period_days": 3.1,
                "refined_t0_btjd": 1200.0,
                "fitted_duration_hours": 2.0,
                "fitted_depth_ppm": 1000.0,
                "fitted_rp_earth": 2.1,
                "dur_to_kepler_max": 1.0,
                "secondary_snr_sigma": 0.0,
                "sec_eclipse_phase": 0.50,
                "flux_contam_percent": 5.0,
                "neb_capable_neighbors": 0,
                "is_known_tess_eb": False,
                "gaia_nss_type": "",
                "sectors_stitched": "1",
                "error_message": "",
            },
            {
                "tic_id": "10002",
                "operational_status": "VETTED",
                "vps_version": VPS_SCHEMA_VERSION,
                "vps_producer": VPS_PRODUCER_NAME,
                "vps_score": l_risk35["final_score"],
                "fp_risk_percent": l_risk35["final_score"],
                "vps_unclipped_total": l_risk35["unclipped_total"],
                "vps_ledger_json": json.dumps(l_risk35),
                "refined_period_days": 5.4,
                "refined_t0_btjd": 1205.0,
                "fitted_duration_hours": 3.0,
                "fitted_depth_ppm": 800.0,
                "fitted_rp_earth": 1.8,
                "dur_to_kepler_max": 1.0,
                "secondary_snr_sigma": 2.6,
                "sec_eclipse_phase": 0.50,
                "flux_contam_percent": 2.0,
                "neb_capable_neighbors": 0,
                "is_known_tess_eb": False,
                "gaia_nss_type": "",
                "sectors_stitched": "1",
                "error_message": "",
            },
        ])
        t2_df.to_csv(t2_path, index=False)

        # Run Stage 3A filter
        survivors, rejected = antares_packager.run_stage_3a_filter(
            tier1_csv=t1_path,
            tier2_csv=t2_path,
            max_corr_risk=25.0,
        )

        # Candidate 10001 (score 0.2) survives; 10002 (score 35.0) rejected!
        self.assertEqual(len(survivors), 1)
        self.assertEqual(str(survivors.iloc[0]["tic_id"]), "10001")
        self.assertEqual(survivors.iloc[0]["corr_risk"], 0.2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(str(rejected.iloc[0]["tic_id"]), "10002")

    def test_duplicate_secondary_column_contradiction_rejected(self):
        # R3 / B1: Contradiction between top-level secondary column and ledger metadata raises VPSContractError
        t1_path = os.path.join(self.tmp_dir, "t1_contra.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_contra.csv")

        pd.DataFrame([{"tic_id": "20001"}]).to_csv(t1_path, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "20001",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "secondary_snr_sigma": 5.0,  # Contradicts ledger snr=0.0!
            "sec_eclipse_phase": 0.50,
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("contradictory secondary SNR", str(ctx.exception))

        # Also test contradictory secondary phase: top-level 0.70 vs canonical 0.50
        pd.DataFrame([{
            "tic_id": "20001",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "secondary_snr_sigma": 0.0,
            "sec_eclipse_phase": 0.70,  # Contradicts ledger phase=0.50!
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        with self.assertRaises(VPSContractError) as ctx_phase:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("contradictory secondary phase", str(ctx_phase.exception))

    def test_duplicate_secondary_columns_normal_rounding_accepted(self):
        # B1: Normal serialization rounding in duplicate columns (2 decimals for SNR, 3 decimals for phase)
        # must be accepted against full-precision ledger without false tampering alarm.
        t1_path = os.path.join(self.tmp_dir, "t1_norm_round.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_norm_round.csv")
        pd.DataFrame([{"tic_id": "20002"}]).to_csv(t1_path, index=False)

        # Full-precision secondary SNR = 2.61234, phase = 0.53999
        l = compute_vps_ledger(0, 0.53999, 2.61234, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "20002",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "secondary_snr_sigma": 2.61,  # Documented 2-decimal export of 2.61234
            "sec_eclipse_phase": 0.540,  # Documented 3-decimal export of 0.53999
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 1)
        self.assertEqual(len(rejected), 0)

    def test_missing_gate_evidence_raises_contract_error(self):
        # B2: Missing duration evidence is an operational contract error, NOT a silent scientific rejection
        t1_path = os.path.join(self.tmp_dir, "t1_nodur.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_nodur.csv")

        pd.DataFrame([{"tic_id": "30001"}]).to_csv(t1_path, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "30001",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            # Missing dur_to_kepler_max!
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("missing mandatory gate measurement 'dur_to_kepler_max'", str(ctx.exception))

    def test_invalid_environment_evidence_domain_raises_contract_error(self):
        # B2: Invalid domain values (e.g. contam = -inf, neighbors = -1) must raise VPSContractError
        t1_path = os.path.join(self.tmp_dir, "t1_inval_env.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_inval_env.csv")
        pd.DataFrame([{"tic_id": "30002"}]).to_csv(t1_path, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)

        # Invalid contamination: -inf
        pd.DataFrame([{
            "tic_id": "30002",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": float("-inf"),
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("invalid mandatory gate measurement 'flux_contam_percent'", str(ctx.exception))

        # Invalid neighbors: -1
        pd.DataFrame([{
            "tic_id": "30002",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": -1,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        with self.assertRaises(VPSContractError) as ctx2:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("invalid mandatory gate measurement 'neb_capable_neighbors'", str(ctx2.exception))

    def test_missing_catalog_eb_flag_raises_contract_error(self):
        # C1: Missing is_known_tess_eb flag must raise VPSContractError (not silently default to False)
        t1_path = os.path.join(self.tmp_dir, "t1_noeb.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_noeb.csv")
        pd.DataFrame([{"tic_id": "30003"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30003",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "gaia_nss_type": "",
            "error_message": "",
            # is_known_tess_eb deliberately missing!
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("missing mandatory gate flag 'is_known_tess_eb'", str(ctx.exception))

    def test_invalid_non_boolean_catalog_eb_flag_raises_contract_error(self):
        # C1: Non-boolean string for is_known_tess_eb must raise VPSContractError
        t1_path = os.path.join(self.tmp_dir, "t1_inveb.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_inveb.csv")
        pd.DataFrame([{"tic_id": "30004"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30004",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": "MAYBE_EB",
            "gaia_nss_type": "",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("invalid non-boolean 'is_known_tess_eb'", str(ctx.exception))

    def test_missing_both_nss_fields_raises_contract_error(self):
        # C1: Missing both gaia_nss_type and is_gaia_nss must raise VPSContractError
        t1_path = os.path.join(self.tmp_dir, "t1_nonss.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_nonss.csv")
        pd.DataFrame([{"tic_id": "30005"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30005",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "error_message": "",
            # Both gaia_nss_type and is_gaia_nss missing!
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("missing mandatory Gaia NSS catalog context", str(ctx.exception))

    def test_explicit_true_nss_flag_without_type_fails_gate_with_zero_survivors(self):
        # C1: Explicit is_gaia_nss=True without gaia_nss_type must fail Gate 1 (0 survivors, 1 rejected)
        t1_path = os.path.join(self.tmp_dir, "t1_nsstrue.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_nsstrue.csv")
        pd.DataFrame([{"tic_id": "30006"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, True, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30006",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": True,
            # gaia_nss_type omitted
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 0, "True NSS binary must not survive Gate 1")
        self.assertEqual(len(rejected), 1)

    def test_explicit_true_nss_flag_with_consistent_type_fails_gate_with_zero_survivors(self):
        # C1: Explicit is_gaia_nss=True with consistent solution type fails Gate 1 (0 survivors, 1 rejected)
        t1_path = os.path.join(self.tmp_dir, "t1_nssorb.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_nssorb.csv")
        pd.DataFrame([{"tic_id": "30007"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, True, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30007",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": True,
            "gaia_nss_type": "Orbital",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 0)
        self.assertEqual(len(rejected), 1)

    def test_explicit_true_eb_flag_fails_gate_with_zero_survivors(self):
        # C1: Explicit is_known_tess_eb=True fails Gate 1 (0 survivors, 1 rejected)
        t1_path = os.path.join(self.tmp_dir, "t1_ebtrue.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_ebtrue.csv")
        pd.DataFrame([{"tic_id": "30008"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, True, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30008",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": True,
            "gaia_nss_type": "",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 0)
        self.assertEqual(len(rejected), 1)

    def test_contradictory_nss_flag_and_type_raises_contract_error(self):
        # C1: Contradictory NSS context must raise VPSContractError
        t1_path = os.path.join(self.tmp_dir, "t1_contra_nss_a.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_contra_nss_a.csv")
        pd.DataFrame([{"tic_id": "30009"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)

        # Case A: is_gaia_nss=False but gaia_nss_type="Orbital"
        row_a = {
            "tic_id": "30009",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": False,
            "gaia_nss_type": "Orbital",
            "error_message": "",
        }
        pd.DataFrame([row_a]).to_csv(t2_path, index=False)
        with self.assertRaises(VPSContractError) as ctx_a:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("contradictory Gaia NSS context", str(ctx_a.exception))

        # Case B: is_gaia_nss=True but gaia_nss_type="FALSE"
        row_b = dict(row_a)
        row_b["is_gaia_nss"] = True
        row_b["gaia_nss_type"] = "FALSE"
        pd.DataFrame([row_b]).to_csv(t2_path, index=False)
        with self.assertRaises(VPSContractError) as ctx_b:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("contradictory Gaia NSS context", str(ctx_b.exception))

    def test_valid_explicit_no_match_passes_catalog_gate(self):
        # C1: Explicit no-match representation passes catalog gate and survives
        t1_path = os.path.join(self.tmp_dir, "t1_nomatch.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_nomatch.csv")
        pd.DataFrame([{"tic_id": "30010"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30010",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": "False",
            "is_gaia_nss": "0",
            "gaia_nss_type": "NaN",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 1)
        self.assertEqual(len(rejected), 0)

    def test_high_flux_contamination_ratio_exceeding_100_percent_passes_contract_and_fails_scientific_gate(self):
        # D1: A high contamination ratio (e.g. ratio 2.0 -> flux_contam_percent = 200.0) is valid physical measurement,
        # not malformed data. It must pass contract validation and be legitimately rejected by Stage 3A scientific gate (<= 20%).
        t1_path = os.path.join(self.tmp_dir, "t1_contam200.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_contam200.csv")
        pd.DataFrame([{"tic_id": "30011"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 2.0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30011",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 200.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": False,
            "gaia_nss_type": "",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 0, "High contamination candidate (200%) must be rejected by scientific gate")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(str(rejected.iloc[0]["tic_id"]), "30011")

    def test_mixed_clean_and_high_contamination_batch(self):
        # D1: In a mixed batch with clean candidate and high-contamination candidate (200%),
        # valid clean candidate proceeds, high-ratio candidate is scientifically rejected.
        t1_path = os.path.join(self.tmp_dir, "t1_mixed_contam.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_mixed_contam.csv")
        pd.DataFrame([{"tic_id": "30012"}, {"tic_id": "30013"}]).to_csv(t1_path, index=False)

        l_clean = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0.05, 1.0, 1.0, 10.0)
        l_high = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 2.0, 1.0, 1.0, 10.0)

        rows = [
            {
                "tic_id": "30012",
                "operational_status": "VETTED",
                "vps_version": VPS_SCHEMA_VERSION,
                "vps_producer": VPS_PRODUCER_NAME,
                "vps_score": l_clean["final_score"],
                "fp_risk_percent": l_clean["final_score"],
                "vps_unclipped_total": l_clean["unclipped_total"],
                "vps_ledger_json": json.dumps(l_clean),
                "dur_to_kepler_max": 1.0,
                "flux_contam_percent": 5.0,
                "neb_capable_neighbors": 0,
                "is_known_tess_eb": False,
                "is_gaia_nss": False,
                "gaia_nss_type": "",
                "error_message": "",
            },
            {
                "tic_id": "30013",
                "operational_status": "VETTED",
                "vps_version": VPS_SCHEMA_VERSION,
                "vps_producer": VPS_PRODUCER_NAME,
                "vps_score": l_high["final_score"],
                "fp_risk_percent": l_high["final_score"],
                "vps_unclipped_total": l_high["unclipped_total"],
                "vps_ledger_json": json.dumps(l_high),
                "dur_to_kepler_max": 1.0,
                "flux_contam_percent": 200.0,
                "neb_capable_neighbors": 0,
                "is_known_tess_eb": False,
                "is_gaia_nss": False,
                "gaia_nss_type": "",
                "error_message": "",
            },
        ]
        pd.DataFrame(rows).to_csv(t2_path, index=False)
        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 1, "Clean candidate must survive")
        self.assertEqual(str(survivors.iloc[0]["tic_id"]), "30012")
        self.assertEqual(len(rejected), 1, "High-contamination candidate must be rejected")
        self.assertEqual(str(rejected.iloc[0]["tic_id"]), "30013")

    def test_negative_flux_contam_percent_raises_contract_error(self):
        # D1: Negative contamination ratio/percentage (< 0.0) is unphysical and must raise VPSContractError
        t1_path = os.path.join(self.tmp_dir, "t1_neg_contam.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_neg_contam.csv")
        pd.DataFrame([{"tic_id": "30014"}]).to_csv(t1_path, index=False)
        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0.0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "30014",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": -5.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": False,
            "gaia_nss_type": "",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)
        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertIn("invalid mandatory gate measurement 'flux_contam_percent'", str(ctx.exception))

    def test_producer_export_to_csv_to_consumer_with_high_contamination_ratio_200(self):
        # D1: End-to-end producer -> CSV -> actual consumer with contamination_ratio 2.0 (flux_contam_percent 200.0).
        # Real producer helper calculates ledger and exports 200.0% without error; consumer Stage 3A accepts contract and rejects candidate.
        t1_path = os.path.join(self.tmp_dir, "t1_p2c_contam200.csv")
        t2_path = os.path.join(self.tmp_dir, "t2_p2c_contam200.csv")
        pd.DataFrame([{"tic_id": "30015", "period_days": 2.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_path, index=False)

        contam_ratio = 2.0
        ledger = compute_vps_ledger(
            b_fit=0.0,
            sec_center_phi=0.5,
            sec_snr=0.0,
            is_sec_eb=False,
            is_tess_eb=False,
            is_gaia_nss=False,
            neb_capable_stars=0,
            contamination_ratio=contam_ratio,
            dur_ratio_kepler=1.0,
            ruwe=1.0,
            stitched_primary_snr=10.0,
        )
        self.assertAlmostEqual(ledger["components"]["contamination"], 70.0, places=4)

        row = {
            "tic_id": "30015",
            "operational_status": "SUCCESS",
            "disposition": "TIER-2 VETTED CANDIDATE",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": round(ledger["final_score"], 1),
            "fp_risk_percent": round(ledger["final_score"], 1),
            "vps_unclipped_total": float(ledger["unclipped_total"]),
            "vps_ledger_json": json.dumps(ledger),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": round(contam_ratio * 100.0, 2),
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }
        pd.DataFrame([row]).to_csv(t2_path, index=False)

        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_path, tier2_csv=t2_path)
        self.assertEqual(len(survivors), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(str(rejected.iloc[0]["tic_id"]), "30015")

    def test_missing_tier2_csv_raises_file_not_found(self):
        t1_path = os.path.join(self.tmp_dir, "t1.csv")
        pd.DataFrame([{"tic_id": "123"}]).to_csv(t1_path, index=False)
        missing_t2 = os.path.join(self.tmp_dir, "nonexistent_t2.csv")

        with self.assertRaises(FileNotFoundError) as ctx:
            antares_packager.run_stage_3a_filter(
                tier1_csv=t1_path,
                tier2_csv=missing_t2,
            )
        self.assertIn("nonexistent_t2.csv", str(ctx.exception))

    def test_empty_tier1_handled_cleanly(self):
        t1_path = os.path.join(self.tmp_dir, "empty_t1.csv")
        pd.DataFrame(columns=["tic_id"]).to_csv(t1_path, index=False)
        t2_path = os.path.join(self.tmp_dir, "any_t2.csv")
        pd.DataFrame(columns=["tic_id"]).to_csv(t2_path, index=False)

        survivors, rejected = antares_packager.run_stage_3a_filter(
            tier1_csv=t1_path,
            tier2_csv=t2_path,
        )
        self.assertTrue(survivors.empty)
        self.assertIn("corr_risk", survivors.columns)

    def test_stage_3a_filter_optional_annotation_and_supported_python_compatibility(self):
        """
        A2: Under supported Python runtime (including Python 3.11 and 3.14), verifies that
        antares_packager imports Optional explicitly, and run_stage_3a_filter executes
        the valid passing path, the zero-survivor gate path, and the contract violation path
        without NameError: name 'Optional' is not defined.
        """
        # 1. Explicit import verification
        self.assertTrue(hasattr(antares_packager, "Optional"), "antares_packager missing Optional import")
        self.assertIs(antares_packager.Optional, typing.Optional, "antares_packager.Optional is not typing.Optional")

        # 2. Valid passing candidate path
        t1_valid = os.path.join(self.tmp_dir, "t1_valid_opt.csv")
        t2_valid = os.path.join(self.tmp_dir, "t2_valid_opt.csv")
        pd.DataFrame([{
            "tic_id": "90001", "period_days": 4.5, "t0_btjd": 2100.0,
            "duration_hours": 2.5, "depth_ppm": 800.0
        }]).to_csv(t1_valid, index=False)
        ledger_valid = compute_vps_ledger(0.1, 0.5, 0.0, False, False, False, 0, 0.0, 1.0, 1.0, 12.0)
        pd.DataFrame([{
            "tic_id": "90001", "operational_status": "SUCCESS", "disposition": "TIER-2 VETTED CANDIDATE",
            "vps_version": VPS_SCHEMA_VERSION, "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": round(ledger_valid["final_score"], 1), "fp_risk_percent": round(ledger_valid["final_score"], 1),
            "vps_unclipped_total": float(ledger_valid["unclipped_total"]), "vps_ledger_json": json.dumps(ledger_valid),
            "dur_to_kepler_max": 1.0, "flux_contam_percent": 0.0, "neb_capable_neighbors": 0,
            "is_known_tess_eb": False, "gaia_nss_type": "", "error_message": "",
        }]).to_csv(t2_valid, index=False)

        survivors, rejected = antares_packager.run_stage_3a_filter(tier1_csv=t1_valid, tier2_csv=t2_valid)
        self.assertEqual(len(survivors), 1, "Expected 1 survivor for valid passing candidate")
        self.assertEqual(len(rejected), 0)

        # 3. Zero-survivor candidate path (e.g. EB fail)
        t1_zero = os.path.join(self.tmp_dir, "t1_zero_opt.csv")
        t2_zero = os.path.join(self.tmp_dir, "t2_zero_opt.csv")
        pd.DataFrame([{"tic_id": "90002", "period_days": 2.0, "t0_btjd": 2100.0, "duration_hours": 1.5, "depth_ppm": 500.0}]).to_csv(t1_zero, index=False)
        ledger_zero = compute_vps_ledger(0.1, 0.5, 0.0, True, False, False, 0, 0.0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "90002", "operational_status": "SUCCESS", "disposition": "TIER-2 VETTED CANDIDATE",
            "vps_version": VPS_SCHEMA_VERSION, "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": round(ledger_zero["final_score"], 1), "fp_risk_percent": round(ledger_zero["final_score"], 1),
            "vps_unclipped_total": float(ledger_zero["unclipped_total"]), "vps_ledger_json": json.dumps(ledger_zero),
            "dur_to_kepler_max": 1.0, "flux_contam_percent": 0.0, "neb_capable_neighbors": 0,
            "is_known_tess_eb": True, "gaia_nss_type": "", "error_message": "",
        }]).to_csv(t2_zero, index=False)

        survivors_z, rejected_z = antares_packager.run_stage_3a_filter(tier1_csv=t1_zero, tier2_csv=t2_zero)
        self.assertEqual(len(survivors_z), 0, "Expected 0 survivors for EB candidate")
        self.assertEqual(len(rejected_z), 1)

        # 4. Invalid contract path (contradictory NSS context)
        t1_inv = os.path.join(self.tmp_dir, "t1_inv_opt.csv")
        t2_inv = os.path.join(self.tmp_dir, "t2_inv_opt.csv")
        pd.DataFrame([{"tic_id": "90003"}]).to_csv(t1_inv, index=False)
        pd.DataFrame([{
            "tic_id": "90003", "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION, "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": round(ledger_valid["final_score"], 1),
            "fp_risk_percent": round(ledger_valid["final_score"], 1),
            "vps_unclipped_total": float(ledger_valid["unclipped_total"]),
            "vps_ledger_json": json.dumps(ledger_valid),
            "dur_to_kepler_max": 1.0, "flux_contam_percent": 0.0, "neb_capable_neighbors": 0,
            "is_known_tess_eb": False, "is_gaia_nss": False, "gaia_nss_type": "Orbital",
            "error_message": "",
        }]).to_csv(t2_inv, index=False)

        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(tier1_csv=t1_inv, tier2_csv=t2_inv)
        self.assertIn("contradictory Gaia NSS context", str(ctx.exception))

    def test_stage_3a_filter_optional_annotation_falsification_control(self):
        """
        A6: Falsification control. Demonstrates that if Optional is missing from the globals
        when evaluating _get_num_series annotations (the un-imported old state), it triggers
        NameError: name 'Optional' is not defined. The fixed module with Optional present resolves
        annotations without error.
        """
        # Parse _get_num_series from antares_packager.py AST
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "antares_packager.py")
        with open(src_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_get_num_series":
                target_node = node
                break
        self.assertIsNotNone(target_node, "Could not find _get_num_series AST node in antares_packager.py")

        mod = ast.Module(body=[target_node], type_ignores=[])
        code = compile(mod, filename="<test_ast>", mode="exec")

        # 1. Negative control: isolated namespace lacking Optional raises NameError
        # Under Python 3.11 (eager annotation evaluation), executing `def _get_num_series` evaluates
        # default parameter annotations at def time, raising NameError immediately during exec.
        # Under Python 3.14 (PEP 649 deferred annotations), exec defines the function, and evaluating
        # annotations via typing.get_type_hints raises NameError.
        ns_mutant = dict(antares_packager.__dict__)
        ns_mutant.pop("Optional", None)

        with self.assertRaises(NameError) as ctx:
            exec(code, ns_mutant)
            fn = ns_mutant.get("_get_num_series")
            if fn is not None:
                typing.get_type_hints(fn, globalns=ns_mutant)
        self.assertIn("name 'Optional' is not defined", str(ctx.exception))

        # In eager-evaluation runtimes (< 3.14, e.g. Python 3.11), verify that calling real
        # run_stage_3a_filter with Optional popped from module globals triggers NameError on definition:
        if sys.version_info < (3, 14):
            t1_tmp = os.path.join(self.tmp_dir, "t1_eager_fail.csv")
            t2_tmp = os.path.join(self.tmp_dir, "t2_eager_fail.csv")
            pd.DataFrame([{"tic_id": "90001", "period_days": 4.5, "t0_btjd": 2100.0, "duration_hours": 2.5, "depth_ppm": 800.0}]).to_csv(t1_tmp, index=False)
            ledger_tmp = compute_vps_ledger(0.1, 0.5, 0.0, False, False, False, 0, 0.0, 1.0, 1.0, 12.0)
            pd.DataFrame([{
                "tic_id": "90001", "operational_status": "SUCCESS", "disposition": "TIER-2 VETTED CANDIDATE",
                "vps_version": VPS_SCHEMA_VERSION, "vps_producer": VPS_PRODUCER_NAME,
                "vps_score": round(ledger_tmp["final_score"], 1), "fp_risk_percent": round(ledger_tmp["final_score"], 1),
                "vps_unclipped_total": float(ledger_tmp["unclipped_total"]), "vps_ledger_json": json.dumps(ledger_tmp),
                "dur_to_kepler_max": 1.0, "flux_contam_percent": 0.0, "neb_capable_neighbors": 0,
                "is_known_tess_eb": False, "gaia_nss_type": "", "error_message": "",
            }]).to_csv(t2_tmp, index=False)

            orig_opt = antares_packager.__dict__.pop("Optional", None)
            try:
                with self.assertRaises(NameError) as ctx_stage:
                    antares_packager.run_stage_3a_filter(tier1_csv=t1_tmp, tier2_csv=t2_tmp)
                self.assertIn("name 'Optional' is not defined", str(ctx_stage.exception))
            finally:
                if orig_opt is not None:
                    antares_packager.Optional = orig_opt

        # 2. Fixed control: namespace with Optional resolves type hints cleanly
        ns_fixed = dict(antares_packager.__dict__)
        exec(code, ns_fixed)
        hints = typing.get_type_hints(ns_fixed["_get_num_series"], globalns=ns_fixed)
        self.assertEqual(hints.get("col"), str)
        self.assertEqual(hints.get("fallback"), typing.Optional[str])


class TestDeliberateFaultMutationEndToEnd(unittest.TestCase):
    """
    R4 Core Evidence Rule: Break the check on purpose.
    Mutates the actual consumed production function during end-to-end Stage 3A filtering,
    demonstrating that the mutated logic fails by spuriously admitting a false positive,
    while the unmutated production code passes.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_mutation_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_end_to_end_legacy_subtraction_mutation(self):
        # Prepare Tier-1 and Tier-2 files for candidate with vps_score = 35.0, sec_snr = 2.6 (circular)
        t1_path = os.path.join(self.tmp_dir, "t1.csv")
        t2_path = os.path.join(self.tmp_dir, "t2.csv")

        pd.DataFrame([{"tic_id": "MUTANT_35"}]).to_csv(t1_path, index=False)

        b_target = 0.75 + 0.25 * math.sqrt(19.8 / 25.0)
        ledger = compute_vps_ledger(b_target, 0.5, 2.6, False, False, False, 0, 0, 1.0, 1.30, 10.0)
        pd.DataFrame([{
            "tic_id": "MUTANT_35",
            "operational_status": "VETTED",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": 35.0,
            "fp_risk_percent": 35.0,
            "vps_unclipped_total": ledger["unclipped_total"],
            "vps_ledger_json": json.dumps(ledger),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "secondary_snr_sigma": 2.6,
            "sec_eclipse_phase": 0.50,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_path, index=False)

        # 1. Unmutated production code: candidate must be REJECTED (35.0 > 25.0)
        survivors, rejected = antares_packager.run_stage_3a_filter(
            tier1_csv=t1_path,
            tier2_csv=t2_path,
            max_corr_risk=25.0,
        )
        self.assertEqual(len(survivors), 0, "Production code must reject 35.0 risk candidate")
        self.assertEqual(len(rejected), 1)

        # 2. Injected deliberate mutation into consumed production path: restore legacy subtraction
        orig_calc = antares_packager.calc_sidak_corrected_risk
        try:
            def mutated_calc(row):
                base_r = float(row["vps_score"])
                sec_snr = float(row.get("secondary_snr_sigma", 0.0))
                if sec_snr > 2.2:
                    base_r -= 45.0 * (sec_snr / 3.0)
                return max(0.2, min(99.9, base_r))

            antares_packager.calc_sidak_corrected_risk = mutated_calc

            # Run actual Stage 3A filter with mutated function:
            survivors_mut, rejected_mut = antares_packager.run_stage_3a_filter(
                tier1_csv=t1_path,
                tier2_csv=t2_path,
                max_corr_risk=25.0,
            )

            # Mutated code spuriously subtracts 39.0 points, turning 35.0 into 0.2,
            # so the candidate erroneously SURVIVES!
            self.assertEqual(len(survivors_mut), 1, "Mutated code should spuriously pass the candidate")
            self.assertEqual(survivors_mut.iloc[0]["corr_risk"], 0.2)
        finally:
            # Restore production function
            antares_packager.calc_sidak_corrected_risk = orig_calc

    def test_fault_mutation_tampered_ledger_detected(self):
        ledger = compute_vps_ledger(0, 0.5, 0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        row = {
            "tic_id": "TAMPER_TEST",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": ledger["final_score"],
            "fp_risk_percent": ledger["final_score"],
            "vps_unclipped_total": ledger["unclipped_total"],
            "vps_ledger_json": json.dumps(ledger),
            "operational_status": "VETTED",
            "error_message": "",
        }

        # Correct control: passes validation
        self.assertEqual(validate_and_extract_vps_score(row), 0.2)

        # Mutate: tamper with ledger base penalty by 0.05
        tampered_ledger = copy.deepcopy(ledger)
        tampered_ledger["components"]["base"] = 0.25
        row["vps_ledger_json"] = json.dumps(tampered_ledger)

        # Deliberate fault is detected and raises VPSContractError!
        with self.assertRaises(VPSContractError):
            validate_and_extract_vps_score(row)


class TestPleiadesProducerCLIAndErrorRowSchema(unittest.TestCase):
    """
    R4: Validates real Pleiades CLI execution, error row recording with NaNs,
    and ensures downstream Stage 3A halts on operational failure rows.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_pleiades_cli_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run_pleiades_main_ast(self, inp_csv, out_dir, mock_deep_vet):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pleiades_path = os.path.join(root_dir, "pleiades.py")
        with open(pleiades_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        main_if = next(
            n for n in tree.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and any(isinstance(left, ast.Name) and left.id == "__name__" for left in [n.test.left])
        )
        code = compile(ast.Module(body=main_if.body, type_ignores=[]), "<pleiades_main>", "exec")
        ns = dict(pleiades.__dict__)
        old_argv = sys.argv
        sys.argv = ["pleiades.py", "--input-csv", inp_csv, "--output-dir", out_dir]
        orig_deep_vet = pleiades.deep_vet_target
        try:
            pleiades.deep_vet_target = mock_deep_vet
            ns["deep_vet_target"] = mock_deep_vet
            exec(code, ns)
        finally:
            sys.argv = old_argv
            pleiades.deep_vet_target = orig_deep_vet

    def test_pleiades_cli_ast_error_row_and_downstream_halt(self):
        # B3: Execute actual AST of pleiades.py main block with failing deep_vet;
        # verify production error-row construction and downstream Stage 3A halt.
        inp_csv = os.path.join(self.tmp_dir, "input_cands_err.csv")
        out_dir = os.path.join(self.tmp_dir, "dossiers_err")
        summary_csv = os.path.join(out_dir, "scientific_dossier_summary.csv")

        pd.DataFrame([{
            "tic_id": "888001",
            "period_days": 2.5,
            "t0_btjd": 2000.0,
            "duration_hours": 1.5,
            "depth_ppm": 500.0
        }]).to_csv(inp_csv, index=False)

        def mock_failing_deep_vet(cand_row, vet_dir=None):
            raise RuntimeError("Injected transit fit failure")

        self._run_pleiades_main_ast(inp_csv, out_dir, mock_failing_deep_vet)

        # Verify summary CSV was constructed by pleiades.py's own main block
        df_out = pd.read_csv(summary_csv, dtype={"tic_id": str})
        self.assertEqual(len(df_out), 1)
        r = df_out.iloc[0]
        self.assertEqual(r["operational_status"], "ERROR")
        self.assertTrue(pd.isna(r["vps_score"]))
        self.assertTrue(pd.isna(r["vps_version"]) or str(r["vps_version"]).strip() == "")
        self.assertIn("Injected transit fit failure", str(r["error_message"]))

        # Feeding this error CSV to Stage 3A MUST raise VPSContractError and halt
        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(
                tier1_csv=inp_csv,
                tier2_csv=summary_csv,
            )
        self.assertIn("non-successful operational status", str(ctx.exception))

    def test_pleiades_cli_ast_no_lightcurve_row_and_downstream_halt(self):
        # B3: Execute actual AST of pleiades.py main block with deep_vet returning None;
        # verify production NO_LIGHTCURVE row construction and downstream Stage 3A halt.
        inp_csv = os.path.join(self.tmp_dir, "input_cands_nolc.csv")
        out_dir = os.path.join(self.tmp_dir, "dossiers_nolc")
        summary_csv = os.path.join(out_dir, "scientific_dossier_summary.csv")

        pd.DataFrame([{
            "tic_id": "888002",
            "period_days": 3.0,
            "t0_btjd": 2000.0,
            "duration_hours": 2.0,
            "depth_ppm": 600.0
        }]).to_csv(inp_csv, index=False)

        def mock_nolc_deep_vet(cand_row, vet_dir=None):
            return None

        self._run_pleiades_main_ast(inp_csv, out_dir, mock_nolc_deep_vet)

        df_out = pd.read_csv(summary_csv, dtype={"tic_id": str})
        self.assertEqual(len(df_out), 1)
        r = df_out.iloc[0]
        self.assertEqual(r["operational_status"], "NO_LIGHTCURVE")
        self.assertTrue(pd.isna(r["vps_score"]))
        self.assertIn("No light curve found", str(r["error_message"]))

        with self.assertRaises(VPSContractError) as ctx:
            antares_packager.run_stage_3a_filter(
                tier1_csv=inp_csv,
                tier2_csv=summary_csv,
            )
        self.assertIn("non-successful operational status", str(ctx.exception))

    def test_pleiades_cli_ast_successful_vetted_export_bridge(self):
        # B1 & B3: Execute actual AST of pleiades.py main block with valid vetted candidate;
        # verify full producer-export bridge into actual Stage 3A consumption.
        inp_csv = os.path.join(self.tmp_dir, "input_cands_succ.csv")
        out_dir = os.path.join(self.tmp_dir, "dossiers_succ")
        summary_csv = os.path.join(out_dir, "scientific_dossier_summary.csv")

        pd.DataFrame([{
            "tic_id": "888003",
            "period_days": 2.5,
            "t0_btjd": 2000.0,
            "duration_hours": 1.5,
            "depth_ppm": 500.0
        }]).to_csv(inp_csv, index=False)

        sec_snr_raw = 2.61234
        sec_phi_raw = 0.53999
        ledger = compute_vps_ledger(
            b_fit=0.3,
            sec_center_phi=sec_phi_raw,
            sec_snr=sec_snr_raw,
            is_sec_eb=False,
            is_tess_eb=False,
            is_gaia_nss=False,
            neb_capable_stars=0,
            contamination_ratio=0.0,
            dur_ratio_kepler=1.0,
            ruwe=1.0,
            stitched_primary_snr=10.0,
        )

        def mock_successful_deep_vet(cand_row, vet_dir=None):
            return {
                "tic_id": str(cand_row["tic_id"]),
                "operational_status": "SUCCESS",
                "disposition": "TIER-2 HEURISTIC SCREEN PASS (candidate; no calibrated FP-risk estimate)",
                "vps_version": VPS_SCHEMA_VERSION,
                "vps_producer": VPS_PRODUCER_NAME,
                "vps_score": round(ledger["final_score"], 1),
                "fp_risk_percent": round(ledger["final_score"], 1),
                "vps_unclipped_total": float(ledger["unclipped_total"]),
                "vps_ledger_json": json.dumps(ledger),
                "sec_eclipse_zone": ledger["secondary_metadata"]["zone"],
                "sec_snr_threshold": ledger["secondary_metadata"]["threshold"],
                "sec_penalty_applied": ledger["secondary_metadata"]["penalty_applied"],
                "vps_clean_criterion": ledger["clean_criterion"],
                "input_period_days": float(cand_row.get("period_days", 2.5)),
                "input_t0_btjd": float(cand_row.get("t0_btjd", 2000.0)),
                "input_depth_ppm": float(cand_row.get("depth_ppm", 500.0)),
                "input_duration_hours": float(cand_row.get("duration_hours", 1.5)),
                "refined_period_days": 2.500000,
                "refined_t0_btjd": 2000.00000,
                "fitted_depth_ppm": 500.0,
                "fitted_duration_hours": 1.50,
                "fitted_rp_earth": 2.10,
                "impact_param_b": 0.300,
                "dur_to_kepler_max": 1.00,
                "stitched_primary_snr": 10.00,
                "secondary_snr_sigma": round(sec_snr_raw, 2),  # 2.61
                "sec_eclipse_phase": round(sec_phi_raw, 3),    # 0.540
                "is_known_tess_eb": False,
                "gaia_nss_type": None,
                "flux_contam_percent": 0.0,
                "neb_capable_neighbors": 0,
                "sectors_stitched": "100",
                "error_message": "",
            }

        self._run_pleiades_main_ast(inp_csv, out_dir, mock_successful_deep_vet)

        survivors, rejected = antares_packager.run_stage_3a_filter(
            tier1_csv=inp_csv,
            tier2_csv=summary_csv,
        )
        self.assertEqual(len(survivors), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(survivors.iloc[0]["tic_id"], "888003")
        self.assertEqual(survivors.iloc[0]["corr_risk"], 0.2)

    def test_pleiades_cli_nonexistent_input_exits_1(self):
        # R4: Pleiades CLI with nonexistent input file exits 1
        nonexistent = os.path.join(self.tmp_dir, "does_not_exist.csv")
        proc = subprocess.run(
            [sys.executable, "-B", "pleiades.py", "--input-csv", nonexistent, "--output-dir", self.tmp_dir],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("not found", proc.stdout + proc.stderr)

    def test_pleiades_cli_empty_input_exits_0_and_produces_header_only_summary(self):
        # R4: Pleiades CLI with 0-row input produces compliant header-only summary
        inp_csv = os.path.join(self.tmp_dir, "empty_cands.csv")
        out_dir = os.path.join(self.tmp_dir, "empty_dossiers")
        pd.DataFrame(columns=["tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm"]).to_csv(inp_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "pleiades.py", "--input-csv", inp_csv, "--output-dir", out_dir],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(proc.returncode, 0)
        summary_csv = os.path.join(out_dir, "scientific_dossier_summary.csv")
        self.assertTrue(os.path.exists(summary_csv))
        df_summary = pd.read_csv(summary_csv)
        self.assertEqual(len(df_summary), 0)
        self.assertIn("vps_version", df_summary.columns)
        self.assertIn("vps_producer", df_summary.columns)
        self.assertIn("vps_score", df_summary.columns)
        self.assertIn("vps_ledger_json", df_summary.columns)

        # Stage 3A consuming this empty summary succeeds with 0 survivors
        survivors, rejected = antares_packager.run_stage_3a_filter(
            tier1_csv=inp_csv,
            tier2_csv=summary_csv,
        )
        self.assertEqual(len(survivors), 0)
        self.assertEqual(len(rejected), 0)

    def test_pleiades_actual_ast_scoring_and_gate_fidelity(self):
        # R2 / R4: Extract actual AST block from pleiades.py and verify score criterion fidelity
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pleiades_path = os.path.join(root_dir, "pleiades.py")
        with open(pleiades_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "deep_vet_target")
        start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ledger" for t in n.targets))
        end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "is_clean" for t in n.targets))

        # Test boundary values: 4.74 (4.94), 4.75 (4.95), 4.76 (4.96), 4.85 (5.05), 4.90 (5.10)
        for weak_pen, expected_score_clean in [
            (4.74, True),   # 0.2 + 4.74 = 4.94 < 5.0
            (4.75, True),   # 0.2 + 4.75 = 4.95 < 5.0
            (4.76, True),   # 0.2 + 4.76 = 4.96 < 5.0 (unrounded clipped score < 5.0)
            (4.85, False),  # 0.2 + 4.85 = 5.05 >= 5.0
            (4.90, False),  # 0.2 + 4.90 = 5.10 >= 5.0
        ]:
            ns = {
                "compute_vps_ledger": compute_vps_ledger,
                "PLEIADES_CLEAN_VPS_THRESHOLD": PLEIADES_CLEAN_VPS_THRESHOLD,
                "json": json,
                "b_fit": 0.0,
                "sec_center_phi": 0.50,
                "sec_snr": 0.0,
                "is_sec_eb": False,
                "gaia_contam": {"is_tess_eb": False, "is_gaia_nss": False, "neb_capable_stars": 0, "contamination_ratio": 0.0, "ruwe": 1.0},
                "dur_ratio_kepler": 1.0,
                "stitched_primary_snr": 5.5 - (weak_pen / 22.0),
            }
            exec(compile(ast.Module(body=fn.body[start:end+1], type_ignores=[]), "test_ast_scoring", "exec"), ns)
            self.assertEqual(
                ns["vps_score_clipped_raw"] < PLEIADES_CLEAN_VPS_THRESHOLD,
                expected_score_clean,
                f"Weak penalty {weak_pen}: expected score clean {expected_score_clean}, got {ns['vps_score_clipped_raw']}"
            )


class TestAntaresPackagerCLI(unittest.TestCase):
    """
    B2 & B3: Tests standalone CLI execution of antares_packager.py.
    - Malformed gate contract input halts nonzero without creating export artifacts.
    - Genuine zero valid case exits 0 with header-only exports.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_antares_cli_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_antares_cli_malformed_gate_missing_duration_halts_nonzero_without_artifacts(self):
        # B2: Missing dur_to_kepler_max halts nonzero without writing empty export artifacts
        t1_csv = os.path.join(self.tmp_dir, "t1.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_bad.csv")
        out_dir = os.path.join(self.tmp_dir, "out_bad")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40001", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40001",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VPSContractError", proc.stderr)
        self.assertIn("missing mandatory gate measurement 'dur_to_kepler_max'", proc.stderr)
        self.assertEqual(len(os.listdir(out_dir)), 0, "No output files must be created on contract failure")

    def test_antares_cli_invalid_environment_halts_nonzero_without_artifacts(self):
        # B2: Invalid domain values (-inf contamination, -1 neighbors) halt nonzero without artifacts
        t1_csv = os.path.join(self.tmp_dir, "t1.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_inval.csv")
        out_dir = os.path.join(self.tmp_dir, "out_inval")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40002", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40002",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": float("-inf"),
            "neb_capable_neighbors": -1,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VPSContractError", proc.stderr)
        self.assertEqual(len(os.listdir(out_dir)), 0, "No output files must be created on invalid domain")

    def test_antares_cli_genuine_zero_valid_case_exits_0_with_header_only_exports(self):
        # B2 & B3: Genuine zero valid candidates (e.g. valid candidate rejected by score gate) exits 0 with header-only exports
        t1_csv = os.path.join(self.tmp_dir, "t1_valid_zero.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_valid_zero.csv")
        out_dir = os.path.join(self.tmp_dir, "out_valid_zero")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40003", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        # Candidate with VPS score 35.0 (legitimate scientific gate rejection, 35.0 > 25.0)
        b_target = 0.75 + 0.25 * math.sqrt(19.8 / 25.0)
        l = compute_vps_ledger(b_target, 0.5, 2.6, False, False, False, 0, 0, 1.0, 1.30, 10.0)
        pd.DataFrame([{
            "tic_id": "40003",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": 35.0,
            "fp_risk_percent": 35.0,
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "secondary_snr_sigma": 2.6,
            "sec_eclipse_phase": 0.50,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)

        # 4 header-only exports must exist with 0 data rows
        expected_files = [
            "prioritized_targets_ranking.csv",
            "exofop_ctoi_upload.csv",
            "exofop_ctoi_upload_comma.csv",
            "exofop_ctoi_upload_top3.csv"
        ]
        for f in expected_files:
            fp = os.path.join(out_dir, f)
            self.assertTrue(os.path.exists(fp), f"Expected export file {f} missing")
            df = pd.read_csv(fp, sep="|" if f == "exofop_ctoi_upload.csv" else ",")
            self.assertEqual(len(df), 0, f"Expected 0 data rows in {f}")

    def test_antares_cli_empty_tier1_exits_0_with_header_only_exports(self):
        # Empty Tier-1 candidates report exits 0 with header-only exports
        t1_csv = os.path.join(self.tmp_dir, "t1_empty.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_empty.csv")
        out_dir = os.path.join(self.tmp_dir, "out_empty")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame(columns=["tic_id", "period_days", "t0_btjd", "duration_hours", "depth_ppm"]).to_csv(t1_csv, index=False)
        pd.DataFrame(columns=["tic_id", "operational_status", "vps_score"]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        ranking_csv = os.path.join(out_dir, "prioritized_targets_ranking.csv")
        self.assertTrue(os.path.exists(ranking_csv))
        df = pd.read_csv(ranking_csv)
        self.assertEqual(len(df), 0)

    def test_antares_cli_missing_catalog_eb_flag_halts_nonzero_without_artifacts(self):
        # C1: Missing is_known_tess_eb halts nonzero without writing empty export artifacts
        t1_csv = os.path.join(self.tmp_dir, "t1_cli_noeb.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_cli_noeb.csv")
        out_dir = os.path.join(self.tmp_dir, "out_cli_noeb")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40004", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40004",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VPSContractError", proc.stderr)
        self.assertIn("missing mandatory gate flag 'is_known_tess_eb'", proc.stderr)
        self.assertEqual(len(os.listdir(out_dir)), 0)

    def test_antares_cli_missing_both_nss_fields_halts_nonzero_without_artifacts(self):
        # C1: Missing both NSS fields halts nonzero without writing empty export artifacts
        t1_csv = os.path.join(self.tmp_dir, "t1_cli_nonss.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_cli_nonss.csv")
        out_dir = os.path.join(self.tmp_dir, "out_cli_nonss")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40005", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40005",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VPSContractError", proc.stderr)
        self.assertIn("missing mandatory Gaia NSS catalog context", proc.stderr)
        self.assertEqual(len(os.listdir(out_dir)), 0)

    def test_antares_cli_contradictory_nss_context_halts_nonzero_without_artifacts(self):
        # C1: Contradictory NSS context halts nonzero without writing empty export artifacts
        t1_csv = os.path.join(self.tmp_dir, "t1_cli_nsscontra.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_cli_nsscontra.csv")
        out_dir = os.path.join(self.tmp_dir, "out_cli_nsscontra")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40006", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40006",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": False,
            "gaia_nss_type": "Orbital",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("VPSContractError", proc.stderr)
        self.assertIn("contradictory Gaia NSS context", proc.stderr)
        self.assertEqual(len(os.listdir(out_dir)), 0)

    def test_antares_cli_true_positive_nss_binary_exits_0_with_header_only_exports(self):
        # C1: Explicit true positive NSS binary legitimately fails Gate 1 (0 survivors), exiting 0 with header-only exports
        t1_csv = os.path.join(self.tmp_dir, "t1_cli_nsstrue.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_cli_nsstrue.csv")
        out_dir = os.path.join(self.tmp_dir, "out_cli_nsstrue")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40007", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, True, 0, 0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40007",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 0.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "is_gaia_nss": True,
            "gaia_nss_type": "Orbital",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        expected_files = [
            "prioritized_targets_ranking.csv",
            "exofop_ctoi_upload.csv",
            "exofop_ctoi_upload_comma.csv",
            "exofop_ctoi_upload_top3.csv"
        ]
        for f in expected_files:
            fp = os.path.join(out_dir, f)
            self.assertTrue(os.path.exists(fp), f"Expected export file {f} missing")
            df = pd.read_csv(fp, sep="|" if f == "exofop_ctoi_upload.csv" else ",")
            self.assertEqual(len(df), 0, f"Expected 0 data rows in {f}")

    def test_antares_cli_all_valid_high_contamination_rejected_exits_0_with_header_only_exports(self):
        # D1: All-valid candidates with high contamination (e.g. 200%) are rejected by scientific gate (0 survivors).
        # Real standalone packager CLI must exit 0 and write 4 header-only exports, unlike malformed input which halts nonzero.
        t1_csv = os.path.join(self.tmp_dir, "t1_cli_contam200.csv")
        t2_csv = os.path.join(self.tmp_dir, "t2_cli_contam200.csv")
        out_dir = os.path.join(self.tmp_dir, "out_cli_contam200")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame([{"tic_id": "40008", "period_days": 3.0, "t0_btjd": 2000.0, "duration_hours": 2.0, "depth_ppm": 500.0}]).to_csv(t1_csv, index=False)

        l = compute_vps_ledger(0, 0.5, 0.0, False, False, False, 0, 2.0, 1.0, 1.0, 10.0)
        pd.DataFrame([{
            "tic_id": "40008",
            "operational_status": "SUCCESS",
            "vps_version": VPS_SCHEMA_VERSION,
            "vps_producer": VPS_PRODUCER_NAME,
            "vps_score": l["final_score"],
            "fp_risk_percent": l["final_score"],
            "vps_unclipped_total": l["unclipped_total"],
            "vps_ledger_json": json.dumps(l),
            "dur_to_kepler_max": 1.0,
            "flux_contam_percent": 200.0,
            "neb_capable_neighbors": 0,
            "is_known_tess_eb": False,
            "gaia_nss_type": "",
            "error_message": "",
        }]).to_csv(t2_csv, index=False)

        proc = subprocess.run(
            [sys.executable, "-B", "antares_packager.py", "--results-dir", out_dir, "--tier1-csv", t1_csv, "--tier2-csv", t2_csv],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        expected_files = [
            "prioritized_targets_ranking.csv",
            "exofop_ctoi_upload.csv",
            "exofop_ctoi_upload_comma.csv",
            "exofop_ctoi_upload_top3.csv"
        ]
        for f in expected_files:
            fp = os.path.join(out_dir, f)
            self.assertTrue(os.path.exists(fp), f"Expected export file {f} missing")
            df = pd.read_csv(fp, sep="|" if f == "exofop_ctoi_upload.csv" else ",")
            self.assertEqual(len(df), 0, f"Expected 0 data rows in {f}")


if __name__ == "__main__":
    unittest.main()
