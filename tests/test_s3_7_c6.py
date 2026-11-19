"""S3.7 (roadmap C6: SCI-07, SCI-17, SCI-23, SCI-32, SCI-33).

Pins the corrected labels, heuristic documentation, unchanged vps_v1 ledger
contract, and Pleiades is_clean AND semantics (catching AND->OR mutants).
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aldebaran
import antares_packager
import pleiades
import test_char_pleiades_deep_vet as vet
import test_s3_9_c1fg
import vps_contract


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestS37LabelsAndDocs(unittest.TestCase):
    def test_pleiades_clean_gate_label_lists_all_and_conditions(self):
        label = pleiades.PLEIADES_CLEAN_GATE_LABEL
        self.assertIn("Pleiades Clean Criterion < 5.0", label)
        for token in (
            "AND b < 0.82",
            "AND sec_snr <= sec_snr_threshold",
            "AND not is_sec_eb",
            "AND not is_tess_eb",
            "AND not is_gaia_nss",
            "AND 0.35 <= dur_ratio <= 1.38",
            "AND primary_snr >= 5.5",
        ):
            self.assertIn(token, label)
        self.assertNotIn("z_crit", label)
        self.assertNotIn("HIGH-CONFIDENCE", label.upper())
        self.assertNotIn("CALIBRATED", label.upper())

    def test_vps_ledger_contract_unchanged(self):
        self.assertEqual(vps_contract.VPS_SCHEMA_VERSION, "vps_v1")
        self.assertEqual(vps_contract.VPS_PRODUCER_NAME, "pleiades")
        self.assertEqual(vps_contract.PLEIADES_CLEAN_VPS_THRESHOLD, 5.0)
        self.assertEqual(vps_contract.PLEIADES_CLEAN_CRITERION_LABEL, "vps_score < 5.0")
        self.assertEqual(vps_contract.ANTARES_DEFAULT_MAX_CORR_RISK, 25.0)

        ledger = vps_contract.compute_vps_ledger(
            b_fit=0.3,
            sec_center_phi=0.5,
            sec_snr=0.5,
            is_sec_eb=False,
            is_tess_eb=False,
            is_gaia_nss=False,
            neb_capable_stars=0,
            contamination_ratio=0.0,
            dur_ratio_kepler=1.0,
            ruwe=1.0,
            stitched_primary_snr=12.0,
        )
        self.assertEqual(
            list(ledger.keys()),
            [
                "version",
                "producer",
                "components",
                "secondary_metadata",
                "unclipped_total",
                "clip_bounds",
                "clipped_score",
                "final_score",
                "clean_criterion",
            ],
        )
        self.assertEqual(ledger["version"], "vps_v1")
        self.assertEqual(ledger["clean_criterion"], "vps_score < 5.0")
        self.assertEqual(ledger["final_score"], 0.2)

    def test_multi_sde_schema_preserved_and_documented_as_heuristic(self):
        self.assertIn("multi_sde", aldebaran.CANDIDATE_COLUMNS)
        self.assertIn("est_multi_sde_proxy", aldebaran.CANDIDATE_COLUMNS)
        with open(os.path.join(REPO_ROOT, "aldebaran.py"), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("Heuristic multi-sector SDE proxy (SCI-07)", src)

    def test_readme_documents_s37_qualifications(self):
        with open(os.path.join(REPO_ROOT, "README.md"), "r", encoding="utf-8") as f:
            readme = f.read()
        self.assertIn("multi_sde", readme)
        self.assertIn("heuristic", readme.lower())
        self.assertIn("uncalibrated heuristic penalty score", readme)
        self.assertIn("is_clean", readme)
        self.assertIn("25.0", readme)


class TestS37PleiadesAndAntaresPaths(unittest.TestCase):
    def test_pleiades_clean_pass_prints_and_draws_and_label(self):
        result, dossier, console = test_s3_9_c1fg.run_vet_rendered(vet.two_sectors(), vet.candidate())
        self.assertTrue(result["disposition"].startswith("TIER-2 HEURISTIC SCREEN PASS"))
        self.assertIn(pleiades.PLEIADES_CLEAN_GATE_LABEL, console)
        self.assertIn("heuristic points, not a probability", console)
        self.assertNotIn("HIGH-CONFIDENCE", console.upper())
        self.assertIn("Pleiades Clean Criterion < 5.0", dossier)
        self.assertIn("AND b < 0.82", dossier)
        self.assertIn("AND sec_snr <= sec_snr_threshold", dossier)
        self.assertNotIn("sec_snr <= z_crit", dossier)

    def test_pleiades_is_clean_requires_all_and_conditions(self):
        # Negative control against AND->OR mutation in pleiades.is_clean:
        # When vps_score_clipped_raw < 5.0 is True (0.2) but b_fit >= 0.82 (0.85),
        # is_clean MUST remain False.
        orig_curve_fit = pleiades.curve_fit

        def high_b_curve_fit(*args, **kwargs):
            popt, pcov = orig_curve_fit(*args, **kwargs)
            popt = np.array(popt, dtype=float)
            popt[1] = 0.85  # force b_fit >= 0.82
            return popt, pcov

        with mock.patch.object(
            pleiades,
            "compute_vps_ledger",
            return_value={
                "version": "vps_v1",
                "producer": "pleiades",
                "components": {},
                "secondary_metadata": {"zone": "circular", "threshold": 2.8, "penalty_applied": 0.0, "snr": 0.5, "phase": 0.5},
                "unclipped_total": 0.2,
                "clip_bounds": [0.1, 99.9],
                "clipped_score": 0.2,
                "final_score": 0.2,
                "clean_criterion": "vps_score < 5.0",
            },
        ), mock.patch.object(pleiades, "curve_fit", side_effect=high_b_curve_fit):
            result, written, _ = vet.run_vet(vet.two_sectors(), vet.candidate())
        self.assertTrue(written)
        self.assertFalse(result["disposition"].startswith("TIER-2 HEURISTIC SCREEN PASS"))
        self.assertEqual(result["disposition"], "SUSPECT / GRAZING OR CONTAMINATED")

    def test_antares_stage_3a_vps_cut_and_wording(self):
        with tempfile.TemporaryDirectory() as tmp:
            t1_csv = os.path.join(tmp, "t1.csv")
            t2_csv = os.path.join(tmp, "t2.csv")
            pd.DataFrame(
                [
                    {"tic_id": "101", "period_days": 4.0, "t0_btjd": 100.0, "depth_ppm": 1000.0, "duration_hours": 2.0},
                    {"tic_id": "102", "period_days": 4.0, "t0_btjd": 100.0, "depth_ppm": 1000.0, "duration_hours": 2.0},
                ]
            ).to_csv(t1_csv, index=False)

            ledger_pass = vps_contract.compute_vps_ledger(
                b_fit=0.2, sec_center_phi=0.5, sec_snr=0.5, is_sec_eb=False,
                is_tess_eb=False, is_gaia_nss=False, neb_capable_stars=0,
                contamination_ratio=0.0, dur_ratio_kepler=1.0, ruwe=1.0,
                stitched_primary_snr=12.0,
            )
            # b_fit=0.95 + ruwe=2.5 + neb=1 -> final_score > 25.0
            ledger_fail = vps_contract.compute_vps_ledger(
                b_fit=0.95, sec_center_phi=0.5, sec_snr=0.5, is_sec_eb=False,
                is_tess_eb=False, is_gaia_nss=False, neb_capable_stars=1,
                contamination_ratio=0.15, dur_ratio_kepler=1.0, ruwe=2.5,
                stitched_primary_snr=12.0,
            )

            def row_dict(tic_id, ledger):
                return {
                    "tic_id": tic_id,
                    "operational_status": "SUCCESS",
                    "fit_status": "CONVERGED",
                    "disposition": "TIER-2 HEURISTIC SCREEN PASS (candidate; no calibrated FP-risk estimate)",
                    "vps_version": "vps_v1",
                    "vps_producer": "pleiades",
                    "vps_score": ledger["final_score"],
                    "fp_risk_percent": ledger["final_score"],
                    "vps_unclipped_total": ledger["unclipped_total"],
                    "vps_ledger_json": json.dumps(ledger),
                    "sec_eclipse_zone": ledger["secondary_metadata"]["zone"],
                    "sec_snr_threshold": ledger["secondary_metadata"]["threshold"],
                    "sec_penalty_applied": ledger["secondary_metadata"]["penalty_applied"],
                    "vps_clean_criterion": ledger["clean_criterion"],
                    "dur_to_kepler_max": 1.0,
                    "flux_contam_percent": 0.0,
                    "neb_capable_neighbors": 0,
                    "is_known_tess_eb": False,
                    "is_gaia_nss": False,
                    "gaia_nss_type": "",
                    "gaia_nss_period": np.nan,
                    "refined_period_days": 4.0,
                    "fitted_rp_earth": 2.0,
                    "sectors_stitched": "1,2",
                }

            pd.DataFrame([row_dict("101", ledger_pass), row_dict("102", ledger_fail)]).to_csv(t2_csv, index=False)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                survivors, rejected = antares_packager.run_stage_3a_filter(t1_csv, t2_csv)

            self.assertEqual(list(survivors["tic_id"]), ["101"])
            self.assertEqual(list(rejected["tic_id"]), ["102"])
            out = buf.getvalue()
            self.assertIn("VPS heuristic points <= 25.0", out)
            self.assertIn("uncalibrated, not a probability", out)
            self.assertNotIn("HIGH-CONFIDENCE", out.upper())


if __name__ == "__main__":
    unittest.main()
