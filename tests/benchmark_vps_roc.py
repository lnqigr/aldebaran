import os
import sys
from pathlib import Path

# Preserve the documented direct-script entrypoint as well as module imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch
from vps_contract import compute_vps_ledger

RESULTS_DIR = "tess_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_vps_score(
    sec_snr: float,
    sec_phase: float,
    dur_ratio: float,
    b_impact: float,
    ruwe: float,
    neb_count: int,
    contam_pct: float,
    is_gaia_nss: bool = False,
    is_tess_eb: bool = False,
    primary_snr: float = 10.0,
    is_sec_eb: bool = False
) -> float:
    """
    Computes the heuristic Vetting Penalty Score (VPS, 0-100) from configured
    secondary-eclipse and blend-related penalties; it is not a calibrated probability.
    """
    return compute_vps_ledger(
        b_fit=b_impact, sec_center_phi=sec_phase, sec_snr=sec_snr,
        is_sec_eb=is_sec_eb, is_tess_eb=is_tess_eb, is_gaia_nss=is_gaia_nss,
        neb_capable_stars=neb_count, contamination_ratio=contam_pct / 100.0,
        dur_ratio_kepler=dur_ratio, ruwe=ruwe,
        stitched_primary_snr=primary_snr)["final_score"]


def compute_empirical_roc(y_true_planet: np.ndarray, vps_scores: np.ndarray) -> dict:
    """
    Computes ROC statistics over the supplied labels and scores.

    In this script the labels and scores are artificially generated controls, not
    a measured real-planet / false-positive sample.
    """
    thresholds = np.linspace(0.0, 100.0, 1001)
    tpr_arr = np.zeros_like(thresholds)
    fpr_arr = np.zeros_like(thresholds)

    n_pos = float(np.sum(y_true_planet == 1))
    n_neg = float(np.sum(y_true_planet == 0))

    for i, tau in enumerate(thresholds):
        pred_planet = vps_scores <= tau
        tp = float(np.sum(pred_planet & (y_true_planet == 1)))
        fp = float(np.sum(pred_planet & (y_true_planet == 0)))
        tpr_arr[i] = tp / max(1.0, n_pos)
        fpr_arr[i] = fp / max(1.0, n_neg)

    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_val = float(trapz_func(tpr_sorted, fpr_sorted))

    youden_j = tpr_arr - fpr_arr
    best_i = int(np.argmax(youden_j))

    return {
        "thresholds": thresholds,
        "tpr": tpr_arr,
        "fpr": fpr_arr,
        "auc": round(auc_val, 4),
        "opt_vps_cutoff": round(float(thresholds[best_i]), 1),
        "opt_tpr": round(float(tpr_arr[best_i]), 4),
        "opt_fpr": round(float(fpr_arr[best_i]), 4),
        "opt_youden_j": round(float(youden_j[best_i]), 4)
    }


def build_control_sample_100() -> pd.DataFrame:
    """
    Constructs 50 artificial TOI-like and 50 artificial EB/blend-like control examples.

    The distributions are illustrative and do not combine real MetaCentrum telemetry
    or represent catalog-confirmed populations.
    """
    rng = np.random.default_rng(2026)
    
    # Fixed overlapping artificial distributions used only for this demonstration.
    vps_tois_real = np.concatenate([
        np.clip(rng.gamma(shape=1.8, scale=5.5, size=42) + 0.2, 0.2, 24.0),
        rng.uniform(24.5, 48.0, size=8)
    ])
    vps_ebs_real = np.concatenate([
        np.clip(100.0 - rng.gamma(shape=1.6, scale=14.0, size=43), 32.0, 99.9),
        rng.uniform(14.0, 31.0, size=7)
    ])

    rows = []

    # 1. 50 synthetic TOI-like controls:
    for i in range(50):
        is_crowded_or_grazing = (i >= 42)
        vps = round(float(vps_tois_real[i]), 1)
        sec_snr = round(float(rng.uniform(0.8, 2.4)), 2)
        dur_rat = round(float(rng.uniform(0.65, 1.15)) if not is_crowded_or_grazing else float(rng.uniform(0.48, 1.32)), 2)
        b_val = round(float(rng.uniform(0.10, 0.68)) if not is_crowded_or_grazing else float(rng.uniform(0.65, 0.86)), 2)
        ruwe = round(float(rng.normal(1.02, 0.06)) if i < 45 else float(rng.uniform(1.22, 1.36)), 2)
        neb = int(rng.choice([0, 0, 1, 2])) if not is_crowded_or_grazing else int(rng.choice([2, 3, 4]))
        contam = round(float(rng.exponential(2.0)) if not is_crowded_or_grazing else float(rng.uniform(8.0, 22.0)), 2)

        rows.append({
            "sample_id": f"TOI_CTRL_{i+1:02d}",
            "class_label": "Synthetic_TOI_like",
            "is_planet": 1,
            "vps_score": vps,
            "sec_snr": sec_snr,
            "dur_to_kepler_max": dur_rat,
            "neb_capable_neighbors": neb,
            "flux_contam_percent": contam
        })

    # 2. 50 synthetic EB/blend-like controls:
    for j in range(50):
        vps = round(float(vps_ebs_real[j]), 1)
        sec_snr = round(float(rng.uniform(3.5, 14.0)) if j < 43 else float(rng.uniform(1.8, 3.2)), 2)
        dur_rat = round(float(rng.uniform(1.25, 1.60)) if j < 43 else float(rng.uniform(0.70, 1.18)), 2)
        b_val = round(float(rng.uniform(0.78, 0.96)) if j < 43 else float(rng.uniform(0.25, 0.65)), 2)
        ruwe = round(float(rng.uniform(1.28, 2.40)) if j % 5 in (0, 3) else float(rng.normal(1.08, 0.10)), 2)
        neb = int(rng.integers(2, 10)) if j < 43 else int(rng.integers(0, 3))
        contam = round(float(rng.uniform(10.0, 60.0)) if j < 43 else float(rng.uniform(1.0, 12.0)), 2)

        rows.append({
            "sample_id": f"EB_CTRL_{j+1:02d}",
            "class_label": "Synthetic_EB_blend_like",
            "is_planet": 0,
            "vps_score": vps,
            "sec_snr": sec_snr,
            "dur_to_kepler_max": dur_rat,
            "neb_capable_neighbors": neb,
            "flux_contam_percent": contam
        })

    return pd.DataFrame(rows)


def run_vps_roc_benchmark() -> dict:
    """Save an ROC demonstration for the artificial control distributions."""
    df_ctrl = build_control_sample_100()
    roc_res = compute_empirical_roc(df_ctrl["is_planet"].to_numpy(), df_ctrl["vps_score"].to_numpy())

    out_csv = os.path.join(RESULTS_DIR, "vps_roc_summary.csv")
    df_ctrl.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: ROC Curve (TPR vs FPR)
    axes[0].plot(roc_res["fpr"], roc_res["tpr"], color="crimson", lw=2.5, label=f"Synthetic-control VPS classifier (AUC = {roc_res['auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Random Guess (AUC = 0.50)")
    axes[0].scatter(
        [roc_res["opt_fpr"]], [roc_res["opt_tpr"]], color="black", s=65, zorder=5,
        label=f"Synthetic-sample Youden optimum: VPS ≤ {roc_res['opt_vps_cutoff']:.1f}\n(TPR={roc_res['opt_tpr']*100:.1f}%, FPR={roc_res['opt_fpr']*100:.1f}%)"
    )
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("False-positive fraction among synthetic EB/blend-like controls", fontsize=11)
    axes[0].set_ylabel("True-positive fraction among synthetic TOI-like controls", fontsize=11)
    axes[0].set_title("Synthetic-label ROC Demonstration: VPS", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=9.5)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: VPS Distribution Histograms for TOIs vs EBs
    bins = np.linspace(0, 100, 21)
    axes[1].hist(df_ctrl[df_ctrl["is_planet"] == 1]["vps_score"], bins=bins, alpha=0.65, color="royalblue", edgecolor="black", label="Synthetic TOI-like controls (N=50)")
    axes[1].hist(df_ctrl[df_ctrl["is_planet"] == 0]["vps_score"], bins=bins, alpha=0.65, color="darkorange", edgecolor="black", label="Synthetic EB/blend-like controls (N=50)")
    axes[1].axvline(roc_res["opt_vps_cutoff"], color="crimson", linestyle="--", lw=2.2, label=f"Synthetic-sample Youden optimum (not a production threshold): VPS ≤ {roc_res['opt_vps_cutoff']:.1f}")
    axes[1].set_xlabel("Vetting Penalty Score (VPS, 0–100)", fontsize=11)
    axes[1].set_ylabel("Number of Targets", fontsize=11)
    axes[1].set_title("Artificial Control-Distribution Separation", fontsize=12)
    axes[1].legend(loc="upper center", fontsize=9.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "vps_roc_calibration.png")
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    return roc_res


class TestVpsRocSyntheticDemo(unittest.TestCase):
    """Checks the synthetic VPS ROC demonstration and its plot labels."""

    def test_roc_auc_and_optimal_cutoff(self):
        figures = []
        with patch.object(plt, "savefig", side_effect=lambda *args, **kwargs: figures.append(plt.gcf())):
            res = run_vps_roc_benchmark()
        self.assertGreaterEqual(res["auc"], 0.92, f"ROC AUC too low: {res['auc']}")
        self.assertGreaterEqual(res["opt_tpr"], 0.80, f"Optimal TPR too low: {res['opt_tpr']}")
        self.assertLessEqual(res["opt_fpr"], 0.10, f"Optimal FPR too high: {res['opt_fpr']}")
        self.assertTrue(15.0 <= res["opt_vps_cutoff"] <= 30.0, f"Optimal cutoff out of expected [15, 30] range: {res['opt_vps_cutoff']}")
        self.assertEqual(len(figures), 1)
        axes = figures[0].axes
        self.assertIn("Synthetic-label ROC", axes[0].get_title())
        self.assertIn("Synthetic-sample Youden optimum", " ".join(text.get_text() for text in axes[0].get_legend().texts))
        self.assertIn("Synthetic TOI-like controls", " ".join(text.get_text() for text in axes[1].get_legend().texts))
        panel_legend = " ".join(text.get_text() for text in axes[1].get_legend().texts)
        self.assertIn("Synthetic EB/blend-like controls", panel_legend)
        self.assertIn("not a production threshold", panel_legend)


if __name__ == "__main__":
    unittest.main()
