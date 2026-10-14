import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from astropy.timeseries import BoxLeastSquares

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = "tess_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_whitened_kovacs_sde(power_arr: np.ndarray, window_pts: int = 501) -> np.ndarray:
    """Computes linearized, running-median whitened Kovacs (2002) SDE from Astropy BLS power."""
    sr = np.sqrt(2.0 * np.maximum(0.0, np.asarray(power_arr, dtype=np.float64)))
    win = min(window_pts, (len(sr) // 4) * 2 + 1)
    local_base = median_filter(sr, size=win, mode="nearest") if win >= 11 else np.nanmedian(sr)
    sr_w = sr - local_base
    med_w = float(np.nanmedian(sr_w))
    mad_w = float(1.4826 * np.nanmedian(np.abs(sr_w - med_w)) + 1e-12)
    return (sr_w - med_w) / mad_w


def piecewise_savgol_baseline(t: np.ndarray, f: np.ndarray, max_win: int = 601) -> np.ndarray:
    """Applies piecewise Savitzky-Golay detrending across continuous segments separated by gaps."""
    trend = np.ones_like(f)
    split_idx = np.where(np.diff(t) > 0.04)[0] + 1
    for seg in np.split(np.arange(len(t)), split_idx):
        if len(seg) < 15:
            trend[seg] = np.nanmedian(f[seg]) if len(seg) > 0 else 1.0
            continue
        win = min(max_win, (len(seg) // 2) * 2 - 1)
        trend[seg] = savgol_filter(f[seg], window_length=win, polyorder=2) if win >= 5 else np.nanmedian(f[seg])
    return np.where(np.isfinite(trend) & (trend > 0.1), trend, 1.0)


def generate_realistic_tess_baseline(n_sectors: int = 1, seed: int = 42, sigma_2min_ppm: float = 1200.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a realistic 2-minute cadence TESS baseline with a mid-sector spacecraft downlink gap
    and correlated low-frequency stellar granulation / spot variability.
    """
    rng = np.random.default_rng(seed)
    cad_days = 120.0 / 86400.0
    t_Orbit1 = np.arange(0.15, 13.15, cad_days)
    t_Orbit2 = np.arange(14.25, 27.25, cad_days)
    t = np.concatenate([t_Orbit1, t_Orbit2])

    white_noise = rng.normal(0.0, sigma_2min_ppm * 1e-6, size=len(t))
    # Correlated quasi-periodic stellar variability (P_rot ~ 6.4 d) + red granulation
    red_raw = rng.normal(0.0, 0.45 * sigma_2min_ppm * 1e-6, size=len(t))
    red_smooth = median_filter(red_raw, size=25)
    rot_wave = 0.0008 * np.sin(2.0 * np.pi * t / 6.4) + 0.0003 * np.cos(2.0 * np.pi * t / 3.2)

    f = 1.0 + white_noise + red_smooth + rot_wave
    ferr = np.full_like(t, sigma_2min_ppm * 1e-6)
    return t, f, ferr


def inject_physical_transit(
    t: np.ndarray,
    f: np.ndarray,
    p_days: float,
    t0_days: float,
    rp_earth: float,
    r_star_sun: float = 0.50,
    m_star_sun: float = 0.50,
    b_impact: float = 0.30
) -> tuple[np.ndarray, float, float]:
    """Injects a limb-darkened transit signal into flux array f and returns (f_injected, depth_ppm, dur_days)."""
    G_SI = 6.67430e-11
    M_SUN = 1.98847e30
    R_SUN = 6.957e8
    R_EARTH = 6.371e6

    rp_rs = (rp_earth * R_EARTH) / (r_star_sun * R_SUN)
    p_sec = p_days * 86400.0
    a_m = (G_SI * (m_star_sun * M_SUN) * (p_sec ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    a_rs = a_m / (r_star_sun * R_SUN)
    half_chord = np.sqrt(max(1e-4, (1.0 + rp_rs) ** 2 - b_impact ** 2))
    dur_days = float((p_days / np.pi) * np.arcsin(min(0.99, half_chord / a_rs)))
    dur_days = max(0.040, dur_days)

    ph = ((t - t0_days + 0.5 * p_days) % p_days) - 0.5 * p_days
    in_tr = np.abs(ph) < (0.5 * dur_days)
    f_inj = f.copy()
    if np.any(in_tr):
        x_norm = ph[in_tr] / (0.5 * dur_days)
        z = np.hypot(b_impact, x_norm * half_chord)
        mu = np.sqrt(np.clip(1.0 - np.clip(z, 0.0, 1.0) ** 2, 0.0, 1.0))
        u1, u2 = 0.50, 0.22
        norm_ld = 1.0 - u1 / 3.0 - u2 / 6.0
        ld_prof = (1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2) / norm_ld
        f_inj[in_tr] -= (rp_rs ** 2) * ld_prof

    return f_inj, float((rp_rs ** 2) * 1e6), dur_days


def run_injection_recovery_suite(n_trials_per_cell: int = 8):
    """
    Executes a systematic synthetic transit injection-recovery grid across Period and Planet Radius
    for a representative M/K dwarf target (R* = 0.50 R_Sun, sigma_2min = 1200 ppm) and outputs
    a publication-grade Completeness Heatmap and CSV summary.
    """
    print("=" * 72)
    print("ALDEBARAN v8.2 SYNTHETIC INJECTION-RECOVERY BENCHMARK")
    print("=" * 72)

    p_bins = np.array([1.0, 2.2, 4.0, 6.5, 9.5, 12.0])
    rp_bins = np.array([1.0, 1.4, 1.8, 2.3, 3.0, 4.0])

    freqs = np.linspace(1.0 / 13.0, 1.0 / 0.8, 6000)
    trial_periods = (1.0 / freqs)[::-1]
    durations = np.linspace(0.040, 0.16, 10)

    completeness_matrix = np.zeros((len(rp_bins), len(p_bins)), dtype=np.float64)
    mean_sde_matrix = np.zeros((len(rp_bins), len(p_bins)), dtype=np.float64)
    rows = []

    for i_r, rp_val in enumerate(rp_bins):
        for j_p, p_val in enumerate(p_bins):
            recovered = 0
            sde_list = []
            for k in range(n_trials_per_cell):
                seed_k = 1000 * i_r + 100 * j_p + k
                t, f_base, ferr = generate_realistic_tess_baseline(seed=seed_k, sigma_2min_ppm=1200.0)
                t0_inj = 0.5 + (k / n_trials_per_cell) * p_val
                f_inj, depth_ppm, _ = inject_physical_transit(t, f_base, p_val, t0_inj, rp_val, r_star_sun=0.50, m_star_sun=0.50)

                trend = piecewise_savgol_baseline(t, f_inj, max_win=601)
                f_flat = f_inj / trend

                bls = BoxLeastSquares(t, f_flat, dy=ferr)
                res = bls.power(trial_periods, durations)
                sde_arr = compute_whitened_kovacs_sde(res.power, window_pts=501)

                best_idx = int(np.nanargmax(sde_arr))
                p_det = float(res.period[best_idx])
                sde_det = float(sde_arr[best_idx])
                sde_list.append(sde_det)

                # Recovery criterion: SDE >= 6.0 and period matches injected P (or 1:2 / 2:1 harmonic) within 1.5%
                p_match = (
                    abs(p_det - p_val) / p_val < 0.015
                    or abs(p_det - 0.5 * p_val) / (0.5 * p_val) < 0.015
                    or abs(p_det - 2.0 * p_val) / (2.0 * p_val) < 0.015
                )
                if sde_det >= 6.0 and p_match:
                    recovered += 1

            rec_frac = (recovered / n_trials_per_cell) * 100.0
            mean_sde = float(np.mean(sde_list))
            completeness_matrix[i_r, j_p] = rec_frac
            mean_sde_matrix[i_r, j_p] = mean_sde
            rows.append({
                "rp_earth": rp_val,
                "period_days": p_val,
                "injected_depth_ppm": round(depth_ppm, 1),
                "n_trials": n_trials_per_cell,
                "recovery_rate_percent": round(rec_frac, 1),
                "mean_recovered_sde": round(mean_sde, 2)
            })
            print(f"Cell (Rp={rp_val:4.1f} R_E, P={p_val:4.1f} d, Depth={depth_ppm:5.0f} ppm) -> Recovery: {rec_frac:5.1f}% | Mean SDE: {mean_sde:5.2f}")

    # Render Publication Completeness Heatmap
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(completeness_matrix, origin="lower", cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pipeline Detection Completeness (%)", fontsize=11)

    ax.set_xticks(np.arange(len(p_bins)))
    ax.set_xticklabels([f"{p:.1f}" for p in p_bins])
    ax.set_yticks(np.arange(len(rp_bins)))
    ax.set_yticklabels([f"{r:.1f}" for r in rp_bins])
    ax.set_xlabel("Injected Orbital Period (days)", fontsize=11)
    ax.set_ylabel("Injected Planet Radius ($R_\\oplus$, for $0.50\\,R_\\odot$ Host)", fontsize=11)
    ax.set_title("Aldebaran v8.2 Injection-Recovery Completeness Map (SDE $\\geq 6.0$)", fontsize=12)

    for i in range(len(rp_bins)):
        for j in range(len(p_bins)):
            val = completeness_matrix[i, j]
            sde_v = mean_sde_matrix[i, j]
            txt_color = "white" if val > 55.0 else "black"
            ax.text(j, i, f"{val:.0f}%\n(SDE {sde_v:.1f})", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_img = os.path.join(RESULTS_DIR, "completeness_heatmap.png")
    out_csv = os.path.join(RESULTS_DIR, "injection_recovery_summary.csv")
    plt.savefig(out_img, dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved Completeness Heatmap to: '{out_img}'")
    print(f"Saved Benchmark Table to:      '{out_csv}'")
    print("=" * 72)


if __name__ == "__main__":
    run_injection_recovery_suite(n_trials_per_cell=8)
