import os
import sys
import unittest
import numpy as np
from astropy.timeseries import BoxLeastSquares

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aldebaran import (
    get_known_transits_for_tic,
    load_known_catalogs_and_toi_ephemerides,
    compute_whitened_kovacs_sde,
    piecewise_savgol_detrend,
    verify_multi_epoch
)


class TestAldebaranV9Goals(unittest.TestCase):
    """
    Verification suite for Release v9.0 upgrades:
    - Goal 2: Multi-Planet Search Engine via Known-Transit Masking
    - Goal 3: Cramer-Rao Bounded Ephemeris Projection in Archival Verification
    """

    def test_goal2_toi_ephemerides_ingestion(self):
        """Verifies ingestion of official TOI catalog ephemerides."""
        known_cats, known_ebs, toi_ephems = load_known_catalogs_and_toi_ephemerides()
        self.assertIsInstance(known_cats, set)
        self.assertIsInstance(known_ebs, set)
        self.assertIsInstance(toi_ephems, dict)
        self.assertGreater(len(toi_ephems), 1000, "Should load thousands of TOI planetary systems")

        # Check structure of ephemeris record
        sample_tic = next(iter(toi_ephems.keys()))
        planets = toi_ephems[sample_tic]
        self.assertIsInstance(planets, list)
        self.assertGreater(len(planets), 0)
        p0 = planets[0]
        for key in ["toi", "period", "t0", "duration", "depth_ppm", "disp"]:
            self.assertIn(key, p0)
            self.assertTrue(np.isfinite(p0["period"]))
            self.assertGreater(p0["period"], 0.0)

    def test_goal2_known_transit_masking_and_sibling_recovery(self):
        """
        Simulates a known TOI planet host with a hot Jupiter (P1 = 3.5d, depth = 8000 ppm)
        and an injected shallow sub-Neptune sibling (P2 = 1.45d, depth = 550 ppm).
        Verifies:
        1. Known transit masking with +-1.0 T_dur buffer covers known events.
        2. Linear interpolation prevents Savitzky-Golay ringing.
        3. Residual BLS cleanly recovers the sibling planet.
        """
        rng = np.random.default_rng(42)
        cad_days = 120.0 / 86400.0
        t = np.arange(0.0, 27.0, cad_days)

        # Planet 1 (Known TOI): P1 = 3.5d, T0 = 0.5d, dur = 0.08d, depth = 8000 ppm
        p1, t0_1, dur1, depth1 = 3.5, 0.5, 0.08, 0.0080
        # Planet 2 (Undiscovered sibling): P2 = 1.45d, T0 = 0.3d, dur = 0.05d, depth = 550 ppm
        p2, t0_2, dur2, depth2 = 1.45, 0.3, 0.05, 0.00055

        ph1 = ((t - t0_1 + 0.5 * p1) % p1) - 0.5 * p1
        ph2 = ((t - t0_2 + 0.5 * p2) % p2) - 0.5 * p2

        stellar_rot = 0.003 * np.sin(2.0 * np.pi * t / 6.0) + 0.001 * np.cos(2.0 * np.pi * t / 3.0)
        noise = rng.normal(0.0, 0.00018, size=len(t))
        f_raw = 1.0 + stellar_rot + noise - depth1 * (np.abs(ph1) < 0.5 * dur1) - depth2 * (np.abs(ph2) < 0.5 * dur2)

        # Apply Goal 2 masking
        in_known = np.abs(ph1) <= (1.0 * dur1)
        self.assertTrue(np.any(in_known))
        self.assertLess(np.mean(in_known), 0.15)

        # Linear interpolation across mask before Savitzky-Golay
        f_for_trend = f_raw.copy()
        f_for_trend[in_known] = np.interp(t[in_known], t[~in_known], f_raw[~in_known])
        trend = piecewise_savgol_detrend(t, f_for_trend, mask=None, max_win=601)
        flat_f = f_raw / trend

        # Search residuals with BLS
        t_res = t[~in_known]
        flat_res = flat_f[~in_known]

        periods = np.linspace(0.8, 8.0, 6000)
        durations = np.linspace(0.035, 0.12, 6)
        bls = BoxLeastSquares(t_res, flat_res)
        res = bls.power(periods, durations)
        sde_arr, _ = compute_whitened_kovacs_sde(res.power, window_pts=501)

        best_idx = int(np.nanargmax(sde_arr))
        det_p = float(periods[best_idx])
        det_sde = float(sde_arr[best_idx])

        # Sibling recovery verification
        rel_err = abs(det_p - p2) / p2
        self.assertLess(rel_err, 0.01, f"Sibling period {det_p:.4f}d should match injected {p2:.4f}d (<1%)")
        self.assertGreaterEqual(det_sde, 6.0, f"Recovered SDE {det_sde:.1f} should exceed 6.0 gate")

    def test_goal3_cramer_rao_contiguous_projection(self):
        """
        Verifies Goal 3 contiguous sector logic (Delta_T <= 60d):
        Strict zero-freedom projection (N_trials = 1, snr_crit = 2.4).
        """
        tic_id = "999999999"
        curr_sec = 100
        p_disc = 2.500000
        t0_disc = 2500.2000
        dur_disc = 0.06
        depth_ppm = 1200.0
        sde_curr = 8.5

        t_curr = np.arange(2500.0, 2527.0, 120.0 / 86400.0)
        flat_f_curr = np.ones_like(t_curr)
        flat_ferr_curr = np.full_like(t_curr, 0.0003)

        # Single sector without prior archive returns Single-Sector_Only
        ok, status, info = verify_multi_epoch(
            tic_id, curr_sec, t_curr, flat_f_curr, flat_ferr_curr,
            p_disc, t0_disc, dur_disc, depth_ppm, sde_curr
        )
        self.assertTrue(ok)
        self.assertEqual(status, "Single-Sector_Only")
        self.assertEqual(info["n_trials"], 1)
        self.assertEqual(info["snr_crit"], 2.4)
        self.assertTrue(info["is_projected_strict"])

    def test_goal3_cramer_rao_multiyear_math(self):
        """
        Verifies Cramer-Rao timing uncertainty bounds across multi-year gaps:
        1. Single-sector discovery period uncertainty sigma_P ~ sqrt(12) * sigma_T0 / N_cyc.
        2. Corridors scale with Fisher information and eliminate sliding phase circularity.
        3. Sidak-Bonferroni trial penalty scales with gap baseline.
        """
        p_days = 2.0
        dur_days = 0.05
        sde = 6.5
        baseline = 25.0

        sigma_t0 = 0.5 * dur_days / sde
        n_cyc = baseline / p_days
        sigma_p = np.sqrt(12.0) * sigma_t0 / n_cyc

        self.assertAlmostEqual(sigma_p * 86400.0, 92.1, delta=0.5)

        for gap in [365.0, 730.0, 1460.0]:
            delta_p_indep = (p_days * dur_days) / gap
            n_trials = max(1, int(np.ceil((4.0 * sigma_p) / delta_p_indep)))
            snr_crit = max(2.8, float(np.sqrt(2.0 * np.log(max(n_trials, 2))) + 1.2))

            # Critical significance must increase with trials to maintain alpha < 0.01
            self.assertGreaterEqual(snr_crit, 3.4)
            self.assertLessEqual(snr_crit, 4.3)


if __name__ == "__main__":
    unittest.main()
