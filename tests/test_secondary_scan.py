import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aldebaran import scan_full_orbit_secondary_eclipse

def test_secondary_scan():
    np.random.seed(42)
    t = np.linspace(0, 25.0, 18000)
    period = 4.0
    t0 = 1.0
    dur = 0.10
    prim_depth_ppm = 1500.0

    # 1. Light curve with ONLY primary transit (no secondary eclipse)
    f_clean = np.ones_like(t) + np.random.normal(0, 0.0003, size=len(t))
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    f_clean[np.abs(phase) < (0.5 * dur)] -= 0.0015

    is_eb, max_snr, depth_ppm, phi, z_crit = scan_full_orbit_secondary_eclipse(
        t, f_clean, period, t0, dur, prim_depth_ppm
    )
    print(f"Clean planet -> is_eb={is_eb}, max_snr={max_snr:.2f}, z_crit={z_crit:.2f}, best_phi={phi:.2f}")
    assert not is_eb, f"False positive EB rejection on clean planet! SNR={max_snr} > z_crit={z_crit}"

    # 2. Light curve with eccentric secondary eclipse at phase = 0.35 (depth = 600 ppm = 40% of primary)
    f_eb = f_clean.copy()
    phase_0to1 = ((t - t0) % period) / period
    sec_phase_diff = np.abs(((phase_0to1 - 0.35 + 0.5) % 1.0) - 0.5)
    f_eb[sec_phase_diff < (0.5 * dur / period)] -= 0.000600

    is_eb2, max_snr2, depth_ppm2, phi2, z_crit2 = scan_full_orbit_secondary_eclipse(
        t, f_eb, period, t0, dur, prim_depth_ppm
    )
    print(f"Eccentric EB (phi=0.35) -> is_eb={is_eb2}, max_snr={max_snr2:.2f}, depth={depth_ppm2:.0f}ppm, best_phi={phi2:.2f}")
    assert is_eb2, "Failed to identify eccentric eclipsing binary!"
    assert abs(phi2 - 0.35) < 0.03, f"Phase mismatch: {phi2} vs 0.35"

    print("[SUCCESS] Full-orbit secondary eclipse scanner passed 100%!")

if __name__ == "__main__":
    test_secondary_scan()
