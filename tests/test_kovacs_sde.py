import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from astropy.timeseries import BoxLeastSquares
from aldebaran import compute_whitened_kovacs_sde

def test_whitened_kovacs_sde():
    np.random.seed(42)
    # 25-day light curve with pure white noise (2-min cadence = 18000 points)
    t = np.linspace(0, 25.0, 18000)
    f_noise = 1.0 + np.random.normal(0, 0.001, size=len(t))
    ferr = np.full_like(t, 0.001)

    freqs = np.linspace(1.0 / 12.0, 1.0 / 0.8, 4500)
    periods = (1.0 / freqs)[::-1]
    durations = np.linspace(0.045, 0.18, 10)

    bls = BoxLeastSquares(t, f_noise, dy=ferr)
    res = bls.power(periods, durations)

    sde_arr, _ = compute_whitened_kovacs_sde(res.power, window_pts=501)
    max_noise_sde = float(np.max(sde_arr))
    print(f"Pure White Noise -> Max Whitened SDE: {max_noise_sde:.2f}")
    assert max_noise_sde < 6.0, f"False positive! Pure noise reached SDE={max_noise_sde:.2f} >= 6.0"

    # Inject a 7.5-sigma transit at P = 4.0 days, dur = 0.10 days
    f_planet = f_noise.copy()
    in_tr = np.abs(((t - 1.2 + 2.0) % 4.0) - 2.0) < 0.05
    f_planet[in_tr] -= 0.000400

    bls_pl = BoxLeastSquares(t, f_planet, dy=ferr)
    res_pl = bls_pl.power(periods, durations)
    sde_pl_arr, _ = compute_whitened_kovacs_sde(res_pl.power, window_pts=501)
    max_pl_sde = float(np.max(sde_pl_arr))
    detected_p = float(periods[np.argmax(sde_pl_arr)])
    print(f"Injected Planet -> Max Whitened SDE: {max_pl_sde:.2f} at P={detected_p:.4f} d")
    assert max_pl_sde >= 6.0, f"False negative! Real planet had SDE={max_pl_sde:.2f} < 6.0"
    assert abs(detected_p - 4.0) < 0.05, f"Period mismatch: {detected_p} vs 4.0"

    print("[SUCCESS] Whitened Kovacs SDE test passed 100%!")

if __name__ == "__main__":
    test_whitened_kovacs_sde()
