"""
Test suite script: Quantifies discretization error and numerical jaggedness
of the concentric N-ring limb-darkened transit model versus a quasi-continuous
20,000-ring Mandel-Agol reference integration.
"""

import numpy as np


def ring_model(
    phase: np.ndarray,
    rp_rs: float,
    b: float,
    dur_total: float,
    u1: float,
    u2: float,
    n_rings: int = 60,
) -> np.ndarray:
    half_chord = np.sqrt(max(1e-6, (1.0 + rp_rs) ** 2 - b ** 2))
    z = np.hypot(b, (phase / (0.5 * dur_total)) * half_chord)
    flux = np.ones_like(phase, dtype=np.float64)
    in_disk = z < (1.0 + rp_rs)

    r_edges = np.linspace(0.0, 1.0, n_rings + 1)
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    mu = np.sqrt(np.clip(1.0 - r_c ** 2, 0.0, 1.0))
    ring_flux = areas * (1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2)

    zi = z[in_disk][:, None]
    rc = r_c[None, :]
    frac = np.zeros((zi.shape[0], n_rings))

    full = (zi + rc) <= rp_rs
    frac[full] = 1.0

    part = (~full) & (np.abs(zi - rc) < rp_rs) & (zi > 0)
    zp = np.broadcast_to(zi, frac.shape)[part]
    rp = np.broadcast_to(rc, frac.shape)[part]
    frac[part] = np.arccos(np.clip((zp ** 2 + rp ** 2 - rp_rs ** 2) / (2.0 * zp * rp), -1.0, 1.0)) / np.pi

    flux[in_disk] = 1.0 - (frac * ring_flux).sum(axis=1) / ring_flux.sum()
    return flux


def benchmark_accuracy():
    print(f"{'Rp/R*':>7} {'~Rp(R_E)':>9} {'Depth ppm':>11} {'Max Err ppm':>13} {'Err/Depth':>11} {'Step Jagged ppm':>17}")
    print("-" * 75)
    ph = np.linspace(-0.06, 0.06, 2001)

    # Test cases: Giant (17 R_E), Neptune (5.9 R_E), Sub-Neptune (3.3 R_E), Super-Earth (2.0 R_E), Earth (1.0 R_E)
    test_radii = [0.155, 0.054, 0.030, 0.018, 0.009]

    for k in test_radii:
        ref = ring_model(ph, k, b=0.3, dur_total=0.1, u1=0.43, u2=0.26, n_rings=20000)
        m60 = ring_model(ph, k, b=0.3, dur_total=0.1, u1=0.43, u2=0.26, n_rings=60)

        depth_ppm = (1.0 - np.min(ref)) * 1e6
        err_ppm = float(np.abs(m60 - ref).max() * 1e6)
        # Jaggedness: second difference of residuals indicates staircase artifacts that disrupt curve_fit gradient
        jaggedness_ppm = float(np.abs(np.diff(m60 - ref, 2)).max() * 1e6)

        print(
            f"{k:7.3f} {k * 109.1:9.2f} {depth_ppm:11.1f} {err_ppm:13.2f} {err_ppm / depth_ppm:11.2%} {jaggedness_ppm:17.2f}"
        )


if __name__ == "__main__":
    benchmark_accuracy()
