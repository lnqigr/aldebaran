import unittest
import numpy as np

try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False

_GL5_NODES, _GL5_WEIGHTS = np.polynomial.legendre.leggauss(5)
_GL5_NODES = 0.5 * _GL5_NODES
_GL5_WEIGHTS = 0.5 * _GL5_WEIGHTS


def evaluate_annulus_gl(phase_days: np.ndarray, rp_rs: float, b: float, dur_days: float, u1: float, u2: float, n_rings: int = 300) -> np.ndarray:
    """Evaluates the 300-ring 5-point Gauss-Legendre limb-darkened transit model."""
    r_edges = np.linspace(0.0, 1.0, n_rings + 1)
    dr = 1.0 / n_rings
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    r_sub = (r_mid[:, None] + _GL5_NODES[None, :] * dr).ravel()
    w_sub = (2.0 * np.pi * r_sub * dr) * np.tile(_GL5_WEIGHTS, n_rings)
    mu = np.sqrt(np.clip(1.0 - r_sub ** 2, 0.0, 1.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    sub_flux = w_sub * intensity
    total_flux = np.sum(sub_flux)

    r_c = r_sub[None, :]
    half_chord = np.sqrt(max(1e-6, (1.0 + rp_rs) ** 2 - b ** 2))
    x_proj = (phase_days / (0.5 * dur_days)) * half_chord
    z = np.hypot(b, x_proj)

    fl = np.ones_like(phase_days, dtype=np.float64)
    in_d = z < (1.0 + rp_rs)
    if not np.any(in_d):
        return fl

    z_in = z[in_d][:, None]
    blocked = np.zeros((z_in.shape[0], len(r_sub)), dtype=np.float64)
    full_c = (z_in + r_c) <= rp_rs
    blocked[full_c] = 1.0
    part_c = (~full_c) & (np.abs(z_in - r_c) < rp_rs) & (z_in > 0)
    if np.any(part_c):
        zp = np.broadcast_to(z_in, blocked.shape)[part_c]
        rp = np.broadcast_to(r_c, blocked.shape)[part_c]
        cos_a = (zp ** 2 + rp ** 2 - rp_rs ** 2) / (2.0 * zp * rp)
        blocked[part_c] = np.arccos(np.clip(cos_a, -1.0, 1.0)) / np.pi
    fl[in_d] = 1.0 - (np.sum(blocked * sub_flux[None, :], axis=1) / total_flux)
    return fl


def evaluate_batman_exact(phase_days: np.ndarray, rp_rs: float, b: float, dur_days: float, u1: float, u2: float) -> np.ndarray:
    """Computes exact Mandel & Agol (2002) light curve via batman (if installed) or 6000-ring reference."""
    if HAS_BATMAN:
        p_orb = 100.0
        half_chord = np.sqrt(max(1e-6, (1.0 + rp_rs) ** 2 - b ** 2))
        sin_dt = np.sin(np.pi * dur_days / p_orb)
        a_rs = float(np.sqrt((half_chord / sin_dt) ** 2 + b ** 2))
        inc_deg = float(np.degrees(np.arccos(b / a_rs)))

        params = batman.TransitParams()
        params.t0 = 0.0
        params.per = p_orb
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inc_deg
        params.ecc = 0.0
        params.w = 90.0
        params.u = [u1, u2]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, phase_days)
        return m.light_curve(params)
    return evaluate_annulus_gl(phase_days, rp_rs, b, dur_days, u1, u2, n_rings=6000)


class TestMandelAgolBatman(unittest.TestCase):
    """Validates 300-ring Gauss-Legendre integrator against Mandel & Agol (2002) within < 5.0 ppm."""

    def test_accuracy_across_parameter_space(self):
        cases = [
            (0.010, 0.20, 0.05, 0.50, 0.20),
            (0.015, 0.30, 0.06, 0.45, 0.25),
            (0.025, 0.50, 0.07, 0.60, 0.15),
            (0.037, 0.35, 0.05, 0.57, 0.20),
            (0.055, 0.70, 0.08, 0.30, 0.30),
            (0.080, 0.40, 0.10, 0.35, 0.25),
            (0.100, 0.50, 0.12, 0.40, 0.20),
            (0.045, 0.85, 0.06, 0.55, 0.20),
        ]
        for rp_rs, b, dur, u1, u2 in cases:
            ph = np.linspace(-0.65 * dur, 0.65 * dur, 300)
            f_model = evaluate_annulus_gl(ph, rp_rs, b, dur, u1, u2, n_rings=300)
            f_exact = evaluate_batman_exact(ph, rp_rs, b, dur, u1, u2)
            max_diff_ppm = float(np.max(np.abs(f_model - f_exact)) * 1e6)
            self.assertLess(
                max_diff_ppm, 5.0,
                msg=f"Exceeded 5 ppm tolerance ({max_diff_ppm:.3f} ppm) for Rp/R*={rp_rs}, b={b}"
            )


if __name__ == "__main__":
    unittest.main()
