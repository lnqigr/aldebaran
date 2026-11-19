"""Transit geometry, model backends and the independent model check of roadmap S4.4 (ARGON_DECISIONS item 31.5).

Backends, selected by name and never silently:
  "pytransit"  pytransit.QuadraticModel(interpolate=False), 120 s exposure, supersampling 5. This is the accepted
               INJ-CAL-01 injector. It needs numpy below 2, so it runs in the injection environment only.
  "reference"  an exact circular orbit with a quadratic limb-darkened disk integrated numerically (Gauss-Legendre
               in a variable that removes the square-root end points). It shares no code with pytransit and is the
               independent reference of the check; it is also the model of the fixture dry run.

model_check() is Control 3 of the injection calibrations: the mean in-transit deficit recomputed by an independent
rule must agree with the injected depth within 1e-4 (relative), and T14 measured from the instantaneous contacts of
the model must agree with the circular-orbit formula within max(1 s, 1e-3 x T14) (item 31.5). No exact-orbit contact
solver is involved.
"""
import math

import numpy as np

G_SI, MSUN, RSUN = 6.67430e-11, 1.98847e30, 6.957e8
EXPTIME_D, NSAMPLES = 120.0 / 86400.0, 5
CAD_D = 2.0 / 1440.0
K_MIN, K_MAX = 0.005, 0.30
DEPTH_REL_TOL = 1e-4                 # INJ-CAL-01 Controls 3
T14_ABS_TOL_S = 1.0                  # item 31.5: max(1 s, 1e-3 x T14)
T14_REL_TOL = 1e-3
BACKENDS = ("pytransit", "reference")
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ------------------------------------------------------------------------------------------------ geometry
def a_over_rs(p_days, m_star, r_star):
    p = p_days * 86400.0
    return (G_SI * m_star * MSUN * p ** 2 / (4.0 * math.pi ** 2)) ** (1.0 / 3.0) / (r_star * RSUN)


def inclination(b, ars):
    return math.acos(b / ars)


def t14(p, k, b, ars):
    """Circular-orbit total duration (days); nan when the planet does not transit."""
    inc = inclination(b, ars); x = (1.0 + k) ** 2 - b ** 2
    if x <= 0: return float('nan')
    s = math.sqrt(x) / (ars * math.sin(inc))
    return float(p / math.pi * math.asin(min(s, 1.0)))


def t14_tolerance_days(t14_days):
    """Item 31.5: max(1 s, 1e-3 x T14), in days."""
    return max(T14_ABS_TOL_S / 86400.0, T14_REL_TOL * float(t14_days))


# ------------------------------------------------------------------------------------------------ reference model
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)   # 48 nodes already agree with 400 nodes to 1e-12 (relative)
_GL_S = 0.5 * (_GL_X + 1.0)          # nodes on [0, 1]
_GL_SW = 0.5 * _GL_W
_GL_C = 0.5 * (1.0 - np.cos(math.pi * _GL_S))            # r = a + width * _GL_C
_GL_J = 0.5 * math.pi * np.sin(math.pi * _GL_S)          # dr = width * _GL_J ds


def _intensity(r, u1, u2):
    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, None))
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


def occulted_fraction(z, k, u1, u2):
    """Fraction of the stellar flux blocked by a dark disk of radius k (stellar radii) whose centre is at projected
    separation z, for quadratic limb darkening I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2.

    The stellar disk is cut into rings of radius r. A ring lies inside the planet over the angle 2 pi (r <= k - z),
    0 (r <= z - k or r >= z + k) or 2 acos((r^2 + z^2 - k^2) / (2 r z)). The ring integral over the partial range
    [|z - k|, min(1, z + k)] uses r = a + (b - a)(1 - cos(pi s)) / 2, which removes the square-root behaviour of the
    angle and of mu at the end points, with 48-point Gauss-Legendre in s."""
    z = np.abs(np.atleast_1d(np.asarray(z, float)))
    k = float(k)
    total = math.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    out = np.zeros(z.shape)
    hit = z < 1.0 + k
    if not np.any(hit):
        return out
    zz = z[hit]
    a = np.abs(zz - k)
    b = np.minimum(1.0, zz + k)
    width = np.clip(b - a, 0.0, None)
    r = a[:, None] + width[:, None] * _GL_C[None, :]
    jac = width[:, None] * _GL_J[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (r * r + zz[:, None] ** 2 - k * k) / (2.0 * r * zz[:, None])
    cosang = np.where(np.isfinite(cosang), cosang, -1.0)   # z = 0: the whole ring is inside (handled below anyway)
    theta = 2.0 * np.arccos(np.clip(cosang, -1.0, 1.0))
    partial = np.sum(_GL_SW[None, :] * _intensity(r, u1, u2) * r * theta * jac, axis=1)
    # Rings entirely inside the planet (only where the planet covers the stellar centre): r in [0, k - z].
    full = np.zeros(zz.shape)
    cover = zz < k
    if np.any(cover):
        inner = np.clip(k - zz[cover], 0.0, 1.0)
        ri = inner[:, None] * _GL_S[None, :]
        full[cover] = np.sum(_GL_SW[None, :] * _intensity(ri, u1, u2) * ri * 2.0 * math.pi, axis=1) * inner
    out[hit] = (partial + full) / total
    return out


def _separation(times, t0, p, ars, inc):
    phi = 2.0 * math.pi * (np.asarray(times, float) - t0) / p
    x = ars * np.sin(phi)
    y = ars * math.cos(inc) * np.cos(phi)
    return np.sqrt(x * x + y * y), np.cos(phi) > 0.0


def _reference_flux(times, k, ld, t0, p, ars, inc, nsamples, exptime):
    times = np.asarray(times, float)
    offs = (np.arange(nsamples) + 0.5) / nsamples * exptime - 0.5 * exptime if nsamples > 1 else np.zeros(1)
    acc = np.zeros(times.size)
    for o in offs:
        z, front = _separation(times + o, t0, p, ars, inc)
        z = np.where(front, z, np.inf)
        acc += 1.0 - occulted_fraction(z, k, float(ld[0]), float(ld[1]))
    return acc / len(offs)


_PYTRANSIT = None


def pytransit_available():
    try:
        _pytransit_class()
        return True
    except Exception:
        return False


def _pytransit_class():
    global _PYTRANSIT
    if _PYTRANSIT is None:
        from pytransit import QuadraticModel
        _PYTRANSIT = QuadraticModel
    return _PYTRANSIT


def _pytransit_flux(times, k, ld, t0, p, ars, inc, nsamples, exptime):
    tm = _pytransit_class()(interpolate=False)
    tm.set_data(np.asarray(times, float), nsamples=[int(nsamples)], exptimes=[float(exptime)])
    return np.atleast_1d(tm.evaluate(float(k), np.asarray(ld, float), float(t0), float(p), float(ars), float(inc)))


def transit_flux(times, k, ld, t0, p, ars, inc, backend, nsamples=NSAMPLES, exptime=EXPTIME_D):
    """Relative flux of the named backend at `times` (non-finite times give 1). The backend must be named."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown transit model backend {backend!r}; expected one of {BACKENDS}")
    times = np.asarray(times, float)
    out = np.ones(times.size)
    ok = np.isfinite(times)
    if not np.any(ok):
        return out
    fn = _pytransit_flux if backend == "pytransit" else _reference_flux
    out[ok] = fn(times[ok], k, ld, t0, p, ars, inc, nsamples, exptime)
    return out


def mean_deficit(k, ld, p, b, ars, backend, n=2001):
    """delta(k): mean model deficit over |t - T0| <= T14 / 2 (fraction, 2-minute exposure), the trapezoidal integral
    over a uniform grid of one transit divided by T14. Returns (delta, T14)."""
    d = t14(p, k, b, ars)
    if not np.isfinite(d) or d <= 0: return float('nan'), d
    grid = np.linspace(-0.5 * d, 0.5 * d, n)
    return float(_trapz(1.0 - transit_flux(grid, k, ld, 0.0, p, ars, inclination(b, ars), backend), grid) / d), d


# ------------------------------------------------------------------------------------------------ S4.4 model check
def contact_t14(k, ld, p, ars, inc, t14_formula, backend):
    """T14 from the instantaneous contacts of the model: bisection for the last time with a deficit, 80 steps."""
    def inst(x):
        return float(transit_flux(np.array([x]), k, ld, 0.0, p, ars, inc, backend, nsamples=1, exptime=0.0)[0])
    lo, hi = 0.0, 0.5 * t14_formula + 0.01
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if inst(mid) < 1.0 else (lo, mid)
    return 2.0 * 0.5 * (lo + hi)


def model_check(truth, backend):
    """Control 3 for one injected planet. truth: period, k, a_rs, inc, b, t0, t14, depth, ld.
    depth is recomputed as the mean of the model at the midpoints of 5000 equal intervals at absolute times (an
    integration rule independent of mean_deficit); T14 comes from the instantaneous contacts."""
    p, k, ars, inc, b = truth['period'], truth['k'], truth['a_rs'], truth['inc'], truth['b']
    d = truth['t14']
    edges = np.linspace(-0.5 * d, 0.5 * d, 5001)
    grid = truth['t0'] + 0.5 * (edges[1:] + edges[:-1])
    m = transit_flux(grid, k, truth['ld'], truth['t0'], p, ars, inc, backend)
    depth_rel = abs(float(np.mean(1.0 - m)) / truth['depth'] - 1.0)
    geo = t14(p, k, b, ars)
    contacts = contact_t14(k, truth['ld'], p, ars, inc, geo, backend)
    diff_d = abs(contacts - geo)
    tol_d = t14_tolerance_days(geo)
    return dict(backend=backend, depth_rel=depth_rel, depth_rel_limit=DEPTH_REL_TOL,
                t14_formula_d=geo, t14_contacts_d=contacts, t14_contacts_minus_formula_s=diff_d * 86400.0,
                t14_tolerance_s=tol_d * 86400.0, a_rs=float(ars),
                depth_passed=bool(depth_rel <= DEPTH_REL_TOL), t14_passed=bool(diff_d <= tol_d),
                passed=bool(depth_rel <= DEPTH_REL_TOL and diff_d <= tol_d))


def geometry_grid(periods=(1.0, 1.5, 2.5, 4.0, 7.0, 11.0, 15.0), impact=(0.02, 0.31, 0.62, 0.79),
                  ratios=(0.005, 0.02, 0.05, 0.10, 0.20, 0.30),
                  stars=((0.20, 0.22), (0.60, 0.60), (1.00, 1.00), (1.20, 1.75)), ld=(0.40, 0.20)):
    """Truth records spanning the injected geometry range: period 1 to 15 d, impact parameter 0.02 to 0.79, radius
    ratio 0.005 to 0.30 and host stars (mass, radius in solar units) from an M dwarf to the R* = 1.75 domain edge,
    which gives the smallest a / R*. Non-transiting combinations are left out. The impact values avoid b = 0 and
    b = 1 - k exactly: pytransit 2.2.0 raises ZeroDivisionError when a sample falls exactly on z = 0 or z = 1 - k
    (seen at b = 0.8, k = 0.2), which a draw from a continuous range does not produce."""
    out = []
    for m_star, r_star in stars:
        for p in periods:
            ars = a_over_rs(p, m_star, r_star)
            for b in impact:
                if b >= ars:
                    continue
                for k in ratios:
                    d = t14(p, k, b, ars)
                    if not np.isfinite(d) or d <= 0:
                        continue
                    out.append(dict(period=p, k=k, b=b, a_rs=ars, inc=inclination(b, ars), t0=0.37 * p, t14=d,
                                    ld=list(ld), m_star=m_star, r_star=r_star))
    return out


def model_check_grid(backend, grid=None):
    """model_check over the geometry grid, with depth taken from mean_deficit of the same backend (the quantity the
    joint solve injects). Returns the per-point rows and a summary with the worst cases."""
    rows = []
    for tr in (geometry_grid() if grid is None else grid):
        depth, _ = mean_deficit(tr['k'], tr['ld'], tr['period'], tr['b'], tr['a_rs'], backend)
        rec = model_check(dict(tr, depth=depth), backend)
        rows.append(dict(period=tr['period'], k=tr['k'], b=tr['b'], m_star=tr['m_star'], r_star=tr['r_star'], **rec))
    worst = max(rows, key=lambda r: r['t14_contacts_minus_formula_s'] / r['t14_tolerance_s'])
    return dict(backend=backend, n=len(rows), n_failed=sum(not r['passed'] for r in rows),
                min_a_rs=min(r['a_rs'] for r in rows), max_a_rs=max(r['a_rs'] for r in rows),
                max_t14_diff_s=max(r['t14_contacts_minus_formula_s'] for r in rows),
                max_t14_diff_over_tolerance=worst['t14_contacts_minus_formula_s'] / worst['t14_tolerance_s'],
                worst_t14_case={k: worst[k] for k in ('period', 'k', 'b', 'a_rs', 't14_contacts_minus_formula_s', 't14_tolerance_s')},
                max_depth_rel=max(r['depth_rel'] for r in rows), passed=all(r['passed'] for r in rows), rows=rows)


def compare_backends(truth, times, a="pytransit", b="reference"):
    """Point-by-point difference of two backends for one planet at the given cadence times (diagnostic)."""
    args = (truth['k'], truth['ld'], truth['t0'], truth['period'], truth['a_rs'], truth['inc'])
    fa, fb = transit_flux(times, *args, a), transit_flux(times, *args, b)
    depth = float(np.max(1.0 - fb))
    return dict(max_abs_diff=float(np.max(np.abs(fa - fb))), max_depth=depth,
                max_abs_diff_over_depth=float(np.max(np.abs(fa - fb)) / depth) if depth > 0 else None)
