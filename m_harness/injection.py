"""M1 injection driver: SNR_inj, the joint depth solve, draws on the declared SNR_inj bins, and injection into
light curve and target pixel files (roadmap 2.I and 0.6; ARGON_DECISIONS items 22, 25, 31.5 and 31.8).

The injection machinery is the accepted INJ-CAL-01 code (agent_exchange/inj_cal_01/inject.py and
run_injections.py), ported with two changes that do not alter a result on valid input: the transit model backend is
named explicitly (transit_model.BACKENDS), and the noise of the SNR definition comes from the production functions
through prodpath instead of INJ-CAL-01's verbatim copies. agent_exchange/s4_m_package/port_equivalence.py compares
the outputs with the accepted code on fixed fixtures.

SNR_inj (the conditioning variable of M1, CROWDSAP-SNR-01 definition):
    SNR_inj = delta x sqrt(N_ev) / sigma_T14
  delta      mean model deficit over |t - T0| <= T14 / 2 (2-minute exposure)
  N_ev       transits with at least half of their expected cadences in the production discovery arrays
  sigma_T14  1.4826 MAD of the means of out-of-transit bins of width T14 (at least 10 bins)
all on the uninjected discovery sector after production extraction, upper clip and detrending.

Bins fixed by item 31.8: SNR_inj [4, 6), [6, 8) and [8, 10]. The declared domain (depth 180 to 30,000 ppm, Rp <= 15 R_E)
bounds every draw. Every other design value (period and impact parameter ranges, T0 conditions, injections per host,
bins above 10, seeds) is a field of InjectionDesign and must come from
the frozen M preregistration; DRAFT_DESIGN carries the values recommended in ARGON_QUEUE and is labelled as a draft.
"""
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage, optimize

from . import matching, prodpath
from .estimands import RP_EARTH_PER_RSUN
from . import transit_model as TM

M1_LOW_SNR_BINS = ((4.0, 6.0), (6.0, 8.0), (8.0, 10.0))     # item 31.8: [4, 6), [6, 8), [8, 10]
GREY_SUPPORT_SNR = (5.0, 9.0)                                 # item 31.8: dC_grey is assessed over SNR_inj 5 to 9
SNR_TOL = 0.05                                                # item 25: |achieved / target - 1| <= 0.05
SOLVE_FIELDS = ('valid', 'stop_reason', 'snr_target', 'snr_achieved', 'snr_residual', 'n_iter', 'min_residual',
                'snr_status', 'noise_n_ev', 'k', 't14', 'depth', 'depth_target', 'sigma_t14', 'n_ev')
STOP_REASONS = ('CONVERGED', 'NOT_CONVERGED_50', 'NOISE_INVALID_START', 'NOISE_INVALID', 'DEPTH_OUTSIDE_K_BRACKET',
                'OUTSIDE_VALID_RANGE', 'MODEL_ERROR')
DEPTH_DOMAIN_PPM = (180.0, 30000.0)                           # declared domain, items 31.2 and 31.9
RP_MAX_EARTH = 15.0                                           # declared domain: Rp <= 15 R_E


@dataclass(frozen=True)
class InjectionDesign:
    """Every design value of the M1 draws. Nothing here has a silent default: a preregistration fills each field."""
    label: str
    backend: str
    snr_bins: tuple
    n_per_bin: int
    period_range_days: tuple
    impact_range: tuple
    min_discovery_events: int
    min_archival_windows: int
    depth_domain_ppm: tuple
    rp_max_earth: float
    seed_base: int
    max_draws: int = 50
    max_t0_tries: int = 20
    status: str = "DRAFT"
    open_questions: tuple = field(default_factory=tuple)

    def as_dict(self):
        return asdict(self)


DRAFT_DESIGN = InjectionDesign(
    label="M1 item 43 design; cohort and release freeze pending",
    backend="pytransit",
    snr_bins=M1_LOW_SNR_BINS + ((10.0, 15.0), (15.0, 25.0), (25.0, 35.0)),
    n_per_bin=1,
    period_range_days=(1.0, 13.0),
    impact_range=(0.0, 0.8),
    min_discovery_events=3,
    min_archival_windows=0,
    depth_domain_ppm=DEPTH_DOMAIN_PPM,
    rp_max_earth=RP_MAX_EARTH,
    seed_base=410000,
    status="DRAFT",
    open_questions=(),  # item 43 fixes the design; the manifest and run authorization remain separate
)


def _in_bin(snr, lo, hi, closed_at):
    """Bins are half-open [lo, hi), except that an edge listed in closed_at belongs to the bin below it:
    with closed_at = (10,), 10.0 is in [8, 10] and the next bin is (10, 15]."""
    left = snr > lo if lo in closed_at else snr >= lo
    right = snr <= hi if hi in closed_at else snr < hi
    return left and right


def snr_bin(snr, bins, closed_at=(10.0, 35.0)):
    """Index of the bin holding snr, or None outside every bin. Item 31.8 closes the low-SNR stratum at 10."""
    if snr is None or not math.isfinite(snr):
        return None
    for i, (lo, hi) in enumerate(bins):
        if _in_bin(snr, lo, hi, closed_at):
            return i
    return None


def bin_label(lo, hi, closed_at=(10.0, 35.0)):
    return f"{'(' if lo in closed_at else '['}{lo:g}, {hi:g}{']' if hi in closed_at else ')'}"


# ------------------------------------------------------------------------------------------------ SNR_inj
def rmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size >= 5 else float("nan")


def snr_context(ald, t_raw, f_raw, q_raw):
    """Production discovery arrays for the SNR definition, computed once per host: (t, flat) or None when the
    production extraction keeps too few cadences."""
    try:
        t, flat = prodpath.production_flat(ald, prodpath.ArrayLightCurve(t_raw, f_raw, q_raw))
    except ValueError:
        return None
    return t, flat


def snr_from_context(ctx, P, T0, T14, depth_ppm):
    """SNR of a box of depth depth_ppm at (P, T0, T14) on the flattened production arrays (CROWDSAP-SNR-01)."""
    if ctx is None:
        return dict(status="INSUFFICIENT_CADENCES")
    t, flat = ctx
    cad = float(np.median(np.diff(t)))
    n0 = math.ceil((t[0] - T0) / P); n1 = math.floor((t[-1] - T0) / P)
    mids = T0 + P * np.arange(n0, n1 + 1)
    exp_in = T14 / cad
    n_ev = int(sum(np.sum(np.abs(t - m) <= T14 / 2) >= 0.5 * exp_in for m in mids))
    near = np.min(np.abs(t[:, None] - mids[None, :]), axis=1) if mids.size else np.full(t.size, np.inf)
    oot = near > T14
    to, fo = t[oot], flat[oot]
    edges = np.arange(to.min(), to.max() + T14, T14); idx = np.digitize(to, edges)
    means = [fo[idx == i].mean() for i in np.unique(idx) if np.sum(idx == i) >= 0.5 * exp_in]
    s14 = rmad(means) if len(means) >= 10 else float("nan")
    val = depth_ppm * 1e-6 / s14 * math.sqrt(n_ev) if (n_ev > 0 and s14 == s14 and s14 > 0 and depth_ppm == depth_ppm) else float("nan")
    return dict(status="OK", sigma_T14_ppm=s14 * 1e6, N_ev=n_ev, n_bins=len(means), SNR=val, cadences=int(t.size))


def snr_inj(ald, t_raw, f_raw, q_raw, P, T0, T14, depth_ppm):
    return snr_from_context(snr_context(ald, t_raw, f_raw, q_raw), P, T0, T14, depth_ppm)


# ------------------------------------------------------------------------------------------------ limb darkening
def ld_coefficients(teff, logg, grid):
    """grid: dict(teff=sorted unique, logg=sorted unique, a=2D[i_teff, j_logg], b=2D). Bilinear, clamped and flagged."""
    tt, gg = np.asarray(grid['teff'], float), np.asarray(grid['logg'], float)
    clamped = bool(teff < tt[0] or teff > tt[-1] or logg < gg[0] or logg > gg[-1])
    x, y = float(np.clip(teff, tt[0], tt[-1])), float(np.clip(logg, gg[0], gg[-1]))
    i = int(np.clip(np.searchsorted(tt, x) - 1, 0, len(tt) - 2)); j = int(np.clip(np.searchsorted(gg, y) - 1, 0, len(gg) - 2))
    fx = (x - tt[i]) / (tt[i + 1] - tt[i]); fy = (y - gg[j]) / (gg[j + 1] - gg[j])
    out = []
    for key in ('a', 'b'):
        v = np.asarray(grid[key], float)
        corner = np.array([v[i, j], v[i + 1, j], v[i, j + 1], v[i + 1, j + 1]])
        assert np.all(np.isfinite(corner)), ('LD_GRID_HOLE', teff, logg)
        out.append(float((1 - fx) * (1 - fy) * corner[0] + fx * (1 - fy) * corner[1] + (1 - fx) * fy * corner[2] + fx * fy * corner[3]))
    return out[0], out[1], clamped


# ------------------------------------------------------------------------------------------------ joint solve
def solve_k_for_depth(target, ld, p, b, ars, backend):
    """k with delta(k) = target, by Brent's method on [1e-4, 0.5]; None if outside that bracket."""
    f = lambda k: TM.mean_deficit(k, ld, p, b, ars, backend)[0] - target
    lo, hi = 1e-4, 0.5
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo > 0 or fhi < 0:
        return None
    return float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


def joint_solve(snr_target, b, p, t0, ld, ars, snr_ctx, backend):
    """Item 22 joint solve with the item 25 stop rule: depth target delta*(T14) = SNR_target x sigma_T14(T14) /
    sqrt(N_ev(T14)), noise from the uninjected discovery sector. At each iterate m = 0..50 the achieved SNR of k_m is
    checked with the noise at T14(k_m) before stepping. Returns one record with SOLVE_FIELDS; a model exception ends
    the solve with stop_reason MODEL_ERROR (recorded, never raised into the draw loop)."""
    out = dict.fromkeys(SOLVE_FIELDS); out.update(valid=False, snr_target=float(snr_target))
    def noise(dur):
        r = snr_from_context(snr_ctx, p, t0, dur, 1.0)
        s, n = r.get('sigma_T14_ppm', float('nan')) * 1e-6, r.get('N_ev')
        out.update(snr_status=r.get('status'), noise_n_ev=n)
        ok = r.get('status') == 'OK' and np.isfinite(s) and s > 0 and isinstance(n, int) and n >= 1
        return (s, n) if ok else None
    def done(reason, **kw):
        out.update(kw, stop_reason=reason)
        return out
    try:
        nz = noise(TM.t14(p, 0.1, b, ars))
        if nz is None: return done('NOISE_INVALID_START')
        k = math.sqrt(snr_target * nz[0] / math.sqrt(nz[1]))
        for m in range(51):
            dur = TM.t14(p, k, b, ars)
            out.update(k=k, t14=dur, n_iter=m)
            nz = noise(dur)
            if nz is None: return done('NOISE_INVALID')
            delta = TM.mean_deficit(k, ld, p, b, ars, backend)[0]
            achieved = delta * math.sqrt(nz[1]) / nz[0]
            residual = achieved / snr_target - 1.0
            best = abs(residual) if out['min_residual'] is None else min(out['min_residual'], abs(residual))
            d_star = snr_target * nz[0] / math.sqrt(nz[1])
            out.update(snr_achieved=achieved, snr_residual=residual, min_residual=best, depth=delta, depth_target=d_star, sigma_t14=nz[0], n_ev=nz[1])
            if abs(residual) <= SNR_TOL:
                if not (TM.K_MIN <= k <= TM.K_MAX and np.isfinite(dur) and dur > 0 and b < 1 + k): return done('OUTSIDE_VALID_RANGE')
                return done('CONVERGED', valid=True)
            if m == 50: return done('NOT_CONVERGED_50')
            k_new = solve_k_for_depth(d_star, ld, p, b, ars, backend)
            if k_new is None: return done('DEPTH_OUTSIDE_K_BRACKET')
            k = k_new
    except (ZeroDivisionError, FloatingPointError, ArithmeticError) as exc:
        return done('MODEL_ERROR', valid=False, model_error=type(exc).__name__)


# ------------------------------------------------------------------------------------------------ T0 conditions
def archival_windows_ok(archival, p, t0, dur):
    """Number of archival windows |t - t_mid| <= T14/2 with >= 75% of expected cadences, all QUALITY == 0."""
    n_ok = 0; expected = dur / TM.CAD_D
    for time, quality in archival:
        good = np.isfinite(time)
        tt, qq = time[good], quality[good]
        if tt.size == 0: continue
        for m in t0 + p * np.arange(math.ceil((tt.min() - t0) / p), math.floor((tt.max() - t0) / p) + 1):
            sel = np.abs(tt - m) <= 0.5 * dur
            if sel.sum() >= 0.75 * expected and np.all(qq[sel] == 0):
                n_ok += 1
    return n_ok


def discovery_supported_events(times, p, t0, dur):
    ids, ev = matching.centers(t0, p, np.sort(times))
    return int(np.sum(matching.support(np.sort(times), ev, 0.5 * dur)))


def search_cap(t_bls):
    return min(13.0, (float(t_bls[-1]) - float(t_bls[0])) * 0.48)


# ------------------------------------------------------------------------------------------------ draws
def host_context(ald, host, root):
    """Everything the draws need from one host's uninjected files: raw discovery arrays, the SNR context, the
    production search timestamps, the archival times and quality flags, and the stellar mass and radius."""
    import lightkurve as lk
    root = Path(root)
    disc = root / host['source_fits']
    raw = prodpath.read_raw(disc)
    main_t = prodpath.main_search_arrays(ald, lk.read(str(disc)))[0]
    archival = []
    for r in host['lightcurves']:
        if int(r['sector']) == int(host['sector']):
            continue
        t, _f, q = prodpath.read_raw(root / r['path'])
        archival.append((t, q))
    st = host['stellar']
    return dict(raw=raw, snr_ctx=snr_context(ald, *raw), main_t=main_t, archival=archival,
                m_star=float(st['m_star']), r_star=float(st['r_star']), disc=disc)


def conditional_period_range(m_star, r_star, design):
    """Item 43.8, corrected: the geometry floor is a lower period bound (days)."""
    if not (math.isfinite(m_star) and math.isfinite(r_star) and m_star > 0 and r_star > 0):
        raise ValueError("INVALID_STELLAR_GEOMETRY")
    p_floor = 2 * math.pi * math.sqrt((2.6 * r_star * TM.RSUN) ** 3 / (TM.G_SI * m_star * TM.MSUN)) / 86400
    lo, hi = max(1.0, design.period_range_days[0], p_floor), min(13.0, design.period_range_days[1])
    return dict(lo=lo, hi=hi, p_floor_days=p_floor, distribution="host-conditional log-uniform",
                status="OK" if lo < hi else "EMPTY_PERIOD_RANGE")


def m1_target_plan(hosts, design):
    """Assign edge halves at random before any outcome; targets survive every draw retry."""
    hosts = list(hosts)
    if len(set(hosts)) != len(hosts):
        raise ValueError("M1_DUPLICATE_HOST")
    rng = np.random.default_rng(design.seed_base)
    plan = {h: {} for h in hosts}
    for bi, (lo, hi) in enumerate(design.snr_bins):
        order = rng.permutation(len(hosts))
        upper_half = set(order[:len(hosts) // 2])
        for i, host in enumerate(hosts):
            a, b = lo, hi
            if (lo, hi) == (4.0, 6.0):
                a, b = (5.0, 6.0) if i in upper_half else (4.0, 5.0)
            elif (lo, hi) == (8.0, 10.0):
                a, b = (9.0, 10.0) if i in upper_half else (8.0, 9.0)
            for slot in range(design.n_per_bin):
                plan[host][(bi, slot)] = float(rng.uniform(a, b))
    return plan


def realized_grey_counts(rows, expected=(150, 300, 150)):
    """Check written injections, not merely planned targets, before equating pooled and weighted gains."""
    counts = [0, 0, 0]
    for r in rows:
        tr = r.get('truth')
        if tr is None or r.get('draw_status', r.get('status')) != 'DRAWN':
            continue
        s = tr['snr_target']
        if 5 <= s < 6: counts[0] += 1
        elif 6 <= s < 8: counts[1] += 1
        elif 8 <= s <= 9: counts[2] += 1
    return dict(counts=counts, expected=list(expected), passed=counts == list(expected))


def draw_planet(rng, snr_lo, snr_hi, host, ld, design, snr_target=None):
    """One slot: an outer loop of at most design.max_draws parameter draws and an inner loop of at most
    design.max_t0_tries epochs. Returns (truth or None, attempts); every rejected draw and try is recorded."""
    attempts = []
    t_min = float(np.nanmin(host['main_t']))
    cap = search_cap(host['main_t'])
    population = conditional_period_range(host['m_star'], host['r_star'], design)
    if population['status'] != 'OK':
        return None, [dict(draw_outcome='EMPTY_PERIOD_RANGE', population=population)]
    p_lo, p_hi = population['lo'], population['hi']
    b_lo, b_hi = design.impact_range
    d_lo, d_hi = design.depth_domain_ppm
    snr_t = float(rng.uniform(snr_lo, snr_hi)) if snr_target is None else float(snr_target)
    if not _in_bin(snr_t, snr_lo, snr_hi, (10.0, 35.0)):
        raise ValueError('TARGET_OUTSIDE_SNR_BIN')
    for draw in range(design.max_draws):
        b = float(rng.uniform(b_lo, b_hi)); p = float(math.exp(rng.uniform(math.log(p_lo), math.log(p_hi))))
        rec = dict(draw=draw, snr_target=snr_t, b=b, period=p, tries=[])
        attempts.append(rec)
        ars = TM.a_over_rs(p, host['m_star'], host['r_star'])
        if ars < 2.6:
            rec['draw_outcome'] = 'BELOW_AR_FLOOR'; continue
        if p > cap:
            rec['draw_outcome'] = 'OUTSIDE_SEARCH_DOMAIN'; continue
        if b >= ars:
            rec['draw_outcome'] = 'NON_TRANSITING_GEOMETRY'; continue
        for _try in range(design.max_t0_tries):
            t0 = float(rng.uniform(t_min, t_min + p))
            sol = joint_solve(snr_t, b, p, t0, ld, ars, host['snr_ctx'], design.backend)
            if not sol['valid']:
                rec['tries'].append(dict(sol, t0=t0, try_outcome='SOLVE_FAILED')); continue
            if not d_lo <= sol['depth'] * 1e6 <= d_hi:
                rec['tries'].append(dict(sol, t0=t0, try_outcome='DEPTH_OUTSIDE_DOMAIN')); continue
            rp_earth = sol['k'] * host['r_star'] * RP_EARTH_PER_RSUN
            if rp_earth > design.rp_max_earth:
                rec['tries'].append(dict(sol, t0=t0, try_outcome='RADIUS_OUTSIDE_DOMAIN', rp_earth=rp_earth)); continue
            n_arch = archival_windows_ok(host['archival'], p, t0, sol['t14'])
            n_disc = discovery_supported_events(host['main_t'], p, t0, sol['t14'])
            if n_arch < design.min_archival_windows or n_disc < design.min_discovery_events:
                rec['tries'].append(dict(sol, t0=t0, try_outcome='T0_CONDITIONS', n_archival_windows=n_arch, n_discovery_events=n_disc)); continue
            truth = dict(b=b, period=p, t0=t0, a_rs=ars, inc=TM.inclination(b, ars), ld=list(ld), **sol,
                         n_archival_windows=n_arch, n_discovery_events=n_disc, backend=design.backend, rp_earth=rp_earth,
                         period_population=population)
            rec['tries'].append(dict(sol, t0=t0, try_outcome='OK')); rec['draw_outcome'] = 'OK'
            return truth, attempts
        else:
            rec['draw_outcome'] = 'INNER_LOOP_TRIES_FAILED'
    return None, attempts


def slot_seed(design, host_index, bin_index, slot):
    return int(design.seed_base) + 1000 * int(host_index) + 10 * int(bin_index) + int(slot)


def draw_m1_host(ald, host, root, design, host_index, targets=None):
    """M1 truth draws for one host (no file written): design.n_per_bin slots in every SNR_inj bin, in fixed order,
    each from its own seeded generator. A slot that cannot be drawn keeps truth None and is reported DRAW_FAILED."""
    if design.n_per_bin > 9:
        raise ValueError("slot_seed supports at most 9 slots per bin")
    ctx = host_context(ald, host, root)
    out = []
    for bi, (lo, hi) in enumerate(design.snr_bins):
        for slot in range(design.n_per_bin):
            seed = slot_seed(design, host_index, bi, slot)
            target = None if targets is None else targets[(bi, slot)]
            truth, attempts = draw_planet(np.random.default_rng(seed), lo, hi, ctx, host['ld'], design, target)
            out.append(dict(family='planet', snr_bin_index=bi, snr_bin=bin_label(lo, hi),
                            snr_target=target if target is not None else next((a['snr_target'] for a in attempts if 'snr_target' in a), None),
                            period_population=conditional_period_range(ctx['m_star'], ctx['r_star'], design),
                            slot=slot, seed=seed, truth=truth, attempts=attempts,
                            status='DRAWN' if truth is not None else 'DRAW_FAILED'))
    return out


# ------------------------------------------------------------------------------------------------ file writing
def model_flux(times, truth):
    if truth.get('null'):
        return np.ones(np.asarray(times).size)   # m = 1 copy through the same writing path (null control)
    return TM.transit_flux(times, truth['k'], truth['ld'], truth['t0'], truth['period'], truth['a_rs'], truth['inc'],
                           truth['backend'])


def _write_with_float64(src, dst, replace):
    """Copy src to dst with the named binary-table columns of HDU 1 replaced by float64 arrays (same name, unit, dim).

    The originals are float32; storing the injected columns as float64 keeps the exact identities of Controls 2
    (1e-9 and 1e-6 relative) instead of float32 rounding. Every other HDU, column and header keyword is copied."""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(src, memmap=False) as h:
        cols = []
        for col in h[1].columns:
            if col.name in replace:
                arr = np.asarray(replace[col.name], dtype=np.float64)
                n = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
                cols.append(fits.Column(name=col.name, format=f'{n}D' if n > 1 else 'D', unit=col.unit, dim=col.dim, array=arr))
            else:
                cols.append(col)
        hdr = h[1].header.copy()
        new = fits.BinTableHDU.from_columns(fits.ColDefs(cols), header=hdr, name=h[1].name)
        out = fits.HDUList([h[0].copy(), new] + [x.copy() for x in h[2:]])
        tmp = dst.with_name(dst.name + '.partial'); out.writeto(tmp, overwrite=True, checksum=False); tmp.replace(dst)


def inject_lightcurve(src, dst, truth):
    """Copy src to dst with PDCSAP_FLUX and PDCSAP_FLUX_ERR multiplied by m(TIME). Returns (CADENCENO, TIME, m)."""
    with fits.open(src, memmap=False) as h:
        d = h[1].data; t = np.asarray(d['TIME'], float); cad = np.asarray(d['CADENCENO']).copy()
        flux, err = np.asarray(d['PDCSAP_FLUX'], float), np.asarray(d['PDCSAP_FLUX_ERR'], float)
    m = model_flux(t, truth)
    _write_with_float64(src, dst, {'PDCSAP_FLUX': flux * m, 'PDCSAP_FLUX_ERR': err * m})
    return cad, t, m


def pixel_weights(flux, quality, aperture):
    """S4 pixels: ring = 8-connected 1-pixel dilation of A minus A, weights w and A_adj medians (item 22.4)."""
    ap = np.asarray(aperture, bool)
    good = np.asarray(quality) == 0
    f_med = np.nanmedian(np.asarray(flux, float)[good], axis=0)
    ring = ndimage.binary_dilation(ap, structure=np.ones((3, 3), bool)) & ~ap
    w = np.zeros(ap.shape)
    w[ap] = np.maximum(np.nan_to_num(f_med[ap], nan=0.0), 0.0)
    for i, j in zip(*np.nonzero(ring)):
        adj = np.zeros(ap.shape, bool); adj[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True; adj &= ap
        cap = 0.5 * float(np.nanmedian(f_med[adj]))
        w[i, j] = min(max(float(np.nan_to_num(f_med[i, j], nan=0.0)), 0.0), cap) if np.isfinite(cap) else 0.0
    return f_med, ring, w


def inject_tpf(src, dst, truth, crowdsap, lc_cadence, lc_time):
    """Copy the discovery TPF and subtract c * w_ij * (1 - m(t)) from FLUX; m at TPF times, gaps filled by CADENCENO."""
    with fits.open(src, memmap=False) as h:
        d = h[1].data
        ap = (np.asarray(h['APERTURE'].data).astype(int) & 2) > 0
        flux = np.asarray(d['FLUX'], float); t = np.asarray(d['TIME'], float).copy()
        cad = np.asarray(d['CADENCENO']); quality = np.asarray(d['QUALITY']).copy()
    lut = dict(zip(np.asarray(lc_cadence).tolist(), np.asarray(lc_time, float).tolist()))
    for idx in np.nonzero(~np.isfinite(t))[0]:
        t[idx] = lut.get(int(cad[idx]), np.nan)
    f_med, ring, w = pixel_weights(flux, quality, ap)
    s_w = float(w[ap].sum())
    if not s_w > 0: return None
    c = float(crowdsap) * float(np.nansum(f_med[ap])) / s_w
    m = model_flux(t, truth)
    if dst is not None:   # dst None: weights and model only, for a fidelity recheck of an existing copy
        _write_with_float64(src, dst, {'FLUX': flux - c * w[None, :, :] * (1.0 - m)[:, None, None]})
    return dict(c=c, sum_w_aperture=s_w, sum_fmed_aperture=float(np.nansum(f_med[ap])), n_ring=int(ring.sum()), m=m, w=w, aperture=ap)


def write_injection(h, root, inj_id, truth, out_root, sha256_file):
    """Separate copies of every host file with the model applied; returns (detector case, pixel record).
    Originals are never opened for writing. The case carries no truth field: it is what the detector may see."""
    root, out_root = Path(root), Path(out_root)
    base = out_root / inj_id
    lcs, disc_cad = [], None
    for r in h['lightcurves']:
        dst = base / Path(r['path']).name
        cad, t, m = inject_lightcurve(root / r['path'], dst, truth)
        if int(r['sector']) == int(h['sector']): disc_cad = (cad, t)
        lcs.append(dict(path=str(dst.relative_to(out_root)).replace('\\', '/'), sector=r['sector'], obsid=r.get('obsid'), sha256=sha256_file(dst)))
    tdst = base / Path(h['tpf_fits']).name
    tinfo = inject_tpf(root / h['tpf_fits'], tdst, truth, h['crowdsap'], *disc_cad)
    if tinfo is None:
        return None, dict(reason='SUM_W_APERTURE_ZERO')
    disc = next(x for x in lcs if int(x['sector']) == int(h['sector']))
    case = dict(tic_id=h['tic_id'], case_id=inj_id, sector=h['sector'], source_fits=disc['path'], source_sha256=disc['sha256'],
                tpf_fits=str(tdst.relative_to(out_root)).replace('\\', '/'), tpf_sha256=sha256_file(tdst), stellar=h['stellar'],
                known_transits=[], lightcurves=lcs)
    return case, tinfo


def fidelity(h, root, case, out_root, tinfo, truth, sha256_file):
    """Injection fidelity (INJ-CAL-01 Controls 2): exact injected-minus-original identities, per injection."""
    root, out_root = Path(root), Path(out_root); res = {}
    worst_lc = 0.0
    for orig, inj in zip(sorted(h['lightcurves'], key=lambda r: r['sector']), sorted(case['lightcurves'], key=lambda r: r['sector'])):
        with fits.open(root / orig['path'], memmap=False) as a, fits.open(out_root / inj['path'], memmap=False) as b:
            t = np.asarray(a[1].data['TIME'], float); m = model_flux(t, truth)
            fa, fb = np.asarray(a[1].data['PDCSAP_FLUX'], float), np.asarray(b[1].data['PDCSAP_FLUX'], float)
            ok = np.isfinite(fa) & np.isfinite(fb) & (fa != 0)
            worst_lc = max(worst_lc, float(np.max(np.abs(fb[ok] / fa[ok] - m[ok]))) if ok.any() else 0.0)
            assert np.array_equal(np.asarray(a[1].data['SAP_FLUX']), np.asarray(b[1].data['SAP_FLUX']), equal_nan=True), 'SAP_FLUX_CHANGED'
    res['lc_max_abs_ratio_minus_m'] = worst_lc
    with fits.open(root / h['tpf_fits'], memmap=False) as a, fits.open(out_root / case['tpf_fits'], memmap=False) as b:
        fa, fb = np.asarray(a[1].data['FLUX'], float), np.asarray(b[1].data['FLUX'], float)
    d = fb - fa; m = tinfo['m']; w = tinfo['w']; ap = tinfo['aperture']
    expect = -tinfo['c'] * w[None, :, :] * (1.0 - m)[:, None, None]
    fin = np.isfinite(d)
    res['pixel_max_abs_minus_expected'] = float(np.max(np.abs(d[fin] - expect[fin]))) if fin.any() else 0.0
    res['outside_pixels_unchanged'] = bool(np.all((d[:, (w == 0)][np.isfinite(d[:, (w == 0)])]) == 0))
    # The 1e-6 relative aperture-sum check is applied where it is resolvable in float64 (|expected| >= 1e6 * eps *
    # sum |F_inj|); every cadence with a signal must also agree within that rounding scale in absolute terms.
    full = np.all(np.isfinite(fa[:, ap]), axis=1) & ((1.0 - m) > 1e-12)
    lhs = d[full][:, ap].sum(axis=1); rhs = -float(h['crowdsap']) * tinfo['sum_fmed_aperture'] * (1.0 - m[full])
    scale = np.finfo(float).eps * np.abs(fb[full][:, ap]).sum(axis=1)
    resolvable = np.abs(rhs) >= 1e6 * scale
    res['aperture_sum_max_rel'] = float(np.max(np.abs(lhs[resolvable] / rhs[resolvable] - 1.0))) if resolvable.any() else None
    res['aperture_sum_n_resolvable'] = int(resolvable.sum()); res['aperture_sum_n_signal'] = int(full.sum())
    res['aperture_sum_max_abs_over_rounding'] = float(np.max(np.abs(lhs - rhs) / scale)) if full.any() else None
    res['originals_unchanged'] = all(sha256_file(root / r['path']) == r['sha256'] for r in h['lightcurves']) and sha256_file(root / h['tpf_fits']) == h['tpf_sha256']
    res['passed'] = (res['lc_max_abs_ratio_minus_m'] <= 1e-9 and res['outside_pixels_unchanged'] and res['originals_unchanged']
                     and res['pixel_max_abs_minus_expected'] <= 1e-6 * max(1.0, float(np.nanmax(np.abs(expect))))
                     and res['aperture_sum_max_rel'] is not None and res['aperture_sum_max_rel'] <= 1e-6
                     and res['aperture_sum_max_abs_over_rounding'] <= 1.0)
    return res
