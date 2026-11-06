import os
import io
import ssl
import json
import shutil
import hashlib
import glob
import logging
import warnings
import urllib.error
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Optional
import certifi
import requests

try:
    from decision_ledger import ObserverContext, SubmissionLedger
except ImportError:
    ObserverContext = None
    SubmissionLedger = None

from ephemeris_contract import (
    days_to_hours,
    BTJD_OFFSET,
    fold_phase_0to1,
    fold_phase_centered,
    cyclic_orbital_distance,
    compute_orbit_number,
    cramer_rao_timing_uncertainty
)

logger = logging.getLogger(__name__)

# Priority P1: Enforce verified TLS context anchored to certifi CA bundle
CA_BUNDLE_PATH = certifi.where()
os.environ["SSL_CERT_FILE"] = CA_BUNDLE_PATH
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE_PATH
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=CA_BUNDLE_PATH)


def safe_http_get(url: str, params: dict = None, headers: dict = None, timeout: int = 45) -> requests.Response:
    """Execute an HTTPS GET with certifi verification; propagate certificate errors."""
    resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=CA_BUNDLE_PATH)
    resp.raise_for_status()
    return resp


def safe_urlopen(req_or_url, timeout: int = 45):
    """Open a URL with certifi verification; propagate certificate errors."""
    ctx = ssl.create_default_context(cafile=CA_BUNDLE_PATH)
    try:
        return urllib.request.urlopen(req_or_url, timeout=timeout, context=ctx)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError):
            raise exc.reason from exc
        raise


# Suppress non-critical library import warnings across multiprocessing workers
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

# Lock internal BLAS/OpenMP threads to 1 per process to prevent CPU oversubscription on i9-14900HX
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import median_filter
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from astroquery.mast import Observations, Catalogs
import known_signals

# Accepted Cp policy: ARGON_DECISIONS items 12-14 and 17-20.
# PORT-CP-01: ordinary source; no runtime replay dependency.
VR_DT_GRID = 0.045 / 10.0           # item 12: main-search minimum duration / astropy oversample 10 (days)
VR_ALPHA_CIRC = 0.001               # item 13
VR_FULL_TRANSIT_FRACTION = 0.75     # preregistration D-3
VR_CADENCE_DAYS = 2.0 / 1440.0      # TESS 2-min cadence used by VALIDATION-02


# ---------------------------------------------------------------- item 12: archival timing and refinement
def vr_timing(best_t0, best_p, best_dur, sde, t_curr, archival_times_by_sector):
    """Pivot epoch, interim floor sigma_T0', sigma_P' (D-1) and per-sector sigma_pred (D-2)."""
    baseline = float(np.max(t_curr) - np.min(t_curr))
    n_cyc = max(1.0, baseline / best_p)
    s0 = 0.5 * best_dur / max(float(sde), 5.0)
    sigma_t0 = float(np.sqrt(s0 ** 2 + VR_DT_GRID ** 2 / 12.0 + (0.08 * best_dur) ** 2))
    sigma_p = float(np.sqrt(12.0) * sigma_t0 / n_cyc)
    t0_pivot = float(best_t0 + np.round((float(np.median(t_curr)) - best_t0) / best_p) * best_p)
    pred, cycles = {}, {}
    for sec, ts in archival_times_by_sector:
        n_s = int(np.round((float(np.median(ts)) - t0_pivot) / best_p))
        pred[int(sec)] = float(np.sqrt(sigma_t0 ** 2 + (n_s * sigma_p) ** 2)); cycles[int(sec)] = n_s
    return dict(t0_pivot=t0_pivot, sigma_t0=sigma_t0, sigma_p=sigma_p, n_cycles=n_cyc, sigma_pred=pred, n_s=cycles,
                max_sigma_pred=max(pred.values()), mode1=max(pred.values()) <= 0.25 * best_dur)


def vr_box_stats(t, y, dy, period, t0, dur):
    """Weighted box depth, its error and the BLS likelihood power at one (P, t0, duration)."""
    w = 1.0 / np.asarray(dy, float) ** 2
    tau = np.abs(((np.asarray(t, float) - t0 + 0.5 * period) % period) - 0.5 * period)
    m = tau < 0.5 * dur
    w_in, w_out = w[m].sum(), w[~m].sum()
    if w_in <= 0 or w_out <= 0:
        return np.nan, np.nan, 0.0
    depth = (w[~m] * y[~m]).sum() / w_out - (w[m] * y[m]).sum() / w_in
    err = np.sqrt(1.0 / w_in + 1.0 / w_out)
    return float(depth), float(err), float(0.5 * depth ** 2 * w_in) if depth > 0 else 0.0


def vr_constrained_box_search(t, y, dy, periods, durations, t0_pivot, half_window, step):
    """Box search over P x t0 x duration with t0 restricted to t0_pivot +/- half_window (D-4).

    Same likelihood objective as astropy BoxLeastSquares (0.5 * depth^2 * W_in, dips only). Returns the best
    (period, t0, duration, power) and the evaluated offsets, so tests can confirm none lies outside the window.
    """
    t = np.asarray(t, float); y = np.asarray(y, float); w = 1.0 / np.asarray(dy, float) ** 2
    W, WY = w.sum(), (w * y).sum()
    offsets = np.arange(-half_window, half_window + 0.5 * step, step)
    offsets = offsets[np.abs(offsets) <= half_window + 1e-12]
    best = (np.nan, np.nan, np.nan, -np.inf)
    reach = half_window + 0.5 * float(np.max(durations))
    for P in np.asarray(periods, float):
        d = ((t - t0_pivot + 0.5 * P) % P) - 0.5 * P
        near = np.abs(d) < reach
        if not np.any(near):
            continue
        order = np.argsort(d[near])
        ds = d[near][order]; cw = np.concatenate([[0.0], np.cumsum(w[near][order])])
        cwy = np.concatenate([[0.0], np.cumsum((w[near] * y[near])[order])])
        for dur in np.asarray(durations, float):
            lo = np.searchsorted(ds, offsets - 0.5 * dur, side="right")
            hi = np.searchsorted(ds, offsets + 0.5 * dur, side="left")
            w_in = cw[hi] - cw[lo]; wy_in = cwy[hi] - cwy[lo]
            ok = (w_in > 0) & (W - w_in > 0)
            if not np.any(ok):
                continue
            depth = np.where(ok, (WY - wy_in) / np.where(ok, W - w_in, 1.0) - wy_in / np.where(ok, w_in, 1.0), 0.0)
            power = np.where(ok & (depth > 0), 0.5 * depth ** 2 * w_in, 0.0)
            k = int(np.argmax(power))
            if power[k] > best[3]:
                best = (float(P), float(t0_pivot + offsets[k]), float(dur), float(power[k]))
    return dict(period=best[0], t0=best[1], duration=best[2], power=best[3], offsets=offsets)


def vr_count_full_transits(t, period, t0, dur):
    """Archival transits with >= 75% of expected 2-min cadences inside |phase| < 0.45 T14 (D-3)."""
    t = np.asarray(t, float)
    tau = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    m = np.abs(tau) < 0.45 * dur
    if not np.any(m):
        return 0
    epochs = np.round((t[m] - t0) / period).astype(np.int64)
    expected = 0.9 * dur / VR_CADENCE_DAYS
    _, counts = np.unique(epochs, return_counts=True)
    return int(np.sum(counts >= VR_FULL_TRANSIT_FRACTION * expected))


def vr_period_scan(t, y, dy, periods, durations, t0_pivot, half_window, step):
    """Per-period best of the constrained search (same objective and epoch window as vr_constrained_box_search)."""
    out = []
    for P in np.asarray(periods, float):
        r = vr_constrained_box_search(t, y, dy, [P], durations, t0_pivot, half_window, step)
        out.append((float(P), r['t0'], r['duration'], r['power']))
    return out


def vr_refine_p3(t, y, dy, p_min, p_max, n_trials, durations, t0_pivot, half_window, step, n_max, dur):
    """Item 18 (P3): period grid with half-step drift <= T14/8 at max|n_s|; two stages above 4000 points."""
    W = float(p_max - p_min)
    n_req = int(np.ceil(4.0 * W * n_max / dur)) + 1
    if n_req <= 4000:
        n = int(np.clip(max(4 * n_trials, n_req), 60, 4000))
        r = vr_constrained_box_search(t, y, dy, np.linspace(p_min, p_max, n), durations, t0_pivot, half_window, step)
        r.update(n_req=n_req, n_grid=n, stages=1)
        return r
    coarse = np.linspace(p_min, p_max, 4000); step1 = W / 3999.0
    scan = vr_period_scan(t, y, dy, coarse, durations, t0_pivot, half_window, step)
    power = np.array([s[3] for s in scan]); order = np.argsort(-power); tops = []
    for k in order:
        if all(abs(k - j) > 2 for j in tops):
            tops.append(int(k))
        if len(tops) == 3:
            break
    step2 = dur / (8.0 * max(n_max, 1))
    best = dict(period=scan[order[0]][0], t0=scan[order[0]][1], duration=scan[order[0]][2], power=scan[order[0]][3])
    n2_total = 0
    for k in tops:
        lo, hi = max(p_min, coarse[k] - 2 * step1), min(p_max, coarse[k] + 2 * step1)
        n2 = int(np.ceil((hi - lo) / step2)) + 1; n2_total += n2
        r = vr_constrained_box_search(t, y, dy, np.linspace(lo, hi, n2), durations, t0_pivot, half_window, step)
        if r['power'] > best['power']:
            best = dict(period=r['period'], t0=r['t0'], duration=r['duration'], power=r['power'])
    best.update(n_req=n_req, n_grid=4000 + n2_total, stages=2, offsets=np.arange(-half_window, half_window + 0.5 * step, step))
    return best


# ---------------------------------------------------------------- item 13: circular-zone Sidak threshold
def vr_z_crit_circ(period, dur, alpha=VR_ALPHA_CIRC):
    n_eff = max(1.0, 1.0 + 0.08 / (0.85 * dur / period))
    p_crit = 1.0 - (1.0 - alpha) ** (1.0 / n_eff)
    return float(stats.norm.isf(p_crit)), float(n_eff)


# ---------------------------------------------------------------- item 14: TPF local noise and hierarchy
def vr_tpf_maps(t, flux, period, t0, dur):
    """Per-transit baseline-subtracted difference image dI, MAD-based sigma_pix and z (frozen windows)."""
    t = np.asarray(t, float); f = np.asarray(flux, float)
    tau = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    epoch = np.round((t - t0) / period).astype(np.int64)
    in_m = np.abs(tau) < 0.45 * dur
    sh_m = (np.abs(tau) > 0.75 * dur) & (np.abs(tau) < 2.5 * dur)
    r_in, r_out = [], []
    for e in np.unique(epoch[in_m | sh_m]):
        sel_sh = sh_m & (epoch == e); sel_in = in_m & (epoch == e)
        if not np.any(sel_sh):
            continue
        base = np.nanmedian(f[sel_sh], axis=0)
        r_out.append(f[sel_sh] - base)
        if np.any(sel_in):
            r_in.append(f[sel_in] - base)
    if not r_in or not r_out:
        return None
    r_in = np.concatenate(r_in); r_out = np.concatenate(r_out)
    n_in = np.sum(np.isfinite(r_in), axis=0); n_out = np.sum(np.isfinite(r_out), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_i = np.nanmean(r_out, axis=0) - np.nanmean(r_in, axis=0)
        mad_in = 1.4826 * np.nanmedian(np.abs(r_in - np.nanmedian(r_in, axis=0)), axis=0)
        mad_out = 1.4826 * np.nanmedian(np.abs(r_out - np.nanmedian(r_out, axis=0)), axis=0)
        sig = np.sqrt(mad_in ** 2 / n_in + mad_out ** 2 / n_out)
        z = d_i / sig
    bad = ~np.isfinite(z) | (sig <= 0)
    d_i = np.where(np.isfinite(d_i), d_i, 0.0); z = np.where(bad, 0.0, z); sig = np.where(bad, np.inf, sig)
    return dict(dI=d_i, sigma=sig, z=z, n_in=int(np.max(n_in)), n_out=int(np.max(n_out)))


def vr_tpf_decide(d_i, sig, z, aperture):
    """Item 14 order: off-target test first, then weak localization, then dipole. Returns (outcome, reason, info).

    outcome is 'REJECT', 'REVIEW' or 'PASS'. Interpretations recorded in the report: the off-target peak is the
    off-aperture pixel with the largest dI; other off-aperture pixels with z >= 3.5 and z_diff >= 2 go to REVIEW.
    """
    ap = np.asarray(aperture, bool); off = ~ap
    nr, nc = ap.shape
    in_flat = np.where(ap, d_i, -np.inf)
    i_max = np.unravel_index(int(np.argmax(in_flat)), ap.shape)
    def zdiff(ix):
        return float((d_i[ix] - d_i[i_max]) / np.sqrt(sig[ix] ** 2 + sig[i_max] ** 2))
    info = dict(in_max_pixel=[int(i_max[1]), int(i_max[0])], dI_in_max=float(d_i[i_max]), z_in_star=float(np.max(np.where(ap, z, -np.inf))))
    if np.any(off):
        off_flat = np.where(off, d_i, -np.inf)
        ip = np.unravel_index(int(np.argmax(off_flat)), ap.shape)
        zd = zdiff(ip); info.update(off_peak_pixel=[int(ip[1]), int(ip[0])], z_off_peak=float(z[ip]), z_diff_peak=zd)
        if z[ip] >= 5.0 and zd >= 3.0:
            interior = 1 <= ip[0] <= nr - 2 and 1 <= ip[1] <= nc - 2
            if interior:
                return "REJECT", f"Off-target source! Pixel (col={ip[1]}, row={ip[0]}) z={z[ip]:.1f}, z_diff={zd:.1f} -> BEB", info
            return "REVIEW", f"REVIEW_TPF_EDGE_OR_MARGINAL: border pixel (col={ip[1]}, row={ip[0]}) z={z[ip]:.1f}, z_diff={zd:.1f}", info
        for ix in zip(*np.nonzero(off & (z >= 3.5))):
            zd_x = zdiff(ix)
            if zd_x >= 2.0:
                return "REVIEW", f"REVIEW_TPF_EDGE_OR_MARGINAL: pixel (col={ix[1]}, row={ix[0]}) z={z[ix]:.1f}, z_diff={zd_x:.1f}", info
    if info["z_in_star"] < 3.0:
        return "PASS", "OK_WEAK_LOCALIZATION", info
    dip = ap & (d_i < -0.25 * d_i[i_max]) & (z <= -4.0)
    if np.any(dip):
        ix = tuple(int(v[0]) for v in np.nonzero(dip))
        return "REVIEW", f"REVIEW_TPF_DIPOLE_JITTER: pixel (col={ix[1]}, row={ix[0]}) z={z[ix]:.1f}", info
    return "PASS", "OK", info


PIPELINE_VERSION = "v9.0"
CANDIDATE_SCHEMA_VERSION = "v1.1"

CANDIDATE_COLUMNS = [
    "schema_version",
    "pipeline_version",
    "run_id",
    "tic_id",
    "sector",
    "is_toi_host",
    "known_toi_context",
    "candidate_type",
    "ephemeris_source",
    "ephemeris_status",
    "ephemeris_fallback_reason",
    "time_system",
    "time_standard",
    "epoch_offset_btjd",
    "period_units",
    "duration_units",
    "depth_units",
    "transit_measurement_source",
    "timing_uncertainty_source",
    "period_days",
    "t0_btjd",
    "discovery_period_days",
    "discovery_t0_btjd",
    "refined_period_days",
    "refined_t0_btjd",
    "duration_hours",
    "min_kepler_dur_hours",
    "max_kepler_dur_hours",
    "n_transits",
    "depth_ppm",
    "crowdsap",
    "sde",
    "r_planet_earth",
    "r_star_sun",
    "m_star_sun",
    "teff_k",
    "gaia_ruwe",
    "odd_even_sigma",
    "centroid_shift_arcsec",
    "archival_sectors",
    "n_archival_sectors",
    "archival_sectors_requested",
    "archival_sectors_attempted",
    "archival_sectors_used",
    "archival_products_requested",
    "archival_products_attempted",
    "archival_products_used",
    "archival_products_failed",
    "archival_snr_sigma",
    "archival_trials",
    "archival_snr_crit",
    "est_multi_sde_proxy",
    "multi_sde",
    "cramer_rao_sigma_p_sec",
    "cramer_rao_sigma_t0_sec",
    "covariance_p_t0",
    "plot_path",
    "exofop_url"
]


def _get_stable_product_id(row, sec_num: int, idx: int) -> str:
    """Extract a stable product ID from search result row if available, otherwise explicit fallback."""
    for attr in ("obsid", "productFilename", "obs_id", "target_name"):
        try:
            val = getattr(row, attr, None)
            if val is not None and str(val).strip():
                return str(val).strip()
        except Exception:
            pass
    try:
        table_row = getattr(row, "table", row)
        for key in ("obsid", "productFilename", "obs_id", "target_name"):
            if hasattr(table_row, "colnames") and key in table_row.colnames:
                val = table_row[key]
                if val is not None and str(val).strip():
                    return str(val).strip()
    except Exception:
        pass
    try:
        if hasattr(row, "__getitem__"):
            for key in ("obsid", "productFilename", "obs_id", "target_name"):
                if key in row:
                    val = row[key]
                    if val is not None and str(val).strip():
                        return str(val).strip()
    except Exception:
        pass
    return f"UNAVAILABLE_attempt_{idx}_sec_{int(sec_num):04d}"


def _serialize_metadata_list(items) -> str:
    """Serializes a list or dict of provenance items into an unambiguous JSON string."""
    if items is None:
        return "[]"
    if isinstance(items, str):
        return items
    return json.dumps(items)


AUTO_PURGE_CACHE = True       # Automatically deletes raw FITS files of non-candidates to keep cache at ~0 MB

# Lightkurve does NOT read a LIGHTKURVE_CACHE_DIR env var by itself; apply it explicitly so HPC jobs
# keep all FITS downloads on node-local scratch (re-executed in every spawned/forked worker on import).
if os.environ.get("LIGHTKURVE_CACHE_DIR"):
    os.makedirs(os.environ["LIGHTKURVE_CACHE_DIR"], exist_ok=True)
    lk.conf.cache_dir = os.environ["LIGHTKURVE_CACHE_DIR"]

# Newest released sectors (108 down to 80; Sectors 99-106 completed in Run 1)
PREFERRED_SECTORS = [
    108, 107, 98, 97, 96, 95, 94, 93,
    92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
    106, 105, 104, 103, 102, 101, 100, 99
]

# Multi-Node PBS Array Coordinator (v8.0 Zero-Lock Sharding)
ARRAY_IDX = int(os.environ.get("PBS_ARRAY_INDEX", "0"))
if ARRAY_IDX > 0:
    target_sec = PREFERRED_SECTORS[(ARRAY_IDX - 1) % len(PREFERRED_SECTORS)]
    PREFERRED_SECTORS = [target_sec]
    OUTPUT_DIR = os.path.join("tess_results", f"sector_{target_sec}")
    DEFAULT_WORKERS = 8
    DEFAULT_TARGETS = 15000
else:
    OUTPUT_DIR = "tess_results"
    DEFAULT_WORKERS = 16
    DEFAULT_TARGETS = 1000

MAX_TARGETS = int(os.environ.get("ALDEBARAN_MAX_TARGETS", DEFAULT_TARGETS))
MAX_WORKERS = int(os.environ.get("ALDEBARAN_WORKERS", DEFAULT_WORKERS))
MAX_PEAKS_PER_STAR = 3    # Inspect top 3 distinct BLS peaks per star before rejecting
PRIORITY_TICS = []        # Add specific TIC IDs here to force-test them first

os.makedirs(OUTPUT_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(OUTPUT_DIR, "scanned_tics.txt")
CATALOG_CACHE_FILE = os.path.join("tess_results", "known_catalog_cache.txt")
TOI_EPHEMERIS_CACHE_FILE = os.path.join("tess_results", "toi_ephemerides_cache.json")
KNOWN_SIGNALS_REGISTRY_CACHE_FILE = os.path.join("tess_results", "known_signals_registry.json")
SEARCH_TOI_COMPANIONS = os.environ.get("ALDEBARAN_SEARCH_TOI_COMPANIONS", "1").lower() not in ("0", "false", "no")
GLOBAL_TOI_EPHEMERIDES = {}
GLOBAL_KNOWN_SIGNALS_REGISTRY = None
REPORT_FILE = os.path.join(OUTPUT_DIR, "candidates_report.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline_run.log")
EVENTS_LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline_events.log")
CURRENT_RUN_ID = os.environ.get("ALDEBARAN_RUN_ID", "default_run")
ARCHIVAL_DIAGNOSTICS_FILE = os.path.join(OUTPUT_DIR, "archival_vetting_diagnostics.jsonl")


def configure_environment(
    output_dir: str = None,
    toi_ephemerides: dict = None,
    log_file: str = None,
    run_id: str = None,
    known_signals_registry = None,
    catalog_cache_file: str = None,
    toi_ephemeris_cache_file: str = None,
    known_signals_registry_cache_file: str = None,
):
    """
    Explicitly configure pipeline paths and global catalog context.
    Ensures that worker processes (under both fork and spawn) have consistent
    output directories, log file locations, in-memory TOI ephemeris tables,
    known signals registry context, cache file locations, and run identifiers for isolated artifact generation.
    """
    global OUTPUT_DIR, LOG_FILE, EVENTS_LOG_FILE, REPORT_FILE, HISTORY_FILE, GLOBAL_TOI_EPHEMERIDES, CURRENT_RUN_ID, ARCHIVAL_DIAGNOSTICS_FILE, GLOBAL_KNOWN_SIGNALS_REGISTRY
    global CATALOG_CACHE_FILE, TOI_EPHEMERIS_CACHE_FILE, KNOWN_SIGNALS_REGISTRY_CACHE_FILE
    if run_id is not None:
        CURRENT_RUN_ID = run_id
    if output_dir is not None:
        OUTPUT_DIR = output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline_run.log") if log_file is None else log_file
        EVENTS_LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline_events.log")
        REPORT_FILE = os.path.join(OUTPUT_DIR, "candidates_report.csv")
        HISTORY_FILE = os.path.join(OUTPUT_DIR, "scanned_tics.txt")
        ARCHIVAL_DIAGNOSTICS_FILE = os.path.join(OUTPUT_DIR, "archival_vetting_diagnostics.jsonl")
    elif log_file is not None:
        LOG_FILE = log_file
    if toi_ephemerides is not None:
        GLOBAL_TOI_EPHEMERIDES = dict(toi_ephemerides)
    if known_signals_registry is not None:
        GLOBAL_KNOWN_SIGNALS_REGISTRY = known_signals_registry
    if catalog_cache_file is not None:
        CATALOG_CACHE_FILE = catalog_cache_file
    if toi_ephemeris_cache_file is not None:
        TOI_EPHEMERIS_CACHE_FILE = toi_ephemeris_cache_file
    if known_signals_registry_cache_file is not None:
        KNOWN_SIGNALS_REGISTRY_CACHE_FILE = known_signals_registry_cache_file



class AnalysisResult(tuple):
    """
    4-element tuple (tic_id, status_code, message, cand_data) preserving strict
    backwards-compatibility for unpacking, while providing result-owned metadata
    for current attempt isolation (run_id, attempt_id, archival_diagnostic, archival_diagnostics).
    """
    def __new__(cls, tic_id, status, message, cand_data, run_id=None, attempt_id=None, archival_diagnostics=None):
        return super().__new__(cls, (tic_id, status, message, cand_data))

    def __init__(self, tic_id, status, message, cand_data, run_id=None, attempt_id=None, archival_diagnostics=None):
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.archival_diagnostics = list(archival_diagnostics or [])
        self.archival_diagnostic = self.archival_diagnostics[-1] if self.archival_diagnostics else None

    def __reduce__(self):
        return (AnalysisResult, (self[0], self[1], self[2], self[3], self.run_id, self.attempt_id, self.archival_diagnostics))


def record_archival_diagnostic(target_diag: dict):
    """
    Persists bounded JSON-safe archival vetting diagnostics for an inspected target/peak.
    Ensures that run_id, attempt_id, timestamp, product IDs, failure stages, and fallback
    reasons survive all exits, including archival rejections and post-preparation evaluation failures.
    On failure to persist, raises RuntimeError to avoid silently dropping evidence.
    """
    global ARCHIVAL_DIAGNOSTICS_FILE
    if not ARCHIVAL_DIAGNOSTICS_FILE:
        ARCHIVAL_DIAGNOSTICS_FILE = os.path.join(OUTPUT_DIR, "archival_vetting_diagnostics.jsonl")
    diag_dir = os.path.dirname(os.path.abspath(ARCHIVAL_DIAGNOSTICS_FILE))
    os.makedirs(diag_dir, exist_ok=True)
    diag_copy = dict(target_diag)
    if "run_id" not in diag_copy or not diag_copy["run_id"]:
        diag_copy["run_id"] = CURRENT_RUN_ID
    if "attempt_id" not in diag_copy or not diag_copy["attempt_id"]:
        diag_copy["attempt_id"] = "unknown_attempt"
    if "timestamp" not in diag_copy or not diag_copy["timestamp"]:
        from datetime import datetime, timezone
        diag_copy["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(diag_copy) + "\n"
    try:
        with open(ARCHIVAL_DIAGNOSTICS_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to persist archival vetting diagnostic to '{ARCHIVAL_DIAGNOSTICS_FILE}': {exc}"
        ) from exc


def get_latest_archival_diagnostic(
    tic_id: str,
    sector: int = None,
    output_dir: str = None,
    run_id: str = None,
    attempt_id: str = None
) -> dict | None:
    """
    Retrieves the most recent archival vetting diagnostic record matching tic_id,
    and optionally sector, run_id, and attempt_id.
    Searches archival_vetting_diagnostics.jsonl in output_dir (or configured OUTPUT_DIR).
    Returns dict or None.
    """
    base_dir = output_dir if output_dir is not None else OUTPUT_DIR
    diag_file = os.path.join(base_dir, "archival_vetting_diagnostics.jsonl")
    if not os.path.exists(diag_file) or os.path.getsize(diag_file) == 0:
        return None
    tid_str = str(tic_id).lstrip("0")
    matching = []
    try:
        with open(diag_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if str(rec.get("tic_id", "")).lstrip("0") != tid_str:
                        continue
                    if sector is not None and int(rec.get("sector", -1)) != int(sector):
                        continue
                    if run_id is not None and str(rec.get("run_id", "")) != str(run_id):
                        continue
                    if attempt_id is not None and str(rec.get("attempt_id", "")) != str(attempt_id):
                        continue
                    matching.append(rec)
                except Exception:
                    continue
    except Exception:
        return None
    return matching[-1] if matching else None


def load_archival_diagnostics(output_dir: str = None) -> list:
    """Reads all diagnostic records from archival_vetting_diagnostics.jsonl."""
    base_dir = output_dir if output_dir is not None else OUTPUT_DIR
    diag_file = os.path.join(base_dir, "archival_vetting_diagnostics.jsonl")
    if not os.path.exists(diag_file) or os.path.getsize(diag_file) == 0:
        return []
    records = []
    try:
        with open(diag_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return records


def write_log(message: str, print_to_console: bool = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if print_to_console:
        print(message, flush=True)
    try:
        log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def write_event(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        event_dir = os.path.dirname(os.path.abspath(EVENTS_LOG_FILE))
        if event_dir:
            os.makedirs(event_dir, exist_ok=True)
        with open(EVENTS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def fetch_csv_with_headers(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = safe_http_get(url, headers=headers, timeout=45)
        return pd.read_csv(io.StringIO(resp.text), comment="#", on_bad_lines="skip")
    except requests.exceptions.SSLError:
        raise
    except Exception:
        req = urllib.request.Request(url, headers=headers)
        with safe_urlopen(req, timeout=45) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(raw_text), comment="#", on_bad_lines="skip")


def _extract_source_input(fix_entry: Any, catalog_name: str, default_url: str):
    """
    Extracts DataFrame, raw payload sha256, payload_hash_kind, raw_source_path_or_url, and original_retrieval_utc.
    Accurately distinguishes raw file bytes vs in-memory string vs dataframe serialization.
    """
    df = None
    raw_sha = None
    hash_kind = "dataframe_serialization"
    source_path_or_url = default_url
    orig_utc = None

    repo_root = os.path.dirname(os.path.abspath(__file__))
    manifest_info = {}
    manifest_paths = [
        os.path.join(repo_root, "24702696_20261016_025524", "audit", "catalog_audit", "source_manifest.json"),
        os.path.join(os.path.dirname(str(fix_entry)), "source_manifest.json") if isinstance(fix_entry, str) else "",
        "24702696_20261016_025524/audit/catalog_audit/source_manifest.json",
    ]
    for mp in manifest_paths:
        if mp and os.path.isfile(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    mf = json.load(f)
                    if catalog_name in mf:
                        manifest_info = mf[catalog_name]
                        break
            except Exception:
                pass

    if isinstance(fix_entry, pd.DataFrame):
        df = fix_entry
        raw_sha = known_signals.compute_payload_sha256(fix_entry)
        hash_kind = "dataframe_serialization"
        source_path_or_url = "in_memory_dataframe"
        orig_utc = None
    elif isinstance(fix_entry, bytes):
        raw_sha = hashlib.sha256(fix_entry).hexdigest()
        hash_kind = "raw_bytes"
        source_path_or_url = "in_memory_bytes"
        df = pd.read_csv(io.BytesIO(fix_entry), comment="#", on_bad_lines="skip")
    elif isinstance(fix_entry, str):
        file_candidate = fix_entry if os.path.isfile(fix_entry) else os.path.join(repo_root, fix_entry)
        if "\n" not in fix_entry and os.path.isfile(file_candidate):
            with open(file_candidate, "rb") as f:
                b = f.read()
            raw_sha = hashlib.sha256(b).hexdigest()
            hash_kind = "raw_file_bytes"
            source_path_or_url = file_candidate
            df = pd.read_csv(io.BytesIO(b), comment="#", on_bad_lines="skip")
            if manifest_info and manifest_info.get("sha256") == raw_sha:
                orig_utc = manifest_info.get("retrieved_utc")
                if manifest_info.get("url"):
                    source_path_or_url = manifest_info.get("url")
        else:
            b = fix_entry.encode("utf-8")
            raw_sha = hashlib.sha256(b).hexdigest()
            hash_kind = "raw_string_bytes"
            source_path_or_url = "in_memory_string"
            df = pd.read_csv(io.StringIO(fix_entry), comment="#", on_bad_lines="skip")
            if manifest_info and manifest_info.get("sha256") == raw_sha:
                orig_utc = manifest_info.get("retrieved_utc")
                if manifest_info.get("url"):
                    source_path_or_url = manifest_info.get("url")

    return df, raw_sha, hash_kind, source_path_or_url, orig_utc


def load_known_catalogs_and_toi_ephemerides(offline_fixtures: Optional[dict] = None) -> tuple:
    """
    Ingests official TOI catalog ephemerides (P, T0, T_dur, depth, disp), confirmed planet hosts
    (preserving all planets, transit ephemerides, durations, and depths), ExoFOP CTOIs,
    and cataloged EBs from VizieR (Prša et al. 2022).
    
    Populates the canonical versioned KnownSignalsRegistry (available via get_known_signals_registry()
    and get_known_signal_context_for_tic(tic_id)) while strictly preserving the legacy return tuple.
    
    Returns:
        known_catalog_tics: set of all cataloged TICs (legacy union)
        known_eb_tics: set of confirmed/cataloged Eclipsing Binary stars (Prša et al. 2022)
        toi_ephemerides: dict[str, list[dict]] mapping TIC string to list of known TOI planets
    """
    global GLOBAL_TOI_EPHEMERIDES, GLOBAL_KNOWN_SIGNALS_REGISTRY
    known_catalog_tics = set()
    known_eb_tics = set()
    toi_ephemerides = {}

    registry = None
    if os.path.exists(KNOWN_SIGNALS_REGISTRY_CACHE_FILE):
        try:
            registry = known_signals.KnownSignalsRegistry.load_from_json(
                KNOWN_SIGNALS_REGISTRY_CACHE_FILE, mark_as_cached=True
            )
            GLOBAL_KNOWN_SIGNALS_REGISTRY = registry
            known_catalog_tics.update(registry.export_legacy_catalog_tics())
            known_eb_tics.update(registry.export_legacy_eb_tics())
            toi_ephemerides.update(registry.export_legacy_toi_ephemerides())
            GLOBAL_TOI_EPHEMERIDES = toi_ephemerides
        except Exception as exc:
            write_log(f"      [!] Warning: Failed to load known_signals_registry from cache ({exc})")

    if registry is None:
        registry = known_signals.KnownSignalsRegistry()

    if not toi_ephemerides and os.path.exists(TOI_EPHEMERIS_CACHE_FILE):
        try:
            with open(TOI_EPHEMERIS_CACHE_FILE, "r", encoding="utf-8") as f:
                toi_ephemerides = json.load(f)
            GLOBAL_TOI_EPHEMERIDES = toi_ephemerides
        except Exception:
            pass

    if not known_catalog_tics and os.path.exists(CATALOG_CACHE_FILE):
        try:
            with open(CATALOG_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = set(line.strip() for line in f if line.strip())
            known_catalog_tics.update(cached)
        except Exception:
            pass

    fixtures = offline_fixtures or {}
    is_offline = os.environ.get("ALDEBARAN_OFFLINE", "0").lower() in ("1", "true", "yes")
    allow_network = (offline_fixtures is None) and (not is_offline)

    def _apply_source_or_preserve(src_name: str, signals: list, snap, df_len: int, log_desc: str):
        if snap.status == "FAILED":
            if src_name in registry.source_snapshots and registry.source_snapshots[src_name].status in ("SUCCESS", "CACHED"):
                existing = registry.source_snapshots[src_name]
                existing.last_attempt_status = "FAILED"
                existing.last_attempt_utc = snap.retrieval_utc
                existing.last_attempt_error = snap.error_message
                registry.record_failed_attempt(snap, signals)
                cov = registry.get_coverage_status()
                for host in registry.hosts.values():
                    host.coverage_status = cov
                write_log(f"      [!] Warning: Failed to parse {log_desc} table: {snap.error_message}. Preserving existing coherent data.")
                return
            registry.replace_source_signals(src_name, signals, snap)
            write_log(f"      [!] Warning: Failed to parse {log_desc} table: {snap.error_message}.")
        else:
            if src_name in registry.source_snapshots and registry.source_snapshots[src_name].status in ("SUCCESS", "CACHED"):
                existing = registry.source_snapshots[src_name]
                if existing.payload_sha256 == snap.payload_sha256:
                    existing.last_attempt_status = "SUCCESS"
                    existing.last_attempt_utc = snap.retrieval_utc
                    existing.last_attempt_error = None
                    existing.status = "SUCCESS"
                    cov = registry.get_coverage_status()
                    statuses = registry.get_source_statuses()
                    for host in registry.hosts.values():
                        host.source_statuses = dict(statuses)
                        host.coverage_status = cov
                    write_log(f"      -> Loaded {df_len} {log_desc} entries (re-verified coherent selected snapshot {existing.snapshot_id}).")
                    return
            registry.replace_source_signals(src_name, signals, snap)
            write_log(f"      -> Loaded {df_len} {log_desc} entries.")

    # [1/5] NASA TOI
    write_log("[1/5] Fetching official TOI catalog & ephemerides from NASA Exoplanet Archive...")
    df_toi = None
    toi_sha = None
    toi_hash_kind = "dataframe_serialization"
    toi_src_path = "offline_fixture"
    toi_orig_utc = None
    toi_retrieval_utc = datetime.now(timezone.utc).isoformat()
    toi_query = (
        "select tid, toi, pl_tranmid, pl_tranmiderr1, pl_tranmiderr2, pl_orbper, "
        "pl_orbpererr1, pl_orbpererr2, pl_trandurh, pl_trandurherr1, pl_trandurherr2, "
        "pl_trandep, pl_trandeperr1, pl_trandeperr2, tfopwg_disp from toi"
    )
    toi_url_or_src = "offline_fixture" if "nasa_toi" in fixtures else "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + urllib.parse.quote(toi_query) + "&format=csv"

    if "nasa_toi" in fixtures:
        raw_fix = fixtures["nasa_toi"]
        try:
            df_toi, toi_sha, toi_hash_kind, toi_src_path, toi_orig_utc = _extract_source_input(raw_fix, "nasa_toi", toi_url_or_src)
        except Exception as exc:
            registry.record_source_attempt("nasa_toi", status="FAILED", error_message=f"Failed to parse fixture: {exc}", provider="NASA Exoplanet Archive", query_or_url=toi_url_or_src)
    elif allow_network:
        try:
            df_toi = fetch_csv_with_headers(toi_url_or_src)
            toi_src_path = toi_url_or_src
        except Exception as e:
            write_log(f"      [!] Warning: Could not load TOI ephemeris table (using cache): {e}")
            registry.record_source_attempt("nasa_toi", status="FAILED", error_message=str(e), provider="NASA Exoplanet Archive", query_or_url=toi_url_or_src)
    else:
        if "nasa_toi" not in registry.source_snapshots:
            registry.record_source_attempt("nasa_toi", status="UNAVAILABLE", error_message="Offline mode and no fixture provided", provider="NASA Exoplanet Archive", query_or_url=toi_url_or_src)

    if df_toi is not None and not df_toi.empty:
        signals_toi, snap_toi = known_signals.ingest_nasa_toi_df(
            df_toi,
            query_or_url=toi_url_or_src,
            retrieval_utc=toi_retrieval_utc,
            raw_payload_sha256=toi_sha,
            payload_hash_kind=toi_hash_kind,
            raw_source_path_or_url=toi_src_path,
            original_retrieval_utc=toi_orig_utc,
        )
        _apply_source_or_preserve("nasa_toi", signals_toi, snap_toi, len(df_toi), "TOI planet")
    elif df_toi is not None and df_toi.empty:
        registry.record_source_attempt("nasa_toi", status="INCOMPLETE", error_message="Empty TOI response/fixture", provider="NASA Exoplanet Archive", query_or_url=toi_url_or_src)

    # [2/5] NASA Confirmed (pscomppars)
    write_log("[2/5] Fetching confirmed exoplanets (pscomppars) from NASA Exoplanet Archive...")
    df_ps = None
    ps_sha = None
    ps_hash_kind = "dataframe_serialization"
    ps_src_path = "offline_fixture"
    ps_orig_utc = None
    ps_retrieval_utc = datetime.now(timezone.utc).isoformat()
    ps_query = (
        "select tic_id, pl_name, hostname, discoverymethod, pl_orbper, pl_orbpererr1, pl_orbpererr2, "
        "pl_tranmid, pl_tranmiderr1, pl_tranmiderr2, pl_trandur, pl_trandurerr1, pl_trandurerr2, "
        "pl_trandep, pl_trandeperr1, pl_trandeperr2 from pscomppars where tic_id is not null"
    )
    ps_url_or_src = "offline_fixture" if "nasa_confirmed" in fixtures else "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + urllib.parse.quote(ps_query) + "&format=csv"

    if "nasa_confirmed" in fixtures:
        raw_fix = fixtures["nasa_confirmed"]
        try:
            df_ps, ps_sha, ps_hash_kind, ps_src_path, ps_orig_utc = _extract_source_input(raw_fix, "nasa_confirmed", ps_url_or_src)
        except Exception as exc:
            registry.record_source_attempt("nasa_confirmed", status="FAILED", error_message=f"Failed to parse fixture: {exc}", provider="NASA Exoplanet Archive", query_or_url=ps_url_or_src)
    elif allow_network:
        try:
            df_ps = fetch_csv_with_headers(ps_url_or_src)
            ps_src_path = ps_url_or_src
        except Exception as e:
            write_log(f"      [!] Warning: Could not load confirmed planets table (using cache): {e}")
            registry.record_source_attempt("nasa_confirmed", status="FAILED", error_message=str(e), provider="NASA Exoplanet Archive", query_or_url=ps_url_or_src)
    else:
        if "nasa_confirmed" not in registry.source_snapshots:
            registry.record_source_attempt("nasa_confirmed", status="UNAVAILABLE", error_message="Offline mode and no fixture provided", provider="NASA Exoplanet Archive", query_or_url=ps_url_or_src)

    if df_ps is not None and not df_ps.empty:
        signals_ps, snap_ps = known_signals.ingest_nasa_confirmed_df(
            df_ps,
            query_or_url=ps_url_or_src,
            retrieval_utc=ps_retrieval_utc,
            raw_payload_sha256=ps_sha,
            payload_hash_kind=ps_hash_kind,
            raw_source_path_or_url=ps_src_path,
            original_retrieval_utc=ps_orig_utc,
        )
        _apply_source_or_preserve("nasa_confirmed", signals_ps, snap_ps, len(signals_ps), "confirmed planet")
    elif df_ps is not None and df_ps.empty:
        registry.record_source_attempt("nasa_confirmed", status="INCOMPLETE", error_message="Empty confirmed planets response/fixture", provider="NASA Exoplanet Archive", query_or_url=ps_url_or_src)

    # [3/5] ExoFOP CTOI
    write_log("[3/5] Fetching Community TOI (CTOI) catalog from ExoFOP-TESS...")
    df_ctoi = None
    ctoi_sha = None
    ctoi_hash_kind = "dataframe_serialization"
    ctoi_src_path = "offline_fixture"
    ctoi_orig_utc = None
    ctoi_retrieval_utc = datetime.now(timezone.utc).isoformat()
    ctoi_url_or_src = "offline_fixture" if "exofop_ctoi" in fixtures else "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"

    if "exofop_ctoi" in fixtures:
        raw_fix = fixtures["exofop_ctoi"]
        try:
            df_ctoi, ctoi_sha, ctoi_hash_kind, ctoi_src_path, ctoi_orig_utc = _extract_source_input(raw_fix, "exofop_ctoi", ctoi_url_or_src)
        except Exception as exc:
            registry.record_source_attempt("exofop_ctoi", status="FAILED", error_message=f"Failed to parse fixture: {exc}", provider="ExoFOP-TESS", query_or_url=ctoi_url_or_src)
    elif allow_network:
        try:
            df_ctoi = fetch_csv_with_headers(ctoi_url_or_src)
            ctoi_src_path = ctoi_url_or_src
        except Exception as e:
            write_log(f"      [!] Warning: Could not load ExoFOP CTOI table (using cache): {e}")
            registry.record_source_attempt("exofop_ctoi", status="FAILED", error_message=str(e), provider="ExoFOP-TESS", query_or_url=ctoi_url_or_src)
    else:
        if "exofop_ctoi" not in registry.source_snapshots:
            registry.record_source_attempt("exofop_ctoi", status="UNAVAILABLE", error_message="Offline mode and no fixture provided", provider="ExoFOP-TESS", query_or_url=ctoi_url_or_src)

    if df_ctoi is not None and not df_ctoi.empty:
        signals_ctoi, snap_ctoi = known_signals.ingest_exofop_ctoi_df(
            df_ctoi,
            query_or_url=ctoi_url_or_src,
            retrieval_utc=ctoi_retrieval_utc,
            raw_payload_sha256=ctoi_sha,
            payload_hash_kind=ctoi_hash_kind,
            raw_source_path_or_url=ctoi_src_path,
            original_retrieval_utc=ctoi_orig_utc,
        )
        _apply_source_or_preserve("exofop_ctoi", signals_ctoi, snap_ctoi, len(signals_ctoi), "Community TOI (CTOI)")
    elif df_ctoi is not None and df_ctoi.empty:
        registry.record_source_attempt("exofop_ctoi", status="INCOMPLETE", error_message="Empty CTOI response/fixture", provider="ExoFOP-TESS", query_or_url=ctoi_url_or_src)

    # [4/5] VizieR TESS EB
    write_log("[4/5] Fetching TESS Eclipsing Binary Catalogue (Prša et al. 2022) from VizieR TAP...")
    df_eb = None
    eb_sha = None
    eb_hash_kind = "dataframe_serialization"
    eb_src_path = "offline_fixture"
    eb_orig_utc = None
    eb_retrieval_utc = datetime.now(timezone.utc).isoformat()
    adql_eb = 'SELECT distinct TIC FROM "J/ApJS/258/16/tess-ebs"'
    eb_url_or_src = "offline_fixture" if "vizier_tess_eb" in fixtures else "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=" + urllib.parse.quote(adql_eb)

    if "vizier_tess_eb" in fixtures:
        raw_fix = fixtures["vizier_tess_eb"]
        try:
            df_eb, eb_sha, eb_hash_kind, eb_src_path, eb_orig_utc = _extract_source_input(raw_fix, "vizier_tess_eb", eb_url_or_src)
        except Exception as exc:
            registry.record_source_attempt("vizier_tess_eb", status="FAILED", error_message=f"Failed to parse fixture: {exc}", provider="VizieR TAP (Prsa et al. 2022)", query_or_url=eb_url_or_src)
    elif allow_network:
        try:
            df_eb = fetch_csv_with_headers(eb_url_or_src)
            eb_src_path = eb_url_or_src
        except Exception as e:
            write_log(f"      [!] Warning: Could not load VizieR TESS EB table (using cache): {e}")
            registry.record_source_attempt("vizier_tess_eb", status="FAILED", error_message=str(e), provider="VizieR TAP (Prsa et al. 2022)", query_or_url=eb_url_or_src)
    else:
        if "vizier_tess_eb" not in registry.source_snapshots:
            registry.record_source_attempt("vizier_tess_eb", status="UNAVAILABLE", error_message="Offline mode and no fixture provided", provider="VizieR TAP (Prsa et al. 2022)", query_or_url=eb_url_or_src)

    if df_eb is not None and not df_eb.empty:
        signals_eb, snap_eb = known_signals.ingest_vizier_eb_df(
            df_eb,
            query_or_url=eb_url_or_src,
            retrieval_utc=eb_retrieval_utc,
            raw_payload_sha256=eb_sha,
            payload_hash_kind=eb_hash_kind,
            raw_source_path_or_url=eb_src_path,
            original_retrieval_utc=eb_orig_utc,
        )
        _apply_source_or_preserve("vizier_tess_eb", signals_eb, snap_eb, len(signals_eb), "TESS Eclipsing Binary (EB)")
    elif df_eb is not None and df_eb.empty:
        registry.record_source_attempt("vizier_tess_eb", status="INCOMPLETE", error_message="Empty VizieR EB response/fixture", provider="VizieR TAP (Prsa et al. 2022)", query_or_url=eb_url_or_src)

    GLOBAL_KNOWN_SIGNALS_REGISTRY = registry

    # Synchronize legacy active return structures
    if "nasa_toi" in registry.source_snapshots and registry.source_snapshots["nasa_toi"].status in ("SUCCESS", "CACHED"):
        toi_ephem_export = registry.export_legacy_toi_ephemerides()
        toi_ephemerides = toi_ephem_export
        GLOBAL_TOI_EPHEMERIDES = dict(toi_ephemerides)
        try:
            with open(TOI_EPHEMERIS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(toi_ephemerides, f)
        except Exception:
            pass

    known_catalog_tics.update(registry.export_legacy_catalog_tics())
    known_eb_tics.update(registry.export_legacy_eb_tics())

    if registry.total_signals() > 0 or len(registry.source_snapshots) > 0:
        try:
            registry.save_to_json(KNOWN_SIGNALS_REGISTRY_CACHE_FILE)
        except Exception as exc:
            registry.persistence_status = "FAILED"
            registry.persistence_error = str(exc)
            write_log(f"      [!] Warning: Could not persist known_signals_registry to cache: {exc}")

    if known_catalog_tics:
        try:
            with open(CATALOG_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(known_catalog_tics)) + "\n")
        except Exception:
            pass

    return known_catalog_tics, known_eb_tics, toi_ephemerides


def get_known_signals_registry():
    """
    Retrieves the canonical KnownSignalsRegistry instance, loading from cache if uninitialized.
    Returns None or KnownSignalsRegistry.
    """
    global GLOBAL_KNOWN_SIGNALS_REGISTRY
    if GLOBAL_KNOWN_SIGNALS_REGISTRY is not None:
        return GLOBAL_KNOWN_SIGNALS_REGISTRY
    if os.path.exists(KNOWN_SIGNALS_REGISTRY_CACHE_FILE):
        try:
            GLOBAL_KNOWN_SIGNALS_REGISTRY = known_signals.KnownSignalsRegistry.load_from_json(
                KNOWN_SIGNALS_REGISTRY_CACHE_FILE, mark_as_cached=True
            )
            return GLOBAL_KNOWN_SIGNALS_REGISTRY
        except Exception as exc:
            logger.warning("Failed to load known_signals_registry from cache %s: %s", KNOWN_SIGNALS_REGISTRY_CACHE_FILE, exc)
    return None


def get_known_signal_context_for_tic(tic_id: str):
    """
    Retrieves structured HostCatalogContext for a target star from the canonical registry.
    Distinguishes star with no signals in complete coverage from unavailable coverage.
    """
    reg = get_known_signals_registry()
    if reg is not None:
        return reg.get_host_context(tic_id)
    return known_signals.HostCatalogContext(
        tic_id=str(tic_id),
        signals=[],
        source_statuses={src: "UNAVAILABLE" for src in known_signals.CANONICAL_SOURCES},
        coverage_status="UNAVAILABLE",
        registry_available=False,
    )


def get_known_transits_for_tic(tic_id: str) -> list:
    """Retrieves cached official TOI planet ephemerides for a target star, if any."""
    tid_str = str(tic_id).lstrip("0")
    if tid_str in GLOBAL_TOI_EPHEMERIDES:
        return GLOBAL_TOI_EPHEMERIDES[tid_str]
    if os.path.exists(TOI_EPHEMERIS_CACHE_FILE):
        try:
            with open(TOI_EPHEMERIS_CACHE_FILE, "r", encoding="utf-8") as f:
                ephems = json.load(f)
                GLOBAL_TOI_EPHEMERIDES.update(ephems)
                return GLOBAL_TOI_EPHEMERIDES.get(tid_str, [])
        except Exception:
            pass
    return []


def get_known_exoplanet_tics() -> set:
    """Legacy compatibility helper."""
    reg = get_known_signals_registry()
    if reg is not None and reg.total_signals() > 0:
        return reg.export_legacy_catalog_tics()
    known_cats, _, _ = load_known_catalogs_and_toi_ephemerides()
    return known_cats


def purge_target_cache(tic_id: str):
    """Removes cached FITS light curves and TPFs for a processed target to keep disk usage near 0 MB."""
    try:
        cache_root = lk.config.get_cache_dir()
    except Exception:
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "lightkurve")
    if not os.path.exists(cache_root):
        return
    padded_tic = str(tic_id).zfill(16)
    # MAST filenames always embed the zero-padded 16-digit TIC; an unpadded substring pattern
    # (e.g. "*674490*") would also match other targets being processed by parallel workers.
    for pattern in [f"*-{padded_tic}-*"]:
        for path in glob.glob(os.path.join(cache_root, "**", pattern), recursive=True):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass


def purge_corrupted_cache(tic_id: str):
    purge_target_cache(tic_id)


def safe_download(search_obj, tic_id: str, download_all: bool = False):
    try:
        return search_obj.download_all() if download_all else search_obj.download()
    except Exception:
        purge_corrupted_cache(tic_id)
        return search_obj.download_all() if download_all else search_obj.download()


def to_numpy(col) -> np.ndarray:
    mask = None
    if hasattr(col, "mask") and col.mask is not None:
        mask = np.asarray(col.mask, dtype=bool)
    for _ in range(3):
        if hasattr(col, "unmasked"):
            col = col.unmasked
        if hasattr(col, "value"):
            col = col.value
    arr = np.array(col, dtype=np.float64, copy=True)
    if mask is not None and mask.shape == arr.shape:
        arr[mask] = np.nan
    return arr


def load_scanned_history() -> set:
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_scanned_history(tic_id: str):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{tic_id}\n")


def select_latest_sector_and_targets(sector_list: list, limit: int) -> tuple:
    known_catalog_tics, known_eb_tics, toi_ephemerides = load_known_catalogs_and_toi_ephemerides()
    scanned_tics = load_scanned_history() - set(PRIORITY_TICS)

    write_log("[5/5] Probing MAST archive for the newest available TESS SPOC sector...")
    chosen_sector = None
    available_tics = []
    raw_count = 0
    toi_in_sec_count = 0
    novel_count = 0
    toi_host_set = set(toi_ephemerides.keys()) - known_eb_tics

    for sec in sector_list:
        try:
            obs_table = Observations.query_criteria(
                obs_collection="TESS",
                dataproduct_type="timeseries",
                sequence_number=sec,
                provenance_name="SPOC"
            )
            if len(obs_table) == 0:
                write_log(f"      -> Sector {sec}: Not released on MAST yet, checking next...")
                continue

            raw_tics = set(str(t).lstrip("0") for t in obs_table["target_name"] if str(t).isdigit())
            
            # Known TOI candidate/confirmed planetary systems (excluding cataloged EBs)
            toi_in_sec = [t for t in sorted(raw_tics & toi_host_set) if t not in scanned_tics]
            novel_tics = [t for t in sorted(raw_tics - known_catalog_tics) if t not in scanned_tics]

            if SEARCH_TOI_COMPANIONS:
                # Prioritize known TOI systems at the head of the queue for companion exoplanet discovery
                combined_unsearched = toi_in_sec + novel_tics
                write_log(
                    f"      -> Sector {sec}: Found {len(raw_tics)} SPOC stars | "
                    f"{len(toi_in_sec)} known TOI hosts queued for sibling search | "
                    f"{len(novel_tics)} unsearched novel stars."
                )
            else:
                combined_unsearched = novel_tics

            if len(combined_unsearched) == 0:
                write_log(f"      -> Sector {sec}: All {len(raw_tics)} targets already scanned, moving to next...")
                continue

            chosen_sector = sec
            available_tics = combined_unsearched
            raw_count = len(raw_tics)
            toi_in_sec_count = len(toi_in_sec)
            novel_count = len(novel_tics)
            break
        except Exception as e:
            write_log(f"      -> Sector {sec} query error ({e}), trying next...")
            continue

    if chosen_sector is None:
        raise RuntimeError("No available unsearched SPOC targets found in any of the PREFERRED_SECTORS.")

    final_queue = []
    for ptic in PRIORITY_TICS:
        if ptic in known_catalog_tics and ptic not in toi_host_set:
            write_log(f"      [i] Note: Priority target TIC {ptic} is already listed in catalogs.")
        final_queue.append(ptic)

    for tic in available_tics:
        if tic not in final_queue:
            final_queue.append(tic)

    n_toi_queued = sum(1 for t in final_queue[:limit] if t in toi_host_set)
    lock_msg = (
        f"Sector {chosen_sector} locked | Total SPOC: {raw_count} | "
        f"Unsearched: {len(available_tics)} (TOI Hosts: {toi_in_sec_count}, Novel: {novel_count}) | "
        f"Queued: {min(limit, len(final_queue))} (TOI Sibling Targets: {n_toi_queued})"
    )
    write_log(f"[5/5] {lock_msg}\n")
    write_event(lock_msg)
    return chosen_sector, final_queue[:limit]


def calc_max_keplerian_duration_hours(p_days: float, r_star_sun: float, m_star_sun: float, rp_rs: float) -> float:
    """Calculates theoretical maximum central transit duration (b=0, circular orbit) from Kepler's 3rd Law."""
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    
    p_sec = p_days * 86400.0
    m_kg = max(0.08, m_star_sun) * M_sun
    r_m = max(0.10, r_star_sun) * R_sun
    
    a_m = (G * m_kg * (p_sec ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    ratio = (r_m * (1.0 + rp_rs)) / a_m
    if ratio >= 1.0:
        return days_to_hours(p_days) * 0.25
    t14_sec = (p_sec / np.pi) * np.arcsin(ratio)
    return float(t14_sec / 3600.0)


def get_stellar_and_gaia_params(tic_id: str) -> dict:
    result = {
        "r_star": np.nan, "e_r_star": np.nan,
        "m_star": np.nan, "e_m_star": np.nan,
        "teff": np.nan, "ruwe": np.nan, "has_params": False
    }
    try:
        tic_data = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
        if len(tic_data) == 0:
            return result
            
        row = tic_data[0]
        r_val = float(to_numpy(row["rad"]))
        t_val = float(to_numpy(row["Teff"]))
        er_val = float(to_numpy(row["e_rad"])) if "e_rad" in row.colnames else np.nan
        m_val = float(to_numpy(row["mass"])) if "mass" in row.colnames else np.nan
        em_val = float(to_numpy(row["e_mass"])) if "e_mass" in row.colnames else np.nan
        
        if np.isfinite(r_val) and r_val > 0 and np.isfinite(t_val) and t_val > 0:
            result["r_star"] = r_val
            result["e_r_star"] = er_val if (np.isfinite(er_val) and er_val > 0) else (0.10 * r_val)
            result["teff"] = t_val
            m_clean = m_val if (np.isfinite(m_val) and m_val > 0) else (r_val ** 1.1)
            result["m_star"] = m_clean
            result["e_m_star"] = em_val if (np.isfinite(em_val) and em_val > 0) else (0.12 * m_clean)
            result["has_params"] = True
            
        ra = float(to_numpy(row["ra"]))
        dec = float(to_numpy(row["dec"]))
        
        # Proper-motion propagation from TIC coordinates (J2000.0) to Gaia DR3 epoch (2016.0)
        dt_yr = 16.0
        pm_ra = float(to_numpy(row["pmra"])) if "pmra" in row and pd.notna(row["pmra"]) else 0.0
        pm_dec = float(to_numpy(row["pmdec"])) if "pmdec" in row and pd.notna(row["pmdec"]) else 0.0
        cos_dec = np.cos(np.radians(dec))
        ra_ep2016 = ra + (pm_ra * dt_yr) / (3600.0 * 1000.0 * (cos_dec if abs(cos_dec) > 1e-4 else 1.0))
        dec_ep2016 = dec + (pm_dec * dt_yr) / (3600.0 * 1000.0)

        adql = (
            f"SELECT TOP 1 ruwe, "
            f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra_ep2016:.6f}, {dec_ep2016:.6f})) AS dist_deg "
            f"FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_ep2016:.6f}, {dec_ep2016:.6f}, 0.001389)) "
            f"ORDER BY dist_deg ASC"
        )
        gaia_url = (
            "https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery"
            f"&LANG=ADQL&FORMAT=csv&QUERY={urllib.parse.quote(adql)}"
        )
        df_gaia = fetch_csv_with_headers(gaia_url)
        if len(df_gaia) > 0:
            ruwe_val = float(df_gaia["ruwe"].iloc[0])
            if np.isfinite(ruwe_val):
                result["ruwe"] = ruwe_val
    except (requests.exceptions.SSLError, ssl.SSLError):
        raise
    except Exception:
        pass
    return result


def compute_keplerian_duration_bounds_hours(
    p_days: float, r_star: float, e_r_star: float, m_star: float, e_m_star: float, rp_rs: float
) -> tuple[float, float, float]:
    """
    Returns (nominal_circular_max_hr, strict_lower_bound_hr, eccentric_upper_bound_hr)
    accounting for 1-sigma TIC stellar uncertainties (e_rad, e_mass) and e <= 0.30 eccentricity.
    """
    nom_max_hr = calc_max_keplerian_duration_hours(p_days, r_star, m_star, rp_rs)
    
    r_upper = r_star + max(e_r_star, 0.08 * r_star)
    m_lower = max(0.08, m_star - max(e_m_star, 0.10 * m_star))
    upper_circ_hr = calc_max_keplerian_duration_hours(p_days, r_upper, m_lower, rp_rs)
    
    r_lower = max(0.08, r_star - max(e_r_star, 0.08 * r_star))
    m_upper = m_star + max(e_m_star, 0.10 * m_star)
    lower_circ_hr = calc_max_keplerian_duration_hours(p_days, r_lower, m_upper, rp_rs)
    
    # 0.35x lower bound (grazing b <= 0.93 + periastron e <= 0.30)
    # 1.38x upper bound on 1-sigma upper stellar radius (apastron e <= 0.30)
    return nom_max_hr, 0.35 * lower_circ_hr, 1.38 * upper_circ_hr


from typing import NamedTuple


class OddEvenV2Result(NamedTuple):
    status: str
    reason: str
    sigma: float
    odd_depth: float
    even_depth: float
    odd_err: float
    even_err: float
    diagnostics: dict


def _student_t_to_gaussian_z(abs_t: float, df: float) -> float:
    """Map a Student reference tail to Z, not calibrate observed event depths.

    norm.sf(Z) == t.sf(abs_t, df). The two-sided factors of two cancel.
    Independent, Gaussian, equal-variance event assumptions are NOT guaranteed
    by detrending, event selection or the OOT noise floor.
    """
    from scipy.special import betaln, ndtri_exp
    if np.isnan(abs_t) or abs_t < 0 or not np.isfinite(df) or df <= 0:
        raise ValueError('Invalid Student reference statistic/degrees of freedom')
    if abs_t == 0:
        return 0.0
    if np.isposinf(abs_t):
        return np.inf
    log_sf = float(stats.t.logsf(abs_t, df=df))
    if not np.isfinite(log_sf):
        # The incomplete-beta small-x expansion has a relative O(x) correction.
        # It is used only after SciPy's tail underflows, not as sqrt(-2 log p).
        log_x = float(np.log(df) - np.logaddexp(np.log(df), 2*np.log(abs_t)))
        a = df/2.0
        if log_x >= np.log(1e-8):
            raise ArithmeticError('Student log-tail failed outside small-x domain')
        log_sf = -np.log(2.) + a*log_x - np.log(a) - betaln(a,.5)
    return max(0.0, float(-ndtri_exp(log_sf)))


def compute_odd_even_v2(
    t: np.ndarray, flat_f: np.ndarray, phase: np.ndarray,
    in_tr: np.ndarray, out_tr: np.ndarray, orbit_num: np.ndarray,
    best_p: float, best_dur: float, baseline: float,
    min_complete_frac: float = .60, prior_block_df: int = 0,
    min_oot_blocks: int = 6,
) -> OddEvenV2Result:
    """Experimental Argon event/OOT-block diagnostic, not a calibrated test.

    Unweighted event depths, pooled empirical event variance and a separate
    OOT floor retain Argon's main construction without adding white variance
    twice. min(Z_emp,Z_oot) is a diagnostic score, not a universal exact tail.
    Relative cadence support .60 and six OOT blocks are heuristic engineering
    settings, not guarantees of full-event coverage or independent blocks.
    No unestimated pseudo-prior df is used: singleton parity is INCONCLUSIVE.
    The errors are component diagnostics; score != delta/hypot(errors) generally.
    """
    if prior_block_df != 0:
        raise ValueError('Unestimated prior_block_df is unsupported; use zero')
    if not(0 < min_complete_frac <= 1) or min_oot_blocks < 2:
        raise ValueError('Invalid support settings')
    if not(np.isfinite(best_p) and np.isfinite(best_dur) and np.isfinite(baseline)
           and best_p > 0 and 0 < best_dur < best_p):
        raise ValueError('Invalid period/duration/baseline')
    t, flat_f, phase, orbit_num = [np.asarray(x,dtype=float) for x in (t,flat_f,phase,orbit_num)]
    in_tr, out_tr = np.asarray(in_tr,dtype=bool), np.asarray(out_tr,dtype=bool)
    arrays=(t,flat_f,phase,orbit_num,in_tr,out_tr)
    if any(x.ndim != 1 or x.shape != t.shape for x in arrays):
        raise ValueError('Unaligned odd/even inputs')
    if np.any(in_tr & out_tr):
        raise ValueError('Overlapping in/out masks')
    order=np.argsort(t,kind='stable')
    t,flat_f,phase,orbit_num,in_tr,out_tr=[x[order] for x in arrays]
    finite=np.isfinite(t)&np.isfinite(flat_f)&np.isfinite(phase)&np.isfinite(orbit_num)
    if np.any(np.diff(t[np.isfinite(t)])<=0):
        raise ValueError('Duplicate timestamps')
    if np.any(orbit_num[finite] != np.rint(orbit_num[finite])):
        raise ValueError('Non-integer orbit indices')
    info=dict(method='ODD_EVEN_V2_EXPERIMENTAL',prior_df=0,calibrated=False,
              support_fraction=min_complete_frac,min_oot_blocks=min_oot_blocks,
              support_definition='count relative to most-supported observed event; not proof of geometric completeness')
    odd_depth=even_depth=odd_err=even_err=np.nan
    def inconclusive(reason):
        return OddEvenV2Result('INCONCLUSIVE',reason,np.nan,odd_depth,even_depth,odd_err,even_err,dict(info))
    events=[]
    for orb in np.unique(orbit_num[finite&in_tr]):
        m=finite&in_tr&(orbit_num==orb);n=int(m.sum())
        if n:events.append((int(orb),n,float((baseline-np.mean(flat_f[m]))*1e6)))
    info['events_before_support']=len(events)
    if not events:return inconclusive('NO_FINITE_TRANSIT_EVENTS')
    max_cad=max(x[1] for x in events)
    min_cad=max(3,int(np.ceil(min_complete_frac*max_cad)))
    supported=[x for x in events if x[1]>=min_cad]
    odd=[x for x in supported if x[0]%2];even=[x for x in supported if not x[0]%2]
    no,ne=len(odd),len(even)
    info.update(n_odd=no,n_even=ne,min_supported_cadences=min_cad,
                supported_orbits=[x[0] for x in supported],excluded_orbits=[x[0] for x in events if x[1]<min_cad])
    if no:odd_depth=float(np.mean([x[2] for x in odd]))
    if ne:even_depth=float(np.mean([x[2] for x in even]))
    if no<2 or ne<2:return inconclusive('UNREPLICATED_PARITY_NO_VALIDATED_PRIOR')
    od=np.array([x[2] for x in odd]);ed=np.array([x[2] for x in even])
    nc_o=np.array([x[1] for x in odd]);nc_e=np.array([x[1] for x in even])
    dt=np.diff(t[finite]);dt=dt[dt>0]
    if not len(dt):return inconclusive('NO_CADENCE_SCALE')
    cadence=float(np.median(dt))
    pure=finite&out_tr&(np.abs(np.abs(phase)-.5*best_p)>max(.06*best_p,1.2*best_dur))
    if np.sum(pure)<min_cad*min_oot_blocks:return inconclusive('INSUFFICIENT_OOT_SUPPORT')
    sigma_cad=float(1.4826*np.median(np.abs(flat_f[finite&out_tr]-baseline))*1e6)
    block_width=.9*best_dur
    ids=np.flatnonzero(pure);bt=t[ids];bf=flat_f[ids]
    block_ids=np.floor((bt-bt[0])/block_width).astype(np.int64)
    # Never join separated OOT islands or bridge a data gap inside a time bin.
    split=np.flatnonzero((np.diff(block_ids)!=0)|(np.diff(bt)>1.5*cadence))+1
    depths=[]
    for inds in np.split(np.arange(len(bt)),split):
        if len(inds)>=min_cad:
            depths.append(float((baseline-np.mean(bf[inds]))*1e6))
    info.update(n_oot_blocks=len(depths),cadence_days=cadence)
    if len(depths)<min_oot_blocks:return inconclusive('INSUFFICIENT_OOT_BLOCKS')
    bd=np.array(depths);sigma_block=float(1.4826*np.median(abs(bd-np.median(bd))))
    var_o=sigma_cad**2*np.sum(1/nc_o)/no**2
    var_e=sigma_cad**2*np.sum(1/nc_e)/ne**2
    oe_oot=float(np.sqrt(max(var_o,sigma_block**2/no)))
    ee_oot=float(np.sqrt(max(var_e,sigma_block**2/ne)))
    se_oot=max(float(np.hypot(oe_oot,ee_oot)),1.0)
    df=no+ne-2
    s2_pool=float((np.sum((od-odd_depth)**2)+np.sum((ed-even_depth)**2))/df)
    oe_emp=float(np.sqrt(s2_pool/no));ee_emp=float(np.sqrt(s2_pool/ne))
    se_emp=float(np.hypot(oe_emp,ee_emp));delta=abs(odd_depth-even_depth)
    z_emp=0. if delta==0 else _student_t_to_gaussian_z(delta/se_emp if se_emp>0 else np.inf,df)
    z_oot=delta/se_oot
    score=float(min(z_emp,z_oot))
    odd_err=max(oe_emp,oe_oot);even_err=max(ee_emp,ee_oot)
    info.update(reference_df=df,pooled_event_variance_ppm2=s2_pool,z_emp=z_emp,z_oot=z_oot,
                se_emp_ppm=se_emp,se_oot_ppm=se_oot,sigma_cad_ppm=sigma_cad,
                sigma_block_ppm=sigma_block,contrast_numerical_floor_ppm=1.,
                empirical_odd_err_ppm=oe_emp,empirical_even_err_ppm=ee_emp,
                oot_odd_err_ppm=oe_oot,oot_even_err_ppm=ee_oot)
    if not np.isfinite(score):return inconclusive('NONFINITE_REFERENCE_SCORE')
    return OddEvenV2Result('ASSESSED','POOLED_T_REFERENCE_WITH_OOT_FLOOR',score,odd_depth,even_depth,odd_err,even_err,info)


def compute_whitened_kovacs_sde(power_arr: np.ndarray, window_pts: int = 501) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Astropy quadratic log-likelihood power (0.5 * dChi^2) into linear Signal Residue (SR),
    removes the low-frequency 1/f periodogram ramp via a running median filter, and normalizes by MAD.
    Returns: (sde_array, whitened_sr_array)
    """
    sr = np.sqrt(2.0 * np.maximum(0.0, np.asarray(power_arr, dtype=np.float64)))
    win = min(window_pts, (len(sr) // 4) * 2 + 1)
    if win >= 11:
        local_baseline = median_filter(sr, size=win, mode="nearest")
    else:
        local_baseline = np.nanmedian(sr)
        
    sr_whitened = sr - local_baseline
    med_w = float(np.nanmedian(sr_whitened))
    mad_w = float(1.4826 * np.nanmedian(np.abs(sr_whitened - med_w)) + 1e-12)
    sde_arr = (sr_whitened - med_w) / mad_w
    return sde_arr, sr_whitened


def scan_full_orbit_secondary_eclipse(
    t: np.ndarray,
    flat_f: np.ndarray,
    period: float,
    t0: float,
    dur: float,
    primary_depth_ppm: float
) -> tuple[bool, float, float, float, float]:
    """
    Scans all out-of-transit orbital phases phi in [min_phi, 1 - min_phi] using a sliding
    box of width 0.85 * dur to detect eccentric secondary eclipses (e > 0.1).
    Applies Sidak-Bonferroni trials-factor correction for alpha = 0.001 false-alarm rate.
    Returns: (is_eb_flag, max_sec_snr, max_sec_depth_ppm, best_sec_phase, z_crit_threshold)
    Note: z_crit is the eccentric trials-factor threshold and is retained in the return tuple
    even when the circular branch is selected. When circular is selected, best_sec_phase is 0.50.
    """
    phase_0to1 = fold_phase_0to1(t, t0, period)
    dur_phase = dur / period
    min_phi = max(0.06, 1.6 * dur_phase)
    
    if min_phi >= 0.45:
        return False, 0.0, 0.0, 0.5, 4.0
        
    oot_mask = (phase_0to1 >= min_phi) & (phase_0to1 <= (1.0 - min_phi))
    if np.sum(oot_mask) < 50:
        return False, 0.0, 0.0, 0.5, 4.0
        
    oot_baseline = float(np.nanmedian(flat_f[oot_mask]))
    oot_noise = float(1.4826 * np.nanmedian(np.abs(flat_f[oot_mask] - oot_baseline)) + 1e-12)
    
    # Number of independent duration-wide windows across the out-of-transit orbit
    n_indep_win = max(5, int((1.0 - 2.0 * min_phi) / max(dur_phase, 0.005)))
    # Sidak-Bonferroni critical sigma for alpha = 0.001 family-wise false-alarm rate
    z_crit = float(stats.norm.isf(0.001 / n_indep_win))
    
    # Evaluate sliding box on a half-duration-spaced phase grid
    n_steps = max(20, n_indep_win * 2)
    trial_phases = np.linspace(min_phi + 0.5 * dur_phase, 1.0 - min_phi - 0.5 * dur_phase, n_steps)
    half_box = 0.425 * dur_phase
    
    ph_oot = phase_0to1[oot_mask]
    f_oot = flat_f[oot_mask]
    sort_idx = np.argsort(ph_oot)
    ph_sorted = ph_oot[sort_idx]
    f_sorted = f_oot[sort_idx]
    
    left_idx = np.searchsorted(ph_sorted, trial_phases - half_box, side="left")
    right_idx = np.searchsorted(ph_sorted, trial_phases + half_box, side="right")
    
    circ_max_snr, circ_max_depth = 0.0, 0.0
    ecc_max_snr, ecc_max_depth, ecc_best_phi = 0.0, 0.0, 0.5
    
    for phi_c, l_i, r_i in zip(trial_phases, left_idx, right_idx):
        n_pts = r_i - l_i
        if n_pts < 5:
            continue
        dip_ppm = float((oot_baseline - np.nanmean(f_sorted[l_i:r_i])) * 1e6)
        err_ppm = (oot_noise * 1e6) / np.sqrt(n_pts)
        snr = dip_ppm / max(err_ppm, 1.0)
        
        if abs(phi_c - 0.50) <= 0.04:
            if snr > circ_max_snr:
                circ_max_snr = snr
                circ_max_depth = dip_ppm
        else:
            if snr > ecc_max_snr:
                ecc_max_snr = snr
                ecc_max_depth = dip_ppm
                ecc_best_phi = float(phi_c)
                
    # Circular zone (|phi - 0.50| <= 0.04): 1 hypothesis test -> threshold SNR > 3.1
    circ_eb = (circ_max_snr > 3.1) and (circ_max_depth > 100.0) and (circ_max_depth >= 0.10 * primary_depth_ppm)
    # Eccentric zone (|phi - 0.50| > 0.04): N independent trials -> Sidak-Bonferroni z_crit
    ecc_eb = (ecc_max_snr > z_crit) and (ecc_max_depth > 100.0) and (ecc_max_depth >= 0.10 * primary_depth_ppm)
    
    # ALD-013: Independent circular and eccentric qualifying branches (logical OR).
    # Selection occurs only among qualifying branches. A nonqualifying raw peak in one
    # zone must not suppress an independently qualifying peak in the other zone.
    if circ_eb and ecc_eb:
        is_eb = True
        if circ_max_snr >= ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    elif circ_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
    elif ecc_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    else:
        is_eb = False
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
            
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit



def _scan_full_orbit_secondary_eclipse_cp(
    t: np.ndarray,
    flat_f: np.ndarray,
    period: float,
    t0: float,
    dur: float,
    primary_depth_ppm: float,
    details: Optional[dict] = None
) -> tuple[bool, float, float, float, float]:
    """
    Scans all out-of-transit orbital phases phi in [min_phi, 1 - min_phi] using a sliding
    box of width 0.85 * dur to detect eccentric secondary eclipses (e > 0.1).
    Applies Sidak-Bonferroni trials-factor correction for alpha = 0.001 false-alarm rate.
    Returns: (is_eb_flag, max_sec_snr, max_sec_depth_ppm, best_sec_phase, z_crit_threshold)
    Note: z_crit is the eccentric trials-factor threshold and is retained in the return tuple
    even when the circular branch is selected. When circular is selected, best_sec_phase is 0.50.
    """
    phase_0to1 = fold_phase_0to1(t, t0, period)
    dur_phase = dur / period
    min_phi = max(0.06, 1.6 * dur_phase)
    
    if min_phi >= 0.45:
        return False, 0.0, 0.0, 0.5, 4.0
        
    oot_mask = (phase_0to1 >= min_phi) & (phase_0to1 <= (1.0 - min_phi))
    if np.sum(oot_mask) < 50:
        return False, 0.0, 0.0, 0.5, 4.0
        
    oot_baseline = float(np.nanmedian(flat_f[oot_mask]))
    oot_noise = float(1.4826 * np.nanmedian(np.abs(flat_f[oot_mask] - oot_baseline)) + 1e-12)
    
    # Number of independent duration-wide windows across the out-of-transit orbit
    n_indep_win = max(5, int((1.0 - 2.0 * min_phi) / max(dur_phase, 0.005)))
    # Sidak-Bonferroni critical sigma for alpha = 0.001 family-wise false-alarm rate
    z_crit = float(stats.norm.isf(0.001 / n_indep_win))
    
    # Evaluate sliding box on a half-duration-spaced phase grid
    n_steps = max(20, n_indep_win * 2)
    trial_phases = np.linspace(min_phi + 0.5 * dur_phase, 1.0 - min_phi - 0.5 * dur_phase, n_steps)
    half_box = 0.425 * dur_phase
    
    ph_oot = phase_0to1[oot_mask]
    f_oot = flat_f[oot_mask]
    sort_idx = np.argsort(ph_oot)
    ph_sorted = ph_oot[sort_idx]
    f_sorted = f_oot[sort_idx]
    
    left_idx = np.searchsorted(ph_sorted, trial_phases - half_box, side="left")
    right_idx = np.searchsorted(ph_sorted, trial_phases + half_box, side="right")
    
    circ_max_snr, circ_max_depth = 0.0, 0.0
    ecc_max_snr, ecc_max_depth, ecc_best_phi = 0.0, 0.0, 0.5
    
    for phi_c, l_i, r_i in zip(trial_phases, left_idx, right_idx):
        n_pts = r_i - l_i
        if n_pts < 5:
            continue
        dip_ppm = float((oot_baseline - np.nanmean(f_sorted[l_i:r_i])) * 1e6)
        err_ppm = (oot_noise * 1e6) / np.sqrt(n_pts)
        snr = dip_ppm / max(err_ppm, 1.0)
        
        if abs(phi_c - 0.50) <= 0.04:
            if snr > circ_max_snr:
                circ_max_snr = snr
                circ_max_depth = dip_ppm
        else:
            if snr > ecc_max_snr:
                ecc_max_snr = snr
                ecc_max_depth = dip_ppm
                ecc_best_phi = float(phi_c)
                
    # Circular zone (|phi - 0.50| <= 0.04): 1 hypothesis test -> threshold SNR > 3.1
    # VETTING-REPLAY-01 A2 (ARGON_DECISIONS item 13): Sidak threshold over N_eff_circ, white-noise error kept.
    z_crit_circ, n_eff_circ = vr_z_crit_circ(period, dur)
    if details is not None:
        details.update(circ_snr=float(circ_max_snr), circ_depth=float(circ_max_depth), z_crit_circ=z_crit_circ, n_eff_circ=n_eff_circ)
    circ_eb = (circ_max_snr > z_crit_circ) and (circ_max_depth > 100.0) and (circ_max_depth >= 0.10 * primary_depth_ppm)
    # Eccentric zone (|phi - 0.50| > 0.04): N independent trials -> Sidak-Bonferroni z_crit
    ecc_eb = (ecc_max_snr > z_crit) and (ecc_max_depth > 100.0) and (ecc_max_depth >= 0.10 * primary_depth_ppm)
    
    # ALD-013: Independent circular and eccentric qualifying branches (logical OR).
    # Selection occurs only among qualifying branches. A nonqualifying raw peak in one
    # zone must not suppress an independently qualifying peak in the other zone.
    if circ_eb and ecc_eb:
        is_eb = True
        if circ_max_snr >= ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    elif circ_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
    elif ecc_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    else:
        is_eb = False
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
            
    return is_eb, max_snr, max_depth_ppm, best_phi, (z_crit_circ if (is_eb and circ_eb and best_phi == 0.50) else z_crit)


def extract_and_stitch_arrays(lc_collection, edge_trim_days: float = 0.10) -> tuple:
    """
    Extracts arrays while enforcing SPOC quality == 0, reading CROWDSAP crowding metric,
    and trimming thermal/scattered-light ramps around spacecraft downlink gaps.
    """
    t_list, f_list, ferr_list, dx_list, dy_list, crowd_list = [], [], [], [], [], []
    
    for lc_sec in lc_collection:
        c_val = lc_sec.meta.get("CROWDSAP", 1.0)
        if c_val is not None and np.isfinite(float(c_val)) and float(c_val) > 0:
            crowd_list.append(float(c_val))
            
        t = to_numpy(lc_sec.time)
        f = to_numpy(lc_sec.flux)
        ferr = to_numpy(lc_sec.flux_err)
        mom1 = to_numpy(lc_sec.mom_centr1)
        mom2 = to_numpy(lc_sec.mom_centr2)
        pos1 = to_numpy(lc_sec.pos_corr1)
        pos2 = to_numpy(lc_sec.pos_corr2)
        
        if hasattr(lc_sec, "quality") and lc_sec.quality is not None:
            qual = np.asarray(lc_sec.quality, dtype=int)
            q_valid = (qual == 0)
        else:
            q_valid = np.ones_like(t, dtype=bool)
            
        valid = np.isfinite(t) & np.isfinite(f) & (f > 0) & q_valid
        if np.sum(valid) < 500:
            continue
            
        t, f, ferr = t[valid], f[valid], ferr[valid]
        mom1, mom2 = mom1[valid], mom2[valid]
        pos1, pos2 = pos1[valid], pos2[valid]
        
        if edge_trim_days > 0 and len(t) > 100:
            dt = np.diff(t)
            gap_indices = np.where(dt > 0.04)[0]
            edge_keep = (t >= (t[0] + edge_trim_days)) & (t <= (t[-1] - edge_trim_days))
            for g_idx in gap_indices:
                t_before_gap = t[g_idx]
                t_after_gap = t[g_idx + 1]
                edge_keep &= ~((t >= (t_before_gap - edge_trim_days)) & (t <= (t_after_gap + edge_trim_days)))
            if np.sum(edge_keep) >= 500:
                t, f, ferr = t[edge_keep], f[edge_keep], ferr[edge_keep]
                mom1, mom2 = mom1[edge_keep], mom2[edge_keep]
                pos1, pos2 = pos1[edge_keep], pos2[edge_keep]
        
        med_flux = np.nanmedian(f)
        f = f / med_flux
        ferr = np.where(np.isfinite(ferr) & (ferr > 0), ferr / med_flux, np.nanstd(f))
        
        dx = mom1 - (pos1 if np.any(np.isfinite(pos1)) else 0.0)
        dy = mom2 - (pos2 if np.any(np.isfinite(pos2)) else 0.0)
        
        dx = np.where(np.isfinite(dx), dx, np.nanmedian(dx))
        dy = np.where(np.isfinite(dy), dy, np.nanmedian(dy))
        dx = np.nan_to_num(dx, nan=0.0)
        dy = np.nan_to_num(dy, nan=0.0)
        
        k_size = min(401, (len(dx) // 2) * 2 - 1)
        if k_size >= 3:
            dx = dx - medfilt(dx, kernel_size=k_size)
            dy = dy - medfilt(dy, kernel_size=k_size)
            
        t_list.append(t)
        f_list.append(f)
        ferr_list.append(ferr)
        dx_list.append(dx)
        dy_list.append(dy)
        
    if not t_list:
        raise ValueError("Insufficient valid cadence points after quality and gap-edge filtering.")
        
    t_all = np.concatenate(t_list)
    sort_idx = np.argsort(t_all)
    crowdsap = float(np.median(crowd_list)) if crowd_list else 1.0
    
    return (
        t_all[sort_idx],
        np.concatenate(f_list)[sort_idx],
        np.concatenate(ferr_list)[sort_idx],
        np.concatenate(dx_list)[sort_idx],
        np.concatenate(dy_list)[sort_idx],
        crowdsap
    )


def piecewise_savgol_detrend(t: np.ndarray, f: np.ndarray, mask: np.ndarray = None, max_win: int = 601) -> np.ndarray:
    if mask is None:
        f_work = f.copy()
    else:
        if np.sum(~mask) < 50:
            return np.ones_like(f)
        f_work = np.interp(t, t[~mask], f[~mask])
        
    trend = np.ones_like(f)
    split_indices = np.where(np.diff(t) > 0.04)[0] + 1
    segments = np.split(np.arange(len(t)), split_indices)
    
    for seg_idx in segments:
        if len(seg_idx) < 15:
            trend[seg_idx] = np.nanmedian(f_work[seg_idx]) if len(seg_idx) > 0 else 1.0
            continue
        win = min(max_win, (len(seg_idx) // 2) * 2 - 1)
        if win < 5:
            trend[seg_idx] = np.nanmedian(f_work[seg_idx])
        else:
            trend[seg_idx] = savgol_filter(f_work[seg_idx], window_length=win, polyorder=2)
            
    trend = np.where(np.isfinite(trend) & (trend > 0.1), trend, 1.0)
    return trend


def clip_upper_outliers(t: np.ndarray, f: np.ndarray, ferr: np.ndarray) -> tuple:
    med_f = np.median(f)
    mad_std = 1.4826 * np.median(np.abs(f - med_f))
    keep = f < (med_f + 3.0 * mad_std)
    return t[keep], f[keep], ferr[keep]


def get_top_distinct_bls_peaks(periods: np.ndarray, power_arr: np.ndarray, max_peaks: int = 3, min_sep_frac: float = 0.04) -> list:
    sorted_indices = np.argsort(power_arr)[::-1]
    selected_indices = []
    selected_periods = []
    
    for idx in sorted_indices:
        p_cand = float(periods[idx])
        is_duplicate = False
        for p_sel in selected_periods:
            if (
                abs(p_cand - p_sel) / p_sel < min_sep_frac
                or abs(p_cand - 2.0 * p_sel) / (2.0 * p_sel) < 0.02
                or abs(2.0 * p_cand - p_sel) / p_sel < 0.02
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            selected_indices.append(int(idx))
            selected_periods.append(p_cand)
            if len(selected_indices) >= max_peaks:
                break
    return selected_indices


def verify_and_filter_tpf(tic_id: str, sector: int, p_val: float, t0_val: float, dur_val: float) -> tuple:
    try:
        tpf_search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector, exptime=120)
        if len(tpf_search) == 0:
            return False, "TPF unavailable on MAST", None, None
            
        tpf = safe_download(tpf_search[0], tic_id, download_all=False)
        t_tpf = to_numpy(tpf.time)
        f_tpf = to_numpy(tpf.flux)
        aperture = np.asarray(tpf.pipeline_mask, dtype=bool)
        
        if not np.any(aperture):
            return False, "Empty SPOC pipeline aperture mask", None, None
            
        phase = fold_phase_centered(t_tpf, t0_val, p_val, in_days=True)
        in_transit = np.abs(phase) < (0.45 * dur_val)
        out_transit = (np.abs(phase) > (0.75 * dur_val)) & (np.abs(phase) < (2.5 * dur_val))
        
        if np.sum(in_transit) < 3 or np.sum(out_transit) < 10:
            return False, "Insufficient TPF cadences in transit window", None, None
            
        diff_image = np.nanmean(f_tpf[out_transit], axis=0) - np.nanmean(f_tpf[in_transit], axis=0)
        diff_image = np.nan_to_num(diff_image, nan=0.0)
        
        max_idx = np.unravel_index(np.argmax(diff_image), diff_image.shape)
        if not aperture[max_idx]:
            return (
                False,
                f"Off-target source! Max flux drop at pixel (col={max_idx[1]}, row={max_idx[0]}) outside aperture -> BEB",
                diff_image,
                aperture
            )
            
        ap_vals = diff_image[aperture]
        max_ap = float(np.max(ap_vals))
        min_ap = float(np.min(ap_vals))
        if max_ap <= 0 or min_ap < (-0.25 * max_ap):
            return (
                False,
                f"Dipole jitter inside aperture (max={max_ap:.1f}, min={min_ap:.1f} e-/s) -> Pointing drift / Variable",
                diff_image,
                aperture
            )
            
        bg_vals = diff_image[~aperture]
        bg_std = float(np.std(bg_vals)) if len(bg_vals) > 0 else 1.0
        if max_ap < 2.8 * bg_std:
            return (
                False,
                f"Weak pixel localization (peak={max_ap:.1f} vs bg_std={bg_std:.1f} e-/s) -> Background scatter",
                diff_image,
                aperture
            )
            
        return True, "OK", diff_image, aperture
        
    except Exception as e:
        return False, f"TPF read error: {str(e)[:40]}", None, None



def _verify_and_filter_tpf_cp(tic_id: str, sector: int, p_val: float, t0_val: float, dur_val: float) -> tuple:
    try:
        tpf_search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector, exptime=120)
        if len(tpf_search) == 0:
            return False, "TPF unavailable on MAST", None, None
            
        tpf = safe_download(tpf_search[0], tic_id, download_all=False)
        t_tpf = to_numpy(tpf.time)
        f_tpf = to_numpy(tpf.flux)
        aperture = np.asarray(tpf.pipeline_mask, dtype=bool)
        
        if not np.any(aperture):
            return False, "Empty SPOC pipeline aperture mask", None, None
            
        phase = fold_phase_centered(t_tpf, t0_val, p_val, in_days=True)
        in_transit = np.abs(phase) < (0.45 * dur_val)
        out_transit = (np.abs(phase) > (0.75 * dur_val)) & (np.abs(phase) < (2.5 * dur_val))
        
        if np.sum(in_transit) < 3 or np.sum(out_transit) < 10:
            return False, "Insufficient TPF cadences in transit window", None, None
            
        # VETTING-REPLAY-01 A3 (ARGON_DECISIONS item 14): per-transit baseline, MAD sigma_pix, new test order.
        maps = vr_tpf_maps(t_tpf, f_tpf, p_val, t0_val, dur_val)
        if maps is None:
            return False, "Insufficient TPF cadences in transit window", None, None
        diff_image = maps['dI']
        outcome, reason, _info = vr_tpf_decide(maps['dI'], maps['sigma'], maps['z'], aperture)
        if outcome == 'REJECT':
            return False, reason, diff_image, aperture
        return True, reason, diff_image, aperture
        
    except Exception as e:
        return False, f"TPF read error: {str(e)[:40]}", None, None


def verify_multi_epoch(
    tic_id: str,
    current_sector: int,
    t_curr: np.ndarray,
    flat_f_curr: np.ndarray,
    flat_ferr_curr: np.ndarray,
    best_p: float,
    best_t0: float,
    best_dur: float,
    depth_ppm_curr: float,
    sde_curr: float
) -> tuple:
    products_requested = []
    sectors_requested = []
    products_attempted = []
    sectors_attempted = []
    products_used = []
    sectors_used = []
    products_failed = []

    # Stage 0: Archival search query
    try:
        all_search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
    except Exception as exc:
        logger.warning("Archival search query failed for TIC %s (%s)", tic_id, exc)
        products_failed.append({
            "product_id": "UNAVAILABLE",
            "sector": int(current_sector),
            "stage": "QUERY",
            "reason": f"{type(exc).__name__}: {str(exc)}"[:150]
        })
        diag_info = {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": [],
            "archival_sectors_attempted": [],
            "archival_sectors_used": [],
            "archival_products_requested": [],
            "archival_products_attempted": [],
            "archival_products_used": [],
            "archival_products_failed": products_failed,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": f"ARCHIVAL_QUERY_FAILED: {type(exc).__name__}"
        }
        if sde_curr < 6.5:
            return False, f"Archival lookup failed ({type(exc).__name__}) and single-sector SDE < 6.5", diag_info
        return True, "Single-Sector_Only (Archival Query Error)", diag_info

    prior_indices = []
    prior_sec_nums = []
    for i, row in enumerate(all_search):
        try:
            mission_val = row.mission[0] if isinstance(row.mission, (list, tuple, np.ndarray)) else row.mission
            sec_num = int(str(mission_val).split()[-1])
        except Exception:
            sec_num = -1
        if sec_num != current_sector and sec_num > 0:
            prior_indices.append(i)
            prior_sec_nums.append(sec_num)

    if len(prior_indices) == 0:
        diag_no_prior = {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": [],
            "archival_sectors_attempted": [],
            "archival_sectors_used": [],
            "archival_products_requested": [],
            "archival_products_attempted": [],
            "archival_products_used": [],
            "archival_products_failed": [],
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "SINGLE_SECTOR_ONLY_NO_PRIOR"
        }
        if sde_curr < 6.5:
            return False, f"Single-sector only and borderline SDE ({sde_curr:.1f} < 6.5)", diag_no_prior
        return True, "Single-Sector_Only", diag_no_prior

    chosen_idx = prior_indices[-3:]
    chosen_secs = [prior_sec_nums[i] for i, idx in enumerate(prior_indices) if idx in chosen_idx]

    products_requested = [_get_stable_product_id(all_search[idx], sec, idx) for idx, sec in zip(chosen_idx, chosen_secs)]
    sectors_requested = sorted(list(dict.fromkeys(chosen_secs)))

    t_raw_list, f_raw_list, ferr_raw_list, f_init_list = [], [], [], []
    for idx, sec_num in zip(chosen_idx, chosen_secs):
        pid = _get_stable_product_id(all_search[idx], sec_num, idx)
        products_attempted.append(pid)
        if sec_num not in sectors_attempted:
            sectors_attempted.append(sec_num)
        stage = "DOWNLOAD"
        try:
            lc_p = safe_download(all_search[idx], tic_id, download_all=False)
            if lc_p is None:
                raise ValueError("safe_download returned None")

            stage = "EXTRACTION"
            tp, fp, ferrp, _, _, _ = extract_and_stitch_arrays([lc_p], edge_trim_days=0.10)
            if tp is None or len(tp) == 0:
                raise ValueError("Extracted arrays empty or None")
            if np.any(~np.isfinite(tp)) or np.any(~np.isfinite(fp)) or np.any(~np.isfinite(ferrp)):
                raise ValueError("Extracted arrays contain non-finite values")

            stage = "CLIPPING"
            tp_c, fp_c, ferrp_c = clip_upper_outliers(tp, fp, ferrp)
            if tp_c is None or len(tp_c) == 0:
                raise ValueError("Empty arrays after outlier clipping")

            stage = "DETRENDING"
            tr_init = piecewise_savgol_detrend(tp_c, fp_c, mask=None, max_win=601)
            if tr_init is None or len(tr_init) != len(tp_c):
                raise ValueError("Detrending returned invalid trend")
            if np.any(~np.isfinite(tr_init)) or np.any(tr_init <= 0):
                raise ValueError("Trend contains non-finite or non-positive values")

            norm_f = fp_c / tr_init
            norm_ferr = ferrp_c / tr_init
            if np.any(~np.isfinite(norm_f)) or np.any(~np.isfinite(norm_ferr)):
                raise ValueError("Normalized flux or error contains non-finite values")

            t_raw_list.append(tp_c)
            f_raw_list.append(fp_c)
            ferr_raw_list.append(norm_ferr)
            f_init_list.append(norm_f)
            products_used.append(pid)
            sectors_used.append(sec_num)
        except Exception as exc:
            products_failed.append({
                "product_id": pid,
                "sector": int(sec_num),
                "stage": stage,
                "reason": f"{type(exc).__name__}: {str(exc)}"[:150]
            })
            continue

    if not t_raw_list:
        diag_all_failed = {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": sectors_requested,
            "archival_sectors_attempted": sorted(list(dict.fromkeys(sectors_attempted))),
            "archival_sectors_used": [],
            "archival_products_requested": products_requested,
            "archival_products_attempted": products_attempted,
            "archival_products_used": [],
            "archival_products_failed": products_failed,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "ARCHIVAL_DOWNLOAD_OR_PREPARATION_FAILED"
        }
        if sde_curr < 6.5:
            return False, "Archival sectors unreadable and SDE < 6.5", diag_all_failed
        return True, "Single-Sector_Only (Archival Download Failed)", diag_all_failed

    unique_secs_used = sorted(list(dict.fromkeys(sectors_used)))
    n_used_secs = len(unique_secs_used)
    used_secs_str = ",".join(str(s) for s in unique_secs_used)

    # Stage 5: Combined multi-sector concatenation, detrending, refinement, and evaluation
    try:
        comb_stage = "CONCATENATION"
        t_prior = np.concatenate(t_raw_list)
        f_prior_raw = np.concatenate(f_raw_list)
        f_prior_init = np.concatenate(f_init_list)
        ferr_prior = np.concatenate(ferr_raw_list)

        t_comb = np.concatenate([t_prior, t_curr])
        f_comb = np.concatenate([f_prior_init, flat_f_curr])
        ferr_comb = np.concatenate([ferr_prior, flat_ferr_curr])
        sort_c = np.argsort(t_comb)
        t_comb, f_comb, ferr_comb = t_comb[sort_c], f_comb[sort_c], ferr_comb[sort_c]

        comb_stage = "TIMING_UNCERTAINTY"
        delta_t_arch = abs(float(np.median(t_curr) - np.median(t_prior)))
        baseline_curr = float(np.max(t_curr) - np.min(t_curr))

        # Cramer-Rao Fisher timing uncertainty from discovery sector (Carter et al. 2008)
        sigma_t0, sigma_p = cramer_rao_timing_uncertainty(best_dur, sde_curr, baseline_curr, best_p)

        if delta_t_arch <= 60.0:
            comb_stage = "PROJECTION"
            # Contiguous or near-contiguous sectors: strict zero-freedom forward projection (N_trials = 1)
            # Refined ephemeris fit not performed; unused refinement is np.nan
            refined_p = np.nan
            refined_t0 = np.nan
            n_trials = 1
            snr_crit = 2.4
            is_projected_strict = True
            ephem_src = "DISCOVERY_SINGLE_SECTOR"
            ephem_stat = "FALLBACK_DISCOVERY"
            fallback_reas = "STRICT_PROJECTION_NEARBY_SECTOR"
        else:
            comb_stage = "REFINEMENT"
            # Multi-year archival baseline: search strictly within Cramer-Rao 2-sigma corridor
            # with Sidak-Bonferroni trial penalty (eliminates circular overfitting of noise dips)
            p_hw = 2.0 * sigma_p
            p_min = max(0.5, best_p - p_hw)
            p_max = best_p + p_hw
            delta_p_indep = (best_p * best_dur) / delta_t_arch
            n_trials = max(1, int(np.ceil((2.0 * p_hw) / max(delta_p_indep, 1e-6))))
            snr_crit = max(2.8, float(np.sqrt(2.0 * np.log(max(n_trials, 2))) + 1.2))
            is_projected_strict = False

            n_grid_pts = int(np.clip(n_trials * 4, 60, 4000))
            fine_periods = np.linspace(p_min, p_max, n_grid_pts)
            fine_durs = np.array([max(0.03, best_dur * 0.90), best_dur, min(0.22, best_dur * 1.10)])

            bls_comb = BoxLeastSquares(t_comb, f_comb, dy=ferr_comb)
            res_comb = bls_comb.power(fine_periods, fine_durs)
            best_c_idx = int(np.nanargmax(res_comb.power))
            cand_refined_p = float(res_comb.period[best_c_idx])
            cand_refined_t0 = float(res_comb.transit_time[best_c_idx])
            if (
                cand_refined_p is not None
                and cand_refined_t0 is not None
                and np.isfinite(cand_refined_p)
                and cand_refined_p > 0
                and np.isfinite(cand_refined_t0)
            ):
                refined_p = cand_refined_p
                refined_t0 = cand_refined_t0
                ephem_src = "REFINED_MULTI_SECTOR"
                ephem_stat = "REFINED_VALID"
                fallback_reas = "NONE"
            else:
                refined_p = np.nan
                refined_t0 = np.nan
                ephem_src = "DISCOVERY_SINGLE_SECTOR"
                ephem_stat = "FALLBACK_DISCOVERY"
                fallback_reas = "REFINED_EPHEMERIS_INVALID"

        comb_stage = "DETRENDING"
        eff_p = refined_p if (refined_p is not None and np.isfinite(refined_p)) else best_p
        eff_t0 = refined_t0 if (refined_t0 is not None and np.isfinite(refined_t0)) else best_t0

        phase_prior = fold_phase_centered(t_prior, eff_t0, eff_p, in_days=True)
        tr_mask_prior = np.abs(phase_prior) < (0.6 * best_dur)
        trend_prior_ref = piecewise_savgol_detrend(t_prior, f_prior_raw, mask=tr_mask_prior, max_win=601)
        f_prior = f_prior_raw / trend_prior_ref

        comb_stage = "EVALUATION"
        in_tr_p = np.abs(phase_prior) < (0.45 * best_dur)
        out_tr_p = np.abs(phase_prior) > (0.65 * best_dur)

        base_diag = {
            "n_archival_sectors": n_used_secs,
            "archival_sectors_str": used_secs_str,
            "refined_p": refined_p,
            "refined_t0": refined_t0,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": n_trials,
            "snr_crit": snr_crit,
            "is_projected_strict": is_projected_strict,
            "delta_t_arch_days": delta_t_arch,
            "sigma_p_sec": float(sigma_p * 86400.0),
            "sigma_t0_sec": float(sigma_t0 * 86400.0),
            "archival_sectors_requested": sectors_requested,
            "archival_sectors_attempted": sorted(list(dict.fromkeys(sectors_attempted))),
            "archival_sectors_used": unique_secs_used,
            "archival_products_requested": products_requested,
            "archival_products_attempted": products_attempted,
            "archival_products_used": products_used,
            "archival_products_failed": products_failed,
            "ephemeris_source": ephem_src,
            "ephemeris_status": ephem_stat,
            "ephemeris_fallback_reason": fallback_reas
        }

        if np.sum(in_tr_p) < 5:
            return False, f"Insufficient archival in-transit points in sectors {used_secs_str}", base_diag

        base_p = float(np.nanmedian(f_prior[out_tr_p]))
        depth_prior_ppm = float((base_p - np.nanmean(f_prior[in_tr_p])) * 1e6)
        err_prior_ppm = float(np.nanstd(f_prior[out_tr_p]) / np.sqrt(np.sum(in_tr_p)) * 1e6)
        archival_snr = depth_prior_ppm / max(err_prior_ppm, 1.0)
        base_diag["archival_snr"] = archival_snr

        # Control Phase Null Test: measure flux deficit at 8 off-transit control phases
        ctrl_phases = [0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85]
        ctrl_snrs = []
        for cp in ctrl_phases:
            # Compare on the periodic orbit: phase_prior is signed, while cp is in [0, 1).
            cp_distance = np.abs(cyclic_orbital_distance(phase_prior, cp * eff_p, period=eff_p))
            in_cp = cp_distance < (0.45 * best_dur)
            if np.sum(in_cp) >= 5:
                cp_depth = float((base_p - np.nanmean(f_prior[in_cp])) * 1e6)
                ctrl_snrs.append(cp_depth / max(err_prior_ppm, 1.0))
        max_ctrl_snr = max(ctrl_snrs) if ctrl_snrs else 0.0

        if archival_snr < snr_crit or depth_prior_ppm < 0.25 * depth_ppm_curr:
            return (
                False,
                f"Archival non-detection in sectors {used_secs_str} (Prior Depth={depth_prior_ppm:.0f}ppm, SNR={archival_snr:.1f}sigma < crit={snr_crit:.1f}sigma, N_trials={n_trials}) -> Transient artifact",
                base_diag
            )

        if not is_projected_strict and (archival_snr < max_ctrl_snr + 1.5):
            return (
                False,
                f"Archival signal failed control phase null test in sectors {used_secs_str} (Transit SNR={archival_snr:.1f}sigma vs control peak {max_ctrl_snr:.1f}sigma) -> Red noise fluctuation",
                base_diag
            )

        # Multi-sector secondary eclipse check on combined baseline
        is_sec_eb, sec_snr, sec_depth_ppm, sec_phi, _ = scan_full_orbit_secondary_eclipse(
            t_comb, f_comb, eff_p, eff_t0, best_dur, depth_prior_ppm
        )
        if is_sec_eb:
            return (
                False,
                f"Multi-sector secondary eclipse in sectors {used_secs_str} at phase {sec_phi:.2f} ({sec_depth_ppm:.0f}ppm, {sec_snr:.1f}sigma) -> Eccentric EB",
                base_diag
            )

        win_phase = min(0.15 * eff_p, 3.0 * best_dur)
        vis_p = np.abs(phase_prior) <= win_phase
        bins = np.linspace(-win_phase, win_phase, 41)
        arch_means, arch_edges, _ = stats.binned_statistic(phase_prior[vis_p], f_prior[vis_p], statistic="mean", bins=bins)
        arch_centers = 0.5 * (arch_edges[:-1] + arch_edges[1:])

        proxy_sde = sde_curr * np.sqrt(1.0 + n_used_secs * 0.5)
        base_diag.update({
            "est_multi_sde_proxy": proxy_sde,
            "multi_sde": proxy_sde,
            "arch_bin_centers": arch_centers,
            "arch_bin_means": arch_means,
        })
        return True, f"Confirmed_in_Sectors_{used_secs_str}", base_diag

    except Exception as exc:
        logger.warning("Archival %s error for TIC %s (%s)", comb_stage.lower(), tic_id, exc)
        failed_stage = comb_stage
        products_failed.append({
            "product_id": "COMBINED_PIPELINE",
            "sector": int(current_sector),
            "stage": failed_stage,
            "reason": f"{type(exc).__name__}: {str(exc)}"[:150]
        })
        diag_refine_failed = {
            "n_archival_sectors": n_used_secs,
            "archival_sectors_str": used_secs_str,
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": sectors_requested,
            "archival_sectors_attempted": sorted(list(dict.fromkeys(sectors_attempted))),
            "archival_sectors_used": unique_secs_used,
            "archival_products_requested": products_requested,
            "archival_products_attempted": products_attempted,
            "archival_products_used": products_used,
            "archival_products_failed": products_failed,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": f"ARCHIVAL_{failed_stage}_FAILED: {type(exc).__name__}"
        }
        if sde_curr < 6.5:
            return False, f"Archival evaluation failed ({failed_stage}: {type(exc).__name__}) and single-sector SDE < 6.5", diag_refine_failed
        return True, f"Single-Sector_Only (Archival {failed_stage.capitalize()} Error: {type(exc).__name__})", diag_refine_failed



def _verify_multi_epoch_cp(
    tic_id: str,
    current_sector: int,
    t_curr: np.ndarray,
    flat_f_curr: np.ndarray,
    flat_ferr_curr: np.ndarray,
    best_p: float,
    best_t0: float,
    best_dur: float,
    depth_ppm_curr: float,
    sde_curr: float
) -> tuple:
    products_requested = []
    sectors_requested = []
    products_attempted = []
    sectors_attempted = []
    products_used = []
    sectors_used = []
    products_failed = []

    # Stage 0: Archival search query
    try:
        all_search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
    except Exception as exc:
        logger.warning("Archival search query failed for TIC %s (%s)", tic_id, exc)
        products_failed.append({
            "product_id": "UNAVAILABLE",
            "sector": int(current_sector),
            "stage": "QUERY",
            "reason": f"{type(exc).__name__}: {str(exc)}"[:150]
        })
        diag_info = {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": [],
            "archival_sectors_attempted": [],
            "archival_sectors_used": [],
            "archival_products_requested": [],
            "archival_products_attempted": [],
            "archival_products_used": [],
            "archival_products_failed": products_failed,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": f"ARCHIVAL_QUERY_FAILED: {type(exc).__name__}"
        }
        if sde_curr < 6.5:
            return False, f"Archival lookup failed ({type(exc).__name__}) and single-sector SDE < 6.5", diag_info
        return True, "Single-Sector_Only (Archival Query Error)", diag_info

    prior_indices = []
    prior_sec_nums = []
    for i, row in enumerate(all_search):
        try:
            mission_val = row.mission[0] if isinstance(row.mission, (list, tuple, np.ndarray)) else row.mission
            sec_num = int(str(mission_val).split()[-1])
        except Exception:
            sec_num = -1
        if sec_num != current_sector and sec_num > 0:
            prior_indices.append(i)
            prior_sec_nums.append(sec_num)

    if len(prior_indices) == 0:
        diag_no_prior = {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": [],
            "archival_sectors_attempted": [],
            "archival_sectors_used": [],
            "archival_products_requested": [],
            "archival_products_attempted": [],
            "archival_products_used": [],
            "archival_products_failed": [],
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "SINGLE_SECTOR_ONLY_NO_PRIOR"
        }
        if sde_curr < 6.5:
            return False, f"Single-sector only and borderline SDE ({sde_curr:.1f} < 6.5)", diag_no_prior
        return True, "Single-Sector_Only", diag_no_prior

    chosen_idx = prior_indices[-3:]
    chosen_secs = [prior_sec_nums[i] for i, idx in enumerate(prior_indices) if idx in chosen_idx]

    products_requested = [_get_stable_product_id(all_search[idx], sec, idx) for idx, sec in zip(chosen_idx, chosen_secs)]
    sectors_requested = sorted(list(dict.fromkeys(chosen_secs)))

    t_raw_list, f_raw_list, ferr_raw_list, f_init_list = [], [], [], []
    for idx, sec_num in zip(chosen_idx, chosen_secs):
        pid = _get_stable_product_id(all_search[idx], sec_num, idx)
        products_attempted.append(pid)
        if sec_num not in sectors_attempted:
            sectors_attempted.append(sec_num)
        stage = "DOWNLOAD"
        try:
            lc_p = safe_download(all_search[idx], tic_id, download_all=False)
            if lc_p is None:
                raise ValueError("safe_download returned None")

            stage = "EXTRACTION"
            tp, fp, ferrp, _, _, _ = extract_and_stitch_arrays([lc_p], edge_trim_days=0.10)
            if tp is None or len(tp) == 0:
                raise ValueError("Extracted arrays empty or None")
            if np.any(~np.isfinite(tp)) or np.any(~np.isfinite(fp)) or np.any(~np.isfinite(ferrp)):
                raise ValueError("Extracted arrays contain non-finite values")

            stage = "CLIPPING"
            tp_c, fp_c, ferrp_c = clip_upper_outliers(tp, fp, ferrp)
            if tp_c is None or len(tp_c) == 0:
                raise ValueError("Empty arrays after outlier clipping")

            stage = "DETRENDING"
            tr_init = piecewise_savgol_detrend(tp_c, fp_c, mask=None, max_win=601)
            if tr_init is None or len(tr_init) != len(tp_c):
                raise ValueError("Detrending returned invalid trend")
            if np.any(~np.isfinite(tr_init)) or np.any(tr_init <= 0):
                raise ValueError("Trend contains non-finite or non-positive values")

            norm_f = fp_c / tr_init
            norm_ferr = ferrp_c / tr_init
            if np.any(~np.isfinite(norm_f)) or np.any(~np.isfinite(norm_ferr)):
                raise ValueError("Normalized flux or error contains non-finite values")

            t_raw_list.append(tp_c)
            f_raw_list.append(fp_c)
            ferr_raw_list.append(norm_ferr)
            f_init_list.append(norm_f)
            products_used.append(pid)
            sectors_used.append(sec_num)
        except Exception as exc:
            products_failed.append({
                "product_id": pid,
                "sector": int(sec_num),
                "stage": stage,
                "reason": f"{type(exc).__name__}: {str(exc)}"[:150]
            })
            continue

    if not t_raw_list:
        diag_all_failed = {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": sectors_requested,
            "archival_sectors_attempted": sorted(list(dict.fromkeys(sectors_attempted))),
            "archival_sectors_used": [],
            "archival_products_requested": products_requested,
            "archival_products_attempted": products_attempted,
            "archival_products_used": [],
            "archival_products_failed": products_failed,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": "ARCHIVAL_DOWNLOAD_OR_PREPARATION_FAILED"
        }
        if sde_curr < 6.5:
            return False, "Archival sectors unreadable and SDE < 6.5", diag_all_failed
        return True, "Single-Sector_Only (Archival Download Failed)", diag_all_failed

    unique_secs_used = sorted(list(dict.fromkeys(sectors_used)))
    n_used_secs = len(unique_secs_used)
    used_secs_str = ",".join(str(s) for s in unique_secs_used)

    # Stage 5: Combined multi-sector concatenation, detrending, refinement, and evaluation
    try:
        comb_stage = "CONCATENATION"
        t_prior = np.concatenate(t_raw_list)
        f_prior_raw = np.concatenate(f_raw_list)
        f_prior_init = np.concatenate(f_init_list)
        ferr_prior = np.concatenate(ferr_raw_list)

        t_comb = np.concatenate([t_prior, t_curr])
        f_comb = np.concatenate([f_prior_init, flat_f_curr])
        ferr_comb = np.concatenate([ferr_prior, flat_ferr_curr])
        sort_c = np.argsort(t_comb)
        t_comb, f_comb, ferr_comb = t_comb[sort_c], f_comb[sort_c], ferr_comb[sort_c]

        comb_stage = "TIMING_UNCERTAINTY"
        delta_t_arch = abs(float(np.median(t_curr) - np.median(t_prior)))
        baseline_curr = float(np.max(t_curr) - np.min(t_curr))

        # VETTING-REPLAY-01 A1 (ARGON_DECISIONS item 12): pivot epoch, interim floor, per-sector mode.
        vr_t = vr_timing(best_t0, best_p, best_dur, sde_curr, t_curr, list(zip(sectors_used, t_raw_list)))
        sigma_t0, sigma_p = vr_t['sigma_t0'], vr_t['sigma_p']
        vr_diag = dict(t0_pivot=vr_t['t0_pivot'], sigma_t0_d=sigma_t0, sigma_p_d=sigma_p,
                       sigma_pred_d={str(k): v for k, v in vr_t['sigma_pred'].items()},
                       max_sigma_pred_d=vr_t['max_sigma_pred'], mode=1 if vr_t['mode1'] else 2)

        if vr_t['mode1']:
            comb_stage = "PROJECTION"
            # Contiguous or near-contiguous sectors: strict zero-freedom forward projection (N_trials = 1)
            # Refined ephemeris fit not performed; unused refinement is np.nan
            refined_p = np.nan
            refined_t0 = np.nan
            n_trials = 1
            snr_crit = 2.4
            is_projected_strict = True
            ephem_src = "DISCOVERY_SINGLE_SECTOR"
            ephem_stat = "FALLBACK_DISCOVERY"
            fallback_reas = "STRICT_PROJECTION_NEARBY_SECTOR"
        else:
            comb_stage = "REFINEMENT"
            # Multi-year archival baseline: search strictly within Cramer-Rao 2-sigma corridor
            # with Sidak-Bonferroni trial penalty (eliminates circular overfitting of noise dips)
            p_hw = 3.5 * sigma_p
            p_min = max(0.5, best_p - p_hw)
            p_max = best_p + p_hw
            delta_p_indep = (best_p * best_dur) / delta_t_arch
            n_trials = max(1, int(np.ceil((2.0 * p_hw) / max(delta_p_indep, 1e-6))))
            snr_crit = max(2.8, float(np.sqrt(2.0 * np.log(max(n_trials, 2))) + 1.2))
            is_projected_strict = False

            n_grid_pts = int(np.clip(n_trials * 4, 60, 4000))
            fine_periods = np.linspace(p_min, p_max, n_grid_pts)
            fine_durs = np.array([max(0.03, best_dur * 0.90), best_dur, min(0.22, best_dur * 1.10)])

            # VETTING-REPLAY-02 item 18 (P3): drift-based grid, two stages above 4000 points.
            _vr_ref = vr_refine_p3(t_comb, f_comb, ferr_comb, p_min, p_max, n_trials, fine_durs,
                                   vr_t['t0_pivot'], 3.0 * sigma_t0, VR_DT_GRID,
                                   max(abs(v) for v in vr_t['n_s'].values()), best_dur)
            vr_diag.update(p3_n_req=_vr_ref['n_req'], p3_n_grid=_vr_ref['n_grid'], p3_stages=_vr_ref['stages'])
            cand_refined_p = _vr_ref['period']
            cand_refined_t0 = _vr_ref['t0']
            if (
                cand_refined_p is not None
                and cand_refined_t0 is not None
                and np.isfinite(cand_refined_p)
                and cand_refined_p > 0
                and np.isfinite(cand_refined_t0)
            ):
                refined_p = cand_refined_p
                refined_t0 = cand_refined_t0
                ephem_src = "REFINED_MULTI_SECTOR"
                ephem_stat = "REFINED_VALID"
                fallback_reas = "NONE"
            else:
                refined_p = np.nan
                refined_t0 = np.nan
                ephem_src = "DISCOVERY_SINGLE_SECTOR"
                ephem_stat = "FALLBACK_DISCOVERY"
                fallback_reas = "REFINED_EPHEMERIS_INVALID"

        vr_diag.update(n_trials=n_trials, snr_crit=snr_crit, refined_p=refined_p, refined_t0=refined_t0)
        comb_stage = "DETRENDING"
        eff_p = refined_p if (refined_p is not None and np.isfinite(refined_p)) else best_p
        eff_t0 = refined_t0 if (refined_t0 is not None and np.isfinite(refined_t0)) else best_t0

        phase_prior = fold_phase_centered(t_prior, eff_t0, eff_p, in_days=True)
        tr_mask_prior = np.abs(phase_prior) < (0.6 * best_dur)
        trend_prior_ref = piecewise_savgol_detrend(t_prior, f_prior_raw, mask=tr_mask_prior, max_win=601)
        f_prior = f_prior_raw / trend_prior_ref

        comb_stage = "EVALUATION"
        in_tr_p = np.abs(phase_prior) < (0.45 * best_dur)
        out_tr_p = np.abs(phase_prior) > (0.65 * best_dur)

        base_diag = {
            "n_archival_sectors": n_used_secs,
            "archival_sectors_str": used_secs_str,
            "refined_p": refined_p,
            "refined_t0": refined_t0,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": n_trials,
            "snr_crit": snr_crit,
            "is_projected_strict": is_projected_strict,
            "delta_t_arch_days": delta_t_arch,
            "sigma_p_sec": float(sigma_p * 86400.0),
            "sigma_t0_sec": float(sigma_t0 * 86400.0),
            "archival_sectors_requested": sectors_requested,
            "archival_sectors_attempted": sorted(list(dict.fromkeys(sectors_attempted))),
            "archival_sectors_used": unique_secs_used,
            "archival_products_requested": products_requested,
            "archival_products_attempted": products_attempted,
            "archival_products_used": products_used,
            "archival_products_failed": products_failed,
            "ephemeris_source": ephem_src,
            "ephemeris_status": ephem_stat,
            "ephemeris_fallback_reason": fallback_reas
        }

        base_diag["vr"] = vr_diag
        if not is_projected_strict and refined_p is not None and np.isfinite(refined_p):
            # VETTING-REPLAY-02 item 19 (P4): the refined ephemeris must keep the discovery-sector transits.
            _tau_c = np.abs(((t_curr - eff_t0 + 0.5 * eff_p) % eff_p) - 0.5 * eff_p)
            _in_c, _out_c = _tau_c < 0.45 * best_dur, _tau_c > 0.65 * best_dur
            _d_disc = float((np.nanmedian(flat_f_curr[_out_c]) - np.nanmean(flat_f_curr[_in_c])) * 1e6) if (np.any(_in_c) and np.any(_out_c)) else float("nan")
            vr_diag.update(p4_discovery_depth_ppm=_d_disc, p4_bls_depth_ppm=float(depth_ppm_curr))
            if not (np.isfinite(_d_disc) and _d_disc >= 0.50 * depth_ppm_curr):
                base_diag.update(refined_p=np.nan, refined_t0=np.nan, ephemeris_source="DISCOVERY_SINGLE_SECTOR",
                                 ephemeris_status="FALLBACK_DISCOVERY", ephemeris_fallback_reason="P4_REFINEMENT_LOST_DISCOVERY_TRANSIT")
                return (True, f"REVIEW_ARCHIVAL_REFINEMENT_LOST_DISCOVERY_TRANSIT: discovery-sector depth {_d_disc:.0f}ppm < 0.50 x "
                              f"{depth_ppm_curr:.0f}ppm at refined P={eff_p:.6f}", base_diag)
        if np.sum(in_tr_p) < 5:
            return False, f"Insufficient archival in-transit points in sectors {used_secs_str}", base_diag

        base_p = float(np.nanmedian(f_prior[out_tr_p]))
        depth_prior_ppm = float((base_p - np.nanmean(f_prior[in_tr_p])) * 1e6)
        err_prior_ppm = float(np.nanstd(f_prior[out_tr_p]) / np.sqrt(np.sum(in_tr_p)) * 1e6)
        archival_snr = depth_prior_ppm / max(err_prior_ppm, 1.0)
        base_diag["archival_snr"] = archival_snr

        # Control Phase Null Test: measure flux deficit at 8 off-transit control phases
        ctrl_phases = [0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85]
        ctrl_snrs = []
        for cp in ctrl_phases:
            # Compare on the periodic orbit: phase_prior is signed, while cp is in [0, 1).
            cp_distance = np.abs(cyclic_orbital_distance(phase_prior, cp * eff_p, period=eff_p))
            in_cp = cp_distance < (0.45 * best_dur)
            if np.sum(in_cp) >= 5:
                cp_depth = float((base_p - np.nanmean(f_prior[in_cp])) * 1e6)
                ctrl_snrs.append(cp_depth / max(err_prior_ppm, 1.0))
        max_ctrl_snr = max(ctrl_snrs) if ctrl_snrs else 0.0

        if archival_snr < snr_crit or depth_prior_ppm < 0.25 * depth_ppm_curr:
            _msg = f"Archival non-detection in sectors {used_secs_str} (Prior Depth={depth_prior_ppm:.0f}ppm, SNR={archival_snr:.1f}sigma < crit={snr_crit:.1f}sigma, N_trials={n_trials}) -> Transient artifact"
            _snr_exp = float(depth_ppm_curr / max(err_prior_ppm, 1.0))  # item 12; R_crowd = 1 (PDCSAP)
            _n_full = vr_count_full_transits(t_prior, eff_p, eff_t0, best_dur)  # D-3
            base_diag["vr"].update(snr_exp_arch=_snr_exp, n_full_archival_transits=_n_full)
            if is_projected_strict and _n_full >= 2 and _snr_exp >= 6.0:
                return (False, _msg, base_diag)
            return (True, "REVIEW_ARCHIVAL_NONDETECTION: " + _msg, base_diag)

        if not is_projected_strict and (archival_snr < max_ctrl_snr + 1.5):
            return (False, f"Archival signal failed control phase null test in sectors {used_secs_str} (Transit SNR={archival_snr:.1f}sigma vs control peak {max_ctrl_snr:.1f}sigma) -> Red noise fluctuation", base_diag)

        if depth_prior_ppm < 0.50 * depth_ppm_curr:
            return (True, f"REVIEW_ARCHIVAL_DEPTH_BAND: archival depth {depth_prior_ppm:.0f}ppm is 25 to 50% of "
                          f"{depth_ppm_curr:.0f}ppm; secondary scan skipped", base_diag)

        # Multi-sector secondary eclipse check on combined baseline
        _sec = {}
        is_sec_eb, sec_snr, sec_depth_ppm, sec_phi, _zc = _scan_full_orbit_secondary_eclipse_cp(
            t_comb, f_comb, eff_p, eff_t0, best_dur, depth_prior_ppm, details=_sec
        )
        if is_sec_eb:
            return (False, f"Multi-sector secondary eclipse in sectors {used_secs_str} at phase {sec_phi:.2f} ({sec_depth_ppm:.0f}ppm, {sec_snr:.1f}sigma) -> {'Circular EB' if abs(sec_phi - 0.50) <= 0.04 else 'Eccentric EB'}, z_crit={_zc:.2f}", base_diag)
        if _sec and 3.1 < _sec['circ_snr'] < _sec['z_crit_circ'] and _sec['circ_depth'] > 100.0 \
                and _sec['circ_depth'] >= 0.10 * depth_prior_ppm:
            return (True, f"REVIEW_ARCHIVAL_MARGINAL_SECONDARY: circular {_sec['circ_snr']:.2f}sigma, "
                          f"3.1 < z < z_crit_circ={_sec['z_crit_circ']:.2f}", base_diag)

        win_phase = min(0.15 * eff_p, 3.0 * best_dur)
        vis_p = np.abs(phase_prior) <= win_phase
        bins = np.linspace(-win_phase, win_phase, 41)
        arch_means, arch_edges, _ = stats.binned_statistic(phase_prior[vis_p], f_prior[vis_p], statistic="mean", bins=bins)
        arch_centers = 0.5 * (arch_edges[:-1] + arch_edges[1:])

        proxy_sde = sde_curr * np.sqrt(1.0 + n_used_secs * 0.5)
        base_diag.update({
            "est_multi_sde_proxy": proxy_sde,
            "multi_sde": proxy_sde,
            "arch_bin_centers": arch_centers,
            "arch_bin_means": arch_means,
        })
        return True, f"Confirmed_in_Sectors_{used_secs_str}", base_diag

    except Exception as exc:
        logger.warning("Archival %s error for TIC %s (%s)", comb_stage.lower(), tic_id, exc)
        failed_stage = comb_stage
        products_failed.append({
            "product_id": "COMBINED_PIPELINE",
            "sector": int(current_sector),
            "stage": failed_stage,
            "reason": f"{type(exc).__name__}: {str(exc)}"[:150]
        })
        diag_refine_failed = {
            "n_archival_sectors": n_used_secs,
            "archival_sectors_str": used_secs_str,
            "refined_p": np.nan,
            "refined_t0": np.nan,
            "archival_snr": np.nan,
            "est_multi_sde_proxy": float(sde_curr),
            "multi_sde": float(sde_curr),
            "arch_bin_centers": None,
            "arch_bin_means": None,
            "n_trials": 1,
            "snr_crit": 2.4,
            "is_projected_strict": True,
            "delta_t_arch_days": 0.0,
            "sigma_p_sec": np.nan,
            "sigma_t0_sec": np.nan,
            "archival_sectors_requested": sectors_requested,
            "archival_sectors_attempted": sorted(list(dict.fromkeys(sectors_attempted))),
            "archival_sectors_used": unique_secs_used,
            "archival_products_requested": products_requested,
            "archival_products_attempted": products_attempted,
            "archival_products_used": products_used,
            "archival_products_failed": products_failed,
            "ephemeris_source": "DISCOVERY_SINGLE_SECTOR",
            "ephemeris_status": "FALLBACK_DISCOVERY",
            "ephemeris_fallback_reason": f"ARCHIVAL_{failed_stage}_FAILED: {type(exc).__name__}"
        }
        if sde_curr < 6.5:
            return False, f"Archival evaluation failed ({failed_stage}: {type(exc).__name__}) and single-sector SDE < 6.5", diag_refine_failed
        return True, f"Single-Sector_Only (Archival {failed_stage.capitalize()} Error: {type(exc).__name__})", diag_refine_failed


def analyze_single_target(
    tic_id: str,
    sector: int,
    min_period: float = 0.8,
    max_period: float = 13.0,
    run_id: str = None,
    attempt_id: str = None,
    *,
    observer_context: Any = None,
    odd_even_method: str = "legacy"
) -> AnalysisResult:
    """
    Worker entrypoint. Guarantees the per-target FITS cache is purged on EVERY non-candidate
    exit path (early REJECT_CROWDSAP / REJECT_RUWE / ... returns previously skipped the purge,
    leaving ~65% of downloads on disk and overflowing scratch quotas on HPC nodes).
    Returns an AnalysisResult tuple (tic_id, status_code, message, cand_data) bound to current
    run_id and attempt_id, isolating archival vetting diagnostics to the current attempt.
    """
    result = None
    ledger = None
    if observer_context is not None and ObserverContext is not None and SubmissionLedger is not None:
        try:
            ctx = ObserverContext.from_any(observer_context)
            if ctx is not None:
                ledger = SubmissionLedger(ctx, tic_id, sector)
                ledger.record_submission_begin()
        except Exception:
            pass

    if run_id is None:
        run_id = CURRENT_RUN_ID
    if attempt_id is None:
        import uuid
        attempt_id = f"att_{str(tic_id).lstrip('0')}_s{sector}_{uuid.uuid4().hex[:8]}"

    if ledger is not None:
        try:
            ledger.bind_attempt_id(attempt_id)
        except Exception:
            pass

    try:
        kwargs = dict(observer_context=observer_context, ledger=ledger)
        if odd_even_method not in ("legacy", "v2"):
            raise ValueError("odd_even_method must be legacy or v2")
        if odd_even_method == "v2":
            kwargs["odd_even_method"] = odd_even_method
        result = _analyze_single_target_impl(
            tic_id, sector, min_period, max_period, run_id=run_id, attempt_id=attempt_id,
            **kwargs
        )
        return result
    finally:
        if AUTO_PURGE_CACHE and (result is None or result[1] != "CANDIDATE"):
            purge_target_cache(tic_id)


def _extract_product_identity(prod) -> dict:
    info = {}
    for attr in ("target_name", "obsid", "mission", "author", "exptime", "sequence_number", "year"):
        val = None
        if hasattr(prod, attr):
            val = getattr(prod, attr)
        elif hasattr(prod, "meta") and isinstance(prod.meta, dict) and attr in prod.meta:
            val = prod.meta.get(attr)
        elif isinstance(prod, dict) and attr in prod:
            val = prod.get(attr)

        if val is not None:
            if hasattr(val, "__getitem__") and not isinstance(val, (str, bytes, dict)):
                try:
                    val = val[0]
                except Exception:
                    pass
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            info[attr] = val

    if not info and isinstance(prod, (str, bytes)):
        info["product_repr"] = str(prod)
    elif not info and isinstance(prod, dict):
        info = dict(prod)
    elif not info:
        info["product_type"] = type(prod).__name__
    return info


def review_event_link(stronger, period, epoch, duration, times):
    """Conservative unresolved-event association, not a planet/alias classifier.

    Require every supported event of the stronger REVIEW peak to overlap a
    distinct supported event of the lower-ranked peak. Each solution uses its
    own cadence counts and support floor: max(3, ceil(0.60 * its maximum)).
    Extra lower-peak events do not resolve the original odd/even uncertainty.
    Period ratios alone are unused. Windows are measured transit intervals,
    with no fitted timing tolerance.
    """
    times = np.asarray(times, dtype=float)
    times = np.sort(times[np.isfinite(times)])
    if not len(times):
        return None
    def events(p, t0, width):
        if not all(np.isfinite(x) for x in (p, t0, width)) or p <= 0 or width <= 0:
            return np.array([])
        cycles = np.arange(np.ceil((times[0] - t0) / p),
                           np.floor((times[-1] - t0) / p) + 1)
        centers = t0 + cycles * p
        # Match compute_odd_even_v2's in-transit cadence mask.
        half_width = 0.45 * width
        left = np.searchsorted(times, centers - half_width, side='right')
        right = np.searchsorted(times, centers + half_width, side='left')
        cadence_counts = right - left
        if not len(cadence_counts):
            return np.array([])
        support_floor = max(3, int(np.ceil(0.60 * np.max(cadence_counts))))
        return np.asarray([c for c, n in zip(centers, cadence_counts)
                           if n >= support_floor])
    a = events(stronger['period_days'], stronger['t0_btjd'], stronger['duration_days'])
    b = events(period, epoch, duration)
    if len(a) < 2 or len(b) < len(a):
        return None
    half_window = (stronger['duration_days'] + duration) / 2
    pairs = []
    j = 0
    for center in a:
        while j < len(b) and b[j] < center - half_window:
            j += 1
        if j == len(b) or abs(b[j] - center) > half_window:
            return None
        pairs.append(float(b[j] - center))
        j += 1
    return dict(stronger_peak_rank=stronger['peak_rank'],
                stronger_supported_events=len(a), lower_supported_events=len(b),
                matched_events=len(pairs), center_residual_days=pairs,
                overlap_half_window_days=float(half_window))


def _analyze_single_target_impl(
    tic_id: str,
    sector: int,
    min_period: float = 0.8,
    max_period: float = 13.0,
    run_id: str = None,
    attempt_id: str = None,
    *,
    observer_context: Any = None,
    ledger: Any = None,
    odd_even_method: str = "legacy"
) -> AnalysisResult:
    if ledger is None and observer_context is not None and ObserverContext is not None and SubmissionLedger is not None:
        try:
            ctx = ObserverContext.from_any(observer_context)
            if ctx is not None:
                ledger = SubmissionLedger(ctx, tic_id, sector)
                ledger.record_submission_begin()
                if attempt_id is not None:
                    ledger.bind_attempt_id(attempt_id)
        except Exception:
            pass

    if run_id is None:
        run_id = CURRENT_RUN_ID
    if attempt_id is None:
        import uuid
        attempt_id = f"att_{str(tic_id).lstrip('0')}_s{sector}_{uuid.uuid4().hex[:8]}"
        if ledger is not None and getattr(ledger, "attempt_id", None) is None:
            try:
                ledger.bind_attempt_id(attempt_id)
            except Exception:
                pass

    # Cp is opt-in. Public helpers retain the baseline legacy contract.
    secondary_scan = _scan_full_orbit_secondary_eclipse_cp if odd_even_method == "v2" else scan_full_orbit_secondary_eclipse
    tpf_check = _verify_and_filter_tpf_cp if odd_even_method == "v2" else verify_and_filter_tpf
    archival_check = _verify_multi_epoch_cp if odd_even_method == "v2" else verify_multi_epoch

    attempt_diagnostics = []
    vetting_diagnostics = []
    review_peaks = []
    peak_evidence = None

    def evidence_safe(value):
        if isinstance(value, dict): return {k: evidence_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [evidence_safe(v) for v in value]
        if isinstance(value, np.generic): return evidence_safe(value.item())
        if isinstance(value, float) and not np.isfinite(value): return None
        return value

    def note_gate(name, status, **metrics):
        if peak_evidence is not None:
            peak_evidence['gates'].append(dict(gate=name, status=status, metrics=metrics))
            if status in ('REJECT', 'MISSING_INPUT'):
                peak_evidence['disposition'] = 'REJECT' if status == 'REJECT' else 'INCOMPLETE'
                peak_evidence['cause'] = name


    def make_res(t_id, status_val, msg_val, cand_val=None):
        if ledger is not None:
            try:
                ledger.record_target_result(
                    status_code=status_val,
                    message=msg_val,
                    has_candidate=(cand_val is not None),
                    cand_data=cand_val
                )
                ledger.record_submission_final(
                    status="COMPLETE" if status_val != "ERROR" else "ERROR"
                )
            except Exception:
                pass
        if odd_even_method == 'v2':
            for item in vetting_diagnostics:
                if item['disposition'] == 'EVALUATING':
                    item['disposition'] = 'ERROR' if status_val == 'ERROR' else 'INCOMPLETE'
                    item['cause'] = status_val
            # Separate fail-closed evidence sink: REVIEW never enters candidate CSV.
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            payload = dict(tic_id=str(t_id), run_id=run_id, attempt_id=attempt_id,
                           final_status=status_val, peaks=vetting_diagnostics)
            with open(os.path.join(OUTPUT_DIR, 'experimental_v2_vetting.jsonl'), 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(evidence_safe(payload), allow_nan=False) + '\n')
                handle.flush()
            rows = [dict(tic_id=str(t_id), run_id=run_id, attempt_id=attempt_id,
                         target_status=status_val, peak_rank=item['peak_rank'],
                         disposition=item['disposition'], cause=item['cause'],
                         evidence_json=json.dumps(evidence_safe(item), allow_nan=False))
                    for item in vetting_diagnostics]
            if rows:
                csv_path = os.path.join(OUTPUT_DIR, 'experimental_v2_vetting.csv')
                pd.DataFrame(rows).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        return AnalysisResult(
            t_id,
            status_val,
            msg_val,
            cand_val,
            run_id=run_id,
            attempt_id=attempt_id,
            archival_diagnostics=attempt_diagnostics
        )

    try:
        try:
            if ledger is not None:
                try:
                    ledger.record_spoc_search_start(target=f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector, exptime=120)
                except Exception:
                    pass
            search_res = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector, exptime=120)
        except Exception as search_exc:
            if ledger is not None:
                try:
                    ledger.record_spoc_search_error(exc=search_exc)
                except Exception:
                    pass
            raise search_exc

        if len(search_res) == 0:
            if ledger is not None:
                try:
                    ledger.record_spoc_search(num_products=0)
                except Exception:
                    pass
            return make_res(tic_id, "SKIP_NO_DATA", "SKIP: No SPOC data in target sector", None)

        if ledger is not None:
            try:
                prod_info = _extract_product_identity(search_res[0])
                ledger.record_spoc_search(num_products=len(search_res), selected_product_info=prod_info)
            except Exception:
                pass

        last_sector = sector
        lc_collection = [safe_download(search_res[0], tic_id, download_all=False)]
            
        t, f, ferr, dx, dy, crowdsap = extract_and_stitch_arrays(lc_collection, edge_trim_days=0.10)
        
        # Check if CROWDSAP was defaulted or measured from metadata
        is_default_crowdsap = True
        if lc_collection and hasattr(lc_collection[0], "meta") and isinstance(lc_collection[0].meta, dict):
            c_val = lc_collection[0].meta.get("CROWDSAP")
            if c_val is not None and np.isfinite(float(c_val)) and float(c_val) > 0:
                is_default_crowdsap = False

        # Filter 0: SPOC Aperture Crowding Metric (CROWDSAP)
        if crowdsap < 0.70:
            if ledger is not None:
                try:
                    ledger.record_crowding_gate(crowdsap=float(crowdsap), passed=False, is_default=is_default_crowdsap)
                except Exception:
                    pass
            return make_res(tic_id, "REJECT_CROWDSAP", f"REJECT: Severe aperture crowding (CROWDSAP={crowdsap:.2f} < 0.70) -> Blended field", None)
        if ledger is not None:
            try:
                ledger.record_crowding_gate(crowdsap=float(crowdsap), passed=True, is_default=is_default_crowdsap)
            except Exception:
                pass

        med_f = np.median(f)
        mad_std = 1.4826 * np.median(np.abs(f - med_f))
        keep = f < (med_f + 3.0 * mad_std)
        t, f, ferr, dx, dy = t[keep], f[keep], ferr[keep], dx[keep], dy[keep]
        
        known_transits = get_known_transits_for_tic(tic_id)
        is_toi_host = len(known_transits) > 0
        in_known_transit = np.zeros(len(t), dtype=bool)

        if is_toi_host:
            for kp in known_transits:
                pk = kp["period"]
                t0k = kp["t0"]
                durk = kp["duration"]
                phk = ((t - t0k + 0.5 * pk) % pk) - 0.5 * pk
                # Expanded buffer: +- 1.0 * duration (covers ingress/egress tails and up to +-45 min TTVs)
                in_known_transit |= (np.abs(phk) <= (1.0 * durk))
            mask_frac = float(np.mean(in_known_transit))
            if mask_frac > 0.40:
                return make_res(tic_id, "REJECT_TOI_MASK_TOO_LARGE", f"REJECT: Known TOI transits cover > 40% of light curve ({mask_frac*100:.1f}%)", None)

        if is_toi_host and np.any(in_known_transit):
            # Goal 2: Linear interpolation across known transits BEFORE Savitzky-Golay detrending prevents filter edge ringing
            f_for_trend = f.copy()
            f_for_trend[in_known_transit] = np.interp(
                t[in_known_transit], t[~in_known_transit], f[~in_known_transit]
            )
            trend = piecewise_savgol_detrend(t, f_for_trend, mask=None, max_win=601)
            flat_f_init = f / trend
            flat_ferr_init = ferr / trend

            # Create masked residual arrays for BLS periodic signal search
            keep_res = ~in_known_transit
            t_bls = t[keep_res]
            flat_f_bls = flat_f_init[keep_res]
            flat_ferr_bls = flat_ferr_init[keep_res]
            dx_bls = dx[keep_res]
            dy_bls = dy[keep_res]
        else:
            trend = piecewise_savgol_detrend(t, f, mask=None, max_win=601)
            flat_f_init = f / trend
            flat_ferr_init = ferr / trend
            keep_res = np.ones(len(t), dtype=bool)
            t_bls = t
            flat_f_bls = flat_f_init
            flat_ferr_bls = flat_ferr_init
            dx_bls = dx
            dy_bls = dy
        
        max_p = min(max_period, (t_bls[-1] - t_bls[0]) * 0.48)
        if max_p <= min_period:
            if ledger is not None:
                try:
                    ledger.record_baseline_gate(
                        span=float(t_bls[-1] - t_bls[0]),
                        min_period=float(min_period),
                        max_period=float(max_period),
                        max_p=float(max_p),
                        passed=False
                    )
                except Exception:
                    pass
            return make_res(tic_id, "REJECT_BASELINE", "REJECT: Time baseline too short", None)
        if ledger is not None:
            try:
                ledger.record_baseline_gate(
                    span=float(t_bls[-1] - t_bls[0]),
                    min_period=float(min_period),
                    max_period=float(max_period),
                    max_p=float(max_p),
                    passed=True
                )
            except Exception:
                pass
            
        freqs = np.linspace(1.0 / max_p, 1.0 / min_period, 9000)
        periods = (1.0 / freqs)[::-1]
        # Start durations at 0.045d (1.08h) to prevent BLS grid boundary collapse
        durations = np.linspace(0.045, 0.18, 12)
        
        if ledger is not None:
            try:
                ledger.record_bls_start(
                    n_freq=len(freqs),
                    min_freq=float(freqs[0]),
                    max_freq=float(freqs[-1]),
                    min_dur=float(durations[0]),
                    max_dur=float(durations[-1]),
                    n_dur=len(durations)
                )
            except Exception:
                pass

        try:
            bls_model = BoxLeastSquares(t_bls, flat_f_bls, dy=flat_ferr_bls)
            bls_res = bls_model.power(periods, durations)
            # Kovács et al. (2002) whitened linear Signal Residue SDE
            sde_arr, _ = compute_whitened_kovacs_sde(bls_res.power, window_pts=501)
            top_peak_indices = get_top_distinct_bls_peaks(periods, sde_arr, max_peaks=MAX_PEAKS_PER_STAR)
        except Exception as bls_exc:
            if ledger is not None:
                try:
                    ledger.record_bls_error(exc=bls_exc)
                except Exception:
                    pass
            raise bls_exc

        if ledger is not None:
            try:
                ledger.record_bls_search(
                    n_freq=len(freqs),
                    min_freq=float(freqs[0]),
                    max_freq=float(freqs[-1]),
                    min_dur=float(durations[0]),
                    max_dur=float(durations[-1]),
                    n_dur=len(durations),
                    window_pts=501,
                    top_peak_indices=[int(idx) for idx in top_peak_indices]
                )
            except Exception:
                pass
        
        star_info_cache = None
        first_reject_code = "EMPTY_LOW_SDE"
        first_reject_msg = "Empty"
        
        for rank_idx, peak_idx in enumerate(top_peak_indices, 1):
            peak_evidence = None
            odd_even_unresolved = False
            vet_holds = []
            best_p = float(bls_res.period[peak_idx])
            best_t0 = float(bls_res.transit_time[peak_idx])
            best_dur = float(bls_res.duration[peak_idx])
            sde = float(sde_arr[peak_idx])
            
            # Filter 1: Signal Detection Efficiency (SDE >= 6.0) & Duty Cycle
            if sde < 6.0:
                if ledger is not None:
                    try:
                        ledger.record_low_sde_gate(
                            peak_idx=int(peak_idx),
                            rank_idx=rank_idx,
                            sde=sde,
                            passed=False
                        )
                    except Exception:
                        pass
                if rank_idx == 1:
                    first_reject_code = "EMPTY_LOW_SDE"
                    first_reject_msg = f"Empty (SDE={sde:.1f})"
                break
            else:
                if ledger is not None:
                    try:
                        ledger.record_low_sde_gate(
                            peak_idx=int(peak_idx),
                            rank_idx=rank_idx,
                            sde=sde,
                            passed=True
                        )
                    except Exception:
                        pass
                
            if (best_dur / best_p) > 0.095:
                if rank_idx == 1:
                    first_reject_code = "REJECT_DUTY_CYCLE"
                    first_reject_msg = f"REJECT [SDE={sde:.1f}, P={best_p:.3f}d]: Duty cycle too wide ({best_dur/best_p:.2f}) -> Sinusoidal variable"
                continue

            # Harmonic protection: reject integer/fractional period multiples of masked known TOI planets
            if is_toi_host:
                is_harmonic = False
                harmonic_reason = ""
                for kp in known_transits:
                    pk = kp["period"]
                    for num, den in [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1)]:
                        exp_p = pk * (num / den)
                        if abs(best_p - exp_p) / exp_p < 0.015:
                            is_harmonic = True
                            harmonic_reason = f"Harmonic {num}/{den} of known TOI {kp['toi']} (P={pk:.4f}d)"
                            break
                    if is_harmonic:
                        break
                if is_harmonic:
                    if rank_idx == 1:
                        first_reject_code = "REJECT_KNOWN_PLANET_HARMONIC"
                        first_reject_msg = f"REJECT [SDE={sde:.1f}, P={best_p:.4f}d]: {harmonic_reason}"
                    continue
                
            phase_full = fold_phase_centered(t, best_t0, best_p, in_days=True)
            phase = phase_full[keep_res]
            transit_mask = (np.abs(phase_full) < (0.6 * best_dur)) | in_known_transit
            
            if np.sum(np.abs(phase) < (0.6 * best_dur)) < 6 or np.sum(~transit_mask) < 100:
                if rank_idx == 1:
                    first_reject_code = "REJECT_POINTS"
                    first_reject_msg = f"REJECT [SDE={sde:.1f}, P={best_p:.3f}d]: Insufficient in-transit points"
                continue
                
            f_for_ref = f.copy()
            if is_toi_host and np.any(in_known_transit):
                f_for_ref[in_known_transit] = np.interp(
                    t[in_known_transit], t[~in_known_transit], f[~in_known_transit]
                )
            trend_refined = piecewise_savgol_detrend(t, f_for_ref, mask=transit_mask, max_win=601)
            flat_f = (f / trend_refined)[keep_res]
            flat_ferr = (ferr / trend_refined)[keep_res]
            t_curr_eval = t_bls
            dx_curr_eval = dx_bls
            dy_curr_eval = dy_bls
            
            in_tr = np.abs(phase) < (0.45 * best_dur)
            out_tr = np.abs(phase) > (0.65 * best_dur)
            baseline = float(np.nanmedian(flat_f[out_tr]))
            depth_ppm = float((baseline - np.nanmean(flat_f[in_tr])) * 1e6)
            # SPOC PDCSAP light curves are already corrected for aperture crowding (CROWDSAP).
            # Do NOT divide by crowdsap again to prevent double dilution overcorrection.
            true_depth_ppm = depth_ppm
            
            base_telemetry = f"[Peak#{rank_idx} | SDE={sde:.1f} | P={best_p:.4f}d | Depth={depth_ppm:.0f}ppm]"
            
            # Filter 2: Transit depth bounds, transit-scaled baseline SNR & mean shoulder check
            if true_depth_ppm > 30000 or depth_ppm < 180:
                if rank_idx == 1:
                    first_reject_code = "REJECT_DEPTH"
                    first_reject_msg = f"REJECT {base_telemetry}: Depth out of planetary bounds (Undiluted={true_depth_ppm:.0f}ppm)"
                continue
                
            win_phase = min(0.15 * best_p, 3.0 * best_dur)
            vis = np.abs(phase) <= win_phase
            bins = np.linspace(-win_phase, win_phase, 41)
            bin_means, bin_edges, _ = stats.binned_statistic(phase[vis], flat_f[vis], statistic="mean", bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = (2.0 * win_phase) / 40.0
            n_bins_in_tr = max(1.0, best_dur / bin_width)
            
            oot_bins = np.abs(bin_centers) > (0.75 * best_dur)
            if np.sum(oot_bins) >= 6:
                oot_std_ppm = float(np.nanstd(bin_means[oot_bins]) * 1e6)
                oot_se_transit_ppm = oot_std_ppm / np.sqrt(n_bins_in_tr)
                
                left_sh = (bin_centers < -0.55 * best_dur) & (bin_centers > -1.50 * best_dur)
                right_sh = (bin_centers > 0.55 * best_dur) & (bin_centers < 1.50 * best_dur)
                l_bump = float((np.nanmean(bin_means[left_sh]) - baseline) * 1e6) if np.any(left_sh) else 0.0
                r_bump = float((np.nanmean(bin_means[right_sh]) - baseline) * 1e6) if np.any(right_sh) else 0.0
                shoulder_bump_ppm = max(l_bump, r_bump)
                
                if depth_ppm < 5.5 * oot_se_transit_ppm or shoulder_bump_ppm > 0.35 * depth_ppm:
                    if rank_idx == 1:
                        first_reject_code = "REJECT_MORPHOLOGY"
                        first_reject_msg = f"REJECT {base_telemetry}: Wavy baseline / shoulder bump (OOT_std={oot_std_ppm:.0f}ppm, Bump={shoulder_bump_ppm:.0f}ppm)"
                    continue
                    
            if true_depth_ppm > 4000:
                core_mask = np.abs(phase) < (0.18 * best_dur)
                wing_mask = (np.abs(phase) > (0.24 * best_dur)) & (np.abs(phase) < (0.44 * best_dur))
                if np.sum(core_mask) >= 3 and np.sum(wing_mask) >= 3:
                    core_depth = float((baseline - np.nanmean(flat_f[core_mask])) * 1e6)
                    wing_depth = float((baseline - np.nanmean(flat_f[wing_mask])) * 1e6)
                    if core_depth > 0 and (wing_depth / core_depth) < 0.48:
                        if rank_idx == 1:
                            first_reject_code = "REJECT_V_SHAPE"
                            first_reject_msg = f"REJECT {base_telemetry}: Sharp V-shaped grazing profile (wing/core={wing_depth/core_depth:.2f}) -> Grazing EB"
                        continue
            
            # Filter 3: Multiplicity (>= 3 transits), Odd-Even symmetry & Phase 0.5 Secondary Eclipse Test
            orbit_num = compute_orbit_number(t_curr_eval, best_t0, best_p)
            unique_transits = len(np.unique(orbit_num[in_tr]))
            if unique_transits < 3:
                if rank_idx == 1:
                    first_reject_code = "REJECT_FEW_TRANSITS"
                    first_reject_msg = f"REJECT {base_telemetry}: Only {unique_transits} transits observed (requires >= 3)"
                continue
                
            odd_mask = in_tr & (orbit_num % 2 != 0)
            even_mask = in_tr & (orbit_num % 2 == 0)
            
            if odd_even_method == "legacy" and (np.sum(odd_mask) < 3 or np.sum(even_mask) < 3):
                if rank_idx == 1:
                    first_reject_code = "REJECT_SINGLE"
                    first_reject_msg = f"REJECT {base_telemetry}: Missing odd or even coverage"
                continue
                
            if odd_even_method == "v2":
                odd_even_result = compute_odd_even_v2(
                    t_curr_eval, flat_f, phase, in_tr, out_tr, orbit_num,
                    best_p, best_dur, baseline
                )
                logger.info("ODD_EVEN_V2_EXPERIMENTAL %s", odd_even_result)
                odd_even_unresolved = odd_even_result.status != "ASSESSED"
                def finite_or_none(value):
                    return float(value) if np.isfinite(value) else None
                peak_evidence = dict(peak_rank=int(rank_idx), period_days=best_p,
                    t0_btjd=best_t0, duration_days=best_dur,
                    odd_even_status='INCONCLUSIVE' if odd_even_unresolved else ('REJECT' if odd_even_result.sigma > 2.8 else 'PASS'),
                    odd_even_reason=odd_even_result.reason,
                    odd_even_metrics={key: finite_or_none(getattr(odd_even_result, key))
                                      for key in ('sigma', 'odd_depth', 'even_depth', 'odd_err', 'even_err')},
                    odd_even_diagnostics=odd_even_result.diagnostics,
                    gates=[], disposition='EVALUATING', cause=None)
                # Existing diagnostics contain nonfinite missing measurements.
                peak_evidence = json.loads(json.dumps(peak_evidence, default=lambda v: v.item()))
                def json_finite(value):
                    if isinstance(value, dict): return {k: json_finite(v) for k, v in value.items()}
                    if isinstance(value, list): return [json_finite(v) for v in value]
                    if isinstance(value, float) and not np.isfinite(value): return None
                    return value
                peak_evidence = json_finite(peak_evidence)
                vetting_diagnostics.append(peak_evidence)
                odd_even_sigma = odd_even_result.sigma
                odd_depth, even_depth = odd_even_result.odd_depth, odd_even_result.even_depth
                odd_err, even_err = odd_even_result.odd_err, odd_even_result.even_err
            elif odd_even_method == "legacy":
                odd_depth = float((baseline - np.nanmean(flat_f[odd_mask])) * 1e6)
                even_depth = float((baseline - np.nanmean(flat_f[even_mask])) * 1e6)
                odd_err = float(np.nanstd(flat_f[odd_mask]) / np.sqrt(np.sum(odd_mask)) * 1e6)
                even_err = float(np.nanstd(flat_f[even_mask]) / np.sqrt(np.sum(even_mask)) * 1e6)

                odd_even_sigma = abs(odd_depth - even_depth) / max(np.hypot(odd_err, even_err), 1.0)
            else:
                raise ValueError("odd_even_method must be legacy or v2")
            if odd_even_method == 'v2':
                note_gate('ODD_EVEN', peak_evidence['odd_even_status'])
            if not odd_even_unresolved and odd_even_sigma > 2.8:
                if rank_idx == 1:
                    first_reject_code = "REJECT_ODD_EVEN"
                    if odd_even_method == "v2":
                        first_reject_msg = f"REJECT {base_telemetry}: Experimental odd/even reference Z={odd_even_sigma:.3f} > 2.8 (uncalibrated; possible EB / Gap Alias)"
                    else:
                        first_reject_msg = f"REJECT {base_telemetry}: Odd/Even asymmetry ({odd_even_sigma:.1f} sigma) -> EB / Gap Alias"
                continue

            # Robust individual transit consistency: require >= 60% of observed transit events to show a flux deficit
            # Rejects random white/red noise clusters in faint stars (e.g. TIC 674490)
            tr_orbits = np.unique(orbit_num[in_tr])
            neg_transits = sum(
                1 for orb in tr_orbits
                if np.sum((orbit_num == orb) & in_tr) >= 2
                and np.nanmean(flat_f[(orbit_num == orb) & in_tr]) < baseline
            )
            consistency_frac = neg_transits / max(len(tr_orbits), 1)
            note_gate('TRANSIT_CONSISTENCY', 'REJECT' if len(tr_orbits) >= 4 and consistency_frac < 0.60 else 'PASS', fraction=float(consistency_frac))
            if len(tr_orbits) >= 4 and consistency_frac < 0.60:
                if rank_idx == 1:
                    first_reject_code = "REJECT_INCONSISTENT_TRANSITS"
                    first_reject_msg = f"REJECT {base_telemetry}: Inconsistent individual transits ({neg_transits}/{len(tr_orbits)} show dip) -> Noise cluster"
                continue
                
            # Secondary Eclipse Test (full out-of-transit scan for eccentric EBs with Sidak-Bonferroni trials correction)
            is_sec_eb, sec_snr, sec_depth_ppm, sec_phi, z_crit = secondary_scan(
                t_curr_eval, flat_f, best_p, best_t0, best_dur, depth_ppm
            )
            note_gate('SECONDARY', 'REJECT' if is_sec_eb else 'PASS', snr=float(sec_snr), depth_ppm=float(sec_depth_ppm))
            if is_sec_eb:
                if rank_idx == 1:
                    first_reject_code = "REJECT_SECONDARY_ECLIPSE"
                    first_reject_msg = (
                        f"REJECT {base_telemetry}: Secondary eclipse at Phase {sec_phi:.2f} "
                        f"({sec_depth_ppm:.0f}ppm, {sec_snr:.1f}sigma > z_crit={z_crit:.2f}) -> {'Circular EB' if abs(sec_phi - 0.50) <= 0.04 else 'Eccentric EB'}"
                        if odd_even_method == "v2" else
                        f"REJECT {base_telemetry}: Secondary eclipse at Phase {sec_phi:.2f} "
                        f"({sec_depth_ppm:.0f}ppm, {sec_snr:.1f}sigma > z_crit={z_crit:.1f}) -> Eccentric EB"
                    )
                continue
                
            # Filter 4: Astrometric centroid stability
            dx_in, dx_out = dx_curr_eval[in_tr], dx_curr_eval[out_tr]
            dy_in, dy_out = dy_curr_eval[in_tr], dy_curr_eval[out_tr]
            
            _, p_val_x = stats.ttest_ind(dx_in, dx_out, equal_var=False)
            _, p_val_y = stats.ttest_ind(dy_in, dy_out, equal_var=False)
            shift_arcsec = float(np.hypot(np.mean(dx_in) - np.mean(dx_out), np.mean(dy_in) - np.mean(dy_out)) * 21.0)
            
            note_gate('CENTROID', 'REJECT' if (p_val_x < 0.001 or p_val_y < 0.001) and shift_arcsec > 2.0 else ('PASS' if np.isfinite(p_val_x) and np.isfinite(p_val_y) and np.isfinite(shift_arcsec) else 'MISSING_INPUT'), shift_arcsec=shift_arcsec)
            if (p_val_x < 0.001 or p_val_y < 0.001) and shift_arcsec > 2.0:
                if rank_idx == 1:
                    first_reject_code = "REJECT_CENTROID"
                    first_reject_msg = f"REJECT {base_telemetry}: Centroid offset ({shift_arcsec:.2f} arcsec) -> BEB"
                continue

            # Filter 5: Stellar parameters, Gaia DR3 RUWE, dilution-corrected Rp & Keplerian duration bound
            if star_info_cache is None:
                star_info_cache = get_stellar_and_gaia_params(tic_id)
            note_gate('STELLAR_INPUT', 'PASS' if star_info_cache['has_params'] else 'MISSING_INPUT')
            if not star_info_cache["has_params"]:
                return make_res(tic_id, "REJECT_NO_STELLAR_PARAMS", f"REJECT {base_telemetry}: Missing measured R_star/Teff in TIC catalog", None)
                
            r_star = star_info_cache["r_star"]
            m_star = star_info_cache["m_star"]
            teff = star_info_cache["teff"]
            ruwe = star_info_cache["ruwe"]
            
            rp_rs = np.sqrt(max(true_depth_ppm, 0.0) * 1e-6)
            r_planet_earth = float(r_star * rp_rs * 109.076)
            ext_telemetry = f"[Peak#{rank_idx} | SDE={sde:.1f} | P={best_p:.4f}d | Depth={depth_ppm:.0f}ppm | Rp={r_planet_earth:.1f}R_E | R*={r_star:.2f}R_Sun | RUWE={ruwe:.2f}]"
            
            note_gate('EVOLVED_STAR', 'REJECT' if r_star > 1.75 or teff > 9500 else 'PASS')
            if r_star > 1.75 or teff > 9500:
                return make_res(tic_id, "REJECT_EVOLVED_STAR", f"REJECT {ext_telemetry}: Evolved giant or hot subdwarf host", None)
                
            note_gate('RUWE', 'REJECT' if np.isfinite(ruwe) and ruwe > 1.4 else ('MISSING_INPUT' if not np.isfinite(ruwe) else 'PASS'))
            if np.isfinite(ruwe) and ruwe > 1.4:
                return make_res(tic_id, "REJECT_RUWE", f"REJECT {ext_telemetry}: Unresolved binary companion (RUWE > 1.4)", None)
                
            note_gate('RADIUS', 'REJECT' if r_planet_earth > 15.0 or r_planet_earth < 0.8 else 'PASS')
            if r_planet_earth > 15.0 or r_planet_earth < 0.8:
                if rank_idx == 1:
                    first_reject_code = "REJECT_RADIUS"
                    first_reject_msg = f"REJECT {ext_telemetry}: Radius out of planetary bounds ({r_planet_earth:.1f} R_E > 15 R_E -> EBLM/Star)"
                continue
                
            # Keplerian Transit Duration vs Stellar Density Check (accounting for e <= 0.30 and 1-sigma stellar errors)
            nom_max_dur_hr, min_kepler_dur_hr, max_kepler_dur_hr = compute_keplerian_duration_bounds_hours(
                best_p, r_star, star_info_cache["e_r_star"], m_star, star_info_cache["e_m_star"], rp_rs
            )
            obs_dur_hr = days_to_hours(best_dur)
            note_gate('KEPLER_DURATION_LONG', 'REJECT' if obs_dur_hr > max_kepler_dur_hr else 'PASS')
            if obs_dur_hr > max_kepler_dur_hr:
                if rank_idx == 1:
                    first_reject_code = "REJECT_KEPLER_DURATION_LONG"
                    first_reject_msg = f"REJECT {ext_telemetry}: Duration ({obs_dur_hr:.2f}h) exceeds Keplerian limit ({max_kepler_dur_hr:.2f}h, nom={nom_max_dur_hr:.2f}h) -> Blend/Subgiant"
                continue
            note_gate('KEPLER_DURATION_SHORT', 'REJECT' if obs_dur_hr < min_kepler_dur_hr else 'PASS')
            if obs_dur_hr < min_kepler_dur_hr:
                if rank_idx == 1:
                    first_reject_code = "REJECT_KEPLER_DURATION_SHORT"
                    first_reject_msg = f"REJECT {ext_telemetry}: Duration ({obs_dur_hr:.2f}h) is unphysically short (Keplerian min={min_kepler_dur_hr:.2f}h, nom={nom_max_dur_hr:.2f}h) -> Glitch/Pulsation"
                continue

            # Filter 6: Pixel-level TPF localization & strict dipole rejection
            tpf_ok, tpf_reason, diff_image, aperture = tpf_check(tic_id, last_sector, best_p, best_t0, best_dur)
            if odd_even_method == 'v2' and tpf_ok and str(tpf_reason).startswith('REVIEW_'):
                vet_holds.append(str(tpf_reason))
            note_gate('TPF', ('HOLD' if str(tpf_reason).startswith('REVIEW_') else 'PASS') if tpf_ok else ('MISSING_INPUT' if diff_image is None else 'REJECT'), reason=str(tpf_reason))
            if not tpf_ok:
                if rank_idx == 1:
                    first_reject_code = "REJECT_TPF_PIXEL"
                    first_reject_msg = f"REJECT {ext_telemetry}: {tpf_reason}"
                continue

            # Filter 7: Multi-Epoch Archival Verification across prior TESS sectors
            arch_ok, arch_status, arch_info = archival_check(
                tic_id, last_sector, t_curr_eval, flat_f, flat_ferr, best_p, best_t0, best_dur, depth_ppm, sde
            )
            arch_display_status = str(arch_status).replace(
                "Confirmed_in_Sectors_", "Archival_match_in_Sectors_"
            )

            # Persist bounded JSON-safe archival diagnostics for every evaluated target/peak
            from datetime import datetime, timezone
            target_diag = {
                "run_id": str(run_id),
                "attempt_id": str(attempt_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tic_id": str(tic_id),
                "sector": int(last_sector),
                "peak_rank": int(rank_idx),
                "arch_ok": bool(arch_ok),
                "arch_status": str(arch_status),
                "ephemeris_source": arch_info.get("ephemeris_source"),
                "ephemeris_status": arch_info.get("ephemeris_status"),
                "ephemeris_fallback_reason": arch_info.get("ephemeris_fallback_reason"),
                "archival_sectors_requested": arch_info.get("archival_sectors_requested", []),
                "archival_sectors_attempted": arch_info.get("archival_sectors_attempted", []),
                "archival_sectors_used": arch_info.get("archival_sectors_used", []),
                "archival_products_requested": arch_info.get("archival_products_requested", []),
                "archival_products_attempted": arch_info.get("archival_products_attempted", []),
                "archival_products_used": arch_info.get("archival_products_used", []),
                "archival_products_failed": arch_info.get("archival_products_failed", []),
                "archival_snr": float(arch_info["archival_snr"]) if (arch_info.get("archival_snr") is not None and np.isfinite(arch_info["archival_snr"])) else None,
                "multi_sde": float(arch_info["multi_sde"]) if (arch_info.get("multi_sde") is not None and np.isfinite(arch_info["multi_sde"])) else None,
                "est_multi_sde_proxy": float(arch_info["est_multi_sde_proxy"]) if (arch_info.get("est_multi_sde_proxy") is not None and np.isfinite(arch_info["est_multi_sde_proxy"])) else None,
            }
            if odd_even_method == "v2":
                target_diag["vr"] = evidence_safe(arch_info.get("vr"))
            record_archival_diagnostic(target_diag)
            attempt_diagnostics.append(target_diag)

            if odd_even_method == 'v2' and arch_ok and str(arch_status).startswith('REVIEW_'):
                vet_holds.append(str(arch_status))
            note_gate('ARCHIVAL', ('HOLD' if str(arch_status).startswith('REVIEW_') else 'PASS') if arch_ok else 'REJECT', reason=str(arch_status))
            if not arch_ok:
                if rank_idx == 1:
                    first_reject_code = "REJECT_ARCHIVAL"
                    first_reject_msg = f"REJECT {ext_telemetry}: {arch_status}"
                continue

            if odd_even_method == 'v2' and vet_holds:
                peak_evidence['vetting_holds'] = list(vet_holds)
                peak_evidence['disposition'] = 'REVIEW'
                peak_evidence['cause'] = 'VETTING_HOLD_DOWNSTREAM_INCONCLUSIVE' if odd_even_unresolved else 'VETTING_HOLD_PASSED_DOWNSTREAM'
                if any(g['status'] == 'MISSING_INPUT' for g in peak_evidence['gates']):
                    peak_evidence['disposition'] = 'INCOMPLETE'
                    peak_evidence['cause'] = 'VETTING_HOLD_DOWNSTREAM_NOT_RUN'
                review_peaks.append(peak_evidence)
                continue
            if odd_even_unresolved:
                peak_evidence['disposition'] = 'REVIEW'
                peak_evidence['cause'] = 'ODD_EVEN_INCONCLUSIVE'
                if any(g['status'] == 'MISSING_INPUT' for g in peak_evidence['gates']):
                    peak_evidence['disposition'] = 'INCOMPLETE'
                    peak_evidence['cause'] = 'MISSING_DOWNSTREAM_EVIDENCE'
                review_peaks.append(peak_evidence)
                continue
            # D1 insertion point: intentionally deferred until ALIAS-D1-D3
            # step 2 and its held-out PASS. No D1 rule is active here.
            # VETTING-REPLAY-02 item 17 (P1): a peak sharing all events with a stronger peak that an EB-indicating gate
            # hard-rejected goes to REVIEW and cannot become CANDIDATE (frozen review_event_link, unchanged).
            if odd_even_method == "v2" and peak_evidence is not None:
                _vr_alias = None
                for stronger in vetting_diagnostics:
                    if stronger is peak_evidence or stronger.get('peak_rank', rank_idx) >= rank_idx or stronger.get('disposition') != 'REJECT':
                        continue
                    _g = next((g for g in stronger.get('gates', []) if g['status'] == 'REJECT'), None)
                    if _g is None or not (_g['gate'] in ('SECONDARY', 'ODD_EVEN', 'TPF', 'RADIUS') or (_g['gate'] == 'ARCHIVAL' and str(_g['metrics'].get('reason', '')).startswith('Multi-sector secondary eclipse'))):
                        continue
                    _vr_alias = review_event_link(stronger, best_p, best_t0, best_dur, t_curr_eval)
                    if _vr_alias is not None:
                        peak_evidence['disposition'] = 'REVIEW'
                        peak_evidence['cause'] = 'SHARED_EVENTS_WITH_STRONGER_REJECTION'
                        peak_evidence['review_event_link'] = _vr_alias
                        peak_evidence['linked_rejected_rank'] = stronger.get('peak_rank')
                        peak_evidence['linked_rejected_gate'] = _g['gate']
                        review_peaks.append(peak_evidence)
                        break
                if _vr_alias is not None:
                    continue
            # A lower-ranked shared-event solution cannot resolve an earlier REVIEW.
            if odd_even_method == "v2":
                for stronger in review_peaks:
                    if stronger.get('cause') == 'SHARED_EVENTS_WITH_STRONGER_REVIEW':
                        continue
                    link = review_event_link(stronger, best_p, best_t0, best_dur, t_curr_eval)
                    if link is not None:
                        peak_evidence['disposition'] = 'REVIEW'
                        peak_evidence['cause'] = 'SHARED_EVENTS_WITH_STRONGER_REVIEW'
                        peak_evidence['review_event_link'] = link
                        review_peaks.append(peak_evidence)
                        break
                else:
                    link = None
                if link is not None:
                    continue
            if peak_evidence is not None:
                peak_evidence['disposition'] = 'CANDIDATE'
                peak_evidence['cause'] = 'ALL_REQUIRED_GATES_PASSED'

            # Coherent ephemeris selection: ensure (period_days, t0_btjd) come strictly from the same solution
            discovery_p = float(best_p)
            discovery_t0 = float(best_t0)
            refined_p_raw = arch_info.get("refined_p", np.nan)
            refined_t0_raw = arch_info.get("refined_t0", np.nan)
            is_strict = arch_info.get("is_projected_strict", True)
            ephem_src = arch_info.get("ephemeris_source", "DISCOVERY_SINGLE_SECTOR")
            ephem_stat = arch_info.get("ephemeris_status", "FALLBACK_DISCOVERY")
            fallback_reas = arch_info.get("ephemeris_fallback_reason", "NONE")

            has_valid_refined = (
                not is_strict
                and refined_p_raw is not None
                and refined_t0_raw is not None
                and np.isfinite(refined_p_raw)
                and np.isfinite(refined_t0_raw)
                and float(refined_p_raw) > 0
                and ephem_src == "REFINED_MULTI_SECTOR"
                and ephem_stat == "REFINED_VALID"
            )

            if has_valid_refined:
                selected_p = float(refined_p_raw)
                selected_t0 = float(refined_t0_raw)
                ephemeris_source = "REFINED_MULTI_SECTOR"
                ephemeris_status = "REFINED_VALID"
                ephemeris_fallback_reason = "NONE"
                refined_p_export = float(refined_p_raw)
                refined_t0_export = float(refined_t0_raw)
            else:
                selected_p = discovery_p
                selected_t0 = discovery_t0
                ephemeris_source = "DISCOVERY_SINGLE_SECTOR"
                ephemeris_status = "FALLBACK_DISCOVERY"
                ephemeris_fallback_reason = fallback_reas if fallback_reas != "NONE" else ("STRICT_PROJECTION_NEARBY_SECTOR" if is_strict else "REFINED_EPHEMERIS_INVALID")
                refined_p_export = np.nan
                refined_t0_export = np.nan

            arch_snr = arch_info.get("archival_snr", np.nan)
            arch_secs_str = arch_info.get("archival_sectors_str", "None")

            # Generate English Vetting Summary Plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(periods, sde_arr, color="black", lw=0.8)
            axes[0, 0].axvline(best_p, color="red", linestyle="--", label=f"Peak #{rank_idx}: P = {selected_p:.5f} d")
            axes[0, 0].axhline(6.0, color="gray", linestyle=":", label="SDE = 6.0")
            axes[0, 0].set_xlabel("Trial Period (days)")
            axes[0, 0].set_ylabel("Kovács Whitened SDE")
            toi_title_suffix = " [TOI Sibling Search]" if is_toi_host else ""
            axes[0, 0].set_title(f"BLS Whitened SDE: TIC {tic_id}{toi_title_suffix} (SDE = {sde:.1f}, N_transits = {unique_transits})")
            axes[0, 0].legend(loc="upper right")

            axes[0, 1].scatter(phase[vis], flat_f[vis], s=2, color="gray", alpha=0.35, label=f"Sector {last_sector} (2-min)")
            axes[0, 1].plot(bin_centers, bin_means, color="red", lw=2, label=f"Sector {last_sector} Binned")
            if arch_info.get("arch_bin_centers") is not None:
                axes[0, 1].plot(
                    arch_info["arch_bin_centers"],
                    arch_info["arch_bin_means"],
                    color="dodgerblue",
                    linestyle="--",
                    lw=1.8,
                    label=f"Archival ({arch_secs_str}) Binned"
                )
            axes[0, 1].set_xlim(-win_phase, win_phase)
            axes[0, 1].set_xlabel("Phase (days from mid-transit)")
            axes[0, 1].set_ylabel("Normalized Flux")
            axes[0, 1].set_title(
                f"Phase-Folded Transit: Rp = {r_planet_earth:.2f} R_Earth | Depth = {depth_ppm:.0f} ppm | {arch_display_status}"
            )
            axes[0, 1].legend(loc="lower right")

            axes[1, 0].scatter(phase[vis], dx_curr_eval[vis] * 21.0, s=2, alpha=0.35, label="dX (arcsec)")
            axes[1, 0].scatter(phase[vis], dy_curr_eval[vis] * 21.0, s=2, alpha=0.35, label="dY (arcsec)")
            axes[1, 0].axvspan(-0.5 * best_dur, 0.5 * best_dur, color="red", alpha=0.15, label="Transit Window")
            axes[1, 0].set_xlim(-win_phase, win_phase)
            axes[1, 0].set_ylim(-5, 5)
            axes[1, 0].set_xlabel("Phase (days)")
            axes[1, 0].set_ylabel("Centroid Offset (arcsec)")
            axes[1, 0].set_title(
                f"Centroid Stability: Shift = {shift_arcsec:.3f} arcsec | Odd/Even = {odd_even_sigma:.2f} {'reference Z (experimental, uncalibrated)' if odd_even_method == 'v2' else 'sigma'} | RUWE = {ruwe:.2f}"
            )
            axes[1, 0].legend(loc="upper right")

            im = axes[1, 1].imshow(diff_image, origin="lower", cmap="viridis")
            plt.colorbar(im, ax=axes[1, 1], label="Flux Deficit (e-/s)")
            for i in range(aperture.shape[0]):
                for j in range(aperture.shape[1]):
                    if aperture[i, j]:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5)
                        axes[1, 1].add_patch(rect)
            axes[1, 1].set_xlabel("Pixel Column")
            axes[1, 1].set_ylabel("Pixel Row")
            axes[1, 1].set_title(
                f"Difference Image Localization Diagnostic (not independent verification; Sector {last_sector}, CROWDSAP={crowdsap:.2f})"
            )

            plt.tight_layout()
            run_suffix = f"_{CURRENT_RUN_ID}" if CURRENT_RUN_ID else ""
            plot_filename = f"candidate_TIC_{tic_id}_s{last_sector}{run_suffix}.png"
            plot_path = os.path.join(OUTPUT_DIR, plot_filename)
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)

            cand_data = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "run_id": str(run_id),
                "tic_id": str(tic_id),
                "sector": int(last_sector),
                "is_toi_host": bool(is_toi_host),
                "known_toi_context": ",".join(f"{kp['toi']}(P={kp['period']:.3f}d)" for kp in known_transits) if is_toi_host else "None",
                "candidate_type": "TOI_SIBLING_COMPANION" if is_toi_host else "NOVEL_DISCOVERY",
                "ephemeris_source": ephemeris_source,
                "ephemeris_status": ephemeris_status,
                "ephemeris_fallback_reason": ephemeris_fallback_reason,
                "time_system": "BTJD",
                "time_standard": "TDB",
                "epoch_offset_btjd": float(BTJD_OFFSET),
                "period_units": "days",
                "duration_units": "hours",
                "depth_units": "ppm",
                "transit_measurement_source": "DISCOVERY_SINGLE_SECTOR",
                "timing_uncertainty_source": "CARTER_2008_DISCOVERY_HEURISTIC",
                "period_days": float(selected_p),
                "t0_btjd": float(selected_t0),
                "discovery_period_days": float(discovery_p),
                "discovery_t0_btjd": float(discovery_t0),
                "refined_period_days": float(refined_p_export) if np.isfinite(refined_p_export) else np.nan,
                "refined_t0_btjd": float(refined_t0_export) if np.isfinite(refined_t0_export) else np.nan,
                "duration_hours": float(obs_dur_hr),
                "min_kepler_dur_hours": float(min_kepler_dur_hr),
                "max_kepler_dur_hours": float(max_kepler_dur_hr),
                "n_transits": int(unique_transits),
                "depth_ppm": float(depth_ppm),
                "crowdsap": float(crowdsap),
                "sde": float(sde),
                "r_planet_earth": float(r_planet_earth),
                "r_star_sun": float(r_star),
                "m_star_sun": float(m_star),
                "teff_k": float(teff),
                "gaia_ruwe": float(ruwe) if np.isfinite(ruwe) else np.nan,
                "odd_even_sigma": float(odd_even_sigma),
                "centroid_shift_arcsec": float(shift_arcsec),
                "archival_sectors": arch_secs_str,
                "n_archival_sectors": int(arch_info.get("n_archival_sectors", 0)),
                "archival_sectors_requested": _serialize_metadata_list(arch_info.get("archival_sectors_requested", [])),
                "archival_sectors_attempted": _serialize_metadata_list(arch_info.get("archival_sectors_attempted", [])),
                "archival_sectors_used": _serialize_metadata_list(arch_info.get("archival_sectors_used", [])),
                "archival_products_requested": _serialize_metadata_list(arch_info.get("archival_products_requested", [])),
                "archival_products_attempted": _serialize_metadata_list(arch_info.get("archival_products_attempted", [])),
                "archival_products_used": _serialize_metadata_list(arch_info.get("archival_products_used", [])),
                "archival_products_failed": _serialize_metadata_list(arch_info.get("archival_products_failed", [])),
                "archival_snr_sigma": float(arch_snr) if np.isfinite(arch_snr) else np.nan,
                "archival_trials": int(arch_info.get("n_trials", 1)),
                "archival_snr_crit": float(arch_info.get("snr_crit", 2.4)),
                "est_multi_sde_proxy": float(arch_info.get("est_multi_sde_proxy", sde)),
                "multi_sde": float(arch_info.get("multi_sde", sde)),
                "cramer_rao_sigma_p_sec": float(arch_info.get("sigma_p_sec", np.nan)) if np.isfinite(arch_info.get("sigma_p_sec", np.nan)) else np.nan,
                "cramer_rao_sigma_t0_sec": float(arch_info.get("sigma_t0_sec", np.nan)) if np.isfinite(arch_info.get("sigma_t0_sec", np.nan)) else np.nan,
                "covariance_p_t0": np.nan,
                "plot_path": plot_filename,
                "exofop_url": f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"
            }
            
            if odd_even_method == "v2":
                cand_data["odd_even_method"] = "ODD_EVEN_V2_EXPERIMENTAL"
                cand_data["odd_even_status"] = odd_even_result.status
                details = {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                           for k, v in odd_even_result.diagnostics.items()}
                cand_data["odd_even_diagnostics_json"] = json.dumps(details, allow_nan=False)

            cand_tag = f"TOI SIBLING (Host of {known_transits[0]['toi']})" if is_toi_host else "CANDIDATE"
            trials_str = f"Trials={arch_info.get('n_trials', 1)}, Crit={arch_info.get('snr_crit', 2.4):.1f}s"
            msg = (
                f"{cand_tag} FOUND (Peak #{rank_idx})! P={selected_p:.5f} d | N={unique_transits} | Depth={depth_ppm:.0f} ppm | "
                f"Rp={r_planet_earth:.2f} R_E | SDE={sde:.1f} | Archival={arch_display_status} ({trials_str})"
            )
            return make_res(tic_id, "CANDIDATE", msg, cand_data)
            
        if AUTO_PURGE_CACHE:
            purge_target_cache(tic_id)
        if review_peaks:
            incomplete = all(p['disposition'] == 'INCOMPLETE' for p in review_peaks)
            code = 'INCOMPLETE_ODD_EVEN_DOWNSTREAM' if incomplete else 'REVIEW_ODD_EVEN_INCONCLUSIVE'
            if any(p.get('cause') == 'SHARED_EVENTS_WITH_STRONGER_REJECTION' for p in review_peaks):
                code = 'REVIEW_ALIAS_OF_REJECTED_PEAK'
            if any(p.get('vetting_holds') for p in review_peaks):
                code = 'INCOMPLETE_VETTING_DOWNSTREAM' if incomplete else 'REVIEW_VETTING_HOLD'
            return make_res(tic_id, code, f'{code}: see experimental_v2_vetting.jsonl; not a CANDIDATE', None)
        return make_res(tic_id, first_reject_code, first_reject_msg, None)
        
    except Exception as e:
        if AUTO_PURGE_CACHE:
            purge_target_cache(tic_id)
        return make_res(tic_id, "ERROR", f"ERROR: {e}", None)


def save_candidates_to_report(report_path: str, candidate_records: list):
    """
    Saves candidate records to report_path using atomic replacement to prevent file corruption.
    """
    if any(r.get('disposition') in ('REVIEW', 'INCOMPLETE', 'REJECT') for r in candidate_records):
        raise ValueError('Non-candidate disposition cannot enter candidate report')
    columns = list(CANDIDATE_COLUMNS)
    if any(r.get("odd_even_method") == "ODD_EVEN_V2_EXPERIMENTAL" for r in candidate_records):
        columns += ["odd_even_method", "odd_even_status", "odd_even_diagnostics_json"]
    df_out = pd.DataFrame(candidate_records, columns=columns)
    dir_name = os.path.dirname(report_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    temp_path = f"{report_path}.tmp.{os.getpid()}"
    df_out.to_csv(temp_path, index=False)
    os.replace(temp_path, report_path)


if __name__ == "__main__":
    import signal
    _MAIN_PID = os.getpid()

    def _handle_sigterm(signum, frame):
        # PBS sends SIGTERM on walltime / qdel: reuse the Ctrl+C path so the Run Summary is still written.
        if os.getpid() == _MAIN_PID:
            raise KeyboardInterrupt
        os._exit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    start_banner = f"=== ALDEBARAN PIPELINE {PIPELINE_VERSION} STARTED (Workers={MAX_WORKERS}, Batch={MAX_TARGETS}) ==="
    write_log("=" * 70, print_to_console=False)
    write_log(start_banner)
    write_event(start_banner)

    # Initialize observer run ledger if configured before report/catalog/queue preflight
    obs_dir = os.environ.get("ALDEBARAN_OBSERVER_DIR")
    run_ledger = None
    if obs_dir and obs_dir.strip():
        try:
            from decision_ledger import RunLedger
            run_ledger = RunLedger(observer_dir=obs_dir.strip(), legacy_run_id=CURRENT_RUN_ID)
        except Exception as exc:
            write_log(f"Warning: Failed to initialize observer RunLedger ({exc}); continuing with observer disabled.")

    # Validate existing candidate report early before remote sector/target selection
    candidates = []
    try:
        if os.path.exists(REPORT_FILE):
            if os.path.getsize(REPORT_FILE) == 0:
                candidates = []
            else:
                try:
                    df_existing = pd.read_csv(REPORT_FILE)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to read existing candidate CSV '{REPORT_FILE}' ({type(exc).__name__}: {exc}). "
                        f"Refusing to proceed to prevent corrupting historical artifacts."
                    ) from exc

                if list(df_existing.columns) != CANDIDATE_COLUMNS:
                    raise RuntimeError(
                        f"Incompatible candidate CSV schema in '{REPORT_FILE}'. "
                        f"Existing header has {len(df_existing.columns)} columns, expected {len(CANDIDATE_COLUMNS)} columns "
                        f"for schema {CANDIDATE_SCHEMA_VERSION}. Refusing to proceed to prevent corrupting historical artifacts."
                    )
                candidates = df_existing.to_dict("records")

        active_sector, targets = select_latest_sector_and_targets(sector_list=PREFERRED_SECTORS, limit=MAX_TARGETS)
    except Exception as prequeue_exc:
        if run_ledger is not None:
            try:
                run_ledger.record_prequeue_failure(str(prequeue_exc))
                run_ledger.finalize_run(completed_count=0, error_count=1, candidate_count=0, status="ERROR")
            except Exception:
                pass
        raise

    write_log(f"Launching parallel pipeline on {MAX_WORKERS} CPU workers for Sector {active_sector}...\n")

    completed = 0
    total = len(targets)
    stats_counter = {}

    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    try:
        if run_ledger is not None:
            future_to_meta = {}
            for tic in targets:
                ctx = None
                sub_id = None
                try:
                    ctx = run_ledger.create_submission_context(tic, active_sector)
                    sub_id = ctx.submission_id
                except Exception:
                    ctx = None
                    sub_id = None
                fut = executor.submit(analyze_single_target, tic, active_sector, observer_context=ctx)
                future_to_meta[fut] = (tic, sub_id)
        else:
            future_to_tic = {
                executor.submit(analyze_single_target, tic, active_sector): tic
                for tic in targets
            }

        iter_futures = future_to_meta if run_ledger is not None else future_to_tic
        for future in as_completed(iter_futures):
            completed += 1
            if run_ledger is not None:
                tic_id, sub_id = future_to_meta[future]
                try:
                    res_tuple = future.result()
                except Exception as worker_exc:
                    if sub_id is not None:
                        try:
                            run_ledger.record_submission_outcome(sub_id, exc=worker_exc)
                        except Exception:
                            pass
                    raise worker_exc

                if sub_id is not None:
                    try:
                        run_ledger.record_submission_outcome(sub_id, future_result=res_tuple)
                    except Exception:
                        pass
                tic_id, code, status_msg, cand_dict = res_tuple
            else:
                tic_id = future_to_tic[future]
                tic_id, code, status_msg, cand_dict = future.result()
            stats_counter[code] = stats_counter.get(code, 0) + 1

            if code != "ERROR":
                append_scanned_history(tic_id)
            else:
                write_event(f"ERROR [{completed}/{total}] [TIC {tic_id}]: {status_msg}")

            if cand_dict is not None:
                write_log(f"\n>>> [{completed}/{total}] [TIC {tic_id}] {status_msg} <<<\n")
                write_event(f"CANDIDATE FOUND [{completed}/{total}] [TIC {tic_id}]: {status_msg}")
                candidates = [c for c in candidates if str(c.get("tic_id")) != str(tic_id)]
                candidates.append(cand_dict)
                save_candidates_to_report(REPORT_FILE, candidates)
            else:
                write_log(f"[{completed}/{total}] [TIC {tic_id}] {status_msg}")
                if code == "REJECT_ARCHIVAL":
                    res_obj = future.result()
                    arch_diag = getattr(res_obj, "archival_diagnostic", None)
                    if arch_diag is None:
                        res_att = getattr(res_obj, "attempt_id", None)
                        if res_att:
                            arch_diag = get_latest_archival_diagnostic(
                                tic_id, active_sector, output_dir=OUTPUT_DIR, run_id=CURRENT_RUN_ID, attempt_id=res_att
                            )
                    if arch_diag and arch_diag.get("archival_products_failed"):
                        write_log(f"    Archival diagnostics: attempted={arch_diag.get('archival_products_attempted')}, failed={arch_diag.get('archival_products_failed')}")

            if completed % 500 == 0:
                cand_cnt = len(candidates)
                err_cnt = stats_counter.get("ERROR", 0)
                pct = (completed / total) * 100
                write_event(f"Progress Checkpoint: [{completed}/{total}] ({pct:.1f}%) | Candidates: {cand_cnt} | Errors: {err_cnt}")
                
    except KeyboardInterrupt:
        write_log("\n[!] Run interrupted by user (Ctrl+C). Shutting down worker pool...")
        write_event("Execution interrupted by user (Ctrl+C).")
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=True)

    if run_ledger is not None:
        try:
            cand_cnt = len(candidates)
            err_cnt = stats_counter.get("ERROR", 0)
            run_ledger.finalize_run(completed_count=completed, error_count=err_cnt, candidate_count=cand_cnt)
        except Exception:
            pass
        
    summary_str = " | ".join(f"{k}: {v}" for k, v in sorted(stats_counter.items()))
    final_summary = f"Run Summary (Sector {active_sector}, {completed}/{total} processed): {summary_str}"
    write_log(f"\n{final_summary}")
    write_log(f"Total candidates saved: {len(candidates)}. Output folder: '{OUTPUT_DIR}'.")
    write_event(final_summary)
    write_event(f"Completed Sector {active_sector}. Total candidates: {len(candidates)}.")