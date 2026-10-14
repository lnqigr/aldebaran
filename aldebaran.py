import os
import io
import ssl
import shutil
import glob
import warnings
import urllib.request
import urllib.parse
from datetime import datetime

# Global SSL verification bypass for Caltech ExoFOP incomplete certificate chain on Windows
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress non-critical library import warnings across multiprocessing workers
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=UserWarning)

# Lock internal BLAS/OpenMP threads to 1 per process to prevent CPU oversubscription on i9-14900HX
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

PIPELINE_VERSION = "v7.7"
AUTO_PURGE_CACHE = True       # Automatically deletes raw FITS files of non-candidates to keep cache at ~0 MB

# Lightkurve does NOT read a LIGHTKURVE_CACHE_DIR env var by itself; apply it explicitly so HPC jobs
# keep all FITS downloads on node-local scratch (re-executed in every spawned/forked worker on import).
if os.environ.get("LIGHTKURVE_CACHE_DIR"):
    os.makedirs(os.environ["LIGHTKURVE_CACHE_DIR"], exist_ok=True)
    lk.conf.cache_dir = os.environ["LIGHTKURVE_CACHE_DIR"]

# Finishes Sector 98 first, then automatically advances to newest released sectors (106 down to 80)
PREFERRED_SECTORS = [
    98, 106, 105, 104, 103, 102, 101, 100, 99,
    97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80
]
# Both values can be overridden from the PBS script (ALDEBARAN_MAX_TARGETS / ALDEBARAN_WORKERS)
MAX_TARGETS = int(os.environ.get("ALDEBARAN_MAX_TARGETS", 1000))   # Number of stars to scan per run
MAX_WORKERS = int(os.environ.get("ALDEBARAN_WORKERS", 16))         # 16 is optimal for i9-14900HX without MAST rate-limits
MAX_PEAKS_PER_STAR = 3    # Inspect top 3 distinct BLS peaks per star before rejecting
PRIORITY_TICS = []        # Add specific TIC IDs here to force-test them first

OUTPUT_DIR = "tess_results"
HISTORY_FILE = os.path.join(OUTPUT_DIR, "scanned_tics.txt")
CATALOG_CACHE_FILE = os.path.join(OUTPUT_DIR, "known_catalog_cache.txt")
REPORT_FILE = os.path.join(OUTPUT_DIR, "candidates_report.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline_run.log")
EVENTS_LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline_events.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_log(message: str, print_to_console: bool = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if print_to_console:
        print(message, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def write_event(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EVENTS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def fetch_csv_with_headers(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=45)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text), comment="#", on_bad_lines="skip")
    except Exception:
        req = urllib.request.Request(url, headers=headers)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=ctx, timeout=45) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(raw_text), comment="#", on_bad_lines="skip")


def get_known_exoplanet_tics() -> set:
    known_tics = set()
    if os.path.exists(CATALOG_CACHE_FILE):
        try:
            with open(CATALOG_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = set(line.strip() for line in f if line.strip())
            known_tics.update(cached)
        except Exception:
            pass

    write_log("[1/5] Fetching official TOI catalog from NASA Exoplanet Archive...")
    try:
        toi_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+tid+from+toi&format=csv"
        df_toi = fetch_csv_with_headers(toi_url)
        tois = set(df_toi["tid"].dropna().astype(int).astype(str))
        known_tics.update(tois)
        write_log(f"      -> Loaded {len(tois)} official TOI targets.")
    except Exception as e:
        write_log(f"      [!] Warning: Could not load TOI table (using cache): {e}")

    write_log("[2/5] Fetching confirmed exoplanets (pscomppars) from NASA Exoplanet Archive...")
    try:
        ps_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+distinct+tic_id+from+pscomppars+where+tic_id+is+not+null&format=csv"
        df_ps = fetch_csv_with_headers(ps_url)
        ps_tics = set(
            str(val).replace("TIC", "").strip().lstrip("0")
            for val in df_ps["tic_id"].dropna()
            if str(val).replace("TIC", "").strip().isdigit()
        )
        known_tics.update(ps_tics)
        write_log(f"      -> Loaded {len(ps_tics)} confirmed planet host stars.")
    except Exception as e:
        write_log(f"      [!] Warning: Could not load confirmed planets table (using cache): {e}")

    write_log("[3/5] Fetching Community TOI (CTOI) catalog from ExoFOP-TESS...")
    try:
        ctoi_url = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
        df_ctoi = fetch_csv_with_headers(ctoi_url)
        tic_col = next((c for c in df_ctoi.columns if "tic" in c.lower()), None)
        if tic_col:
            ctois = set(df_ctoi[tic_col].dropna().astype(int).astype(str))
            known_tics.update(ctois)
            write_log(f"      -> Loaded {len(ctois)} Community TOI (CTOI) targets.")
    except Exception as e:
        write_log(f"      [!] Warning: Could not load ExoFOP CTOI table (using cache): {e}")

    write_log("[4/5] Fetching TESS Eclipsing Binary Catalogue (Prša et al. 2022) from VizieR TAP...")
    try:
        adql_eb = 'SELECT distinct TIC FROM "J/ApJS/258/16/tess-ebs"'
        eb_url = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=" + urllib.parse.quote(adql_eb)
        df_eb = fetch_csv_with_headers(eb_url)
        eb_tics = set(df_eb["TIC"].dropna().astype(int).astype(str))
        known_tics.update(eb_tics)
        write_log(f"      -> Loaded {len(eb_tics)} cataloged TESS Eclipsing Binary (EB) stars.")
    except Exception as e:
        write_log(f"      [!] Warning: Could not load VizieR TESS EB table (using cache): {e}")

    if known_tics:
        try:
            with open(CATALOG_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(known_tics)) + "\n")
        except Exception:
            pass

    return known_tics


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
    known_catalog_tics = get_known_exoplanet_tics()
    scanned_tics = load_scanned_history() - set(PRIORITY_TICS)

    write_log("[4/5] Probing MAST archive for the newest available TESS SPOC sector...")
    chosen_sector = None
    available_tics = []
    raw_count = 0

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
            rem_tics = list(raw_tics - known_catalog_tics - scanned_tics)
            if len(rem_tics) == 0:
                write_log(f"      -> Sector {sec}: All {len(raw_tics)} targets already scanned, moving to next...")
                continue

            chosen_sector = sec
            available_tics = rem_tics
            raw_count = len(raw_tics)
            break
        except Exception as e:
            write_log(f"      -> Sector {sec} query error ({e}), trying next...")
            continue

    if chosen_sector is None:
        raise RuntimeError("No available unsearched SPOC targets found in any of the PREFERRED_SECTORS.")

    final_queue = []
    for ptic in PRIORITY_TICS:
        if ptic in known_catalog_tics:
            write_log(f"      [i] Note: Priority target TIC {ptic} is already listed in TOI/CTOI/Confirmed catalogs.")
        final_queue.append(ptic)

    for tic in available_tics:
        if tic not in final_queue:
            final_queue.append(tic)

    lock_msg = (
        f"Sector {chosen_sector} locked | Total SPOC: {raw_count} | "
        f"Unsearched: {len(available_tics)} | Queued: {min(limit, len(final_queue))}"
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
        return p_days * 24.0 * 0.25
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
        
        adql = (
            f"SELECT TOP 1 ruwe FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.000833))"
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
    """
    phase_0to1 = ((t - t0) % period) / period
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
    
    max_snr = 0.0
    max_depth_ppm = 0.0
    best_phi = 0.5
    
    for phi_c, l_i, r_i in zip(trial_phases, left_idx, right_idx):
        n_pts = r_i - l_i
        if n_pts < 5:
            continue
        dip_ppm = float((oot_baseline - np.nanmean(f_sorted[l_i:r_i])) * 1e6)
        err_ppm = (oot_noise * 1e6) / np.sqrt(n_pts)
        snr = dip_ppm / max(err_ppm, 1.0)
        if snr > max_snr:
            max_snr = snr
            max_depth_ppm = dip_ppm
            best_phi = float(phi_c)
            
    # Flag as eccentric EB if SNR exceeds trials-corrected threshold AND depth >= 12% of primary
    is_eb = (max_snr > z_crit) and (max_depth_ppm > 220.0) and (max_depth_ppm >= 0.12 * primary_depth_ppm)
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit


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
            
        phase = ((t_tpf - t0_val + 0.5 * p_val) % p_val) - 0.5 * p_val
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


def verify_multi_epoch(
    tic_id: str,
    current_sector: int,
    t_curr: np.ndarray,
    flat_f_curr: np.ndarray,
    flat_ferr_curr: np.ndarray,
    best_p: float,
    best_dur: float,
    depth_ppm_curr: float,
    sde_curr: float
) -> tuple:
    try:
        all_search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
        prior_indices = []
        prior_sec_nums = []
        
        for i, row in enumerate(all_search):
            sec_num = int(str(row.mission[0]).split()[-1])
            if sec_num != current_sector:
                prior_indices.append(i)
                prior_sec_nums.append(sec_num)
                
        if len(prior_indices) == 0:
            if sde_curr < 6.5:
                return False, f"Single-sector only and borderline SDE ({sde_curr:.1f} < 6.5)", None
            return True, "Single-Sector_Only", {
                "n_archival_sectors": 0,
                "archival_sectors_str": "None",
                "refined_p": best_p,
                "archival_snr": np.nan,
                "multi_sde": sde_curr,
                "arch_bin_centers": None,
                "arch_bin_means": None
            }
            
        chosen_idx = prior_indices[-3:]
        chosen_secs = prior_sec_nums[-3:]
        
        t_raw_list, f_raw_list, ferr_raw_list, f_init_list = [], [], [], []
        for idx in chosen_idx:
            try:
                lc_p = safe_download(all_search[idx], tic_id, download_all=False)
                tp, fp, ferrp, _, _, _ = extract_and_stitch_arrays([lc_p], edge_trim_days=0.10)
                tp_c, fp_c, ferrp_c = clip_upper_outliers(tp, fp, ferrp)
                tr_init = piecewise_savgol_detrend(tp_c, fp_c, mask=None, max_win=601)
                t_raw_list.append(tp_c)
                f_raw_list.append(fp_c)
                ferr_raw_list.append(ferrp_c / tr_init)
                f_init_list.append(fp_c / tr_init)
            except Exception:
                continue
                
        if not t_raw_list:
            if sde_curr < 6.5:
                return False, "Archival sectors unreadable and SDE < 6.5", None
            return True, "Single-Sector_Only", {
                "n_archival_sectors": 0,
                "archival_sectors_str": "None",
                "refined_p": best_p,
                "archival_snr": np.nan,
                "multi_sde": sde_curr,
                "arch_bin_centers": None,
                "arch_bin_means": None
            }
            
        t_prior = np.concatenate(t_raw_list)
        f_prior_raw = np.concatenate(f_raw_list)
        f_prior_init = np.concatenate(f_init_list)
        ferr_prior = np.concatenate(ferr_raw_list)
        
        t_comb = np.concatenate([t_prior, t_curr])
        f_comb = np.concatenate([f_prior_init, flat_f_curr])
        ferr_comb = np.concatenate([ferr_prior, flat_ferr_curr])
        sort_c = np.argsort(t_comb)
        t_comb, f_comb, ferr_comb = t_comb[sort_c], f_comb[sort_c], ferr_comb[sort_c]
        
        baseline_days = float(np.max(t_comb) - np.min(t_comb))
        n_orbits = max(10.0, baseline_days / best_p)
        n_grid_pts = int(np.clip(n_orbits * 45.0, 5000, 16000))
        
        fine_periods = np.linspace(best_p * 0.994, best_p * 1.006, n_grid_pts)
        fine_durs = np.array([max(0.03, best_dur * 0.85), best_dur, min(0.22, best_dur * 1.15)])
        
        bls_comb = BoxLeastSquares(t_comb, f_comb, dy=ferr_comb)
        res_comb = bls_comb.power(fine_periods, fine_durs)
        best_c_idx = int(np.nanargmax(res_comb.power))
        refined_p = float(res_comb.period[best_c_idx])
        refined_t0 = float(res_comb.transit_time[best_c_idx])
        
        phase_prior = ((t_prior - refined_t0 + 0.5 * refined_p) % refined_p) - 0.5 * refined_p
        tr_mask_prior = np.abs(phase_prior) < (0.6 * best_dur)
        trend_prior_ref = piecewise_savgol_detrend(t_prior, f_prior_raw, mask=tr_mask_prior, max_win=601)
        f_prior = f_prior_raw / trend_prior_ref
        
        in_tr_p = np.abs(phase_prior) < (0.45 * best_dur)
        out_tr_p = np.abs(phase_prior) > (0.65 * best_dur)
        
        if np.sum(in_tr_p) < 5:
            return False, f"Insufficient archival in-transit points in sectors {chosen_secs}", None
            
        base_p = float(np.nanmedian(f_prior[out_tr_p]))
        depth_prior_ppm = float((base_p - np.nanmean(f_prior[in_tr_p])) * 1e6)
        err_prior_ppm = float(np.nanstd(f_prior[out_tr_p]) / np.sqrt(np.sum(in_tr_p)) * 1e6)
        archival_snr = depth_prior_ppm / max(err_prior_ppm, 1.0)
        
        if archival_snr < 2.4 or depth_prior_ppm < 0.25 * depth_ppm_curr:
            return (
                False,
                f"Archival non-detection in sectors {chosen_secs} (Prior Depth={depth_prior_ppm:.0f}ppm, SNR={archival_snr:.1f}sigma) -> Transient artifact",
                None
            )
            
        win_phase = min(0.15 * best_p, 3.0 * best_dur)
        vis_p = np.abs(phase_prior) <= win_phase
        bins = np.linspace(-win_phase, win_phase, 41)
        arch_means, arch_edges, _ = stats.binned_statistic(phase_prior[vis_p], f_prior[vis_p], statistic="mean", bins=bins)
        arch_centers = 0.5 * (arch_edges[:-1] + arch_edges[1:])
        
        secs_str = ",".join(str(s) for s in chosen_secs)
        return True, f"Confirmed_in_Sectors_{secs_str}", {
            "n_archival_sectors": len(chosen_secs),
            "archival_sectors_str": secs_str,
            "refined_p": refined_p,
            "archival_snr": archival_snr,
            "multi_sde": sde_curr * np.sqrt(1.0 + len(chosen_secs) * 0.5),
            "arch_bin_centers": arch_centers,
            "arch_bin_means": arch_means
        }
    except Exception:
        if sde_curr < 6.5:
            return False, "Archival lookup failed and single-sector SDE < 6.5", None
        return True, "Single-Sector_Only", {
            "n_archival_sectors": 0,
            "archival_sectors_str": "None",
            "refined_p": best_p,
            "archival_snr": np.nan,
            "multi_sde": sde_curr,
            "arch_bin_centers": None,
            "arch_bin_means": None
        }


def analyze_single_target(tic_id: str, sector: int, min_period: float = 0.8, max_period: float = 13.0) -> tuple:
    """
    Worker entrypoint. Guarantees the per-target FITS cache is purged on EVERY non-candidate
    exit path (early REJECT_CROWDSAP / REJECT_RUWE / ... returns previously skipped the purge,
    leaving ~65% of downloads on disk and overflowing scratch quotas on HPC nodes).
    """
    result = None
    try:
        result = _analyze_single_target_impl(tic_id, sector, min_period, max_period)
        return result
    finally:
        if AUTO_PURGE_CACHE and (result is None or result[1] != "CANDIDATE"):
            purge_target_cache(tic_id)


def _analyze_single_target_impl(tic_id: str, sector: int, min_period: float = 0.8, max_period: float = 13.0) -> tuple:
    try:
        search_res = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector, exptime=120)
        if len(search_res) == 0:
            return tic_id, "SKIP_NO_DATA", "SKIP: No SPOC data in target sector", None
        last_sector = sector
        lc_collection = [safe_download(search_res[0], tic_id, download_all=False)]
            
        t, f, ferr, dx, dy, crowdsap = extract_and_stitch_arrays(lc_collection, edge_trim_days=0.10)
        
        # Filter 0: SPOC Aperture Crowding Metric (CROWDSAP)
        if crowdsap < 0.70:
            return tic_id, "REJECT_CROWDSAP", f"REJECT: Severe aperture crowding (CROWDSAP={crowdsap:.2f} < 0.70) -> Blended field", None
            
        med_f = np.median(f)
        mad_std = 1.4826 * np.median(np.abs(f - med_f))
        keep = f < (med_f + 3.0 * mad_std)
        t, f, ferr, dx, dy = t[keep], f[keep], ferr[keep], dx[keep], dy[keep]
        
        trend = piecewise_savgol_detrend(t, f, mask=None, max_win=601)
        flat_f_init = f / trend
        flat_ferr_init = ferr / trend
        
        max_p = min(max_period, (t[-1] - t[0]) * 0.48)
        if max_p <= min_period:
            return tic_id, "REJECT_BASELINE", "REJECT: Time baseline too short", None
            
        freqs = np.linspace(1.0 / max_p, 1.0 / min_period, 9000)
        periods = (1.0 / freqs)[::-1]
        # Start durations at 0.045d (1.08h) to prevent BLS grid boundary collapse
        durations = np.linspace(0.045, 0.18, 12)
        
        bls_model = BoxLeastSquares(t, flat_f_init, dy=flat_ferr_init)
        bls_res = bls_model.power(periods, durations)
        
        # Kovács et al. (2002) whitened linear Signal Residue SDE
        sde_arr, _ = compute_whitened_kovacs_sde(bls_res.power, window_pts=501)
        top_peak_indices = get_top_distinct_bls_peaks(periods, sde_arr, max_peaks=MAX_PEAKS_PER_STAR)
        
        star_info_cache = None
        first_reject_code = "EMPTY_LOW_SDE"
        first_reject_msg = "Empty"
        
        for rank_idx, peak_idx in enumerate(top_peak_indices, 1):
            best_p = float(bls_res.period[peak_idx])
            best_t0 = float(bls_res.transit_time[peak_idx])
            best_dur = float(bls_res.duration[peak_idx])
            sde = float(sde_arr[peak_idx])
            
            # Filter 1: Signal Detection Efficiency (SDE >= 6.0) & Duty Cycle
            if sde < 6.0:
                if rank_idx == 1:
                    first_reject_code = "EMPTY_LOW_SDE"
                    first_reject_msg = f"Empty (SDE={sde:.1f})"
                break
                
            if (best_dur / best_p) > 0.095:
                if rank_idx == 1:
                    first_reject_code = "REJECT_DUTY_CYCLE"
                    first_reject_msg = f"REJECT [SDE={sde:.1f}, P={best_p:.3f}d]: Duty cycle too wide ({best_dur/best_p:.2f}) -> Sinusoidal variable"
                continue
                
            phase = ((t - best_t0 + 0.5 * best_p) % best_p) - 0.5 * best_p
            transit_mask = np.abs(phase) < (0.6 * best_dur)
            
            if np.sum(transit_mask) < 6 or np.sum(~transit_mask) < 100:
                if rank_idx == 1:
                    first_reject_code = "REJECT_POINTS"
                    first_reject_msg = f"REJECT [SDE={sde:.1f}, P={best_p:.3f}d]: Insufficient in-transit points"
                continue
                
            trend_refined = piecewise_savgol_detrend(t, f, mask=transit_mask, max_win=601)
            flat_f = f / trend_refined
            flat_ferr = ferr / trend_refined
            
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
            orbit_num = np.round((t - best_t0) / best_p).astype(int)
            unique_transits = len(np.unique(orbit_num[in_tr]))
            if unique_transits < 3:
                if rank_idx == 1:
                    first_reject_code = "REJECT_FEW_TRANSITS"
                    first_reject_msg = f"REJECT {base_telemetry}: Only {unique_transits} transits observed (requires >= 3)"
                continue
                
            odd_mask = in_tr & (orbit_num % 2 != 0)
            even_mask = in_tr & (orbit_num % 2 == 0)
            
            if np.sum(odd_mask) < 3 or np.sum(even_mask) < 3:
                if rank_idx == 1:
                    first_reject_code = "REJECT_SINGLE"
                    first_reject_msg = f"REJECT {base_telemetry}: Missing odd or even coverage"
                continue
                
            odd_depth = float((baseline - np.nanmean(flat_f[odd_mask])) * 1e6)
            even_depth = float((baseline - np.nanmean(flat_f[even_mask])) * 1e6)
            odd_err = float(np.nanstd(flat_f[odd_mask]) / np.sqrt(np.sum(odd_mask)) * 1e6)
            even_err = float(np.nanstd(flat_f[even_mask]) / np.sqrt(np.sum(even_mask)) * 1e6)
            
            odd_even_sigma = abs(odd_depth - even_depth) / max(np.hypot(odd_err, even_err), 1.0)
            if odd_even_sigma > 2.8:
                if rank_idx == 1:
                    first_reject_code = "REJECT_ODD_EVEN"
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
            if len(tr_orbits) >= 4 and consistency_frac < 0.60:
                if rank_idx == 1:
                    first_reject_code = "REJECT_INCONSISTENT_TRANSITS"
                    first_reject_msg = f"REJECT {base_telemetry}: Inconsistent individual transits ({neg_transits}/{len(tr_orbits)} show dip) -> Noise cluster"
                continue
                
            # Secondary Eclipse Test (full out-of-transit scan for eccentric EBs with Sidak-Bonferroni trials correction)
            is_sec_eb, sec_snr, sec_depth_ppm, sec_phi, z_crit = scan_full_orbit_secondary_eclipse(
                t, flat_f, best_p, best_t0, best_dur, depth_ppm
            )
            if is_sec_eb:
                if rank_idx == 1:
                    first_reject_code = "REJECT_SECONDARY_ECLIPSE"
                    first_reject_msg = (
                        f"REJECT {base_telemetry}: Secondary eclipse at Phase {sec_phi:.2f} "
                        f"({sec_depth_ppm:.0f}ppm, {sec_snr:.1f}sigma > z_crit={z_crit:.1f}) -> Eccentric EB"
                    )
                continue
                
            # Filter 4: Astrometric centroid stability
            dx_in, dx_out = dx[in_tr], dx[out_tr]
            dy_in, dy_out = dy[in_tr], dy[out_tr]
            
            _, p_val_x = stats.ttest_ind(dx_in, dx_out, equal_var=False)
            _, p_val_y = stats.ttest_ind(dy_in, dy_out, equal_var=False)
            shift_arcsec = float(np.hypot(np.mean(dx_in) - np.mean(dx_out), np.mean(dy_in) - np.mean(dy_out)) * 21.0)
            
            if (p_val_x < 0.001 or p_val_y < 0.001) and shift_arcsec > 2.0:
                if rank_idx == 1:
                    first_reject_code = "REJECT_CENTROID"
                    first_reject_msg = f"REJECT {base_telemetry}: Centroid offset ({shift_arcsec:.2f} arcsec) -> BEB"
                continue

            # Filter 5: Stellar parameters, Gaia DR3 RUWE, dilution-corrected Rp & Keplerian duration bound
            if star_info_cache is None:
                star_info_cache = get_stellar_and_gaia_params(tic_id)
            if not star_info_cache["has_params"]:
                return tic_id, "REJECT_NO_STELLAR_PARAMS", f"REJECT {base_telemetry}: Missing measured R_star/Teff in TIC catalog", None
                
            r_star = star_info_cache["r_star"]
            m_star = star_info_cache["m_star"]
            teff = star_info_cache["teff"]
            ruwe = star_info_cache["ruwe"]
            
            rp_rs = np.sqrt(max(true_depth_ppm, 0.0) * 1e-6)
            r_planet_earth = float(r_star * rp_rs * 109.076)
            ext_telemetry = f"[Peak#{rank_idx} | SDE={sde:.1f} | P={best_p:.4f}d | Depth={depth_ppm:.0f}ppm | Rp={r_planet_earth:.1f}R_E | R*={r_star:.2f}R_Sun | RUWE={ruwe:.2f}]"
            
            if r_star > 1.75 or teff > 9500:
                return tic_id, "REJECT_EVOLVED_STAR", f"REJECT {ext_telemetry}: Evolved giant or hot subdwarf host", None
                
            if np.isfinite(ruwe) and ruwe > 1.4:
                return tic_id, "REJECT_RUWE", f"REJECT {ext_telemetry}: Unresolved binary companion (RUWE > 1.4)", None
                
            if r_planet_earth > 15.0 or r_planet_earth < 0.8:
                if rank_idx == 1:
                    first_reject_code = "REJECT_RADIUS"
                    first_reject_msg = f"REJECT {ext_telemetry}: Radius out of planetary bounds ({r_planet_earth:.1f} R_E > 15 R_E -> EBLM/Star)"
                continue
                
            # Keplerian Transit Duration vs Stellar Density Check (accounting for e <= 0.30 and 1-sigma stellar errors)
            nom_max_dur_hr, min_kepler_dur_hr, max_kepler_dur_hr = compute_keplerian_duration_bounds_hours(
                best_p, r_star, star_info_cache["e_r_star"], m_star, star_info_cache["e_m_star"], rp_rs
            )
            obs_dur_hr = best_dur * 24.0
            if obs_dur_hr > max_kepler_dur_hr:
                if rank_idx == 1:
                    first_reject_code = "REJECT_KEPLER_DURATION_LONG"
                    first_reject_msg = f"REJECT {ext_telemetry}: Duration ({obs_dur_hr:.2f}h) exceeds Keplerian limit ({max_kepler_dur_hr:.2f}h, nom={nom_max_dur_hr:.2f}h) -> Blend/Subgiant"
                continue
            if obs_dur_hr < min_kepler_dur_hr:
                if rank_idx == 1:
                    first_reject_code = "REJECT_KEPLER_DURATION_SHORT"
                    first_reject_msg = f"REJECT {ext_telemetry}: Duration ({obs_dur_hr:.2f}h) is unphysically short (Keplerian min={min_kepler_dur_hr:.2f}h, nom={nom_max_dur_hr:.2f}h) -> Glitch/Pulsation"
                continue

            # Filter 6: Pixel-level TPF localization & strict dipole rejection
            tpf_ok, tpf_reason, diff_image, aperture = verify_and_filter_tpf(tic_id, last_sector, best_p, best_t0, best_dur)
            if not tpf_ok:
                if rank_idx == 1:
                    first_reject_code = "REJECT_TPF_PIXEL"
                    first_reject_msg = f"REJECT {ext_telemetry}: {tpf_reason}"
                continue

            # Filter 7: Multi-Epoch Archival Verification across prior TESS sectors
            arch_ok, arch_status, arch_info = verify_multi_epoch(
                tic_id, last_sector, t, flat_f, flat_ferr, best_p, best_dur, depth_ppm, sde
            )
            if not arch_ok:
                if rank_idx == 1:
                    first_reject_code = "REJECT_ARCHIVAL"
                    first_reject_msg = f"REJECT {ext_telemetry}: {arch_status}"
                continue

            refined_p = arch_info["refined_p"]
            arch_snr = arch_info["archival_snr"]
            arch_secs_str = arch_info["archival_sectors_str"]

            # Generate English Vetting Summary Plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].plot(periods, sde_arr, color="black", lw=0.8)
            axes[0, 0].axvline(best_p, color="red", linestyle="--", label=f"Peak #{rank_idx}: P = {refined_p:.5f} d")
            axes[0, 0].axhline(6.0, color="gray", linestyle=":", label="SDE = 6.0")
            axes[0, 0].set_xlabel("Trial Period (days)")
            axes[0, 0].set_ylabel("Kovács Whitened SDE")
            axes[0, 0].set_title(f"BLS Whitened SDE: TIC {tic_id} (SDE = {sde:.1f}, N_transits = {unique_transits})")
            axes[0, 0].legend(loc="upper right")
            
            axes[0, 1].scatter(phase[vis], flat_f[vis], s=2, color="gray", alpha=0.35, label=f"Sector {last_sector} (2-min)")
            axes[0, 1].plot(bin_centers, bin_means, color="red", lw=2, label=f"Sector {last_sector} Binned")
            if arch_info["arch_bin_centers"] is not None:
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
                f"Phase-Folded Transit: Rp = {r_planet_earth:.2f} R_Earth | Depth = {depth_ppm:.0f} ppm | {arch_status}"
            )
            axes[0, 1].legend(loc="lower right")
            
            axes[1, 0].scatter(phase[vis], dx[vis] * 21.0, s=2, alpha=0.35, label="dX (arcsec)")
            axes[1, 0].scatter(phase[vis], dy[vis] * 21.0, s=2, alpha=0.35, label="dY (arcsec)")
            axes[1, 0].axvspan(-0.5 * best_dur, 0.5 * best_dur, color="red", alpha=0.15, label="Transit Window")
            axes[1, 0].set_xlim(-win_phase, win_phase)
            axes[1, 0].set_ylim(-5, 5)
            axes[1, 0].set_xlabel("Phase (days)")
            axes[1, 0].set_ylabel("Centroid Offset (arcsec)")
            axes[1, 0].set_title(
                f"Centroid Stability: Shift = {shift_arcsec:.3f} arcsec | Odd/Even = {odd_even_sigma:.2f} sigma | RUWE = {ruwe:.2f}"
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
            axes[1, 1].set_title(f"Verified On-Target Difference Image (Sector {last_sector}, CROWDSAP={crowdsap:.2f})")
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"candidate_TIC_{tic_id}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            
            cand_data = {
                "pipeline_version": PIPELINE_VERSION,
                "tic_id": tic_id,
                "sector": last_sector,
                "archival_sectors": arch_secs_str,
                "archival_snr_sigma": round(arch_snr, 2) if np.isfinite(arch_snr) else "Single-Sector",
                "period_days": round(refined_p, 6),
                "t0_btjd": round(best_t0, 5),
                "duration_hours": round(obs_dur_hr, 2),
                "min_kepler_dur_hours": round(min_kepler_dur_hr, 2),
                "max_kepler_dur_hours": round(max_kepler_dur_hr, 2),
                "n_transits": unique_transits,
                "depth_ppm": round(depth_ppm, 1),
                "crowdsap": round(crowdsap, 3),
                "sde": round(sde, 2),
                "r_planet_earth": round(r_planet_earth, 2),
                "r_star_sun": round(r_star, 2),
                "m_star_sun": round(m_star, 2),
                "teff_k": round(teff, 0),
                "gaia_ruwe": round(ruwe, 3) if np.isfinite(ruwe) else np.nan,
                "odd_even_sigma": round(odd_even_sigma, 2),
                "centroid_shift_arcsec": round(shift_arcsec, 3),
                "exofop_url": f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"
            }
            
            msg = (
                f"CANDIDATE FOUND (Peak #{rank_idx})! P={refined_p:.5f} d | N={unique_transits} | Depth={depth_ppm:.0f} ppm | "
                f"Rp={r_planet_earth:.2f} R_E | SDE={sde:.1f} | Archival={arch_status}"
            )
            return tic_id, "CANDIDATE", msg, cand_data
            
        if AUTO_PURGE_CACHE:
            purge_target_cache(tic_id)
        return tic_id, first_reject_code, first_reject_msg, None
        
    except Exception as e:
        if AUTO_PURGE_CACHE:
            purge_target_cache(tic_id)
        return tic_id, "ERROR", f"ERROR: {e}", None


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
    
    active_sector, targets = select_latest_sector_and_targets(sector_list=PREFERRED_SECTORS, limit=MAX_TARGETS)
    candidates = []
    
    if os.path.exists(REPORT_FILE):
        try:
            candidates = pd.read_csv(REPORT_FILE).to_dict("records")
        except Exception:
            candidates = []
            
    write_log(f"Launching parallel pipeline on {MAX_WORKERS} CPU workers for Sector {active_sector}...\n")
    
    completed = 0
    total = len(targets)
    stats_counter = {}
    
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    try:
        future_to_tic = {
            executor.submit(analyze_single_target, tic, active_sector): tic
            for tic in targets
        }
        
        for future in as_completed(future_to_tic):
            completed += 1
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
                pd.DataFrame(candidates).to_csv(REPORT_FILE, index=False)
            else:
                write_log(f"[{completed}/{total}] [TIC {tic_id}] {status_msg}")

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
        
    summary_str = " | ".join(f"{k}: {v}" for k, v in sorted(stats_counter.items()))
    final_summary = f"Run Summary (Sector {active_sector}, {completed}/{total} processed): {summary_str}"
    write_log(f"\n{final_summary}")
    write_log(f"Total candidates saved: {len(candidates)}. Output folder: '{OUTPUT_DIR}'.")
    write_event(final_summary)
    write_event(f"Completed Sector {active_sector}. Total candidates: {len(candidates)}.")