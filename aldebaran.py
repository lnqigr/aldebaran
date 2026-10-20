import os
import io
import ssl
import json
import shutil
import hashlib
import glob
import logging
import warnings
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
    """
    Executes an HTTPS GET request with strict certifi TLS verification.
    Falls back to unverified TLS strictly on legacy HPC nodes raising SSLError.
    """
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=CA_BUNDLE_PATH)
        resp.raise_for_status()
        return resp
    except requests.exceptions.SSLError as ssl_err:
        logger.warning("TLS verification failed (%s); falling back to unverified context for %s", ssl_err, url)
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
        resp.raise_for_status()
        return resp


def safe_urlopen(req_or_url, timeout: int = 45):
    """
    Executes urllib.request.urlopen with explicit certifi CA bundle and logged SSLError fallback.
    """
    try:
        ctx = ssl.create_default_context(cafile=CA_BUNDLE_PATH)
        return urllib.request.urlopen(req_or_url, timeout=timeout, context=ctx)
    except ssl.SSLError as ssl_err:
        logger.warning("urllib TLS handshake failed (%s); using unverified fallback.", ssl_err)
        unverified_ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(req_or_url, timeout=timeout, context=unverified_ctx)


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


def analyze_single_target(
    tic_id: str,
    sector: int,
    min_period: float = 0.8,
    max_period: float = 13.0,
    run_id: str = None,
    attempt_id: str = None,
    *,
    observer_context: Any = None
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
        result = _analyze_single_target_impl(
            tic_id, sector, min_period, max_period, run_id=run_id, attempt_id=attempt_id,
            observer_context=observer_context, ledger=ledger
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


def _analyze_single_target_impl(
    tic_id: str,
    sector: int,
    min_period: float = 0.8,
    max_period: float = 13.0,
    run_id: str = None,
    attempt_id: str = None,
    *,
    observer_context: Any = None,
    ledger: Any = None
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

    attempt_diagnostics = []

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
                t_curr_eval, flat_f, best_p, best_t0, best_dur, depth_ppm
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
            dx_in, dx_out = dx_curr_eval[in_tr], dx_curr_eval[out_tr]
            dy_in, dy_out = dy_curr_eval[in_tr], dy_curr_eval[out_tr]
            
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
                return make_res(tic_id, "REJECT_NO_STELLAR_PARAMS", f"REJECT {base_telemetry}: Missing measured R_star/Teff in TIC catalog", None)
                
            r_star = star_info_cache["r_star"]
            m_star = star_info_cache["m_star"]
            teff = star_info_cache["teff"]
            ruwe = star_info_cache["ruwe"]
            
            rp_rs = np.sqrt(max(true_depth_ppm, 0.0) * 1e-6)
            r_planet_earth = float(r_star * rp_rs * 109.076)
            ext_telemetry = f"[Peak#{rank_idx} | SDE={sde:.1f} | P={best_p:.4f}d | Depth={depth_ppm:.0f}ppm | Rp={r_planet_earth:.1f}R_E | R*={r_star:.2f}R_Sun | RUWE={ruwe:.2f}]"
            
            if r_star > 1.75 or teff > 9500:
                return make_res(tic_id, "REJECT_EVOLVED_STAR", f"REJECT {ext_telemetry}: Evolved giant or hot subdwarf host", None)
                
            if np.isfinite(ruwe) and ruwe > 1.4:
                return make_res(tic_id, "REJECT_RUWE", f"REJECT {ext_telemetry}: Unresolved binary companion (RUWE > 1.4)", None)
                
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
            record_archival_diagnostic(target_diag)
            attempt_diagnostics.append(target_diag)

            if not arch_ok:
                if rank_idx == 1:
                    first_reject_code = "REJECT_ARCHIVAL"
                    first_reject_msg = f"REJECT {ext_telemetry}: {arch_status}"
                continue

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
            
            cand_tag = f"TOI SIBLING (Host of {known_transits[0]['toi']})" if is_toi_host else "CANDIDATE"
            trials_str = f"Trials={arch_info.get('n_trials', 1)}, Crit={arch_info.get('snr_crit', 2.4):.1f}s"
            msg = (
                f"{cand_tag} FOUND (Peak #{rank_idx})! P={selected_p:.5f} d | N={unique_transits} | Depth={depth_ppm:.0f} ppm | "
                f"Rp={r_planet_earth:.2f} R_E | SDE={sde:.1f} | Archival={arch_display_status} ({trials_str})"
            )
            return make_res(tic_id, "CANDIDATE", msg, cand_data)
            
        if AUTO_PURGE_CACHE:
            purge_target_cache(tic_id)
        return make_res(tic_id, first_reject_code, first_reject_msg, None)
        
    except Exception as e:
        if AUTO_PURGE_CACHE:
            purge_target_cache(tic_id)
        return make_res(tic_id, "ERROR", f"ERROR: {e}", None)


def save_candidates_to_report(report_path: str, candidate_records: list):
    """
    Saves candidate records to report_path using atomic replacement to prevent file corruption.
    """
    df_out = pd.DataFrame(candidate_records, columns=CANDIDATE_COLUMNS)
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