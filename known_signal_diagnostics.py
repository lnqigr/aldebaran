"""
known_signal_diagnostics.py - Defensive Known-Signal Geometry Diagnostics,
Explicit Uncertainty Provenance, and Interval Coverage (Task ALD-007B1).

Strictly offline, non-destructive, and diagnostic-only.
Does NOT activate production masking, alter candidate veto thresholds,
or modify existing catalog/search pipelines.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Constants and Scientific Boundary Declarations
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "known_signal_diagnostics_v1"
MODE: str = "DIAGNOSTIC_ONLY"
PRODUCTION_MASK_ALLOWED: bool = False

ALLOWED_DURATION_FRAMES: tuple[str, ...] = (
    "INTRINSIC_DECONVOLVED",
    "SMEARED_OBSERVED",
)

DEFAULT_STAGE_MASKABLE_DISPOSITIONS: tuple[str, ...] = (
    "CP",
    "KP",
    "CONFIRMED",
    "PC",
    "FP",
    "EB",
)

# Robust floating-point tolerance bounds to prevent numerical underflow/overflow.
# ULP/operation-aware integer rebase tolerance avoids unmodeled phase error
# (e.g. 7.776s phase shift rejected) while safely handling floating-point cancellation
# across all time ranges (including large BJDs, BTJD, zero, or negative epochs).
MIN_REPRESENTABLE_VARIANCE: float = 1e-150
MAX_REPRESENTABLE_VARIANCE: float = 1e150
REBASE_INTEGER_TOLERANCE: float = 1e-11


def check_integer_rebase(
    t0: float,
    orig_t0: float,
    period: float,
) -> tuple[bool, int, float, float]:
    """
    Operation-aware ULP bound check for integer epoch rebase:
        m = (t0 - orig_t0) / period.
    Avoids unmodeled phase error (e.g. 7.776s phase shift rejected) while safely
    handling floating-point cancellation across all time ranges (including large BJDs,
    BTJD, zero, or negative epochs).

    Returns:
        (is_integer, round_m, cycle_diff, max_allowed_tol)
    """
    m = (float(t0) - float(orig_t0)) / float(period)
    round_m = int(round(m))
    cycle_diff = abs(m - round_m)
    ulp_t0 = math.ulp(float(t0))
    ulp_orig = math.ulp(float(orig_t0))
    ulp_period = math.ulp(float(period))
    max_tol = max(1e-11, 8.0 * max(ulp_t0, ulp_orig, ulp_period) / float(period))
    return (cycle_diff <= max_tol, round_m, cycle_diff, max_tol)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class TransformedCovariance(tuple):
    """
    4-tuple (t0_new, var_t0_new, cov_new, var_p_new) with preserved provenance attributes.
    Supports standard tuple unpacking:
        t0, v, c, vp = transform_ephemeris_epoch_covariance(...)
    as well as attribute and dictionary access (.epoch_identity, .cycle_shift_m, .to_dict()).
    """
    def __new__(
        cls,
        t0: float,
        var_t0: float,
        cov_t0_p: float,
        var_p: float,
        cycle_shift_m: int = 0,
        epoch_identity: Optional[str] = None,
    ):
        return super().__new__(
            cls,
            (float(t0), float(var_t0), float(cov_t0_p), float(var_p)),
        )

    def __init__(
        self,
        t0: float,
        var_t0: float,
        cov_t0_p: float,
        var_p: float,
        cycle_shift_m: int = 0,
        epoch_identity: Optional[str] = None,
    ):
        self.epoch_btjd = float(t0)
        self.var_t0_days2 = float(var_t0)
        self.cov_t0_p_days2 = float(cov_t0_p)
        self.var_p_days2 = float(var_p)
        self.cycle_shift_m = int(cycle_shift_m)
        self.epoch_identity = str(epoch_identity) if epoch_identity is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_btjd": self.epoch_btjd,
            "var_t0_days2": self.var_t0_days2,
            "cov_t0_p_days2": self.cov_t0_p_days2,
            "var_p_days2": self.var_p_days2,
            "cycle_shift_m": self.cycle_shift_m,
            "epoch_identity": self.epoch_identity,
        }


# ---------------------------------------------------------------------------
# Type and Numerical Validation Helpers
# ---------------------------------------------------------------------------

def is_valid_integer_shift(m: object) -> bool:
    """
    Checks whether m is a genuine integer scalar.
    Strictly rejects boolean values, None, non-finite floats, and fractional floats.
    """
    if isinstance(m, bool) or m is None:
        return False
    if isinstance(m, (int, np.integer)):
        return True
    if isinstance(m, (float, np.floating)):
        if not math.isfinite(float(m)):
            return False
        return float(m).is_integer()
    return False


def is_psd_covariance_2x2(
    var_t0: float,
    cov_t0_p: float,
    var_p: float,
    rel_tol: float = 1e-12,
) -> bool:
    """
    Scale-invariant and underflow-safe check for 2x2 positive semidefinite covariance matrix:
        Sigma = [[var_t0,   cov_t0_p],
                 [cov_t0_p, var_p   ]]

    Invariants:
      1. Inputs must be finite numeric scalars (strictly non-boolean).
      2. Diagonal variances must be non-negative: var_t0 >= 0 and var_p >= 0.
      3. Zero marginal variance on either axis requires identically zero cross-covariance.
      4. Positive variances must lie within stable representable range [1e-150, 1e150].
      5. Stable correlation check: |cov_t0_p| / (sqrt(var_t0) * sqrt(var_p)) <= 1.0 + rel_tol.
         Computing via separate square roots avoids catastrophic underflow of var_t0 * var_p
         to zero when variances are tiny (e.g. 1e-200).
    """
    if isinstance(var_t0, bool) or isinstance(cov_t0_p, bool) or isinstance(var_p, bool):
        return False
    try:
        v0 = float(var_t0)
        cov = float(cov_t0_p)
        vp = float(var_p)
    except (TypeError, ValueError):
        return False

    if not (math.isfinite(v0) and math.isfinite(cov) and math.isfinite(vp)):
        return False
    if v0 < 0.0 or vp < 0.0:
        return False
    if v0 == 0.0 or vp == 0.0:
        return cov == 0.0

    # Underflow/overflow range guard: reject non-representable extreme scales
    if v0 < MIN_REPRESENTABLE_VARIANCE or v0 > MAX_REPRESENTABLE_VARIANCE:
        return False
    if vp < MIN_REPRESENTABLE_VARIANCE or vp > MAX_REPRESENTABLE_VARIANCE:
        return False

    # Stable correlation computation using separate square roots
    std_t0 = math.sqrt(v0)
    std_p = math.sqrt(vp)
    denom = std_t0 * std_p
    if denom == 0.0 or not math.isfinite(denom):
        return False

    corr = abs(cov) / denom
    if not math.isfinite(corr):
        return False
    return corr <= (1.0 + rel_tol)


def transform_ephemeris_epoch_covariance(
    t0: float,
    period: float,
    var_t0: float,
    cov_t0_p: float,
    var_p: float,
    cycle_shift_m: int,
    *,
    epoch_identity: Optional[str] = None,
) -> TransformedCovariance:
    """
    Strict Jacobian transport J * Sigma * J^T for an exact integer cycle shift m:
        T0' = T0 + m * P
        Var(T0') = Var(T0) + 2*m*Cov(T0, P) + m^2 * Var(P)
        Cov(T0', P) = Cov(T0, P) + m * Var(P)
        Var(P') = Var(P)

    Preserves full predictive timing variance:
        Var(T0' + (n - m)*P) == Var(T0 + n*P) for any cycle n.

    Invariants:
      - cycle_shift_m must be an exact integer (rejects bool, fractional float).
      - period must be a finite, strictly positive number.
      - Input covariance matrix must be positive semidefinite.
      - Derived quantities must be finite numbers within numeric representability bounds.
    """
    if not is_valid_integer_shift(cycle_shift_m):
        raise ValueError(f"cycle_shift_m must be an exact integer, got {cycle_shift_m!r}")
    if isinstance(period, bool) or not math.isfinite(period) or period <= 0.0:
        raise ValueError(f"period must be a finite positive number, got {period!r}")
    if isinstance(t0, bool) or not math.isfinite(t0):
        raise ValueError(f"t0 must be a finite number, got {t0!r}")
    if not is_psd_covariance_2x2(var_t0, cov_t0_p, var_p):
        raise ValueError("Input covariance matrix is not positive semidefinite")

    m = int(cycle_shift_m)
    t0_new = float(t0) + m * float(period)
    var_t0_new = float(var_t0) + 2.0 * m * float(cov_t0_p) + (m ** 2) * float(var_p)
    cov_new = float(cov_t0_p) + m * float(var_p)
    var_p_new = float(var_p)

    if not (
        math.isfinite(t0_new)
        and math.isfinite(var_t0_new)
        and math.isfinite(cov_new)
        and math.isfinite(var_p_new)
    ):
        raise ValueError("Transformed covariance contains non-finite values (numeric overflow)")

    if var_t0_new < 0.0:
        scale = float(var_t0) + (m ** 2) * float(var_p)
        if abs(var_t0_new) <= 1e-12 * scale:
            var_t0_new = 0.0
        else:
            raise ValueError("Transformed variance is materially negative")

    return TransformedCovariance(
        t0_new,
        var_t0_new,
        cov_new,
        var_p_new,
        cycle_shift_m=m,
        epoch_identity=epoch_identity,
    )


# ---------------------------------------------------------------------------
# Interval Union and Coverage Geometry
# ---------------------------------------------------------------------------

def measure_interval_union(intervals: Sequence[tuple[float, float]]) -> float:
    """
    Exact Lebesgue measure of 1D interval union with deduplication and touching endpoint merge.
    Enforces strict interval validity: each interval must have finite bounds with b >= a.
    Reversed intervals (b < a) or non-finite bounds raise ValueError.
    """
    if intervals is None:
        return 0.0
    clean_intervals: list[tuple[float, float]] = []
    for item in intervals:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"Interval must be a 2-tuple (start, end), got {item!r}")
        a, b = item
        if isinstance(a, bool) or isinstance(b, bool):
            raise ValueError(f"Interval bounds cannot be boolean: [{a}, {b}]")
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            raise ValueError(f"Non-numeric interval bounds: [{a}, {b}]")
        if not (math.isfinite(fa) and math.isfinite(fb)):
            raise ValueError(f"Non-finite interval bounds: [{fa}, {fb}]")
        if fb < fa:
            raise ValueError(f"Reversed interval: [{fa}, {fb}] (end < start)")
        if fb > fa:
            clean_intervals.append((fa, fb))

    if not clean_intervals:
        return 0.0

    clean_intervals.sort(key=lambda x: (x[0], x[1]))
    total = 0.0
    cur_a, cur_b = clean_intervals[0]
    for a, b in clean_intervals[1:]:
        if a <= cur_b:
            if b > cur_b:
                cur_b = b
        else:
            total += (cur_b - cur_a)
            cur_a, cur_b = a, b
    total += (cur_b - cur_a)
    return float(total)


def compute_window_interval_coverage(
    exposures: Union[Sequence[tuple[float, float]], np.ndarray],
    window: tuple[float, float],
    *,
    exp_time_days: Optional[float] = None,
    clean_mask: Optional[Union[Sequence[bool], np.ndarray]] = None,
) -> dict[str, Any]:
    """
    Computes geometric coverage G and clean coverage C over a transit window:
        G = measure(Union(exposures) cap window) / measure(window)
        C = measure(Union(clean_exposures) cap window) / measure(window)

    Invariants:
      - window must be a finite interval [w_start, w_end] with w_end > w_start.
        Zero or negative window denominator returns structured unavailable.
      - Input exposures must have valid dimensions and finite bounds.
        Non-finite or reversed exposures return structured unavailable with exact reasons,
        never silently dropped.
      - clean_mask elements must be strictly booleans (rejects string 'False').
      - Duplicate exposures with conflicting quality are handled conservatively:
        a contaminated copy of an exposure marks that region unclean.
      - Guaranteed invariant on valid inputs: 0.0 <= C <= G <= 1.0.
      - Returns directly JSON-serializable structure with excluded-time fractions and metrics.
    """
    def _cov_err(status: str, reason: str, raw_cnt: int = 0, w_l: float = 0.0) -> dict[str, Any]:
        return {
            "status": status,
            "coverage_valid": False,
            "G": None,
            "C": None,
            "window_length_days": float(w_l),
            "covered_time_days": None,
            "clean_covered_time_days": None,
            "excluded_observation_time_days": None,
            "excluded_observation_time_fraction": None,
            "raw_exposure_count": int(raw_cnt),
            "deduplicated_exposure_count": 0,
            "merged_interval_count": 0,
            "intersecting_exposure_count": 0,
            "reasons": [reason],
        }

    if window is None or len(window) != 2:
        return _cov_err("INVALID_WINDOW_SPECIFICATION", "Window specification must be a 2-tuple (start, end)")
    try:
        w_start, w_end = float(window[0]), float(window[1])
    except (TypeError, ValueError):
        return _cov_err("NON_FINITE_WINDOW_BOUNDS", "Window bounds must be real numbers")

    if not (math.isfinite(w_start) and math.isfinite(w_end)):
        return _cov_err("NON_FINITE_WINDOW_BOUNDS", "Window bounds must be finite")

    w_len = w_end - w_start
    if w_len <= 0.0:
        return _cov_err("ZERO_OR_NEGATIVE_WINDOW_DENOMINATOR", f"Window length {w_len} <= 0 (zero or negative denominator)", w_l=w_len)

    # Extract raw exposures
    if exposures is None:
        raw_exposure_count = 0
    else:
        raw_exposure_count = len(exposures)

    # 1. Validate clean_mask strictly
    validated_clean: list[bool] = []
    if clean_mask is not None:
        c_list = list(clean_mask)
        if len(c_list) != raw_exposure_count:
            return _cov_err("INVALID_CLEAN_MASK_LENGTH", f"clean_mask length {len(c_list)} != exposures length {raw_exposure_count}", raw_cnt=raw_exposure_count, w_l=w_len)
        for c_idx, c_val in enumerate(c_list):
            if isinstance(c_val, (str, bytes)):
                return _cov_err("INVALID_CLEAN_MASK", f"clean_mask element at index {c_idx} cannot be string: {c_val!r}", raw_cnt=raw_exposure_count, w_l=w_len)
            if isinstance(c_val, (bool, np.bool_)):
                validated_clean.append(bool(c_val))
            elif isinstance(c_val, (int, np.integer)) and c_val in (0, 1):
                validated_clean.append(bool(c_val))
            else:
                return _cov_err("INVALID_CLEAN_MASK", f"clean_mask element at index {c_idx} is not boolean: {c_val!r}", raw_cnt=raw_exposure_count, w_l=w_len)
    else:
        validated_clean = [True] * raw_exposure_count

    # 2. Extract and validate exposure intervals
    intervals: list[tuple[float, float]] = []

    if isinstance(exposures, np.ndarray) and exposures.ndim == 1 and exp_time_days is not None:
        if isinstance(exp_time_days, bool) or not math.isfinite(exp_time_days) or exp_time_days <= 0.0:
            return _cov_err("INVALID_EXP_TIME", f"exp_time_days must be finite positive, got {exp_time_days}", raw_cnt=raw_exposure_count, w_l=w_len)
        half_exp = 0.5 * float(exp_time_days)
        for idx, tc in enumerate(exposures):
            if isinstance(tc, (bool, np.bool_)):
                return _cov_err("INVALID_EXPOSURE_BOUNDS", f"Exposure center at index {idx} cannot be boolean", raw_cnt=raw_exposure_count, w_l=w_len)
            try:
                tc_f = float(tc)
            except (TypeError, ValueError):
                return _cov_err("INVALID_EXPOSURE_BOUNDS", f"Non-numeric exposure center at index {idx}: {tc!r}", raw_cnt=raw_exposure_count, w_l=w_len)
            if not math.isfinite(tc_f):
                return _cov_err("NON_FINITE_EXPOSURE_BOUNDS", f"Non-finite exposure center at index {idx}: {tc_f}", raw_cnt=raw_exposure_count, w_l=w_len)
            intervals.append((tc_f - half_exp, tc_f + half_exp))

    else:
        # Sequence or 2D array of (start, end)
        if isinstance(exposures, np.ndarray) and exposures.ndim not in (1, 2):
            return _cov_err("INVALID_EXPOSURE_DIMENSIONS", f"Invalid exposure array dimensions: ndim={exposures.ndim}", raw_cnt=raw_exposure_count, w_l=w_len)

        for idx, item in enumerate(exposures if exposures is not None else []):
            if not isinstance(item, (tuple, list, np.ndarray)) or len(item) != 2:
                return _cov_err("INVALID_EXPOSURE_BOUNDS", f"Exposure at index {idx} must be a 2-element interval (start, end)", raw_cnt=raw_exposure_count, w_l=w_len)
            a_val, b_val = item[0], item[1]
            if isinstance(a_val, (bool, np.bool_)) or isinstance(b_val, (bool, np.bool_)):
                return _cov_err("INVALID_EXPOSURE_BOUNDS", f"Exposure bounds at index {idx} cannot be boolean: [{a_val}, {b_val}]", raw_cnt=raw_exposure_count, w_l=w_len)
            try:
                a, b = float(a_val), float(b_val)
            except (TypeError, ValueError):
                return _cov_err("INVALID_EXPOSURE_BOUNDS", f"Non-numeric exposure bounds at index {idx}: [{a_val}, {b_val}]", raw_cnt=raw_exposure_count, w_l=w_len)
            if not (math.isfinite(a) and math.isfinite(b)):
                return _cov_err("NON_FINITE_EXPOSURE_BOUNDS", f"Non-finite exposure bounds at index {idx}: [{a}, {b}]", raw_cnt=raw_exposure_count, w_l=w_len)
            if b < a:
                return _cov_err("REVERSED_EXPOSURE_BOUNDS", f"Reversed exposure bounds at index {idx}: [{a}, {b}] (end < start)", raw_cnt=raw_exposure_count, w_l=w_len)
            intervals.append((a, b))

    # 3. Clip to window and separate clean vs unclean intervals
    all_clipped: list[tuple[float, float]] = []
    unclean_clipped: list[tuple[float, float]] = []
    clean_clipped_candidates: list[tuple[float, float]] = []

    for (a, b), is_c in zip(intervals, validated_clean):
        ca = max(a, w_start)
        cb = min(b, w_end)
        if cb > ca:
            all_clipped.append((ca, cb))
            if is_c:
                clean_clipped_candidates.append((ca, cb))
            else:
                unclean_clipped.append((ca, cb))

    # Total geometric covered time
    tot_covered_time = measure_interval_union(all_clipped)
    g_k = tot_covered_time / w_len

    # Conservative clean handling: any overlapping unclean exposure removes that portion
    if not unclean_clipped:
        clean_covered_time = measure_interval_union(clean_clipped_candidates)
    else:
        unclean_union = measure_interval_union(unclean_clipped)
        combined_all = measure_interval_union(clean_clipped_candidates + unclean_clipped)
        clean_covered_time = max(0.0, combined_all - unclean_union)

    c_k = clean_covered_time / w_len
    # Enforce guaranteed physical invariant
    c_k = min(c_k, g_k)

    # Excluded time metrics
    excluded_time_days = max(0.0, tot_covered_time - clean_covered_time)
    cov_reasons: list[str] = []
    if tot_covered_time <= 0.0:
        excluded_time_fraction: Optional[float] = None
        cov_reasons.append("Total observed time in window is zero; excluded_observation_time_fraction is undefined")
    else:
        excluded_time_fraction = float(excluded_time_days / tot_covered_time)

    # Deduplicated exposure intervals count (unique (start, end) tuples across exposures)
    unique_tuples = set((round(float(a), 12), round(float(b), 12)) for a, b in intervals)
    dedup_count = len(unique_tuples)

    # Merged contiguous intervals in window (disjoint union connected components)
    if all_clipped:
        sorted_clipped = sorted(all_clipped, key=lambda x: (x[0], x[1]))
        merged_chunks = 1
        cur_end = sorted_clipped[0][1]
        for a_c, b_c in sorted_clipped[1:]:
            if a_c > cur_end:
                merged_chunks += 1
                cur_end = b_c
            else:
                cur_end = max(cur_end, b_c)
        merged_interval_count = merged_chunks
    else:
        merged_interval_count = 0

    return {
        "status": "VALID_INTERVAL_COVERAGE",
        "coverage_valid": True,
        "G": float(g_k),
        "C": float(c_k),
        "window_length_days": float(w_len),
        "covered_time_days": float(tot_covered_time),
        "clean_covered_time_days": float(clean_covered_time),
        "excluded_observation_time_days": float(excluded_time_days),
        "excluded_observation_time_fraction": excluded_time_fraction,
        "raw_exposure_count": int(raw_exposure_count),
        "deduplicated_exposure_count": int(dedup_count),
        "merged_interval_count": int(merged_interval_count),
        "intersecting_exposure_count": int(len(all_clipped)),
        "reasons": cov_reasons,
    }


# ---------------------------------------------------------------------------
# Core Evaluation API
# ---------------------------------------------------------------------------

def _safe_finite_float(val: object) -> Optional[float]:
    """Safely extracts a finite float scalar without throwing on non-scalars."""
    if val is None or isinstance(val, (bool, np.bool_, str, bytes, list, tuple, dict, set)):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _validate_numeric_param(
    val: object,
    name: str,
    *,
    allow_zero: bool = False,
    allow_negative: bool = False,
) -> tuple[Optional[float], Optional[str]]:
    """Strictly validates a numeric parameter: rejects strings, bools, and non-finites."""
    if val is None:
        return None, f"Missing required parameter: {name}"
    if isinstance(val, (bool, np.bool_)):
        return None, f"Parameter {name} cannot be boolean"
    if isinstance(val, (str, bytes)):
        return None, f"Parameter {name} must be numeric, got string {val!r}"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None, f"Parameter {name} must be numeric, got {val!r}"
    if not math.isfinite(f):
        return None, f"Parameter {name} must be finite, got {f}"
    if not allow_negative and f < 0.0:
        return None, f"Parameter {name} cannot be negative, got {f}"
    if not allow_zero and f == 0.0:
        return None, f"Parameter {name} must be strictly positive, got {f}"
    return f, None


def _clean_str_id(val: Any) -> Optional[str]:
    """Return stripped non-empty string or None for missing, blank, or boolean values."""
    if val is None or isinstance(val, (bool, np.bool_)):
        return None
    s = str(val).strip()
    return s if s else None


def evaluate_signal_mask_contract(
    t_btjd: Union[Sequence[float], np.ndarray],
    *,
    period_days: Optional[float],
    epoch_btjd: Optional[float],
    duration_days: Optional[float],
    raw_disposition: Optional[str] = None,
    stage_policy_action: Optional[str] = None,
    allowed_dispositions: Optional[Sequence[str]] = None,
    normalized_time_format: Optional[str] = None,
    normalized_time_scale: Optional[str] = None,
    original_time_format: Optional[str] = None,
    original_time_scale: Optional[str] = None,
    ephemeris_coherence_status: Optional[str] = None,
    has_unmodeled_ttv: bool = False,
    var_t0_days2: Optional[float] = None,
    cov_t0_p_days2: Optional[float] = None,
    var_p_days2: Optional[float] = None,
    covariance_epoch_btjd: Optional[float] = None,
    covariance_solution_id: Optional[str] = None,
    sigma_t0_days: Optional[float] = None,
    sigma_p_days: Optional[float] = None,
    sigma_dur_days: Optional[float] = None,
    published_epoch_btjd: Optional[float] = None,
    marginal_errors_at_published_epoch: bool = False,
    marginal_errors_solution_id: Optional[str] = None,
    published_solution_id: Optional[str] = None,
    duration_frame_type: Optional[str] = None,
    exp_time_days: Optional[float] = None,
    k_sigma: Optional[float] = None,
    k_dur: Optional[float] = None,
    alpha_stage: Optional[float] = None,
    max_stage_half_width_days: Optional[float] = None,
    max_cycle_ambiguity_fraction: Optional[float] = None,
    max_half_orbit_fraction: Optional[float] = None,
    signal_id: Optional[str] = None,
    solution_id: Optional[str] = None,
    source_snapshot_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Defensive known-signal geometry evaluation and hypothetical masking contract.

    Invariants:
      1. Pure diagnostic helper: returns mode="DIAGNOSTIC_ONLY" and production_mask_allowed=False.
      2. Rejection of booleans masquerading as numbers, invalid dimensions, non-finite values,
         and invalid duration frames.
      3. Strict covariance verification: 2x2 PSD check with scale-underflow guard.
         Requires covariance_epoch_btjd and evaluates variance relative to covariance epoch.
      4. Cauchy-Schwarz bound branch requires:
         - Strict boolean provenance: marginal_errors_at_published_epoch must be True.
         - Explicit published_epoch_btjd.
         - Verified integer rebase between epoch_btjd and published_epoch_btjd.
      5. Explicit hypothetical stage authorization: stage_policy_action="ALLOW_ACTIVE_MASK"
         and raw_disposition in allowed_dispositions required to grant hypothetical mask validity.
         No default whitelist.
      6. Output is guaranteed directly JSON-serializable (mask is list[bool], all numbers finite).
    """
    reasons: list[str] = []

    def _make_record(
        status: str,
        geometry_status: str,
        mask_valid: bool,
        mask_list: list[bool],
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        marg_sol = marginal_errors_solution_id if marginal_errors_solution_id is not None else published_solution_id
        rec: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode": MODE,
            "production_mask_allowed": PRODUCTION_MASK_ALLOWED,
            "status": status,
            "geometry_status": geometry_status,
            "mask_valid": bool(mask_valid),
            "mask": list(mask_list),
            "time_format": "BTJD",
            "time_scale": "TDB",
            "normalized_time_format": str(normalized_time_format) if normalized_time_format is not None else None,
            "normalized_time_scale": str(normalized_time_scale) if normalized_time_scale is not None else None,
            "original_time_format": str(original_time_format) if original_time_format is not None else None,
            "original_time_scale": str(original_time_scale) if original_time_scale is not None else None,
            "ephemeris_coherence_status": str(ephemeris_coherence_status) if ephemeris_coherence_status is not None else None,
            "has_unmodeled_ttv": bool(has_unmodeled_ttv),
            "duration_frame_type": str(duration_frame_type) if duration_frame_type is not None else None,
            "period_days": _safe_finite_float(period_days),
            "epoch_btjd": _safe_finite_float(epoch_btjd),
            "duration_days": _safe_finite_float(duration_days),
            "var_t0_days2": _safe_finite_float(var_t0_days2),
            "cov_t0_p_days2": _safe_finite_float(cov_t0_p_days2),
            "var_p_days2": _safe_finite_float(var_p_days2),
            "covariance_epoch_btjd": _safe_finite_float(covariance_epoch_btjd),
            "covariance_solution_id": str(covariance_solution_id) if covariance_solution_id is not None else None,
            "sigma_t0_days": _safe_finite_float(sigma_t0_days),
            "sigma_p_days": _safe_finite_float(sigma_p_days),
            "sigma_dur_days": _safe_finite_float(sigma_dur_days),
            "published_epoch_btjd": _safe_finite_float(published_epoch_btjd),
            "marginal_errors_at_published_epoch": bool(marginal_errors_at_published_epoch) if isinstance(marginal_errors_at_published_epoch, (bool, np.bool_)) else None,
            "marginal_errors_solution_id": str(marg_sol) if marg_sol is not None else None,
            "k_sigma": _safe_finite_float(k_sigma),
            "k_dur": _safe_finite_float(k_dur),
            "alpha_stage": _safe_finite_float(alpha_stage),
            "exp_time_days": _safe_finite_float(exp_time_days),
            "max_stage_half_width_days": _safe_finite_float(max_stage_half_width_days),
            "max_cycle_ambiguity_fraction": _safe_finite_float(max_cycle_ambiguity_fraction),
            "max_half_orbit_fraction": _safe_finite_float(max_half_orbit_fraction),
            "stage_policy_action": str(stage_policy_action) if stage_policy_action is not None else None,
            "allowed_dispositions": [str(d) for d in allowed_dispositions] if (allowed_dispositions is not None and isinstance(allowed_dispositions, (list, tuple, set))) else None,
            "raw_disposition": str(raw_disposition) if raw_disposition is not None else None,
            "signal_id": str(signal_id) if signal_id is not None else None,
            "solution_id": str(solution_id) if solution_id is not None else None,
            "source_snapshot_id": str(source_snapshot_id) if source_snapshot_id is not None else None,
            "reasons": list(reasons),
            "time_mask_fraction": None,
            "excluded_observation_time_days": 0.0,
            "total_observation_time_days": 0.0,
            "observation_point_count": len(mask_list),
            "masked_point_count": int(sum(1 for m in mask_list if m)),
        }
        if extra:
            rec.update(extra)
        return rec

    # 1. Validate time array
    if t_btjd is None:
        reasons.append("Time array is None")
        return _make_record("INVALID_TIME_ARRAY", "UNCHECKED", False, [])

    # Check for boolean time array (reject bools masquerading as numbers)
    if isinstance(t_btjd, np.ndarray) and t_btjd.dtype == bool:
        reasons.append("Time array has boolean dtype")
        return _make_record("INVALID_TIME_ARRAY", "UNCHECKED", False, [False] * len(t_btjd))

    try:
        if any(isinstance(x, (bool, np.bool_)) for x in (t_btjd if not isinstance(t_btjd, np.ndarray) else t_btjd.ravel())):
            reasons.append("Time array contains boolean values")
            raw_len = len(t_btjd) if hasattr(t_btjd, "__len__") else 0
            return _make_record("INVALID_TIME_ARRAY", "UNCHECKED", False, [False] * raw_len)
    except (TypeError, ValueError):
        pass

    try:
        t_arr = np.asarray(t_btjd, dtype=float)
    except (TypeError, ValueError):
        reasons.append("Time array contains non-numeric values")
        return _make_record("INVALID_TIME_ARRAY", "UNCHECKED", False, [])

    if t_arr.ndim != 1 or len(t_arr) == 0 or not np.all(np.isfinite(t_arr)):
        reasons.append("Time array must be a 1D non-empty array of finite real numbers")
        return _make_record("INVALID_TIME_ARRAY", "UNCHECKED", False, [False] * (len(t_arr) if t_arr.ndim == 1 else 0))

    empty_mask = [False] * len(t_arr)

    # 2. Validate policy inputs
    k_sig_val, err = _validate_numeric_param(k_sigma, "k_sigma", allow_zero=False)
    if err:
        reasons.append(err)
        return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)

    k_dur_val, err = _validate_numeric_param(k_dur, "k_dur", allow_zero=True)
    if err:
        reasons.append(err)
        return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)

    alpha_val, err = _validate_numeric_param(alpha_stage, "alpha_stage", allow_zero=False)
    if err:
        reasons.append(err)
        return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)

    exp_val, err = _validate_numeric_param(exp_time_days, "exp_time_days", allow_zero=False)
    if err:
        reasons.append(err)
        return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)

    if max_stage_half_width_days is not None:
        max_budget_val, err = _validate_numeric_param(max_stage_half_width_days, "max_stage_half_width_days", allow_zero=False)
        if err:
            reasons.append(err)
            return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)
    else:
        max_budget_val = None

    if max_cycle_ambiguity_fraction is not None:
        max_cycle_ambig_val, err = _validate_numeric_param(max_cycle_ambiguity_fraction, "max_cycle_ambiguity_fraction", allow_zero=False)
        if err:
            reasons.append(err)
            return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)
    else:
        max_cycle_ambig_val = None

    if max_half_orbit_fraction is not None:
        max_orbit_val, err = _validate_numeric_param(max_half_orbit_fraction, "max_half_orbit_fraction", allow_zero=False)
        if err:
            reasons.append(err)
            return _make_record("INVALID_POLICY_PARAMETERS", "UNCHECKED", False, empty_mask)
    else:
        max_orbit_val = None

    if duration_frame_type not in ALLOWED_DURATION_FRAMES:
        reasons.append(f"duration_frame_type must be one of {ALLOWED_DURATION_FRAMES}, got {duration_frame_type!r}")
        return _make_record("INVALID_DURATION_FRAME_TYPE", "UNCHECKED", False, empty_mask)

    # 3. Validate ephemeris coherence and time scale
    if normalized_time_format != "BTJD" or normalized_time_scale != "TDB":
        reasons.append(f"Requires explicit normalized BTJD format with TDB scale, got normalized_time_format={normalized_time_format!r}, normalized_time_scale={normalized_time_scale!r}")
        return _make_record("UNSAFE_OR_UNKNOWN_TIME_SCALE", "UNSAFE_TIME_SCALE", False, empty_mask)

    if ephemeris_coherence_status != "COHERENT":
        reasons.append(f"ephemeris_coherence_status must be 'COHERENT', got {ephemeris_coherence_status!r}")
        return _make_record("COMPOSITE_OR_UNVERIFIED_EPHEMERIS", "UNVERIFIED_SOURCE", False, empty_mask)

    if original_time_format not in ("BTJD", "BJD") or original_time_scale != "TDB":
        reasons.append(f"Requires original BTJD/BJD format with TDB scale, got format={original_time_format!r}, scale={original_time_scale!r}")
        return _make_record("UNSAFE_OR_UNKNOWN_TIME_SCALE", "UNSAFE_TIME_SCALE", False, empty_mask)

    if has_unmodeled_ttv:
        reasons.append("Ephemeris has unmodeled transit timing variations (TTV)")
        return _make_record("UNMODELED_TTV_UNCERTAINTY", "UNMODELED_TTV", False, empty_mask)

    # 4. Validate base geometry
    p_val, err = _validate_numeric_param(period_days, "period_days", allow_zero=False)
    if err:
        reasons.append(err)
        return _make_record("INVALID_OR_MISSING_GEOMETRY", "INVALID_GEOMETRY", False, empty_mask)

    t0_val, err = _validate_numeric_param(epoch_btjd, "epoch_btjd", allow_zero=True, allow_negative=True)
    if err:
        reasons.append(err)
        return _make_record("INVALID_OR_MISSING_GEOMETRY", "INVALID_GEOMETRY", False, empty_mask)

    dur_val, err = _validate_numeric_param(duration_days, "duration_days", allow_zero=False)
    if err:
        reasons.append(err)
        return _make_record("INVALID_OR_MISSING_GEOMETRY", "INVALID_GEOMETRY", False, empty_mask)

    # Validate duration error
    if sigma_dur_days is None or isinstance(sigma_dur_days, (bool, np.bool_)) or isinstance(sigma_dur_days, (str, bytes)):
        reasons.append(f"sigma_dur_days must be a non-negative real number, got {sigma_dur_days!r}")
        return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_DURATION_ERR", False, empty_mask)
    try:
        sig_d_val = float(sigma_dur_days)
    except (TypeError, ValueError):
        reasons.append(f"sigma_dur_days must be numeric, got {sigma_dur_days!r}")
        return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_DURATION_ERR", False, empty_mask)
    if not math.isfinite(sig_d_val) or sig_d_val < 0.0:
        reasons.append(f"sigma_dur_days must be finite non-negative, got {sig_d_val}")
        return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_DURATION_ERR", False, empty_mask)

    P = p_val
    T0 = t0_val
    D = dur_val
    sig_D = sig_d_val

    n_cycle = np.round((t_arr - T0) / P)
    t_pred = T0 + n_cycle * P
    dt_abs = np.abs(t_arr - t_pred)

    # 5. Timing uncertainty branch
    has_any_cov_input = any(x is not None for x in (var_t0_days2, cov_t0_p_days2, var_p_days2))

    if has_any_cov_input:
        uncertainty_branch = "EXACT_COVARIANCE"
        if var_t0_days2 is None or cov_t0_p_days2 is None or var_p_days2 is None:
            reasons.append("Incomplete 2x2 covariance elements provided")
            return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "INCOMPLETE_COVARIANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if not is_psd_covariance_2x2(var_t0_days2, cov_t0_p_days2, var_p_days2):
            reasons.append("Supplied covariance matrix is not positive semidefinite or has invalid numerical scale")
            return _make_record("INVALID_COVARIANCE_MATRIX", "NON_PSD_COVARIANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        # Require covariance epoch binding
        if covariance_epoch_btjd is None or isinstance(covariance_epoch_btjd, (bool, np.bool_)) or isinstance(covariance_epoch_btjd, (str, bytes)):
            reasons.append("Missing explicit covariance_epoch_btjd for supplied covariance matrix")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_COVARIANCE_EPOCH_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        try:
            t0_cov = float(covariance_epoch_btjd)
        except (TypeError, ValueError):
            reasons.append(f"covariance_epoch_btjd must be a real number, got {covariance_epoch_btjd!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "INVALID_COVARIANCE_EPOCH", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if not math.isfinite(t0_cov):
            reasons.append(f"covariance_epoch_btjd must be finite, got {t0_cov}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "INVALID_COVARIANCE_EPOCH", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        # Check explicit non-blank solution identity and matching covariance solution identity
        sol_id_clean = _clean_str_id(solution_id)
        cov_sol_clean = _clean_str_id(covariance_solution_id)

        if sol_id_clean is None:
            reasons.append("Missing or blank ephemeris solution_id required for verified ephemeris provenance")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_EPHEMERIS_SOLUTION_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if cov_sol_clean is None:
            reasons.append(f"Missing or blank covariance_solution_id required to match ephemeris solution_id {solution_id!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_COVARIANCE_SOLUTION_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if cov_sol_clean != sol_id_clean:
            reasons.append(f"Covariance solution_id {covariance_solution_id!r} does not match ephemeris solution_id {solution_id!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "COVARIANCE_SOLUTION_MISMATCH", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        # Enforce verified operation/ULP-aware integer rebase between epoch_btjd (T0) and covariance_epoch_btjd (t0_cov)
        is_int_cov, round_m_cov, diff_cov, max_tol_cov = check_integer_rebase(T0, t0_cov, P)
        if not is_int_cov:
            m_cov_rebase = (T0 - t0_cov) / P
            reasons.append(f"Non-integer epoch rebase between epoch_btjd ({T0}) and covariance_epoch_btjd ({t0_cov}): delta/P={m_cov_rebase}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "COVARIANCE_EPOCH_MISMATCH", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        # Cycle count relative to the covariance epoch t0_cov
        n_cov = np.round((t_pred - t0_cov) / P)
        raw_var = (
            float(var_t0_days2)
            + 2.0 * n_cov * float(cov_t0_p_days2)
            + (n_cov ** 2) * float(var_p_days2)
        )
        scale_bound = 1e-12 * (float(var_t0_days2) + (n_cov ** 2) * float(var_p_days2))
        if np.any(raw_var < -scale_bound):
            reasons.append("Transformed timing variance is materially negative")
            return _make_record("INVALID_COVARIANCE_MATRIX", "NEGATIVE_PREDICTED_VARIANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        sig_pred = np.sqrt(np.where(raw_var < 0.0, 0.0, raw_var))
        geom_status = "VALID_EXACT_COVARIANCE"

    else:
        # Cauchy-Schwarz bound branch
        uncertainty_branch = "CAUCHY_SCHWARZ_UPPER_BOUND"
        # Strict boolean type requirement for provenance flag: rejects string 'False' or non-bools
        if not (isinstance(marginal_errors_at_published_epoch, bool) and marginal_errors_at_published_epoch is True):
            reasons.append("marginal_errors_at_published_epoch must be strict boolean True")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_PUBLISHED_EPOCH_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if (
            published_epoch_btjd is None
            or isinstance(published_epoch_btjd, (bool, np.bool_))
            or isinstance(published_epoch_btjd, (str, bytes))
        ):
            reasons.append("Missing explicit published_epoch_btjd for Cauchy-Schwarz bound")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_PUBLISHED_EPOCH_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        try:
            t0_pub = float(published_epoch_btjd)
        except (TypeError, ValueError):
            reasons.append(f"published_epoch_btjd must be a real number, got {published_epoch_btjd!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_PUBLISHED_EPOCH_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if not math.isfinite(t0_pub):
            reasons.append(f"published_epoch_btjd must be finite, got {t0_pub}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_PUBLISHED_EPOCH_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        # Check explicit non-blank solution identity and matching marginal errors solution identity
        marg_sol = marginal_errors_solution_id if marginal_errors_solution_id is not None else published_solution_id
        sol_id_clean = _clean_str_id(solution_id)
        marg_sol_clean = _clean_str_id(marg_sol)

        if sol_id_clean is None:
            reasons.append("Missing or blank ephemeris solution_id required for verified ephemeris provenance")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_EPHEMERIS_SOLUTION_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if marg_sol_clean is None:
            reasons.append(f"Missing or blank marginal_errors_solution_id required to match ephemeris solution_id {solution_id!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MISSING_MARGINAL_ERRORS_SOLUTION_PROVENANCE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if marg_sol_clean != sol_id_clean:
            reasons.append(f"Marginal errors solution_id {marg_sol!r} does not match ephemeris solution_id {solution_id!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "MARGINAL_ERRORS_SOLUTION_MISMATCH", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        # Enforce verified operation/ULP-aware integer rebase between epoch_btjd (T0) and published_epoch_btjd (t0_pub)
        is_int_pub, round_m_pub, diff_pub, max_tol_pub = check_integer_rebase(T0, t0_pub, P)
        if not is_int_pub:
            m_rebase = (T0 - t0_pub) / P
            reasons.append(f"Non-integer epoch rebase between epoch_btjd ({T0}) and published_epoch_btjd ({t0_pub}): delta/P={m_rebase}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_PROVENANCE", "NON_INTEGER_EPOCH_REBASE", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if (
            sigma_t0_days is None
            or isinstance(sigma_t0_days, (bool, np.bool_))
            or isinstance(sigma_t0_days, (str, bytes))
        ):
            reasons.append("Missing sigma_t0_days for Cauchy-Schwarz bound")
            return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_EPOCH_ERR", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        try:
            sig_t0 = float(sigma_t0_days)
        except (TypeError, ValueError):
            reasons.append(f"sigma_t0_days must be a real number, got {sigma_t0_days!r}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_EPOCH_ERR", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        if not math.isfinite(sig_t0) or sig_t0 < 0.0:
            reasons.append(f"sigma_t0_days must be finite non-negative, got {sig_t0}")
            return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_EPOCH_ERR", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

        n_pub = np.round((t_pred - t0_pub) / P)
        if np.any(n_pub != 0):
            if (
                sigma_p_days is None
                or isinstance(sigma_p_days, (bool, np.bool_))
                or isinstance(sigma_p_days, (str, bytes))
            ):
                reasons.append("Missing sigma_p_days for cycles away from published epoch")
                return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_PERIOD_ERR", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

            try:
                sig_p_val = float(sigma_p_days)
            except (TypeError, ValueError):
                reasons.append(f"sigma_p_days must be a real number, got {sigma_p_days!r}")
                return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_PERIOD_ERR", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

            if not math.isfinite(sig_p_val) or sig_p_val < 0.0:
                reasons.append(f"sigma_p_days must be finite non-negative, got {sig_p_val}")
                return _make_record("INSUFFICIENT_UNCERTAINTY_METADATA", "MISSING_PERIOD_ERR", False, empty_mask, {"uncertainty_branch": uncertainty_branch})
            sig_P = sig_p_val
        else:
            sig_P = float(sigma_p_days) if (sigma_p_days is not None and not isinstance(sigma_p_days, bool) and not isinstance(sigma_p_days, str) and math.isfinite(sigma_p_days)) else 0.0

        sig_pred = sig_t0 + np.abs(n_pub) * sig_P
        geom_status = "VALID_CAUCHY_SCHWARZ_UPPER_BOUND"

    # 6. Cycle ambiguity fraction, stage width budget checks, and finite overflow guards
    if not np.all(np.isfinite(sig_pred)):
        reasons.append("Derived timing uncertainty vector contains non-finite values (numeric overflow)")
        return _make_record("DERIVED_QUANTITY_OVERFLOW", "DERIVED_NUMERICAL_OVERFLOW", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

    k_sig_pred = k_sig_val * sig_pred
    if not np.all(np.isfinite(k_sig_pred)):
        reasons.append("Scaled timing uncertainty (k_sigma * sigma_pred) overflowed float representation")
        return _make_record("DERIVED_QUANTITY_OVERFLOW", "DERIVED_NUMERICAL_OVERFLOW", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

    exposure_pad = 0.5 * exp_val if duration_frame_type == "INTRINSIC_DECONVOLVED" else 0.0
    half_width_req = 0.5 * alpha_val * (D + k_dur_val * sig_D) + k_sig_pred + exposure_pad
    if not np.all(np.isfinite(half_width_req)):
        reasons.append("Required half-width calculation overflowed float representation")
        return _make_record("DERIVED_QUANTITY_OVERFLOW", "DERIVED_NUMERICAL_OVERFLOW", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

    cycle_ambiguity_frac = float(np.median(k_sig_pred / P))
    half_orbit_frac = float(np.median(2.0 * half_width_req / P))
    med_width_raw = float(np.median(half_width_req))

    if (
        not math.isfinite(cycle_ambiguity_frac)
        or not math.isfinite(half_orbit_frac)
        or not math.isfinite(med_width_raw)
        or med_width_raw > 1e305
    ):
        reasons.append("Derived ambiguity metrics or median width overflowed float representation")
        return _make_record("DERIVED_QUANTITY_OVERFLOW", "DERIVED_NUMERICAL_OVERFLOW", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

    med_width_min = med_width_raw * 1440.0
    if not math.isfinite(med_width_min):
        reasons.append("Derived ambiguity metrics or median width overflowed float representation")
        return _make_record("DERIVED_QUANTITY_OVERFLOW", "DERIVED_NUMERICAL_OVERFLOW", False, empty_mask, {"uncertainty_branch": uncertainty_branch})

    ambiguity_extra = {
        "uncertainty_branch": uncertainty_branch,
        "cycle_ambiguity_fraction": cycle_ambiguity_frac,
        "half_orbit_fraction": half_orbit_frac,
    }

    # Only apply explicit caller-supplied ambiguity thresholds (no unapproved default cutoffs)
    if max_cycle_ambig_val is not None and np.any(k_sig_pred >= max_cycle_ambig_val * P):
        reasons.append(f"Cycle ambiguity fraction {cycle_ambiguity_frac:.4f} exceeded caller threshold {max_cycle_ambig_val}")
        return _make_record("AMBIGUOUS_CYCLE_COUNT_DEGENERATE", geom_status, False, empty_mask, ambiguity_extra)

    if max_orbit_val is not None and np.any(2.0 * half_width_req >= max_orbit_val * P):
        reasons.append(f"Half-orbit fraction {half_orbit_frac:.4f} exceeded caller threshold {max_orbit_val}")
        return _make_record("INSUFFICIENT_TIMING_PRECISION_FOR_STAGE", geom_status, False, empty_mask, ambiguity_extra)

    if max_budget_val is not None and np.any(half_width_req > max_budget_val):
        reasons.append(f"Required half-width ({float(np.max(half_width_req)):.4f} d) exceeded stage budget ({max_budget_val:.4f} d)")
        return _make_record("INSUFFICIENT_TIMING_PRECISION_FOR_STAGE", geom_status, False, empty_mask, ambiguity_extra)

    # 7. Separate valid geometry from stage authorization & disposition policy
    disp_norm = "" if raw_disposition is None else str(raw_disposition).strip().upper()
    if disp_norm in ("FA", "FALSE ALARM"):
        reasons.append(f"Disposition {raw_disposition!r} is a non-astrophysical false alarm")
        return _make_record("SKIPPED_NON_ASTROPHYSICAL_FA", geom_status, False, empty_mask, ambiguity_extra)

    allowed_set = {str(d).strip().upper() for d in allowed_dispositions} if allowed_dispositions is not None else set()
    if stage_policy_action != "ALLOW_ACTIVE_MASK" or disp_norm not in allowed_set:
        if stage_policy_action != "ALLOW_ACTIVE_MASK":
            reasons.append(f"Caller stage_policy_action {stage_policy_action!r} does not grant 'ALLOW_ACTIVE_MASK'")
        if disp_norm not in allowed_set:
            reasons.append(f"Disposition {raw_disposition!r} not in caller allowed_dispositions {sorted(allowed_set)}")
        ambiguity_extra["diagnostic_half_width_median_minutes"] = med_width_min
        return _make_record("DISPOSITION_NOT_PERMITTED_FOR_ACTIVE_MASK", geom_status, False, empty_mask, ambiguity_extra)

    # Valid active mask construction
    mask_arr = dt_abs <= half_width_req
    mask_list = [bool(m) for m in mask_arr]

    # Compute continuous interval fraction of masked observations
    all_intervals = [
        (float(tc - 0.5 * exp_val), float(tc + 0.5 * exp_val))
        for tc in t_arr
    ]
    masked_intervals = [iv for iv, m in zip(all_intervals, mask_list) if m]
    tot_time = measure_interval_union(all_intervals)
    masked_time = measure_interval_union(masked_intervals)
    if tot_time <= 0.0:
        f_mask_time: Optional[float] = None
        reasons.append("Total observation time is zero; time_mask_fraction is undefined")
    else:
        f_mask_time = float(masked_time / tot_time)

    ambiguity_extra.update({
        "required_half_width_median_minutes": med_width_min,
        "time_mask_fraction": f_mask_time,
        "excluded_observation_time_days": float(masked_time),
        "total_observation_time_days": float(tot_time),
    })

    return _make_record(f"ACTIVE_MASK_{geom_status}", geom_status, True, mask_list, ambiguity_extra)


def evaluate_known_signal_geometry_diagnostics(
    t_btjd: Union[Sequence[float], np.ndarray],
    *,
    period_days: Optional[float],
    epoch_btjd: Optional[float],
    duration_days: Optional[float],
    raw_disposition: Optional[str] = None,
    stage_policy_action: Optional[str] = None,
    allowed_dispositions: Optional[Sequence[str]] = None,
    normalized_time_format: Optional[str] = None,
    normalized_time_scale: Optional[str] = None,
    original_time_format: Optional[str] = None,
    original_time_scale: Optional[str] = None,
    ephemeris_coherence_status: Optional[str] = None,
    has_unmodeled_ttv: bool = False,
    var_t0_days2: Optional[float] = None,
    cov_t0_p_days2: Optional[float] = None,
    var_p_days2: Optional[float] = None,
    covariance_epoch_btjd: Optional[float] = None,
    covariance_solution_id: Optional[str] = None,
    sigma_t0_days: Optional[float] = None,
    sigma_p_days: Optional[float] = None,
    sigma_dur_days: Optional[float] = None,
    published_epoch_btjd: Optional[float] = None,
    marginal_errors_at_published_epoch: bool = False,
    marginal_errors_solution_id: Optional[str] = None,
    published_solution_id: Optional[str] = None,
    duration_frame_type: Optional[str] = None,
    exp_time_days: Optional[float] = None,
    k_sigma: Optional[float] = None,
    k_dur: Optional[float] = None,
    alpha_stage: Optional[float] = None,
    max_stage_half_width_days: Optional[float] = None,
    max_cycle_ambiguity_fraction: Optional[float] = None,
    max_half_orbit_fraction: Optional[float] = None,
    signal_id: Optional[str] = None,
    solution_id: Optional[str] = None,
    source_snapshot_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Public entry point for defensive known-signal geometry diagnostics.
    Preserves signal and solution identity metadata, ensures JSON-safe outputs,
    and guarantees mode="DIAGNOSTIC_ONLY" with production_mask_allowed=False.
    """
    return evaluate_signal_mask_contract(
        t_btjd,
        period_days=period_days,
        epoch_btjd=epoch_btjd,
        duration_days=duration_days,
        raw_disposition=raw_disposition,
        stage_policy_action=stage_policy_action,
        allowed_dispositions=allowed_dispositions,
        normalized_time_format=normalized_time_format,
        normalized_time_scale=normalized_time_scale,
        original_time_format=original_time_format,
        original_time_scale=original_time_scale,
        ephemeris_coherence_status=ephemeris_coherence_status,
        has_unmodeled_ttv=has_unmodeled_ttv,
        var_t0_days2=var_t0_days2,
        cov_t0_p_days2=cov_t0_p_days2,
        var_p_days2=var_p_days2,
        covariance_epoch_btjd=covariance_epoch_btjd,
        covariance_solution_id=covariance_solution_id,
        sigma_t0_days=sigma_t0_days,
        sigma_p_days=sigma_p_days,
        sigma_dur_days=sigma_dur_days,
        published_epoch_btjd=published_epoch_btjd,
        marginal_errors_at_published_epoch=marginal_errors_at_published_epoch,
        marginal_errors_solution_id=marginal_errors_solution_id,
        published_solution_id=published_solution_id,
        duration_frame_type=duration_frame_type,
        exp_time_days=exp_time_days,
        k_sigma=k_sigma,
        k_dur=k_dur,
        alpha_stage=alpha_stage,
        max_stage_half_width_days=max_stage_half_width_days,
        max_cycle_ambiguity_fraction=max_cycle_ambiguity_fraction,
        max_half_orbit_fraction=max_half_orbit_fraction,
        signal_id=signal_id,
        solution_id=solution_id,
        source_snapshot_id=source_snapshot_id,
    )
