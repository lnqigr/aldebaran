"""
ephemeris_contract.py — Aldebaran Cross-Tier Shared Ephemeris & Time Consistency Contract
========================================================================================
Implements canonical, versioned time systems, epoch references, phase folding/wrapping,
orbit indexing, and ephemeris conversions across Tier 1 (discovery), Tier 2 (vetting/fit),
Tier 3 (CTOI/Antares packaging), and Tier 4 (Antares detective & TRICERATOPS validation).

Resolves finding ALD-MATH-03-S2: unifies pure time and ephemeris mathematics across
all pipeline tiers, guaranteeing:
  - Exact BTJD/BJD_TDB offset (2457000.0); round trips incur float64 rounding.
  - Consistent phase folding: orbital phase [0, 1), centered phase [-P/2, P/2) in days,
    and centered dimensionless phase [-0.5, 0.5) in cycles.
  - Integer-period shift equivalence within floating-point precision, away from boundaries.
  - Cyclic periodic distance on orbital manifolds (eliminating sign/branch mismatches).
  - Consistent integer orbit numbering and the retained SDE-based timing approximation.
"""

from datetime import datetime, timedelta
import math
from typing import Optional, Tuple, Union

import numpy as np

# ==============================================================================
# SCHEMA & METADATA CONSTANTS
# ==============================================================================
EPHEMERIS_SCHEMA_VERSION = "ephemeris_v1"
BTJD_OFFSET = 2457000.0
BJD_BTJD_OFFSET = 2457000.0  # Backward-compatible alias
TIME_SYSTEM_BTJD = "BTJD"
TIME_STANDARD_TDB = "TDB"

# Standard TESS mission reference epoch (BJD 2457000.0 = 2014-12-08 12:00:00 TDB)
TESS_EPOCH_DATETIME_TDB = datetime(2014, 12, 8, 12, 0, 0)


# ==============================================================================
# TIME SYSTEM CONVERSIONS (BTJD <-> BJD_TDB)
# ==============================================================================

def bjd_to_btjd(
    bjd: Union[float, int, np.ndarray],
    strict: bool = False
) -> Union[float, np.ndarray]:
    """
    Converts Barycentric Julian Date (BJD_TDB) to TESS Barycentric Julian Date (BTJD).
    Definition: BTJD = BJD - 2457000.0
    Offset arithmetic assumes the input is already on the TDB time standard;
    it neither verifies that provenance nor converts UTC/TT to TDB.

    Parameters:
      bjd: Scalar float/int or NumPy array in BJD (or BTJD if auto-detecting).
      strict: If True, unconditionally subtracts 2457000.0.
              If False (default), only subtracts 2457000.0 if value > 2,400,000.0,
              preventing double-conversion of values already in BTJD.

    Returns:
      Converted time in BTJD (same shape as input).
    """
    if isinstance(bjd, np.ndarray):
        arr = np.asarray(bjd, dtype=np.float64)
        if strict:
            return arr - BTJD_OFFSET
        return np.where(arr > 2400000.0, arr - BTJD_OFFSET, arr)

    val = float(bjd)
    if not math.isfinite(val):
        return val
    if strict or val > 2400000.0:
        return val - BTJD_OFFSET
    return val


def btjd_to_bjd(
    btjd: Union[float, int, np.ndarray],
    strict: bool = False
) -> Union[float, np.ndarray]:
    """
    Converts TESS Barycentric Julian Date (BTJD) to Barycentric Julian Date (BJD_TDB).
    Definition: BJD = BTJD + 2457000.0

    Parameters:
      btjd: Scalar float/int or NumPy array in BTJD (or BJD if auto-detecting).
      strict: If True, unconditionally adds 2457000.0.
              If False (default), only adds 2457000.0 if value < 2,400,000.0,
              preventing double-conversion of values already in BJD.

    Returns:
      Converted time in BJD_TDB (same shape as input).
    """
    if isinstance(btjd, np.ndarray):
        arr = np.asarray(btjd, dtype=np.float64)
        if strict:
            return arr + BTJD_OFFSET
        return np.where(arr < 2400000.0, arr + BTJD_OFFSET, arr)

    val = float(btjd)
    if not math.isfinite(val):
        return val
    if strict or val < 2400000.0:
        return val + BTJD_OFFSET
    return val


def btjd_to_year(btjd_val: Union[float, int]) -> int:
    """
    Converts TESS Barycentric Julian Date (BTJD = BJD - 2457000.0) to calendar year.
    Base date: 2014-12-08 12:00:00 (BTJD 0.0).
    """
    target_dt = TESS_EPOCH_DATETIME_TDB + timedelta(days=float(btjd_val))
    return int(target_dt.year)


# ==============================================================================
# ORBITAL PHASE FOLDING & WRAPPING
# ==============================================================================

def fold_phase_0to1(
    time: Union[float, int, np.ndarray],
    t0: float,
    period: float
) -> Union[float, np.ndarray]:
    """
    Folds time onto the orbital interval [0, 1) cycles from mid-transit (t0).
    Formula: phi = ((time - t0) % period) / period

    Invariant:
      Equivalent under integer-period shifts in exact arithmetic. Floating-point
      rounding can affect phase/contact boundaries; no bitwise invariance is promised.
      Values are strictly in [0.0, 1.0).
    """
    if period <= 0 or not math.isfinite(period):
        raise ValueError(f"Period must be strictly positive and finite: {period}")

    if isinstance(time, np.ndarray):
        arr = np.asarray(time, dtype=np.float64)
        ph = ((arr - t0) % period) / period
        # Clamp exact 1.0 floating edge to 0.0 to guarantee [0.0, 1.0)
        return np.where(ph >= 1.0, 0.0, ph)

    t_val = float(time)
    if not math.isfinite(t_val):
        return t_val
    ph = ((t_val - t0) % period) / period
    return 0.0 if ph >= 1.0 else ph


def fold_phase_centered(
    time: Union[float, int, np.ndarray],
    t0: float,
    period: float,
    in_days: bool = True
) -> Union[float, np.ndarray]:
    """
    Folds time onto the centered interval around mid-transit (t0 = 0).

    When in_days=True (default):
      tau in [-period/2, period/2) in days (time offset from nearest transit center).
      Formula: tau = ((time - t0 + 0.5 * period) % period) - 0.5 * period
      Used across Tier 1, 2, 3, 4 for transit masking, fitting, and visualization.

    When in_days=False:
      phi_c in [-0.5, 0.5) in orbital cycles.
      Formula: phi_c = ((time - t0) / period + 0.5) % 1.0 - 0.5
      The operation order preserves legacy dimensionless consumers. It is not
      bitwise interchangeable with folding in days and dividing by period.
      Used for normalized box models and dimensionless phase comparisons.

    Invariant:
      Equivalent under integer-period shifts in exact arithmetic. Floating-point
      rounding can affect phase/contact boundaries; no bitwise invariance is promised.
    """
    if period <= 0 or not math.isfinite(period):
        raise ValueError(f"Period must be strictly positive and finite: {period}")

    half_p = 0.5 * period

    if isinstance(time, np.ndarray):
        arr = np.asarray(time, dtype=np.float64)
        if in_days:
            return ((arr - t0 + half_p) % period) - half_p
        return ((arr - t0) / period + 0.5) % 1.0 - 0.5

    t_val = float(time)
    if not math.isfinite(t_val):
        return t_val
    if in_days:
        return ((t_val - t0 + half_p) % period) - half_p
    return ((t_val - t0) / period + 0.5) % 1.0 - 0.5


def cyclic_orbital_distance(
    phase1: Union[float, int, np.ndarray],
    phase2: float,
    period: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Calculates the minimal signed distance between two phases on the periodic orbital circle.

    Parameters:
      phase1: Scalar or array of phases.
      phase2: Reference phase scalar.
      period: If None, phase1 and phase2 are assumed to be in orbital cycles [0, 1) or [-0.5, 0.5),
              and the returned distance is in [-0.5, 0.5) cycles.
              If period is given (in days), phase1 and phase2 are assumed to be in days,
              and the returned distance is in [-period/2, period/2) days.

    Formula (cycles): delta_phi = ((phase1 - phase2 + 0.5) % 1.0) - 0.5
    Formula (days):   delta_t   = ((phase1 - phase2 + 0.5 * period) % period) - 0.5 * period

    Resolves SCI01 finding: correctly connects signed or unsigned phase coordinates
    to control centers anywhere on the orbit without branch discontinuity.
    """
    if period is None:
        if isinstance(phase1, np.ndarray):
            arr = np.asarray(phase1, dtype=np.float64)
            return ((arr - phase2 + 0.5) % 1.0) - 0.5
        val = float(phase1)
        if not math.isfinite(val):
            return val
        return ((val - phase2 + 0.5) % 1.0) - 0.5

    if period <= 0 or not math.isfinite(period):
        raise ValueError(f"Period must be strictly positive: {period}")

    half_p = 0.5 * period
    if isinstance(phase1, np.ndarray):
        arr = np.asarray(phase1, dtype=np.float64)
        return ((arr - phase2 + half_p) % period) - half_p
    val = float(phase1)
    if not math.isfinite(val):
        return val
    return ((val - phase2 + half_p) % period) - half_p


# ==============================================================================
# TRANSIT EPOCH PROJECTION & ORBIT NUMBERING
# ==============================================================================

def compute_orbit_number(
    time: Union[float, int, np.ndarray],
    t0: float,
    period: float
) -> Union[int, np.ndarray]:
    """
    Computes the nearest integer orbit index N for each cadence timestamp:
      N = round((time - t0) / period)

    Used for odd/even transit splitting, transit multiplicity counting,
    and individual transit consistency checks across all tiers.
    """
    if period <= 0 or not math.isfinite(period):
        raise ValueError(f"Period must be strictly positive: {period}")

    if isinstance(time, np.ndarray):
        arr = np.asarray(time, dtype=np.float64)
        return np.round((arr - t0) / period).astype(int)

    val = float(time)
    if not math.isfinite(val):
        raise ValueError(f"Cannot compute orbit number for non-finite time: {val}")
    return int(round((val - t0) / period))


def predict_transit_time(
    t0: float,
    period: float,
    cycle_number: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Predicts transit mid-time for integer orbit cycle N:
      T_N = t0 + N * period
    """
    if period <= 0 or not math.isfinite(period):
        raise ValueError(f"Period must be strictly positive: {period}")

    if isinstance(cycle_number, np.ndarray):
        return t0 + np.asarray(cycle_number, dtype=np.float64) * period
    return float(t0 + int(cycle_number) * period)


def normalize_epoch_to_baseline(
    t0: float,
    period: float,
    reference_time: float
) -> float:
    """
    Shifts epoch t0 by integer multiples of period to the cycle closest to reference_time.
    Formula:
      N = round((reference_time - t0) / period)
      t0_norm = t0 + N * period

    Invariant:
      The predicted transit sequence {t0_norm + k * period} is identical to
      {t0 + k * period} for all k in Z.
      abs(t0_norm - reference_time) <= 0.5 * period.
    """
    if period <= 0 or not math.isfinite(period):
        raise ValueError(f"Period must be strictly positive: {period}")

    shift_cycles = int(round((reference_time - t0) / period))
    return float(t0 + shift_cycles * period)


def calculate_cyclic_timing_residual(
    cand_p: float,
    cand_t0: float,
    cat_p: float,
    cat_t0: float,
    pivot_time: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Calculates physical timing difference between candidate and catalog solutions,
    equivalent under integer-cycle epoch shifts in exact arithmetic; float64 rounding remains.

    Methodology:
      1. Uses pivot_time as a common fixed time anchor (if None, defaults to cand_t0).
      2. Projects candidate ephemeris to nearest cycle around pivot_time:
           N_cand = round((pivot_time - cand_t0) / cand_p)
           T_cand_proj = cand_t0 + N_cand * cand_p
      3. Projects catalog ephemeris to nearest cycle around T_cand_proj:
           N_cat = round((T_cand_proj - cat_t0) / cat_p)
           T_cat_proj = cat_t0 + N_cat * cat_p
      4. Timing residual dt = T_cand_proj - T_cat_proj.
      5. Centered cyclic residual:
           dt_cyclic = dt - round(dt / cat_p) * cat_p
      6. Phase offset in [-0.5, 0.5]:
           phase = dt_cyclic / cat_p

    Returns:
      (dt_cyclic_days, dt_minutes, phase_offset)
    """
    if cand_p <= 0 or cat_p <= 0:
        raise ValueError(f"Periods must be strictly positive: cand_p={cand_p}, cat_p={cat_p}")

    anchor = cand_t0 if pivot_time is None else float(pivot_time)

    # Transport candidate to anchor
    n_cand = round((anchor - cand_t0) / cand_p)
    t_cand_proj = cand_t0 + n_cand * cand_p

    # Transport catalog to nearest cycle at candidate predicted time
    n_cat = round((t_cand_proj - cat_t0) / cat_p)
    t_cat_proj = cat_t0 + n_cat * cat_p

    dt_days = t_cand_proj - t_cat_proj
    # Ensure cyclic centering within [-P_cat / 2, +P_cat / 2]
    dt_cyclic_days = dt_days - round(dt_days / cat_p) * cat_p
    dt_minutes = dt_cyclic_days * 1440.0
    phase_offset = dt_cyclic_days / cat_p

    return float(dt_cyclic_days), float(dt_minutes), float(phase_offset)


# ==============================================================================
# DURATION & TIMING UNCERTAINTY CONVERSIONS
# ==============================================================================

def days_to_hours(days: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert numeric durations without changing dtype or multiplication order."""
    if isinstance(days, (np.ndarray, np.generic)):
        return days * 24.0
    return float(days) * 24.0


def hours_to_days(hours: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert numeric durations without changing dtype or division order."""
    if isinstance(hours, (np.ndarray, np.generic)):
        return hours / 24.0
    return float(hours) / 24.0


def cramer_rao_timing_uncertainty(
    duration_days: float,
    sde: float,
    baseline_days: float,
    period_days: float
) -> Tuple[float, float]:
    """
    Preserves the existing SDE-based timing-uncertainty approximation:
      sigma_T0 = (0.5 * duration) / max(SDE, 5.0)
      N_cyc = max(1.0, baseline / period)
      sigma_P = sqrt(12.0) * sigma_T0 / N_cyc

    SDE is used as a heuristic signal-strength proxy. This arithmetic is not an
    independently validated Fisher-information bound or uncertainty coverage model.
    The legacy public function name is retained for compatibility.

    Returns:
      (sigma_t0_days, sigma_p_days)
    """
    if duration_days <= 0 or period_days <= 0 or baseline_days <= 0:
        raise ValueError("Duration, period, and baseline must be strictly positive")

    sigma_t0 = (0.5 * duration_days) / max(float(sde), 5.0)
    n_cyc = max(1.0, float(baseline_days) / float(period_days))
    sigma_p = math.sqrt(12.0) * sigma_t0 / n_cyc
    return float(sigma_t0), float(sigma_p)
