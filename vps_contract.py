"""
vps_contract.py — Aldebaran / Pleiades / Antares Shared VPS Contract & Ledger Module
====================================================================================
Implements versioned Vetting Penalty Score (VPS) component accounting, immutable
provenance tracking, full-precision serialization, and authoritative boundary validation
between Pleiades (producer) and Antares Stage 3A (consumer).

Resolves finding SCI-16: eliminates legacy unsupported secondary-penalty subtraction
in Antares while enforcing strict mathematical reconstruction of all score penalties.
"""

from collections import OrderedDict
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Schema and Versioning Constants
VPS_SCHEMA_VERSION = "vps_v1"
VPS_PRODUCER_NAME = "pleiades"
VPS_BASE_PENALTY = 0.2
VPS_CLIP_MIN = 0.1
VPS_CLIP_MAX = 99.9
VPS_ROUNDING_DECIMALS = 1
VPS_RECONSTRUCTION_TOLERANCE = 1e-4

# Pleiades Clean Score Criterion
PLEIADES_CLEAN_VPS_THRESHOLD = 5.0
PLEIADES_CLEAN_CRITERION_LABEL = f"vps_score < {PLEIADES_CLEAN_VPS_THRESHOLD:.1f}"

# Antares Default Stage 3A Risk Filter Threshold
ANTARES_DEFAULT_MAX_CORR_RISK = 25.0

# Secondary Eclipse Detection Zone & Significance Parameters
SEC_CIRCULAR_HALF_WIDTH = 0.04
SEC_CIRCULAR_CENTER = 0.50
SEC_CIRCULAR_THRESHOLD_SNR = 2.8
SEC_ECCENTRIC_THRESHOLD_SNR = 3.4
SEC_PENALTY_SCALE = 45.0
SEC_PENALTY_SNR_DIVISOR = 3.0

# Mandatory Component Ledger Keys in Explicit Evaluation Order
MANDATORY_COMPONENT_KEYS = (
    "base",
    "impact_parameter",
    "secondary_significance",
    "secondary_eb",
    "catalog_tess_eb",
    "gaia_nss",
    "neb_neighbors",
    "contamination",
    "duration_ratio",
    "ruwe",
    "weak_primary"
)


class VPSContractError(ValueError):
    """
    Raised when a VPS score record fails schema versioning, component ledger
    integrity, or mathematical reconstruction validation.
    """
    pass


CTOI_SECONDARY_THRESHOLD_SNR = 3.5

def secondary_evidence(sec_center_phi: float, sec_snr: float) -> Dict[str, Any]:
    """Full-precision VPS predicate, phase in cycles and SNR in sigma.

    Retains production binary64 zone arithmetic, strict comparisons and penalties.
    A heuristic evidence predicate, not an astrophysical probability.
    """
    if sec_snr is None or isinstance(sec_snr, bool) or not math.isfinite(sec_snr) or sec_snr < 0:
        raise VPSContractError("Secondary SNR must be finite and nonnegative")
    if sec_center_phi is None or isinstance(sec_center_phi, bool) or not math.isfinite(sec_center_phi) or not 0 <= sec_center_phi <= 1:
        raise VPSContractError("Secondary phase must be finite in [0,1]")
    sec_phase = float(sec_center_phi)
    is_circular_sec = abs(sec_phase - SEC_CIRCULAR_CENTER) <= SEC_CIRCULAR_HALF_WIDTH
    sec_snr_threshold = SEC_CIRCULAR_THRESHOLD_SNR if is_circular_sec else SEC_ECCENTRIC_THRESHOLD_SNR

    sec_snr_val = float(sec_snr)
    sec_penalty_applied = sec_snr_val > sec_snr_threshold
    if sec_penalty_applied:
        sec_penalty = float(SEC_PENALTY_SCALE * (sec_snr_val / SEC_PENALTY_SNR_DIVISOR))
    else:
        sec_penalty = 0.0
    # R1: Store full-precision floats in secondary_metadata so validator does not diverge from producer
    secondary_metadata = {
        "phase": float(sec_phase),
        "snr": float(sec_snr_val),
        "is_circular": bool(is_circular_sec),
        "zone": "circular" if is_circular_sec else "eccentric",
        "threshold": float(sec_snr_threshold),
        "penalty_applied": bool(sec_penalty_applied),
        "penalty_amount": float(sec_penalty)
    }

    return secondary_metadata


def compute_vps_ledger(
    b_fit: float,
    sec_center_phi: float,
    sec_snr: float,
    is_sec_eb: bool,
    is_tess_eb: bool,
    is_gaia_nss: bool,
    neb_capable_stars: int,
    contamination_ratio: float,
    dur_ratio_kepler: float,
    ruwe: float,
    stitched_primary_snr: float
) -> Dict[str, Any]:
    """
    Computes all VPS component penalties in exact Pleiades production order,
    building an unclipped ledger, clipped score, and auditing metadata.

    Parameters:
        b_fit: Fitted transit impact parameter (b >= 0.0). Mandatory finite measurement.
        sec_center_phi: Best secondary eclipse phase in [0.0, 1.0]. Mandatory finite measurement.
        sec_snr: Secondary eclipse peak detection SNR (sigma >= 0.0). Mandatory finite measurement.
        is_sec_eb: Boolean flag indicating eccentric secondary eclipse geometry.
        is_tess_eb: Boolean flag indicating cataloged TESS eclipsing binary match.
        is_gaia_nss: Boolean flag indicating Gaia non-single star (NSS) two-body solution.
        neb_capable_stars: Count of nearby Gaia neighbors capable of causing NEB (optional catalog match).
        contamination_ratio: Gaia flux contamination ratio in target aperture (optional catalog match).
        dur_ratio_kepler: Ratio of transit duration to nominal Keplerian duration. Mandatory finite measurement.
        ruwe: Gaia DR3 Renormalised Unit Weight Error (optional catalog match; NaN means unknown).
        stitched_primary_snr: Detection SNR of phase-folded stitched primary transit. Mandatory finite measurement.

    Returns:
        Dict containing full ledger:
            - version: "vps_v1"
            - producer: "pleiades"
            - components: OrderedDict of all 11 component penalties (float)
            - secondary_metadata: Dict with full-precision secondary phase, SNR, zone, threshold, penalty details
            - unclipped_total: Unclipped sum of components
            - clip_bounds: [0.1, 99.9]
            - clipped_score: Score clipped to [0.1, 99.9] (unrounded float)
            - final_score: Rounded to 1 decimal place
            - clean_criterion: "vps_score < 5.0"
    """
    # Strict validation of mandatory measurement inputs (R2: no silent physical defaults for missing measurements)
    if b_fit is None or not math.isfinite(b_fit) or b_fit < 0.0:
        raise VPSContractError(f"Mandatory measurement 'b_fit' must be a non-negative finite float, got {b_fit}")
    if dur_ratio_kepler is None or not math.isfinite(dur_ratio_kepler) or dur_ratio_kepler <= 0.0:
        raise VPSContractError(f"Mandatory measurement 'dur_ratio_kepler' must be a positive finite float, got {dur_ratio_kepler}")
    if stitched_primary_snr is None or not math.isfinite(stitched_primary_snr):
        raise VPSContractError(f"Mandatory measurement 'stitched_primary_snr' must be a finite float, got {stitched_primary_snr}")
    if sec_snr is None or not math.isfinite(sec_snr) or sec_snr < 0.0:
        raise VPSContractError(f"Mandatory measurement 'sec_snr' must be a non-negative finite float, got {sec_snr}")
    if sec_center_phi is None or not math.isfinite(sec_center_phi) or not (0.0 <= sec_center_phi <= 1.0):
        raise VPSContractError(f"Mandatory measurement 'sec_center_phi' must be a finite float in [0.0, 1.0], got {sec_center_phi}")

    components: Dict[str, float] = OrderedDict()

    # 1. Base score (always 0.2)
    components["base"] = float(VPS_BASE_PENALTY)

    # 2. Impact parameter penalty (b > 0.75)
    if b_fit > 0.75:
        components["impact_parameter"] = float(25.0 * ((b_fit - 0.75) / 0.25) ** 2)
    else:
        components["impact_parameter"] = 0.0

    # 3. Secondary eclipse significance penalty (Šidák-Bonferroni dual-zone threshold)
    # Circular zone (|phase - 0.50| <= 0.04): SNR > 2.8 sigma
    # Eccentric zone (|phase - 0.50| > 0.04): SNR > 3.4 sigma
    secondary_metadata = secondary_evidence(sec_center_phi, sec_snr)
    sec_penalty = secondary_metadata["penalty_amount"]
    components["secondary_significance"] = sec_penalty

    # 4. Secondary EB flag
    components["secondary_eb"] = 65.0 if bool(is_sec_eb) else 0.0

    # 5. Cataloged TESS EB match
    components["catalog_tess_eb"] = 90.0 if bool(is_tess_eb) else 0.0

    # 6. Gaia NSS two-body architecture
    components["gaia_nss"] = 80.0 if bool(is_gaia_nss) else 0.0

    # 7. Nearby NEB-capable neighbor stars (optional Gaia catalog count; 8.0 pts per neighbor)
    neb_count = int(neb_capable_stars) if (neb_capable_stars is not None and math.isfinite(neb_capable_stars) and neb_capable_stars > 0) else 0
    if neb_count > 0:
        components["neb_neighbors"] = float(8.0 * neb_count)
    else:
        components["neb_neighbors"] = 0.0

    # 8. Aperture flux contamination (optional Gaia catalog ratio; > 25%)
    contam = float(contamination_ratio) if (contamination_ratio is not None and math.isfinite(contamination_ratio)) else 0.0
    if contam > 0.25:
        components["contamination"] = float(35.0 * contam)
    else:
        components["contamination"] = 0.0

    # 9. Kinematic transit duration self-consistency
    dur_ratio = float(dur_ratio_kepler)
    if dur_ratio > 1.38:
        components["duration_ratio"] = float(40.0 * (dur_ratio - 1.0))
    elif dur_ratio < 0.35:
        components["duration_ratio"] = float(85.0 * ((0.35 - dur_ratio) / 0.35))
    else:
        components["duration_ratio"] = 0.0

    # 10. Gaia RUWE (> 1.25, documented optional-unknown semantics: if NaN/unavailable, penalty is 0.0)
    ruwe_val = float(ruwe) if (ruwe is not None and math.isfinite(ruwe)) else 1.0
    if ruwe_val > 1.25:
        components["ruwe"] = 15.0
    else:
        components["ruwe"] = 0.0

    # 11. Stitched primary transit SNR gate (< 5.5 sigma)
    pri_snr = float(stitched_primary_snr)
    if pri_snr < 5.5:
        components["weak_primary"] = float(min(65.0, max(0.0, (5.5 - pri_snr) * 22.0)))
    else:
        components["weak_primary"] = 0.0

    # Unclipped total
    unclipped_total = float(sum(components.values()))

    # Clipped score to [0.1, 99.9]
    clipped_score = float(np.clip(unclipped_total, VPS_CLIP_MIN, VPS_CLIP_MAX))

    # Serialized rounded score (1 decimal place)
    final_score = round(clipped_score, VPS_ROUNDING_DECIMALS)


    return {
        "version": VPS_SCHEMA_VERSION,
        "producer": VPS_PRODUCER_NAME,
        "components": components,
        "secondary_metadata": secondary_metadata,
        "unclipped_total": unclipped_total,
        "clip_bounds": [VPS_CLIP_MIN, VPS_CLIP_MAX],
        "clipped_score": clipped_score,
        "final_score": final_score,
        "clean_criterion": PLEIADES_CLEAN_CRITERION_LABEL
    }


def validate_and_extract_vps_score(row: Union[pd.Series, Mapping[str, Any]]) -> float:
    """
    Validates score provenance, schema version, component ledger integrity,
    and mathematical reconstruction from a Tier-2 Pleiades dossier row.

    Returns:
        The authoritative, uncorrected score (vps_score) as a float.

    Raises:
        VPSContractError: If the record lacks versioned provenance, contains
            inconsistent aliases, has an unparseable or altered ledger, or represents
            an operational failure.
    """
    tic_id = str(row.get("tic_id", "UNKNOWN")).strip()

    # 1. Operational error check: operational errors cannot masquerade as valid scores (R3: explicit successful status required)
    raw_op = row.get("operational_status")
    if raw_op is None or pd.isna(raw_op) or not str(raw_op).strip():
        raise VPSContractError(f"Target TIC {tic_id} missing required 'operational_status'.")
    op_status = str(raw_op).strip().upper()
    if op_status not in ("VETTED", "TIER-2", "PASS", "CLEAN", "SUCCESS"):
        raise VPSContractError(
            f"Target TIC {tic_id} has non-successful operational status '{op_status}'. "
            f"Unavailable/error scores cannot enter Stage 3A filtering."
        )

    raw_err = row.get("error_message")
    if raw_err is not None and not pd.isna(raw_err):
        err_msg = str(raw_err).strip()
        if err_msg and err_msg.upper() not in ("NAN", "NONE", ""):
            raise VPSContractError(
                f"Target TIC {tic_id} has non-empty error_message: '{err_msg}'. "
                f"Unavailable scores cannot enter Stage 3A filtering."
            )

    # 2. Schema version check: must be explicitly present and equal to VPS_SCHEMA_VERSION
    raw_version = row.get("vps_version")
    if raw_version is None or pd.isna(raw_version) or not str(raw_version).strip():
        raise VPSContractError(
            f"Target TIC {tic_id} missing required 'vps_version'. "
            f"Historical/legacy records without versioned ledger require recomputation."
        )
    version_str = str(raw_version).strip()
    if version_str != VPS_SCHEMA_VERSION:
        raise VPSContractError(
            f"Target TIC {tic_id} has unsupported VPS version '{version_str}'. "
            f"Expected '{VPS_SCHEMA_VERSION}'."
        )

    # 3. Producer check: must be recognized producer matching VPS_PRODUCER_NAME (R3: reject unrecognized)
    raw_producer = row.get("vps_producer")
    if raw_producer is None or pd.isna(raw_producer) or not str(raw_producer).strip():
        raise VPSContractError(f"Target TIC {tic_id} missing required 'vps_producer'.")
    prod_str = str(raw_producer).strip()
    if prod_str != VPS_PRODUCER_NAME:
        raise VPSContractError(
            f"Target TIC {tic_id} has unrecognized producer '{prod_str}'. Expected '{VPS_PRODUCER_NAME}'."
        )

    # 4. Score availability & numeric validity
    raw_score = row.get("vps_score")
    if raw_score is None or pd.isna(raw_score):
        raise VPSContractError(f"Target TIC {tic_id} has missing 'vps_score'.")
    try:
        vps_score = float(raw_score)
        if not math.isfinite(vps_score):
            raise ValueError(f"Non-finite vps_score: {vps_score}")
    except (TypeError, ValueError) as exc:
        raise VPSContractError(f"Target TIC {tic_id} has invalid non-numeric 'vps_score': {exc}")

    # R3: Enforce that exported score adheres to declared 1-decimal rounding identity
    if abs(vps_score - round(vps_score, VPS_ROUNDING_DECIMALS)) > 1e-6:
        raise VPSContractError(
            f"Target TIC {tic_id} vps_score ({vps_score}) is not rounded to {VPS_ROUNDING_DECIMALS} decimal places."
        )

    # 5. Consistency check between vps_score and fp_risk_percent alias (R3: reject inf/nan and enforce strict identity)
    raw_alias = row.get("fp_risk_percent")
    if raw_alias is not None and not pd.isna(raw_alias) and str(raw_alias).strip() != "":
        try:
            alias_val = float(raw_alias)
            if not math.isfinite(alias_val):
                raise ValueError(f"Non-finite alias: {alias_val}")
        except (TypeError, ValueError):
            raise VPSContractError(f"Target TIC {tic_id} has unparseable alias 'fp_risk_percent': '{raw_alias}'.")
        if abs(vps_score - alias_val) > 1e-6:
            raise VPSContractError(
                f"Target TIC {tic_id} has conflicting score aliases: "
                f"vps_score={vps_score} vs fp_risk_percent={alias_val} (diff={abs(vps_score - alias_val)})."
            )

    # 6. Unclipped total availability & numeric validity
    raw_unclipped = row.get("vps_unclipped_total")
    if raw_unclipped is None or pd.isna(raw_unclipped):
        raise VPSContractError(f"Target TIC {tic_id} missing required 'vps_unclipped_total'.")
    try:
        unclipped_total = float(raw_unclipped)
        if not math.isfinite(unclipped_total):
            raise ValueError(f"Non-finite vps_unclipped_total: {unclipped_total}")
    except (TypeError, ValueError) as exc:
        raise VPSContractError(f"Target TIC {tic_id} has invalid 'vps_unclipped_total': {exc}")

    # 7. Component ledger parsing & validation
    raw_ledger = row.get("vps_ledger_json")
    if raw_ledger is None or pd.isna(raw_ledger) or not str(raw_ledger).strip():
        raise VPSContractError(f"Target TIC {tic_id} missing required 'vps_ledger_json'.")

    if isinstance(raw_ledger, dict):
        ledger = raw_ledger
    else:
        try:
            ledger = json.loads(str(raw_ledger))
        except Exception as exc:
            raise VPSContractError(f"Target TIC {tic_id} has malformed 'vps_ledger_json': {exc}")

    if not isinstance(ledger, dict):
        raise VPSContractError(f"Target TIC {tic_id} ledger must be a JSON object.")

    if ledger.get("version") != VPS_SCHEMA_VERSION:
        raise VPSContractError(
            f"Target TIC {tic_id} ledger version mismatch: expected '{VPS_SCHEMA_VERSION}', got '{ledger.get('version')}'."
        )

    if ledger.get("producer") != VPS_PRODUCER_NAME:
        raise VPSContractError(
            f"Target TIC {tic_id} ledger producer mismatch: expected '{VPS_PRODUCER_NAME}', got '{ledger.get('producer')}'."
        )

    # R3: Verify agreement between embedded ledger fields and top-level row fields
    if "unclipped_total" in ledger:
        led_unclipped = float(ledger["unclipped_total"])
        if not math.isfinite(led_unclipped) or abs(led_unclipped - unclipped_total) > 1e-6:
            raise VPSContractError(
                f"Target TIC {tic_id} ledger unclipped_total ({ledger.get('unclipped_total')}) "
                f"contradicts top-level vps_unclipped_total ({unclipped_total})."
            )

    if "final_score" in ledger:
        led_final = float(ledger["final_score"])
        if not math.isfinite(led_final) or abs(led_final - vps_score) > 1e-6:
            raise VPSContractError(
                f"Target TIC {tic_id} ledger final_score ({ledger.get('final_score')}) "
                f"contradicts top-level vps_score ({vps_score})."
            )

    components = ledger.get("components")
    if not isinstance(components, dict):
        raise VPSContractError(f"Target TIC {tic_id} ledger missing 'components' dictionary.")

    # Validate all mandatory component keys and compute sum
    computed_sum = 0.0
    for key in MANDATORY_COMPONENT_KEYS:
        if key not in components:
            raise VPSContractError(f"Target TIC {tic_id} ledger missing mandatory component '{key}'.")
        val = components[key]
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise VPSContractError(f"Target TIC {tic_id} component '{key}' is non-finite or non-numeric: {val}")
        if val < 0.0:
            raise VPSContractError(f"Target TIC {tic_id} component '{key}' is negative: {val}")
        computed_sum += float(val)

    # 8. Reconstruct unclipped total
    if abs(computed_sum - unclipped_total) > VPS_RECONSTRUCTION_TOLERANCE:
        raise VPSContractError(
            f"Target TIC {tic_id} ledger component sum ({computed_sum:.6f}) differs from "
            f"vps_unclipped_total ({unclipped_total:.6f}) by {abs(computed_sum - unclipped_total):.6e} "
            f"(tolerance: {VPS_RECONSTRUCTION_TOLERANCE})."
        )

    # 9. Reconstruct clipped & rounded score (R3: tight numerical tolerance)
    expected_clipped = min(VPS_CLIP_MAX, max(VPS_CLIP_MIN, computed_sum))
    if "clipped_score" in ledger:
        led_clipped = float(ledger["clipped_score"])
        if not math.isfinite(led_clipped) or abs(led_clipped - expected_clipped) > 1e-6:
            raise VPSContractError(
                f"Target TIC {tic_id} ledger clipped_score ({ledger.get('clipped_score')}) "
                f"does not match expected clipped value ({expected_clipped:.6f})."
            )

    expected_score = round(expected_clipped, VPS_ROUNDING_DECIMALS)
    if abs(vps_score - expected_score) > 1e-6:
        raise VPSContractError(
            f"Target TIC {tic_id} vps_score ({vps_score}) does not match reconstructed "
            f"score ({expected_score}) from ledger component sum {computed_sum:.6f}."
        )

    # 10. Validate secondary eclipse audit metadata consistency (R1: full-precision; R3: finite types and booleans)
    sec_meta = ledger.get("secondary_metadata")
    if not isinstance(sec_meta, dict):
        raise VPSContractError(f"Target TIC {tic_id} ledger missing 'secondary_metadata' object.")

    # Validate finite types
    for num_key in ("phase", "snr", "threshold", "penalty_amount"):
        v = sec_meta.get(num_key)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise VPSContractError(f"Target TIC {tic_id} secondary metadata '{num_key}' is non-finite or non-numeric: {v}")
        if v < 0.0 and num_key != "phase":
            raise VPSContractError(f"Target TIC {tic_id} secondary metadata '{num_key}' cannot be negative: {v}")

    for bool_key in ("is_circular", "penalty_applied"):
        v = sec_meta.get(bool_key)
        if not isinstance(v, bool):
            raise VPSContractError(f"Target TIC {tic_id} secondary metadata '{bool_key}' must be boolean, got {type(v)}: {v}")

    sec_phase = float(sec_meta["phase"])
    if not (0.0 <= sec_phase <= 1.0):
        raise VPSContractError(f"Target TIC {tic_id} secondary phase must be in [0.0, 1.0], got {sec_phase}")

    expected_secondary = secondary_evidence(sec_phase, float(sec_meta["snr"]))
    is_circ = expected_secondary["is_circular"]
    if sec_meta["is_circular"] != is_circ:
        raise VPSContractError(
            f"Target TIC {tic_id} secondary metadata 'is_circular' contradiction: expected {is_circ}, got {sec_meta['is_circular']}."
        )

    expected_zone = "circular" if is_circ else "eccentric"
    actual_zone = sec_meta.get("zone")
    if actual_zone != expected_zone:
        raise VPSContractError(
            f"Target TIC {tic_id} secondary zone inconsistency: expected '{expected_zone}', got '{actual_zone}'."
        )

    expected_threshold = expected_secondary["threshold"]
    actual_threshold = float(sec_meta["threshold"])
    if abs(actual_threshold - expected_threshold) > 1e-6:
        raise VPSContractError(
            f"Target TIC {tic_id} secondary threshold inconsistency: expected {expected_threshold}, got {actual_threshold}."
        )

    sec_snr = float(sec_meta["snr"])
    expected_applied = expected_secondary["penalty_applied"]
    actual_applied = bool(sec_meta["penalty_applied"])
    if expected_applied != actual_applied:
        raise VPSContractError(
            f"Target TIC {tic_id} secondary penalty applied flag contradiction: "
            f"expected {expected_applied} for SNR={sec_snr} vs threshold={expected_threshold}, got {actual_applied}."
        )

    expected_penalty_amt = expected_secondary["penalty_amount"]
    actual_penalty_amt = float(sec_meta["penalty_amount"])
    if abs(expected_penalty_amt - actual_penalty_amt) > VPS_RECONSTRUCTION_TOLERANCE:
        raise VPSContractError(
            f"Target TIC {tic_id} secondary penalty amount inconsistency: "
            f"expected {expected_penalty_amt:.6f}, got {actual_penalty_amt:.6f}."
        )

    comp_sec_penalty = float(components["secondary_significance"])
    if abs(comp_sec_penalty - actual_penalty_amt) > VPS_RECONSTRUCTION_TOLERANCE:
        raise VPSContractError(
            f"Target TIC {tic_id} component 'secondary_significance' ({comp_sec_penalty:.6f}) differs from "
            f"secondary metadata penalty amount ({actual_penalty_amt:.6f})."
        )

    # 11. Authoritative score return: NO secondary subtraction
    return vps_score


def resolve_secondary_evidence(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical validated ledger wins over display columns; legacy stays explicit.

    Invalid supplied ledgers fail closed. Missing legacy SNR is unknown, not
    absence. CTOI >3.5 and VPS phase-dependent thresholds are distinct policies.
    """
    raw = row.get("vps_ledger_json")
    supplied = isinstance(raw, dict) or (raw is not None and not pd.isna(raw) and str(raw).strip() != "")
    if not supplied and row.get("vps_version") is not None and not pd.isna(row.get("vps_version")):
        raise VPSContractError("Versioned VPS record missing canonical ledger")
    if supplied:
        validate_and_extract_vps_score(row)
        ledger = raw if isinstance(raw, dict) else json.loads(str(raw))
        meta = secondary_evidence(ledger["secondary_metadata"]["phase"], ledger["secondary_metadata"]["snr"])
        # Existing Antares representation allowance; never used for decisions.
        for column, key, decimals in (("sec_eclipse_phase", "phase", 3), ("secondary_snr_sigma", "snr", 2)):
            value = row.get(column)
            if value is not None and not pd.isna(value):
                try:
                    number = float(value)
                except (ValueError, TypeError) as exc:
                    raise VPSContractError(f"Invalid secondary display {column}") from exc
                if not math.isfinite(number) or all(abs(number - expected) > VPS_RECONSTRUCTION_TOLERANCE
                                                     for expected in (meta[key], round(meta[key], decimals))):
                    raise VPSContractError(f"Contradictory secondary display {column}")
        source, snr = "canonical_vps_ledger", meta["snr"]
    else:
        source = "legacy_display_only"
        try:
            snr = float(row.get("secondary_snr_sigma"))
        except (TypeError, ValueError):
            snr = math.nan
        if not math.isfinite(snr) or snr < 0:
            return {"source": source, "snr": None, "phase": None, "vps_penalty_applied": None, "ctoi_reject": None}
        meta = {"phase": None, "penalty_applied": None}
    return {"source": source, "snr": snr, "phase": meta["phase"],
            "vps_penalty_applied": meta["penalty_applied"],
            "ctoi_reject": snr > CTOI_SECONDARY_THRESHOLD_SNR}
