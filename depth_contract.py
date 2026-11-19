"""Versioned depth lineage; no transit model, thresholds or dilution corrections.

Legacy fitted_depth_ppm is geometric k² (or an explicitly flagged initial guess),
never an implicit photometric substitute. Export policy keeps measured box depth;
model midpoint depth is separate and does not replace the measured estimator.
"""
import math
import os
import csv

DEPTH_SCHEMA = "depth_contract_v1"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def fit_flag(value):
    if value is None or str(value).strip().lower() in ("", "nan", "none"):
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "1.0"):
        return True
    if text in ("false", "0", "0.0"):
        return False
    raise ValueError("Invalid fit_converged flag: " + repr(value))


def photometric_depth(row):
    """Return positive box depth and source; never reinterpret untyped fitted_depth."""
    version = row.get("depth_schema_version")
    if version is not None and str(version) not in ("", "nan", DEPTH_SCHEMA):
        raise ValueError("Unsupported depth schema: " + str(version))
    for field in ("measured_box_depth_ppm", "input_depth_ppm", "depth_ppm", "depth_ppm_t1"):
        value = number(row.get(field))
        if value > 0:
            return value, field
    raise ValueError("Missing positive measured photometric box depth")


def geometric_radius(row):
    """No sqrt(box-depth) refit. Preserve labelled legacy compatibility only."""
    flag = fit_flag(row.get("fit_converged"))
    if row.get("stellar_parameter_provenance") == "initial_model_assumptions_present":
        raise ValueError("Fitted physical radius relies on assumed host parameters")
    if flag is False:
        raise ValueError("Physical fit failed; initial radius is not a fitted observable")
    radius = number(row.get("fitted_rp_earth"))
    if radius > 0:
        return radius, "converged_geometric_fit" if flag is True else "legacy_fit_unverified"
    if flag is None:
        radius = number(row.get("r_planet_earth"))
        if radius > 0:
            return radius, "tier1_box_radius_approximation"
    raise ValueError("Missing geometric radius or labelled legacy radius estimate")


def fit_depth_fields(rp_rs, model_mid_flux, measured_box, converged, error="", measured_source="tier1_input_box"):
    """Publish distinct quantities. Model flux is evaluated by the existing model."""
    good = bool(converged)
    return {
        "depth_schema_version": DEPTH_SCHEMA,
        "fit_converged": good,
        "fit_status": "CONVERGED" if good else "INITIAL_GUESS_FALLBACK",
        "fit_error": str(error),
        "fitted_rp_rs": float(rp_rs) if good else float("nan"),
        "geometric_rp_rs_sq_ppm": float(rp_rs) ** 2 * 1e6 if good else float("nan"),
        "model_transit_depth_ppm": (1.0 - float(model_mid_flux)) * 1e6 if good else float("nan"),
        "model_depth_definition": "native_model_midpoint_flux_deficit_not_box_average",
        "measured_box_depth_ppm": number(measured_box),
        "measured_box_depth_source": measured_source,
        "fitted_depth_kind": "geometric_k_squared" if good else "geometric_initial_guess",
        "fitted_radius_kind": "geometric_fit" if good else "box_depth_initial_guess",
        "impact_parameter_kind": "geometric_fit" if good else "initial_guess",
        "fit_uncertainty_status": "NOT_CALIBRATED",
    }


def validate_observables(period, epoch, duration, depth, radius):
    values = [number(v) for v in (period, epoch, duration, depth, radius)]
    if not all(math.isfinite(v) for v in values) or not all(v > 0 for v in (values[0], *values[2:])):
        raise ValueError("Missing/nonfinite/nonpositive physical observable; no nominal defaults allowed")


def check_output_schema(path, columns, sep=","):
    """Refuse replacing a different table schema at a shared historical path."""
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle, delimiter=sep), [])
        if header != list(columns):
            raise ValueError("Refusing conflicting output schema at " + str(path))
