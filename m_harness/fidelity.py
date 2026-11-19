"""Development fidelity gate of the N1 null (ARGON_DECISIONS item 31.7, operational definitions of item 32
Amendment 1; prospective item 44 replaces the ACF maximum rule for the M4 noise-only pre-gate).

Noise criteria, on the production arrays of every bound star, unmodified (U) and N1:
  C1  |median_N1 / median_U - 1| <= 0.05 for BOTH the 1-hour CDPP and the MAD scatter (ratio of cohort medians).
  C2  lag 1, C3 lag 10: all three of
        |median_N1 - median_U| <= max(0.005, 0.10 |median_U|)   (difference of arm medians)
        type-7 P90 of the per-star |N1 - U| <= 0.015
        maximum of the per-star |N1 - U| <= 0.20 (development only)
        M4 item 44 instead requires at most 5 % with per-star |N1 - U| > 0.05, at each lag.
Signal-destruction criteria, on the bound injected planets with N1 applied after injection:
  C4  recovered <= 5 % of all bound planets (floor of 0.05 n; 2 of 50)
  C5  recovered <= 5 % of the bound planets with true period below 2.0 d (1 of 22)
Equality passes every threshold. Every bound star and planet is required: a missing, duplicate, non-finite or
boolean value is a failed evaluation, never a smaller denominator. A failed condition or evaluation blocks the gate
and returns the numbers for consultation; nothing here tunes a threshold.

The arrays come from the production functions themselves (extract_and_stitch_arrays, clip_upper_outliers,
piecewise_savgol_detrend); the harness passes the production module in, it holds no copy of them.
"""
import math

import numpy as np

from . import prodpath, stats

SCATTER_REL_TOL = 0.05
ACF_LAGS = (1, 10)
ACF_MEDIAN_ABS_TOL = 0.005
ACF_MEDIAN_REL_TOL = 0.10
ACF_P90_TOL = 0.015
ACF_MAX_TOL = 0.20
RECOVERY_LIMIT_FRACTION = 0.05
SHORT_PERIOD_DAYS = 2.0
RECOVERY_PERIOD_TOL = 0.02          # gate match of the development gate: |P_peak / P_true - 1| <= 0.02
CADENCE_S = 120.0
MIN_BIN_CADENCES = 20
NOISE_FIELDS = ("cdpp_1h", "mad", "acf1", "acf10")


def production_arrays(ald, lightcurve):
    """Production discovery arrays of one light curve object: extraction, upper 3 MAD clip, Savitzky-Golay trend.
    Returns (t, x) with x = f / trend - 1. No known-signal mask (null and injection hosts have none)."""
    t, flat = prodpath.production_flat(ald, lightcurve)
    return t, flat - 1.0


def noise_metrics(t, x):
    """MAD scatter, 1-hour CDPP and lag-1 / lag-10 autocorrelation of a residual series (NULL-FIDELITY-01
    preregistration section 6). CDPP: bins of 1/24 d from the first timestamp, bins with at least 20 cadences,
    1.4826 MAD of the bin means. Autocorrelation: pairs whose time difference is within 60 s of k x 120 s."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    b = np.floor((t - t[0]) * 24.0).astype(np.int64)
    _ub, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=x)
    means = (sums / cnt)[cnt >= MIN_BIN_CADENCES]
    cdpp = 1.4826 * float(np.median(np.abs(means - np.median(means)))) if means.size else float("nan")
    y = x - np.mean(x)
    den = float(np.mean(y * y))
    out = dict(mad=mad, cdpp_1h=cdpp, n_bins=int(means.size), n=int(x.size))
    for k in ACF_LAGS:
        dt = (t[k:] - t[:-k]) * 86400.0
        ok = np.abs(dt - k * CADENCE_S) <= 0.5 * CADENCE_S
        out[f"acf{k}"] = float(np.mean(y[:-k][ok] * y[k:][ok]) / den) if ok.any() and den > 0 else float("nan")
        out[f"n_pairs_{k}"] = int(ok.sum())
    return out


def star_row(star_id, t_u, x_u, t_n1, x_n1):
    """One star's paired noise row from its unmodified and N1 production arrays."""
    return dict(star=star_id, U=noise_metrics(t_u, x_u), N1=noise_metrics(t_n1, x_n1))


def _number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and math.isfinite(float(v))


def validate_noise_rows(rows, expected_stars):
    """Item 32 validity rule: exactly one row for every bound star, no extra star, every required field a finite
    number in both arms. Returns the list of problems (empty when valid)."""
    problems = []
    expected = list(expected_stars)
    seen = [r.get("star") for r in rows]
    for s in expected:
        c = seen.count(s)
        if c == 0:
            problems.append(f"missing star {s!r}")
        elif c > 1:
            problems.append(f"duplicate star {s!r} ({c} rows)")
    for s in seen:
        if s not in expected:
            problems.append(f"star {s!r} is not in the bound set")
    for r in rows:
        for arm in ("U", "N1"):
            for field in NOISE_FIELDS:
                v = (r.get(arm) or {}).get(field)
                if not _number(v):
                    problems.append(f"star {r.get('star')!r} {arm}.{field} is not a finite number ({v!r})")
    return problems


def noise_gate(rows, expected_stars, *, m4=False):
    """C1 to C3: Item32 by default; explicit m4=True selects prospective item 44."""
    expected = list(expected_stars)
    problems = validate_noise_rows(rows, expected)
    n = len(expected)
    out = dict(n_expected=n, n_rows=len(rows), problems=problems, criteria={})
    if problems or n == 0:
        if n == 0:
            out["problems"] = problems + ["empty bound set"]
        out["status"] = "FAILED_EVALUATION"
        return out

    def med(arm, field):
        return stats.median_fixed(r[arm][field] for r in rows)
    scatter = {}
    for field in ("cdpp_1h", "mad"):
        mu, mn = med("U", field), med("N1", field)
        if not mu > 0:
            out["problems"].append(f"cohort median of U.{field} is not strictly positive ({mu})")
            out["status"] = "FAILED_EVALUATION"
            return out
        d = abs(mn / mu - 1.0)
        scatter[field] = dict(median_U=mu, median_N1=mn, D=d, limit=SCATTER_REL_TOL, passed=bool(d <= SCATTER_REL_TOL))
    out["criteria"]["C1"] = dict(status="PASS" if all(v["passed"] for v in scatter.values()) else "FAIL", n=n, **scatter)
    for name, k in (("C2", 1), ("C3", 10)):
        field = f"acf{k}"
        mu, mn = med("U", field), med("N1", field)
        diffs = [abs(r["N1"][field] - r["U"][field]) for r in rows]
        median_limit = max(ACF_MEDIAN_ABS_TOL, ACF_MEDIAN_REL_TOL * abs(mu))
        parts = dict(
            median=dict(median_U=mu, median_N1=mn, abs_diff=abs(mn - mu), limit=median_limit,
                        passed=bool(abs(mn - mu) <= median_limit)),
            p90=dict(value=stats.quantile_type7(diffs, 0.90), limit=ACF_P90_TOL),
            max=dict(value=max(diffs), limit=ACF_MAX_TOL))
        parts["p90"]["passed"] = bool(parts["p90"]["value"] <= ACF_P90_TOL)
        parts["max"]["passed"] = bool(parts["max"]["value"] <= ACF_MAX_TOL)
        if m4:
            # Item 44 replaces only the M4 maximum condition, separately at each lag.
            parts.pop("max")
            count = sum(d > 0.05 for d in diffs)
            parts["tail"] = dict(count=count, n=n, threshold=0.05, fraction=count / n,
                                 limit=0.05, passed=bool(count * 20 <= n))
        out["criteria"][name] = dict(status="PASS" if all(p["passed"] for p in parts.values()) else "FAIL",
                                     lag=k, n=n, **parts)
    out["status"] = "PASS" if all(c["status"] == "PASS" for c in out["criteria"].values()) else "FAIL"
    return out


def m4_tail_report(rows, expected_stars, outcomes):
    """Item 44.3 listing. Outcomes are N1 case_outcome rows keyed by TIC; absent runs stay unknown.
    Each noise row must carry its sector. Ranking uses the larger absolute shift across the two lags."""
    problems = validate_noise_rows(rows, expected_stars)
    problems += [f"star {r.get('star')!r} has no sector" for r in rows if r.get("sector") is None]
    if problems:
        return dict(status="FAILED_EVALUATION", problems=problems)
    ranked = []
    for r in rows:
        acf = {}
        for k in ACF_LAGS:
            u, n1 = float(r["U"][f"acf{k}"]), float(r["N1"][f"acf{k}"])
            acf[str(k)] = dict(U=u, N1=n1, signed_difference=n1 - u, absolute_difference=abs(n1 - u))
        outcome = outcomes.get(r["star"], {})
        sde = outcome.get("top_sde")
        evaluated = outcome.get("status_class") not in (None, "FAILED", "UNAVAILABLE") and _number(sde)
        ranked.append(dict(tic_id=r["star"], sector=r["sector"], acf=acf,
                           max_absolute_difference=max(a["absolute_difference"] for a in acf.values()),
                           max_tier1_sde=sde if _number(sde) else None,
                           false_alarm_sde_6=bool(sde >= 6.0) if evaluated else None,
                           outcome_status=outcome.get("status"), outcome_evaluated=bool(evaluated)))
    ranked.sort(key=lambda r: (-r["max_absolute_difference"], str(r["tic_id"]), r["sector"]))
    return dict(status="COMPLETE", above_005=[r for r in ranked if r["max_absolute_difference"] > 0.05],
                top_10=ranked[:10], cohort_maximum=ranked[0] if ranked else None)


def recovery_limit(n):
    """Largest recovered count that is still at most 5 % of n."""
    return int(math.floor(RECOVERY_LIMIT_FRACTION * n + 1e-12))


def recovery_gate(rows, expected_planets):
    """C4 and C5 over the bound injected planets. rows: dict(planet, period_days, recovered (bool), error (bool)).
    A run error counts as recovered (conservative); a missing or duplicate planet is a failed evaluation."""
    expected = list(expected_planets)
    seen = [r.get("planet") for r in rows]
    problems = [f"missing planet {p!r}" for p in expected if seen.count(p) == 0]
    problems += [f"duplicate planet {p!r}" for p in expected if seen.count(p) > 1]
    problems += [f"planet {p!r} is not in the bound set" for p in seen if p not in expected]
    for r in rows:
        if not _number(r.get("period_days")) or not float(r["period_days"]) > 0:
            problems.append(f"planet {r.get('planet')!r} has no valid true period")
        if not isinstance(r.get("recovered"), (bool, np.bool_)) and not r.get("error"):
            problems.append(f"planet {r.get('planet')!r} has no recovery outcome")
    out = dict(n_expected=len(expected), n_rows=len(rows), problems=problems, criteria={})
    if problems or not expected:
        if not expected:
            out["problems"] = problems + ["empty bound set"]
        out["status"] = "FAILED_EVALUATION"
        return out

    def count(sel):
        return sum(1 for r in sel if r.get("error") or r["recovered"])
    short = [r for r in rows if float(r["period_days"]) < SHORT_PERIOD_DAYS]
    for name, sel in (("C4", rows), ("C5", short)):
        k, n = count(sel), len(sel)
        if n == 0:
            out["criteria"][name] = dict(status="FAILED_EVALUATION", recovered=k, n=0,
                                         reason="no bound planet in this stratum")
            continue
        out["criteria"][name] = dict(status="PASS" if k <= recovery_limit(n) else "FAIL", recovered=k, n=n,
                                     limit=recovery_limit(n), rate=k / n, errors=sum(1 for r in sel if r.get("error")))
    st = [c["status"] for c in out["criteria"].values()]
    out["status"] = "FAILED_EVALUATION" if "FAILED_EVALUATION" in st else ("PASS" if all(s == "PASS" for s in st) else "FAIL")
    return out


def fidelity_gate(noise_rows, expected_stars, recovery_rows=None, expected_planets=None, *, m4=False):
    """The whole Item32 development gate. With recovery_rows None only noise is evaluated;
    explicit m4=True selects the item 44 tail condition and is forbidden for the full development gate."""
    if m4 and recovery_rows is not None:
        raise ValueError("Item 44 applies only to the M4 noise pre-gate")
    noise = noise_gate(noise_rows, expected_stars, m4=m4)
    out = dict(noise=noise, recovery=None, scope="NOISE_ONLY_PRE_GATE" if recovery_rows is None else "FULL_DEVELOPMENT_GATE")
    parts = [noise["status"]]
    if recovery_rows is not None:
        out["recovery"] = recovery_gate(recovery_rows, expected_planets or [])
        parts.append(out["recovery"]["status"])
    out["status"] = "FAILED_EVALUATION" if "FAILED_EVALUATION" in parts else ("PASS" if all(p == "PASS" for p in parts) else "FAIL")
    out["blocks_m4"] = out["status"] != "PASS"
    return out


def recovered_after_n1(record, p_true):
    """Gate match of the development gate (NULL-FIDELITY-01 preregistration section 6, conservative and epoch-free):
    the planet counts as recovered when any CANDIDATE evidence peak of the run has |P_peak / P_true - 1| <= 0.02.
    N1 moves the transit epochs, so an event match could miss a coherent train at the true period; this counts it.
    Returns (recovered, error): a run that failed or had no data is an error, which the gate counts as recovered."""
    if record["status"] == "ERROR" or str(record["status"]).startswith("SKIP_"):
        return True, True
    hit = any(p.get("disposition") == "CANDIDATE" and abs(float(p["period_days"]) / float(p_true) - 1.0) <= RECOVERY_PERIOD_TOL
              for p in record.get("peaks", []))
    return bool(hit), False
