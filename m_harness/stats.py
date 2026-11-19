"""Exact binomial bounds, fixed order statistics and host-clustered resampling for the M estimands.

Interval conventions (ARGON_DECISIONS item 31; M preregistration draft section 2):
  one-sided 95 % Clopper-Pearson upper bound   false-alarm and false-positive rates (M3, M4), "at most" claims
  one-sided 95 % Clopper-Pearson lower bound   reported where a "at least" claim is made
  two-sided 95 % Clopper-Pearson               completeness, false rejection and coverage (M1, M2, M5)
The two-sided interval puts 2.5 % in each tail, so its upper limit is larger than the one-sided 95 % upper bound of
the same count. Using one where the other is required is a defect, and the tests pin both.
"""
import math

import numpy as np
from scipy import stats as _stats

LEVEL = 0.95


def _check(k, n):
    if isinstance(k, bool) or isinstance(n, bool) or int(k) != k or int(n) != n:
        raise ValueError(f"counts must be integers, got k={k!r}, n={n!r}")
    k, n = int(k), int(n)
    if n <= 0:
        raise ValueError(f"denominator must be positive, got n={n}")
    if not 0 <= k <= n:
        raise ValueError(f"count out of range: k={k}, n={n}")
    return k, n


def cp_upper(k, n, level=LEVEL):
    """One-sided Clopper-Pearson upper bound at confidence `level`; 1 when k == n."""
    k, n = _check(k, n)
    return 1.0 if k == n else float(_stats.beta.ppf(level, k + 1, n - k))


def cp_lower(k, n, level=LEVEL):
    """One-sided Clopper-Pearson lower bound at confidence `level`; 0 when k == 0."""
    k, n = _check(k, n)
    return 0.0 if k == 0 else float(_stats.beta.ppf(1.0 - level, k, n - k + 1))


def cp_two_sided(k, n, level=LEVEL):
    """Two-sided (equal-tailed) Clopper-Pearson interval at confidence `level`."""
    k, n = _check(k, n)
    tail = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(_stats.beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(_stats.beta.ppf(1.0 - tail, k + 1, n - k))
    return lo, hi


SIDES = ("two-sided", "upper", "lower")


def rate(k, n, side):
    """A reported rate: count, denominator, point estimate and its 95 % Clopper-Pearson interval.

    side: "two-sided", "upper" (one-sided upper bound, lower limit 0) or "lower" (one-sided lower bound, upper
    limit 1). n == 0 gives NOT_AVAILABLE, never a rate."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if int(n) == 0:
        return dict(k=int(k), n=0, rate=None, lo=None, hi=None, interval=f"Clopper-Pearson 95 % {side}",
                    status="NOT_AVAILABLE_EMPTY_DENOMINATOR")
    k, n = _check(k, n)
    if side == "two-sided":
        lo, hi = cp_two_sided(k, n)
    elif side == "upper":
        lo, hi = 0.0, cp_upper(k, n)
    else:
        lo, hi = cp_lower(k, n), 1.0
    return dict(k=k, n=n, rate=k / n, lo=lo, hi=hi, interval=f"Clopper-Pearson 95 % {side}", status="OK")


def median_fixed(values):
    """Median by the item 32 convention: sorted middle value for odd n, mean of the two middle values for even n."""
    x = sorted(float(v) for v in values)
    n = len(x)
    if n == 0:
        raise ValueError("median of an empty sample")
    return x[n // 2] if n % 2 == 1 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def quantile_type7(values, q):
    """Empirical quantile, type 7 linear interpolation (item 32): h = (n - 1) q, j = floor(h), g = h - j,
    Q = (1 - g) x[j] + g x[j + 1], and x[j] alone when g == 0."""
    x = sorted(float(v) for v in values)
    n = len(x)
    if n == 0:
        raise ValueError("quantile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile level out of range: {q}")
    h = (n - 1) * q
    j = int(math.floor(h))
    g = h - j
    return x[j] if g == 0 else (1.0 - g) * x[j] + g * x[j + 1]


def host_cluster_bootstrap(rows, host_key, statistic, n_boot=2000, seed=0, level=LEVEL):
    """Percentile interval of `statistic(rows)` under resampling of whole hosts with replacement (roadmap section 12:
    several cases of one host are not independent). rows: list of dicts; host_key: the field naming the host.
    Returns dict(point, lo, hi, n_hosts, n_rows, n_boot, n_undefined). A resample whose statistic is None is counted
    in n_undefined and left out; with fewer than 2 hosts the interval is NOT_AVAILABLE."""
    by_host = {}
    for r in rows:
        by_host.setdefault(r[host_key], []).append(r)
    hosts = sorted(by_host, key=str)
    point = statistic(list(rows))
    out = dict(point=point, lo=None, hi=None, n_hosts=len(hosts), n_rows=len(rows), n_boot=int(n_boot), n_undefined=0,
               interval=f"host-cluster bootstrap {int(round(level * 100))} % percentile", seed=int(seed))
    if len(hosts) < 2:
        out["status"] = "NOT_AVAILABLE_FEWER_THAN_TWO_HOSTS"
        return out
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(hosts), len(hosts))
        sample = [r for i in pick for r in by_host[hosts[i]]]
        v = statistic(sample)
        if v is None:
            out["n_undefined"] += 1
        else:
            vals.append(float(v))
    if not vals:
        out["status"] = "NOT_AVAILABLE_STATISTIC_UNDEFINED"
        return out
    tail = (1.0 - level) / 2.0
    out.update(lo=float(quantile_type7(vals, tail)), hi=float(quantile_type7(vals, 1.0 - tail)), status="OK")
    return out


def n_for_upper_bound(k, target, level=LEVEL, n_max=10_000_000):
    """Smallest n whose one-sided upper bound with k events is at most `target` (a rate between 0 and 1)."""
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be a rate strictly between 0 and 1, got {target}")
    lo, hi = max(int(k), 1), max(int(k), 1)
    while cp_upper(k, hi, level) > target:
        hi *= 2
        if hi > n_max:
            raise ValueError("required sample size exceeds n_max")
    while lo < hi:
        mid = (lo + hi) // 2
        if cp_upper(k, mid, level) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def halfwidth_two_sided(n, p):
    """Half-width of the two-sided 95 % Clopper-Pearson interval at the count nearest to n * p."""
    k = int(round(n * p))
    lo, hi = cp_two_sided(k, n)
    return 0.5 * (hi - lo)


def n_for_halfwidth(halfwidth, p=0.5, n_max=1_000_000):
    """Smallest n whose two-sided 95 % Clopper-Pearson half-width at expected rate p is at most `halfwidth`, and
    stays at most `halfwidth` for the next 20 values of n (the exact half-width is not monotone in n)."""
    if not 0.0 < halfwidth < 0.5:
        raise ValueError(f"halfwidth must be between 0 and 0.5, got {halfwidth}")
    n = 2
    while n <= n_max:
        if all(halfwidth_two_sided(m, p) <= halfwidth for m in range(n, n + 21)):
            return n
        n += 1
    raise ValueError("required sample size exceeds n_max")
