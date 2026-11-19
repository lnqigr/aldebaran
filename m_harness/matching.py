"""Frozen matching rules: when does a detector peak count as the true signal?

centers, support and match are the accepted bank_valid_01 scorer (agent_exchange/bank_valid_01/scorer.py, SHA256
0d0a9c4c...), moved unchanged; it is the scorer of INJ-CAL-01, ALIAS-D1-D3 and NULL-FIDELITY-01.
agent_exchange/s4_m_package/port_equivalence.py checks AST equality function by function.

A peak is FUNDAMENTAL when every supported true event (within the true T14 / 2, plus half an exposure) pairs one to
one with a supported peak event, with equal cycle increments, and the solution can be told apart from its half and
double period on the observed timestamps. AMBIGUOUS, ALIAS and MISSED are not matches. A truth with fewer than three
supported events is UNAVAILABLE, which is reported on its own line and never counted as a miss or a match.
"""
import math
import numpy as np

# fmt: off
def centers(epoch,period,times):
    first=math.ceil((float(times[0])-epoch)/period);last=math.floor((float(times[-1])-epoch)/period)
    ids=np.arange(first,last+1,dtype=int)
    return ids,epoch+ids*period

def support(times,events,halfwidth,exposure=120/86400):
    return np.asarray([np.any(np.abs(times-c)<=halfwidth+exposure/2) for c in events],bool)

def match(candidate,truth,times):
    times=np.asarray(times,float);times=times[np.isfinite(times)]
    if len(times)<2:return dict(label='UNAVAILABLE',reason='NO_OBSERVABLE_TIMESTAMPS',fundamental=False)
    times=np.sort(times);p=float(truth['period_days']);t0=float(truth['t0_btjd_tdb']);half=float(truth['T14_hours'])/48
    kp,ep=centers(float(candidate['epoch_btjd_tdb']),float(candidate['period_days']),times)
    kt,et=centers(t0,p,times)
    kt,et=kt[support(times,et,half)],et[support(times,et,half)]
    kp,ep=kp[support(times,ep,half)],ep[support(times,ep,half)]
    if len(et)<3:return dict(label='UNAVAILABLE',reason='FEWER_THAN_THREE_SUPPORTED_TRUTH_EVENTS',fundamental=False,truth_events=len(et))
    # Optimal monotone interval matching by earliest feasible pair. It cannot
    # reuse events and is independent of candidate duration or grid cells.
    pairs=[];i=j=0
    while i<len(et) and j<len(ep):
        if abs(et[i]-ep[j])<=half:pairs.append((i,j));i+=1;j+=1
        elif ep[j]<et[i]-half:j+=1
        else:i+=1
    cover_truth=len(pairs)==len(et);cover_candidate=len(pairs)==len(ep)
    cycles=bool(pairs) and all(kt[b[0]]-kt[a[0]]==kp[b[1]]-kp[a[1]] for a,b in zip(pairs,pairs[1:]))
    period_ratio=float(candidate['period_days'])/p
    full=cover_truth and cover_candidate
    if full and cycles:
        # A fundamental must also be distinguishable from its half/double
        # alternatives through at least one observable differentiating phase.
        degeneracy=[]
        for ratio in (.5,2.):
            alternatives=[]
            offsets=(0.,p) if ratio==2 else (0.,)
            for offset in offsets:
                _,ev=centers(t0+offset,p*ratio,times)
                ev=ev[support(times,ev,half)]
                extra=any(np.all(np.abs(et-e)>half) for e in ev)
                missing=any(np.all(np.abs(ev-e)>half) for e in et) if len(ev) else True
                alternatives.append(not(extra or missing))
            if any(alternatives):degeneracy.append(ratio)
        label='AMBIGUOUS' if degeneracy else 'FUNDAMENTAL'
    elif full:label='AMBIGUOUS'
    else:
        rational=False
        for numerator,denominator in ((1,3),(1,2),(2,3),(3,2),(2,1),(3,1)):
            ratio=numerator/denominator
            locked=len(pairs)>=2 and all(numerator*(kp[b[1]]-kp[a[1]])==denominator*(kt[b[0]]-kt[a[0]]) for a,b in zip(pairs,pairs[1:]))
            if abs(period_ratio-ratio)<.02*ratio and locked:rational=True
        label='ALIAS' if rational else 'MISSED'
    return dict(label=label,fundamental=label=='FUNDAMENTAL',truth_events=len(et),candidate_events=len(ep),matched_events=len(pairs),cycle_increments_1to1=cycles,period_ratio=period_ratio,max_center_residual_hours=max((abs(et[i]-ep[j])*24 for i,j in pairs),default=None))
# fmt: on


MATCH_LABELS = ("FUNDAMENTAL", "AMBIGUOUS", "ALIAS", "MISSED", "UNAVAILABLE")


def match_peak(peak, truth, times):
    """Scorer result of one detector peak against one truth on the discovery search timestamps.
    peak: period_days, t0_btjd, duration_days (the discovery solution, as written in the vetting evidence).
    truth: period (d), t0 (BTJD), t14 (d)."""
    return match(dict(period_days=peak["period_days"], epoch_btjd_tdb=peak["t0_btjd"],
                      duration_hours=peak["duration_days"] * 24.0, sde=peak.get("sde", 0.0)),
                 dict(period_days=truth["period"], t0_btjd_tdb=truth["t0"], T14_hours=truth["t14"] * 24.0), times)


def recover(peaks, truth, times, sde_min=6.0):
    """Search-level recovery: the scorer result of every evaluated peak with SDE >= sde_min; the lowest rank wins.
    Returns outcome MATCH_FUNDAMENTAL, NO_MATCH or UNAVAILABLE, the matched rank and SDE, and the per-peak labels."""
    rows = []
    for p in peaks:
        sde = p.get("sde")
        if sde is None or sde < sde_min:
            continue
        m = match_peak(p, truth, times)
        rows.append(dict(rank=p["rank"], sde=float(sde), label=m["label"], reason=m.get("reason"),
                         fundamental=bool(m["fundamental"]), period_ratio=m.get("period_ratio")))
    fund = [r for r in rows if r["fundamental"]]
    if fund:
        best = min(fund, key=lambda r: r["rank"])
        return dict(outcome="MATCH_FUNDAMENTAL", rank=best["rank"], sde=best["sde"], peaks=rows)
    if rows and all(r["label"] == "UNAVAILABLE" for r in rows):
        return dict(outcome="UNAVAILABLE", rank=None, sde=None, peaks=rows)
    return dict(outcome="NO_MATCH", rank=None, sde=None, peaks=rows)


ALIAS_RATIOS = tuple((n, d) for n in range(1, 6) for d in range(1, 6) if math.gcd(n, d) == 1)


def period_relation(p_peak, p_true, tol_fundamental=0.04, tol_alias=0.02):
    """Period relation of a peak to a reference period by the item 26.2 ratio tolerances (4 % for 1:1, 2 % for the
    other small-integer ratios n:d up to 5): '1:1', 'n/d' or None. Descriptive label for the alias-promotion line of
    M3; the matching decision itself is `match`."""
    if not (p_peak and p_true and p_peak > 0 and p_true > 0):
        return None
    ratio = float(p_peak) / float(p_true)
    best = None
    for n, d in ALIAS_RATIOS:
        target = n / d
        tol = tol_fundamental if (n, d) == (1, 1) else tol_alias
        err = abs(ratio / target - 1.0)
        if err <= tol and (best is None or err < best[0]):
            best = (err, "1:1" if (n, d) == (1, 1) else f"{n}/{d}")
    return None if best is None else best[1]
