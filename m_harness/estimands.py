"""Estimand calculators for M1 to M5, the grey-zone estimand and the item 31 diagnostics.

Input: outcome records of routing.run_case, each joined with its design row (cohort, host, CROWDSAP, truth or
reference, SNR). Nothing here decides a detection: statuses, gates, candidates and archival diagnostics are read from
the production records, and the only arithmetic is counting and the intervals of stats.py.

Conventions (M preregistration draft):
  - M1 excludes DRAW_FAILED planning losses under item 46; the attrition ledger retains every planned row.
  - Other failed, unavailable and incomplete cases are lines of the table,
    never dropped rows; a ledger that does not sum to its denominator raises.
  - An unknown status raises: a new production status must be classified here before any table is built.
  - Completeness, false rejection and coverage carry two-sided 95 % Clopper-Pearson intervals; false-alarm and
    false-positive rates carry the one-sided 95 % upper bound (ARGON_DECISIONS item 31.3, 31.7).
  - Rates over several cases of one host also get a host-cluster bootstrap interval (roadmap section 12).
"""
import math

from . import matching, stats

SDE_PRODUCTION = 6.0
SDE_GREY_UPPER = 7.5
SDE_SINGLE_SECTOR = 6.5
CROWDSAP_DOMAIN_MIN = 0.70
CROWDSAP_STRATUM = 0.90
DOMAIN = dict(crowdsap_min=0.70, ruwe_max=1.4, r_star_max=1.75, teff_max=9500.0, depth_ppm_min=180.0,
              depth_ppm_max=30000.0, rp_earth_max=15.0)                       # items 31.2, 31.7, 31.9
M3_POINT_LIMIT = 0.10               # item 31.3: label rule, strictly above
M3_UPPER_LIMIT = 0.15
GREY_GAIN_MIN = 0.05                # item 31.8: dC_grey >= 5.0 percentage points
GREY_FAR_UPPER_MAX = 0.015          # item 31.8: M4(c) one-sided 95 % upper bound at SDE >= 6.0 <= 1.5 %
PERIOD_OFFSET_N_SIGMA = 3.5
NOMINAL_COVERAGE = {1: 0.6827, 2: 0.9545, 3: 0.9973}
RP_EARTH_PER_RSUN = 109.076         # the detector's own conversion: r_planet_earth = r_star * rp_rs * 109.076

HOST_DOMAIN_STOPS = ("REJECT_CROWDSAP", "REJECT_EVOLVED_STAR", "REJECT_RUWE")
HOST_DATA_STOPS = ("REJECT_BASELINE", "REJECT_TOI_MASK_TOO_LARGE", "REJECT_NO_STELLAR_PARAMS")
REVIEW_SUBROWS = ("REVIEW_SECONDARY_NOT_EVALUATED", "REVIEW_ARCHIVAL_SECONDARY_NOT_EVALUATED",
                  "REVIEW_CENTROID_NOT_EVALUATED")


def classify_status(status):
    """Class of a final host status. Raises on a status this harness does not know how to count."""
    s = str(status)
    if s == "CANDIDATE":
        return "CANDIDATE"
    if s.startswith("REVIEW_"):
        return "REVIEW"
    if s.startswith("INCOMPLETE_"):
        return "INCOMPLETE"
    if s == "ERROR":
        return "FAILED"
    if s.startswith("SKIP_"):
        return "UNAVAILABLE"
    if s in HOST_DOMAIN_STOPS:
        return "DOMAIN_STOP"
    if s in HOST_DATA_STOPS:
        return "HOST_STOP"
    if s.startswith("EMPTY_"):
        return "NO_DETECTION"
    if s.startswith("REJECT_"):
        return "GATE_REJECT"
    raise ValueError(f"UNCLASSIFIED_STATUS: {status!r}")


def crowd_stratum(crowdsap):
    """Reporting stratum of item 31.9."""
    if crowdsap is None or not math.isfinite(crowdsap):
        return "CROWDSAP unknown"
    if crowdsap < CROWDSAP_DOMAIN_MIN:
        return "CROWDSAP < 0.70 (outside the domain)"
    return "CROWDSAP >= 0.90" if crowdsap >= CROWDSAP_STRATUM else "CROWDSAP [0.70, 0.90)"


def domain_check(meta):
    """Declared-domain membership from metadata alone (item 31.9). meta: crowdsap, ruwe, r_star, teff, and for a
    known planet depth_ppm and rp_earth. A missing value is reported as unknown, never as inside.
    Returns dict(in_domain True / False / None, reasons, unknown)."""
    reasons, unknown = [], []

    def val(key):
        v = meta.get(key)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            unknown.append(key)
            return None
        return float(v)
    c, ruwe, r, t = val("crowdsap"), val("ruwe"), val("r_star"), val("teff")
    if c is not None and c < DOMAIN["crowdsap_min"]:
        reasons.append("CROWDSAP < 0.70")
    if ruwe is not None and ruwe > DOMAIN["ruwe_max"]:
        reasons.append("RUWE > 1.4")
    if r is not None and r > DOMAIN["r_star_max"]:
        reasons.append("R* > 1.75 R_sun")
    if t is not None and t > DOMAIN["teff_max"]:
        reasons.append("Teff > 9500 K")
    if "depth_ppm" in meta:
        d = val("depth_ppm")
        if d is not None and d < DOMAIN["depth_ppm_min"]:
            reasons.append("depth < 180 ppm")
        if d is not None and d > DOMAIN["depth_ppm_max"]:
            reasons.append("depth > 30,000 ppm")
    if "rp_earth" in meta:
        rp = val("rp_earth")
        if rp is not None and rp > DOMAIN["rp_earth_max"]:
            reasons.append("Rp > 15 R_E")
    in_domain = False if reasons else (None if unknown else True)
    return dict(in_domain=in_domain, reasons=reasons, unknown=unknown)


# ------------------------------------------------------------------------------------------- per-case outcome
def _top_sde(record):
    top = (record.get("search") or {}).get("top_peaks")
    if top:
        return float(top[0]["sde"])
    first = next((j for j in record.get("journal", []) if j.get("rank") == 1), None)
    return None if first is None or first.get("sde") is None else float(first["sde"])


def _peak_evidence(record, rank):
    return next((p for p in record.get("peaks", []) if p.get("peak_rank") == rank), None)


def _review_subrow(record):
    return record["status"] if record["status"] in REVIEW_SUBROWS else None


def _centroid_not_evaluated(peak):
    return any(g["gate"] == "CENTROID" and g["status"] == "NOT_EVALUATED" for g in (peak or {}).get("gates", []))


def case_outcome(record, truth=None, times=None):
    """One case reduced to the fields the tables count.

    With a truth (injected planet or blinded reference ephemeris) and the discovery search timestamps, the selected
    BLS peaks are matched by the frozen scorer and the matched rank is followed through the cascade:
      stage RUN      the run failed, had no data or stayed incomplete at host level
      stage HOST     a host-level gate stopped the star before the search
      stage SEARCH   no selected peak with SDE >= 6.0 is a FUNDAMENTAL match (line: NO_PEAK_ABOVE_6, ALIAS_ONLY,
                     NOT_MATCHED or MATCH_UNAVAILABLE)
      stage VETTING  the matched peak was rejected, held for review, left incomplete or never evaluated
      stage FINAL    the matched peak is CANDIDATE evidence but the reported candidate is an earlier peak
      stage NONE     no loss: the reported CANDIDATE is the matched peak
    Without a truth (nulls, EBs counted raw) only the host status, classes and SDE fields are filled."""
    status = record["status"]
    klass = classify_status(status)
    top_sde = _top_sde(record)
    cand = record.get("candidate")
    out = dict(case_id=record["case_id"], status=status, status_class=klass, top_sde=top_sde,
               candidate_sde=None if cand is None else float(cand["sde"]),
               review_subrow=_review_subrow(record), is_candidate=klass == "CANDIDATE",
               reached_archival=any(g["gate"] == "ARCHIVAL" for p in record.get("peaks", []) for g in p.get("gates", [])),
               matched_rank=None, matched_sde=None, search_recovered=None, matched_candidate=None,
               matched_candidate_or_review=None, match_outcome=None, centroid_not_evaluated=None,
               stage=None, line=None)
    if truth is None:
        return out
    if klass in ("FAILED", "UNAVAILABLE"):
        return dict(out, stage="RUN", line=status, search_recovered=False, matched_candidate=False,
                    matched_candidate_or_review=False, match_outcome="NOT_RUN")
    top = (record.get("search") or {}).get("top_peaks")
    if not top:
        if klass in ("DOMAIN_STOP", "HOST_STOP"):
            return dict(out, stage="HOST", line=status, search_recovered=False, matched_candidate=False,
                        matched_candidate_or_review=False, match_outcome="NOT_SEARCHED")
        raise ValueError(f"NO_SEARCH_RECORD: case {record['case_id']} ended {status} without selected peaks")
    if times is None:
        raise ValueError(f"NO_SEARCH_TIMES: case {record['case_id']}")
    rec = matching.recover(top, truth, times, sde_min=SDE_PRODUCTION)
    out["match_outcome"] = rec["outcome"]
    if rec["outcome"] != "MATCH_FUNDAMENTAL":
        if rec["outcome"] == "UNAVAILABLE":
            line = "MATCH_UNAVAILABLE"
        elif not rec["peaks"]:
            line = "NO_PEAK_ABOVE_6"
        elif any(p["label"] == "ALIAS" for p in rec["peaks"]):
            line = "ALIAS_ONLY"
        else:
            line = "NOT_MATCHED"
        return dict(out, stage="SEARCH", line=line, search_recovered=False, matched_candidate=False,
                    matched_candidate_or_review=False)
    rank = rec["rank"]
    out.update(matched_rank=rank, matched_sde=float(rec["sde"]), search_recovered=True)
    jr = next((j for j in record.get("journal", []) if j.get("rank") == rank), None)
    peak = _peak_evidence(record, rank)
    out["centroid_not_evaluated"] = _centroid_not_evaluated(peak)
    if jr is None:
        return dict(out, stage="VETTING", line=f"NOT_EVALUATED:{status}", matched_candidate=False,
                    matched_candidate_or_review=False)
    disp = jr.get("disposition")
    if disp == "CANDIDATE":
        first = (record.get("host_end") or {}).get("first_candidate_rank")
        if klass == "CANDIDATE" and first == rank:
            return dict(out, stage="NONE", line="MATCHED_CANDIDATE", matched_candidate=True,
                        matched_candidate_or_review=True)
        return dict(out, stage="FINAL", line=f"CANDIDATE_EVIDENCE_NOT_REPORTED:{status}", matched_candidate=False,
                    matched_candidate_or_review=klass in ("CANDIDATE", "REVIEW"))
    if disp == "REVIEW":
        return dict(out, stage="VETTING", line=f"REVIEW:{jr.get('cause')}", matched_candidate=False,
                    matched_candidate_or_review=klass in ("CANDIDATE", "REVIEW"))
    if disp == "INCOMPLETE":
        return dict(out, stage="VETTING", line=f"INCOMPLETE:{jr.get('cause')}", matched_candidate=False,
                    matched_candidate_or_review=False)
    gate = jr.get("first_reject_gate") or jr.get("exit_code") or jr.get("cause") or "UNKNOWN"
    return dict(out, stage="VETTING", line=f"GATE:{gate}", matched_candidate=False, matched_candidate_or_review=False)


STAGES = ("PLANNING", "RUN", "HOST", "SEARCH", "VETTING", "FINAL")


def attrition_table(rows, n_planned):
    """Per-stage attrition ledger. rows: one dict per planned case with stage and line; a case that was planned but
    never became a run has stage PLANNING (line DRAW_FAILED, INJECTION_FAILED, NOT_RUN ...); survivors have stage
    NONE. Returns the ordered lines with counts and the remaining count after each stage. The ledger must balance."""
    if len(rows) != n_planned:
        raise ValueError(f"LEDGER_UNBALANCED: {len(rows)} rows for {n_planned} planned cases")
    table, remaining = [], n_planned
    for stage in STAGES:
        lost = [r for r in rows if r["stage"] == stage]
        lines = {}
        for r in lost:
            lines[r["line"]] = lines.get(r["line"], 0) + 1
        remaining -= len(lost)
        table.append(dict(stage=stage, lost=len(lost), remaining=remaining,
                          lines=[dict(line=k, n=v) for k, v in sorted(lines.items(), key=lambda kv: (-kv[1], kv[0]))]))
    survivors = sum(1 for r in rows if r["stage"] == "NONE")
    unknown = [r["stage"] for r in rows if r["stage"] not in STAGES + ("NONE",)]
    if unknown or survivors != remaining:
        raise ValueError(f"LEDGER_UNBALANCED: survivors {survivors}, remaining {remaining}, unknown stages {unknown}")
    return dict(n_planned=n_planned, n_survivors=survivors, stages=table)


def _strata(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def _with_bootstrap(entry, rows, flag, host_key, n_boot, seed):
    """Add a host-cluster bootstrap interval when hosts repeat."""
    hosts = {r[host_key] for r in rows}
    if rows and len(hosts) < len(rows):
        entry["host_cluster"] = stats.host_cluster_bootstrap(
            rows, host_key, lambda s: (sum(1 for x in s if x[flag]) / len(s)) if s else None, n_boot=n_boot, seed=seed)
    entry["n_hosts"] = len(hosts)
    return entry


# ------------------------------------------------------------------------------------------- M1
def _grey(r, level):
    """SDE of the case at the chosen level, or None when the case does not count at that level.
    level "candidate": the reported, matched CANDIDATE's SDE. level "search": the matched peak's SDE."""
    if level == "candidate":
        return r["candidate_sde"] if r["matched_candidate"] else None
    if level == "search":
        return r["matched_sde"] if r["search_recovered"] else None
    raise ValueError(f"level must be 'candidate' or 'search', got {level!r}")


def completeness_cell(rows, level, host_key="host", n_boot=2000, seed=0):
    """Completeness figures of one cell (item 31.8): C(SDE >= 6.0), C(SDE >= 7.5), dC_grey = their difference (the
    share recovered with 6.0 <= SDE < 7.5), matched CANDIDATE completeness and CANDIDATE-or-REVIEW completeness.
    Denominator: every injected case of the cell, whatever happened to it."""
    planned = list(rows)
    rows = [r for r in planned if not (r.get("stage") == "PLANNING" and r.get("line") == "DRAW_FAILED")]
    n = len(rows)
    sde = [_grey(r, level) for r in rows]
    k6 = sum(1 for s in sde if s is not None and s >= SDE_PRODUCTION)
    k75 = sum(1 for s in sde if s is not None and s >= SDE_GREY_UPPER)
    flagged = [dict(r, _grey=(s is not None and SDE_PRODUCTION <= s < SDE_GREY_UPPER)) for r, s in zip(rows, sde)]
    out = dict(n=n, n_planned=len(planned), n_draw_failed=len(planned) - n, n_evaluated=n,
               n_recovered=k6, primary_completeness=stats.rate(k6, n, "two-sided"),
               unconditioned_completeness=stats.rate(k6, len(planned), "two-sided"), level=level,
               c_sde_6=stats.rate(k6, n, "two-sided"), c_sde_7_5=stats.rate(k75, n, "two-sided"),
               dc_grey=_with_bootstrap(stats.rate(k6 - k75, n, "two-sided"), flagged, "_grey", host_key, n_boot, seed),
               matched_candidate=_with_bootstrap(stats.rate(sum(1 for r in rows if r["matched_candidate"]), n, "two-sided"),
                                                 rows, "matched_candidate", host_key, n_boot, seed),
               candidate_or_review=stats.rate(sum(1 for r in rows if r["matched_candidate_or_review"]), n, "two-sided"),
               search_recovered=stats.rate(sum(1 for r in rows if r["search_recovered"]), n, "two-sided"),
               centroid_not_evaluated=sum(1 for r in rows if r.get("centroid_not_evaluated")),
               review_subrows={s: sum(1 for r in rows if r.get("review_subrow") == s) for s in REVIEW_SUBROWS})
    for key, flags in (('c_sde_6', [s is not None and s >= SDE_PRODUCTION for s in sde]),
                       ('c_sde_7_5', [s is not None and s >= SDE_GREY_UPPER for s in sde]),
                       ('candidate_or_review', [bool(r['matched_candidate_or_review']) for r in rows]),
                       ('search_recovered', [bool(r['search_recovered']) for r in rows])):
        _with_bootstrap(out[key], [dict(r, _flag=f) for r, f in zip(rows, flags)], '_flag', host_key, n_boot, seed)
    return out


# Item 47 is a pre-detector amendment after planning outcomes were observed.
ITEM47_COHORT_SHA256 = 'ed81ec376b3d8d6436ac0c02101b50868aac896a8f87202b13f67d36c3797890'
ITEM47_ANNOTATION = (
    'Preregistered C14 planning-loss limit of 30/300 (10.0 %) exceeded by 1 family '
    '(31/300 = 10.3 % lost; 269/300 = 89.7 % evaluated); accepted by pre-run amendment '
    '(items 46 and 47, Q39) before any detector execution.')
M1_BOUNDARY_MECHANISMS = (
    'Low-SNR floor ([4, 6), with a small tail up to 15): on quiet hosts, a transit with at least '
    '3 discovery events and depth >= 180 ppm already exceeds the target SNR; reaching it needs '
    'depth < 180 ppm or k < 0.005. High-SNR ceiling ([15, 25) and [25, 35]): on noisy hosts or '
    'larger stars limited by Rp <= 15 R_E, SNR >= 15 needs depth > 30,000 ppm, Rp > 15 R_E or k > 0.30.')


def m1_planning_limit(bin_index, cohort_sha256=None):
    if bin_index == 0 and cohort_sha256 == ITEM47_COHORT_SHA256:
        return 31
    return 30 if bin_index < 4 else 45


def m1_population(cohort_sha256=None):
    note=dict(mechanisms=M1_BOUNDARY_MECHANISMS)
    if cohort_sha256 == ITEM47_COHORT_SHA256:
        note.update(status='PASS_WITH_AMENDMENT',cohort_sha256=cohort_sha256,annotation=ITEM47_ANNOTATION,
                    amendment_after_draw_outcomes=True,before_detector_execution=True,
                    n_planned=1800,n_drawn=1736,n_draw_failed_by_bin=[31,11,6,3,6,7],
                    realized_grey_counts=[141,289,147],planned_grey_counts=[150,300,150],
                    aggregation_equivalence_claimed=False)
    return note


def m1_completeness(rows, bin_labels, level="candidate", min_cell=30, host_key="host", n_boot=2000, seed=0,
                    cohort_sha256=None, draw_records=None):
    """M1 tables. rows: case_outcome rows of injected cases, each with snr_bin (label), crowdsap, host and optional
    period_bin and radius_bin labels. Returns the pooled cell, one cell per SNR_inj bin, per CROWDSAP stratum, per
    bin and stratum, and the period and radius marginals. A cell below min_cell distinct hosts is UNSUPPORTED;
    counts and intervals stay visible (item 43.4)."""
    population = m1_population(cohort_sha256)
    if draw_records is not None:
        draws = [r for host in draw_records.values() for r in host]
        failed = [sum(r['status'] == 'DRAW_FAILED' for r in draws if r['snr_bin'] == b) for b in bin_labels]
        drawn = sum(r['status'] == 'DRAWN' for r in draws)
        row_failed = [sum(r.get('stage') == 'PLANNING' and r.get('line') == 'DRAW_FAILED'
                          for r in rows if r['snr_bin'] == b) for b in bin_labels]
        if (len(rows) != len(draws) or row_failed != failed or
                sum(r.get('line') != 'DRAW_FAILED' for r in rows) != drawn or
                any(sum(r['snr_bin'] == b for r in rows) != sum(r['snr_bin'] == b for r in draws)
                    for b in bin_labels) or any(r['snr_bin'] not in bin_labels for r in draws)):
            raise ValueError('M1_DRAW_COUNT_MISMATCH: table rows disagree with draw records')
        if 'annotation' in population and (len(draws) != population['n_planned'] or
                drawn != population['n_drawn'] or failed != population['n_draw_failed_by_bin']):
            raise ValueError('M1_DRAW_COUNT_MISMATCH: draw records disagree with frozen population')
    elif 'annotation' in population:
        raise ValueError('M1_DRAW_RECORDS_REQUIRED: amended population counts must be verified')
    def cell(sel):
        c = completeness_cell(sel, level, host_key, n_boot, seed)
        c["n_hosts"] = len({r[host_key] for r in sel if not (r.get("stage") == "PLANNING" and r.get("line") == "DRAW_FAILED")})
        c["supported"] = None if min_cell is None else bool(c["n_hosts"] >= min_cell)
        c["population"] = m1_population(cohort_sha256)
        return c
    out = dict(level=level, min_cell=min_cell, population=m1_population(cohort_sha256), pooled=cell(rows), by_snr_bin={}, by_crowding={}, by_snr_bin_and_crowding={},
               by_period_bin={}, by_radius_bin={})
    for label in bin_labels:
        out["by_snr_bin"][label] = cell([r for r in rows if r["snr_bin"] == label])
    for stratum, sel in sorted(_strata(rows, lambda r: crowd_stratum(r["crowdsap"])).items()):
        out["by_crowding"][stratum] = cell(sel)
        out["by_snr_bin_and_crowding"][stratum] = {label: cell([r for r in sel if r["snr_bin"] == label]) for label in bin_labels}
    for field, key in (("by_period_bin", "period_bin"), ("by_radius_bin", "radius_bin")):
        if all(key in r for r in rows):
            for label, sel in sorted(_strata(rows, lambda r: r[key]).items()):
                out[field][label] = cell(sel)
    unbinned = [r["case_id"] for r in rows if r["snr_bin"] not in bin_labels]
    if unbinned:
        raise ValueError(f"M1_UNBINNED_CASES: {unbinned[:5]} ({len(unbinned)} cases have no declared SNR_inj bin)")
    out['headline_rp_ge_0_8'] = cell([r for r in rows if r.get('rp_true_earth') is not None and r['rp_true_earth'] >= 0.8])
    out['radius_below_2'] = cell([r for r in rows if r.get('rp_true_earth') is not None and r['rp_true_earth'] < 2])
    from .injection import snr_bin
    out['achieved_snr'] = {}
    # Bin by the preassigned target; achieved values can cross a boundary within the solve tolerance.
    from .injection import bin_label, DRAFT_DESIGN
    for bi, bounds in enumerate(DRAFT_DESIGN.snr_bins):
        label = bin_label(*bounds)
        sel = [r for r in rows if r['snr_bin'] == label]
        have = [r['snr_achieved'] for r in sel if r.get('snr_achieved') is not None and math.isfinite(r['snr_achieved'])]
        q = lambda x: stats.quantile_type7(have, x) if have else None
        out['achieved_snr'][label] = dict(population=m1_population(cohort_sha256),n=len(have), unavailable=len(sel) - len(have), median=q(0.5),
            q25=q(0.25), q75=q(0.75), iqr=None if not have else q(0.75) - q(0.25),
            out_of_bin_fraction=None if not have else sum(snr_bin(v, DRAFT_DESIGN.snr_bins) != bi for v in have) / len(have))
    return out


def grey_zone_gain(rows, snr_field, support=(5.0, 9.0), level="candidate", bin_edges=None, host_key="host",
                   n_boot=2000, seed=0, cohort_sha256=None):
    """dC_grey over the SNR_inj support of item 31.8 (5.0 to 9.0, both ends included).

    pooled: every injected case whose `snr_field` lies in the support, each with equal weight.
    bin_weighted: when bin_edges is given, the mean of the dC_grey of the parts of the support inside each bin,
    weighted by the SNR width of each part (reported beside the pooled value; which one decides is a preregistered
    choice, see ARGON_DECISIONS item 31 point b)."""
    rows = [r for r in rows if not (r.get("stage") == "PLANNING" and r.get("line") == "DRAW_FAILED")]
    lo, hi = support
    sel = [r for r in rows if r.get(snr_field) is not None and lo <= r[snr_field] <= hi]
    out = dict(snr_field=snr_field, support=[lo, hi], level=level, population=m1_population(cohort_sha256), pooled=completeness_cell(sel, level, host_key, n_boot, seed))
    if bin_edges:
        parts, total = [], 0.0
        for b_lo, b_hi in bin_edges:
            a, b = max(lo, b_lo), min(hi, b_hi)
            if b <= a:
                continue
            last = (b_lo, b_hi) == tuple(bin_edges[-1])
            part = [r for r in sel if a <= r[snr_field] < b or (r[snr_field] == b and (last or b == hi))]
            c = completeness_cell(part, level, host_key, n_boot, seed)
            parts.append(dict(range=[a, b], width=b - a, n=len(part), dc_grey=c["dc_grey"]["rate"]))
            total += b - a
        if parts and all(p["dc_grey"] is not None for p in parts):
            out["bin_weighted"] = dict(value=sum(p["width"] * p["dc_grey"] for p in parts) / total, parts=parts)
        else:
            out["bin_weighted"] = dict(value=None, parts=parts, status="NOT_AVAILABLE_EMPTY_PART")
    return out


def grey_zone_decision(dc_grey, m4c_upper_at_6):
    """Item 31.8, as worded: the README grey-zone claim stays only if dC_grey >= 5.0 percentage points over SNR_inj
    5 to 9 and the M4(c) one-sided 95 % upper bound at SDE >= 6.0 is <= 1.5 %. Equality satisfies both. A missing
    input gives NOT_EVALUABLE, and then the claim is not kept."""
    if dc_grey is None or m4c_upper_at_6 is None:
        return dict(decision="NOT_EVALUABLE", keep_claim=False, dc_grey=dc_grey, m4c_upper_at_6=m4c_upper_at_6)
    gain_ok, far_ok = dc_grey >= GREY_GAIN_MIN, m4c_upper_at_6 <= GREY_FAR_UPPER_MAX
    return dict(decision="KEEP" if gain_ok and far_ok else "WITHDRAW", keep_claim=bool(gain_ok and far_ok),
                gain_condition=bool(gain_ok), far_condition=bool(far_ok), dc_grey=dc_grey,
                m4c_upper_at_6=m4c_upper_at_6, gain_min=GREY_GAIN_MIN, far_upper_max=GREY_FAR_UPPER_MAX)


def grey_zone_m4_decision(dc_grey, m4):
    """Item 43.11: unresolved runs cannot certify a favourable N1 decision."""
    c = m4['pooled']['c_candidate_sde_6']
    planned = m4['n_planned']
    observed = stats.rate(c['k'], planned, 'upper')
    worst = c['worst_case']
    upper = None if worst is None else worst['hi']
    out = grey_zone_decision(dc_grey, upper)
    out.update(observed_planned=observed, unresolved=m4['n_failed_or_unavailable'])
    if observed['hi'] is not None and observed['hi'] > GREY_FAR_UPPER_MAX:
        out.update(decision='FAIL', keep_claim=False)
    elif m4['n_failed_or_unavailable'] and upper is not None and upper > GREY_FAR_UPPER_MAX:
        out.update(decision='NOT_EVALUABLE', keep_claim=False)
    return out


# ------------------------------------------------------------------------------------------- M2
def m2_false_rejection(rows, snr_bins=None, min_cell=30, n_boot=2000, seed=0):
    """False rejection of blinded known planets (item 31.9). rows: case_outcome rows with in_domain (True, False or
    None), domain_reasons, crowdsap and optional snr.

    in_domain rate: 1 - accepted / (planets inside the declared domain). end_to_end rate: 1 - accepted / (all cohort
    planets), with the out-of-domain planets and the pre-search domain stops on their own design-exclusion lines.
    accepted = the reported CANDIDATE is the FUNDAMENTAL match of the reference ephemeris. A planet whose domain
    membership is unknown stays in the end-to-end denominator and is listed; it is not counted as inside."""
    n = len(rows)
    acc = lambda sel: sum(1 for r in sel if r["matched_candidate"])
    inside = [r for r in rows if r["in_domain"] is True]
    outside = [r for r in rows if r["in_domain"] is False]
    unknown = [r for r in rows if r["in_domain"] is None]

    def fr(sel):
        base = stats.rate(len(sel) - acc(sel), len(sel), "two-sided")
        base["accepted"] = acc(sel)
        if all('host' in r for r in sel):
            flagged = [dict(r, _loss=not r['matched_candidate']) for r in sel]
            _with_bootstrap(base, flagged, '_loss', 'host', n_boot, seed)
            base['supported'] = len({r['host'] for r in sel}) >= min_cell
        else:
            base.update(n_hosts=None, supported=None)
        return base
    reasons = {}
    for r in outside:
        for why in r.get("domain_reasons", []):
            reasons[why] = reasons.get(why, 0) + 1
    out = dict(n_cohort=n, in_domain=fr(inside), end_to_end=fr(rows),
               design_exclusion=dict(out_of_domain_planets=len(outside), by_reason=reasons,
                                     pre_search_domain_stops={s: sum(1 for r in rows if r["status"] == s) for s in HOST_DOMAIN_STOPS},
                                     domain_unknown=len(unknown)),
               lines=_status_lines(rows),
               by_crowding={k: fr(v) for k, v in sorted(_strata(inside, lambda r: crowd_stratum(r["crowdsap"])).items())},
               review_subrows={s: sum(1 for r in rows if r.get("review_subrow") == s) for s in REVIEW_SUBROWS},
               centroid_not_evaluated=sum(1 for r in rows if r.get("centroid_not_evaluated")))
    if snr_bins:
        out["by_snr"] = {}
        for lo, hi in snr_bins:
            sel = [r for r in inside if r.get("snr") is not None and lo <= r["snr"] < hi]
            out["by_snr"][f"[{lo:g}, {hi:g})"] = fr(sel)
        out["by_snr"]["SNR unavailable"] = fr([r for r in inside if r.get("snr") is None])
    out['by_multiplicity'] = {}
    for name, number in (('single-planet', 1), ('multi-planet', 2)):
        sel = [r for r in rows if r.get('n_reference_planets') is not None and
               (r['n_reference_planets'] == 1 if number == 1 else r['n_reference_planets'] > 1)]
        out['by_multiplicity'][name] = dict(end_to_end=fr(sel), in_domain=fr([r for r in sel if r['in_domain'] is True]))
    out['multiplicity_unavailable'] = sum(r.get('n_reference_planets') is None for r in rows)
    out['sibling_wins'] = dict(n=sum(bool(r.get('sibling_win')) for r in rows),
                              case_ids=[r['case_id'] for r in rows if r.get('sibling_win')],
                              unavailable=sum(r.get('sibling_win') is None for r in rows))
    return out


def m2_case_outcome(record, reference, times, host, frozen_references, *, sibling_references):
    """Match the target first; a different frozen reference can explain, but cannot rescue, its loss."""
    out = case_outcome(record, reference, times)
    if not frozen_references:
        out.update(host=host, n_reference_planets=None, sibling_win=None,
                   multiplicity_status='NOT_AVAILABLE', sibling_win_status='NOT_AVAILABLE')
        return out
    cand = record.get('candidate')
    sibling = False
    if cand is not None and not out['matched_candidate']:
        selected = [dict(rank=1, period_days=cand['period_days'], t0_btjd=cand['t0_btjd'],
                         duration_days=cand['duration_hours'] / 24, sde=cand['sde'])]
        sibling = any(matching.recover(selected, r, times)['outcome'] == 'MATCH_FUNDAMENTAL'
                      for r in sibling_references if r is not None)
    if not sibling and any(r is None for r in sibling_references):
        sibling = None
    out.update(host=host, n_reference_planets=len(frozen_references), sibling_win=sibling,
               multiplicity_status='OK', sibling_win_status='NOT_AVAILABLE' if sibling is None else 'OK')
    return out


def _status_lines(rows):
    lines = {}
    for r in rows:
        key = (r["status_class"], r["status"])
        lines[key] = lines.get(key, 0) + 1
    return [dict(status_class=k[0], status=k[1], n=v) for k, v in sorted(lines.items())]


# ------------------------------------------------------------------------------------------- M3
def m3_false_positive(rows):
    """Astrophysical false-positive burden (item 31.3). rows: case_outcome rows of verified EBs and false positives,
    each with crowdsap, and for a CANDIDATE the period relation of the reported candidate to the reference period
    (candidate_relation: '1:1', 'n/d' or None).

    The raw CANDIDATE rate counts every host that ends CANDIDATE, whatever the period (alias promotion included).
    Label rule: point estimate above 10.0 % or one-sided 95 % Clopper-Pearson upper bound above 15.0 % keeps the
    code frozen and labels the release output a preliminary transit candidate list."""
    n = len(rows)
    cands = [r for r in rows if r["is_candidate"]]
    raw = stats.rate(len(cands), n, "upper")
    relation = {}
    for r in cands:
        key = r.get("candidate_relation") or "unrelated period"
        relation[key] = relation.get(key, 0) + 1
    out = dict(n=n, raw_candidate=raw, candidate_period_relation=relation,
               alias_promotions=sum(v for k, v in relation.items() if k not in ("1:1", "unrelated period")),
               review=stats.rate(sum(1 for r in rows if r["status_class"] == "REVIEW"), n, "two-sided") if n else None,
               lines=_status_lines(rows),
               by_crowding={k: stats.rate(sum(1 for r in v if r["is_candidate"]), len(v), "upper")
                            for k, v in sorted(_strata(rows, lambda r: crowd_stratum(r["crowdsap"])).items())},
               review_subrows={s: sum(1 for r in rows if r.get("review_subrow") == s) for s in REVIEW_SUBROWS})
    bad = [r for r in rows if r["status_class"] in ("FAILED", "UNAVAILABLE")]
    if bad:
        out["failed_or_unavailable"] = [dict(case_id=r["case_id"], status=r["status"]) for r in bad]
        # Item 43.11: this reporting convention cannot resolve the release label.
        out["worst_case"] = stats.rate(len(cands) + len(bad), n, "upper")
    if raw["rate"] is None or bad:
        out["label_rule"] = dict(status="NOT_EVALUABLE", preliminary_label=None)
    else:
        over_point, over_upper = raw["rate"] > M3_POINT_LIMIT, raw["hi"] > M3_UPPER_LIMIT
        out["label_rule"] = dict(status="EVALUATED", point_above_limit=bool(over_point), upper_above_limit=bool(over_upper),
                                 preliminary_label=bool(over_point or over_upper), point_limit=M3_POINT_LIMIT,
                                 upper_limit=M3_UPPER_LIMIT)
    return out


# ------------------------------------------------------------------------------------------- M4
def _candidate_kind(record):
    cand = record.get("candidate")
    if cand is None:
        return None
    return "Multi-Epoch" if int(cand.get("n_archival_sectors", 0)) > 0 else "Single-Sector_Only"


def mode2_triggers(record):
    """Discovery-vetted triggers that entered archival Mode 2 in one run, each with its archival result.
    passed: the archival gate confirmed the trigger (status Confirmed_in_Sectors...). A REVIEW_ hold is reported as
    hold, not as a pass."""
    out = []
    for d in record.get("archival_diagnostics") or []:
        vr = d.get("vr") or {}
        if vr.get("mode") != 2:
            continue
        status = str(d.get("arch_status", ""))
        out.append(dict(case_id=record["case_id"], peak_rank=d.get("peak_rank"), arch_ok=d.get("arch_ok"),
                        passed=bool(d.get("arch_ok") is True and status.startswith("Confirmed_in_Sectors")),
                        hold=bool(d.get("arch_ok") is True and status.startswith("REVIEW_")),
                        not_evaluated=d.get("arch_ok") is None, status=status[:120]))
    return out


def m4_false_alarm(records, crowdsap_by_case, label):
    """Empirical false-alarm rates on a null arm (item 31.7), per star-sector, with one-sided 95 % upper bounds.

    (a) top peak SDE >= 6.0 and >= 7.5; (b) at least one peak passed every discovery gate and reached the archival
    stage; (c) final CANDIDATE, split into Multi-Epoch and Single-Sector_Only (SDE >= 6.5), and at SDE >= 6.0 and
    >= 7.5 for dFAR_grey; (d) Mode 2 triggers passing the archival gate, per trigger (use the N1-Archival arm).

    Denominators. `evaluated` leaves out runs that failed or had no data (ERROR, SKIP_NO_DATA); they are listed and
    must be rerun or explained before a result is quoted. Hosts stopped by a host-level gate, incomplete hosts
    (item 37.1) and capped REVIEW hosts (item 38.4) stay in the denominator and are never counted as CANDIDATE.
    `worst_case` counts every failed or unavailable run as an alarm in the planned denominator."""
    rows = []
    for rec in records:
        o = case_outcome(rec)
        o.update(crowdsap=crowdsap_by_case.get(rec["case_id"]), candidate_kind=_candidate_kind(rec),
                 triggers=mode2_triggers(rec))
        rows.append(o)
    planned = len(rows)
    evaluated = [r for r in rows if r["status_class"] not in ("FAILED", "UNAVAILABLE")]
    n_bad = planned - len(evaluated)

    def tables(sel, n_fail):
        n = len(sel)

        def r(flag):
            k = sum(1 for x in sel if flag(x))
            entry = stats.rate(k, n, "upper")
            entry["worst_case"] = stats.rate(k + n_fail, n + n_fail, "upper") if n + n_fail else None
            return entry
        top = lambda x, thr: x["top_sde"] is not None and x["top_sde"] >= thr
        cand = lambda x, thr: x["is_candidate"] and x["candidate_sde"] is not None and x["candidate_sde"] >= thr
        trig = [t for x in sel for t in x["triggers"]]
        a6, a75 = r(lambda x: top(x, SDE_PRODUCTION)), r(lambda x: top(x, SDE_GREY_UPPER))
        c6, c75 = r(lambda x: cand(x, SDE_PRODUCTION)), r(lambda x: cand(x, SDE_GREY_UPPER))
        return dict(
            n=n,
            a_top_peak_sde_6=a6, a_top_peak_sde_7_5=a75,
            b_reached_archival=r(lambda x: x["reached_archival"]),
            c_candidate=r(lambda x: x["is_candidate"]),
            c_candidate_multi_epoch=r(lambda x: x["candidate_kind"] == "Multi-Epoch"),
            c_candidate_single_sector=r(lambda x: x["candidate_kind"] == "Single-Sector_Only"),
            c_candidate_sde_6=c6, c_candidate_sde_7_5=c75,
            d_mode2=dict(per_trigger=stats.rate(sum(1 for t in trig if t["passed"]), len(trig), "upper"),
                         holds=sum(1 for t in trig if t["hold"]), not_evaluated=sum(1 for t in trig if t["not_evaluated"]),
                         hosts_with_trigger=sum(1 for x in sel if x["triggers"]),
                         hosts_with_pass=sum(1 for x in sel if any(t["passed"] for t in x["triggers"]))),
            dfar_grey=dict(candidate_level=stats.rate(c6["k"] - c75["k"], n, "upper"),
                           search_level=stats.rate(a6["k"] - a75["k"], n, "upper")))
    out = dict(arm=label, n_planned=planned, n_evaluated=len(evaluated), n_failed_or_unavailable=n_bad,
               pooled=tables(evaluated, n_bad), lines=_status_lines(rows),
               by_crowding={k: tables([r for r in v if r["status_class"] not in ("FAILED", "UNAVAILABLE")],
                                     sum(r["status_class"] in ("FAILED", "UNAVAILABLE") for r in v))
                            for k, v in sorted(_strata(rows, lambda r: crowd_stratum(r["crowdsap"])).items())},
               review_subrows={s: sum(1 for r in rows if r.get("review_subrow") == s) for s in REVIEW_SUBROWS},
               incomplete_lines={r["status"]: sum(1 for x in rows if x["status"] == r["status"])
                                 for r in rows if r["status_class"] == "INCOMPLETE"},
               candidates=[dict(case_id=r["case_id"], sde=r["candidate_sde"], kind=r["candidate_kind"]) for r in rows if r["is_candidate"]])
    return out


# ------------------------------------------------------------------------------------------- M5
def _wrap(dt, period):
    return ((dt + 0.5 * period) % period) - 0.5 * period


def m5_parameters(cases):
    """Parameter accuracy and uncertainty coverage of matched CANDIDATEs. cases: dicts with candidate (the reported
    record), truth (period, t0, depth as a fraction, k) and r_star (solar radii).

    Period and epoch have a production uncertainty (cramer_rao_sigma_p_sec, cramer_rao_sigma_t0_sec: the discovery
    Carter heuristic, also when the reported ephemeris is the refined one). Coverage is the share of truths inside
    1, 2 and 3 sigma, with two-sided Clopper-Pearson intervals, against the Gaussian reference levels. Depth and
    radius have no production interval: their coverage is NOT_AVAILABLE and only the ratio to the truth is reported.
    A missing or non-positive sigma is NOT_AVAILABLE, never replaced."""
    rows = []
    for c in cases:
        cand, truth = c["candidate"], c["truth"]
        p, t0 = float(cand["period_days"]), float(cand["t0_btjd"])
        sp, st = cand.get("cramer_rao_sigma_p_sec"), cand.get("cramer_rao_sigma_t0_sec")
        sp = float(sp) / 86400.0 if sp is not None and math.isfinite(float(sp)) and float(sp) > 0 else None
        st = float(st) / 86400.0 if st is not None and math.isfinite(float(st)) and float(st) > 0 else None
        dp = p - float(truth["period"])
        dt0 = _wrap(t0 - float(truth["t0"]), float(truth["period"]))
        true_depth_ppm = float(truth["depth"]) * 1e6
        true_rp = float(truth["k"]) * float(c["r_star"]) * RP_EARTH_PER_RSUN if c.get("r_star") is not None else None
        rows.append(dict(case_id=c["case_id"], ephemeris_source=cand.get("ephemeris_source"),
                         period_error_d=dp, t0_error_d=dt0, z_period=None if sp is None else abs(dp) / sp,
                         z_t0=None if st is None else abs(dt0) / st,
                         depth_ratio=float(cand["depth_ppm"]) / true_depth_ppm if true_depth_ppm > 0 else None,
                         radius_ratio=float(cand["r_planet_earth"]) / true_rp if true_rp else None))

    def coverage(key, sel):
        have = [r[key] for r in sel if r[key] is not None]
        out = dict(n=len(sel), n_with_sigma=len(have), not_available=len(sel) - len(have), levels={})
        for k, nominal in NOMINAL_COVERAGE.items():
            entry = stats.rate(sum(1 for z in have if z <= k), len(have), "two-sided")
            entry["nominal"] = nominal
            out["levels"][f"{k} sigma"] = entry
        return out

    def summary(vals, reference=0.0):
        vals = [v for v in vals if v is not None]
        if not vals:
            return dict(n=0, median=None, median_bias=None, p16=None, p84=None, p2_5=None, p97_5=None)
        return dict(n=len(vals), median=stats.median_fixed(vals), p16=stats.quantile_type7(vals, 0.16),
                    median_bias=stats.median_fixed(vals) - reference,
                    p84=stats.quantile_type7(vals, 0.84), p2_5=stats.quantile_type7(vals, 0.025),
                    p97_5=stats.quantile_type7(vals, 0.975))
    out = dict(n=len(rows), period=dict(error_d=summary(r["period_error_d"] for r in rows), coverage=coverage("z_period", rows)),
               t0=dict(error_d=summary(r["t0_error_d"] for r in rows), coverage=coverage("z_t0", rows)),
               depth=dict(ratio=summary((r["depth_ratio"] for r in rows), 1.0), coverage="NOT_AVAILABLE (no production interval)"),
               radius=dict(ratio=summary((r["radius_ratio"] for r in rows), 1.0), coverage="NOT_AVAILABLE (no production interval)"),
               by_ephemeris_source={}, rows=rows)
    for src, sel in sorted(_strata(rows, lambda r: str(r["ephemeris_source"])).items()):
        out["by_ephemeris_source"][src] = dict(n=len(sel), period=coverage("z_period", sel), t0=coverage("z_t0", sel))
    return out


# ------------------------------------------------------------------------------------------- item 31.4 diagnostics
def archival_secondary_diagnostics(ald, records, p_cat_by_case=None, sigma_p_cat_by_case=None):
    """Item 31.4 and 36.2 diagnostics over every archival circular-zone secondary evaluation of the given runs,
    read from the production record (vr.sci74). The frozen rule is unchanged; these are reports.

    archival_sec_flip_rate   share whose secondary SNR over P_ref, P_ref +/- one item 18 grid step and
                             P_ref +/- sigma_P,ref has maximum >= z_crit and minimum < z_crit (ref_sigma_flip)
    near_threshold           share with |SNR_sec - z_crit| <= 0.5
    refinement beyond 3.5 sigma: |P_ref - P_bls| / sigma_P,disc (M3) and, with a catalog or truth period,
                             |P_ref - P_cat| / sqrt(sigma_P,ref^2 + sigma_P,cat^2) (M2), from the production function
                             sci78_period_offsets; the ratios to sigma_P,ref and sigma_P,disc alone are kept per row.
    An evaluation whose points are missing is counted NOT_AVAILABLE in that line and left out of its denominator."""
    rows = []
    for rec in records:
        for d in rec.get("archival_diagnostics") or []:
            s = (d.get("vr") or {}).get("sci74")
            if not s:
                continue
            row = dict(case_id=rec["case_id"], peak_rank=d.get("peak_rank"), status=s.get("status"),
                       flip=s.get("ref_sigma_flip"), step_flip=s.get("step_flip"), near_threshold=s.get("near_threshold"),
                       bls_above=(s.get("sci78") or {}).get("bls_above_n_sigma"), cat_above=None,
                       cat_offset_ref_sigma=None, cat_offset_disc_sigma=None)
            if p_cat_by_case is not None and rec["case_id"] in p_cat_by_case:
                jr = next((j for j in rec.get("journal", []) if j.get("rank") == d.get("peak_rank")), None)
                ref = s.get("sigma_p_ref") or {}
                off = ald.sci78_period_offsets(
                    s.get("p_ref_d"), None if jr is None else jr.get("period_days"), s.get("sigma_p_disc_projected_d"),
                    ref.get("sigma_p_ref_d") if ref.get("status") == "DEFINED" else None,
                    p_cat=p_cat_by_case[rec["case_id"]],
                    sigma_p_cat_d=(sigma_p_cat_by_case or {}).get(rec["case_id"], 0.0))
                row.update(cat_above=off.get("cat_above_n_sigma"), cat_offset_ref_sigma=off.get("cat_offset_ref_sigma"),
                           cat_offset_disc_sigma=off.get("cat_offset_disc_sigma"),
                           cat_offset_combined_sigma=off.get("cat_offset_combined_sigma"))
            rows.append(row)

    def share(key):
        have = [r[key] for r in rows if r[key] is not None]
        entry = stats.rate(sum(1 for v in have if v), len(have), "two-sided")
        entry["not_available"] = len(rows) - len(have)
        return entry
    return dict(n_evaluations=len(rows), archival_sec_flip_rate=share("flip"), step_flip_rate=share("step_flip"),
                near_threshold=share("near_threshold"), refinement_above_3_5_sigma_from_p_bls=share("bls_above"),
                refinement_above_3_5_sigma_from_p_cat=share("cat_above"), rows=rows)


# Item 46: this assessment never redraws or relaxes the fixed seed, target or bounds.
def m1_draw_assessment(row, r_star, max_draws=50, max_t0_tries=20):
    if row['truth'] is not None:
        return dict(actual_drawn=True, verified_domain_exhaustion=False, accepted=True)
    allowed = {'NOISE_INVALID_START', 'NOISE_INVALID', 'OUTSIDE_VALID_RANGE',
               'NOT_CONVERGED_50', 'DEPTH_OUTSIDE_K_BRACKET'}
    geometry = {'BELOW_AR_FLOOR', 'OUTSIDE_SEARCH_DOMAIN', 'NON_TRANSITING_GEOMETRY'}
    k_range = {'DEPTH_OUTSIDE_K_BRACKET', 'OUTSIDE_VALID_RANGE'}
    attempts = row['attempts']; target = row['snr_target']
    tries = [t for a in attempts for t in a.get('tries', [])]
    caps = {'depth_below_180_ppm': 0, 'depth_above_30000_ppm': 0, 'rp_above_15_earth': 0}
    converged = []; domain_evidence = False; geometry_rejected = 0; k_stops = 0; event_rejections = 0
    valid = (row.get('status') == 'DRAW_FAILED' and len(attempts) == max_draws
             and [a.get('draw') for a in attempts] == list(range(max_draws))
             and math.isfinite(target) and target > 0)
    for a in attempts:
        tt = a.get('tries', [])
        valid = valid and a.get('snr_target') == target
        if a.get('draw_outcome') in geometry:
            geometry_rejected += 1
            valid = valid and not tt
        else:
            valid = valid and a.get('draw_outcome') == 'INNER_LOOP_TRIES_FAILED' and len(tt) == max_t0_tries
    for t in tries:
        if t.get('stop_reason') != 'CONVERGED':
            valid = valid and t.get('stop_reason') in allowed and t.get('valid') is False and t.get('try_outcome') == 'SOLVE_FAILED'
            if t.get('stop_reason') in k_range:
                k_stops += 1
                domain_evidence = True
            continue
        converged.append(t)
        depth = t.get('depth', float('nan')) * 1e6
        rp = t.get('k', float('nan')) * r_star * RP_EARTH_PER_RSUN
        achieved = t.get('snr_achieved', float('nan'))
        outside = math.isfinite(depth) and math.isfinite(rp) and (depth < 180 or depth > 30000 or rp > 15)
        solved = t.get('valid') is True and math.isfinite(achieved) and abs(achieved / target - 1) <= 0.05
        events = t.get('n_discovery_events')
        inside = math.isfinite(depth) and math.isfinite(rp) and 180 <= depth <= 30000 and 0 <= rp <= 15
        event_rejected = (inside and t.get('try_outcome') == 'T0_CONDITIONS'
                          and type(events) in (int, float) and math.isfinite(events)
                          and events == int(events) and 0 <= events < 3)
        event_rejections += int(event_rejected)
        valid = valid and solved and ((outside and t.get('try_outcome') in ('DEPTH_OUTSIDE_DOMAIN', 'RADIUS_OUTSIDE_DOMAIN')) or event_rejected)
        domain_evidence = domain_evidence or (outside and solved)
        caps['depth_below_180_ppm'] += int(depth < 180)
        caps['depth_above_30000_ppm'] += int(depth > 30000)
        caps['rp_above_15_earth'] += int(rp > 15)
    caps['k_outside_model_range'] = k_stops
    accepted = bool(valid and domain_evidence)
    return dict(actual_drawn=False, verified_domain_exhaustion=accepted, accepted=accepted,
                n_converged_out_of_domain=sum(1 for t in converged if t.get('try_outcome') in ('DEPTH_OUTSIDE_DOMAIN', 'RADIUS_OUTSIDE_DOMAIN')),
                n_solver_failures=len(tries) - len(converged), n_tries=len(tries), binding_caps=caps,
                n_geometry_rejected=geometry_rejected, n_k_range_stops=k_stops, n_discovery_event_rejections=event_rejections)

def m1_draw_failure_hosts(draws, hosts):
    table = []
    for h in hosts:
        failed = []
        for row in draws[h['tic_id']]:
            if row['truth'] is None:
                failed.append(dict(snr_bin=row['snr_bin'], snr_target=row['snr_target'],
                                   **m1_draw_assessment(row, h['stellar']['r_star'])))
        if failed:
            table.append(dict(tic_id=h['tic_id'], r_star=h['stellar']['r_star'], teff=h['stellar']['teff'],
                              sector_count=len(h['lightcurves']), baseline=h.get('discovery_baseline_days'),
                              failed_families=failed))
    return table
