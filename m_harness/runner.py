"""Cohort runner with the S3.6 bound resume, and the end-to-end dry run on synthetic fixtures.

run_cases() is the long runner of the M measurement. Its run record is a run_identity.BoundRun: the SHA256 of the
harness and detector code, of the configuration and preregistration, and of every input file are bound into it, a
resume reuses a finished case record only when all three still match, and otherwise it stops. Interruptions and
resumes are logged in the record, a partial case record is refused, and the detector output of an interrupted case
is cleared and the case is run again.

    python -I -B -m m_harness.runner dry-run --out DIR [--hosts N] [--backend reference|pytransit]

The dry run builds a small synthetic cohort and carries it through preflight, the null fidelity pre-gate, injection,
the detector and every M table. Its numbers are ENGINEERING CHECKS of the pipeline from cohort list to report. They
are not measurements of the detector and must never be quoted as scientific results.
"""
import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import HARNESS_VERSION, estimands, fidelity, fixtures, injection, matching, nulls, preflight, prodpath, routing
from . import transit_model as TM

ENGINEERING_LABEL = ("ENGINEERING CHECK ON SYNTHETIC FIXTURES. Not a measurement of the detector; "
                     "no number below is a scientific result.")
DRAFT_PERIOD_BINS = ((1.0, 2.0), (2.0, 5.0), (5.0, 13.0))
DRAFT_RADIUS_BINS = ((0.0, 0.8), (0.8, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 15.0))


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def harness_code_files(repo_dir):
    """Code bound into the run identity: the harness and the detector with its local dependencies."""
    import run_identity as ri
    repo_dir = Path(repo_dir)
    files = sorted(str(p) for p in (repo_dir / "m_harness").glob("*.py"))
    return files + ri.discover_local_code_dependencies(repo_dir, ["aldebaran.py"])


def job_inputs(job):
    root = Path(job["root"])
    paths = [str(root / r["path"]) for r in job["case"]["lightcurves"]]
    if job["case"]["tpf_fits"] is not None:
        paths.append(str(root / job["case"]["tpf_fits"]))
    return paths


def run_cases(ald, *, run_dir, run_id, jobs, config, repo_dir, preregistration_path=None, resume=True, on_case=None):
    """Run every job through the detector with bound resume. jobs: dict(case, root, arm, originals (optional list
    of SHA256 of the uninjected files)). Returns dict(records by case_id, reused, ran, bound run data).

    on_case(case_id, n_done) is called after each case is saved; the tests use it to interrupt the run."""
    import run_identity as ri
    if not preregistration_path or not Path(preregistration_path).is_file():
        raise ValueError("a preregistration file is required before creating or resuming a run")
    for job in jobs:
        if job['arm'] in ('M1', 'M2', 'M3', 'N1', 'N2') and job['case'].get('known_transits') != []:
            raise ValueError('M_RUN_REQUIRES_EMPTY_KNOWN_SIGNAL_CONTEXT')
    run_dir = Path(run_dir)
    ids = [j["case"]["case_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case_id in jobs")
    spec = [dict(case_id=j["case"]["case_id"], arm=j["arm"], case=j["case"]) for j in jobs]
    bound = ri.BoundRun.open_or_resume(
        run_dir / "run_identity.json", run_id=run_id, runner=f"m_harness.runner ({HARNESS_VERSION})",
        code_files=harness_code_files(repo_dir), config=config, preregistration_path=preregistration_path,
        input_paths=sorted({p for j in jobs for p in job_inputs(j)}), input_spec=spec, repo_dir=repo_dir, resume=resume)
    records, reused, ran = {}, [], []
    try:
        for job in jobs:
            cid = job["case"]["case_id"]
            rec_path = run_dir / "records" / f"{cid}.json"
            det_dir = run_dir / "detector" / cid
            if rec_path.exists():
                # Identity-bound reuse; a truncated or stale record raises RunIdentityError and stops the run.
                item = bound.load_verified_item_record(rec_path, required_keys=("case_id", "arm", "record", "routing_check"))
                if item.get("times_path"):
                    item = bound.load_verified_item_record(rec_path, required_keys=("case_id", "arm", "record"),
                                                           artifact_fields={"times_path": "times_sha256"})
                records[cid] = item
                reused.append(cid)
                continue
            if det_dir.exists():
                shutil.rmtree(det_dir)                  # this run's own output of an interrupted case
                bound.data.setdefault("partial_outputs_cleared", []).append(cid)
                bound.save()
            attempt_kw = dict(attempt_id=job['attempt_id']) if job.get('attempt_id') else {}
            record = routing.run_case(ald, job["case"], job["root"], det_dir, run_id, **attempt_kw)
            check = routing.verify_routing(record, job["case"], originals=job.get("originals"))
            times_path = None if record["search"]["times_file"] is None else str(det_dir / record["search"]["times_file"])
            payload = dict(case_id=cid, arm=job["arm"], record=record, routing_check=check,
                           times_path=times_path, times_sha256=record["search"]["times_sha256"])
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            bound.save_item_record(rec_path, payload)
            records[cid] = _load_json(rec_path)
            ran.append(cid)
            if on_case is not None:
                on_case(cid, len(reused) + len(ran))
    except (KeyboardInterrupt, SystemExit) as exc:
        bound.record_interruption(type(exc).__name__, completed_count=len(reused) + len(ran))
        raise
    bound.record_completion("COMPLETE", extra=dict(n_cases=len(jobs), n_reused=len(reused), n_ran=len(ran)))
    return dict(records=records, reused=reused, ran=ran, run=bound.data)


def load_times(item):
    return None if not item.get("times_path") else np.load(item["times_path"])


def load_m2_reference_catalog(manifest, path=None):
    """Read only the frozen metadata sidecar; a missing or changed catalog is unavailable."""
    try:
        binding = manifest['scoring_sidecars']['M2']
        data = Path(path or binding['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != binding['sha256']:
            return None
        return json.loads(data)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def m2_table_rows(items, design, manifest, reference_catalog=None, reference_path=None, include_missing_truth=False):
    """Join every host reference gate to its frozen ephemeris, without reading photometry."""
    if reference_catalog is None:
        reference_catalog = load_m2_reference_catalog(manifest, reference_path)
    rows = []
    for cid, d in design.items():
        if d['truth'] is None and not include_missing_truth:
            continue
        references, siblings, gates = None, [], []
        try:
            gates = manifest['M2_reference_gates'][str(d['host'])]
            if not isinstance(gates, list) or not all(isinstance(g, dict) for g in gates):
                gates = []
                raise ValueError('M2_REFERENCE_GATES_INVALID')
            catalog = reference_catalog['hosts'][str(d['host'])]['reference_planets']
            by_name = {r['catalog']['pl_name']: r for r in catalog}
            if not gates or len(by_name) != len(catalog) or {g['planet'] for g in gates} != set(by_name):
                raise ValueError('M2_REFERENCE_SET_MISMATCH')
            target_planet = d.get('target_planet')
            if target_planet not in by_name:
                raise ValueError('M2_TARGET_IDENTITY_UNAVAILABLE')
            references = []
            for gate in gates:
                entry = by_name[gate['planet']]
                ref = entry.get('selected_reference')
                if not entry['reference_status'].startswith('VALID_') or ref is None:
                    references.append(None)
                    if gate['planet'] != target_planet:
                        siblings.append(None)
                    continue
                p, t0, dur = (float(ref[k]) for k in ('period_days', 'epoch_btjd', 'T14_hours'))
                if not np.all(np.isfinite([p, t0, dur])) or p <= 0 or dur <= 0:
                    raise ValueError('M2_REFERENCE_QUANTITIES_INVALID')
                references.append(dict(period=p, t0=t0, t14=dur / 24))
                if gate['planet'] != target_planet:
                    siblings.append(references[-1])
        except (KeyError, TypeError, ValueError):
            references = None
        o = estimands.m2_case_outcome(items[cid]['record'], d['truth'], load_times(items[cid]),
                                      d['host'], references, sibling_references=siblings)
        if d['truth'] is None:
            o.update(stage='SEARCH', line='MATCH_UNAVAILABLE', match_outcome='MATCH_UNAVAILABLE',
                     matched_candidate=False, matched_candidate_or_review=False, search_recovered=False)
        o.update(crowdsap=d['crowdsap'], in_domain=d['in_domain'], domain_reasons=d['domain_reasons'], snr=d['snr'])
        # Derive the clearer display label; never rewrite the frozen gates.
        o['reference_gates'] = [dict(g, SNR_status='NO_DISCOVERY_EVENT')
                                if g.get('domain', {}).get('in_domain') is True and
                                g.get('snr_support', {}).get('N_ev') == 0 else dict(g) for g in gates]
        rows.append(o)
    return rows


# ----------------------------------------------------------------------------------------------- dry run
def _case_from_host(h, case_id):
    return dict(tic_id=h["tic_id"], case_id=case_id, sector=h["sector"], source_fits=h["source_fits"],
                source_sha256=h["source_sha256"], tpf_fits=h["tpf_fits"], tpf_sha256=h["tpf_sha256"],
                stellar=h["stellar"], known_transits=[], lightcurves=[dict(r) for r in h["lightcurves"]])


def _copy_case(case, src_root, dst_root, case_id):
    """Copy a case's files under dst_root / case_id and return the case pointing at the copies."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    new = dict(case, case_id=case_id, lightcurves=[])
    for r in case["lightcurves"]:
        dst = dst_root / case_id / Path(r["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_root / r["path"], dst)
        new["lightcurves"].append(dict(r, path=str(dst.relative_to(dst_root)).replace("\\", "/")))
    tdst = dst_root / case_id / Path(case["tpf_fits"]).name
    shutil.copyfile(src_root / case["tpf_fits"], tdst)
    new["tpf_fits"] = str(tdst.relative_to(dst_root)).replace("\\", "/")
    disc = next(x for x in new["lightcurves"] if int(x["sector"]) == int(case["sector"]))
    new["source_fits"] = disc["path"]
    return new


def _label(bins, value):
    for i, (lo, hi) in enumerate(bins):
        if lo <= value < hi or (i == len(bins) - 1 and value == hi):
            return f"[{lo:g}, {hi:g}{']' if i == len(bins) - 1 else ')'}"
    return "outside the declared bins"


def _add_secondary(path, period, t0, duration, depth):
    """Fixture only: a box dip of the given depth at phase 0.5 in PDCSAP_FLUX, to make an EB-like light curve."""
    from astropy.io import fits
    with fits.open(path, mode="update", memmap=False) as h:
        t = np.asarray(h[1].data["TIME"], float)
        ph = ((t - t0) / period) % 1.0
        dip = np.abs(ph - 0.5) <= 0.5 * duration / period
        f = np.asarray(h[1].data["PDCSAP_FLUX"], float)
        f[dip] *= (1.0 - depth)
        h[1].data["PDCSAP_FLUX"] = f
        h.flush()


def _host_set(root, first_tic, n, seed, sigma, **special):
    hosts = []
    for i in range(n):
        archival = () if i % 5 == 4 else (12, 13)
        crowd = 0.80 if i % 4 == 1 else 0.95
        hosts.append(fixtures.synthetic_host(root, first_tic + i, 40 + i % 3, archival_sectors=archival, seed=seed + i,
                                             sigma=sigma, crowdsap=special.get("crowdsap", {}).get(i, crowd),
                                             n_segments=3 if i % 3 == 2 else 2))
    return hosts


def dry_run(out_dir, n_hosts=10, backend="reference", seed=20261114, repo_dir=None, stop_after=None, snr_bins=None):
    """End-to-end engineering run on synthetic fixtures. Returns the summary that is also written to out_dir.
    snr_bins restricts the M1 bins (the tests use two bins to stay short); None uses the draft design."""
    out_dir = Path(out_dir)
    repo_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent
    ald = prodpath.load_production()
    t_start = time.time()
    design = dataclasses.replace(injection.DRAFT_DESIGN, backend=backend,
                                 label=injection.DRAFT_DESIGN.label + " [dry run on fixtures]",
                                 snr_bins=tuple(snr_bins) if snr_bins else injection.DRAFT_DESIGN.snr_bins)
    bins = design.snr_bins
    bin_labels = [injection.bin_label(lo, hi) for lo, hi in bins]
    hosts_root = out_dir / "inputs" / "hosts"
    sigma = 1.5e-3
    m1_hosts = _host_set(hosts_root, 910000, n_hosts, seed, sigma)
    m4_hosts = _host_set(hosts_root, 940000, n_hosts, seed + 1000, sigma)
    m2_hosts = _host_set(hosts_root, 920000, 4, seed + 2000, sigma, crowdsap={3: 0.60})
    m3_hosts = _host_set(hosts_root, 930000, 3, seed + 3000, sigma)
    by_tic = {h["tic_id"]: h for h in m1_hosts + m4_hosts + m2_hosts + m3_hosts}
    summary = dict(label=ENGINEERING_LABEL, harness=HARNESS_VERSION, backend=backend, design=design.as_dict(),
                   n_hosts=dict(m1=len(m1_hosts), m4=len(m4_hosts), m2=len(m2_hosts), m3=len(m3_hosts)), steps={})

    # 1. Preflight (C14). The M4 null hosts and the known-planet benchmark are held out: metadata only.
    draws = {}
    target_plan = injection.m1_target_plan([h['tic_id'] for h in m1_hosts], design)
    summary['target_plan'] = {str(h): {f'{b}:{s}': t for (b, s), t in v.items()} for h, v in target_plan.items()}

    def draw(cohort, host):
        slots = injection.draw_m1_host(ald, by_tic[host], hosts_root, design, [h["tic_id"] for h in m1_hosts].index(host),
                                       target_plan[host])
        draws[host] = slots
        return {s["snr_bin"] + f" slot {s['slot']}": dict(
            drawn=estimands.m1_draw_assessment(s, by_tic[host]['stellar']['r_star'], design.max_draws, design.max_t0_tries)['accepted'],
            attempts=len(s['attempts']), **estimands.m1_draw_assessment(s, by_tic[host]['stellar']['r_star'], design.max_draws, design.max_t0_tries)) for s in slots}
    cohorts = dict(m1_injection=dict(blinded=False, root=str(hosts_root), hosts=m1_hosts),
                   m4_null=dict(blinded=True, root=str(hosts_root), hosts=m4_hosts),
                   m2_known_planets=dict(blinded=True, root=str(hosts_root), hosts=m2_hosts),
                   m3_eclipsing_binaries=dict(blinded=True, root=str(hosts_root), hosts=m3_hosts))
    pre_out = next(out_dir / name for name in ["preflight.json"] + [f"preflight_resume_{k}.json" for k in range(1, 1000)]
                   if not (out_dir / name).exists())       # each preflight writes a new file; a resume gets its own
    pre = preflight.run(source=repo_dir / "aldebaran.py", cohorts=cohorts, out=pre_out,
                        allow_gate_rejects=("m2_known_planets",), max_attempts=design.max_draws,
                        draw=lambda c, h: draw(c, h) if c == "m1_injection" else {},
                        note="dry run on synthetic fixtures; gate rejects allowed in the known-planet benchmark, "
                             "where an out-of-domain planet is a design-exclusion line")
    summary["steps"]["preflight"] = dict(passed=pre["passed"], failures=pre["failures"], m_preflight=pre["m_preflight"],
                                         cohorts={k: dict(n_hosts=v["n_hosts"], n_passed=v["n_passed"], blinded=v["blinded"],
                                                          gate_rejects={g: x for g, x in v["gate_rejects"].items() if x})
                                                  for k, v in pre["cohorts"].items()})
    if not pre["passed"]:
        summary["status"] = "STOPPED_PREFLIGHT_FAILED"
        _write_summary(out_dir, summary)
        return summary

    # 2. M1 injections: fidelity (exact identities) and the item 31.5 model check for every drawn planet.
    inj_root = out_dir / "inputs" / "injections"
    jobs, m1_design, planned = [], {}, []
    fid_fail, model_fail = [], []
    for hi, h in enumerate(m1_hosts):
        originals = [r["sha256"] for r in h["lightcurves"]] + [h["tpf_sha256"]]
        for s in draws[h["tic_id"]]:
            cid = f"m1_h{hi:03d}_b{s['snr_bin_index']}_s{s['slot']}"
            row = dict(case_id=cid, host=h["tic_id"], crowdsap=h["crowdsap"], snr_bin=s["snr_bin"], truth=s["truth"],
                       draw_status=s["status"], snr_target=s['snr_target'], period_population=s['period_population'])
            planned.append(row)
            if s["truth"] is None:
                continue
            case, tinfo = injection.write_injection(h, hosts_root, cid, s["truth"], inj_root, fixtures.sha256_file)
            if case is None:
                row["draw_status"] = "INJECTION_FAILED"
                continue
            fid = injection.fidelity(h, hosts_root, case, inj_root, tinfo, s["truth"], fixtures.sha256_file)
            chk = TM.model_check(s["truth"], backend)
            if not fid["passed"]:
                fid_fail.append(cid)
            if not chk["passed"]:
                model_fail.append(cid)
            row.update(fidelity_passed=fid["passed"], model_check=chk)
            m1_design[cid] = row
            jobs.append(dict(case=case, root=str(inj_root), arm="M1", originals=originals))
    summary["steps"]["m1_injection"] = dict(n_planned=len(planned), n_injected=len(m1_design),
                                            realized_grey_counts=injection.realized_grey_counts(planned),
                                            conditional_population=[dict(host=r['host'], **r['period_population'])
                                                                    for r in planned if r['snr_bin'] == bin_labels[0]],
                                            draw_failed=[r["case_id"] for r in planned if r["draw_status"] != "DRAWN"],
                                            fidelity_failed=fid_fail, model_check_failed=model_fail,
                                            max_t14_diff_s=max((r["model_check"]["t14_contacts_minus_formula_s"] for r in m1_design.values()), default=None))

    # 3. Null arms. The noise-only pre-gate (item 44) is evaluated before any null detector run.
    null_root = out_dir / "inputs" / "nulls"
    noise_rows, n1_cases = [], {}
    import lightkurve as lk
    for hi, h in enumerate(m4_hosts):
        base = _case_from_host(h, f"n2_h{hi:03d}")
        n1 = _copy_case(base, hosts_root, null_root, f"n1_h{hi:03d}")
        tinfo = nulls.transform_case(n1, null_root, fixtures.sha256_file)
        n1_cases[h["tic_id"]] = (n1, tinfo)
        if tinfo["ok"]:
            t_u, x_u = fidelity.production_arrays(ald, lk.read(str(hosts_root / h["source_fits"])))
            t_n, x_n = fidelity.production_arrays(ald, lk.read(str(null_root / n1["source_fits"])))
            noise_rows.append(dict(fidelity.star_row(h["tic_id"], t_u, x_u, t_n, x_n), sector=h["sector"]))
    pre_gate = fidelity.fidelity_gate(noise_rows, [h["tic_id"] for h in m4_hosts], m4=True)
    summary["steps"]["m4_noise_pre_gate"] = pre_gate
    crowd = {}
    if not pre_gate["blocks_m4"]:
        for hi, h in enumerate(m4_hosts):
            n1, tinfo = n1_cases[h["tic_id"]]
            jobs.append(dict(case=n1, root=str(null_root), arm="N1"))
            jobs.append(dict(case=_case_from_host(h, f"n2_h{hi:03d}"), root=str(hosts_root), arm="N2"))
            crowd[f"n1_h{hi:03d}"] = crowd[f"n2_h{hi:03d}"] = h["crowdsap"]

    # 4. Development-style signal destruction: N1 applied after injection on the strongest planet of each M1 host.
    dev_root = out_dir / "inputs" / "n1_after_injection"
    dev_planets = {}
    for hi, h in enumerate(m1_hosts):
        strongest = [r for r in m1_design.values() if r["host"] == h["tic_id"]]
        if not strongest:
            continue
        row = max(strongest, key=lambda r: r["truth"]["snr_target"])
        src = next(j["case"] for j in jobs if j["case"]["case_id"] == row["case_id"])
        dev = _copy_case(src, inj_root, dev_root, f"dev_h{hi:03d}")
        tinfo = nulls.transform_case(dev, dev_root, fixtures.sha256_file)
        dev_planets[dev["case_id"]] = dict(period=row["truth"]["period"], transform_ok=tinfo["ok"])
        if tinfo["ok"]:
            jobs.append(dict(case=dev, root=str(dev_root), arm="N1_AFTER_INJECTION"))

    # 5. M2 blinded known-planet benchmark and M3 EB-like hosts (fixtures: an injected planet stands in for a known
    #    planet with its truth as the reference ephemeris; a deep eclipse with a secondary stands in for an EB).
    m2_design, m3_design = {}, {}
    strong = dataclasses.replace(design, snr_bins=((25.0, 35.0),), n_per_bin=1)
    for hi, h in enumerate(m2_hosts):
        cid = f"m2_h{hi:03d}"
        truth, _att = injection.draw_planet(np.random.default_rng(seed + 5000 + hi), 25.0, 35.0,
                                            injection.host_context(ald, h, hosts_root), h["ld"], strong)
        if truth is None:
            m2_design[cid] = dict(case_id=cid, host=h["tic_id"], crowdsap=h["crowdsap"], truth=None)
            continue
        case, _ = injection.write_injection(h, hosts_root, cid, truth, inj_root, fixtures.sha256_file)
        dom = estimands.domain_check(dict(crowdsap=h["crowdsap"], ruwe=h["stellar"]["ruwe"], r_star=h["stellar"]["r_star"],
                                          teff=h["stellar"]["teff"], depth_ppm=truth["depth"] * 1e6,
                                          rp_earth=truth["k"] * h["stellar"]["r_star"] * estimands.RP_EARTH_PER_RSUN))
        m2_design[cid] = dict(case_id=cid, target_planet=cid, host=h["tic_id"], crowdsap=h["crowdsap"], truth=truth, snr=truth["snr_achieved"],
                              in_domain=dom["in_domain"], domain_reasons=dom["reasons"])
        jobs.append(dict(case=case, root=str(inj_root), arm="M2"))
    for hi, h in enumerate(m3_hosts):
        cid = f"m3_h{hi:03d}"
        ars = TM.a_over_rs(2.3 + 0.4 * hi, h["stellar"]["m_star"], h["stellar"]["r_star"])
        truth = dict(period=2.3 + 0.4 * hi, k=0.10, b=0.3, a_rs=ars, inc=TM.inclination(0.3, ars), ld=h["ld"], backend=backend)
        truth["t0"] = float(fixtures.sector_times(h["sector"])[0][0] + 0.9)
        truth["t14"] = TM.t14(truth["period"], truth["k"], truth["b"], ars)
        case, _ = injection.write_injection(h, hosts_root, cid, truth, inj_root, fixtures.sha256_file)
        for r in case["lightcurves"]:
            _add_secondary(inj_root / r["path"], truth["period"], truth["t0"], truth["t14"], 0.004)
            r["sha256"] = fixtures.sha256_file(inj_root / r["path"])
        case["source_sha256"] = next(x for x in case["lightcurves"] if int(x["sector"]) == int(case["sector"]))["sha256"]
        m3_design[cid] = dict(case_id=cid, host=h["tic_id"], crowdsap=h["crowdsap"], reference_period=truth["period"])
        jobs.append(dict(case=case, root=str(inj_root), arm="M3"))

    # 6. Detector runs with bound resume.
    config = dict(harness=HARNESS_VERSION, design=design.as_dict(), purpose="dry run on synthetic fixtures", seed=seed)
    state = dict(n=0)

    def on_case(cid, n_done):
        if stop_after is not None and n_done >= stop_after and state["n"] == 0:
            state["n"] = 1
            raise KeyboardInterrupt("dry run interrupted on request")
    # Bind the synthetic exercise to its own declaration, never to a scientific authorization.
    declaration = out_dir / "SYNTHETIC_DESIGN.json"
    if not declaration.exists():
        declaration.write_text(json.dumps(dict(label=ENGINEERING_LABEL, config=config), sort_keys=True), encoding="utf-8")
    res = run_cases(ald, run_dir=out_dir / "run", run_id="m_dry_run", jobs=jobs, config=config, repo_dir=repo_dir,
                    preregistration_path=declaration, on_case=on_case)
    items = res["records"]
    runtimes = [items[j["case"]["case_id"]]["record"]["runtime_s"] for j in jobs]
    routing_bad = [cid for cid, it in items.items() if not it["routing_check"]["passed"]]
    summary["steps"]["detector_runs"] = dict(
        n_jobs=len(jobs), n_ran=len(res["ran"]), n_reused=len(res["reused"]), run_status=res["run"]["status"],
        interruptions=len(res["run"]["interruptions"]), resumes=len(res["run"]["resumes"]),
        routing_failed=routing_bad, by_arm={a: sum(1 for j in jobs if j["arm"] == a) for a in sorted({j["arm"] for j in jobs})},
        runtime_s=dict(n=len(runtimes), median=float(np.median(runtimes)), p90=float(np.percentile(runtimes, 90)),
                       max=float(np.max(runtimes)), total=float(np.sum(runtimes))))

    summary["steps"]["m4_acf_tail"] = fidelity.m4_tail_report(
        noise_rows, [h["tic_id"] for h in m4_hosts],
        {h["tic_id"]: estimands.case_outcome(items[f"n1_h{hi:03d}"]["record"])
         for hi, h in enumerate(m4_hosts) if f"n1_h{hi:03d}" in items})

    # 7. Tables.
    tables = {}
    m1_rows = []
    for row in planned:
        cid = row["case_id"]
        if cid not in m1_design:
            tr = row['truth']
            rp = None if tr is None else tr['rp_earth']
            m1_rows.append(dict(case_id=cid, stage="PLANNING", line=row["draw_status"], host=row["host"],
                                crowdsap=row["crowdsap"], snr_bin=row["snr_bin"], search_recovered=False,
                                matched_candidate=False, matched_candidate_or_review=False, candidate_sde=None,
                                matched_sde=None, snr_target=row['snr_target'], snr_achieved=None,
                                period_bin='not drawn' if tr is None else _label(DRAFT_PERIOD_BINS, tr['period']),
                                radius_bin='not drawn' if rp is None else _label(DRAFT_RADIUS_BINS, rp), rp_true_earth=rp))
            continue
        it, tr = items[cid], row["truth"]
        o = estimands.case_outcome(it["record"], tr, load_times(it))
        rp = tr["k"] * by_tic[row["host"]]["stellar"]["r_star"] * estimands.RP_EARTH_PER_RSUN
        o.update(host=row["host"], crowdsap=row["crowdsap"], snr_bin=row["snr_bin"], snr_target=tr["snr_target"],
                 snr_achieved=tr["snr_achieved"], period_bin=_label(DRAFT_PERIOD_BINS, tr["period"]),
                 radius_bin=_label(DRAFT_RADIUS_BINS, rp), rp_true_earth=rp)
        m1_rows.append(o)
    tables["m1_attrition"] = estimands.attrition_table(m1_rows, len(planned))
    tables["m1_draw_failure_hosts"] = estimands.m1_draw_failure_hosts(draws, m1_hosts)
    for level in ("candidate", "search"):
        tables[f"m1_completeness_{level}_level"] = estimands.m1_completeness(
            m1_rows, bin_labels, level=level, draw_records=draws)
        tables[f"grey_zone_gain_{level}_level"] = estimands.grey_zone_gain(
            m1_rows, "snr_target", level=level, bin_edges=[b for b in bins if b[0] < 10.0])
        tables[f"grey_zone_gain_{level}_level"]['aggregation_equivalence_verified'] = (
            summary['steps']['m1_injection']['realized_grey_counts']['passed'])
    if not pre_gate["blocks_m4"]:
        n1 = [items[c]["record"] for c in sorted(items) if items[c]["arm"] == "N1"]
        n2 = [items[c]["record"] for c in sorted(items) if items[c]["arm"] == "N2"]
        tables["m4_n1"] = estimands.m4_false_alarm(n1, crowd, "N1 (with N1-Archival)")
        tables["m4_n2"] = estimands.m4_false_alarm(n2, crowd, "N2 (unmodified)")
        tables["grey_zone_decision"] = estimands.grey_zone_m4_decision(
            tables["grey_zone_gain_candidate_level"]["pooled"]["dc_grey"]["rate"],
            tables["m4_n1"])
    else:
        tables["m4_n1"] = tables["m4_n2"] = dict(status="NOT_RUN", reason="noise-only pre-gate did not pass")
        tables["grey_zone_decision"] = estimands.grey_zone_decision(None, None)
    rec_rows = []
    for cid, d in dev_planets.items():
        if cid in items:
            recovered, err = fidelity.recovered_after_n1(items[cid]["record"], d["period"])
        else:
            recovered, err = True, True                 # the transform could not run: counted as recovered
        rec_rows.append(dict(planet=cid, period_days=d["period"], recovered=recovered, error=err))
    tables["development_signal_destruction"] = fidelity.recovery_gate(rec_rows, sorted(dev_planets))
    # This fixture catalog declares one injected stand-in per host; real tables use the bound frozen sidecar.
    fixture_manifest = dict(M2_reference_gates={})
    fixture_catalog = dict(hosts={})
    for cid, d in m2_design.items():
        if d['truth'] is not None:
            tr, host = d['truth'], str(d['host'])
            fixture_manifest['M2_reference_gates'][host] = [dict(planet=cid)]
            fixture_catalog['hosts'][host] = dict(reference_planets=[dict(catalog=dict(pl_name=cid),
                reference_status='VALID_SYNTHETIC', selected_reference=dict(
                    period_days=tr['period'], epoch_btjd=tr['t0'], T14_hours=tr['t14'] * 24))])
    m2_rows = m2_table_rows(items, m2_design, fixture_manifest, fixture_catalog)
    tables['m2_reference_rows'] = m2_rows
    tables["m2_false_rejection"] = estimands.m2_false_rejection(m2_rows, snr_bins=((0.0, 10.0), (10.0, 25.0), (25.0, 1e9)))
    m3_rows = []
    for cid, d in m3_design.items():
        o = estimands.case_outcome(items[cid]["record"])
        cand = items[cid]["record"]["candidate"]
        o.update(crowdsap=d["crowdsap"], candidate_relation=None if cand is None else
                 matching.period_relation(cand["discovery_period_days"], d["reference_period"]))
        m3_rows.append(o)
    tables["m3_false_positive"] = estimands.m3_false_positive(m3_rows)
    matched = [dict(case_id=r["case_id"], candidate=items[r["case_id"]]["record"]["candidate"], truth=m1_design[r["case_id"]]["truth"],
                    r_star=by_tic[r["host"]]["stellar"]["r_star"]) for r in m1_rows if r["matched_candidate"]]
    tables["m5_parameters"] = estimands.m5_parameters(matched)
    tables["item_31_4_diagnostics_m2_m3"] = estimands.archival_secondary_diagnostics(
        ald, [items[c]["record"] for c in sorted(items) if items[c]["arm"] in ("M2", "M3")],
        p_cat_by_case={cid: d["truth"]["period"] for cid, d in m2_design.items() if d["truth"] is not None})
    summary["tables"] = tables
    summary["elapsed_s"] = round(time.time() - t_start, 1)
    summary["status"] = "COMPLETE"
    _write_summary(out_dir, summary)
    return summary


def _fmt(rate):
    if rate is None or rate.get("rate") is None:
        return "not available (empty denominator)"
    return f"{rate['k']} of {rate['n']} = {100 * rate['rate']:.1f} % (95 % {rate['lo'] * 100:.1f} to {rate['hi'] * 100:.1f} %)"


def m1_report_notes(table):
    note=table.get('population', estimands.m1_population())
    lines=[note['mechanisms']]
    if 'annotation' in note:
        lines += [note['status']+': '+note['annotation'],
                  'Planning losses per bin: '+str(note['n_draw_failed_by_bin'])+
                  '; 1,736 of 1,800 (96.4 %) available. Realized grey-zone counts: 141/150, 289/300, 147/150; '
                  'equality of the two grey-zone aggregations is not claimed.']
    return lines


def _write_summary(out_dir, summary):
    out_dir = Path(out_dir)
    (out_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    L = ["# M harness dry run on synthetic fixtures", "", "**" + summary["label"] + "**", "",
         f"Status: {summary.get('status')}. Transit model backend: {summary['backend']}. Hosts: {summary['n_hosts']}.", ""]
    st = summary["steps"]
    if "preflight" in st:
        p = st["preflight"]
        L += ["## Preflight (C14)", "", f"Passed: {p['passed']}. Elapsed {p['m_preflight']['elapsed_s']} s; files opened for "
              f"contents: {p['m_preflight']['files_opened']}; files of blinded cohorts opened: {len(p['m_preflight']['blinded_files_opened'])}.", ""]
        L += [f"- {name}: {c['n_passed']} of {c['n_hosts']} hosts pass, blinded {c['blinded']}, gate rejects {c['gate_rejects'] or 'none'}"
              for name, c in p["cohorts"].items()] + [""]
    tail = st.get("m4_acf_tail")
    if tail:
        L += ["## M4 ACF tail (item 44)", "", f"Listing status: {tail['status']}.", ""]
        if tail["status"] == "COMPLETE":
            for title, rows in (("All stars above 0.05 at either lag", tail["above_005"]),
                                ("Top 10 stars by largest absolute shift", tail["top_10"]),
                                ("Cohort maximum", [tail["cohort_maximum"]] if tail["cohort_maximum"] else [])):
                L += ["### " + title, "", "| TIC | Sector | Lag | U | N1 | Signed difference | Absolute difference | Max Tier 1 SDE | False alarm SDE >= 6.0 |",
                      "|---|---|---|---|---|---|---|---|---|"]
                for row in rows:
                    for lag, acf in row["acf"].items():
                        L.append(f"| {row['tic_id']} | {row['sector']} | {lag} | {acf['U']} | {acf['N1']} | "
                                 f"{acf['signed_difference']} | {acf['absolute_difference']} | {row['max_tier1_sde']} | {row['false_alarm_sde_6']} |")
                L.append("")
    t = summary.get("tables")
    if t:
        inj, det = st["m1_injection"], st["detector_runs"]
        L += ["## Injection and runs", "",
              f"Planned M1 injections {inj['n_planned']}, injected {inj['n_injected']}, draw failures {len(inj['draw_failed'])}, "
              f"fidelity failures {len(inj['fidelity_failed'])}, model check failures {len(inj['model_check_failed'])}.",
              f"Detector jobs {det['n_jobs']} ({det['by_arm']}); routing ledger failures {len(det['routing_failed'])}; "
              f"interruptions {det['interruptions']}, resumes {det['resumes']}.",
              f"Runtime per case: median {det['runtime_s']['median']:.2f} s, 90th percentile {det['runtime_s']['p90']:.2f} s, "
              f"maximum {det['runtime_s']['max']:.2f} s.", "",
              "## M1 attrition (planned denominator)", "", "| Stage | Lost | Remaining | Lines |", "|---|---|---|---|"]
        for s in t["m1_attrition"]["stages"]:
            L.append(f"| {s['stage']} | {s['lost']} | {s['remaining']} | " + "; ".join(f"{x['line']} {x['n']}" for x in s["lines"]) + " |")
        L += ["", "## M1 completeness by SNR_inj bin (candidate level)", "",
              "Completeness for signals reachable inside the declared domain (depth 180 to 30,000 ppm, Rp <= 15 R_E).",
              "Denominators condition on the depth, radius and at least 3 discovery-event population.", "",
              *m1_report_notes(t["m1_completeness_candidate_level"]), "",
              "| SNR_inj bin | planned | draw failed | evaluated | recovered | primary C, 95 % CP | recovered / planned | C(SDE >= 7.5) | dC_grey |", "|---|---|---|---|---|---|---|---|---|"]
        for label, c in t["m1_completeness_candidate_level"]["by_snr_bin"].items():
            L.append(f"| {label} | {c['n_planned']} | {c['n_draw_failed']} | {c['n_evaluated']} | {c['n_recovered']} | "
                     f"{_fmt(c['primary_completeness'])} | {_fmt(c['unconditioned_completeness'])} | "
                     f"{_fmt(c['c_sde_7_5'])} | {_fmt(c['dc_grey'])} |")
        L += ["", "## M1 hosts with DRAW_FAILED families", "",
              "| TIC | R* | Teff | sectors | baseline (days) | failed families and diagnostics |", "|---|---|---|---|---|---|"]
        for h in t['m1_draw_failure_hosts']:
            detail='; '.join(f"{r['snr_bin']}, target {r['snr_target']}, converged outside {r['n_converged_out_of_domain']}, solver failures {r['n_solver_failures']}, caps {r['binding_caps']}" for r in h['failed_families'])
            L.append(f"| {h['tic_id']} | {h['r_star']} | {h['teff']} | {h['sector_count']} | {h['baseline']} | {detail} |")
        g = t["grey_zone_gain_candidate_level"]
        L += ["", f"Grey-zone gain over SNR_inj 5 to 9, pooled: {_fmt(g['pooled']['dc_grey'])}; bin-weighted: "
              f"{g.get('bin_weighted', {}).get('value')}. Decision function: {t['grey_zone_decision']['decision']}.", ""]
        if "pooled" in t["m4_n1"]:
            L += ["## M4 false-alarm rates (one-sided 95 % upper bounds)", "", "| Arm | n | (a) top SDE >= 6.0 | (a) >= 7.5 | (b) reached archival | (c) CANDIDATE | (d) Mode 2 pass per trigger |", "|---|---|---|---|---|---|---|"]
            for key in ("m4_n1", "m4_n2"):
                p = t[key]["pooled"]
                L.append(f"| {t[key]['arm']} | {p['n']} | {_fmt(p['a_top_peak_sde_6'])} | {_fmt(p['a_top_peak_sde_7_5'])} | "
                         f"{_fmt(p['b_reached_archival'])} | {_fmt(p['c_candidate'])} | {_fmt(p['d_mode2']['per_trigger'])} |")
            L.append("")
        L += ["## Other tables", "",
              f"- Noise-only pre-gate (item 44): {st['m4_noise_pre_gate']['status']}.",
              f"- Signal destruction after N1: {t['development_signal_destruction']['status']} "
              f"({ {k: (v.get('recovered'), v.get('n')) for k, v in t['development_signal_destruction']['criteria'].items()} }).",
              f"- M2 in-domain false rejection: {_fmt(t['m2_false_rejection']['in_domain'])}; end to end: "
              f"{_fmt(t['m2_false_rejection']['end_to_end'])}; out-of-domain planets {t['m2_false_rejection']['design_exclusion']['out_of_domain_planets']}.",
              f"- M2 multiplicity: {t['m2_false_rejection']['by_multiplicity']}; unavailable {t['m2_false_rejection']['multiplicity_unavailable']}.",
              f"- M2 sibling wins (target non-recovery): {t['m2_false_rejection']['sibling_wins']}.",
              f"- M3 raw CANDIDATE rate: {_fmt(t['m3_false_positive']['raw_candidate'])}; label rule {t['m3_false_positive']['label_rule']}.",
              f"- M5: {t['m5_parameters']['n']} matched CANDIDATEs; period coverage at 1 sigma "
              f"{_fmt(t['m5_parameters']['period']['coverage']['levels']['1 sigma'])}.",
              f"- Item 31.4 diagnostics: {t['item_31_4_diagnostics_m2_m3']['n_evaluations']} archival secondary evaluations.", ""]
        L += ['## M2 frozen reference status', '', '| Host | Planet | SNR status |', '|---|---|---|']
        for row in t['m2_reference_rows']:
            for gate in row['reference_gates']:
                L.append(f"| {row['host']} | {gate['planet']} | {gate.get('SNR_status', 'NOT_AVAILABLE')} |")
        L.append('')
    L += [f"Elapsed: {summary.get('elapsed_s')} s. Full tables: `dry_run_summary.json`.", ""]
    (out_dir / "DRY_RUN_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="end-to-end engineering run on synthetic fixtures")
    d.add_argument("--out", required=True)
    d.add_argument("--hosts", type=int, default=10)
    d.add_argument("--backend", default="reference", choices=TM.BACKENDS)
    d.add_argument("--stop-after", type=int, default=None, help="interrupt after this many cases (resume test)")
    a = ap.parse_args(argv)
    s = dry_run(a.out, n_hosts=a.hosts, backend=a.backend, stop_after=a.stop_after)
    print(json.dumps(dict(status=s.get("status"), elapsed_s=s.get("elapsed_s"), steps={k: (v if k != "m4_noise_pre_gate" else v["status"])
                                                                                       for k, v in s["steps"].items()}), indent=1, default=str))
    return 0 if s.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
