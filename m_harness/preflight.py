"""C14 preflight over the M cohorts (AGENTS.md "Preflight before long runs"; roadmap S4.5 and section 11 C14).

preflight_core.py is the accepted shared tool (agent_exchange/preflight_01/preflight.py, SHA256 496620ec...),
ported with the round 1 correction that blinded input digests are declarations only. It reads every host-level return gate from the frozen detector source and evaluates those
exact conditions on each host's metadata. This module supplies what an M run must declare:

  host_values      gate variables from the cohort manifest only (crowding, stellar parameters, RUWE, known-signal
                   mask fraction, discovery baseline). A value the manifest lacks is left out, so its gate is
                   NOT_EVALUATED and the preflight fails; nothing is defaulted.
  inputs           every light curve and target pixel file a run would read, with its SHA256.
  metadata_checks  ranges and counts from the manifest: the search period range and the discovery cadence count.
  data_checks      ranges and counts read from the files themselves: valid cadences, baseline, N1 segments, and
                   target pixel coverage of the valid light curve cadences (needed by N1).
  draw             the planned injection families, when the run passes its draw function.

Held-out and blinded cohorts are checked from metadata only. The core never calls data_checks or draw for them, and
the file reader here refuses a blinded cohort by itself (BlindedCohortRead), so a change in either place cannot open
held-out photometry unnoticed. Blinded files are checked for presence and a declared SHA256 only; their contents are not read or
verified. Non-blinded input digests are verified against file contents. The preflight writes its one result file and nothing else.
"""
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import nulls
from . import preflight_core as core

MIN_VALID_CADENCES = 500        # extract_and_stitch_arrays keeps a sector only with at least 500 valid cadences
SEARCH_BASELINE_FACTOR = 0.48   # max_p = min(max_period, span * 0.48), read against the frozen source by the test


class BlindedCohortRead(RuntimeError):
    """An attempt to open photometry of a held-out or blinded cohort."""


class MPreflight:
    """Callables for preflight_core.run_preflight, bound to one cohort manifest.

    cohorts: {name: dict(blinded=bool, root=dir, hosts=[host record])}. A host record has tic_id, sector, crowdsap,
    stellar (r_star, teff, ruwe, has_params), lightcurves [path, sector, sha256], tpf_fits, tpf_sha256,
    known_transits, and the manifest metadata n_valid_cadences, discovery_baseline_days and, for a host with known
    transits, known_mask_fraction."""

    def __init__(self, cohorts, min_period=0.8, max_period=13.0):
        self.cohorts = cohorts
        self.min_period, self.max_period = float(min_period), float(max_period)
        self.opened = []                                 # (cohort, path) of every file opened for its contents

    def _host(self, cohort, host):
        for h in self.cohorts[cohort]["hosts"]:
            if str(h["tic_id"]) == str(host):
                return h
        raise KeyError(f"host {host!r} is not in cohort {cohort!r}")

    def _root(self, cohort):
        return Path(self.cohorts[cohort]["root"])

    def _open(self, cohort, path):
        if self.cohorts[cohort].get("blinded"):
            raise BlindedCohortRead(f"cohort {cohort!r} is blinded; {path} may not be opened by the preflight")
        self.opened.append((cohort, str(path)))
        return fits.open(path, memmap=False)

    def host_values(self, cohort, host):
        h = self._host(cohort, host)
        st = h.get("stellar") or {}
        vals = {}
        disc = [r for r in h.get("lightcurves", []) if int(r["sector"]) == int(h["sector"])]
        vals["search_res"] = disc
        if h.get("crowdsap") is not None:
            vals["crowdsap"] = float(h["crowdsap"])
        known = h.get("known_transits")
        if known is not None:
            vals["is_toi_host"] = len(known) > 0
            if len(known) == 0:
                vals["mask_frac"] = 0.0
            elif h.get("known_mask_fraction") is not None:
                vals["mask_frac"] = float(h["known_mask_fraction"])
        if h.get("discovery_baseline_days") is not None:
            vals["max_p"] = min(self.max_period, float(h["discovery_baseline_days"]) * SEARCH_BASELINE_FACTOR)
        if "has_params" in st:
            vals["star_info_cache"] = dict(has_params=bool(st["has_params"]))
        for key in ("r_star", "teff", "ruwe"):
            if st.get(key) is not None:
                vals[key] = float(st[key])
        return vals

    def inputs(self, cohort, host):
        h, root = self._host(cohort, host), self._root(cohort)
        items = [dict(path=str(root / r["path"]), sha256=r["sha256"]) for r in h["lightcurves"]]
        if h.get("tpf_fits") is not None:
            items.append(dict(path=str(root / h["tpf_fits"]), sha256=h["tpf_sha256"]))
        return items

    def metadata_checks(self, cohort, host):
        h = self._host(cohort, host)
        base = h.get("discovery_baseline_days")
        max_p = None if base is None else min(self.max_period, float(base) * SEARCH_BASELINE_FACTOR)
        return dict(ranges={"search_period_range_days": (self.min_period, max_p)},
                    counts={"discovery_valid_cadences": (h.get("n_valid_cadences"), MIN_VALID_CADENCES),
                            "discovery_light_curves": (len([r for r in h.get("lightcurves", [])
                                                            if int(r["sector"]) == int(h["sector"])]), 1)})

    def data_checks(self, cohort, host):
        h, root = self._host(cohort, host), self._root(cohort)
        disc = next(r for r in h["lightcurves"] if int(r["sector"]) == int(h["sector"]))
        with self._open(cohort, root / disc["path"]) as hdul:
            d = hdul[1].data
            valid = nulls.valid_mask(d)
            tv = np.asarray(d["TIME"], float)[valid]
            cad = np.asarray(d["CADENCENO"])[valid]
        n_valid = int(valid.sum())
        baseline = float(tv[-1] - tv[0]) if n_valid >= 2 else 0.0
        segs = len(nulls.segments(tv)) if n_valid >= 2 else 0
        counts = {"valid_cadences_in_file": (n_valid, MIN_VALID_CADENCES), "n1_segments": (segs, 2)}
        ranges = {"search_period_range_in_file_days": (self.min_period, min(self.max_period, baseline * SEARCH_BASELINE_FACTOR))}
        if h.get("n_valid_cadences") is not None:
            # The manifest count must be the file's count: the blinded cohorts are checked from that number alone.
            counts["manifest_cadence_count_matches_file"] = (int(int(h["n_valid_cadences"]) == n_valid), 1)
        if h.get("tpf_fits") is not None:
            with self._open(cohort, root / h["tpf_fits"]) as hdul:
                tcad = set(np.asarray(hdul[1].data["CADENCENO"]).tolist())
            covered = int(sum(1 for c in cad.tolist() if c in tcad))
            counts["tpf_covers_valid_lc_cadences"] = (covered, n_valid)
        return dict(ranges=ranges, counts=counts)


def run(*, source, cohorts, out, expected_source_sha256=None, exempt=None, allow_gate_rejects=(), draw=None,
        max_attempts=None, min_period=0.8, max_period=13.0, note=None):
    """Run the C14 preflight over an M cohort manifest and write its result file. Returns the result with the
    elapsed time, the list of files opened for their contents and the blinded cohorts it left unopened."""
    mp = MPreflight(cohorts, min_period=min_period, max_period=max_period)
    spec = {name: dict(hosts=[str(h["tic_id"]) for h in c["hosts"]], blinded=bool(c.get("blinded")))
            for name, c in cohorts.items()}
    t0 = time.time()
    res = core.run_preflight(source=source, cohorts=spec, host_values=mp.host_values, out=out,
                             expected_source_sha256=expected_source_sha256, exempt=exempt,
                             allow_gate_rejects=allow_gate_rejects, inputs=mp.inputs,
                             metadata_checks=mp.metadata_checks, data_checks=mp.data_checks, draw=draw,
                             max_attempts=max_attempts, note=note)
    res["m_preflight"] = dict(elapsed_s=round(time.time() - t0, 2), files_opened=len(mp.opened),
                              blinded_cohorts=sorted(n for n, c in spec.items() if c["blinded"]),
                              blinded_files_opened=[p for c, p in mp.opened if spec[c]["blinded"]])
    return res


def passing_hosts(result, cohort):
    return core.passing_hosts(result, cohort)
