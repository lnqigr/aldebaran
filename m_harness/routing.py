"""Offline product routing into the production detector, and the product routing controls of roadmap 2.I (S4.3).

What is routed. The Tier 1 detector (aldebaran.analyze_single_target) reaches the outside world through five
seams, and the router replaces exactly these with the case's own local files and frozen metadata:
    lightkurve.search_lightcurve        discovery sector (one query) and all sectors (archival query)
    lightkurve.search_targetpixelfile   discovery sector target pixel file
    product.download()                  through the unchanged production safe_download, which also writes the
                                        production SHA256 record of every light curve it hands on (S3.6)
    get_stellar_and_gaia_params         TIC and Gaia values, served from the frozen cohort metadata
    get_known_transits_for_tic          known-signal context, served from the case (empty for injection and null
                                        hosts, and for a catalog-blinded benchmark)
Every other network attempt is blocked at the socket and recorded; a run with one is not a valid measurement.

What is not routed. Pleiades, Antares, the Antares packager and TRICERATOPS fetch their own products
(UNROUTED_STAGES). They are outside the measured boundary of this harness until their routing is validated; the
boundary inventory fails if the detector gains a fetch site that the router does not cover.

Detection sees only the case: file paths, SHA256, sector, stellar metadata and known-signal context. A case with
any other field is refused (DETECTOR_TRUTH_LEAK), and a file whose SHA256 differs from the case is refused
(INPUT_HASH_MISMATCH) before the detector starts.
"""
import ast
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from . import fixtures, prodpath
from .preflight_core import detector_stages

CASE_KEYS = frozenset({"tic_id", "case_id", "sector", "source_fits", "source_sha256", "tpf_fits", "tpf_sha256",
                       "stellar", "known_transits", "lightcurves"})
LIGHTCURVE_KEYS = frozenset({"path", "sector", "obsid", "sha256"})
STELLAR_KEYS = frozenset({"r_star", "e_r_star", "m_star", "e_m_star", "teff", "ruwe", "has_params",
                          "tic_query_error", "gaia_query_error"})
ROUTED_SEAMS = ("lk.search_lightcurve", "lk.search_targetpixelfile", "get_stellar_and_gaia_params",
                "get_known_transits_for_tic")
DETECTOR_FUNCTIONS = ("_analyze_single_target_impl", "verify_multi_epoch", "_verify_and_filter_tpf_cp",
                      "get_stellar_and_gaia_params", "safe_download", "fetch_csv_with_headers",
                      "_prepare_archival_product")  # C7: the archival product stage of verify_multi_epoch
# plus every stage the detector source lists in DETECTOR_STAGES (RC2-FREEZE-01 C7), read by boundary_inventory
UNROUTED_STAGES = {
    "pleiades.py": "Tier 2 Pleiades: own light curve search and download, TIC, VizieR and Gaia queries",
    "antares.py": "Tier 3 Antares: own multi-author light curve search and download, TIC query",
    "antares_packager.py": "Tier 3 packager: own multi-author light curve search and download",
    "validate_triceratops.py": "Tier 4 TRICERATOPS: own light curve and target pixel search and download",
    "stellar_metrics.py": "packager stellar metrics: TIC queries",
}
FETCH_CALLS = ("lk.search_lightcurve", "lk.search_targetpixelfile", "lk.search_tesscut", "Catalogs.query_criteria",
               "Catalogs.query_object", "Catalogs.query_region", "safe_http_get", "safe_urlopen", "requests.get",
               "fetch_csv_with_headers", "Vizier", "lk.read")


class RoutingError(RuntimeError):
    """A routing guard refused the case or the query. The detector is not run, or the run is not valid."""


def sha256_file(path):
    return fixtures.sha256_file(path)


def check_case(case, root):
    """Refuse a case that carries a field detection must not see, or whose files do not match their SHA256."""
    extra = sorted(set(case) - CASE_KEYS)
    if extra:
        raise RoutingError(f"DETECTOR_TRUTH_LEAK: case has fields outside the detector contract: {extra}")
    missing = sorted(CASE_KEYS - set(case))
    if missing:
        raise RoutingError(f"CASE_INCOMPLETE: missing fields {missing}")
    if set(case["stellar"]) - STELLAR_KEYS:
        raise RoutingError(f"DETECTOR_TRUTH_LEAK: stellar metadata has unexpected fields "
                           f"{sorted(set(case['stellar']) - STELLAR_KEYS)}")
    root = Path(root)
    sectors = [int(r["sector"]) for r in case["lightcurves"]]
    if sectors.count(int(case["sector"])) != 1:
        raise RoutingError("CASE_INCOMPLETE: the discovery sector must appear exactly once in lightcurves")
    for r in case["lightcurves"]:
        if set(r) - LIGHTCURVE_KEYS:
            raise RoutingError(f"DETECTOR_TRUTH_LEAK: light curve row has unexpected fields {sorted(set(r) - LIGHTCURVE_KEYS)}")
        p = root / r["path"]
        if not p.is_file():
            raise RoutingError(f"INPUT_MISSING: {r['path']}")
        if sha256_file(p) != r["sha256"]:
            raise RoutingError(f"INPUT_HASH_MISMATCH: {r['path']}")
    disc = next(r for r in case["lightcurves"] if int(r["sector"]) == int(case["sector"]))
    if disc["path"] != case["source_fits"] or disc["sha256"] != case["source_sha256"]:
        raise RoutingError("CASE_INCONSISTENT: source_fits or source_sha256 differs from the discovery light curve row")
    if case["tpf_fits"] is not None:
        p = root / case["tpf_fits"]
        if not p.is_file():
            raise RoutingError(f"INPUT_MISSING: {case['tpf_fits']}")
        if sha256_file(p) != case["tpf_sha256"]:
            raise RoutingError(f"INPUT_HASH_MISMATCH: {case['tpf_fits']}")


class Product:
    """One local product in the shape the detector reads from a lightkurve search row."""

    def __init__(self, router, kind, path, sector, obsid, sha256):
        self._router, self.kind, self.path, self.sha256 = router, kind, str(path), sha256
        self.sector = int(sector)
        self.mission = np.array([f"TESS Sector {int(sector)}"])
        self.obsid = obsid
        self.author = "SPOC"
        self.exptime = 120
        self.target_name = str(router.case["tic_id"])
        self.productFilename = os.path.basename(self.path)

    def download(self, **kwargs):
        import lightkurve as lk
        actual = sha256_file(self.path)
        self._router.consumed.append(dict(kind=self.kind, sector=self.sector, path=self.path, sha256=actual,
                                          expected_sha256=self.sha256, match=actual == self.sha256))
        if actual != self.sha256:
            raise RoutingError(f"INPUT_HASH_MISMATCH at download: {self.path}")
        return lk.read(self.path)

    def download_all(self, **kwargs):
        raise RoutingError("ROUTING_UNEXPECTED_CALL: download_all is not part of the detector contract")


class Search(list):
    """A search result: a list of products (len, index and iteration are all the detector uses)."""


class OfflineRouter:
    """Context manager that binds one case to the production detector module for the duration of one run."""

    def __init__(self, ald, case, root, out_dir, run_id, observe=True):
        self.ald, self.case, self.root, self.out_dir = ald, case, Path(root), Path(out_dir)
        self.run_id, self.observe = str(run_id), observe
        self.queries, self.consumed, self.network_attempts, self.search = [], [], [], {}
        self._saved = {}

    # ---------------------------------------------------------------------------------------- routed seams
    def _check_query(self, kind, target, kw):
        expected = f"TIC {self.case['tic_id']}"
        problems = []
        if str(target) != expected:
            problems.append(f"target {target!r} is not {expected!r}")
        if kw.get("author") != "SPOC":
            problems.append(f"author {kw.get('author')!r} is not 'SPOC'")
        if kw.get("exptime") != 120:
            problems.append(f"exptime {kw.get('exptime')!r} is not 120")
        if kw.get("mission") != "TESS":
            problems.append(f"mission {kw.get('mission')!r} is not 'TESS'")
        self.queries.append(dict(kind=kind, target=str(target), sector=kw.get("sector"), problems=problems))
        if problems:
            raise RoutingError(f"ROUTING_UNEXPECTED_QUERY ({kind}): " + "; ".join(problems))

    def search_lightcurve(self, target, *args, **kw):
        self._check_query("lightcurve", target, kw)
        rows = sorted(self.case["lightcurves"], key=lambda r: int(r["sector"]))
        if kw.get("sector") is not None:
            rows = [r for r in rows if int(r["sector"]) == int(kw["sector"])]
        return Search(Product(self, "discovery_lc" if int(r["sector"]) == int(self.case["sector"]) else "archival_lc",
                              self.root / r["path"], r["sector"], r.get("obsid"), r["sha256"]) for r in rows)

    def search_targetpixelfile(self, target, *args, **kw):
        self._check_query("targetpixelfile", target, kw)
        # Item 37.2: the pixel test is tied to the discovery sector; no other sector is ever offered.
        if self.case["tpf_fits"] is None or kw.get("sector") is None or int(kw["sector"]) != int(self.case["sector"]):
            return Search()
        return Search([Product(self, "discovery_tpf", self.root / self.case["tpf_fits"], self.case["sector"], "tpf",
                               self.case["tpf_sha256"])])

    def stellar(self, tic_id):
        if str(tic_id).lstrip("0") != str(self.case["tic_id"]).lstrip("0"):
            raise RoutingError(f"ROUTING_UNEXPECTED_QUERY (stellar): TIC {tic_id}")
        self.queries.append(dict(kind="stellar", target=str(tic_id), sector=None, problems=[]))
        return dict(self.case["stellar"])

    def known_transits(self, tic_id):
        if str(tic_id).lstrip("0") != str(self.case["tic_id"]).lstrip("0"):
            raise RoutingError(f"ROUTING_UNEXPECTED_QUERY (known transits): TIC {tic_id}")
        self.queries.append(dict(kind="known_transits", target=str(tic_id), sector=None, problems=[]))
        return [dict(k) for k in self.case["known_transits"]]

    # ---------------------------------------------------------------------------------------- observation
    def _recording_bls(self, real):
        sink = self.search

        class RecordingBLS(real):                      # same computation; keeps the inputs and the result
            def __init__(self, t, y, dy=None, **kw):
                sink["times"] = np.array(t, dtype=float, copy=True)
                super().__init__(t, y, dy=dy, **kw)

            def power(self, *a, **k):
                res = super().power(*a, **k)
                sink["result"] = res
                return res
        return RecordingBLS

    def _recording_peaks(self, real):
        sink = self.search

        def get_top_distinct_bls_peaks(periods, power_arr, *a, **k):
            idx = real(periods, power_arr, *a, **k)
            sink["peak_indices"] = [int(i) for i in idx]
            sink["peak_sde"] = [float(power_arr[i]) for i in idx]
            return idx
        return get_top_distinct_bls_peaks

    def top_peaks(self):
        """The ranked BLS peaks the detector selected: rank, period, epoch, duration and SDE."""
        res, idx = self.search.get("result"), self.search.get("peak_indices")
        if res is None or idx is None:
            return None
        return [dict(rank=n, period_days=float(res.period[i]), t0_btjd=float(res.transit_time[i]),
                     duration_days=float(res.duration[i]), sde=float(s))
                for n, (i, s) in enumerate(zip(idx, self.search["peak_sde"]), 1)]

    # ---------------------------------------------------------------------------------------- bind and release
    def verify_offline_binding(self):
        """Freeze-time check: TIC, Gaia and registry seams and socket exits are actually replaced."""
        import lightkurve as lk
        seams = ((self.ald.get_stellar_and_gaia_params, self.stellar),
                 (self.ald.get_known_transits_for_tic, self.known_transits),
                 (lk.search_lightcurve, self.search_lightcurve),
                 (lk.search_targetpixelfile, self.search_targetpixelfile))
        exits = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
        if any(actual != expected for actual, expected in seams) or any(f is not self._blocked for f in exits):
            raise RoutingError('LIVE_QUERY_BINDING_NOT_BLOCKED')
        return dict(passed=True, catalogs='frozen metadata', known_signals='case context', network='blocked')

    def __enter__(self):
        check_case(self.case, self.root)
        ald = self.ald
        import lightkurve as lk
        if ald.lk is not lk:
            raise RoutingError("ROUTING_SETUP: the detector module does not use the imported lightkurve module")
        self._saved = dict(
            lk=(lk.search_lightcurve, lk.search_targetpixelfile),
            ald={name: getattr(ald, name) for name in (
                "get_stellar_and_gaia_params", "get_known_transits_for_tic", "BoxLeastSquares",
                "get_top_distinct_bls_peaks", "AUTO_PURGE_CACHE", "OUTPUT_DIR", "LOG_FILE", "EVENTS_LOG_FILE",
                "REPORT_FILE", "HISTORY_FILE", "ARCHIVAL_DIAGNOSTICS_FILE", "CURRENT_RUN_ID")},
            sock=(socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo))
        lk.search_lightcurve = self.search_lightcurve
        lk.search_targetpixelfile = self.search_targetpixelfile
        ald.get_stellar_and_gaia_params = self.stellar
        ald.get_known_transits_for_tic = self.known_transits
        ald.AUTO_PURGE_CACHE = False                   # the harness owns its inputs; nothing is purged
        if self.observe:
            ald.BoxLeastSquares = self._recording_bls(self._saved["ald"]["BoxLeastSquares"])
            ald.get_top_distinct_bls_peaks = self._recording_peaks(self._saved["ald"]["get_top_distinct_bls_peaks"])
        ald.configure_environment(output_dir=str(self.out_dir), run_id=self.run_id)
        attempts = self.network_attempts

        def blocked(*a, **k):
            attempts.append(repr(a)[:200])
            raise OSError("NETWORK_BLOCKED_M_HARNESS")
        self._blocked = blocked
        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked
        socket.getaddrinfo = blocked
        self.verify_offline_binding()
        return self

    def __exit__(self, *exc):
        import lightkurve as lk
        lk.search_lightcurve, lk.search_targetpixelfile = self._saved["lk"]
        for name, value in self._saved["ald"].items():
            setattr(self.ald, name, value)
        (socket.socket.connect, socket.socket.connect_ex, socket.create_connection,
         socket.getaddrinfo) = self._saved["sock"]
        return False


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_jsonl(path):
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_case(ald, case, root, out_dir, run_id, observe=True, attempt_id=None):
    """Run the production detector on one case through the router and collect its own records.

    Returns the outcome record: final status, message, candidate record, per-peak vetting evidence, the all-ranks
    journal, archival diagnostics, the selected BLS peaks, and the routing ledger (queries, every file consumed with
    its SHA256, the production SHA256 records, network attempts). out_dir must be new: one case, one directory."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RoutingError(f"OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with OfflineRouter(ald, case, root, out_dir, run_id, observe=observe) as router:
        result = ald.analyze_single_target(str(case["tic_id"]), int(case["sector"]), run_id=str(run_id),
                                           attempt_id=attempt_id or f"att_{case['case_id']}")
        top = router.top_peaks()
        times = router.search.get("times")
    runtime = time.time() - t_start
    vetting = [r for r in _read_jsonl(out_dir / "experimental_v2_vetting.jsonl") if r.get("run_id") == str(run_id)]
    journal = _read_jsonl(out_dir / ald.AR_JOURNAL_FILENAME)
    mast = []
    mast_dir = out_dir / "mast_products"
    if mast_dir.is_dir():
        for p in sorted(mast_dir.glob("mast_*.json")):
            mast.append(json.loads(p.read_text(encoding="utf-8")))
    times_file = None
    if times is not None:
        times_file = out_dir / "search_times.npy"
        np.save(times_file, times)
    record = dict(
        case_id=case["case_id"], tic_id=str(case["tic_id"]), sector=int(case["sector"]), run_id=str(run_id),
        status=result[1], message=str(result[2]), candidate=_json_safe(result[3]),
        peaks=vetting[-1]["peaks"] if vetting else [],
        journal=[r for r in journal if r.get("record") == "rank"],
        host_end=next((r for r in journal if r.get("record") == "host_end"), None),
        archival_diagnostics=_json_safe(result.archival_diagnostics),
        search=dict(observed=bool(observe), top_peaks=top, n_times=None if times is None else int(times.size),
                    times_file=None if times_file is None else times_file.name,
                    times_sha256=None if times_file is None else sha256_file(times_file)),
        routing=dict(queries=router.queries, consumed=router.consumed, production_sha256_records=mast,
                     network_attempts=router.network_attempts),
        runtime_s=round(runtime, 3))
    return record


def verify_routing(record, case, originals=None):
    """Routing ledger of one finished run: every consumed product is one of the case's files with the case's
    SHA256; the production SHA256 records (light curves and target pixel files) agree with it; nothing came from the network; and, when the
    hashes of the uninjected originals are given, no consumed file is an original.

    Returns dict(passed, problems, products) where products lists each consumed file once with its kind."""
    problems = []
    allowed = {r["sha256"]: ("discovery_lc" if int(r["sector"]) == int(case["sector"]) else "archival_lc")
               for r in case["lightcurves"]}
    if case["tpf_fits"] is not None:
        allowed[case["tpf_sha256"]] = "discovery_tpf"
    consumed = record["routing"]["consumed"]
    for c in consumed:
        if not c["match"]:
            problems.append(f"consumed file changed after the case was written: {c['path']}")
        if c["sha256"] not in allowed:
            problems.append(f"consumed file is not a case product: {c['path']}")
        elif allowed[c["sha256"]] != c["kind"]:
            problems.append(f"consumed file has kind {c['kind']}, the case says {allowed[c['sha256']]}: {c['path']}")
    kinds = [c["kind"] for c in consumed]
    if kinds.count("discovery_lc") != 1:
        problems.append(f"the discovery light curve was consumed {kinds.count('discovery_lc')} times, expected once")
    lc_hashes = {c["sha256"] for c in consumed}
    prod_hashes = {m["sha256"] for m in record["routing"]["production_sha256_records"]}
    if prod_hashes != lc_hashes:
        problems.append("the production SHA256 records of consumed products differ from the router's ledger "
                        f"({len(prod_hashes)} against {len(lc_hashes)} files)")
    if record["routing"]["network_attempts"]:
        problems.append(f"{len(record['routing']['network_attempts'])} network attempt(s) during the run")
    for q in record["routing"]["queries"]:
        if q["problems"]:
            problems.append(f"unexpected query: {q}")
    if originals is not None:
        hit = sorted({c["path"] for c in consumed if c["sha256"] in set(originals)})
        if hit:
            problems.append(f"an uninjected original was consumed: {hit}")
    products = {}
    for c in consumed:
        products.setdefault(c["sha256"], dict(kind=c["kind"], sector=c["sector"], path=c["path"], sha256=c["sha256"], n_reads=0))
        products[c["sha256"]]["n_reads"] += 1
    return dict(passed=not problems, problems=problems, products=sorted(products.values(), key=lambda p: (p["kind"], p["sector"])))


def signal_in_production_arrays(ald, case, root, host, host_root, model_flux, truth):
    """The injected signal as the production reader sees it: the ratio of the injected to the uninjected production
    discovery arrays (extraction and upper clip) against the model at the same timestamps. Returns the maximum
    absolute difference over cadences kept in both, and the deepest model point among them."""
    import lightkurve as lk
    t_i, f_i, _e, _c = prodpath.main_search_arrays(ald, lk.read(str(Path(root) / case["source_fits"])))
    t_o, f_o, _e, _c = prodpath.main_search_arrays(ald, lk.read(str(Path(host_root) / host["source_fits"])))
    common, ia, ib = np.intersect1d(np.round(t_i, 9), np.round(t_o, 9), return_indices=True)
    m = model_flux(t_i[ia], truth)
    # Both arrays are normalised by their own median; the injected median is lower by the mean in-sector deficit.
    ratio = (f_i[ia] / f_o[ib]) / np.median(f_i[ia] / f_o[ib] / m)
    return dict(n_common=int(common.size), n_injected=int(t_i.size), n_original=int(t_o.size),
                max_abs_ratio_minus_model=float(np.max(np.abs(ratio - m))) if common.size else None,
                max_model_deficit=float(np.max(1.0 - m)) if common.size else None,
                n_in_transit=int(np.sum(m < 1.0)))


# ---------------------------------------------------------------------------------------- boundary inventory
def fetch_sites(source_path):
    """Every call in a source file that fetches a product or queries a catalog: (function, line, call)."""
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    out = []

    def walk(node, owner):
        for child in ast.iter_child_nodes(node):
            name = owner
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name if owner == "<module>" else owner
            if isinstance(child, ast.Call):
                text = ast.unparse(child.func)
                if text in FETCH_CALLS or text.endswith(".download") or text.endswith(".download_all"):
                    out.append(dict(function=name, line=child.lineno, call=text))
            walk(child, name)
    walk(tree, "<module>")
    return sorted(out, key=lambda r: r["line"])


def boundary_inventory(repo_dir):
    """Which fetch sites exist in each stage and whether this harness routes them.

    The Tier 1 detector functions must contain only routed or covered sites: the two lightkurve searches, the
    downloads inside safe_download (they act on router products) and the catalog queries inside
    get_stellar_and_gaia_params (the whole function is replaced). Anything else in a detector function is an
    unrouted fetch and fails the inventory. The other stages are listed as outside the boundary."""
    repo_dir = Path(repo_dir)
    covered = {
        ("_load_discovery_light_curve", "lk.search_lightcurve"), ("verify_multi_epoch", "lk.search_lightcurve"),
        ("_verify_and_filter_tpf_cp", "lk.search_targetpixelfile"),
        ("safe_download", "search_obj.download"), ("safe_download", "search_obj.download_all"),
        ("get_stellar_and_gaia_params", "Catalogs.query_criteria"), ("get_stellar_and_gaia_params", "fetch_csv_with_headers"),
        ("fetch_csv_with_headers", "safe_http_get"), ("fetch_csv_with_headers", "safe_urlopen"),
    }
    source = repo_dir / "aldebaran.py"
    functions = set(DETECTOR_FUNCTIONS) | set(detector_stages(ast.parse(source.read_text(encoding="utf-8"))) or ())
    detector = [s for s in fetch_sites(source) if s["function"] in functions]
    unrouted = [s for s in detector if (s["function"], s["call"]) not in covered]
    stages = {}
    for name, what in UNROUTED_STAGES.items():
        p = repo_dir / name
        stages[name] = dict(description=what, present=p.is_file(), fetch_sites=fetch_sites(p) if p.is_file() else [],
                            routed=False, inside_boundary=False)
    return dict(detector_sites=detector, unrouted_detector_sites=unrouted, passed=not unrouted and bool(detector),
                outside_boundary=stages, routed_seams=list(ROUTED_SEAMS))
