"""Production CSV contract and isolated replay-corruption controls for S3.4."""
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import aldebaran as ald
import pleiades
import search_toi_companions as toi
import test_char_pleiades_deep_vet as vet


class ToiMaskPersistence(unittest.TestCase):
    def test_controller_worker_csv_parser_and_resume(self):
        masks = [dict(toi="companion", period=5.1, t0=2001., duration=.1)]
        ctx = {"12345": masks}
        row = vet.candidate(tic_id="12345", known_transit_masks_json=json.dumps(masks),
                            known_signal_cache_state="NOT_EVALUATED")
        with tempfile.TemporaryDirectory() as td:
            argv = ["--sectors", "106", "--workers", "1", "--output-dir", td, "--run-id", "s34_revision"]
            with patch.object(toi, "query_sector_toi_hosts", return_value=(["12345"], None)), \
                 patch.object(ald, "load_known_catalogs_and_toi_ephemerides", return_value=(set(), set(), ctx)), \
                 patch.object(ald, "analyze_single_target", return_value=("12345", "CANDIDATE", "fixture", row)), \
                 contextlib.redirect_stdout(io.StringIO()):
                code, summary = toi.run_search(toi.parse_args(argv))
                self.assertEqual(code, 0)
                self.assertEqual(summary["run_status"], "CANDIDATES_FOUND")
                for path in (Path(td) / "toi_companions_summary.csv", Path(td) / "runs/s34_revision/candidates.csv"):
                    # Locate the run CSV using the controller's public paths below.
                    if not path.exists():
                        continue
                    frame = pd.read_csv(path)
                    self.assertEqual(len(frame), 1)
                    self.assertEqual(pleiades._parse_known_transit_masks(frame.iloc[0].to_dict()), masks)
                    self.assertEqual(frame.iloc[0]["known_signal_cache_state"], "NOT_EVALUATED")
                paths = list(Path(td).rglob("*.csv"))
                candidate_paths = [p for p in paths if "known_transit_masks_json" in pd.read_csv(p, nrows=0).columns]
                self.assertEqual(len(candidate_paths), 2)
                # Repeat under append: no duplicate or mask loss in either table.
                toi.run_search(toi.parse_args(argv + ["--append"]))
                for path in candidate_paths:
                    frame = pd.read_csv(path)
                    self.assertEqual(len(frame), 1)
                    self.assertEqual(pleiades._parse_known_transit_masks(frame.iloc[0].to_dict()), masks)

    def test_historical_schema_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as td:
            paths = toi.initialize_output_artifacts(td, "old", False)
            for path in (paths[0], paths[2]):
                pd.DataFrame(columns=toi.CANDIDATE_COLUMNS).to_csv(path, index=False)
            before = {p: Path(p).read_bytes() for p in paths}
            with self.assertRaisesRegex(RuntimeError, "Incompatible candidate CSV schema"):
                toi.initialize_output_artifacts(td, "old", True)
            self.assertEqual(before, {p: Path(p).read_bytes() for p in paths})


def replay_module():
    paths = [Path("/mnt/c/Users/user/Desktop/exoplanet search"), Path("C:/Users/user/Desktop/exoplanet search")]
    root = next(p for p in paths if (p / "agent_exchange/s3_4_known_signals/replay.py").exists())
    spec = importlib.util.spec_from_file_location("s34_revision_replay", root / "agent_exchange/s3_4_known_signals/replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_fixture_row(rank):
    fields = ('ephemeris_source', 'ephemeris_status', 'ephemeris_fallback_reason', 'archival_snr',
              'multi_sde', 'est_multi_sde_proxy', 'archival_confirmation_label', 'vr')
    row = dict.fromkeys(fields)
    for name in ('archival_sectors_requested', 'archival_sectors_attempted', 'archival_sectors_used',
                 'archival_products_requested', 'archival_products_attempted', 'archival_products_used', 'archival_products_failed'):
        row[name] = []
    row.update(run_id='fixture', tic_id='12345', sector=106, peak_rank=rank, arch_ok=True, arch_status='OK')
    return row


def fixture(e4=False):
    def peak(rank, period):
        return dict(peak_rank=rank, period_days=period, t0_btjd=2001., duration_days=.1,
                    depth_ppm=1000., sde=9., disposition="CANDIDATE", cause="ALL_REQUIRED_GATES_PASSED",
                    gates=[dict(gate="SECONDARY", status="PASS", metrics=dict(depth_ppm=20., snr=1.)),
                           dict(gate="TPF", status="PASS", metrics=dict(reason="OK")),
                           dict(gate="ARCHIVAL", status="PASS", metrics=dict(reason="OK"))])
    peaks = [peak(1, 3.1)] if e4 else [peak(1, 3.1), peak(2, 5.2)]
    if e4:
        peaks[0]['gates'].insert(1, dict(gate='RUWE', status='MISSING_INPUT', metrics={}))
    record = dict(raw_status="CANDIDATE", message="ok", detection={}, candidate=dict(tic_id="12345", period_days=3.1),
                  gates=[dict(peak_rank=p["peak_rank"], gate="REJECT_SECONDARY_ECLIPSE", status="PASS",
                              metrics=dict(is_toi_host=e4, sec_depth_ppm=20., sec_snr=1., is_sec_eb=False)) for p in peaks],
                  archival=[archive_fixture_row(p["peak_rank"]) for p in peaks], review_evidence=[dict(final_status="CANDIDATE", peaks=peaks, run_id="fixture", tic_id="12345")])
    def journal(ps):
        return [dict(record="rank", rank=p["peak_rank"], period_days=p["period_days"], evidence=True,
                     disposition=p["disposition"], cause=p["cause"], first_reject_gate=None,
                     linked_candidate_rank=None, linked_rejected_rank=None, stop="continue", exit_code=None) for p in ps] + [
                         dict(record="host_end", selected_ranks=len(ps), evaluated_ranks=[p["peak_rank"] for p in ps],
                              not_evaluated_ranks=[], first_candidate_rank=next((p["peak_rank"] for p in ps if p["disposition"] == "CANDIDATE"), None))]
    base = dict(input_digest="input", arrays={}, record=record, journal=journal(peaks))
    new = copy.deepcopy(base)
    new["record"]["candidate"].update(known_transit_masks_json="[]", known_signal_cache_state="OK")
    if e4:
        p = new["record"]["review_evidence"][0]["peaks"][0]
        p.update(disposition="INCOMPLETE", cause="VETTING_HOLD_DOWNSTREAM_NOT_RUN", vetting_holds=["REVIEW_TPF_DIPOLE_JITTER: fixture"])
        p["gates"][2].update(status="HOLD", metrics=dict(reason=p["vetting_holds"][0]))
        new["record"].update(raw_status="INCOMPLETE_VETTING_DOWNSTREAM", candidate=None, message="hold")
        new["record"]["review_evidence"][0]["final_status"] = "INCOMPLETE_VETTING_DOWNSTREAM"
    else:
        p = new["record"]["review_evidence"][0]["peaks"][1]
        p["gates"][0]["metrics"]["snr"] = .5
        new["record"]["gates"][1]["metrics"]["sec_snr"] = .5
    new["journal"] = journal(new["record"]["review_evidence"][0]["peaks"])
    return base, new


def producer_fixture(base, new):
    arms = {}
    for label, arm in [('base', base), ('new', new)]:
        traces = []
        for peak in arm['record']['review_evidence'][0]['peaks']:
            rank = peak['peak_rank']
            metrics = next(g['metrics'] for g in arm['record']['gates'] if g['peak_rank'] == rank)
            traces.append(dict(kind='secondary', rank=rank, returned=[metrics['is_sec_eb'], metrics['sec_snr'], metrics['sec_depth_ppm'], .5, 4.]))
            for kind, name in [('tpf', 'TPF'), ('archival', 'ARCHIVAL')]:
                gate = next(g for g in peak['gates'] if g['gate'] == name)
                result = [True, gate['metrics']['reason']]
                if kind == 'archival': result.append({})
                traces.append(dict(kind=kind, rank=rank, returned=result))
            traces.append(dict(kind='rank_exit', rank=rank, evidence=copy.deepcopy(peak), locals=copy.deepcopy(metrics)))
        arms[label] = traces
    return dict(case=dict(tic_id='12345', sector=106, known_signal_cache_state='OK', host_messages={a: r['record']['message'] for a, r in [('base', base), ('new', new)]}), arms=arms)


class ReplayDeltaValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = replay_module()

    def test_valid_e3_e4_and_each_isolated_mutation(self):
        for e4 in (False, True):
            base, new = fixture(e4)
            production = producer_fixture(base, new)
            def checked(arm):
                with patch.object(self.replay, '_branch_evidence', return_value=(production, []), create=True):
                    return self.replay.compare(base, arm)[0]
            self.assertEqual(checked(new), [])
            for kind in ("status", "candidate", "gate", "journal", "cause", "hold", "input", "arrays", "detection", "exit_code", "hold_cause", "ledger_status", "gate_metric"):
                with self.subTest(e4=e4, mutation=kind):
                    arm = copy.deepcopy(new)
                    peak = arm["record"]["review_evidence"][0]["peaks"][-1]
                    if kind == "status": arm["record"]["raw_status"] = "REJECT_ARBITRARY"
                    elif kind == "candidate": arm["record"]["candidate"] = dict(tic_id="wrong", period_days=123456.)
                    elif kind == "gate": peak["gates"].append(dict(gate="UNRELATED", status="PASS", metrics={}))
                    elif kind == "journal": arm["journal"][0]["cause"] = "UNRELATED"
                    elif kind == "cause": peak["cause"] = "UNRELATED"
                    elif kind == "hold": peak["vetting_holds"] = ["UNRELATED"]
                    elif kind == "ledger_status": arm['record']['gates'][-1]['status'] = 'REJECT'
                    elif kind == "gate_metric": peak['gates'][0]['metrics']['snr'] = 123456.
                    elif kind == "exit_code": arm['journal'][0]['exit_code'] = 'REJECT_ARBITRARY'
                    elif kind == "hold_cause":
                        peak['cause'] = 'VETTING_HOLD_PASSED_DOWNSTREAM'
                        arm['journal'][-2]['cause'] = peak['cause']
                    elif kind == "input": arm["input_digest"] = "wrong"
                    elif kind == "arrays": arm["arrays"] = dict(wrong=1)
                    else: arm["record"]["detection"] = dict(wrong=1)
                    self.assertTrue(checked(arm))

    def test_null_and_missing_journals_fail_even_on_identity(self):
        base, _ = fixture()
        for missing in (False, True):
            arm = copy.deepcopy(base)
            if missing: del arm["journal"]
            else: arm["journal"] = None
            self.assertTrue(self.replay.compare(arm, arm)[0])

    def test_reads_production_path_and_requires_host_end(self):
        base, _ = fixture()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            with self.assertRaisesRegex(AssertionError, "MISSING_ALL_RANKS_JOURNAL"):
                self.replay.read_journal(folder, base["record"])
            path = folder / "v2/all_ranks_journal.jsonl"
            path.parent.mkdir()
            path.write_text("\n".join(json.dumps(r) for r in base["journal"]))
            self.assertEqual(self.replay.read_journal(folder, base["record"]), base["journal"])
            path.write_text(json.dumps(base["journal"][0]))
            with self.assertRaisesRegex(AssertionError, "INCOMPLETE_ALL_RANKS_JOURNAL"):
                self.replay.read_journal(folder, base["record"])

    def test_terminal_host_return_is_complete_without_host_end(self):
        base, _ = fixture(e4=True)
        p = base['record']['review_evidence'][0]['peaks'][0]
        p.update(disposition='REJECT', cause='RUWE', gates=[dict(gate='RUWE', status='REJECT', metrics={})])
        base['record'].update(raw_status='REJECT_RUWE', candidate=None, archival=[], gates=[dict(peak_rank=1, gate='REJECT_RUWE', status='REJECT', metrics=dict(ruwe=2.))])
        base['record']['review_evidence'][0]['final_status'] = 'REJECT_RUWE'
        base['journal'] = [dict(record='rank', rank=1, period_days=p['period_days'], evidence=True,
                                disposition='REJECT', cause='RUWE', first_reject_gate='RUWE',
                                linked_candidate_rank=None, linked_rejected_rank=None,
                                stop='return', exit_code='REJECT_RUWE')]
        self.assertEqual(self.replay.compare(base, base)[0], [])
        bad = copy.deepcopy(base)
        bad['journal'][0]['exit_code'] = 'REJECT_ARBITRARY'
        self.assertTrue(self.replay.compare(base, bad)[0])

    def test_real_branch_returns_reject_numeric_and_reason_mutations(self):
        ev = self.replay.HERE
        for key in ['inj_h000_s1', 'inj_h005_s0', 'v03_286712787', 'vr02_281837575']:
            row = json.loads((ev / 'cases_r3_complete' / (key + '.json')).read_text())
            b, n = row['arms']['base'], row['arms']['new']
            self.assertEqual(self.replay.compare(b, n)[0], [])
            for kind in (['restored_delta'] if key == 'inj_h000_s1' else ['threshold', 'phase', 'stale_snr', 'missing_threshold', 'missing_new_gate_depth'] if key.startswith('inj') else ['archive_snr', 'archive_reason', 'archive_vr', 'message', 'module_path', 'host_local', 'missing_archive_snr'] + (['missing_masks', 'missing_cache'] if n['record']['candidate'] else [])):
                with self.subTest(case=key, mutation=kind):
                    arm = copy.deepcopy(n)
                    if kind == 'restored_delta':
                        arm['record'] = copy.deepcopy(b['record'])
                        arm['record']['candidate'].update({k: n['record']['candidate'][k] for k in ('known_transit_masks_json', 'known_signal_cache_state')})
                    elif kind in ('threshold', 'phase'):
                        g = next(g for g in arm['record']['gates'] if g.get('peak_rank') == 3 and g['gate'] == 'REJECT_SECONDARY_ECLIPSE')
                        g['metrics']['z_crit' if kind == 'threshold' else 'sec_phi'] = 123456.
                    elif kind == 'missing_new_gate_depth':
                        next(g for g in arm['record']['gates'] if g.get('peak_rank') == 2 and g['gate'] == 'REJECT_CENTROID')['metrics'].pop('depth_ppm')
                    elif kind == 'missing_threshold':
                        next(g for g in arm['record']['gates'] if g.get('peak_rank') == 3 and g['gate'] == 'REJECT_SECONDARY_ECLIPSE')['metrics'].pop('z_crit')
                    elif kind == 'message': arm['record']['message'] = 'UNRELATED'
                    elif kind == 'module_path': arm['module_path'] = '/unrelated/module.py'
                    elif kind == 'host_local': arm['record']['gates'][0]['metrics']['arch_snr'] = 123456.
                    elif kind == 'missing_archive_snr': arm['record']['archival'][0].pop('archival_snr')
                    elif kind == 'missing_masks': arm['record']['candidate'].pop('known_transit_masks_json')
                    elif kind == 'missing_cache': arm['record']['candidate'].pop('known_signal_cache_state')
                    elif kind == 'stale_snr':
                        g = next(g for g in arm['record']['gates'] if g.get('peak_rank') == 3 and g['gate'] == 'REJECT_DUTY_CYCLE')
                        g['metrics']['sec_snr'] = 123456.
                    elif kind == 'archive_snr': arm['record']['archival'][0]['archival_snr'] = 123456.
                    elif kind == 'archive_vr': arm['record']['archival'][0]['vr']['refined_p'] = 123456.
                    else:
                        p = self.replay.peaks_of(arm['record'])[0]
                        next(g for g in p['gates'] if g['gate'] == 'ARCHIVAL')['metrics']['reason'] = 'UNRELATED'
                    self.assertTrue(self.replay.compare(b, arm)[0])

    def test_production_branch_evidence_integrity_fails_closed(self):
        row = json.loads((self.replay.HERE / 'cases_r3_complete/v03_286712787.json').read_text())
        b, n = row['arms']['base'], row['arms']['new']
        original = self.replay.BRANCH_FILE.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'branches.jsonl'
            path.write_bytes(original)
            with patch.object(self.replay, 'BRANCH_FILE', path):
                self.assertEqual(self.replay.compare(b, n)[0], [])
                path.write_bytes(original + b'\n')
                self.assertTrue(self.replay.compare(b, n)[0])
            with patch.object(self.replay, 'BRANCH_FILE', Path(td) / 'missing'):
                self.assertTrue(self.replay.compare(b, n)[0])
