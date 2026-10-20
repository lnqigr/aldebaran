"""
Tier-1 Decision Ledger (S1 Slice).

Standard-library-only module for context, identities, event serialization,
submission-local persistence, and independent completeness validation.
Preserves existing scientific computations, thresholds, and failure propagation.
"""

from __future__ import annotations

import os
import sys
import json
import math
import uuid
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

SCHEMA_VERSION: str = "obs.tier1.v0"

# Explicit coverage manifest for Tier-1 S1 slice
COVERAGE_MANIFEST: Dict[str, str] = {
    # Tier-1 S1 instrumented gates
    "gate_spoc_search": "INSTRUMENTED",
    "gate_crowding": "INSTRUMENTED",
    "gate_baseline": "INSTRUMENTED",
    "search_bls": "INSTRUMENTED",
    "gate_low_sde": "INSTRUMENTED",
    # Gates explicitly NOT instrumented in this first slice
    "gate_duty_cycle": "NOT_INSTRUMENTED",
    "gate_harmonic": "NOT_INSTRUMENTED",
    "gate_points": "NOT_INSTRUMENTED",
    "gate_depth": "NOT_INSTRUMENTED",
    "gate_morphology": "NOT_INSTRUMENTED",
    "gate_v_shape": "NOT_INSTRUMENTED",
    "gate_multiplicity": "NOT_INSTRUMENTED",
    "gate_odd_even_support": "NOT_INSTRUMENTED",
    "gate_odd_even_depth": "NOT_INSTRUMENTED",
    "gate_consistency": "NOT_INSTRUMENTED",
    "gate_secondary": "NOT_INSTRUMENTED",
    "gate_centroid": "NOT_INSTRUMENTED",
    "gate_stellar_params": "NOT_INSTRUMENTED",
    "gate_radius": "NOT_INSTRUMENTED",
    "gate_duration": "NOT_INSTRUMENTED",
    "gate_tpf": "NOT_INSTRUMENTED",
    "gate_archival": "NOT_INSTRUMENTED",
    "attrition_quality_mask": "NOT_INSTRUMENTED",
    "attrition_detrend": "NOT_INSTRUMENTED",
}


def get_module_source_hash(module_path: Optional[str] = None) -> str:
    """Computes SHA256 of the specified module or aldebaran.py in same directory."""
    if module_path is None:
        module_path = os.path.join(os.path.dirname(__file__), "aldebaran.py")
    if os.path.exists(module_path):
        try:
            with open(module_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "UNKNOWN_UNREADABLE"
    return "UNKNOWN_NOT_FOUND"


def sanitize_for_strict_json(obj: Any) -> Any:
    """
    Recursively replaces NaN, Inf, and -Inf with None and annotates nonfinite state
    to adhere strictly to RFC 8259 JSON standards (allow_nan=False).
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out[k] = None
                out[f"{k}_availability"] = "NONFINITE"
                out[f"{k}_raw"] = str(v)
            else:
                out[k] = sanitize_for_strict_json(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_strict_json(item) for item in obj]
    return obj


def strict_json_dumps(obj: Any) -> str:
    """Serializes object to strict JSON (no NaN/Infinity) with deterministic formatting."""
    sanitized = sanitize_for_strict_json(obj)
    return json.dumps(sanitized, allow_nan=False, ensure_ascii=True, separators=(",", ":"))


def is_safely_contained(parent_dir: str, target_dir: str) -> bool:
    """
    Strictly verifies that target_dir resolves inside parent_dir without
    allowing sibling-prefix matches (e.g. /tmp/obs vs /tmp/obs_sibling)
    or symlink traversal.
    """
    try:
        real_parent = os.path.realpath(os.path.abspath(parent_dir))
        real_target = os.path.realpath(os.path.abspath(target_dir))
        if real_target == real_parent:
            return False
        common = os.path.commonpath([real_parent, real_target])
        return common == real_parent
    except Exception:
        return False


@dataclass(frozen=True)
class ObserverContext:
    """
    Lightweight, pickleable container of primitive strings for worker IPC.
    Never holds open file handles or callbacks.
    """
    observer_dir: str
    observer_run_id: str
    submission_id: str

    @classmethod
    def from_any(cls, obj: Any) -> Optional[ObserverContext]:
        if obj is None:
            return None
        if isinstance(obj, ObserverContext):
            return obj
        if isinstance(obj, (tuple, list)) and len(obj) == 3:
            return cls(str(obj[0]), str(obj[1]), str(obj[2]))
        if isinstance(obj, dict):
            return cls(
                str(obj.get("observer_dir", "")),
                str(obj.get("observer_run_id", "")),
                str(obj.get("submission_id", "")),
            )
        return None


class SubmissionLedger:
    """
    Worker-side submission-local append-only event ledger.
    Writes exclusively to observer_dir/observer_run_id/submission_id/events.jsonl.
    Fails closed on observer issues without interrupting the scientific pipeline.
    """

    def __init__(self, context: ObserverContext, tic_id: Union[str, int], sector: int):
        self.context = context
        self.tic_id = str(tic_id)
        self.sector = int(sector)
        self.attempt_id: Optional[str] = None
        self.sequence: int = 0
        self.search_call_index: int = 0
        self.has_error: bool = False
        self.error_message: Optional[str] = None
        self.source_hash: str = get_module_source_hash()

        # Submission-local directory layout
        run_dir = os.path.join(self.context.observer_dir, self.context.observer_run_id)
        self.submission_dir = os.path.join(run_dir, self.context.submission_id)
        self.events_path = os.path.join(self.submission_dir, "events.jsonl")

        # Verify path containment before writing or creating directories
        if not is_safely_contained(self.context.observer_dir, run_dir) or not is_safely_contained(run_dir, self.submission_dir):
            self.has_error = True
            self.error_message = f"Path escape detected: submission path '{self.submission_dir}' escapes observer_dir"

    def _append_event(self, record_kind: str, event_name: str, **kwargs: Any) -> None:
        if self.has_error:
            # If an earlier write or serialization failed, do not attempt further broken writes
            return
        self.sequence += 1
        event_id = f"evt_{self.context.submission_id}_{self.sequence}"
        now_utc = datetime.now(timezone.utc).isoformat()

        event = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.context.observer_run_id,
            "submission_id": self.context.submission_id,
            "source_hash": self.source_hash,
            "attempt_id": self.attempt_id,
            "tic_id": self.tic_id,
            "sector": self.sector,
            "record_kind": record_kind,
            "event_name": event_name,
            "sequence": self.sequence,
            "event_id": event_id,
            "timestamp": now_utc,
            **kwargs
        }

        try:
            os.makedirs(self.submission_dir, exist_ok=True)
            line = strict_json_dumps(event) + "\n"
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception as exc:
            # Observer persistence is its own channel: catch observer errors, mark invalid, do not raise
            self.has_error = True
            self.error_message = str(exc)

    def record_submission_begin(self) -> None:
        """Writes the initial submission BEGIN record before attempt_id generation."""
        self._append_event(
            record_kind="submission",
            event_name="BEGIN",
            attempt_id=None
        )

    def bind_attempt_id(self, attempt_id: str) -> None:
        """Binds the worker-generated attempt ID and logs attempt BEGIN."""
        self.attempt_id = str(attempt_id)
        self._append_event(
            record_kind="attempt",
            event_name="BEGIN",
            attempt_id=self.attempt_id
        )

    def record_spoc_search_start(
        self,
        target: str,
        mission: str,
        author: str,
        sector: int,
        exptime: int
    ) -> None:
        """Records SPOC search query start."""
        self._append_event(
            record_kind="search",
            event_name="START",
            gate_id="gate_spoc_search",
            query_params={
                "target": target,
                "mission": mission,
                "author": author,
                "sector": sector,
                "exptime": exptime
            },
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "source_hash": self.source_hash
            }
        )

    def record_spoc_search(
        self,
        num_products: int,
        selected_product_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Records SPOC search gate outcome."""
        eval_state = "EVALUATED" if error is None else "ERROR"
        prod_res = "pass" if num_products > 0 and error is None else "return"
        diag_out = "PASS" if num_products > 0 and error is None else ("REJECT" if num_products == 0 else "ERROR")
        reason = "SKIP_NO_DATA" if num_products == 0 and error is None else (f"ERROR: {error}" if error else None)

        evidence = {
            "num_products": {"value": num_products, "unit": "count", "availability": "OBSERVED"}
        }
        if selected_product_info:
            evidence["selected_product"] = {
                "value": selected_product_info,
                "unit": "metadata",
                "availability": "OBSERVED"
            }

        self._append_event(
            record_kind="gate",
            event_name="EVALUATE",
            gate_id="gate_spoc_search",
            evaluation_state=eval_state,
            production_result=prod_res,
            diagnostic_outcome=diag_out,
            reason_code=reason,
            evidence=evidence,
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "len(search_res) == 0",
                "source_hash": self.source_hash
            }
        )

    def record_spoc_search_error(self, exc: Exception) -> None:
        """Records SPOC search exception."""
        self.record_spoc_search(num_products=0, error=str(exc))

    def record_crowding_gate(
        self,
        crowdsap: float,
        passed: bool,
        threshold: float = 0.70,
        comparator: str = "<",
        is_default: bool = False
    ) -> None:
        """Records SPOC CROWDSAP aperture crowding gate."""
        prod_res = "pass" if passed else "return"
        diag_out = "PASS" if passed else "REJECT"
        reason = None if passed else "REJECT_CROWDSAP"
        provenance = "DEFAULT" if is_default else "OBSERVED"

        self._append_event(
            record_kind="gate",
            event_name="EVALUATE",
            gate_id="gate_crowding",
            evaluation_state="EVALUATED",
            production_result=prod_res,
            diagnostic_outcome=diag_out,
            reason_code=reason,
            provenance=provenance,
            thresholds=[{
                "name": "crowdsap_min",
                "value": threshold,
                "comparator": comparator,
                "unit": "dimensionless"
            }],
            evidence={
                "crowdsap": {
                    "value": crowdsap,
                    "unit": "dimensionless",
                    "availability": "DEFAULTED" if is_default else "OBSERVED",
                    "provenance": provenance
                }
            },
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "crowdsap < 0.70",
                "source_hash": self.source_hash
            }
        )

    def record_baseline_gate(
        self,
        span: float,
        min_period: float,
        max_period: float,
        max_p: float,
        passed: bool,
        threshold: Optional[float] = None,
        comparator: str = "<="
    ) -> None:
        """Records time baseline and period limit predicate."""
        prod_res = "pass" if passed else "return"
        diag_out = "PASS" if passed else "REJECT"
        reason = None if passed else "REJECT_BASELINE"
        thresh_val = min_period if threshold is None else threshold

        self._append_event(
            record_kind="gate",
            event_name="EVALUATE",
            gate_id="gate_baseline",
            evaluation_state="EVALUATED",
            production_result=prod_res,
            diagnostic_outcome=diag_out,
            reason_code=reason,
            thresholds=[{
                "name": "min_period",
                "value": thresh_val,
                "comparator": comparator,
                "unit": "day"
            }],
            evidence={
                "time_span": {"value": span, "unit": "day", "availability": "OBSERVED"},
                "min_period_configured": {"value": min_period, "unit": "day", "availability": "CONFIGURED"},
                "max_period_configured": {"value": max_period, "unit": "day", "availability": "CONFIGURED"},
                "effective_max_p": {"value": max_p, "unit": "day", "availability": "DERIVED"}
            },
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "max_p <= min_period",
                "source_hash": self.source_hash
            }
        )

    def record_bls_start(
        self,
        n_freq: int,
        min_freq: float,
        max_freq: float,
        min_dur: float,
        max_dur: float,
        n_dur: int
    ) -> None:
        """Records BLS search grid start."""
        self.search_call_index += 1
        search_id = f"{self.attempt_id}_s{self.search_call_index}"
        self._append_event(
            record_kind="search",
            event_name="START",
            gate_id="search_bls",
            search_id=search_id,
            evidence={
                "n_freq": {"value": n_freq, "unit": "count", "availability": "CONFIGURED"},
                "min_freq": {"value": min_freq, "unit": "1/day", "availability": "DERIVED"},
                "max_freq": {"value": max_freq, "unit": "1/day", "availability": "DERIVED"},
                "min_duration": {"value": min_dur, "unit": "day", "availability": "CONFIGURED"},
                "max_duration": {"value": max_dur, "unit": "day", "availability": "CONFIGURED"},
                "n_durations": {"value": n_dur, "unit": "count", "availability": "CONFIGURED"}
            },
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "BoxLeastSquares.power",
                "source_hash": self.source_hash
            }
        )

    def record_bls_search(
        self,
        n_freq: int,
        min_freq: float,
        max_freq: float,
        min_dur: float,
        max_dur: float,
        n_dur: int,
        window_pts: int,
        top_peak_indices: List[int],
        error: Optional[str] = None
    ) -> None:
        """Records BLS search grid parameters and returned top distinct peak indices."""
        search_id = f"{self.attempt_id}_s{self.search_call_index}"
        eval_state = "EVALUATED" if error is None else "ERROR"
        diag_out = "PASS" if error is None else "ERROR"

        self._append_event(
            record_kind="search",
            event_name="END",
            gate_id="search_bls",
            search_id=search_id,
            evaluation_state=eval_state,
            diagnostic_outcome=diag_out,
            evidence={
                "n_freq": {"value": n_freq, "unit": "count", "availability": "CONFIGURED"},
                "min_freq": {"value": min_freq, "unit": "1/day", "availability": "DERIVED"},
                "max_freq": {"value": max_freq, "unit": "1/day", "availability": "DERIVED"},
                "min_duration": {"value": min_dur, "unit": "day", "availability": "CONFIGURED"},
                "max_duration": {"value": max_dur, "unit": "day", "availability": "CONFIGURED"},
                "n_durations": {"value": n_dur, "unit": "count", "availability": "CONFIGURED"},
                "whitened_sde_window_pts": {"value": window_pts, "unit": "count", "availability": "CONFIGURED"},
                "top_peak_indices": {"value": list(top_peak_indices), "unit": "indices", "availability": "OBSERVED"},
                "error": {"value": error, "unit": "message", "availability": "OBSERVED" if error else "NONE"}
            },
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "BoxLeastSquares.power",
                "source_hash": self.source_hash
            }
        )

    def record_bls_error(self, exc: Exception) -> None:
        """Records BLS execution exception."""
        search_id = f"{self.attempt_id}_s{self.search_call_index}"
        self._append_event(
            record_kind="search",
            event_name="ERROR",
            gate_id="search_bls",
            search_id=search_id,
            evaluation_state="ERROR",
            diagnostic_outcome="ERROR",
            reason_code=f"BLS_ERROR: {exc}",
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "BoxLeastSquares.power",
                "source_hash": self.source_hash
            }
        )

    def record_low_sde_gate(
        self,
        peak_idx: int,
        rank_idx: int,
        sde: float,
        passed: bool,
        threshold: float = 6.0,
        comparator: str = "<"
    ) -> None:
        """Records low-SDE gate evaluation for a visited peak."""
        search_id = f"{self.attempt_id}_s{self.search_call_index}"
        peak_id = f"{search_id}_p{peak_idx}"
        prod_res = "pass" if passed else "break"
        diag_out = "PASS" if passed else "REJECT"
        reason = None if passed else "EMPTY_LOW_SDE"

        self._append_event(
            record_kind="gate",
            event_name="EVALUATE",
            gate_id="gate_low_sde",
            search_id=search_id,
            peak_id=peak_id,
            rank_idx=rank_idx,
            peak_idx=peak_idx,
            evaluation_state="EVALUATED",
            production_result=prod_res,
            diagnostic_outcome=diag_out,
            reason_code=reason,
            thresholds=[{
                "name": "sde_min",
                "value": threshold,
                "comparator": comparator,
                "unit": "dimensionless"
            }],
            evidence={
                "sde": {"value": sde, "unit": "dimensionless", "availability": "OBSERVED"}
            },
            source_ref={
                "module": "aldebaran.py",
                "function": "_analyze_single_target_impl",
                "line_predicate": "sde < 6.0",
                "source_hash": self.source_hash
            }
        )

    def record_target_result(
        self,
        status_code: str,
        message: str,
        has_candidate: bool = False,
        cand_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records terminal outcome of the target analysis."""
        cand_summary = None
        if cand_data:
            cand_summary = {
                "period_days": cand_data.get("period_days", cand_data.get("period")),
                "t0_btjd": cand_data.get("t0_btjd", cand_data.get("t0")),
                "duration_hours": cand_data.get("duration_hours"),
                "depth_ppm": cand_data.get("depth_ppm"),
                "sde": cand_data.get("sde"),
                "r_planet_earth": cand_data.get("r_planet_earth"),
            }

        self._append_event(
            record_kind="target_result",
            event_name="RESULT",
            status_code=status_code,
            message=message,
            has_candidate=has_candidate,
            candidate_summary=cand_summary,
            source_ref={
                "module": "aldebaran.py",
                "function": "make_res",
                "source_hash": self.source_hash
            }
        )

    def record_submission_final(self, status: str = "COMPLETE") -> None:
        """Finalizes the submission event stream."""
        final_status = "ERROR" if self.has_error else status
        self._append_event(
            record_kind="submission",
            event_name="FINAL",
            status=final_status,
            total_events=self.sequence + 1,
            observer_has_error=self.has_error,
            observer_error_message=self.error_message
        )


class RunLedger:
    """
    Parent-side run orchestrator.
    Manages run-level directory, run.jsonl, and manifest.json.
    Assigns submission contexts before dispatching to ProcessPoolExecutor.
    """

    def __init__(
        self,
        observer_dir: str,
        legacy_run_id: str = "default_run",
        observer_run_id: Optional[str] = None
    ):
        self.observer_dir = os.path.abspath(observer_dir)
        self.legacy_run_id = legacy_run_id
        if observer_run_id is None:
            self.observer_run_id = f"obs_{legacy_run_id}_{uuid.uuid4().hex[:8]}"
        else:
            self.observer_run_id = observer_run_id

        self.run_dir = os.path.join(self.observer_dir, self.observer_run_id)
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.run_log_path = os.path.join(self.run_dir, "run.jsonl")
        self.submissions: Dict[str, Dict[str, Any]] = {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.source_hash = get_module_source_hash()

        os.makedirs(self.run_dir, exist_ok=True)
        self._init_run_records()

    def _init_run_records(self) -> None:
        manifest_data = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.observer_run_id,
            "legacy_run_id": self.legacy_run_id,
            "created_at": self.created_at,
            "source_hash": self.source_hash,
            "coverage_manifest": COVERAGE_MANIFEST,
            "status": "RUNNING",
            "submissions": {}
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(strict_json_dumps(manifest_data) + "\n")

        run_begin_event = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.observer_run_id,
            "legacy_run_id": self.legacy_run_id,
            "record_kind": "run",
            "event_name": "BEGIN",
            "timestamp": self.created_at,
            "source_hash": self.source_hash,
            "coverage_manifest": COVERAGE_MANIFEST
        }
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(strict_json_dumps(run_begin_event) + "\n")

    def create_submission_context(
        self,
        tic_id: Union[str, int],
        sector: int,
        submission_id: Optional[str] = None
    ) -> ObserverContext:
        """
        Creates and registers a submission context before dispatching to worker.
        """
        tic_str = str(tic_id)
        if submission_id is None:
            clean_tic = tic_str.lstrip("0") or "0"
            submission_id = f"sub_{clean_tic}_s{sector}_{uuid.uuid4().hex[:6]}"

        events_rel = os.path.join(submission_id, "events.jsonl")
        self.submissions[submission_id] = {
            "submission_id": submission_id,
            "tic_id": tic_str,
            "sector": int(sector),
            "events_path": events_rel,
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
            "status": "DISPATCHED",
            "terminal_code": None,
            "attempt_id": None,
            "worker_error": None
        }
        self._update_manifest()

        # Write run-level dispatch event
        run_dispatch_event = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.observer_run_id,
            "legacy_run_id": self.legacy_run_id,
            "record_kind": "submission",
            "event_name": "DISPATCH",
            "submission_id": submission_id,
            "source_hash": self.source_hash,
            "tic_id": tic_str,
            "sector": int(sector),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self.run_log_path, "a", encoding="utf-8") as f:
                f.write(strict_json_dumps(run_dispatch_event) + "\n")
        except Exception:
            pass

        return ObserverContext(
            observer_dir=self.observer_dir,
            observer_run_id=self.observer_run_id,
            submission_id=submission_id
        )

    def record_submission_outcome(
        self,
        submission_id: str,
        future_result: Optional[Tuple[Any, ...]] = None,
        exc: Optional[Exception] = None
    ) -> None:
        """
        Reconciles submission outcome after future completion or worker failure.
        """
        if submission_id not in self.submissions:
            return

        sub = self.submissions[submission_id]
        if exc is not None:
            sub["status"] = "WORKER_FAILED"
            sub["worker_error"] = str(exc)
        elif future_result is not None:
            # AnalysisResult is (tic_id, status_code, message, cand_data)
            status_code = future_result[1] if len(future_result) > 1 else "UNKNOWN"
            sub["terminal_code"] = status_code
            sub["attempt_id"] = getattr(future_result, "attempt_id", None)
            sub["status"] = "COMPLETED" if status_code != "ERROR" else "ERROR"

        self._update_manifest()

        # Write run-level outcome event
        outcome_event = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.observer_run_id,
            "legacy_run_id": self.legacy_run_id,
            "record_kind": "submission",
            "event_name": "OUTCOME",
            "submission_id": submission_id,
            "source_hash": self.source_hash,
            "status": sub["status"],
            "terminal_code": sub.get("terminal_code"),
            "attempt_id": sub.get("attempt_id"),
            "worker_error": sub.get("worker_error"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self.run_log_path, "a", encoding="utf-8") as f:
                f.write(strict_json_dumps(outcome_event) + "\n")
        except Exception:
            pass

    def record_prequeue_failure(self, error_msg: str) -> None:
        """Records a pre-queue failure in run.jsonl before target enumeration."""
        prequeue_event = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.observer_run_id,
            "legacy_run_id": self.legacy_run_id,
            "record_kind": "run",
            "event_name": "PREQUEUE_FAILURE",
            "source_hash": self.source_hash,
            "error_message": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self.run_log_path, "a", encoding="utf-8") as f:
                f.write(strict_json_dumps(prequeue_event) + "\n")
        except Exception:
            pass

    def finalize_run(
        self,
        completed_count: int = 0,
        error_count: int = 0,
        candidate_count: int = 0,
        status: str = "FINALIZED"
    ) -> None:
        """Writes run FINAL record and finalizes manifest."""
        now_utc = datetime.now(timezone.utc).isoformat()
        run_final_event = {
            "schema_version": SCHEMA_VERSION,
            "observer_run_id": self.observer_run_id,
            "legacy_run_id": self.legacy_run_id,
            "record_kind": "run",
            "event_name": "FINAL",
            "source_hash": self.source_hash,
            "timestamp": now_utc,
            "status": status,
            "total_submissions": len(self.submissions),
            "completed_count": completed_count,
            "error_count": error_count,
            "candidate_count": candidate_count
        }
        with open(self.run_log_path, "a", encoding="utf-8") as f:
            f.write(strict_json_dumps(run_final_event) + "\n")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        manifest_data["status"] = status
        manifest_data["finalized_at"] = now_utc
        manifest_data["summary"] = {
            "total_submissions": len(self.submissions),
            "completed_count": completed_count,
            "error_count": error_count,
            "candidate_count": candidate_count
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(strict_json_dumps(manifest_data) + "\n")

    def _update_manifest(self) -> None:
        if not os.path.exists(self.manifest_path):
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["submissions"] = self.submissions
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                f.write(strict_json_dumps(data) + "\n")
        except Exception:
            pass


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def format_report(self) -> str:
        lines = [
            f"Validation Result: {'PASS' if self.valid else 'FAIL'}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Summary: {json.dumps(self.summary, indent=2)}"
        ]
        if self.errors:
            lines.append("Errors List:")
            for err in self.errors:
                lines.append(f"  - {err}")
        return "\n".join(lines)


def validate_run_ledger(
    observer_dir: str,
    observer_run_id: Optional[str] = None,
    expected_source_hash: Optional[str] = None
) -> ValidationResult:
    """
    Independent completeness validator.
    Inspects run manifest, run log, and all declared submission events.
    Fails closed on missing records, sequence breaks, tampered thresholds, or invalid states.
    """
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {
        "submissions_audited": 0,
        "events_audited": 0,
        "gates_verified": 0
    }

    if observer_run_id is None:
        if os.path.exists(os.path.join(observer_dir, "manifest.json")):
            run_dir = os.path.abspath(observer_dir)
        else:
            errors.append(f"Cannot locate manifest.json in {observer_dir}")
            return ValidationResult(False, errors, warnings, summary)
    else:
        run_dir = os.path.abspath(os.path.join(observer_dir, observer_run_id))

    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        errors.append(f"Missing run manifest: {manifest_path}")
        return ValidationResult(False, errors, warnings, summary)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        errors.append(f"Invalid JSON in manifest.json: {exc}")
        return ValidationResult(False, errors, warnings, summary)

    # 1. Manifest level validation
    schema_ver = manifest.get("schema_version")
    if schema_ver != SCHEMA_VERSION:
        errors.append(f"Manifest schema_version mismatch: got '{schema_ver}', expected '{SCHEMA_VERSION}'")

    manifest_status = manifest.get("status")
    if manifest_status not in ("FINALIZED", "ERROR"):
        errors.append(f"Run manifest is unfinalized: status='{manifest_status}'")

    manifest_run_id = manifest.get("observer_run_id")
    if not manifest_run_id:
        errors.append("Manifest missing observer_run_id")
    if expected_source_hash is not None:
        manifest_source_hash = manifest.get("source_hash")
        if not manifest_source_hash:
            errors.append("Manifest missing required source hash")
        elif manifest_source_hash != expected_source_hash:
            errors.append(
                f"Stale source hash in manifest: got {manifest_source_hash}, expected {expected_source_hash}"
            )

    # 2. Run log validation
    run_log_path = os.path.join(run_dir, "run.jsonl")
    run_events = []
    dispatched_sub_ids: List[str] = []
    reconciled_sub_ids: List[str] = []
    run_final_event = None
    run_final_events = []
    prequeue_failure_count = 0

    if not os.path.exists(run_log_path):
        errors.append(f"Missing run log: {run_log_path}")
    else:
        try:
            with open(run_log_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    evt = json.loads(line_s)
                    run_events.append((line_idx, evt))
                    if expected_source_hash is not None:
                        event_source_hash = evt.get("source_hash")
                        if not event_source_hash:
                            errors.append(f"Run event on line {line_idx} missing required source hash")
                        elif event_source_hash != expected_source_hash:
                            errors.append(
                                f"Stale source hash on run event line {line_idx}: "
                                f"got {event_source_hash}, expected {expected_source_hash}"
                            )
                    if evt.get("record_kind") == "submission" and evt.get("event_name") == "DISPATCH":
                        sub_id = evt.get("submission_id")
                        if sub_id:
                            dispatched_sub_ids.append(sub_id)
                    elif evt.get("record_kind") == "submission" and evt.get("event_name") == "OUTCOME":
                        sub_id = evt.get("submission_id")
                        if sub_id:
                            reconciled_sub_ids.append(sub_id)
                    elif evt.get("record_kind") == "run" and evt.get("event_name") == "PREQUEUE_FAILURE":
                        prequeue_failure_count += 1
                    elif evt.get("record_kind") == "run" and evt.get("event_name") == "FINAL":
                        run_final_events.append(evt)
                        run_final_event = evt
        except Exception as exc:
            errors.append(f"Invalid JSON in run.jsonl: {exc}")

        has_begin = any(e[1].get("record_kind") == "run" and e[1].get("event_name") == "BEGIN" for e in run_events)
        has_final = run_final_event is not None

        if not has_begin:
            errors.append("run.jsonl missing run BEGIN event")
        if not has_final:
            errors.append("run.jsonl missing run FINAL event")
        if len(run_final_events) > 1:
            errors.append(f"run.jsonl contains {len(run_final_events)} run FINAL events; expected exactly one")

    # 3. Submissions reconciliation
    submissions = manifest.get("submissions", {})
    if not isinstance(submissions, dict):
        errors.append("Manifest submissions field is not a dictionary")
        submissions = {}

    # Check for empty submissions when run log claims completed or dispatched
    completed_in_final = run_final_event.get("completed_count", 0) if run_final_event else 0
    if len(submissions) == 0 and completed_in_final > 0:
        errors.append(f"Manifest claims 0 submissions but run.jsonl FINAL claims {completed_in_final} completed submissions")
    if len(submissions) == 0 and len(dispatched_sub_ids) > 0:
        errors.append(f"Manifest claims 0 submissions but run.jsonl dispatched {len(dispatched_sub_ids)} submissions")

    # Verify dispatched submissions in run.jsonl match manifest
    if set(dispatched_sub_ids) != set(submissions.keys()) and (dispatched_sub_ids or submissions):
        missing_in_manifest = set(dispatched_sub_ids) - set(submissions.keys())
        if missing_in_manifest:
            errors.append(f"Submissions dispatched in run.jsonl missing from manifest: {sorted(missing_in_manifest)}")
        missing_in_log = set(submissions.keys()) - set(dispatched_sub_ids)
        if missing_in_log:
            errors.append(f"Submissions in manifest not found in run.jsonl dispatch log: {sorted(missing_in_log)}")

    if len(dispatched_sub_ids) != len(set(dispatched_sub_ids)):
        errors.append("run.jsonl contains duplicate submission DISPATCH events")
    if len(reconciled_sub_ids) != len(set(reconciled_sub_ids)):
        errors.append("run.jsonl contains duplicate submission OUTCOME events")
    if set(reconciled_sub_ids) != set(dispatched_sub_ids):
        missing_outcomes = set(dispatched_sub_ids) - set(reconciled_sub_ids)
        unexpected_outcomes = set(reconciled_sub_ids) - set(dispatched_sub_ids)
        if missing_outcomes:
            errors.append(f"Dispatched submissions missing run OUTCOME events: {sorted(missing_outcomes)}")
        if unexpected_outcomes:
            errors.append(f"run OUTCOME events without matching DISPATCH: {sorted(unexpected_outcomes)}")

    outcome_by_id = {
        evt.get("submission_id"): evt
        for _, evt in run_events
        if evt.get("record_kind") == "submission" and evt.get("event_name") == "OUTCOME"
    }
    for sub_id, outcome in outcome_by_id.items():
        manifest_sub = submissions.get(sub_id)
        if not isinstance(manifest_sub, dict):
            continue
        if manifest_sub.get("status") != outcome.get("status"):
            errors.append(
                f"Submission '{sub_id}' manifest status '{manifest_sub.get('status')}' "
                f"does not match run OUTCOME status '{outcome.get('status')}'"
            )
        if manifest_sub.get("terminal_code") != outcome.get("terminal_code"):
            errors.append(f"Submission '{sub_id}' manifest terminal_code does not match run OUTCOME")

    if run_final_event is not None:
        expected_total = len(dispatched_sub_ids)
        returned_statuses = {"COMPLETED", "ERROR"}
        error_statuses = {"ERROR", "WORKER_FAILED"}
        expected_completed = sum(
            1 for evt in outcome_by_id.values() if evt.get("status") in returned_statuses
        )
        expected_errors = (
            sum(1 for evt in outcome_by_id.values() if evt.get("status") in error_statuses)
            + prequeue_failure_count
        )
        if run_final_event.get("total_submissions") != expected_total:
            errors.append(
                f"Run FINAL total_submissions mismatch: got {run_final_event.get('total_submissions')}, "
                f"derived {expected_total} DISPATCH events"
            )
        if run_final_event.get("completed_count") != expected_completed:
            errors.append(
                f"Run FINAL completed_count mismatch: got {run_final_event.get('completed_count')}, "
                f"derived {expected_completed} completed worker outcomes"
            )
        if run_final_event.get("error_count") != expected_errors:
            errors.append(
                f"Run FINAL error_count mismatch: got {run_final_event.get('error_count')}, "
                f"derived {expected_errors} worker/prequeue errors"
            )
        if manifest_status != run_final_event.get("status"):
            errors.append(
                f"Manifest status '{manifest_status}' does not match run FINAL status "
                f"'{run_final_event.get('status')}'"
            )

        manifest_summary = manifest.get("summary")
        if not isinstance(manifest_summary, dict):
            errors.append("Finalized manifest missing summary")
        else:
            for count_name in ("total_submissions", "completed_count", "error_count", "candidate_count"):
                if manifest_summary.get(count_name) != run_final_event.get(count_name):
                    errors.append(
                        f"Manifest summary {count_name} does not match run FINAL "
                        f"({manifest_summary.get(count_name)} != {run_final_event.get(count_name)})"
                    )

    # Check disk directories against manifest
    try:
        disk_dirs = [
            d for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d)) and not d.startswith(".") and d != "__pycache__"
        ]
        disk_set = set(disk_dirs)
        manifest_set = set(submissions.keys())
        if disk_set != manifest_set:
            orphan_on_disk = disk_set - manifest_set
            if orphan_on_disk:
                errors.append(f"Undeclared submission directories found on disk: {sorted(orphan_on_disk)}")
            declared_missing_on_disk = manifest_set - disk_set
            if declared_missing_on_disk:
                errors.append(f"Manifest declared submissions missing on disk: {sorted(declared_missing_on_disk)}")
    except Exception as exc:
        errors.append(f"Failed to scan run directory on disk: {exc}")

    summary["submissions_audited"] = len(submissions)

    # 4. Detailed audit of each submission
    for sub_id, sub_info in submissions.items():
        sub_dir = os.path.join(run_dir, sub_id)
        if not is_safely_contained(run_dir, sub_dir):
            errors.append(f"Path escape detected in submission '{sub_id}'")
            continue

        rel_events_path = sub_info.get("events_path", os.path.join(sub_id, "events.jsonl"))
        sub_events_path = os.path.normpath(os.path.join(run_dir, rel_events_path))

        if not is_safely_contained(run_dir, sub_events_path):
            errors.append(f"Submission path escape: '{sub_events_path}' outside '{run_dir}'")
            continue

        if not os.path.exists(sub_events_path):
            errors.append(f"Missing events file for submission '{sub_id}': {sub_events_path}")
            continue

        sub_events: List[Dict[str, Any]] = []
        try:
            with open(sub_events_path, "r", encoding="utf-8") as f:
                for l_idx, line in enumerate(f, 1):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    sub_events.append(json.loads(line_s))
        except Exception as exc:
            errors.append(f"Invalid JSON in {sub_events_path}: {exc}")
            continue

        if not sub_events:
            errors.append(f"Submission '{sub_id}' has empty events.jsonl")
            continue

        summary["events_audited"] += len(sub_events)

        seen_seqs = set()
        seen_event_ids = set()
        attempt_id_bound: Optional[str] = None
        has_sub_begin = False
        has_sub_final = False
        has_attempt_begin = False
        terminal_status_code = None
        target_result_events: List[Dict[str, Any]] = []
        recorded_gate_ids = set()
        recorded_search_ids = set()
        bls_lifecycle: Dict[str, List[str]] = {}
        visited_peaks: List[int] = []
        sde_break_observed = False

        for idx, evt in enumerate(sub_events):
            seq = evt.get("sequence")
            eid = evt.get("event_id")

            # Monotonic sequence
            if seq != idx + 1:
                errors.append(f"Submission '{sub_id}' sequence mismatch at index {idx}: got {seq}, expected {idx+1}")
            if seq in seen_seqs:
                errors.append(f"Duplicate sequence number {seq} in submission '{sub_id}'")
            seen_seqs.add(seq)

            if eid in seen_event_ids:
                errors.append(f"Duplicate event_id '{eid}' in submission '{sub_id}'")
            seen_event_ids.add(eid)

            # Metadata envelope
            if evt.get("schema_version") != SCHEMA_VERSION:
                errors.append(f"Schema mismatch on event '{eid}': {evt.get('schema_version')}")
            if evt.get("observer_run_id") != manifest_run_id:
                errors.append(f"observer_run_id mismatch on event '{eid}': got {evt.get('observer_run_id')}, expected {manifest_run_id}")
            if evt.get("submission_id") != sub_id:
                errors.append(f"submission_id mismatch on event '{eid}': got {evt.get('submission_id')}, expected {sub_id}")
            if str(evt.get("tic_id")) != str(sub_info.get("tic_id")):
                errors.append(f"tic_id mismatch on event '{eid}': got {evt.get('tic_id')}, expected {sub_info.get('tic_id')}")
            if int(evt.get("sector", -1)) != int(sub_info.get("sector", -2)):
                errors.append(f"sector mismatch on event '{eid}': got {evt.get('sector')}, expected {sub_info.get('sector')}")

            # Source hash check
            if expected_source_hash is not None:
                event_source_hash = evt.get("source_hash")
                if not event_source_hash:
                    errors.append(f"Event '{eid}' missing required source hash in '{sub_id}'")
                elif event_source_hash != expected_source_hash:
                    errors.append(
                        f"Stale source hash on event '{eid}': got {event_source_hash}, "
                        f"expected {expected_source_hash}"
                    )
                src_ref = evt.get("source_ref", {})
                if isinstance(src_ref, dict):
                    shash = src_ref.get("source_hash")
                    if shash and shash != expected_source_hash:
                        errors.append(
                            f"Stale source hash in source_ref on event '{eid}': "
                            f"got {shash}, expected {expected_source_hash}"
                        )

            r_kind = evt.get("record_kind")
            e_name = evt.get("event_name")

            if r_kind == "submission" and e_name == "BEGIN":
                has_sub_begin = True
                if idx != 0:
                    errors.append(f"submission BEGIN must be first event in '{sub_id}', found at index {idx}")
                if evt.get("attempt_id") is not None:
                    errors.append(f"submission BEGIN must have attempt_id=null before generation in '{sub_id}'")

            elif r_kind == "attempt" and e_name == "BEGIN":
                has_attempt_begin = True
                attempt_id_bound = evt.get("attempt_id")
                if not attempt_id_bound:
                    errors.append(f"attempt BEGIN missing valid attempt_id in '{sub_id}'")

            elif r_kind == "gate":
                summary["gates_verified"] += 1
                gate_id = evt.get("gate_id")
                recorded_gate_ids.add(gate_id)

                if COVERAGE_MANIFEST.get(gate_id) == "NOT_INSTRUMENTED":
                    errors.append(f"Gate '{gate_id}' is declared NOT_INSTRUMENTED in S1 coverage manifest but was recorded as evaluated")

                if attempt_id_bound and evt.get("attempt_id") != attempt_id_bound:
                    errors.append(f"Gate '{gate_id}' attempt_id '{evt.get('attempt_id')}' != bound '{attempt_id_bound}'")

                # Missing provenance check
                src_ref = evt.get("source_ref")
                if not src_ref or not isinstance(src_ref, dict) or not src_ref.get("module") or not src_ref.get("function"):
                    errors.append(f"Gate '{gate_id}' missing required source_ref provenance in '{sub_id}'")

                if gate_id == "gate_crowding":
                    prov = evt.get("provenance")
                    if prov not in ("OBSERVED", "DEFAULT"):
                        errors.append(f"gate_crowding missing valid provenance ('OBSERVED' or 'DEFAULT'): got '{prov}'")

                    thresh_list = evt.get("thresholds", [])
                    c_thresh = next((t for t in thresh_list if t.get("name") == "crowdsap_min"), None)
                    if not c_thresh or c_thresh.get("value") != 0.70 or c_thresh.get("comparator") != "<":
                        errors.append(f"gate_crowding threshold altered in '{sub_id}': {c_thresh}")
                    if c_thresh and c_thresh.get("unit") != "dimensionless":
                        errors.append(f"gate_crowding unit mismatch: {c_thresh.get('unit')}")

                    ev_crowd = evt.get("evidence", {}).get("crowdsap", {}).get("value")
                    if ev_crowd is not None:
                        if ev_crowd < 0.70 and evt.get("diagnostic_outcome") == "PASS":
                            errors.append(f"gate_crowding mutation: crowdsap={ev_crowd} < 0.70 falsely marked PASS")

                elif gate_id == "gate_baseline":
                    thresh_list = evt.get("thresholds", [])
                    b_thresh = next((t for t in thresh_list if t.get("name") == "min_period"), None)
                    if not b_thresh or b_thresh.get("comparator") != "<=":
                        errors.append(f"gate_baseline comparator altered in '{sub_id}': {b_thresh}")
                    if b_thresh and b_thresh.get("unit") != "day":
                        errors.append(f"gate_baseline unit mismatch: {b_thresh.get('unit')}")

                elif gate_id == "gate_low_sde":
                    thresh_list = evt.get("thresholds", [])
                    s_thresh = next((t for t in thresh_list if t.get("name") == "sde_min"), None)
                    if not s_thresh or s_thresh.get("value") != 6.0 or s_thresh.get("comparator") != "<":
                        errors.append(f"gate_low_sde threshold altered in '{sub_id}': {s_thresh}")
                    if s_thresh and s_thresh.get("unit") != "dimensionless":
                        errors.append(f"gate_low_sde unit mismatch: {s_thresh.get('unit')}")

                    ev_sde = evt.get("evidence", {}).get("sde", {}).get("value")
                    peak_idx = evt.get("peak_idx")
                    visited_peaks.append(peak_idx)

                    if ev_sde is not None and ev_sde < 6.0:
                        sde_break_observed = True
                        if evt.get("production_result") != "break":
                            errors.append(f"gate_low_sde production_result should be 'break' for SDE < 6.0 in '{sub_id}'")
                        if evt.get("diagnostic_outcome") == "PASS":
                            errors.append(f"gate_low_sde mutation: sde={ev_sde} < 6.0 falsely marked PASS")
                    elif sde_break_observed:
                        errors.append(f"Subsequent peak evaluated after low-SDE break in '{sub_id}'")

            elif r_kind == "search":
                gate_id = evt.get("gate_id", "search_bls")
                recorded_search_ids.add(gate_id)
                if gate_id == "search_bls":
                    search_id = evt.get("search_id")
                    if not search_id:
                        errors.append(f"BLS search event '{eid}' missing search_id in '{sub_id}'")
                    else:
                        bls_lifecycle.setdefault(search_id, []).append(e_name)
                        if attempt_id_bound and evt.get("attempt_id") != attempt_id_bound:
                            errors.append(
                                f"BLS search '{search_id}' attempt_id '{evt.get('attempt_id')}' "
                                f"!= bound '{attempt_id_bound}'"
                            )

            elif r_kind == "target_result":
                terminal_status_code = evt.get("status_code")
                target_result_events.append(evt)
                if attempt_id_bound and evt.get("attempt_id") != attempt_id_bound:
                    errors.append(f"target_result attempt_id '{evt.get('attempt_id')}' != bound '{attempt_id_bound}'")

            elif r_kind == "submission" and e_name == "FINAL":
                has_sub_final = True
                if idx != len(sub_events) - 1:
                    errors.append(f"submission FINAL is not the last event in '{sub_id}' (index {idx}/{len(sub_events)})")
                if evt.get("status") not in ("COMPLETE", "ERROR"):
                    errors.append(f"submission FINAL invalid status '{evt.get('status')}' in '{sub_id}'")

        if not has_sub_begin:
            errors.append(f"Submission '{sub_id}' missing submission BEGIN")
        if not has_sub_final:
            errors.append(f"Submission '{sub_id}' missing submission FINAL (unfinalized submission)")
        if has_attempt_begin and terminal_status_code is None and not any(e.get("status") == "ERROR" for e in sub_events):
            errors.append(f"Submission '{sub_id}' had active attempt but missing target_result event")

        parent_outcome = outcome_by_id.get(sub_id)
        if parent_outcome is not None:
            parent_status = parent_outcome.get("status")
            if parent_status in ("COMPLETED", "ERROR"):
                if len(target_result_events) != 1:
                    errors.append(
                        f"Submission '{sub_id}' parent outcome '{parent_status}' requires exactly one "
                        f"worker target_result event; found {len(target_result_events)}"
                    )
                else:
                    target_result = target_result_events[0]
                    worker_status = target_result.get("status_code")
                    parent_terminal_code = parent_outcome.get("terminal_code")
                    manifest_terminal_code = sub_info.get("terminal_code")
                    if worker_status != parent_terminal_code:
                        errors.append(
                            f"Submission '{sub_id}' worker target_result status_code '{worker_status}' "
                            f"does not match parent terminal_code '{parent_terminal_code}'"
                        )
                    if worker_status != manifest_terminal_code:
                        errors.append(
                            f"Submission '{sub_id}' worker target_result status_code '{worker_status}' "
                            f"does not match manifest terminal_code '{manifest_terminal_code}'"
                        )
                    has_candidate = target_result.get("has_candidate")
                    expected_candidate = worker_status == "CANDIDATE"
                    if has_candidate is not expected_candidate:
                        errors.append(
                            f"Submission '{sub_id}' worker target_result has_candidate '{has_candidate}' "
                            f"does not match status_code '{worker_status}'"
                        )
                    if (target_result.get("candidate_summary") is not None) != expected_candidate:
                        errors.append(
                            f"Submission '{sub_id}' worker target_result candidate_summary presence "
                            f"does not match status_code '{worker_status}'"
                        )
            elif parent_status == "WORKER_FAILED" and target_result_events:
                errors.append(
                    f"Submission '{sub_id}' has a worker target_result but parent outcome is WORKER_FAILED"
                )

        for search_id, event_names in bls_lifecycle.items():
            if event_names.count("START") != 1:
                errors.append(
                    f"BLS search '{search_id}' must have exactly one START event in '{sub_id}'"
                )
            terminal_count = event_names.count("END") + event_names.count("ERROR")
            if terminal_count != 1:
                errors.append(
                    f"BLS search '{search_id}' must have exactly one END or ERROR terminal event "
                    f"in '{sub_id}'"
                )
            unexpected_names = [
                name for name in event_names if name not in ("START", "END", "ERROR")
            ]
            if unexpected_names:
                errors.append(
                    f"BLS search '{search_id}' has invalid lifecycle event(s) {unexpected_names} "
                    f"in '{sub_id}'"
                )

        # 5. Branch-required event coverage validation
        if terminal_status_code is not None:
            required_gates = []
            if terminal_status_code == "SKIP_NO_DATA":
                required_gates = ["gate_spoc_search"]
            elif terminal_status_code == "REJECT_CROWDSAP":
                required_gates = ["gate_spoc_search", "gate_crowding"]
            elif terminal_status_code == "REJECT_BASELINE":
                required_gates = ["gate_spoc_search", "gate_crowding", "gate_baseline"]
            elif terminal_status_code == "EMPTY_LOW_SDE":
                required_gates = ["gate_spoc_search", "gate_crowding", "gate_baseline", "gate_low_sde"]
                if "search_bls" not in recorded_search_ids and "search_bls" not in recorded_gate_ids:
                    errors.append(f"Branch required search 'search_bls' is missing for outcome 'EMPTY_LOW_SDE' in '{sub_id}'")
            elif terminal_status_code == "CANDIDATE" or terminal_status_code.startswith("REJECT_"):
                # Downstream outcome past low SDE
                required_gates = ["gate_spoc_search", "gate_crowding", "gate_baseline", "gate_low_sde"]
                if "search_bls" not in recorded_search_ids and "search_bls" not in recorded_gate_ids:
                    errors.append(f"Branch required search 'search_bls' is missing for outcome '{terminal_status_code}' in '{sub_id}'")

            for req_gate in required_gates:
                if req_gate not in recorded_gate_ids:
                    errors.append(f"Branch required gate '{req_gate}' is missing for outcome '{terminal_status_code}' in submission '{sub_id}'")

    valid = (len(errors) == 0)
    return ValidationResult(valid=valid, errors=errors, warnings=warnings, summary=summary)
