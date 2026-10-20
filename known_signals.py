"""
known_signals.py - Canonical Versioned Known-Signals Registry and Diagnostic Association.

Part of ALDEBARAN Exoplanet Search Engine (Task ALD-007A).
Provides:
  - Canonical data models: KnownSignal, HostCatalogContext, SourceSnapshot, KnownSignalsRegistry.
  - Ingestion parsers for NASA TOI, NASA Confirmed (pscomppars), ExoFOP CTOI, and VizieR TESS EB.
  - Preserves distinct signal rows and conflicting solutions per host (no silent deduplication or collapsing).
  - Preserves measured transit ephemerides regardless of discovery method (e.g. RV-discovered planets
    such as HIP 29442 c/d that have measured transit midpoints/durations).
  - Explicit provenance contract: original time format (BJD/BTJD), timescale (TDB/UTC), epoch zero point,
    asymmetric uncertainties (err1/err2), and distinct stable solution identifiers.
  - Idempotent catalog refresh with coherent per-source replacement and retirement.
  - Full 4-source tracking with truthful availability, caching, failure, and persistence status reporting.
  - Lossless positive integer TIC normalization rejecting invalid non-numeric/negative strings.
  - Strict geometry validation requiring finite positive period and epoch, and requiring duration for maskability.
  - Non-destructive diagnostic association metrics invariant to integer-cycle epoch shifts and
    properly transporting unequal periods to a common anchor/pivot without arbitrary acceptance cutoffs.
  - Diagnostic hypothetical transit mask geometry helper (labeled hypothetical/non-active).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from ephemeris_contract import (
    days_to_hours,
    hours_to_days,
    BTJD_OFFSET,
    bjd_to_btjd,
    btjd_to_bjd,
    calculate_cyclic_timing_residual as contract_calculate_cyclic_timing_residual
)

logger = logging.getLogger("aldebaran.known_signals")

SCHEMA_VERSION = "known_signals_v1"
BJD_BTJD_OFFSET = BTJD_OFFSET
CANONICAL_SOURCES = ("nasa_toi", "nasa_confirmed", "exofop_ctoi", "vizier_tess_eb")


def normalize_tic_id(val: Any) -> str:
    """
    Losslessly canonicalize TIC ID to cleaned string representation.
    TIC IDs must be valid positive integers.
    Strips 'TIC', whitespace, and leading zeros.
    Raises ValueError for None, NaN, non-numeric strings, negative numbers, or non-integer floats.
    """
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        raise ValueError("Cannot normalize None, NaN, or infinite TIC ID")

    # Handle int directly to preserve arbitrary integer precision
    if isinstance(val, int) and not isinstance(val, bool):
        if val <= 0:
            raise ValueError(f"Invalid TIC ID: {val}; must be a strictly positive integer")
        return str(val)

    if isinstance(val, float):
        if not val.is_integer() or val <= 0:
            raise ValueError(f"Invalid TIC ID: {val}; must be a strictly positive integer")
        if val >= 9007199254740992:  # 2**53 IEEE-754 double precision limit
            raise ValueError(f"Unsafe float TIC ID: {val}; exceeds IEEE-754 precision limit (2^53). Pass as int or str.")
        return str(int(val))

    s = str(val).strip()
    if s.upper().startswith("TIC"):
        s = s[3:].strip()
    if s.endswith(".0"):
        s_base = s[:-2]
        if s_base.isdigit():
            if int(s_base) >= 9007199254740992:
                raise ValueError(f"Unsafe float-formatted TIC ID string {val!r}; exceeds IEEE-754 precision limit (2^53). Pass as int or integer string.")
            s = s_base

    if not s.isdigit() or int(s) <= 0:
        raise ValueError(f"Invalid TIC ID: {val!r}; must be a strictly positive integer")

    return s.lstrip("0") or "0"


def convert_bjd_to_btjd(bjd: float) -> float:
    """Converts BJD to BTJD if > 2,400,000, otherwise preserves value."""
    return float(bjd_to_btjd(bjd, strict=False))


def convert_btjd_to_bjd(btjd: float) -> float:
    """Converts BTJD to BJD if < 100,000, otherwise preserves value."""
    if btjd < 100000.0:
        return float(btjd_to_bjd(btjd, strict=True))
    return float(btjd)


def compute_payload_sha256(data: Union[str, bytes, pd.DataFrame]) -> str:
    """Computes a reproducible SHA256 hex digest for string, bytes, or DataFrame."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    if isinstance(data, str):
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    if isinstance(data, pd.DataFrame):
        buf = data.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(buf).hexdigest()
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


def make_solution_id(
    prefix: str,
    record_id: Any,
    period: Any = None,
    epoch: Any = None,
    duration: Any = None,
    depth: Any = None,
    disambiguator: Any = None,
) -> str:
    """
    Constructs a stable, deterministic astronomical solution identifier.
    Decoupled from row order and presence/absence of other solutions in the table.
    Distinct conflicting solutions (different P, different T0, different duration)
    yield distinct IDs, while identical astrophysical solutions receive identical IDs.
    """
    clean_rec = str(record_id).strip().replace(" ", "_")
    base = f"{prefix}:{clean_rec}"
    parts = []
    if period is not None and pd.notna(period):
        parts.append(f"P{period}")
    if epoch is not None and pd.notna(epoch):
        parts.append(f"T0{epoch}")
    if duration is not None and pd.notna(duration):
        parts.append(f"D{duration}")
    if depth is not None and pd.notna(depth):
        parts.append(f"dep{depth}")
    if disambiguator is not None and pd.notna(disambiguator):
        parts.append(str(disambiguator).strip())
    if not parts:
        return f"{base}#unsolved"
    return f"{base}#" + "_".join(parts)



def normalize_disposition(source_catalog: str, raw_disp: Any) -> tuple[str, str]:
    """
    Maps source-specific disposition string to canonical category and retains raw string.
    Categories:
      - 'CONFIRMED_PLANET'
      - 'PLANET_CANDIDATE'
      - 'FALSE_POSITIVE'
      - 'ECLIPSING_BINARY'
      - 'UNKNOWN'
    """
    raw_str = "" if pd.isna(raw_disp) or raw_disp is None else str(raw_disp).strip()
    norm = raw_str.upper()

    if source_catalog in ("nasa_toi", "exofop_ctoi"):
        if norm in ("CP", "KP", "CONFIRMED"):
            return "CONFIRMED_PLANET", raw_str
        if norm in ("PC", "CANDIDATE", "PLANET CANDIDATE"):
            return "PLANET_CANDIDATE", raw_str
        if norm in ("FP", "FA", "FALSE POSITIVE", "FALSE ALARM"):
            return "FALSE_POSITIVE", raw_str
        if norm in ("EB", "ECLIPSING BINARY"):
            return "ECLIPSING_BINARY", raw_str
        return "UNKNOWN", raw_str

    if source_catalog == "nasa_confirmed":
        return "CONFIRMED_PLANET", raw_str or "CONFIRMED"

    if source_catalog == "vizier_tess_eb":
        return "ECLIPSING_BINARY", raw_str or "EB"

    return "UNKNOWN", raw_str


@dataclass
class KnownSignal:
    signal_id: str
    tic_id: str
    source_catalog: str
    source_record_id: str
    planet_or_signal_name: str
    disposition: str
    raw_disposition: str
    solution_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    discovery_method: Optional[str] = None
    period_days: Optional[float] = None
    period_err_days: Optional[float] = None
    period_err_pos: Optional[float] = None
    period_err_neg: Optional[float] = None
    epoch_bjd: Optional[float] = None
    epoch_btjd: Optional[float] = None
    epoch_err_days: Optional[float] = None
    epoch_err_pos: Optional[float] = None
    epoch_err_neg: Optional[float] = None
    original_time_format: Optional[str] = None  # 'BTJD', 'BJD', 'JD', 'UNKNOWN'
    original_time_scale: Optional[str] = None   # 'TDB', 'UTC', 'UNKNOWN'
    epoch_zero_point: Optional[float] = None    # 2457000.0, 0.0, etc.
    duration_hours: Optional[float] = None
    duration_days: Optional[float] = None
    duration_err_pos: Optional[float] = None
    duration_err_neg: Optional[float] = None
    depth_ppm: Optional[float] = None
    depth_err_pos: Optional[float] = None
    depth_err_neg: Optional[float] = None
    has_transit_ephemeris: bool = False
    maskable: bool = False
    unmaskable_reason: Optional[str] = None
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    raw_fields: dict[str, Any] = field(default_factory=dict)
    source_row_index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "tic_id": self.tic_id,
            "source_catalog": self.source_catalog,
            "source_record_id": self.source_record_id,
            "planet_or_signal_name": self.planet_or_signal_name,
            "disposition": self.disposition,
            "raw_disposition": self.raw_disposition,
            "solution_id": self.solution_id,
            "snapshot_id": self.snapshot_id,
            "discovery_method": self.discovery_method,
            "period_days": self.period_days,
            "period_err_days": self.period_err_days,
            "period_err_pos": self.period_err_pos,
            "period_err_neg": self.period_err_neg,
            "epoch_bjd": self.epoch_bjd,
            "epoch_btjd": self.epoch_btjd,
            "epoch_err_days": self.epoch_err_days,
            "epoch_err_pos": self.epoch_err_pos,
            "epoch_err_neg": self.epoch_err_neg,
            "original_time_format": self.original_time_format,
            "original_time_scale": self.original_time_scale,
            "epoch_zero_point": self.epoch_zero_point,
            "duration_hours": self.duration_hours,
            "duration_days": self.duration_days,
            "duration_err_pos": self.duration_err_pos,
            "duration_err_neg": self.duration_err_neg,
            "depth_ppm": self.depth_ppm,
            "depth_err_pos": self.depth_err_pos,
            "depth_err_neg": self.depth_err_neg,
            "has_transit_ephemeris": self.has_transit_ephemeris,
            "maskable": self.maskable,
            "unmaskable_reason": self.unmaskable_reason,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "aliases": list(self.aliases),
            "raw_fields": dict(self.raw_fields),
            "source_row_index": self.source_row_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnownSignal:
        return cls(
            signal_id=str(d["signal_id"]),
            tic_id=str(d["tic_id"]),
            source_catalog=str(d["source_catalog"]),
            source_record_id=str(d["source_record_id"]),
            planet_or_signal_name=str(d["planet_or_signal_name"]),
            disposition=str(d["disposition"]),
            raw_disposition=str(d.get("raw_disposition", "")),
            solution_id=d.get("solution_id"),
            snapshot_id=d.get("snapshot_id"),
            discovery_method=d.get("discovery_method"),
            period_days=float(d["period_days"]) if d.get("period_days") is not None else None,
            period_err_days=float(d["period_err_days"]) if d.get("period_err_days") is not None else None,
            period_err_pos=float(d["period_err_pos"]) if d.get("period_err_pos") is not None else None,
            period_err_neg=float(d["period_err_neg"]) if d.get("period_err_neg") is not None else None,
            epoch_bjd=float(d["epoch_bjd"]) if d.get("epoch_bjd") is not None else None,
            epoch_btjd=float(d["epoch_btjd"]) if d.get("epoch_btjd") is not None else None,
            epoch_err_days=float(d["epoch_err_days"]) if d.get("epoch_err_days") is not None else None,
            epoch_err_pos=float(d["epoch_err_pos"]) if d.get("epoch_err_pos") is not None else None,
            epoch_err_neg=float(d["epoch_err_neg"]) if d.get("epoch_err_neg") is not None else None,
            original_time_format=d.get("original_time_format"),
            original_time_scale=d.get("original_time_scale"),
            epoch_zero_point=float(d["epoch_zero_point"]) if d.get("epoch_zero_point") is not None else None,
            duration_hours=float(d["duration_hours"]) if d.get("duration_hours") is not None else None,
            duration_days=float(d["duration_days"]) if d.get("duration_days") is not None else None,
            duration_err_pos=float(d["duration_err_pos"]) if d.get("duration_err_pos") is not None else None,
            duration_err_neg=float(d["duration_err_neg"]) if d.get("duration_err_neg") is not None else None,
            depth_ppm=float(d["depth_ppm"]) if d.get("depth_ppm") is not None else None,
            depth_err_pos=float(d["depth_err_pos"]) if d.get("depth_err_pos") is not None else None,
            depth_err_neg=float(d["depth_err_neg"]) if d.get("depth_err_neg") is not None else None,
            has_transit_ephemeris=bool(d.get("has_transit_ephemeris", False)),
            maskable=bool(d.get("maskable", False)),
            unmaskable_reason=d.get("unmaskable_reason"),
            quarantined=bool(d.get("quarantined", False)),
            quarantine_reason=d.get("quarantine_reason"),
            aliases=list(d.get("aliases", [])),
            raw_fields=dict(d.get("raw_fields", {})),
            source_row_index=d.get("source_row_index"),
        )


@dataclass
class HostCatalogContext:
    tic_id: str
    signals: list[KnownSignal] = field(default_factory=list)
    source_statuses: dict[str, str] = field(default_factory=dict)
    coverage_status: str = "COMPLETE"  # 'COMPLETE', 'INCOMPLETE', 'UNAVAILABLE'
    registry_available: bool = True

    @property
    def has_signals(self) -> bool:
        return len(self.signals) > 0

    @property
    def has_confirmed_planets(self) -> bool:
        return any(s.disposition == "CONFIRMED_PLANET" and not s.quarantined for s in self.signals)

    @property
    def has_toi(self) -> bool:
        return any(s.source_catalog == "nasa_toi" and not s.quarantined for s in self.signals)

    @property
    def has_ctoi(self) -> bool:
        return any(s.source_catalog == "exofop_ctoi" and not s.quarantined for s in self.signals)

    @property
    def has_eb(self) -> bool:
        return any(
            (s.source_catalog == "vizier_tess_eb" or s.disposition == "ECLIPSING_BINARY") and not s.quarantined
            for s in self.signals
        )

    @property
    def has_false_positive(self) -> bool:
        return any(s.disposition == "FALSE_POSITIVE" and not s.quarantined for s in self.signals)

    @property
    def maskable_signals(self) -> list[KnownSignal]:
        return [s for s in self.signals if s.maskable and not s.quarantined]

    @property
    def confirmed_planets(self) -> list[KnownSignal]:
        return [s for s in self.signals if s.disposition == "CONFIRMED_PLANET" and not s.quarantined]

    @property
    def toi_signals(self) -> list[KnownSignal]:
        return [s for s in self.signals if s.source_catalog == "nasa_toi" and not s.quarantined]

    @property
    def ctoi_signals(self) -> list[KnownSignal]:
        return [s for s in self.signals if s.source_catalog == "exofop_ctoi" and not s.quarantined]

    @property
    def eb_signals(self) -> list[KnownSignal]:
        return [
            s for s in self.signals
            if (s.source_catalog == "vizier_tess_eb" or s.disposition == "ECLIPSING_BINARY") and not s.quarantined
        ]

    def get_signals(self, source_catalog: Optional[str] = None, maskable_only: bool = False) -> list[KnownSignal]:
        res = self.signals
        if source_catalog is not None:
            res = [s for s in res if s.source_catalog == source_catalog]
        if maskable_only:
            res = [s for s in res if s.maskable and not s.quarantined]
        return res

    def to_dict(self) -> dict[str, Any]:
        return {
            "tic_id": self.tic_id,
            "signals": [s.to_dict() for s in self.signals],
            "source_statuses": dict(self.source_statuses),
            "coverage_status": self.coverage_status,
            "registry_available": self.registry_available,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HostCatalogContext:
        return cls(
            tic_id=str(d["tic_id"]),
            signals=[KnownSignal.from_dict(s) for s in d.get("signals", [])],
            source_statuses=dict(d.get("source_statuses", {})),
            coverage_status=str(d.get("coverage_status", "COMPLETE")),
            registry_available=bool(d.get("registry_available", True)),
        )


@dataclass
class SourceSnapshot:
    source_name: str
    provider: str
    query_or_url: str
    retrieval_utc: str
    status: str  # 'SUCCESS', 'CACHED', 'FAILED', 'STALE', 'INCOMPLETE', 'UNAVAILABLE'
    row_count_raw: int
    row_count_valid: int
    row_count_quarantined: int
    payload_sha256: Optional[str] = None
    payload_hash_kind: str = "dataframe_serialization"
    raw_source_path_or_url: Optional[str] = None
    original_retrieval_utc: Optional[str] = None
    snapshot_id: Optional[str] = None
    error_message: Optional[str] = None
    last_attempt_status: Optional[str] = None
    last_attempt_utc: Optional[str] = None
    last_attempt_error: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if not self.snapshot_id:
            self.snapshot_id = f"{self.source_name}@{self.retrieval_utc}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "provider": self.provider,
            "query_or_url": self.query_or_url,
            "retrieval_utc": self.retrieval_utc,
            "status": self.status,
            "row_count_raw": self.row_count_raw,
            "row_count_valid": self.row_count_valid,
            "row_count_quarantined": self.row_count_quarantined,
            "payload_sha256": self.payload_sha256,
            "payload_hash_kind": self.payload_hash_kind,
            "raw_source_path_or_url": self.raw_source_path_or_url,
            "original_retrieval_utc": self.original_retrieval_utc,
            "snapshot_id": self.snapshot_id,
            "error_message": self.error_message,
            "last_attempt_status": self.last_attempt_status,
            "last_attempt_utc": self.last_attempt_utc,
            "last_attempt_error": self.last_attempt_error,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceSnapshot:
        return cls(
            source_name=str(d["source_name"]),
            provider=str(d.get("provider", "Unknown")),
            query_or_url=str(d.get("query_or_url", "")),
            retrieval_utc=str(d.get("retrieval_utc", "")),
            status=str(d.get("status", "UNKNOWN")),
            row_count_raw=int(d.get("row_count_raw", 0)),
            row_count_valid=int(d.get("row_count_valid", 0)),
            row_count_quarantined=int(d.get("row_count_quarantined", 0)),
            payload_sha256=d.get("payload_sha256"),
            payload_hash_kind=str(d.get("payload_hash_kind", "dataframe_serialization")),
            raw_source_path_or_url=d.get("raw_source_path_or_url"),
            original_retrieval_utc=d.get("original_retrieval_utc"),
            snapshot_id=d.get("snapshot_id"),
            error_message=d.get("error_message"),
            last_attempt_status=d.get("last_attempt_status"),
            last_attempt_utc=d.get("last_attempt_utc"),
            last_attempt_error=d.get("last_attempt_error"),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )


class KnownSignalsRegistry:
    """
    Canonical, explicit, versioned known-signals registry.
    Retains distinct source signal rows across NASA TOI, NASA Confirmed, ExoFOP CTOI, and VizieR EB.
    Ensures complete traceability, provenance, atomic JSON caching, schema validation, and idempotent refresh.
    """
    schema_version: str = SCHEMA_VERSION

    def __init__(self, created_utc: Optional[str] = None):
        self.created_utc = created_utc or datetime.now(timezone.utc).isoformat()
        self.source_snapshots: dict[str, SourceSnapshot] = {}
        self.attempt_snapshots: list[SourceSnapshot] = []
        self.hosts: dict[str, HostCatalogContext] = {}
        self.quarantine_records: list[KnownSignal] = []
        self.persistence_status: str = "NOT_ATTEMPTED"  # 'NOT_ATTEMPTED', 'PERSISTED', 'FAILED'
        self.persistence_error: Optional[str] = None
        self.persistence_path: Optional[str] = None

    def add_signal(self, signal: KnownSignal) -> None:
        if signal.tic_id == "quarantine" or signal.tic_id is None:
            self.quarantine_records.append(signal)
            return
        try:
            tid = normalize_tic_id(signal.tic_id)
        except Exception:
            self.quarantine_records.append(signal)
            return

        if not signal.snapshot_id and signal.source_catalog in self.source_snapshots:
            signal.snapshot_id = self.source_snapshots[signal.source_catalog].snapshot_id

        if tid not in self.hosts:
            self.hosts[tid] = HostCatalogContext(
                tic_id=tid,
                source_statuses=self.get_source_statuses(),
                coverage_status=self.get_coverage_status(),
                registry_available=True,
            )
        self.hosts[tid].signals.append(signal)

    def register_snapshot(self, snapshot: SourceSnapshot) -> None:
        self.source_snapshots[snapshot.source_name] = snapshot
        for host in self.hosts.values():
            host.source_statuses[snapshot.source_name] = snapshot.status
            host.coverage_status = self.get_coverage_status()
            for s in host.signals:
                if s.source_catalog == snapshot.source_name and not s.snapshot_id:
                    s.snapshot_id = snapshot.snapshot_id
        for s in self.quarantine_records:
            if s.source_catalog == snapshot.source_name and not s.snapshot_id:
                s.snapshot_id = snapshot.snapshot_id

    def record_failed_attempt(self, snapshot: SourceSnapshot, signals: Sequence[KnownSignal]) -> None:
        """
        Preserves failed attempt snapshot and its quarantine/raw signals without corrupting
        the selected successful source data or causing count mismatches in cache serialization.
        """
        if not snapshot.snapshot_id or "@attempt_" not in snapshot.snapshot_id:
            snapshot.snapshot_id = f"{snapshot.source_name}@attempt_{snapshot.retrieval_utc}#FAILED"
        for sig in signals:
            sig.snapshot_id = snapshot.snapshot_id
            sig.quarantined = True
            self.quarantine_records.append(sig)
        self.attempt_snapshots.append(snapshot)
        cov = self.get_coverage_status()
        for host in self.hosts.values():
            host.coverage_status = cov

    def replace_source_signals(
        self,
        source_catalog: str,
        signals: Sequence[KnownSignal],
        snapshot: SourceSnapshot,
    ) -> None:
        """
        Idempotently replaces all signals for a given source_catalog.
        Purges prior signals from this source across all hosts, removes empty hosts,
        purges prior quarantine_records for this source, adds new signals,
        registers the snapshot, and synchronizes source statuses.
        """
        # If an existing snapshot with a different snapshot_id is being replaced, archive it
        if source_catalog in self.source_snapshots:
            old_snap = self.source_snapshots[source_catalog]
            if old_snap.snapshot_id and old_snap.snapshot_id != snapshot.snapshot_id:
                if not any(a.snapshot_id == old_snap.snapshot_id for a in self.attempt_snapshots):
                    archived_snap = SourceSnapshot(
                        source_name=old_snap.source_name,
                        provider=old_snap.provider,
                        query_or_url=old_snap.query_or_url,
                        retrieval_utc=old_snap.retrieval_utc,
                        status="SUPERSEDED",
                        row_count_raw=old_snap.row_count_raw,
                        row_count_valid=0,
                        row_count_quarantined=old_snap.row_count_raw,
                        payload_sha256=old_snap.payload_sha256,
                        payload_hash_kind=old_snap.payload_hash_kind,
                        raw_source_path_or_url=old_snap.raw_source_path_or_url,
                        original_retrieval_utc=old_snap.original_retrieval_utc,
                        snapshot_id=old_snap.snapshot_id,
                        error_message="Superseded by fresh snapshot",
                        last_attempt_status=old_snap.last_attempt_status,
                        last_attempt_utc=old_snap.last_attempt_utc,
                        last_attempt_error=old_snap.last_attempt_error,
                    )
                    self.attempt_snapshots.append(archived_snap)
                    for tid, host in self.hosts.items():
                        for s in host.signals:
                            if s.source_catalog == source_catalog and (s.snapshot_id == old_snap.snapshot_id or not s.snapshot_id):
                                s_arch = copy.deepcopy(s)
                                s_arch.snapshot_id = old_snap.snapshot_id
                                s_arch.quarantined = True
                                s_arch.quarantine_reason = f"Superseded by snapshot {snapshot.snapshot_id}"
                                self.quarantine_records.append(s_arch)

        attempt_snapshot_ids = {a.snapshot_id for a in self.attempt_snapshots if a.snapshot_id}

        # 1. Remove previous signals for this source catalog from active hosts
        for tid, host in list(self.hosts.items()):
            host.signals = [s for s in host.signals if s.source_catalog != source_catalog]
            if not host.signals:
                del self.hosts[tid]

        # Preserve quarantine records belonging to historical/attempt snapshots
        self.quarantine_records = [
            s for s in self.quarantine_records
            if s.snapshot_id in attempt_snapshot_ids or s.source_catalog != source_catalog
        ]

        # 2. Add incoming signals
        for sig in signals:
            sig.snapshot_id = snapshot.snapshot_id
            self.add_signal(sig)

        # 3. Register snapshot
        self.source_snapshots[source_catalog] = snapshot

        # 4. Synchronize source statuses on all hosts
        cov = self.get_coverage_status()
        statuses = self.get_source_statuses()
        for host in self.hosts.values():
            host.source_statuses = dict(statuses)
            host.coverage_status = cov

    def record_source_attempt(
        self,
        source_name: str,
        status: str,
        error_message: Optional[str] = None,
        provider: str = "Unknown",
        query_or_url: str = "",
    ) -> None:
        """
        Truthfully records an attempt status for a source without destroying good cached data.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        if source_name in self.source_snapshots:
            snap = self.source_snapshots[source_name]
            snap.last_attempt_status = status
            snap.last_attempt_utc = now_utc
            snap.last_attempt_error = error_message
            if snap.status == "SUCCESS":
                snap.status = "CACHED"
        else:
            self.source_snapshots[source_name] = SourceSnapshot(
                source_name=source_name,
                provider=provider,
                query_or_url=query_or_url,
                retrieval_utc=now_utc,
                status=status,
                row_count_raw=0,
                row_count_valid=0,
                row_count_quarantined=0,
                error_message=error_message,
                last_attempt_status=status,
                last_attempt_utc=now_utc,
                last_attempt_error=error_message,
            )
        cov = self.get_coverage_status()
        statuses = self.get_source_statuses()
        for host in self.hosts.values():
            host.source_statuses = dict(statuses)
            host.coverage_status = cov

    def get_source_statuses(self) -> dict[str, str]:
        statuses = {}
        for src in CANONICAL_SOURCES:
            if src in self.source_snapshots:
                statuses[src] = self.source_snapshots[src].status
            else:
                statuses[src] = "UNAVAILABLE"
        return statuses

    def get_coverage_status(self) -> str:
        statuses = self.get_source_statuses()
        for snap in self.source_snapshots.values():
            if snap.last_attempt_status in ("FAILED", "INCOMPLETE", "UNAVAILABLE"):
                return "INCOMPLETE"
        if all(st in ("SUCCESS", "CACHED") for st in statuses.values()):
            return "COMPLETE"
        if all(st in ("UNAVAILABLE", "FAILED") for st in statuses.values()):
            return "UNAVAILABLE"
        return "INCOMPLETE"

    def get_host_context(self, tic_id: Union[str, int]) -> HostCatalogContext:
        try:
            tid_norm = normalize_tic_id(tic_id)
        except Exception:
            return HostCatalogContext(
                tic_id=str(tic_id),
                signals=[],
                source_statuses=self.get_source_statuses(),
                coverage_status="UNAVAILABLE",
                registry_available=True,
            )
        if tid_norm in self.hosts:
            ctx = self.hosts[tid_norm]
            ctx.coverage_status = self.get_coverage_status()
            ctx.source_statuses = self.get_source_statuses()
            ctx.registry_available = True
            return ctx
        return HostCatalogContext(
            tic_id=tid_norm,
            signals=[],
            source_statuses=self.get_source_statuses(),
            coverage_status=self.get_coverage_status(),
            registry_available=True,
        )

    def total_signals(self) -> int:
        return sum(len(h.signals) for h in self.hosts.values()) + len(self.quarantine_records)

    def total_valid_signals(self) -> int:
        host_valid = sum(len([s for s in h.signals if not s.quarantined]) for h in self.hosts.values())
        quar_valid = len([s for s in self.quarantine_records if not s.quarantined])
        return host_valid + quar_valid

    def total_quarantined_signals(self) -> int:
        host_quar = sum(len([s for s in h.signals if s.quarantined]) for h in self.hosts.values())
        quar_quar = len([s for s in self.quarantine_records if s.quarantined])
        return host_quar + quar_quar

    def export_legacy_catalog_tics(self) -> set[str]:
        """Returns set of all TIC IDs known to the registry for legacy compatibility."""
        return set(self.hosts.keys())

    def export_legacy_eb_tics(self) -> set[str]:
        """Returns set of all TIC IDs known to host EBs for legacy compatibility."""
        return {tid for tid, h in self.hosts.items() if h.has_eb}

    def export_legacy_toi_ephemerides(self) -> dict[str, list[dict[str, Any]]]:
        """
        Exports TOI signals in the legacy dict format required by existing Aldebaran consumers:
          {'toi': str, 'period': float, 't0': float, 'duration': float, 'depth_ppm': float, 'disp': str}
        NOTE:
          - duration is converted to days, with legacy 0.08d fallback if unavailable.
          - disp defaults to 'PC' if unavailable.
          These fallbacks are STRICTLY confined to this legacy adapter; canonical KnownSignal objects
          preserve duration=None and unaugmented raw dispositions.
        """
        legacy: dict[str, list[dict[str, Any]]] = {}
        for tid, host in self.hosts.items():
            toi_list = []
            for s in host.signals:
                if s.source_catalog == "nasa_toi" and not s.quarantined and s.period_days is not None and s.epoch_btjd is not None:
                    dur_days = (hours_to_days(s.duration_hours)) if s.duration_hours is not None else 0.08
                    toi_list.append({
                        "toi": str(s.source_record_id),
                        "period": float(s.period_days),
                        "t0": float(s.epoch_btjd),
                        "duration": float(dur_days),
                        "depth_ppm": float(s.depth_ppm) if s.depth_ppm is not None else float("nan"),
                        "disp": str(s.raw_disposition) if s.raw_disposition else "PC",
                    })
            if toi_list:
                legacy[tid] = toi_list
        return legacy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "source_snapshots": {k: s.to_dict() for k, s in sorted(self.source_snapshots.items())},
            "attempt_snapshots": [s.to_dict() for s in self.attempt_snapshots],
            "hosts": {k: h.to_dict() for k, h in sorted(self.hosts.items())},
            "quarantine_records": [s.to_dict() for s in self.quarantine_records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], mark_as_cached: bool = True) -> KnownSignalsRegistry:
        if not isinstance(data, dict):
            raise ValueError(f"Registry data root must be dict, got {type(data).__name__}")
        ver = data.get("schema_version")
        if ver != cls.schema_version:
            raise ValueError(f"Registry schema version mismatch: expected '{cls.schema_version}', got '{ver}'")
        if "source_snapshots" not in data or "hosts" not in data:
            raise ValueError("Registry JSON missing required keys: 'source_snapshots' and/or 'hosts'")

        reg = cls(created_utc=data.get("created_utc"))

        # 1. Parse source snapshots
        for k, snap_dict in data.get("source_snapshots", {}).items():
            snap = SourceSnapshot.from_dict(snap_dict)
            if snap.source_name != k:
                raise ValueError(
                    f"Malformed cache: snapshot internal source_name '{snap.source_name}' does not match key '{k}'"
                )
            if snap.status in ("SUCCESS", "CACHED") or snap.row_count_raw > 0:
                if not snap.payload_sha256 or not isinstance(snap.payload_sha256, str) or not snap.payload_sha256.strip():
                    raise ValueError(f"Malformed cache: missing or empty payload_sha256 for source '{k}'")
            if mark_as_cached and snap.status == "SUCCESS":
                snap.status = "CACHED"
            reg.source_snapshots[k] = snap

        # 2. Parse attempt / historical snapshots
        for snap_dict in data.get("attempt_snapshots", []):
            snap = SourceSnapshot.from_dict(snap_dict)
            reg.attempt_snapshots.append(snap)

        all_snapshots: dict[str, SourceSnapshot] = {}
        for s in list(reg.source_snapshots.values()) + reg.attempt_snapshots:
            if s.snapshot_id:
                all_snapshots[s.snapshot_id] = s

        # 3. Parse quarantine records
        for q_dict in data.get("quarantine_records", []):
            sig = KnownSignal.from_dict(q_dict)
            if not sig.quarantined:
                raise ValueError(f"Malformed cache: unquarantined signal in quarantine_records: {sig.signal_id}")
            reg.quarantine_records.append(sig)

        # 4. Check for orphan signals: if hosts has signals but source_snapshots is empty
        host_dict_raw = data.get("hosts", {})
        if host_dict_raw and not reg.source_snapshots:
            raise ValueError("Malformed cache: host signals present but source_snapshots is empty")

        # 5. Parse hosts and validate signals
        for k, host_dict in host_dict_raw.items():
            norm_k = normalize_tic_id(k)
            if norm_k != str(k).strip():
                raise ValueError(f"Malformed cache: host key '{k}' is not canonically normalized (expected '{norm_k}')")
            host = HostCatalogContext.from_dict(host_dict)
            if host.tic_id != norm_k:
                raise ValueError(f"Malformed cache: host TIC mismatch (key '{norm_k}' != content '{host.tic_id}')")
            for sig in host.signals:
                if sig.tic_id != norm_k:
                    raise ValueError(f"Malformed cache: signal TIC '{sig.tic_id}' does not match host TIC '{norm_k}'")

                # Validate snapshot reference
                if not sig.snapshot_id:
                    if sig.source_catalog in reg.source_snapshots:
                        sig.snapshot_id = reg.source_snapshots[sig.source_catalog].snapshot_id
                    else:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} has missing snapshot_id with no matching source snapshot")
                if sig.snapshot_id not in all_snapshots:
                    raise ValueError(f"Malformed cache: signal {sig.signal_id} references unknown snapshot_id '{sig.snapshot_id}'")
                linked_snap = all_snapshots[sig.snapshot_id]
                if sig.source_catalog != linked_snap.source_name:
                    raise ValueError(
                        f"Malformed cache: signal {sig.signal_id} source_catalog '{sig.source_catalog}' "
                        f"disagrees with snapshot source_name '{linked_snap.source_name}'"
                    )

                # C3: Required presence / consistency implied by flags
                if sig.maskable:
                    if sig.quarantined:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} cannot be both maskable=True and quarantined=True")
                    if not sig.has_transit_ephemeris:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but has_transit_ephemeris=False")
                    if sig.period_days is None:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but period_days is None")
                    if not math.isfinite(sig.period_days) or sig.period_days <= 0:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but has non-positive/non-finite period_days: {sig.period_days}")
                    if sig.epoch_btjd is None or not math.isfinite(sig.epoch_btjd):
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but has missing/non-finite epoch_btjd: {sig.epoch_btjd}")
                    if sig.duration_hours is None and sig.duration_days is None:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but has neither duration_hours nor duration_days")
                    if sig.duration_hours is not None and (not math.isfinite(sig.duration_hours) or sig.duration_hours <= 0):
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but has invalid duration_hours: {sig.duration_hours}")
                    if sig.duration_days is not None and (not math.isfinite(sig.duration_days) or sig.duration_days <= 0):
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} is maskable=True but has invalid duration_days: {sig.duration_days}")
                    if sig.duration_hours is not None and sig.duration_days is not None:
                        if abs(sig.duration_hours - days_to_hours(sig.duration_days)) > 1e-3:
                            raise ValueError(f"Malformed cache: signal {sig.signal_id} duration_hours ({sig.duration_hours}) inconsistent with duration_days ({sig.duration_days})")

                if sig.has_transit_ephemeris:
                    if sig.period_days is None or not math.isfinite(sig.period_days) or sig.period_days <= 0:
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} has has_transit_ephemeris=True but invalid/missing period_days: {sig.period_days}")
                    if sig.epoch_btjd is None or not math.isfinite(sig.epoch_btjd):
                        raise ValueError(f"Malformed cache: signal {sig.signal_id} has has_transit_ephemeris=True but invalid/missing epoch_btjd: {sig.epoch_btjd}")

                if not sig.quarantined:
                    if sig.period_days is not None:
                        if not math.isfinite(sig.period_days) or sig.period_days <= 0:
                            raise ValueError(f"Malformed cache: non-finite or non-positive period {sig.period_days} on unquarantined signal {sig.signal_id}")
                    if sig.epoch_btjd is not None and (not math.isfinite(sig.epoch_btjd) or math.isnan(sig.epoch_btjd)):
                        raise ValueError(f"Malformed cache: non-finite epoch_btjd {sig.epoch_btjd} on unquarantined signal {sig.signal_id}")
                    if sig.epoch_bjd is not None and (not math.isfinite(sig.epoch_bjd) or math.isnan(sig.epoch_bjd)):
                        raise ValueError(f"Malformed cache: non-finite epoch_bjd {sig.epoch_bjd} on unquarantined signal {sig.signal_id}")
                    # C3: Validate BJD / BTJD mathematical consistency
                    if sig.epoch_bjd is not None and sig.epoch_btjd is not None:
                        expected_bjd = sig.epoch_btjd + BJD_BTJD_OFFSET
                        if abs(sig.epoch_bjd - expected_bjd) > 1e-3:
                            raise ValueError(
                                f"Malformed cache: inconsistent epoch_bjd ({sig.epoch_bjd}) and epoch_btjd ({sig.epoch_btjd}) on signal {sig.signal_id}; "
                                f"expected epoch_bjd ~ {expected_bjd}"
                            )

            if mark_as_cached:
                for src in host.source_statuses:
                    if host.source_statuses[src] == "SUCCESS":
                        host.source_statuses[src] = "CACHED"
            reg.hosts[norm_k] = host

        # Also validate snapshot links on quarantine records
        for sig in reg.quarantine_records:
            if not sig.snapshot_id and sig.source_catalog in reg.source_snapshots:
                sig.snapshot_id = reg.source_snapshots[sig.source_catalog].snapshot_id
            if sig.snapshot_id and sig.snapshot_id not in all_snapshots:
                raise ValueError(f"Malformed cache: quarantined signal {sig.signal_id} references unknown snapshot_id '{sig.snapshot_id}'")

        # 6. Strict count consistency validation per snapshot
        all_signals = [s for h in reg.hosts.values() for s in h.signals] + reg.quarantine_records
        for snap in list(reg.source_snapshots.values()) + reg.attempt_snapshots:
            matching = [s for s in all_signals if s.snapshot_id == snap.snapshot_id]
            actual_total = len(matching)
            actual_valid = sum(1 for s in matching if not s.quarantined)
            actual_quar = sum(1 for s in matching if s.quarantined)

            if snap.row_count_raw != actual_total or snap.row_count_valid != actual_valid or snap.row_count_quarantined != actual_quar:
                raise ValueError(
                    f"Malformed cache: row count mismatch for snapshot '{snap.snapshot_id}': "
                    f"snapshot claims {snap.row_count_raw} raw ({snap.row_count_valid} valid, {snap.row_count_quarantined} quarantined), "
                    f"but records contain {actual_total} raw ({actual_valid} valid, {actual_quar} quarantined)"
                )

        return reg

    def save_to_json(self, path: str) -> None:
        """Atomically persists registry to JSON using a unique temporary file and atomic replace."""
        temp_path = None
        try:
            dir_name = os.path.dirname(os.path.abspath(path))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            temp_path = os.path.join(dir_name, f".tmp_{os.path.basename(path)}_{uuid.uuid4().hex}.json")
            payload = self.to_dict()
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
            self.persistence_status = "PERSISTED"
            self.persistence_path = path
            self.persistence_error = None
        except Exception as exc:
            self.persistence_status = "FAILED"
            self.persistence_error = str(exc)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise

    @classmethod
    def load_from_json(cls, path: str, mark_as_cached: bool = True) -> KnownSignalsRegistry:
        """Loads and schema-validates a registry from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, mark_as_cached=mark_as_cached)


# ---------------------------------------------------------------------------
# Ingestion Parsers for Individual Sources
# ---------------------------------------------------------------------------

def ingest_nasa_toi_df(
    df: pd.DataFrame,
    source_name: str = "nasa_toi",
    provider: str = "NASA Exoplanet Archive",
    query_or_url: str = "toi",
    retrieval_utc: Optional[str] = None,
    raw_payload_sha256: Optional[str] = None,
    payload_hash_kind: str = "dataframe_serialization",
    raw_source_path_or_url: Optional[str] = None,
    original_retrieval_utc: Optional[str] = None,
) -> tuple[list[KnownSignal], SourceSnapshot]:
    """
    Parses NASA TOI catalog rows into KnownSignal records.
    Preserves all rows, units (pl_trandurh in hours, pl_trandep in ppm), and TFOPWG disposition.
    """
    retrieval_utc = retrieval_utc or datetime.now(timezone.utc).isoformat()
    raw_sha = raw_payload_sha256 or compute_payload_sha256(df)
    signals: list[KnownSignal] = []
    quarantined_count = 0

    col_map = {c.lower(): c for c in df.columns}
    tid_col = col_map.get("tid")
    toi_col = col_map.get("toi")
    tmid_col = col_map.get("pl_tranmid")
    per_col = col_map.get("pl_orbper")
    dur_col = col_map.get("pl_trandurh")
    dep_col = col_map.get("pl_trandep")
    disp_col = col_map.get("tfopwg_disp")
    per_err_col = col_map.get("pl_orbpererr1")
    per_err2_col = col_map.get("pl_orbpererr2")
    tmid_err_col = col_map.get("pl_tranmiderr1")
    tmid_err2_col = col_map.get("pl_tranmiderr2")
    dur_err_col = col_map.get("pl_trandurherr1")
    dur_err2_col = col_map.get("pl_trandurherr2")
    dep_err_col = col_map.get("pl_trandeperr1")
    dep_err2_col = col_map.get("pl_trandeperr2")

    if not tid_col or not toi_col:
        error_msg = f"Missing required columns in NASA TOI table: tid={tid_col}, toi={toi_col}"
        snapshot_id = f"{source_name}@attempt_{retrieval_utc}#FAILED"
        for idx, row in df.iterrows():
            raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            signals.append(
                KnownSignal(
                    signal_id=f"toi:missing_header_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=str(row.get(toi_col, f"row_{idx}") if toi_col else f"row_{idx}"),
                    planet_or_signal_name=f"TOI-row-{idx}",
                    disposition="UNKNOWN",
                    raw_disposition="",
                    solution_id=f"toi:missing_header_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=error_msg,
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
        snapshot = SourceSnapshot(
            source_name=source_name,
            provider=provider,
            query_or_url=query_or_url,
            retrieval_utc=retrieval_utc,
            status="FAILED",
            row_count_raw=len(df),
            row_count_valid=0,
            row_count_quarantined=len(df),
            payload_sha256=raw_sha,
            payload_hash_kind=payload_hash_kind,
            raw_source_path_or_url=raw_source_path_or_url,
            original_retrieval_utc=original_retrieval_utc,
            snapshot_id=snapshot_id,
            error_message=error_msg,
        )
        return signals, snapshot

    for idx, row in df.iterrows():
        raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        snapshot_id = f"{source_name}@{retrieval_utc}"
        try:
            tid_str = normalize_tic_id(row[tid_col])
        except Exception as exc:
            quarantined_count += 1
            signals.append(
                KnownSignal(
                    signal_id=f"toi:invalid_tic_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=str(row.get(toi_col, f"row_{idx}")),
                    planet_or_signal_name=f"TOI-{row.get(toi_col, idx)}",
                    disposition="UNKNOWN",
                    raw_disposition=str(row.get(disp_col, "")),
                    solution_id=f"toi:invalid_tic_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=f"Invalid TIC ID: {exc}",
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
            continue

        toi_num = str(row[toi_col]).strip()
        signal_id = f"toi:{toi_num}"
        raw_disp = row.get(disp_col)
        disp_cat, raw_disp_str = normalize_disposition("nasa_toi", raw_disp)

        period = None
        period_err = None
        period_err_pos = None
        period_err_neg = None
        epoch_bjd = None
        epoch_btjd = None
        epoch_err = None
        epoch_err_pos = None
        epoch_err_neg = None
        orig_time_format = "BTJD"
        orig_time_scale = "TDB"
        epoch_zero_point = BJD_BTJD_OFFSET
        dur_h = None
        dur_d = None
        dur_err_pos = None
        dur_err_neg = None
        depth_ppm = None
        depth_err_pos = None
        depth_err_neg = None
        quarantined = False
        quarantine_reason = None

        if per_col and pd.notna(row.get(per_col)):
            try:
                period = float(row[per_col])
                if not math.isfinite(period) or period <= 0:
                    quarantined = True
                    quarantine_reason = f"Non-finite or non-positive orbital period: {period}"
            except Exception:
                quarantined = True
                quarantine_reason = f"Unparseable period: {row.get(per_col)}"

        if per_err_col and pd.notna(row.get(per_err_col)):
            try:
                period_err_pos = float(row[per_err_col])
                period_err = abs(period_err_pos)
            except Exception:
                pass
        if per_err2_col and pd.notna(row.get(per_err2_col)):
            try:
                period_err_neg = float(row[per_err2_col])
            except Exception:
                pass

        if tmid_col and pd.notna(row.get(tmid_col)):
            try:
                tmid = float(row[tmid_col])
                if not math.isfinite(tmid):
                    quarantined = True
                    quarantine_reason = f"Non-finite transit midpoint: {tmid}"
                else:
                    # In NASA TOI table, pl_tranmid is officially in BJD (timescale TDB, zero-point 0.0).
                    # If tmid > 2,400,000, convert to BTJD by subtracting 2457000.
                    if tmid > 2400000.0:
                        epoch_bjd = tmid
                        epoch_btjd = tmid - BJD_BTJD_OFFSET
                        orig_time_format = "BJD"
                        orig_time_scale = "TDB"
                        epoch_zero_point = 0.0
                    else:
                        epoch_btjd = tmid
                        epoch_bjd = tmid + BJD_BTJD_OFFSET
                        orig_time_format = "BTJD"
                        orig_time_scale = "TDB"
                        epoch_zero_point = BJD_BTJD_OFFSET
            except Exception:
                quarantined = True
                quarantine_reason = f"Unparseable transit midpoint: {row.get(tmid_col)}"

        if tmid_err_col and pd.notna(row.get(tmid_err_col)):
            try:
                epoch_err_pos = float(row[tmid_err_col])
                epoch_err = abs(epoch_err_pos)
            except Exception:
                pass
        if tmid_err2_col and pd.notna(row.get(tmid_err2_col)):
            try:
                epoch_err_neg = float(row[tmid_err2_col])
            except Exception:
                pass

        if dur_col and pd.notna(row.get(dur_col)):
            try:
                dur_val = float(row[dur_col])
                if math.isfinite(dur_val) and dur_val > 0:
                    dur_h = dur_val
                    dur_d = hours_to_days(dur_h)
            except Exception:
                pass
        if dur_err_col and pd.notna(row.get(dur_err_col)):
            try:
                dur_err_pos = float(row[dur_err_col])
            except Exception:
                pass
        if dur_err2_col and pd.notna(row.get(dur_err2_col)):
            try:
                dur_err_neg = float(row[dur_err2_col])
            except Exception:
                pass

        if dep_col and pd.notna(row.get(dep_col)):
            try:
                dep_val = float(row[dep_col])
                if math.isfinite(dep_val) and dep_val > 0:
                    depth_ppm = dep_val
            except Exception:
                pass
        if dep_err_col and pd.notna(row.get(dep_err_col)):
            try:
                depth_err_pos = float(row[dep_err_col])
            except Exception:
                pass
        if dep_err2_col and pd.notna(row.get(dep_err2_col)):
            try:
                depth_err_neg = float(row[dep_err2_col])
            except Exception:
                pass

        has_ephem = bool(period is not None and epoch_btjd is not None and not quarantined)
        if has_ephem:
            if dur_h is not None and dur_h > 0:
                maskable = True
                unmaskable_reason = None
            else:
                maskable = False
                unmaskable_reason = "Missing or invalid transit duration required for mask geometry"
        else:
            maskable = False
            unmaskable_reason = "Incomplete transit ephemeris in catalog"

        if quarantined:
            quarantined_count += 1

        solution_id = make_solution_id(
            "toi",
            toi_num,
            period=period,
            epoch=epoch_btjd,
            duration=dur_h,
            depth=depth_ppm,
        )

        signals.append(
            KnownSignal(
                signal_id=signal_id,
                tic_id=tid_str,
                source_catalog=source_name,
                source_record_id=toi_num,
                planet_or_signal_name=f"TOI-{toi_num}",
                disposition=disp_cat,
                raw_disposition=raw_disp_str,
                solution_id=solution_id,
                snapshot_id=snapshot_id,
                discovery_method="Transit",
                period_days=period,
                period_err_days=period_err,
                period_err_pos=period_err_pos,
                period_err_neg=period_err_neg,
                epoch_bjd=epoch_bjd,
                epoch_btjd=epoch_btjd,
                epoch_err_days=epoch_err,
                epoch_err_pos=epoch_err_pos,
                epoch_err_neg=epoch_err_neg,
                original_time_format=orig_time_format,
                original_time_scale=orig_time_scale,
                epoch_zero_point=epoch_zero_point,
                duration_hours=dur_h,
                duration_days=dur_d,
                duration_err_pos=dur_err_pos,
                duration_err_neg=dur_err_neg,
                depth_ppm=depth_ppm,
                depth_err_pos=depth_err_pos,
                depth_err_neg=depth_err_neg,
                has_transit_ephemeris=has_ephem,
                maskable=maskable,
                unmaskable_reason=unmaskable_reason,
                quarantined=quarantined,
                quarantine_reason=quarantine_reason,
                aliases=[f"TOI-{toi_num}"],
                raw_fields=raw_fields,
                source_row_index=int(idx),
            )
        )

    valid_count = len(signals) - quarantined_count
    snapshot = SourceSnapshot(
        source_name=source_name,
        provider=provider,
        query_or_url=query_or_url,
        retrieval_utc=retrieval_utc,
        status="SUCCESS",
        row_count_raw=len(df),
        row_count_valid=valid_count,
        row_count_quarantined=quarantined_count,
        payload_sha256=raw_sha,
        payload_hash_kind=payload_hash_kind,
        raw_source_path_or_url=raw_source_path_or_url,
        original_retrieval_utc=original_retrieval_utc,
        snapshot_id=f"{source_name}@{retrieval_utc}",
    )
    return signals, snapshot


def ingest_nasa_confirmed_df(
    df: pd.DataFrame,
    source_name: str = "nasa_confirmed",
    provider: str = "NASA Exoplanet Archive",
    query_or_url: str = "pscomppars",
    retrieval_utc: Optional[str] = None,
    raw_payload_sha256: Optional[str] = None,
    payload_hash_kind: str = "dataframe_serialization",
    raw_source_path_or_url: Optional[str] = None,
    original_retrieval_utc: Optional[str] = None,
) -> tuple[list[KnownSignal], SourceSnapshot]:
    """
    Parses NASA Confirmed Planets (pscomppars) table rows into KnownSignal records.
    CRITICAL SCIENTIFIC INVARIANT:
      Does NOT exclude transit ephemerides merely because discoverymethod != 'Transit'.
      Planets discovered by Radial Velocity (e.g. HIP 29442 c and d) frequently have measured
      pl_tranmid and pl_trandur; these are fully preserved.
    Unit conversions:
      - pl_trandur is in hours.
      - pl_trandep is in percent; converted to ppm via pl_trandep * 10000.0.
    """
    retrieval_utc = retrieval_utc or datetime.now(timezone.utc).isoformat()
    raw_sha = raw_payload_sha256 or compute_payload_sha256(df)
    signals: list[KnownSignal] = []
    quarantined_count = 0

    col_map = {c.lower(): c for c in df.columns}
    tic_col = col_map.get("tic_id")
    plname_col = col_map.get("pl_name")
    hostname_col = col_map.get("hostname")
    method_col = col_map.get("discoverymethod")
    per_col = col_map.get("pl_orbper")
    per_err_col = col_map.get("pl_orbpererr1")
    per_err2_col = col_map.get("pl_orbpererr2")
    tmid_col = col_map.get("pl_tranmid")
    tmid_err_col = col_map.get("pl_tranmiderr1")
    tmid_err2_col = col_map.get("pl_tranmiderr2")
    dur_col = col_map.get("pl_trandur")
    dur_err_col = col_map.get("pl_trandurerr1")
    dur_err2_col = col_map.get("pl_trandurerr2")
    dep_col = col_map.get("pl_trandep")
    dep_err_col = col_map.get("pl_trandeperr1")
    dep_err2_col = col_map.get("pl_trandeperr2")

    if not tic_col or not plname_col:
        error_msg = f"Missing required columns in NASA Confirmed table: tic_id={tic_col}, pl_name={plname_col}"
        snapshot_id = f"{source_name}@attempt_{retrieval_utc}#FAILED"
        for idx, row in df.iterrows():
            raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            signals.append(
                KnownSignal(
                    signal_id=f"confirmed:missing_header_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=str(row.get(plname_col, f"row_{idx}") if plname_col else f"row_{idx}"),
                    planet_or_signal_name=f"Confirmed-row-{idx}",
                    disposition="CONFIRMED_PLANET",
                    raw_disposition="CONFIRMED",
                    solution_id=f"confirmed:missing_header_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=error_msg,
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
        snapshot = SourceSnapshot(
            source_name=source_name,
            provider=provider,
            query_or_url=query_or_url,
            retrieval_utc=retrieval_utc,
            status="FAILED",
            row_count_raw=len(df),
            row_count_valid=0,
            row_count_quarantined=len(df),
            payload_sha256=raw_sha,
            payload_hash_kind=payload_hash_kind,
            raw_source_path_or_url=raw_source_path_or_url,
            original_retrieval_utc=original_retrieval_utc,
            snapshot_id=snapshot_id,
            error_message=error_msg,
        )
        return signals, snapshot

    for idx, row in df.iterrows():
        raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        snapshot_id = f"{source_name}@{retrieval_utc}"
        try:
            tid_str = normalize_tic_id(row[tic_col])
        except Exception as exc:
            quarantined_count += 1
            signals.append(
                KnownSignal(
                    signal_id=f"confirmed:invalid_tic_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=str(row.get(plname_col, f"row_{idx}")),
                    planet_or_signal_name=str(row.get(plname_col, f"row_{idx}")),
                    disposition="CONFIRMED_PLANET",
                    raw_disposition="CONFIRMED",
                    solution_id=f"confirmed:invalid_tic_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=f"Invalid TIC ID: {exc}",
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
            continue

        pl_name = str(row[plname_col]).strip()
        signal_id = f"confirmed:{pl_name.replace(' ', '_')}"
        disc_method = str(row.get(method_col, "Unknown")).strip() if pd.notna(row.get(method_col)) else None

        period = None
        period_err = None
        period_err_pos = None
        period_err_neg = None
        epoch_bjd = None
        epoch_btjd = None
        epoch_err = None
        epoch_err_pos = None
        epoch_err_neg = None
        dur_h = None
        dur_d = None
        dur_err_pos = None
        dur_err_neg = None
        depth_ppm = None
        depth_err_pos = None
        depth_err_neg = None
        quarantined = False
        quarantine_reason = None

        if per_col and pd.notna(row.get(per_col)):
            try:
                period = float(row[per_col])
                if not math.isfinite(period) or period <= 0:
                    quarantined = True
                    quarantine_reason = f"Non-finite or non-positive orbital period: {period}"
            except Exception:
                quarantined = True
                quarantine_reason = f"Unparseable period: {row.get(per_col)}"

        if per_err_col and pd.notna(row.get(per_err_col)):
            try:
                period_err_pos = float(row[per_err_col])
                period_err = abs(period_err_pos)
            except Exception:
                pass
        if per_err2_col and pd.notna(row.get(per_err2_col)):
            try:
                period_err_neg = float(row[per_err2_col])
            except Exception:
                pass

        if tmid_col and pd.notna(row.get(tmid_col)):
            try:
                tmid = float(row[tmid_col])
                if not math.isfinite(tmid):
                    quarantined = True
                    quarantine_reason = f"Non-finite transit midpoint: {tmid}"
                else:
                    # In NASA Confirmed pscomppars table, pl_tranmid is in BJD_TDB (epoch zero 0.0)
                    epoch_bjd = tmid
                    epoch_btjd = tmid - BJD_BTJD_OFFSET
            except Exception:
                pass

        if tmid_err_col and pd.notna(row.get(tmid_err_col)):
            try:
                epoch_err_pos = float(row[tmid_err_col])
                epoch_err = abs(epoch_err_pos)
            except Exception:
                pass
        if tmid_err2_col and pd.notna(row.get(tmid_err2_col)):
            try:
                epoch_err_neg = float(row[tmid_err2_col])
            except Exception:
                pass

        if dur_col and pd.notna(row.get(dur_col)):
            try:
                dur_val = float(row[dur_col])
                if math.isfinite(dur_val) and dur_val > 0:
                    dur_h = dur_val
                    dur_d = hours_to_days(dur_h)
            except Exception:
                pass
        if dur_err_col and pd.notna(row.get(dur_err_col)):
            try:
                dur_err_pos = float(row[dur_err_col])
            except Exception:
                pass
        if dur_err2_col and pd.notna(row.get(dur_err2_col)):
            try:
                dur_err_neg = float(row[dur_err2_col])
            except Exception:
                pass

        if dep_col and pd.notna(row.get(dep_col)):
            try:
                # pl_trandep in pscomppars is percent (e.g. 0.11791% -> 1179.1 ppm)
                depth_pct = float(row[dep_col])
                if math.isfinite(depth_pct) and depth_pct > 0:
                    depth_ppm = depth_pct * 10000.0
            except Exception:
                pass
        if dep_err_col and pd.notna(row.get(dep_err_col)):
            try:
                depth_err_pos = float(row[dep_err_col]) * 10000.0
            except Exception:
                pass
        if dep_err2_col and pd.notna(row.get(dep_err2_col)):
            try:
                depth_err_neg = float(row[dep_err2_col]) * 10000.0
            except Exception:
                pass

        has_ephem = bool(period is not None and epoch_btjd is not None and not quarantined)
        if has_ephem:
            if dur_h is not None and dur_h > 0:
                maskable = True
                unmaskable_reason = None
            else:
                maskable = False
                unmaskable_reason = "Missing or invalid transit duration required for mask geometry"
        else:
            maskable = False
            unmaskable_reason = "Confirmed planet without measured transit midpoint ephemeris"

        if quarantined:
            quarantined_count += 1

        solution_id = make_solution_id(
            "confirmed",
            pl_name,
            period=period,
            epoch=epoch_btjd,
            duration=dur_h,
            depth=depth_ppm,
        )

        signals.append(
            KnownSignal(
                signal_id=signal_id,
                tic_id=tid_str,
                source_catalog=source_name,
                source_record_id=pl_name,
                planet_or_signal_name=pl_name,
                disposition="CONFIRMED_PLANET",
                raw_disposition="CONFIRMED",
                solution_id=solution_id,
                snapshot_id=snapshot_id,
                discovery_method=disc_method,
                period_days=period,
                period_err_days=period_err,
                period_err_pos=period_err_pos,
                period_err_neg=period_err_neg,
                epoch_bjd=epoch_bjd,
                epoch_btjd=epoch_btjd,
                epoch_err_days=epoch_err,
                epoch_err_pos=epoch_err_pos,
                epoch_err_neg=epoch_err_neg,
                original_time_format="BJD",
                original_time_scale="TDB",
                epoch_zero_point=0.0,
                duration_hours=dur_h,
                duration_days=dur_d,
                duration_err_pos=dur_err_pos,
                duration_err_neg=dur_err_neg,
                depth_ppm=depth_ppm,
                depth_err_pos=depth_err_pos,
                depth_err_neg=depth_err_neg,
                has_transit_ephemeris=has_ephem,
                maskable=maskable,
                unmaskable_reason=unmaskable_reason,
                quarantined=quarantined,
                quarantine_reason=quarantine_reason,
                aliases=[pl_name],
                raw_fields=raw_fields,
                source_row_index=int(idx),
            )
        )

    valid_count = len(signals) - quarantined_count
    snapshot = SourceSnapshot(
        source_name=source_name,
        provider=provider,
        query_or_url=query_or_url,
        retrieval_utc=retrieval_utc,
        status="SUCCESS",
        row_count_raw=len(df),
        row_count_valid=valid_count,
        row_count_quarantined=quarantined_count,
        payload_sha256=raw_sha,
        payload_hash_kind=payload_hash_kind,
        raw_source_path_or_url=raw_source_path_or_url,
        original_retrieval_utc=original_retrieval_utc,
        snapshot_id=f"{source_name}@{retrieval_utc}",
    )
    return signals, snapshot


def ingest_exofop_ctoi_df(
    df: pd.DataFrame,
    source_name: str = "exofop_ctoi",
    provider: str = "ExoFOP-TESS",
    query_or_url: str = "download_ctoi.php",
    retrieval_utc: Optional[str] = None,
    raw_payload_sha256: Optional[str] = None,
    payload_hash_kind: str = "dataframe_serialization",
    raw_source_path_or_url: Optional[str] = None,
    original_retrieval_utc: Optional[str] = None,
) -> tuple[list[KnownSignal], SourceSnapshot]:
    """
    Parses ExoFOP Community TOI (CTOI) table rows into KnownSignal records.
    Robust column matching for ExoFOP headers.
    """
    retrieval_utc = retrieval_utc or datetime.now(timezone.utc).isoformat()
    raw_sha = raw_payload_sha256 or compute_payload_sha256(df)
    signals: list[KnownSignal] = []
    quarantined_count = 0

    col_map = {c.strip().lower(): c for c in df.columns}
    tic_col = col_map.get("tic id") or col_map.get("tic") or next((c for k, c in col_map.items() if "tic" in k), None)
    ctoi_col = col_map.get("ctoi") or next((c for k, c in col_map.items() if "ctoi" in k), None)
    per_col = col_map.get("period (days)") or next((c for k, c in col_map.items() if "period" in k), None)
    per_err_col = col_map.get("period (days) error")
    epoch_col = col_map.get("transit epoch (bjd)") or next((c for k, c in col_map.items() if "epoch" in k), None)
    dur_col = col_map.get("duration (hrs)") or next((c for k, c in col_map.items() if "duration" in k), None)
    dep_col = col_map.get("depth ppm") or next((c for k, c in col_map.items() if "depth ppm" in k), None)
    disp_col = col_map.get("tfopwg disposition") or col_map.get("user disposition")

    if not tic_col:
        error_msg = f"Missing TIC column in ExoFOP CTOI table: available={list(df.columns[:5])}"
        snapshot_id = f"{source_name}@attempt_{retrieval_utc}#FAILED"
        for idx, row in df.iterrows():
            raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            signals.append(
                KnownSignal(
                    signal_id=f"ctoi:missing_header_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=str(row.get(ctoi_col, f"row_{idx}") if ctoi_col else f"row_{idx}"),
                    planet_or_signal_name=f"CTOI-row-{idx}",
                    disposition="UNKNOWN",
                    raw_disposition="",
                    solution_id=f"ctoi:missing_header_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=error_msg,
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
        snapshot = SourceSnapshot(
            source_name=source_name,
            provider=provider,
            query_or_url=query_or_url,
            retrieval_utc=retrieval_utc,
            status="FAILED",
            row_count_raw=len(df),
            row_count_valid=0,
            row_count_quarantined=len(df),
            payload_sha256=raw_sha,
            payload_hash_kind=payload_hash_kind,
            raw_source_path_or_url=raw_source_path_or_url,
            original_retrieval_utc=original_retrieval_utc,
            snapshot_id=snapshot_id,
            error_message=error_msg,
        )
        return signals, snapshot

    for idx, row in df.iterrows():
        raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        snapshot_id = f"{source_name}@{retrieval_utc}"
        try:
            tid_str = normalize_tic_id(row[tic_col])
        except Exception as exc:
            quarantined_count += 1
            signals.append(
                KnownSignal(
                    signal_id=f"ctoi:invalid_tic_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=str(row.get(ctoi_col, f"row_{idx}")),
                    planet_or_signal_name=f"CTOI-{row.get(ctoi_col, idx)}",
                    disposition="UNKNOWN",
                    raw_disposition="",
                    solution_id=f"ctoi:invalid_tic_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=f"Invalid TIC ID: {exc}",
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
            continue

        ctoi_val = str(row[ctoi_col]).strip() if ctoi_col and pd.notna(row.get(ctoi_col)) else ""
        rec_id = ctoi_val if ctoi_val else f"{tid_str}_row_{idx}"
        signal_id = f"ctoi:{rec_id}"
        raw_disp = row.get(disp_col)
        disp_cat, raw_disp_str = normalize_disposition("exofop_ctoi", raw_disp)

        period = None
        period_err = None
        epoch_bjd = None
        epoch_btjd = None
        dur_h = None
        dur_d = None
        depth_ppm = None
        quarantined = False
        quarantine_reason = None

        if per_col and pd.notna(row.get(per_col)):
            try:
                period = float(row[per_col])
                if not math.isfinite(period) or period <= 0:
                    quarantined = True
                    quarantine_reason = f"Non-finite or non-positive orbital period: {period}"
            except Exception:
                quarantined = True
                quarantine_reason = f"Unparseable period: {row.get(per_col)}"

        if per_err_col and pd.notna(row.get(per_err_col)):
            try:
                period_err = abs(float(row[per_err_col]))
            except Exception:
                pass

        time_format = "UNKNOWN"
        time_scale = "UNKNOWN"
        zero_pt = None
        if epoch_col and pd.notna(row.get(epoch_col)):
            try:
                ep = float(row[epoch_col])
                if not math.isfinite(ep):
                    quarantined = True
                    quarantine_reason = f"Non-finite transit epoch: {ep}"
                else:
                    col_name_lower = str(epoch_col).lower()
                    if "btjd" in col_name_lower:
                        epoch_btjd = ep
                        epoch_bjd = ep + BJD_BTJD_OFFSET
                        time_format = "BTJD"
                        time_scale = "TDB"
                        zero_pt = BJD_BTJD_OFFSET
                    elif "bjd" in col_name_lower or ep > 2400000.0:
                        epoch_bjd = ep
                        epoch_btjd = ep - BJD_BTJD_OFFSET
                        time_format = "BJD"
                        time_scale = "TDB"
                        zero_pt = 0.0
                    else:
                        epoch_btjd = ep
                        epoch_bjd = ep + BJD_BTJD_OFFSET
                        time_format = "BTJD"
                        time_scale = "UNKNOWN"
            except Exception:
                pass

        if dur_col and pd.notna(row.get(dur_col)):
            try:
                dur_val = float(row[dur_col])
                if math.isfinite(dur_val) and dur_val > 0:
                    dur_h = dur_val
                    dur_d = hours_to_days(dur_h)
            except Exception:
                pass

        if dep_col and pd.notna(row.get(dep_col)):
            try:
                dep_val = float(row[dep_col])
                if math.isfinite(dep_val) and dep_val > 0:
                    depth_ppm = dep_val
            except Exception:
                pass

        has_ephem = bool(period is not None and epoch_btjd is not None and not quarantined)
        if has_ephem:
            if dur_h is not None and dur_h > 0:
                maskable = True
                unmaskable_reason = None
            else:
                maskable = False
                unmaskable_reason = "Missing or invalid transit duration required for mask geometry"
        else:
            maskable = False
            unmaskable_reason = "CTOI entry missing complete orbital ephemeris"

        if quarantined:
            quarantined_count += 1

        solution_id = make_solution_id(
            "ctoi",
            rec_id,
            period=period,
            epoch=epoch_btjd,
            duration=dur_h,
            depth=depth_ppm,
        )

        aliases = [f"CTOI-{ctoi_val}"] if ctoi_val else []

        signals.append(
            KnownSignal(
                signal_id=signal_id,
                tic_id=tid_str,
                source_catalog=source_name,
                source_record_id=rec_id,
                planet_or_signal_name=f"CTOI-{rec_id}",
                disposition=disp_cat,
                raw_disposition=raw_disp_str,
                solution_id=solution_id,
                snapshot_id=snapshot_id,
                discovery_method="Transit",
                period_days=period,
                period_err_days=period_err,
                epoch_bjd=epoch_bjd,
                epoch_btjd=epoch_btjd,
                original_time_format=time_format,
                original_time_scale=time_scale,
                epoch_zero_point=zero_pt,
                duration_hours=dur_h,
                duration_days=dur_d,
                depth_ppm=depth_ppm,
                has_transit_ephemeris=has_ephem,
                maskable=maskable,
                unmaskable_reason=unmaskable_reason,
                quarantined=quarantined,
                quarantine_reason=quarantine_reason,
                aliases=aliases,
                raw_fields=raw_fields,
                source_row_index=int(idx),
            )
        )

    valid_count = len(signals) - quarantined_count
    snapshot = SourceSnapshot(
        source_name=source_name,
        provider=provider,
        query_or_url=query_or_url,
        retrieval_utc=retrieval_utc,
        status="SUCCESS",
        row_count_raw=len(df),
        row_count_valid=valid_count,
        row_count_quarantined=quarantined_count,
        payload_sha256=raw_sha,
        payload_hash_kind=payload_hash_kind,
        raw_source_path_or_url=raw_source_path_or_url,
        original_retrieval_utc=original_retrieval_utc,
        snapshot_id=f"{source_name}@{retrieval_utc}",
    )
    return signals, snapshot


def ingest_vizier_eb_df(
    df: pd.DataFrame,
    source_name: str = "vizier_tess_eb",
    provider: str = "VizieR TAP (Prsa et al. 2022)",
    query_or_url: str = 'J/ApJS/258/16/tess-ebs',
    retrieval_utc: Optional[str] = None,
    raw_payload_sha256: Optional[str] = None,
    payload_hash_kind: str = "dataframe_serialization",
    raw_source_path_or_url: Optional[str] = None,
    original_retrieval_utc: Optional[str] = None,
) -> tuple[list[KnownSignal], SourceSnapshot]:
    """
    Parses VizieR TESS Eclipsing Binary table rows into KnownSignal records.
    Supports tables with TIC only (standard search query) or with period/epoch if available.
    """
    retrieval_utc = retrieval_utc or datetime.now(timezone.utc).isoformat()
    raw_sha = raw_payload_sha256 or compute_payload_sha256(df)
    signals: list[KnownSignal] = []
    quarantined_count = 0

    col_map = {c.strip().lower(): c for c in df.columns}
    tic_col = col_map.get("tic") or col_map.get("tic_id") or next((c for k, c in col_map.items() if "tic" in k), None)
    per_col = col_map.get("period") or col_map.get("b_period") or col_map.get("per")
    bjd_col = col_map.get("bjd0") or col_map.get("t0") or col_map.get("epoch")

    if not tic_col:
        error_msg = f"Missing TIC column in VizieR EB table: columns={list(df.columns)}"
        snapshot_id = f"{source_name}@attempt_{retrieval_utc}#FAILED"
        for idx, row in df.iterrows():
            raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            signals.append(
                KnownSignal(
                    signal_id=f"eb:missing_header_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=f"row_{idx}",
                    planet_or_signal_name=f"EB-row-{idx}",
                    disposition="ECLIPSING_BINARY",
                    raw_disposition="EB",
                    solution_id=f"eb:missing_header_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=error_msg,
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
        snapshot = SourceSnapshot(
            source_name=source_name,
            provider=provider,
            query_or_url=query_or_url,
            retrieval_utc=retrieval_utc,
            status="FAILED",
            row_count_raw=len(df),
            row_count_valid=0,
            row_count_quarantined=len(df),
            payload_sha256=raw_sha,
            payload_hash_kind=payload_hash_kind,
            raw_source_path_or_url=raw_source_path_or_url,
            original_retrieval_utc=original_retrieval_utc,
            snapshot_id=snapshot_id,
            error_message=error_msg,
        )
        return signals, snapshot

    for idx, row in df.iterrows():
        raw_fields = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        snapshot_id = f"{source_name}@{retrieval_utc}"
        try:
            tid_str = normalize_tic_id(row[tic_col])
        except Exception as exc:
            quarantined_count += 1
            signals.append(
                KnownSignal(
                    signal_id=f"eb:invalid_tic_row_{idx}",
                    tic_id="quarantine",
                    source_catalog=source_name,
                    source_record_id=f"row_{idx}",
                    planet_or_signal_name=f"EB-row-{idx}",
                    disposition="ECLIPSING_BINARY",
                    raw_disposition="EB",
                    solution_id=f"eb:invalid_tic_row_{idx}#quarantine",
                    snapshot_id=snapshot_id,
                    quarantined=True,
                    quarantine_reason=f"Invalid TIC ID: {exc}",
                    raw_fields=raw_fields,
                    source_row_index=int(idx),
                )
            )
            continue

        period = None
        epoch_bjd = None
        epoch_btjd = None
        quarantined = False
        quarantine_reason = None

        if per_col and pd.notna(row.get(per_col)):
            try:
                period_val = float(row[per_col])
                if math.isfinite(period_val) and period_val > 0:
                    period = period_val
                else:
                    quarantined = True
                    quarantine_reason = f"Non-finite or non-positive period: {period_val}"
            except Exception:
                pass
        if bjd_col and pd.notna(row.get(bjd_col)):
            try:
                bjd_val = float(row[bjd_col])
                if math.isfinite(bjd_val):
                    epoch_bjd = bjd_val
                    epoch_btjd = convert_bjd_to_btjd(bjd_val)
            except Exception:
                pass

        has_ephem = bool(period is not None and epoch_btjd is not None and not quarantined)
        # EB without explicit duration cannot be masked
        maskable = False
        unmaskable_reason = "VizieR EB catalog provides host TIC without full transit mask ephemeris"

        if quarantined:
            quarantined_count += 1

        solution_id = make_solution_id(
            "eb",
            tid_str,
            period=period,
            epoch=epoch_btjd,
        )

        signals.append(
            KnownSignal(
                signal_id=f"eb:{tid_str}",
                tic_id=tid_str,
                source_catalog=source_name,
                source_record_id=tid_str,
                planet_or_signal_name=f"EB-TIC-{tid_str}",
                disposition="ECLIPSING_BINARY",
                raw_disposition="EB",
                solution_id=solution_id,
                snapshot_id=snapshot_id,
                discovery_method="Transit",
                period_days=period,
                epoch_bjd=epoch_bjd,
                epoch_btjd=epoch_btjd,
                has_transit_ephemeris=has_ephem,
                maskable=maskable,
                unmaskable_reason=unmaskable_reason,
                quarantined=quarantined,
                quarantine_reason=quarantine_reason,
                aliases=[],
                raw_fields=raw_fields,
                source_row_index=int(idx),
            )
        )

    valid_count = len(signals) - quarantined_count
    snapshot = SourceSnapshot(
        source_name=source_name,
        provider=provider,
        query_or_url=query_or_url,
        retrieval_utc=retrieval_utc,
        status="SUCCESS",
        row_count_raw=len(df),
        row_count_valid=valid_count,
        row_count_quarantined=quarantined_count,
        payload_sha256=raw_sha,
        payload_hash_kind=payload_hash_kind,
        raw_source_path_or_url=raw_source_path_or_url,
        original_retrieval_utc=original_retrieval_utc,
        snapshot_id=f"{source_name}@{retrieval_utc}",
    )
    return signals, snapshot


# ---------------------------------------------------------------------------
# Non-Destructive Diagnostic Association & Cycle-Wrap Invariance
# ---------------------------------------------------------------------------

def find_nearest_harmonic_ratio(
    period_a: float,
    period_b: float,
    max_p: int = 7,
    max_q: int = 7,
) -> tuple[int, int, float, float]:
    """
    Finds the integers (p, q) <= 7 that best approximate period_a / period_b.
    Returns: (p, q, harmonic_ratio, relative_difference).
    """
    if period_a <= 0 or period_b <= 0:
        raise ValueError(f"Periods must be strictly positive: {period_a}, {period_b}")
    actual_ratio = period_a / period_b
    ratios = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            if math.gcd(p, q) == 1:
                ratios.append((p, q))
    ratios.sort(key=lambda item: abs(actual_ratio - (item[0] / item[1])))
    best_p, best_q = 1, 1
    best_diff = float("inf")
    for p, q in ratios:
        target = p / q
        diff = abs(actual_ratio - target) / target
        if diff < best_diff:
            best_diff = diff
            best_p, best_q = p, q
    return best_p, best_q, actual_ratio, best_diff


def calculate_cyclic_timing_residual(
    cand_p: float,
    cand_t0: float,
    cat_p: float,
    cat_t0: float,
    pivot_time: Optional[float] = None,
) -> tuple[float, float, float]:
    """
    Calculates physical timing difference between candidate and catalog solutions,
    strictly invariant to integer-cycle epoch shifts in either candidate or catalog.
    Delegates to canonical implementation in ephemeris_contract.
    """
    return contract_calculate_cyclic_timing_residual(
        cand_p, cand_t0, cat_p, cat_t0, pivot_time=pivot_time
    )


@dataclass
class SignalAssociationMetric:
    candidate_period: float
    candidate_t0: float
    catalog_signal_id: str
    catalog_source: str
    catalog_name: str
    catalog_disposition: str
    candidate_observation_id: Optional[str] = None
    candidate_time_format: str = "BTJD"
    candidate_time_scale: str = "TDB"
    pivot_time: Optional[float] = None
    catalog_solution_id: Optional[str] = None
    catalog_period: Optional[float] = None
    catalog_epoch_btjd: Optional[float] = None
    catalog_duration_hours: Optional[float] = None
    catalog_depth_ppm: Optional[float] = None
    discovery_method: Optional[str] = None
    delta_period_days: Optional[float] = None
    relative_period_diff: Optional[float] = None
    period_ratio: Optional[float] = None
    nearest_harmonic_p: Optional[int] = None
    nearest_harmonic_q: Optional[int] = None
    harmonic_ratio: Optional[float] = None
    harmonic_relative_diff: Optional[float] = None
    epoch_difference_days: Optional[float] = None
    epoch_difference_minutes: Optional[float] = None
    phase_offset: Optional[float] = None
    duration_ratio: Optional[float] = None
    depth_ratio: Optional[float] = None
    status: str = "MEASURED"  # 'MEASURED', 'TIMING_UNAVAILABLE', 'UNAVAILABLE', 'QUARANTINED'
    status_reason: Optional[str] = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_period": self.candidate_period,
            "candidate_t0": self.candidate_t0,
            "catalog_signal_id": self.catalog_signal_id,
            "catalog_source": self.catalog_source,
            "catalog_name": self.catalog_name,
            "catalog_disposition": self.catalog_disposition,
            "candidate_observation_id": self.candidate_observation_id,
            "candidate_time_format": self.candidate_time_format,
            "candidate_time_scale": self.candidate_time_scale,
            "pivot_time": self.pivot_time,
            "catalog_solution_id": self.catalog_solution_id,
            "catalog_period": self.catalog_period,
            "catalog_epoch_btjd": self.catalog_epoch_btjd,
            "catalog_duration_hours": self.catalog_duration_hours,
            "catalog_depth_ppm": self.catalog_depth_ppm,
            "discovery_method": self.discovery_method,
            "delta_period_days": self.delta_period_days,
            "relative_period_diff": self.relative_period_diff,
            "period_ratio": self.period_ratio,
            "nearest_harmonic_p": self.nearest_harmonic_p,
            "nearest_harmonic_q": self.nearest_harmonic_q,
            "harmonic_ratio": self.harmonic_ratio,
            "harmonic_relative_diff": self.harmonic_relative_diff,
            "epoch_difference_days": self.epoch_difference_days,
            "epoch_difference_minutes": self.epoch_difference_minutes,
            "phase_offset": self.phase_offset,
            "duration_ratio": self.duration_ratio,
            "depth_ratio": self.depth_ratio,
            "status": self.status,
            "status_reason": self.status_reason,
            "flags": list(self.flags),
        }


def calculate_association_metrics(
    candidate_p: float,
    candidate_t0: float,
    signal: KnownSignal,
    candidate_observation_id: Optional[str] = None,
    candidate_time_format: str = "BTJD",
    candidate_time_scale: str = "TDB",
    pivot_time: Optional[float] = None,
    candidate_duration_hours: Optional[float] = None,
    candidate_depth_ppm: Optional[float] = None,
) -> SignalAssociationMetric:
    """
    Computes rigorous non-destructive diagnostic association metrics between candidate
    and a specific catalog signal.
    Does NOT assert automatic planet vetoes or match thresholds.
    If pivot_time is None or time scale is unknown, timing residual is marked UNAVAILABLE.
    """
    if signal.quarantined:
        return SignalAssociationMetric(
            candidate_period=candidate_p,
            candidate_t0=candidate_t0,
            catalog_signal_id=signal.signal_id,
            catalog_source=signal.source_catalog,
            catalog_name=signal.planet_or_signal_name,
            catalog_disposition=signal.disposition,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=pivot_time,
            catalog_solution_id=signal.solution_id,
            catalog_period=signal.period_days,
            catalog_epoch_btjd=signal.epoch_btjd,
            catalog_duration_hours=signal.duration_hours,
            catalog_depth_ppm=signal.depth_ppm,
            discovery_method=signal.discovery_method,
            status="QUARANTINED",
            status_reason=f"Catalog signal is quarantined: {signal.quarantine_reason}",
            flags=["CATALOG_SIGNAL_QUARANTINED"],
        )

    if signal.period_days is None or signal.period_days <= 0 or not math.isfinite(signal.period_days):
        return SignalAssociationMetric(
            candidate_period=candidate_p,
            candidate_t0=candidate_t0,
            catalog_signal_id=signal.signal_id,
            catalog_source=signal.source_catalog,
            catalog_name=signal.planet_or_signal_name,
            catalog_disposition=signal.disposition,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=pivot_time,
            catalog_solution_id=signal.solution_id,
            catalog_period=signal.period_days,
            catalog_epoch_btjd=signal.epoch_btjd,
            catalog_duration_hours=signal.duration_hours,
            catalog_depth_ppm=signal.depth_ppm,
            discovery_method=signal.discovery_method,
            status="UNAVAILABLE",
            status_reason="Missing or invalid period in catalog",
            flags=["MISSING_CATALOG_PERIOD"],
        )

    dp = candidate_p - signal.period_days
    rel_dp = abs(dp) / signal.period_days
    p_rat = candidate_p / signal.period_days

    p, q, harm_ratio, harm_diff = find_nearest_harmonic_ratio(candidate_p, signal.period_days)

    dur_ratio = None
    if candidate_duration_hours is not None and signal.duration_hours is not None and signal.duration_hours > 0:
        dur_ratio = candidate_duration_hours / signal.duration_hours

    dep_ratio = None
    if candidate_depth_ppm is not None and signal.depth_ppm is not None and signal.depth_ppm > 0:
        dep_ratio = candidate_depth_ppm / signal.depth_ppm

    flags: list[str] = []
    if signal.disposition == "FALSE_POSITIVE":
        flags.append("SOURCE_DISPOSITION_FALSE_POSITIVE")
    if signal.disposition == "CONFIRMED_PLANET":
        flags.append("SOURCE_DISPOSITION_CONFIRMED_PLANET")

    if signal.epoch_btjd is None or not math.isfinite(signal.epoch_btjd):
        return SignalAssociationMetric(
            candidate_period=candidate_p,
            candidate_t0=candidate_t0,
            catalog_signal_id=signal.signal_id,
            catalog_source=signal.source_catalog,
            catalog_name=signal.planet_or_signal_name,
            catalog_disposition=signal.disposition,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=pivot_time,
            catalog_solution_id=signal.solution_id,
            catalog_period=signal.period_days,
            catalog_epoch_btjd=signal.epoch_btjd,
            catalog_duration_hours=signal.duration_hours,
            catalog_depth_ppm=signal.depth_ppm,
            discovery_method=signal.discovery_method,
            delta_period_days=dp,
            relative_period_diff=rel_dp,
            period_ratio=p_rat,
            nearest_harmonic_p=p,
            nearest_harmonic_q=q,
            harmonic_ratio=harm_ratio,
            harmonic_relative_diff=harm_diff,
            duration_ratio=dur_ratio,
            depth_ratio=dep_ratio,
            status="UNAVAILABLE",
            status_reason="Missing transit epoch in catalog",
            flags=flags + ["MISSING_CATALOG_EPOCH"],
        )

    if pivot_time is None:
        return SignalAssociationMetric(
            candidate_period=candidate_p,
            candidate_t0=candidate_t0,
            catalog_signal_id=signal.signal_id,
            catalog_source=signal.source_catalog,
            catalog_name=signal.planet_or_signal_name,
            catalog_disposition=signal.disposition,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=None,
            catalog_solution_id=signal.solution_id,
            catalog_period=signal.period_days,
            catalog_epoch_btjd=signal.epoch_btjd,
            catalog_duration_hours=signal.duration_hours,
            catalog_depth_ppm=signal.depth_ppm,
            discovery_method=signal.discovery_method,
            delta_period_days=dp,
            relative_period_diff=rel_dp,
            period_ratio=p_rat,
            nearest_harmonic_p=p,
            nearest_harmonic_q=q,
            harmonic_ratio=harm_ratio,
            harmonic_relative_diff=harm_diff,
            epoch_difference_days=None,
            epoch_difference_minutes=None,
            phase_offset=None,
            duration_ratio=dur_ratio,
            depth_ratio=dep_ratio,
            status="TIMING_UNAVAILABLE",
            status_reason="Common pivot time not provided; timing residual requires common reference anchor",
            flags=flags,
        )

    # Validate candidate time format
    if candidate_time_format is None or candidate_time_format not in ("BTJD", "BJD"):
        return SignalAssociationMetric(
            candidate_period=candidate_p,
            candidate_t0=candidate_t0,
            catalog_signal_id=signal.signal_id,
            catalog_source=signal.source_catalog,
            catalog_name=signal.planet_or_signal_name,
            catalog_disposition=signal.disposition,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=pivot_time,
            catalog_solution_id=signal.solution_id,
            catalog_period=signal.period_days,
            catalog_epoch_btjd=signal.epoch_btjd,
            catalog_duration_hours=signal.duration_hours,
            catalog_depth_ppm=signal.depth_ppm,
            discovery_method=signal.discovery_method,
            delta_period_days=dp,
            relative_period_diff=rel_dp,
            period_ratio=p_rat,
            nearest_harmonic_p=p,
            nearest_harmonic_q=q,
            harmonic_ratio=harm_ratio,
            harmonic_relative_diff=harm_diff,
            epoch_difference_days=None,
            epoch_difference_minutes=None,
            phase_offset=None,
            duration_ratio=dur_ratio,
            depth_ratio=dep_ratio,
            status="TIMING_UNAVAILABLE",
            status_reason=f"Unknown or unsupported candidate time format: {candidate_time_format}",
            flags=flags,
        )

    # Validate time scale compatibility
    if (
        candidate_time_scale is None
        or candidate_time_scale == "UNKNOWN"
        or signal.original_time_scale is None
        or signal.original_time_scale == "UNKNOWN"
        or candidate_time_scale != signal.original_time_scale
    ):
        return SignalAssociationMetric(
            candidate_period=candidate_p,
            candidate_t0=candidate_t0,
            catalog_signal_id=signal.signal_id,
            catalog_source=signal.source_catalog,
            catalog_name=signal.planet_or_signal_name,
            catalog_disposition=signal.disposition,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=pivot_time,
            catalog_solution_id=signal.solution_id,
            catalog_period=signal.period_days,
            catalog_epoch_btjd=signal.epoch_btjd,
            catalog_duration_hours=signal.duration_hours,
            catalog_depth_ppm=signal.depth_ppm,
            discovery_method=signal.discovery_method,
            delta_period_days=dp,
            relative_period_diff=rel_dp,
            period_ratio=p_rat,
            nearest_harmonic_p=p,
            nearest_harmonic_q=q,
            harmonic_ratio=harm_ratio,
            harmonic_relative_diff=harm_diff,
            epoch_difference_days=None,
            epoch_difference_minutes=None,
            phase_offset=None,
            duration_ratio=dur_ratio,
            depth_ratio=dep_ratio,
            status="TIMING_UNAVAILABLE",
            status_reason=f"Unknown, missing, or incompatible time scale: candidate={candidate_time_scale}, catalog={signal.original_time_scale}",
            flags=flags,
        )

    # Normalize candidate epoch and pivot time to common BTJD anchor
    norm_cand_t0 = candidate_t0 - BJD_BTJD_OFFSET if candidate_time_format == "BJD" else candidate_t0
    norm_pivot = pivot_time - BJD_BTJD_OFFSET if pivot_time > 2400000.0 else pivot_time
    norm_cat_t0 = signal.epoch_btjd

    dt_days, dt_min, phase = calculate_cyclic_timing_residual(
        candidate_p, norm_cand_t0, signal.period_days, norm_cat_t0, pivot_time=norm_pivot
    )

    return SignalAssociationMetric(
        candidate_period=candidate_p,
        candidate_t0=candidate_t0,
        catalog_signal_id=signal.signal_id,
        catalog_source=signal.source_catalog,
        catalog_name=signal.planet_or_signal_name,
        catalog_disposition=signal.disposition,
        candidate_observation_id=candidate_observation_id,
        candidate_time_format=candidate_time_format,
        candidate_time_scale=candidate_time_scale,
        pivot_time=pivot_time,
        catalog_solution_id=signal.solution_id,
        catalog_period=signal.period_days,
        catalog_epoch_btjd=signal.epoch_btjd,
        catalog_duration_hours=signal.duration_hours,
        catalog_depth_ppm=signal.depth_ppm,
        discovery_method=signal.discovery_method,
        delta_period_days=dp,
        relative_period_diff=rel_dp,
        period_ratio=p_rat,
        nearest_harmonic_p=p,
        nearest_harmonic_q=q,
        harmonic_ratio=harm_ratio,
        harmonic_relative_diff=harm_diff,
        epoch_difference_days=dt_days,
        epoch_difference_minutes=dt_min,
        phase_offset=phase,
        duration_ratio=dur_ratio,
        depth_ratio=dep_ratio,
        status="MEASURED",
        status_reason=None,
        flags=flags,
    )


def associate_candidate_with_known_signals(
    candidate_p: float,
    candidate_t0: float,
    host_context: Optional[HostCatalogContext],
    candidate_observation_id: Optional[str] = None,
    candidate_time_format: str = "BTJD",
    candidate_time_scale: str = "TDB",
    pivot_time: Optional[float] = None,
    candidate_duration_hours: Optional[float] = None,
    candidate_depth_ppm: Optional[float] = None,
) -> list[SignalAssociationMetric]:
    """
    Evaluates candidate parameters against all signals associated with host star.
    Returns per-signal association metrics without auto-classifying novelty or vetoing.
    """
    if host_context is None or not host_context.signals:
        return []

    metrics: list[SignalAssociationMetric] = []
    for signal in host_context.signals:
        metric = calculate_association_metrics(
            candidate_p=candidate_p,
            candidate_t0=candidate_t0,
            signal=signal,
            candidate_observation_id=candidate_observation_id,
            candidate_time_format=candidate_time_format,
            candidate_time_scale=candidate_time_scale,
            pivot_time=pivot_time,
            candidate_duration_hours=candidate_duration_hours,
            candidate_depth_ppm=candidate_depth_ppm,
        )
        metrics.append(metric)
    return metrics


# ---------------------------------------------------------------------------
# Hypothetical / Non-Active Masking Policy Helper
# ---------------------------------------------------------------------------

def build_diagnostic_transit_mask(
    time: np.ndarray,
    period_days: float,
    epoch_btjd: float,
    duration_days: float,
    width_factor: float = 1.0,
    label: str = "hypothetical_non_active",
) -> np.ndarray:
    """
    Diagnostic helper for computing an in-transit boolean mask array.

    WARNING / SCIENTIFIC INTEGRITY CONTRACT:
      This function is strictly for diagnostic calculations and offline unit tests.
      It is explicitly LABELED HYPOTHETICAL/NON-ACTIVE. It does NOT choose production
      masking policy or width defaults, and MUST NOT be wired into active detection,
      detrending, queue filtering, or candidate vetoing without explicit scientific
      authorization and Argon consultation (ALD-007B).

    Parameters:
      time: 1D numpy array of observation times (BTJD).
      period_days: orbital period in days (must be > 0).
      epoch_btjd: transit midpoint in BTJD.
      duration_days: transit duration in days (must be > 0).
      width_factor: diagnostic scaling factor applied to duration (default 1.0).
      label: explicit contract label identifying non-active status.

    Returns:
      boolean numpy array of same shape as time, True where in-transit.
    """
    if label != "hypothetical_non_active":
        logger.warning("build_diagnostic_transit_mask called with non-standard label: %s", label)
    if period_days <= 0 or duration_days <= 0:
        raise ValueError("period_days and duration_days must be strictly positive")
    if time is None or len(time) == 0:
        return np.array([], dtype=bool)

    t_arr = np.asarray(time, dtype=float)
    half_width = 0.5 * duration_days * width_factor
    cycles = np.round((t_arr - epoch_btjd) / period_days)
    t_nearest = epoch_btjd + cycles * period_days
    time_diff = np.abs(t_arr - t_nearest)
    return time_diff <= half_width
