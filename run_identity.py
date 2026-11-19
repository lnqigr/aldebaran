"""
Shared run identity, frozen environment, MAST product provenance, and
interruption-safe bound resume helpers (S3.6: OPS-13, OPS-14, OPS-33).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

RUN_IDENTITY_SCHEMA_VERSION = "run_identity.v1"

_REQUIRED_RECORD_KEYS = frozenset({
    "schema_version",
    "run_id",
    "runner",
    "status",
    "identity",
    "environment",
    "mast_products",
    "interruptions",
    "resumes",
})
_REQUIRED_IDENTITY_KEYS = frozenset({"code_sha256", "config_sha256", "input_sha256"})

_ENV_CACHE: Dict[str, Dict[str, Any]] = {}


class RunIdentityError(RuntimeError):
    """Raised when a run record is partial/corrupt or its identity/MAST hashes do not match."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    """Convert objects to strict RFC-8259 JSON-compatible Python types."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    return str(obj)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(
        _json_safe(obj),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Atomically write bytes via a same-directory temporary file, fsync, and os.replace."""
    target = os.path.abspath(str(path))
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(
        parent,
        f".tmp_{os.path.basename(target)}.{os.getpid()}.{uuid.uuid4().hex[:8]}",
    )
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    payload = json.dumps(
        _json_safe(data),
        indent=indent,
        allow_nan=False,
        ensure_ascii=True,
    )
    atomic_write_text(path, payload + "\n", encoding="utf-8")


def atomic_write_csv(
    path: Union[str, Path],
    df: Any,
    index: bool = False,
    header: bool = True,
    sep: str = ",",
) -> None:
    """Atomically write a pandas DataFrame to CSV via a same-directory temp file and os.replace."""
    target = os.path.abspath(str(path))
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(
        parent,
        f".tmp_{os.path.basename(target)}.{os.getpid()}.{uuid.uuid4().hex[:8]}",
    )
    try:
        df.to_csv(tmp_path, index=index, header=header, sep=sep)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def compute_code_sha256(
    module_paths: Sequence[Union[str, Path]],
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Compute {filename: sha256} for the given code files plus run_identity.py."""
    here = Path(__file__).resolve().parent
    base = Path(base_dir).resolve() if base_dir else here
    seen: Dict[str, Path] = {}
    all_paths = list(module_paths) + [Path(__file__).resolve()]
    for raw in all_paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (base / p).resolve()
        else:
            p = p.resolve()
        try:
            key = p.relative_to(base).as_posix()
        except ValueError:
            key = p.name
        seen[key] = p
    out: Dict[str, str] = {}
    for key in sorted(seen):
        p = seen[key]
        if not p.is_file():
            raise RunIdentityError(f"CODE_FILE_MISSING: '{p}' does not exist")
        out[key] = sha256_file(p)
    return out


def compute_config_sha256(
    config: Any = None,
    preregistration_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compute SHA256 binding configuration and optional preregistration document."""
    prereg_sha: Optional[str] = None
    if preregistration_path is not None:
        p = Path(preregistration_path)
        if not p.is_file():
            raise RunIdentityError(f"PREREGISTRATION_MISSING: '{p}' does not exist")
        prereg_sha = sha256_file(p)
    return sha256_obj({
        "config": config,
        "preregistration_sha256": prereg_sha,
    })


def resolve_code_files(
    base_dir: Union[str, Path],
    filenames: Sequence[str],
) -> List[str]:
    """Return existing absolute paths for a runner's local code dependencies."""
    root = Path(base_dir)
    out: List[str] = []
    for name in filenames:
        p = root / name
        if p.is_file():
            out.append(str(p))
    return out


def discover_local_code_dependencies(
    base_dir: Union[str, Path],
    entry_filenames: Sequence[str],
) -> List[str]:
    """Transitively discover local .py dependencies imported by entry_filenames under base_dir."""
    import ast

    root = Path(base_dir)
    local_mods = {p.stem: p.name for p in root.glob("*.py") if p.is_file()}
    visited: List[str] = []
    seen = set()
    queue = [name for name in entry_filenames if (root / name).is_file()]
    while queue:
        fname = queue.pop(0)
        if fname in seen:
            continue
        fpath = root / fname
        if not fpath.is_file():
            continue
        seen.add(fname)
        visited.append(str(fpath))
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in local_mods and local_mods[mod] not in seen:
                        queue.append(local_mods[mod])
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split(".")[0]
                if mod in local_mods and local_mods[mod] not in seen:
                    queue.append(local_mods[mod])
    return sorted(set(visited))


def compute_input_sha256(
    input_paths: Sequence[Union[str, Path]] = (),
    input_spec: Any = None,
) -> str:
    """Compute SHA256 binding input file contents and explicit input specification."""
    file_hashes: Dict[str, str] = {}
    for raw in sorted(str(x) for x in input_paths if x is not None):
        p = Path(raw)
        if not p.is_file():
            raise RunIdentityError(f"INPUT_FILE_MISSING: '{p}' does not exist")
        norm_key = os.path.abspath(str(p)).replace("\\", "/")
        digest = sha256_file(p)
        if norm_key in file_hashes and file_hashes[norm_key] != digest:
            raise RunIdentityError(f"INPUT_PATH_COLLISION: conflicting digests for '{norm_key}'")
        file_hashes[norm_key] = digest
    return sha256_obj({
        "files": file_hashes,
        "spec": input_spec,
    })


def capture_frozen_environment(
    repo_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Capture read-only executed environment (OPS-13): Python, pip freeze, platform, git."""
    resolved_repo = str(Path(repo_dir).resolve()) if repo_dir else str(Path(__file__).resolve().parent)
    if use_cache and resolved_repo in _ENV_CACHE:
        return dict(_ENV_CACHE[resolved_repo], captured_at=_utc_now_iso())

    pip_freeze: List[str] = []
    pip_source = "pip_freeze"
    pip_error: Optional[str] = None
    try:
        env = dict(os.environ)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            check=False,
        )
        if proc.returncode == 0:
            pip_freeze = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        else:
            pip_error = (proc.stderr or f"exit_code={proc.returncode}").strip()
    except Exception as exc:
        pip_error = f"{type(exc).__name__}: {exc}"

    if pip_error is not None:
        pip_source = "importlib.metadata"
        try:
            pip_freeze = sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            )
        except Exception as meta_exc:
            pip_error = f"{pip_error}; metadata_fallback={type(meta_exc).__name__}: {meta_exc}"

    git_revision: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_error: Optional[str] = None
    try:
        rev_proc = subprocess.run(
            ["git", "-C", resolved_repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if rev_proc.returncode == 0:
            git_revision = rev_proc.stdout.strip()
            st_proc = subprocess.run(
                ["git", "-C", resolved_repo, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if st_proc.returncode == 0:
                git_dirty = bool(st_proc.stdout.strip())
            else:
                git_error = (st_proc.stderr or f"git_status_exit={st_proc.returncode}").strip()
        else:
            git_error = (rev_proc.stderr or f"git_rev_parse_exit={rev_proc.returncode}").strip()
    except Exception as exc:
        git_error = f"{type(exc).__name__}: {exc}"

    snapshot: Dict[str, Any] = {
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "pip_freeze_source": pip_source,
        "pip_freeze": pip_freeze,
        "pip_freeze_error": pip_error,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "git_error": git_error,
        "captured_at": _utc_now_iso(),
    }
    if use_cache:
        _ENV_CACHE[resolved_repo] = dict(snapshot)
    return snapshot


def load_verified_json(
    path: Union[str, Path],
    expected_identity: Optional[Dict[str, Any]] = None,
    required_keys: Iterable[str] = (),
) -> Dict[str, Any]:
    """Load JSON strictly; reject truncated/partial records and identity mismatches."""
    p = str(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            raise RunIdentityError(f"PARTIAL_OR_CORRUPT_RECORD: '{p}' is empty")
        data = json.loads(raw)
    except RunIdentityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunIdentityError(f"PARTIAL_OR_CORRUPT_RECORD: failed to parse '{p}': {exc}") from exc

    if not isinstance(data, dict):
        raise RunIdentityError(f"PARTIAL_OR_CORRUPT_RECORD: '{p}' must be a JSON object")

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise RunIdentityError(
            f"PARTIAL_OR_CORRUPT_RECORD: '{p}' missing required keys {sorted(missing)}"
        )

    if expected_identity is not None:
        verify_identity_match(data.get("identity"), expected_identity, record_path=p)
    return data


def verify_identity_match(
    saved_identity: Any,
    current_identity: Dict[str, Any],
    record_path: str = "<record>",
) -> None:
    """Verify code, config/preregistration, and input SHA256s all match (OPS-14)."""
    if not isinstance(saved_identity, dict) or not _REQUIRED_IDENTITY_KEYS.issubset(saved_identity.keys()):
        raise RunIdentityError(
            f"PARTIAL_OR_CORRUPT_RECORD: '{record_path}' has missing or invalid identity block"
        )
    if saved_identity.get("code_sha256") != current_identity.get("code_sha256"):
        raise RunIdentityError(
            f"STALE_CODE_SHA256: code SHA256 mismatch in '{record_path}'; refusing to reuse record"
        )
    if saved_identity.get("config_sha256") != current_identity.get("config_sha256"):
        raise RunIdentityError(
            f"STALE_CONFIG_SHA256: config/preregistration SHA256 mismatch in '{record_path}'; refusing to reuse record"
        )
    if saved_identity.get("input_sha256") != current_identity.get("input_sha256"):
        raise RunIdentityError(
            f"STALE_INPUT_SHA256: input SHA256 mismatch in '{record_path}'; refusing to reuse record"
        )


def extract_mast_product_paths(obj: Any) -> List[str]:
    """Extract local FITS/product file paths from a lightkurve object or collection."""
    paths: List[str] = []
    if obj is None:
        return paths
    for attr in ("filename", "path"):
        fname = getattr(obj, attr, None)
        if isinstance(fname, (str, Path)) and str(fname).strip():
            p = str(fname)
            if os.path.isfile(p):
                paths.append(p)
    if isinstance(obj, (list, tuple)) or hasattr(obj, "data"):
        items = obj if isinstance(obj, (list, tuple)) else getattr(obj, "data", ())
        if isinstance(items, (list, tuple)):
            for item in items:
                if item is not obj:
                    paths.extend(extract_mast_product_paths(item))
    seen = set()
    unique: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return unique


class BoundRun:
    """Manages an atomic, identity-bound run record with OPS-13/14/33 and resume/interruption logs."""

    def __init__(
        self,
        record_path: Union[str, Path],
        data: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        self.record_path = os.path.abspath(str(record_path))
        self.data = data
        self.logger = logger

    @property
    def run_id(self) -> str:
        return str(self.data["run_id"])

    @property
    def identity(self) -> Dict[str, Any]:
        return dict(self.data["identity"])

    @property
    def environment(self) -> Dict[str, Any]:
        return dict(self.data["environment"])

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    def save(self) -> None:
        self.data["updated_at"] = _utc_now_iso()
        atomic_write_json(self.record_path, self.data)

    @classmethod
    def open_or_resume(
        cls,
        record_path: Union[str, Path],
        *,
        run_id: str,
        runner: str,
        code_files: Sequence[Union[str, Path]],
        config: Any = None,
        preregistration_path: Optional[Union[str, Path]] = None,
        input_paths: Sequence[Union[str, Path]] = (),
        input_spec: Any = None,
        repo_dir: Optional[Union[str, Path]] = None,
        resume: bool = True,
        require_existing: bool = False,
        resume_reason: str = "resume",
        logger: Optional[logging.Logger] = None,
    ) -> "BoundRun":
        abs_path = os.path.abspath(str(record_path))
        base_dir = Path(repo_dir).resolve() if repo_dir else Path(__file__).resolve().parent
        identity = {
            "code_sha256": compute_code_sha256(code_files, base_dir=base_dir),
            "config_sha256": compute_config_sha256(config, preregistration_path=preregistration_path),
            "input_sha256": compute_input_sha256(input_paths=input_paths, input_spec=input_spec),
        }

        if os.path.exists(abs_path):
            if not resume:
                raise RunIdentityError(
                    f"RUN_RECORD_EXISTS: '{abs_path}' already exists for run '{run_id}'; refusing to overwrite without resume"
                )
            saved = load_verified_json(
                abs_path,
                expected_identity=identity,
                required_keys=_REQUIRED_RECORD_KEYS,
            )
            if str(saved.get("run_id")) != str(run_id):
                raise RunIdentityError(
                    f"STALE_RUN_ID: record '{abs_path}' belongs to run '{saved.get('run_id')}', not '{run_id}'"
                )
            inst = cls(abs_path, saved, logger=logger)
            inst.verify_mast_products()
            prev_status = saved.get("status")
            saved.setdefault("resumes", []).append({
                "timestamp": _utc_now_iso(),
                "pid": os.getpid(),
                "previous_status": prev_status,
                "reason": resume_reason,
            })
            saved["status"] = "IN_PROGRESS"
            inst.save()
            inst._log(
                f"[RUN_IDENTITY] Resumed {runner} run_id='{run_id}' (previous_status={prev_status}, resume_count={len(saved['resumes'])})"
            )
            return inst

        if require_existing:
            raise RunIdentityError(
                f"MISSING_RUN_RECORD: cannot resume '{run_id}', record '{abs_path}' not found"
            )

        now = _utc_now_iso()
        data: Dict[str, Any] = {
            "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
            "run_id": str(run_id),
            "runner": str(runner),
            "status": "IN_PROGRESS",
            "started_at": now,
            "updated_at": now,
            "identity": identity,
            "environment": capture_frozen_environment(repo_dir=base_dir),
            "mast_products": {},
            "interruptions": [],
            "resumes": [],
            "completed_items": {},
        }
        inst = cls(abs_path, data, logger=logger)
        inst.save()
        return inst

    def verify_mast_products(self) -> None:
        """Verify all recorded MAST products on disk still match their recorded SHA256 (OPS-33)."""
        products = self.data.get("mast_products")
        if not isinstance(products, dict):
            raise RunIdentityError(
                f"PARTIAL_OR_CORRUPT_RECORD: '{self.record_path}' has invalid mast_products"
            )
        for key, info in products.items():
            if not isinstance(info, dict) or "sha256" not in info:
                raise RunIdentityError(
                    f"PARTIAL_OR_CORRUPT_RECORD: invalid MAST product entry '{key}' in '{self.record_path}'"
                )
            p = info.get("path")
            if p and os.path.isfile(p):
                actual = sha256_file(p)
                if actual != info["sha256"]:
                    raise RunIdentityError(
                        f"STALE_MAST_PRODUCT: MAST product '{p}' changed since prior run "
                        f"(expected {info['sha256']}, got {actual})"
                    )

    def record_mast_product(
        self,
        path: Union[str, Path],
        *,
        tic_id: Optional[Union[str, int]] = None,
        sector: Optional[int] = None,
        product_type: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Record a MAST product file with its SHA256; detect if a previously recorded file changed."""
        abs_p = os.path.abspath(str(path))
        if not os.path.isfile(abs_p):
            raise RunIdentityError(f"MAST_PRODUCT_MISSING: '{abs_p}' does not exist")
        digest = sha256_file(abs_p)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RunIdentityError(
                f"STALE_MAST_PRODUCT: MAST product '{abs_p}' changed since sidecar was written "
                f"(expected {expected_sha256}, got {digest})"
            )
        size = os.path.getsize(abs_p)
        products = self.data.setdefault("mast_products", {})
        existing = products.get(abs_p)
        if existing is not None and existing.get("sha256") != digest:
            raise RunIdentityError(
                f"STALE_MAST_PRODUCT: MAST product '{abs_p}' changed during run "
                f"(recorded {existing.get('sha256')}, got {digest})"
            )
        entry = {
            "path": abs_p,
            "sha256": digest,
            "size_bytes": size,
            "tic_id": str(tic_id) if tic_id is not None else None,
            "sector": int(sector) if sector is not None else None,
            "product_type": product_type,
            "recorded_at": _utc_now_iso(),
        }
        products[abs_p] = entry
        if save:
            self.save()
        return entry

    def record_mast_products_from_lk(
        self,
        lk_obj: Any,
        *,
        tic_id: Optional[Union[str, int]] = None,
        sector: Optional[int] = None,
        product_type: Optional[str] = None,
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        entries = []
        paths = extract_mast_product_paths(lk_obj)
        for p in paths:
            entries.append(
                self.record_mast_product(
                    p,
                    tic_id=tic_id,
                    sector=sector,
                    product_type=product_type,
                    save=False,
                )
            )
        if entries and save:
            self.save()
        return entries

    def record_interruption(
        self,
        reason: str,
        *,
        completed_count: Optional[int] = None,
    ) -> None:
        self.data.setdefault("interruptions", []).append({
            "timestamp": _utc_now_iso(),
            "pid": os.getpid(),
            "reason": reason,
            "completed_count": completed_count,
        })
        self.data["status"] = "INTERRUPTED"
        self.save()
        self._log(
            f"[RUN_IDENTITY] Interrupted {self.data.get('runner')} run_id='{self.run_id}' "
            f"(reason={reason}, completed={completed_count})"
        )

    def record_completion(
        self,
        status: str = "COMPLETE",
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data["status"] = status
        self.data["finished_at"] = _utc_now_iso()
        if extra:
            self.data.update(_json_safe(extra))
        self.save()

    def save_item_record(self, item_path: Union[str, Path], payload: Dict[str, Any]) -> None:
        record = dict(payload)
        record["run_id"] = self.run_id
        record["identity"] = self.identity
        atomic_write_json(item_path, record)

    def load_verified_item_record(
        self,
        item_path: Union[str, Path],
        *,
        required_keys: Sequence[str] = ("run_id", "identity"),
        artifact_fields: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        req = tuple(dict.fromkeys(["run_id", "identity", *required_keys]))
        data = load_verified_json(
            item_path,
            expected_identity=self.identity,
            required_keys=req,
        )
        for key in req:
            val = data.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise RunIdentityError(
                    f"PARTIAL_OR_CORRUPT_RECORD: '{item_path}' has empty/null required field '{key}'"
                )
        if artifact_fields:
            for path_key, sha_key in artifact_fields.items():
                art_path = data.get(path_key)
                art_sha = data.get(sha_key)
                if not isinstance(art_path, str) or not art_path.strip():
                    raise RunIdentityError(
                        f"PARTIAL_OR_CORRUPT_RECORD: '{item_path}' missing artifact path '{path_key}'"
                    )
                if not isinstance(art_sha, str) or not art_sha.strip():
                    raise RunIdentityError(
                        f"PARTIAL_OR_CORRUPT_RECORD: '{item_path}' missing artifact digest '{sha_key}'"
                    )
                if not os.path.isfile(art_path):
                    raise RunIdentityError(
                        f"PARTIAL_OR_CORRUPT_RECORD: artifact '{art_path}' referenced by '{item_path}' is missing"
                    )
                actual_sha = sha256_file(art_path)
                if actual_sha != art_sha:
                    raise RunIdentityError(
                        f"PARTIAL_OR_CORRUPT_RECORD: artifact '{art_path}' SHA256 mismatch "
                        f"(expected {art_sha}, got {actual_sha})"
                    )
        return data
