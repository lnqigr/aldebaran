#!/usr/bin/env python3
"""
tests/test_run_merge_pbs.py
===========================
Comprehensive offline test suite for cluster/run_merge.pbs:
Safe downstream orchestration PBS wrapper, failure propagation, signal trapping,
and guarded cleanup.

Verifies:
1. Real Bash execution of cluster/run_merge.pbs with local shell stubs.
2. Normal execution flow: Python merge_shards invoked with structured arguments,
   clean_scratch executed only on success, exit code 0.
3. Python failure propagation: when merge_shards fails (e.g. exit 7), clean_scratch
   is NOT executed, scratch is retained, and exit code 7 is propagated.
4. Preflight / setup failure: missing manifest exits with code 2 without running science.
5. Setup failure: failed venv creation exits with code 3, scratch retained.
6. Cleanup failure: clean_scratch returning 1 exits with code 5.
7. Signal handling: catching SIGTERM terminates child process tree, including resistant
   grandchild, retains scratch, and exits 143.
8. Unsafe overlap: refusing OUTPUT_DIR inside SCRATCHDIR with exit 1.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


def find_bash():
    """Locate bash executable on the host machine."""
    for candidate in [
        r"C:\Program Files\Git\bin\bash.exe",
        "bash",
        "bash.exe",
    ]:
        found = shutil.which(candidate)
        if found:
            return found
    return None


BASH_EXE = find_bash()


@unittest.skipIf(not BASH_EXE, "Bash executable not found in environment")
class TestRunMergePBS(unittest.TestCase):

    def setUp(self):
        self.test_root = tempfile.mkdtemp(prefix="ald_test_merge_pbs_")
        self.repo_dir = os.path.join(self.test_root, "repo")
        self.scratch_dir = os.path.join(self.test_root, "scratch")
        self.bin_dir = os.path.join(self.test_root, "bin")

        os.makedirs(self.repo_dir, exist_ok=True)
        os.makedirs(self.scratch_dir, exist_ok=True)
        os.makedirs(self.bin_dir, exist_ok=True)

        self.pbs_script_src = os.path.abspath(os.path.join("cluster", "run_merge.pbs"))
        if not os.path.exists(self.pbs_script_src):
            self.fail(f"Production PBS script not found at {self.pbs_script_src}")

        # Copy production PBS script to repo and ensure executable mode
        self.pbs_script = os.path.join(self.repo_dir, "run_merge.pbs")
        shutil.copy2(self.pbs_script_src, self.pbs_script)
        os.chmod(self.pbs_script, 0o755)

        # Create dummy merge_shards.py sentinel in repo
        self.merge_shards_py = os.path.join(self.repo_dir, "merge_shards.py")
        with open(self.merge_shards_py, "w", encoding="utf-8") as f:
            f.write("# Dummy merge_shards.py sentinel\n")

        # Setup standard shell stubs
        self._create_shell_stubs()

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _create_shell_stubs(self):
        # module stub
        with open(os.path.join(self.bin_dir, "module"), "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/bin/bash\necho MODULE_STUB_CALLED >> \"${SCRATCHDIR:-.}/.module_stub_called\"\nexit 0\n")

        # pip stub
        with open(os.path.join(self.bin_dir, "pip"), "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/bin/bash\necho GLOBAL_PIP_CALLED >> \"${SCRATCHDIR:-.}/.pip_stub_called\"\nexit 0\n")

        # clean_scratch stub
        with open(os.path.join(self.bin_dir, "clean_scratch"), "w", encoding="utf-8", newline="\n") as f:
            f.write("""#!/bin/bash
if [ "${INJECT_CLEAN_SCRATCH_FAIL:-0}" = "1" ]; then
    echo "[stub clean_scratch] Simulated failure" >&2
    exit 1
fi
echo CLEAN_SCRATCH_CALLED >> "${SCRATCHDIR:-.}/clean_call.log"
exit 0
""")

        real_python = sys.executable.replace("\\", "/")

        # python3 and python stubs
        with open(os.path.join(self.bin_dir, "python3"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"""#!/bin/bash
echo PYTHON3_STUB_CALLED >> "${{SCRATCHDIR:-.}}/.python3_stub_called"
if [ "$1" = "-c" ]; then
    shift
    exec "{real_python}" -c "$@"
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    if [ "${{INJECT_VENV_FAIL:-0}}" = "1" ]; then
        echo "Simulated venv creation failure" >&2
        exit 1
    fi
    target="$3"
    mkdir -p "$target/bin" "$target/Scripts"
    cat << VENV_ACT > "$target/bin/activate"
export PATH="$target/bin:\\$PATH"
deactivate () {{ :; }}
VENV_ACT
    cat << 'VENV_PY' > "$target/bin/python"
#!/bin/bash
echo VENV_PYTHON_CALLED >> "${{SCRATCHDIR:-.}}/.venv_python_called"
if [ "$1" = "-c" ]; then
    shift
    exec "{real_python}" -c "$@"
fi
if [ -n "$INJECT_PYTHON_EXIT" ]; then
    echo "Injected python exit $INJECT_PYTHON_EXIT" >&2
    exit $INJECT_PYTHON_EXIT
fi
if [ -n "$SPAWN_RESISTANT_CHILD" ]; then
    (
        trap '' TERM
        while true; do sleep 0.2; done
    ) &
    GRANDCHILD_PID=$!
    echo "$GRANDCHILD_PID" > "$SCRATCHDIR/.grandchild_pid"
    echo "$$" > "$SCRATCHDIR/.py_started"
    wait "$GRANDCHILD_PID" 2>/dev/null
    exit 0
fi
if [ -n "$SLEEP_IN_PYTHON" ]; then
    on_sig() {{
        if [ -n "$SLEEP_PID" ]; then
            kill -TERM "$SLEEP_PID" 2>/dev/null || true
            kill -KILL "$SLEEP_PID" 2>/dev/null || true
        fi
        exit 143
    }}
    trap on_sig TERM INT
    if [ "${{SLEEP_WITHOUT_MARKER:-0}}" != "1" ]; then
        echo "$$" > "$SCRATCHDIR/.py_started" || {{ echo "Failed to write .py_started" >&2; exit 1; }}
    else
        echo "$$" > "$SCRATCHDIR/.unadvertised_child_pid" || true
    fi
    sleep "$SLEEP_IN_PYTHON" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null
    exit 0
fi
echo "$@" > "$SCRATCHDIR/python_args.txt"
# Create output directory and manifests if requested
out_dir=""
run_id=""
for ((i=1;i<=$#;i++)); do
    if [ "${{!i}}" = "--output-dir" ]; then
        j=$((i+1))
        out_dir="${{!j}}"
    elif [ "${{!i}}" = "--run-id" ]; then
        j=$((i+1))
        run_id="${{!j}}"
    fi
done
eff_run_id="${{INJECT_RUN_ID:-$run_id}}"
st_status="${{INJECT_RUN_STATUS:-SUCCESS}}"
if [ -n "$out_dir" ]; then
    mkdir -p "$out_dir"
    echo '{{"status": "'"$st_status"'", "run_id": "'"$eff_run_id"'"}}' > "$out_dir/run_status.json"
    echo "dummy candidate content" > "$out_dir/candidates_consolidated.csv"
    dummy_hash="$("{real_python}" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest().upper())' "$out_dir/candidates_consolidated.csv")"
    if [ "${{INJECT_CORRUPT_MANIFEST:-0}}" = "1" ]; then
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"candidates_consolidated.csv": {{"sha256": "BAD_HASH_CORRUPT"}}}}}}' > "$out_dir/manifest.json"
    elif [ "${{INJECT_MALFORMED_MANIFEST:-0}}" = "1" ]; then
        echo '{{malformed json' > "$out_dir/manifest.json"
    elif [ "${{INJECT_NON_OBJECT_ARTIFACT:-0}}" = "1" ]; then
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"missing_required_output.csv": null}}}}' > "$out_dir/manifest.json"
    elif [ "${{INJECT_MIXED_MALFORMED_ARTIFACT:-0}}" = "1" ]; then
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"candidates_consolidated.csv": {{"sha256": "'"$dummy_hash"'"}}, "broken.csv": null}}}}' > "$out_dir/manifest.json"
    elif [ "${{INJECT_ALL_MALFORMED_ARTIFACTS:-0}}" = "1" ]; then
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"a.csv": 123, "b.csv": "not an object"}}}}' > "$out_dir/manifest.json"
    elif [ "${{INJECT_TRAVERSAL_ARTIFACT:-0}}" = "1" ]; then
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"../escape.csv": {{"sha256": "1234567890123456789012345678901234567890123456789012345678901234"}}}}}}' > "$out_dir/manifest.json"
    else
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"candidates_consolidated.csv": {{"sha256": "'"$dummy_hash"'"}}}}}}' > "$out_dir/manifest.json"
    fi
fi

exit 0
VENV_PY
    cat << 'VENV_PIP' > "$target/bin/pip"
#!/bin/bash
echo "VENV_PIP_CALLED" >> "${{SCRATCHDIR:-.}}/.pip_stub_called"
echo "VENV_PIP_CALLED" >> "${{SCRATCHDIR:-.}}/.venv_pip_called"
exit 0
VENV_PIP
    chmod +x "$target/bin/python" "$target/bin/pip"
    exit 0
fi
exit 0
""")

        with open(os.path.join(self.bin_dir, "python"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"""#!/bin/bash
echo GLOBAL_PYTHON_CALLED >> "${{SCRATCHDIR:-.}}/.python_stub_called"
if [ "$1" = "-c" ]; then
    shift
    exec "{real_python}" -c "$@"
fi
if [ -n "$INJECT_PYTHON_EXIT" ]; then
    echo "Injected python exit $INJECT_PYTHON_EXIT" >&2
    exit $INJECT_PYTHON_EXIT
fi
if [ -n "$SPAWN_RESISTANT_CHILD" ]; then
    (
        trap '' TERM
        while true; do sleep 0.2; done
    ) &
    GRANDCHILD_PID=$!
    echo "$GRANDCHILD_PID" > "$SCRATCHDIR/.grandchild_pid"
    echo "$$" > "$SCRATCHDIR/.py_started"
    wait "$GRANDCHILD_PID" 2>/dev/null
    exit 0
fi
if [ -n "$SLEEP_IN_PYTHON" ]; then
    on_sig() {{
        if [ -n "$SLEEP_PID" ]; then
            kill -TERM "$SLEEP_PID" 2>/dev/null || true
            kill -KILL "$SLEEP_PID" 2>/dev/null || true
        fi
        exit 143
    }}
    trap on_sig TERM INT
    if [ "${{SLEEP_WITHOUT_MARKER:-0}}" != "1" ]; then
        echo "$$" > "$SCRATCHDIR/.py_started" || {{ echo "Failed to write .py_started" >&2; exit 1; }}
    else
        echo "$$" > "$SCRATCHDIR/.unadvertised_child_pid" || true
    fi
    sleep "$SLEEP_IN_PYTHON" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null
    exit 0
fi
echo "$@" > "$SCRATCHDIR/python_args.txt"
out_dir=""
run_id=""
for ((i=1;i<=$#;i++)); do
    if [ "${{!i}}" = "--output-dir" ]; then
        j=$((i+1))
        out_dir="${{!j}}"
    elif [ "${{!i}}" = "--run-id" ]; then
        j=$((i+1))
        run_id="${{!j}}"
    fi
done
eff_run_id="${{INJECT_RUN_ID:-$run_id}}"
st_status="${{INJECT_RUN_STATUS:-SUCCESS}}"
if [ -n "$out_dir" ]; then
    mkdir -p "$out_dir"
    echo '{{"status": "'"$st_status"'", "run_id": "'"$eff_run_id"'"}}' > "$out_dir/run_status.json"
    echo "dummy candidate content" > "$out_dir/candidates_consolidated.csv"
    if [ "${{INJECT_CORRUPT_MANIFEST:-0}}" = "1" ]; then
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"candidates_consolidated.csv": {{"sha256": "BAD_HASH_CORRUPT"}}}}}}' > "$out_dir/manifest.json"
    elif [ "${{INJECT_MALFORMED_MANIFEST:-0}}" = "1" ]; then
        echo '{{malformed json' > "$out_dir/manifest.json"
    else
        dummy_hash="$("{real_python}" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest().upper())' "$out_dir/candidates_consolidated.csv")"
        echo '{{"manifest_version": "1.0", "run_id": "'"$eff_run_id"'", "artifacts": {{"candidates_consolidated.csv": {{"sha256": "'"$dummy_hash"'"}}}}}}' > "$out_dir/manifest.json"
    fi
fi

exit 0
""")

        # Ensure all stubs in bin_dir have POSIX executable permissions (0o755)
        for s in ["module", "pip", "clean_scratch", "python3", "python"]:
            p = os.path.join(self.bin_dir, s)
            if os.path.exists(p):
                os.chmod(p, 0o755)

    def _create_dummy_manifest(self, run_id: str = "test_run") -> str:
        manifest_path = os.path.join(self.repo_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "manifest_version": "1.0",
                "run_id": run_id,
                "shards": []
            }, f)
        return manifest_path

    def _get_posix_group_snapshot(self, pgid: int) -> list:
        """Capture diagnostic snapshot of processes in the given POSIX process group.
        Returns a list of dicts: [{'pid': int, 'ppid': int, 'pgid': int, 'sid': int, 'stat': str, 'comm': str}].
        """
        snapshot = []
        try:
            clean_env = self._sanitize_env()
            res = subprocess.run(
                ["ps", "-eo", "pid,ppid,pgid,sid,stat,comm"],
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=5
            )
            if res.returncode == 0 and res.stdout:
                for line in res.stdout.strip().splitlines():
                    parts = line.strip().split(None, 5)
                    if len(parts) >= 6 and parts[0].isdigit() and parts[2].isdigit():
                        p_pid = int(parts[0])
                        p_ppid = int(parts[1]) if parts[1].isdigit() else -1
                        p_pgid = int(parts[2])
                        p_sid = int(parts[3]) if parts[3].isdigit() else -1
                        p_stat = parts[4]
                        p_comm = parts[5]
                        if p_pgid == pgid:
                            snapshot.append({
                                "pid": p_pid,
                                "ppid": p_ppid,
                                "pgid": p_pgid,
                                "sid": p_sid,
                                "stat": p_stat,
                                "comm": p_comm
                            })
        except Exception:
            pass
        return snapshot

    def _cleanup_process_tree(self, proc):
        """Clean only fixture-owned process descendants on timeout/cancellation.
        Raises RuntimeError if any owned process or group survives cleanup.
        """
        clean_env = self._sanitize_env()
        pgid = getattr(proc, "_owned_pgid", proc.pid)
        surviving_entities = []

        if os.name == "posix":
            # 1. Capture snapshot before signals
            initial_snap = self._get_posix_group_snapshot(pgid)

            # 2. Escalating signals to process group
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError as pe:
                surviving_entities.append(f"group {pgid} (SIGTERM PermissionError: {pe})")

            time.sleep(0.2)
            try:
                os.killpg(pgid, getattr(signal, "SIGKILL", 9))
            except ProcessLookupError:
                pass
            except PermissionError as pe:
                surviving_entities.append(f"group {pgid} (SIGKILL PermissionError: {pe})")

            # 3. REAP DIRECT CHILD FIRST with bounded wait.
            # In POSIX, proc is the process group leader. Terminated children become zombies (stat='Z')
            # and retain their group membership in pgid until their parent reaps them via wait().
            # If we probe killpg(pgid, 0) before reaping proc, the unreaped zombie causes killpg to return 0,
            # falsely diagnosing the group as surviving. Reaping proc clears its zombie entry from pgid.
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            if proc.poll() is None:
                surviving_entities.append(f"parent process PID {proc.pid}")

            # 4. Bounded probe of remaining process group members
            group_is_dead = False
            for _ in range(20):
                try:
                    os.killpg(pgid, 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    group_is_dead = True
                    break
                except PermissionError as pe:
                    surviving_entities.append(f"group {pgid} (killpg(0) PermissionError: {pe})")
                    break

            # 5. State-aware diagnostics if group appears to survive
            if not group_is_dead and not any(f"group {pgid}" in s for s in surviving_entities):
                snap = self._get_posix_group_snapshot(pgid)
                if snap:
                    living = [p for p in snap if not p.get("stat", "").startswith("Z")]
                    zombies = [p for p in snap if p.get("stat", "").startswith("Z")]
                    if living:
                        desc = ", ".join(f"PID {p['pid']} (stat={p.get('stat')}, comm={p.get('comm')})" for p in living)
                        surviving_entities.append(f"process group {pgid} (active living descendants: {desc})")
                    elif zombies:
                        # Reparented zombies waiting for init/subreaper (PID 1) to reap
                        zombies_cleared = False
                        for _ in range(10):
                            time.sleep(0.1)
                            try:
                                os.killpg(pgid, 0)
                            except ProcessLookupError:
                                zombies_cleared = True
                                break
                            except Exception:
                                pass
                        if not zombies_cleared:
                            zdesc = ", ".join(f"PID {p['pid']} (ppid={p.get('ppid')}, comm={p.get('comm')})" for p in zombies)
                            surviving_entities.append(f"process group {pgid} (unreaped zombie descendants: {zdesc})")
                else:
                    surviving_entities.append(f"process group {pgid} (unknown active members)")
        else:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    env=clean_env,
                    capture_output=True,
                    check=False,
                    timeout=5
                )
            except Exception:
                pass

            if BASH_EXE:
                # Under Git-Bash/MSYS2, background subshells spawned by the PBS wrapper can detach from
                # the Windows Win32 parent process tree. Terminate any MSYS2 processes running within this test root.
                try:
                    ps_out = subprocess.run(
                        [BASH_EXE, "-c", "ps -ef"],
                        env=clean_env,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    test_root_norm = self.test_root.replace("\\", "/").lower()
                    test_root_win = self.test_root.replace("/", "\\").lower()
                    for line in ps_out.stdout.splitlines():
                        line_lower = line.lower()
                        if test_root_norm in line_lower or test_root_win in line_lower:
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                bg_pid = parts[1]
                                subprocess.run(
                                    [BASH_EXE, "-c", f"kill -9 {bg_pid} 2>/dev/null || true"],
                                    env=clean_env,
                                    capture_output=True,
                                    timeout=5
                                )
                except Exception:
                    pass

            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
                if proc.poll() is None:
                    surviving_entities.append(f"parent process PID {proc.pid}")

        # Also terminate and verify death of any fixture-recorded child PIDs
        for marker_name in [".py_started", ".grandchild_pid"]:
            marker_file = os.path.join(self.scratch_dir, marker_name)
            if os.path.exists(marker_file):
                try:
                    with open(marker_file, "r") as f:
                        target_pid = f.read().strip()
                    if target_pid and target_pid.isdigit():
                        pid_int = int(target_pid)
                        if os.name == "posix":
                            try:
                                os.kill(pid_int, getattr(signal, "SIGKILL", 9))
                            except ProcessLookupError:
                                pass
                            except PermissionError as pe:
                                surviving_entities.append(f"PID {pid_int} ({marker_name} PermissionError: {pe})")

                            pid_is_dead = False
                            for _ in range(20):
                                try:
                                    os.kill(pid_int, 0)
                                    time.sleep(0.05)
                                except ProcessLookupError:
                                    pid_is_dead = True
                                    break
                                except PermissionError as pe:
                                    surviving_entities.append(f"PID {pid_int} ({marker_name} kill(0) PermissionError: {pe})")
                                    break
                            if not pid_is_dead and not any(f"PID {pid_int}" in s for s in surviving_entities):
                                surviving_entities.append(f"PID {pid_int} ({marker_name})")
                        else:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", str(target_pid)],
                                env=clean_env,
                                capture_output=True,
                                check=False,
                                timeout=5
                            )
                            subprocess.run(
                                [BASH_EXE, "-c", f"kill -9 {target_pid} 2>/dev/null || true"],
                                env=clean_env,
                                capture_output=True,
                                check=False,
                                timeout=5
                            )
                            pid_is_dead = False
                            for _ in range(20):
                                chk = subprocess.run(
                                    [BASH_EXE, "-c", f"kill -0 {target_pid} 2>/dev/null && echo ALIVE || echo DEAD"],
                                    env=clean_env,
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if chk.stdout.strip() == "DEAD":
                                    pid_is_dead = True
                                    break
                                time.sleep(0.05)
                            if not pid_is_dead:
                                surviving_entities.append(f"PID {target_pid} ({marker_name})")
                except Exception as ex:
                    surviving_entities.append(f"error verifying {marker_name}: {ex}")

        if surviving_entities:
            raise RuntimeError(f"Process cleanup failed; surviving owned processes: {', '.join(surviving_entities)}")

    def _sanitize_env(self, extra_env: dict = None) -> dict:
        """Create an isolated, sanitized shell environment for PBS wrapper execution."""
        env = os.environ.copy()

        # Clear ALL recognized ambient test / injection / PBS / workflow controls
        controlled_keys = [
            "MANIFEST_PATH", "RUN_ID", "OUTPUT_DIR",
            "SLEEP_IN_PYTHON", "SPAWN_RESISTANT_CHILD",
            "RUN_PLEIADES", "RUN_ANTARES", "RUN_TRICERATOPS", "ALLOW_PARTIAL",
            "INJECT_RUN_ID", "INJECT_RUN_STATUS", "INJECT_PYTHON_EXIT",
            "INJECT_VENV_FAIL", "INJECT_CLEAN_SCRATCH_FAIL", "INJECT_CORRUPT_MANIFEST",
            "INJECT_MALFORMED_MANIFEST", "INJECT_NON_OBJECT_ARTIFACT",
            "INJECT_MIXED_MALFORMED_ARTIFACT", "INJECT_ALL_MALFORMED_ARTIFACTS",
            "INJECT_TRAVERSAL_ARTIFACT", "SLEEP_WITHOUT_MARKER"
        ]
        for k in controlled_keys:
            env.pop(k, None)

        # Force fixture-safe defaults independent of ambient external environment
        env["SCRATCHDIR"] = self.scratch_dir
        env["PBS_O_WORKDIR"] = self.repo_dir
        env["PBS_JOBID"] = "12345.test_node"
        env["MANIFEST_PATH"] = os.path.join(self.repo_dir, "manifest.json")
        env["RUN_ID"] = "default_run_id"
        env["OUTPUT_DIR"] = os.path.join(self.repo_dir, "merged_output")
        env["ALDEBARAN_OFFLINE"] = "1"

        if extra_env:
            env.update(extra_env)

        # Remove inherited or injected shell function exports (e.g. BASH_FUNC_module%% or Lmod functions)
        for k in list(env.keys()):
            if k.startswith("BASH_FUNC_") or k.endswith("()"):
                del env[k]

        # Prevent shell startup scripts from running hostile code or overriding stubs
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)

        # Place test bin_dir at the front of PATH so all stubs are selected
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"

        return env

    def _run_pbs(self, extra_env: dict = None, timeout: int = 15):
        # 1. Sanitize environment before ANY subprocess or preflight check
        env = self._sanitize_env(extra_env)

        # 2. Preflight verification: verify required stubs exist and are executable in self.bin_dir
        # using the sanitized environment and finite timeouts to prevent ambient BASH_ENV hooks or hangs.
        required_stubs = ["module", "pip", "clean_scratch", "python3", "python"]
        for s in required_stubs:
            sp = os.path.join(self.bin_dir, s)
            if not os.path.exists(sp):
                raise RuntimeError(f"Preflight error: required stub '{s}' missing from fixture bin_dir '{self.bin_dir}'")
            if os.name == "posix":
                mode = os.stat(sp).st_mode
                if not (mode & 0o111) or not os.access(sp, os.X_OK):
                    raise RuntimeError(f"Preflight error: required stub '{s}' in '{self.bin_dir}' is not executable (mode {oct(mode)})")
            else:
                sp_bash = sp.replace("\\", "/")
                chk = subprocess.run(
                    [BASH_EXE, "-c", f"[ -x '{sp_bash}' ]"],
                    env=env,
                    capture_output=True,
                    timeout=5
                )
                if chk.returncode != 0:
                    raise RuntimeError(f"Preflight error: required stub '{s}' in '{self.bin_dir}' is not executable in bash")

        kwargs = {
            "cwd": self.repo_dir,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        cmd = [BASH_EXE, self.pbs_script.replace("\\", "/")]
        proc = subprocess.Popen(cmd, **kwargs)
        proc._owned_pgid = proc.pid

        def _to_str(val):
            if val is None:
                return ""
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr
            )
        except subprocess.TimeoutExpired as exc:
            exc.proc = proc
            self._cleanup_process_tree(proc)
            try:
                out, err = proc.communicate(timeout=1)
            except Exception:
                out, err = "", ""
            exc.stdout = _to_str(getattr(exc, "stdout", None)) + _to_str(out)
            exc.stderr = _to_str(getattr(exc, "stderr", None)) + _to_str(err)
            raise exc

    def test_normal_success_invokes_orchestration_and_cleans_scratch(self):
        """
        R7 / NC-14: Normal execution runs Python merge_shards with structured arguments,
        preserves exit code 0, and runs clean_scratch exactly once on success.
        """
        manifest_path = self._create_dummy_manifest("test_run_success")
        out_dir = os.path.join(self.repo_dir, "merged_output")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_success",
            "OUTPUT_DIR": out_dir,
            "RUN_PLEIADES": "1",
            "RUN_ANTARES": "0",
            "RUN_TRICERATOPS": "0"
        })

        self.assertEqual(proc.returncode, 0, f"PBS script failed: {proc.stderr}")
        self.assertIn("Merge and orchestration completed successfully", proc.stdout)

        # Verify arguments passed to merge_shards.py
        args_file = os.path.join(self.scratch_dir, "python_args.txt")
        self.assertTrue(os.path.exists(args_file))
        with open(args_file, "r") as f:
            captured_args = f.read().strip()
        self.assertIn("merge_shards.py", captured_args)
        self.assertIn("--manifest", captured_args)
        self.assertIn("--run-pleiades", captured_args)
        self.assertNotIn("--run-triceratops", captured_args)

        # Verify clean_scratch was called
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertTrue(os.path.exists(clean_log))
        with open(clean_log, "r") as f:
            self.assertEqual(f.read().strip(), "CLEAN_SCRATCH_CALLED")

    def test_python_failure_propagates_exit_code_and_retains_scratch(self):
        """
        R7 / NC-15: When merge_shards fails (exit 7), clean_scratch is NOT called,
        scratch is retained, and exit code 7 is propagated.
        """
        manifest_path = self._create_dummy_manifest("test_run_fail")
        out_dir = os.path.join(self.repo_dir, "merged_output_fail")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_fail",
            "OUTPUT_DIR": out_dir,
            "INJECT_PYTHON_EXIT": "7"
        })

        self.assertEqual(proc.returncode, 7)
        self.assertIn("merge_shards.py failed with exit code 7", proc.stderr)
        self.assertIn("Scratch retained for debugging", proc.stderr)

        # clean_scratch must NOT have been called
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_missing_manifest_fails_fast_with_exit_2(self):
        """
        R7 / NC-16: Missing manifest file fails fast with exit code 2 before python is called.
        """
        proc = self._run_pbs({
            "MANIFEST_PATH": os.path.join(self.repo_dir, "nonexistent_manifest.json")
        })

        self.assertEqual(proc.returncode, 2)
        self.assertIn("Manifest file not found", proc.stderr)

        # Python should never have been invoked
        self.assertFalse(os.path.exists(os.path.join(self.scratch_dir, "python_args.txt")))

    def test_venv_creation_failure_fails_fast_with_exit_3(self):
        """
        R7 / NC-17: Failed venv creation exits with code 3 and retains scratch.
        """
        manifest_path = self._create_dummy_manifest("test_run_venv_fail")
        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "INJECT_VENV_FAIL": "1"
        })

        self.assertEqual(proc.returncode, 3)
        self.assertIn("Failed to create venv", proc.stderr)

    def test_clean_scratch_failure_propagates_exit_5(self):
        """
        R7 / NC-18: When clean_scratch returns 1 on exit, PBS wrapper exits with code 5.
        """
        manifest_path = self._create_dummy_manifest("test_run_clean_fail")
        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "INJECT_CLEAN_SCRATCH_FAIL": "1"
        })

        self.assertEqual(proc.returncode, 5)
        self.assertIn("clean_scratch failed", proc.stderr)

    def test_output_dir_inside_scratch_refused_with_exit_1(self):
        """
        R7: Setting OUTPUT_DIR inside SCRATCHDIR is refused with exit 1 to prevent data destruction.
        """
        manifest_path = self._create_dummy_manifest("test_run_overlap")
        unsafe_out = os.path.join(self.scratch_dir, "unsafe_output")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "OUTPUT_DIR": unsafe_out
        })

        self.assertEqual(proc.returncode, 1)
        self.assertIn("OUTPUT_DIR cannot reside inside or equal SCRATCHDIR", proc.stderr)


    def test_signal_handling_terminates_child_and_exits_143(self):
        """
        R7 / NC-19: Catching SIGTERM terminates python child process, retains scratch,
        and exits with code 143.
        """
        manifest_path = self._create_dummy_manifest("test_run_signal")

        bin_posix = self.bin_dir.replace("\\", "/")
        target_posix = self.pbs_script.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")
        manifest_posix = manifest_path.replace("\\", "/")

        driver_script = os.path.join(self.test_root, "signal_driver.sh")
        driver_content = """#!/bin/bash
TARGET_PBS="$1"
MARKER="$SCRATCHDIR/.py_started"

"$TARGET_PBS" &
PBS_PID=$!

waited=0
while [ ! -f "$MARKER" ] && [ "$waited" -lt 40 ]; do
    sleep 0.25
    waited=$((waited + 1))
done

if [ ! -f "$MARKER" ]; then
    echo "ERROR: Marker $MARKER never appeared within timeout" >&2
    kill -KILL "$PBS_PID" 2>/dev/null || true
    exit 98
fi

echo "[driver] Sending SIGTERM to PBS_PID=$PBS_PID" >&2
kill -TERM "$PBS_PID" 2>/dev/null || true

# Bounded wait for PBS process to finish
term_waited=0
while kill -0 "$PBS_PID" 2>/dev/null && [ "$term_waited" -lt 20 ]; do
    sleep 0.25
    term_waited=$((term_waited + 1))
done

if kill -0 "$PBS_PID" 2>/dev/null; then
    echo "[driver] PBS_PID survived SIGTERM, killing with SIGKILL" >&2
    kill -KILL "$PBS_PID" 2>/dev/null || true
fi

wait "$PBS_PID" 2>/dev/null
EXIT_CODE=$?
echo "[driver] PBS wrapper exited with code $EXIT_CODE" >&2
exit $EXIT_CODE
"""
        with open(driver_script, "w", encoding="utf-8", newline="\n") as f:
            f.write(driver_content)
        os.chmod(driver_script, 0o755)

        env = self._sanitize_env({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_signal",
            "OUTPUT_DIR": os.path.join(self.repo_dir, "signal_out"),
            "SLEEP_IN_PYTHON": "30",
        })

        kwargs = {
            "cwd": self.repo_dir,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        cmd = [BASH_EXE, driver_script.replace("\\", "/"), target_posix]
        proc = subprocess.Popen(cmd, **kwargs)
        proc._owned_pgid = proc.pid

        def _to_str(val):
            if val is None:
                return ""
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val)

        try:
            stdout, stderr = proc.communicate(timeout=15)
            proc_res = subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr
            )
        except subprocess.TimeoutExpired as exc:
            exc.proc = proc
            self._cleanup_process_tree(proc)
            try:
                out, err = proc.communicate(timeout=1)
            except Exception:
                out, err = "", ""
            exc.stdout = _to_str(getattr(exc, "stdout", None)) + _to_str(out)
            exc.stderr = _to_str(getattr(exc, "stderr", None)) + _to_str(err)
            raise exc

        self.assertEqual(proc_res.returncode, 143, f"Expected exit 143, got {proc_res.returncode}.\nStderr: {proc_res.stderr}\nStdout: {proc_res.stdout}")
        self.assertIn("Caught signal SIGTERM", proc_res.stderr)
        self.assertIn("Scratch retained for diagnostic review", proc_res.stderr)

        # clean_scratch must NOT have been called
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_signal_handling_terminates_resistant_grandchild_and_exits_143(self):
        """
        R7 / NC: Deliberately resistant grandchild ignoring SIGTERM is terminated
        via wrapper's process-tree KILL escalation, and wrapper exits 143.
        """
        manifest_path = self._create_dummy_manifest("test_run_resistant")

        bin_posix = self.bin_dir.replace("\\", "/")
        target_posix = self.pbs_script.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")

        driver_script = os.path.join(self.test_root, "resistant_driver.sh")
        driver_content = """#!/bin/bash
TARGET_PBS="$1"
MARKER="$SCRATCHDIR/.py_started"

"$TARGET_PBS" &
PBS_PID=$!

waited=0
while [ ! -f "$MARKER" ] && [ "$waited" -lt 40 ]; do
    sleep 0.25
    waited=$((waited + 1))
done

if [ ! -f "$MARKER" ]; then
    echo "ERROR: Marker never appeared" >&2
    kill -KILL "$PBS_PID" 2>/dev/null || true
    exit 98
fi

echo "[driver] Sending SIGTERM to PBS_PID=$PBS_PID" >&2
kill -TERM "$PBS_PID" 2>/dev/null || true

# Wait for PBS process to finish signal handling and KILL escalation to resistant child
wait "$PBS_PID" 2>/dev/null
EXIT_CODE=$?
echo "[driver] PBS wrapper exited with code $EXIT_CODE" >&2
exit $EXIT_CODE
"""
        with open(driver_script, "w", encoding="utf-8", newline="\n") as f:
            f.write(driver_content)
        os.chmod(driver_script, 0o755)

        env = self._sanitize_env({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_resistant",
            "OUTPUT_DIR": os.path.join(self.repo_dir, "resistant_out"),
            "SPAWN_RESISTANT_CHILD": "1",
        })

        kwargs = {
            "cwd": self.repo_dir,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        cmd = [BASH_EXE, driver_script.replace("\\", "/"), target_posix]
        proc = subprocess.Popen(cmd, **kwargs)
        proc._owned_pgid = proc.pid

        def _to_str(val):
            if val is None:
                return ""
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val)

        try:
            stdout, stderr = proc.communicate(timeout=15)
            proc_res = subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr
            )
        except subprocess.TimeoutExpired as exc:
            exc.proc = proc
            self._cleanup_process_tree(proc)
            try:
                out, err = proc.communicate(timeout=1)
            except Exception:
                out, err = "", ""
            exc.stdout = _to_str(getattr(exc, "stdout", None)) + _to_str(out)
            exc.stderr = _to_str(getattr(exc, "stderr", None)) + _to_str(err)
            raise exc

        self.assertEqual(proc_res.returncode, 143, f"Expected exit 143, got {proc_res.returncode}.\nStderr: {proc_res.stderr}\nStdout: {proc_res.stdout}")
        self.assertIn("Caught signal SIGTERM", proc_res.stderr)

        # Verify resistant grandchild was terminated by KILL escalation
        grandchild_pid_file = os.path.join(self.scratch_dir, ".grandchild_pid")
        self.assertTrue(os.path.exists(grandchild_pid_file))
        with open(grandchild_pid_file, "r") as f:
            gc_pid = f.read().strip()
        self.assertTrue(gc_pid)

        # Check in bash that grandchild PID is indeed dead
        check_proc = subprocess.run(
            [BASH_EXE, "-c", f"kill -0 {gc_pid} 2>/dev/null && echo ALIVE || echo DEAD"],
            env=self._sanitize_env(),
            capture_output=True,
            text=True,
            timeout=5
        )
        self.assertEqual(check_proc.stdout.strip(), "DEAD", f"Resistant grandchild {gc_pid} survived wrapper termination!")

    def test_output_dir_equals_scratchdir_refused_with_exit_1(self):
        """
        B5 / NC: Setting OUTPUT_DIR identical to SCRATCHDIR is refused with exit 1 to prevent data destruction.
        """
        manifest_path = self._create_dummy_manifest("test_run_equal_scratch")
        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "OUTPUT_DIR": self.scratch_dir
        })
        self.assertEqual(proc.returncode, 1)
        self.assertIn("OUTPUT_DIR cannot reside inside or equal SCRATCHDIR", proc.stderr)

    def test_setup_failure_invalid_scratchdir_exits_3(self):
        """
        B5 / NC: Invalid or non-existent SCRATCHDIR fails setup fast with exit code 3.
        """
        manifest_path = self._create_dummy_manifest("test_run_bad_scratch")
        block_file = os.path.join(self.test_root, "not_a_dir")
        with open(block_file, "w") as f:
            f.write("block")
        invalid_scratch = os.path.join(block_file, "nested_scratch")
        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "SCRATCHDIR": invalid_scratch
        })
        self.assertEqual(proc.returncode, 3)
        self.assertIn("Failed to create SCRATCHDIR", proc.stderr)

    def test_pre_cleanup_missing_status_or_not_success_exits_4(self):
        """
        B5 / NC: If run_status.json has non-SUCCESS status or is missing before cleanup, clean_scratch is refused and exits 4.
        """
        manifest_path = self._create_dummy_manifest("test_run_failed_status")
        out_dir = os.path.join(self.repo_dir, "status_fail_dir")
        os.makedirs(out_dir, exist_ok=True)
        # Create output files where status is FAILED
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            f.write('{"manifest_version": "1.0"}')
        with open(os.path.join(out_dir, "run_status.json"), "w") as f:
            f.write('{"status": "FAILED", "run_id": "test_run_failed_status"}')

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_failed_status",
            "OUTPUT_DIR": out_dir,
            "INJECT_RUN_STATUS": "FAILED"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("Refusing scratch cleanup", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_c4_success_but_invalid_status_refused_exit_4(self):
        """
        C4 / NC: run_status.json with prefix 'SUCCESS_BUT_INVALID' is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_prefix_invalid")
        out_dir = os.path.join(self.repo_dir, "prefix_invalid_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_prefix_invalid",
            "OUTPUT_DIR": out_dir,
            "INJECT_RUN_STATUS": "SUCCESS_BUT_INVALID"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("must be exact SUCCESS or SUCCESS_PARTIAL_SURVEY", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_c4_success_partial_survey_status_accepted_exit_0(self):
        """
        C4: run_status.json with exact status 'SUCCESS_PARTIAL_SURVEY' is accepted and cleans scratch.
        """
        manifest_path = self._create_dummy_manifest("test_run_partial_valid")
        out_dir = os.path.join(self.repo_dir, "partial_valid_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_partial_valid",
            "OUTPUT_DIR": out_dir,
            "INJECT_RUN_STATUS": "SUCCESS_PARTIAL_SURVEY"
        })
        self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}")
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertTrue(os.path.exists(clean_log))

    def test_c4_malformed_manifest_json_refused_exit_4(self):
        """
        C4 / NC: Malformed manifest.json is rejected by JSON parser with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_malformed_man")
        out_dir = os.path.join(self.repo_dir, "malformed_man_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_malformed_man",
            "OUTPUT_DIR": out_dir,
            "INJECT_MALFORMED_MANIFEST": "1"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("manifest.json is malformed or unreadable", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_c4_corrupt_artifact_hash_refused_exit_4(self):
        """
        C4 / NC: Corrupt artifact SHA256 in manifest.json is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_corrupt_hash")
        out_dir = os.path.join(self.repo_dir, "corrupt_hash_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_corrupt_hash",
            "OUTPUT_DIR": out_dir,
            "INJECT_CORRUPT_MANIFEST": "1"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("hash mismatch", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_c4_mismatched_run_id_refused_exit_4(self):
        """
        C4 / NC: run_status.json or manifest.json with mismatched run_id is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("declared_run_id")
        out_dir = os.path.join(self.repo_dir, "mismatched_id_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "different_env_run_id",
            "OUTPUT_DIR": out_dir,
            "INJECT_RUN_ID": "unrelated_different_run_id"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("mismatch", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))


    def test_d3_non_object_artifact_entry_refused_exit_4(self):
        """
        D3 / NC: Manifest with non-object artifact entry (e.g. None/null) is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_non_obj_art")
        out_dir = os.path.join(self.repo_dir, "non_obj_art_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_non_obj_art",
            "OUTPUT_DIR": out_dir,
            "INJECT_NON_OBJECT_ARTIFACT": "1"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("is not an object", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_d3_mixed_valid_and_malformed_artifact_entries_refused_exit_4(self):
        """
        D3 / NC: Manifest with mixed valid and malformed entries is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_mixed_mal_art")
        out_dir = os.path.join(self.repo_dir, "mixed_mal_art_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_mixed_mal_art",
            "OUTPUT_DIR": out_dir,
            "INJECT_MIXED_MALFORMED_ARTIFACT": "1"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("is not an object", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_d3_all_entries_malformed_refused_exit_4(self):
        """
        D3 / NC: Manifest with all malformed entries is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_all_mal_art")
        out_dir = os.path.join(self.repo_dir, "all_mal_art_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_all_mal_art",
            "OUTPUT_DIR": out_dir,
            "INJECT_ALL_MALFORMED_ARTIFACTS": "1"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("is not an object", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_d3_artifact_path_traversal_refused_exit_4(self):
        """
        D3 / NC: Manifest artifact path with '..' directory traversal is rejected with exit 4, clean_scratch refused.
        """
        manifest_path = self._create_dummy_manifest("test_run_traversal_art")
        out_dir = os.path.join(self.repo_dir, "traversal_art_dir")

        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_traversal_art",
            "OUTPUT_DIR": out_dir,
            "INJECT_TRAVERSAL_ARTIFACT": "1"
        })
        self.assertEqual(proc.returncode, 4)
        self.assertIn("contains traversal or is absolute", proc.stderr)
        clean_log = os.path.join(self.scratch_dir, "clean_call.log")
        self.assertFalse(os.path.exists(clean_log))

    def test_stubs_are_executable_and_resolved_from_fixture_bin(self):
        """
        A3 / A4 / A6: Preflight check verifies that created stubs in bin_dir and copied run_merge.pbs
        have executable permissions and resolve to fixture bin_dir.
        Deliberately breaking a required stub in a disposable fixture proves early preflight refusal
        before any fallback command in PATH can be reached, leaving outside fallback sentinels untouched.
        Normal execution verifies that module, pip, clean_scratch, python3, and the activated venv python
        are all selected and executed from fixture stubs.
        """
        stub_names = ["module", "pip", "clean_scratch", "python3", "python"]
        for s in stub_names:
            sp = os.path.join(self.bin_dir, s)
            self.assertTrue(os.path.exists(sp), f"Stub {s} missing from bin_dir")
            if os.name == "posix":
                mode = os.stat(sp).st_mode
                self.assertTrue(bool(mode & 0o111), f"Stub {s} missing executable permission bits: {oct(mode)}")
                self.assertTrue(os.access(sp, os.X_OK), f"Stub {s} is not executable via os.access")
            else:
                res = subprocess.run(
                    [BASH_EXE, "-c", f"[ -x '{sp.replace(chr(92), '/')}' ] && echo EXEC || echo NOEXEC"],
                    env=self._sanitize_env(),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                self.assertEqual(res.stdout.strip(), "EXEC", f"Stub {s} is not executable in bash")

            # Check that command resolution with fixture PATH resolves strictly to bin_dir
            env_chk = self._sanitize_env({"PATH": f"{self.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"})
            res_which = subprocess.run(
                [BASH_EXE, "-c", f"command -v {s}"],
                env=env_chk,
                capture_output=True,
                text=True,
                timeout=5
            )
            resolved_path = res_which.stdout.strip()
            self.assertEqual(os.path.basename(resolved_path), s, f"Stub {s} basename mismatch in command -v: {resolved_path}")
            self.assertIn(os.path.basename(self.test_root).lower(), resolved_path.lower(),
                          f"Stub {s} did not resolve to fixture bin_dir ({resolved_path} vs {self.bin_dir})")

        # Verify copied run_merge.pbs has executable bit set
        if os.name == "posix":
            pbs_mode = os.stat(self.pbs_script).st_mode
            self.assertTrue(bool(pbs_mode & 0o111), f"PBS script missing executable bits: {oct(pbs_mode)}")
            self.assertTrue(os.access(self.pbs_script, os.X_OK), "PBS script not executable via os.access")
        else:
            res = subprocess.run(
                [BASH_EXE, "-c", f"[ -x '{self.pbs_script.replace(chr(92), '/')}' ] && echo EXEC || echo NOEXEC"],
                env=self._sanitize_env(),
                capture_output=True,
                text=True,
                timeout=5
            )
            self.assertEqual(res.stdout.strip(), "EXEC", "PBS script is not executable in bash")

        # Negative control: Deliberately break a required stub's permissions in fixture
        # and set up an outside fallback sentinel directory in PATH. Prove fail-closed early refusal
        # occurs before any fallback command in PATH is reached, keeping the fallback sentinel untouched.
        fallback_dir = os.path.join(self.test_root, "outside_fallback_bin")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_pip = os.path.join(fallback_dir, "pip")
        fallback_sentinel = os.path.join(self.scratch_dir, ".outside_fallback_hit")
        with open(fallback_pip, "w", encoding="utf-8", newline="\n") as f:
            f.write(f'#!/bin/bash\necho FALLBACK_HIT >> "{fallback_sentinel.replace(chr(92), "/")}"\nexit 0\n')
        os.chmod(fallback_pip, 0o755)

        pip_stub = os.path.join(self.bin_dir, "pip")
        with open(pip_stub, "r", encoding="utf-8") as f:
            orig_pip_content = f.read()
        # To make stub non-executable portably across POSIX and Git-Bash on Windows:
        # On POSIX: os.chmod 0o644 strips 0o111 execute bits.
        # On Git-Bash (Windows): Git-Bash determines executable status by checking for '#!' shebang.
        # Stripping the shebang makes '[ -x stub ]' evaluate to false.
        with open(pip_stub, "w", encoding="utf-8", newline="\n") as f:
            f.write("echo 'broken non-executable stub plain text'\nexit 1\n")
        os.chmod(pip_stub, 0o644)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self._run_pbs({
                    "PATH": f"{self.bin_dir}{os.pathsep}{fallback_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                })
            self.assertIn("not executable", str(ctx.exception))
            # Assert outside fallback sentinel was NEVER touched
            self.assertFalse(os.path.exists(fallback_sentinel), "Outside fallback sentinel was executed despite broken stub!")
        finally:
            with open(pip_stub, "w", encoding="utf-8", newline="\n") as f:
                f.write(orig_pip_content)
            os.chmod(pip_stub, 0o755)
            shutil.rmtree(fallback_dir, ignore_errors=True)

        # Execute normal run to verify stubs in bin_dir are actually resolved and executed
        manifest_path = self._create_dummy_manifest("test_run_stub_resolution")
        out_dir = os.path.join(self.repo_dir, "stub_res_out")
        proc = self._run_pbs({
            "MANIFEST_PATH": manifest_path,
            "RUN_ID": "test_run_stub_resolution",
            "OUTPUT_DIR": out_dir,
        })
        self.assertEqual(proc.returncode, 0, f"PBS run failed: {proc.stderr}")

        # Prove all required stubs were executed from bin_dir and venv
        self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, ".module_stub_called")), "module stub was not executed")
        self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, ".pip_stub_called")), "pip stub was not executed")
        self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, ".venv_pip_called")), "venv pip stub was not executed")
        self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, "clean_call.log")), "clean_scratch stub was not executed")
        self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, ".python3_stub_called")), "python3 stub was not executed")
        self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, ".venv_python_called")), "venv python was not executed")

    def test_environment_sanitization_isolates_hostile_bash_env_and_functions(self):
        """
        A4 / A6: Ambient hostile BASH_ENV and exported shell functions (BASH_FUNC_module%%)
        present in outer os.environ are purged by _sanitize_env prior to ANY child shell invocation,
        including Windows preflight stub checks, verification commands, and wrapper execution.
        Also includes a negative falsification control proving that un-sanitized execution
        DOES trigger the hostile trap.
        """
        manifest_path = self._create_dummy_manifest("test_run_hostile_env")
        out_dir = os.path.join(self.repo_dir, "hostile_env_out")

        hostile_bash_env_script = os.path.join(self.test_root, "hostile_bash_env.sh")
        hostile_sentinel = os.path.join(self.scratch_dir, ".hostile_bash_env_hit")
        with open(hostile_bash_env_script, "w", encoding="utf-8", newline="\n") as f:
            f.write(f'echo HOSTILE_BASH_ENV >> "{hostile_sentinel.replace(chr(92), "/")}"\n')
        os.chmod(hostile_bash_env_script, 0o755)

        hostile_func_sentinel = os.path.join(self.scratch_dir, ".hostile_func_hit")
        hostile_func_def = "() { echo HOSTILE_FUNC >> " + hostile_func_sentinel.replace("\\", "/") + "; }"

        # 1. Negative falsification control: demonstrate that without sanitization,
        # ambient BASH_ENV executes on child shell invocations
        raw_env = os.environ.copy()
        raw_env["BASH_ENV"] = hostile_bash_env_script
        subprocess.run([BASH_EXE, "-c", "exit 0"], env=raw_env, capture_output=True, timeout=5)
        self.assertTrue(os.path.exists(hostile_sentinel), "Falsification failure: ambient BASH_ENV did not trigger outside test harness!")
        os.remove(hostile_sentinel)

        # 2. Isolated execution: place hostile controls in actual outer os.environ
        orig_bash_env = os.environ.get("BASH_ENV")
        orig_bash_func = os.environ.get("BASH_FUNC_module%%")
        try:
            os.environ["BASH_ENV"] = hostile_bash_env_script
            os.environ["BASH_FUNC_module%%"] = hostile_func_def

            # _run_pbs must sanitize prior to preflight checks and wrapper execution
            proc = self._run_pbs({
                "MANIFEST_PATH": manifest_path,
                "RUN_ID": "test_run_hostile_env",
                "OUTPUT_DIR": out_dir,
            })
            self.assertEqual(proc.returncode, 0, f"Run failed: {proc.stderr}")

            # Assert hostile traps were NOT executed either during preflight or wrapper execution
            self.assertFalse(os.path.exists(hostile_sentinel), "Hostile BASH_ENV was executed!")
            self.assertFalse(os.path.exists(hostile_func_sentinel), "Hostile exported shell function was executed!")

            # Assert our own fixture stubs were executed
            self.assertTrue(os.path.exists(os.path.join(self.scratch_dir, ".module_stub_called")),
                             "Fixture module stub was not executed")
        finally:
            if orig_bash_env is not None:
                os.environ["BASH_ENV"] = orig_bash_env
            else:
                os.environ.pop("BASH_ENV", None)
            if orig_bash_func is not None:
                os.environ["BASH_FUNC_module%%"] = orig_bash_func
            else:
                os.environ.pop("BASH_FUNC_module%%", None)
            if os.path.exists(hostile_sentinel):
                os.remove(hostile_sentinel)
            if os.path.exists(hostile_func_sentinel):
                os.remove(hostile_func_sentinel)

    def test_timeout_cleanup_terminates_owned_process_tree(self):
        """
        A5 / A6: Deliberately hanging child causes TimeoutExpired; _run_pbs cleanup ensures
        the fixture-owned process tree is terminated without leaking running background processes,
        while an external sentinel process outside the fixture remains intact.
        Also tests missing-marker cleanup and resistant descendant teardown with strict postconditions.
        """
        manifest_path = self._create_dummy_manifest("test_run_timeout_hanging")
        out_dir = os.path.join(self.repo_dir, "timeout_hanging_out")
        clean_env = self._sanitize_env()

        # Spawn an external sentinel process outside the fixture in its own process group
        if os.name == "posix":
            sentinel_proc = subprocess.Popen(
                [BASH_EXE, "-c", "while true; do sleep 0.2; done"],
                env=clean_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        else:
            sentinel_proc = subprocess.Popen(
                [BASH_EXE, "-c", "while true; do sleep 0.2; done"],
                env=clean_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

        try:
            # Case 1: Normal hanging child with marker written
            with self.assertRaises(subprocess.TimeoutExpired) as ctx:
                self._run_pbs({
                    "MANIFEST_PATH": manifest_path,
                    "RUN_ID": "test_run_timeout_hanging",
                    "OUTPUT_DIR": out_dir,
                    "SLEEP_IN_PYTHON": "30",
                }, timeout=4)

            self.assertIn("TimeoutExpired", type(ctx.exception).__name__)

            # Assert child actually wrote marker
            marker = os.path.join(self.scratch_dir, ".py_started")
            self.assertTrue(os.path.exists(marker), "Hanging child failed to create .py_started marker")
            with open(marker, "r") as f:
                py_pid = f.read().strip()
            self.assertTrue(py_pid, "Marker was empty")

            # Check that after timeout cleanup, PID is dead
            check = subprocess.run(
                [BASH_EXE, "-c", f"kill -0 {py_pid} 2>/dev/null && echo ALIVE || echo DEAD"],
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=5
            )
            self.assertEqual(check.stdout.strip(), "DEAD", f"Hanging python PID {py_pid} survived timeout cleanup!")

            # Verify external sentinel outside fixture remained completely intact
            self.assertIsNone(sentinel_proc.poll(), "External sentinel process was killed by cleanup!")

            # Case 2: Hanging child without marker (exercises missing-marker cleanup via group/taskkill)
            os.remove(marker)
            with self.assertRaises(subprocess.TimeoutExpired) as ctx2:
                self._run_pbs({
                    "MANIFEST_PATH": manifest_path,
                    "RUN_ID": "test_run_timeout_hanging_nomark",
                    "OUTPUT_DIR": out_dir,
                    "SLEEP_IN_PYTHON": "30",
                    "SLEEP_WITHOUT_MARKER": "1"
                }, timeout=3)
            # Postconditions for missing marker:
            self.assertFalse(os.path.exists(marker), ".py_started should not exist in nomark case")
            unadv_file = os.path.join(self.scratch_dir, ".unadvertised_child_pid")
            self.assertTrue(os.path.exists(unadv_file), "Unadvertised child failed to record PID")
            with open(unadv_file, "r") as f:
                unadv_pid = f.read().strip()
            self.assertTrue(unadv_pid.isdigit(), "Unadvertised child PID is not a digit")
            unadv_dead = False
            for _ in range(20):
                chk_unadv = subprocess.run(
                    [BASH_EXE, "-c", f"kill -0 {unadv_pid} 2>/dev/null && echo ALIVE || echo DEAD"],
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if chk_unadv.stdout.strip() == "DEAD":
                    unadv_dead = True
                    break
                time.sleep(0.05)
            self.assertTrue(unadv_dead, f"Unmarked child PID {unadv_pid} survived group cleanup!")
            exc_proc = getattr(ctx2.exception, "proc", None)
            if exc_proc is not None:
                self.assertIsNotNone(exc_proc.poll(), "Parent process was not terminated by cleanup")
            self.assertIsNone(sentinel_proc.poll(), "External sentinel was killed during nomark cleanup!")

            # Case 3: Resistant descendant timeout
            with self.assertRaises(subprocess.TimeoutExpired) as ctx3:
                self._run_pbs({
                    "MANIFEST_PATH": manifest_path,
                    "RUN_ID": "test_run_timeout_resistant",
                    "OUTPUT_DIR": out_dir,
                    "SPAWN_RESISTANT_CHILD": "1",
                    "SLEEP_IN_PYTHON": "30"
                }, timeout=3)
            gc_file = os.path.join(self.scratch_dir, ".grandchild_pid")
            self.assertTrue(os.path.exists(gc_file), "Expected grandchild marker .grandchild_pid was never created")
            with open(gc_file, "r") as f:
                gc_pid = f.read().strip()
            self.assertTrue(gc_pid.isdigit(), "Grandchild PID is not a digit")
            chk_gc = subprocess.run(
                [BASH_EXE, "-c", f"kill -0 {gc_pid} 2>/dev/null && echo ALIVE || echo DEAD"],
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=5
            )
            self.assertEqual(chk_gc.stdout.strip(), "DEAD", f"Resistant grandchild {gc_pid} survived timeout cleanup!")

            self.assertIsNone(sentinel_proc.poll(), "External sentinel was killed during resistant cleanup!")
        finally:
            if sentinel_proc.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(sentinel_proc.pid, getattr(signal, "SIGKILL", 9))
                    except Exception:
                        pass
                else:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(sentinel_proc.pid)], env=clean_env, capture_output=True, check=False, timeout=5)
                try:
                    sentinel_proc.kill()
                    sentinel_proc.wait(timeout=2)
                except Exception:
                    pass

    def test_cleanup_process_tree_detects_surviving_process_fault(self):
        """
        A5 / A6: Negative control for cleanup verification (isolated simulated processes).
        Demonstrates that if an owned process group, marker PID, or parent process survives
        cleanup escalation, _cleanup_process_tree raises RuntimeError reporting the surviving entity.
        Also verifies passing controls when processes are cleanly dead.

        All simulated-process branches are hermetically isolated via mock boundaries:
        external OS / process enumeration / kill commands (subprocess.run, os.killpg, os.kill)
        are intercepted in memory, recorded, and verified never to dispatch fictitious PIDs to the host OS.
        Additionally verifies that removal of the mock barrier would be detected by a safety guard
        without sending commands to the OS.
        """
        import unittest.mock

        FICTITIOUS_PIDS = {"77777", "88888", "99999", "66666"}
        captured_subprocess_cmds = []
        captured_killpg_calls = []
        captured_kill_calls = []
        simulated_state = {
            "marker_alive": True,
            "ps_lines": [],
            "posix_ps_lines": ["PID PPID PGID SID STAT COMMAND"]
        }

        def mock_subprocess_run(args, *a, **kw):
            cmd_list = list(args) if isinstance(args, (list, tuple)) else [str(args)]
            cmd_str = " ".join(str(x) for x in cmd_list)
            captured_subprocess_cmds.append(cmd_list)

            # Security assertion: ensure fictitious PIDs and ps inspection are handled strictly in memory
            if any(fpid in cmd_str for fpid in FICTITIOUS_PIDS) or "ps -ef" in cmd_str or (cmd_list[0] == "ps" and "-eo" in cmd_list):
                if "kill -0" in cmd_str:
                    out = "ALIVE" if simulated_state.get("marker_alive", True) else "DEAD"
                    return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")
                elif "ps -ef" in cmd_str:
                    lines = simulated_state.get("ps_lines", [])
                    return subprocess.CompletedProcess(args, 0, stdout="\n".join(lines), stderr="")
                elif cmd_list[0] == "ps" and "-eo" in cmd_list:
                    lines = simulated_state.get("posix_ps_lines", ["PID PPID PGID SID STAT COMMAND"])
                    return subprocess.CompletedProcess(args, 0, stdout="\n".join(lines), stderr="")
                else:
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            raise AssertionError(f"Unexpected unmocked subprocess command in simulated fault test: {cmd_list}")

        def mock_killpg(pgid, sig):
            captured_killpg_calls.append((pgid, sig))
            if pgid == 99999:
                if simulated_state.get("killpg_perm_error", False):
                    raise PermissionError(f"Operation not permitted on pgid {pgid}")
                if simulated_state.get("model_zombie_reap", False):
                    if simulated_state.get("zombie_reaped", False):
                        raise ProcessLookupError(f"No such process group {pgid}")
                    return None
                if simulated_state.get("group_dead", False):
                    raise ProcessLookupError(f"No such process group {pgid}")
                return None  # survives
            raise AssertionError(f"Unexpected unmocked killpg call with pgid: {pgid}")

        def mock_kill(pid, sig):
            captured_kill_calls.append((pid, sig))
            if pid == 88888:
                if simulated_state.get("kill_perm_error", False):
                    raise PermissionError(f"Operation not permitted on pid {pid}")
                if simulated_state.get("pid_dead", False):
                    raise ProcessLookupError(f"No such process {pid}")
                return None  # survives
            raise AssertionError(f"Unexpected unmocked kill call with pid: {pid}")

        # Safety barrier detection control:
        # Prove that if an unmocked command containing a fictitious PID attempts to reach the OS,
        # the safety barrier refuses execution before OS dispatch
        def barrier_guard(args, *a, **kw):
            cmd_str = " ".join(str(x) for x in args) if isinstance(args, (list, tuple)) else str(args)
            for fpid in FICTITIOUS_PIDS:
                if fpid in cmd_str:
                    raise PermissionError(f"Safety barrier prevented fictitious PID {fpid} dispatch to host OS")
            return subprocess.CompletedProcess(args, 0)

        with self.assertRaises(PermissionError) as barrier_ctx:
            barrier_guard(["taskkill", "/F", "/T", "/PID", "77777"])
        self.assertIn("Safety barrier prevented fictitious PID 77777 dispatch to host OS", str(barrier_ctx.exception))

        fake_proc = unittest.mock.MagicMock()
        fake_proc.pid = 99999
        fake_proc._owned_pgid = 99999
        fake_proc.poll.return_value = 0  # parent exited

        with unittest.mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            with unittest.mock.patch("os.killpg", side_effect=mock_killpg, create=True):
                with unittest.mock.patch("os.kill", side_effect=mock_kill, create=True):
                    with unittest.mock.patch("time.sleep", return_value=None):

                        # 1a. POSIX delayed parent reap / zombie observation ordering:
                        # Counterexample: old ordering (probe before wait) falsely fails on unreaped zombie
                        simulated_state["model_zombie_reap"] = True
                        simulated_state["zombie_reaped"] = False
                        zombie_proc = unittest.mock.MagicMock()
                        zombie_proc.pid = 99999
                        zombie_proc._owned_pgid = 99999
                        def fake_reap(timeout=None):
                            simulated_state["zombie_reaped"] = True
                        zombie_proc.wait = fake_reap
                        zombie_proc.poll.side_effect = lambda: 0 if simulated_state["zombie_reaped"] else None

                        old_order_surviving = []
                        for _ in range(20):
                            try:
                                mock_killpg(99999, 0)
                            except ProcessLookupError:
                                break
                        else:
                            old_order_surviving.append("process group 99999")
                        self.assertEqual(old_order_surviving, ["process group 99999"], "Old probe-before-wait order failed to reproduce zombie failure")

                        # 1b. Fixed resolution: new ordering reaps direct child first, clearing zombie from group
                        with unittest.mock.patch("os.name", "posix"):
                            self._cleanup_process_tree(zombie_proc)
                        self.assertTrue(simulated_state["zombie_reaped"], "Direct child was not reaped before process group probing")
                        simulated_state["model_zombie_reap"] = False

                        # 2. POSIX real resistant living descendant: strict failure must remain
                        simulated_state["group_dead"] = False
                        simulated_state["posix_ps_lines"] = [
                            "PID PPID PGID SID STAT COMMAND",
                            "88888 99999 99999 99999 S sleep 30"
                        ]
                        living_proc = unittest.mock.MagicMock()
                        living_proc.pid = 99999
                        living_proc._owned_pgid = 99999
                        living_proc.poll.return_value = 0
                        with unittest.mock.patch("os.name", "posix"):
                            with self.assertRaises(RuntimeError) as ctx_living:
                                self._cleanup_process_tree(living_proc)
                            self.assertIn("Process cleanup failed", str(ctx_living.exception))
                            self.assertIn("process group 99999", str(ctx_living.exception))
                            self.assertIn("active living descendants", str(ctx_living.exception))
                            self.assertIn("PID 88888", str(ctx_living.exception))
                            self.assertIn("stat=S", str(ctx_living.exception))

                        # 3. POSIX parent exit with live child: cleanup still covers owned PGID
                        simulated_state["group_dead"] = False
                        simulated_state["posix_ps_lines"] = [
                            "PID PPID PGID SID STAT COMMAND",
                            "88888 1 99999 99999 R python3 worker.py"
                        ]
                        exited_parent = unittest.mock.MagicMock()
                        exited_parent.pid = 77777
                        exited_parent._owned_pgid = 99999
                        exited_parent.poll.return_value = 0
                        with unittest.mock.patch("os.name", "posix"):
                            with self.assertRaises(RuntimeError) as ctx_exited:
                                self._cleanup_process_tree(exited_parent)
                            self.assertIn("Process cleanup failed", str(ctx_exited.exception))
                            self.assertIn("process group 99999", str(ctx_exited.exception))
                            self.assertIn("PID 88888", str(ctx_exited.exception))
                            self.assertIn("stat=R", str(ctx_exited.exception))

                        # 4a. POSIX PermissionError: must not treat as dead
                        simulated_state["killpg_perm_error"] = True
                        with unittest.mock.patch("os.name", "posix"):
                            with self.assertRaises(RuntimeError) as ctx_perm:
                                self._cleanup_process_tree(fake_proc)
                            self.assertIn("Process cleanup failed", str(ctx_perm.exception))
                            self.assertIn("PermissionError", str(ctx_perm.exception))
                        simulated_state["killpg_perm_error"] = False

                        # 4b. POSIX unknown active members: killpg(0) succeeds but snapshot is empty
                        simulated_state["group_dead"] = False
                        simulated_state["posix_ps_lines"] = ["PID PPID PGID SID STAT COMMAND"]
                        with unittest.mock.patch("os.name", "posix"):
                            with self.assertRaises(RuntimeError) as ctx_unknown:
                                self._cleanup_process_tree(fake_proc)
                            self.assertIn("Process cleanup failed", str(ctx_unknown.exception))
                            self.assertIn("unknown active members", str(ctx_unknown.exception))

                        # 4c. POSIX reparented unreaped zombie descendant failure
                        simulated_state["group_dead"] = False
                        simulated_state["posix_ps_lines"] = [
                            "PID PPID PGID SID STAT COMMAND",
                            "88888 1 99999 99999 Z [sleep] <defunct>"
                        ]
                        with unittest.mock.patch("os.name", "posix"):
                            with self.assertRaises(RuntimeError) as ctx_zomb:
                                self._cleanup_process_tree(fake_proc)
                            self.assertIn("Process cleanup failed", str(ctx_zomb.exception))
                            self.assertIn("unreaped zombie descendants", str(ctx_zomb.exception))
                            self.assertIn("PID 88888", str(ctx_zomb.exception))

                        # 5. POSIX surviving marker PID fault control
                        marker = os.path.join(self.scratch_dir, ".py_started")
                        with open(marker, "w") as f:
                            f.write("88888")
                        try:
                            simulated_state["group_dead"] = True
                            simulated_state["pid_dead"] = False
                            with unittest.mock.patch("os.name", "posix"):
                                with self.assertRaises(RuntimeError) as ctx:
                                    self._cleanup_process_tree(fake_proc)
                                self.assertIn("Process cleanup failed", str(ctx.exception))
                                self.assertIn("PID 88888", str(ctx.exception))
                            self.assertTrue(any(call[0] == 88888 for call in captured_kill_calls), "os.kill was not called for PID 88888")

                            # 6. POSIX positive control: cleanly dead group and PID pass without error
                            simulated_state["pid_dead"] = True
                            with unittest.mock.patch("os.name", "posix"):
                                self._cleanup_process_tree(fake_proc)
                        finally:
                            if os.path.exists(marker):
                                os.remove(marker)

                        # 7. Windows surviving parent process fault control
                        surviving_proc = unittest.mock.MagicMock()
                        surviving_proc.pid = 77777
                        surviving_proc._owned_pgid = 77777
                        surviving_proc.poll.return_value = None  # Parent remains alive
                        with unittest.mock.patch("os.name", "nt"):
                            with self.assertRaises(RuntimeError) as ctx_win:
                                self._cleanup_process_tree(surviving_proc)
                            self.assertIn("Process cleanup failed", str(ctx_win.exception))
                            self.assertIn("parent process PID 77777", str(ctx_win.exception))
                        # Explicit command capture assertion: taskkill /F /T /PID 77777 was intercepted
                        tk_calls = [c for c in captured_subprocess_cmds if "77777" in " ".join(str(x) for x in c)]
                        self.assertTrue(len(tk_calls) > 0, "taskkill command for PID 77777 was not captured")
                        self.assertTrue(any("taskkill" in str(c[0]).lower() and "/PID" in c for c in tk_calls), "Malformed taskkill call")

                        # 8. Windows positive control: parent exited cleanly (poll() == 0) passes without error
                        clean_proc = unittest.mock.MagicMock()
                        clean_proc.pid = 77777
                        clean_proc._owned_pgid = 77777
                        clean_proc.poll.return_value = 0  # Parent exited
                        with unittest.mock.patch("os.name", "nt"):
                            self._cleanup_process_tree(clean_proc)

                        # 9. Windows surviving marker PID fault control
                        win_marker = os.path.join(self.scratch_dir, ".py_started")
                        with open(win_marker, "w") as f:
                            f.write("66666")
                        try:
                            simulated_state["marker_alive"] = True
                            with unittest.mock.patch("os.name", "nt"):
                                with self.assertRaises(RuntimeError) as ctx_win_m:
                                    self._cleanup_process_tree(clean_proc)
                                self.assertIn("Process cleanup failed", str(ctx_win_m.exception))
                                self.assertIn("PID 66666", str(ctx_win_m.exception))
                            # Assert taskkill and kill probes were captured in memory
                            m_calls = [c for c in captured_subprocess_cmds if "66666" in " ".join(str(x) for x in c)]
                            self.assertTrue(len(m_calls) > 0, "No commands captured for marker PID 66666")

                            # 10. Windows marker PID dead positive control: returns DEAD, passes cleanly
                            simulated_state["marker_alive"] = False
                            with unittest.mock.patch("os.name", "nt"):
                                self._cleanup_process_tree(clean_proc)
                        finally:
                            if os.path.exists(win_marker):
                                os.remove(win_marker)

        # Final verification: confirm all captured subprocess commands targeted strictly simulated PIDs / safe commands
        self.assertTrue(len(captured_subprocess_cmds) > 0, "No subprocess commands were captured")
        for cmd in captured_subprocess_cmds:
            cmd_str = " ".join(str(x) for x in cmd)
            self.assertTrue(
                any(fpid in cmd_str for fpid in FICTITIOUS_PIDS) or "ps -ef" in cmd_str or (cmd[0] == "ps" and "-eo" in cmd),
                f"Command escaped fictitious PID boundary: {cmd}"
            )


if __name__ == "__main__":
    unittest.main()


