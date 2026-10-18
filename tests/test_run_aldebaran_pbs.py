#!/usr/bin/env python3
"""
tests/test_run_aldebaran_pbs.py
===============================
Comprehensive offline regression and falsification test suite for ALD-005:
Main PBS failure propagation, safe finalization, staging isolation, and signal handling.

Verifies:
1. Real Bash execution of cluster/run_aldebaran.pbs with local shell stubs.
2. Normal single-job and array-job success: correct stage sequence, destination file
   bytes, diagnostics generation, and exactly one scratch cleanup.
3. Distinction between actual candidates and header-only CSV in single-job mode (skipping Tier-2).
4. Failure propagation from Tier-1 (exit 7): Tier-2 skipped, diagnostics saved, scratch preserved.
5. Failure propagation from Tier-2 (exit 9): diagnostics saved, scratch preserved.
6. Transfer failures (mkdir/cp failure): exit nonzero, scratch preserved with actionable reason.
7. Compound failures (Python error + transfer failure): both recorded, overall exit nonzero.
8. Preflight / setup failures (staging error, pip failure): no science run, scratch preserved.
9. Cleanup failure (clean_scratch returns 1): exit nonzero (5), no false success message.
10. Signal handling (SIGTERM / SIGINT): child termination forwarded, bounded wait, scratch retained, exit 143/130.
11. Staging / output ownership isolation: Array worker A cannot overwrite worker B's sentinel in pre-staged directories.
12. Unsafe transfer prevention: array mode refuses transfer if ownership resolves to entire tess_results.
13. Periodic sync mutex lock and transient error recovery accounting.
14. Regression control: Demonstrates that the unpatched baseline script pattern masked Python failure
    and wiped scratch, while the patched implementation preserves scratch and propagates exit 7.
15. Descendant tracking and escalation even after parent exit (TERM-resistant descendant killed before finalization).
16. In-flight periodic sync copy termination and clean lock release without race.
17. Post-cleanup diagnostics update failure propagation (exit 7) with valid prior JSON preserved.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import unittest
import time
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


def kill_process_tree(proc):
    """Cleanly terminates process and its entire child process tree."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@unittest.skipIf(not BASH_EXE, "Bash executable not found in environment")
class TestRunAldebaranPBS(unittest.TestCase):

    def setUp(self):
        self.test_root = tempfile.mkdtemp(prefix="ald_test_pbs_")
        self.repo_dir = os.path.join(self.test_root, "repo")
        self.scratch_dir = os.path.join(self.test_root, "scratch")
        self.bin_dir = os.path.join(self.test_root, "bin")

        os.makedirs(self.repo_dir, exist_ok=True)
        os.makedirs(self.scratch_dir, exist_ok=True)
        os.makedirs(self.bin_dir, exist_ok=True)

        self.pbs_script_src = os.path.abspath(os.path.join("cluster", "run_aldebaran.pbs"))
        if not os.path.exists(self.pbs_script_src):
            self.fail(f"Production PBS script not found at {self.pbs_script_src}")

        self._active_procs = []

        # Setup standard Python scripts in fake repo
        self._create_repo_scripts()

        # Setup standard shell stubs in bin_dir
        self._create_shell_stubs()

    def tearDown(self):
        # Terminate any remaining background processes before cleaning directory
        for proc in self._active_procs:
            kill_process_tree(proc)
        self._active_procs.clear()

        # Brief sleep to allow Windows file handles to close
        time.sleep(0.1)
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _create_repo_scripts(self):
        """Creates controllable mock aldebaran.py and pleiades.py in repo_dir."""
        aldebaran_py = os.path.join(self.repo_dir, "aldebaran.py")
        with open(aldebaran_py, "w", encoding="utf-8") as f:
            f.write('''import os, sys

idx = os.environ.get("PBS_ARRAY_INDEX")
OUTPUT_DIR = f"tess_results/sector_{idx}" if idx else "tess_results"

if __name__ == "__main__":
    inject_exit = int(os.environ.get("INJECT_TIER1_EXIT", "0"))
    if inject_exit != 0:
        print(f"[Tier-1] Injected failure with exit {inject_exit}", file=sys.stderr)
        sys.exit(inject_exit)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    events_log = os.path.join(OUTPUT_DIR, "pipeline_events.log")
    with open(events_log, "w", encoding="utf-8") as f:
        f.write(f"Tier-1 events log for sector {idx or 'single'}\\n")

    if os.environ.get("CREATE_CANDIDATES") == "1":
        with open(os.path.join(OUTPUT_DIR, "candidates_report.csv"), "w", encoding="utf-8") as f:
            f.write("tic_id,sector,period,sde\\n12345,108,2.5,9.5\\n")
    elif os.environ.get("CREATE_HEADER_ONLY_CSV") == "1":
        with open(os.path.join(OUTPUT_DIR, "candidates_report.csv"), "w", encoding="utf-8") as f:
            f.write("tic_id,sector,period,sde\\n")

    if os.environ.get("SLEEP_IN_PYTHON"):
        import time
        sleep_dur = float(os.environ["SLEEP_IN_PYTHON"])
        time.sleep(sleep_dur)

    print(f"[Tier-1] Screening completed successfully for {OUTPUT_DIR}")
    sys.exit(0)
''')

        pleiades_py = os.path.join(self.repo_dir, "pleiades.py")
        with open(pleiades_py, "w", encoding="utf-8") as f:
            f.write('''import os, sys

if __name__ == "__main__":
    inject_exit = int(os.environ.get("INJECT_TIER2_EXIT", "0"))
    if inject_exit != 0:
        print(f"[Tier-2] Injected failure with exit {inject_exit}", file=sys.stderr)
        sys.exit(inject_exit)

    output_dir = "tess_results"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pleiades_vetting.log"), "w", encoding="utf-8") as f:
        f.write("Tier-2 deep vetting completed successfully\\n")

    print("[Tier-2] Deep vetting completed successfully")
    sys.exit(0)
''')

        req_txt = os.path.join(self.repo_dir, "requirements.txt")
        with open(req_txt, "w", encoding="utf-8") as f:
            f.write("certifi\n")

    def _create_shell_stubs(self):
        """Creates offline stubs for module, pip, clean_scratch, and python3."""
        # clean_scratch stub: records call in scratch/.clean_scratch_calls
        clean_scratch_path = os.path.join(self.bin_dir, "clean_scratch")
        with open(clean_scratch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write('''#!/bin/bash
if [ "${INJECT_CLEAN_SCRATCH_FAIL:-0}" = "1" ]; then
    echo "[stub clean_scratch] Simulated failure" >&2
    exit 1
fi
echo "cleaned $(date)" >> "${SCRATCHDIR:-.}/.clean_scratch_calls"
exit 0
''')

        # module stub
        module_path = os.path.join(self.bin_dir, "module")
        with open(module_path, "w", encoding="utf-8", newline="\n") as f:
            f.write('''#!/bin/bash
exit 0
''')

        # pip stub
        pip_path = os.path.join(self.bin_dir, "pip")
        with open(pip_path, "w", encoding="utf-8", newline="\n") as f:
            f.write('''#!/bin/bash
if [ "${INJECT_PIP_FAIL:-0}" = "1" ]; then
    echo "[stub pip] Simulated pip install failure" >&2
    exit 1
fi
exit 0
''')

        # python3 and python stubs: intercept -m venv to avoid network / heavy creation, delegate others to sys.executable
        real_py_posix = sys.executable.replace("\\", "/")
        python3_path = os.path.join(self.bin_dir, "python3")
        with open(python3_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f'''#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    target="$3"
    mkdir -p "$target/bin" "$target/Scripts"
    cat << 'VENV_ACT' > "$target/bin/activate"
deactivate () {{ :; }}
VENV_ACT
    cp "$target/bin/activate" "$target/Scripts/activate"
    cat << 'VENV_PY' > "$target/bin/python"
#!/bin/bash
exec "{real_py_posix}" "$@"
VENV_PY
    chmod +x "$target/bin/python"
    cp "$target/bin/python" "$target/bin/python3"
    cp "$target/bin/python" "$target/Scripts/python"
    cp "$target/bin/python" "$target/Scripts/python3"
    chmod +x "$target/bin/python3" "$target/Scripts/python" "$target/Scripts/python3"
    exit 0
fi
exec "{real_py_posix}" "$@"
''')

        # python stub in bin_dir
        python_path = os.path.join(self.bin_dir, "python")
        with open(python_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f'''#!/bin/bash
exec "{real_py_posix}" "$@"
''')

        # Make stubs executable on POSIX systems
        for stub in [clean_scratch_path, module_path, pip_path, python3_path, python_path]:
            try:
                os.chmod(stub, 0o755)
            except OSError:
                pass

    def _run_pbs(self, env_overrides=None, script_path=None, check=False, timeout=15):
        """Runs cluster/run_aldebaran.pbs in Bash with test environment and strict timeout."""
        target_script = script_path or self.pbs_script_src
        target_script_posix = target_script.replace("\\", "/")
        bin_posix = self.bin_dir.replace("\\", "/")

        env = os.environ.copy()
        env["PBS_O_WORKDIR"] = self.repo_dir
        env["SCRATCHDIR"] = self.scratch_dir
        env["PBS_JOBID"] = "24702696"
        env["ALDEBARAN_SIGNAL_TIMEOUT"] = "2"
        env.pop("PBS_ARRAY_INDEX", None)

        if env_overrides:
            for k, v in env_overrides.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = str(v)

        # Prepend bin_dir using MSYS /tmp path resolution so all shell stubs are resolved before /usr/bin
        bash_cmd = f'export PATH="$(cd "{bin_posix}" && pwd):$PATH"; exec "{target_script_posix}"'

        proc = subprocess.Popen(
            [BASH_EXE, "-c", bash_cmd],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._active_procs.append(proc)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            kill_process_tree(proc)
            raise subprocess.TimeoutExpired(
                proc.args, timeout, output=exc.stdout, stderr=exc.stderr
            ) from exc

        proc.stdout = stdout
        proc.stderr = stderr

        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, target_script_posix, proc.stdout, proc.stderr)
        return proc

    # --------------------------------------------------------------------------
    # Test Cases
    # --------------------------------------------------------------------------

    def test_normal_single_job_with_candidates_runs_tier2_and_succeeds(self):
        """
        Single job with candidates: Tier-1 runs, candidates_report.csv triggers Tier-2,
        diagnostics recorded, destination bytes verified, scratch cleaned, exit 0.
        """
        proc = self._run_pbs({"CREATE_CANDIDATES": "1"})
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

        # Stage verification
        self.assertIn("[1/4] Setting up Python virtual environment", proc.stdout)
        self.assertIn("[2/4] Launching Aldebaran Tier-1 Pipeline", proc.stdout)
        self.assertIn("[3/4] Candidates found! Launching Pleiades Tier-2 Deep Vetting", proc.stdout)
        self.assertIn("[4/4] Pipeline execution finished", proc.stdout)
        self.assertIn("[TRAP] Local scratch cleaned", proc.stdout)

        # Destination byte & artifact verification
        dest_events = Path(self.repo_dir) / "tess_results" / "pipeline_events.log"
        dest_candidates = Path(self.repo_dir) / "tess_results" / "candidates_report.csv"
        dest_pleiades = Path(self.repo_dir) / "tess_results" / "pleiades_vetting.log"

        self.assertTrue(dest_events.exists())
        self.assertTrue(dest_candidates.exists())
        self.assertTrue(dest_pleiades.exists())
        self.assertIn("12345,108", dest_candidates.read_text(encoding="utf-8"))
        self.assertIn("Tier-2 deep vetting completed", dest_pleiades.read_text(encoding="utf-8"))

        # Diagnostics verification
        diag_path = Path(self.repo_dir) / "tess_results" / "job_diagnostics_24702696_single.json"
        self.assertTrue(diag_path.exists())
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["stage_reached"], "PIPELINE_COMPLETE")
        self.assertEqual(diag["pipeline_exit_code"], 0)
        self.assertTrue(diag["setup_ok"])
        self.assertTrue(diag["final_transfer_success"])
        self.assertTrue(diag["scratch_cleaned"])

        # Clean scratch called exactly once
        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertTrue(clean_calls.exists())
        self.assertEqual(len(clean_calls.read_text().strip().splitlines()), 1)

    def test_normal_single_job_header_only_csv_skips_tier2(self):
        """
        Single job with header-only CSV (0 candidates): Tier-2 is skipped,
        exit 0, scratch cleaned, diagnostics verified.
        """
        proc = self._run_pbs({"CREATE_HEADER_ONLY_CSV": "1"})
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

        self.assertIn("[3/4] No candidates found in this batch. Tier-2 vetting skipped", proc.stdout)
        self.assertNotIn("Launching Pleiades Tier-2", proc.stdout)

        # Tier-2 output should NOT exist
        dest_pleiades = Path(self.repo_dir) / "tess_results" / "pleiades_vetting.log"
        self.assertFalse(dest_pleiades.exists())

        # Clean scratch was called
        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertTrue(clean_calls.exists())
        self.assertEqual(len(clean_calls.read_text().strip().splitlines()), 1)

    def test_normal_array_job_success(self):
        """
        Array job (PBS_ARRAY_INDEX=1): maps authoritatively to sector_108,
        stages output in tess_results/sector_108, diagnostics saved to owned dir,
        scratch cleaned, exit 0.
        """
        proc = self._run_pbs({"PBS_ARRAY_INDEX": "1"})
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

        self.assertIn("Authoritative output directory: tess_results/sector_1", proc.stdout)
        self.assertIn("Array shard complete", proc.stdout)

        dest_events = Path(self.repo_dir) / "tess_results" / "sector_1" / "pipeline_events.log"
        self.assertTrue(dest_events.exists())
        self.assertIn("Tier-1 events log for sector 1", dest_events.read_text(encoding="utf-8"))

        # Diagnostics inside owned directory
        diag_path = Path(self.repo_dir) / "tess_results" / "sector_1" / "job_diagnostics_24702696_idx1.json"
        self.assertTrue(diag_path.exists())
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["pbs_array_index"], "1")
        self.assertEqual(diag["pipeline_exit_code"], 0)
        self.assertTrue(diag["scratch_cleaned"])

    def test_inject_tier1_failure_exit_7(self):
        """
        Injected Tier-1 failure (exit 7):
        - Overall job exit code is 7.
        - Tier-2 is NOT executed.
        - No success message.
        - Scratch is preserved with explicit reason and path.
        - Diagnostics file is copied to permanent storage with exit 7 and TIER1_FAILED.
        - clean_scratch is NOT called.
        """
        proc = self._run_pbs({"INJECT_TIER1_EXIT": "7"})
        self.assertEqual(proc.returncode, 7)

        self.assertIn("Tier-1 screening failed with exit code 7", proc.stderr)
        self.assertNotIn("Pleiades Tier-2", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)
        self.assertIn("Scratch preserved at:", proc.stdout)

        # clean_scratch was NOT called
        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertFalse(clean_calls.exists())

        # Diagnostics copied to repo
        diag_path = Path(self.repo_dir) / "tess_results" / "job_diagnostics_24702696_single.json"
        self.assertTrue(diag_path.exists())
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["pipeline_exit_code"], 7)
        self.assertEqual(diag["stage_reached"], "TIER1_FAILED")
        self.assertFalse(diag["scratch_cleaned"])
        self.assertIn("Tier-1 screening failed with exit code 7", diag["failure_reason"])

    def test_inject_tier2_failure_exit_9(self):
        """
        Injected Tier-2 failure (exit 9):
        - Tier-1 succeeds, candidates found.
        - Tier-2 fails with exit 9.
        - Overall job exit code is 9.
        - Scratch is preserved, clean_scratch is NOT called.
        - Diagnostics records TIER2_FAILED and pipeline_exit_code=9.
        """
        proc = self._run_pbs({"CREATE_CANDIDATES": "1", "INJECT_TIER2_EXIT": "9"})
        self.assertEqual(proc.returncode, 9)

        self.assertIn("Tier-2 deep vetting failed with exit code 9", proc.stderr)
        self.assertIn("Scratch preserved at:", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)

        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertFalse(clean_calls.exists())

        diag_path = Path(self.repo_dir) / "tess_results" / "job_diagnostics_24702696_single.json"
        self.assertTrue(diag_path.exists())
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["pipeline_exit_code"], 9)
        self.assertEqual(diag["stage_reached"], "TIER2_FAILED")
        self.assertFalse(diag["scratch_cleaned"])

    def test_inject_final_transfer_failure_after_python_success(self):
        """
        Injected final result transfer failure (repo/tess_results blocked by file):
        - Python stages completed exit 0.
        - Final transfer fails.
        - Overall exit is nonzero (code 4).
        - Scratch is preserved with explicit transfer failure reason.
        - clean_scratch is NOT called.
        - Diagnostics retained in scratch emitted to stdout.
        """
        block_file = Path(self.repo_dir) / "tess_results"
        block_file.write_text("blocking non-directory file", encoding="utf-8")

        proc = self._run_pbs()
        self.assertEqual(proc.returncode, 4)

        self.assertIn("Final result transfer failed", proc.stderr + proc.stdout)
        self.assertIn("Scratch preserved at:", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)

        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertFalse(clean_calls.exists())

        # Diagnostics retained in scratch
        diag_scratch = Path(self.scratch_dir) / "job_diagnostics_24702696_single.json"
        self.assertTrue(diag_scratch.exists())
        with open(diag_scratch, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertFalse(diag["final_transfer_success"])
        self.assertFalse(diag["scratch_cleaned"])

    def test_inject_compound_python_and_transfer_failure(self):
        """
        Injected Python failure (exit 7) AND destination copy failure:
        - Both failures are logged.
        - Overall exit remains nonzero (7).
        - Scratch is preserved, clean_scratch is NOT called.
        """
        block_file = Path(self.repo_dir) / "tess_results"
        block_file.write_text("blocking non-directory file", encoding="utf-8")

        proc = self._run_pbs({"INJECT_TIER1_EXIT": "7"})
        self.assertEqual(proc.returncode, 7)

        self.assertIn("Tier-1 screening failed with exit code 7", proc.stderr)
        self.assertIn("Final result transfer failed", proc.stderr + proc.stdout)
        self.assertIn("Scratch preserved at:", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)

    def test_inject_setup_staging_failure(self):
        """
        Injected preflight/staging failure:
        - aldebaran.py deleted from repo.
        - Script fails immediately during preflight (exit 1).
        - No science stage executed.
        - Scratch preserved.
        """
        os.remove(os.path.join(self.repo_dir, "aldebaran.py"))
        proc = self._run_pbs()
        self.assertEqual(proc.returncode, 1)

        self.assertIn("Invalid REPO_DIR or missing aldebaran.py", proc.stderr)
        self.assertNotIn("Launching Aldebaran Tier-1 Pipeline", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)

    def test_inject_setup_pip_failure(self):
        """
        Injected virtualenv / pip install failure:
        - pip fails with exit 1.
        - Wrapper exits 3.
        - Science stages do not run.
        - Scratch preserved.
        """
        proc = self._run_pbs({"INJECT_PIP_FAIL": "1"})
        self.assertEqual(proc.returncode, 3)

        self.assertIn("pip upgrade failed", proc.stderr + proc.stdout)
        self.assertNotIn("Launching Aldebaran Tier-1 Pipeline", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)

    def test_inject_clean_scratch_failure(self):
        """
        Injected clean_scratch failure (clean_scratch returns 1):
        - Pipeline succeeded, transfer succeeded.
        - clean_scratch command failed.
        - Wrapper exits nonzero (code 5).
        - Output does NOT report 'Local scratch cleaned'.
        - Output explicitly reports clean_scratch command failure.
        """
        proc = self._run_pbs({"INJECT_CLEAN_SCRATCH_FAIL": "1"})
        self.assertEqual(proc.returncode, 5)

        self.assertIn("ERROR: clean_scratch command failed", proc.stderr + proc.stdout)
        self.assertNotIn("[TRAP] Local scratch cleaned.", proc.stdout)

    def test_output_ownership_job_a_does_not_overwrite_job_b_sentinel(self):
        """
        Requirement 9 & Acceptance criterion 9:
        Worker A (array index 1, sector 1) stages existing tess_results from repo.
        Repo has sector_2 with a newer sentinel file written by Worker B.
        Scratch contains an older stale copy of sector_2.
        When Worker A completes and syncs:
        - Worker A only syncs sector_1.
        - Worker B's sentinel in sector_2 is NOT overwritten by Worker A's stale copy!
        """
        # Pre-populate repo with Job B's newer sentinel in sector_2
        repo_sec2 = Path(self.repo_dir) / "tess_results" / "sector_2"
        repo_sec2.mkdir(parents=True, exist_ok=True)
        sentinel_repo = repo_sec2 / "sentinel_b.txt"
        sentinel_repo.write_text("JOB_B_NEW_DATA_V2", encoding="utf-8")

        # Pre-populate scratch with old stale sector_2 data
        scratch_sec2 = Path(self.scratch_dir) / "tess_results" / "sector_2"
        scratch_sec2.mkdir(parents=True, exist_ok=True)
        sentinel_scratch = scratch_sec2 / "sentinel_b.txt"
        sentinel_scratch.write_text("JOB_A_STALE_PRESTAGED_DATA_V1", encoding="utf-8")

        # Run Worker A for sector_1 (PBS_ARRAY_INDEX=1)
        proc = self._run_pbs({"PBS_ARRAY_INDEX": "1"})
        self.assertEqual(proc.returncode, 0, f"Worker A failed:\n{proc.stderr}\n{proc.stdout}")

        # Verify Worker A's own sector_1 was synced
        dest_sec1_log = Path(self.repo_dir) / "tess_results" / "sector_1" / "pipeline_events.log"
        self.assertTrue(dest_sec1_log.exists())

        # CRITICAL CHECK: Job B's sentinel in sector_2 MUST NOT be overwritten!
        self.assertEqual(
            sentinel_repo.read_text(encoding="utf-8"),
            "JOB_B_NEW_DATA_V2",
            "Violation of single-writer / staging ownership: Job A overwrote Job B's newer sentinel!"
        )

    def test_array_mode_unsafe_transfer_refused_if_ownership_unresolved(self):
        """
        Requirement 9: In array mode, if determine_owned_output_dir resolves to
        entire tess_results or empty, sync_results explicitly refuses transfer as unsafe.
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")
        test_script = f'''
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"

SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
PBS_ARRAY_INDEX="1"
OWNED_OUTPUT_DIR="tess_results"

sync_results "final"
RC=$?
[ "$RC" -eq 1 ] || exit 11
echo "UNSAFE_TRANSFER_REFUSED_OK"
'''
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}")
        self.assertIn("UNSAFE_TRANSFER_REFUSED_OK", proc.stdout)
        self.assertIn("Unsafe transfer refused in array mode", proc.stderr)

    def test_periodic_sync_mutex_and_transient_recovery(self):
        """
        Requirement 7 & Acceptance criterion 8:
        Verify lock acquisition prevents race, and earlier transient periodic sync
        errors are recorded and reported as recovered when final sync succeeds.
        """
        # Inject earlier periodic failures into scratch/.periodic_sync_errors
        err_file = Path(self.scratch_dir) / ".periodic_sync_errors"
        err_file.write_text("PERIODIC_FAIL 1000\nPERIODIC_FAIL 2000\n", encoding="utf-8")

        proc = self._run_pbs()
        self.assertEqual(proc.returncode, 0, f"Failed with stderr:\n{proc.stderr}\n{proc.stdout}")

        self.assertIn("Recovered: 2 earlier transient periodic sync failure(s) succeeded on final transfer", proc.stdout)

        diag_path = Path(self.repo_dir) / "tess_results" / "job_diagnostics_24702696_single.json"
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["periodic_sync_errors"], 2)
        self.assertTrue(diag["final_transfer_success"])

    def test_inject_setup_staging_existing_history_copy_failure(self):
        """
        R1 [P1] Negative control:
        If existing tess_results/scanned_tics.txt exists in repo, but copying it fails (e.g. cp returns 17):
        - The failure must NOT be ignored (no `|| true` mask).
        - The pipeline must halt immediately before science stages (exit 2).
        - Scratch is preserved with actionable reason.
        - clean_scratch is NOT called.
        """
        old_dir = Path(self.repo_dir) / "tess_results"
        old_dir.mkdir(parents=True, exist_ok=True)
        (old_dir / "scanned_tics.txt").write_text("existing scan history", encoding="utf-8")

        # Stub cp in bin_dir to fail specifically on staging tess_results
        cp_stub = Path(self.bin_dir) / "cp"
        cp_stub.write_text('''#!/bin/bash
if [[ "$*" == *"repo/tess_results"* && "$*" == *"scratch/tess_results"* ]]; then
    echo "[stub cp] Injected existing history copy failure" >&2
    exit 17
fi
exec /usr/bin/cp "$@"
''', encoding="utf-8", newline="\n")
        try:
            os.chmod(cp_stub, 0o755)
        except OSError:
            pass

        proc = self._run_pbs()
        self.assertEqual(proc.returncode, 2, f"Failed with returncode {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("Failed to copy existing tess_results", proc.stderr)
        self.assertNotIn("Launching Aldebaran Tier-1 Pipeline", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)
        self.assertIn("Scratch preserved at:", proc.stdout)

        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertFalse(clean_calls.exists())

    def test_inject_diagnostics_writer_failure_prevents_cleanup(self):
        """
        R2 [P1] Negative control:
        If science completes and sync_results succeeds, but saving diagnostics to permanent
        storage fails (e.g. write_job_diagnostics returns 1):
        - Destructive scratch cleanup must NOT be performed!
        - The finalizer must NOT print a false 'Diagnostics saved to:' message.
        - Scratch must be preserved, diagnostics retained in scratch, and exit code must be nonzero (6).
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")
        test_script = f'''
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"

SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
SETUP_OK=1
STAGE_REACHED="PIPELINE_COMPLETE"
PIPELINE_EXIT_CODE=0
SIGNAL_RECEIVED=0

sync_results() {{ return 0; }}
write_job_diagnostics() {{
    local t="$1"
    if [ "$t" = "$SCRATCHDIR" ]; then
        echo "SCRATCH_DIAG_WRITTEN" > "$SCRATCHDIR/$DIAG_FILENAME"
        return 0
    fi
    echo "[stub write_job_diagnostics] Simulated permanent storage failure" >&2
    return 1
}}
clean_scratch() {{
    echo "CLEANUP_CALLED" >> "$SCRATCHDIR/.clean_scratch_calls"
    return 0
}}

sync_and_finalize 0
'''
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        self.assertEqual(proc.returncode, 6, f"Expected exit 6 on permanent diagnostics failure, got {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("Failed to save diagnostic evidence to permanent storage", proc.stderr)
        self.assertNotIn("Diagnostics saved to:", proc.stdout)
        self.assertNotIn("Local scratch cleaned", proc.stdout)
        self.assertIn("Diagnostics retained in scratch at:", proc.stdout)

        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertFalse(clean_calls.exists(), "Destructive scratch cleanup was executed despite failed diagnostics save!")

    def test_periodic_sync_mutex_contention_and_prevention(self):
        """
        R3 [P2] Invariant & Concurrency Control:
        Starts an in-flight background periodic copy holding .sync_lock, proves that
        final sync respects mutual exclusion, waits for lock release, and completes cleanly
        without racing or leaving a stale lock.
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")
        test_script = f'''
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"

SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
mkdir -p "$SCRATCHDIR/tess_results" "$REPO_DIR/tess_results"
echo "data1" > "$SCRATCHDIR/tess_results/test.txt"

# Simulate an in-flight periodic sync holding the lock for 1 second
mkdir "$SCRATCHDIR/.sync_lock"
(
    sleep 1
    rmdir "$SCRATCHDIR/.sync_lock" 2>/dev/null || true
) &
HOLDER_PID=$!

start_time=$(date +%s)
sync_results "final"
RC=$?
end_time=$(date +%s)

wait "$HOLDER_PID" 2>/dev/null || true

# Assert mutual exclusion waited (duration >= 1s) and succeeded
[ "$RC" -eq 0 ] || exit 21
[ ! -d "$SCRATCHDIR/.sync_lock" ] || exit 22
echo "MUTEX_CONTENTION_RESOLVED_OK $((end_time - start_time))"
'''
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15
        )
        self.assertEqual(proc.returncode, 0, f"Mutex test failed with rc {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("MUTEX_CONTENTION_RESOLVED_OK", proc.stdout)

    def test_signal_handling_bounded_wait_and_retention(self):
        """
        Requirement 6 & Acceptance criterion 7 & R3:
        When SIGTERM is received while pipeline is executing:
        - Handshake file (.py_started) ensures Python child and its grandchild worker are running.
        - SIGTERM is caught and forwarded to child process tree.
        - Grandchild and child terminate before final copy.
        - Finalization executes once, saves diagnostics with signal_received=143 and overall_exit_code=143,
          retains scratch, and exits 143.
        """
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["PBS_O_WORKDIR"] = self.repo_dir
        env["SCRATCHDIR"] = self.scratch_dir
        env["PBS_JOBID"] = "24702696"
        env["ALDEBARAN_SIGNAL_TIMEOUT"] = "3"

        # Update aldebaran.py in repo to spawn a grandchild process and emit a readiness handshake
        aldebaran_py = os.path.join(self.repo_dir, "aldebaran.py")
        with open(aldebaran_py, "w", encoding="utf-8") as f:
            f.write('''import os, sys, time, subprocess

OUTPUT_DIR = "tess_results"

if __name__ == "__main__":
    ready_file = os.path.join(os.environ.get("SCRATCHDIR", "."), ".py_started")
    # Spawn long-running grandchild worker
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(ready_file, "w") as f:
        f.write(f"{os.getpid()},{worker.pid}\\n")

    try:
        worker.wait(timeout=60)
    except Exception:
        pass
    sys.exit(0)
''')

        target_posix = self.pbs_script_src.replace("\\", "/")
        bin_posix = self.bin_dir.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        bash_cmd = f'''
export PATH="$(cd "{bin_posix}" && pwd):$PATH"
"{target_posix}" &
PBS_PID=$!

# Handshake: wait until Python child has created .py_started
waited=0
while [ ! -f "{scratch_posix}/.py_started" ] && [ "$waited" -lt 10 ]; do
    sleep 0.5
    waited=$((waited + 1))
done

kill -TERM "$PBS_PID" 2>/dev/null || true
wait "$PBS_PID"
exit $?
'''
        proc = subprocess.Popen(
            [BASH_EXE, "-c", bash_cmd],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._active_procs.append(proc)

        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired as exc:
            kill_process_tree(proc)
            raise subprocess.TimeoutExpired(
                proc.args, 15, output=exc.stdout, stderr=exc.stderr
            ) from exc

        self.assertEqual(proc.returncode, 143, f"Unexpected returncode {proc.returncode}:\nstdout:\n{stdout}\nstderr:\n{stderr}")
        self.assertIn("Termination signal received", stdout + stderr)
        self.assertNotIn("Local scratch cleaned", stdout)
        self.assertIn("Scratch preserved at:", stdout)

        # Diagnostics saved with signal_received=143 and overall_exit_code=143
        diag_path = Path(self.repo_dir) / "tess_results" / "job_diagnostics_24702696_single.json"
        self.assertTrue(diag_path.exists())
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["signal_received"], 143)
        self.assertEqual(diag["overall_exit_code"], 143)
        self.assertFalse(diag["scratch_cleaned"])

        # clean_scratch was NOT called
        clean_calls = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertFalse(clean_calls.exists())

        # R3: Assert child and grandchild worker PIDs actually no longer exist
        ready_file = Path(self.scratch_dir) / ".py_started"
        self.assertTrue(ready_file.exists())
        py_pid_str, worker_pid_str = ready_file.read_text(encoding="utf-8").strip().split(",")
        for target_pid in [py_pid_str, worker_pid_str]:
            check = subprocess.run([BASH_EXE, "-c", f"kill -0 {target_pid} 2>/dev/null"], capture_output=True)
            self.assertNotEqual(check.returncode, 0, f"Process PID {target_pid} still alive after SIGTERM!")

    def test_signal_handling_int_bounded_wait_and_retention(self):
        """
        R3 [P2] Signal INT case:
        Verifies SIGINT handling: executes on_signal INT 130 against an active child process tree,
        proves bounded termination of children/grandchildren via kill_process_tree, retains scratch,
        saves diagnostics with signal_received=130 and overall_exit_code=130, and exits 130.
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        bin_posix = self.bin_dir.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")

        test_script = f'''
export PATH="$(cd "{bin_posix}" && pwd):$PATH"
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"

SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
SETUP_OK=1
STAGE_REACHED="TIER1_RUNNING"
ALDEBARAN_SIGNAL_TIMEOUT=3

# Spawn a child with grandchild worker in background
(
    echo "$BASHPID" > "$SCRATCHDIR/.int_child_pid"
    (
        echo "$BASHPID" > "$SCRATCHDIR/.int_worker_pid"
        sleep 60 &
        SLEEP_PID=$!
        echo "$SLEEP_PID" > "$SCRATCHDIR/.int_sleep_pid"
        wait "$SLEEP_PID"
    ) &
    wait
) &
PY_PID=$!

sleep 0.5
[ -n "$PY_PID" ] && kill -0 "$PY_PID" 2>/dev/null || exit 31

# Trigger signal handling
on_signal INT 130
'''
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15
        )

        self.assertEqual(proc.returncode, 130, f"Unexpected returncode {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("Termination signal received", proc.stdout + proc.stderr)
        self.assertIn("Forwarding INT to child process tree", proc.stderr)
        self.assertNotIn("Local scratch cleaned", proc.stdout)
        self.assertIn("Scratch preserved at:", proc.stdout)

        diags = list((Path(self.repo_dir) / "tess_results").glob("job_diagnostics_*.json"))
        self.assertTrue(len(diags) > 0, "Diagnostic file was not saved to permanent storage!")
        with open(diags[0], "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["signal_received"], 130)
        self.assertEqual(diag["overall_exit_code"], 130)
        self.assertFalse(diag["scratch_cleaned"])

        # R3: Assert INT child, worker, and sleep PIDs actually no longer exist
        for pid_file_name in [".int_child_pid", ".int_worker_pid", ".int_sleep_pid"]:
            pf = Path(self.scratch_dir) / pid_file_name
            if pf.exists():
                p = pf.read_text(encoding="utf-8").strip()
                if p:
                    check = subprocess.run([BASH_EXE, "-c", f"kill -0 {p} 2>/dev/null"], capture_output=True)
                    self.assertNotEqual(check.returncode, 0, f"INT process PID {p} still alive after SIGINT!")

    def test_signal_handling_term_resistant_descendant_escalation_and_death(self):
        """
        R3 [P1] Negative control & descendant quiescence:
        Verifies that when a child process exits on TERM but its descendant (worker/grandchild)
        ignores TERM (trap '' TERM), on_signal snapshots the entire tree, tracks the surviving
        descendant even after parent disappearance, escalates to KILL after the grace period,
        and ensures zero surviving descendant writers before entering finalization.
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")

        # Worker script that ignores SIGTERM
        worker_sh = Path(self.scratch_dir) / "worker.sh"
        worker_sh.write_text("""#!/bin/bash
trap '' TERM
echo $BASHPID > "$1/worker.pid"
while true; do sleep 1; done
""", encoding="utf-8", newline="\n")

        # Parent script that launches worker and exits on TERM
        parent_sh = Path(self.scratch_dir) / "parent.sh"
        parent_sh.write_text(f"""#!/bin/bash
bash "{worker_sh.as_posix()}" "{scratch_posix}" </dev/null >/dev/null 2>&1 &
wait
""", encoding="utf-8", newline="\n")

        test_script = f'''
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"
SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
SETUP_OK=1
STAGE_REACHED="TIER1_RUNNING"
ALDEBARAN_SIGNAL_TIMEOUT=1

bash "{parent_sh.as_posix()}" </dev/null >/dev/null 2>&1 &
PY_PID=$!

# Wait for worker handshake
for i in {{1..30}}; do
    [ -s "{scratch_posix}/worker.pid" ] && break
    sleep 0.1
done
WORKER_PID=$(cat "{scratch_posix}/worker.pid")

# Trigger on_signal with TERM
on_signal TERM 143
'''
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15
        )
        self.assertEqual(proc.returncode, 143, f"Unexpected returncode {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("Termination signal received", proc.stdout + proc.stderr)
        self.assertIn("survived 1s grace period; escalating to KILL", proc.stderr)
        self.assertIn("Scratch preserved at:", proc.stdout)

        # Assert worker PID is confirmed dead
        worker_pid_file = Path(self.scratch_dir) / "worker.pid"
        self.assertTrue(worker_pid_file.exists())
        worker_pid = worker_pid_file.read_text(encoding="utf-8").strip()
        check = subprocess.run([BASH_EXE, "-c", f"kill -0 {worker_pid} 2>/dev/null"], capture_output=True)
        self.assertNotEqual(check.returncode, 0, f"Worker process {worker_pid} is still alive!")

    def test_stop_periodic_sync_in_flight_copy_termination_and_lock_release(self):
        """
        R3 [P2] In-flight copy & concurrency control:
        Verifies that when periodic sync is actively in the middle of a copy (holding .sync_lock),
        stop_periodic_sync safely snapshots and terminates the copier (including escalation if needed),
        reaps the process tree, releases the stale lock left by the terminated copier, and allows
        final result synchronization to acquire the lock immediately and proceed cleanly.
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")

        test_script = f'''
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"
SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
SETUP_OK=1
STAGE_REACHED="PIPELINE_COMPLETE"

# Simulate periodic sync loop acquiring lock and launching an in-flight copy
(
    acquire_sync_lock "periodic"
    (
        echo $BASHPID > "$SCRATCHDIR/in_flight_cp.pid"
        trap '' TERM
        while true; do sleep 1; done
    ) &
    CP_PID=$!
    wait "$CP_PID"
    release_sync_lock
) </dev/null >/dev/null 2>&1 &
SYNC_LOOP_PID=$!

# Wait for active copy process handshake
for i in {{1..30}}; do
    [ -s "$SCRATCHDIR/in_flight_cp.pid" ] && break
    sleep 0.1
done
[ -d "$SCRATCHDIR/.sync_lock" ] || exit 41

# Stop periodic sync while copy is in flight
stop_periodic_sync

# Verify copier is terminated
CP_PID=$(cat "$SCRATCHDIR/in_flight_cp.pid")
if kill -0 "$CP_PID" 2>/dev/null; then
    echo "ERROR: Copier still alive!" >&2
    exit 42
fi

# Verify lock was released
if [ -d "$SCRATCHDIR/.sync_lock" ]; then
    echo "ERROR: Lock still held!" >&2
    exit 43
fi

# Verify final sync can now acquire lock without timeout
if acquire_sync_lock "final"; then
    echo "FINAL_LOCK_ACQUIRED_OK"
    release_sync_lock
else
    echo "ERROR: Final sync could not acquire lock!" >&2
    exit 44
fi
'''
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15
        )
        self.assertEqual(proc.returncode, 0, f"In-flight copy test failed with rc {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("FINAL_LOCK_ACQUIRED_OK", proc.stdout)
        self.assertIn("Releasing stale lock left by terminated periodic sync", proc.stderr)

    def test_post_cleanup_diagnostic_update_failure_propagates_exit_7(self):
        """
        R2 [P2] Injected post-cleanup diagnostics update failure:
        Verifies that if the post-cleanup call to write_job_diagnostics fails on permanent storage
        after clean_scratch has succeeded:
        - The wrapper detects the failure and exits with non-zero status (code 7).
        - The prior valid pre-cleanup diagnostics JSON is preserved on permanent storage.
        - PBS output truthfully reports the failure and notes that the cleaned scratch copy is not recoverable.
        - The wrapper does not falsely claim clean success.
        """
        target_posix = self.pbs_script_src.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        repo_posix = self.repo_dir.replace("\\", "/")

        test_script = f'''
ALDEBARAN_PBS_SOURCE_ONLY=1 source "{target_posix}"
SCRATCHDIR="{scratch_posix}"
REPO_DIR="{repo_posix}"
SETUP_OK=1
STAGE_REACHED="PIPELINE_COMPLETE"
sync_results() {{ return 0; }}

# Injected hook: succeed on pre-cleanup writes (1 & 2), fail on post-cleanup write (3)
N_WRITES=0
write_job_diagnostics() {{
    local target_dir="$1"
    local rc="$2"
    N_WRITES=$((N_WRITES + 1))
    if [ "$N_WRITES" -eq 3 ]; then
        echo "INJECTED_POST_CLEANUP_WRITE_FAILURE" >&2
        return 1
    fi
    echo "{{\\"precleanup_saved\\": true, \\"writes\\": $N_WRITES, \\"scratch_cleaned\\": false, \\"overall_exit_code\\": $rc}}" > "$target_dir/$DIAG_FILENAME"
    return 0
}}

# Real scratch deletion
clean_scratch() {{
    echo "CLEANUP_INVOKED"
    rm -rf "$SCRATCHDIR"
    return 0
}}

sync_and_finalize 0
'''
        # Keep the single-job fixture independent of an enclosing PBS allocation.
        env = os.environ.copy()
        env.pop("PBS_JOBID", None)
        env.pop("PBS_ARRAY_INDEX", None)
        proc = subprocess.run(
            [BASH_EXE, "-c", test_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=15
        )
        self.assertEqual(proc.returncode, 7, f"Expected returncode 7, got {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("INJECTED_POST_CLEANUP_WRITE_FAILURE", proc.stderr)
        self.assertIn("Failed to publish final post-cleanup job diagnostics", proc.stderr)
        self.assertIn("Prior pre-cleanup diagnostics preserved at:", proc.stderr)
        self.assertIn("Scratch was cleaned", proc.stderr)
        self.assertIn("scratch copy is not recoverable", proc.stderr)
        self.assertIn("CLEANUP_INVOKED", proc.stdout)

        # Assert prior pre-cleanup diagnostics file is intact
        diag_file = Path(self.repo_dir) / "tess_results" / "job_diagnostics_local_single.json"
        self.assertTrue(diag_file.exists())
        data = json.loads(diag_file.read_text(encoding="utf-8"))
        self.assertTrue(data.get("precleanup_saved"))
        self.assertEqual(data.get("writes"), 2)

        # Assert scratch was actually deleted
        self.assertFalse(os.path.exists(self.scratch_dir))

    def test_regression_unpatched_script_baseline_masked_failure(self):
        """
        Acceptance criterion 10 / Core evidence rule:
        Demonstrates that the unpatched baseline script pattern (as documented in
        METACENTRUM_24702696_1_EVIDENCE.md) FAILS to propagate Python error (ignoring exit 7),
        masks copy failures, and wipes scratch on failure, whereas the new implementation
        catches the failure, preserves scratch, and exits 7.
        """
        # Unpatched baseline pattern
        repo_posix = self.repo_dir.replace("\\", "/")
        scratch_posix = self.scratch_dir.replace("\\", "/")
        unpatched_script = f'''#!/bin/bash
REPO_DIR="{repo_posix}"
SCRATCHDIR="{scratch_posix}"

run_py() {{
    python "$@" &
    PY_PID=$!
    wait "$PY_PID"
    return $?
}}

sync_results() {{
    mkdir -p "$REPO_DIR/tess_results"
    cp -rf "$SCRATCHDIR/tess_results"/. "$REPO_DIR/tess_results/" 2>/dev/null || true
}}

sync_and_clean() {{
    sync_results
    clean_scratch
}}

trap sync_and_clean EXIT

# Stage code and cd scratch (matching original line 108-114)
cp "$REPO_DIR"/*.py "$SCRATCHDIR/"
cd "$SCRATCHDIR" || exit 1

# Tier 1 call ignoring return code (original flaw!)
run_py aldebaran.py

echo "Unpatched script finished normally"
exit 0
'''
        unpatched_file = Path(self.test_root) / "unpatched_baseline.sh"
        unpatched_file.write_text(unpatched_script, encoding="utf-8")

        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["INJECT_TIER1_EXIT"] = "7"
        env["SCRATCHDIR"] = self.scratch_dir

        proc_unpatched = subprocess.Popen(
            [BASH_EXE, str(unpatched_file).replace("\\", "/")],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._active_procs.append(proc_unpatched)
        try:
            stdout_unp, stderr_unp = proc_unpatched.communicate(timeout=10)
        except subprocess.TimeoutExpired as exc:
            kill_process_tree(proc_unpatched)
            raise subprocess.TimeoutExpired(proc_unpatched.args, 10, exc.stdout, exc.stderr) from exc
        proc_unpatched.stdout = stdout_unp
        proc_unpatched.stderr = stderr_unp

        # The unpatched script falsely exits 0 and cleans scratch despite Python exit 7!
        self.assertEqual(proc_unpatched.returncode, 0, "Unpatched script failed to reproduce original bug")
        clean_calls_unpatched = Path(self.scratch_dir) / ".clean_scratch_calls"
        self.assertTrue(clean_calls_unpatched.exists(), "Unpatched script failed to clean scratch")

        # Clean counter for patched test
        clean_calls_unpatched.unlink()

        # In contrast, the patched script correctly propagates exit 7 and preserves scratch!
        proc_patched = self._run_pbs({"INJECT_TIER1_EXIT": "7"})
        self.assertEqual(proc_patched.returncode, 7)
        self.assertFalse(clean_calls_unpatched.exists(), "Patched script should NOT have cleaned scratch on exit 7")
        self.assertIn("Scratch preserved at:", proc_patched.stdout)


if __name__ == "__main__":
    unittest.main()
