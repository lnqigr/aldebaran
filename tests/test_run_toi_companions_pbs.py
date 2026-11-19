#!/usr/bin/env python3
"""
PBS contract tests for cluster/run_toi_companions.pbs (roadmap S2.6, IMPROVE-QUICK
plan 004 with the plan 006 PBS tests).

The whole wrapper runs under Bash with stub executables on PATH (python3, python,
and for the guard tests cp and mkdir), so nothing is installed, downloaded or
searched. The tests pin three contracts:

1. REPO_DIR and SCRATCHDIR guards: an unset, empty, missing or unsafe SCRATCHDIR
   (or a REPO_DIR without search_toi_companions.py) exits 1 before any copy,
   cache directory, venv or workload.
2. constraints-linux-py311.txt is copied to scratch and passed to pip as
   "-c constraints-linux-py311.txt" when the repository ships it, and not
   otherwise.
3. SIGTERM and SIGINT stop the workload process tree within the grace period,
   copy the partial results back through sync_and_finalize, retain scratch and
   exit 143 or 130.

The existing sed-sliced tests of the same script live in
tests/test_toi_companions_auditable.py.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

PBS_PATH = Path(__file__).resolve().parent.parent / "cluster" / "run_toi_companions.pbs"

PYTHON3_STUB = """#!/bin/sh
echo "VENV $*" >> "$STUB_LOG"
if [ "${STUB_VENV_FAIL:-0}" = "1" ]; then
    exit 1
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    /bin/mkdir -p "$3/bin"
    echo ":" > "$3/bin/activate"
fi
exit 0
"""

PYTHON_STUB = """#!/bin/sh
case "$1" in
    -c)
        exit 0
        ;;
    -m)
        echo "PIP $*" >> "$STUB_LOG"
        exit 0
        ;;
    search_toi_companions.py)
        echo "WORKLOAD $*" >> "$STUB_LOG"
        out=""
        prev=""
        for a in "$@"; do
            if [ "$prev" = "--output-dir" ]; then
                out="$a"
            fi
            prev="$a"
        done
        /bin/mkdir -p "$out"
        echo "tic_id,status" > "$out/toi_companions_summary.csv"
        echo "1,PARTIAL" >> "$out/toi_companions_summary.csv"
        case "${STUB_MODE:-exit0}" in
            exit0)
                exit 0
                ;;
            hang|stubborn)
                if [ "$STUB_MODE" = "stubborn" ]; then
                    trap '' TERM
                fi
                sleep 60 >/dev/null 2>&1 &
                echo "$!" > "$STUB_DIR/worker.pid"
                echo "$$" > "$STUB_DIR/main.pid"
                exec sleep 60 >/dev/null 2>&1
                ;;
        esac
        ;;
esac
exit 0
"""

LOGGING_STUB = """#!/bin/sh
echo "{name} $*" >> "$STUB_LOG"
exit 0
"""


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("State:"):
                    return "Z" not in line
    except OSError:
        pass
    return True


class TestRunToiCompanionsPbs(unittest.TestCase):
    def setUp(self):
        self.bash = shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash executable not found in environment")
        self.root = Path(tempfile.mkdtemp(prefix="toi_pbs_"))
        self.repo = self.root / "repo"
        self.scratch = self.root / "scratch"
        self.stub_dir = self.root / "stub"
        self.bin = self.stub_dir / "bin"
        for d in (self.repo, self.scratch, self.bin):
            d.mkdir(parents=True)
        (self.repo / "search_toi_companions.py").write_text("# stub\n", encoding="utf-8")
        (self.repo / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        self.log = self.stub_dir / "calls.log"
        self.log.write_text("", encoding="utf-8")
        self._write_stub("python3", PYTHON3_STUB)
        self._write_stub("python", PYTHON_STUB)
        # Run an LF copy so the test does not depend on the checkout's line endings.
        self.script = self.root / "run_toi_companions.pbs"
        self.script.write_bytes(PBS_PATH.read_bytes().replace(b"\r\n", b"\n"))
        self._spawned = []

    def tearDown(self):
        for name in ("main.pid", "worker.pid"):
            p = self.stub_dir / name
            if p.exists():
                try:
                    os.kill(int(p.read_text().strip()), signal.SIGKILL)
                except (OSError, ValueError):
                    pass
        for proc in self._spawned:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_stub(self, name, body):
        path = self.bin / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def _env(self, **overrides):
        env = dict(os.environ)
        env.update({
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "PBS_O_WORKDIR": str(self.repo),
            "SCRATCHDIR": str(self.scratch),
            "PBS_JOBID": "test_job",
            "STUB_LOG": str(self.log),
            "STUB_DIR": str(self.stub_dir),
            "TOI_SIGNAL_GRACE_SECONDS": "2",
        })
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return env

    def _run(self, **overrides):
        return subprocess.run([self.bash, str(self.script)], env=self._env(**overrides),
                              capture_output=True, text=True, timeout=60)

    def _calls(self):
        return self.log.read_text(encoding="utf-8").splitlines()

    # ------------------------------------------------------------------ guards

    def _install_copy_stubs(self):
        # cp and mkdir only log, so a guard that is missing cannot touch "/" or $HOME;
        # venv creation fails, so a missing guard ends with exit 3, not 1.
        self._write_stub("cp", LOGGING_STUB.format(name="CP"))
        self._write_stub("mkdir", LOGGING_STUB.format(name="MKDIR"))

    def test_scratchdir_guard_fails_closed_before_any_copy(self):
        self._install_copy_stubs()
        home = self.root / "home"
        home.mkdir()
        cases = {
            "unset": dict(SCRATCHDIR=None),
            "empty": dict(SCRATCHDIR=""),
            "missing": dict(SCRATCHDIR=str(self.root / "no_such_scratch")),
            "root": dict(SCRATCHDIR="/"),
            "home": dict(SCRATCHDIR=str(home), HOME=str(home)),
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                self.log.write_text("", encoding="utf-8")
                proc = self._run(STUB_VENV_FAIL="1", **overrides)
                self.assertEqual(proc.returncode, 1, f"stderr:\n{proc.stderr}")
                self.assertIn("ERROR: SCRATCHDIR", proc.stderr)
                self.assertEqual(self._calls(), [], "no copy, cache dir, venv or workload may run")

    def test_repo_dir_guard_fails_closed(self):
        self._install_copy_stubs()
        no_script = self.root / "elsewhere" / "deeper"
        no_script.mkdir(parents=True)
        for label, workdir in {"unset": None, "empty": "", "no_script": str(no_script)}.items():
            with self.subTest(case=label):
                self.log.write_text("", encoding="utf-8")
                proc = self._run(STUB_VENV_FAIL="1", PBS_O_WORKDIR=workdir)
                self.assertEqual(proc.returncode, 1, f"stderr:\n{proc.stderr}")
                self.assertIn("ERROR: Invalid REPO_DIR", proc.stderr)
                self.assertEqual(self._calls(), [])

    def test_repo_dir_parent_fallback_still_works(self):
        cluster_dir = self.repo / "cluster"
        cluster_dir.mkdir()
        proc = self._run(PBS_O_WORKDIR=str(cluster_dir))
        self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}")
        self.assertTrue((self.repo / "tess_results" / "toi_companions" / "toi_companions_summary.csv").exists())

    # ------------------------------------------------------------- constraints

    def _pip_install_lines(self):
        return [c for c in self._calls() if c.startswith("PIP -m pip install") and "--upgrade" not in c]

    def test_constraints_file_copied_and_forwarded_to_pip(self):
        (self.repo / "constraints-linux-py311.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
        proc = self._run()
        self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}")
        self.assertTrue((self.scratch / "constraints-linux-py311.txt").exists())
        self.assertEqual(self._pip_install_lines(), [
            "PIP -m pip install --no-cache-dir --quiet -c constraints-linux-py311.txt -r requirements.txt certifi"])
        self.assertTrue(any(c.startswith("WORKLOAD search_toi_companions.py") for c in self._calls()))
        summary = self.repo / "tess_results" / "toi_companions" / "toi_companions_summary.csv"
        self.assertTrue(summary.exists())

    def test_no_constraints_file_installs_without_c(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}")
        self.assertFalse((self.scratch / "constraints-linux-py311.txt").exists())
        self.assertEqual(self._pip_install_lines(), [
            "PIP -m pip install --no-cache-dir --quiet -r requirements.txt certifi"])

    # ----------------------------------------------------------------- signals

    def _start_hanging(self, mode="hang"):
        if os.name != "posix" or not shutil.which("pgrep"):
            self.skipTest("POSIX signals and pgrep are required")
        proc = subprocess.Popen([self.bash, str(self.script)], env=self._env(STUB_MODE=mode),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._spawned.append(proc)
        deadline = time.monotonic() + 30
        main_pid_file = self.stub_dir / "main.pid"
        while time.monotonic() < deadline:
            if main_pid_file.exists() and main_pid_file.read_text().strip():
                break
            if proc.poll() is not None:
                self.fail(f"wrapper exited early: {proc.communicate()}")
            time.sleep(0.05)
        else:
            self.fail("workload stub did not start")
        main_pid = int(main_pid_file.read_text().strip())
        worker_pid = int((self.stub_dir / "worker.pid").read_text().strip())
        return proc, main_pid, worker_pid

    def _assert_signal_outcome(self, sig, expected_rc, mode="hang"):
        proc, main_pid, worker_pid = self._start_hanging(mode)
        started = time.monotonic()
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=30)
        elapsed = time.monotonic() - started
        self.assertEqual(proc.returncode, expected_rc, f"stdout:\n{out}\nstderr:\n{err}")
        self.assertIn(f"Signal received (exit {expected_rc})", err)
        self.assertLess(elapsed, 15, "grace period must bound the shutdown")
        # The workload and its worker are gone.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (_pid_alive(main_pid) or _pid_alive(worker_pid)):
            time.sleep(0.05)
        self.assertFalse(_pid_alive(main_pid), "workload process survived the signal")
        self.assertFalse(_pid_alive(worker_pid), "workload worker survived the signal")
        # Partial results reached permanent storage; scratch is retained.
        synced = self.repo / "tess_results" / "toi_companions" / "toi_companions_summary.csv"
        self.assertTrue(synced.exists(), f"partial results were not synced; stderr:\n{err}")
        self.assertIn("1,PARTIAL", synced.read_text(encoding="utf-8"))
        self.assertTrue((self.scratch / "tess_results" / "toi_companions" / "toi_companions_summary.csv").exists())
        self.assertIn("Retaining scratch", err)
        return err

    def test_sigterm_stops_workload_syncs_partial_results_and_exits_143(self):
        self._assert_signal_outcome(signal.SIGTERM, 143)

    def test_sigint_stops_workload_syncs_partial_results_and_exits_130(self):
        self._assert_signal_outcome(signal.SIGINT, 130)

    def test_sigterm_escalates_to_kill_after_grace_period(self):
        err = self._assert_signal_outcome(signal.SIGTERM, 143, mode="stubborn")
        self.assertIn("survived the 2s grace period", err)


if __name__ == "__main__":
    unittest.main()
