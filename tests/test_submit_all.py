"""Offline Bash production-launcher controls; never invoke a real scheduler."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH = shutil.which('bash') or shutil.which(r'C:\Program Files\Git\bin\bash.exe')
LAUNCHER = Path(__file__).resolve().parents[1] / 'cluster' / 'submit_all.sh'

class TestSubmitAll(unittest.TestCase):
    def invoke(self, args, result='12345[].pbs.example', code=0, script=None, missing=False):
        self.assertIsNotNone(BASH, 'Bash is required for launcher acceptance')
        with tempfile.TemporaryDirectory(prefix='submit_contract_') as tmp:
            root=Path(tmp); (root/'cluster').mkdir()
            launcher=root/'cluster'/'submit_all.sh'
            launcher.write_text(script if script is not None else LAUNCHER.read_text(), encoding='utf-8', newline='\n')
            log=root/'calls'
            # Export a Bash function: executable lookup never reaches a host qsub.
            prelude='qsub() { printf "%s\\n" "$*" >> calls; printf "%s\\n" "$STUB_RESULT"; return "$STUB_CODE"; }; export -f qsub; '
            if missing:
                prelude='unset -f qsub; PATH=/nonexistent; '
            env=os.environ.copy();env.update(STUB_RESULT=result,STUB_CODE=str(code))
            proc=subprocess.run([BASH,'--noprofile','--norc','-c',prelude+'source cluster/submit_all.sh "$@"','contract',*args],cwd=tmp,env=env,text=True,capture_output=True,timeout=15)
            return proc, log.read_text().splitlines() if log.exists() else []

    def assert_scan(self, script=None):
        p,calls=self.invoke(['--scan-only','--array','2-4'],script=script)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(calls,['-J 2-4 cluster/run_aldebaran.pbs'])
        self.assertIn('Scan array submitted:',p.stdout)

    def test_scan_once(self):
        self.assert_scan()
        p,calls=self.invoke(['--scan-only'])
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(calls,['-J 1-8 cluster/run_aldebaran.pbs'])

    def test_refuse_before_submission(self):
        for args in [[],['--merge'],['--scan-only','--merge'],['--scan-only','--array','8-1'],['--scan-only','--array','0'],['--scan-only','--array','1;echo bad'],['--scan-only','--array','1','--array','2']]:
            with self.subTest(args=args):
                p,calls=self.invoke(args)
                self.assertNotEqual(p.returncode,0)
                self.assertEqual(calls,[])

    def test_scheduler_faults(self):
        for result,code in [('123.pbs',7),('',0),('garbage',0),('123.pbs\n456.pbs',0)]:
            with self.subTest(result=result,code=code):
                p,calls=self.invoke(['--scan-only'],result,code)
                self.assertNotEqual(p.returncode,0)
                self.assertNotIn('Scan array submitted:',p.stdout)
        p,calls=self.invoke(['--scan-only'],missing=True)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(calls,[])

    def test_unconditional_reducer_mutation_is_detected(self):
        mutated=LAUNCHER.read_text().replace('echo "Scan array submitted:', 'qsub cluster/run_merge.pbs\necho "Scan array submitted:',1)
        self.assertNotEqual(mutated,LAUNCHER.read_text())
        with self.assertRaises(AssertionError):
            self.assert_scan(mutated)
