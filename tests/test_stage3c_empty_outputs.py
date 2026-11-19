import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antares_packager


EXPECTED = {
    "antares_exofop_ctoi_upload.csv": b"TIC ID|Camera|CCD|Epoch BJD|Period|Depth|Duration|Rp|User Comments / Disposition\n",
    "antares_exofop_ctoi_upload_comma.csv": b"TIC ID,Camera,CCD,Epoch BJD,Period,Depth,Duration,Rp,User Comments / Disposition\n",
    "antares_exofop_ctoi_upload_top3.csv": b"TIC ID,Camera,CCD,Epoch BJD,Period,Depth,Duration,Rp,User Comments / Disposition\n",
    # S2.5: the zero-survivor ranking carries the packaged ranking header (ANTARES_RANKING_COLS).
    "antares_prioritized_targets_ranking.csv": (",".join(antares_packager.ANTARES_RANKING_COLS) + "\n").encode(),
}


class TestStage3CEmptyOutputs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def no_pass(self):
        results = [
            {"status": status, "candidate": {"tic_id": str(i)}}
            for i, status in enumerate(("ARCHIVAL_REJECT", "ARCHIVAL_REJECT", "ARCHIVAL_ERROR"), 1)
        ]
        return antares_packager.run_stage_3c_packaging(results, results_dir=self.d)

    def assert_empty_tables(self):
        actual = {
            p.name: p.read_bytes().replace(b"\r", b"")
            for p in Path(self.d).iterdir()
        }
        self.assertEqual(actual, EXPECTED)

    def test_no_pass_writes_four_empty_tables(self):
        self.no_pass()
        self.assert_empty_tables()

    def test_no_pass_returns_empty_frames(self):
        ranking, exofop = self.no_pass()
        self.assertIsInstance(ranking, pd.DataFrame)
        self.assertIsInstance(exofop, pd.DataFrame)
        self.assertEqual(len(ranking), 0)
        self.assertEqual(len(exofop), 0)

    def test_zero_survivor_content_unchanged(self):
        with mock.patch.object(
            antares_packager, "run_stage_3a_filter",
            return_value=(pd.DataFrame(columns=["tic_id"]), pd.DataFrame()),
        ):
            antares_packager.run_antares_packager(
                results_dir=self.d, tier1_csv="unused", tier2_csv="unused",
            )
        self.assert_empty_tables()

    def test_conflicting_exofop_schema_refuses(self):
        exofop = Path(self.d, "antares_exofop_ctoi_upload.csv")
        exofop.write_bytes(b"a|b\n")
        with self.assertRaisesRegex(ValueError, "Refusing conflicting output schema"):
            self.no_pass()
        self.assertFalse(Path(self.d, "antares_prioritized_targets_ranking.csv").exists())
        self.assertEqual(exofop.read_bytes(), b"a|b\n")

    def test_stale_empty_schema_files_replaced(self):
        for name, content in EXPECTED.items():
            Path(self.d, name).write_bytes(content + b"stale data row\n")
        self.no_pass()
        self.assert_empty_tables()

    def test_existing_foreign_ranking_refuses(self):
        # S2.5: a packaged ranking now has the zero-survivor header; a foreign header still refuses.
        ranking = Path(self.d, "antares_prioritized_targets_ranking.csv")
        content = b"rank,tic_id,depth_schema_version\n1,123,x\n"
        ranking.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "Refusing conflicting output schema"):
            self.no_pass()
        self.assertEqual(ranking.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
