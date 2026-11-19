import os
import re
import sys
import unittest
from unittest import mock
from urllib.parse import unquote

import numpy as np
import pandas as pd
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aldebaran


class TestTicProperMotion(unittest.TestCase):
    def tic_table(self, **proper_motion):
        values = {
            "rad": [1.0], "Teff": [5800.0], "e_rad": [0.1],
            "mass": [1.0], "e_mass": [0.1], "ra": [150.0], "dec": [20.0],
        }
        values.update({name: [value] for name, value in proper_motion.items()})
        return Table(values, masked=True)

    def query_centre(self, table):
        with mock.patch.object(aldebaran, "Catalogs") as catalogs, \
             mock.patch.object(aldebaran, "fetch_csv_with_headers") as fetch:
            catalogs.query_criteria.return_value = table
            fetch.return_value = pd.DataFrame({"ruwe": [1.1], "dist_deg": [0.0]})
            result = aldebaran.get_stellar_and_gaia_params("123")
        catalogs.query_criteria.assert_called_once_with(catalog="Tic", ID=123)
        fetch.assert_called_once()
        circle = re.search(
            r"CIRCLE\('ICRS', ([-0-9.]+), ([-0-9.]+), 0\.001389\)",
            unquote(fetch.call_args.args[0]),
        )
        self.assertIsNotNone(circle)
        self.assertEqual(result["ruwe"], 1.1)
        return float(circle.group(1)), float(circle.group(2))

    def assert_shifted(self, table):
        ra, dec = self.query_centre(table)
        self.assertAlmostEqual(ra, 150.0 + 1000.0 * 16.0 / (3.6e6 * np.cos(np.radians(20.0))), places=6)
        self.assertAlmostEqual(dec, 20.0 - 500.0 * 16.0 / 3.6e6, places=6)

    def test_tic_pmRA_pmDEC_columns_shift_cone_centre(self):
        self.assert_shifted(self.tic_table(pmRA=1000.0, pmDEC=-500.0))

    def test_lowercase_columns_also_accepted(self):
        self.assert_shifted(self.tic_table(pmra=1000.0, pmdec=-500.0))

    def test_missing_proper_motion_columns_leave_position_unchanged(self):
        self.assertEqual(self.query_centre(self.tic_table()), (150.0, 20.0))

    def test_masked_proper_motion_treated_as_zero(self):
        table = self.tic_table(pmRA=1000.0, pmDEC=0.0)
        table["pmRA"].mask = [True]
        self.assertEqual(self.query_centre(table), (150.0, 20.0))


if __name__ == "__main__":
    unittest.main()
