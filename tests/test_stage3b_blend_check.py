import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antares_packager


def _old_blend(author_stats, contam):
    blended_suspect = False
    blend_reason = ""
    if "SPOC" in author_stats and ("QLP" in author_stats or "TGLC" in author_stats):
        d_spoc = author_stats["SPOC"]["depth_ppm"]
        d_alt = max(
            author_stats.get("QLP", {}).get("depth_ppm", 0.0),
            author_stats.get("TGLC", {}).get("depth_ppm", 0.0)
        )
        if d_spoc > 2.5 * max(d_alt, 1.0) and contam > 5.0 and d_alt < 200.0:
            blended_suspect = True
            blend_reason = f"SPOC depth ({d_spoc:.0f} ppm) diluted > 2.5x in QLP/TGLC ({d_alt:.0f} ppm)"
    return blended_suspect, blend_reason


class TestStage3BBlendCheck(unittest.TestCase):
    def test_finite_inputs_match_old_logic(self):
        combinations = 0
        for spoc in (50, 400, 1000, 5000):
            for qlp in (None, 10, 150, 400, 900):
                for tglc in (None, 10, 150, 400, 900):
                    if qlp is None and tglc is None:
                        continue
                    for contam in (2.0, 10.0):
                        stats = {"SPOC": {"depth_ppm": spoc}}
                        if qlp is not None:
                            stats["QLP"] = {"depth_ppm": qlp}
                        if tglc is not None:
                            stats["TGLC"] = {"depth_ppm": tglc}
                        with self.subTest(spoc=spoc, qlp=qlp, tglc=tglc, contam=contam):
                            actual = antares_packager.assess_blend(stats, contam)
                            self.assertEqual(actual[:2], _old_blend(stats, contam))
                            self.assertEqual(actual[2], "EVALUATED")
                        combinations += 1
        self.assertEqual(combinations, 192)

    def test_qlp_nan_uses_tglc(self):
        stats = {"SPOC": {"depth_ppm": 1000}, "QLP": {"depth_ppm": np.nan}, "TGLC": {"depth_ppm": 100}}
        self.assertFalse(_old_blend(stats, 10.0)[0])
        blended, reason, check = antares_packager.assess_blend(stats, 10.0)
        self.assertTrue(blended)
        self.assertEqual(reason, "SPOC depth (1000 ppm) diluted > 2.5x in QLP/TGLC (100 ppm)")
        self.assertEqual(check, "EVALUATED")

    def test_tglc_nan_qlp_absent_is_not_evaluated(self):
        stats = {"SPOC": {"depth_ppm": 1000}, "TGLC": {"depth_ppm": np.nan}}
        self.assertTrue(_old_blend(stats, 10.0)[0])
        self.assertEqual(antares_packager.assess_blend(stats, 10.0), (False, "", "NOT_EVALUATED"))

    def test_all_alternatives_nan_not_evaluated(self):
        stats = {"SPOC": {"depth_ppm": 1000}, "QLP": {"depth_ppm": np.nan}, "TGLC": {"depth_ppm": np.nan}}
        self.assertEqual(antares_packager.assess_blend(stats, 10.0), (False, "", "NOT_EVALUATED"))

    def test_spoc_nan_not_evaluated(self):
        stats = {"SPOC": {"depth_ppm": np.nan}, "QLP": {"depth_ppm": 100}}
        self.assertEqual(antares_packager.assess_blend(stats, 10.0), (False, "", "NOT_EVALUATED"))

    def test_no_alternative_author(self):
        self.assertEqual(antares_packager.assess_blend({"SPOC": {"depth_ppm": 1000}}, 10.0), (False, "", "NOT_EVALUATED"))


if __name__ == "__main__":
    unittest.main()
