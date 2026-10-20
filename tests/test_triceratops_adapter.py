import unittest
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from unittest import mock
import types
import pandas as pd
import validate_triceratops as adapter
from types import SimpleNamespace

import numpy as np

from validate_triceratops import (
    _download_spoc_sector,
    _pipeline_aperture_pixels,
    _sap_flux_only,
)


class _FakeTPF:
    pipeline_mask = np.array([[False, True, False], [True, True, False]])
    column = 100
    row = 200
    sector = 4


class _FakeLightCurve:
    sector = 4
    meta = {"CROWDSAP": 0.7}
    flux = SimpleNamespace(value=np.array([100.0, 99.0, 100.0]))
    flux_err = SimpleNamespace(value=np.array([0.1, 0.1, 0.1]))
    time = SimpleNamespace(btjd=np.array([1.0, 2.0, 3.0]))

    def remove_nans(self):
        return self

    def remove_outliers(self, sigma):
        return self


class _FakeSearchResult:
    def __init__(self, exposures, products):
        self.table = {"exptime": np.asarray(exposures)}
        self.products = products

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        product = self.products[index]
        def download(**kwargs):
            return product.download(**kwargs) if hasattr(product, "download") else product

        return SimpleNamespace(download=download)


class _FailingProduct:
    def download(self, **kwargs):
        if kwargs.get("flux_column") != "sap_flux":
            raise AssertionError("The adapter must request SAP explicitly.")
        raise KeyError("sap_flux missing")


class TriceratopsAdapterTests(unittest.TestCase):
    def test_uses_pipeline_aperture_in_global_tess_pixel_coordinates(self):
        actual = _pipeline_aperture_pixels(_FakeTPF())
        np.testing.assert_array_equal(actual, [[101, 200], [100, 201], [101, 201]])

    def test_sap_flux_is_unchanged(self):
        flux = np.array([100.0, 99.0, 100.0])
        np.testing.assert_array_equal(_sap_flux_only(flux, "sap_flux"), flux)

    def test_pdcsap_is_rejected_without_approved_conversion(self):
        with self.assertRaisesRegex(ValueError, "not enabled without an approved"):
            _sap_flux_only([1.0, 0.99], "pdcsap_flux")

    def test_rejects_empty_pipeline_aperture(self):
        invalid_tpf = _FakeTPF()
        invalid_tpf.pipeline_mask = np.zeros((2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "aperture is empty"):
            _pipeline_aperture_pixels(invalid_tpf)

    def test_download_pairs_cadence_and_uses_sap_with_pipeline_aperture(self):
        lc = _FakeLightCurve()
        tpf = _FakeTPF()
        lc_results = _FakeSearchResult([20.0, 120.0], [_FailingProduct(), lc])
        tpf_results = _FakeSearchResult([20.0, 120.0], [object(), tpf])
        lk = SimpleNamespace(
            search_lightcurve=lambda **kwargs: lc_results,
            search_targetpixelfile=lambda **kwargs: tpf_results,
        )

        times, flux, errors, pixels = _download_spoc_sector(
            lk, 123, 4, "sap_flux"
        )

        np.testing.assert_array_equal(times, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(flux, [100.0, 99.0, 100.0])
        np.testing.assert_allclose(errors, [0.1, 0.1, 0.1])
        np.testing.assert_array_equal(pixels, [[101, 200], [100, 201], [101, 201]])

    def test_pdcsap_request_fails_before_any_product_search(self):
        def unexpected_search(**kwargs):
            raise AssertionError("PDCSAP request must not search or substitute products.")

        lk = SimpleNamespace(
            search_lightcurve=unexpected_search,
            search_targetpixelfile=unexpected_search,
        )
        with self.assertRaisesRegex(ValueError, "no automatic product substitution"):
            _download_spoc_sector(lk, 123, 4, "pdcsap_flux")

    def test_missing_sap_fails_without_pdcsap_fallback(self):
        lc_search = _FakeSearchResult([120.0], [_FailingProduct()])
        tpf_search = _FakeSearchResult([120.0], [_FakeTPF()])
        lk = SimpleNamespace(
            search_lightcurve=lambda **kwargs: lc_search,
            search_targetpixelfile=lambda **kwargs: tpf_search,
        )
        with self.assertRaisesRegex(RuntimeError, "no PDCSAP fallback was attempted"):
            _download_spoc_sector(lk, 123, 4, "sap_flux")

    def test_download_rejects_unmatched_lc_and_tpf_cadence(self):
        lk = SimpleNamespace(
            search_lightcurve=lambda **kwargs: _FakeSearchResult([20.0], [object()]),
            search_targetpixelfile=lambda **kwargs: _FakeSearchResult([120.0], [object()]),
        )
        with self.assertRaisesRegex(ValueError, "No cadence-matched"):
            _download_spoc_sector(lk, 123, 4, "sap_flux")


class IntegrationContractTests(unittest.TestCase):
    def test_unsupported_cadence_fails_instead_of_approximating_exposure(self):
        for exposure in (20.0, 1800.0, 0.00139):
            with self.subTest(exposure=exposure):
                lk = SimpleNamespace(
                    search_lightcurve=lambda **kw: _FakeSearchResult([exposure], [object()]),
                    search_targetpixelfile=lambda **kw: _FakeSearchResult([exposure], [object()]))
                with self.assertRaisesRegex(ValueError, "No supported 120-second"):
                    _download_spoc_sector(lk, 123, 4, "sap_flux")

    def test_wrong_downloaded_sector_and_ambiguous_products_fail(self):
        lc = _FakeLightCurve()
        lc.sector = 5
        lk = SimpleNamespace(
            search_lightcurve=lambda **kw: _FakeSearchResult([120.0], [lc]),
            search_targetpixelfile=lambda **kw: _FakeSearchResult([120.0], [_FakeTPF()]))
        with self.assertRaisesRegex(ValueError, "light-curve sector"):
            _download_spoc_sector(lk, 123, 4, "sap_flux")
        lk.search_lightcurve = lambda **kw: _FakeSearchResult([120., 120.], [lc, lc])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            _download_spoc_sector(lk, 123, 4, "sap_flux")

    @staticmethod
    def sector_data(baseline, depth, dense_inside):
        centers = np.linspace(-.2, .2, 61)[:-1] + .4 / 120
        time, flux = [], []
        for center in centers:
            inside = abs(center) <= .05
            count = 9 if inside == dense_inside else 3
            time.extend([2000.0 + center] * count)
            flux.extend([baseline * (1.0 - depth if inside else 1.0)] * count)
        return (np.array(time), np.array(flux), np.full(len(time), baseline * .001),
                np.array([[100, 200]]))

    def run_profile(self, baseline1, baseline2, depth):
        # Actual production folding/binning/normalization; only external I/O and MC are seams.
        target = mock.Mock()
        target.sectors = np.array([4, 5])
        target.stars = pd.DataFrame([{"rad": 1., "Teff": 5700.}])
        target.FPP, target.NFPP = .5, .1
        target.probs = pd.DataFrame({"scenario": ["TP", "EB"], "prob": [.5, .5]})
        package = types.ModuleType("triceratops")
        module = types.ModuleType("triceratops.triceratops")
        module.target = mock.Mock(return_value=target)
        package.triceratops = module
        parts = [self.sector_data(baseline1, depth, True),
                 self.sector_data(baseline2, depth, False)]
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.dict(sys.modules, {"triceratops": package, "triceratops.triceratops": module}), \
             mock.patch.object(adapter, "version", return_value="1.0.21"), \
             mock.patch.object(adapter, "_download_spoc_sector", side_effect=parts), \
             mock.patch.object(adapter.plt, "savefig"), \
             contextlib.redirect_stdout(io.StringIO()):
            result = adapter.run_triceratops_validation(123, 4., 2000., 2.4, 1200., [4, 5],
                                                        n_mc=1, results_dir=out)
            if result:
                saved = pd.read_csv(Path(out)/"triceratops/triceratops_probs_TIC_123.csv")
                self.assertIn("SCIENTIFIC_REVIEW_REQUIRED", saved.input_contract_json.iloc[0])
        return result, target

    def test_flat_sectors_cannot_manufacture_a_half_depth(self):
        result, target = self.run_profile(100., 200., 0.)
        self.assertIs(result, False)
        target.calc_depths.assert_not_called()
        target.calc_probs.assert_not_called()

    def test_true_sap_depth_is_scale_invariant_and_not_reference_depth(self):
        reference, first = self.run_profile(100., 100., .01)
        changed, second = self.run_profile(100., 200., .01)
        for result, target in ((reference, first), (changed, second)):
            self.assertTrue(result)
            self.assertAlmostEqual(target.calc_depths.call_args.kwargs["tdepth"], .01)
            self.assertNotEqual(target.calc_depths.call_args.kwargs["tdepth"], .0012)
            self.assertEqual(len(target.calc_depths.call_args.kwargs["all_ap_pixels"]), 2)
            self.assertEqual(result["input_contract"]["depth_estimator"],
                             "median_binned_full_duration_provisional")
        np.testing.assert_allclose(first.calc_probs.call_args.kwargs["flux_0"],
                                   second.calc_probs.call_args.kwargs["flux_0"], rtol=0, atol=1e-15)

    def test_normalization_scales_errors_and_requires_oot_evidence(self):
        time, flux, errors, _ = self.sector_data(200., .01, True)
        f, e, baseline = adapter._normalize_sap_sector(time, flux, errors, 2000., 4., .1)
        self.assertEqual(baseline, 200.)
        np.testing.assert_allclose(e, .001)
        self.assertAlmostEqual(min(f), .99)
        for flux in (np.zeros(3), np.array([100., np.nan, 100.])):
            with self.assertRaises(ValueError):
                adapter._normalize_sap_sector(np.array([1999.85, 2000., 2000.15]),
                                               flux, np.ones(3), 2000., 4., .1)
        with self.assertRaisesRegex(ValueError, "no out-of-transit"):
            adapter._normalize_sap_sector(np.array([2000.]), np.ones(1), np.ones(1), 2000., 4., .1)

    def test_old_raw_concatenation_mutation_is_detected(self):
        # Actual new production regression must fail if pre-combination scaling is removed.
        with mock.patch.object(adapter, "_normalize_sap_sector",
                               side_effect=lambda t, f, e, *a: (f, e, 1.)):
            with self.assertRaises(AssertionError):
                self.test_flat_sectors_cannot_manufacture_a_half_depth()

    def test_wrong_cadence_unit_mutation_is_detected(self):
        with mock.patch.object(adapter, "SPOC_EXPOSURE_SECONDS", 0.00139):
            with self.assertRaises(ValueError):
                # Directly execute the positive control; wrong unit rejects real-second inputs.
                TriceratopsAdapterTests().test_download_pairs_cadence_and_uses_sap_with_pipeline_aperture()

    def auto_fixture(self, directory, prefix="antares_", sectors="[93]", top_sectors=None):
        top = Path(directory)/(prefix+"exofop_ctoi_upload_top3.csv")
        row = {"TIC ID": 123, "Period": 4., "Epoch": 2459000.,
               "depth_schema_version": "depth_contract_v1", "Depth": 1200., "Duration": 2.4}
        if top_sectors is not None:
            row["sectors"] = top_sectors
        pd.DataFrame([row]).to_csv(top, index=False)
        pd.DataFrame([{"tic_id": 123, "sectors": sectors}]).to_csv(
            Path(directory)/(prefix+"prioritized_targets_ranking.csv"), index=False)
        return top

    def test_auto_uses_explicit_photometric_depth_and_matching_producer_ranking(self):
        with tempfile.TemporaryDirectory() as out:
            top = self.auto_fixture(out)
            pd.DataFrame([{"tic_id": 123, "sectors": "[100]"}]).to_csv(
                Path(out)/"prioritized_targets_ranking.csv", index=False)
            with mock.patch.object(adapter, "run_triceratops_validation") as run:
                adapter.validate_auto(top_csv=str(top), results_dir=out)
            self.assertEqual(run.call_args.kwargs["sectors"], [93])
            self.assertEqual(run.call_args.kwargs["depth_ppm"], 1200.)
            self.assertEqual(run.call_args.kwargs["t0_btjd"], 2000.)

    def test_auto_missing_conflicting_or_malformed_sectors_fail_before_dispatch(self):
        for value, top_value in ((None, None), ("[93]", "[100]"), ("garbage93", None)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as out:
                top = self.auto_fixture(out, top_sectors=top_value)
                ranking = Path(out)/"antares_prioritized_targets_ranking.csv"
                if value is None:
                    ranking.unlink()
                else:
                    pd.DataFrame([{"tic_id": 123, "sectors": value}]).to_csv(ranking, index=False)
                with mock.patch.object(adapter, "run_triceratops_validation") as run:
                    with self.assertRaises(ValueError):
                        adapter.validate_auto(top_csv=str(top), results_dir=out)
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
