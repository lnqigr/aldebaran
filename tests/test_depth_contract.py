"""S3 actual producer/consumer contracts; synthetic/offline, no calibration claim."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import depth_contract as contract
import pleiades
import ctoi_packager as ctoi
import antares_packager as pack
import validate_triceratops as validation
import antares
import merge_shards


class TestDepthContract(unittest.TestCase):
    def test_three_quantities_are_not_aliases(self):
        for k, b in ((0.05, 0.0), (0.1, 0.95)):
            flux = pleiades.exact_annulus_transit_model(np.array([0.0]), k, b, .1, .4, .2)[0]
            fields = contract.fit_depth_fields(k, flux, 1234., True)
            self.assertEqual(fields["geometric_rp_rs_sq_ppm"], k*k*1e6)
            self.assertEqual(fields["model_transit_depth_ppm"], (1-flux)*1e6)
            self.assertEqual(fields["measured_box_depth_ppm"], 1234.)
            self.assertNotEqual(fields["model_transit_depth_ppm"], fields["geometric_rp_rs_sq_ppm"])
            self.assertEqual(contract.photometric_depth(fields), (1234., "measured_box_depth_ppm"))
        # No universal grazing ratio or copied approximate Argon number is imposed.

    def test_legacy_geometric_field_never_substitutes_for_flux(self):
        self.assertEqual(contract.photometric_depth({"fitted_depth_ppm": 10000., "depth_ppm": 1500.}),
                         (1500., "depth_ppm"))
        for row in ({"fitted_depth_ppm": 10000.}, {"depth_ppm": np.nan},
                    {"depth_schema_version": "unknown", "depth_ppm": 1000.}):
            with self.assertRaises(ValueError):
                contract.photometric_depth(row)

    def test_failed_fit_and_legacy_radius_provenance(self):
        fields = contract.fit_depth_fields(.05, .997, 1234., False, "RuntimeError: forced")
        self.assertFalse(fields["fit_converged"])
        self.assertTrue(np.isnan(fields["fitted_rp_rs"]))
        self.assertTrue(np.isnan(fields["model_transit_depth_ppm"]))
        with self.assertRaises(ValueError):
            contract.geometric_radius(dict(fields, fitted_rp_earth=5.45, r_planet_earth=2.0))
        self.assertEqual(contract.geometric_radius({"fitted_rp_earth": 5.45}),
                         (5.45, "legacy_fit_unverified"))
        self.assertEqual(contract.geometric_radius({"r_planet_earth": 2.0}),
                         (2.0, "tier1_box_radius_approximation"))

    def test_output_schema_conflict_is_detected_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"output.csv"
            path.write_text("foreign,columns\n1,2\n")
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                contract.check_output_schema(path, ["tic_id", "depth_ppm"])
            self.assertEqual(path.read_bytes(), before)


class TestPleiadesProducer(unittest.TestCase):
    def run_producer(self, fail=False):
        t = np.linspace(1992., 2008., 1000)
        rng = np.random.default_rng(19020)
        f = 1.0 + rng.normal(0, .0003, len(t))
        phase = ((t-2000.+2.)%4.)-2.
        f[np.abs(phase)<.045] -= .0012
        lc = SimpleNamespace(time=t, flux=f, quality=np.zeros(len(t), dtype=int))
        class Search(list):
            mission = ["TESS Sector 93"]
            def __getitem__(self, key):
                result = super().__getitem__(key)
                return Search(result) if isinstance(key, slice) else result
        search = Search([SimpleNamespace(download=lambda: lc)])
        fit_result = (np.array([.05, .2, .1]), np.eye(3))
        fit = mock.Mock(side_effect=RuntimeError("forced fit failure")) if fail else mock.Mock(return_value=fit_result)
        bls = SimpleNamespace(power=lambda *a, **kw: SimpleNamespace(
            power=np.array([1.]), period=np.array([4.]), transit_time=np.array([2000.]), duration=np.array([.1])))
        gaia = {"neb_capable_stars": 0, "contamination_ratio": 0., "ruwe": 1., "n_neighbors": 0,
                "is_tess_eb": False, "is_gaia_nss": False}
        row = dict(tic_id="123",period_days=4.,t0_btjd=2000.,duration_hours=2.4,
                   depth_ppm=1200.,r_star_sun=1.,m_star_sun=1.,teff_k=5700.)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(pleiades.lk, "search_lightcurve", return_value=search), \
             mock.patch.object(pleiades, "tukey_biweight_detrend", side_effect=lambda t,f,**kw:f), \
             mock.patch.object(pleiades, "BoxLeastSquares", return_value=bls), \
             mock.patch.object(pleiades, "curve_fit", fit), \
             mock.patch.object(pleiades, "query_gaia_contamination", return_value=gaia), \
             mock.patch.object(pleiades, "scan_full_orbit_secondary_eclipse", return_value=(False,0.,0.,.5,4.)), \
             mock.patch.object(pleiades.plt, "savefig"):
            result = pleiades.deep_vet_target(row, vet_dir=directory)
        return result

    def test_actual_producer_success(self):
        row = self.run_producer()
        self.assertTrue(row["fit_converged"])
        self.assertEqual(row["fitted_rp_rs"], .05)
        self.assertAlmostEqual(row["geometric_rp_rs_sq_ppm"], 2500.)
        self.assertEqual(row["fitted_depth_ppm"], 2500.)  # Legacy meaning retained.
        self.assertEqual(row["fitted_depth_kind"], "geometric_k_squared")
        self.assertGreater(row["model_transit_depth_ppm"], row["geometric_rp_rs_sq_ppm"])
        self.assertEqual(row["measured_box_depth_ppm"], 1200.)

    def test_actual_fit_failure_is_not_a_measurement(self):
        row = self.run_producer(fail=True)
        self.assertFalse(row["fit_converged"])
        self.assertIn("forced fit failure", row["fit_error"])
        self.assertEqual(row["fitted_radius_kind"], "box_depth_initial_guess")
        self.assertTrue(np.isnan(row["geometric_rp_rs_sq_ppm"]))
        with self.assertRaises(ValueError):
            contract.geometric_radius(row)

    def test_missing_input_depth_fails_before_download(self):
        with mock.patch.object(pleiades.lk, "search_lightcurve") as query:
            with self.assertRaises(ValueError):
                pleiades.deep_vet_target({"tic_id":"123", "depth_ppm":np.nan})
        query.assert_not_called()


class TestActualPackagers(unittest.TestCase):
    @staticmethod
    def candidate():
        return dict(tic_id="123",period_days=4.,t0_btjd=2000.,duration_hours=2.4,depth_ppm=1200.,
                    r_star_sun=1.,m_star_sun=1.,teff_k=5700.,r_planet_earth=2.0,sector=93)

    @staticmethod
    def dossier():
        return dict(tic_id="123", fit_converged=True, fitted_rp_earth=5.45, fitted_rp_rs=.05,
                    geometric_rp_rs_sq_ppm=2500., model_transit_depth_ppm=3000.,
                    measured_box_depth_ppm=1200., fitted_depth_ppm=2500.,
                    depth_schema_version=contract.DEPTH_SCHEMA, fitted_depth_kind="geometric_k_squared",
                    fitted_duration_hours=2.4,impact_param_b=.2,gaia_nss_type=np.nan,
                    is_known_tess_eb=False,secondary_snr_sigma=0.,sec_eclipse_phase=.5,
                    dur_to_kepler_max=1.,flux_contam_percent=0.,neb_capable_neighbors=0,
                    sectors_stitched="93", vps_score=0.)

    def run_ctoi(self, directory, cand=None, dossier=None):
        d=Path(directory); cand=self.candidate() if cand is None else cand
        dossier=self.dossier() if dossier is None else dossier
        pd.DataFrame([cand]).to_csv(d/"t1.csv",index=False)
        pd.DataFrame([dossier]).to_csv(d/"t2.csv",index=False)
        with mock.patch.multiple(ctoi, RESULTS_DIR=str(d),TIER1_CSV=str(d/"t1.csv"),TIER2_CSV=str(d/"t2.csv"),
                                 RANKING_OUT_CSV=str(d/"prioritized_targets_ranking.csv"),
                                 EXOFOP_OUT_CSV=str(d/"exofop_ctoi_upload.csv")), \
             mock.patch.object(ctoi,"batch_query_tic_photometry",return_value={"123":{
                 "r_star":1.,"m_star":1.,"teff":5700.,"Tmag":10.,"Jmag":9.,"Kmag":8.8}}), \
             mock.patch.object(ctoi,"extract_camera_ccd_from_cache",return_value=("1","2")), \
             contextlib.redirect_stdout(io.StringIO()):
            ctoi.run_ctoi_packager()
        return pd.read_csv(d/"prioritized_targets_ranking.csv")

    def test_actual_ctoi_exports_box_not_geometric_depth(self):
        with tempfile.TemporaryDirectory() as d:
            rank=self.run_ctoi(d)
            upload=pd.read_csv(Path(d)/"exofop_ctoi_upload.csv",sep="|")
        self.assertEqual(upload.iloc[0]["Depth"],1200.)
        self.assertEqual(rank.iloc[0]["depth_ppm"],1200.)
        self.assertEqual(rank.iloc[0]["rp_earth"],5.45)
        self.assertEqual(rank.iloc[0]["radius_source"],"converged_geometric_fit")

    def test_missing_impact_is_not_fabricated_and_model_assumption_is_explicit(self):
        dossier=self.dossier();dossier.pop("impact_param_b")
        with tempfile.TemporaryDirectory() as d:
            rank=self.run_ctoi(d,dossier=dossier)
            upload=pd.read_csv(Path(d)/"exofop_ctoi_upload.csv",sep="|")
        self.assertTrue(np.isnan(rank.iloc[0]["impact_param_b"]))
        self.assertEqual(rank.iloc[0]["impact_parameter_source"],"missing")
        self.assertEqual(rank.iloc[0]["characterization_assumptions"],"legacy_model_b_default_0.3")
        self.assertIn("assumes b=0.3",upload.iloc[0]["User Comments / Disposition"])

    def test_missing_and_failed_fit_never_fabricate_exports(self):
        for field_case in ("missing_depth", "missing_radius", "failed_fit"):
            cand=self.candidate(); dossier=self.dossier()
            if field_case=="missing_depth":
                cand["depth_ppm"]=np.nan; dossier["measured_box_depth_ppm"]=np.nan
            elif field_case=="missing_radius":
                dossier["fitted_rp_earth"]=np.nan; cand["r_planet_earth"]=np.nan
            else:dossier["fit_converged"]=False
            with self.subTest(case=field_case),tempfile.TemporaryDirectory() as d:
                path=Path(d)/"prioritized_targets_ranking.csv";path.write_text("historical\n42\n")
                before=path.read_bytes()
                with self.assertRaises(ValueError):self.run_ctoi(d,cand,dossier)
                self.assertEqual(path.read_bytes(),before)
                self.assertFalse((Path(d)/"exofop_ctoi_upload.csv").exists())

    def test_ctoi_and_antares_keep_geometric_radius_separate_and_outputs_disjoint(self):
        cand=dict(self.candidate(),**self.dossier(),corr_risk=0.,contam_pct=0.,neb_count=0)
        archive=dict(status="ARCHIVAL_PASS",candidate=cand,p_ref=4.,t0_ref=2000.,depth_ppm=800.,
                     stack_author="SPOC",snr_red=10.,snr_white=12.,n_transits=5)
        with tempfile.TemporaryDirectory() as d:
            ctoi_rank=self.run_ctoi(d)
            ctoi_bytes=(Path(d)/"prioritized_targets_ranking.csv").read_bytes()
            with mock.patch.object(pack,"batch_query_tic_photometry",return_value={"123":{
                "r_star":1.,"m_star":1.,"teff":5700.,"Tmag":10.,"Jmag":9.,"Kmag":8.8}}), \
                 mock.patch.object(pack,"extract_camera_ccd_from_cache",return_value=("1","2")), \
                 contextlib.redirect_stdout(io.StringIO()):
                rank,upload=pack.run_stage_3c_packaging([archive],results_dir=d)
            self.assertEqual((Path(d)/"prioritized_targets_ranking.csv").read_bytes(),ctoi_bytes)
            self.assertTrue((Path(d)/"antares_prioritized_targets_ranking.csv").exists())
        self.assertEqual(rank.iloc[0]["rp_earth"],ctoi_rank.iloc[0]["rp_earth"])
        self.assertEqual(rank.iloc[0]["depth_ppm"],800.)
        self.assertEqual(upload.iloc[0]["Depth"],800.)
        self.assertNotAlmostEqual(rank.iloc[0]["rp_earth"],np.sqrt(.0008)*109.076,places=2)

    def test_auto_reader_refuses_ambiguous_geometric_column(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"geometry.csv"
            pd.DataFrame([{"TIC ID":123,"Period":4.,"Epoch BJD":2459000.,
                           "geometric_rp_rs_sq_ppm":2500.,"Duration":2.4}]).to_csv(path,index=False)
            with self.assertRaises(ValueError),mock.patch.object(validation,"run_triceratops_validation") as run:
                validation.validate_auto(top_csv=str(path),results_dir=d)
            run.assert_not_called()


class TestAntaresFitProvenance(unittest.TestCase):
    def test_actual_failed_fit_summary_is_truthful(self):
        t=np.linspace(1992.,2008.,1200)
        f=1.+np.random.default_rng(20).normal(0,.0002,len(t))
        phase=((t-2000.+2.)%4.)-2.
        f[np.abs(phase)<.045]-=.0012
        item=SimpleNamespace(author=["SPOC"],mission=["TESS Sector 93"],exptime=[120.],download=lambda:None)
        star=dict(r_star=1.,m_star=1.,teff=5700.,tmag=10.,u1=.4,u2=.2)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(antares,"RESULTS_DIR",d), \
             mock.patch.object(antares,"get_stellar_limb_darkening",return_value=star), \
             mock.patch.object(antares.lk,"search_lightcurve",return_value=[item]), \
             mock.patch.object(antares,"extract_multi_author_lc",return_value=(t,f,np.full_like(t,.0002),120/86400)), \
             mock.patch.object(antares,"hybrid_tukey_savgol_detrend",side_effect=lambda t,f,*a,**kw:f), \
             mock.patch.object(antares,"refine_multi_year_ephemeris",return_value=(4.,2000.,np.array([4.]),np.array([10.]))), \
             mock.patch.object(antares,"curve_fit",side_effect=RuntimeError("forced Antares failure")), \
             mock.patch.object(antares.plt,"savefig"):
            antares.run_antares_detective("123",4.,2000.,2.4)
            table=pd.read_csv(Path(d)/"antares_summary_123.csv")
        master=table[table["group_type"]=="MASTER_DEDUPLICATED_ALL_YEARS"].iloc[0]
        self.assertFalse(master["fit_converged"])
        self.assertIn("forced Antares failure",master["fit_error"])
        self.assertTrue(np.isnan(master["fitted_rp_earth"]))
        self.assertGreater(master["legacy_initial_radius_earth"],0.)
        self.assertIn("NOT_CALIBRATED",master["radius_uncertainty_method"])

    def test_actual_catalog_fallback_is_explicit_without_changing_model_values(self):
        with mock.patch.object(antares.Catalogs,"query_criteria",side_effect=RuntimeError("offline catalog fault")):
            info=antares.get_stellar_limb_darkening("123")
        self.assertEqual(info["r_star"],.61)
        self.assertEqual(info["m_star"],.60)
        self.assertEqual(info["stellar_parameter_sources"]["r_star"],"fixed_initial_assumption")
        self.assertIn("offline catalog fault",info["catalog_error"])
        self.assertEqual(set(info["stellar_parameter_sources"]),{"r_star","m_star","teff","tmag"})

    def test_observe_native_exposure_branch_without_repairing_it(self):
        phase=np.linspace(-.06,.06,601)
        instantaneous=antares.annulus_300_transit_model(phase,.05,.5,.06,.4,.2,exp_time_days=0.)
        short=antares.annulus_300_transit_model(phase,.05,.5,.06,.4,.2,exp_time_days=120./86400.)
        long=antares.annulus_300_transit_model(phase,.05,.5,.06,.4,.2,exp_time_days=180./86400.)
        np.testing.assert_array_equal(short,instantaneous)
        self.assertGreater(float(np.max(np.abs(long-instantaneous))),0.)
        print("NATIVE_EXPOSURE_120S_VS_0_MAX_PPM",float(np.max(np.abs(short-instantaneous)))*1e6)
        print("NATIVE_EXPOSURE_180S_VS_0_MAX_PPM",float(np.max(np.abs(long-instantaneous)))*1e6)


class TestProducerOrchestration(unittest.TestCase):
    def test_legacy_ctoi_file_does_not_fake_antares_success(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            Path(d,"exofop_ctoi_upload.csv").write_text("TIC ID|Depth\n123|1200\n")
            with mock.patch.object(merge_shards.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="",stderr="")):
                with self.assertRaises(merge_shards.StageExecutionError) as error:
                    merge_shards.orchestrate_downstream_stages(
                        pd.DataFrame([{"tic_id":"123"}]),d,run_pleiades=False,run_antares=True)
            self.assertIn("antares_exofop_ctoi_upload.csv",str(error.exception))

    def test_fresh_antares_paths_and_tier4_never_select_stale_ctoi(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            Path(d,"exofop_ctoi_upload_top3.csv").write_text("TIC ID,Period,Epoch BJD,Depth,Duration\n999,4,2459000,9999,2\n")
            def run(command,**kwargs):
                self.assertIn("antares_packager.py",command)
                Path(d,"antares_exofop_ctoi_upload.csv").write_text("TIC ID|Depth\n")
                Path(d,"antares_prioritized_targets_ranking.csv").write_text("tic_id,period_days,epoch_bjd,depth_ppm,duration_hours\n")
                return SimpleNamespace(returncode=0,stdout="",stderr="")
            with mock.patch.object(merge_shards.subprocess,"run",side_effect=run) as invocation:
                stages=merge_shards.orchestrate_downstream_stages(
                    pd.DataFrame([{"tic_id":"123"}]),d,run_pleiades=False,run_antares=True,run_triceratops=True)
            self.assertEqual(invocation.call_count,1)
            self.assertEqual(stages[-1]["status"],"SKIPPED_ZERO_TARGETS")
            completed=next(x for x in stages if x.get("stage")=="antares" and x.get("status")=="COMPLETED")
            self.assertTrue(completed["output_csv"].endswith("antares_exofop_ctoi_upload.csv"))


if __name__=="__main__":unittest.main()
