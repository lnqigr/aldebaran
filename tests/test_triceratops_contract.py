"""Pinned external-contract checks, not FPP/NFPP calibration or an upgrade policy."""
import inspect
import ast
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import hashlib
import contextlib
import io
import unittest

import numpy as np
import pandas as pd
from scipy.special import ndtr

# Full package import is a SEPARATE FAIL (pytransit2.2.0 / scipy1.17.1 trapz).
# Execute the unchanged installed method ASTs for bounded source-contract controls;
# never patch installed packages or represent this as a working Bayesian runtime.
_DIST = importlib.metadata.distribution("triceratops")
_SOURCE = Path(_DIST.locate_file("triceratops/triceratops.py"))
_NAMESPACE = {"np": np, "ndtr": ndtr}
_METHODS = {}
for _node in ast.walk(ast.parse(_SOURCE.read_text())):
    if isinstance(_node, ast.FunctionDef) and _node.name in ("calc_depths", "calc_probs"):
        exec(compile(ast.Module(body=[_node], type_ignores=[]), str(_SOURCE), "exec"), _NAMESPACE)
        _METHODS[_node.name] = _NAMESPACE[_node.name]
tr = SimpleNamespace(target=type("InstalledSourceMethodsOnly", (), _METHODS))
_FUNCS = Path(_DIST.locate_file("triceratops/funcs.py"))
_node = next(n for n in ast.walk(ast.parse(_FUNCS.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == "renorm_flux")
exec(compile(ast.Module(body=[_node], type_ignores=[]), str(_FUNCS), "exec"), _NAMESPACE)
renorm_flux = _NAMESPACE["renorm_flux"]
from lightkurve.io.tess import read_tess_lightcurve


class TestInstalledTriceratopsSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("SOURCE_METHOD_CONTROLS_ONLY_FULL_RUNTIME_IMPORT_FAILED",
              hashlib.sha256(_SOURCE.read_bytes()).hexdigest())
        actual = importlib.metadata.version("triceratops")
        if actual != "1.0.21":
            raise RuntimeError("This evidence pins TRICERATOPS1.0.21, installed="+actual+
                               "; re-evaluate version contract, do not silently apply these assertions")

    @staticmethod
    def local_target(fraction):
        # No target.__init__, catalog request, pixel download or TRILEGAL.
        target = object.__new__(tr.target)
        mags = [10.] if fraction == 1. else [10., 10.-2.5*np.log10((1.-fraction)/fraction)]
        target.stars = pd.DataFrame(dict(ID=np.arange(len(mags))+1,Tmag=mags,
                                        mass=1.,rad=1.,Teff=5700.,plx=10.))
        target.pix_coords = [np.zeros((len(mags),2))]
        return target

    def test_actual_signatures_and_lightkurve_reader_default(self):
        depth = inspect.signature(tr.target.calc_depths).parameters
        probs = inspect.signature(tr.target.calc_probs).parameters
        self.assertEqual(list(depth),["self","tdepth","all_ap_pixels"])
        self.assertNotIn("dilution_corrected",probs)
        self.assertEqual(probs["N"].default,1000000)
        self.assertEqual(probs["exptime"].default,0.00139)
        self.assertEqual(probs["nsamples"].default,20)
        self.assertEqual(inspect.signature(read_tess_lightcurve).parameters["flux_column"].default,"pdcsap_flux")
        # Confirms the old API's internal renormalization rather than assuming 1.1.0 defaults.
        self.assertIn("renorm_flux(",inspect.getsource(tr.target.calc_probs))

    def test_actual_calc_depths_fraction_units_and_aperture_fraction(self):
        for fraction in (1., .98, .85, .7):
            target=self.local_target(fraction)
            with contextlib.redirect_stdout(io.StringIO()):
                target.calc_depths(tdepth=.001,all_ap_pixels=[np.array([[0.,0.]])])
            actual=target.stars.iloc[0]
            self.assertAlmostEqual(actual["fluxratio"],fraction,places=12)
            self.assertAlmostEqual(actual["tdepth"],.001/fraction,places=12)
            print("INSTALLED_CALC_DEPTHS",fraction,float(actual["fluxratio"]),float(actual["tdepth"]))
        # Wrong ppm input is rejected by the actual method's depth>1 feasibility rule.
        target=self.local_target(.7)
        target.calc_depths(tdepth=1000.,all_ap_pixels=[np.array([[0.,0.]])])
        self.assertTrue((target.stars["tdepth"]==0).all())

    def test_actual_raw_sap_and_corrected_pdcsap_conventions_differ(self):
        for fraction in (1., .98, .85, .7):
            intrinsic=.001
            aperture=fraction*intrinsic
            raw=self.local_target(fraction)
            raw.calc_depths(tdepth=aperture,all_ap_pixels=[np.array([[0.,0.]])])
            self.assertAlmostEqual(raw.stars.iloc[0]["tdepth"],intrinsic,places=12)
            corrected_flux=np.array([1.,1.-intrinsic])
            actual_flux,actual_error=renorm_flux(corrected_flux,1e-4,fraction)
            self.assertAlmostEqual(1.-actual_flux[1],intrinsic/fraction,places=12)
            self.assertEqual(actual_error,1e-4/fraction)
            print("CORRECTED_INPUT_RENORMALIZED",fraction,1.-actual_flux[1],actual_error)
        # These are real library depth/flux consequences, not a simulated FPP estimate.


if __name__=="__main__":unittest.main()
