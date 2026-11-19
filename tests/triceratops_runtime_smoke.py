"""Bounded real-library smoke and dilution comparison for TRICERATOPS 1.0.21."""

import csv
import importlib.metadata
import json
from math import erf, sqrt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import triceratops.triceratops as tr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPECTED_VERSIONS = {
    "numpy": "1.23.5",
    "scipy": "1.13.1",
    "astropy": "6.1.7",
    "lightkurve": "2.6.0",
    "triceratops": "1.0.21",
    "PyTransit": "2.2.0",
}


def _assert_runtime_versions():
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if actual != EXPECTED_VERSIONS:
        raise RuntimeError(
            f"Runtime smoke requires the exact pinned profile; expected "
            f"{EXPECTED_VERSIONS}, found {actual}."
        )


def _target_for_dilution(target_fraction, trilegal_file):
    target_pixel_flux = (
        (0.5 * (1 + erf(0.5 / (0.75 * sqrt(2))))
         - 0.5 * (1 + erf(-0.5 / (0.75 * sqrt(2)))))
        ** 2
    )
    neighbor_pixel_flux = (
        (0.5 * (1 + erf(-0.5 / (0.75 * sqrt(2))))
         - 0.5 * (1 + erf(-1.5 / (0.75 * sqrt(2)))))
        * target_pixel_flux ** 0.5
    )
    neighbor_relative_flux = (
        target_pixel_flux * (1 - target_fraction)
        / (target_fraction * neighbor_pixel_flux)
    )
    neighbor_tmag = 10.0 - 2.5 * np.log10(neighbor_relative_flux)

    obj = tr.target.__new__(tr.target)
    obj.ID = 100000001
    obj.mission = "TESS"
    obj.sectors = np.array([1], dtype=int)
    obj.trilegal_fname = str(trilegal_file)
    obj.pix_coords = [np.array([[0.0, 0.0], [1.0, 0.0]])]
    obj.stars = pd.DataFrame(
        [
            [100000001, 10.0, 9.0, 8.8, 8.7, 12.0, 20.0, 1.0, 1.0, 5777.0, 10.0],
            [100000002, neighbor_tmag, 10.0, 9.8, 9.7, 12.0001, 20.0001, 0.8, 0.8, 5000.0, 9.9],
        ],
        columns=["ID", "Tmag", "Jmag", "Hmag", "Kmag", "ra", "dec",
                 "mass", "rad", "Teff", "plx"],
    )
    return obj


def _run_case(target_fraction, trilegal_file):
    intrinsic_depth = 0.001
    sap_depth = intrinsic_depth * target_fraction
    time = np.linspace(-0.12, 0.12, 121)
    sap_flux = np.where(np.abs(time) <= 0.02, 1.0 - sap_depth, 1.0)

    obj = _target_for_dilution(target_fraction, trilegal_file)
    obj.calc_depths(
        tdepth=sap_depth,
        all_ap_pixels=[np.array([[0, 0]], dtype=int)],
    )
    actual_fraction = float(obj.stars.loc[0, "fluxratio"])
    actual_host_depth = float(obj.stars.loc[0, "tdepth"])
    if not np.isclose(actual_fraction, target_fraction, rtol=0, atol=1e-8):
        raise AssertionError(f"modelled dilution {actual_fraction} != requested {target_fraction}")
    if not np.isclose(actual_host_depth, intrinsic_depth, rtol=0, atol=1e-10):
        raise AssertionError(f"host depth {actual_host_depth} != {intrinsic_depth}")

    np.random.seed(41021)
    obj.calc_probs(
        time=time,
        flux_0=sap_flux,
        flux_err_0=1e-4,
        P_orb=1.0,
        N=32,
        parallel=False,
        verbose=0,
    )
    if not np.isfinite(obj.FPP) or not np.isfinite(obj.NFPP):
        raise AssertionError(f"non-finite probabilities: FPP={obj.FPP}, NFPP={obj.NFPP}")
    if obj.FPP_degenerate:
        raise AssertionError("TRICERATOPS marked the bounded scenario calculation degenerate.")
    if len(obj.probs) != 18 or not np.isclose(obj.probs["prob"].sum(), 1.0):
        raise AssertionError("Expected 18 normalized scenario probabilities.")
    return {
        "requested_target_flux_fraction": target_fraction,
        "measured_target_flux_fraction": actual_fraction,
        "input_product": "SAP",
        "sap_box_depth_ppm": sap_depth * 1e6,
        "FPP": float(obj.FPP),
        "NFPP": float(obj.NFPP),
        "probability_sum": float(obj.probs["prob"].sum()),
        "scenario_rows": len(obj.probs),
    }


def main():
    _assert_runtime_versions()
    with tempfile.TemporaryDirectory(prefix="triceratops-runtime-") as directory:
        trilegal_file = Path(directory) / "trilegal.csv"
        with trilegal_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Mact", "logg", "logTe", "[M/H]", "TESS", "J", "H", "Ks"])
            writer.writerow([0.8, 4.5, 3.70, 0.0, 12.0, 11.0, 10.8, 10.7])
            writer.writerow([0.7, 4.6, 3.60, 0.0, 13.0, 12.0, 11.8, 11.7])
            writer.writerow([0.7, 4.6, 3.60, 0.0, 15.0, 14.0, 13.8, 13.7])
            writer.writerow([0.6, 4.7, 3.50, 0.0, 17.0, 16.0, 15.8, 15.7])
        results = [
            _run_case(0.7, trilegal_file),
            _run_case(0.9, trilegal_file),
        ]
        results[1]["delta_FPP_vs_0.70"] = results[1]["FPP"] - results[0]["FPP"]
        results[1]["delta_NFPP_vs_0.70"] = results[1]["NFPP"] - results[0]["NFPP"]
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
