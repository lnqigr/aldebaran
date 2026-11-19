"""The one place where the harness reaches the production light-curve path.

Nothing here re-implements a production step. production_flat() calls, in the production order,
aldebaran.extract_and_stitch_arrays (quality 0, finite positive flux, 0.10 d edge trim, normalisation),
aldebaran.clip_upper_outliers (upper 3 MAD clip) and aldebaran.piecewise_savgol_detrend (window up to 601,
order 2), on a light curve object or on raw file arrays wrapped in ArrayLightCurve.
"""
import os

import numpy as np
from astropy.io import fits


def load_production():
    """Import the production detector module with the offline and headless settings the harness always uses."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("ALDEBARAN_OFFLINE", "1")
    import aldebaran
    return aldebaran


class ArrayLightCurve:
    """Raw file arrays (TIME, PDCSAP_FLUX, QUALITY) in the shape extract_and_stitch_arrays reads. Flux errors and
    centroids are absent (NaN): the callers use only the time and flux that the production function returns."""

    def __init__(self, time, flux, quality, crowdsap=None):
        n = len(time)
        self.time = np.asarray(time, float)
        self.flux = np.asarray(flux, float)
        self.flux_err = np.full(n, np.nan)
        self.quality = np.asarray(quality, int)
        self.mom_centr1 = np.full(n, np.nan)
        self.mom_centr2 = np.full(n, np.nan)
        self.pos_corr1 = np.full(n, np.nan)
        self.pos_corr2 = np.full(n, np.nan)
        self.meta = {} if crowdsap is None else {"CROWDSAP": float(crowdsap)}


def read_raw(path):
    """TIME, PDCSAP_FLUX and QUALITY of a SPOC light curve file, exactly as stored."""
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        return np.asarray(d["TIME"], float), np.asarray(d["PDCSAP_FLUX"], float), np.asarray(d["QUALITY"], int)


def main_search_arrays(ald, lightcurve):
    """Discovery arrays after extraction and the upper clip, before any peak: (t, f, ferr, crowdsap)."""
    t, f, ferr, _dx, _dy, crowdsap = ald.extract_and_stitch_arrays([lightcurve], edge_trim_days=0.10)
    t, f, ferr = ald.clip_upper_outliers(t, f, ferr)
    return t, f, ferr, crowdsap


def production_flat(ald, lightcurve):
    """(t, f / trend) on the production discovery path, no known-signal mask."""
    t, f, _ferr, _crowd = main_search_arrays(ald, lightcurve)
    trend = ald.piecewise_savgol_detrend(t, f, mask=None, max_win=601)
    return t, f / trend
