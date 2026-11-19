"""Synthetic SPOC-format products for the harness tests and the fixture dry run.

ENGINEERING ONLY. These files have the structure that lightkurve and the production reader expect (2-minute light
curve with PDCSAP flux, centroids and quality flags; target pixel file with a pipeline aperture), filled with white
noise around a Gaussian PSF. They carry no astrophysical realism and no number derived from them is a measurement.
"""
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

from run_identity import sha256_file  # one copy; same digest

CADENCE_D = 2.0 / 1440.0
SECTOR_DAYS = 27.0
BTJD_SECTOR0 = 1325.0          # start of the fixture's sector numbering, BTJD
ORBIT_GAP_D = 1.0


def sector_times(sector, n_segments=2, gap_days=ORBIT_GAP_D):
    """Cadence times (BTJD) and cadence numbers of one fixture sector: n_segments equal blocks separated by gaps."""
    start = BTJD_SECTOR0 + SECTOR_DAYS * int(sector)
    n_total = int(round((SECTOR_DAYS - gap_days * (n_segments - 1)) / CADENCE_D))
    per = n_total // n_segments
    times, cads, c0 = [], [], int(sector) * 100000
    for j in range(n_segments):
        seg_start = start + j * (per * CADENCE_D + gap_days)
        times.append(seg_start + CADENCE_D * np.arange(per))
        first = c0 + int(round((seg_start - start) / CADENCE_D))
        cads.append(first + np.arange(per))
    return np.concatenate(times), np.concatenate(cads).astype(np.int32)


def _primary(tic_id, sector, creator):
    hdr = fits.Header()
    hdr["TELESCOP"] = "TESS"
    hdr["INSTRUME"] = "TESS Photometer"
    hdr["OBJECT"] = f"TIC {int(tic_id)}"
    hdr["TICID"] = int(tic_id)
    hdr["SECTOR"] = int(sector)
    hdr["CAMERA"] = 1
    hdr["CCD"] = 1
    hdr["ORIGIN"] = "NASA/Ames"
    hdr["CREATOR"] = creator
    hdr["PROCVER"] = "m_harness-fixture"
    hdr["RA_OBJ"] = 120.0
    hdr["DEC_OBJ"] = -30.0
    hdr["TESSMAG"] = 10.0
    hdr["FIXTURE"] = (True, "synthetic engineering fixture, not flight data")
    return fits.PrimaryHDU(header=hdr)


def _time_keys(hdr):
    hdr["TELESCOP"] = "TESS"
    hdr["BJDREFI"] = 2457000
    hdr["BJDREFF"] = 0.0
    hdr["TIMEUNIT"] = "d"
    hdr["TIMESYS"] = "TDB"
    hdr["EXPOSURE"] = 20.0
    hdr["INT_TIME"] = 1.98
    hdr["FRAMETIM"] = 2.0
    hdr["NUM_FRM"] = 60
    hdr["TIMEDEL"] = CADENCE_D


def write_lightcurve(path, tic_id, sector, time, flux, flux_err, quality=None, cadenceno=None, crowdsap=0.95,
                     centroid_xy=None, flux_scale=1.0e4):
    """One SPOC-format 2-minute light curve file. flux is relative (about 1.0) and is stored times flux_scale."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, float)
    n = time.size
    quality = np.zeros(n, np.int32) if quality is None else np.asarray(quality, np.int32)
    cadenceno = np.arange(n, dtype=np.int32) if cadenceno is None else np.asarray(cadenceno, np.int32)
    f = np.asarray(flux, float) * flux_scale
    e = np.asarray(flux_err, float) * flux_scale
    cx, cy = centroid_xy if centroid_xy is not None else (np.full(n, 5.0), np.full(n, 5.0))
    cols = [
        fits.Column("TIME", "D", unit="BJD - 2457000, days", array=time),
        fits.Column("TIMECORR", "E", unit="d", array=np.zeros(n, np.float32)),
        fits.Column("CADENCENO", "J", array=cadenceno),
        fits.Column("SAP_FLUX", "E", unit="e-/s", array=f.astype(np.float32)),
        fits.Column("SAP_FLUX_ERR", "E", unit="e-/s", array=e.astype(np.float32)),
        fits.Column("SAP_BKG", "E", unit="e-/s", array=np.full(n, 100.0, np.float32)),
        fits.Column("SAP_BKG_ERR", "E", unit="e-/s", array=np.full(n, 1.0, np.float32)),
        fits.Column("PDCSAP_FLUX", "E", unit="e-/s", array=f.astype(np.float32)),
        fits.Column("PDCSAP_FLUX_ERR", "E", unit="e-/s", array=e.astype(np.float32)),
        fits.Column("QUALITY", "J", array=quality),
        fits.Column("PSF_CENTR1", "D", unit="pixel", array=np.full(n, np.nan)),
        fits.Column("PSF_CENTR1_ERR", "E", unit="pixel", array=np.full(n, np.nan, np.float32)),
        fits.Column("PSF_CENTR2", "D", unit="pixel", array=np.full(n, np.nan)),
        fits.Column("PSF_CENTR2_ERR", "E", unit="pixel", array=np.full(n, np.nan, np.float32)),
        fits.Column("MOM_CENTR1", "D", unit="pixel", array=np.asarray(cx, float)),
        fits.Column("MOM_CENTR1_ERR", "E", unit="pixel", array=np.full(n, 1e-3, np.float32)),
        fits.Column("MOM_CENTR2", "D", unit="pixel", array=np.asarray(cy, float)),
        fits.Column("MOM_CENTR2_ERR", "E", unit="pixel", array=np.full(n, 1e-3, np.float32)),
        fits.Column("POS_CORR1", "E", unit="pixel", array=np.zeros(n, np.float32)),
        fits.Column("POS_CORR2", "E", unit="pixel", array=np.zeros(n, np.float32)),
    ]
    table = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    _time_keys(table.header)
    table.header["OBJECT"] = f"TIC {int(tic_id)}"
    table.header["TICID"] = int(tic_id)
    table.header["SECTOR"] = int(sector)
    table.header["CROWDSAP"] = float(crowdsap)
    table.header["FLFRCSAP"] = 0.9
    table.header["CDPP0_5"] = 100.0
    table.header["CDPP1_0"] = 80.0
    table.header["CDPP2_0"] = 60.0
    aperture = fits.ImageHDU(_aperture_bits(), name="APERTURE")
    fits.HDUList([_primary(tic_id, sector, "m_harness LightCurveExporterPipelineModule"), table, aperture]).writeto(
        path, overwrite=True, checksum=False)
    return path


def _aperture_bits(shape=(9, 9), radius=1.6):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    inside = (yy - shape[0] // 2) ** 2 + (xx - shape[1] // 2) ** 2 <= radius ** 2
    return np.where(inside, 3, 1).astype(np.int32)      # bit 1: collected; bit 2: in the optimal aperture


def write_tpf(path, tic_id, sector, time, rel_flux, quality=None, cadenceno=None, shape=(9, 9), total=1.0e4,
              pixel_sigma=2.0, seed=0, offset_source=None):
    """One SPOC-format target pixel file: Gaussian PSF at the centre whose total follows rel_flux, plus white noise.
    Only the columns the reader and the pixel gate use are written (no raw counts or background cubes), on a small
    9 x 9 stamp, to keep the fixtures at about 12 MB each.

    offset_source = (row, col, rel_flux_array) adds a second PSF away from the target, for off-target controls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, float)
    n = time.size
    rng = np.random.default_rng(seed)
    quality = np.zeros(n, np.int32) if quality is None else np.asarray(quality, np.int32)
    cadenceno = np.arange(n, dtype=np.int32) if cadenceno is None else np.asarray(cadenceno, np.int32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]

    def psf(row, col):
        p = np.exp(-0.5 * ((yy - row) ** 2 + (xx - col) ** 2) / 1.0 ** 2)
        return p / p.sum()
    cube = total * np.asarray(rel_flux, float)[:, None, None] * psf(shape[0] // 2, shape[1] // 2)[None, :, :]
    if offset_source is not None:
        row, col, rel = offset_source
        cube = cube + total * np.asarray(rel, float)[:, None, None] * psf(row, col)[None, :, :]
    cube = cube + rng.normal(0.0, pixel_sigma, cube.shape)
    dim = f"({shape[1]},{shape[0]})"
    npix = shape[0] * shape[1]
    cols = [
        fits.Column("TIME", "D", unit="BJD - 2457000, days", array=time),
        fits.Column("TIMECORR", "E", unit="d", array=np.zeros(n, np.float32)),
        fits.Column("CADENCENO", "J", array=cadenceno),
        fits.Column("FLUX", f"{npix}E", unit="e-/s", dim=dim, array=cube.astype(np.float32)),
        fits.Column("FLUX_ERR", f"{npix}E", unit="e-/s", dim=dim, array=np.full(cube.shape, pixel_sigma, np.float32)),
        fits.Column("QUALITY", "J", array=quality),
        fits.Column("POS_CORR1", "E", unit="pixel", array=np.zeros(n, np.float32)),
        fits.Column("POS_CORR2", "E", unit="pixel", array=np.zeros(n, np.float32)),
    ]
    table = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    _time_keys(table.header)
    table.header["OBJECT"] = f"TIC {int(tic_id)}"
    table.header["TICID"] = int(tic_id)
    table.header["SECTOR"] = int(sector)
    for idx in (4, 5):                                        # image columns (FLUX, FLUX_ERR): a plain pixel grid
        table.header[f"1CRV{idx}P"] = 1
        table.header[f"2CRV{idx}P"] = 1
        table.header[f"1CRPX{idx}"] = 1.0
        table.header[f"2CRPX{idx}"] = 1.0
    aperture = fits.ImageHDU(_aperture_bits(shape), name="APERTURE")
    fits.HDUList([_primary(tic_id, sector, "m_harness TargetPixelExporterPipelineModule"), table, aperture]).writeto(
        path, overwrite=True, checksum=False)
    return path


SOLAR = dict(r_star=1.0, e_r_star=0.1, m_star=1.0, e_m_star=0.12, teff=5777.0, ruwe=1.0, has_params=True,
             tic_query_error=None, gaia_query_error=None)
FIXTURE_LD = (0.40, 0.20)


def synthetic_host(root, tic_id, sector, archival_sectors=(), seed=0, sigma=4.0e-4, crowdsap=0.95, n_segments=2,
                   stellar=None, red_noise=0.0):
    """Write one synthetic host (discovery light curve, archival light curves, discovery target pixel file) under
    root and return its host record: paths relative to root, SHA256 of every file, stellar metadata, crowding."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    lcs = []
    disc_rel = None
    for s in sorted(set([int(sector)] + [int(x) for x in archival_sectors])):
        t, cad = sector_times(s, n_segments=n_segments if s == int(sector) else 2)
        flux = 1.0 + rng.normal(0.0, sigma, t.size)
        if red_noise > 0:
            flux = flux + red_noise * np.sin(2.0 * math.pi * (t - t[0]) / 3.7 + rng.uniform(0, 2 * math.pi))
        name = f"tess-s{s:04d}-{int(tic_id):016d}-fixture_lc.fits"
        # Centroids get a small jitter: a constant series has zero variance and the production Filter 4 would end
        # CENTROID_NOT_EVALUATED (item 34.2 c) on every fixture.
        centroid = (5.0 + rng.normal(0.0, 2e-3, t.size), 5.0 + rng.normal(0.0, 2e-3, t.size))
        p = write_lightcurve(root / f"TIC{int(tic_id)}" / name, tic_id, s, t, flux, np.full(t.size, sigma),
                             cadenceno=cad, crowdsap=crowdsap, centroid_xy=centroid)
        rel = str(p.relative_to(root)).replace("\\", "/")
        lcs.append(dict(path=rel, sector=s, obsid=f"fixture-{int(tic_id)}-{s}", sha256=sha256_file(p)))
        if s == int(sector):
            disc_rel, disc_t, disc_cad, disc_flux = rel, t, cad, flux
    tpf_name = f"tess-s{int(sector):04d}-{int(tic_id):016d}-fixture_tp.fits"
    tp = write_tpf(root / f"TIC{int(tic_id)}" / tpf_name, tic_id, sector, disc_t, disc_flux, cadenceno=disc_cad,
                   seed=seed + 7)
    disc = next(x for x in lcs if x["path"] == disc_rel)
    return dict(tic_id=str(int(tic_id)), sector=int(sector), crowdsap=float(crowdsap), source_fits=disc_rel,
                source_sha256=disc["sha256"], tpf_fits=str(tp.relative_to(root)).replace("\\", "/"),
                tpf_sha256=sha256_file(tp), lightcurves=lcs, stellar=dict(stellar or SOLAR), ld=list(FIXTURE_LD),
                known_transits=[], n_valid_cadences=int(disc_t.size), discovery_baseline_days=float(disc_t[-1] - disc_t[0]),
                role="synthetic_fixture")
