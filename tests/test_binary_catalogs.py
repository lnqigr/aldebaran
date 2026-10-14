import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.parse
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from pleiades import fetch_csv_no_ssl, safe_float

def check_binary_catalogs(ra: float, dec: float) -> dict:
    res = {
        "is_tess_eb": False,
        "tess_eb_tic": None,
        "tess_eb_period": None,
        "is_gaia_nss": False,
        "gaia_nss_type": None,
        "gaia_nss_period": None
    }
    # 1. VizieR TESS Eclipsing Binary Catalogue (Prša et al. 2022)
    try:
        v = Vizier(columns=["TIC", "Per", "Morph"], row_limit=1)
        coord = SkyCoord(ra, dec, unit="deg", frame="icrs")
        v_res = v.query_region(coord, radius=8.0 * u.arcsec, catalog="J/ApJS/258/16/tess-ebs")
        if len(v_res) > 0 and len(v_res[0]) > 0:
            res["is_tess_eb"] = True
            res["tess_eb_tic"] = str(v_res[0]["TIC"][0])
            res["tess_eb_period"] = safe_float(v_res[0]["Per"][0], None)
    except Exception:
        pass

    # 2. Gaia DR3 Non-Single Star (NSS) Two-Body Orbit Table
    try:
        adql_nss = (
            f"SELECT TOP 1 n.nss_solution_type, n.period FROM gaiadr3.nss_two_body_orbit AS n "
            f"JOIN gaiadr3.gaia_source AS g ON n.source_id=g.source_id "
            f"WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra}, {dec}, 0.001))"
        )
        url_nss = (
            "https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery"
            f"&LANG=ADQL&FORMAT=csv&QUERY={urllib.parse.quote(adql_nss)}"
        )
        df_nss = fetch_csv_no_ssl(url_nss)
        if len(df_nss) > 0 and "nss_solution_type" in df_nss.columns:
            sol = str(df_nss["nss_solution_type"].iloc[0]).strip()
            if sol and sol.upper() != "NAN":
                res["is_gaia_nss"] = True
                res["gaia_nss_type"] = sol
                res["gaia_nss_period"] = safe_float(df_nss["period"].iloc[0], None)
    except Exception:
        pass
    return res

if __name__ == "__main__":
    # Test 1: Known single exoplanet host HD 209458 (should return clean False)
    r1 = check_binary_catalogs(330.7949, 18.8843)
    print("Single star (HD 209458) result:", r1)
    assert not r1["is_tess_eb"] and not r1["is_gaia_nss"]

    # Test 2: Known TESS Eclipsing Binary TIC 167812449 (caught in both Vizier and Gaia NSS SB1)
    r2 = check_binary_catalogs(104.78244, -61.84007)
    print("Known EB (TIC 167812449) result:", r2)
    assert r2["is_tess_eb"] is True, "Failed to identify known TESS EB TIC 167812449!"
    assert r2["is_gaia_nss"] is True and r2["gaia_nss_type"] == "SB1", "Failed to identify Gaia SB1!"

    # Test 3: Known Gaia DR3 Non-Single Star binary
    r3 = check_binary_catalogs(126.334044, -20.799923)
    print("Known Gaia NSS result:", r3)
    assert r3["is_gaia_nss"] is True, "Failed to identify Gaia NSS binary!"

    print("[SUCCESS] All binary catalog cross-matches work perfectly!")


