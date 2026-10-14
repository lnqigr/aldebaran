import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.parse
from aldebaran import fetch_csv_with_headers

def fetch_tess_ebs():
    try:
        # In ADQL on TAPVizieR, table name with slashes is enclosed in double quotes: "J/ApJS/258/16/tess-ebs"
        adql = 'SELECT distinct TIC FROM "J/ApJS/258/16/tess-ebs"'
        url = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=" + urllib.parse.quote(adql)
        df = fetch_csv_with_headers(url)
        print(f"Successfully fetched {len(df)} TESS EBs from TAPVizieR.")
        print(df.head())
        return df
    except Exception as e:
        print("TAPVizieR error:", e)
        # Fallback via astroquery.vizier
        from astroquery.vizier import Vizier
        v = Vizier(columns=["TIC"], row_limit=-1)
        res = v.get_catalogs("J/ApJS/258/16/tess-ebs")
        print(f"Astroquery fallback fetched {len(res[0])} rows.")
        return res[0].to_pandas()

if __name__ == "__main__":
    fetch_tess_ebs()
