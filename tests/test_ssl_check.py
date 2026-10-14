"""
Test suite script: Verifies SSL connection and certificate validation across
all remote data providers (NASA Exoplanet Archive, ExoFOP-TESS, Gaia ESA TAP, STScI MAST).
Tests standard SSL context vs certifi bundle vs truststore.
"""

import ssl
import sys
import urllib.request

print("Python version:", sys.version)

URLS = [
    "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv",
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+1+tid+from+toi&format=csv",
    "https://gea.esac.esa.int/tap-server/tap/availability",
    "https://mast.stsci.edu/api/v0/",
]


def test_ssl_context(context_name: str, ctx: ssl.SSLContext):
    print(f"\n--- Testing SSL context: {context_name} ---")
    all_ok = True
    for url in URLS:
        host = url.split("/")[2]
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=20) as response:
                response.read(200)
            print(f"  [OK]   {host}")
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {host} -> {type(e).__name__}: {str(e)[:80]}")
    return all_ok


if __name__ == "__main__":
    test_ssl_context("Python Default Context", ssl.create_default_context())

    try:
        import certifi

        test_ssl_context("certifi CA bundle", ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        print("\n[!] 'certifi' is not installed in the current environment.")

    try:
        import truststore

        test_ssl_context("truststore (Windows System Cert Store)", truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    except ImportError:
        print("\n[i] 'truststore' is not installed.")
