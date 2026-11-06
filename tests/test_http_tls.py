"""Offline TLS regression controls on the production helpers and callers."""
import io
import os
import ssl
import tempfile
import unittest
import urllib.error
from unittest import mock

from astropy.table import Table
import requests

import aldebaran
import pleiades


URL = "https://example.invalid/catalog.csv"


class HttpTlsTests(unittest.TestCase):
    def test_certificate_failure_propagates_without_unverified_retry(self):
        for module in (aldebaran, pleiades):
            with self.subTest(module=module.__name__):
                failure = requests.exceptions.SSLError("certificate verify failed")
                # An insecure retry would succeed, so merely raising eventually is insufficient.
                with mock.patch.object(module.requests, "get", side_effect=[failure, mock.Mock()]) as get:
                    with self.assertRaises(requests.exceptions.SSLError) as caught:
                        module.safe_http_get(URL)
                    self.assertIs(caught.exception, failure)
                    get.assert_called_once_with(URL, params=None, headers=None, timeout=45,
                                                verify=module.CA_BUNDLE_PATH)

    def test_success_preserves_response_and_request_arguments(self):
        for module in (aldebaran, pleiades):
            with self.subTest(module=module.__name__):
                response = mock.Mock()
                with mock.patch.object(module.requests, "get", return_value=response) as get:
                    self.assertIs(module.safe_http_get(URL, {"a": "b"}, {"X-Test": "yes"}, 7), response)
                get.assert_called_once_with(URL, params={"a": "b"}, headers={"X-Test": "yes"},
                                            timeout=7, verify=module.CA_BUNDLE_PATH)
                response.raise_for_status.assert_called_once_with()

    def test_other_request_errors_propagate_without_retry(self):
        for module in (aldebaran, pleiades):
            for failure in (requests.exceptions.Timeout("timeout"), requests.exceptions.HTTPError("503")):
                with self.subTest(module=module.__name__, error=type(failure).__name__):
                    response = mock.Mock()
                    if isinstance(failure, requests.exceptions.HTTPError):
                        response.raise_for_status.side_effect = failure
                    with mock.patch.object(module.requests, "get", return_value=response,
                                           side_effect=None if isinstance(failure, requests.exceptions.HTTPError) else failure) as get:
                        with self.assertRaises(type(failure)) as caught:
                            module.safe_http_get(URL)
                        self.assertIs(caught.exception, failure)
                        self.assertEqual(get.call_count, 1)

    def test_csv_tls_failure_reaches_caller_without_urllib_fallback(self):
        for module, fetch in ((aldebaran, aldebaran.fetch_csv_with_headers),
                              (pleiades, pleiades.fetch_csv_no_ssl)):
            with self.subTest(module=module.__name__):
                failure = requests.exceptions.SSLError("certificate verify failed")
                with mock.patch.object(module.requests, "get", side_effect=failure) as get, \
                        mock.patch.object(aldebaran, "safe_urlopen") as fallback:
                    fallback.return_value.__enter__.return_value = io.BytesIO(b"tic,value\n123,4\n")
                    with self.assertRaises(requests.exceptions.SSLError) as caught:
                        fetch(URL)
                    self.assertIs(caught.exception, failure)
                    self.assertEqual(get.call_count, 1)
                    self.assertTrue(all(c.kwargs["verify"] == module.CA_BUNDLE_PATH for c in get.call_args_list))
                    fallback.assert_not_called()

    def test_csv_success_is_unchanged(self):
        for module, fetch in ((aldebaran, aldebaran.fetch_csv_with_headers),
                              (pleiades, pleiades.fetch_csv_no_ssl)):
            with self.subTest(module=module.__name__):
                response = mock.Mock(text="# metadata\ntic,value\n123,4\n")
                with mock.patch.object(module.requests, "get", return_value=response):
                    self.assertEqual(fetch(URL).to_dict("list"), {"tic": [123], "value": [4]})

    def test_non_tls_csv_failure_keeps_existing_urllib_fallback(self):
        with mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.Timeout("timeout")), \
                mock.patch.object(aldebaran, "safe_urlopen") as fallback:
            fallback.return_value.__enter__.return_value = io.BytesIO(b"tic,value\n123,4\n")
            self.assertEqual(aldebaran.fetch_csv_with_headers(URL).to_dict("list"), {"tic": [123], "value": [4]})
            fallback.assert_called_once()

    def test_urllib_tls_failure_propagates_without_unverified_context(self):
        failure = ssl.SSLCertVerificationError("certificate verify failed")
        with mock.patch.object(aldebaran.urllib.request, "urlopen", side_effect=[failure, mock.Mock()]) as open_url:
            with self.assertRaises(ssl.SSLCertVerificationError) as caught:
                aldebaran.safe_urlopen(URL, timeout=7)
            self.assertIs(caught.exception, failure)
            open_url.assert_called_once()
            context = open_url.call_args.kwargs["context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertEqual(open_url.call_args.kwargs["timeout"], 7)

    def test_csv_non_tls_fallback_cannot_accept_urllib_tls_failure(self):
        failure = ssl.SSLCertVerificationError("certificate verify failed")
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(b"tic,value\n123,4\n")
        with mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.Timeout("timeout")), \
                mock.patch.object(aldebaran.urllib.request, "urlopen", side_effect=[failure, response]) as open_url:
            with self.assertRaises(ssl.SSLCertVerificationError) as caught:
                aldebaran.fetch_csv_with_headers(URL)
            self.assertIs(caught.exception, failure)
            open_url.assert_called_once()

    def test_catalog_loader_records_tls_as_failed_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(aldebaran, "CATALOG_CACHE_FILE", os.path.join(tmp, "catalog.txt")), \
                mock.patch.object(aldebaran, "TOI_EPHEMERIS_CACHE_FILE", os.path.join(tmp, "toi.json")), \
                mock.patch.object(aldebaran, "KNOWN_SIGNALS_REGISTRY_CACHE_FILE", os.path.join(tmp, "registry.json")), \
                mock.patch.object(aldebaran, "GLOBAL_KNOWN_SIGNALS_REGISTRY", None), \
                mock.patch.object(aldebaran, "GLOBAL_TOI_EPHEMERIDES", {}), \
                mock.patch.object(aldebaran, "write_log"), \
                mock.patch.dict(os.environ, {"ALDEBARAN_OFFLINE": "0"}), \
                mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.SSLError("bad certificate")) as get, \
                mock.patch.object(aldebaran, "safe_urlopen") as fallback:
            aldebaran.load_known_catalogs_and_toi_ephemerides()
            registry = aldebaran.get_known_signals_registry()
            for name in ("nasa_toi", "nasa_confirmed", "exofop_ctoi", "vizier_tess_eb"):
                self.assertEqual(registry.source_snapshots[name].last_attempt_status, "FAILED")
            self.assertEqual(get.call_count, 4)
            fallback.assert_not_called()

    def test_csv_wrapped_urllib_tls_failure_propagates_original_reason(self):
        failure = ssl.SSLCertVerificationError("certificate verify failed")
        wrapped = urllib.error.URLError(failure)
        with mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.Timeout("timeout")) as get, \
                mock.patch.object(aldebaran.urllib.request, "urlopen", side_effect=wrapped) as open_url:
            with self.assertRaises(ssl.SSLCertVerificationError) as caught:
                aldebaran.fetch_csv_with_headers(URL)
            self.assertIs(caught.exception, failure)
            self.assertIs(caught.exception.__cause__, wrapped)
            get.assert_called_once()
            self.assertEqual(get.call_args.kwargs["verify"], aldebaran.CA_BUNDLE_PATH)
            open_url.assert_called_once()
            self.assertEqual(open_url.call_args.kwargs["context"].verify_mode, ssl.CERT_REQUIRED)

    def test_stellar_caller_propagates_wrapped_urllib_tls_failure(self):
        tic = Table({"rad": [1.0], "Teff": [5500.0], "ra": [1.0], "dec": [2.0]})
        for failure in (ssl.SSLCertVerificationError("bad certificate"), ssl.SSLError("TLS handshake failed")):
            with self.subTest(error=type(failure).__name__), \
                    mock.patch.object(aldebaran.Catalogs, "query_criteria", return_value=tic), \
                    mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.Timeout("timeout")), \
                    mock.patch.object(aldebaran.urllib.request, "urlopen", side_effect=urllib.error.URLError(failure)) as open_url:
                with self.assertRaises(ssl.SSLError) as caught:
                    aldebaran.get_stellar_and_gaia_params("123")
                self.assertIs(caught.exception, failure)
                open_url.assert_called_once()

    def test_non_tls_url_error_keeps_existing_fetch_and_stellar_behavior(self):
        tic = Table({"rad": [1.0], "Teff": [5500.0], "ra": [1.0], "dec": [2.0]})
        for reason in (OSError("connection refused"), "name resolution failed"):
            failure = urllib.error.URLError(reason)
            with self.subTest(reason=str(reason)), \
                    mock.patch.object(aldebaran.Catalogs, "query_criteria", return_value=tic), \
                    mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.Timeout("timeout")), \
                    mock.patch.object(aldebaran.urllib.request, "urlopen", side_effect=failure) as open_url:
                with self.assertRaises(urllib.error.URLError) as caught:
                    aldebaran.fetch_csv_with_headers(URL)
                self.assertIs(caught.exception, failure)
                result = aldebaran.get_stellar_and_gaia_params("123")
                self.assertTrue(result["has_params"])
                self.assertTrue(aldebaran.np.isnan(result["ruwe"]))
                self.assertEqual(open_url.call_count, 2)

    def test_stellar_successful_urllib_fallback_keeps_measured_ruwe(self):
        tic = Table({"rad": [1.0], "Teff": [5500.0], "ra": [1.0], "dec": [2.0]})
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(b"ruwe\n1.1\n")
        with mock.patch.object(aldebaran.Catalogs, "query_criteria", return_value=tic), \
                mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.Timeout("timeout")), \
                mock.patch.object(aldebaran.urllib.request, "urlopen", return_value=response) as open_url:
            result = aldebaran.get_stellar_and_gaia_params("123")
            self.assertTrue(result["has_params"])
            self.assertEqual(result["ruwe"], 1.1)
            open_url.assert_called_once()
            self.assertTrue(open_url.call_args.kwargs["context"].check_hostname)

    def test_stellar_caller_does_not_silence_tls_failure(self):
        tic = Table({"rad": [1.0], "Teff": [5500.0], "ra": [1.0], "dec": [2.0]})
        with mock.patch.object(aldebaran.Catalogs, "query_criteria", return_value=tic), \
                mock.patch.object(aldebaran.requests, "get", side_effect=requests.exceptions.SSLError("bad certificate")), \
                mock.patch.object(aldebaran, "safe_urlopen") as fallback:
            with self.assertRaises(requests.exceptions.SSLError):
                aldebaran.get_stellar_and_gaia_params("123")
            fallback.assert_not_called()

    def test_binary_caller_does_not_return_false_on_tls_failure(self):
        with mock.patch.object(pleiades, "Vizier", return_value=mock.Mock(query_region=mock.Mock(return_value=[]))), \
                mock.patch.object(pleiades.requests, "get", side_effect=requests.exceptions.SSLError("bad certificate")):
            with self.assertRaises(requests.exceptions.SSLError):
                pleiades.check_binary_catalogs(1.0, 2.0)

    def test_contamination_caller_does_not_return_zero_on_tls_failure(self):
        tic = Table({"ra": [1.0], "dec": [2.0]})
        with mock.patch.object(pleiades.Catalogs, "query_criteria", return_value=tic), \
                mock.patch.object(pleiades, "check_binary_catalogs", return_value={}), \
                mock.patch.object(pleiades.requests, "get", side_effect=requests.exceptions.SSLError("bad certificate")):
            with self.assertRaises(requests.exceptions.SSLError):
                pleiades.query_gaia_contamination("123", 500.0)

    def test_binary_tls_failure_survives_contamination_caller(self):
        tic = Table({"ra": [1.0], "dec": [2.0]})
        with mock.patch.object(pleiades.Catalogs, "query_criteria", return_value=tic), \
                mock.patch.object(pleiades, "Vizier", return_value=mock.Mock(query_region=mock.Mock(return_value=[]))), \
                mock.patch.object(pleiades.requests, "get", side_effect=requests.exceptions.SSLError("bad certificate")):
            with self.assertRaises(requests.exceptions.SSLError):
                pleiades.query_gaia_contamination("123", 500.0)


if __name__ == "__main__":
    unittest.main()
