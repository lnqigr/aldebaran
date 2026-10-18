"""Opt-in real-input replay: python -m unittest tests.historical_known_signals.

Reuses the contract assertions; these archived observations and their provenance
are distinct from the mandatory SYNTHETIC suite. No missing-input skips.
"""
import os
from pathlib import Path
from unittest import mock
from tests import test_known_signals as contracts


class TestHistoricalKnownSignals(contracts.TestKnownSignalsSuite):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = str(Path(__file__).resolve().parents[1] /
                              "24702696_20261016_025524" / "audit" / "catalog_audit")
        cls.confirmed_path = os.path.join(cls.fixture_dir, "nasa_confirmed.csv")
        cls.toi_path = os.path.join(cls.fixture_dir, "nasa_toi.csv")
        cls.ctoi_path = os.path.join(cls.fixture_dir, "exofop_ctoi.csv")
        for name in (cls.confirmed_path, cls.toi_path, cls.ctoi_path,
                     os.path.join(cls.fixture_dir, "source_manifest.json")):
            if not os.path.isfile(name):
                raise FileNotFoundError("Historical replay requires original archived input: " + name)
        cls.has_fixtures = True
        cls.expected_toi_sha256 = "9d33a9baadc332fdc9ca69cb77d945ab198ea07f27412288deeaec2e66ff78c5"
        cls.expected_raw_retrieval_utc = "2026-10-16T01:16:12.688595+00:00"
        cls._orig_offline = os.environ.get("ALDEBARAN_OFFLINE")
        os.environ["ALDEBARAN_OFFLINE"] = "1"
        cls._network = mock.patch("socket.socket.connect", side_effect=AssertionError("Historical offline replay attempted network"))
        cls._network.start()
        cls.addClassCleanup(cls._network.stop)
