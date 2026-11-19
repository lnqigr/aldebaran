"""RC2-FREEZE-01 C10: the Tier 1 worker pool gets the parent's run configuration explicitly.

A spawn worker re-imports aldebaran with default module state. The control shows that a spawn worker without the
initializer does not see the parent's configuration, and that with run_config/apply_run_config it sees all of it."""
import multiprocessing
import os
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor

import aldebaran as ald
import known_signals


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.saved = ald.run_config()
        self.tmp = tempfile.TemporaryDirectory()
        ald.configure_environment(
            output_dir=os.path.join(self.tmp.name, "out"), run_id="c10_run",
            toi_ephemerides={"123": [{"period": 3.5, "t0": 2000.25, "duration_hours": 2.0}]},
            known_signals_registry=known_signals.KnownSignalsRegistry(),
            catalog_cache_file=os.path.join(self.tmp.name, "cat.txt"),
            toi_ephemeris_cache_file=os.path.join(self.tmp.name, "toi.json"),
            known_signals_registry_cache_file=os.path.join(self.tmp.name, "reg.json"))
        ald.KNOWN_SIGNAL_CACHE_STATUS.update(state="NOT_EVALUATED", toi_cache_failures=2, last_error="toi (x): OSError")

    def tearDown(self):
        ald.apply_run_config(self.saved)
        self.tmp.cleanup()

    @staticmethod
    def comparable(config):
        out = dict(config)
        out["GLOBAL_KNOWN_SIGNALS_REGISTRY"] = type(out["GLOBAL_KNOWN_SIGNALS_REGISTRY"]).__name__
        return out

    def worker_config(self, with_initializer):
        kwargs = dict(initializer=ald.apply_run_config, initargs=(ald.run_config(),)) if with_initializer else {}
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), **kwargs) as pool:
            return pool.submit(ald.run_config).result(timeout=300)

    def test_spawn_worker_receives_the_whole_configuration(self):
        parent = ald.run_config()
        self.assertEqual(set(parent), set(ald.RUN_CONFIG_NAMES) | {"KNOWN_SIGNAL_CACHE_STATUS"})
        self.assertEqual(self.comparable(self.worker_config(True)), self.comparable(parent))

    def test_control_spawn_worker_without_initializer_differs(self):
        child = self.worker_config(False)
        self.assertNotEqual(child["CURRENT_RUN_ID"], "c10_run")
        self.assertNotEqual(child["OUTPUT_DIR"], ald.OUTPUT_DIR)
        self.assertEqual(child["KNOWN_SIGNAL_CACHE_STATUS"]["state"], "OK")

    def test_apply_is_the_inverse_of_run_config_and_rejects_partial_configs(self):
        config = ald.run_config()
        ald.apply_run_config(self.saved)
        self.assertNotEqual(ald.CURRENT_RUN_ID, "c10_run")
        ald.apply_run_config(config)
        self.assertEqual(ald.run_config(), config)
        partial = dict(config)
        partial.pop("OUTPUT_DIR")
        with self.assertRaises(ValueError):
            ald.apply_run_config(partial)
        with self.assertRaises(ValueError):
            ald.apply_run_config(dict(config, EXTRA_NAME=1))
        self.assertEqual(ald.run_config(), config)  # a refused config changes nothing


if __name__ == "__main__":
    unittest.main()
