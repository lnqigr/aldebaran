"""
Targeted regression test suite for ALD-013:
Preserve independently passing circular/eccentric secondary-detection branches in Aldebaran;
faithful selected-peak tuple and regression tests.

Covers:
- A2: Compact SEC03 fixture demonstration (old baseline fails, fixed detector passes)
- A3: Complete truth-table coverage (C/E combinations, selected metadata tuples, competing branches)
- A4: Scientific boundary invariants (strict SNR >, depth > 100, ratio >= 0.10)
- A5: Contract early returns (min_phi >= 0.45, oot_points < 50) and metadata conventions
- A6: Deliberate executable mutant negative controls (old guard, OR->AND, metadata swap, inverted tie, threshold shift)
"""

import sys
import os
import unittest
import hashlib
import inspect
import re
import ast
import numpy as np
import scipy.stats as stats

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import aldebaran
from aldebaran import scan_full_orbit_secondary_eclipse


# ---------------------------------------------------------------------------
# AST-Derived Scoped Decision Block Extractor
# ---------------------------------------------------------------------------

def extract_decision_core_from_ast(source_code: str = None):
    """
    Extracts and compiles the actual decision and qualification block directly from
    the AST of scan_full_orbit_secondary_eclipse in aldebaran.py (or supplied source code).
    Scope: isolated decision-block scope (from circ_eb = ... up to return ...).
    Guarantees that boundary and truth-table assertions execute the actual source statements
    rather than a handwritten duplicate implementation.
    """
    if source_code is None:
        aldebaran_path = os.path.join(REPO_ROOT, "aldebaran.py")
        with open(aldebaran_path, "r", encoding="utf-8") as f:
            source_code = f.read()

    tree = ast.parse(source_code)
    fn_node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "scan_full_orbit_secondary_eclipse"
    )

    # Locate the assignment to circ_eb
    start_idx = None
    for i, stmt in enumerate(fn_node.body):
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == "circ_eb":
                    start_idx = i
                    break
            if start_idx is not None:
                break

    if start_idx is None:
        raise RuntimeError("Could not find assignment to circ_eb in scan_full_orbit_secondary_eclipse AST")

    decision_stmts = fn_node.body[start_idx:]

    new_fn = ast.FunctionDef(
        name="_source_extracted_decision_core",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="circ_max_snr"),
                ast.arg(arg="circ_max_depth"),
                ast.arg(arg="ecc_max_snr"),
                ast.arg(arg="ecc_max_depth"),
                ast.arg(arg="ecc_best_phi"),
                ast.arg(arg="primary_depth_ppm"),
                ast.arg(arg="z_crit"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=decision_stmts,
        decorator_list=[]
    )
    ast.fix_missing_locations(new_fn)
    mod = ast.Module(body=[new_fn], type_ignores=[])
    ast.fix_missing_locations(mod)

    code = compile(mod, "<extracted_decision_core_from_aldebaran_ast>", "exec")
    ns = {}
    exec(code, {}, ns)
    return ns["_source_extracted_decision_core"]


# Global production decision core compiled from aldebaran.py AST
PRODUCTION_AST_DECISION_CORE = extract_decision_core_from_ast()


# ---------------------------------------------------------------------------
# Deterministic Fixture Generators
# ---------------------------------------------------------------------------

def generate_sec03_fixtures():
    """
    Generates deterministic SEC03 test fixtures matching ARGON-PHYS-01 / Codex reproduced SEC03:
    - Period = 4.00 d, dur = 2.0/24.0 d, primary_depth_ppm = 1800.0 ppm
    - Baseline with 10-point cycle plateau
    - Circular injection at phi = 0.50 (SNR ~ 3.2356 > 3.1, depth ~ 208.4 ppm > 100 and >= 180)
    - Eccentric injection at phi = 0.22 (SNR ~ 3.3531 < z_crit ~ 4.067, depth ~ 213.0 ppm)
    Returns:
        dict with t, f_circ_only, f_combined, period, t0, dur, primary_depth_ppm, and sha256 hashes
    """
    p_s3 = 4.00
    dur_s3 = 2.0 / 24.0
    t_s3 = np.linspace(0.0, 27.0, 19440)
    ph_s3 = ((t_s3 - 1.0) / p_s3) % 1.0
    idx_sort_s3 = np.argsort(ph_s3)

    f_sec03 = np.ones_like(t_s3)
    cycle10 = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0])[np.arange(len(t_s3)) % 10]
    f_sec03[idx_sort_s3] = 1.0 + (1200e-6 / 1.4826) * cycle10

    dur_phase_s3 = dur_s3 / p_s3
    half_box_s3 = 0.425 * dur_phase_s3
    oot_mask_s3 = (ph_s3 > 2.0 * dur_phase_s3) & (ph_s3 < 1.0 - 2.0 * dur_phase_s3)
    oot_med_s3 = np.nanmedian(f_sec03[oot_mask_s3])
    oot_noise_s3 = 1.4826 * np.nanmedian(np.abs(f_sec03[oot_mask_s3] - oot_med_s3))

    w_circ = np.abs(ph_s3 - 0.50) <= half_box_s3
    n_circ = int(np.sum(w_circ))
    se_circ_ppm = (oot_noise_s3 / np.sqrt(n_circ)) * 1e6
    f_sec03[w_circ] -= (4.57 * se_circ_ppm) * 1e-6

    w_ecc = np.abs(ph_s3 - 0.22) <= half_box_s3
    n_ecc = int(np.sum(w_ecc))
    se_ecc_ppm = (oot_noise_s3 / np.sqrt(n_ecc)) * 1e6
    f_sec03[w_ecc] -= (4.60 * se_ecc_ppm) * 1e-6

    f_sec03_circ_only = f_sec03.copy()
    f_sec03_circ_only[w_ecc] += (4.60 * se_ecc_ppm) * 1e-6

    t_hash = hashlib.sha256(t_s3.tobytes()).hexdigest()
    f_circ_hash = hashlib.sha256(f_sec03_circ_only.tobytes()).hexdigest()
    f_comb_hash = hashlib.sha256(f_sec03.tobytes()).hexdigest()

    return {
        "t": t_s3,
        "f_circ_only": f_sec03_circ_only,
        "f_combined": f_sec03,
        "period": p_s3,
        "t0": 1.0,
        "dur": dur_s3,
        "primary_depth_ppm": 1800.0,
        "t_hash": t_hash,
        "f_circ_hash": f_circ_hash,
        "f_comb_hash": f_comb_hash,
    }


def generate_exact_tie_waveform():
    """
    Constructs a deterministic full-orbit waveform where both circular and eccentric
    secondary eclipses qualify AND produce exactly identical measured SNR and depth values.
    Grid alignment: dips are centered on exact grid nodes with identical point counts (425),
    guaranteeing circ_max_snr == ecc_max_snr down to floating-point bit equality.
    """
    t = np.linspace(0.0, 20.0, 20000)
    p = 4.0
    t0 = 1.0
    dur = 0.10
    f = np.ones_like(t)

    ph = ((t - t0) % p) / p
    dur_ph = dur / p
    min_ph = max(0.06, 1.6 * dur_ph)
    n_win = max(5, int((1.0 - 2.0 * min_ph) / max(dur_ph, 0.005)))
    n_steps = max(20, n_win * 2)
    grid = np.linspace(min_ph + 0.5 * dur_ph, 1.0 - min_ph - 0.5 * dur_ph, n_steps)
    half_box = 0.425 * dur_ph

    p_circ = [x for x in grid if abs(x - 0.50) <= 0.04][0]  # 0.4690217...
    p_ecc = [x for x in grid if abs(x - 0.50) > 0.04 and x > 0.20][0]  # 0.2088043...

    w_c = np.abs(ph - p_circ) <= half_box
    w_e = np.abs(ph - p_ecc) <= half_box

    # Both dips have exactly 600 ppm depth
    f[w_c] -= 0.000600
    f[w_e] -= 0.000600

    return t, f, p, t0, dur, 1500.0


def generate_circ_near_threshold_3_15_waveform():
    """
    Constructs a deterministic waveform where circular secondary has evaluated SNR ~ 3.15,
    strictly between the production threshold (3.10) and shifted mutant threshold (3.20).
    Depth is ~ 202.9 ppm, passing both > 100.0 ppm floor and >= 180.0 ppm ratio.
    """
    p = 4.0
    dur = 2.0 / 24.0
    t = np.linspace(0.0, 27.0, 19440)
    ph = ((t - 1.0) / p) % 1.0
    idx = np.argsort(ph)
    f = np.ones_like(t)

    cycle10 = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0])[np.arange(len(t)) % 10]
    f[idx] = 1.0 + (1200e-6 / 1.4826) * cycle10

    dur_ph = dur / p
    half_box = 0.425 * dur_ph
    oot_m = (ph > 2.0 * dur_ph) & (ph < 1.0 - 2.0 * dur_ph)
    oot_med = np.nanmedian(f[oot_m])
    oot_noise = 1.4826 * np.nanmedian(np.abs(f[oot_m] - oot_med))

    w_circ = np.abs(ph - 0.50) <= half_box
    se_circ = (oot_noise / np.sqrt(np.sum(w_circ))) * 1e6
    f[w_circ] -= (4.45 * se_circ) * 1e-6  # Evaluated SNR = 3.1497

    return t, f, p, 1.0, dur, 1800.0


def generate_clean_fixture(seed: int = 42):
    """Generates a clean planet light curve with only primary transit, no secondary eclipse."""
    np.random.seed(seed)
    t = np.linspace(0.0, 25.0, 18000)
    period = 4.0
    t0 = 1.0
    dur = 0.10
    primary_depth_ppm = 1500.0
    f = np.ones_like(t) + np.random.normal(0, 0.0003, size=len(t))
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    f[np.abs(phase) < (0.5 * dur)] -= 0.0015
    return t, f, period, t0, dur, primary_depth_ppm


def generate_eccentric_only_fixture(seed: int = 42):
    """Generates an eccentric-only EB with secondary dip at phase 0.35."""
    t, f, period, t0, dur, primary_depth_ppm = generate_clean_fixture(seed=seed)
    phase_0to1 = ((t - t0) % period) / period
    sec_phase_diff = np.abs(((phase_0to1 - 0.35 + 0.5) % 1.0) - 0.5)
    f[sec_phase_diff < (0.5 * dur / period)] -= 0.000600  # 600 ppm dip
    return t, f, period, t0, dur, primary_depth_ppm


def generate_both_qualify_fixture(dominant: str = "circ", seed: int = 42):
    """
    Generates a light curve where both circular and eccentric secondary eclipses qualify.
    dominant: 'circ' for circular SNR > eccentric SNR, 'ecc' for eccentric SNR > circular SNR.
    """
    t, f, period, t0, dur, primary_depth_ppm = generate_clean_fixture(seed=seed)
    phase_0to1 = ((t - t0) % period) / period
    circ_phase_diff = np.abs(((phase_0to1 - 0.50 + 0.5) % 1.0) - 0.5)
    sec_phase_diff = np.abs(((phase_0to1 - 0.35 + 0.5) % 1.0) - 0.5)

    if dominant == "circ":
        f[circ_phase_diff < (0.5 * dur / period)] -= 0.001000  # 1000 ppm circ
        f[sec_phase_diff < (0.5 * dur / period)] -= 0.000600   # 600 ppm ecc
    else:
        f[circ_phase_diff < (0.5 * dur / period)] -= 0.000350  # 350 ppm circ (qualifies > 3.1)
        f[sec_phase_diff < (0.5 * dur / period)] -= 0.001000   # 1000 ppm ecc (qualifies > z_crit)

    return t, f, period, t0, dur, primary_depth_ppm


# ---------------------------------------------------------------------------
# Isolated Executable Mutants Factory (for A6 Deliberate Negative Controls)
# ---------------------------------------------------------------------------

def _build_mutant_detector(mutant_kind: str):
    """
    Constructs an isolated, executable copy of scan_full_orbit_secondary_eclipse
    derived from the production function but with a specific mutated selection block.
    This guarantees that negative controls execute the REAL pipeline up to the
    selection block, testing executable mutant code paths rather than static booleans.
    """
    src = inspect.getsource(aldebaran.scan_full_orbit_secondary_eclipse)

    pattern = r'    # ALD-013: Independent circular and eccentric qualifying branches.*?return is_eb, max_snr, max_depth_ppm, best_phi, z_crit'

    if mutant_kind == "old_baseline":
        mutant_block = '''    # MUTANT: Old baseline selection guard
    if circ_eb and (circ_max_snr >= ecc_max_snr):
        is_eb = True
        max_snr = circ_max_snr
        max_depth_ppm = circ_max_depth
        best_phi = 0.50
    elif ecc_eb:
        is_eb = True
        max_snr = ecc_max_snr
        max_depth_ppm = ecc_max_depth
        best_phi = ecc_best_phi
    else:
        is_eb = False
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit'''

    elif mutant_kind == "or_to_and":
        mutant_block = '''    # MUTANT: OR-to-AND fault
    if circ_eb and ecc_eb:
        is_eb = True
        if circ_max_snr >= ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    else:
        is_eb = False
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit'''

    elif mutant_kind == "metadata_swap":
        mutant_block = '''    # MUTANT: Selected metadata swap
    if circ_eb and ecc_eb:
        is_eb = True
        if circ_max_snr >= ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    elif circ_eb:
        is_eb = True
        # FAULT: returning eccentric metadata instead of circular
        max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    elif ecc_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    else:
        is_eb = False
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit'''

    elif mutant_kind == "inverted_tie":
        mutant_block = '''    # MUTANT: Inverted tie breaker (strict > instead of >=)
    if circ_eb and ecc_eb:
        is_eb = True
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    elif circ_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
    elif ecc_eb:
        is_eb = True
        max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    else:
        is_eb = False
        if circ_max_snr > ecc_max_snr:
            max_snr, max_depth_ppm, best_phi = circ_max_snr, circ_max_depth, 0.50
        else:
            max_snr, max_depth_ppm, best_phi = ecc_max_snr, ecc_max_depth, ecc_best_phi
    return is_eb, max_snr, max_depth_ppm, best_phi, z_crit'''

    elif mutant_kind == "circ_threshold_shift":
        # Mutates circular threshold from > 3.1 to > 3.2
        mutated_src = src.replace("circ_max_snr > 3.1", "circ_max_snr > 3.2", 1)
        globs = dict(aldebaran.__dict__)
        locs = {}
        exec(compile(mutated_src, "<mutant_circ_threshold_shift>", "exec"), globs, locs)
        return locs["scan_full_orbit_secondary_eclipse"]

    else:
        raise ValueError(f"Unknown mutant kind: {mutant_kind}")

    match = re.search(pattern, src, re.DOTALL)
    if not match:
        raise RuntimeError("Failed to locate decision block in scan_full_orbit_secondary_eclipse")

    mutated_src = src[:match.start()] + mutant_block
    globs = dict(aldebaran.__dict__)
    locs = {}
    exec(compile(mutated_src, f"<mutant_{mutant_kind}>", "exec"), globs, locs)
    return locs["scan_full_orbit_secondary_eclipse"]


MUTANT_OLD_BASELINE = _build_mutant_detector("old_baseline")
MUTANT_OR_TO_AND = _build_mutant_detector("or_to_and")
MUTANT_METADATA_SWAP = _build_mutant_detector("metadata_swap")
MUTANT_INVERTED_TIE = _build_mutant_detector("inverted_tie")
MUTANT_CIRC_THRESHOLD_SHIFT = _build_mutant_detector("circ_threshold_shift")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestSecondaryBranchSelectionA2(unittest.TestCase):
    """A2: Compact SEC03 fixture demonstration and baseline-failure reproduction."""

    @classmethod
    def setUpClass(cls):
        cls.sec03 = generate_sec03_fixtures()

    def test_sec03_deterministic_fixture_hashes(self):
        """Verifies deterministic SHA256 of frozen SEC03 input arrays."""
        expected_t_hash = "1920cecacd1bf8b2b18c6b142aa28b4867060a21458668cb3cbd9c5c337d11e0"
        expected_circ_hash = "13564d9441f79001b15205cd061a71cc90ed1fb8c66d4564d1f50b8c26d9b7b4"
        expected_comb_hash = "3bcc05c7a429582538324791ec82bce8761ff55fe336f8a5031b9ddd32f23889"

        self.assertEqual(self.sec03["t_hash"], expected_t_hash)
        self.assertEqual(self.sec03["f_circ_hash"], expected_circ_hash)
        self.assertEqual(self.sec03["f_comb_hash"], expected_comb_hash)

    def test_sec03_circular_only_detection(self):
        """SEC03 circular-only fixture returns is_eb=True with expected SNR ~ 3.2356."""
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(
            self.sec03["t"],
            self.sec03["f_circ_only"],
            self.sec03["period"],
            self.sec03["t0"],
            self.sec03["dur"],
            self.sec03["primary_depth_ppm"],
        )
        self.assertTrue(is_eb, "Circular secondary eclipse must be detected")
        self.assertAlmostEqual(snr, 3.235606, places=4)
        self.assertAlmostEqual(depth, 208.435724, places=3)
        self.assertEqual(phi, 0.50)
        self.assertAlmostEqual(z_crit, 4.067016, places=4)

    def test_sec03_combined_baseline_fails_and_production_passes(self):
        """
        Critical A2 assertion:
        Adding an off-phase nonqualifying feature (SNR ~ 3.353 < z_crit ~ 4.067):
        - Baseline old detector FAILS by returning is_eb=False (suppression bug)
        - Production fixed detector PASSES by preserving is_eb=True and returning circular tuple
        """
        # 1. Run old baseline detector -> verify suppression bug
        old_is_eb, old_snr, old_depth, old_phi, old_zc = MUTANT_OLD_BASELINE(
            self.sec03["t"],
            self.sec03["f_combined"],
            self.sec03["period"],
            self.sec03["t0"],
            self.sec03["dur"],
            self.sec03["primary_depth_ppm"],
        )
        self.assertFalse(old_is_eb, "Old baseline detector must fail by returning is_eb=False on combined SEC03")
        self.assertAlmostEqual(old_snr, 3.353098, places=4)
        self.assertAlmostEqual(old_phi, 0.215336, places=3)

        # 2. Run fixed production detector -> verify bug is resolved
        prod_is_eb, prod_snr, prod_depth, prod_phi, prod_zc = scan_full_orbit_secondary_eclipse(
            self.sec03["t"],
            self.sec03["f_combined"],
            self.sec03["period"],
            self.sec03["t0"],
            self.sec03["dur"],
            self.sec03["primary_depth_ppm"],
        )
        self.assertTrue(prod_is_eb, "Fixed detector must return is_eb=True on combined SEC03")
        self.assertAlmostEqual(prod_snr, 3.235606, places=4)
        self.assertAlmostEqual(prod_depth, 208.435724, places=3)
        self.assertEqual(prod_phi, 0.50)
        self.assertAlmostEqual(prod_zc, 4.067016, places=4)


class TestSecondaryBranchSelectionA3TruthTable(unittest.TestCase):
    """A3: Truth-table coverage for C/E cases, qualifying branch selection, and competing branches."""

    def test_waveform_clean_both_false(self):
        """C=False, E=False: clean light curve returns is_eb=False with diagnostic raw maximum."""
        t, f, p, t0, dur, p_depth = generate_clean_fixture()
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertFalse(is_eb)
        self.assertGreater(z_crit, 3.5)
        self.assertLess(snr, z_crit)

    def test_waveform_eccentric_only_qualifies(self):
        """C=False, E=True: eccentric secondary at phi=0.35 is detected and returns eccentric tuple."""
        t, f, p, t0, dur, p_depth = generate_eccentric_only_fixture()
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertTrue(is_eb)
        self.assertGreater(snr, z_crit)
        self.assertGreater(depth, 100.0)
        self.assertGreaterEqual(depth, 0.10 * p_depth)
        self.assertAlmostEqual(phi, 0.35, delta=0.03)

    def test_waveform_both_qualify_circ_dominant(self):
        """C=True, E=True with circ_snr > ecc_snr selects circular branch (phi=0.50)."""
        t, f, p, t0, dur, p_depth = generate_both_qualify_fixture(dominant="circ")
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertTrue(is_eb)
        self.assertEqual(phi, 0.50, "Circular branch must be selected when circ_snr > ecc_snr")
        self.assertGreater(depth, 500.0)

    def test_waveform_both_qualify_ecc_dominant(self):
        """C=True, E=True with ecc_snr > circ_snr selects eccentric branch (phi ~ 0.35)."""
        t, f, p, t0, dur, p_depth = generate_both_qualify_fixture(dominant="ecc")
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertTrue(is_eb)
        self.assertAlmostEqual(phi, 0.35, delta=0.03, msg="Eccentric branch must be selected when ecc_snr > circ_snr")
        self.assertGreater(depth, 500.0)

    def test_waveform_both_qualify_exact_tie(self):
        """
        C=True, E=True on full deterministic waveform with circ_max_snr == ecc_max_snr.
        Enforces that actual detector selects circular branch (phi=0.50) under >= tie convention.
        Kills actual_tie_flip mutant (if circ_max_snr > ecc_max_snr).
        """
        t, f, p, t0, dur, p_depth = generate_exact_tie_waveform()
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertTrue(is_eb, "Tied qualifying branches must return is_eb=True")
        self.assertEqual(phi, 0.50, "Exact tie between qualifying branches must select circular branch (phi=0.50)")
        self.assertAlmostEqual(snr, 600.0, places=1)
        self.assertAlmostEqual(depth, 600.0, places=1)

    def test_waveform_circ_near_threshold_3_15(self):
        """
        Waveform test with circular secondary SNR ~ 3.15, strictly between 3.10 and 3.20.
        Enforces that production detector flags it as EB (3.15 > 3.10).
        Kills actual_circ_threshold_shift mutant (circ_max_snr > 3.2).
        """
        t, f, p, t0, dur, p_depth = generate_circ_near_threshold_3_15_waveform()
        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertTrue(is_eb, "Secondary eclipse with SNR ~ 3.15 must qualify under threshold > 3.10")
        self.assertAlmostEqual(snr, 3.1497, places=3)
        self.assertGreater(depth, 100.0)
        self.assertEqual(phi, 0.50)

    def test_waveform_competing_ecc_shallow_fails_depth_floor(self):
        """
        C=True, E=False: Eccentric raw peak has higher SNR than circular, but fails the depth floor (depth <= 100).
        Fixed detector must select the qualifying circular branch.
        """
        np.random.seed(42)
        t = np.linspace(0.0, 25.0, 25000)
        period = 4.0
        t0 = 1.0
        dur = 0.10
        primary_depth_ppm = 1500.0
        f = np.ones_like(t) + np.random.normal(0, 0.000005, size=len(t))

        phase_0to1 = ((t - t0) % period) / period
        circ_phase_diff = np.abs(((phase_0to1 - 0.50 + 0.5) % 1.0) - 0.5)
        sec_phase_diff = np.abs(((phase_0to1 - 0.35 + 0.5) % 1.0) - 0.5)

        # Circ: depth 200 ppm (passes > 100 and >= 150)
        f[circ_phase_diff < (0.5 * dur / period)] -= 0.000200
        # Ecc: depth 80 ppm (fails depth floor <= 100), but very high SNR
        f[sec_phase_diff < (0.25 * dur / period)] -= 0.000080

        is_eb, snr, depth, phi, z_crit = scan_full_orbit_secondary_eclipse(t, f, period, t0, dur, primary_depth_ppm)
        self.assertTrue(is_eb)
        self.assertEqual(phi, 0.50, "Qualifying circular branch must be selected despite higher-SNR shallow eccentric peak")
        self.assertGreater(depth, 100.0)

    def test_ast_extracted_decision_core_truth_table_all_combinations(self):
        """
        Exhaustive test of source-derived AST decision core across all combinations:
        - C=True, E=False
        - C=False, E=True
        - C=True, E=True (circ > ecc, ecc > circ, and exact tie)
        - C=False, E=False (circ > ecc, ecc > circ, and exact tie)
        """
        core = PRODUCTION_AST_DECISION_CORE
        z_crit = 4.0
        p_depth = 1500.0

        # 1. C=True, E=False (circ qualifies, ecc fails significance)
        res = core(circ_max_snr=3.5, circ_max_depth=200.0,
                   ecc_max_snr=3.8, ecc_max_depth=250.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 3.5, 200.0, 0.50, z_crit))

        # 2. C=False, E=True (ecc qualifies, circ fails significance even if circ_snr > ecc_snr)
        res = core(circ_max_snr=3.0, circ_max_depth=200.0,
                   ecc_max_snr=4.5, ecc_max_depth=250.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 4.5, 250.0, 0.35, z_crit))

        # 3. C=True, E=True, circ > ecc
        res = core(circ_max_snr=5.0, circ_max_depth=300.0,
                   ecc_max_snr=4.5, ecc_max_depth=250.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 5.0, 300.0, 0.50, z_crit))

        # 4. C=True, E=True, ecc > circ
        res = core(circ_max_snr=4.2, circ_max_depth=200.0,
                   ecc_max_snr=6.0, ecc_max_depth=400.0, ecc_best_phi=0.28,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 6.0, 400.0, 0.28, z_crit))

        # 5. C=True, E=True, exact tie (>= convention gives tie to circular)
        res = core(circ_max_snr=5.0, circ_max_depth=300.0,
                   ecc_max_snr=5.0, ecc_max_depth=300.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 5.0, 300.0, 0.50, z_crit), "Tie between qualifying branches must go to circular")

        # 6. C=False, E=False, circ > ecc -> diagnostic returns circ
        res = core(circ_max_snr=2.5, circ_max_depth=50.0,
                   ecc_max_snr=2.0, ecc_max_depth=40.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (False, 2.5, 50.0, 0.50, z_crit))

        # 7. C=False, E=False, ecc > circ -> diagnostic returns ecc
        res = core(circ_max_snr=2.0, circ_max_depth=40.0,
                   ecc_max_snr=2.8, ecc_max_depth=60.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (False, 2.8, 60.0, 0.35, z_crit))

        # 8. C=False, E=False, exact tie -> diagnostic tie goes to ecc (> convention falls to else)
        res = core(circ_max_snr=2.5, circ_max_depth=50.0,
                   ecc_max_snr=2.5, ecc_max_depth=50.0, ecc_best_phi=0.35,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (False, 2.5, 50.0, 0.35, z_crit), "Legacy diagnostic tie behavior must be preserved")

    def test_ast_extracted_competing_branch_nonqualifying_due_to_depth_or_ratio(self):
        """
        Tests competing branches where higher raw SNR fails depth floor (<= 100)
        or primary depth ratio (< 0.10 * primary) on source-extracted AST core.
        """
        core = PRODUCTION_AST_DECISION_CORE
        z_crit = 4.0
        p_depth = 2000.0  # 10% is 200 ppm

        # Competing Ecc has higher SNR (10.0 > 4.0), but depth=90 ppm <= 100 ppm floor
        res = core(circ_max_snr=4.0, circ_max_depth=300.0,
                   ecc_max_snr=10.0, ecc_max_depth=90.0, ecc_best_phi=0.25,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 4.0, 300.0, 0.50, z_crit))

        # Competing Ecc has higher SNR (10.0 > 4.0), depth=150 ppm (> 100), but ratio=150 < 200 (10% of 2000)
        res = core(circ_max_snr=4.0, circ_max_depth=300.0,
                   ecc_max_snr=10.0, ecc_max_depth=150.0, ecc_best_phi=0.25,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 4.0, 300.0, 0.50, z_crit))

        # Competing Circ has higher SNR (10.0 > 5.0), but depth=90 ppm <= 100 ppm floor
        res = core(circ_max_snr=10.0, circ_max_depth=90.0,
                   ecc_max_snr=5.0, ecc_max_depth=400.0, ecc_best_phi=0.25,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 5.0, 400.0, 0.25, z_crit))

        # Competing Circ has higher SNR (10.0 > 5.0), depth=150 ppm, but ratio=150 < 200
        res = core(circ_max_snr=10.0, circ_max_depth=150.0,
                   ecc_max_snr=5.0, ecc_max_depth=400.0, ecc_best_phi=0.25,
                   primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertEqual(res, (True, 5.0, 400.0, 0.25, z_crit))


class TestSecondaryBranchSelectionA4Boundaries(unittest.TestCase):
    """A4: Strict scientific threshold boundaries (> 3.1, > z_crit, > 100, >= 0.10 * primary)."""

    def test_strict_snr_thresholds(self):
        """Verifies strict inequality for SNR thresholds (> 3.10 and > z_crit) on AST core."""
        core = PRODUCTION_AST_DECISION_CORE
        z_crit = 4.05
        p_depth = 1000.0

        # Circular SNR exactly 3.1000000 fails (strict >)
        res_fail = core(circ_max_snr=3.1000000, circ_max_depth=200.0,
                        ecc_max_snr=1.0, ecc_max_depth=50.0, ecc_best_phi=0.2,
                        primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertFalse(res_fail[0], "SNR == 3.10 must not qualify for circular")

        # Circular SNR 3.1000001 passes
        res_pass = core(circ_max_snr=3.1000001, circ_max_depth=200.0,
                        ecc_max_snr=1.0, ecc_max_depth=50.0, ecc_best_phi=0.2,
                        primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertTrue(res_pass[0], "SNR > 3.10 must qualify for circular")

        # Eccentric SNR exactly z_crit fails (strict >)
        res_fail_ecc = core(circ_max_snr=1.0, circ_max_depth=50.0,
                            ecc_max_snr=z_crit, ecc_max_depth=200.0, ecc_best_phi=0.2,
                            primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertFalse(res_fail_ecc[0], "SNR == z_crit must not qualify for eccentric")

        # Eccentric SNR z_crit + 1e-6 passes
        res_pass_ecc = core(circ_max_snr=1.0, circ_max_depth=50.0,
                            ecc_max_snr=z_crit + 1e-6, ecc_max_depth=200.0, ecc_best_phi=0.2,
                            primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertTrue(res_pass_ecc[0], "SNR > z_crit must qualify for eccentric")

    def test_strict_depth_and_ratio_thresholds(self):
        """Verifies depth floor (> 100 ppm) and ratio threshold (>= 0.10 * primary) on AST core."""
        core = PRODUCTION_AST_DECISION_CORE
        z_crit = 4.0
        p_depth = 1000.0

        # Depth exactly 100.0 ppm fails (strict > 100.0)
        res_fail_depth = core(circ_max_snr=5.0, circ_max_depth=100.000000,
                              ecc_max_snr=1.0, ecc_max_depth=50.0, ecc_best_phi=0.2,
                              primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertFalse(res_fail_depth[0], "Depth == 100.0 ppm must not qualify")

        # Depth 100.000001 ppm passes
        res_pass_depth = core(circ_max_snr=5.0, circ_max_depth=100.000001,
                              ecc_max_snr=1.0, ecc_max_depth=50.0, ecc_best_phi=0.2,
                              primary_depth_ppm=p_depth, z_crit=z_crit)
        self.assertTrue(res_pass_depth[0], "Depth > 100.0 ppm must qualify")

        # Primary depth = 1500 ppm -> 10% is 150.0 ppm
        # Depth exactly 150.0 ppm passes (ratio >= 0.10)
        res_pass_ratio = core(circ_max_snr=5.0, circ_max_depth=150.000000,
                              ecc_max_snr=1.0, ecc_max_depth=50.0, ecc_best_phi=0.2,
                              primary_depth_ppm=1500.0, z_crit=z_crit)
        self.assertTrue(res_pass_ratio[0], "Ratio == 0.10 must qualify (>= convention)")

        # Depth 149.99999 ppm fails (< 0.10)
        res_fail_ratio = core(circ_max_snr=5.0, circ_max_depth=149.999990,
                              ecc_max_snr=1.0, ecc_max_depth=50.0, ecc_best_phi=0.2,
                              primary_depth_ppm=1500.0, z_crit=z_crit)
        self.assertFalse(res_fail_ratio[0], "Ratio < 0.10 must not qualify")


class TestSecondaryBranchSelectionA5Contract(unittest.TestCase):
    """A5: Early return contracts, metadata types, and z_crit meaning preservation."""

    def test_min_phi_early_return(self):
        """When min_phi >= 0.45, returns exact 5-tuple (False, 0.0, 0.0, 0.5, 4.0)."""
        t = np.linspace(0.0, 25.0, 1000)
        f = np.ones_like(t)
        res = scan_full_orbit_secondary_eclipse(t, f, period=0.3, t0=1.0, dur=0.1, primary_depth_ppm=1500.0)
        self.assertEqual(res, (False, 0.0, 0.0, 0.5, 4.0))

    def test_insufficient_oot_points_early_return(self):
        """When oot points < 50, returns exact 5-tuple (False, 0.0, 0.0, 0.5, 4.0)."""
        t = np.linspace(0.0, 25.0, 30)
        f = np.ones_like(t)
        res = scan_full_orbit_secondary_eclipse(t, f, period=4.0, t0=1.0, dur=0.1, primary_depth_ppm=1500.0)
        self.assertEqual(res, (False, 0.0, 0.0, 0.5, 4.0))

    def test_return_tuple_contract_and_z_crit_preservation(self):
        """
        Public return contract is 5-tuple (is_eb, max_sec_snr, max_sec_depth_ppm, best_sec_phase, z_crit).
        z_crit remains the eccentric trials-factor threshold even when circular branch is selected.
        """
        sec03 = generate_sec03_fixtures()
        res = scan_full_orbit_secondary_eclipse(
            sec03["t"], sec03["f_circ_only"], sec03["period"], sec03["t0"], sec03["dur"], sec03["primary_depth_ppm"]
        )
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 5)
        is_eb, snr, depth, phi, z_crit = res
        self.assertIsInstance(is_eb, (bool, np.bool_))
        self.assertIsInstance(float(snr), float)
        self.assertIsInstance(float(depth), float)
        self.assertIsInstance(float(phi), float)
        self.assertIsInstance(float(z_crit), float)
        self.assertAlmostEqual(z_crit, 4.067016, places=4)
        self.assertEqual(phi, 0.50)


class TestSecondaryBranchSelectionA6Mutants(unittest.TestCase):
    """
    A6: Actual deliberate negative controls.
    Executes isolated copies of the detector with injected selection faults
    and demonstrates that the mutant fails the test oracle while production passes.
    """

    @classmethod
    def setUpClass(cls):
        cls.sec03 = generate_sec03_fixtures()

    def test_mutant_1_old_selection_guard_fails_sec03_oracle(self):
        """
        Mutant 1: Restores the old selection guard (suppression of circ by higher ecc raw SNR).
        Oracle: is_eb must be True on combined SEC03 fixture.
        Mutant FAILS (returns False); production PASSES (returns True).
        """
        mut_res = MUTANT_OLD_BASELINE(
            self.sec03["t"], self.sec03["f_combined"], self.sec03["period"],
            self.sec03["t0"], self.sec03["dur"], self.sec03["primary_depth_ppm"]
        )
        self.assertFalse(mut_res[0], "Mutant 1 (old selection guard) must fail the oracle by returning False")

        prod_res = scan_full_orbit_secondary_eclipse(
            self.sec03["t"], self.sec03["f_combined"], self.sec03["period"],
            self.sec03["t0"], self.sec03["dur"], self.sec03["primary_depth_ppm"]
        )
        self.assertTrue(prod_res[0], "Production detector must pass the oracle by returning True")

    def test_mutant_2_or_to_and_fault_fails_single_qualifier_oracle(self):
        """
        Mutant 2: OR-to-AND fault requires BOTH circular and eccentric to qualify.
        Oracle: is_eb must be True on circular-only fixture.
        Mutant FAILS (returns False); production PASSES (returns True).
        """
        mut_res = MUTANT_OR_TO_AND(
            self.sec03["t"], self.sec03["f_circ_only"], self.sec03["period"],
            self.sec03["t0"], self.sec03["dur"], self.sec03["primary_depth_ppm"]
        )
        self.assertFalse(mut_res[0], "Mutant 2 (OR-to-AND fault) must fail the oracle by returning False on circ-only")

        prod_res = scan_full_orbit_secondary_eclipse(
            self.sec03["t"], self.sec03["f_circ_only"], self.sec03["period"],
            self.sec03["t0"], self.sec03["dur"], self.sec03["primary_depth_ppm"]
        )
        self.assertTrue(prod_res[0], "Production detector must pass the oracle by returning True on circ-only")

    def test_mutant_3_metadata_swap_fault_fails_metadata_oracle(self):
        """
        Mutant 3: Swaps circular selected metadata with eccentric metadata.
        Oracle: best_phi must be 0.50 when circular branch qualifies on SEC03.
        Mutant FAILS (returns eccentric best_phi != 0.50); production PASSES (returns 0.50).
        """
        mut_res = MUTANT_METADATA_SWAP(
            self.sec03["t"], self.sec03["f_combined"], self.sec03["period"],
            self.sec03["t0"], self.sec03["dur"], self.sec03["primary_depth_ppm"]
        )
        self.assertNotEqual(mut_res[3], 0.50, "Mutant 3 (metadata swap) must fail the metadata oracle (best_phi != 0.50)")

        prod_res = scan_full_orbit_secondary_eclipse(
            self.sec03["t"], self.sec03["f_combined"], self.sec03["period"],
            self.sec03["t0"], self.sec03["dur"], self.sec03["primary_depth_ppm"]
        )
        self.assertEqual(prod_res[3], 0.50, "Production detector must pass the metadata oracle (best_phi == 0.50)")

    def test_mutant_4_inverted_tie_breaker_fails_tie_oracle(self):
        """
        Mutant 4: Uses strict > instead of >= for tie between circular and eccentric qualifying branches.
        Oracle: When circ_max_snr == ecc_max_snr and both qualify, tie must go to circular (phi == 0.50).
        Executes real MUTANT_INVERTED_TIE against the exact-tie waveform.
        Mutant FAILS (selects eccentric phi != 0.50); production PASSES (selects circular phi == 0.50).
        """
        t, f, p, t0, dur, p_depth = generate_exact_tie_waveform()

        # Production execution on waveform
        prod_res = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertEqual(prod_res[3], 0.50, "Production detector must select circular branch (phi=0.50) on exact tie")

        # Mutant execution on same waveform
        mut_res = MUTANT_INVERTED_TIE(t, f, p, t0, dur, p_depth)
        self.assertNotEqual(mut_res[3], 0.50, "Mutant 4 (inverted tie) must fail the tie oracle by not returning 0.50")

    def test_mutant_5_circ_threshold_shift_fails_threshold_oracle(self):
        """
        Mutant 5: Shifts circular threshold from > 3.1 to > 3.2.
        Oracle: An eclipse with SNR ~ 3.15 must qualify as EB (3.15 > 3.10).
        Executes real MUTANT_CIRC_THRESHOLD_SHIFT against the 3.15 SNR waveform.
        Mutant FAILS (returns is_eb=False); production PASSES (returns is_eb=True).
        """
        t, f, p, t0, dur, p_depth = generate_circ_near_threshold_3_15_waveform()

        prod_res = scan_full_orbit_secondary_eclipse(t, f, p, t0, dur, p_depth)
        self.assertTrue(prod_res[0], "Production detector must detect secondary with SNR ~ 3.15")

        mut_res = MUTANT_CIRC_THRESHOLD_SHIFT(t, f, p, t0, dur, p_depth)
        self.assertFalse(mut_res[0], "Mutant 5 (threshold shift > 3.2) must fail the oracle by returning False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
