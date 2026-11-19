"""N1 and N1-Archival null constructions (ARGON_DECISIONS item 31.7; roadmap 0.6; NULL-FIDELITY-01 preregistration
sections 4 and 5, carried unchanged by item 32).

The transforms act on the product files, before any production detrending or masking: the production reader then
sees a real light curve whose valid flux values are in a different order. TIME, TIMECORR, CADENCENO and QUALITY,
headers and invalid cadences stay as they are; every other per-cadence column moves with its cadence.

  N1           split the valid cadences at gaps above 0.5 d (the orbit gap). Two segments: orbit 1 is time-reversed
               onto its own valid timestamps, orbit 2 stays forward. More than two: segments are taken in pairs, each
               pair contributes the later segment forward and then the earlier one reversed; an odd last segment is
               reversed. The discovery target pixel file gets the same cadence permutation.
  N1-Archival  the N1 permutation, then a circular shift of the permuted valid values by round(f_s * N_s) cadences,
               f_s = 0.25 + 0.5 * frac(s * phi), phi = (sqrt(5) - 1) / 2, s the sector number.

The first block of functions is the accepted NULL-FIDELITY-01 code (agent_exchange/null_fidelity_01/n1.py), moved
unchanged; agent_exchange/s4_m_package/port_equivalence.py checks AST equality function by function.
"""
import math
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
from astropy.io import fits

KEEP_COLUMNS = ('TIME', 'TIMECORR', 'CADENCENO', 'QUALITY')
GAP_DAYS = 0.5
PHI = (math.sqrt(5.0) - 1.0) / 2.0


def valid_mask(data):
    """Section 4.1: cadences extract_and_stitch_arrays can keep."""
    t = np.asarray(data['TIME'], float)
    f = np.asarray(data['PDCSAP_FLUX'], float)
    q = np.asarray(data['QUALITY'])
    with np.errstate(invalid='ignore'):
        return np.isfinite(t) & np.isfinite(f) & (f > 0) & (q == 0)


def segments(tv):
    """Section 4.2: positions (into the valid cadences) of each segment."""
    n = len(tv)
    if n < 2:
        return [np.arange(n)]
    d = np.diff(tv)
    cuts = np.where(d > GAP_DAYS)[0] + 1
    if len(cuts) == 0:
        cuts = np.array([int(np.argmax(d)) + 1])
    return np.split(np.arange(n), cuts)


def n1_order(tv):
    """Sections 4.3 and 4.4: src[i] = valid position whose value is placed on valid position i."""
    segs = segments(tv)
    if len(segs) == 2:
        return np.concatenate([segs[0][::-1], segs[1]]), len(segs)
    parts = []
    for j in range(0, len(segs) - 1, 2):
        parts += [segs[j + 1], segs[j][::-1]]
    if len(segs) % 2 == 1:
        parts.append(segs[-1][::-1])
    return np.concatenate(parts), len(segs)


def archival_fraction(sector):
    """Section 5.2: f_s = 0.25 + 0.5 x frac(s x phi)."""
    return 0.25 + 0.5 * math.modf(int(sector) * PHI)[0]


def permutation(data, archival_sector=None):
    """Full-row permutation (identity on invalid rows) and a description."""
    v = np.where(valid_mask(data))[0]
    tv = np.asarray(data['TIME'], float)[v]
    src, nseg = n1_order(tv)
    info = dict(n_rows=len(data), n_valid=int(len(v)), n_segments=int(nseg),
                segment_lengths=[int(len(s)) for s in segments(tv)])
    if archival_sector is not None:
        f = archival_fraction(archival_sector)
        k = int(round(f * len(v)))
        src = np.roll(src, k)
        info.update(archival_fraction=f, roll_cadences=k, baseline_days=float(tv[-1] - tv[0]),
                    shift_days_nominal=float(f * (tv[-1] - tv[0])))
    perm = np.arange(len(data))
    perm[v] = v[src]
    return perm, info


def _permute_table(hdu, perm):
    for name in hdu.columns.names:
        if name in KEEP_COLUMNS:
            continue
        col = hdu.data[name]
        hdu.data[name] = col[perm].copy()


def transform_lc(path, archival_sector=None):
    """In place. Returns (info, cadence permutation as dict CADENCENO -> source CADENCENO for valid rows)."""
    path = Path(path)
    with fits.open(path, memmap=False) as h:
        perm, info = permutation(h[1].data, archival_sector)
        cad = np.asarray(h[1].data['CADENCENO']).copy()
        valid_cad = cad[valid_mask(h[1].data)].tolist()
        _permute_table(h[1], perm)
        h.writeto(path, overwrite=True, output_verify='exception')
    moved = perm != np.arange(len(perm))
    return info, dict(zip(cad[moved].tolist(), cad[perm[moved]].tolist())), valid_cad


def transform_tpf(path, lc_cad_map, lc_valid_cadences):
    """Section 4.6: the light curve permutation applied to target pixel rows matched by CADENCENO."""
    path = Path(path)
    with fits.open(path, memmap=False) as h:
        headers = []
        with path.open('rb') as stream:
            for hdu in h:
                info = hdu.fileinfo()
                stream.seek(info['hdrLoc'])
                headers.append((info['hdrLoc'], stream.read(info['datLoc'] - info['hdrLoc']),
                                Counter(card.image for card in hdu.header.cards)))
        cad = np.asarray(h[1].data['CADENCENO'])
        pos = {int(c): i for i, c in enumerate(cad)}
        missing = [c for c in lc_valid_cadences if int(c) not in pos]
        if missing:
            return dict(ok=False, reason='LC_VALID_CADENCE_MISSING_IN_TPF', n_missing=len(missing))
        perm = np.arange(len(cad))
        for dst, src in lc_cad_map.items():
            perm[pos[int(dst)]] = pos[int(src)]
        _permute_table(h[1], perm)
        h.writeto(path, overwrite=True, output_verify='exception')
    # Astropy reorders column-display cards during serialization. Restore the exact
    # original blocks only after proving that layout and every card image survived.
    with fits.open(path, memmap=False) as written:
        if len(written) != len(headers):
            raise ValueError('TPF_HDU_LAYOUT_CHANGED')
        for hdu, (offset, block, cards) in zip(written, headers):
            info = hdu.fileinfo()
            if (info['hdrLoc'] != offset or info['datLoc'] != offset + len(block) or
                    Counter(card.image for card in hdu.header.cards) != cards):
                raise ValueError('TPF_HEADER_CONTENT_OR_LAYOUT_CHANGED')
    with path.open('r+b') as stream:
        for offset, block, _ in headers:
            stream.seek(offset)
            stream.write(block)
    return dict(ok=True, n_rows=int(len(cad)), n_moved=int(np.sum(perm != np.arange(len(cad)))))


# ------------------------------------------------------------------------------------------- harness additions
NULL_KINDS = ("N1", "N2")


def n1_values(time, values, archival_sector=None):
    """The N1 (or N1-Archival) value sequence for already valid, time-ordered cadences: values[src] placed on the
    unchanged timestamps. Array form of `permutation`, for tests and for arrays that never were a file."""
    time = np.asarray(time, float)
    src, nseg = n1_order(time)
    if archival_sector is not None:
        src = np.roll(src, int(round(archival_fraction(archival_sector) * len(time))))
    return np.asarray(values)[src], dict(n_segments=int(nseg), source_index=src)


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    floats = np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating)
    return bool(np.array_equal(a, b, equal_nan=floats))


def invariants(orig, new, archival_sector=None):
    """Transform invariants of one file pair (NULL-FIDELITY-01 control V2): headers, the kept columns, the other
    HDUs and the invalid rows are identical; the production valid mask is identical; every transformed column holds
    the same multiset of valid values; a two-segment N1 applied twice is the identity; and the file really changed
    (a transform that keeps the original order fails `values_moved`)."""
    out = dict(file=Path(new).name)
    with fits.open(orig, memmap=False) as a, fits.open(new, memmap=False) as b:
        out['n_hdu_equal'] = len(a) == len(b)
        out['headers_equal'] = all(x.header.tostring() == y.header.tostring() for x, y in zip(a, b))
        out['other_hdus_equal'] = all(_eq(a[i].data, b[i].data) for i in range(len(a)) if i != 1 and a[i].data is not None)
        da, db = a[1].data, b[1].data
        out['keep_columns_equal'] = all(_eq(da[c], db[c]) for c in KEEP_COLUMNS if c in da.columns.names)
        if 'PDCSAP_FLUX' in da.columns.names:
            va, vb = valid_mask(da), valid_mask(db)
            out['valid_mask_equal'] = _eq(va, vb)
            inv_ok, multiset_ok = True, True
            for c in da.columns.names:
                if c in KEEP_COLUMNS:
                    continue
                xa, xb = np.asarray(da[c]), np.asarray(db[c])
                inv_ok &= _eq(xa[~va], xb[~va])
                multiset_ok &= _eq(np.sort(xa[va], axis=0), np.sort(xb[va], axis=0))
            out['invalid_rows_equal'] = bool(inv_ok)
            out['valid_multiset_equal'] = bool(multiset_ok)
            perm, info = permutation(da, archival_sector)
            out['n_segments'] = info['n_segments']
            out['values_placed_by_permutation'] = _eq(np.asarray(da['PDCSAP_FLUX'])[perm], np.asarray(db['PDCSAP_FLUX']))
            out['values_moved'] = bool(np.any(perm != np.arange(len(perm))))
            if info['n_segments'] == 2 and archival_sector is None:
                out['involution'] = bool(np.array_equal(perm[perm], np.arange(len(perm))))
    out['passed'] = all(v for v in out.values() if isinstance(v, bool))
    return out


def transform_case(case, root, sha256_file):
    """Apply N1 to the discovery light curve and target pixel file of one detector case and N1-Archival to every
    other light curve, in place under root, and update the SHA256 fields of the case. Returns the transform record
    with the invariants of every file. A target pixel file that lacks a valid light curve cadence leaves the case
    NOT_RUN (ok False) and the caller reports it; nothing is skipped silently."""
    root = Path(root)
    info = dict(ok=True, discovery=None, tpf=None, archival=[], invariants=[])
    backups = {}
    for r in case['lightcurves']:
        p = root / r['path']
        backups[r['path']] = p.with_name(p.name + '.orig')
        shutil.copyfile(p, backups[r['path']])
    tp = root / case['tpf_fits']
    tp_orig = tp.with_name(tp.name + '.orig')
    shutil.copyfile(tp, tp_orig)
    try:
        for r in case['lightcurves']:
            p = root / r['path']
            if int(r['sector']) == int(case['sector']):
                i, cmap, vcad = transform_lc(p)
                info['discovery'] = i
                info['tpf'] = transform_tpf(tp, cmap, vcad)
                info['invariants'].append(invariants(backups[r['path']], p))
                if info['tpf']['ok']:
                    info['invariants'].append(dict(invariants(tp_orig, tp), tpf=True))
            else:
                i, _, _ = transform_lc(p, archival_sector=r['sector'])
                info['archival'].append(dict(i, sector=int(r['sector'])))
                info['invariants'].append(invariants(backups[r['path']], p, archival_sector=r['sector']))
            r['sha256'] = sha256_file(p)
        disc = next(x for x in case['lightcurves'] if int(x['sector']) == int(case['sector']))
        case['source_sha256'] = disc['sha256']
        case['tpf_sha256'] = sha256_file(tp)
    finally:
        for o in list(backups.values()) + [tp_orig]:
            if o.exists():
                o.unlink()
    info['ok'] = bool(info['tpf'] and info['tpf']['ok'] and all(v['passed'] for v in info['invariants']))
    if info['tpf'] and not info['tpf']['ok']:
        info['reason'] = info['tpf']['reason']
    elif not info['ok']:
        info['reason'] = 'TRANSFORM_INVARIANT_FAILED'
    return info


def corridor_reach_days(sigma_p_days, n_cycles, n_sigma=3.5):
    """Timing reach of the item 18 period corridor (+/- n_sigma sigma_P') at an archival sector n_cycles away."""
    return float(n_sigma * abs(sigma_p_days) * abs(n_cycles))


def shift_outside_corridor(shift_days, sigma_p_days, n_cycles, period_days, n_sigma=3.5):
    """Diagnostic for "outside the item 18 corridor": the N1-Archival shift, folded to the nearest transit of the
    trial period, compared with the corridor's timing reach. None when an input is missing."""
    vals = (shift_days, sigma_p_days, n_cycles, period_days)
    if any(v is None or not np.isfinite(v) for v in vals) or period_days <= 0:
        return None
    folded = abs(((shift_days + 0.5 * period_days) % period_days) - 0.5 * period_days)
    reach = corridor_reach_days(sigma_p_days, n_cycles, n_sigma)
    return dict(shift_days=float(shift_days), folded_offset_days=float(folded), corridor_reach_days=reach,
                outside=bool(folded > reach))
