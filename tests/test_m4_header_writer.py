"""Header-byte preservation regression and strict invariant controls."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m_harness import fixtures, nulls


class HeaderWriterTests(unittest.TestCase):
    def test_serializer_card_value_change_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = fixtures.synthetic_host(root, 990003, 40, archival_sectors=(), seed=25)
            path = root / host['tpf_fits']
            cad = fits.getdata(path, 1)['CADENCENO'].tolist()
            original_write = fits.HDUList.writeto
            def corrupted(hdus, filename, *args, **kwargs):
                original_write(hdus, filename, *args, **kwargs)
                with fits.open(filename, mode='update', memmap=False) as written:
                    written[1].header['OBJECT'] = 'CORRUPTED'
            with patch.object(fits.HDUList, 'writeto', corrupted):
                with self.assertRaisesRegex(ValueError, 'TPF_HEADER_CONTENT_OR_LAYOUT_CHANGED'):
                    nulls.transform_tpf(path, dict(zip(cad, cad[::-1])), cad)
    def test_noncanonical_column_card_order_survives_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = fixtures.synthetic_host(root, 990001, 40, archival_sectors=(), seed=23)
            original = root / host['tpf_fits']
            # Add a real TPF-style display card, deliberately away from Astropy's preferred position.
            with fits.open(original, mode='update', memmap=False) as hdus:
                hdus[1].header.append(('TDISP4', 'E14.7', 'display format'))
            transformed = root / 'transformed.fits'
            shutil.copyfile(original, transformed)
            cad = fits.getdata(original, 1)['CADENCENO'].tolist()
            info = nulls.transform_tpf(transformed, dict(zip(cad, cad[::-1])), cad)
            self.assertTrue(info['ok'])
            with fits.open(original) as a, fits.open(transformed) as b:
                self.assertEqual(a[1].header.tostring(), b[1].header.tostring())
                np.testing.assert_array_equal(b[1].data['FLUX'], a[1].data['FLUX'][::-1])

    def test_reordered_header_still_fails_exact_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = fixtures.synthetic_host(root, 990002, 40, archival_sectors=(), seed=24)
            original = root / host['source_fits']
            transformed = root / 'changed.fits'
            shutil.copyfile(original, transformed)
            nulls.transform_lc(transformed)
            with fits.open(transformed) as hdus:
                start = hdus[1].fileinfo()['hdrLoc']
            with transformed.open('r+b') as stream:
                stream.seek(start)
                # COMMENT-free non-structural cards near the end; swap two actual card images.
                with fits.open(original) as hdus:
                    indices = [i for i, c in enumerate(hdus[1].header.cards) if c.keyword in ('TICID', 'SECTOR')]
                header_size = fits.getheader(original, 1).tostring().encode('ascii')
                stream.seek(start)
                full = bytearray(stream.read(len(header_size)))
                i, j = indices
                full[i*80:(i+1)*80], full[j*80:(j+1)*80] = full[j*80:(j+1)*80], full[i*80:(i+1)*80]
                stream.seek(start)
                stream.write(full)
            self.assertFalse(nulls.invariants(original, transformed)['headers_equal'])


if __name__ == '__main__':
    unittest.main()
