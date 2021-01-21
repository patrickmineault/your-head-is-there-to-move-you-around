import sys
sys.path.append('../')

from loaders import pvc4

import numpy as np
from pathlib import Path
import tempfile
import time
import unittest
import torch
from pprint import pprint


PVC4_PATH = Path('/mnt/e/data_derived/crcns-pvc4')
V2_PATH = Path('/mnt/e/data_derived/crcns-v2')

class TestPvc4Loader(unittest.TestCase):
    def test_virtual(self):
        loader = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='traintune',
                           virtual='rot'
                           )
        self.assertEqual(loader.total_electrodes, 100)
        X0, _, _, _ = loader[0]
        X1, _, _, _ = loader[len(loader) // 2]
        self.assertEqual(X0[0, 0, 0, 0], X1[0, 0, -1, -1])

    def test_openimfile(self):
        framecount, iconsize, iconside, filetype = pvc4._openimfile(
            str(PVC4_PATH / "Nat/r0208D/test.review.mountlake.30_pix.2sizes.imsm"))
        self.assertEqual(framecount, 756)
        self.assertEqual(iconsize, 14400)
        self.assertEqual(iconside, 120)
        self.assertEqual(filetype, 2)

    def test_loadimfile(self):
        data = pvc4._loadimfile(
            str(PVC4_PATH / 'Nat/r0208D/test.review.mountlake.30_pix.2sizes.imsm'))
        self.assertEqual(data.shape[2], 120)
        self.assertEqual(data.shape[0], 756)

    def test_loadimfile_iconsize0(self):
        data = pvc4._loadimfile(
            str(PVC4_PATH / 'NatRev/r0156A/test.natrev.size.mountlake.imsm'))
        self.assertEqual(data.shape[2], 96)
        self.assertEqual(data.shape[0], 7228)

    def test_disjoint(self):
        """Check that the train and report sets are disjoint."""
        traintune = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='traintune',
                           single_cell=0,
                           )
        reportset = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='report',
                           single_cell=0,
                           )

        traintune_paths = {s['images_path'] for s in traintune.sequence}
        reportset_paths = {s['images_path'] for s in reportset.sequence}

        self.assertEqual(len(traintune_paths), 5)
        self.assertEqual(len(reportset_paths), 1)
        self.assertEqual(len(traintune_paths.intersection(reportset_paths)), 0)

        # This one has a shared reportset
        traintune = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='traintune',
                           single_cell=6,
                           )
        reportset = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='report',
                           single_cell=6,
                           )
        traintune_paths = {s['images_path'] for s in traintune.sequence}
        reportset_paths = {s['images_path'] for s in reportset.sequence}

        self.assertEqual(len(traintune_paths), 2)
        self.assertEqual(len(reportset_paths), 1)
        assert len(traintune_paths.intersection(reportset_paths)) == 1

        traintune_seq = {str(PVC4_PATH / 'Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm'):
                        np.zeros(3600, dtype=np.bool)}
        reportset_seq = {str(PVC4_PATH / 'Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm'):
                        np.zeros(3600, dtype=np.bool)}

        for s in traintune.sequence:
            if s['images_path'] in traintune_seq:
                traintune_seq[s['images_path']][s['start_spktime']:s['end_spktime']] = True

        for s in reportset.sequence:
            if s['images_path'] in reportset_seq:
                reportset_seq[s['images_path']][s['start_spktime']:s['end_spktime']] = True

        self.assertEqual(np.sum(
            traintune_seq[str(PVC4_PATH / 'Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm')] & 
            reportset_seq[str(PVC4_PATH / 'Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm')]), 0)


    def test_train(self):
        loader = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='train',
                           )
        
        self.assertEqual(len({x['cellnum']: 1 for x in loader.sequence}), 
                         loader.total_electrodes)
        
        self.assertEqual(len({x['cellid']: 1 for x in loader.sequence}), 
                         loader.total_electrodes)

        X, m, W, y = loader[0]
        self.assertEqual(X.shape[3], 64)
        self.assertEqual(X.shape[1], loader.nt + loader.ntau - 1)
        self.assertEqual(m.shape, W.shape)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], 32)

    def test_traintune(self):
        loader = pvc4.PVC4(str(PVC4_PATH), 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='traintune',
                           )

        loader[0]

    def test_tune(self):
        _ = pvc4.PVC4(str(PVC4_PATH), nt=32, split='tune')

    def test_report(self):
        _ = pvc4.PVC4(str(PVC4_PATH), nt=32, split='report')

    @unittest.skip("Slow")
    def test_v2(self):
        for i in range(123):
            print(i)
            loader = pvc4.PVC4(str(V2_PATH), 
                            nt=32, 
                            nx=64,
                            ny=64,
                            split='train',
                            single_cell=i
                            )
            X, _, _, Y = loader[0]
            self.assertEqual(np.isnan(X).sum(), 0)
            self.assertEqual(np.isnan(Y).sum(), 0)

    def test_v2_nonzero(self):
        for i in range(117):
            with self.subTest(i=i):
                loader = pvc4.PVC4(str(V2_PATH),
                                nt=32, 
                                nx=64,
                                ny=64,
                                split='report',
                                single_cell=i
                                )
                sums = 0
                for _, _, _, Y in loader:
                    sums += Y.sum()
                self.assertGreater(sums, 0)

    def test_v2_nonzero_onecell(self):
        for single_cell in [24, 84, 86, 109]:
            with self.subTest(i=single_cell):
                loader = pvc4.PVC4(str(V2_PATH),
                                nt=32, 
                                nx=64,
                                ny=64,
                                split='train',
                                single_cell=single_cell
                                )

                loader = pvc4.PVC4(str(V2_PATH),
                                nt=32, 
                                nx=64,
                                ny=64,
                                split='tune',
                                single_cell=single_cell
                                )


                loader = pvc4.PVC4(str(V2_PATH),
                                nt=32, 
                                nx=64,
                                ny=64,
                                split='report',
                                single_cell=single_cell
                                )


if __name__ == '__main__':
    unittest.main()