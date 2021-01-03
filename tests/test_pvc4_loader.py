import sys
sys.path.append('../')

from loaders import pvc4
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestPvc4Loader(unittest.TestCase):
    def test_openimfile(self):
        framecount, iconsize, iconside, filetype = pvc4._openimfile(
            '../data/crcns-pvc4/Nat/r0208D/test.review.mountlake.30_pix.2sizes.imsm')
        self.assertEqual(framecount, 756)
        self.assertEqual(iconsize, 14400)
        self.assertEqual(iconside, 120)
        self.assertEqual(filetype, 2)

    def test_loadimfile(self):
        data = pvc4._loadimfile(
            '../data/crcns-pvc4/Nat/r0208D/test.review.mountlake.30_pix.2sizes.imsm')
        self.assertEqual(data.shape[2], 120)
        self.assertEqual(data.shape[0], 756)

    def test_loadimfile_iconsize0(self):
        data = pvc4._loadimfile(
            '../data/crcns-pvc4/NatRev/r0156A/test.natrev.size.mountlake.imsm')
        self.assertEqual(data.shape[2], 96)
        self.assertEqual(data.shape[0], 7228)

    def test_train(self):
        loader = pvc4.PVC4('../data/crcns-pvc4', 
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

    @unittest.skip("Slow")
    def test_tune(self):
        _ = pvc4.PVC4('../data/crcns-pvc4', nt=32, split='tune')

    @unittest.skip("Slow")
    def test_report(self):
        _ = pvc4.PVC4('../data/crcns-pvc4', nt=32, split='report')


if __name__ == '__main__':
    unittest.main()