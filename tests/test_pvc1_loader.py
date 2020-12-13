import sys
sys.path.append('../')

import pvc1_loader
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestPvc1Loader(unittest.TestCase):
    def test_batching(self):
        loader = pvc1_loader.PVC1('../crcns-ringach-data/',
                                  ntau=7)

        trainloader = torch.utils.data.DataLoader(loader, 
                                                  batch_size=8, 
                                                  shuffle=True)
        
        data = next(iter(trainloader))
        (X, rg, y) = data

        self.assertEqual(X.shape[0], rg.shape[0])
        self.assertEqual(rg.shape[0], y.shape[0])
        self.assertEqual(rg.shape[1], y.shape[1])
        self.assertEqual(y.shape[2], loader.nt)

    def test_sequential(self):
        loader = pvc1_loader.PVC1('../crcns-ringach-data/',
                                  ntau=7)
        
        (X, rg, y) = loader[0]

        self.assertEqual(X.shape[1], loader.nt + loader.ntau - 1)
        self.assertEqual(y.shape[1], loader.nt)

        # We shouldn't have missing data between the frames.
        self.assertEqual(loader.sequence[0]['spike_frames'][-1], 
                         loader.sequence[1]['spike_frames'][0])

        self.assertEqual(loader.sequence[0]['end_frame'] - loader.ntau + 1,
                         loader.sequence[1]['start_frame'])

        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[2], loader.ny)
        self.assertEqual(X.shape[3], loader.nx)

        # Check it doesn't crash at the edges
        for i, data in enumerate(loader):
            if i > 100:
                break

        self.assertTrue(True)

    @unittest.skip("Slow")
    def test_download_cached(self):
        tmp_dir = tempfile.mkdtemp()

        # The first time around, it should be very slow.
        self.assertTrue(pvc1_loader.download(tmp_dir))

        # The second time around, it should be much faster.
        t0 = time.time()
        self.assertTrue(pvc1_loader.download(tmp_dir))
        self.assertLess(time.time() - t0, 2)


if __name__ == '__main__':
    unittest.main()