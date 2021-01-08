import sys
sys.path.append('../')

from loaders import pvc1
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestPvc1Loader(unittest.TestCase):
    def test_delays(self):
        loader = pvc1.PVC1('../data/crcns-ringach-data/',
                    nt=9,
                    ntau=7,
                    nframedelay=0)
        s0 = loader.sequence[0]
        # according to the docs, frame k runs from kT to (k+1)T,
        # where T = 1/30. Hence the end of the last frame should coincide
        # with the start of the response.
        self.assertEqual(s0['end_frame'], s0['spike_frames'][-2])

        loader = pvc1.PVC1('../data/crcns-ringach-data/',
                           nt=1,
                           ntau=7,
                           nframedelay=0)

        s0 = loader.sequence[0]
        self.assertEqual(s0['end_frame'], s0['spike_frames'][-2])

    @unittest.skip("Slow")
    def test_batching(self):
        loader = pvc1.PVC1('../data/crcns-ringach-data/',
                                  ntau=7)

        trainloader = torch.utils.data.DataLoader(loader, 
                                                  batch_size=8, 
                                                  shuffle=True)
        
        data = next(iter(trainloader))
        (X, rg, w, y) = data

        self.assertEqual(X.shape[0], rg.shape[0])
        self.assertEqual(rg.shape[0], y.shape[0])
        self.assertEqual(rg.shape[1], y.shape[1])
        
        self.assertEqual(y.ndim, 3)
        self.assertEqual(y.shape[2], loader.nt)
        self.assertEqual(y.shape[0], 8)

    @unittest.skip("Slow")
    def test_sequential(self):
        loader = pvc1.PVC1('../data/crcns-ringach-data/',
                                  ntau=7)
        
        (X, rg, w, y) = loader[0]

        self.assertEqual(X.ndim, 4)
        self.assertEqual(rg.ndim, 1)
        self.assertEqual(w.ndim, 1)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[0], len(w))
        self.assertEqual(y.shape[0], len(rg))

        self.assertEqual(X.shape[1], loader.nt + loader.ntau - 1)
        self.assertEqual(y.shape[1], loader.nt)

        # Same stimulus, different electrode.
        self.assertEqual(loader.sequence[0]['spike_frames'][-1], 
                         loader.sequence[1]['spike_frames'][-1])

        # self.assertEqual(loader.sequence[0]['end_frame'] - loader.ntau + 1,
        #                  loader.sequence[1]['start_frame'])

        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[2], 112)
        self.assertEqual(X.shape[3], 112)

        # Check it doesn't crash at the edges
        for i, data in enumerate(loader):
            if i > 100:
                break

        self.assertTrue(True)

    @unittest.skip("Slow")
    def test_download_cached(self):
        tmp_dir = tempfile.mkdtemp()

        # The first time around, it should be very slow.
        self.assertTrue(pvc1.download(tmp_dir))

        # The second time around, it should be much faster.
        t0 = time.time()
        self.assertTrue(pvc1.download(tmp_dir))
        self.assertLess(time.time() - t0, 2)


if __name__ == '__main__':
    unittest.main()