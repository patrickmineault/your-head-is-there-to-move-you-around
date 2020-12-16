import sys
sys.path.append('../')

import pvc2_loader
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestPvc1Loader(unittest.TestCase):
    def test_sequential(self):
        loader = pvc2_loader.PVC2('../data/crcns-pvc2/', ntau=7)

        for spktimes in loader.spktimes:
            for i in range(1, len(spktimes)):
                self.assertGreaterEqual(spktimes[i], spktimes[i-1])

        for framerate in loader.framerates:
            self.assertGreaterEqual(framerate, 23)
            self.assertGreaterEqual(24, framerate)

    def test_data(self):
        loader = pvc2_loader.PVC2('../data/crcns-pvc2/', nt=32, ntau=7)

        X, m, y = loader[0]
        
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[1], 38)
        self.assertEqual(X.shape[2], 12)
        self.assertEqual(X.shape[3], 12)

        self.assertEqual(m.size, y.shape[0])
        self.assertEqual(32, y.shape[1])

if __name__ == '__main__':
    unittest.main()