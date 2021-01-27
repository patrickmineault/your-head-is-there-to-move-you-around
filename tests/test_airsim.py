import sys
sys.path.append('../')

from loaders import airsim
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestAirsimLoader(unittest.TestCase):
    def test_train(self):
        loader = airsim.AirSim('/mnt/e/data_derived/airsim', 
                           split='train',
                           )
        
        X, Y = loader[0]
        self.assertEqual(tuple(X.shape), (3, 11, 112, 112))
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(Y.size, 5)
        self.assertLess(X.min(), 128)
        self.assertGreater(X.min(), -1)
        self.assertGreater(X.max(), 128)
        self.assertLess(X.max(),  256)

    def test_multiclass(self):
        loader = airsim.AirSim('/mnt/e/data_derived/airsim', 
                           split='train',
                           regression=False
                           )
        
        X, Y = loader[0]
        self.assertEqual(tuple(X.shape), (3, 11, 112, 112))
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(Y.size, 4)
        self.assertLess(X.min(), 128)
        self.assertGreater(X.min(), -1)
        self.assertGreater(X.max(), 128)
        self.assertLess(X.max(),  256)
        self.assertEqual(Y.dtype, np.int64)
        self.assertTrue(np.any(Y > 0))

if __name__ == '__main__':
    unittest.main()