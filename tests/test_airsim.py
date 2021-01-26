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
        self.assertLess(X.min(), -1)
        self.assertGreater(X.min(), -3)
        self.assertGreater(X.max(), 1)
        self.assertLess(X.max(),  3)

if __name__ == '__main__':
    unittest.main()