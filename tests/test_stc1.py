import sys
sys.path.append('../')

from loaders import stc1
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestAirsimLoader(unittest.TestCase):
    def test_mst(self):
        loader = stc1.Stc1('/mnt/e/data_derived/crcns-stc1', 
                           split='report',
                           subset='MSTd'
                           )
        
        X, Y = loader[0]
        self.assertEqual(tuple(X.shape), (3, 11, 112, 112))
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(Y.size, 129)

    def test_vip(self):
        loader = stc1.Stc1('/mnt/e/data_derived/crcns-stc1', 
                           split='report',
                           subset='VIP'
                           )
        
        X, Y = loader[0]
        self.assertEqual(tuple(X.shape), (3, 11, 112, 112))
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(Y.size, 95)

if __name__ == '__main__':
    unittest.main()