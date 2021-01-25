import sys
sys.path.append('../')

from loaders import mt2
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint

class TestMt2Loader(unittest.TestCase):
    def test_train(self):
        loader = mt2.MT2('/mnt/e/data_derived/crcns-mt2', 
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
        loader = mt2.MT2('/mnt/e/data_derived/crcns-mt2', 
                           nt=32, 
                           nx=64,
                           ny=64,
                           split='traintune',
                           single_cell=0,
                           )

        data = loader[0]
        self.assertEqual(len(data), 4)

    def test_disjoint(self):
        loader1 = mt2.MT2('/mnt/e/data_derived/crcns-mt2', nt=32, split='traintune', single_cell=0)
        loader2 = mt2.MT2('/mnt/e/data_derived/crcns-mt2', nt=32, split='report', single_cell=0)

        self.assertLess(max([x['end_frame'] for x in loader1.sequence]),
                        min([x['start_frame'] for x in loader2.sequence]))



if __name__ == '__main__':
    unittest.main()