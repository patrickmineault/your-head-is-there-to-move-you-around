import sys
sys.path.append('../')

from loaders import vim2
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint

data_root = '../data/crcns-vim2/derived/'

class TestVim2Loader(unittest.TestCase):
    def test_sequence(self):
        loader = vim2.Vim2(data_root, 
                           nx=112, 
                           ny=112,
                           split='train')
        self.assertGreater(len(loader.sequence), 0)
        self.assertTrue(np.all([np.all(x['stim_idx'] >= -1) for x in loader.sequence]))

        loader = vim2.Vim2(data_root, 
                           nx=112, 
                           ny=112,
                           split='tune')
        self.assertGreater(len(loader.sequence), 0)
        self.assertTrue(np.all([np.all(x['stim_idx'] >= -1) for x in loader.sequence]))

        loader = vim2.Vim2(data_root, 
                           nx=112, 
                           ny=112,
                           split='report')
        self.assertGreater(len(loader.sequence), 0)
        self.assertTrue(np.all([np.all(x['stim_idx'] >= -1) for x in loader.sequence]))

    def test_rois(self):
        loader = vim2.Vim2(data_root, 
                           nx=112, 
                           ny=112,
                           split='train')
        rois = loader._get_rois()
        self.assertGreater(len(rois), 0)
        self.assertIn('v4rh', rois.keys())

    def test_train(self):
        for split in ('train', 'tune', 'report'):
            loader = vim2.Vim2(data_root, 
                            nt=9,
                            ntau=3,
                            nx=112, 
                            ny=112,
                            split=split)

            X, Y = loader[0]

            self.assertFalse(np.isnan(Y).any())
            self.assertEqual(X.shape[-1], 112)
            self.assertEqual(X.shape[-2], 112)
            self.assertEqual(X.shape[0], 3)
            self.assertEqual(X.shape[1], (loader.nt + loader.ntau - 1) * 15)

            self.assertEqual(Y.shape[0], loader.nt)
            self.assertEqual(Y.shape[1], loader.total_electrodes)
            if split in ('train', 'report'):
                self.assertEqual(X[0, 0, 0, 0], 0)
                self.assertEqual(loader.sequence[-1]['stim_idx'][-1] + 1,
                                (loader.sequence[-1]['resp_idx'][-1] + 1) * 15)

            # Also check it doesn't crash for the last element.
            X, Y = loader[-1]

        self.assertGreater(loader.total_electrodes, 0)

    def test_valid(self):
        loader = vim2.Vim2(data_root, 
                        nt=9,
                        ntau=4,
                        nx=112, 
                        ny=112,
                        split='train')

        for _, Y in loader:
            self.assertEqual(np.isnan(Y).sum(), 0)

if __name__ == '__main__':
    unittest.main()