import sys
sys.path.append('../')

from loaders import vim2
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint

data_root = '../data/crcns-vim2/derived'

class TestVim2Loader(unittest.TestCase):
    def test_sequence(self):
        loader = vim2.Vim2(data_root, 
                           split='train')
        self.assertGreater(len(loader.sequence), 0)
        self.assertTrue(np.all([np.all(x['stim_idx'] >= -1) for x in loader.sequence]))

        loader = vim2.Vim2(data_root, 
                           split='tune')
        self.assertGreater(len(loader.sequence), 0)
        self.assertTrue(np.all([np.all(x['stim_idx'] >= -1) for x in loader.sequence]))

        loader = vim2.Vim2(data_root, 
                           split='report')
        self.assertGreater(len(loader.sequence), 0)
        self.assertTrue(np.all([np.all(x['stim_idx'] >= -1) for x in loader.sequence]))

    def test_rois(self):
        loader = vim2.Vim2(data_root, 
                           split='train')
        rois = loader._get_rois()
        self.assertGreater(len(rois), 0)
        self.assertIn('v4rh', rois.keys())

    def test_train(self):
        for split in ('train', 'tune', 'report'):
            loader = vim2.Vim2(data_root, 
                            nt=1,
                            ntau=16,
                            split=split)
            X, Y = loader[0]

            self.assertFalse(np.isnan(Y).any())
            self.assertEqual(X.shape[-1], 128)
            self.assertEqual(X.shape[-2], 128)
            self.assertEqual(X.shape[0], 3)
            self.assertEqual(X.shape[1], (loader.nt - 1) * 15 + loader.ntau)

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
                        split='train')

        for _, Y in loader:
            self.assertEqual(np.isnan(Y).sum(), 0)

    def test_subject(self):
        for s in ['s1', 's2', 's3']:
            loader = vim2.Vim2(data_root, 
                            nt=9,
                            ntau=4,
                            split='report',
                            subject=s)
            X, Y = loader[-1]
            self.assertGreater(Y.size, 0)
            self.assertFalse(np.isnan(Y).any(), s)

            loader = vim2.Vim2(data_root, 
                            nt=9,
                            ntau=4,
                            split='traintune',
                            subject=s)
            X, Y = loader[-1]
            self.assertGreater(Y.size, 0)
            self.assertFalse(np.isnan(Y).any(), s)


if __name__ == '__main__':
    unittest.main()