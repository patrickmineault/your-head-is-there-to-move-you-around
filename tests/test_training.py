import sys
sys.path.append('../')

import numpy as np
import os
from python_dict_wrapper import wrap
import tempfile
import time
import torch
import unittest

from training import (compute_corr, 
                      compute_rdm, 
                      compute_rdm_distance,
                      bootstrap_rdm_distance,
                      bootstrap_ycols)

class TestTraining(unittest.TestCase):
    def test_corr(self):
        X = torch.tensor(np.random.randn(100, 10))
        Y = torch.tensor(np.random.randn(100, 10))

        c1 = compute_corr(X, Y)

        X = torch.cat((X, np.nan * torch.zeros(1, 10)), axis=0)
        Y = torch.cat((Y, np.nan * torch.zeros(1, 10)), axis=0)

        c2 = compute_corr(X, Y)
        np.testing.assert_allclose(c1.cpu().detach().numpy(), c2.cpu().detach().numpy())

    def test_rdm(self):
        X = torch.tensor(np.random.randn(100, 10))
        rdm = compute_rdm(X)
        self.assertEqual(rdm.shape[0], X.shape[0])

    def test_rdm_distance(self):
        X = torch.tensor(np.random.randn(100, 10))
        rdm = compute_rdm(X)
        d = compute_rdm_distance(rdm, rdm)
        self.assertLess(abs(d.item()), 1e-15)

        Y = torch.tensor(np.random.randn(100, 10))
        rdm_y = compute_rdm(Y)
        d = compute_rdm_distance(rdm, rdm_y)
        self.assertGreater(d.item(), 0.9)
        self.assertLess(d.item(), 1.03) # Can be a tiny bit negative

    def test_rdm_distances(self):
        X = torch.tensor(np.random.randn(100, 10))
        rdm = compute_rdm(X)
        a = compute_rdm_distance(rdm, rdm + 1, 'pearson')
        b = compute_rdm_distance(rdm, rdm, 'pearson')
        c = compute_rdm_distance(rdm, rdm ** 3, 'pearson')
        self.assertLess(abs(a.item()), 1e-15)
        self.assertLess(abs(b.item()), 1e-15)
        self.assertGreater(abs(c.item()), 1e-3)

        a = compute_rdm_distance(rdm, rdm + 1, 'cosine')
        b = compute_rdm_distance(rdm, rdm, 'cosine')
        self.assertGreater(abs(a.item()), 1e-15)
        self.assertLess(abs(b.item()), 1e-15)

        c = compute_rdm_distance(rdm, rdm ** 3, 'rank')
        self.assertLess(abs(b.item()), 1e-3)

        a = compute_rdm_distance(rdm, rdm, 'r2')
        b = compute_rdm_distance(rdm, rdm*2, 'r2')
        self.assertLess(abs(a.item()), 1e-15)
        self.assertGreater(abs(b.item()), .1)

    def test_bootstrap_rdm_distance(self):
        X = torch.tensor(np.random.randn(100, 10))
        Y = torch.tensor(np.random.randn(100, 10))
        m, s = bootstrap_rdm_distance(X, Y, 'pearson')
        self.assertLess(abs(m - 1), s * 3)

        m2, s2 = bootstrap_rdm_distance(X, Y, 'rank')
        self.assertGreater(abs(m2 - m), 1e-4)

        def f(X, Y):
            return compute_rdm_distance(
                compute_rdm(X), compute_rdm(Y), 'rank'
            )

        m3, s3 = bootstrap_ycols(f, X, Y)

        self.assertLess(abs(m3 - m2), 3 * s)
        self.assertLess(abs(s2 - s3)/np.sqrt(s2 ** 2 + s3**2) / 2, .1)

if __name__ == '__main__':
    unittest.main()