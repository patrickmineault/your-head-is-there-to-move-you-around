import unittest
import sys
sys.path.append('../')
import transforms

import numpy as np
import torch

class TestTransforms(unittest.TestCase):
    def test_color_jitter(self):
        X = torch.tensor(np.random.rand(100, 3, 10, 10, 10) * 255)
        t = transforms.ThreedColorJitter(.2, .2, .2, .2)
        Y = t(X)

        self.assertEqual(tuple(X.shape), tuple(Y.shape))
        self.assertEqual(np.sum(np.isnan(X.cpu().detach().numpy())), 0)

    def test_identity(self):
        X = torch.tensor(np.random.rand(100, 3, 10, 10, 10) * 255)
        t = transforms.ThreedColorJitter(0, 0, 0, 0)
        Y = t(X)

        self.assertEqual(tuple(X.shape), tuple(Y.shape))
        np.testing.assert_allclose(X.detach().numpy(), Y.detach().numpy())


    def test_gaussian_blur(self):
        X = torch.tensor(np.random.rand(100, 3, 10, 10, 10) * 255)
        t = transforms.ThreedGaussianBlur(5)
        Y = t(X)

        self.assertEqual(tuple(X.shape), tuple(Y.shape))
        self.assertEqual(np.sum(np.isnan(X.cpu().detach().numpy())), 0)


    def test_gaussian_identity(self):
        X = torch.tensor(np.random.rand(100, 3, 10, 10, 10) * 255)
        t = transforms.ThreedGaussianBlur(5, sigma=(.1, .1))
        Y = t(X)

        self.assertEqual(tuple(X.shape), tuple(Y.shape))
        np.testing.assert_allclose(X.detach().numpy(), Y.detach().numpy())

if __name__ == '__main__':
    unittest.main()