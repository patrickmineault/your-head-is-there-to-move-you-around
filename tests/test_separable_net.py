import sys
sys.path.append('../')

import torch

import tempfile
import time
import unittest

# Test out separable net
import xception
from separable_net import LowRankNet

class TestSeparableNet(unittest.TestCase):
    def test_init(self):
        basenet = xception.Xception()
        net = LowRankNet(
            basenet,
            6,
            128,
            14,
            14, 
            2,
            sampler_size=9
        )
        self.assertGreater((net.wx > 0).sum(), 0)
        self.assertGreater((net.wy > 0).sum(), 0)
        self.assertGreater((net.xgrid > 0).sum(), 0)
        self.assertGreater((net.ygrid > 0).sum(), 0)

    def test_cuda(self):
        """CUDA test."""
        basenet = xception.Xception()
        X = torch.randn(8, 3, 15, 224, 224, device=torch.device('cuda'))
        outputs = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool, device=torch.device('cuda'))

        net = LowRankNet(
            basenet,
            6,
            128,
            14,
            14, 
            2,
            sampler_size=9
        )
        basenet.to(torch.device('cuda'))
        net.to(torch.device('cuda'))

        net.train()
        Yt = net.forward((X, outputs))
        net.train(False)
        Ye = net.forward((X, outputs))

        self.assertEqual(Yt.shape, Ye.shape)

    def test_sampler(self):
        """Smoke test."""
        basenet = xception.Xception()
        X = torch.randn(8, 3, 15, 224, 224)
        outputs = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool)

        net = LowRankNet(
            basenet,
            6,
            128,
            14,
            14, 
            2,
            sampler_size=9
        )

        net.train()
        Yt = net.forward((X, outputs))
        net.train(False)
        Ye = net.forward((X, outputs))

        self.assertEqual(Yt.shape, Ye.shape)

    def test_forward(self):
        """Smoke test."""
        basenet = xception.Xception()
        X = torch.randn(8, 3, 15, 224, 224)
        outputs = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool)

        net = LowRankNet(
            basenet,
            6,
            128,
            14,
            14, 
            2
        )

        Y = net.forward((X, outputs))
        self.assertEqual(Y.shape, (8, 4, 14))


if __name__ == '__main__':
    unittest.main()


