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


