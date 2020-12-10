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
        X = torch.randn(1, 3, 224, 224, 7)
        outputs = torch.tensor([1, 3, 5])

        net = LowRankNet(
            basenet,
            2,
            6,
            128,
            14,
            14, 
            2
        )

        Y = net.forward((X, outputs))
        self.assertEqual(Y.shape, (1, 6, 3))


if __name__ == '__main__':
    unittest.main()


