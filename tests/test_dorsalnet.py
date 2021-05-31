import sys

sys.path.append("../")
import unittest

from python_dict_wrapper import wrap

from models import get_dataset, preprocess_data, get_feature_model, get_aggregator
from modelzoo import dorsalnet

import torch


class TestDorsalNet(unittest.TestCase):
    def test_symmetric(self):
        """Smoke test"""
        model = dorsalnet.ShallowNet(nstartfeats=16, symmetric=True)
        X = torch.randn(1, 3, 80, 112, 112)
        X_forwarded = model.forward(X)
        self.assertEqual(X_forwarded.shape[0], 1)

    def test_dorsal(self):
        """Smoke test"""

        model = dorsalnet.DorsalNet()
        X = torch.randn(1, 3, 80, 112, 112)
        X_forwarded = model.forward(X)

        self.assertEqual(X_forwarded.shape[-1], 28)


if __name__ == "__main__":
    unittest.main()