import sys
sys.path.append('../')
import unittest

from python_dict_wrapper import wrap

from fmri_models import (get_dataset, 
                         preprocess_data, 
                         get_feature_model,
                         get_aggregator)
from modelzoo import monkeynet

import torch

class TestMonkeyNet(unittest.TestCase):
    def test_symmetric(self):
        """Smoke test"""
        model = monkeynet.ShallowNet(nstartfeats=16, 
                                     symmetric=True)
        X = torch.randn(1, 3, 80, 112, 112)
        X_forwarded = model.forward(X)
        self.assertEqual(X_forwarded.shape[0], 1)

    def test_v1(self):
        """Smoke test"""

        model = monkeynet.V1Net()
        X = torch.randn(1, 3, 80, 112, 112)
        X_forwarded = model.forward(X)
        self.assertEqual(X_forwarded.shape[0], 1)


if __name__ == '__main__':
    unittest.main()