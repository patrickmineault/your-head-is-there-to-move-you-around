import sys
sys.path.append('../')
import unittest

from python_dict_wrapper import wrap

from fmri_models import (get_dataset, 
                         preprocess_data, 
                         get_feature_model,
                         get_aggregator)
from modelzoo import shiftnet

import torch

class TestShiftNet(unittest.TestCase):
    def test_forward(self):
        """Smoke test"""
        args = wrap({'features': 'ShiftNet',
                     'ckpt_root': '../pretrained'})
        model = shiftnet.ShiftNet(args)
        X = torch.randn(1, 3, 80, 112, 112)
        X_forwarded = model.forward(X)
        self.assertEqual(X_forwarded.shape[0], 1)


if __name__ == '__main__':
    unittest.main()