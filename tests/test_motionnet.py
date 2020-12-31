import sys
sys.path.append('../')
import unittest

from python_dict_wrapper import wrap

from fmri_models import (get_dataset, 
                         preprocess_data, 
                         get_feature_model,
                         get_aggregator)
from modelzoo import motionnet

import torch

class TestMotionNet(unittest.TestCase):
    def test_forward(self):
        """Smoke test"""
        args = wrap({'features': 'MotionNet',
                     'ckpt_root': '../pretrained'})
        model = motionnet.MotionNet(args)
        X = torch.randn(1, 3, 80, 112, 112)
        X_forwarded = model.forward(X)
        print(X_forwarded.shape)
        self.assertEqual(X_forwarded.shape[0], 1)


if __name__ == '__main__':
    unittest.main()