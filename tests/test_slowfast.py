import sys
sys.path.append('../')
import unittest

from python_dict_wrapper import wrap

from fmri_models import (get_dataset, 
                         preprocess_data, 
                         get_feature_model,
                         get_aggregator)
from modelzoo import slowfast_wrapper

import torch

class TestSlowFast(unittest.TestCase):
    def test_each(self):
        """Smoke tests"""
        for features in ['I3D', 'Slow', 'SlowFast']:
            args = wrap({'features': features,
                         'slowfast_path': '../../slowfast/',
                         'ckpt_root': '../pretrained'})
            model = slowfast_wrapper.SlowFast(args)
            X = torch.randn(1, 3, 64, 224, 224, device='cuda')
            X_forwarded = model.forward(X)
            self.assertEqual(X_forwarded.shape[0], 1)

if __name__ == '__main__':
    unittest.main()