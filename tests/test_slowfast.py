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
    @unittest.skip("Slow")
    def test_each(self):
        """Smoke tests"""
        for features in ['I3D', 'Slow', 'SlowFast']:
            args = wrap({'features': features,
                         'slowfast_root': '../../slowfast/',
                         'ckpt_root': '../pretrained'})
            model = slowfast_wrapper.SlowFast(args)
            X = torch.randn(1, 3, 80, 224, 224, device='cuda')
            X_forwarded = model.forward(X)
            self.assertEqual(X_forwarded.shape[0], 1)

    def test_end_to_end(self):
        """Smoke tests"""
        for features in ['SlowFast', 'Slow', 'I3D']:
            args = wrap({'features': features,
                         'slowfast_root': '../../slowfast/',
                         'ckpt_root': '../pretrained',
                         'aggregator': 'average',
                         'data_root': '../data',
                         'dataset': 'vim2',
                         'batch_size': 1,
                         'layer': 0,
                         'subject': 's1',
                         'width': 224,
                         'ntau': 80,
                         'nt': 1})

            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)

            X = torch.randn(1, 3, 80, 224, 224, device='cuda')
            _ = feature_model(X)
            
            for k, v in activations.items():
                aggregator(v)

    @unittest.skip("Broken")
    def test_X3DM(self):
        """Smoke tests"""
        for features in ['X3DM']:
            args = wrap({'features': features,
                         'slowfast_root': '../../slowfast/',
                         'ckpt_root': '../pretrained'})
            model = slowfast_wrapper.SlowFast(args)
            X = torch.randn(1, 3, 80, 224, 224, device='cuda')
            X_forwarded = model.forward(X)
            self.assertEqual(X_forwarded.shape[0], 1)


if __name__ == '__main__':
    unittest.main()