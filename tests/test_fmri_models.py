import sys
sys.path.append('../')
import unittest

import tempfile
import time
import os

from python_dict_wrapper import wrap

from fmri_models import (get_dataset, 
                         preprocess_data, 
                         get_feature_model,
                         get_aggregator,
                         Averager)

import torch

class TestFmriModels(unittest.TestCase):
    def test_averaging(self):
        avg = Averager()
        for sz in [10, 20, 40, 80]:
            X = torch.randn(1, 6, sz, 10, 10)
            X_ = avg(X)
            self.assertEqual(X_.shape[1], 24)
            self.assertEqual(X_.ndim, 2)

    def test_caching(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap({'subject': 's1',
                        'batch_size': 10,
                        'features': 'gaborpyramid3d',
                        'aggregator': 'average',
                        'dataset': 'vim2',
                        'data_root': '../data',
                        'layer': 0,
                        'width': 112,
                        'cache_root': tmpdirname})


            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)
            reportset = get_dataset(args, 'report')

            reportloader = torch.utils.data.DataLoader(reportset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=False,
                                                    pin_memory=True
                                                    )
            # Second call should be a cache miss.
            t = time.time()
            feature_model.to(device='cuda')
            X_report, Y_report = preprocess_data(reportloader, 
                                feature_model, 
                                aggregator,
                                activations, 
                                metadata,
                                args)

            self.assertEqual(X_report.shape[0], 540)
            self.assertEqual(X_report.ndim, 2)
            self.assertEqual(Y_report.shape[0], 540)

            dt = time.time() - t

            # Second call should be a cache hit.
            t = time.time()
            feature_model.to(device='cuda')
            X_report, Y_report = preprocess_data(reportloader, 
                                feature_model, 
                                aggregator,
                                activations, 
                                metadata,
                                args)

            dt2 = time.time() - t

            # Should be at least five times faster
            self.assertGreater(dt, dt2 * 5)

    def test_caching_2d(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap({'subject': 's1',
                        'batch_size': 10,
                        'features': 'resnet18',
                        'aggregator': 'average',
                        'dataset': 'vim2',
                        'data_root': '../data',
                        'layer': 0,
                        'width': 112,
                        'cache_root': tmpdirname})


            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)
            reportset = get_dataset(args, 'report')

            reportloader = torch.utils.data.DataLoader(reportset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=False,
                                                    pin_memory=True
                                                    )
            # First call should be a cache miss.
            t = time.time()
            feature_model.to(device='cuda')
            X_report, Y_report = preprocess_data(reportloader, 
                                feature_model, 
                                aggregator,
                                activations, 
                                metadata,
                                args)

            self.assertEqual(X_report.shape[0], 540)
            self.assertEqual(X_report.shape[1], 256)
            self.assertEqual(X_report.ndim, 2)
            self.assertEqual(Y_report.shape[0], 540)

            dt = time.time() - t

            # Second call should be a cache hit.
            t = time.time()
            feature_model.to(device='cuda')
            X_report, Y_report = preprocess_data(reportloader, 
                                feature_model, 
                                aggregator,
                                activations, 
                                metadata,
                                args)

            dt2 = time.time() - t

            # Should be at least five times faster
            self.assertGreater(dt, dt2 * 5)

if __name__ == '__main__':
    unittest.main()