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
                         Averager,
                         Downsampler,
                         RP)

import torch

class TestFmriModels(unittest.TestCase):
    def test_averaging(self):
        avg = Averager()
        for sz in [10, 20, 40, 80]:
            X = torch.randn(1, 6, sz, 10, 10)
            X_ = avg(X)
            self.assertEqual(X_.shape[1], 24)
            self.assertEqual(X_.ndim, 2)

    def test_downsampling(self):
        ds = Downsampler(8)

        sz = 10
        X = torch.zeros(1, 6, sz, 55, 55)
        X[:, :, :, :7, :7] = 1
        X_ = ds(X)
        self.assertGreater(X_[0, 0], 0.9)
        self.assertLess(X_[0, 2], 0.2)
        for sz in [10, 20, 40, 80]:
            X = torch.randn(1, 6, sz, 8*7, 8*7)
            X_ = ds(X)
            self.assertEqual(X_.shape[1], 24*8*8)
            self.assertEqual(X_.ndim, 2)

    def test_rp(self):
        rp = RP(100)
        for sz in [10, 20, 40, 80]:
            X = torch.randn(1, 6, sz, 20, 20)
            X_ = rp(X)
            self.assertEqual(X_.shape[1], 100)
            self.assertEqual(X_.ndim, 2)

        # Now check a different size
        X = torch.randn(1, 128, 80, 20, 20)
        X_ = rp(X)
        self.assertEqual(X_.shape[1], 100)

    def test_rp_cuda(self):
        rp = RP(100)

        # Now check a different size
        X = torch.randn(1, 128, 80, 20, 20, device='cuda')
        X_ = rp(X)
        self.assertEqual(X_.shape[1], 100)

    @unittest.skip("Slow")
    def test_caching(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap({'subset': 's1',
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

    @unittest.skip("Slow")
    def test_caching_2d(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap({'subset': 's1',
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