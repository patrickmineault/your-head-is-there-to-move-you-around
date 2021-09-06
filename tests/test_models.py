import sys

sys.path.append("../")

import numpy as np
import os
from python_dict_wrapper import wrap
import tempfile
import time
import torch
import unittest


from models import (
    get_dataset,
    preprocess_data,
    get_feature_model,
    get_aggregator,
    Averager,
    Downsampler,
)


class TestModels(unittest.TestCase):
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
            X = torch.randn(1, 6, sz, 8 * 7, 8 * 7)
            X_ = ds(X)
            self.assertEqual(X_.shape[1], 24 * 8 * 8)
            self.assertEqual(X_.ndim, 2)

    def test_downsampling_t(self):
        ds = Downsampler(None, only_t=True)

        sz = 10
        X = torch.zeros(1, 6, sz, 55, 55)
        X[:, :, :, :7, :7] = 1
        X_ = ds(X)
        self.assertGreater(X_[0, 0], 0.9)
        self.assertEqual(X_[0, 2], 1.0)
        for sz in [10, 20, 40, 80]:
            X = torch.randn(1, 6, sz, 8 * 7, 8 * 7)
            X_ = ds(X)
            self.assertEqual(X_.shape[1], 24 * 8 * 8 * 7 * 7)
            self.assertEqual(X_.ndim, 2)

    @unittest.skip("Slow")
    def test_caching(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap(
                {
                    "subset": "s1",
                    "batch_size": 10,
                    "features": "gaborpyramid3d",
                    "aggregator": "average",
                    "dataset": "vim2",
                    "data_root": "../data",
                    "layer": 0,
                    "width": 112,
                    "cache_root": tmpdirname,
                }
            )

            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)
            reportset = get_dataset(args, "report")

            reportloader = torch.utils.data.DataLoader(
                reportset, batch_size=args.batch_size, shuffle=False, pin_memory=True
            )
            # Second call should be a cache miss.
            t = time.time()
            feature_model.to(device="cuda")
            X_report, Y_report = preprocess_data(
                reportloader, feature_model, aggregator, activations, metadata, args
            )

            self.assertEqual(X_report.shape[0], 540)
            self.assertEqual(X_report.ndim, 2)
            self.assertEqual(Y_report.shape[0], 540)

            dt = time.time() - t

            # Second call should be a cache hit.
            t = time.time()
            feature_model.to(device="cuda")
            X_report, Y_report = preprocess_data(
                reportloader, feature_model, aggregator, activations, metadata, args
            )

            dt2 = time.time() - t

            # Should be at least five times faster
            self.assertGreater(dt, dt2 * 5)

    @unittest.skip("Slow")
    def test_caching_2d(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap(
                {
                    "subset": "s1",
                    "batch_size": 10,
                    "features": "resnet18",
                    "aggregator": "average",
                    "dataset": "vim2",
                    "data_root": "../data",
                    "layer": 0,
                    "width": 112,
                    "cache_root": tmpdirname,
                }
            )

            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)
            reportset = get_dataset(args, "report")

            reportloader = torch.utils.data.DataLoader(
                reportset, batch_size=args.batch_size, shuffle=False, pin_memory=True
            )
            # First call should be a cache miss.
            t = time.time()
            feature_model.to(device="cuda")
            X_report, Y_report = preprocess_data(
                reportloader, feature_model, aggregator, activations, metadata, args
            )

            self.assertEqual(X_report.shape[0], 540)
            self.assertEqual(X_report.shape[1], 256)
            self.assertEqual(X_report.ndim, 2)
            self.assertEqual(Y_report.shape[0], 540)

            dt = time.time() - t

            # Second call should be a cache hit.
            t = time.time()
            feature_model.to(device="cuda")
            X_report, Y_report = preprocess_data(
                reportloader, feature_model, aggregator, activations, metadata, args
            )

            dt2 = time.time() - t

            # Should be at least five times faster
            self.assertGreater(dt, dt2 * 5)

    def test_mst_data_report(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap(
                {
                    "subset": 0,
                    "batch_size": 1,
                    "features": "airsim_04",
                    "aggregator": "average",
                    "dataset": "mst",
                    "data_root": "/mnt/e/data_derived/",
                    "ckpt_root": "../pretrained",
                    "layer": 0,
                    "layer_name": "layer00",
                    "width": 112,
                    "cache_root": tmpdirname,
                }
            )

            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)
            reportset = get_dataset(args, "report")

            reportloader = torch.utils.data.DataLoader(
                reportset, batch_size=args.batch_size, shuffle=False, pin_memory=True
            )
            feature_model.to(device="cuda")

            # First call should be a cache miss.
            X_report, Y_report = preprocess_data(
                reportloader, feature_model, aggregator, activations, metadata, args
            )
            self.assertEqual(X_report.shape[0], 216)

    def test_mst_data_traintune(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            args = wrap(
                {
                    "subset": 0,
                    "batch_size": 1,
                    "features": "airsim_04",
                    "aggregator": "average",
                    "dataset": "mst",
                    "data_root": "/mnt/e/data_derived/",
                    "ckpt_root": "../pretrained",
                    "layer": 0,
                    "layer_name": "layer00",
                    "width": 112,
                    "cache_root": tmpdirname,
                }
            )

            feature_model, activations, metadata = get_feature_model(args)
            aggregator = get_aggregator(metadata, args)
            reportset = get_dataset(args, "traintune")

            reportloader = torch.utils.data.DataLoader(
                reportset, batch_size=args.batch_size, shuffle=False, pin_memory=True
            )
            feature_model.to(device="cuda")

            # First call should be a cache miss.
            X_report, Y_report = preprocess_data(
                reportloader, feature_model, aggregator, activations, metadata, args
            )


if __name__ == "__main__":
    unittest.main()