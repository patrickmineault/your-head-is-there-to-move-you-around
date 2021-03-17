import sys

sys.path.append("../")

from convex_models import compute_boosting_estimate, compute_ridge_estimate
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint


class TestConvexModels(unittest.TestCase):
    def test_ridge(self):
        X = torch.randn((1000, 10), dtype=torch.float32, device="cuda")
        w = torch.exp(-torch.arange(10, dtype=torch.float32, device="cuda"))
        y = X @ w + torch.randn(1000, dtype=torch.float32, device="cuda")
        y = y.reshape(y.shape[0], 1)

        X_report = torch.randn((1000, 10), dtype=torch.float32, device="cuda")
        y_report = X_report @ w
        y_report = y_report.reshape(y_report.shape[0], 1)

        splits = np.arange(1000) % 5

        results, weights = compute_ridge_estimate(X, y, X_report, y_report, splits)
        self.assertGreater(
            np.corrcoef(weights["W"].squeeze(), w.cpu().detach().numpy())[0, 1], 0.95
        )

        self.assertEqual((weights["W"] != 0).sum(), 10)
        self.assertGreater(results["corrs_report_mean"], 0.95)

    def test_ridge_big(self):
        X = torch.randn((5000, 5500), dtype=torch.float32, device="cuda")
        w = torch.exp(-torch.arange(5500, dtype=torch.float32, device="cuda"))
        y = X @ w + torch.randn(5000, dtype=torch.float32, device="cuda")
        y = y.reshape(y.shape[0], 1)

        X_report = torch.randn((5000, 5500), dtype=torch.float32, device="cuda")
        y_report = X_report @ w
        y_report = y_report.reshape(y_report.shape[0], 1)

        splits = np.arange(5000) % 5

        results, weights = compute_ridge_estimate(X, y, X_report, y_report, splits)
        C = np.corrcoef(weights["W"].squeeze(), w.cpu().detach().numpy())
        self.assertEqual(C.shape[1], 2)
        self.assertLess(C[0, 1], 0.9)

        self.assertEqual((weights["W"] != 0).sum(), 5500)
        self.assertLess(results["corrs_report_mean"], 0.9)

    def test_boosting(self):
        X = torch.randn((1000, 10), dtype=torch.float32, device="cuda")
        w = torch.exp(-torch.arange(10, dtype=torch.float32, device="cuda"))
        y = X @ w + torch.randn(1000, dtype=torch.float32, device="cuda")
        y = y.reshape(y.shape[0], 1)

        X_report = torch.randn((1000, 10), dtype=torch.float32, device="cuda")
        y_report = X_report @ w
        y_report = y_report.reshape(y_report.shape[0], 1)

        splits = np.arange(1000) % 5

        results, weights = compute_boosting_estimate(X, y, X_report, y_report, splits)
        self.assertGreater(
            np.corrcoef(weights["W"].squeeze(), w.cpu().detach().numpy())[0, 1], 0.95
        )

        self.assertLess((weights["W"] != 0).sum(), 10)
        self.assertGreater(results["corrs_report_mean"], 0.95)

    def test_boosting_big(self):
        X = torch.randn((5000, 20000), dtype=torch.float32, device="cuda")
        w = torch.exp(-torch.arange(20000, dtype=torch.float32, device="cuda"))
        y = X @ w + torch.randn(5000, dtype=torch.float32, device="cuda")
        y = y.reshape(y.shape[0], 1)

        X_report = torch.randn((5000, 20000), dtype=torch.float32, device="cuda")
        y_report = X_report @ w
        y_report = y_report.reshape(y_report.shape[0], 1)

        splits = np.arange(5000) % 5

        results, weights = compute_boosting_estimate(X, y, X_report, y_report, splits)
        self.assertGreater(
            np.corrcoef(weights["W"].squeeze(), w.cpu().detach().numpy())[0, 1], 0.95
        )

        self.assertLess((weights["W"] != 0).sum(), 100)
        self.assertGreater(results["corrs_report_mean"], 0.95)


if __name__ == "__main__":
    unittest.main()