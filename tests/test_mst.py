import sys

sys.path.append("../")

from loaders import mst
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint

ROOT = "/mnt/e/data_derived/packlab-mst"
ROOT_DORSAL = "/mnt/e/data_derived/packlab-dorsal"


class TestMstLoader(unittest.TestCase):
    def test_train(self):
        loader = mst.MST(
            ROOT,
            nt=1,
            split="train",
        )

        self.assertEqual(
            len({x["cellnum"]: 1 for x in loader.sequence}), loader.total_electrodes
        )

        X, m, W, y = loader[0]
        self.assertEqual(X.shape, (3, 10, 112, 112))
        self.assertEqual(m.shape, W.shape)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], 1)

    def test_pad(self):
        loader = mst.MST(
            ROOT,
            nt=1,
            ntau=12,
            split="train",
        )

        self.assertEqual(
            len({x["cellnum"]: 1 for x in loader.sequence}), loader.total_electrodes
        )

        X, m, W, y = loader[0]
        self.assertEqual(X.shape, (3, 12, 112, 112))
        self.assertEqual(m.shape, W.shape)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], 1)

    def test_singlecell(self):
        loader = mst.MST(
            ROOT,
            nt=1,
            split="traintune",
            single_cell=0,
        )

        X, m, W, y = loader[0]
        self.assertEqual(X.shape, (3, 10, 112, 112))
        self.assertEqual(m.shape, W.shape)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], 1)

    def test_singlecell_dorsal(self):
        loader = mst.MST(
            ROOT_DORSAL,
            nt=1,
            split="traintune",
            single_cell=0,
        )

        X, m, W, y = loader[0]
        self.assertEqual(X.shape, (3, 10, 112, 112))
        self.assertEqual(m.shape, W.shape)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], 1)

    def test_report(self):
        loader = mst.MST(
            ROOT,
            nt=1,
            split="report",
            single_cell=0,
        )

        for el in loader:
            X, m, W, y = el
            self.assertEqual(X.shape, (3, 10, 112, 112))
            self.assertEqual(m.shape, W.shape)
            self.assertEqual(y.ndim, 2)
            self.assertEqual(y.shape[1], 1)


if __name__ == "__main__":
    unittest.main()