import sys

sys.path.append("../")

from loaders import stc1, st
import numpy as np
import tempfile
import time
import unittest
import torch
from pprint import pprint


class TestAirsimLoader(unittest.TestCase):
    def test_loading(self):
        loader = st.St("/mnt/e/data_derived/packlab-st", split="report", subset="MSTd")

        X, Y = loader[0]
        self.assertEqual(tuple(X.shape), (3, 12, 112, 112))
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(Y.size, 55)

        loader = st.St("/mnt/e/data_derived/packlab-st", split="report", subset="V3A")

        X, Y = loader[0]
        self.assertEqual(tuple(X.shape), (3, 12, 112, 112))
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(Y.size, 181)


if __name__ == "__main__":
    unittest.main()