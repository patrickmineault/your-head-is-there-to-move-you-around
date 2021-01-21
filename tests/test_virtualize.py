import sys
sys.path.append('../')

import numpy as np
import os
from python_dict_wrapper import wrap
import tempfile
import time
import torch
import unittest

from loaders import virtualize

class TestVirtualize(unittest.TestCase):
    def test_transform_list(self):
        t = virtualize.list_transformations()
        self.assertEqual(len(t), 16)

    def test_transforms(self):
        X = torch.randn(3, 5, 18, 18)
        ts = virtualize.list_transformations()
        for i, t in enumerate(ts):
            Y = virtualize.transform(X, t)
            self.assertEqual(tuple(X.shape), tuple(Y.shape))
            if i in (0, 5):
                # Both id and mirror + rot90 keep the top pixel constant.
                self.assertEqual(Y[0, 0, 0, 0], X[0, 0, 0, 0])
            else:
                self.assertNotEqual(Y[0, 0, 0, 0], X[0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()