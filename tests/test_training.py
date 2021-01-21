import sys
sys.path.append('../')

import numpy as np
import os
from python_dict_wrapper import wrap
import tempfile
import time
import torch
import unittest

from training import compute_corr

class TestFmriModels(unittest.TestCase):
    def test_corr(self):
        X = torch.tensor(np.random.randn(100, 10))
        Y = torch.tensor(np.random.randn(100, 10))

        c1 = compute_corr(X, Y)

        X = torch.cat((X, np.nan * torch.zeros(1, 10)), axis=0)
        Y = torch.cat((Y, np.nan * torch.zeros(1, 10)), axis=0)

        c2 = compute_corr(X, Y)
        np.testing.assert_allclose(c1.cpu().detach().numpy(), c2.cpu().detach().numpy())

if __name__ == '__main__':
    unittest.main()