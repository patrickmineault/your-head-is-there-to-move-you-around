import sys
sys.path.append('../')

from python_dict_wrapper import wrap
import train_net

import tempfile
import time
import torch
import unittest

class TestTrain(unittest.TestCase):
    @unittest.skip("Runs only on Unix")
    def test_train(self):
        """Smoke test."""
        train_net.main('/tmp/data', '/tmp/models')

        self.assertTrue(True)


    def test_v2_mt(self):
        """Check that v2 and mt dataset work together."""
        args = wrap({'dataset': 'v2-mt',
                     'data_root': '/mnt/e/data_derived',
                     'image_size': 112,
                     'subset': 'coreset'})
        dataset, _, _, _ = train_net.get_dataset(args)
        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=4, 
                                             shuffle=True,
                                             pin_memory=True)

        self.assertEqual(dataset.total_electrodes, 60)
        i = 0
        for data in loader:
            X0, _, _, Y0 = data
            self.assertEqual(X0.shape[-1], 112)
            self.assertEqual(X0.shape[0], 4)
            self.assertEqual(Y0.shape[1], 60)


            i += 1
            if i > 10:
                break


if __name__ == '__main__':
    unittest.main()