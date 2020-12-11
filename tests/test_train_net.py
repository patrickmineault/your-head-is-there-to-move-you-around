import sys
sys.path.append('../')

import train_net
import tempfile
import time
import unittest

class TestTrain(unittest.TestCase):
    def test_train(self):
        """Smoke test."""
        train_net.main('/tmp/data', '/tmp/models')

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()