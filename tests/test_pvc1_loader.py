import sys
sys.path.append('../')

import pvc1_loader
import tempfile
import time
import unittest

class TestPvc1Loader(unittest.TestCase):
    def test_download_cached(self):
        tmp_dir = tempfile.mkdtemp()

        # The first time around, it should be very slow.
        self.assertTrue(pvc1_loader.download(tmp_dir))

        # The second time around, it should be much faster.
        t0 = time.time()
        self.assertTrue(pvc1_loader.download(tmp_dir))
        self.assertLess(time.time() - t0, 2)


if __name__ == '__main__':
    unittest.main()