import sys
sys.path.append('../')

import gabor_pyramid
import torch
import unittest

class TestGaborPyramid(unittest.TestCase):
    def test_levels(self):
        net = gabor_pyramid.GaborPyramid(3)
        inputs = torch.ones(4, 3, 127, 127)
        outputs = net.forward(inputs)

        print(outputs.shape)
        self.assertEqual(outputs.shape[1], 12)

        net = gabor_pyramid.GaborPyramid(1)
        outputs = net.forward(inputs)

        self.assertEqual(len(outputs), 4)

    def test_mapping(self):
        net = gabor_pyramid.GaborPyramid(4)
        inputs = torch.ones(4, 3, 127, 127)
        outputs = net.forward(inputs)

        self.assertEqual(outputs.shape, (4, 16, 127, 127))

    def test_shapes(self):
        net = gabor_pyramid.GaborPyramid(4)
        for i in range(120, 128):
            inputs = torch.ones(4, 3, i, i)
            outputs = net.forward(inputs)

        self.assertTrue(True)


    def test_mapping_gpu(self):
        device = torch.device('cuda')
        net = gabor_pyramid.GaborPyramid(4)
        net.to(device)
        inputs = torch.ones(4, 3, 127, 127)
        outputs = net.forward(inputs.to(device))

        self.assertEqual(len(outputs), 4)


if __name__ == "__main__":
    unittest.main()