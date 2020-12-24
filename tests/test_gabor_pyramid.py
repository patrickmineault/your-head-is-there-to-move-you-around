import sys
sys.path.append('../')

from modelzoo import gabor_pyramid
import torch
import unittest

class TestGaborPyramid(unittest.TestCase):
    def test_levels(self):
        net = gabor_pyramid.GaborPyramid(3)
        inputs = torch.ones(4, 3, 127, 127)
        outputs = net.forward(inputs)

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

    def test_3d_pyramid(self):
        net = gabor_pyramid.GaborPyramid3d(nlevels=2, nt=7)

        frames = torch.zeros(3, 3, 7, 9, 9)

        # Moving or static horizontal bars.
        for i in range(7):
            frames[0, :, i, i+1, :] = 1
            frames[1, :, i, 4, :] = 1
            frames[2, :, i, 6-i, :] = 1
        
        output = net.forward(frames)
        self.assertEqual(output.shape[4], 9)
        self.assertEqual(output.shape[3], 9)
        self.assertEqual(output.shape[2], 7)

        output = output[:, :, 3, 4, 4].T

        # Check that horizontal responds more than vertical
        self.assertGreater(output[7, 1].item(), output[1, 1].item())
        self.assertGreater(output[6, 0].item(), output[6, 1].item())
        self.assertGreater(output[7, 1].item(), output[7, 0].item())
        self.assertGreater(output[8, 2].item(), output[8, 1].item())


if __name__ == "__main__":
    unittest.main()