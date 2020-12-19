import sys
sys.path.append('../')

import torch

import tempfile
import time
import unittest

# Test out separable net
import xception
from separable_net import LowRankNet, GaussianSampler

class TestSeparableNet(unittest.TestCase):
    
    def test_init(self):
        """Test initialization."""
        nx = 17
        rg = (nx - 1) / nx
        net = GaussianSampler(9, nx, nx, 9)
        self.assertGreater((net.wx > 0).sum(), 0)
        self.assertGreater((net.wy > 0).sum(), 0)
        self.assertGreater((net.xgrid > 0).sum(), 0)
        self.assertGreater((net.ygrid > 0).sum(), 0)

        self.assertAlmostEqual(net.xgrid.min().item(), -rg)
        self.assertAlmostEqual(net.xgrid.max().item(), rg)
        self.assertAlmostEqual(net.ygrid.min().item(), -rg)
        self.assertAlmostEqual(net.ygrid.max().item(), rg)

    def test_gaussian_nonsampler(self):
        nx = 17
        net = GaussianSampler(9, nx, nx, sample=False)
        net.wx.data = torch.tensor([-(nx - 1)/nx / 2, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.float32)
        net.wy.data = torch.tensor([-(nx - 1)/nx / 2, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
        net.wsigmax.data = .05 * torch.ones(9, dtype=torch.float32)
        net.wsigmay.data = .05 * torch.ones(9, dtype=torch.float32)

        # Create a test pattern that's one dot in the top left, 
        # one in the middle.
        X = torch.zeros(1, 9, nx, nx, dtype=torch.float32)
        X[:, :, 4, 4] = 1
        X[:, :, 8, 8] = 1

        mask = torch.ones(9, dtype=torch.bool)
        forwarded = net.forward((X, mask)).cpu().detach().numpy().squeeze()

        self.assertAlmostEqual(forwarded[0], forwarded[4])
        self.assertGreater(forwarded[0], .1)
        self.assertLess(forwarded[1], .001)
        self.assertLess(forwarded[2], .001)
        self.assertLess(forwarded[3], .001)
        self.assertLess(forwarded[5], .001)
        self.assertLess(forwarded[6], .001)
        self.assertLess(forwarded[7], .001)
        self.assertLess(forwarded[8], .001)

    
    def test_gaussian_sampler(self):
        nx = 17
        net = GaussianSampler(9, nx, nx, sample=True)
        net.wx.data = torch.tensor([-(nx - 1)/nx / 2, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.float32)
        net.wy.data = torch.tensor([-(nx - 1)/nx / 2, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
        net.wsigmax.data = .01 * torch.ones(9, dtype=torch.float32)
        net.wsigmay.data = .01 * torch.ones(9, dtype=torch.float32)

        # Create a test pattern that's one dot in the top left, 
        # one in the middle.
        X = torch.zeros(1, 9, nx, nx, dtype=torch.float32)
        X[:, :, 4, 4] = 1
        X[:, :, 8, 8] = 1

        mask = torch.ones(9, dtype=torch.bool)
        net.eval()
        forwarded = net.forward((X, mask)).cpu().detach().numpy().squeeze()

        self.assertAlmostEqual(forwarded[0], forwarded[4], delta=.05*forwarded[0])
        self.assertGreater(forwarded[0], .1)
        self.assertLess(forwarded[1], .001)
        self.assertLess(forwarded[2], .001)
        self.assertLess(forwarded[3], .001)
        self.assertLess(forwarded[5], .001)
        self.assertLess(forwarded[6], .001)
        self.assertLess(forwarded[7], .001)
        self.assertLess(forwarded[8], .001)


    def test_gaussian_sampler_train(self):
        nx = 17
        net = GaussianSampler(9, nx, nx, sample=True)
        net.wx.data = torch.tensor([-(nx - 1)/nx, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.float32)
        net.wy.data = torch.tensor([-(nx - 1)/nx, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
        net.wsigmax.data = .01 * torch.ones(9, dtype=torch.float32)
        net.wsigmay.data = .01 * torch.ones(9, dtype=torch.float32)

        # Create a test pattern that's one dot in the top left, 
        # one in the middle.
        X = torch.zeros(1, 9, nx, nx, dtype=torch.float32)
        X[:, :, 0, 0] = 1
        X[:, :, 8, 8] = 1

        mask = torch.ones(9, dtype=torch.bool)
        net.train()
        
        m = 0
        for i in range(100):
            forwarded = net.forward((X, mask)).cpu().detach().numpy().squeeze()
            m += forwarded
        
        forwarded = m / 100
        
        self.assertAlmostEqual(forwarded[0], forwarded[4], delta=.3*forwarded[0])
        self.assertGreater(forwarded[0], .1)
        self.assertLess(forwarded[1], .001)
        self.assertLess(forwarded[2], .001)
        self.assertLess(forwarded[3], .001)
        self.assertLess(forwarded[5], .001)
        self.assertLess(forwarded[6], .001)
        self.assertLess(forwarded[7], .001)
        self.assertLess(forwarded[8], .001)

    def test_clamped_edges(self):
        """Check that edges are clamped as expected.
        
        In deterministic mode, the gaussian does not extend beyond borders.
        In sampling mode, samples close to the border are clamped there.
        """
        nx = 17
        X = torch.zeros(1, 1, nx, nx, dtype=torch.float32)
        X[:, :, 0, 0] = 1

        mask = torch.ones(1, dtype=torch.bool)
        for sample in [False, True]:

            net = GaussianSampler(9, nx, nx, sample=sample)
            net.wx.data = torch.tensor([-2], dtype=torch.float32)
            net.wy.data = torch.tensor([-2], dtype=torch.float32)
            net.wsigmax.data = .01 * torch.ones(1, dtype=torch.float32)
            net.wsigmay.data = .01 * torch.ones(1, dtype=torch.float32)

            net.eval()
            
            forwarded = net.forward((X, mask)).cpu().detach().numpy().squeeze()
            
            if sample:
                self.assertAlmostEqual(forwarded, 1.0)
            else:
                self.assertAlmostEqual(forwarded, 0.0)

    def test_gradient_propagates(self):
        nx = 17
        net = GaussianSampler(9, nx, nx, sample=True)
        net.wx.data = torch.tensor([-(nx - 1)/nx, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.float32)
        net.wy.data = torch.tensor([-(nx - 1)/nx, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
        net.wsigmax.data = .05 * torch.ones(9, dtype=torch.float32)
        net.wsigmay.data = .05 * torch.ones(9, dtype=torch.float32)

        # Create a test pattern that's one dot in the top left, 
        # one in the middle.
        X = torch.zeros(1, 9, nx, nx, dtype=torch.float32)
        X[:, :, 0, 0] = 1
        X[:, :, 8, 8] = 1

        mask = torch.ones(9, dtype=torch.bool)

        # zero the parameter gradients
        net.eval()
        outputs = net((X, mask))
        outputs.sum().backward()
        self.assertTrue((abs(net.wx.grad.cpu().detach().numpy()) > 0).any())
        self.assertTrue((abs(net.wy.grad.cpu().detach().numpy()) > 0).any())
        self.assertTrue((abs(net.wsigmax.grad.cpu().detach().numpy()) > 0).any())
        self.assertTrue((abs(net.wsigmay.grad.cpu().detach().numpy()) > 0).any())
        
        
    def test_cuda(self):
        """CUDA test."""
        basenet = xception.Xception(nblocks=0)
        X = torch.randn(8, 3, 15, 64, 64, device=torch.device('cuda'))
        outputs = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool, device=torch.device('cuda'))

        net = LowRankNet(
            basenet,
            6,
            16,
            31,
            31, 
            2,
            sample=False
        )
        basenet.to(torch.device('cuda'))
        net.to(torch.device('cuda'))

        net.train()
        Yt = net.forward((X, outputs))
        net.train(False)
        Ye = net.forward((X, outputs))

        self.assertEqual(Yt.shape, Ye.shape)


    def test_sampler(self):
        """Smoke test."""
        basenet = xception.Xception(nblocks=0)
        X = torch.randn(8, 3, 15, 64, 64)
        outputs = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool)

        net = LowRankNet(
            basenet,
            6,
            16,
            31,
            31, 
            2,
            sample=False
        )

        net.train()
        Yt = net.forward((X, outputs))
        net.train(False)
        Ye = net.forward((X, outputs))

        self.assertEqual(Yt.shape, Ye.shape)

    def test_forward(self):
        """Smoke test."""
        basenet = xception.Xception(nblocks=0)
        X = torch.randn(8, 3, 15, 64, 64)
        outputs = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool)

        net = LowRankNet(
            basenet,
            6,
            16,
            31,
            31, 
            2
        )

        Y = net.forward((X, outputs))
        self.assertEqual(Y.shape, (8, 4, 14))


if __name__ == '__main__':
    unittest.main()


