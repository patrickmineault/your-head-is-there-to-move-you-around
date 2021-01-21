import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

def mysoftplus(x, beta):
    print(x.max())
    print(beta.max())
    return torch.where(abs(x) < 20, 
                       1 / beta * torch.log(1 + torch.exp(beta * x)),
                       torch.where( x < 0, 0.0 * x, x))


class GaussianSampler(nn.Module):
    def __init__(self, 
                 nchannels: int, 
                 height_out: int, 
                 width_out: int,
                 sample: bool):
        super(GaussianSampler, self).__init__()
        self.nchannels = nchannels
        self.height_out = height_out
        self.width_out = width_out
        self.sample = sample
        
        self.init_params()

    def init_params(self):
        self.wx = nn.Parameter(-.4 + .8 * torch.rand(self.nchannels))
        self.wy = nn.Parameter(-.4 + .8 * torch.rand(self.nchannels))
        self.wsigmax = nn.Parameter(.5 + .1 * torch.rand(self.nchannels))
        self.wsigmay = nn.Parameter(.5 + .1 * torch.rand(self.nchannels))

        # Create the grid to evaluate the parameters on.

        offset_y = (self.height_out - 1) / self.height_out
        offset_x = (self.width_out - 1) / self.width_out
        # The weird offset is to deal with align_corners = False
        ygrid, xgrid = torch.meshgrid(
            torch.linspace(-offset_y, offset_y, self.height_out),
            torch.linspace(-offset_x, offset_x, self.width_out)
        )

        assert ygrid[1, 0] - ygrid[0, 0] > 0
        assert xgrid[1, 0] - xgrid[0, 0] == 0
        assert xgrid.shape[0] == self.height_out
        assert xgrid.shape[1] == self.width_out
        xgrid = xgrid.view(1, self.height_out, self.width_out)
        ygrid = ygrid.view(1, self.height_out, self.width_out)
        
        self.register_buffer('xgrid', xgrid, False)
        self.register_buffer('ygrid', ygrid, False)

    def forward(self, inputs):
        """Takes a stack of images and a mask and samples them using Gaussians.
        
        Dimension order: batch_dim, nchannels, wy, wx
        Output: batch_dim, nchannels
        """
        X, mask = inputs

        batch_dim, nchannels, ny, nx = X.shape
        assert ny == nx
        
        assert mask.dtype == torch.bool
        assert X.ndim == 4
        assert X.shape[1] == mask.sum()

        if not self.sample:
            # Deterministic sampling
            wsigmax = self.wsigmax[mask].view(-1, 1)
            wsigmay = self.wsigmay[mask].view(-1, 1)

            dx = (((self.xgrid.reshape(1, -1) - 
                    self.wx[mask].view(-1, 1)) ** 2 / 2 / wsigmax ** 2))
            dy = (((self.ygrid.reshape(1, -1) - 
                    self.wy[mask].view(-1, 1)) ** 2 / 2 / wsigmay ** 2))
            
            # We normalize so this matches up with the sampling version exactly.
            ws = torch.exp(-dx -dy) / (
                np.sqrt(2 * np.pi) * 
                 wsigmax * 
                 wsigmay * 
                 self.width_out * 
                 self.height_out / 4
            ) 

            R = torch.einsum('ijk,jk->ij', X.reshape(batch_dim,
                                                     nchannels,
                                                     -1), ws)
        else:
            nnz_mask = mask.sum().item()
            if self.training:
                # Single point sampling
                precision = 1
                x0 = (self.wx[mask] + torch.randn(nnz_mask, device=self.wx.device) * self.wsigmax[mask]).reshape((-1, 1, 1))
                y0 = (self.wy[mask] + torch.randn(nnz_mask, device=self.wx.device) * self.wsigmay[mask]).reshape((-1, 1, 1))
            else:
                # Put things in eval mode - use 1000 points.
                # Single point sampling
                precision = 1000
                x0 = (self.wx[mask].reshape(-1, 1, 1) + 
                      torch.randn(nnz_mask, precision, 1, device=self.wx.device) * self.wsigmax[mask].reshape(-1, 1, 1))
                y0 = (self.wy[mask].reshape(-1, 1, 1) + 
                      torch.randn(nnz_mask, precision, 1, device=self.wx.device) * self.wsigmay[mask].reshape(-1, 1, 1))
            
            x0 = torch.clamp(x0, self.xgrid.min().item(), self.xgrid.max().item())
            y0 = torch.clamp(y0, self.ygrid.min().item(), self.ygrid.max().item())
            
            grid = torch.stack([x0, y0], dim=3)
            assert grid.shape[1] == precision
            assert grid.shape[2] == 1
            assert grid.shape[3] == 2

            R = F.grid_sample(X.permute(1, 0, 2, 3), 
                              grid, 
                              align_corners=False)
            assert R.shape[0] == nchannels
            assert R.shape[1] == batch_dim
            R = R.sum(axis=2).sum(axis=2) / precision
            R = R.T
    
        assert R.shape[0] == X.shape[0]
        return R


class LowRankNet(nn.Module):
    """Creates a low-rank lagged network for prediction.
    
    Takes a network that maps images to stacks of features and maps
    to multi-target networks which are separable combinations along channel, 
    time, and space dimensions.

    Right now, only rank = 1 is implemented.

    Arguments:
        subnet: a nn.Module whose forward op takes batches of 
            if threed is False:
                channel_in X height_in X width_in images (if threed is False) and
                maps them to channels_out X height_out X width_out images.
            if threed is True:
                channel_in X time_in X height_in X width_in and maps them to 
                channels_out X time_in X height_out X width_out
        ntargets: the number of targets total
        channels_out: the number of channels coming out of subnet
        height_out: the number of vertical pixels coming out of subnet
        width_out: the number of horizontal pixels coming out of subnet
        nt: the number of weights in the convolution over time
        sample: whether to sample the Gaussian envelope or 
                not or do it deterministically.
        threed: whether the subnet is 3d.
    """
    def __init__(self, subnet: nn.Module, 
                       ntargets: int, 
                       channels_out: int, 
                       height_out: int, 
                       width_out: int, 
                       nt: int,
                       sample: bool = False,
                       threed: bool = False,
                       output_nl: str = "relu"):
        super(LowRankNet, self).__init__()
        self.subnet = subnet
        self.ntargets = ntargets
        self.channels_out = channels_out
        self.width_out  = width_out
        self.height_out = height_out
        self.nt = nt
        self.sample = sample
        self.threed = threed
        self.output_nl = output_nl

        if self.output_nl == 'softplus':
            beta_ = nn.Parameter(torch.ones(self.ntargets))
            self.register_parameter('beta', beta_)

        self.init_params()

    def init_params(self):
        self.wc = nn.Parameter(
            (.1 * torch.randn(self.channels_out, self.ntargets)/10 + 1) / self.channels_out
        )

        self.wt = nn.Parameter((1 + .1 * torch.randn(self.nt, self.ntargets)) / self.nt)
        self.wb = nn.Parameter(torch.randn(self.ntargets)*.05 + .6)

        self.inner_parameters = [self.wc, self.wt, self.wb]

        self.sampler = GaussianSampler(self.ntargets, 
                                       self.height_out,
                                       self.width_out,
                                       self.sample)

    def forward(self, inputs):
        x, targets = inputs
        input_shape = x.shape
        batch_dim, nchannels, nt, ny, nx = input_shape
        mask = torch.any(targets, dim=0)
        ntargets = mask.sum().item()
        assert nx == ny
        assert x.ndim == 5

        # Start by mapping X through the sub-network.
        if self.threed:
            x = self.subnet.forward(x)
            assert x.shape[-1] == self.width_out
            assert x.shape[-2] == self.height_out
            R = torch.tensordot(x, self.wc[:, mask], dims=([1], [0]))

            # Now we're at batch_dim x nt x Y x X x ntargets
            R = R.reshape(-1, R.shape[2], R.shape[3], R.shape[4])

        else:
            # The subnet doesn't know anything about time, so move time to the front dimension of batches
            # batch_dim x channels x time x Y x X
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_dim * nt, nchannels, ny, nx)
            x = self.subnet.forward(x)

            assert x.shape[0] == batch_dim * nt
            assert x.shape[1] == self.channels_out
            assert x.shape[2] == x.shape[3]

            # Now we're at (batch_dim x nt) x ntargets x Y x X
            R = torch.tensordot(x, self.wc[:, mask], dims=([1], [0]))

        # assert R.shape[0] == batch_dim * nt
        assert R.shape[1] == R.shape[2]
        assert R.shape[3] == ntargets

        # Now at (batch_dim x nt) x Y x X x ntargets
        R = R.permute(0, 3, 1, 2)

        # assert R.shape[0] == batch_dim * nt
        assert R.shape[1] == ntargets
        assert R.shape[2] == R.shape[3]

        R = self.sampler.forward((R, mask))

        assert R.ndim == 2  
        assert R.shape[1] == ntargets

        # Finally, we're at batch_dim x outputs x nt
        # (batch_dim x nt) x ntargets
        R = R.reshape(batch_dim, -1, ntargets).permute(0, 2, 1)

        # Now at batch_dim x ntargets x nt
        R = F.conv1d(R, 
                     self.wt[:, mask].T.view(ntargets, 1, self.wt.shape[0]), 
                     bias=self.wb[mask], 
                     groups=ntargets)

        if self.output_nl:
            if self.output_nl == 'relu':
                R = F.leaky_relu(R, negative_slope=.1)
            elif self.output_nl == 'softplus':
                assert mask.sum() == 1, "Softplus only works one output at a time."
                R = F.softplus(R, beta=self.beta[mask].item())
            else:
                raise NotImplementedError(self.output_nl)

        # Finally, we want to be at batch_dim x nt_out x noutputs
        R = R.permute(0, 2, 1)

        return R
