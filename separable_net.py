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
                 sampler_size: int):
        super(GaussianSampler, self).__init__()
        self.nchannels = nchannels
        self.height_out = height_out
        self.width_out = width_out
        self.sampler_size = sampler_size
        
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

        if self.sampler_size > 1:
            # Use a sampler
            ygrid_small, xgrid_small = torch.meshgrid(
                torch.linspace(-2.0, 2.0, self.sampler_size),
                torch.linspace(-2.0, 2.0, self.sampler_size)
            )

            pgrid_small = (
                torch.exp(-xgrid_small ** 2 - ygrid_small ** 2)
            )
            pgrid_small = pgrid_small.view(-1) / pgrid_small.sum()
            self.register_buffer('xgrid_small', xgrid_small, False)
            self.register_buffer('ygrid_small', ygrid_small, False)
            self.register_buffer('pgrid_small', pgrid_small, False)

    def forward(self, inputs):
        """Takes a stack of images and a mask and samples them using Gaussians.
        
        Dimension order: batch_dim, nchannels, wx, wy
        Output: batch_dim, nchannels
        """
        X, mask = inputs

        assert mask.dtype == torch.bool
        assert X.ndim == 4
        assert X.shape[1] == mask.sum()

        if self.sampler_size <= 1:
            # Don't use sampling
            wsigmax = 0.1 + F.relu(self.wsigmax[mask].view(-1, 1))
            wsigmay = 0.1 + F.relu(self.wsigmay[mask].view(-1, 1))

            dx = (((self.xgrid.reshape(1, -1) - 
                    self.wx[mask].view(-1, 1)) ** 2 / 2 / wsigmax ** 2))
            dy = (((self.ygrid.reshape(1, -1) - 
                    self.wy[mask].view(-1, 1)) ** 2 / 2 / wsigmay ** 2))
            
            # We normalize so this roughly matches up with the sampling version.
            ws = torch.exp(-dx -dy) / (
                np.sqrt(2 * np.pi) * 
                 wsigmax * 
                 wsigmay * 
                 self.width_out * 
                 self.height_out / 4
            ) 

            R = torch.einsum('ijk,jk->ij', X.reshape(X.shape[0],
                                                     X.shape[1],
                                                     -1), ws)

        else:
            grid = torch.stack([
                    self.wx[mask].reshape(-1, 1, 1) + 
                    self.xgrid_small.reshape(1, 
                        self.xgrid_small.shape[0], 
                        self.xgrid_small.shape[1]) * (.1 + F.relu(self.wsigmax[mask].reshape(-1, 1, 1))),
                    self.wy[mask].reshape(-1, 1, 1) + 
                    self.ygrid_small.reshape(1, 
                        self.ygrid_small.shape[0], 
                        self.ygrid_small.shape[1]) * (.1 + F.relu(self.wsigmay[mask].reshape(-1, 1, 1))),
                    ], dim=3)
            
            # Right now, we have resampled the grid
            R = F.grid_sample(X.permute(1, 0, 2, 3), grid, align_corners=False)

            # Now R is ntargets x (batch_dim x nt) x Y_small x X_small

            # Depending on evaluation mode, we can either sample or take a weighted mean
            if self.training:
                # Use a consistent position in space during evaluation
                cat = torch.distributions.Categorical(self.pgrid_small)
                idx = cat.sample([self.nchannels])
                assert len(idx) == R.shape[0]

                # Haven't figured out how to do this with one call to index_select
                R = torch.stack([
                    R.reshape(R.shape[0], R.shape[1], -1)[i, :, idx[i]] for i in range(R.shape[0])
                    ], axis=0)
            else:
                R = torch.tensordot(R.view(R.shape[0], R.shape[1], -1), 
                              self.pgrid_small.view(-1), dims=([2], [0]))
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
                channel_in X height_in X width_in images
        and maps them to channels_out X height_out X width_out images.
        ntargets: the number of targets total
        channels_out: the number of channels coming out of subnet
        height_out: the number of vertical pixels coming out of subnet
        width_out: the number of horizontal pixels coming out of subnet
        nt: the number of weights in the convolution over time
        sampler_size: how big the sampling kernel should be. If 0, don't use 
            normal sampling.
    """
    def __init__(self, subnet: nn.Module, 
                       ntargets: int, 
                       channels_out: int, 
                       height_out: int, 
                       width_out: int, 
                       nt: int,
                       sampler_size: int = 0):
        super(LowRankNet, self).__init__()
        self.subnet = subnet
        self.ntargets = ntargets
        self.channels_out = channels_out
        self.width_out  = width_out
        self.height_out = height_out
        self.nt = nt
        self.sampler_size = sampler_size

        self.init_params()

    def init_params(self):
        self.wc = nn.Parameter(
            (.1 * torch.randn(self.channels_out, self.ntargets)/10 + 1) / self.channels_out
        )

        self.wt = nn.Parameter((1 + .1 * torch.randn(self.nt, self.ntargets)) / self.nt)
        self.wb = nn.Parameter(torch.randn(self.ntargets)*.05 + .6)
        self.sampler = GaussianSampler(self.ntargets, 
                                       self.height_out,
                                       self.width_out,
                                       self.sampler_size)

    def forward(self, inputs):
        x, targets = inputs
        input_shape = x.shape
        batch_dim, nchannels, nt, ny, nx = input_shape
        mask = torch.any(targets, dim=0)
        ntargets = mask.sum().item()

        assert x.ndim == 5

        # Start by mapping X through the sub-network.
        # The subnet doesn't know anything about time, so move time to the front dimension of batches
        # batch_dim x channels x time x Y x X
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_dim * nt, nchannels, ny, nx)

        x = self.subnet.forward(x)

        # Now we're at (batch_dim x nt) x channels x Y x X
        R = torch.tensordot(x, self.wc[:, mask], dims=([1], [0]))

        # Now at (batch_dim x nt) x Y x X x ntargets
        R = R.permute(0, 3, 1, 2)

        # Now R is (batch_dim x nt) x ntargets x Y x X
        # assert R.shape == (batch_dim * nt, ntargets, ny, nx)
        R = self.sampler.forward((R, mask))

        assert R.ndim == 2  
        assert R.shape[1] == ntargets

        # Finally, we're at batch_dim x outputs x nt
        # (batch_dim x nt) x ntargets

        R = R.reshape(batch_dim, nt, ntargets).permute(0, 2, 1)

        # Now at batch_dim x ntargets x nt
        R = F.conv1d(R, 
                     self.wt[:, mask].T.view(ntargets, 1, self.wt.shape[0]), 
                     bias=self.wb[mask], 
                     groups=ntargets)

        return R

    
        