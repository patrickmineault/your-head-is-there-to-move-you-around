import torch
from torch import nn
import torch.nn.functional as F

def mysoftplus(x, beta):
    print(x.max())
    print(beta.max())
    return torch.where(abs(x) < 20, 
                       1 / beta * torch.log(1 + torch.exp(beta * x)),
                       torch.where( x < 0, 0.0 * x, x))

class LowRankNet(nn.Module):
    """Creates a low-rank lagged network for prediction.
    
    Takes a network that maps images to stacks of features and maps
    to multi-target networks which are separable combinations along channel, 
    time, and space dimensions.

    Right now, only rank = 1 is implemented.

    Arguments:
        subnet: a nn.Module whose forward op takes batches of channel_in X height_in X width_in images
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

        self.wx = nn.Parameter(-.4 + .8 * torch.rand(self.ntargets))
        self.wy = nn.Parameter(-.4 + .8 * torch.rand(self.ntargets))
        self.wsigmax = nn.Parameter(.5 + .1 * torch.rand(self.ntargets))
        self.wsigmay = nn.Parameter(.5 + .1 * torch.rand(self.ntargets))
        self.wt = nn.Parameter((1 + .1 * torch.randn(self.nt, self.ntargets)) / self.nt)
        self.wb = nn.Parameter(torch.randn(self.ntargets)*.05 + .6)

        # Create the grid to evaluate the parameters on.
        ygrid, xgrid = torch.meshgrid(
            torch.arange(-self.height_out / 2 + .5, self.height_out / 2 + 0.5) / self.height_out,
            torch.arange(-self.width_out / 2 + .5, self.width_out / 2 + 0.5) / self.width_out,
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
            small_grid = (self.sampler_size - 1) // 2
            ygrid_small, xgrid_small = torch.meshgrid(
                torch.arange(-small_grid, small_grid + 1) / small_grid * 2.0,
                torch.arange(-small_grid, small_grid + 1) / small_grid * 2.0,
            )

            pgrid_small = (
                torch.exp(-xgrid_small ** 2 - ygrid_small ** 2)
            )
            pgrid_small = pgrid_small.view(-1) / pgrid_small.sum()
            self.register_buffer('xgrid_small', xgrid_small, False)
            self.register_buffer('ygrid_small', ygrid_small, False)
            self.register_buffer('pgrid_small', pgrid_small, False)


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

        if self.sampler_size <= 1:
            # Don't use sampling
            dx = (((self.xgrid.reshape(-1, 1) - 
                    self.wx[mask].view(1, -1)) ** 2 / 2 ) / 
                (0.1 + F.relu(self.wsigmax[mask].view(1, -1)) ** 2))
            dy = (((self.ygrid.reshape(-1, 1) - 
                    self.wy[mask].view(1, -1)) ** 2 / 2 ) / 
                (0.1 + F.relu(self.wsigmay[mask].view(1, -1)) ** 2))

            ws = torch.exp(-dx -dy)
            ws = ws / (.1 + ws.sum(axis=0, keepdims=True))

            R = torch.einsum('ijk,kj->ij', R.reshape(R.shape[0],
                                                     R.shape[1],
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
            assert R.shape[0] == batch_dim * nt
            R = F.grid_sample(R.permute(1, 0, 2, 3), grid, align_corners=False)

            # Now R is ntargets x (batch_dim x nt) x Y_small x X_small

            # Depending on evaluation mode, we can either sample or take a weighted mean
            if self.training:
                # Use a consistent position in space during evaluation
                cat = torch.distributions.Categorical(self.pgrid_small)
                idx = cat.sample([ntargets])
                assert len(idx) == R.shape[0]

                # Haven't figured out how to do this with one call to index_select
                R = torch.stack([
                    R.reshape(R.shape[0], R.shape[1], -1)[i, :, idx[i]] for i in range(R.shape[0])
                    ], axis=0)
            else:
                R = torch.tensordot(R.view(R.shape[0], R.shape[1], -1), 
                              self.pgrid_small.view(-1), dims=([2], [0]))
            R = R.T

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

    
        