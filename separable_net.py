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
    """
    def __init__(self, subnet: nn.Module, 
                       ntargets: int, 
                       channels_out: int, 
                       height_out: int, 
                       width_out: int, 
                       nt: int):
        super(LowRankNet, self).__init__()
        self.subnet = subnet
        self.ntargets = ntargets
        self.channels_out = channels_out
        self.width_out  = width_out
        self.height_out = height_out
        self.nt = nt

        self.init_params()

    def init_params(self):
        self.wc = nn.Parameter(
            (.1 * torch.randn(self.channels_out, self.ntargets)/10 + 1) / self.channels_out
        )

        self.wx = nn.Parameter(.3 + .4 * torch.rand(self.ntargets))
        self.wy = nn.Parameter(.3 + .4 * torch.rand(self.ntargets))
        self.wsigmax = nn.Parameter(.4 + .1 * torch.rand(self.ntargets))
        self.wsigmay = nn.Parameter(.4 + .1 * torch.rand(self.ntargets))
        self.wt = nn.Parameter((1 + .1 * torch.randn(self.nt, self.ntargets)) / self.nt)
        self.wb = nn.Parameter(torch.randn(self.ntargets)*.05 + .6)

        # Create the grid to evaluate the parameters on.
        ygrid, xgrid = torch.meshgrid(
            torch.arange(0, self.height_out) / self.height_out,
            torch.arange(0, self.width_out) / self.width_out,
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
        x, targets = inputs
        mask = torch.any(targets, dim=0)
        ntargets = mask.sum().item()

        assert x.ndim == 5

        # Start by mapping X through the sub-network.
        # The subnet doesn't know anything about time, so move time to the front dimension of batches
        # batch_dim x channels x time x Y x X
        front_dims = (x.shape[0], x.shape[2])
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], x.shape[3], x.shape[4])

        x = self.subnet.forward(x)


        # Now we're at (batch_dim x nt) x channels x Y x X
        dx = (((self.xgrid.reshape(-1, 1) - self.wx[mask].view(1, -1)) ** 2 / 2 ) / 
               (0.01 + self.wsigmax[mask].view(1, -1) ** 2))
        dy = (((self.ygrid.reshape(-1, 1) - self.wy[mask].view(1, -1)) ** 2 / 2 ) / 
               (0.01 + self.wsigmay[mask].view(1, -1) ** 2))

        ws = torch.exp(- dx - dy)
        ws = ws / (.1 + ws.sum(axis=0, keepdims=True))

        R = torch.tensordot(x.view(x.shape[0], x.shape[1], -1), ws, dims=([2], [0]))

        # (batch_dim x nt) x channels x outputs
        R = (R * self.wc[:, mask].view(-1, self.wc.shape[0], ntargets)).sum(axis=1)

        # (batch_dim x nt) x outputs
        R = R.reshape(front_dims[0], front_dims[1], -1).permute(0, 2, 1)

        # Now we're at batch_dim x outputs x nt
        R = F.conv1d(R, 
                     self.wt[:, mask].T.view(ntargets, 1, self.wt.shape[0]), 
                     bias=self.wb[mask], 
                     groups=ntargets)

        # Finally, we're at batch_dim x outputs x nt
        return R

    
        