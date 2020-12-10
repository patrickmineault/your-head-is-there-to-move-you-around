import torch
from torch import nn
import torch.nn.functional as F

class LowRankNet(nn.Module):
    """Creates a low-rank lagged network for prediction.
    
    Takes a network that maps images to stacks of features and maps
    to multi-target networks which are separable combinations along channel, 
    time, and space dimensions.

    Arguments:
        subnet: a nn.Module whose forward op takes batches of channel_in X height_in X width_in images
        and maps them to channels_out X height_out X width_out images.
        rank: the rank of the transformation
        ntargets: the number of targets total
        channels_out: the number of channels coming out of subnet
        height_out: the number of vertical pixels coming out of subnet
        width_out: the number of horizontal pixels coming out of subnet
        nt: the number of weights in the convolution over time
    """
    def __init__(self, subnet: nn.Module, 
                       rank: int, 
                       ntargets: int, 
                       channels_out: int, 
                       height_out: int, 
                       width_out: int, 
                       nt: int):
        super(LowRankNet, self).__init__()
        self.subnet = subnet
        self.rank = rank
        self.ntargets = ntargets
        self.channels_out = channels_out
        self.width_out  = width_out
        self.height_out = height_out
        self.nt = nt

        self.init_params()

    def init_params(self):
        self.wc = nn.Parameter(torch.randn(self.channels_out, self.rank, self.ntargets))
        self.wx = nn.Parameter(torch.rand(self.rank, self.ntargets) * self.width_out)
        self.wy = nn.Parameter(torch.rand(self.rank, self.ntargets) * self.height_out)
        self.wsigmax = nn.Parameter(torch.randn(self.rank, self.ntargets))
        self.wsigmay = nn.Parameter(torch.randn(self.rank, self.ntargets))
        self.wt = nn.Parameter(torch.randn(self.nt, self.ntargets))
        self.wb = nn.Parameter(torch.randn(self.ntargets))

        # Create the grid to evaluate the parameters on.
        xgrid, ygrid = torch.meshgrid(
            torch.arange(0, self.width_out),
            torch.arange(0, self.height_out),
        )
        assert xgrid.shape[0] == self.height_out
        assert xgrid.shape[1] == self.width_out
        xgrid = xgrid.view(1, self.height_out, self.width_out)
        ygrid = ygrid.view(1, self.height_out, self.width_out)
        
        self.register_buffer('xgrid', xgrid, False)
        self.register_buffer('ygrid', ygrid, False)

    def forward(self, inputs):
        x, targets = inputs
        targets = targets.squeeze()

        if x.shape[0] != 1:
            raise NotImplementedError("Cannot deal with more than 1 batch at a time")

        assert x.ndim == 5

        # Start by mapping X through the sub-networks.
        # The subnet doesn't know anything about time, so move time to the front dimension of batches
        front_dims = (x.shape[0], x.shape[4])
        x = x.permute(0, 4, 1, 2, 3).reshape(-1, x.shape[1], x.shape[2], x.shape[3])
        x = self.subnet.forward(x)

        # x has shape (batch size x nt) X channels_out X height_out X width_out
        results = []
        for target in targets:
            r = torch.tensordot(x, self.wc[:, :, target], dims=([1], [0]))
            # r has shape (batch size x nt) X height_out X width_out X rank

            ws = torch.exp(-(self.xgrid - self.wx[:, target].view(-1, 1, 1)) ** 2 / 2 / (0.5 + self.wsigmax[:, target].view(-1, 1, 1) ** 2) - 
                            (self.ygrid - self.wy[:, target].view(-1, 1, 1)) ** 2 / 2 / (0.5 + self.wsigmay[:, target].view(-1, 1, 1) ** 2))
            
            # ws has shape rank x height_out X width_out
            r = torch.tensordot(r, ws, ([1, 2, 3], [1, 2, 0]))

            # r has shape (batch_size X nt)
            r = F.conv1d(r.view(front_dims[0], 1, front_dims[1]), 
                         self.wt[:, target].view(1, 1, -1), 
                         self.wb[target].view(1))

            results.append(r.view(front_dims[0], -1))
        return torch.stack(results, 2)

    
        