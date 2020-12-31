import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class SymmetricConv3d(nn.Module):
    """Convolution, adding symmetric versions for equivariance."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride,
                 padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k = 1 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        w = 2 * np.sqrt(k) * (torch.rand(out_channels, in_channels, *kernel_size) - 0.5)
        w = nn.Parameter(w)
        self.register_parameter('weight', w)

    def forward(self, X):
        w = torch.cat(
            (torch.rot90(self.weight, 0, [3, 4]),
             torch.rot90(self.weight, 1, [3, 4]), 
             torch.rot90(self.weight, 2, [3, 4]), 
             torch.rot90(self.weight, 3, [3, 4])), axis=0
        )

        return F.conv3d(X, 
                        w, 
                        padding=self.padding, 
                        stride=self.stride)

class ShallowNet(nn.Module):
    """
    Create a tiny shallow network to approximate the whole freakin brain.
    """
    def __init__(self, 
                 nstartfeats=16,
                 symmetric=False):
        super().__init__()

        if symmetric:
            assert nstartfeats % 4 == 0
            self.conv1 = SymmetricConv3d(
                3, 
                nstartfeats // 4,
                [5, 7, 7],
                [1, 2, 2],
                padding=[2, 3, 3],
            )
        else:
            self.conv1 = nn.Conv3d(3, 
                                nstartfeats,
                                [5, 7, 7],
                                [1, 2, 2],
                                padding=[2, 3, 3],
                                bias=False
                                )

        self.bn1 = nn.BatchNorm3d(nstartfeats)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3],
                                       stride=[1, 2, 2],
                                       padding=[0, 1, 1])

        self.dropout = nn.Dropout3d(.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        x = self.dropout(x)

        # Endpoint
        return x