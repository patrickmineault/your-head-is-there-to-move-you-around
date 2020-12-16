import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class GaborPyramid(nn.Module):
    """
    Create a module that maps stacks of images to a Gabor pyramid.
    Only works in grayscale
    """
    def __init__(self, 
                 nlevels=5):
        super(GaborPyramid, self).__init__()
        self.nlevels = nlevels
        self.setup()

    def setup(self):
        # The filters will be 8x1x9x9
        xi, yi = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5))
        filters = []
        for ii in range(4):
            coso = np.cos(ii * np.pi / 4)
            sino = np.sin(ii * np.pi / 4)
            G = torch.exp(-(xi**2+yi**2)/2/2**2)
            thefilt1 = torch.cos((coso*xi+sino*yi)*.8) * G
            thefilt2 = torch.sin((coso*xi+sino*yi)*.8) * G
            thefilt1 = thefilt1 - G / G.mean() * thefilt1.mean()
            thefilt2 = thefilt2 - G / G.mean() * thefilt2.mean()
            scale = 1 / torch.sqrt((thefilt1 ** 2).sum())

            filters += [thefilt1 * scale, thefilt2 * scale]

        downsample_filt = torch.tensor([[.25, .5, .25], [.5, 1.0, .5], [.25, .5, .25]]).view(1, 1, 3, 3)
        downsample_filt /= 4.0

        filters = torch.stack(filters, dim=0).view(8, 1, 9, 9)
        self.register_buffer('filters', filters, False)
        self.register_buffer('downsample_filt', downsample_filt, False)

    def forward(self, X):
        X_ = X.sum(axis=1, keepdims=True)
        maps = []
        for i in range(self.nlevels):
            outputs = F.conv2d(X_, self.filters, padding=4)
            magnitude = torch.sqrt((outputs ** 2)[:, ::2, :, :] + 
                                   (outputs ** 2)[:, 1::2, :, :])
            if i == 0:
                maps.append(magnitude)
            else:
                maps.append(F.interpolate(magnitude, scale_factor=2**i, mode='bilinear', align_corners=False)[:, :, :X.shape[2], :X.shape[3]])

            X_ = F.conv2d(X_, self.downsample_filt, padding=1, stride=2)
        
        return torch.cat(maps, axis=1)
            

