import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d,self).__init__()

        # separable convolution from groups=in_channels
        self.conv1 = nn.Conv2d(in_channels, 
                               in_channels, 
                               kernel_size, 
                               stride, 
                               padding, 
                               dilation, 
                               groups=in_channels, 
                               bias=bias)

        # 1x1 convolution to mix the channels together
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Xception(nn.Module):
    """
    Create a truncated, smaller version of Xception.
    * Remove the computationally expensive 3x3 non-separable all-channel convolution.
    * Uses less filters overall
    * Removes some of the middle layers.

    Code adapted from https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
    """
    def __init__(self, 
                start_kernel_size=3, 
                nblocks=5,
                nstartfeats=16):
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(3, 
                               nstartfeats,
                               start_kernel_size,
                               2,
                               0,
                               bias=False
                               )
        # self.bn1 = nn.BatchNorm2d(nstartfeats)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.nblocks = nblocks

        if self.nblocks >= 1:
            # 112
            self.block1 = Block(nstartfeats, 
                                nstartfeats*2, 2, 2, start_with_relu=False)

        if self.nblocks >= 2:
            # 56
            self.block2 = Block(nstartfeats*2, 
                                nstartfeats*4, 2, 2, start_with_relu=True)
        
        if self.nblocks >= 3:
            # 28
            self.block3 = Block(nstartfeats*4, 
                                nstartfeats*8, 2, 2, start_with_relu=True)

        # Middle 
        # 14
        if self.nblocks >= 4:
            self.block4 = Block(nstartfeats*8, 
                                nstartfeats*8, 1, 1, start_with_relu=False)
        if self.nblocks >= 5:
            self.block5 = Block(nstartfeats*8, 
                                nstartfeats*8, 1, 1, start_with_relu=False)
        

    def forward(self, x):
        # Head of xception
        # 224 x 224
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        if self.nblocks >= 1:
            x = self.block1(x)
        if self.nblocks >= 2:
            x = self.block2(x)
        if self.nblocks >= 3:
            x = self.block3(x)
        if self.nblocks >= 4:
            x = self.block4(x)
        if self.nblocks >= 5:
            x = self.block5(x)

        # Endpoint
        return x

if __name__ == "__main__":
    """Smoke test."""
    net = Xception()
    the_input = torch.randn(1, 3, 224, 224)
    the_output = net.forward(the_input)

    assert the_output.ndim == 4
    assert the_output.shape == (1, 128, 14, 14)
