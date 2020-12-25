import collections
import numpy as np
import os

from modelzoo import gabor_pyramid, separable_net
from loaders import vim2

from torch import nn
import torch.nn.functional as F

from torchvision.models.vgg import vgg19
from torchvision.models.video import r3d_18

class Passthrough(nn.Module):
    def __init__(self):
        super(Passthrough, self).__init__()

    def forward(self, data):
        return data


class Downsampler(nn.Module):
    def __init__(self, max_size):
        super(Downsampler, self).__init__()
        self.max_size = max_size

    def forward(self, data):
        if data.shape[-1] > self.max_size:
            data = F.interpolate(data, 
                                (data.shape[2], self.max_size, self.max_size), 
                                mode='trilinear',
                                align_corners=True)
        return data


class Averager(nn.Module):
    def __init__(self):
        super(Averager, self).__init__()

    def forward(self, X):
        assert X.shape[2] in (10, 20, 40, 80)
        stride = X.shape[2] // 5  # Always use at most 4 time points
        delta = (X.shape[2] - 3 * stride) // 2
        slc = slice(delta, delta + stride * 3, stride)
        return X[:, :, slc, :, :].mean(4).mean(3).reshape(X.shape[0], -1)

def get_aggregator(metadata, args):
    if args.aggregator == 'average':
        return Averager()
    elif args.aggregator == 'downsample':
        max_size = int(np.sqrt(args.max_size / metadata['sz']))
        return Downsampler(max_size)

def get_dataset(args, fold):
    if args.dataset == 'vim2':
        nframedelay = -3
        data = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2'), 
                                    split=fold, 
                                    nt=1, 
                                    nx=112,
                                    ny=112,
                                    ntau=80, 
                                    nframedelay=nframedelay,
                                    subject=args.subject,
                                    subset=args.subset)
    else:
        raise NotImplementedError("Only vim2 implemented")

    return data


def get_feature_model(args):
    activations = collections.OrderedDict()
    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o
        return hook_fn

    if args.features == 'gaborpyramid3d':
        model = gabor_pyramid.GaborPyramid3d(nlevels=5, stride=(1, 1, 1))
        model.register_forward_hook(hook(0))
        metadata = {'sz': 112,
                    'threed': True}  # The pyramid itself deals with the stride.
    elif args.features == 'r3d_18':
        model = r3d_18(pretrained=True)
        layers = (
            model.stem[2],
            model.layer1[0].conv1[2],
            model.layer1[0].relu,
            model.layer1[1].conv1[2],
            model.layer1[1].relu,
            model.layer2[0].conv1[2],
            model.layer2[0].relu,
            model.layer2[1].conv1[2],
            model.layer2[1].relu,
            model.layer3[0].conv1[2],
            model.layer3[0].relu,
            model.layer3[1].conv1[2],
            model.layer3[1].relu,
            model.layer4[0].conv1[2],
            model.layer4[0].relu,
            model.layer4[1].conv1[2],
            model.layer4[1].relu,
        )
        for i, layer in enumerate(layers):
            if i == args.layer:
                layer.register_forward_hook(hook(i))
        
        metadata = {'sz': 112,
                    'threed': True}  # The pyramid itself deals with the stride.
    elif args.features == 'vgg19':
        model = vgg19(pretrained=True)
        layers = [layer for layer in model.features if 
                  layer.__repr__().startswith('ReLU')]

        for i, layer in enumerate(layers):
            if i == args.layer:
                layer.register_forward_hook(hook(i))

        metadata = {'sz': 112,
                    'threed': False}
    else:
        raise NotImplementedError('Model not implemented yet')

    model.eval()
    return model, activations, metadata


def get_readout_model(args, threed, trainset):
    if args.readout == 'gaussian':
        subnet = Downsampler(max_size=32)
        net = separable_net.LowRankNet(subnet, 
                                    trainset.total_electrodes, 
                                    args.nfeats, 
                                    32, 
                                    32, 
                                    trainset.ntau - 1,
                                    sample=(not args.no_sample), 
                                    threed=threed,
                                    output_nl=False)
    elif args.readout == 'average':
        subnet = Passthrough()
        net = separable_net.AverageNet(subnet, 
                                    trainset.total_electrodes, 
                                    args.nfeats, 
                                    32, 
                                    32, 
                                    trainset.ntau - 1,
                                    sample=(not args.no_sample), 
                                    threed=threed,
                                    output_nl=False)
    else:
        raise NotImplementedError(f"{args.readout} readout not implemented")
    return net, subnet
