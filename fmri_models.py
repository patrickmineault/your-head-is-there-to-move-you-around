import collections
import numpy as np
import os

from modelzoo import gabor_pyramid, separable_net
from loaders import vim2

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.vgg import vgg19
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18

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
        nt = 4
        assert X.shape[2] in (10, 20, 40, 80), "X.shape[2] must be 10 x a power of 2"
        stride = X.shape[2] // (nt + 1)  # Always use at most 4 time points
        delta = (X.shape[2] - (nt - 1) * stride) // 2
        slc = slice(delta, delta + stride * (nt - 1))
        return X[:, :, slc, :, :].mean(4).mean(3).reshape(
            X.shape[0], 
            X.shape[1], 
            nt, -1).mean(3).reshape(X.shape[0], -1)


def downsample(movie, width):
    data = F.interpolate(movie, 
                        (movie.shape[2], width, width), 
                        mode='trilinear',
                        align_corners=False)
    return movie

def preprocess_data(loader, model, aggregator, activations, metadata, args):
    Xs = []
    Ys = []
    for X, Y in loader:
        X, Y = X.to(device='cuda'), Y.to(device='cuda')

        with torch.no_grad():
            X = downsample(X, args.width)
            if metadata['threed']:
                result = model(X)
                fit_layer = activations[args.layer]
            else:
                result = model(X.permute(0, 2, 1, 3, 4).reshape(-1, 
                                                                X.shape[1], 
                                                                X.shape[3], 
                                                                X.shape[4]))
                fit_layer = activations[args.layer]
                fit_layer = fit_layer.reshape(X.shape[0], X.shape[2], *fit_layer.shape[1:])
                fit_layer = fit_layer.permute(0, 2, 1, 3, 4)

            stim = aggregator(fit_layer)
            Xs.append(stim)
            Ys.append(Y.squeeze())

    return torch.cat(Xs, axis=0), torch.cat(Ys, axis=0)


def get_aggregator(metadata, args):
    if args.aggregator == 'average':
        return Averager()
    elif args.aggregator == 'downsample':
        max_size = int(np.sqrt(args.max_size / metadata['sz']))
        return Downsampler(max_size)


def get_dataset(args, fold):
    if args.dataset == 'vim2':
        nframedelay = -3
        data = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2/derived'), 
                                    split=fold, 
                                    nt=1, 
                                    ntau=80, 
                                    nframedelay=nframedelay,
                                    subject=args.subject)
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
        model = gabor_pyramid.GaborPyramid3d(nlevels=args.layer+1, stride=(1, 1, 1))
        model.register_forward_hook(hook(args.layer))
        metadata = {'sz': 112,
                    'threed': True}  # The pyramid itself deals with the stride.
    elif args.features in ('r3d_18', 'mc3_18', 'r2plus1d_18'):
        if args.features == 'r3d_18':
            model = r3d_18(pretrained=True)
        elif args.features == 'mc3_18':
            model = mc3_18(pretrained=True)
        elif args.features == 'r2plus1d_18':
            model = r2plus1d_18(pretrained=True)

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
            assert layer.__repr__().startswith('ReLU')
            if i == args.layer:
                layer.register_forward_hook(hook(i))
        
        metadata = {'sz': 112,
                    'threed': True}
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
