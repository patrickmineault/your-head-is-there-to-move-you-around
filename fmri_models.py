import collections
import git
import numpy as np
import os
import tables
from tqdm import tqdm

from modelzoo import gabor_pyramid, separable_net
from loaders import vim2

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18
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
        delta = (X.shape[2] - nt * stride) // 2
        slc = slice(delta, delta + stride * nt)
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

def preprocess_data(loader, 
                    model, 
                    aggregator, 
                    activations, 
                    metadata, 
                    args):

    # Check if cache exists for this model.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    cache_file = f'{args.features}_{args.width}_{args.dataset}_{args.subject}_{loader.dataset.split}_{args.aggregator}_{sha}.h5'
    cache_file = os.path.join(args.cache_root, cache_file)

    if not os.path.exists(cache_file):
        print("Create cache file")
        h5file = tables.open_file(cache_file, mode="w", title="Cache file")
        layers = {}
        outputs = None
        nrows = len(loader) * loader.batch_size

        progress_bar = tqdm(total=len(loader), unit='batches', unit_scale=True)

        for i, (X, Y) in enumerate(loader):
            progress_bar.update(1)
            X, Y = X.to(device='cuda'), Y.to(device='cuda')

            with torch.no_grad():
                X = downsample(X, args.width)
                if metadata['threed']:
                    result = model(X)
                    
                    for layer in activations.keys():
                        fit_layer = aggregator(activations[layer]).cpu().detach().numpy()
                        
                        if outputs is None:
                            layers[layer] = h5file.create_earray('/', f'layer{layer}', obj=fit_layer, expectedrows=nrows)
                        else:
                            layers[layer].append(fit_layer)
                else:
                    result = model(X.permute(0, 2, 1, 3, 4).reshape(-1, 
                                                                    X.shape[1], 
                                                                    X.shape[3], 
                                                                    X.shape[4]))
                    
                    for layer in activations.keys():
                        fit_layer = activations[layer]
                        fit_layer = fit_layer.reshape(X.shape[0], X.shape[2], *fit_layer.shape[1:])
                        fit_layer = fit_layer.permute(0, 2, 1, 3, 4).cpu().detach().numpy()
                        fit_layer = aggregator(fit_layer)

                        if outputs is None:
                            layers[layer] = h5file.create_earray('/', f'layer{layer}', obj=fit_layer, expectedrows=nrows)
                        else:
                            layers[layer].append(fit_layer)

                if outputs is None:
                    outputs = h5file.create_earray('/', 
                                                f'outputs', 
                                                obj=Y.cpu().detach().numpy(), 
                                                expectedrows=nrows)
                else:
                    outputs.append(Y.cpu().detach().numpy())

        progress_bar.close()
        h5file.close()
    
    h5file = tables.open_file(cache_file, mode="r")
    X = torch.tensor(h5file.get_node(f'/layer{args.layer}')[:], device='cuda')
    Y = torch.tensor(h5file.get_node(f'/outputs')[:], device='cuda').squeeze()
    h5file.close()
    return X, Y


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
            layer.register_forward_hook(hook(i))
        
        metadata = {'sz': 112,
                    'threed': True}
    elif args.features in ('vgg19'):
        model = vgg19(pretrained=True)
        layers = [layer for layer in model.features if 
                  layer.__repr__().startswith('ReLU')]

        for i, layer in enumerate(layers):
            if i == args.layer:
                layer.register_forward_hook(hook(i))

        metadata = {'sz': 112,
                    'threed': False}
    elif args.features in ('resnet18'):
        model = resnet18(pretrained=True)

        layers = (
            model.relu,
            model.layer1[0].relu,
            model.layer1[0],
            model.layer1[1].relu,
            model.layer1[1],
            model.layer2[0].relu,
            model.layer2[0],
            model.layer2[1].relu,
            model.layer2[1],
            model.layer3[0].relu,
            model.layer3[0],
            model.layer3[1].relu,
            model.layer3[1],
            model.layer4[0].relu,
            model.layer4[0],
            model.layer4[1].relu,
            model.layer4[1],
        )

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
