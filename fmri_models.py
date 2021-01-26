import collections
import git
import numpy as np
import os
import sklearn
import sklearn.decomposition
import sklearn.random_projection
import tables
from tqdm import tqdm

from loaders import pvc4, pvc1, vim2, mt2
from modelzoo import gabor_pyramid, separable_net
from modelzoo.motionnet import MotionNet
from modelzoo.slowfast_wrapper import SlowFast
from modelzoo.shiftnet import ShiftNet
from modelzoo.monkeynet import ShallowNet, V1Net, DorsalNet

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


def downsample_3d(X, sz):
    """
    Spatially downsamples a stack of square videos.
    
    Args:
        X: a stack of images (batch, channels, nt, ny, ny).
        sz: the desired size of the videos.
        
    Returns:
        The downsampled videos, a tensor of shape (batch, channel, nt, sz, sz)
    """
    kernel = torch.tensor([[.25, .5, .25], 
                           [.5, 1, .5], 
                           [.25, .5, .25]], device=X.device).reshape(1, 1, 1, 3, 3)
    kernel = kernel.repeat((X.shape[1], 1, 1, 1, 1))
    while sz < X.shape[-1] / 2:
        # Downsample by a factor 2 with smoothing
        mask = torch.ones(1, *X.shape[1:], device=X.device)
        mask = F.conv3d(mask, kernel, groups=X.shape[1], stride=(1, 2, 2), padding=(0, 1, 1))
        X = F.conv3d(X, kernel, groups=X.shape[1], stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Normalize the edges and corners.
        X = X = X / mask
    
    return F.interpolate(X, size=(X.shape[2], sz, sz), mode='trilinear', align_corners=True)

"""
def get_downsampler(X0_shape, X_shape, Y_shape, ntau):
    downsample_factor = X0_shape[2] // X_shape[2]
    assert downsample_factor in (1, 2, 4, 8)

    nt = 1
    t = torch.arange(0, X_shape[2]) * downsample_factor
    stride = ntau // (nt + 1)

    dt = (X0_shape[2] - ntau) / (Y0_shape[1] - 1)

    for i in range(Y_shape[1]):
        t0 = (i + 1) * dt
        (t - t0)
"""

class Downsampler(nn.Module):
    def __init__(self, sz):
        super(Downsampler, self).__init__()
        self.sz = sz

    def forward(self, data):
        nt = 4
        if isinstance(data, tuple):
            X, X0_shape, Y0_shape, ntau = data
            ny = Y0_shape[1]
        else:
            X = data
            ntau = X.shape[2]
            ny = 1
            
            # 6 and 12 are one-offs for FastSlow_Fast
            if X.shape[2] not in (5, 6, 10, 12, 20, 40, 80):
                raise NotImplementedError("X.shape[2] must be 10 x a power of 2")

        # assert X.shape[2] in (10, 20, 40, 80, 200), "X.shape[2] must be 10 x a power of 2"
        stride = ntau // (nt + 1)  # Always use at most 4 time points
        delta = stride // 2
        
        X_ = downsample_3d(X, self.sz)

        if ny == 1:
            slc = slice(delta, delta + nt * stride)
            return X_[:, :, slc, :, :].reshape(
                X.shape[0], 
                X.shape[1], 
                nt, -1, X_.shape[-2], X_.shape[-1]).mean(3).reshape(X_.shape[0], -1)
        else:
            # Need to restride the data. 
            downsample_amount = X.shape[2] / X0_shape[2]

            dt = (X0_shape[2] - ntau) / (Y0_shape[1] - 1)

            Xs = []
            for i in range(ny):
                slc = slice(int(downsample_amount * (i * dt + delta)), 
                            int(downsample_amount * (i * dt + ntau - delta)))
                assert (slc.stop - slc.start) % nt == 0

                
                Xs.append(
                    X_[:, :, slc, :, :].reshape(
                        X_.shape[0],
                        X_.shape[1],
                        nt, -1, X_.shape[-2], X_.shape[-1]).mean(3).reshape(X_.shape[0], -1)
                )
            
            return torch.stack(Xs, axis=1).reshape(-1, Xs[0].shape[-1])

class RP(nn.Module):
    def __init__(self, ncomp, sparse=True):
        super().__init__()
        self.downsampler = Downsampler(8)
        self.ncomp = ncomp
        self.Ps = {}
        self.sparse = sparse

    def forward(self, X):
        X = self.downsampler(X)
        P = self._get_projection_matrix(X.shape[1], device=X.device)
        if self.sparse:
            # Note: Currently, PyTorch does not support matrix multiplication 
            # with the layout signature M[strided] @ M[sparse_coo]
            result = torch.matmul(P, X.T).T
        else:
            result = torch.matmul(X, P)

        return result

    def _get_projection_matrix(self, nfeatures, device):
        if nfeatures not in self.Ps:
            if self.sparse:
                rp = sklearn.random_projection.SparseRandomProjection(
                    self.ncomp, 
                    density = .05,
                    random_state=0xdeadbeef)

                mat = rp._make_random_matrix(self.ncomp, nfeatures)
                coo = mat.tocoo()

                values = coo.data
                indices = np.vstack((coo.row, coo.col))

                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = coo.shape

                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = coo.shape

                self.Ps[nfeatures] = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)
            else:
                rp = sklearn.random_projection.GaussianRandomProjection(
                    self.ncomp, 
                    random_state=0xdeadbeef)

                mat = rp._make_random_matrix(self.ncomp, nfeatures)
                self.Ps[nfeatures] = torch.tensor(mat.T, dtype=torch.float, device=device)

            
        return self.Ps[nfeatures]


class Averager(nn.Module):
    def __init__(self):
        super(Averager, self).__init__()

    def forward(self, data):
        nt = 4
        if isinstance(data, tuple):
            X, X0_shape, Y0_shape, ntau = data
            ny = Y0_shape[1]
        else:
            X = data
            ntau = X.shape[2]
            ny = 1
            assert X.shape[2] in (10, 20, 40, 80), "X.shape[2] must be 10 x a power of 2"

        if X.ndim == 4:
            X = X.unsqueeze(0)

        # assert X.shape[2] in (10, 20, 40, 80, 200), "X.shape[2] must be 10 x a power of 2"
        stride = ntau // (nt + 1)  # Always use at most 4 time points
        delta = stride // 2
        
        X_ = X.mean(4).mean(3)

        if ny == 1:
            slc = slice(delta, ntau - delta)
            return X_[:, :, slc].reshape(
                X.shape[0], 
                X.shape[1], 
                nt, -1).mean(3).reshape(X_.shape[0], -1)
        else:
            # Need to restride the data. 
            dt = (X0_shape[2] - ntau) / (Y0_shape[1] - 1)
            Xs = []
            for i in range(ny):
                slc = slice(int(delta + i * dt), 
                            int(ntau - delta + i * dt))
                assert (slc.stop - slc.start) % nt == 0
                Xs.append(
                    X_[:, :, slc].reshape(
                        X_.shape[0],
                        X_.shape[1],
                        nt, -1).mean(3).reshape(X_.shape[0], -1)
                )
            
            return torch.stack(Xs, axis=1).reshape(-1, Xs[0].shape[-1])

def get_projection_matrix(X, n):
    X_ = X.cpu().detach().numpy()
    svd = sklearn.decomposition.TruncatedSVD(n_components=n, random_state=0xadded)
    r = svd.fit_transform(X_)
    return torch.tensor(svd.components_.T / r[:, 0].std(), device=X.device)

def resize(movie, width):
    data = F.interpolate(movie, 
                        (movie.shape[2], width, width), 
                        mode='trilinear',
                        align_corners=False)
    return data

def preprocess_data_consolidated(loader, 
                                model, 
                                aggregator, 
                                activations, 
                                metadata, 
                                args):

    # Check if cache exists for this model.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    cache_file = f'{args.features}_{metadata["sz"]}_{args.dataset}_{args.subset}_{loader.dataset.split}_{args.aggregator}_cons_{sha}.h5'
    cache_file = os.path.join(args.cache_root, cache_file)

    if not os.path.exists(cache_file):
        print("Create cache file")
        h5file = tables.open_file(cache_file, mode="w", title="Cache file")
        layers = {}
        outputs = None
        nrows = len(loader) * loader.batch_size

        progress_bar = tqdm(total=len(loader), unit='batches', unit_scale=True)

        for i, loaded in enumerate(loader):
            if len(loaded) == 2:
                X, Y = loaded
            else:
                X, _, _, Y = loaded

                # So the order of the indices matches the order in vim2
                Y = Y.permute(0, 2, 1)

            progress_bar.update(1)
            X, Y = X.to(device='cuda'), Y.to(device='cuda')

            with torch.no_grad():
                X = resize(X, metadata['sz'])
                if metadata['threed']:
                    result = model(X)
                    
                    for layer in activations.keys():
                        try:
                            fit_layer = aggregator((activations[layer], 
                                                    X.shape, 
                                                    Y.shape,
                                                    loader.dataset.ntau)).cpu().detach().numpy()
                        except NotImplementedError:
                            # This is because the output is too small, so the aggregator doesn't work.
                            continue
                        
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
                        fit_layer = fit_layer.permute(0, 2, 1, 3, 4)
                        fit_layer = aggregator(fit_layer).cpu().detach().numpy()

                        if outputs is None:
                            layers[layer] = h5file.create_earray('/', f'layer{layer}', obj=fit_layer, expectedrows=nrows)
                        else:
                            layers[layer].append(fit_layer)

                #Y = Y.permute(1, 0, 2).reshape(-1, Y.shape[2])
                Y = Y.reshape(-1, Y.shape[2])

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
    try:
        X = torch.tensor(h5file.get_node(f'/layer{args.layer_name}')[:], device='cpu')
    except tables.exceptions.NoSuchNodeError:
        h5file.close()
        return None, None
    Y = torch.tensor(h5file.get_node(f'/outputs')[:], device='cpu', dtype=torch.float).squeeze()
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    h5file.close()
    return X, Y

def tune_batch_size(model, loader, metadata):
    """This doesn't help _that_ much, about 20%."""
    print("Tuning batch size")

    sampler_size = 4

    Xs = []
    Ys = []
    for idx in range(sampler_size):
        loaded = loader[idx]
        if len(loaded) == 2:
            X, Y = loaded
        else:
            X, _, _, Y = loaded
        Xs.append(torch.tensor(X, device='cuda'))
        Ys.append(torch.tensor(Y, device='cuda'))

    X = torch.stack(Xs, axis=0)
    Y = torch.stack(Ys, axis=0)

    X, Y = X.to(device='cuda'), Y.to(device='cuda')

    X = resize(X, metadata['sz'])
    
    _ = model(X)

    # Tune the batch size to maximize throughput.
    import GPUtil

    devices = GPUtil.getGPUs()
    multiplier = devices[0].memoryTotal // devices[0].memoryUsed

    batch_size = int(multiplier * sampler_size)
    print(f"Automatic batch size of {batch_size}")
    return batch_size

def preprocess_data(loader, 
                    model, 
                    aggregator, 
                    activations, 
                    metadata, 
                    args):

    # Check if cache exists for this model.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    cache_file = f'{args.features}_{metadata["sz"]}_{args.dataset}_{args.subset}_{loader.dataset.split}_{args.aggregator}_{sha}.h5'
    cache_file = os.path.join(args.cache_root, cache_file)

    if not os.path.exists(cache_file):
        print("Create cache file")
        h5file = tables.open_file(cache_file, mode="w", title="Cache file")
        layers = {}
        outputs = None
        nrows = len(loader) * loader.batch_size

        progress_bar = tqdm(total=len(loader), unit='batches', unit_scale=True)

        for i, loaded in enumerate(loader):
            if len(loaded) == 2:
                X, Y = loaded
            else:
                X, _, _, Y = loaded
            progress_bar.update(1)
            X, Y = X.to(device='cuda'), Y.to(device='cuda')

            with torch.no_grad():
                X = resize(X, metadata['sz'])
                if metadata['threed']:
                    result = model(X)
                    
                    for layer in activations.keys():
                        try:
                            al = activations[layer]
                            # print(al.shape)
                            fit_layer = aggregator(al).cpu().detach().numpy()
                        except NotImplementedError as e:
                            # This is because the output is too small, so the aggregator doesn't work.
                            # print(e)
                            # raise(e)
                            continue
                        
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
                        fit_layer = fit_layer.permute(0, 2, 1, 3, 4)
                        fit_layer = aggregator(fit_layer).cpu().detach().numpy()

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
    try:
        X = torch.tensor(h5file.get_node(f'/layer{args.layer_name}')[:], device='cpu')
    except tables.exceptions.NoSuchNodeError:
        h5file.close()
        return None, None
    Y = torch.tensor(h5file.get_node(f'/outputs')[:], device='cpu', dtype=torch.float).squeeze()
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    h5file.close()
    return X, Y


def get_aggregator(metadata, args):
    if args.aggregator == 'average':
        return Averager()
    elif args.aggregator == 'downsample':
        return Downsampler(args.aggregator_sz)
    elif args.aggregator == 'rp':
        return RP(args.aggregator_sz, sparse=True)
    else:
        raise NotImplementedError(f"Aggregator {args.aggregator} not implemented.")


def get_dataset(args, fold):
    if args.consolidated:
        nt = 9
    else:
        nt = 1

    # SlowFast_Fast has a limitation that it doesn't work with small inputs,
    # so fudge things here.
    if args.features == 'SlowFast_Fast':
        ntau = 12
    else:
        ntau = 10
        
    if args.dataset == 'vim2':
        nframedelay = -3
        data = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2'), 
                                    split=fold, 
                                    nt=nt, 
                                    ntau=80, 
                                    nframedelay=nframedelay,
                                    subject=args.subset)
    elif args.dataset == 'vim2_deconv':
        nframedelay = -3
        data = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2'), 
                                    split=fold, 
                                    nt=nt, 
                                    ntau=80, 
                                    nframedelay=nframedelay,
                                    subject=args.subset,
                                    deconvolved=True)
    elif args.dataset == 'pvc1':
        data = pvc1.PVC1(os.path.join(args.data_root, 'crcns-ringach-data'), 
                         split=fold, 
                         nt=nt, 
                         nx=112,
                         ny=112,
                         ntau=ntau, 
                         nframedelay=0,
                         single_cell=int(args.subset))
    elif args.dataset == 'pvc4':
        data = pvc4.PVC4(os.path.join(args.data_root, 'crcns-pvc4'), 
                            split=fold, 
                             nt=nt, 
                             nx=112,
                             ny=112,
                             ntau=ntau, 
                             nframedelay=0,
                             single_cell=int(args.subset))
    elif args.dataset == 'v2':
        data = pvc4.PVC4(os.path.join(args.data_root, 'crcns-v2'), 
                            split=fold, 
                             nt=nt, 
                             nx=112,
                             ny=112,
                             ntau=ntau, 
                             nframedelay=0,
                             single_cell=int(args.subset))
    elif args.dataset == 'mt2':
        data = mt2.MT2(os.path.join(args.data_root, 'crcns-mt2'), 
                            split=fold, 
                             nt=nt, 
                             nx=112,
                             ny=112,
                             ntau=ntau, 
                             nframedelay=1,
                             single_cell=int(args.subset))

    else:
        raise NotImplementedError(f"{args.dataset} implemented")

    return data


def get_feature_model(args):
    activations = collections.OrderedDict()
    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o
        return hook_fn

    if args.features == 'gaborpyramid3d':
        model = gabor_pyramid.GaborPyramid3d(nlevels=4, stride=(1, 1, 1))
        layers = collections.OrderedDict([
            ('layer00', model)
        ])
        metadata = {'sz': 112,
                    'threed': True}  # The pyramid itself deals with the stride.
    elif args.features == 'gaborpyramid3d_motionless':
        model = gabor_pyramid.GaborPyramid3d(nlevels=4, stride=(1, 1, 1), motionless=True)
        layers = collections.OrderDict([
            ('layer00', model)
        ])
        metadata = {'sz': 112,
                    'threed': True}  # The pyramid itself deals with the stride.
    elif args.features in ('r3d_18', 'mc3_18', 'r2plus1d_18'):
        if args.features == 'r3d_18':
            model = r3d_18(pretrained=True)
        elif args.features == 'mc3_18':
            model = mc3_18(pretrained=True)
        elif args.features == 'r2plus1d_18':
            model = r2plus1d_18(pretrained=True)

        layers = collections.OrderedDict([
            ('layer00', model.stem[2]),
            ('layer01', model.layer1[0].conv1[2]),
            ('layer02', model.layer1[0].relu),
            ('layer03', model.layer1[1].conv1[2]),
            ('layer04', model.layer1[1].relu),
            ('layer05', model.layer2[0].conv1[2]),
            ('layer06', model.layer2[0].relu),
            ('layer07', model.layer2[1].conv1[2]),
            ('layer08', model.layer2[1].relu),
            ('layer09', model.layer3[0].conv1[2]),
            ('layer10', model.layer3[0].relu),
            ('layer11', model.layer3[1].conv1[2]),
            ('layer12', model.layer3[1].relu),
            ('layer13', model.layer4[0].conv1[2]),
            ('layer14', model.layer4[0].relu),
            ('layer15', model.layer4[1].conv1[2]),
            ('layer16', model.layer4[1].relu)
        ])

        if args.subsample_layers:
            nums = [0, 1, 2, 4, 6, 8, 10, 12]
            l = []
            for i, (layer_name, layer) in enumerate(layers.items()):
                if i in nums:
                    l.append((layer_name, layer))

            layers = collections.OrderedDict(l)

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features in ('vgg19'):
        model = vgg19(pretrained=True)
        layers = [layer for layer in model.features if 
                  layer.__repr__().startswith('ReLU')]

        metadata = {'sz': 224,
                    'threed': False}
    elif args.features in ('resnet18'):
        model = resnet18(pretrained=True)

        layers = collections.OrderedDict([
            ('layer00', model.relu),
            ('layer01', model.layer1[0].relu),
            ('layer02', model.layer1[0]),
            ('layer03', model.layer1[1].relu),
            ('layer04', model.layer1[1]),
            ('layer05', model.layer2[0].relu),
            ('layer06', model.layer2[0]),
            ('layer07', model.layer2[1].relu),
            ('layer08', model.layer2[1]),
            ('layer09', model.layer3[0].relu),
            ('layer10', model.layer3[0]),
            ('layer11', model.layer3[1].relu),
            ('layer12', model.layer3[1]),
            ('layer13', model.layer4[0].relu),
            ('layer14', model.layer4[0]),
            ('layer15', model.layer4[1].relu),
            ('layer16', model.layer4[1]),
        ])

        metadata = {'sz': 224,
                    'threed': False}
    elif args.features in ('SlowFast_Slow', 'SlowFast_Fast', 'Slow', 'I3D'):
        model = SlowFast(args)

        if args.features == 'SlowFast_Fast':
            layers = collections.OrderedDict([
                ('layer00', model.model.s1.pathway1_stem.relu),
                ('layer01', model.model.s1.pathway1_stem),
                ('layer02', model.model.s2.pathway1_res0),
                ('layer03', model.model.s2.pathway1_res1),
                ('layer04', model.model.s2.pathway1_res2),
                ('layer05', model.model.s3.pathway1_res0),
                ('layer06', model.model.s3.pathway1_res1),
                ('layer07', model.model.s3.pathway1_res2),
                ('layer08', model.model.s3.pathway1_res3),
                ('layer09', model.model.s4.pathway1_res0),
                ('layer10', model.model.s4.pathway1_res1),
                ('layer11', model.model.s4.pathway1_res2),
                ('layer12', model.model.s4.pathway1_res3),
                ('layer13', model.model.s4.pathway1_res4),
                ('layer14', model.model.s4.pathway1_res5),
                ('layer15', model.model.s5.pathway1_res0),
                ('layer16', model.model.s5.pathway1_res1),
                ('layer17', model.model.s5.pathway1_res2),
            ])
        else:
            layers = collections.OrderedDict([
                ('layer00', model.model.s1.pathway0_stem.relu),
                ('layer01', model.model.s1.pathway0_stem),
                ('layer02', model.model.s2.pathway0_res0),
                ('layer03', model.model.s2.pathway0_res1),
                ('layer04', model.model.s2.pathway0_res2),
                ('layer05', model.model.s3.pathway0_res0),
                ('layer06', model.model.s3.pathway0_res1),
                ('layer07', model.model.s3.pathway0_res2),
                ('layer08', model.model.s3.pathway0_res3),
                ('layer09', model.model.s4.pathway0_res0),
                ('layer10', model.model.s4.pathway0_res1),
                ('layer11', model.model.s4.pathway0_res2),
                ('layer12', model.model.s4.pathway0_res3),
                ('layer13', model.model.s4.pathway0_res4),
                ('layer14', model.model.s4.pathway0_res5),
                ('layer15', model.model.s5.pathway0_res0),
                ('layer16', model.model.s5.pathway0_res1),
                ('layer17', model.model.s5.pathway0_res2),
            ])

        if args.subsample_layers:
            nums = [0, 1, 2, 4, 6, 8]
            l = []
            for i, (layer_name, layer) in enumerate(layers.items()):
                if i in nums:
                    l.append((layer_name, layer))

            layers = collections.OrderedDict(l)

        metadata = {'sz': 224,
                    'threed': True}
    elif args.features in ('ShiftNet'):
        model = ShiftNet(args)
        layers = model.layers

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features in ('MotionNet'):
        model = MotionNet(args)
        layers = collections.OrderedDict([
            model.relu,
            model.softmax
        ])

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features.startswith('ShallowMonkeyNet'):
        if 'pvc4' in args.features:
            # Load peach-wildflower-102
            # https://wandb.ai/pmin/crcns-train_net.py/runs/2l21idn1/overview?workspace=user-pmin
            path = os.path.join(args.ckpt_root, 'shallownet_symmetric_model.ckpt-1040000-2020-12-31 03-29-51.517721.pt')
        elif 'pvc1' in args.features:
            # This model was never saved because of a crash
            # From run Jan01_15-45-00_DESKTOP-I8HN3PB_pvc1_shallownet
            path = os.path.join(args.ckpt_root, 'model.ckpt-8700000-2021-01-03 22-34-02.540594.pt')
        else:
            raise NotImplementedError('Not implemented')
        checkpoint = torch.load(path)

        subnet_dict = extract_subnet_dict(checkpoint)

        model = ShallowNet(nstartfeats=subnet_dict['bn1.weight'].shape[0],
                           symmetric=subnet_dict['bn1.weight'].shape[0] > subnet_dict['conv1.weight'].shape[0])
        model.load_state_dict(subnet_dict)
        layers = model.layers

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features == 'V1Net':
        path = os.path.join(args.ckpt_root, 'model.ckpt-1259280-2021-01-10 17-09-33.770070.pt')
        checkpoint = torch.load(path)

        subnet_dict = extract_subnet_dict(checkpoint)

        model = V1Net()
        model.load_state_dict(subnet_dict)
        layers = model.layers

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features == 'dorsalnet':
        path = os.path.join(args.ckpt_root, 'dorsalnet-model.ckpt-1360000-2021-01-21 13-24-29.132420.pt')
        checkpoint = torch.load(path)

        subnet_dict = extract_subnet_dict(checkpoint)

        model = DorsalNet()
        model.load_state_dict(subnet_dict)
        layers = model.layers

        metadata = {'sz': 112,
                    'threed': True}

    else:
        raise NotImplementedError('Model not implemented yet')

    for key, layer in layers.items():
        layer.register_forward_hook(hook(key))

    metadata['layers'] = layers
    
    # Put model in eval mode (for batch_norm, dropout, etc.)
    model.eval()
    return model, activations, metadata

def extract_subnet_dict(d):
    out = {}
    for k, v in d.items():
        if k.startswith('subnet.'):
            out[k[7:]] = v

    return out