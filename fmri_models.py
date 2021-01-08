import collections
import git
import numpy as np
import os
import sklearn
import sklearn.decomposition
import sklearn.random_projection
import tables
from tqdm import tqdm

from loaders import vim2, pvc4
from modelzoo import gabor_pyramid, separable_net
from modelzoo.motionnet import MotionNet
from modelzoo.slowfast_wrapper import SlowFast
from modelzoo.shiftnet import ShiftNet
from modelzoo.monkeynet import ShallowNet

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

class Downsampler(nn.Module):
    def __init__(self, sz):
        super(Downsampler, self).__init__()
        self.sz = sz

    def forward(self, X):
        nt = 4
        assert X.shape[2] in (10, 20, 40, 80), "X.shape[2] must be 10 x a power of 2"
        big_pixel = X.shape[-1] // self.sz
        
        stride = X.shape[2] // (nt + 1)  # Always use at most 4 time points
        delta = (X.shape[2] - nt * stride) // 2
        slc = slice(delta, delta + stride * nt)
        X = X[:, :, slc, :, :].reshape(
            X.shape[0], X.shape[1], nt, stride, X.shape[-2], X.shape[-1]
        ).mean(3)
        X = downsample_3d(X, self.sz)
        assert X.shape[-1] == self.sz
        assert X.shape[2] == nt
        return X.reshape(X.shape[0], -1)

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

def preprocess_data(loader, 
                    model, 
                    aggregator, 
                    activations, 
                    metadata, 
                    args):

    # Check if cache exists for this model.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    cache_file = f'{args.features}_{args.width}_{args.dataset}_{args.subset}_{loader.dataset.split}_{args.aggregator}_{sha}.h5'
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
                X = resize(X, args.width)
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
    X = torch.tensor(h5file.get_node(f'/layer{args.layer}')[:], device='cpu')
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
    if args.dataset == 'vim2':
        nframedelay = -3
        data = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2/derived'), 
                                    split=fold, 
                                    nt=1, 
                                    ntau=80, 
                                    nframedelay=nframedelay,
                                    subject=args.subset)
    elif args.dataset == 'pvc4':
        data = pvc4.PVC4(os.path.join(args.data_root, 'crcns-pvc4'), 
                            split=fold, 
                             nt=1, 
                             nx=112,
                             ny=112,
                             ntau=10, 
                             nframedelay=0,
                             single_cell=int(args.subset))
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
        model = gabor_pyramid.GaborPyramid3d(nlevels=4, stride=(1, 1, 1))
        layers = [model]
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

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features in ('vgg19'):
        model = vgg19(pretrained=True)
        layers = [layer for layer in model.features if 
                  layer.__repr__().startswith('ReLU')]

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

        metadata = {'sz': 112,
                    'threed': False}
    elif args.features in ('SlowFast_Slow', 'SlowFast_Fast', 'Slow', 'I3D'):
        model = SlowFast(args)

        if args.features == 'SlowFast_Fast':
            layers = (
                model.model.s1.pathway1_stem.relu,
                model.model.s1.pathway1_stem,
                model.model.s2.pathway1_res0,
                model.model.s2.pathway1_res1,
                model.model.s2.pathway1_res2,
                model.model.s3.pathway1_res0,
                model.model.s3.pathway1_res1,
                model.model.s3.pathway1_res2,
                model.model.s3.pathway1_res3,
                model.model.s4.pathway1_res0,
                model.model.s4.pathway1_res1,
                model.model.s4.pathway1_res2,
                model.model.s4.pathway1_res3,
                model.model.s4.pathway1_res4,
                model.model.s4.pathway1_res5,
                model.model.s5.pathway1_res0,
                model.model.s5.pathway1_res1,
                model.model.s5.pathway1_res2,
            )
        else:
            layers = (
                model.model.s1.pathway0_stem.relu,
                model.model.s1.pathway0_stem,
                model.model.s2.pathway0_res0,
                model.model.s2.pathway0_res1,
                model.model.s2.pathway0_res2,
                model.model.s3.pathway0_res0,
                model.model.s3.pathway0_res1,
                model.model.s3.pathway0_res2,
                model.model.s3.pathway0_res3,
                model.model.s4.pathway0_res0,
                model.model.s4.pathway0_res1,
                model.model.s4.pathway0_res2,
                model.model.s4.pathway0_res3,
                model.model.s4.pathway0_res4,
                model.model.s4.pathway0_res5,
                model.model.s5.pathway0_res0,
                model.model.s5.pathway0_res1,
                model.model.s5.pathway0_res2,
            )

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features in ('ShiftNet'):
        model = ShiftNet(args)
        layers = model.layers

        metadata = {'sz': 112,
                    'threed': True}
    elif args.features in ('MotionNet'):
        model = MotionNet(args)
        layers = (
            model.relu,
            model.softmax
        )

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
        layers = [x for _, x in model.layers]

        metadata = {'sz': 112,
                    'threed': True}
    else:
        raise NotImplementedError('Model not implemented yet')

    for i, layer in enumerate(layers):
        layer.register_forward_hook(hook(i))

    metadata['nlayers'] = len(layers)
    
    # Put model in eval mode (for batch_norm, dropout, etc.)
    model.eval()
    return model, activations, metadata

def extract_subnet_dict(d):
    out = {}
    for k, v in d.items():
        if k.startswith('subnet.'):
            out[k[7:]] = v

    return out