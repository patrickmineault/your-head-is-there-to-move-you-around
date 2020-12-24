import argparse
import collections
import datetime
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os

from loaders import vim2
import gabor_pyramid
import separable_net

import torch
from torch import nn
import torchvision
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def get_all_layers(net, prefix=[]):
    if hasattr(net, '_modules'):
        lst = []
        for name, layer in net._modules.items():
            full_name = '_'.join((prefix + [name]))
            lst = lst + [(full_name, layer)] + get_all_layers(layer, prefix + [name])
        return lst
    else:
        return []

def save_state(net, title, output_dir):
    datestr = str(datetime.datetime.now()).replace(':', '-')
    filename = os.path.join(output_dir, f'{title}-{datestr}.pt')
    torch.save(net.state_dict(), filename)
    return filename

def get_feature_model(args):
    activations = collections.OrderedDict()
    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o
        return hook_fn

    if args.features == 'gaborpyramid3d':
        model = gabor_pyramid.GaborPyramid3d(nlevels=5, stride=(15, 1, 1))
        model.register_forward_hook(hook('layer1'))
        sz = 112
        threed = True
    else:
        raise NotImplementedError('Model not implemented yet')

    return model, activations, sz, threed


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


class Passthrough(nn.Module):
    def __init__(self):
        super(Passthrough, self).__init__()

    def forward(self, data):
        return (data - 5) / 5


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
        return (data - 5) / 5


def log_net(net, layers, writer, n):
    for name, layer in layers:
        if hasattr(layer, 'weight'):
            writer.add_scalar(f'Weights/{name}/mean', 
                            layer.weight.mean(), n)
            writer.add_scalar(f'Weights/{name}/std', 
                            layer.weight.std(), n)
            writer.add_histogram(f'Weights/{name}/hist', 
                            layer.weight.view(-1), n)

    if hasattr(net, 'sampler'):
        for name, param in net.sampler._parameters.items():
            writer.add_scalar(f'Weights/{name}/mean', 
                            param.mean(), n)
            writer.add_scalar(f'Weights/{name}/std', 
                            param.std(), n)
            writer.add_histogram(f'Weights/{name}/hist', 
                            param.view(-1), n)

        # Plot the positions of the receptive fields
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        for i in range(0, net.ntargets, 100):
            ellipse = Ellipse((net.sampler.wx[i].item(), net.sampler.wy[i].item()), 
                            width=2.35*(.1 + F.relu(net.sampler.wsigmax[i]).item()),
                            height=2.35*(.1 + F.relu(net.sampler.wsigmay[i]).item()),
                            facecolor='none',
                            edgecolor=[0, 0, 0, .5]
                            )
            ax.add_patch(ellipse)
            ax.text(net.sampler.wx[i].item() + .05, net.sampler.wy[i].item(), str(i))


        ax.plot(net.sampler.wx[::100].cpu().detach().numpy(), 
                net.sampler.wy[::100].cpu().detach().numpy(), 'r.')
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((1.1, -1.1))

        writer.add_figure('RF', fig, n)

    for name, param in net._parameters.items():
        writer.add_scalar(f'Weights/{name}/mean', 
                        param.mean(), n)
        writer.add_scalar(f'Weights/{name}/std', 
                        param.std(), n)
        writer.add_histogram(f'Weights/{name}/hist', 
                        param.view(-1), n)

    if hasattr(net, 'wt'):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(net.wt[:, ::100].cpu().detach().numpy())
        writer.add_figure('wt', fig, n)

        fig = plt.figure(figsize=(6, 4))
        plt.plot(abs(net.wt).mean(axis=1).cpu().detach().numpy())
        writer.add_figure('abs_wt', fig, n)


def compute_corr(Yl, Yp):
    corr = torch.zeros(Yl.shape[1], device=Yl.device)
    for i in range(Yl.shape[1]):
        yl, yp = (Yl[:, i].cpu().detach().numpy(), 
                  Yp[:, i].cpu().detach().numpy())
        yl = yl[~np.isnan(yl)]
        yp = yp[~np.isnan(yp)]
        corr[i] = np.corrcoef(yl, yp)[0, 1]
    return corr


def main(args):
    print("Main")
    output_dir = os.path.join(args.output_dir, args.exp_name)
    # Train a network
    try:
        os.makedirs(args.data_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    
    
    writer = SummaryWriter(comment=args.exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("No CUDA! Sad!")

    nframedelay = -3
    trainset = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2'), 
                                split='train', 
                                nt=1, 
                                nx=112,
                                ny=112,
                                ntau=5, 
                                nframedelay=nframedelay,
                                subset=args.subset)

    tuneset = vim2.Vim2(os.path.join(args.data_root, 'crcns-vim2'), 
                            split='report', 
                            nt=1,
                            nx=112,
                            ny=112,
                            ntau=5,
                            nframedelay=nframedelay,
                            subset=args.subset)

    feature_model, activations, sz, threed = get_feature_model(args)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True,
                                              pin_memory=True
                                              )

    tuneloader = torch.utils.data.DataLoader(tuneset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True,
                                             pin_memory=True
                                             )

    tuneloader_iter = iter(tuneloader)

    feature_model.to(device=device)
    net, subnet = get_readout_model(args, threed, trainset)
    subnet.to(device=device)
    net.to(device=device)
    net.wb.data *= 0

    layers = get_all_layers(net)
    
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    #optimizer = optim.SGD(net.parameters(), 
    #                      lr=args.learning_rate, momentum=.9)

    print_frequency = 25

    # Keep this off of the GPU, since it takes so much memory.
    Yl = np.nan * torch.ones(len(tuneset) * tuneset.nt, trainset.total_electrodes)
    Yp = np.nan * torch.ones_like(Yl)
    total_timesteps = 0
    
    corr = torch.ones(0)
    m, n = 0, 0
    tune_loss = 0
    base_loss = 0

    try:
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                X, labels = data
                X, labels = X.to(device), labels.to(device)

                with torch.no_grad():
                    _ = feature_model(X)
                    fit_layer = activations[args.target_layer]
                
                assert fit_layer.ndim == 5
                fit_layer = fit_layer[:, :, 1:, :, :]
                optimizer.zero_grad()

                # zero the parameter gradients
                M = torch.ones(fit_layer.shape[0], trainset.total_electrodes, 
                               device=device, 
                               dtype=torch.bool)
                outputs = net((fit_layer, M))
                outputs = outputs.permute(0, 2, 1)

                #print(outputs.shape)
                #print(labels.shape)
                # masked mean squared error
                assert tuple(outputs.shape) == tuple(labels.shape)
                loss = ((outputs - labels) ** 2).mean()
                loss.backward()

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                writer.add_scalar('Labels/mean', labels.mean(), n)
                writer.add_scalar('Labels/std', labels.std(), n)
                writer.add_scalar('Outputs/mean', outputs.mean(), n)
                writer.add_scalar('Outputs/std', outputs.std(), n)
                writer.add_scalar('Loss/train', loss.item(), n)
                
                if i % print_frequency == print_frequency - 1:
                    # log_net(net, layers, writer, n)
                    print('[%02d, %04d] average train loss: %.3f' % (epoch + 1, i + 1, running_loss / print_frequency ))
                    running_loss = 0

                if i % 100 == 0:
                    log_net(net, layers, writer, n)

                if i % 4 == 0:
                    net.eval()
                    try:
                        tune_data = next(tuneloader_iter)
                    except StopIteration:
                        tuneloader_iter = iter(tuneloader)
                        tune_data = next(tuneloader_iter)

                        if n > 0:
                            corr = compute_corr(Yl, Yp)
                            print(f'     --> mean tune corr: {corr.mean():.3f}')
                            print(corr.max())
                            print(corr.median())
                            writer.add_histogram('tune/corr', corr, n)
                            writer.add_scalar('tune/corr', corr.mean(), n)
                            
                        Yl[:, :] = np.nan
                        Yp[:, :] = np.nan
                        total_timesteps = 0
                    
                    # get the inputs; data is a list of [inputs, labels]
                    with torch.no_grad():
                        X, labels = tune_data
                        X, labels = X.to(device), labels.to(device)

                        _ = feature_model(X)
                        fit_layer = activations[args.target_layer]
                        fit_layer = fit_layer[:, :, 1:, :, :]

                        M = torch.ones(fit_layer.shape[0], trainset.total_electrodes, device=device, dtype=torch.bool)
                        outputs = net((fit_layer, M))
                        outputs = outputs.permute(0, 2, 1)
                        assert tuple(outputs.shape) == tuple(labels.shape)
                        
                        slc = slice(total_timesteps, 
                                    total_timesteps + labels.shape[0] * labels.shape[1])

                        Yl[slc, :] = labels.cpu().detach().squeeze()
                        Yp[slc, :] = outputs.cpu().detach().squeeze()
                        total_timesteps += labels.shape[0] * labels.shape[1]

                        loss = ((outputs - labels) ** 2).mean()

                        writer.add_scalar('Loss/tune', loss.item(), n)

                        tune_loss += loss.item()
                        base_loss += (labels ** 2).mean().item()
                    m += 1

                    if m == print_frequency:
                        print(f"tune accuracy: {tune_loss /  print_frequency:.3f}")
                        print(f"tune base accuracy: {base_loss /  print_frequency:.3f}")
                        tune_loss = 0
                        base_loss = 0
                        m = 0

                n += 1

                if n % args.ckpt_frequency == 0:
                    save_state(net, f'model.ckpt-{n:07}', output_dir)
                    
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    desc = "Train a neural net"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--exp_name", required=True, help='Friendly name of experiment')
    
    parser.add_argument("--readout", default='gaussian', type=str, help='Readout model (either average or sampler)')
    parser.add_argument("--features", default='gaborpyramid3d', type=str, help='What kind of features to use')
    parser.add_argument("--target_layer", default='layer1', type=str, help='Which layer to use as features')
    parser.add_argument("--learning_rate", default=5e-3, type=float, help='Learning rate')
    parser.add_argument("--num_epochs", default=20, type=int, help='Number of epochs to train')
    parser.add_argument("--nfeats", default=64, type=int, help='Number of features')
    parser.add_argument("--num_blocks", default=0, type=int, help="Num Xception blocks")
    parser.add_argument("--warmup", default=5000, type=int, help="Number of iterations before unlocking tuning RFs and filters")
    parser.add_argument("--single_cell", default=-1, type=int, help="Fit data to a single cell with this index if true")
    parser.add_argument("--ckpt_frequency", default=2500, type=int, help="Checkpoint frequency")
    parser.add_argument("--batch_size", default=4, type=int, help="Back size")

    parser.add_argument("--no_sample", default=False, help='Whether to use a normal gaussian layer rather than a sampled one', action='store_true')
    parser.add_argument("--no_wandb", default=False, help='Skip using W&B', action='store_true')
    parser.add_argument("--subset", default=False, help='Use a subset of the data', action='store_true')
    
    parser.add_argument("--load_conv1_weights", default='', help="Load conv1 weights in .npy format")
    parser.add_argument("--load_ckpt", default='', help="Load checkpoint")
    parser.add_argument("--dataset", default='pvc4', help='Dataset (currently pvc1, pvc2 or pvc4)')
    parser.add_argument("--data_root", default='./data', help='Data path')
    parser.add_argument("--output_dir", default='./models', help='Output path for models')
    
    args = parser.parse_args()
    main(args)