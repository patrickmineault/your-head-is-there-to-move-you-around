from modelzoo import xception, separable_net, gabor_pyramid
from loaders import pvc1, pvc2, pvc4

import argparse
import datetime
import itertools
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

import wandb

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

def get_dataset(args):
    if args.dataset == 'pvc1':
        trainset = pvc1.PVC1(os.path.join(args.data_root, 'crcns-ringach-data'), 
                                    split='train', 
                                    nt=32, 
                                    ntau=9, 
                                    nframedelay=0)

        tuneset = pvc1.PVC1(os.path.join(args.data_root, 'crcns-ringach-data'), 
                                split='tune', 
                                nt=32,
                                ntau=9,
                                nframedelay=0)

        transform = transforms.RandomCrop(223)
        sz = 223
    elif args.dataset == 'pvc2':
        trainset = pvc2.PVC2(os.path.join(args.data_root, 'pvc2'), 
                                    split='train', 
                                    nt=32, 
                                    ntau=9, 
                                    nframedelay=0)

        tuneset = pvc2.PVC2(os.path.join(args.data_root, 'pvc2'), 
                                split='tune', 
                                nt=32,
                                ntau=9,
                                nframedelay=0)
        sz = 12
    elif args.dataset == 'pvc4':
        trainset = pvc4.PVC4(os.path.join(args.data_root, 'crcns-pvc4'), 
                                    split='train', 
                                    nt=32, 
                                    nx=65,
                                    ny=65,
                                    ntau=9, 
                                    nframedelay=0,
                                    single_cell=args.single_cell)

        tuneset = pvc4.PVC4(os.path.join(args.data_root, 'crcns-pvc4'), 
                                split='tune', 
                                nt=32,
                                nx=65,
                                ny=65,
                                ntau=9,
                                nframedelay=0,
                                single_cell=args.single_cell)

        transform = lambda x: x
        sz = 30
    return trainset, tuneset, transform, sz


def constraints(net, mask):
    return 10 * (
        F.relu(abs(net.sampler.wx) - 1) ** 2 + 
        F.relu(abs(net.sampler.wy) - 1) ** 2 + 
        F.relu(net.sampler.wsigmax - 1) ** 2 + 
        F.relu(net.sampler.wsigmay - 1) ** 2
    ).sum() + .001 * (
        abs(net.sampler.wx[mask]) + 
        abs(net.sampler.wy[mask]) + 
        abs(net.sampler.wsigmax[mask]) + 
        abs(net.sampler.wsigmay[mask])
    ).sum()


def log_net(net, layers, writer, n):
    for name, layer in layers:
        if hasattr(layer, 'weight'):
            writer.add_scalar(f'Weights/{name}/mean', 
                            layer.weight.mean(), n)
            writer.add_scalar(f'Weights/{name}/std', 
                            layer.weight.std(), n)
            writer.add_histogram(f'Weights/{name}/hist', 
                            layer.weight.view(-1), n)

    for name, param in net.sampler._parameters.items():
        writer.add_scalar(f'Weights/{name}/mean', 
                        param.mean(), n)
        writer.add_scalar(f'Weights/{name}/std', 
                        param.std(), n)
        writer.add_histogram(f'Weights/{name}/hist', 
                        param.view(-1), n)

    for name, param in net._parameters.items():
        writer.add_scalar(f'Weights/{name}/mean', 
                        param.mean(), n)
        writer.add_scalar(f'Weights/{name}/std', 
                        param.std(), n)
        writer.add_histogram(f'Weights/{name}/hist', 
                        param.view(-1), n)

    if hasattr(net.subnet, 'conv1'):
        writer.add_images('Weights/conv1d/img', 
                        .25*net.subnet.conv1.weight + .5, n)

    # Plot the positions of the receptive fields
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for i in range(net.ntargets):
        ellipse = Ellipse((net.sampler.wx[i].item(), net.sampler.wy[i].item()), 
                        width=2.35*(.1 + F.relu(net.sampler.wsigmax[i]).item()),
                        height=2.35*(.1 + F.relu(net.sampler.wsigmay[i]).item()),
                        facecolor='none',
                        edgecolor=[0, 0, 0, .5]
                        )
        ax.add_patch(ellipse)
        ax.text(net.sampler.wx[i].item() + .05, net.sampler.wy[i].item(), str(i))


    ax.plot(net.sampler.wx.cpu().detach().numpy(), net.sampler.wy.cpu().detach().numpy(), 'r.')
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((1.1, -1.1))

    writer.add_figure('RF', fig, n)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(net.wt.cpu().detach().numpy())
    writer.add_figure('wt', fig, n)


def compute_corr(Yl, Yp):
    corr = torch.zeros(Yl.shape[1], device=Yl.device)
    for i in range(Yl.shape[1]):
        yl, yp = (Yl[:, i].cpu().detach().numpy(), 
                  Yp[:, i].cpu().detach().numpy())
        yl = yl[~np.isnan(yl)]
        yp = yp[~np.isnan(yp)]
        corr[i] = np.corrcoef(yl, yp)[0, 1]
    return corr


def get_subnet(args):
    threed = False
    if args.submodel == 'xception2d':
        subnet = xception.Xception(start_kernel_size=7, 
                                   nblocks=args.num_blocks, 
                                   nstartfeats=args.nfeats)
    elif args.submodel == 'gaborpyramid2d':
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid(4),
            transforms.Normalize(2.2, 2.2)
        )
    elif args.submodel == 'gaborpyramid3d':
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid3d(4),
            transforms.Normalize(2.2, 2.2)
        )
        threed = True
    return subnet, threed


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
    writer.add_hparams(vars(args), {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("No CUDA! Sad!")

    trainset, tuneset, transform, sz = get_dataset(args)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=1, 
                                              shuffle=True,
                                              pin_memory=True)

    tuneloader = torch.utils.data.DataLoader(tuneset, 
                                             batch_size=1, 
                                             shuffle=True,
                                             pin_memory=True)

    tuneloader_iter = iter(tuneloader)

    print("Init models")
    
    subnet, threed = get_subnet(args)

    if args.load_conv1_weights:
        W = np.load(args.load_conv1_weights)
        subnet.conv1.weight.data = torch.tensor(W)
        
    subnet.to(device=device)
    net = separable_net.LowRankNet(subnet, 
                                   trainset.total_electrodes, 
                                   args.nfeats, 
                                   sz, 
                                   sz, 
                                   trainset.ntau,
                                   sample=(not args.no_sample), 
                                   threed=threed).to(device)


    net.to(device=device)

    # Load a baseline with pre-trained weights
    if args.load_ckpt != '':
        net.load_state_dict(
            torch.load(args.load_ckpt)
        )

    layers = get_all_layers(net)

    if args.single_cell == -1:
        # Make sure to scale things properly because of double counting.
        optimizer = optim.Adam([
            {'params': net.inner_parameters, 'lr': args.learning_rate / np.sqrt(trainset.total_electrodes)},
            {'params': net.sampler.parameters(), 'lr': args.learning_rate / np.sqrt(trainset.total_electrodes)},
            {'params': subnet.parameters(), 'lr': args.learning_rate},
        ])
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    
    net.requires_grad_(True)
    net.sampler.requires_grad_(False)
    subnet.requires_grad_(False)

    activations = {}
    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o
        return hook_fn

    if hasattr(net.subnet, 'relu'):
        net.subnet.relu.register_forward_hook(hook('relu'))

    m, n = 0, 0
    print_frequency = 100
    tune_loss = 0.0

    Yl = np.nan * torch.ones(100000, trainset.total_electrodes, device=device)
    Yp = np.nan * torch.ones_like(Yl)
    total_timesteps = torch.zeros(trainset.total_electrodes, device=device, dtype=torch.long)
    
    corr = torch.ones(0)
    try:
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                net.train()
                if n > args.warmup:
                    # Release the Kraken!
                    subnet.requires_grad_(True)
                    net.sampler.requires_grad_(True)

                # get the inputs; data is a list of [inputs, labels]
                X, M, w, labels = data
                X, M, w, labels = X.to(device), M.to(device), w.to(device), labels.to(device)                

                optimizer.zero_grad()

                # zero the parameter gradients
                X = transform(X)
                outputs = net((X, M))

                mask = torch.any(M, dim=0)
                M = M[:, mask]

                # Add some soft constraints
                if n > args.warmup:
                    loss += constraints(net, mask)

                # masked mean squared error
                loss = w[:, mask] * ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels[:, mask, :])) ** 2).sum() / M.sum() / labels.shape[-1]
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
                    log_net(net, layers, writer, n)
                    print('[%02d, %04d] average train loss: %.3f' % (epoch + 1, i + 1, running_loss / print_frequency ))
                    running_loss = 0
                    
                    if 'xception' in args.submodel:
                        the_max = abs(activations['relu']).max()
                        the_X_max = abs(X).max()
                        writer.add_images('Activations/input', 
                                        .5 * X.squeeze() / the_X_max + .5, 
                                        n, dataformats='CNHW')
                        writer.add_images('Activations/relu', 
                                        .5 * activations['relu'].reshape(-1, 1, net.height_out, net.width_out) / the_max + .5, 
                                        n, dataformats='NCHW')
                if i % 10 == 0:
                    net.eval()
                    try:
                        tune_data = next(tuneloader_iter)
                    except StopIteration:
                        tuneloader_iter = iter(tuneloader)
                        tune_data = next(tuneloader_iter)

                        if n > 0:
                            corr = compute_corr(Yl, Yp)
                            print(f'     --> mean tune corr: {corr.mean():.3f}')
                            writer.add_histogram('tune/corr', corr, n)
                            writer.add_scalar('tune/corr', corr.mean(), n)
                            
                        Yl[:, :] = np.nan
                        Yp[:, :] = np.nan
                        total_timesteps *= 0
                    
                    # get the inputs; data is a list of [inputs, labels]
                    with torch.no_grad():
                        X, M, _, labels = tune_data
                        X, M, labels = X.to(device), M.to(device), labels.to(device)

                        X = transform(X)
                        outputs = net((X, M))
                        mask = torch.any(M, dim=0)
                        M = M[:, mask]

                        nnz = torch.nonzero(mask)[0]
                        
                        for k, j in enumerate(nnz):
                            slc = slice(total_timesteps[j].item(), 
                                        total_timesteps[j].item()+labels.shape[2])
                            Yl[slc, j.item()] = labels[:, j, :]
                            Yp[slc, j.item()] = outputs[:, k, :]
                            total_timesteps[j.item()] += labels.shape[2]

                        loss = ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels[:, mask, :])) ** 2).sum() / M.sum() / labels.shape[-1]

                        writer.add_scalar('Loss/tune', loss.item(), n)

                        tune_loss += loss.item()
                    m += 1

                    if m == print_frequency:
                        print(f"tune accuracy: {tune_loss /  print_frequency:.3f}")
                        tune_loss = 0
                        m = 0

                n += 1

                if n % args.ckpt_frequency == 0:
                    save_state(net, f'model.ckpt-{n:07}', output_dir)
                    
    except KeyboardInterrupt:
        pass

    filename = save_state(net, f'model.ckpt-{n:07}', output_dir)

    if args.no_wandb:
        print("Skipping W&B per config")
    else:
        if n > 10000:
            print("Saving to weight and biases")
            wandb.init(project="crcns-train_net.py", 
                    config=vars(args))
            config = wandb.config
            corr = corr.cpu().detach().numpy()
            corr = corr[~np.isnan(corr)]
            wandb.log({"tune_corr": corr})
            wandb.watch(net, log="all")
            torch.save(net.state_dict(), 
                    os.path.join(wandb.run.dir, 'model.pt'))
            print("done")
        else:
            print("Aborted too early, did not save results")

if __name__ == "__main__":
    desc = "Train a neural net"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--exp_name", required=True, help='Friendly name of experiment')
    
    parser.add_argument("--submodel", default='xception2d', type=str, help='Sub-model type (currently, either xception2d, gaborpyramid2d, gaborpyramid3d')
    parser.add_argument("--learning_rate", default=5e-3, type=float, help='Learning rate')
    parser.add_argument("--num_epochs", default=20, type=int, help='Number of epochs to train')
    parser.add_argument("--nfeats", default=64, type=int, help='Number of features')
    parser.add_argument("--num_blocks", default=0, type=int, help="Num Xception blocks")
    parser.add_argument("--warmup", default=5000, type=int, help="Number of iterations before unlocking tuning RFs and filters")
    parser.add_argument("--single_cell", default=-1, type=int, help="Fit data to a single cell with this index if true")
    parser.add_argument("--ckpt_frequency", default=2500, type=int, help="Checkpoint frequency")

    parser.add_argument("--no_sample", default=False, help='Whether to use a normal gaussian layer rather than a sampled one', action='store_true')
    parser.add_argument("--no_wandb", default=False, help='Skip using W&B', action='store_true')
    
    parser.add_argument("--load_conv1_weights", default='', help="Load conv1 weights in .npy format")
    parser.add_argument("--load_ckpt", default='', help="Load checkpoint")
    parser.add_argument("--dataset", default='pvc4', help='Dataset (currently pvc1, pvc2 or pvc4)')
    parser.add_argument("--data_root", default='./data', help='Data path')
    parser.add_argument("--output_dir", default='./models', help='Output path for models')
    
    args = parser.parse_args()
    main(args)
