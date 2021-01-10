from modelzoo import xception, separable_net, gabor_pyramid, monkeynet
from loaders import pvc1, pvc2, pvc4, mt2

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

        transform = lambda x: x
        sz = 112
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
                                    nx=args.image_size,
                                    ny=args.image_size,
                                    ntau=10, 
                                    nframedelay=0,
                                    single_cell=args.single_cell)

        tuneset = pvc4.PVC4(os.path.join(args.data_root, 'crcns-pvc4'), 
                                split='tune', 
                                nt=32,
                                nx=args.image_size,
                                ny=args.image_size,
                                ntau=10,
                                nframedelay=0,
                                single_cell=args.single_cell)
        transform = lambda x:x
        sz = args.image_size
    elif args.dataset == 'mt2':
        trainset = mt2.MT2(os.path.join(args.data_root, 'crcns-mt2'), 
                                    split='train', 
                                    nt=32, 
                                    nx=args.image_size,
                                    ny=args.image_size,
                                    ntau=10, 
                                    nframedelay=1,
                                    single_cell=args.single_cell)

        tuneset = mt2.MT2(os.path.join(args.data_root, 'crcns-mt2'), 
                                split='tune', 
                                nt=32,
                                nx=args.image_size,
                                ny=args.image_size,
                                ntau=10,
                                nframedelay=1,
                                single_cell=args.single_cell)
        transform = lambda x:x
        sz = args.image_size
    elif args.dataset == 'v2':
        trainset = pvc4.PVC4(os.path.join(args.data_root, 'crcns-v2'), 
                                    split='train', 
                                    nt=32, 
                                    nx=args.image_size,
                                    ny=args.image_size,
                                    ntau=10, 
                                    nframedelay=0,
                                    single_cell=args.single_cell)

        tuneset = pvc4.PVC4(os.path.join(args.data_root, 'crcns-v2'), 
                                split='tune', 
                                nt=32,
                                nx=args.image_size,
                                ny=args.image_size,
                                ntau=10,
                                nframedelay=0,
                                single_cell=args.single_cell)
        transform = lambda x:x
        sz = args.image_size
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

        if hasattr(layer, 'bias') and layer.bias is not None:
            writer.add_scalar(f'Biases/{name}/mean', 
                            layer.bias.mean(), n)
            writer.add_histogram(f'Biases/{name}/hist', 
                            layer.bias.view(-1), n)


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
        # NCHW
        if net.subnet.conv1.weight.ndim == 4:
            writer.add_images('Weights/conv1d/img', 
                            .25*net.subnet.conv1.weight + .5, n)
        else:
            # NTCHW
            scale = .5 / abs(net.subnet.conv1.weight).max()
            writer.add_video('Weights/conv1d/img', 
                            scale * net.subnet.conv1.weight.permute(0, 2, 1, 3, 4) + .5, 
                            n)

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


def get_subnet(args, start_size):
    threed = False
    if args.submodel == 'xception2d':
        subnet = xception.Xception(start_kernel_size=7, 
                                   nblocks=args.num_blocks, 
                                   nstartfeats=args.nfeats)
        sz = start_size // 2
        nfeats = args.nfeats
    if args.submodel.startswith('shallownet'):
        symmetric = 'symmetric' in args.submodel
        subnet = monkeynet.ShallowNet(nstartfeats=args.nfeats,
                                      symmetric=symmetric)
        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        nfeats = args.nfeats
    if args.submodel.startswith('v1net'):
        subnet = monkeynet.V1Net()
        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        nfeats = args.nfeats
    elif args.submodel == 'gaborpyramid2d':
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid(4),
            transforms.Normalize(2.2, 2.2)
        )
        sz = start_size // 2
        nfeats = args.nfeats
    elif args.submodel == 'gaborpyramid3d':
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid3d(4),
            transforms.Normalize(2.2, 2.2)
        )
        threed = True
        sz = start_size
        nfeats = args.nfeats
    elif args.submodel == 'gaborpyramid3d_tiny':
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid3d(2),
            transforms.Normalize(2.2, 2.2)
        )
        threed = True
        sz = start_size
        nfeats = args.nfeats
    return subnet, threed, sz, nfeats


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

    trainset, tuneset, transform, start_sz = get_dataset(args)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True,
                                              pin_memory=True)

    tuneloader = torch.utils.data.DataLoader(tuneset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True,
                                             pin_memory=True)

    tuneloader_iter = iter(tuneloader)

    print("Init models")
    
    subnet, threed, sz, nfeats = get_subnet(args, start_sz)

    if args.load_conv1_weights:
        W = np.load(args.load_conv1_weights)
        subnet.conv1.weight.data = torch.tensor(W)
        
    subnet.to(device=device)
    net = separable_net.LowRankNet(subnet, 
                                   trainset.total_electrodes, 
                                   nfeats, 
                                   sz, 
                                   sz, 
                                   trainset.ntau,
                                   sample=(not args.no_sample), 
                                   threed=threed,
                                   output_nl='relu').to(device)


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
            {'params': net.inner_parameters, 'lr': args.learning_rate_outer / np.sqrt(trainset.total_electrodes)},
            {'params': net.sampler.parameters(), 'lr': args.learning_rate_outer / np.sqrt(trainset.total_electrodes)},
            {'params': subnet.parameters(), 'lr': args.learning_rate},
        ])

        # Use a ramp-up for sensitive components like BN.
        def ramp_up_one(epoch): 
            alpha = min(max(epoch - args.warmup / args.batch_size, 0.0) / (args.warmup / args.batch_size), 1.0)
            return alpha

        # Ramp up the subnet (features) after the exterior components.
        def ramp_up_two(epoch): 
            alpha = min(max(epoch - 2 * args.warmup / args.batch_size, 0.0) / (args.warmup / args.batch_size), 1.0)
            return alpha

        lambdas = [
            lambda epoch: 1.0,
            ramp_up_one,
            ramp_up_two,
        ]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        scheduler = None

    activations = {}
    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o
        return hook_fn

    if hasattr(net.subnet, 'layers'):
        # Hook the activations
        for name, layer in net.subnet.layers:
            layer.register_forward_hook(hook(name))

    net.requires_grad_(True)
    subnet.requires_grad_(True)
    net.sampler.requires_grad_(True)

    ll, m, n = 0, 0, 0    
    tune_loss = 0.0

    Yl = np.nan * torch.ones(100000, trainset.total_electrodes, device=device)
    Yp = np.nan * torch.ones_like(Yl)
    total_timesteps = torch.zeros(trainset.total_electrodes, device=device, dtype=torch.long)
    
    corr = torch.ones(0)
    running_loss = 0.0
    try:
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            for data in trainloader:
                net.train()

                # get the inputs; data is a list of [inputs, labels]
                X, M, w, labels = data
                X, M, w, labels = X.to(device), M.to(device), w.to(device), labels.to(device)                

                optimizer.zero_grad()

                # zero the parameter gradients
                X = transform(X)
                outputs = net((X, M))
                outputs = outputs.permute(0, 2, 1)

                mask = torch.any(M, dim=0)
                M = M[:, mask]

                labels = labels[:, mask, :]

                assert tuple(outputs.shape) == tuple(labels.shape)

                sum_loss = (w[:, mask].view(-1, mask.sum(), 1) * (M.view(M.shape[0], M.shape[1], 1) * ((outputs - labels) ** 2))).sum()
                loss = sum_loss / M.sum() / labels.shape[-1]

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                label_mean = (M.view(M.shape[0], M.shape[1], 1) * labels).sum() / M.sum() / labels.shape[-1]
                output_mean = (M.view(M.shape[0], M.shape[1], 1) * outputs).sum() / M.sum() / labels.shape[-1]

                writer.add_scalar('Labels/mean', label_mean, n)
                writer.add_scalar('Outputs/mean', output_mean, n)
                writer.add_scalar('Loss/train', loss.item(), n)
                
                if ll % args.print_frequency == args.print_frequency - 1:
                    log_net(net, layers, writer, n)
                    print('[%02d, %07d] average train loss: %.3f' % (epoch + 1, n, running_loss / args.print_frequency ))
                    running_loss = 0
                    ll = 0
                    
                    if hasattr(net.subnet, 'layers'):
                        for name, layer in net.subnet.layers:
                            writer.add_histogram(f'Activations/{name}/hist', 
                                            activations[name].view(-1), 
                                            n)

                            writer.add_scalar(f'Activations/{name}/mean', 
                                            activations[name].mean(), 
                                            n)

                            writer.add_scalar(f'Activations/{name}/std', 
                                            activations[name].permute(1, 0, 2, 3, 4).reshape(activations[name].shape[1], -1).std(dim=1).mean(), 
                                            n)
                    
                if ll % 10 == 0:
                    net.eval()
                    try:
                        tune_data = next(tuneloader_iter)
                    except StopIteration:
                        tuneloader_iter = iter(tuneloader)
                        tune_data = next(tuneloader_iter)

                        if n > 0:
                            corr = compute_corr(Yl, Yp)
                            print(corr)
                            print(f'     --> mean tune corr: {corr.mean():.3f}')
                            writer.add_histogram('Tune/corr/hist', corr, n)
                            writer.add_scalar('Tune/corr/mean', corr.mean().item(), n)
                            
                        Yl[:, :] = np.nan
                        Yp[:, :] = np.nan
                        total_timesteps *= 0
                    
                    # get the inputs; data is a list of [inputs, labels]
                    with torch.no_grad():
                        X, M, w, labels = tune_data
                        X, M, w, labels = X.to(device), M.to(device), w.to(device), labels.to(device)

                        X = transform(X)
                        outputs = net((X, M))
                        outputs = outputs.permute(0, 2, 1)
                        mask = torch.any(M, dim=0)
                        M = M[:, mask]

                        nnz = torch.nonzero(mask).view(-1)
                        
                        for k, j in enumerate(nnz):
                            m_ = M[:, k].sum()
                            slc = slice(total_timesteps[j].item(), 
                                        total_timesteps[j].item() + m_ * labels.shape[2])
                            Yl[slc, j.item()] = labels[M[:, k], j, :].view(-1)
                            Yp[slc, j.item()] = outputs[M[:, k], k, :].view(-1)
                            total_timesteps[j.item()] += m_ * labels.shape[2]

                        sum_loss = ((M.view(M.shape[0], M.shape[1], 1) * ((outputs - labels[:, mask, :]) ** 2))).sum()
                        loss = sum_loss / M.sum() / labels.shape[-1]

                        writer.add_scalar('Loss/tune', loss.item(), n)

                        tune_loss += loss.item()
                    m += 1

                    if m == args.print_frequency:
                        print(f"tune accuracy: {tune_loss /  args.print_frequency:.3f}")
                        tune_loss = 0
                        m = 0

                if scheduler is not None:
                    scheduler.step()

                n += args.batch_size
                ll += 1

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
            wandb.log({"tune_corr": corr, "tune_corr_mean": corr.mean()})
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
    parser.add_argument("--learning_rate_outer", default=5e-3, type=float, help='Outer learning rate')
    parser.add_argument("--num_epochs", default=20, type=int, help='Number of epochs to train')
    parser.add_argument("--image_size", default=112, type=int, help='Image size')
    parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
    parser.add_argument("--nfeats", default=64, type=int, help='Number of features')
    parser.add_argument("--num_blocks", default=0, type=int, help="Num Xception blocks")
    parser.add_argument("--warmup", default=5000, type=int, help="Number of iterations before unlocking tuning RFs and filters")
    parser.add_argument("--single_cell", default=-1, type=int, help="Fit data to a single cell with this index if true")
    parser.add_argument("--ckpt_frequency", default=2500, type=int, help="Checkpoint frequency")
    parser.add_argument("--print_frequency", default=100, type=int, help="Print frequency")

    parser.add_argument("--no_sample", default=False, help='Whether to use a normal gaussian layer rather than a sampled one', action='store_true')
    parser.add_argument("--no_wandb", default=False, help='Skip using W&B', action='store_true')
    
    parser.add_argument("--load_conv1_weights", default='', help="Load conv1 weights in .npy format")
    parser.add_argument("--load_ckpt", default='', help="Load checkpoint")
    parser.add_argument("--dataset", default='pvc4', help='Dataset (currently pvc1, pvc2 or pvc4)')
    parser.add_argument("--data_root", default='./data_derived', help='Data path')
    parser.add_argument("--output_dir", default='./models', help='Output path for models')
    
    args = parser.parse_args()
    main(args)
