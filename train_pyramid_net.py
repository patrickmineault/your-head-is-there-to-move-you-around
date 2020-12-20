import gabor_pyramid
import separable_net
from loaders import pvc1, pvc2, pvc4
import xception

import datetime
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os


import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler


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
    torch.save(net.state_dict(), os.path.join(output_dir, f'{title}-{datestr}.pt'))


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

def compute_corr(Yl, Yp):
    corr = torch.zeros(Yl.shape[1], device=Yl.device)
    for i in range(Yl.shape[1]):
        yl, yp = (Yl[:, i].cpu().detach().numpy(), 
                  Yp[:, i].cpu().detach().numpy())
        yl = yl[~np.isnan(yl)]
        yp = yp[~np.isnan(yp)]
        corr[i] = np.corrcoef(yl, yp)[0, 1]
    return corr


def main(dataset='pvc1',
         experiment_name='',
         data_root='.',
         output_dir='/storage/trained/xception2d'):

    print("Main")
    # Train a network
    try:
        os.makedirs(data_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    
    writer = SummaryWriter(comment=experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("No CUDA! Sad!")

    if dataset == 'pvc1':
        trainset = pvc1.PVC1(os.path.join(data_root, 'crcns-ringach-data'), 
                                    split='train', 
                                    nt=32, 
                                    ntau=9, 
                                    nframedelay=0)

        tuneset = pvc1.PVC1(os.path.join(data_root, 'crcns-ringach-data'), 
                                split='tune', 
                                nt=32,
                                ntau=9,
                                nframedelay=0)
        transform = transforms.RandomCrop(223)
        sz = 223

    elif dataset == 'pvc2':
        trainset = pvc2.PVC2(os.path.join(data_root, 'pvc2'), 
                                    split='train', 
                                    nt=32, 
                                    ntau=9, 
                                    nframedelay=0)

        tuneset = pvc2.PVC2(os.path.join(data_root, 'pvc2'), 
                                split='tune', 
                                nt=32,
                                ntau=9,
                                nframedelay=0)
        sz = 12

    elif dataset == 'pvc4':
        nframedelay = 0
        trainset = pvc4.PVC4(os.path.join(data_root, 'crcns-pvc4'), 
                                    split='train', 
                                    nt=32, 
                                    nx=65,
                                    ny=65,
                                    ntau=9, 
                                    nframedelay=nframedelay)

        tuneset = pvc4.PVC4(os.path.join(data_root, 'crcns-pvc4'), 
                                split='tune', 
                                nt=32,
                                nx=65,
                                ny=65,
                                ntau=9,
                                nframedelay=nframedelay)

        transform = lambda x: x
        sz = 65

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

    subnet = nn.Sequential(
        gabor_pyramid.GaborPyramid(4),
        transforms.Normalize(2.2, 2.2)
    )

    subnet.to(device=device)

    net = separable_net.LowRankNet(subnet, 
                                   trainset.total_electrodes, 
                                   16, 
                                   sz, 
                                   sz, 
                                   trainset.ntau, 
                                   sampler_size=9).to(device)

    net.to(device=device)

    # Freeze the receptive fields
    net.sampler.requires_grad_(False)

    layers = get_all_layers(net)

    # optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=3e-3)

    m, n = 0, 0
    tune_loss = 0.0
    print_frequency = 25
    ckpt_frequency = 2000
    
    inner_opt = 1000

    Yl = np.nan * torch.ones(100000, trainset.total_electrodes, device=device)
    Yp = np.nan * torch.ones_like(Yl)
    total_timesteps = torch.zeros(trainset.total_electrodes, device=device, dtype=torch.long)

    try:
        for epoch in range(50):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                net.train()
                X, M, labels = data
                X, M, labels = X.to(device), M.to(device), labels.to(device)

                X = transform(X)

                if n > inner_opt:
                    # Turn on the heat.
                    net.sampler.requires_grad_(True)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = net((X, M))
                mask = torch.any(M, dim=0)
                M = M[:, mask]

                # masked mean squared error
                loss = ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels[:, mask, :])) ** 2).sum() / M.sum() / labels.shape[-1]

                # Add some soft constraints
                if n > inner_opt:
                    loss += constraints(net, mask)


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
                    for name, layer in layers:
                        if hasattr(layer, 'weight'):
                            writer.add_scalar(f'Weights/{name}/mean', 
                                            layer.weight.mean(), n)
                            writer.add_scalar(f'Weights/{name}/std', 
                                            layer.weight.std(), n)
                            writer.add_histogram(f'Weights/{name}/hist', 
                                            layer.weight.view(-1), n)

                    for name, param in net._parameters.items():
                        writer.add_scalar(f'Weights/{name}/mean', 
                                        param.mean(), n)
                        writer.add_scalar(f'Weights/{name}/std', 
                                        param.std(), n)
                        writer.add_histogram(f'Weights/{name}/hist', 
                                        param.view(-1), n)

                    for name, param in net.sampler._parameters.items():
                        writer.add_scalar(f'Weights/{name}/mean', 
                                        param.mean(), n)
                        writer.add_scalar(f'Weights/{name}/std', 
                                        param.std(), n)
                        writer.add_histogram(f'Weights/{name}/hist', 
                                        param.view(-1), n)

                    # writer.add_images('Weights/conv1d/img', subnet.conv1.weight, n)

                    print('[%d, %5d] average train loss: %.3f' % (epoch + 1, i + 1, running_loss / print_frequency ))
                    running_loss = 0

                    # Plot the positions of the receptive fields
                    fig = plt.figure(figsize=(6, 6))
                    ax = plt.gca()
                    for i in range(trainset.total_electrodes):
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

                if i % 10 == 0:
                    net.eval()
                    try:
                        tune_data = next(tuneloader_iter)
                    except StopIteration:
                        tuneloader_iter = iter(tuneloader)
                        tune_data = next(tuneloader_iter)

                        if n > 0:
                            r2 = compute_corr(Yl, Yp)
                            writer.add_histogram('tune/r2', r2, n)
                            
                        Yl = np.nan * torch.ones(100000, trainset.total_electrodes, device=device)
                        Yp = np.nan * torch.ones_like(Yl)
                        total_timesteps = torch.zeros(trainset.total_electrodes, device=device, dtype=torch.long)
                    
                    with torch.no_grad():
                        # get the inputs; data is a list of [inputs, labels]
                        X, M, labels = tune_data
                        X, M, labels = X.to(device), M.to(device), labels.to(device)

                        X = transform(X)

                        outputs = net((X, M))

                        mask = torch.any(M, dim=0)
                        labels = labels[:, mask, :]
                        M = M[:, mask]

                        nnz = torch.nonzero(mask)[0]
                        assert labels.shape[1] == 1
                        assert outputs.shape[1] == 1
                        
                        for k, j in enumerate(nnz):
                            slc = slice(total_timesteps[j].item(), 
                                        total_timesteps[j].item()+labels.shape[2])
                            Yl[slc, j.item()] = labels[:, k, :]
                            Yp[slc, j.item()] = outputs[:, k, :]
                            total_timesteps[j.item()] += labels.shape[2]

                        loss = ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels)) ** 2).sum() / M.sum() / labels.shape[-1]

                        writer.add_scalar('Loss/tune', loss.item(), n)

                        tune_loss += loss.item()
                        m += 1

                    if m == print_frequency:
                        print(f"tune accuracy: {tune_loss /  print_frequency:.3f}")
                        tune_loss = 0
                        m = 0

                n += 1

                if n % ckpt_frequency == 0:
                    save_state(net, f'xception.ckpt{n}', output_dir)
                    
    except KeyboardInterrupt:
        save_state(net, f'xception.ckpt{n}', output_dir)

if __name__ == "__main__":
    print("Getting into main")
    main('pvc4', 
         'pvc4_pyramid_stochastic_constrained', 
         'data/', 
         'models/pyramid')

