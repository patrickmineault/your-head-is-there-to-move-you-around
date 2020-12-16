import xception
import separable_net
import pvc1_loader

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
    datuner = str(datetime.datetime.now()).replace(':', '-')
    torch.save(net.state_dict(), os.path.join(output_dir, f'{title}-{datuner}.pt'))

def main(data_root='/storage/crcns/pvc1/', 
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
    
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("No CUDA! Sad!")

    print("Download data")
    # pvc1_loader.download(data_root, 'https://storage.googleapis.com/vpl-bucket/')

    print("Loading data")

    trainset = pvc1_loader.PVC1(os.path.join(data_root, 'crcns-ringach-data'), 
                                split='train', 
                                nt=32, 
                                ntau=9, 
                                nframedelay=0)
                                
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=1, 
                                              shuffle=True,
                                              pin_memory=True)

    tuneset = pvc1_loader.PVC1(os.path.join(data_root, 'crcns-ringach-data'), 
                               split='tune', 
                               nt=32,
                               ntau=9,
                               nframedelay=0)
    tuneloader = torch.utils.data.DataLoader(tuneset, 
                                             batch_size=1, 
                                             shuffle=True,
                                             pin_memory=True)
    tuneloader_iter = iter(tuneloader)

    print("Init models")    

    nfeats = 32
    subnet = xception.Xception(start_kernel_size=7, 
                               nblocks=0, 
                               nstartfeats=nfeats)

    """
    resnet18 = models.resnet18(pretrained=True)
    subnet.conv1.weight.data = resnet18.conv1.weight.data
    subnet.to(device=device)
    """

    net = separable_net.LowRankNet(subnet, 
                                   trainset.total_electrodes, 
                                   nfeats, 
                                   53, 
                                   53, 
                                   trainset.ntau,
                                   9).to(device)

    def ds():
        downsample_filt = torch.tensor([[.25, .5, .25], [.5, 1.0, .5], [.25, .5, .25]]).view(1, 1, 3, 3).to(device=device)
        downsample_filt /= 4.0

        def d(X):
            return F.conv2d(X.reshape(-1, 1, X.shape[3], X.shape[4]), 
                            downsample_filt, 
                            stride=2).reshape(X.shape[0], 
                                X.shape[1], 
                                X.shape[2], 
                                (X.shape[3]-1)//2,
                                (X.shape[4]-1)//2)

        return d

    # Downsampling helps reduce compute and decreases the 
    rc = transforms.Compose([ds()])

    net.to(device=device)

    # Load a baseline with pre-trained weights
    """
    net.load_state_dict(
        torch.load('models/shallow/xception.ckpt-0007215-2020-12-15 00-02-15.355483.pt')
    )
    """

    layers = get_all_layers(net)

    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    subnet.requires_grad_(False)

    m, n = 0, 0
    print_frequency = 25
    tune_loss = 0.0
    ckpt_frequency = 2500
    
    try:
        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                net.train()
                if n > 1000:
                    # Time to turn on the heat.
                    subnet.requires_grad_(True)

                # get the inputs; data is a list of [inputs, labels]
                X, M, labels = data
                X, M, labels = X.to(device), M.to(device), labels.to(device)

                optimizer.zero_grad()

                # zero the parameter gradients
                X = rc(X)
                outputs = net((X, M))

                mask = torch.any(M, dim=0)
                M = M[:, mask]

                # masked mean squared error
                loss = ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels[:, mask, :])) ** 2).sum() / M.sum() / labels.shape[-1]
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

                    writer.add_images('Weights/conv1d/img', subnet.conv1.weight * .5 + .5, n)

                    print('[%d, %5d] average train loss: %.3f' % (epoch + 1, i + 1, running_loss / print_frequency ))
                    running_loss = 0

                    # Plot the positions of the receptive fields
                    fig = plt.figure(figsize=(6, 6))
                    ax = plt.gca()
                    for i in range(trainset.total_electrodes):
                        ellipse = Ellipse((net.wx[i].item(), net.wy[i].item()), 
                                        width=2.35*(.1 + abs(net.wsigmax[i].item())),
                                        height=2.35*(.1 + abs(net.wsigmay[i].item())),
                                        facecolor='none',
                                        edgecolor=[0, 0, 0, .5]
                                        )
                        ax.add_patch(ellipse)
                    ax.plot(net.wx.cpu().detach().numpy(), net.wy.cpu().detach().numpy(), 'r.')
                    ax.set_xlim((-1.1, 1.1))
                    ax.set_ylim((1.1, -1.1))

                    writer.add_figure('RF', fig, n)

                if i % 10 == 0:
                    net.eval()
                    try:
                        tune_data = next(tuneloader_iter)
                    except StopIteration:
                        tuneloader_iter = iter(tuneloader)
                        tune_data = next(tuneloader_iter)
                    
                    # get the inputs; data is a list of [inputs, labels]
                    with torch.no_grad():
                        X, M, labels = tune_data
                        X, M, labels = X.to(device), M.to(device), labels.to(device)

                        X = rc(X)
                        outputs = net((X, M))
                        mask = torch.any(M, dim=0)
                        M = M[:, mask]
                        loss = ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels[:, mask, :])) ** 2).sum() / M.sum() / labels.shape[-1]

                        writer.add_scalar('Loss/tune', loss.item(), n)

                        tune_loss += loss.item()
                    m += 1

                    if m == print_frequency:
                        print(f"tune accuracy: {tune_loss /  print_frequency:.3f}")
                        tune_loss = 0
                        m = 0

                n += 1

                if n % ckpt_frequency == 0:
                    save_state(net, f'xception.ckpt-{n:07}', output_dir)
                    
    except KeyboardInterrupt:
        save_state(net, f'xception.ckpt-{n:07}', output_dir)

if __name__ == "__main__":
    print("Getting into main")
    main('./data', 'models/shallow')

