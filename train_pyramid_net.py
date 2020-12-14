import gabor_pyramid
import xception
import separable_net
import pvc1_loader

import datetime
import itertools
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch import nn
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
                                              batch_size=8, 
                                              shuffle=True,
                                              pin_memory=True)

    testset = pvc1_loader.PVC1(os.path.join(data_root, 'crcns-ringach-data'), 
                               split='tune', 
                               nt=32,
                               ntau=9,
                               nframedelay=0)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=8, 
                                             shuffle=True,
                                             pin_memory=True)
    testloader_iter = iter(testloader)

    print("Init models")

    subnet = nn.Sequential(
        gabor_pyramid.GaborPyramid(5),
        transforms.Normalize(2.2, 2.2)
    )
    subnet.to(device=device)

    net = separable_net.LowRankNet(subnet, 
                                   trainset.total_electrodes, 
                                   20, 
                                   223, 
                                   223, 
                                   trainset.ntau).to(device)

    rc = transforms.RandomCrop(223)

    net.to(device=device)

    layers = get_all_layers(net)

    # optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    m, n = 0, 0
    test_loss = 0.0
    print_frequency = 25
    ckpt_frequency = 2000
    
    try:
        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                X, M, labels = data
                X, M, labels = X.to(device), M.to(device), labels.to(device)

                X = rc(X)

                # zero the parameter gradients
                optimizer.zero_grad()
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

                    # writer.add_images('Weights/conv1d/img', subnet.conv1.weight, n)

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
                    ax.set_xlim((-.1, 1.1))
                    ax.set_ylim((1.1, -0.1))

                    writer.add_figure('RF', fig, n)

                if i % 10 == 0:
                    try:
                        test_data = next(testloader_iter)
                    except StopIteration:
                        testloader_iter = iter(testloader)
                        test_data = next(testloader_iter)
                    
                    # get the inputs; data is a list of [inputs, labels]
                    X, M, labels = test_data
                    X, M, labels = X.to(device), M.to(device), labels.to(device)

                    X = X[:, :, :, :-1, :-1]

                    outputs = net((X, M))
                    mask = torch.any(M, dim=0)
                    M = M[:, mask]
                    loss = ((M.view(M.shape[0], M.shape[1], 1) * (outputs - labels[:, mask, :])) ** 2).sum() / M.sum() / labels.shape[-1]

                    writer.add_scalar('Loss/test', loss.item(), n)

                    test_loss += loss.item()
                    m += 1

                    if m == print_frequency:
                        print(f"Test accuracy: {test_loss /  print_frequency:.3f}")
                        test_loss = 0
                        m = 0

                n += 1

                if n % ckpt_frequency == 0:
                    save_state(net, f'xception.ckpt{n}', output_dir)
                    
    except KeyboardInterrupt:
        save_state(net, f'xception.ckpt{n}', output_dir)

if __name__ == "__main__":
    print("Getting into main")
    main('.', 'models/pyramid')

