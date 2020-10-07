import xception
import separable_net
import pvc1_loader

import torch
from torch import nn
from torch import optim

if __name__ == "__main__":
    # Train a network
    trainset = pvc1_loader.PVC1(split='train', ntau=6)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    testset = pvc1_loader.PVC1(split='test', ntau=6)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    subnet = xception.Xception()
    net = separable_net.LowRankNet(subnet, 
                                   2, 
                                   trainset.total_electrodes, 128, 14, 14, trainset.ntau)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0