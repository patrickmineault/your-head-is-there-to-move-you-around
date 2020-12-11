import xception
import separable_net
import pvc1_loader

import datetime
import itertools

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler

def save_state(net, title):
    datestr = str(datetime.datetime.now()).replace(':', '-')
    torch.save(net.state_dict(), f'models/{title}-{datestr}.pt')

if __name__ == "__main__":
    print_frequency = 1

    writer = SummaryWriter()

    # Train a network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = pvc1_loader.PVC1(split='train', ntau=6)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True)

    testset = pvc1_loader.PVC1(split='test', ntau=6)
    testloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True)

    testloader_iter = iter(testloader)
    subnet = xception.Xception()
    net = separable_net.LowRankNet(subnet, 
                                   2, 
                                   trainset.total_electrodes, 
                                   128, 
                                   14, 14, trainset.ntau).to(device)

    # Restart evaluation in this network
    net.load_state_dict(torch.load('models\\xception.ckpt90000-2020-10-08 09-31-19.917481.pt'))
    net.eval()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=3e-5, momentum=0.9)

    n = 0
    try:
        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                (inputs, neurons), labels = ((data[0][0].to(device), 
                                            data[0][1].to(device)),
                                            data[1].to(device))

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = net((inputs, neurons))

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                writer.add_scalar('Outputs/mean', outputs.mean(), n)
                writer.add_scalar('Outputs/std', outputs.std(), n)
                writer.add_scalar('Loss/train', loss.item(), n)
                
                if i % print_frequency == print_frequency - 1:
                    print('[%d, %5d] average train loss: %.3f' % (epoch + 1, i + 1, running_loss / print_frequency ))
                    running_loss = 0

                if i % 7 == 0:
                    try:
                        test_data = next(testloader_iter)
                    except StopIteration:
                        testloader_iter = iter(testloader)
                        test_data = next(testloader_iter)
                    
                    # get the inputs; data is a list of [inputs, labels]
                    (inputs, neurons), labels = ((test_data[0][0].to(device), 
                                                test_data[0][1].to(device)),
                                                test_data[1].to(device))
                    outputs = net((inputs, neurons))
                    loss = criterion(outputs, labels)
                    writer.add_scalar('Loss/test', loss.item(), n)       

                n += 1

                if n % 10000 == 0:
                    save_state(net, f'xception.ckpt{n}')
    except KeyboardInterrupt:
        save_state(net, f'xception.ckpt{n}')

        torch.save(net.state_dict(), f'models/xception.{str(datetime.datetime.now())}.pt'.replace(':', '-'))