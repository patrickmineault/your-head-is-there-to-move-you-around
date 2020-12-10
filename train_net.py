import xception
import separable_net
import pvc1_loader

import torch
from torch import nn
from torch import optim

import os

from gradient_utils.metrics import MetricsLogger

def main(data_root='/storage/crcns/pvc1/', output_dir='/storage/trained/xception2d'):
    # Train a network
    try:
        os.makedirs(data_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    logger = MetricsLogger()
    logger.add_gauge("mse_train")
    logger.add_gauge("mse_test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("No CUDA! Sad!")

    print("Download data")
    pvc1_loader.download(data_root)

    print("Loading data")

    trainset = pvc1_loader.PVC1(os.path.join(data_root, 'crcns-ringach-data'), split='train', ntau=6)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    testset = pvc1_loader.PVC1(os.path.join(data_root, 'crcns-ringach-data'), split='test', ntau=6)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    print("Init models")
    subnet = xception.Xception()
    subnet.to(device=device)
    net = separable_net.LowRankNet(subnet, 
                                   2, 
                                   trainset.total_electrodes, 128, 14, 14, trainset.ntau)

    net.to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Fitting model")
    for epoch in range(1):  # loop over the dataset multiple times
        print(f"Epoch {epoch}")
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (X, rg), labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net((X.to(device=device), rg.to(device=device)))
            loss = criterion(outputs, labels.to(device=device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if epoch % 5 == 0:
            total_loss = 0
            for i, data in enumerate(testloader, 0):
                (X, rg), labels = data
                outputs = net((X.to(device=device), rg.to(device=device)))
                total_loss += criterion(outputs, labels.to(device=device))

            print(f"CV loss {total_loss:.2f}")

            logger['mse_test'] = total_loss
            logger.push_metrics()

            # After a full epoch, print out the status
            torch.save(net.state_dict(), output_dir + '_{epoch}.ckpt')

    torch.save(net.state_dict(), output_dir)

if __name__ == "__main__":
    print("Getting into main")
    main()