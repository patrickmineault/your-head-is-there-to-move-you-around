import os
import pickle
import requests
from tqdm import tqdm

import torch
from torch import nn


class MotionNet(nn.Module):
    """An PyTorch implementation of Rideaux & Welchman (2020).
    
    https://www.jneurosci.org/content/40/12/2538

    I saved their Tensorflow checkpoint as a pickle file to remove any 
    dependency on TF1.

    Note that I re-interpret their final readout layer as a convolutional layer
    by repeating the fully connected action all over space.
    """
    def __init__(self, args):
        super().__init__()

        # Now load the model.
        ckpt = 'motionnet.pkl'
        # TODO: automatically download the file if it doesn't exist.
        local_path = os.path.join(args.ckpt_root, ckpt)

        with open(local_path, 'rb') as f:
            results = pickle.load(f)

        self.conv1 = nn.Conv3d(1, 
                              128, 
                              (6, 6, 6),
                              (1, 1, 1),
                              padding=(3, 3, 3)
                             )

        self.conv1.weight.data = torch.tensor(
            results['wconv'].transpose((3, 2, 1, 0))
        ).unsqueeze(1)

        self.conv1.bias.data = torch.tensor(results['bconv'])

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(128,
                               64,
                               (1, 27, 27),
                               (1, 9, 9),
                               padding=(0, 17, 17))

        self.conv2.weight.data = torch.tensor(
            results['wout'].reshape(27, 27, 128, 64).transpose((3, 2, 1, 0))
        ).unsqueeze(2)

        self.conv2.bias.data = torch.tensor(results['bout'])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = X.mean(axis=1, keepdims=True)
        X = self.conv1(X)
        X = X[:, :, :-1, :, :]
        X = self.relu(X)
        X = self.conv2(X)
        X = self.softmax(X)

        return X


        