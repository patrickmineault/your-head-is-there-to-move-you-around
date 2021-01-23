import collections
import numpy as np
import os
import pickle
import requests
from tqdm import tqdm

import torch
from torch import nn

remote = "https://www.repository.cam.ac.uk/bitstream/handle/1810/300898/data_and_code.zip?sequence=1&isAllowed=y"

class Block3(nn.Module):
    def __init__(self, nf, dim, stride=1):
        super().__init__()
        if stride != 1:
            self.pool1 = nn.MaxPool3d([1, 1, 1], stride)
        else:
            self.pool1 = lambda x: x

        # Note that tfslim conv includes its own ReLU.
        self.conv1 = nn.Conv3d(nf, nf, dim, stride=stride, padding=(dim[0]//2, dim[1]//2, dim[2]//2), bias=True)
        self.conv1_relu = nn.ReLU()
        self.conv2 = nn.Conv3d(nf, nf, dim, padding=(dim[0]//2, dim[1]//2, dim[2]//2))
        self.bn = nn.BatchNorm3d(nf)
        self.relu = nn.ReLU()

    def forward(self, X):
        short = self.pool1(X)
        X = self.conv1(X)
        X = self.conv1_relu(X)
        X = self.conv2(X)
        X = X + short
        X = self.bn(X)
        return self.relu(X)

class ShiftNet(nn.Module):
    """A PyTorch implementation of Owens & Efros (2018).
    
    Audio-Visual Scene Analysis with Self-Supervised Multisensory Features.

    I saved their Tensorflow checkpoint as a pickle file to remove any 
    dependency on TF1.

    Note that I only implement the purely visual representations (conv1, conv2_1, conv2_2).
    """
    def __init__(self, args):
        super().__init__()

        # Now load the model.
        #ckpt = 'shiftnet.pkl'
        #local_path = os.path.join(args.ckpt_root, ckpt)
        #if not os.path.exists(os.path.join(args.ckpt_root, ckpt)):
        #    util.download(remote, local_path)

        #with open(local_path, 'rb') as f:
        #    results = pickle.load(f)

        self.conv1 = nn.Conv3d(3, 
                              64, 
                              (5, 7, 7),
                              stride=2,
                              padding=(2, 3, 3),
                              bias=False
                             )
        self.conv1_relu = nn.ReLU()

        self.pool1 = nn.MaxPool3d([1, 3, 3], [1, 2, 2], [0, 1, 1])
        self.block1 = Block3(64, [3, 3, 3], stride=1)
        self.block2 = Block3(64, [3, 3, 3], stride=2)

        self.load_weights(args)

        self.layers = collections.OrderedDict([
                       ('conv1_relu', self.conv1_relu), 
                       ('pool1', self.pool1), 
                       ('block1_conv1_relu', self.block1.conv1_relu),
                       ('block1_relu', self.block1.relu),
                       ('block2_conv1_relu', self.block2.conv1_relu),
                       ('block2_relu', self.block2.relu)])

    def load_weights(self, args):
        ckpt = 'shiftnet.pkl'
        local_path = os.path.join(args.ckpt_root, ckpt)
        with open(local_path, 'rb') as f:
            results = pickle.load(f)

        _fill(results['im/conv1/weights'].transpose(4, 3, 0, 1, 2), self.conv1.weight)
        for prefix, block in [('im/conv2_1', self.block1),
                              ('im/conv2_2', self.block2)]:

            _fill(results[f'{prefix}_1/weights'].transpose(4, 3, 0, 1, 2), block.conv1.weight)
            _fill(results[f'{prefix}_2/weights'].transpose(4, 3, 0, 1, 2), block.conv2.weight)
            _fill(results[f'{prefix}_2/biases'], block.conv2.bias)
            _fill(results[f'{prefix}_bn/moving_mean'], block.bn.running_mean)
            _fill(results[f'{prefix}_bn/moving_variance'], block.bn.running_var)


    def forward(self, X):
        # Fudge factor for this using a [-1, 1] normalization, 
        # while we normalize to have a s.d. of 1
        X = X / 2 
        X = self.conv1(X)
        X = self.conv1_relu(X)
        X = self.pool1(X)
        X = self.block1(X)
        X = self.block2(X)

        return X

def _fill(np_array, recipient):
    data = torch.tensor(np_array)
    assert data.shape == recipient.data.shape
    recipient.data = data


        