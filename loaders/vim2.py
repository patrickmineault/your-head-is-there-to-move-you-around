from . import utils

import collections
import datetime
import glob
import hashlib
import matplotlib
import matplotlib.image
import numpy as np
import os
import pandas as pd
import requests
import struct
import subprocess
import tables, numpy
import time

import torch.nn.functional as F
import torch.utils.data

stim_t = None
stim_r = None
resp_t = None
resp_r = None
rois = {}

framerate = 15

class Vim2(torch.utils.data.Dataset):
    """
    Loads a segment from Gallant crcns vim-2 dataset.

    Each call to __get_item__ returns a tuple

    X: a numpy array with size (3, (nt + ntau) * 15 * ndelays, ny, nx)
    y: a numpy array with size (nt, nvoxels).
    """
    def __init__(self, 
                 root='./data/crcns-vim2',
                 nx=112,
                 ny=112,
                 nt=4,
                 ntau=1,
                 nframedelay=0,
                 split='train',
                 subject='s1',
                 subset=False,
                 ):

        block_len = 10  # in seconds

        if split not in ('traintune', 'train', 'tune', 'report'):
            raise NotImplementedError('Split is set to an unknown value')

        if subject not in ('s1', 's2', 's3'):
            raise NotImplementedError("Subject not s1, s2 or s3")

        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay
        self.subject = subject
        self.split = split
        self.framerate = 15
        self.sampling_freq = 16

        if 540 % nt != 0:
            raise NotImplementedError("nt must divide 540")

        sequences = []
        if subset:
            seq_end = 540
        else:
            seq_end = 540 if split == 'report' else 7200

        for i in range(0, seq_end, self.nt):
            which_bucket = i // block_len
            if (split == 'traintune' or
                (split == 'train' and (which_bucket % 10) != 4) or 
                (split == 'tune' and (which_bucket % 10) == 4) or
                (split == 'report')):
                sequences.append(
                    {'subject': self.subject, 
                    'split': split,
                    'stim_idx': np.fmin(
                                    np.fmax(
                                        np.arange((i + nframedelay + 1) * framerate - ntau, 
                                                  (i + nframedelay + nt) * framerate), 
                                    -1), 
                                framerate * seq_end - 1),
                    'resp_idx': np.arange(i, i + self.nt)
                    }
                )

        self.total_reponses = seq_end
        self.sequence = sequences

        # Use a lazy loading strategy
        self.root = root
        self.rois = self._get_rois()
        self.mask = self._get_master_mask()

        if subset:
            self.total_voxels = 10000
        else:
            self.total_voxels = 18*64*64

        self.mask = self.mask[:self.total_voxels]
        self.total_electrodes = self.mask.sum()

        assert self.total_electrodes > 0


    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        # Cache are module variables
        global stim_t, stim_r, resp_t, resp_r, rois
        tgt = self.sequence[idx]

        # Use a common cache for everyone
        if ((tgt['split'] in ('traintune', 'train', 'tune') and stim_t is None) or
            (tgt['split'] in ('report') and stim_r is None)):
            # Lazy load
            f = tables.open_file(
                os.path.join(self.root, 
                f'VoxelResponses_subject{self.subject[1]}.mat'))

            if tgt['split'] == 'report':
                resp_r = f.get_node('/rv')[:self.total_voxels].T
                resp_r = resp_r[:, self.mask]
            else:
                resp_t = f.get_node('/rt')[:self.total_voxels].T[:self.total_reponses, :]
                resp_t = resp_t[:, self.mask]

            f.close()

            f = tables.open_file(
                os.path.join(self.root, 'Stimuli.mat')
            )

            if tgt['split'] == 'report':
                stim_r = f.get_node('/sv')[:]
                stim_r[-1, :, :, :] = 85
            else:
                stim_t = f.get_node('/st')[:(self.total_reponses * framerate)]
                stim_t[-1, :, :, :] = 85

            f.close()
        
        if tgt['split'] == 'report':
            stim = stim_r
            resp = resp_r
        else:
            stim = stim_t
            resp = resp_t

        rgx = slice((stim.shape[-1] - self.nx) // 2,
                    (stim.shape[-1] - self.nx) // 2 + self.nx)
        rgy = slice((stim.shape[-1] - self.ny) // 2,
                    (stim.shape[-1] - self.ny) // 2 + self.ny)

        assert tgt['split'] == self.split

        X = stim[tgt['stim_idx'], :, rgy, rgx].transpose((1, 0, 2, 3))
        Y = resp[tgt['resp_idx'], :].astype(np.float32)

        X = (X.astype(np.float32) - 85) / 63

        return (X, Y)

    def _get_master_mask(self):
        mask = None

        valid_keys = ['v1lh', 'v1rh', 'v2lh', 'v2rh', 'v3alh', 'v3arh', 'v3blh', 
                      'v3brh', 'v3lh', 'v3rh', 'v4lh', 'v4rh', 
                      'MTlh', 'MTplh', 'MTprh', 'MTrh']

        for k, v in self.rois.items():
            if k not in valid_keys:
                continue

            if mask is None:
                mask = (v > 0)
            else:
                mask |= (v > 0)

        assert mask.shape[0] == 18
        mask[0, :, :] = False
        mask[:, 0, :] = False
        mask[:, :, :2] = False

        mask = mask.ravel()
        return mask

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)

    def _get_rois(self):
        f = tables.open_file(
            os.path.join(self.root, 
            f'VoxelResponses_subject{self.subject[1]}.mat'))

        nodes = f.list_nodes('/roi')
        rois = {}
        for node in nodes:
            rois[node.name] = node[:]

        f.close()

        return rois

