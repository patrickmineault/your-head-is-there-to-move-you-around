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
resp_ts = {}
resp_rs = {}
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
                 nt=4,
                 ntau=1,
                 nframedelay=0,
                 split='train',
                 subject='s1',
                 mask_type='roi'
                 ):

        block_len = 10  # in seconds

        if split not in ('traintune', 'train', 'tune', 'report'):
            raise NotImplementedError('Split is set to an unknown value')

        if subject not in ('s1', 's2', 's3'):
            raise NotImplementedError("Subject not s1, s2 or s3")

        if mask_type not in ('early', 'roi', 'valid'):
            raise NotImplementedError("Mask must be early, roi, valid")

        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay
        self.subject = subject
        self.split = split
        self.framerate = 15
        self.sampling_freq = 16
        self.mask_type = mask_type


        if 540 % nt != 0:
            raise NotImplementedError("nt must divide 540")

        sequences = []

        if split == 'report':
            seq_end = 540
        else:
            seq_end = 7200

        nblocks = 10

        for i in range(0, seq_end, self.nt):
            which_bucket = i // block_len
            if (split == 'traintune' or
                (split == 'train' and (which_bucket % nblocks) != 4) or 
                (split == 'tune' and (which_bucket % nblocks) == 4) or
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
        self.mask = self._get_master_mask(mask_type)

        self.total_electrodes = self.mask.sum()
        self.max_r2 = self._get_max_r2s()

        assert self.total_electrodes > 0


    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        # Cache are module variables
        global stim_t, stim_r, resp_ts, resp_rs, rois
        tgt = self.sequence[idx]

        # Use a common cache for everyone
        if ((tgt['split'] in ('traintune', 'train', 'tune') and stim_t is None) or
            (tgt['split'] in ('report') and stim_r is None)):

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

        if ((tgt['split'] in ('traintune', 'train', 'tune') and self.subject not in resp_ts) or
            (tgt['split'] in ('report') and self.subject not in resp_rs)):
            # Lazy load
            f = tables.open_file(
                os.path.join(self.root, 
                f'VoxelResponses_subject{self.subject[1]}.mat'))

            if tgt['split'] == 'report':
                resp_rs[self.subject] = f.get_node('/rv')[:].T
                resp_rs[self.subject] = resp_rs[self.subject][:, self.mask]
            else:
                resp_ts[self.subject] = f.get_node('/rt')[:].T[:self.total_reponses, :]
                resp_ts[self.subject] = resp_ts[self.subject][:, self.mask]

            f.close()

        
        if tgt['split'] == 'report':
            stim = stim_r
            resp = resp_rs[self.subject]
        else:
            stim = stim_t
            resp = resp_ts[self.subject]

        assert tgt['split'] == self.split

        X = stim[tgt['stim_idx'], ...].transpose((1, 0, 2, 3))
        Y = resp[tgt['resp_idx'], :].astype(np.float32)

        X = (X.astype(np.float32) - 85) / 63

        return (X, Y)

    def _get_master_mask(self, mask_type):
        f = tables.open_file(
            os.path.join(self.root, 
            f'VoxelResponses_subject{self.subject[1]}.mat'))
        master_mask = f.get_node('/mask')[:]
        f.close()

        if mask_type == 'early':
            mask = np.zeros_like(master_mask)
            for k, v in self.rois.items():
                if k.startswith('v') and k.startswith('MT'):
                    mask |= (v > 0).ravel()
        elif mask_type == 'roi':
            mask = np.zeros_like(master_mask)
            for k, v in self.rois.items():
                mask |= (v > 0).ravel()
        elif mask_type == 'valid':
            mask = np.ones_like(master_mask)
        else:
            raise NotImplementedError(f"Mask type `{mask_type}` not implemented.")

        mask = mask & master_mask
        return mask.ravel()


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

    def _get_max_r2s(self):
        f = tables.open_file(
            os.path.join(self.root, 
            f'VoxelResponses_subject{self.subject[1]}.mat'))

        max_r2 = f.get_node('/maxr2')[:]
        f.close()

        return max_r2