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
import tables
import time

import torch.nn.functional as F
import torch.utils.data

cache = {}

class MT2(torch.utils.data.Dataset):
    """
    Loads a segment from Gallant crcns mt-2 dataset.

    Each call to __get_item__ returns a tuple (X, mask, y)

    X: a numpy array with size (3, nt, ny, nx)
    mask: a mask that says which parts of the dataset to look at towards
    y: a numpy array with size (nt - ntau + 1, nelectrodes[experiment]).

    Arguments:
        root:        the root folder where the data is stored
        nt:          the number of images per micro-batch
        ntau:        the number of time lags that the y response listens to
        nframedelay: the number of frames the neural response is delayed by compared to the neural data.
        nframestart: the number of frames after the onset of a sequence to start at. 15 by default ~ 500ms
        split: either train, tune or report (if tune or report, returns a 1 / 10 tune/report set, if train, 8/10)
    """
    def __init__(self, 
                 root='./data/crcns-mt2',
                 nx=None,
                 ny=None,
                 nt=20,
                 ntau=1,
                 nframedelay=0,
                 nframestart=15,
                 split='train',
                 single_cell=-1,
                 ):

        block_len = 6  # in seconds
        framerate = 72
        min_seconds = 60  # At least one minute of data

        if split not in ('train', 'tune', 'report', 'traintune'):
            raise NotImplementedError('Split is set to an unknown value')

        if ntau + nframedelay > nframestart:
            raise NotImplementedError('ntau + nframedelay > nframestart, sequence starts before frame 0')

        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay
        self.nframestart = nframestart
        self.single_cell = single_cell
        self.split = split

        cells = []
        for item in glob.glob(os.path.join(root, '*.mat')):
            cells.append(item)

        cells = sorted(cells)

        if single_cell != -1:
            cells = [cells[single_cell]]

        images = []
        spktimes = []

        cell_info = collections.OrderedDict()
        i = 0
        for cell in cells:
            f = tables.open_file(cell, 'r')
            spikes = f.get_node('/psths')[:].squeeze()
            assert spikes.ndim == 1
            cellid = cell.split('/')[-1][:6]
            framecount = len(f.get_node('/rawStims'))
            f.close()

            info = {
                    'cellid': cellid,                        
                    'spktimes': spikes,
                    'ntrainingframes': framecount,
                    'images_path': cell,
                    'nrepeats': 1,
                    }

            cell_info[cellid] = [info]
            i += 1


        # Average a tune or report block to 10 seconds
        block_size = (block_len * framerate) // nt
        sequence = []
        n_electrodes = 0
        
        for j, cell_files in enumerate(cell_info.values()):
            ntraining_frames = sum([x['ntrainingframes'] for x in cell_files])
            
            if ntraining_frames < min_seconds * framerate:
                # This is less than 2 minutes of data
                print(ntraining_frames)
                print(cell_files[0]['cellid'])
                raise Exception("less than 1 minute of data")
                continue

            splits = {'train': [0, 1, 2, 3, 5, 6, 7, 8],
                'tune': [4],
                'report': [9],
                'traintune': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
            nblocks = 10
            
            n = 0
            nskip = nt

            # Calculate mean spikes.
            total_spikes = 0
            total_frames = 0
            for i, experiment in enumerate(cell_files):
                all_spks = np.array(experiment['spktimes'])
                total_spikes += all_spks[all_spks > -1].sum() * experiment['nrepeats']
                total_frames += (all_spks > -1).sum() * experiment['nrepeats']

            mean_spk = total_spikes / total_frames

            for i, experiment in enumerate(cell_files):
                nframes = len(experiment['spktimes'])
                all_spks = np.array(experiment['spktimes'])

                for start_time in range(self.nframestart, nframes, nskip):
                    if start_time + nskip + 1 > nframes:
                        # Incomplete frame.
                        # print("incomplete frame")
                        continue

                    end_time = min((nframes, start_time + nskip + 1))

                    spk = np.array(experiment['spktimes'][start_time+1:end_time])
                    if np.any(spk < 0) or np.any(np.isnan(spk)):
                        # Skip this chunk
                        # print("nan")
                        continue

                    if (int(n / block_size) % nblocks in splits[split]):
                        sequence.append({
                            'images_path': experiment['images_path'],
                            'start_frame': start_time - self.nframedelay - self.ntau + 2,
                            'end_frame': end_time - self.nframedelay,
                            'spikes': spk,
                            'split': split,
                            'cellid': experiment['cellid'],
                            'cellnum': n_electrodes,
                            'nrepeats': experiment['nrepeats'],
                            'mean_spk': mean_spk,
                        })

                    n += 1

            assert n > 0
            n_electrodes += 1
        
        self.sequence = sequence
        self.total_electrodes = n_electrodes

        if self.total_electrodes == 0:
            raise Exception("Didn't find any data")

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        global cache
        tgt = self.sequence[idx]

        # The images are natively different sizes, grayscale.
        
        if tgt['images_path'] not in cache:
            f = tables.open_file(tgt['images_path'], 'r')
            X_ = f.get_node('/rawStims')[:].squeeze()
            f.close()

            cache[tgt['images_path']] = X_

        ims = cache[tgt['images_path']]
        ims = ims[tgt['start_frame']:tgt['end_frame'], :, :].astype(np.float32)
        X = np.stack([ims, ims, ims], axis=0)

        if self.nx is not None:
            X = F.interpolate(torch.tensor(X), 
                             [self.nx, self.ny], 
                             align_corners=False,
                             mode='bilinear')

        # Mean and std for ct0001_arg0466d_128.mat
        X = (X - 115.0) / 59.0

        # Create a mask from the electrode range
        M = np.zeros((self.total_electrodes), dtype=np.bool)
        M[tgt['cellnum']] = True

        Y = np.zeros((self.total_electrodes, self.nt))
        Y[tgt['cellnum'], :] = tgt['spikes']

        w = np.sqrt(tgt['nrepeats'] / max(tgt['mean_spk'], .1))
        W = np.zeros((self.total_electrodes))
        W[tgt['cellnum']] = w # max(w, .1)

        return (X, M, W, Y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)
