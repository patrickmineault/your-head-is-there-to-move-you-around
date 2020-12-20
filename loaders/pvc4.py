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
import time

import torch.nn.functional as F
import torch.utils.data

def _openimfile(filepath):
    """translation of openimfile.m"""
    with open(filepath, 'rb') as f:
        framecount, iconsize = struct.unpack('<II', f.read(8))
        if framecount == 0 and iconsize == 1:
            # Little-endian
            filetype = 2
            framecount, = struct.unpack('<I', f.read(4))
            iconsize, = struct.unpack('<I', f.read(4))
            iconside = np.sqrt(iconsize)

            assert abs(iconside - int(iconside)) < 1e-6
            iconside = int(iconside)
            return framecount, iconsize, iconside, filetype
        elif framecount == 0 and iconsize == 0:
            # Big-endian
            filetype = 2
            framecount, = struct.unpack('>I', f.read(4))
            iconsize, = struct.unpack('>I', f.read(4))
            iconside = np.sqrt(iconsize)

            assert abs(iconside - int(iconside)) < 1e-6
            iconside = int(iconside)
            return framecount, iconsize, iconside, filetype
        else:
            raise NotImplementedError("Function not fully implemented")


def _loadimfile(filepath):
    """translation of loadimfile.m and readfromimfile"""
    framecount, _, iconside, filetype = _openimfile(filepath)

    if filetype != 2:
        raise NotImplementedError("Function not fully implemented")

    with open(filepath, 'rb') as f:
        f.read(16)        
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    data = data.reshape((framecount, iconside, iconside))[:, ::-1, :]
    return data


class PVC4(torch.utils.data.Dataset):
    """
    Loads a segment from Gallant crcns pvc-4 dataset.

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
                 root='./data/crcns-pvc4',
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

        if split not in ('train', 'tune', 'report'):
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

        paths = []
        for item in glob.glob(os.path.join(root, 'Nat', '*', '*summary_file.mat')):
            paths.append(item)
        
        for item in glob.glob(os.path.join(root, 'NatRev', '*', '*summary_file.mat')):
            paths.append(item)

        paths = sorted(paths)

        images = []
        cells = []
        spktimes = []

        cell_info = collections.defaultdict(list)
        i = 0
        for path in paths:
            summary = utils.load_mat_as_dict(path)
            
            respfiles = summary['celldata']['respfile']
            stimfiles = summary['celldata']['stimfile']
            cellids = summary['celldata']['cellid']
            repcounts = summary['celldata']['repcount']

            ntraining_frames = np.array(summary['celldata']['resplen']).sum()
            if ntraining_frames < framerate * min_seconds:
                continue

            if not isinstance(respfiles, list):
                respfiles = [respfiles]
                stimfiles = [stimfiles]
                cellids = [cellids]
                repcounts = [repcounts]

            for respfile, stimfile, cellid, repcount in zip(respfiles, stimfiles, cellids, repcounts):
                assert '+' not in respfile
                resppath = os.path.join(os.path.dirname(path), 
                                    respfile)

                try:
                    data = pd.read_csv(resppath, sep='\t', header=None)
                except FileNotFoundError:
                    continue

                spikes = data.iloc[:, 2].tolist()

                stimpath = os.path.join(os.path.dirname(path), 
                                        stimfile)

                stim = _loadimfile(stimpath)
                if stim.shape[1] < 60:
                    continue

                info = {
                        'cellid': cellid,
                        'images': stim,
                        'spktimes': spikes,
                        'nrepeats': repcount,
                        }

                cell_info[cellid].append(info)
                i += 1


        # Average a tune or report block to 10 seconds
        block_size = (block_len * framerate) // nt
        sequence = []
        n_electrodes = 0
        
        for cell_files in cell_info.values():
            ntraining_frames = sum([x['images'].shape[0] for x in cell_files])
            if ntraining_frames < min_seconds * framerate:
                # This is less than 2 minutes of data
                continue

            largest_size = max([x['images'].shape[1] for x in cell_files])
            
            # Figure out if there's a natural report fold for this dataset
            repeats = np.array([x['nrepeats'] for x in cell_files])
            the_report = repeats.argmax()
            report_set = -1
            if len(cell_files) > 1 and repeats[the_report] >= 10:
                report_set = the_report
                splits = {'train': [0, 1, 2, 3, 5, 6, 7, 8, 9],
                  'tune': [4],
                  'report': [],
                }
            else:
                splits = {'train': [0, 1, 2, 3, 5, 6, 7, 8],
                  'tune': [4],
                  'report': [9],
                }
            
            n = 0
            nskip = nt

            for i, experiment in enumerate(cell_files):
                sz = experiment['images'].shape[1]
                if sz < largest_size:
                    # Pad the movie to this size
                    X = 20 * np.ones((experiment['images'].shape[0], largest_size, largest_size), dtype=np.uint8)
                    delta = (X.shape[1] - sz) // 2
                    assert delta > 0
                    rg = slice(delta, delta + sz)
                    X[:, rg, rg] = experiment['images']
                    experiment['images'] = X

                nframes = len(experiment['spktimes'])

                for start_time in range(self.nframestart, nframes, nskip):
                    if start_time + nskip + 1 > nframes:
                        # Incomplete frame.
                        continue

                    end_time = min((nframes, start_time + nskip + 1))

                    spk = np.array(experiment['spktimes'][start_time+1:end_time])
                    if np.any(spk < 0):
                        # Skip this chunk
                        continue

                    if (int(n / block_size) % block_size in splits[split]
                        or (split == 'report' and i == report_set)):
                        sequence.append({
                            'images': experiment['images'],
                            'start_frame': start_time - self.nframedelay - self.ntau + 2,
                            'end_frame': end_time - self.nframedelay,
                            'spikes': spk,
                            'split': split,
                            'cellid': experiment['cellid'],
                            'cellnum': n_electrodes,
                        })

                    n += 1

            assert n > 0
            n_electrodes += 1
        
        self.sequence = sequence
        self.total_electrodes = n_electrodes

        if self.single_cell != -1:
            # Pick a single cell
            self.sequence = [x for x in self.sequence if x['cellnum'] == self.single_cell]
            for s in self.sequence:
                s['cellnum'] = 0
            self.total_electrodes = 1

        if self.total_electrodes == 0:
            raise Exception("Didn't find any data")

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]

        # The images are natively different sizes, grayscale.
        ims = tgt['images']
        ims = ims[tgt['start_frame']:tgt['end_frame'], :, :].astype(np.float32)
        X = np.stack([ims, ims, ims], axis=0)

        if self.nx is not None:
            X = F.interpolate(torch.tensor(X), 
                             [self.nx, self.ny], 
                             align_corners=False,
                             mode='bilinear')

        X = (X - 40.0) / 40.0

        # Create a mask from the electrode range
        m = np.zeros((self.total_electrodes), dtype=np.bool)
        m[tgt['cellnum']] = True

        Y = np.zeros((self.total_electrodes, self.nt))
        Y[tgt['cellnum'], :] = tgt['spikes']

        return (X, m, Y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)
