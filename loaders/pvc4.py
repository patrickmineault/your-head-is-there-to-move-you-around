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

cache = {}

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
        elif framecount == 0 and iconsize == 3:
            filetype = 3
            framecount, = struct.unpack('<I', f.read(4))
            spacedimcount, = struct.unpack('<I', f.read(4))
            assert spacedimcount == 2
            iconside1, = struct.unpack('<I', f.read(4))
            iconside2, = struct.unpack('<I', f.read(4))
            assert iconside1 == iconside2
            iconside = iconside1
            iconsize = iconside1 * iconside2
            return framecount, iconsize, iconside, filetype
        else:
            raise NotImplementedError("Function not fully implemented")


def _loadimfile(filepath):
    """translation of loadimfile.m and readfromimfile"""
    framecount, _, iconside, filetype = _openimfile(filepath)

    if filetype == 2:
        with open(filepath, 'rb') as f:
            f.read(16)        
            data = np.frombuffer(f.read(), dtype=np.uint8)
    elif filetype == 3:
        with open(filepath, 'rb') as f:
            f.read(24)        
            data = np.frombuffer(f.read(), dtype=np.uint8)
    else:
        raise NotImplementedError("Function not fully implemented")

    
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
        for item in glob.glob(os.path.join(root, 'Nat', '*')):
            cells.append(item)
        
        for item in glob.glob(os.path.join(root, 'NatRev', '*')):
            cells.append(item)

        cells = sorted(cells)

        if single_cell != -1:
            cells = [cells[single_cell]]

        images = []
        spktimes = []

        cell_info = collections.OrderedDict()
        i = 0
        for cell in cells:
            paths = glob.glob(os.path.join(cell, '*summary_file.mat'))
            for path in paths:
                summary = utils.load_mat_as_dict(path)
                
                respfiles = summary['celldata']['respfile']
                stimfiles = summary['celldata']['stimfile']
                cellids = summary['celldata']['cellid']
                repcounts = summary['celldata']['repcount']

                ntraining_frames = np.array(summary['celldata']['resplen']).sum()

                if not isinstance(respfiles, list):
                    respfiles = [respfiles]
                    stimfiles = [stimfiles]
                    cellids = [cellids]
                    repcounts = [repcounts]

                for respfile, stimfile, cellid, repcount in zip(respfiles, stimfiles, cellids, repcounts):
                    assert '+' not in respfile
                    resppath = os.path.join(os.path.dirname(path), 
                                        respfile)

                    if resppath.endswith('.mat'):
                        # Mat format
                        mat = utils.load_mat_as_dict(resppath)
                        spikes = mat['psth']                        
                    else:
                        # Text format
                        try:
                            data = pd.read_csv(resppath, sep='\t', header=None)
                        except FileNotFoundError:
                            continue

                        spikes = np.array(data.iloc[:, 2].tolist())

                    stimpath = os.path.join(os.path.dirname(path), stimfile)
                    framecount, iconsize, iconside, filetype = _openimfile(stimpath)
                    if iconside < 32:
                        print("iconside is small")
                        print(iconside)

                    info = {
                            'cellid': cellid,                        
                            'images_path': stimpath,
                            'spktimes': spikes,
                            'nrepeats': repcount,
                            'ntrainingframes': framecount,
                            'iconside': iconside,
                            }

                    if cellid not in cell_info:
                        cell_info[cellid] = [info]
                    else:
                        cell_info[cellid].append(info)

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

            largest_size = max([x['iconside'] for x in cell_files])
            
            # Figure out if there's a natural report fold for this dataset
            repeats = np.array([x['nrepeats'] for x in cell_files])
            the_report = repeats.argmax()
            report_set = -1

            if len(cell_files) > 1 and repeats[the_report] >= 10:
                report_set = the_report
                splits = {'train': [0, 1, 2, 3, 5, 6, 7, 8, 9],
                  'tune': [4],
                  'report': [],
                  'traintune': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                }
            else:
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
                sz = experiment['iconside']
                nframes = len(experiment['spktimes'])
                all_spks = np.array(experiment['spktimes'])

                for start_time in range(self.nframestart, nframes, nskip):
                    
                    if start_time + nskip + 1 > nframes:
                        # Incomplete frame.
                        continue

                    end_time = min((nframes, start_time + nskip + 1))

                    spk = np.array(experiment['spktimes'][start_time+1:end_time])
                    if np.any(spk < 0) or np.any(np.isnan(spk)):
                        # Skip this chunk
                        continue

                    if (((int(n / block_size) % nblocks in splits[split]) and (i != report_set))
                        or (split == 'report' and i == report_set)):
                        
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
                            'iconside': largest_size,
                            'start_spktime': start_time + 1,
                            'end_spktime': end_time,
                        })

                    n += 1

            assert n > 0
            n_electrodes += 1
        
        self.sequence = sequence
        self.total_electrodes = n_electrodes

        if self.total_electrodes == 0:
            raise Exception("Didn't find any data")

        if self.total_electrodes == 1:
            print(f"Loaded cell {self.sequence[-1]['cellid']}")

        self.root = root

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        global cache
        tgt = self.sequence[idx]

        # The images are natively different sizes, grayscale.
        
        # Mean and standard deviation vary widely across the image and across 
        # sequences. We normalize against the mean of means and standard 
        # deviations across images.
        if 'pvc4' in self.root:
            mm, ss, infill = 54, 43, 20
        elif 'v2' in self.root:
            mm, ss, infill = 73, 47, 73
        else:
            raise NotImplementedError()

        if tgt['images_path'] not in cache:
            X_ = _loadimfile(tgt['images_path'])
            if X_.shape[1] < tgt['iconside']:
                # The top left background is systematically at value 20.0
                X = infill * np.ones((X_.shape[0], tgt['iconside'], tgt['iconside']), dtype=np.uint8)
                delta = (X.shape[1] - X_.shape[1]) // 2
                assert delta > 0
                rg = slice(delta, delta + X_.shape[1])
                X[:, rg, rg] = X_
                X_ = X

            cache[tgt['images_path']] = X_

        ims = cache[tgt['images_path']]
        ims = ims[tgt['start_frame']:tgt['end_frame'], :, :].astype(np.float32)
        X = np.stack([ims, ims, ims], axis=0)

        if self.nx is not None:
            X = F.interpolate(torch.tensor(X), 
                             [self.nx, self.ny], 
                             align_corners=False,
                             mode='bilinear')

        X = (X - mm) / ss

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
