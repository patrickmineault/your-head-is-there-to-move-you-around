from . import utils

import datetime
import glob
import matplotlib
import matplotlib.image
import numpy as np
import os
import requests
import subprocess
import time

import torch.utils.data

def _find_framerate(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('FrameRate\x00'):
                return float(line[-12:].strip('\x00'))
    return None

def _fget_spk(path):
    """
    Translation of fget_spk.m, marker Revision 1.1 - brian 09.16.99
    """
    spkhdrsize = 828
    assert 'mq' not in path
    assert 'film02' not in path
    assert 'film32' not in path
    assert 'cell' not in path

    with open(path, 'rb') as f:
        data = f.read(7)
        _ = f.read(641-7)

    if data[:7] == b'DAN_SPK':
        return np.fromfile(path, np.int32, offset=spkhdrsize).astype(np.float32) / 10000.0
    else:
        raise NotImplementedError("DAN_SPK header not found!")

    

class PVC2(torch.utils.data.Dataset):
    """
    Loads a segment from the Yang Dan crcns pvc-2 dataset.

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
                 root='./data/crcns-pvc2',
                 nt=20,
                 ntau=1,
                 nframedelay=0,
                 nframestart=15,
                 split='train',
                 ):

        if split not in ('train', 'tune', 'report'):
            raise NotImplementedError('Split is set to an unknown value')

        if ntau + nframedelay > nframestart:
            raise NotImplementedError('ntau + nframedelay > nframestart, sequence starts before frame 0')

        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay
        self.nframestart = nframestart

        paths = []
        for item in glob.glob(os.path.join(root, '2D_noise_natural', 'Spike_and_Log_files', '*', '*equalpower*.log')):
            paths.append(item)

        paths = sorted(paths)

        datasets = []
        spktimes = []
        framerates = []

        for path in paths:
            framerate = _find_framerate(path)
            if framerate is None:
                continue

            dataset = path[-6:-4]
            if dataset not in ('B1', 'B2', 'C1', 'C2', 'C3'):
                continue

            datasets.append(dataset)
            spktimes.append(_fget_spk(path[:-4] + '.sa0'))
            framerates.append(framerate)

        im_mat = [('B1', 'Equalpower_B1_25hz.mat'), 
                  ('B2', 'Equalpower_B2_25hz.mat'), 
                  # ('B3', 'Equalpower_B3_25hz.mat'), 
                  ('C1', 'Equalpower_C1_25hz.mat'), 
                  ('C2', 'Equalpower_C2_25hz.mat')]

        ims = []
        start_times = []
        t0 = 0

        splits = {'train': [0, 1, 2, 3, 5, 6, 7, 8],
                  'tune': [4],
                  'report': [9],
                  }

        sequence = []
        movies = {}

        n = 0
        for dataset, mat_path in im_mat:
            the_dict = utils.load_mat_as_dict(
                os.path.join(root, '2D_noise_natural', 'Stimulus_Files', mat_path)
            )
            ims = the_dict['mov']

            movies[dataset] = ims

            start_times.append(t0)
            t0 += the_dict['mov'].shape[-1]

            nframes = ims.shape[2]
            nskip = nt
            ntrainings = int(np.ceil((nframes - nt) / nskip) + 1)

            electrode_range = np.where([x == dataset for x in datasets])[0].astype(np.uint32)

            Y = []
            for el in electrode_range:
                Y.append(np.histogram(
                    spktimes[el], bins=(self.nframedelay + np.arange(nframes + 10)) / framerates[el])[0])
            Y = np.stack(Y, axis=0)

            for start_time in range(self.nframestart, nframes, nskip):
                if int(n / 10) % 10 not in splits[split]:
                    continue

                if start_time + nskip + 1 > nframes:
                    # Incomplete frame.
                    continue

                end_time = min((nframes, start_time + nskip + 1))
                spike_frames = np.arange(start_time, 
                                            end_time)

                sequence.append({
                    'dataset': dataset,
                    'start_frame': start_time - self.nframedelay - self.ntau + 2,
                    'end_frame': end_time - self.nframedelay,
                    'electrode_range': electrode_range,
                    'spike_frames': spike_frames,
                    'nframes': nframes,
                    'spikes': Y[:, start_time+1:end_time]})

                n = n + 1

        # Create a mapping from a single index to the necessary information needed to load the corresponding data
        self.sequence = sequence
        self.total_electrodes = len(spktimes)
        self.spktimes = spktimes
        self.framerates = framerates
        self.datasets = datasets
        self.movies = movies

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]

        # The images are natively 12 x 12, grayscale.
        ims = self.movies[tgt['dataset']]
        ims = ims[:, :, tgt['start_frame']:tgt['end_frame']].astype(np.float32).transpose((2, 0, 1))
        X = np.stack([ims, ims, ims], axis=0)

        # Create a mask from the electrode range
        m = np.zeros((self.total_electrodes), dtype=np.bool)
        m[tgt['electrode_range']] = True

        Y = np.zeros((self.total_electrodes, self.nt))
        Y[tgt['electrode_range'], :] = tgt['spikes']

        return (X, m, Y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)
