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

cache = {
    "traintune": {},
    "report": {},
}


class MST(torch.utils.data.Dataset):
    """
    Loads a segment from the packlab mst dataset (Mineault et al. 2011).

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

    def __init__(
        self,
        root="./data/packlab-mst",
        nt=1,
        ntau=10,
        split="train",
        single_cell=-1,
        norm_scheme="neutralbg",
    ):

        if nt != 1:
            raise NotImplementedError("nt = 1 implemented only")

        if ntau < 10:
            raise NotImplementedError("ntau >= 10 implemented only")

        block_len = 6  # in seconds
        framerate = 30
        block_size = framerate * block_len

        if split not in ("traintune", "train", "tune", "report"):
            raise NotImplementedError("Split is set to an unknown value")

        self.nt = nt
        self.ntau = ntau
        self.split = split
        self.framerate = framerate
        self.nframestart = 0

        cells = []
        for item in glob.glob(os.path.join(root, "*.h5")):
            cells.append(item)

        cells = sorted(cells)

        if single_cell != -1:
            cells = [cells[single_cell]]

        n_electrodes = 0
        sequence = []
        for cell in cells:
            f = tables.open_file(cell)

            splits = {
                "train": [0, 1, 2, 3, 5, 6, 7, 8, 9],
                "tune": [4],
                "report": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "traintune": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
            nblocks = 10

            if split == "report":
                rg = np.arange(f.get_node("/Xidx_report").shape[0])
                Xidx = np.array(f.get_node("/Xidx_report")[:])
                all_spks = np.array(f.get_node("/Y_report")[:])
            else:
                rg = np.arange(f.get_node("/Xidx_traintune").shape[0])
                Xidx = np.array(f.get_node("/Xidx_traintune")[:])
                all_spks = np.array(f.get_node("/Y_traintune")[:])

            all_spks = all_spks.squeeze()
            assert np.all(np.diff(rg) == 1)
            nskip = nt

            n = 0
            for start_time in range(rg[0] + self.nframestart, rg[-1] + 1, nskip):
                if start_time + nskip > rg[-1] + 1:
                    # Incomplete frame.
                    # print("incomplete frame")
                    continue

                end_time = min((rg[-1] + 2, start_time + nskip + 1))

                if all_spks.ndim > 1:
                    spk = np.array(all_spks[start_time : end_time - 1, :])
                else:
                    spk = np.array(all_spks[start_time : end_time - 1]).reshape((-1, 1))

                if np.any(np.isnan(spk)) or np.any(spk < 0):
                    # Skip this chunk
                    # print("nan")
                    continue

                if int(n / block_size) % nblocks in splits[split]:
                    padded_idx = Xidx[start_time : end_time - 1, :].astype(np.int)

                    if self.ntau > padded_idx.shape[1]:
                        # Pad.
                        lpad = (self.ntau - padded_idx.shape[1]) // 2
                        idxrg = np.zeros(self.ntau)
                        idxrg[lpad:-lpad] = np.arange(padded_idx.shape[1])
                        idxrg[:lpad] = 0
                        idxrg[-lpad:] = padded_idx.shape[1] - 1

                        padded_idx = padded_idx[:, idxrg.astype(np.int)]

                    sequence.append(
                        {
                            "stim_idx": padded_idx,
                            "spikes": spk,
                            "split": split,
                            "cellnum": n_electrodes,
                            "cellnum_end": n_electrodes + spk.shape[1],
                            "path": cell,
                        }
                    )

                n += 1

            assert n > 0
            n_electrodes += spk.shape[1]

            try:
                self.max_r = 1 / f.get_node("/corr_multiplier").read()
            except tables.exceptions.NoSuchNodeError:
                self.max_r = 1

            f.close()

        self.sequence = sequence

        # Use a lazy loading strategy
        self.total_electrodes = n_electrodes

        #
        self.offset = 0
        self.norm_scheme = norm_scheme

        assert self.total_electrodes > 0

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        # Cache are module variables
        global cache
        tgt = self.sequence[idx]

        # Use a common cache for everyone
        if (
            tgt["split"] in ("traintune", "train", "tune")
            and tgt["path"] not in cache["traintune"]
        ) or (tgt["split"] == "report" and tgt["path"] not in cache["report"]):

            f = tables.open_file(tgt["path"], "r")

            if tgt["split"] == "report":
                try:
                    stim = np.array(f.get_node("/X_report")[:])
                except tables.NoSuchNodeError:
                    stim = np.array(f.get_node("/X_traintune")[:])

                cache["report"][tgt["path"]] = stim
            else:
                stim = np.array(f.get_node("/X_traintune")[:])
                cache["traintune"][tgt["path"]] = stim

            f.close()
        else:
            if tgt["split"] == "report":
                stim = cache["report"][tgt["path"]]
            else:
                stim = cache["traintune"][tgt["path"]]

        assert tgt["split"] == self.split

        # Create a mask from the electrode range
        M = np.zeros((self.total_electrodes), dtype=np.bool)
        M[(tgt["cellnum"] + self.offset) : (tgt["cellnum_end"] + self.offset)] = True

        Y = np.zeros((self.total_electrodes, self.nt))
        Y[(tgt["cellnum"] + self.offset) : (tgt["cellnum_end"] + self.offset), :] = tgt[
            "spikes"
        ].T

        W = np.zeros((self.total_electrodes))
        W[
            (tgt["cellnum"] + self.offset) : (tgt["cellnum_end"] + self.offset)
        ] = 1.0  # max(w, .1)

        # channel, time, ny, nx
        # Add a bounds checks here.
        X = stim[np.fmin(tgt["stim_idx"], stim.shape[0] - 1), ...]
        X = np.concatenate([X, X, X], axis=0)

        if self.norm_scheme == "neutralbg":
            X = (X.astype(np.float32)) / 100.0
        elif self.norm_scheme == "airsim":
            X = (X.astype(np.float32) - 123.0) / 75.0
        elif self.norm_scheme == "cpc":
            X = (X.astype(np.float32) - 0.48) / 0.225

        return (X, M, W, Y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)
