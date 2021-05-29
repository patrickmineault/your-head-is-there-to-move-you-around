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
from pathlib import Path
import requests
import struct
import subprocess
import tables
import time

import torch.nn.functional as F
import torch.utils.data

cache = {10: {}, 40: {}}

nclasses = 72  # 5 degree precision in heading discrimination.
max_speed = 3  # Max 3 m/s movement


def to_class(theta):
    theta = theta % (2 * np.pi)
    return int(theta / (2 * np.pi) * nclasses)


def to_linear_class(speed, maxspeed):
    return int(speed / maxspeed * nclasses)


class AirSim(torch.utils.data.Dataset):
    """
    Loads a segment from the Airsim flythrough data.
    """

    def __init__(self, root="./airsim", split="train", regression=True, nt=10):

        if split not in ("train", "tune", "report", "traintune"):
            raise NotImplementedError("Split is set to an unknown value")

        assert nt in (10, 40)

        self.split = split
        self.root = root

        cells = []
        for item in Path(root).glob("*/*/*.h5"):
            cells.append(item)

        cells = sorted(cells)

        splits = {
            "train": [0, 1, 2, 3, 5, 6, 7, 8],
            "tune": [4],
            "report": [9],
            "traintune": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
        nblocks = 10

        sequence = []
        for cell in cells:
            f = tables.open_file(cell, "r")
            labels = f.get_node("/labels")[:]
            f.close()

            for j in range(labels.shape[0]):
                if (j % nblocks) in splits[split]:
                    if regression:
                        # Outputs appropriate for regression
                        sequence.append(
                            {
                                "images_path": cell,
                                "labels": np.array(
                                    [
                                        labels[j]["heading_pitch"],
                                        labels[j]["heading_yaw"],
                                        labels[j]["rotation_pitch"],
                                        labels[j]["rotation_yaw"],
                                        labels[j]["speed"] / max_speed,
                                    ],
                                    dtype=np.float32,
                                ),
                                "idx": j,
                            }
                        )
                    else:
                        # Outputs appropriate for multi-class
                        hp = to_class(labels[j]["heading_pitch"])
                        hy = to_class(labels[j]["heading_yaw"])
                        rp = to_class(labels[j]["rotation_pitch"])
                        ry = to_class(labels[j]["rotation_yaw"])

                        # TODO(pmin): make max_speed not hard-coded.
                        speed = to_linear_class(labels[j]["speed"], max_speed)

                        sequence.append(
                            {
                                "images_path": cell,
                                "labels": np.array(
                                    [hp, hy, rp, ry, speed], dtype=np.int64
                                ),  # Torch requires long ints
                                "idx": j,
                            }
                        )

        if regression:
            self.noutputs = 5
            self.nclasses = 1
        else:
            self.noutputs = 5
            self.nclasses = nclasses

        self.sequence = sequence
        self.nt = nt

        if len(self.sequence) == 0:
            raise Exception("Didn't find any data")

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        global cache
        tgt = self.sequence[idx]

        if tgt["images_path"] not in cache[self.nt]:
            f = tables.open_file(tgt["images_path"], "r")
            if self.nt == 40:
                X_ = f.get_node("/videos")[:].squeeze()
            else:
                X_ = f.get_node("/short_videos")[:].squeeze()

            f.close()

            cache[self.nt][tgt["images_path"]] = X_

        X_ = cache[self.nt][tgt["images_path"]]
        X_ = X_[tgt["idx"], :].astype(np.float32)
        # The images are natively different sizes, grayscale.
        return (X_.transpose((1, 0, 2, 3)), tgt["labels"])

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)
