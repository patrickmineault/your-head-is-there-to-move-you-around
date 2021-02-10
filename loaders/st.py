from pathlib import Path

from .utils import load_mat_as_dict

import numpy as np

import tables
import torch


class St(torch.utils.data.Dataset):
    """
    Loads data from a SuperTune dataset.
    """

    def __init__(self, root="./airsim", split="report", subset="MSTd"):

        if split not in ("report"):
            raise NotImplementedError("Split is set to an unknown value")

        if subset not in ("MSTd", "VIP", "V3A"):
            raise NotImplementedError("Subset is set to an unknown value")

        self.split = split
        self.root = root

        f = tables.open_file(str(Path(root) / "starfields.h5"), "r")
        stims = f.get_node("/stim")[:]
        f.close()

        f = tables.open_file(str(Path(root) / f"{subset}_resp.h5"), "r")
        responses = f.get_node("/resp")[:]
        f.close()

        assert stims.shape[0] == responses.shape[0]

        self.responses = responses
        self.stims = stims

        sequence = [{"idx": i} for i in range(stims.shape[0])]

        self.sequence = sequence
        self.ntau = stims.shape[2]

        # Average over space
        A = np.zeros((216, 24), dtype=np.float32)
        for i in range(24):
            A[i::24, i] = 1

        self.avg_mat = A.T

        # Use the center position to compare reps only
        A = np.zeros((216, 24), dtype=np.float32)
        for i in range(24):
            A[i + 24 * 4, i] = 1

        self.center_mat = A.T

        if len(self.sequence) == 0:
            raise Exception("Didn't find any data")

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]

        return (
            (self.stims[tgt["idx"], ...].astype(np.float32) - 123.0) / 75.0,
            self.responses[tgt["idx"], ...],
        )

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)