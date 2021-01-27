from pathlib import Path

from .utils import load_mat_as_dict

import numpy as np

import tables
import torch

class Stc1(torch.utils.data.Dataset):
    """
    Loads data from the stc-1 dataset.
    """
    def __init__(self, 
                 root='./airsim',
                 split='report',
                 subset='MSTd'
                 ):

        if split not in ('report'):
            raise NotImplementedError('Split is set to an unknown value')

        if subset not in ('MSTd', 'VIP'):
            raise NotImplementedError('Subset is set to an unknown value')

        self.split = split
        self.root = root

        f = load_mat_as_dict(str(Path(root) / f"{subset}.mat"))
        data = f['experiment1']['units']['vis']
        labels = np.array([r['stim_global'] for r in data])
        responses = np.array([r['resp_global'] for r in data])

        f = tables.open_file(str(Path(root) / "stc1-starfields.h5"), "r")
        stim = f.get_node("/stim")[:]
        labels_stim = f.get_node("/labels")[:]
        f.close()

        # Re-order responses to correspond to the right columns.
        if subset == 'MSTd':
            canonical_labels = [-180, -135, -90, -45, -22.5, 0, 22.5, 45, 90, 135]
        else:
            canonical_labels = [-180, -135, -90, -45, 0, 45, 90, 135]

        R = []
        S = []
        for l in canonical_labels:
            R.append(responses[:, labels[0, :] == l].squeeze())
            S.append(stim[labels_stim == l, ...].squeeze())

        responses = np.array(R)
        stims = np.array(S)

        assert stims.shape[0] == responses.shape[0]

        sequence = []
        self.responses = responses
        self.stims = stims
        self.labels = canonical_labels

        sequence = [{'idx': i} for i in range(stims.shape[0])]

        self.sequence = sequence

        if len(self.sequence) == 0:
            raise Exception("Didn't find any data")

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]

        return (self.stims[tgt['idx'], ...], 
                self.responses[tgt['idx'], ...])

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)