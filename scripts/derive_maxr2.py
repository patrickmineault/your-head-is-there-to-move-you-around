import numpy as np
import sys

sys.path.append("../")
# sys.path.append("../loaders/")

from derive_dataset import get_max_r2, get_max_r2_alt
from loaders import pvc1

if __name__ == "__main__":
    """Only for pvc1. See generate_hyperflow for the method for HyperFlow."""
    maxr2s = []
    for single_cell in range(23):
        loader = pvc1.PVC1(
            "/mnt/e/data_derived/crcns-ringach-data/",
            nt=1,
            ntau=10,
            nframedelay=0,
            repeats=True,
            single_cell=single_cell,
            split="report",
        )
        Ys = []
        for _, _, Y, _ in loader:
            Ys.append(Y)

        Y = np.concatenate(Ys, axis=0)
        Y = Y.reshape((1, Y.shape[0], Y.shape[1])).transpose((0, 2, 1))

        maxr2 = get_max_r2(Y)
        maxr2_alt = get_max_r2_alt(Y)

        print(single_cell, maxr2, maxr2_alt)
        maxr2s.append(maxr2_alt)

    print(maxr2s)