import glob
import numpy as np
import scipy
import scipy.signal

import matplotlib.pyplot as plt

import sys

sys.path.append("../")
import tables
from loaders import airsim


def main():
    """
    files = glob.glob('../data_derived/crcns-pvc4/*/*/*.imsm')
    ms = []
    ss = []
    for filename in files:
        data = pvc4._loadimfile(filename)
        m = data.shape[1] // 2
        print([data[:, 0, 0].mean(),
               data[:, m, m].mean(),
               data.mean(),
               data[:, m, m].std(),
               data.std()])
        ms.append(data.mean())
        ss.append(data.std())
    """
    """
    files = glob.glob('../data_derived/crcns-v2/*/*/*.imsm')
    ms = []
    ss = []
    for filename in files:
        data = pvc4._loadimfile(filename)
        m = data.shape[1] // 2
        print([data[:, 0, 0].mean(), 
               data[:, m, m].mean(),
               data.mean(), 
               data[:, m, m].std(),
               data.std()])
        ms.append(data.mean())
        ss.append(data.std())
    print(np.mean(ms))
    print(np.mean(ss))
    """

    files = glob.glob("/mnt/e/data_derived/airsim/batch2/*/*/output.h5")
    ms = []
    ss = []
    ns = []
    for filename in files:
        f = tables.open_file(filename, "r")
        data = f.get_node("/short_videos")[:]
        print(
            [
                data.mean(),
                data.std(),
            ]
        )
        ms.append(data.shape[0] * data.mean())
        ss.append(data.shape[0] * data.std())
        ns.append(data.shape[0])
    print(np.sum(ms) / np.sum(ns))
    print(np.sum(ss) / np.sum(ns))

    """
    files = glob.glob("/mnt/e/data_derived/packlab-mst/*.h5")
    ms = []
    ss = []
    ns = []
    for filename in files:
        f = tables.open_file(filename, "r")
        data = f.get_node("/X_traintune")[:]
        xi, yi = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))
        g = np.exp(-(xi ** 2 + yi ** 2) / 2 / 1.3 ** 2)
        g = g / g.sum()
        mask = (
            scipy.signal.convolve(
                data > 0, g.reshape((1, g.shape[0], g.shape[1])), "same"
            )
            > 0.002
        )

        plt.imshow(mask[10, :, :])
        plt.savefig("figures/mask.png")

        m = data[mask].mean()
        s = data[mask].std()
        print(
            [
                # Only sum over non-zero elements.
                filename,
                m,
                s,
            ]
        )
        data = f.get_node("/X_report")[:]
        xi, yi = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))
        g = np.exp(-(xi ** 2 + yi ** 2) / 2 / 1.3 ** 2)
        g = g / g.sum()
        mask = (
            scipy.signal.convolve(
                data > 0, g.reshape((1, g.shape[0], g.shape[1])), "same"
            )
            > 0.002
        )

        plt.imshow(mask[10, :, :])
        plt.savefig("figures/mask.png")

        m = data[mask].mean()
        s = data[mask].std()
        print(
            [
                # Only sum over non-zero elements.
                filename,
                m,
                s,
            ]
        )

        ms.append(m)
        ss.append(s)
        ns.append(data.shape[0])
    print(np.sum(ms) / np.sum(ns))
    print(np.sum(ss) / np.sum(ns))
    """


if __name__ == "__main__":
    main()