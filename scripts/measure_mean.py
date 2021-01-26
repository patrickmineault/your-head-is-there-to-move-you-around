import glob
import numpy as np

import sys
sys.path.append('../')
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

    files = glob.glob('/mnt/e/data_derived/airsim/*/*/*.h5')
    ms = []
    ss = []
    ns = []
    for filename in files:
        f = tables.open_file(filename, 'r')
        data = f.get_node('/videos')[:]
        print([data.mean(), 
               data.std(),
               ])
        ms.append(data.shape[0] * data.mean())
        ss.append(data.shape[0] * data.std())
        ns.append(data.shape[0])
    print(np.sum(ms) / np.sum(ns))
    print(np.sum(ss) / np.sum(ns))


if __name__ == '__main__':
    main()