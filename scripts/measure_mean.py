import glob
import numpy as np

import sys
sys.path.append('../')
from loaders import pvc4

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

if __name__ == '__main__':
    main()