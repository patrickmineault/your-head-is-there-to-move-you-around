import glob
import numpy as np
from python_dict_wrapper import wrap

import sys
sys.path.append('../')
from loaders import pvc4
import fmri_models

def main():
    args = wrap({'data_root': '../data_derived',
                 'subset': 0,
                 'dataset': 'pvc4'})
    traintune = fmri_models.get_dataset(args, 'traintune')
    reportset = fmri_models.get_dataset(args, 'report')

    traintune_seq = {'../data_derived/crcns-pvc4/Nat/r0206B/test.review.clown.10sec.imsm':
                     np.zeros(707)}
    reportset_seq = {'../data_derived/crcns-pvc4/Nat/r0206B/test.review.clown.10sec.imsm':
                     np.zeros(707)}

    for s in traintune.sequence:
        if s['images_path'] in traintune_seq:
            traintune_seq[s['images_path']][s['start_spktime']:s['end_spktime']] = 1

    for s in reportset.sequence:
        if s['images_path'] in reportset_seq:
            reportset_seq[s['images_path']][s['start_spktime']:s['end_spktime']] = 1

    assert np.sum(traintune_seq['../data_derived/crcns-pvc4/Nat/r0206B/test.review.clown.10sec.imsm']) == 0
    assert np.sum(reportset_seq['../data_derived/crcns-pvc4/Nat/r0206B/test.review.clown.10sec.imsm']) > 0

if __name__ == '__main__':
    main()