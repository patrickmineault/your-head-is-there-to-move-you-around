import glob
import numpy as np
from python_dict_wrapper import wrap

import sys
sys.path.append('../')
from loaders import pvc4
import fmri_models

def main():
    # This one has a dedicated reportset
    args = wrap({'data_root': '../data_derived',
                 'subset': 0,
                 'dataset': 'pvc4'})
    traintune = fmri_models.get_dataset(args, 'traintune')
    reportset = fmri_models.get_dataset(args, 'report')

    traintune_paths = {s['images_path'] for s in traintune.sequence}
    reportset_paths = {s['images_path'] for s in reportset.sequence}

    assert len(traintune_paths) == 5
    assert len(reportset_paths) == 1
    assert len(traintune_paths.intersection(reportset_paths)) == 0

    # This one has a shared reportset
    args = wrap({'data_root': '../data_derived',
                 'subset': 6,
                 'dataset': 'pvc4'})
    traintune = fmri_models.get_dataset(args, 'traintune')
    reportset = fmri_models.get_dataset(args, 'report')

    traintune_paths = {s['images_path'] for s in traintune.sequence}
    reportset_paths = {s['images_path'] for s in reportset.sequence}

    assert len(traintune_paths) == 2
    assert len(reportset_paths) == 1
    assert len(traintune_paths.intersection(reportset_paths)) == 1

    traintune_seq = {'../data_derived/crcns-pvc4/Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm':
                     np.zeros(3600, dtype=np.bool)}
    reportset_seq = {'../data_derived/crcns-pvc4/Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm':
                     np.zeros(3600, dtype=np.bool)}

    for s in traintune.sequence:
        if s['images_path'] in traintune_seq:
            traintune_seq[s['images_path']][s['start_spktime']:s['end_spktime']] = True

    for s in reportset.sequence:
        if s['images_path'] in reportset_seq:
            reportset_seq[s['images_path']][s['start_spktime']:s['end_spktime']] = True

    assert np.sum(
        traintune_seq['../data_derived/crcns-pvc4/Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm'] & 
        reportset_seq['../data_derived/crcns-pvc4/Nat/r0214A/test.review.mountlake.20_pix.50sec.imsm']) == 0

if __name__ == '__main__':
    main()