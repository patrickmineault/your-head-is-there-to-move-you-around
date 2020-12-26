import sys
sys.path.append('../')
import unittest

from python_dict_wrapper import wrap

from fmri_models import (get_dataset, 
                         preprocess_data, 
                         get_feature_model,
                         get_aggregator)

import torch

class TestFmriModels(unittest.TestCase):
    def test_s2(self):
        args = wrap({'subject': 's1',
                     'batch_size': 10,
                     'features': 'gaborpyramid3d',
                     'aggregator': 'average',
                     'dataset': 'vim2',
                     'data_root': '../data',
                     'subset': False,
                     'layer': 0})

        feature_model, activations, metadata = get_feature_model(args)
        aggregator = get_aggregator(metadata, args)
        reportset = get_dataset(args, 'report')

        reportloader = torch.utils.data.DataLoader(reportset, 
                                                batch_size=args.batch_size, 
                                                shuffle=False,
                                                pin_memory=True
                                                )

        feature_model.to(device='cuda')
        X_report, Y_report = preprocess_data(reportloader, 
                            feature_model, 
                            aggregator,
                            activations, 
                            metadata,
                            args)

        print(Y_report.shape)

if __name__ == '__main__':
    unittest.main()