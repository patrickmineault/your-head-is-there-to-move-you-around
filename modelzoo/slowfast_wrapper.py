import os
import requests
from torch import nn
from tqdm import tqdm

from .util import download

from slowfast.models.build import build_model
from slowfast.config.defaults import get_cfg
from slowfast.utils.checkpoint import load_checkpoint

kinetics_root = 'https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/'
x3d_root = 'https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/'

paths = {'I3D': ('Kinetics/c2/I3D_8x8_R50.yaml', kinetics_root + 'I3D_8x8_R50.pkl'),
         'Slow': ('Kinetics/c2/SLOW_8x8_R50.yaml', kinetics_root + 'SLOWONLY_8x8_R50.pkl'),
         'SlowFast': ('Kinetics/c2/SLOWFAST_8x8_R50.yaml', kinetics_root + 'SLOWFAST_8x8_R50.pkl'),
         'SlowFast_Fast': ('Kinetics/c2/SLOWFAST_8x8_R50.yaml', kinetics_root + 'SLOWFAST_8x8_R50.pkl'),
         'SlowFast_Slow': ('Kinetics/c2/SLOWFAST_8x8_R50.yaml', kinetics_root + 'SLOWFAST_8x8_R50.pkl'),
         'X3DM': ('Kinetics/X3D_M.yaml', x3d_root + 'x3d_m.pyth')}

class SlowFast(nn.Module):
    def __init__(self, args):
        super(SlowFast, self).__init__()
        assert args.features in paths

        yaml, remote = paths[args.features]

        cfg = get_cfg()
        cfg.merge_from_file(
            os.path.join(args.slowfast_root, 'configs', yaml)
        )
        cfg.NUM_GPUS = 1
        cfg.DATA.NUM_FRAMES = args.ntau
        model = build_model(cfg)

        # Now load the weights.
        ckpt = os.path.basename(remote)
        local_path = os.path.join(args.ckpt_root, ckpt)
        if not os.path.exists(
            os.path.join(args.ckpt_root, ckpt)):
            download(remote, local_path)

        load_checkpoint(
            local_path,
            model,
            False,
            None,
            inflation=False,
            convert_from_caffe2='c2' in yaml,
        )
        self.model = model
        self.features = args.features

    def forward(self, X):
        if self.features.startswith('SlowFast'):
            slc = slice(2, X.shape[2] - 1, 4)
            inputs = [X[:, :, slc, :, :], X]
        else:
            inputs = [X]
        return self.model(inputs)