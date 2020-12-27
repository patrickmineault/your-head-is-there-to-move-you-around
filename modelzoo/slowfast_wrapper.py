import os
import requests
from torch import nn
from tqdm import tqdm

from slowfast.models.build import build_model
from slowfast.config.defaults import get_cfg
from slowfast.utils.checkpoint import load_checkpoint

kinetics_root = 'https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/'
x3d_root = 'https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/'

paths = {'I3D': ('Kinetics/c2/I3D_8x8_R50.yaml', kinetics_root + 'I3D_8x8_R50.pkl'),
         'Slow': ('Kinetics/c2/SLOW_8x8_R50.yaml', kinetics_root + 'SLOW_8x8_R50.pkl'),
         'SlowFast': ('Kinetics/c2/SLOWFAST_8x8_R50.yaml', kinetics_root + 'SLOWFAST_8x8_R50.pkl'),
         'X3DL': ('Kinetics/X3D_L.yaml', x3d_root + 'x3d_l.pyth')}

class SlowFast(nn.Module):
    def __init__(self, args):
        super(SlowFast, self).__init__()
        assert args.features in paths

        yaml, remote = paths[args.features]

        cfg = get_cfg()
        cfg.merge_from_file(
            os.path.join(args.slowfast_path, 'configs', yaml)
        )
        cfg.NUM_GPUS = 1
        cfg.DATA.NUM_FRAMES = 64
        model = build_model(cfg)

        # Now load the weights.
        ckpt = os.path.basename(remote)
        local_path = os.path.join(args.ckpt_root, ckpt)
        if not os.path.exists(
            os.path.join(args.ckpt_root, ckpt)):
            self._download(remote, local_path)

        load_checkpoint(
            local_path,
            model,
            False,
            None,
            inflation=False,
            convert_from_caffe2='c2' in yaml,
        )
        self.model = model

    def _download(self, url, local_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 2 ** 16 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    def forward(self, X):
        slc = slice(2, X.shape[2] - 1, 4)
        inputs = [X[:, :, slc, :, :], X]
        print(inputs[0].shape)
        print(inputs[1].shape)
        return self.model(inputs)