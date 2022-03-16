import unittest
import sys

sys.path.append("../")
import paths
from models import get_feature_model
from python_dict_wrapper import wrap


class TestCPC(unittest.TestCase):
    def test_cpc(self):
        sys.path.append(paths.CPC_DPC)
        sys.path.append(paths.CPC_BACKBONE)
        args = wrap(
            {
                "features": "cpc-01",
                "ckpt_root": "/mnt/d/Documents/brain-scorer/pretrained/",
            }
        )
        model, activations, metadata = get_feature_model(args)

    def test_visualnet_ufc(self):
        sys.path.append(paths.CPC_DPC)
        sys.path.append(paths.CPC_BACKBONE)
        args = wrap(
            {
                "features": "cpc-02",
                "ckpt_root": "/mnt/d/Documents/brain-scorer/pretrained/",
            }
        )
        model, activations, metadata = get_feature_model(args)


if __name__ == "__main__":
    unittest.main()
