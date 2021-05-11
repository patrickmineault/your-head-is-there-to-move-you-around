import sys

sys.path.append("../")
sys.path.append("../../CPC/dpc")
sys.path.append("../../CPC/backbone")

from tqdm import tqdm

import numpy as np
from pathlib import Path
import pickle

from python_dict_wrapper import wrap
import tables
import torch

from fmri_models import get_feature_model, get_projection_matrix, downsample_3d
from convex_models import compute_ridge_estimate, compute_boosting_estimate


block_size = 64
ff = 1e-6


def fit_one(model, filename):
    core_name = filename.stem
    f = tables.open_file(str(filename))
    X = f.get_node("/X_traintune")[:]

    X_preds = []
    for i in tqdm(range(X.shape[0] // block_size + 1)):
        with torch.no_grad():
            pred = model(
                torch.tensor(
                    np.tile(
                        X[i * block_size : (i + 1) * block_size, :, :]
                        .reshape((1, 1, -1, X.shape[-2], X.shape[-1]))
                        .astype(np.float32)
                        / 100.0,
                        (1, 3, 1, 1, 1),
                    ),
                    device="cuda",
                    dtype=torch.float,
                )
            )

        pred = downsample_3d(pred, 28)
        X_preds.append(
            pred.squeeze().permute(1, 0, 2, 3).reshape(pred.shape[2], -1).cpu()
        )

    X_preds = torch.cat(X_preds, axis=0)
    assert X_preds.shape[0] == X.shape[0]
    m = X_preds.mean(axis=0, keepdims=True)
    s = X_preds.std(axis=0, keepdims=True) + ff

    X_preds.add_(-m)
    X_preds.divide_(s)

    V = get_projection_matrix(X_preds, n=500)
    X_preds = torch.matmul(X_preds, V)

    idx_traintune = f.get_node("/Xidx_traintune")[:, -1]
    idx_report = f.get_node("/Xidx_report")[:, -1]

    idx_traintune = (idx_traintune).astype(np.int)
    idx_report = (idx_report).astype(np.int)

    Y_traintune = f.get_node("/Y_traintune")[:]
    Y_report = f.get_node("/Y_report")[:]
    f.close()

    kfold = 5
    splits = (np.arange(idx_traintune.size) / 100).astype(np.int) % kfold

    print(X_preds.shape)
    print(Y_traintune.shape)
    print(Y_report.shape)
    print(idx_traintune.size)

    for whichhalf in [0, 1]:
        resultss = []
        for delta in tqdm(range(-15, 15)):

            half = np.arange(len(idx_traintune))
            half_report = np.arange(len(idx_report))
            if whichhalf == 0:
                half = half < len(idx_traintune) / 2
                half_report = half_report < len(idx_report) / 2
            elif whichhalf == 1:
                half = half > len(idx_traintune) / 2
                half_report = half_report > len(idx_report) / 2

            idx_traintune_delta = np.fmin(
                np.fmax(idx_traintune + delta, 0), X_preds.shape[0] - 1
            )

            idx_report_delta = np.fmin(
                np.fmax(idx_report + delta, 0), X_preds.shape[0] - 1
            )
            results, _ = compute_ridge_estimate(
                torch.tensor(X_preds[idx_traintune_delta, :])[half, :],
                    dtype=torch.float,
                ),
                torch.tensor(Y_traintune[half, :], dtype=torch.float),
                torch.tensor(X_preds[idx_report_delta, :])[half_report, :],
                    dtype=torch.float,
                ),
                torch.tensor(Y_report[half_report, :], dtype=torch.float),
                splits[half],
            )
            resultss.append(results)

        with open(f"delay/results-{core_name}_motionless_h{whichhalf}.pkl", "wb") as f:
            pickle.dump(resultss, f)


def main():
    features = "gaborpyramid3d_motionless"

    args = wrap(
        {
            "features": features,  #'airsim_02',
            "ckpt_root": "../pretrained",
            "slowfast_root": "../../slowfast",
            "ntau": block_size - 10 + 1,
            "subsample_layers": False,
        }
    )

    model, _, data = get_feature_model(args)
    model.to(device="cuda")

    for filename in Path("/mnt/e/data_derived/packlab-dorsal/").glob("*.h5"):
        fit_one(model, filename)

    #    X = f.get_node('/X_hf')[:]


if __name__ == "__main__":
    main()
    # files = sorted(Path("/mnt/e/data_derived/packlab-dorsal/").glob("*.h5"))
    # for file_ in files:
    #    f = tables.read_file(file_)
    #    X = f.get_node('/X_hf')[:]