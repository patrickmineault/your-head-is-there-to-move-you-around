"""Fast bounded end-to-end smoke test: real data -> ViT features -> ridge -> corr.

Uses only the first N clips of a cell (the full cell has ~9000) so it finishes in
minutes on CPU, exercising the model + aggregator + ridge tail on real responses.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_feature_model, get_aggregator, resize, get_dataset, get_projection_matrix  # noqa: E402
from convex_models import compute_ridge_estimate  # noqa: E402


def featurize(loader_ds, idxs, model, aggregator, metadata, device, bs=8):
    feats, Ys = {}, []
    idxs = list(idxs)
    with torch.no_grad():
        for start in range(0, len(idxs), bs):
            chunk = idxs[start:start + bs]
            Xs = [torch.tensor(loader_ds[i][0]) for i in chunk]
            Ys += [torch.tensor(np.asarray(loader_ds[i][-1])) for i in chunk]
            X = resize(torch.stack(Xs).to(device).float(), metadata["sz"])
            if metadata["threed"]:
                model(X)
                for layer, al in list(model_acts.items()):
                    feats.setdefault(layer, []).append(aggregator(al).cpu())
            else:
                xr = X.permute(0, 2, 1, 3, 4).reshape(-1, X.shape[1], X.shape[3], X.shape[4])
                model(xr)
                for layer, al in list(model_acts.items()):
                    fit = al.reshape(X.shape[0], X.shape[2], *al.shape[1:]).permute(0, 2, 1, 3, 4)
                    feats.setdefault(layer, []).append(aggregator(fit).cpu())
    feats = {k: torch.cat(v, 0) for k, v in feats.items()}
    Y = torch.stack(Ys).float().reshape(len(idxs), -1)
    return feats, Y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="midway_bdd_vitb")
    ap.add_argument("--dataset", default="pvc4")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    c = ap.parse_args()

    args = SimpleNamespace(
        features=c.features, dataset=c.dataset, subset="0",
        ckpt_root=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "checkpoints"),
        data_root=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data_derived"),
        input_adapt="resize", vjepa_pad_t=(16 if "vjepa" in c.features else 0),
        subsample_layers=True,
        aggregator="average", aggregator_sz=8, device=c.device, resize=112,
    )
    model, model_acts, metadata = get_feature_model(args)
    metadata["sz"] = args.resize
    model.to(c.device)
    aggregator = get_aggregator(metadata, args)

    tr = get_dataset(args, "traintune")
    rp = get_dataset(args, "report")
    ntr, nrp = min(c.n, len(tr)), min(c.n, len(rp))
    print(f"{c.features} {c.dataset}: using {ntr}/{len(tr)} train, {nrp}/{len(rp)} report clips")

    Ftr, Ytr = featurize(tr, range(ntr), model, aggregator, metadata, c.device)
    Frp, Yrp = featurize(rp, range(nrp), model, aggregator, metadata, c.device)

    splits = (np.arange(ntr) / 10).astype(int) % 5
    for layer in Ftr:
        X, Xr = Ftr[layer], Frp[layer]
        m, s = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 0.1
        X = (X - m) / s
        Xr = (Xr - m) / s
        if 0 < c.pca < X.shape[1]:
            V = get_projection_matrix(X, n=c.pca)
            X, Xr = X.matmul(V), Xr.matmul(V)
        Ym = Ytr.mean(0, keepdim=True)
        res, _ = compute_ridge_estimate(X, Ytr - Ym, Xr, Yrp - Ym, splits, device=c.device)
        print(f"  {layer}: feat_dim={X.shape[1]:6d}  report_corr_mean={res['corrs_report_mean']:.4f}")
    print("OK")
