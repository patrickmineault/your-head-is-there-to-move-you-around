import argparse
import faulthandler
import numpy as np
import pickle
import os
import time
import wandb

from training import compute_corr
from models import (
    get_dataset,
    get_feature_model,
    get_aggregator,
    preprocess_data,
    get_projection_matrix,
    tune_batch_size,
)

from convex_models import compute_ridge_estimate, compute_boosting_estimate

from research_code.cka_step4 import cka

import torch

ff = 0.1


def save_to_wandb(results, weights, args, offline=False):
    if offline:
        os.environ["WANDB_MODE"] = "dryrun"
    else:
        os.environ["WANDB_MODE"] = "run"

    run = wandb.init(project="train_fmri_convex.py", config=vars(args), reinit=True)
    config = wandb.config

    # This allows the info to be visible in the dashboard
    wandb.log(results)

    # This saves the whole results (no histogramming). Has to happen after log.
    wandb.run.summary.update(results)

    # Also save the best weights.
    if not args.no_save:
        weight_path = os.path.join(wandb.run.dir, "optimal_weights.pkl")
        with open(weight_path, "wb") as f:
            pickle.dump(weights, f)

        # These weights are big, only save if necessary
        wandb.save(weight_path)

        results["weight_path"] = weight_path

    out_path = os.path.join(wandb.run.dir, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    wandb.save(out_path)
    run.finish()


def compute_layer(
    trainloader,
    reportloader,
    feature_model,
    aggregator,
    activations,
    metadata,
    args,
    max_r,
):
    print(f"Processing layer {args.layer_name}")
    t0 = time.time()

    X, Y = preprocess_data(
        trainloader, feature_model, aggregator, activations, metadata, args
    )

    if X is None:
        print(f"Skipping layer {args.layer_name}")
        return

    # Use k-fold cross-validation
    kfold = 5
    splits = (np.arange(X.shape[0]) / 100).astype(np.int) % kfold

    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True) + ff

    Ym = Y.mean(axis=0, keepdims=True)
    Y = Y - Ym

    # Use in-place operators instead of (X - m) / s to save memory.
    X.add_(-m)
    X.divide_(s)

    if args.pca > -1:
        V = get_projection_matrix(X, n=args.pca)
        X = torch.matmul(X, V)

    X_report, Y_report = preprocess_data(
        reportloader, feature_model, aggregator, activations, metadata, args
    )

    if X is None:
        print(f"Skipping layer {args.layer_name}")
        return

    Y_report = Y_report - Ym

    # Use in-place operators instead of (X - m) / s to save memory.
    X_report.add_(-m)
    X_report.divide_(s)

    if args.pca > -1:
        X_report = torch.matmul(X_report, V)

    if args.method == "ridge":
        results, weights = compute_ridge_estimate(X, Y, X_report, Y_report, splits)
    elif args.method == "boosting":
        results, weights = compute_boosting_estimate(X, Y, X_report, Y_report, splits)
    else:
        raise NotImplementedError("Method not implemented")

    cka_report = cka(X_report, Y_report)

    if not args.save_predictions:
        del weights["Y_preds"]

    results["feature_mean"] = m.squeeze().cpu().detach().numpy()
    results["fit_time"] = time.time() - t0
    results["cka_report"] = cka_report.item()
    results["layer"] = args.layer
    results["subset"] = args.subset
    results["max_r"] = max_r

    if not args.no_wandb:
        try:
            save_to_wandb(results, weights, args, offline=False)
        except wandb.errors.error.UsageError:
            print(">>> Could not save to cloud, using offline save this once.")
            save_to_wandb(results, weights, args, offline=True)
    else:
        print(results)


def check_existing(args, metadata):
    api = wandb.Api()

    runs = api.runs(
        "pmin/train_fmri_convex.py",
        {
            "$and": [
                {"config.exp_name": args.exp_name},
                {"config.dataset": args.dataset},
                {"config.aggregator": args.aggregator},
                {"config.aggregator_sz": args.aggregator_sz},
                {"config.pca": args.pca},
                {"config.features": args.features},
                {"config.subset": args.subset},
                {"config.method": args.method},
                {"state": "finished"},
            ]
        },
    )
    return len(runs) >= len(metadata["layers"])


def main(args):
    print("Fitting model")
    print(args)
    device = "cuda"

    try:
        os.makedirs(args.ckpt_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(args.cache_root)
    except FileExistsError:
        pass

    trainset = get_dataset(args, "traintune")
    reportset = get_dataset(args, "report")

    if hasattr(trainset, "max_r"):
        # Store it in the dataset
        max_r = trainset.max_r
    else:
        max_r = 1.0

    args.ntau = trainset.ntau
    feature_model, activations, metadata = get_feature_model(args)

    if args.skip_existing:
        exists = check_existing(args, metadata)
        if exists:
            print(">>> Run already exists, skipping")
            return

    aggregator = get_aggregator(metadata, args)
    feature_model.to(device=device)

    if args.autotune:
        batch_size = tune_batch_size(feature_model, trainset, metadata)
    else:
        batch_size = args.batch_size

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    reportloader = torch.utils.data.DataLoader(
        reportset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Do this for every layer under the sun.
    for layer_num, layer_name in enumerate(metadata["layers"].keys()):
        args.layer = layer_num  # For backwards compatibility
        args.layer_name = layer_name
        compute_layer(
            trainloader,
            reportloader,
            feature_model,
            aggregator,
            activations,
            metadata,
            args,
            max_r,
        )


if __name__ == "__main__":
    faulthandler.enable()
    desc = "Map a pretrained neural net to a time series of brain data (neurons or brains) using ridge regression."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--exp_name", required=True, help="Friendly name of experiment")
    parser.add_argument(
        "--features",
        default="gaborpyramid3d",
        type=str,
        help="What kind of features to use",
    )
    parser.add_argument(
        "--aggregator",
        default="average",
        type=str,
        help="What kind of aggregator to use",
    )
    parser.add_argument(
        "--aggregator_sz",
        default=8,
        type=int,
        help="The size the aggregator will be used with",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument(
        "--pca",
        default=-1,
        type=int,
        help="Size of PCA before model fit (if applicable)",
    )

    parser.add_argument(
        "--no_wandb", default=False, help="Skip using W&B", action="store_true"
    )
    parser.add_argument(
        "--no_save", default=False, help="Skip saving weights", action="store_true"
    )
    parser.add_argument(
        "--skip_existing", default=False, help="Skip existing runs", action="store_true"
    )
    parser.add_argument(
        "--autotune",
        default=False,
        help="Tune the batch size to maximize memory consumption",
        action="store_true",
    )
    parser.add_argument(
        "--subsample_layers",
        default=False,
        help="Subsample layers (saves disk space & mem)",
        action="store_true",
    )
    parser.add_argument(
        "--save_predictions",
        default=False,
        help="Save predictions among weights",
        action="store_true",
    )

    parser.add_argument(
        "--dataset", default="vim2", help="Dataset (currently vim2, pvc4)"
    )
    parser.add_argument(
        "--subset", default="s1", help="Either subject name or neuron num"
    )
    parser.add_argument("--data_root", default="./data_derived", help="Data path")
    parser.add_argument(
        "--cache_root", default="./cache", help="Precomputed cache path"
    )
    parser.add_argument(
        "--slowfast_root", default="", help="Path where SlowFast is installed"
    )
    parser.add_argument(
        "--ckpt_root",
        default="./pretrained",
        help="Path where trained model checkpoints will be downloaded",
    )
    parser.add_argument(
        "--method",
        default="ridge",
        help="Method to fit the model (ridge or boosting)",
    )

    args = parser.parse_args()
    main(args)