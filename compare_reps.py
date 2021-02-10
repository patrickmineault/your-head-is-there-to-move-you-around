import argparse
import numpy as np
import pickle
import os
import time
import wandb

from training import (
    compute_corr,
    compute_rdm,
    compute_rdm_distance,
    bootstrap_rdm_distance,
    bootstrap_ycols,
)
from fmri_models import (
    get_dataset,
    get_feature_model,
    get_aggregator,
    preprocess_data,
    preprocess_data_consolidated,
    get_projection_matrix,
    tune_batch_size,
)

from research_code.cka_step4 import cka

import torch

ff = 0.1


def save_to_wandb(results, weights, args, offline=False):
    if offline:
        os.environ["WANDB_MODE"] = "dryrun"
    else:
        os.environ["WANDB_MODE"] = "run"

    run = wandb.init(project="compare_reps", config=vars(args), reinit=True)
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


def compute_layer(reportloader, feature_model, activations, metadata, args):
    print(f"Processing layer {args.layer_name}")
    t0 = time.time()

    X_, Y_ = preprocess_data(
        reportloader,
        feature_model,
        lambda x: x.reshape(x.shape[0], -1),
        activations,
        metadata,
        args,
    )

    assert X_.shape[0] == Y_.shape[0]
    assert X_.ndim == 2
    assert Y_.ndim == 2

    if X_ is None:
        print(f"Skipping layer {args.layer_name}")
        return

    nboot = 1
    results = {}
    rdms = {}
    for norm in ["raw", "demean", "z"]:
        for reduction in ["all", "center", "avg"]:
            print(f"{norm}_{reduction}")
            if reduction == "avg":
                if hasattr(reportloader.dataset, "avg_mat"):
                    # Do a reduction
                    avg_mat = torch.tensor(
                        reportloader.dataset.avg_mat, device=X_.device
                    )
                    X_report = avg_mat @ X_
                    Y_report = avg_mat @ Y_
                else:
                    continue
            elif reduction == "center":
                if hasattr(reportloader.dataset, "center_mat"):
                    # Do a reduction
                    center_mat = torch.tensor(
                        reportloader.dataset.center_mat, device=X_.device
                    )
                    X_report = center_mat @ X_
                    Y_report = center_mat @ Y_
                else:
                    continue
            elif reduction == "all":
                X_report = X_
                Y_report = Y_
            else:
                raise NotImplementedError(f"{reduction} not implemented")

            if norm == "z":
                X_report = X_report - X_report.mean(axis=0, keepdims=True)
                X_report = X_report / (1e-6 + X_report.std(axis=0, keepdims=True))

                Y_report = Y_report - Y_report.mean(axis=0, keepdims=True)
                Y_report = Y_report / (1e-6 + Y_report.std(axis=0, keepdims=True))
            elif norm == "demean":
                X_report = X_report - X_report.mean(axis=0, keepdims=True)
                Y_report = Y_report - Y_report.mean(axis=0, keepdims=True)
            elif norm == "raw":
                pass
            else:
                raise NotImplementedError(f"{norm} not implemented")

            # Compute CKA for report fold
            cka_m, cka_s = bootstrap_ycols(cka, X_report, Y_report, nboot=nboot)

            # Different types of RDM distances
            d_rank, s_rank = bootstrap_rdm_distance(
                X_report, Y_report, "rank", nboot=nboot
            )

            # d_r2, s_r2 = bootstrap_rdm_distance(X_report, Y_report, "r2", nboot=nboot)

            # Population spiking similarity
            def tuning_sim(X, Y):
                # and sum
                the_X, the_Y = X.sum(axis=1), Y.sum(axis=1)
                return np.corrcoef(the_X, the_Y)[0, 1]

            p_m, p_s = bootstrap_ycols(tuning_sim, X_report, Y_report, nboot=nboot)

            postfix = f"_{reduction}_{norm}"
            results = {
                **results,
                f"cka{postfix}": cka_m.item(),
                f"cka_s{postfix}": cka_s.item(),
                f"rdm_dist_rank{postfix}": d_rank.item(),
                f"rdm_dist_rank_s{postfix}": s_rank.item(),
                # f"rdm_dist_r2{postfix}": d_r2.item(),
                # f"rdm_dist_r2_s{postfix}": s_r2.item(),
                f"pop_tc{postfix}": p_m,
                f"pop_tc_s{postfix}": p_s,
            }

            rdm_X = compute_rdm(X_report)
            rdm_Y = compute_rdm(Y_report)

            rdms = {
                **rdms,
                f"rdm_X{postfix}": rdm_X,
                f"rdm_Y{postfix}": rdm_Y,
            }

    results = {
        **results,
        "fit_time": time.time() - t0,
        "layer": args.layer,
        "subset": args.subset,
    }

    if not args.no_wandb:
        try:
            save_to_wandb(results, rdms, args, offline=False)
        except wandb.errors.error.UsageError:
            print(">>> Could not save to cloud, using offline save this once.")
            save_to_wandb(results, rdms, args, offline=True)
    else:
        print(results)


def check_existing(args, metadata):
    api = wandb.Api()

    runs = api.runs(
        "pmin/compare_reps",
        {
            "$and": [
                {"config.exp_name": args.exp_name},
                {"config.dataset": args.dataset},
                {"config.aggregator": args.aggregator},
                {"config.aggregator_sz": args.aggregator_sz},
                {"config.pca": args.pca},
                {"config.features": args.features},
                {"config.subset": args.subset},
                {"config.consolidated": args.consolidated},
                {"state": "finished"},
            ]
        },
    )
    return len(runs) >= len(metadata["layers"])


def main(args):
    print("Fitting model")
    print(args)
    device = "cuda"

    args.aggregator = "id"  # For compatibility
    args.aggregator_sz = 1  # For compatibility

    reportset = get_dataset(args, "report")
    args.ntau = reportset.ntau
    feature_model, activations, metadata = get_feature_model(args)

    if args.skip_existing:
        exists = check_existing(args, metadata)
        if exists:
            print(">>> Run already exists, skipping")
            return

    feature_model.to(device=device)
    batch_size = 4

    reportloader = torch.utils.data.DataLoader(
        reportset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Do this for every layer under the sun.
    for layer_num, layer_name in enumerate(metadata["layers"].keys()):
        args.layer = layer_num  # For backwards compatibility
        args.layer_name = layer_name
        compute_layer(reportloader, feature_model, activations, metadata, args)


if __name__ == "__main__":
    desc = "Map a pretrained neural net to a time series of brain data (neurons or brains) using ridge regression."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--exp_name", required=True, help="Friendly name of experiment")
    parser.add_argument(
        "--features",
        default="gaborpyramid3d",
        type=str,
        help="What kind of features to use",
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
        "--consolidated",
        default=False,
        help="Consolidated batches",
        action="store_true",
    )
    parser.add_argument(
        "--subsample_layers",
        default=False,
        help="Subsample layers (saves disk space & mem)",
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

    args = parser.parse_args()
    main(args)