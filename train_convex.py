import argparse
import numpy as np
import pickle
import os
import time
import wandb

from training import compute_corr
from fmri_models import (get_dataset, 
                         get_feature_model, 
                         get_aggregator,
                         preprocess_data,
                         preprocess_data_consolidated,
                         get_projection_matrix)

from research_code.cka_step4 import cka

import torch

ff = .1

def save_to_wandb(results, weights, args, offline=False):
    if offline:
        os.environ["WANDB_MODE"] = "dryrun"
    else:
        os.environ["WANDB_MODE"] = "run"

    run = wandb.init(project="train_fmri_convex.py", 
                    config=vars(args),
                    reinit=True)
    config = wandb.config
    
    # This allows the info to be visible in the dashboard
    wandb.log(results)

    # This saves the whole results (no histogramming). Has to happen after log.
    wandb.run.summary.update(results)

    # Also save the best weights.
    if not args.no_save:
        weight_path = os.path.join(wandb.run.dir, 'optimal_weights.pkl')
        with open(weight_path, 'wb') as f:
            pickle.dump(weights, f)
    
        # These weights are big, only save if necessary
        wandb.save(weight_path)

        results['weight_path'] = weight_path
        
    out_path = os.path.join(wandb.run.dir, 'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    wandb.save(out_path)
    run.finish()

def compute_layer(trainloader, reportloader, feature_model, aggregator, 
                  activations, metadata, args):
    print(f"Processing layer {args.layer_name}")
    t0 = time.time()

    if args.consolidated:
        X, Y = preprocess_data_consolidated(trainloader, 
                            feature_model, 
                            aggregator,
                            activations, 
                            metadata,
                            args)
    else:
        X, Y = preprocess_data(trainloader, 
                            feature_model, 
                            aggregator,
                            activations, 
                            metadata,
                            args)

    if X is None:
        print(f"Skipping layer {args.layer_name}")
        return

    print(X.shape)
    print(Y.shape)

    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True) + ff

    Ym = Y.mean(axis=0, keepdims=True)

    # Use in-place operators instead of (X - m) / s to save memory.
    X.add_(-m)
    X.divide_(s)

    if args.pca > -1:
        V = get_projection_matrix(X, n=args.pca)
        X = torch.matmul(X, V)

    Y = Y - Ym
    Y = Y.to(device='cuda')
    X = X.to(device='cuda')

    print(X.std(axis=0)[:10])

    # Use k-fold cross-validation
    kfold = 5
    lambdas = np.logspace(0, 5, num=11)
    splits = (np.arange(X.shape[0]) / 100).astype(np.int) % kfold

    # Store predictions in main memory to prevent out-of-memory errors.
    Y_preds = torch.zeros(Y.shape[0], Y.shape[1], len(lambdas))

    for i in range(kfold):
        X_train, Y_train, X_test = X[splits != i, :], Y[splits != i, :], X[splits == i, :]
        C = X_train.T.matmul(X_train)

        for j, lambda_ in enumerate(lambdas):
            H = C + lambda_ * torch.eye(X_train.shape[1], device='cuda')
            # w = torch.inverse(H).matmul(X_train.T.matmul(Y_train))
            # w = torch.linalg.solve(H, X_train.T @ Y_train)
            w, _ = torch.solve(X_train.T @ Y_train, H)
            Y_pred = X_test.matmul(w)
            Y_preds[splits == i, :, j] = Y_pred.to(device='cpu')

    Y = Y.to(device='cpu')
    var_baseline = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y.reshape(Y.shape[0], Y.shape[1], 1) - Y_preds) ** 2).mean(0)
    r2_cvs = 1 - var_after / var_baseline.reshape((-1, 1))

    # Now we find the best lambdas
    best_lambdas = lambdas[np.argmax(r2_cvs, axis=1)]

    assert best_lambdas.size == Y.shape[1]

    # this is in case there's only one output. This is a no-op when best_lambdas 
    # is an array already.
    best_lambdas = np.array(best_lambdas) 
    r2_cv = np.array([r2_cvs[j, i] for j, i in enumerate(np.argmax(r2_cvs, axis=1))])

    if args.consolidated:
        X_report, Y_report = preprocess_data_consolidated(reportloader, 
                            feature_model, 
                            aggregator,
                            activations, 
                            metadata,
                            args)
    else:
        X_report, Y_report = preprocess_data(reportloader, 
                            feature_model, 
                            aggregator,
                            activations, 
                            metadata,
                            args)

    # Compute CKA for report fold
    cka_report = cka(X_report, Y_report)

    X_report = (X_report - m) / s

    if args.pca > -1:
        X_report = torch.matmul(X_report, V)

    Y_report = Y_report - Ym
    X_report = X_report.to(device='cuda')
    Y_report = Y_report.to(device='cuda')
    
    best_lambda_vals = np.unique(best_lambdas)

    Y_preds = torch.zeros(Y_report.shape, device='cuda')

    best_W = np.zeros((X.shape[1], Y_report.shape[1]))
    Y = Y.to(device='cuda')

    C = X.T.matmul(X)
    for lambda_ in best_lambda_vals:
        H = C + lambda_ * torch.eye(X.shape[1], device='cuda')
        # w = torch.inverse(H).matmul(X.T.matmul(Y))
        # This would be ideal, but it's not in torch stable yet.
        # w = torch.linalg.solve(H, X.T @ Y)
        w, _ = torch.solve(X.T @ Y, H)
        Y_pred = X_report.matmul(w)
        to_replace = (best_lambdas == lambda_)

        # In case to_replace is a scalar
        to_replace = to_replace.reshape(to_replace.size)
        
        Y_preds[:, to_replace] = Y_pred[:, to_replace]
        best_W[:, to_replace] = w[:, to_replace].cpu().detach().numpy()

    var_baseline = ((Y_report - Y_report.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y_report - Y_preds) ** 2).mean(0)
    r2_report = 1 - var_after / var_baseline

    corrs_report = compute_corr(Y_report, Y_preds)

    weights = {
        'W': best_W
    }

    results = {
        'r2_cvs': r2_cvs.cpu().detach().numpy(),
        'r2_report': r2_report.cpu().detach().numpy(),
        'corrs_report': corrs_report.cpu().detach().numpy(),
        'corrs_report_mean': corrs_report.cpu().detach().numpy().mean(),
        'corrs_report_median': np.median(corrs_report.cpu().detach().numpy()),
        'w_shape': w.shape,
        'feature_mean': m.squeeze().cpu().detach().numpy(),
        'feature_std': s.squeeze().cpu().detach().numpy(),
        'fit_time': time.time() - t0,
        'cka_report': cka_report.item(),
        'layer': args.layer,
        'subset': args.subset,
    }

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

    runs = api.runs("pmin/train_fmri_convex.py", 
                    {"$and": 
                    [
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
                    }
                )
    return len(runs) >= len(metadata['layers'])


def main(args):
    print("Fitting model")
    print(args)
    device = 'cuda'

    try:
        os.makedirs(args.ckpt_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(args.cache_root)
    except FileExistsError:
        pass

    trainset = get_dataset(args, 'traintune')
    reportset = get_dataset(args, 'report')

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False,
                                              pin_memory=True
                                              )

    reportloader = torch.utils.data.DataLoader(reportset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False,
                                             pin_memory=True
                                             )

    args.ntau = trainset.ntau
    feature_model, activations, metadata = get_feature_model(args)

    if args.skip_existing:
        exists = check_existing(args, metadata)
        if exists:
            print(">>> Run already exists, skipping")
            return

    aggregator = get_aggregator(metadata, args)
    feature_model.to(device=device)

    # Do this for every layer under the sun.
    for layer_num, layer_name in enumerate(metadata['layers'].keys()):
        args.layer = layer_num  # For backwards compatibility
        args.layer_name = layer_name
        compute_layer(trainloader, 
                      reportloader, 
                      feature_model, 
                      aggregator, 
                      activations, 
                      metadata, 
                      args)


if __name__ == "__main__":
    desc = "Map a pretrained neural net to a time series of brain data (neurons or brains) using ridge regression."
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--exp_name", required=True, help='Friendly name of experiment')
    parser.add_argument("--features", default='gaborpyramid3d', type=str, help='What kind of features to use')
    parser.add_argument("--aggregator", default='average', type=str, help='What kind of aggregator to use')
    parser.add_argument("--aggregator_sz", default=8, type=int, help='The size the aggregator will be used with')
    parser.add_argument("--batch_size", default=4, type=int, help='Batch size')
    parser.add_argument("--pca", default=-1, type=int, help='Size of PCA before model fit (if applicable)')

    parser.add_argument("--no_wandb", default=False, help='Skip using W&B', action='store_true')
    parser.add_argument("--no_save", default=False, help='Skip saving weights', action='store_true')
    parser.add_argument("--skip_existing", default=False, help='Skip existing runs', action='store_true')
    parser.add_argument("--consolidated", default=False, help='Consolidated batches', action='store_true')
    parser.add_argument("--autotune", default=False, help='Tune the batch size to maximize memory consumption', action='store_true')
    parser.add_argument("--subsample_layers", default=False, help='Subsample layers (saves disk space & mem)', action='store_true')
    
    parser.add_argument("--dataset", default='vim2', help='Dataset (currently vim2, pvc4)')
    parser.add_argument("--subset", default='s1', help='Either subject name or neuron num')
    parser.add_argument("--data_root", default='./data_derived', help='Data path')
    parser.add_argument("--cache_root", default='./cache', help='Precomputed cache path')
    parser.add_argument("--slowfast_root", default='', help='Path where SlowFast is installed')
    parser.add_argument("--ckpt_root", default='./pretrained', help='Path where trained model checkpoints will be downloaded')
    
    args = parser.parse_args()
    main(args)