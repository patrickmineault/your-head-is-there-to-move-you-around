import argparse
import numpy as np
import pickle
import os
import time
import wandb

from training import compute_corr
from fmri_models import (get_dataset, 
                         get_feature_model, 
                         get_readout_model,
                         get_aggregator,
                         preprocess_data)

import torch

ff = .1

def main(args):
    print("Fitting model")
    print(args)
    t0 = time.time()
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

    feature_model, activations, metadata = get_feature_model(args)
    aggregator = get_aggregator(metadata, args)
    feature_model.to(device=device)

    X, Y = preprocess_data(trainloader, 
                           feature_model, 
                           aggregator,
                           activations, 
                           metadata,
                           args)

    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True) + ff

    X = (X - m) / s

    # Use k-fold cross-validation
    kfold = 5
    lambdas = np.array([1, 10, 100, 1000, 10000, 100000])
    splits = (np.arange(X.shape[0]) / 100).astype(np.int) % kfold

    # Store predictions in main memory to prevent out-of-memory errors.
    Y_preds = torch.zeros(Y.shape[0], Y.shape[1], len(lambdas))

    for i in range(kfold):
        X_train, Y_train, X_test, Y_test = X[splits != i, :], Y[splits != i, :], X[splits == i, :], Y[splits == i, :]
        for j, lambda_ in enumerate(lambdas):
            H = X_train.T.matmul(X_train) + lambda_ * torch.eye(X_train.shape[1], device='cuda')
            w = torch.inverse(H).matmul(X_train.T.matmul(Y_train))
            Y_pred = X_test.matmul(w)
            Y_preds[splits == i, :, j] = Y_pred.to(device='cpu')

    Y = Y.to(device='cpu')
    var_baseline = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y.reshape(Y.shape[0], Y.shape[1], 1) - Y_preds) ** 2).mean(0)
    r2_cvs = 1 - var_after / var_baseline.reshape((-1, 1))

    # Now we find the best lambdas
    best_lambdas = lambdas[np.argmax(r2_cvs, axis=1)]
    r2_cv = np.array([r2_cvs[j, i] for j, i in enumerate(np.argmax(r2_cvs, axis=1))])

    X_report, Y_report = preprocess_data(reportloader, 
                           feature_model, 
                           aggregator,
                           activations, 
                           metadata,
                           args)

    X_report = (X_report - m) / s

    best_lambda_vals = np.unique(best_lambdas)

    Y_preds = torch.zeros(Y_report.shape, device='cuda')

    best_W = np.zeros((X.shape[1], Y_report.shape[1]))
    Y = Y.to(device='cuda')
    for lambda_ in best_lambda_vals:
        H = X.T.matmul(X) + lambda_ * torch.eye(X.shape[1], device='cuda')
        w = torch.inverse(H).matmul(X.T.matmul(Y))
        Y_pred = X_report.matmul(w)
        to_replace = (best_lambdas == lambda_)
        
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
        'best_lambdas': best_lambdas,
        'r2_cv': r2_cv,
        'r2_cvs': r2_cvs.cpu().detach().numpy(),
        'r2_report': r2_report.cpu().detach().numpy(),
        'corrs_report': corrs_report.cpu().detach().numpy(),
        'corrs_report_mean': corrs_report.cpu().detach().numpy().mean(),
        'corrs_report_median': np.median(corrs_report.cpu().detach().numpy()),
        'w_shape': w.shape,
        'feature_mean': m.squeeze().cpu().detach().numpy(),
        'feature_std': s.squeeze().cpu().detach().numpy(),
        'fit_time': time.time() - t0,
    }

    if not args.no_wandb:
        wandb.init(project="train_fmri_convex.py", 
                config=vars(args))
        config = wandb.config
        
        # This allows the info to be visible in the dashboard
        wandb.log(results)

        # This saves the whole results (no histogramming). Has to happen after log.
        wandb.run.summary.update(results)

        # Also save the best weights.
        out_path = os.path.join(wandb.run.dir, 'optimal_weights.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(weights, f)

        wandb.save(out_path)

        out_path = os.path.join(wandb.run.dir, 'results.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)

        wandb.save(out_path)


if __name__ == "__main__":
    desc = "Map a neural net representation to fmri"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--exp_name", required=True, help='Friendly name of experiment')
    parser.add_argument("--width", default=112, type=int, help='Width the video will be resized to')
    parser.add_argument("--features", default='gaborpyramid3d', type=str, help='What kind of features to use')
    parser.add_argument("--layer", default=0, type=int, help='Which layer to use as features')
    parser.add_argument("--aggregator", default='average', type=str, help='What kind of aggregator to use')
    parser.add_argument("--batch_size", default=4, type=int, help='Batch size')

    parser.add_argument("--no_sample", default=False, help='Whether to use a normal gaussian layer rather than a sampled one', action='store_true')
    parser.add_argument("--no_wandb", default=False, help='Skip using W&B', action='store_true')
    
    parser.add_argument("--dataset", default='vim2', help='Dataset (currently only vim2)')
    parser.add_argument("--subject", default='s1', help='Dataset (for vim2: s1, s2 or s3)')
    parser.add_argument("--data_root", default='./data', help='Data path')
    parser.add_argument("--cache_root", default='./cache', help='Precomputed cache path')
    parser.add_argument("--slowfast_root", default='', help='Path where SlowFast is installed')
    parser.add_argument("--ckpt_root", default='./pretrained', help='Path where trained model checkpoints will be downloaded')
    
    args = parser.parse_args()
    main(args)