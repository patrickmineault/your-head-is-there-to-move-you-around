import numpy as np
import time
import torch

import sklearn
import sklearn.model_selection
import sklearn.linear_model

from training import compute_corr


def compute_ridge_estimate(X, Y, X_report, Y_report, splits):
    kfold = splits.max() + 1
    Y = Y.to(device="cuda")
    X = X.to(device="cuda")

    print(X.std(axis=0)[:10])

    # Store predictions in main memory to prevent out-of-memory errors.
    lambdas = np.logspace(0, 5, num=11)
    Y_preds = torch.zeros(Y.shape[0], Y.shape[1], len(lambdas))

    for i in range(kfold):
        X_train, Y_train, X_test = (
            X[splits != i, :],
            Y[splits != i, :],
            X[splits == i, :],
        )
        C = X_train.T.matmul(X_train)

        for j, lambda_ in enumerate(lambdas):
            H = C + lambda_ * torch.eye(X_train.shape[1], device="cuda")
            w = torch.inverse(H).matmul(X_train.T.matmul(Y_train))
            # w = torch.linalg.solve(H, X_train.T @ Y_train)
            # w, _ = torch.solve(X_train.T @ Y_train, H)
            Y_pred = X_test.matmul(w)
            Y_preds[splits == i, :, j] = Y_pred.to(device="cpu")

    Y = Y.to(device="cpu")
    var_baseline = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y.reshape(Y.shape[0], Y.shape[1], 1) - Y_preds) ** 2).mean(0)
    r2_cvs = 1 - var_after / var_baseline.reshape((-1, 1))

    # Now we find the best lambdas
    best_lambdas = lambdas[np.argmax(r2_cvs, axis=1)]

    assert best_lambdas.size == Y.shape[1]

    # this is in case there's only one output. This is a no-op when best_lambdas
    # is an array already.
    best_lambdas = np.array(best_lambdas)

    X_report = X_report.to(device="cuda")
    Y_report = Y_report.to(device="cuda")

    best_lambda_vals = np.unique(best_lambdas)

    Y_preds = torch.zeros(Y_report.shape, device="cuda")

    best_W = np.zeros((X.shape[1], Y_report.shape[1]))
    Y = Y.to(device="cuda")

    C = X.T.matmul(X)
    for lambda_ in best_lambda_vals:
        H = C + lambda_ * torch.eye(X.shape[1], device="cuda")
        w = torch.inverse(H).matmul(X.T.matmul(Y))
        # This would be ideal, but it's not in torch stable yet.
        # w = torch.linalg.solve(H, X.T @ Y)
        # w, _ = torch.solve(X.T @ Y, H)
        Y_pred = X_report.matmul(w)
        to_replace = best_lambdas == lambda_

        # In case to_replace is a scalar
        to_replace = to_replace.reshape(to_replace.size)

        Y_preds[:, to_replace] = Y_pred[:, to_replace]
        best_W[:, to_replace] = w[:, to_replace].cpu().detach().numpy()

    var_baseline = ((Y_report - Y_report.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y_report - Y_preds) ** 2).mean(0)
    r2_report = 1 - var_after / var_baseline

    corrs_report = compute_corr(Y_report, Y_preds)

    weights = {
        "W": best_W,
        "Y_preds": Y_preds,
    }

    results = {
        "r2_cvs": r2_cvs.cpu().detach().numpy(),
        "r2_report": r2_report.cpu().detach().numpy(),
        "corrs_report": corrs_report.cpu().detach().numpy(),
        "corrs_report_mean": corrs_report.cpu().detach().numpy().mean(),
        "corrs_report_median": np.median(corrs_report.cpu().detach().numpy()),
        "w_shape": w.shape,
    }

    return results, weights


def compute_boosting_estimate(X, Y, X_report, Y_report, splits):
    alpha = 0.1
    max_iter = 100
    kfold = splits.max() + 1

    t = torch.cuda.get_device_properties(0).total_memory
    
    nums = X.numel() * 4 * 2.5
    print(t, nums)
    if t > X.numel() * nums:
        # CUDA is much faster, but has less memory.
        target = "cuda"
    else:
        target = "cpu"

    # Store predictions in main memory to prevent out-of-memory errors.
    Y_preds = torch.zeros(Y.shape[0], Y.shape[1], max_iter, dtype=torch.float32)

    for i in range(kfold):
        X_train, Y_train, X_test = (
            X[splits != i, :],
            Y[splits != i, :],
            X[splits == i, :],
        )

        X_train = X_train.to(device=target)
        Y_train = Y_train.to(device=target)
        X_test = X_test.to(device=target)

        m = X_train.mean(axis=0, keepdims=True)
        s = X_train.std(axis=0, keepdims=True) + 1e-6

        X_train.add_(-m)
        X_train.divide_(s)

        w = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.float32, device=target)
        R = Y_train - Y_train.mean(axis=0, keepdims=True)

        for j in range(max_iter):
            dw = (X_train.T @ R) / X_train.shape[0]
            the_best = abs(dw).argmax(axis=0)
            w[the_best, np.arange(Y.shape[1])] += (
                alpha * dw[the_best, np.arange(Y.shape[1])]
            )

            R = Y_train - X_train @ w
            R = R - R.mean(axis=0, keepdims=True)
            R = R / R.std(axis=0, keepdims=True)

            Y_pred = X_test @ w
            Y_preds[splits == i, :, j] = Y_pred.to(device="cpu")

        del X_train, Y_train, X_test

    Y = Y.to(device="cpu")
    var_baseline = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y.reshape(Y.shape[0], Y.shape[1], 1) - Y_preds) ** 2).mean(0)
    r2_cvs = 1 - var_after / var_baseline.reshape((-1, 1))

    # Now we find the best number of iterations.
    best_iters = np.argmax(r2_cvs, axis=1).cpu().detach().numpy()

    # this is in case there's only one output. This is a no-op when best_lambdas
    # is an array already.
    best_iters = np.array(best_iters)

    X_report = X_report - X_report.mean(axis=0, keepdims=True)
    Y_report = Y_report - Y_report.mean(axis=0, keepdims=True)

    X = X.to(device=target)
    X_report = X_report.to(device=target)
    Y_report = Y_report.to(device=target)

    Y = Y.to(device=target)
    R = Y - Y.mean(axis=0, keepdims=True)
    R = R / R.std(axis=0, keepdims=True)

    w = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.float32, device=target)

    for j in range(max(best_iters)):
        dw = X.T @ R / X.shape[0]
        the_best = abs(dw).argmax(axis=0)

        # Only advance w when necessary
        w[the_best[j <= best_iters], np.arange(Y.shape[1])[j <= best_iters]] += (
            alpha
            * dw[the_best[j <= best_iters], np.arange(Y.shape[1])[j <= best_iters]]
        )

        R = Y - X @ w
        R = R - R.mean(axis=0, keepdims=True)
        R = R / R.std(axis=0, keepdims=True)

    Y_preds = X_report @ w

    var_baseline = ((Y_report - Y_report.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y_report - Y_preds) ** 2).mean(0)
    r2_report = 1 - var_after / var_baseline

    corrs_report = compute_corr(Y_report, Y_preds)

    results = {
        "r2_cvs": r2_cvs.cpu().detach().numpy(),
        "r2_report": r2_report.cpu().detach().numpy(),
        "corrs_report": corrs_report.cpu().detach().numpy(),
        "corrs_report_mean": corrs_report.cpu().detach().numpy().mean(),
        "corrs_report_median": np.median(corrs_report.cpu().detach().numpy()),
        "w_shape": w.shape,
    }

    weights = {"W": w.cpu().detach().numpy(), "Y_preds": Y_preds.cpu().detach().numpy()}

    return results, weights


def compute_l1_estimate(X, Y, X_report, Y_report, splits):
    if Y.shape[1] > 1:
        raise NotImplementedError("Y.shape[1] > 1 not implemented")

    cv = sklearn.model_selection.PredefinedSplit(splits)
    std = sklearn.preprocessing.StandardScaler()

    X = std.fit_transform(X.detach().cpu().numpy())
    X_report = std.transform(X_report.detach().cpu().numpy())
    Y = Y.detach().cpu().numpy()
    Y_report = Y_report.detach().cpu().numpy()

    model = sklearn.linear_model.LassoCV(cv=cv, n_alphas=25)
    model.fit(X, Y.ravel())

    Y_preds = model.predict(X).reshape((-1, 1))

    var_baseline = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y.reshape(Y.shape[0], Y.shape[1], 1) - Y_preds) ** 2).mean(0)
    r2_cvs = 1 - var_after / var_baseline.reshape((-1, 1))

    Y_preds = model.predict(X_report).reshape((-1, 1))

    var_baseline = ((Y_report - Y_report.mean(axis=0, keepdims=True)) ** 2).mean(0)
    var_after = ((Y_report - Y_preds) ** 2).mean(0)
    r2_report = 1 - var_after / var_baseline

    corrs_report = compute_corr(
        torch.tensor(Y_report, dtype=torch.float32, device="cuda"),
        torch.tensor(Y_preds, dtype=torch.float32, device="cuda"),
    )

    results = {
        "r2_cvs": r2_cvs,
        "r2_report": r2_report,
        "corrs_report": corrs_report.cpu().detach().numpy(),
        "corrs_report_mean": corrs_report.cpu().detach().numpy().mean(),
        "corrs_report_median": np.median(corrs_report.cpu().detach().numpy()),
        "w_shape": model.coef_.shape,
    }

    weights = {"W": model.coef_, "Y_preds": Y_preds}

    return results, weights
