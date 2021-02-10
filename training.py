import datetime
import numpy as np
import os
import scipy.stats
import torch


def bootstrap_ycols(fun, X, Y, nboot=100):
    vals = []
    for _ in range(nboot):
        idx = np.random.randint(low=0, high=Y.shape[1], size=Y.shape[1])
        vals.append(fun(X, Y[:, idx]))

    val0 = fun(X, Y)

    return val0, np.std(vals)


def bootstrap_rdm_distance(X, Y, method, nboot=100):
    rdm_X = compute_rdm(X)

    ds = []
    for _ in range(nboot):
        idx = np.random.randint(low=0, high=Y.shape[1], size=Y.shape[1])
        rdm_Y = compute_rdm(Y[:, idx])
        ds.append(compute_rdm_distance(rdm_X, rdm_Y, method))

    rdm_Y = compute_rdm(Y)
    m = compute_rdm_distance(rdm_X, rdm_Y, method)

    return m, np.std(ds)


def compute_rdm(X):
    rdm = 1 - np.corrcoef(X.detach().cpu().numpy())
    assert rdm.shape[0] == rdm.shape[1]
    assert rdm.shape[0] == X.shape[0]
    return rdm


def compute_rdm_distance(rdm_0, rdm_1, method="pearson"):
    assert rdm_0.shape[0] == rdm_0.shape[1]
    ii, jj = np.triu_indices(rdm_0.shape[0], k=1)
    if method == "pearson":
        corr = np.corrcoef(rdm_0[ii, jj], rdm_1[ii, jj])
        assert corr.shape[0] == 2
        corr = corr[0, 1]
    elif method == "cosine":
        corr = (rdm_0[ii, jj] * rdm_1[ii, jj]).sum()
        corr /= np.sqrt((rdm_0[ii, jj] ** 2).sum() * (rdm_1[ii, jj] ** 2).sum())
    elif method == "rank":
        corr = scipy.stats.spearmanr(rdm_0[ii, jj], rdm_1[ii, jj]).correlation
    elif method == "r2":
        corr = 1 - (
            ((rdm_0[ii, jj] - rdm_1[ii, jj]) ** 2).sum()
            / ((rdm_1[ii, jj] - rdm_1[ii, jj].mean()) ** 2).sum()
        )
    else:
        raise NotImplementedError(f"{method} not implemented")

    return 1 - corr


def compute_rdm_corr(rdm_0, rdm_1):
    assert rdm_0.shape[0] == rdm_0.shape[0]
    ii, jj = torch.triu_indices(rdm_0.shape[0], rdm_0.shape[1], offset=1)
    corr = np.corrcoef(rdm_0[ii, jj], rdm_1[ii, jj])
    assert corr.shape[0] == 2
    return 1 - corr[0, 1]


def compute_rdm_cosine(rdm_0, rdm_1):
    assert rdm_0.shape[0] == rdm_0.shape[0]
    ii, jj = torch.triu_indices(rdm_0.shape[0], rdm_0.shape[1], offset=1)
    corr = (rdm_0[ii, jj] * rdm_1[ii, jj]).sum()
    corr /= torch.sqrt((rdm_0[ii, jj] ** 2).sum() * (rdm_1[ii, jj] ** 2).sum())
    assert corr.shape[0] == 2
    return 1 - corr[0, 1]


def compute_corr(Yl, Yp):
    if torch.any(torch.isnan(Yl)) or torch.any(torch.isnan(Yp)):
        corr = torch.zeros(Yl.shape[1], device=Yl.device)
        for i in range(Yl.shape[1]):
            yl, yp = (Yl[:, i].cpu().detach().numpy(), Yp[:, i].cpu().detach().numpy())
            yl = yl[~np.isnan(yl)]
            yp = yp[~np.isnan(yp)]
            corr[i] = np.corrcoef(yl, yp)[0, 1]
    else:
        Yl = Yl - Yl.mean(axis=0, keepdims=True)
        Yp = Yp - Yp.mean(axis=0, keepdims=True)
        Yl = Yl / torch.linalg.norm(Yl, axis=0, keepdims=True)
        Yp = Yp / torch.linalg.norm(Yp, axis=0, keepdims=True)
        corr = (Yl * Yp).sum(axis=0)
    return corr


def get_all_layers(net, prefix=[]):
    if hasattr(net, "_modules"):
        lst = []
        for name, layer in net._modules.items():
            full_name = "_".join((prefix + [name]))
            lst = lst + [(full_name, layer)] + get_all_layers(layer, prefix + [name])
        return lst
    else:
        return []


def save_state(net, title, output_dir):
    datestr = str(datetime.datetime.now()).replace(":", "-")
    filename = os.path.join(output_dir, f"{title}-{datestr}.pt")
    torch.save(net.state_dict(), filename)
    return filename