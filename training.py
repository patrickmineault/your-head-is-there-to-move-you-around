import datetime
import numpy as np
import os
import torch

def compute_corr(Yl, Yp):
    if torch.any(torch.isnan(Yl)) or torch.any(torch.isnan(Yp)):
        corr = torch.zeros(Yl.shape[1], device=Yl.device)
        for i in range(Yl.shape[1]):
            yl, yp = (Yl[:, i].cpu().detach().numpy(), 
                    Yp[:, i].cpu().detach().numpy())
            yl = yl[~np.isnan(yl)]
            yp = yp[~np.isnan(yp)]
            corr[i] = np.corrcoef(yl, yp)[0, 1]
    else:
        Yl = (Yl - Yl.mean(axis=0, keepdims=True))
        Yp = (Yp - Yp.mean(axis=0, keepdims=True))
        Yl = Yl / torch.linalg.norm(Yl, axis=0, keepdims=True)
        Yp = Yp / torch.linalg.norm(Yp, axis=0, keepdims=True)
        corr = (Yl * Yp).sum(axis=0)
    return corr


def get_all_layers(net, prefix=[]):
    if hasattr(net, '_modules'):
        lst = []
        for name, layer in net._modules.items():
            full_name = '_'.join((prefix + [name]))
            lst = lst + [(full_name, layer)] + get_all_layers(layer, prefix + [name])
        return lst
    else:
        return []

def save_state(net, title, output_dir):
    datestr = str(datetime.datetime.now()).replace(':', '-')
    filename = os.path.join(output_dir, f'{title}-{datestr}.pt')
    torch.save(net.state_dict(), filename)
    return filename