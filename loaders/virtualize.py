import torch
import torch.nn.functional as F

def list_transformations(lst="rot"):
    transformations = [
         (), 
         ('rot90',), 
         ('rot180',), 
         ('rot270',),
         ('mirror',), 
         ('mirror', 'rot90',), 
         ('mirror', 'rot180',), 
         ('mirror', 'rot270',),
         ('scale',), 
         ('scale', 'rot90',), 
         ('scale', 'rot180',), 
         ('scale', 'rot270',),
         ('scale', 'mirror',), 
         ('scale', 'mirror', 'rot90',), 
         ('scale', 'mirror', 'rot180',), 
         ('scale', 'mirror', 'rot270',)
         ]

    if lst == 'rot':
        return transformations[:4]
    elif lst == 'all':
        return transformations
    else:
        raise NotImplementedError("Transformations not implemented")

def transform(X, trans):
    if 'scale' in trans:
        old_sz = X.shape[-1]
        rg = slice(int(old_sz * 1/6), int(old_sz * 5/6))
        X = F.interpolate(X[..., rg, rg], 
                            [old_sz, old_sz], 
                            align_corners=False,
                            mode='bilinear')

    if 'mirror' in trans:
        X = torch.flip(X, [X.ndim - 1])

    if 'rot90' in trans:
        X = torch.rot90(X, 1, (X.ndim - 2, X.ndim - 1))

    if 'rot180' in trans:
        X = torch.rot90(X, 2, (X.ndim - 2, X.ndim - 1))

    if 'rot270' in trans:
        X = torch.rot90(X, 3, (X.ndim - 2, X.ndim - 1))

    return X