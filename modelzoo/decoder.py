from torch import nn

class Average(nn.Module):
    def __init__(self, 
                 subnet, 
                 noutputs,
                 nfeats,
                 threed
                 ):
        super().__init__()
        self.subnet = subnet
        self.noutputs = noutputs
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(self.nfeats, self.noutputs, bias=True)

    def forward(self, X):
        X = self.subnet(X)
        Xs = X.mean(axis=4).mean(axis=3).mean(axis=2)
        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        return self.fully_connected.forward(Xs)