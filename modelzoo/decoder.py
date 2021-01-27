from torch import nn

class Average(nn.Module):
    def __init__(self, 
                 subnet, 
                 noutputs,
                 nclasses,
                 nfeats,
                 threed
                 ):
        super().__init__()
        self.subnet = subnet
        self.noutputs = noutputs
        self.nclasses = nclasses
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(self.nfeats, 
                                         self.noutputs * self.nclasses, 
                                         bias=True)

    def forward(self, X):
        X = self.subnet(X)
        Xs = X.mean(axis=4).mean(axis=3).mean(axis=2)

        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        Y = self.fully_connected.forward(Xs)
        if self.nclasses > 1:
            Y = Y.reshape(Y.shape[0], self.nclasses, self.noutputs)
        return Y

class Center(nn.Module):
    def __init__(self, 
                 subnet, 
                 noutputs,
                 nclasses,
                 nfeats,
                 threed
                 ):
        super().__init__()
        self.subnet = subnet
        self.noutputs = noutputs
        self.nclasses = nclasses
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(self.nfeats, 
                                         self.noutputs * self.nclasses, 
                                         bias=True)

    def forward(self, X):
        X = self.subnet(X)
        ct, cy, cx = X.shape[2] // 2, X.shape[3] // 2, X.shape[4] // 2, 
        Xs = X[:, :, ct, cy, cx]

        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        Y = self.fully_connected.forward(Xs)
        if self.nclasses > 1:
            Y = Y.reshape(Y.shape[0], self.nclasses, self.noutputs)
        return Y