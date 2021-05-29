from torch import nn


class Average(nn.Module):
    def __init__(self, noutputs, nclasses, nfeats, threed):
        super().__init__()
        self.noutputs = noutputs
        self.nclasses = nclasses
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(
            self.nfeats, self.noutputs * self.nclasses, bias=True
        )

    def forward(self, X):
        Xs = X.mean(axis=4).mean(axis=3).mean(axis=2)

        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        Y = self.fully_connected.forward(Xs)
        if self.nclasses > 1:
            Y = Y.reshape(Y.shape[0], self.nclasses, self.noutputs)
        return Y


class Point(nn.Module):
    def __init__(self, noutputs, nclasses, nfeats, threed):
        super().__init__()
        self.noutputs = noutputs
        self.nclasses = nclasses
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(
            self.nfeats, self.noutputs * self.nclasses, bias=True
        )

    def forward(self, X):
        ct, cy, cx = (
            X.shape[2] // 2,
            X.shape[3] // 2,
            X.shape[4] // 2,
        )
        Xs = (
            X[:, :, ct, cy, cx]
            + X[:, :, ct, cy - 1, cx]
            + X[:, :, ct, cy, cx - 1]
            + X[:, :, ct, cy - 1, cx - 1]
        ) / 4

        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        Y = self.fully_connected.forward(Xs)
        if self.nclasses > 1:
            Y = Y.reshape(Y.shape[0], self.nclasses, self.noutputs)
        return Y


class Center(nn.Module):
    def __init__(self, noutputs, nclasses, nfeats, threed):
        super().__init__()
        self.noutputs = noutputs
        self.nclasses = nclasses
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(
            self.nfeats, self.noutputs * self.nclasses, bias=True
        )

    def forward(self, X):
        ct, cy, cx = (
            X.shape[2] // 2,
            slice(X.shape[3] // 4, int(3 * X.shape[3] / 4)),
            slice(X.shape[4] // 4, int(3 * X.shape[4] / 4)),
        )

        Xs = X[:, :, ct, cy, cx].mean(3).mean(2)

        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        Y = self.fully_connected.forward(Xs)
        if self.nclasses > 1:
            Y = Y.reshape(Y.shape[0], self.nclasses, self.noutputs)
        return Y


class Weighted(nn.Module):
    def __init__(self, noutputs, nclasses, nfeats, threed):
        super().__init__()
        self.noutputs = noutputs
        self.nclasses = nclasses
        self.nfeats = nfeats
        self.threed = threed
        self.fully_connected = nn.Linear(
            self.nfeats * 14 * 14, self.noutputs * self.nclasses, bias=True
        )

    def forward(self, X):
        ct, cy, cx = (
            X.shape[2] // 2,
            slice(X.shape[3] // 4, int(3 * X.shape[3] / 4)),
            slice(X.shape[4] // 4, int(3 * X.shape[4] / 4)),
        )

        Xs = X[:, :, ct, cy, cx].reshape(X.shape[0], -1)

        assert Xs.ndim == 2
        assert Xs.shape[1] == self.nfeats
        Y = self.fully_connected.forward(Xs)
        if self.nclasses > 1:
            Y = Y.reshape(Y.shape[0], self.nclasses, self.noutputs)
        return Y