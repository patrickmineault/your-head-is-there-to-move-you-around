import torch
from torchvision.transforms import ColorJitter, GaussianBlur

class ThreedExposure(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, X):
        assert X.ndim == 5
        assert X.shape[1] == 3
        X.add_(torch.zeros(X.shape[0], 1, 1, 1, 1, device=X.device).uniform_(-self.brightness, self.brightness))
        X.multiply_(torch.zeros(X.shape[0], 1, 1, 1, 1, device=X.device).uniform_(1-self.contrast, 1+self.contrast))
        return X


class ThreedGaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.t = GaussianBlur(kernel_size, sigma)

    def forward(self, X):
        return self.t(X.reshape((-1, X.shape[-3], X.shape[-2], X.shape[-1]))).reshape(X.shape)
