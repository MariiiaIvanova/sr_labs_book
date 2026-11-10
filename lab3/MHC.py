import torch
import torch.nn as nn

class ManifoldConstrainedHyperConnection(nn.Module):
    """Static learnable stream mixing with Sinkhorn-Knopp projection."""
    def __init__(self, channels, num_streams=4, sinkhorn_iters=3):
        super().__init__()
        if channels % num_streams:
            raise ValueError(f"{channels=} must be divisible by {num_streams=}")
        self.channels = channels
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        # Identity-biased initialization: engineering choice, not specified in the paper.
        theta = torch.full((num_streams, num_streams), -2.0)
        theta.fill_diagonal_(2.0)
        self.theta = nn.Parameter(theta)

    def mixing_matrix(self, dtype=None, device=None):
        a = torch.exp(self.theta.float() - self.theta.float().amax())
        for _ in range(self.sinkhorn_iters):
            a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-12)
            a = a / a.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return a.to(device=device or a.device, dtype=dtype or a.dtype)

    def forward(self, shortcut, transformed):
        if shortcut.shape != transformed.shape:
            raise ValueError("mHC shortcut and residual branch must have equal shapes")
        b, c = shortcut.shape[:2]
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        shape = (b, self.num_streams, c // self.num_streams, *shortcut.shape[2:])
        hs = shortcut.reshape(shape)
        fs = transformed.reshape(shape)
        w = self.mixing_matrix(dtype=shortcut.dtype, device=shortcut.device)
        mixed = torch.einsum("ij,bj...->bi...", w, hs)
        
        return (mixed + fs).reshape_as(shortcut)
