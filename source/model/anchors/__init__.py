from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AnchorsBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def positions(self):
        """Anchor positions, shape (N, D)."""
        raise NotImplementedError

    @abstractmethod
    def scores_for(self, coords: torch.Tensor, idx: torch.Tensor):
        """
        Args:
            coords: (B, D)
            idx:    (B, k) indices of selected neighbors
        Returns:
            (B, k) unnormalized scores s(x,i).
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_grid(cls, resolution: int, aspect_ratio: float = 1.0):
        """Factory: initialize anchors on a grid."""
        raise NotImplementedError

    @abstractmethod
    def show_stats(self):
        """Return dict of stats for logging."""
        raise NotImplementedError


from .gaussian import GaussianAnchors
from .scalar   import ScalarAnchors


__all__ = ["GaussianAnchors", "ScalarAnchors"]