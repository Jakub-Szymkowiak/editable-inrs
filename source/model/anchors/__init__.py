from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AnchorsBase(nn.Module, ABC):
    def __init__(self, positions: torch.Tensor, margin: float=1e-3):
        super().__init__()

        assert 0 <= margin < 0.1, \
            f"Margin must be in [0, 0.1); got margin={margin}"
        
        self._margin = margin

        assert positions.ndim == 2 and positions.size(-1) == 2, \
            "positions must be (N,2)"
        
        _positions_raw = self._positions_activation_inverse(positions)
        self._positions_raw = nn.Parameter(_positions_raw)

    def _positions_activation(self, latent: torch.Tensor):
        m = self._margin
        return m + (1.0 - 2.0 * m) * latent.sigmoid()

    def _positions_activation_inverse(self, pos: torch.Tensor):
        m = self._margin
        return torch.logit((pos - m) / (1.0 - 2.0 * m), eps=1e-6)

    @property
    def positions(self):
        return self._positions_activation(self._positions_raw)
    
    def expose_param_dict(self):
        return { "pos": self._positions_raw }

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