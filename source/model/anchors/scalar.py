import math
import torch
import torch.nn as nn

from . import AnchorsBase


class ScalarAnchors(AnchorsBase):
    def __init__(
                self, 
                positions: torch.Tensor, 
                weights:   torch.Tensor, 
                eps:       float = 1e-12
        ):
        
        super().__init__(positions=positions)

        assert weights.ndim == 1 and weights.size(0) == positions.size(0), \
            "weights must be (N,)"
        
        self._weights   = nn.Parameter(weights)
        self.eps = eps

    @property
    def positions(self):
        return self._positions

    def forward(self):
        return self._positions, self._weights
    
    def expose_param_dict(self):
        return super().expose_param_dict() | { 
            "weights": self._weights
        }

    @classmethod
    def from_grid(cls, resolution: int, aspect_ratio: float = 1.0):
        ny, nx = resolution, round(aspect_ratio * resolution)

        xs = torch.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx, dtype=torch.float32)
        ys = torch.linspace(0.5 / ny, 1.0 - 0.5 / ny, ny, dtype=torch.float32)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")

        N = ny * nx
        w0 = 1.0 / math.sqrt(N)

        positions = torch.stack([xv.flatten(), yv.flatten()], dim=-1)
        weights   = torch.full((N,), w0, dtype=torch.float32)

        return cls(positions, weights)

    def scores_for(
            self, 
            coords:  torch.Tensor, 
            idx:     torch.Tensor, 
            sqdist:  torch.Tensor,
        ):
        """
        Compute isotropic Gaussian scores for query-anchor pairs, based on Euclidean distance.

        Args:
            coords:  (B,2) float32 queries
            idx:     (B,k) long indices of neighbors
            sqdist:  precomputed Euclidean squared distances
        Returns:
            (B,k) unnormalized scores
        """

        assert coords.ndim == 2 and coords.size(-1) == 2
        assert idx.ndim == 2 and idx.size(0) == coords.size(0)

        precision = (self._weights[idx] ** 2).clamp_min(self.eps).reciprocal()
        return torch.exp(-0.5 * precision * sqdist)

    def show_stats(self): return {
            "anchors/num":    self.positions.size(0),
            "anchors/mean_x": self.positions[:, 0].mean().item(),
            "anchors/mean_y": self.positions[:, 1].mean().item(),
            "anchors/mean_w": self._weights.mean().item(),
            "anchors/std_w":  self._weights.std().item(),
        }