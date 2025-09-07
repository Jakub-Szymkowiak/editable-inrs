import math
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, positions: torch.Tensor, weights: torch.Tensor):
        super().__init__()

        self.positions = nn.Parameter(positions)
        self.weights   = nn.Parameter(weights)

    def forward(self):
        return self.positions, self.weights

    @classmethod
    def from_grid(cls, resolution: int, aspect_ratio: float=1.0):
        ny, nx = resolution, round(aspect_ratio * resolution)

        xs = torch.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
        ys = torch.linspace(0.5 / ny, 1 - 0.5 / ny, ny)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")

        N = ny * nx
        w = 1.0 / math.sqrt(N)

        positions = torch.stack([xv.flatten(), yv.flatten()], dim=-1)
        weights   = torch.full((N, ), w)

        return cls(positions, weights)

    def show_stats(self): return {
            "anchors/num":    self.positions.size(0),
            "anchors/mean_x": self.positions[:, 0].mean().item(),
            "anchors/mean_y": self.positions[:, 1].mean().item(),
            "anchors/mean_w": self.weights.mean().item(),
            "anchors/std_w":  self.weights.std().item(),
        }