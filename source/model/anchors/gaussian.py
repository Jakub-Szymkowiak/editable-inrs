import math
import torch
import torch.nn as nn

from . import AnchorsBase


class GaussianAnchors(AnchorsBase):
    def __init__(
            self,
            positions:  torch.Tensor,
            log_scales: torch.Tensor,
            angles:     torch.Tensor,
            beta:       float = 1.0,
            eps:        float = 1e-12,
        ):

        super().__init__(positions=positions)

        assert log_scales.shape == positions.shape, \
            "log_scales must be (N,2)"
        assert angles.ndim == 1 and angles.size(0) == positions.size(0), \
            "angles must be (N,)"

        self._log_scales = nn.Parameter(log_scales)
        self._angles     = nn.Parameter(angles)

        self.beta = beta
        self.eps  = eps

    def forward(self):
        return self.positions, self._log_scales, self._angles
    
    def expose_param_dict(self):
        return super().expose_param_dict() | { 
            "scl": self._log_scales,
            "ang": self._angles
        }

    @classmethod
    def from_grid(cls, resolution: int, aspect_ratio: float = 1.0):
        ny, nx = resolution, round(aspect_ratio * resolution)

        xs = torch.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx, dtype=torch.float32)
        ys = torch.linspace(0.5 / ny, 1.0 - 0.5 / ny, ny, dtype=torch.float32)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")

        N = ny * nx
        positions  = torch.stack([xv.flatten(), yv.flatten()], dim=-1)

        init_scales = 0.8 * math.sqrt(1.0 / (nx * ny))
        log_scales  = torch.full((N, 2), math.log(init_scales), dtype=torch.float32)
        angles      = torch.zeros(N, dtype=torch.float32)

        return cls(positions, log_scales, angles)

    @torch.no_grad()
    def show_stats(self):
        scales = torch.exp(self._log_scales)
        return {
            "anchors/num":     self.positions.size(0),
            "anchors/mean_x":  self.positions[:, 0].mean(),
            "anchors/mean_y":  self.positions[:, 1].mean(),
            "anchors/mean_sx": scales[:, 0].mean(),
            "anchors/mean_sy": scales[:, 1].mean(),
        }

    def scores_for(
            self,
            coords:  torch.Tensor,
            idx:     torch.Tensor,
            sqdist:  torch.Tensor | None = None,
        ):

        assert coords.ndim == 2 and coords.size(-1) == 2
        assert idx.ndim == 2 and idx.size(0) == coords.size(0)

        p  = self.positions[idx]
        offsets  = coords.unsqueeze(1) - p

        cosines = torch.cos(self._angles[idx])
        sines   = torch.sin(self._angles[idx])

        dx, dy = offsets.unbind(dim=-1)

        dxp = cosines * dx + sines * dy
        dyp = cosines * dy - sines * dx

        scales = torch.exp(self._log_scales[idx]).clamp_min(self.eps)
        sx, sy = scales.unbind(dim=-1)

        inv_var_x = (sx * sx).clamp_min(self.eps).reciprocal()
        inv_var_y = (sy * sy).clamp_min(self.eps).reciprocal()

        q = (dxp * dxp) * inv_var_x + (dyp * dyp) * inv_var_y

        return torch.exp(-0.5 * q)