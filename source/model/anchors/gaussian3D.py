import math
import torch
import torch.nn as nn

from ...utils.quaternions import quat_to_rmat

from . import AnchorsBase


class GaussianAnchors3D(AnchorsBase):
    def __init__(
            self,
            positions:   torch.Tensor,
            log_scales:  torch.Tensor,
            quaternions: torch.Tensor,
            beta:        float = 1.0,
            eps:         float = 1e-12,
        ):
        super().__init__(positions=positions)

        assert positions.ndim == 2 and positions.size(-1) == 3, \
            "positions must be (N,3)"
        assert log_scales.shape == positions.shape, \
            "log_scales must be (N,3)"
        assert quaternions.ndim == 2 and quaternions.shape[0] == positions.shape[0] and quaternions.shape[1] == 4, \
            "quaternions must be (N,4)"

        self._log_scales = nn.Parameter(log_scales)
        self._quats      = nn.Parameter(quaternions)

        self.beta = beta
        self.eps  = eps

    def forward(self):
        return self.positions, self._log_scales, self._quats

    def expose_params(self):
        return super().expose_params() | {
            "scl":  ("_log_scales", self._log_scales),
            "quat": ("_quats",      self._quats),
        }

    @classmethod
    def from_grid(
            cls, 
            resolution: int, 
            aspect_ratio: float = 1.0, 
            dataset=None
        ):

        ny, nx = resolution, round(aspect_ratio * resolution)

        xs = torch.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx, dtype=torch.float32)
        ys = torch.linspace(0.5 / ny, 1.0 - 0.5 / ny, ny, dtype=torch.float32)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")

        N = ny * nx

        init_z_from_depth = dataset is not None and \
            hasattr(dataset, "has_depth")       and \
            dataset.has_depth()
        
        z = dataset.draw_depth_batch(size=N, device="cpu") \
            if init_z_from_depth else torch.rand(N, dtype=torch.float32) 

        positions  = torch.stack([xv.reshape(N), yv.reshape(N), z], dim=-1)

        init_scales = 0.8 * math.sqrt(1.0 / (nx * ny))
        log_scales  = torch.full((N, 3), math.log(init_scales), dtype=torch.float32)

        quats = torch.zeros((N, 4), dtype=torch.float32)
        quats[:, 0] = 1.0

        return cls(positions, log_scales, quats)

    @torch.no_grad()
    def show_stats(self):
        scales = torch.exp(self._log_scales)
        return {
            "anchors/num":     self.positions.size(0),
            "anchors/mean_x":  self.positions[:, 0].mean(),
            "anchors/mean_y":  self.positions[:, 1].mean(),
            "anchors/mean_z":  self.positions[:, 2].mean(),
            "anchors/mean_sx": scales[:, 0].mean(),
            "anchors/mean_sy": scales[:, 1].mean(),
            "anchors/mean_sz": scales[:, 2].mean(),
        }

    
    def scores_for(
            self, 
            coords: torch.Tensor, 
            idx: torch.Tensor, 
            sqdist: torch.Tensor | None = None
        ):
        assert coords.ndim == 2 and coords.size(-1) == 3
        assert idx.ndim == 2 and idx.size(0) == coords.size(0)

        p   = self.positions[idx]
        offsets  = coords.unsqueeze(1) - p              
        Rs    = quat_to_rmat(self._quats[idx], eps=self.eps)

        local = torch.matmul(
            Rs.transpose(-1, -2), 
            offsets.unsqueeze(-1)
        ).squeeze(-1)

        scales  = torch.exp(self._log_scales[idx]).clamp_min(self.eps)
        local_w = local / scales

        q = (local_w * local_w).sum(dim=-1)
        if self.beta != 1.0:
            q = self.beta * q

        return torch.exp(-0.5 * q)

