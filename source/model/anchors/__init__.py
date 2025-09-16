from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn


from ...utils.structures import TensorDict

class AnchorsBase(nn.Module, ABC):
    def __init__(
            self, 
            positions:    torch.Tensor, 
            margin:       float = 1e-3
        ):

        super().__init__()

        assert 0 <= margin < 0.1, \
            f"Margin must be in [0, 0.1); got margin={margin}"
        
        self._margin = margin

        assert positions.ndim == 2 and positions.size(-1) == 2, \
            "positions must be (N,2)"
        
        _positions_raw = self._positions_activation_inverse(positions)
        self._positions_raw = nn.Parameter(_positions_raw)

        self.trimmer = None

    def _positions_activation(self, latent: torch.Tensor):
        m = self._margin
        return m + (1.0 - 2.0 * m) * latent.sigmoid()

    def _positions_activation_inverse(self, pos: torch.Tensor):
        m = self._margin
        return torch.logit((pos - m) / (1.0 - 2.0 * m), eps=1e-6)

    @property
    def positions(self):
        return self._positions_activation(self._positions_raw)
    
    @property
    def size(self):    return self.positions.size(0)
    
    @property
    def device(self):   return self.positions.device

    @property
    def grad_ema(self): return self.trimmer._grad_ema
    
    def expose_params(self):
        return { "pos": ("_positions_raw", self._positions_raw) }
    
    def expose_attrs_names(self):
        return [ pair[0] for pair in self.expose_params().values() ]
    
    def expose_attrs_values(self):
        return [ pair[1] for pair in self.expose_params().values() ]

    @torch.no_grad()
    def add_new_anchors(
            self, 
            new_attrs: TensorDict
        ):

        assert self.device == new_attrs.device

        # assert the number of new entries is consistent
        new_attrs.check_dimsize(dim=0)

        # assert that the new attrs dims match existing anchors
        new_attrs.check_shape_compatibility(
            names=self.expose_attrs_names(), 
            values=self.expose_attrs_values(),
            dims=(0), 
        )

        for _, (attr_name, param) in self.expose_params().items():
            block = new_attrs[attr_name]
            cat   = torch.cat([param.detach(), block], dim=0)
            setattr(self, attr_name, nn.Parameter(cat, requires_grad=True))

    @torch.no_grad()
    def remove_anchors(self, ids: torch.Tensor):
        if ids.numel() == 0: return

        N = self._positions_raw.size(0)
        ids = torch.unique(ids.to(device=self.device, dtype=torch.long))

        assert torch.all((0 <= ids) & (ids < N)), \
            "remove_anchors: indices out of range"

        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        mask[ids] = False

        for _, (attr_name, param) in self.expose_params().items():
            kept = param[mask]
            new_param = nn.Parameter(kept, requires_grad=True)
            setattr(self, attr_name, new_param)

    @property
    def device(self):
        return self._positions_raw.device

    def specify_trimmer(
            self, trimmer_cfg: dict[str, Any]
        ):

        type = trimmer_cfg.pop("type", None)
        if type is None: raise KeyError("Trimemr config does not specify trimmer type")

        from . import trimmer
        trimmer_cls = getattr(trimmer, type, None)
        if trimmer_cls is None: 
            raise KeyError(f"Requested trimmer type {type} not found")
        
        self.trimmer = trimmer_cls(**(trimmer_cfg | {"N": self.size }))

        return self

    @classmethod
    def from_grid(
            cls, 
            resolution:   int, 
            aspect_ratio: float = 1.0,
            *args, **kwargs
        ):
        """Factory: initialize anchors on a grid."""
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

    @abstractmethod
    def show_stats(self):
        """Return dict of stats for logging."""
        raise NotImplementedError


from .gaussian import GaussianAnchors
from .scalar   import ScalarAnchors


__all__ = ["GaussianAnchors", "ScalarAnchors"]