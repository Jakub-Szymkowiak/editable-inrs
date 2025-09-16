from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from ... import EditableINR


class TrimmerBase(ABC, nn.Module):
    def __init__(
            self,
            N,
            *,
            ema_beta:     float=0.95,
            prune_thrs:   float=5e-6,
            prune_cap:    float=0.10,
            densf_thrs:   float=0.90,
            densf_cap:    float=0.10,
            jitter:       float=1e-3
        ):

        super().__init__()

        self.ema_beta = ema_beta

        assert 0 <= prune_cap <= 1.0 and 0 <= densf_cap <= 1.0
        self.prune_thrs = prune_thrs ; self.prune_cap = prune_cap
        self.densf_thrs = densf_thrs ; self.densf_cap = densf_cap

        self.jitter = jitter

        self.register_buffer("_grad_ema", torch.zeros(N, 1))

    def forward(
            self, 
            model:   EditableINR, 
            densify: bool=False, 
            prune:   bool=False
        ):

        if prune:   self.prune(model)
        if densify: self.densify(model)
    
    def update_grad_ema(
            self, model: EditableINR, lrs: dict[str, float] 
        ):

        total = sum(
            param.grad.reshape(param.size(0), -1).norm(dim=1) * lrs.get(short, 1.0)
            for short, (_, param) in model.anchors.expose_params().items()
            if param.grad is not None
        )

        beta = self.ema_beta
        self._grad_ema.mul_(beta).add_((1-beta) * total.unsqueeze(-1))
    
    def prune(self, model: EditableINR):
        g = self._grad_ema.squeeze(-1) ; N = g.numel()
        if N == 0: return

        ids = torch.nonzero(g < self.prune_thrs, as_tuple=False).flatten()
        if ids.numel() == 0: return

        max_remove = int(torch.floor(
            torch.tensor(self.prune_cap * N, device=g.device)
        ).item())

        if ids.numel() > max_remove:
            _, order = torch.sort(g[ids])
            ids = ids[order[:max_remove]]

        mask = torch.ones(N, dtype=torch.bool, device=g.device)
        mask[ids] = False

        model.anchors.remove_anchors(ids)
        self._grad_ema = self._grad_ema[mask]

    def densify(self):
        pass
        

from .gaussian import GaussianTrimmer

__all__ = ["GaussianTrimmer"]