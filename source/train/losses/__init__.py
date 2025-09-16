from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...model import AnchorsBase

import torch 
import torch.nn as nn


@dataclass
class LossContext:
    preds: torch.Tensor
    colors: torch.Tensor
    anchors: AnchorsBase


class LossBase(ABC, nn.Module):
    def __init__(self, cfg: dict | None = None): 
        super().__init__()

        self.cfg = cfg or {}

    @abstractmethod
    def forward(self, ctx: LossContext):
        raise NotImplementedError


class LossFunction(nn.Module):
    def __init__(self, losses):
        super().__init__()

        self.losses = nn.ModuleDict(losses)

    def forward(self, ctx: LossContext, lws: torch.Tensor):        
        lossdict = {
            name: lossfn(ctx) * lws[name]
            for name, lossfn in self.losses.items()
        }

        return (sum(lossdict.values()), {
            name: value.item() for name, value in lossdict.items()
        })

    @classmethod
    def from_config(cls, cfg: dict):
        from .. import losses as Ls
        
        def _make_loss(name, args):
            if not hasattr(Ls, name): raise KeyError(f"Unknown loss {name} in config")
            return getattr(Ls, name)(**args)

        return cls({
            key: _make_loss(spec["fn"]["name"], spec["fn"].get("args", {}))
            for key, spec in cfg.items()
        })


from .l1loss import L1Loss

__all__ = [
    LossContext, LossFunction,
    L1Loss,
]