import torch.nn.functional as F

from . import LossBase, LossContext


class L1Loss(LossBase):
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

    def forward(self, ctx: LossContext):
        return F.l1_loss(ctx.preds, ctx.colors)