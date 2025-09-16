import torch


from ..model import EditableINR, Bridge, Decoder, HashGrid
from ..model.anchors import ScalarAnchors, GaussianAnchors

from .optimizer import Optimizer


_ANCHOR_TYPES = {
    "scalar":   ScalarAnchors,
    "gaussian": GaussianAnchors,
}


def build_model(cfg: dict | None = None, device="cuda"):
    anchors_cfg = cfg.get("anchors")

    anchors_type = anchors_cfg.pop("type", None)
    if anchors_type is None: raise KeyError(f"Must specify Anchors type")

    anchors_cls = _ANCHOR_TYPES.get(anchors_type, None)
    if anchors_cls is None: raise KeyError(f"Unrecognized Anchors type: {anchors_type}")

    anchors  = anchors_cls                   \
        .from_grid(**cfg.get("anchors" ))    \
        .specify_trimmer(cfg.get("trimmer"))
    
    bridge   = Bridge(               **cfg.get("bridge"  ))
    decoder  = Decoder(              **cfg.get("decoder" ))
    hashgrid = HashGrid(             **cfg.get("hashgrid"))

    model = EditableINR(anchors, bridge, decoder, hashgrid).to(device)

    return model
    