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

    anchors  = anchors_cls.from_grid(**cfg.get("anchors" ))
    bridge   = Bridge(               **cfg.get("bridge"  ))
    decoder  = Decoder(              **cfg.get("decoder" ))
    hashgrid = HashGrid(             **cfg.get("hashgrid"))

    model = EditableINR(anchors, bridge, decoder, hashgrid).to(device)

    return model



def build_optimizer(model, cfg):
    param_groups, groups, base_lrs, rules = [], {}, {}, {}

    def _rule(d): 
        from . import rules
        return getattr(rules, d["name"])(**d.get("args", {}))

    for short, params in model.anchors.expose_param_dict().items():
        key = f"anchors_{short}"
        settings = cfg["groups"][key]
        rule = _rule(settings.pop("rule"))
        plist = [params]
        pg = {"params": plist, **settings}
        param_groups.append(pg)
        groups[key] = [pg]
        base_lrs[key] = float(settings.get("lr", 0.0))
        rules[key] = rule

    for key in ("decoder", "hashgrid"):
        settings = cfg["groups"][key]
        rule = _rule(settings.pop("rule"))
        pg = {"params": list(getattr(model, key).parameters()), **settings}
        param_groups.append(pg)
        groups[key] = [pg]
        base_lrs[key] = float(settings.get("lr", 0.0))
        rules[key] = rule

    opt_cls = getattr(torch.optim, cfg["name"])
    opt = opt_cls(param_groups, **cfg.get("args", {}))

    wrapper = Optimizer(opt, rules, groups, base_lrs, mode=cfg.get("mode", "mul"))
    wrapper.lr_min = cfg.get("lr_min", None)
    wrapper.lr_max = cfg.get("lr_max", None)
    return wrapper

    