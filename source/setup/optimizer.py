from copy import deepcopy

import torch

from ..utils.structures import TensorDict
from ..model import EditableINR


class Optimizer:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            groups:    dict[str, list[dict]],
            base_lrs:  dict[str, float],
            lr_min:    float | None = None,
            lr_max:    float | None = None,
            cfg:       dict  | None = None
        ):

        assert cfg is not None, \
            "Must provide optimizer config for post-densification rebuilds"

        self.opt = optimizer
        self.groups = groups
        self.base_lrs = base_lrs
        self.lr_min = lr_min
        self.lr_max = lr_max

        self._cfg = cfg

    def zero_grad(self, *args, **kwargs):
        return self.opt.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.opt.step(*args, **kwargs)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        return self.opt.load_state_dict(sd)

    def update_lrs(
            self, lr_multipliers: dict[str, float]
        ):

        new_lrs = {}
        for name, pgs in self.groups.items():
            base = self.base_lrs[name]
            mult = lr_multipliers.get(name, 1.0)

            lr = base * mult
            if self.lr_min is not None: lr = max(lr, self.lr_min)
            if self.lr_max is not None: lr = min(lr, self.lr_max)
            
            new_lrs[name] = lr
            for pg in pgs: pg["lr"] = lr

        return new_lrs

    def get_anchors_lrs(self):
        return {
            name.split("_", 1)[1]: pgs[0]["lr"]
            for name, pgs in self.groups.items()
            if name.startswith("anchors_")
        }

    def show_stats(self):
        return { 
            f"lr/{name}": pgs[0]["lr"] 
            for name, pgs in self.groups.items()
        }
    
    @classmethod
    def from_config(cls, model, cfg):
        param_groups, groups, base_lrs = [], {}, {}
        CFG = deepcopy(cfg)

        # anchors
        for short, (_, params) in model.anchors.expose_params().items():
            key = f"anchors_{short}"
            settings = cfg["groups"][key].copy()
            pg = {"params": [params], **settings}
            param_groups.append(pg)
            groups[key] = [pg]
            base_lrs[key] = float(settings.get("lr", 0.0))

        # decoder + hashgrid
        for key in ("decoder", "hashgrid"):
            settings = cfg["groups"][key].copy()
            pg = {"params": list(getattr(model, key).parameters()), **settings}
            param_groups.append(pg)
            groups[key] = [pg]
            base_lrs[key] = float(settings.get("lr", 0.0))

        opt_cls = getattr(torch.optim, cfg["name"])
        opt = opt_cls(param_groups, **cfg.get("args", {}))

        lr_min = cfg.get("lr_min")
        lr_max = cfg.get("lr_max")

        return cls(opt, groups, base_lrs, lr_min, lr_max, CFG)
    
    def rebuild(self, model: EditableINR):
        return self.from_config(model, self._cfg)