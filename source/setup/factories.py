import torch


from ..model import EditableINR, Anchors, Bridge, HashGrid, Decoder
from .optimizer import Optimizer


def build_model(cfg: dict | None = None, device="cuda"):

    anchors  = Anchors.from_grid(**cfg.get("anchors" ))
    bridge   = Bridge(           **cfg.get("bridge"  ))
    decoder  = Decoder(          **cfg.get("decoder" ))
    hashgrid = HashGrid(         **cfg.get("hashgrid"))

    model = EditableINR(anchors, bridge, decoder, hashgrid).to(device)

    return model



def build_optimizer(model: EditableINR, cfg: dict | None = None):
    param_groups, groups, base_lrs, rules = [], {}, {}, {}

    def _rule(ruledict): 
        from . import rules
        return getattr(rules, ruledict.get("name"))(**ruledict.get("args", {}))

    def _add(name, params, settings):
        rules[name] = _rule(settings.pop("rule"))
        groups[name] = [ settings | { "params": list(params) } ]
        base_lrs[name] = float(settings.get("lr", 0.0))
        
        param_groups.append(groups[name][0])

    for name, pair in cfg.get("groups").items():
        module = getattr(model, name, None)
        
        if module is None: 
            raise KeyError(f"Model has no attribute '{name}' for optimizer group")
        if not hasattr(module, "parameters"):
            raise TypeError(f"Attribute '{name}' is not an nn.Module with .parameters()")
        
        _add(name, module.parameters(), pair)

    opt_cls  = getattr(torch.optim, cfg.get("name"))
    opt_args = cfg.get("args", {})
    opt      = opt_cls(param_groups, **opt_args)

    mode = cfg.get("mode", "mul")
    wrapper = Optimizer(
        optimizer=opt, 
        rules=rules, 
        groups=groups, 
        base_lrs=base_lrs, 
        mode=mode
    )

    wrapper.lr_min = cfg.get("lr_min", None)
    wrapper.lr_max = cfg.get("lr_max", None)

    return wrapper

    