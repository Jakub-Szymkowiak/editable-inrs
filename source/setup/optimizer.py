import torch


from .rules import RuleFn

class Optimizer:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            rules:     dict[str, RuleFn],
            groups:    dict[str, list[dict]],
            base_lrs:  dict[str, float],
            mode:      str | None = None
        ):

        self.opt      = optimizer
        self.rules    = rules
        self.groups   = groups
        self.base_lrs = base_lrs
        self.mode     = mode or "mul"

    def zero_grad(self, *args, **kwargs): 
        return self.opt.zero_grad(*args, **kwargs)
    
    def step(self, *args, **kwargs): 
        return self.opt.step(*args, **kwargs)
    
    def state_dict(self):          
        return self.opt.state_dict()
    
    def load_state_dict(self, sd): 
        return self.opt.load_state_dict(sd)

    def update_lrs(self, iteration: int):
        new_lrs: dict[str, float] = {}
        for name, pgs in self.groups.items():
            value = float(self.rules[name](iteration))
            base  = float(self.base_lrs[name])

            if   self.mode == "mul": lr = base * value
            elif self.mode == "set": lr = value
            elif self.mode == "add": lr = base + value
            else: raise KeyError(f"Unrecognized Optimizer mode: {self.mode}")

            if self.lr_min is not None: lr = max(lr, self.lr_min)
            if self.lr_max is not None: lr = min(lr, self.lr_max)

            new_lrs[name] = lr
            for pg in pgs: pg["lr"] = lr

        return new_lrs
    
    def show_stats(self):
        return {f"lr/{name}": pgs[0]["lr"] for name, pgs in self.groups.items()}