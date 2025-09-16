

from . import Directives
from . import rules as Rl

class Schedule:
    def __init__(
            self,
            loss_rules:  dict[str, Rl.RuleFn],
            lr_rules:    dict[str, Rl.RuleFn],
            prune_iters: set[int],
            densf_iters: set[int],
            eval_iters:  set[int]
        ):

        self.rules = { "lws": loss_rules, "lrs": lr_rules }

        self.prune_iters = prune_iters
        self.densf_iters = densf_iters
        self.eval_iters  = eval_iters


    def make_directives(self, iteration: int):
        return Directives(
            loss_weights=self._evaluate_rules(iteration, "lws"), 
            lr_multipliers=self._evaluate_rules(iteration, "lrs"),
            do_densify=iteration in self.densf_iters,
            do_prune=iteration in self.prune_iters,
            eval=iteration in self.eval_iters
        )

    def _evaluate_rules(self, iteration: int, ruleset: str):
        return { n: float(rl(iteration)) for n, rl in self.rules[ruleset].items() }
    
    @classmethod
    def from_config(cls, cfg: dict):
        losses_cfg = cfg.get("losses", {})
        loss_rules = {
            name: getattr(Rl, spec["rule"]["name"])(**spec["rule"].get("args", {}))
            for name, spec in losses_cfg.items()
        }

        opt_groups = cfg["optimizer"]["groups"]
        lr_rules = {
            group_name: getattr(Rl, gspec["rule"]["name"])(**gspec["rule"].get("args", {}))
            for group_name, gspec in opt_groups.items()
        }

        def mk_every(start: int, end: int, step: int):
            return set() if step <= 0 or end <= start \
                else set(range(start, end + 1, step))

        dens_cfg = cfg.get("densification",{})

        prune_int = dens_cfg.get("prune_int")
        densf_int = dens_cfg.get("densf_int")
        start_it  = dens_cfg.get("start_at_iter")
        end_it    = dens_cfg.get("end_at_iter")

        prune_iters = mk_every(start_it, end_it, prune_int)
        densf_iters = mk_every(start_it, end_it, densf_int)

        eval_cfg   = cfg.get("eval")
        eval_int   = eval_cfg.get("int")
        eval_start = eval_cfg.get("start")
        eval_end   = eval_cfg.get("end")

        eval_iters = mk_every(eval_start, eval_end, eval_int)

        return cls(loss_rules, lr_rules, prune_iters, densf_iters, eval_iters)