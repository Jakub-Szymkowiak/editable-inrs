from dataclasses import dataclass


@dataclass(frozen=True)
class Directives:
    loss_weights:   dict[str, float]
    lr_multipliers: dict[str, float]

    do_densify:   bool
    do_prune:     bool

    eval: bool


from .schedule import Schedule

__all__ = ["Directives", "Schedule"]