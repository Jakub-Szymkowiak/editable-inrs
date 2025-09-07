import math

from typing import Callable


RuleFn = Callable[[int], float]

def constant(value: float):
    def f(t: int) -> float:
        return float(value)
    return f


def linear(
        value_start: float,
        value_end: float,
        steps: int
    ):

    def f(t: int):
        if steps <= 0:
            return float(value_end)
        clamped = max(0, min(t, steps))
        fraction = clamped / steps
        return float(value_start + (value_end - value_start) * fraction)
    return f


def cosine(
        value_start: float,
        value_end: float,
        steps: int
    ):

    def f(t: int):
        if steps <= 0:
            return float(value_end)
        clamped = max(0, min(t, steps))
        fraction = clamped / steps
        return float(
            value_end + 0.5 * (value_start - value_end) * (1.0 + math.cos(math.pi * fraction))
        )
    return f


def warmup_then_decay(
        warmup_steps: int,
        peak: float,
        final: float,
        decay_steps: int
    ) :

    warm = linear(0.0, peak, warmup_steps)
    decay = cosine(peak, final, decay_steps)

    def f(t: int):
        if t <= warmup_steps:
            return warm(t)
        return decay(t - warmup_steps)
    return f


def step_decay(
        initial: float,
        factor: float,
        step_size: int
    ):

    def f(t: int):
        if step_size <= 0:
            return float(initial)
        k = t // step_size
        return float(initial * (factor ** k))
    return f


def piecewise_constant(
        milestones: list[int],
        values: list[float]
    ):

    assert len(values) == len(milestones) + 1

    def f(t: int):
        for i, m in enumerate(milestones):
            if t < m:
                return float(values[i])
        return float(values[-1])
    return f