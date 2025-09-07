from dataclasses import dataclass
from typing import Any

from tqdm import tqdm


class ProgressBar:
    def __init__(self, total: int, ema_beta: float = 0.95):
        self._bar = tqdm(
            total=total,
            desc="training",
            dynamic_ncols=True,
            leave=True
        )

        self.ema_beta = ema_beta
        self.ema_loss = None

    def update(self, loss: float):

        self._update_ema_loss(loss)
        self._bar.set_postfix({ f"EMA Loss": f"{self.ema_loss:.4f}" })
        self._bar.update(1)

    def _update_ema_loss(self, loss: float):
        self.ema_loss = sum([
            self.ema_beta * self.ema_loss,
            (1 - self.ema_beta) * loss
        ]) if self.ema_loss is not None else loss

    def close(self):
        self._bar.close()