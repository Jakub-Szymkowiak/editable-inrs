import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(
            self,
            log_dir: str
        ):

        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, value: float, iteration: int, key: str):
        self.writer.add_scalar(key, value, iteration)

    def log_image(self, image: torch.Tensor, iteration: int, key: str, in_HWC: bool=True):
        if in_HWC: image = image.permute(2, 0, 1)
        self.writer.add_image(key, image, iteration)

    def log_scalar_dict(self, scalars: dict, iteration: int, key: str):
        for name, value in scalars.items():
            self.writer.add_scalar(f"{key}/{name}", value, iteration)

    def close(self): self.writer.close()