from typing import Collection

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

from ..model import EditableINR

from ..setup.dataset   import RGBImageDataset
from ..setup.logger    import Logger
from ..setup.optimizer import Optimizer

from .metrics import psnr
from .progress_bar import ProgressBar


class Trainer:
    def __init__(
            self,
            model:      EditableINR, 
            dataset:    RGBImageDataset,
            optimizer:  Optimizer,
            logger:     Logger
        ):
        
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.logger = logger

    def start(
            self,
            eval_iterations: Collection[int],
            batch_size:      int = 512,
            num_iterations:  int = 1_000,
            output_path:     Path | None = None
        ):

        progress_bar = ProgressBar(num_iterations)
        for _ in range(num_iterations):
            iteration = _ + 1

            loss = self._backprop(batch_size).item()
            progress_bar.update(loss)

            eval = iteration in eval_iterations
            self._post_backprop(loss, iteration, eval)

        progress_bar.close()
        self.logger.close()

        self.save_result(output_path)
    
    def _backprop(self, batch_size: int):
        coords, colors = self.dataset.draw_pixels_batch(size=batch_size)

        preds = self.model(coords)
        loss = F.l1_loss(preds, colors)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    @torch.no_grad
    def _post_backprop(self, loss: float, iteration: int, eval: bool):
        self.optimizer.update_lrs(iteration)

        self.logger.log_scalar(loss, iteration, key="loss")

        stats = self.optimizer.show_stats()
        stats |= self.model.anchors.show_stats()

        if eval:
            coords, colors = self.dataset.draw_pixels_batch()
            preds = self.model(coords).clamp(0.0, 1.0)
            pred_image = self.dataset.reshape_pixels_to_image(preds)

            gt_image   = self.dataset.image.to(pred_image.device)

            metrics = { "psnr": psnr(pred_image, gt_image).item() }
            self.logger.log_scalar_dict(metrics, iteration, key="metrics")

            self.logger.log_image(pred_image, iteration, key="pred_image")

            stats |= {
                "eval_mean": pred_image.mean().item(),
                "eval_std":  pred_image.std().item()
            }

        self.logger.log_scalar_dict(stats, iteration, "stats")



    




