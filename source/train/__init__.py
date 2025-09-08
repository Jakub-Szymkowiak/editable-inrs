from typing import Any, Collection

from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision.utils import save_image

from fused_ssim import fused_ssim

from ..model import EditableINR

from ..setup.dataset   import RGBImageDataset
from ..setup.logger    import Logger
from ..setup.optimizer import Optimizer

from .metrics import psnr
from .progress_bar import ProgressBar


class Trainer:
    def __init__(
            self,
            model:        EditableINR, 
            dataset:      RGBImageDataset,
            optimizer:    Optimizer,
            logger:       Logger,
            lambda_dssim: float = 0.2
        ):
        
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.logger = logger

        self.lambda_dssim = lambda_dssim

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

            loss, regress_extra = self._regress(batch_size)
            progress_bar.update(loss)

            eval = iteration in eval_iterations
            self._post_backprop(loss, iteration, eval, regress_extra)

        progress_bar.close()
        self.logger.close()

        self._finalize(output_path)
    
    def _regress(self, batch_size: int):
        # L1 on random pixels
        coords, colors = self.dataset.draw_pixels_batch(size=batch_size)

        preds = self.model(coords)
        l1 = F.l1_loss(preds, colors)

        # FSSIM on patches
        H, W = 64, 64
        coords, colors, _ = self.dataset.draw_random_patch(H=H, W=W)

        preds = self.model(coords).view(1, H, W, 3).permute(0, 3, 1, 2)
        ssim = fused_ssim(preds, colors)
        
        loss = (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * (1.0 - ssim)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), { "l1": l1.item() }, # { "ssim": ssim.item(), "l1": l1.item() }

    @torch.no_grad
    def _render_eval_image(self, upscale: int=1):
        coords = self.dataset.draw_upscaled_coords(scale=upscale)
        preds  = self.model(coords)
        return self.dataset.reshape_pixels_to_image(preds, scale=upscale)
              
    @torch.no_grad
    def _post_backprop(self, loss: float, iteration: int, eval: bool, regress_extra: dict[str, float | int]):
        self.optimizer.update_lrs(iteration)

        self.logger.log_scalar(loss, iteration, key="loss")
        self.logger.log_scalar_dict(regress_extra, iteration, key="loss_terms")

        stats = self.optimizer.show_stats() | self.model.anchors.show_stats()

        if eval: 
            pred_image, metrics, eval_stats = self._eval()
            stats |= eval_stats

            self.logger.log_scalar_dict(metrics, iteration, key="metrics")
            self.logger.log_image(pred_image, iteration, key="pred_image")

        self.logger.log_scalar_dict(stats, iteration, "stats")

    @torch.no_grad
    def _eval(self):
        pred_image = self._render_eval_image()
        gt_image   = self.dataset.image.to(pred_image.device)

        metrics = { 
            "psnr": psnr(pred_image, gt_image).item() 
        }

        stats = {
            "eval_mean": pred_image.mean().item(),
            "eval_std":  pred_image.std().item()
        }

        return pred_image, metrics, stats
    

    @torch.no_grad
    def _finalize(self, path: Path):
        self.model.freeze()
        self.model.save(path / "checkpoint.pth")

        pred_image = self._render_eval_image().permute(2, 0, 1)
        save_image(pred_image, path / "image.png")

        pred_image_16x = self._render_eval_image(upscale=16).permute(2, 0, 1)
        save_image(pred_image_16x, path, "image_16x.png")

        self.model.export_anchors(path / "anchors.")





    




