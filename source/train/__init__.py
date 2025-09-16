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
from ..setup.schedule  import Directives, Schedule

from .metrics import psnr
from .progress_bar import ProgressBar
from .losses import LossContext, LossFunction


class Trainer:
    def __init__(
            self,
            model:        EditableINR, 
            dataset:      RGBImageDataset,
            lossfunc:     LossFunction,
            optimizer:    Optimizer,
            schedule:     Schedule,
            logger:       Logger
        ):
        
        self.model     = model
        self.dataset   = dataset
        self.lossfunc  = lossfunc
        self.optimizer = optimizer
        self.schedule  = schedule
        self.logger    = logger


    def start(
            self,
            batch_size:      int = 512,
            num_iterations:  int = 1_000,
            output_path:     Path | None = None
        ):

        print(self.optimizer.get_anchors_lrs())

        progress_bar = ProgressBar(num_iterations)
        for _ in range(num_iterations):
            iteration = _ + 1

            directives = self.schedule.make_directives(iteration)

            total, losses = self._regress(batch_size, directives)
            progress_bar.update(total)

            self._post_backprop(losses, directives, iteration)

        progress_bar.close()
        self.logger.close()

        self._finalize(output_path)
    
    def _regress(self, batch_size: int, directives: Directives):
        coords, colors = self.dataset.draw_pixels_batch(size=batch_size)
        preds = self.model(coords)

        ctx = LossContext(preds=preds, colors=colors, anchors=self.model.anchors)
        total, losses = self.lossfunc(ctx=ctx, lws=directives.loss_weights)

        total.backward()

        # gather gradients for densification / pruning decision
        lrs = self.optimizer.get_anchors_lrs()
        self.model.anchors.trimmer.update_grad_ema(self.model, lrs)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return total.item(), losses

    @torch.no_grad
    def _render_eval_image(self, upscale: int=1):
        coords = self.dataset.draw_upscaled_coords(scale=upscale)
        preds  = self.model(coords)
        return self.dataset.reshape_pixels_to_image(preds, scale=upscale)
              
    @torch.no_grad
    def _post_backprop(
                self, 
                losses:        dict[str | float], 
                directives:    Directives, 
                iteration:     int
            ):
        
        # model maintenance
        rebuild = self.model.anchors.trimmer(self.model, 
            densify=directives.do_densify, prune=directives.do_prune)
        if rebuild: self.optimizer = self.optimizer.rebuild(self.model)

        self.optimizer.update_lrs(directives.lr_multipliers)

        # logging and eval
        self.logger.log_scalar_dict(losses, iteration, key="loss")

        stats = self.optimizer.show_stats() | self.model.anchors.show_stats()
        stats |= { "grad_ema_mean": self.model.anchors.grad_ema.mean() }

        if directives.eval: 
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

        # pred_image_4x = self._render_eval_image(upscale=4).permute(2, 0, 1)
        # save_image(pred_image_4x, path, "image_4x.png")

        self.model.export_anchors(path / "anchors.")





    




