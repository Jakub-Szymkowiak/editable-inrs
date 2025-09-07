import torch


def psnr(pred: torch.Tensor, gt:   torch.Tensor):   
    return -10 * torch.log10((pred - gt).square().mean().clamp_min(1e-10))