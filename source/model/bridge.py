import torch
import torch.nn as nn

from .anchors import AnchorsBase


class Bridge(nn.Module):
    def __init__(
            self,
            k:          int   = 8, 
            chunk_size: int   = 4096,
            bandwidth:  float = 0.05, 
            eps:        float = 1e-8,
        ):

        super().__init__()

        self.k = k
        self.eps = eps
        self.chunk_size = chunk_size
        self.temp = 2.0 * (bandwidth ** 2)

    def forward(
            self,
            anchors:  AnchorsBase,
            coords:   torch.Tensor,
            features: torch.Tensor
        ):

        B = coords.size(0)
        positions = anchors.positions
        outputs = []

        for start in range(0, B, self.chunk_size):
            end = min(start + self.chunk_size, B)
            q = coords[start:end]

            sqdist = torch.cdist(q, positions, p=2).pow(2)
            top_sqdist, top_idx = torch.topk(sqdist, self.k, dim=1, largest=False)

            scores = anchors.scores_for(q, top_idx, top_sqdist)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)

            out = (w * features[top_idx]).sum(dim=1).to(coords.dtype)
            outputs.append(out)

        return torch.cat(outputs, dim=0)