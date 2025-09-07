import torch
import torch.nn as nn


class Bridge(nn.Module):
    def __init__(self, k: int=8, bandwidth: float=0.05, eps: float=1e-8, chunk_size: int= 4096):
        super().__init__()

        self.k = k
        self.eps = eps
        self.chunk_size = chunk_size
        self.softmax_temperature = 2 * bandwidth * bandwidth

    def forward(
            self,
            coords:    torch.Tensor,
            positions: torch.Tensor,
            weights:   torch.Tensor,
            features:  torch.Tensor,
        ):

        B = coords.size(0)
        temp = self.softmax_temperature
        outputs = [] 

        for start in range(0, B, self.chunk_size):
            end = min(start + self.chunk_size, B)
            coords_chunk = coords[start:end]

            sqdist = torch.cdist(coords_chunk, positions, p=2).pow(2)
            sqdist = sqdist / (weights.unsqueeze(0) ** 2 + self.eps)

            sq_k, idx = torch.topk(sqdist, self.k, dim=1, largest=False)
            w = torch.softmax(-sq_k / temp, dim=1).unsqueeze(-1)

            out_chunk = (w * features[idx]).sum(dim=1)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=0)