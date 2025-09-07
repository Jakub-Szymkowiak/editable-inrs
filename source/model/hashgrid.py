import torch
import torch.nn as nn
import tinycudann as tcnn


class HashGrid(nn.Module):
    def __init__(
            self,
            n_input_dims:          int = 2,
            n_levels:              int = 16,
            n_features_per_level:  int = 2,
            log2_hashmap_size:     int = 19,
            base_resolution:       int = 16,
            per_level_scale:       float = 1.5,
            dtype = torch.float16
        ):

        super().__init__()

        otype = "HashGrid"

        self.encoding = tcnn.Encoding(
            n_input_dims, {
                "otype":                otype,
                "n_levels":             n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size":    log2_hashmap_size,
                "base_resolution":      base_resolution,
                "per_level_scale":      per_level_scale,
            }, dtype=dtype)

        self.out_dim = n_levels * n_features_per_level

    def forward(self, x: torch.Tensor):
        return self.encoding(x)