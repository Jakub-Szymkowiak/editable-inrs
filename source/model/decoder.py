import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
            self,
            out_dim:          int = 3,
            hidden_width:     int = 128,
            num_layers:       int = 2,
            activation:       nn.Module = nn.SiLU,
            final_activation: nn.Module | None = None,
        ):

        super().__init__()

        assert all([num_layers > 0, hidden_width > 0, out_dim > 0])
        assert activation is not None

        layers = [nn.LazyLinear(hidden_width), activation()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(activation())
        layers.append(nn.Linear(hidden_width, out_dim))
        
        if final_activation is not None:
            layers.append(final_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)