import torch
import torch.nn as nn

from .anchors  import Anchors
from .bridge   import Bridge
from .decoder  import Decoder
from .hashgrid import HashGrid


class EditableINR(nn.Module):
    def __init__(
            self,
            anchors:    Anchors,
            bridge:     Bridge,
            decoder:    Decoder,
            hashgrid:   HashGrid
        ):

        super().__init__()
        
        self.anchors    = anchors
        self.bridge     = bridge
        self.decoder    = decoder
        self.hashgrid   = hashgrid

    def forward(self, coords: torch.Tensor):
        positions, weights = self.anchors()
        features  = self.hashgrid(positions)
        queries   = self.bridge(coords, positions, weights, features)

        return self.decoder(queries)

