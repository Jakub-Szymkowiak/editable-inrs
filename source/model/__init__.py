import torch
import torch.nn as nn

from .anchors  import AnchorsBase
from .bridge   import Bridge
from .decoder  import Decoder
from .hashgrid import HashGrid


class EditableINR(nn.Module):
    def __init__(
            self,
            anchors:    AnchorsBase,
            bridge:     Bridge,
            decoder:    Decoder,
            hashgrid:   HashGrid 
        ):

        super().__init__()
        
        self.anchors  = anchors
        self.bridge   = bridge
        self.decoder  = decoder
        self.hashgrid = hashgrid

    def forward(self, coords: torch.Tensor):
        features  = self.hashgrid(self.anchors.positions)
        queries   = self.bridge(self.anchors, coords, features)

        return self.decoder(queries)