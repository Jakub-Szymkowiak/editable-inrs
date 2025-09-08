from pathlib import Path

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

        self.register_buffer("_frozen_features", torch.empty(0), persistent=True)
        
        def _set_param_list(n, pl): 
            setattr(self, f"anchors_{n}", nn.ParameterList(pl))

        for name, params in self.anchors.expose_param_dict().items():
            _set_param_list(name, list(params))

    @property
    def is_frozen(self): 
        return self._frozen_features.numel() > 0

    def forward(self, coords: torch.Tensor):
        features = (self._frozen_features if self.is_frozen 
            else self.hashgrid(self.anchors.positions))
        queries  = self.bridge(self.anchors, coords, features)
        return self.decoder(queries)
    
    @torch.no_grad
    def freeze(self):
        features = self.hashgrid(self.anchors.positions)
        self._frozen_features = features.detach()

    @torch.no_grad()
    def unfreeze(self):
        self._frozen_features = torch.empty(0, device=self.anchors.positions.device)

    def save(self, path: Path):
        torch.save(self.state_dict(), str(path))

    def load(self, path: Path, device="cuda"):
        sd = torch.load(str(path), map_location=device)
        self.load_statate_dict(sd, strict=True)

    # def export_anchors(self, path: Path):
    #     self.anchors.export(path)

    # def inport_anchors(self, path: Path):
    #     self.anchors.inport(path)