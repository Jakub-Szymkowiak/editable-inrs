import math
import torch

from ....utils.structures import TensorDict

from ..gaussian import GaussianAnchors

from . import TrimmerBase


class GaussianTrimmer(TrimmerBase):
    def __init__(
            self,
            N,
            *,
            ema_beta:     float=0.95,
            prune_thrs:   float=5e-6,
            prune_cap:    float=0.10,
            densf_thrs:   float=1.00,
            jitter:       float=1e-3,
            split_thrs:   float=0.02,
            split_shrink: float=1.00,
            clone_shrink: float=1.00
        ):

        super().__init__(
            N, ema_beta=ema_beta, jitter=jitter,
            prune_thrs=prune_thrs, prune_cap=prune_cap,
            densf_thrs=densf_thrs
        )

        self.split_thrs   = split_thrs
        self.split_shrink = split_shrink
        self.clone_shrink = clone_shrink

    @torch.no_grad()
    def densify(self, model):
        a: GaussianAnchors = model.anchors
        g = self._grad_ema.squeeze(-1) ; g = g / g.max()
        N = g.numel() ; device = g.device

        parents = torch.nonzero(g >= self.densf_thrs, as_tuple=False).flatten()
        if parents.numel() == 0: return

        pos = a._positions_raw[parents]
        scl = a._log_scales[parents]
        ang = a._angles[parents]
                
        logS_axis = a._log_scales.max(dim=1).values     
        logS_max  = logS_axis.max()
        log_tau   = math.log(self.split_thrs)
        mask      = (logS_axis - logS_max) >= log_tau
        big = mask[parents] ; sml = ~big 


        def make_children(pos, scl, ang, add_log, n):
            if pos.numel() == 0:
                return (
                    torch.empty_like(pos),
                    torch.empty_like(scl),
                    torch.empty_like(ang),
                )
            
            ch_ang =  ang.repeat_interleave(n, 0)

            ch_scl = (scl + add_log).repeat_interleave(n, 0)
            jmag   = ch_scl.exp().mean(-1, keepdim=True) * self.jitter

            ch_pos = a._positions_activation(pos).repeat_interleave(n, 0)
            ch_pos = (ch_pos + torch.randn_like(ch_pos) * jmag).clamp_(0.0, 1.0)
            ch_pos = a._positions_activation_inverse(ch_pos)

            return (ch_pos, ch_scl, ch_ang)
        
        log_split_shrink = math.log(self.split_shrink)
        log_clone_shrink = math.log(self.clone_shrink)

        split_ch = make_children(pos[big], scl[big], ang[big], log_split_shrink, 2)
        clone_ch = make_children(pos[sml], scl[sml], ang[sml], log_clone_shrink, 1)

        ch_pos, ch_scl, ch_ang = (torch.cat(attrs, 0) for attrs in zip(split_ch, clone_ch))

        a.add_new_anchors(TensorDict(
            keys   = ("_positions_raw",  "_log_scales",  "_angles" ),
            values = (ch_pos, ch_scl, ch_ang)
        ))

        self._grad_ema = torch.cat([self._grad_ema, torch.zeros(
            ch_pos.size(0), 1, device=device, dtype=self._grad_ema.dtype
        )], dim=0)
