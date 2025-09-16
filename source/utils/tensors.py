from typing import Collection, Sequence
import torch

import torch


def infer_tensors_dtype_device(
        tensors: Sequence[torch.Tensor],
        check_device: bool = False,
        check_dtype:  bool = False
    ):

    inf_device, inf_dtype = tensors[0].device, tensors[0].dtype

    if check_device:
        assert all(t.device == inf_device for t in tensors)
    if check_dtype:
        assert all(t.dtype  == inf_dtype  for t in tensors)

    return inf_dtype, inf_device


def infer_tensors_size_at_dim(
        tensors: Sequence[torch.Tensor], 
        dim:     int,
        check:   bool=False
    ):

    inf_size = tensors[0].size(dim)

    if check:
        assert all(tensor.size(dim) == inf_size for tensor in tensors)

    return inf_size

def shape_except(
        shape: tuple[int, ...], 
        dims:  int | Collection[int] | None = None
    ):

    if dims is None:          return shape
    if isinstance(dims, int): dims = (dims,)

    rank = len(shape)
    norm = { d if d >= 0 else rank + d for d in dims }
    
    return tuple(size for d, size in enumerate(shape) if d not in norm)

    