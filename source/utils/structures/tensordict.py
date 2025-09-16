from typing import Collection, Sequence 

import torch

from ..tensors import (
    infer_tensors_dtype_device,
    infer_tensors_size_at_dim,
    shape_except
)


class TensorDict:
    def __init__(
            self, 
            keys:   Sequence[str],
            values: Sequence[torch.Tensor],
            device: str | None = None,
            dtype:  torch.dtype | None = None
        ):

        L = len(keys)
        assert L == len(values), "Number of keys and values does not match"

        check_device, check_dtype = device is None, dtype is None

        inf_dtype, inf_device = infer_tensors_dtype_device(
            tensors=values, 
            check_device=check_device, 
            check_dtype=check_dtype
        )

        device = device or inf_device ; dtype = dtype or inf_dtype

        self._dict = { keys[i]: values[i] for i in range(L) }
        self.to(device=device, dtype=dtype)

    def keys(  self): return self._dict.keys(  )
    def values(self): return self._dict.values()
    def items( self): return self._dict.items( )

    def __getitem__(self, key: str): 
        return self._dict[key]

    def __setitem__(
            self, key: str, value: torch.Tensor, move: bool=True
        ): 

        if not move: 
            assert value.device == self.device, "Device mismatch"
            assert value.dtype  == self.dtype,  "Dtype mismatch"
        else:        
            value = value.to(device=self.device, dtype=self.dtype)
        
        self._dict[key] = value

    @property
    def dtype(self):
        return infer_tensors_dtype_device(
            list(self.values()), check_dtype=True)[0]
    
    @property
    def device(self):
        return infer_tensors_dtype_device(
            list(self.values()), check_device=True)[1]
    
    def to(self, device: str | None = None, dtype: torch.dtype | None = None):
        device = device or self.device ; dtype = dtype or self.dtype

        for key, value in self.items():
            self[key] = value.to(device=device, dtype=dtype)

    def check_missing_extra(
            self, keys: Collection[str], allow_mismatch: bool = True
        ):

        int_keys = set(self.keys())
        ext_keys = set(keys)

        missing, extra = int_keys - ext_keys, ext_keys - int_keys

        if not allow_mismatch: 
            assert len(extra) == len(missing) == 0

        return missing, extra

    def check_dimsize(self, dim: int, ref: int | None = None):
        dimsize = infer_tensors_size_at_dim(
            list(self.values()), dim=dim, check=True
        )

        if ref is not None: assert ref == dimsize

        return dimsize
    
    def check_shape_compatibility(
            self, 
            names:  Sequence[str],
            values: Sequence[torch.Tensor],
            dims:   Collection[int] | None = None
        ):

        assert len(names) == len(values)

        return all(
            shape_except(v.shape, dims) 
            == shape_except(self[n].shape, dims)
            for n, v in zip(names, values)
        )



    


