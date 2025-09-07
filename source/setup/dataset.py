from pathlib import Path

import torch
import imageio.v3 as iio


class RGBImageDataset:
    def __init__(self, image: torch.Tensor):
        assert image.ndim == 3 and image.size(-1) == 3, \
            f"Expected (H,W,3) tensor shape; got shape={image.shape}"

        self.image = image.to(torch.float32)
        self.H, self.W, _ = self.image.shape
        self.coords, self.colors = self._get_coords_and_colors()

    @property
    def aspect_ratio(self): return self.W / self.H

    @property
    def num_pixels(self):   return self.H * self.W

    def _get_coords_and_colors(self):
        ys, xs = torch.meshgrid(
            torch.arange(self.H), 
            torch.arange(self.W),
            indexing="ij"
        )

        coords = torch.stack([
            (xs.flatten() + 0.5) / self.W,
            (ys.flatten() + 0.5) / self.H
        ], dim=-1)

        colors = self.image.view(-1, 3)

        return coords, colors

    def draw_pixels_batch(self, size: int | None = None, device="cuda"):
        if size is None or size >= self.num_pixels:
            return self.coords.to(device), self.colors.to(device)
        
        random_ids = torch.randint(0, self.num_pixels, (size, ))
        
        return self.coords[random_ids].to(device), self.colors[random_ids].to(device)

    def reshape_pixels_to_image(self, pixels: torch.Tensor, out_HW3: bool=True):
        assert pixels.ndim == 2 and pixels.size(-1) == 3, \
            f"Expected (N, 3) shape tensor; got {pixels.shape}"
        assert pixels.size(0) == self.num_pixels, \
            f"Need N={self.num_pixels} (= H * W) pixels to compose the image; got {pixels.size(0)}"
        
        out_shape = (self.H, self.W, 3) if out_HW3 else (3, self.H, self.W) 
        return pixels.view(*out_shape)
    
    @classmethod
    def from_path(cls, path: Path, device="cpu"):
        return cls(torch.from_numpy(iio.imread(path) / 255.0).to(device))
        


    
        