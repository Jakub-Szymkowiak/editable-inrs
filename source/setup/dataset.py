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

    def reshape_pixels_to_image(
            self, pixels: torch.Tensor, 
            out_HW3: bool=True, 
            scale: int=1
        ):

        assert pixels.ndim == 2 and pixels.size(-1) == 3, \
            f"Expected (N, 3) shape tensor; got {pixels.shape}"
        assert pixels.size(0) == self.num_pixels, \
            f"Need N={self.num_pixels} (= H * W) pixels to compose the image; got {pixels.size(0)}"
        
        H, W = self.H * scale, self.W * scale
        out_shape = (H, W, 3) if out_HW3 else (3, H, W) 
        return pixels.view(*out_shape)
    
    def _coords_grid_crop(self, X: int, Y: int, W: int, H: int, device="cuda"):
        xs = torch.linspace(X + 0.5, X + W - 0.5, W, device=device) / self.W
        ys = torch.linspace(Y + 0.5, Y + H - 0.5, H, device=device) / self.H

        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([xv.flatten(), yv.flatten()], dim=-1)

    def draw_patch_at(self, X: int, Y: int, W: int, H: int, device="cuda"):
        assert 0 <= X < self.W and 0 <= Y < self.H
        assert X + W <= self.W and Y + H <= self.H

        coords = self._coords_grid_crop(X, Y, W, H, device=device)
        gt     = self.image[Y:Y+H, X:X+W].permute(2, 0, 1).unsqueeze(0).to(device)
        return coords, gt, (X, Y, W, H)

    def draw_random_patch(self, H: int, W: int, device="cuda"):
        X = torch.randint(0, self.W - W + 1, (1,)).item()
        Y = torch.randint(0, self.H - H + 1, (1,)).item()
        return self.draw_patch_at(X, Y, W, H, device=device)
    
    def draw_upscaled_coords(self, scale: int=2, device="cuda"):
        Hs, Ws = self.H * scale, self.W * scale

        ys, xs = torch.meshgrid(
            torch.arange(Hs, device=device),
            torch.arange(Ws, device=device),
            indexing="ij"
        )

        coords = torch.stack([
            (xs.flatten() + 0.5) / Ws,
            (ys.flatten() + 0.5) / Hs
        ], dim=-1)

        return coords


    @classmethod
    def from_path(cls, path: Path, device="cpu"):
        return cls(torch.from_numpy(iio.imread(path) / 255.0).to(device))
        


    
        