"""Prithvi-EO-2.0 encoder + a small decoder that predicts NBR at t + horizon for every pixel.

The decoder predicts the change from the last clear NBR (base), so a model that learns nothing
still does as well as persistence. Terrain is concatenated with the decoder features at full
resolution (it's on the same 30 m grid as the image).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def up_block(c_in, c_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class PrithviRecovery(nn.Module):
    def __init__(self, backbone="prithvi_eo_v2_100_tl", num_frames=4, freeze_encoder=True, use_terrain=True):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.use_coords = backbone.endswith("_tl")
        kwargs = {"pretrained": True, "num_frames": num_frames, "bands": PRITHVI_BANDS}
        if self.use_coords:
            kwargs["coords_encoding"] = ["time", "location"]
        self.encoder = BACKBONE_REGISTRY.build(backbone, **kwargs)
        self.num_frames = num_frames
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        dim = self._embed_dim()
        self.neck = nn.Sequential(nn.Conv2d(dim * num_frames, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(up_block(256, 128), up_block(128, 64), up_block(64, 32), up_block(32, 16))
        self.use_terrain = use_terrain
        extra = 1 + (4 if use_terrain else 0)  # base NBR (+ terrain)
        self.out = nn.Sequential(nn.Conv2d(16 + extra, 16, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.out[-1].weight)    # start as persistence
        nn.init.zeros_(self.out[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _embed_dim(self):
        out = getattr(self.encoder, "out_channels", None)
        if out:
            return out[-1]
        return getattr(self.encoder, "embed_dim")

    def encode(self, x, temporal_coords=None, location_coords=None):
        if self.use_coords:
            feats = self.encoder(x, temporal_coords=temporal_coords, location_coords=location_coords)
        else:
            feats = self.encoder(x)
        tokens = feats[-1] if isinstance(feats, (list, tuple)) else feats      # (B, N, D)
        B, N, D = tokens.shape
        T = self.num_frames
        if (N - 1) % T == 0 and math.isqrt((N - 1) // T) ** 2 == (N - 1) // T:
            tokens = tokens[:, 1:]                                             # drop the CLS token
        g = math.isqrt(tokens.shape[1] // T)
        return tokens.reshape(B, T, g, g, D).permute(0, 1, 4, 2, 3).reshape(B, T * D, g, g)

    def forward(self, batch):
        x = batch["x"]
        if self.freeze_encoder:
            with torch.no_grad():
                f = self.encode(x, batch.get("temporal_coords"), batch.get("location_coords"))
        else:
            f = self.encode(x, batch.get("temporal_coords"), batch.get("location_coords"))
        y = self.decoder(self.neck(f.float()))
        H, W = x.shape[-2:]
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        base = batch["base"].unsqueeze(1)
        parts = [y, base] + ([batch["terrain"]] if self.use_terrain else [])
        return batch["base"] + self.out(torch.cat(parts, 1)).squeeze(1)
