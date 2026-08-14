"""A PatchCore-style unsupervised detector: patch features + memory bank + kNN.

Faithful to the parts of the method that matter for the claim rather than to
every implementation detail. What has to be right is the shape of the pipeline --
mid-level pretrained features, a memory bank of *normal* patches, and an anomaly
score that is a distance to the nearest normal patch -- because that is what
produces a spatial anomaly map, and the spatial map is what this project
interrogates.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from .data import IMG


class PatchCore:
    def __init__(self, coreset_frac: float = 0.10, device: str | None = None,
                 seed: int = 0):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        net.train(False)
        net.to(self.device)
        self.net = net
        self.coreset_frac = coreset_frac
        self.seed = seed
        self.memory: torch.Tensor | None = None
        self._feat: dict[str, torch.Tensor] = {}
        # layer2 + layer3: deep enough to be semantic, shallow enough to keep
        # spatial resolution. layer4 localises too coarsely to score a 32px box.
        net.layer2.register_forward_hook(self._hook("l2"))
        net.layer3.register_forward_hook(self._hook("l3"))

    def _hook(self, name):
        def fn(_m, _i, out):
            self._feat[name] = out
        return fn

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) -> (B, C, h, w) locally-averaged concatenated features."""
        x = x.repeat(1, 3, 1, 1)  # grayscale -> the 3 channels ImageNet expects
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        with torch.no_grad():
            self.net((x - mean) / std)
        l2, l3 = self._feat["l2"], self._feat["l3"]
        # Local average pooling, as in PatchCore: a patch descriptor should
        # summarise a neighbourhood, not a single activation.
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([l2, l3], dim=1)

    def fit(self, images: np.ndarray, batch: int = 16) -> "PatchCore":
        feats = []
        h = w = None
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).unsqueeze(1).to(self.device)
            f = self._embed(x)                      # B,C,h,w
            b, c, h, w = f.shape
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, c).cpu())
        bank = torch.cat(feats)
        # Random subsample instead of greedy k-center coreset selection. Greedy
        # selection is the paper's contribution to *efficiency*; it does not
        # change what the memory represents, and this project's claim concerns
        # localisation, not throughput. Stated rather than silently substituted.
        g = torch.Generator().manual_seed(self.seed)
        k = max(1, int(len(bank) * self.coreset_frac))
        idx = torch.randperm(len(bank), generator=g)[:k]
        self.memory = F.normalize(bank[idx], dim=1).to(self.device)
        self.fmap_size = (h, w)
        return self

    def anomaly_map(self, images: np.ndarray, batch: int = 16) -> np.ndarray:
        """(N, IMG, IMG) map of distance-to-nearest-normal-patch."""
        assert self.memory is not None, "fit() first"
        maps = []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).unsqueeze(1).to(self.device)
            f = self._embed(x)
            b, c, h, w = f.shape
            q = F.normalize(f.permute(0, 2, 3, 1).reshape(-1, c), dim=1)
            # Cosine distance to nearest memory patch, chunked to bound memory.
            best = torch.full((len(q),), 1e9, device=self.device)
            for j in range(0, len(self.memory), 8192):
                sim = q @ self.memory[j:j + 8192].T
                best = torch.minimum(best, 1 - sim.max(dim=1).values)
            m = best.reshape(b, h, w).unsqueeze(1)
            m = F.interpolate(m, size=(IMG, IMG), mode="bilinear", align_corners=False)
            maps.append(m.squeeze(1).cpu().numpy())
        return np.concatenate(maps)

    @staticmethod
    def image_score(maps: np.ndarray) -> np.ndarray:
        """Image-level score = peak of the map, as in PatchCore."""
        return maps.reshape(len(maps), -1).max(axis=1)
