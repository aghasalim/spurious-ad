"""A PatchCore-style unsupervised detector: patch features + memory bank + kNN.

Faithful to the parts of the method that matter for the claim rather than to
every implementation detail. What has to be right is the shape of the pipeline --
mid-level pretrained features, a memory bank of *normal* patches, and an anomaly
score that is a distance to the nearest normal patch, because that is what
produces a spatial anomaly map, and the spatial map is what this project
interrogates.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

from .data import IMG

# Backbone sensitivity is a stated limitation, so the extractor is a parameter
# rather than a hard-coded call. Both nets expose layer2/layer3 identically.
ARCHS = {
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights),
    "resnet18": (models.resnet18, models.ResNet18_Weights),
}


def image_score(maps: np.ndarray) -> np.ndarray:
    """Image-level score = peak of the anomaly map, as in PatchCore and PaDiM."""
    return maps.reshape(len(maps), -1).max(axis=1)


class Backbone:
    """Frozen ImageNet features at layer2+layer3, shared by both detectors.

    Shared deliberately: if the two detector families used different features,
    a difference in their results would be unattributable.
    """

    def __init__(self, arch: str = "wide_resnet50_2", device: str | None = None):
        ctor, weights = ARCHS[arch]
        self.arch = arch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        net = ctor(weights=weights.IMAGENET1K_V1)
        net.train(False)
        net.to(self.device)
        self.net = net
        self._feat: dict[str, torch.Tensor] = {}
        # layer2 + layer3: deep enough to be semantic, shallow enough to keep
        # spatial resolution. layer4 localises too coarsely to score a 32px box.
        net.layer2.register_forward_hook(self._hook("l2"))
        net.layer3.register_forward_hook(self._hook("l3"))

    def _hook(self, name):
        def fn(_m, _i, out):
            self._feat[name] = out
        return fn

    def to_tensor(self, images: np.ndarray) -> torch.Tensor:
        """(B,H,W) grayscale synthetic or (B,3,H,W) real RGB -> (B,C,H,W)."""
        x = torch.from_numpy(np.ascontiguousarray(images))
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x.to(self.device)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1|3,H,W) -> (B, C, h, w) locally-averaged concatenated features."""
        if x.shape[1] == 1:
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


class PatchCore:
    def __init__(self, coreset_frac: float = 0.10, device: str | None = None,
                 seed: int = 0, arch: str = "wide_resnet50_2"):
        self.bk = Backbone(arch, device)
        self.device = self.bk.device
        self.coreset_frac = coreset_frac
        self.seed = seed
        self.memory: torch.Tensor | None = None

    def fit(self, images: np.ndarray, batch: int = 16) -> "PatchCore":
        feats = []
        h = w = None
        for i in range(0, len(images), batch):
            f = self.bk.embed(self.bk.to_tensor(images[i:i + batch]))  # B,C,h,w
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
            f = self.bk.embed(self.bk.to_tensor(images[i:i + batch]))
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

    image_score = staticmethod(image_score)


class PaDiM:
    """The second detector family: a Gaussian per patch location, Mahalanobis score.

    PatchCore is non-parametric (nearest neighbour in a memory bank); PaDiM is
    parametric (a fitted multivariate normal per spatial position). They share
    only the "model normal, flag departures" structure and the frozen backbone,
    which is exactly the thing the repo's conclusion is supposed to rest on --
    so if the conclusion is a property of that structure rather than of kNN, both
    must behave the same way.

    Reduced to `n_dims` randomly-chosen feature channels, as in the paper. The
    paper uses 550 of 1792 at a 56x56 grid; this uses 100 of 1536 at 32x32, which
    keeps the 1024 covariance matrices small enough to invert on a laptop and
    keeps n_train (200-400 images) comfortably above the dimension. Statistics
    are fitted and applied on CPU, MPS has no `linalg.inv`, while the
    backbone stays on the GPU, where the cost actually is.
    """

    def __init__(self, n_dims: int = 100, device: str | None = None, seed: int = 0,
                 arch: str = "wide_resnet50_2", ridge: float = 0.01):
        self.bk = Backbone(arch, device)
        self.device = self.bk.device
        self.n_dims = n_dims
        self.seed = seed
        self.ridge = ridge
        self.mu: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None

    def _reduced(self, images: np.ndarray, batch: int) -> torch.Tensor:
        """(N, n_dims, h, w) on CPU."""
        out = []
        for i in range(0, len(images), batch):
            f = self.bk.embed(self.bk.to_tensor(images[i:i + batch]))
            if self.idx is None:
                g = torch.Generator().manual_seed(self.seed)
                self.idx = torch.randperm(f.shape[1], generator=g)[:self.n_dims]
            out.append(f[:, self.idx.to(f.device)].cpu())
        return torch.cat(out)

    def fit(self, images: np.ndarray, batch: int = 16) -> "PaDiM":
        self.idx = None
        f = self._reduced(images, batch)                  # N,d,h,w
        n, d, h, w = f.shape
        if n <= d:
            raise ValueError(f"PaDiM needs more images than dims: n={n}, d={d}")
        x = f.permute(2, 3, 0, 1).reshape(h * w, n, d).double()   # L,N,d
        self.mu = x.mean(dim=1)                                   # L,d
        xc = x - self.mu[:, None]
        cov = xc.transpose(1, 2) @ xc / (n - 1)
        cov += self.ridge * torch.eye(d, dtype=cov.dtype)   # keeps it invertible
        self.inv = torch.linalg.inv(cov)                    # L,d,d
        self.fmap_size = (h, w)
        return self

    def anomaly_map(self, images: np.ndarray, batch: int = 16) -> np.ndarray:
        assert self.mu is not None, "fit() first"
        maps = []
        for i in range(0, len(images), batch):
            f = self._reduced(images[i:i + batch], batch)
            b, d, h, w = f.shape
            x = f.permute(2, 3, 0, 1).reshape(h * w, b, d).double() - self.mu[:, None]
            m = torch.einsum("lbi,lij,lbj->lb", x, self.inv, x)
            m = m.clamp(min=0).sqrt().reshape(h, w, b).permute(2, 0, 1).float()
            m = F.interpolate(m.unsqueeze(1), size=(IMG, IMG), mode="bilinear",
                              align_corners=False)
            maps.append(m.squeeze(1).numpy())
        return np.concatenate(maps)

    image_score = staticmethod(image_score)


DETECTORS = {"patchcore": PatchCore, "padim": PaDiM}
