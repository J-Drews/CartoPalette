"""CartoPalette v4.1 / v4.2 CVAE model architecture.

v4.1 condition layout (loaded from existing checkpoint):
    [CNN features (256) | metadata one-hot (10)]   = 266 dims

v4.2 extension (optional, activated by config['dominant_colors_dim'] > 0):
    [CNN features (256) | metadata one-hot (10) | dominant LAB colours (30)] = 296 dims

The dominant-colour block is the 10 K-Means centroids of the basemap in CIELAB,
flattened and divided by 100 (so L*∈[0,1], a*/b*≈[-1,1]). This gives the decoder
direct access to "which colours must I avoid" instead of forcing the CVAE to
recover that information from the EfficientNet bottleneck.

To stay backward-compatible, ChromaMapCVAE inspects ``config`` and only enables
the extra block when ``dominant_colors_dim`` is set and > 0.
"""

import torch
import torch.nn as nn
from torchvision import models


class BasemapEncoder(nn.Module):
    """EfficientNet-B0 with projection head for basemap-discriminative features."""

    def __init__(self, projection_dim=256):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = efficientnet.features
        self.pool = efficientnet.avgpool

        for i, block in enumerate(self.features):
            if i < 3:
                for param in block.parameters():
                    param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x


class CVAEEncoder(nn.Module):
    """CVAE encoder: (condition, palette) -> (mu, logvar)."""

    def __init__(self, condition_dim, palette_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim + palette_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, condition, palette):
        x = torch.cat([condition, palette], dim=-1)
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """CVAE decoder: (condition, z) -> reconstructed palette."""

    def __init__(self, condition_dim, latent_dim, hidden_dim, palette_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, palette_dim),
        )

    def forward(self, condition, z):
        return self.net(torch.cat([condition, z], dim=-1))


class ChromaMapCVAE(nn.Module):
    """CartoPalette CVAE: Basemap image -> Projected CNN features (+ optional
    dominant-colour block) -> CVAE -> Palette.

    Default v4.1 condition:
        [projected_features (256) + metadata (10)] = 266 dims.

    v4.2 condition (when ``config['dominant_colors_dim'] > 0``):
        [projected_features (256) + metadata (10) + dominant_LAB (30)] = 296 dims.

    The dominant-LAB block is supplied externally at training/inference time
    (the same K-Means dominant colours already computed by the scoring module).
    Pre-normalisation: L*/100, a*/100, b*/100 — keeps the block in roughly the
    same magnitude as the projected CNN features.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]

        self.cnn = BasemapEncoder(projection_dim=config["cnn_projection_dim"])
        self.metadata_scale = nn.Parameter(torch.ones(10) * 1.0)

        # v4.2: dominant-colour block size (0 = disabled, fall back to v4.1)
        self.dominant_colors_dim = int(config.get("dominant_colors_dim", 0))
        if self.dominant_colors_dim > 0:
            # Learnable scale per dimension so the model can downweight noisy
            # K-Means centroids if they are not useful.
            self.dominant_scale = nn.Parameter(torch.ones(self.dominant_colors_dim))

        condition_dim = config["condition_dim"]
        self.encoder = CVAEEncoder(
            condition_dim, config["palette_dim"], config["hidden_dim"], config["latent_dim"]
        )
        self.decoder = CVAEDecoder(
            condition_dim, config["latent_dim"], config["hidden_dim"], config["palette_dim"]
        )

    # ------------------------------------------------------------------
    # build_condition has TWO signatures:
    #   v4.1 (legacy):  build_condition(cnn_features, metadata)
    #   v4.2:           build_condition(cnn_features, metadata, dominant_lab_norm)
    #
    # The v4.2 path is only taken when dominant_colors_dim > 0 AND
    # dominant_lab_norm is provided. This keeps the call-site in inference.py
    # unchanged for the v4.1 checkpoint.
    # ------------------------------------------------------------------
    def build_condition(self, cnn_features, metadata, dominant_lab_norm=None):
        scaled_meta = metadata * self.metadata_scale
        parts = [cnn_features, scaled_meta]
        if self.dominant_colors_dim > 0 and dominant_lab_norm is not None:
            parts.append(dominant_lab_norm * self.dominant_scale)
        return torch.cat(parts, dim=-1)

    def forward(self, image, metadata, palette, dominant_lab_norm=None):
        cnn_features = self.cnn(image)
        condition = self.build_condition(cnn_features, metadata, dominant_lab_norm)
        mu, logvar = self.encoder(condition, palette)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(condition, z)
        return recon, mu, logvar, cnn_features

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    @torch.no_grad()
    def generate(self, image, metadata, n_samples=5, dominant_lab_norm=None):
        self.eval()
        cnn_features = self.cnn(image)
        condition = self.build_condition(cnn_features, metadata, dominant_lab_norm)
        condition = condition.repeat(n_samples, 1)
        z = torch.randn(n_samples, self.latent_dim, device=image.device)
        palettes = self.decoder(condition, z)
        return palettes.view(n_samples, 9, 3)


# ──────────────────────────────────────────────────────────────────────
# Helpers for v4.2 training: pack dominant colours into a normalised block.
# ──────────────────────────────────────────────────────────────────────

def pack_dominant_lab(dominant_lab, target_n=10):
    """Pad/truncate (n,3) dominant CIELAB centroids to a fixed-size flat tensor.

    Returns: torch.Tensor of shape (target_n * 3,), values normalised so each
    component is roughly in [-1, 1]: L*/100, a*/100, b*/100.
    """
    import numpy as np
    import torch as _torch

    arr = np.asarray(dominant_lab, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if len(arr) < target_n:
        pad = np.zeros((target_n - len(arr), 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:target_n]

    arr = arr / 100.0  # normalize to ~[-1, 1]
    return _torch.tensor(arr.flatten(), dtype=_torch.float32)
