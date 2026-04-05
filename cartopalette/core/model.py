"""CartoPalette v4.1 CVAE model architecture (must match training notebook exactly)."""

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
    """CartoPalette v4.1 CVAE: Basemap image -> Projected CNN features -> CVAE -> Palette.

    Condition = [projected_features (256) + metadata (10)] = 266 dims.
    v4.1: Only sequential + diverging (no qualitative), metadata_dim = 10.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]

        self.cnn = BasemapEncoder(projection_dim=config["cnn_projection_dim"])
        self.metadata_scale = nn.Parameter(torch.ones(10) * 1.0)

        condition_dim = config["condition_dim"]
        self.encoder = CVAEEncoder(
            condition_dim, config["palette_dim"], config["hidden_dim"], config["latent_dim"]
        )
        self.decoder = CVAEDecoder(
            condition_dim, config["latent_dim"], config["hidden_dim"], config["palette_dim"]
        )

    def build_condition(self, cnn_features, metadata):
        scaled_meta = metadata * self.metadata_scale
        return torch.cat([cnn_features, scaled_meta], dim=-1)

    def forward(self, image, metadata, palette):
        cnn_features = self.cnn(image)
        condition = self.build_condition(cnn_features, metadata)
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
    def generate(self, image, metadata, n_samples=5):
        self.eval()
        cnn_features = self.cnn(image)
        condition = self.build_condition(cnn_features, metadata)
        condition = condition.repeat(n_samples, 1)
        z = torch.randn(n_samples, self.latent_dim, device=image.device)
        palettes = self.decoder(condition, z)
        return palettes.view(n_samples, 9, 3)
