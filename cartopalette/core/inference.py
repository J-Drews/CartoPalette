"""CartoPalette v4.1 inference pipeline — the main user-facing API.

Pipeline: Generate k=20 → Composite Score → Guardrail Filter → MMR Top-3 (λ=0.85)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cartopalette.core.model import ChromaMapCVAE


# ── Encoding maps (must match training) ──
SCHEME_TO_IDX = {"sequential": 0, "diverging": 1}
SCALE_TO_IDX = {"overview": 0, "regional": 1, "local": 2}
NCLASSES_TO_IDX = {3: 0, 4: 1, 5: 2, 7: 3, 9: 4}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_SCHEMES = list(SCHEME_TO_IDX.keys())
VALID_SCALES = list(SCALE_TO_IDX.keys())
VALID_NCLASSES = list(NCLASSES_TO_IDX.keys())

# ── Default pipeline config (v4.1) ──
DEFAULT_PIPELINE_CONFIG = {
    "k": 30,
    "lambda_mmr": 0.85,
    "n_suggestions": 3,
    "guardrails": {
        "light_bm_threshold": 75,
        "medium_bm_threshold": 55,
        "min_basemap_contrast": 0.15,
        "min_distinguishability": 0.28,
        "sequential": {
            "light": {"min_mean_L": 45, "max_dark_colors": 1, "dark_threshold": 35},
            "medium": {"min_mean_L": 30},
        },
        "diverging": {
            "light": {"min_mean_L": 40, "max_dark_colors": 2, "dark_threshold": 35},
            "medium": {"min_mean_L": 30},
        },
    },
}


def _lab_to_rgb(lab_colors: np.ndarray) -> np.ndarray:
    """Convert CIELAB (N, 3) to sRGB (N, 3) clipped to [0, 1]."""
    L, a, b = lab_colors[:, 0], lab_colors[:, 1], lab_colors[:, 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    x = np.where(fx > delta, fx**3, 3 * delta**2 * (fx - 4.0 / 29.0))
    y = np.where(fy > delta, fy**3, 3 * delta**2 * (fy - 4.0 / 29.0))
    z = np.where(fz > delta, fz**3, 3 * delta**2 * (fz - 4.0 / 29.0))

    x *= 0.95047
    z *= 1.08883

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    def gamma(c):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055)

    rgb = np.stack([gamma(r_lin), gamma(g_lin), gamma(b_lin)], axis=1)
    return np.clip(rgb, 0.0, 1.0)


def _rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert single RGB [0,1] color to hex string."""
    r, g, b = (np.clip(rgb * 255, 0, 255)).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class Palette:
    """A generated color palette with metadata."""

    def __init__(self, lab: np.ndarray, scheme_type: str, n_classes: int,
                 score: Optional[float] = None, metrics: Optional[dict] = None):
        self.lab = lab  # (n_classes, 3) CIELAB values
        self.scheme_type = scheme_type
        self.n_classes = n_classes
        self.score = score
        self.metrics = metrics  # dict with individual metric scores (from score_palette_detailed)

    @property
    def rgb(self) -> np.ndarray:
        """sRGB values (n_classes, 3) in [0, 1]."""
        return _lab_to_rgb(self.lab)

    @property
    def rgb_255(self) -> np.ndarray:
        """sRGB values (n_classes, 3) in [0, 255] as integers."""
        return (self.rgb * 255).astype(int)

    @property
    def hex_colors(self) -> list:
        """List of hex color strings."""
        return [_rgb_to_hex(c) for c in self.rgb]

    def __repr__(self):
        colors = " ".join(self.hex_colors)
        score_str = f", score={self.score:.3f}" if self.score is not None else ""
        return f"Palette({self.scheme_type}/{self.n_classes}cls: {colors}{score_str})"


class CartoPalette:
    """CartoPalette v4.1 inference engine.

    Pipeline: Generate k=20 → Score → Guardrail Filter → MMR Top-3 (λ=0.85)

    Usage:
        cp = CartoPalette("path/to/cartopalette_v4.pt")
        palettes = cp.suggest("basemap.png", scheme="sequential", n_classes=5)
        for p in palettes:
            print(p.hex_colors, p.score)
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """Load a trained CartoPalette model.

        Args:
            model_path: Path to exported .pt model file.
            device: "cuda", "cpu", or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]

        # Pipeline config (from export or defaults)
        self.pipeline_config = checkpoint.get("pipeline_config", DEFAULT_PIPELINE_CONFIG)

        # Build and load model
        self.model = ChromaMapCVAE(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Store metadata
        self.best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", "?"))

    def _build_metadata(self, scheme_type: str, n_classes: int, scale_class: str) -> torch.Tensor:
        """Build one-hot metadata vector (10,) for v4.1."""
        n_classes_oh = np.zeros(5, dtype=np.float32)
        n_classes_oh[NCLASSES_TO_IDX[n_classes]] = 1.0
        scheme_oh = np.zeros(2, dtype=np.float32)
        scheme_oh[SCHEME_TO_IDX[scheme_type]] = 1.0
        scale_oh = np.zeros(3, dtype=np.float32)
        scale_oh[SCALE_TO_IDX[scale_class]] = 1.0
        return torch.tensor(np.concatenate([n_classes_oh, scheme_oh, scale_oh]))

    def _denormalize(self, palette_norm: np.ndarray, n_colors: int) -> np.ndarray:
        """Convert model output to CIELAB."""
        pal = palette_norm.copy()
        pal[:, 0] *= 100.0  # L*
        pal[:, 1] *= 128.0  # a*
        pal[:, 2] *= 128.0  # b*
        return pal[:n_colors]

    def _load_image(self, image_input) -> torch.Tensor:
        """Load and transform an image from path, PIL Image, or ndarray."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _get_basemap_mean_L(self, image_input) -> float:
        """Estimate mean L* of the basemap for guardrail classification."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            return 50.0  # default to medium

        # Quick estimate: sample center region, convert to L*
        img_small = img.resize((64, 64))
        pixels = np.array(img_small).astype(np.float64) / 255.0
        # Approximate L* from luminance: Y = 0.2126R + 0.7152G + 0.0722B
        luminance = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
        # Approximate CIE L* from Y
        Y = luminance.mean()
        L_star = 116.0 * (Y ** (1/3)) - 16.0 if Y > 0.008856 else 903.3 * Y
        return float(L_star)

    def _apply_guardrails(self, palettes: list, scheme_type: str, basemap_mean_L: float,
                          basemap_colors: np.ndarray) -> list:
        """Apply v4.1b darkness guardrail filters.

        Returns filtered list. Falls back to best unfiltered palette if all rejected.
        """
        gc = self.pipeline_config.get("guardrails", DEFAULT_PIPELINE_CONFIG["guardrails"])
        light_thresh = gc.get("light_bm_threshold", 75)
        medium_thresh = gc.get("medium_bm_threshold", 55)
        min_bm_contrast = gc.get("min_basemap_contrast", 0.15)
        min_distinguish = gc.get("min_distinguishability", 0.28)

        filtered = []
        for pal in palettes:
            palette_L = pal.lab[:, 0]
            mean_L = float(palette_L.mean())

            # Rule: minimum distinguishability — reject palettes with near-identical colors
            if pal.metrics is not None:
                if pal.metrics.get("distinguishability", 1.0) < min_distinguish:
                    continue

            # Rule: basemap_contrast minimum
            if pal.score is not None and hasattr(pal, '_basemap_contrast'):
                if pal._basemap_contrast < min_bm_contrast:
                    continue

            scheme_rules = gc.get(scheme_type, {})

            if basemap_mean_L > light_thresh:
                rules = scheme_rules.get("light", {})
                min_mean = rules.get("min_mean_L", 40)
                max_dark = rules.get("max_dark_colors", 2)
                dark_thresh = rules.get("dark_threshold", 35)

                if mean_L < min_mean:
                    continue
                n_dark = int(np.sum(palette_L < dark_thresh))
                if n_dark > max_dark:
                    continue

            elif basemap_mean_L > medium_thresh:
                rules = scheme_rules.get("medium", {})
                min_mean = rules.get("min_mean_L", 30)
                if mean_L < min_mean:
                    continue

            # Dark basemaps: no darkness filter needed

            filtered.append(pal)

        # Fallback: if all filtered out, pick palettes with highest distinguishability
        if not filtered and palettes:
            by_dist = sorted(
                palettes,
                key=lambda p: p.metrics.get("distinguishability", 0) if p.metrics else 0,
                reverse=True,
            )
            filtered = [by_dist[0]]

        return filtered

    @staticmethod
    def _palette_distance(pal_a: 'Palette', pal_b: 'Palette') -> float:
        """Mean color-wise Euclidean distance in CIELAB between two palettes."""
        n = min(len(pal_a.lab), len(pal_b.lab))
        return float(np.mean(np.sqrt(np.sum((pal_a.lab[:n] - pal_b.lab[:n]) ** 2, axis=1))))

    @staticmethod
    def _mmr_select(palettes: list, n: int, lambda_mmr: float) -> list:
        """MMR (Maximal Marginal Relevance) selection.

        Selects n palettes balancing quality (λ) and diversity (1-λ).
        λ=1.0: pure quality ranking. λ=0.0: pure diversity.

        Algorithm:
        1. Pick highest-scoring palette.
        2. For each subsequent pick:
           combined = λ * score_normalized - (1-λ) * max_similarity_to_selected
        3. Pick candidate maximizing combined score.
        """
        if not palettes or n <= 0:
            return []

        # Sort by score descending first
        scored = [(i, p.score if p.score is not None else 0) for i, p in enumerate(palettes)]

        # Normalize scores to [0, 1]
        scores = np.array([s for _, s in scored])
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = np.ones_like(scores)

        selected_indices = []
        remaining = list(range(len(palettes)))

        # First pick: highest quality
        best_idx = int(np.argmax(scores_norm))
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

        # Subsequent picks: MMR
        for _ in range(min(n - 1, len(remaining))):
            best_mmr = -float("inf")
            best_cand = remaining[0]

            for idx in remaining:
                # Quality term
                quality = scores_norm[idx]

                # Diversity term: min distance to any selected palette
                min_dist = min(
                    CartoPalette._palette_distance(palettes[idx], palettes[sel])
                    for sel in selected_indices
                )
                # Normalize (typical CIELAB distances 0-60 dE)
                div_norm = min(min_dist / 50.0, 1.0)

                # MMR score
                mmr = lambda_mmr * quality + (1 - lambda_mmr) * div_norm

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_cand = idx

            selected_indices.append(best_cand)
            remaining.remove(best_cand)

        return [palettes[i] for i in selected_indices]

    @torch.no_grad()
    def suggest(
        self,
        image,
        scheme: str = "sequential",
        n_classes: int = 5,
        scale: str = "regional",
        n_suggestions: int = 3,
        k: int = 30,
        lambda_mmr: float = 0.85,
    ) -> list:
        """Generate palette suggestions for a basemap.

        v4.1 pipeline: Generate k → Score → Guardrail Filter → MMR Top-n

        Args:
            image: File path (str/Path), PIL Image, or numpy array of the basemap.
            scheme: "sequential" or "diverging".
            n_classes: Number of colors (3, 4, 5, 7, or 9).
            scale: Map scale — "overview", "regional", or "local".
            n_suggestions: Number of palettes to return (default 3).
            k: Number of candidates to generate (default 20).
            lambda_mmr: MMR quality-diversity tradeoff (default 0.85).

        Returns:
            List of Palette objects, selected by MMR (best quality + diversity).
        """
        # Validate inputs
        if scheme not in VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {VALID_SCHEMES}, got '{scheme}'")
        if n_classes not in VALID_NCLASSES:
            raise ValueError(f"n_classes must be one of {VALID_NCLASSES}, got {n_classes}")
        if scale not in VALID_SCALES:
            raise ValueError(f"scale must be one of {VALID_SCALES}, got '{scale}'")

        # Prepare inputs
        img_tensor = self._load_image(image)
        metadata = self._build_metadata(scheme, n_classes, scale).unsqueeze(0).to(self.device)

        # Generate k candidates (iid from latent space)
        cnn_features = self.model.cnn(img_tensor)
        condition = self.model.build_condition(cnn_features, metadata)
        condition = condition.repeat(k, 1)
        z = torch.randn(k, self.model.latent_dim, device=self.device)
        raw = self.model.decoder(condition, z)
        raw = raw.view(k, 9, 3).cpu().numpy()

        # Denormalize and create Palette objects
        palettes = []
        for i in range(k):
            lab = self._denormalize(raw[i], n_classes)
            palettes.append(Palette(lab, scheme, n_classes))

        # Score using composite score (with individual metric breakdown)
        try:
            from cartopalette.scoring.composite import score_palette_detailed, extract_basemap_colors

            if isinstance(image, (str, Path)):
                basemap_colors = extract_basemap_colors(str(image))
            else:
                import tempfile
                if isinstance(image, np.ndarray):
                    pil_img = Image.fromarray(image)
                else:
                    pil_img = image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    pil_img.save(f.name)
                    basemap_colors = extract_basemap_colors(f.name)

            for pal in palettes:
                details = score_palette_detailed(pal.lab, basemap_colors, scheme)
                pal.score = details["composite"]
                pal.metrics = details

        except ImportError:
            # Scoring unavailable — assign random scores
            for pal in palettes:
                pal.score = 0.5
                pal.metrics = None

        # Apply guardrail filters
        basemap_mean_L = self._get_basemap_mean_L(image)
        filtered = self._apply_guardrails(palettes, scheme, basemap_mean_L,
                                          basemap_colors if 'basemap_colors' in dir() else np.zeros((1, 3)))

        # MMR selection for diverse top-n
        selected = self._mmr_select(filtered, n_suggestions, lambda_mmr)

        return selected

    @torch.no_grad()
    def get_cnn_features(self, image) -> np.ndarray:
        """Extract CNN features for a basemap (for debugging/analysis)."""
        img_tensor = self._load_image(image)
        features = self.model.cnn(img_tensor)
        return features.cpu().numpy().flatten()

    def __repr__(self):
        return f"CartoPalette(v4.1, epoch={self.best_epoch})"
