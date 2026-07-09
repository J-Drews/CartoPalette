"""CartoPalette v4.2 inference pipeline — the main user-facing API.

v4.2 pipeline (default):
    Extract basemap colors → adaptive k → generate → score → guardrail filter
    → constrained ε-Pareto reranking with β · basemap_contrast tilt → top-3

Legacy v4.1 behaviour (MMR with λ=0.85) is preserved via the `reranker="mmr"`
argument and used as the ablation baseline.
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

# ── Default pipeline config (v4.2) ──
# v4.2 adds: constrained reranking + adaptive k + LAB contrast repair.
# Legacy "k" / "lambda_mmr" remain for backward compatibility with v4.1 checkpoints.
DEFAULT_PIPELINE_CONFIG = {
    "k": 20,
    "k_hard": 80,                   # Adaptive k: more CVAE candidates on hard basemaps
    "use_cartographic_candidates": True,
    "n_cartographic_candidates": 320,
    "lambda_mmr": 0.85,
    "n_suggestions": 3,
    "reranker": "constrained",      # "constrained" (v4.2) or "mmr" (v4.1 legacy)
    "beta_basemap": 0.20,           # Constrained reranker: extra weight on basemap_contrast
    "epsilon_constraints": {
        # ε-floors relative to the best candidate per non-target metric
        "distinguishability":  0.04,
        "cvd_robustness":      0.02,
        "perceptual_ordering": 0.02,
        "hue_contrast":        0.02,
        "lightness_contrast":  0.03,
    },
    "quality_floors": {
        "distinguishability_by_n": {3: 0.65, 4: 0.45, 5: 0.40, 7: 0.32, 9: 0.25},
        "cvd_robustness_by_n":     {3: 0.60, 4: 0.40, 5: 0.25, 7: 0.18, 9: 0.12},
        "perceptual_ordering": 0.85,
        "hue_contrast": 0.50,
        "lightness_contrast": 0.35,
    },
    "adaptive_k": {
        # Trigger conditions for using k_hard. Properties evaluated on basemap K-Means centroids.
        "enabled": True,
        "medium_L_lo": 25,          # Hard if median L* in [lo, hi]
        "medium_L_hi": 75,
        "chroma_threshold": 15,     # Hard if mean chroma > threshold (saturated basemap)
        "L_std_threshold": 14,      # Hard if L* variance > threshold (high contrast basemap)
    },
    "lab_repair": {
        # Post-selection surgical fix on the top-N palettes. Cheap (only on top-N),
        # gated by an accept rule: basemap_contrast must strictly improve and no
        # other metric may drop by more than ε.
        "enabled": True,
        "target_min_dE":  18.0,
        "trigger_min_dE": 12.0,
        "step_size_ab":    3.0,
        "max_step_size_L": 4.0,
        "max_iterations":   8,
        "allow_L_repair": True,
    },
    "guardrails": {
        "light_bm_threshold": 75,
        "medium_bm_threshold": 55,
        "min_basemap_contrast": 0.15,
        "sequential": {
            "light": {"min_mean_L": 32, "max_dark_colors": 2, "dark_threshold": 35},
            "medium": {"min_mean_L": 25},
        },
        "diverging": {
            "light": {"min_mean_L": 40, "max_dark_colors": 2, "dark_threshold": 35},
            "medium": {"min_mean_L": 30},
        },
    },
}


def _angle_distance_deg(a: float, b: float) -> float:
    """Smallest angular distance between two hue angles in degrees."""
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def _lab_from_lch(L: np.ndarray, C: np.ndarray, H_deg: np.ndarray) -> np.ndarray:
    """Build LAB colors from lightness, chroma, and hue arrays."""
    H = np.radians(H_deg)
    return np.stack([L, C * np.cos(H), C * np.sin(H)], axis=1)


def _roundtrip_to_display_lab(palette_lab: np.ndarray) -> np.ndarray:
    """Project LAB through clipped sRGB and back so scoring matches display."""
    from cartopalette.scoring.composite import _rgb_to_lab_pixel

    rgb = _lab_to_rgb(np.asarray(palette_lab, dtype=np.float64))
    return np.array([_rgb_to_lab_pixel(c) for c in rgb], dtype=np.float64)


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
    """CartoPalette v4.2 inference engine.

    v4.2 pipeline (default):
        1. Extract basemap dominant colors (K-Means in CIELAB).
        2. Adaptive k: more candidates on chromatically complex / medium-bright basemaps.
        3. Generate k candidates from the CVAE.
        4. Score with Composite Score v4.1 (six metrics).
        5. Darkness guardrails (v4.1b rules).
        6. Constrained ε-Pareto reranking with β · basemap_contrast tilt:
             utility = composite + β · basemap_contrast,
             subject to other metrics ≥ best_in_pool − ε.

    Usage:
        cp = CartoPalette("path/to/cartopalette_v4.pt")
        palettes = cp.suggest("basemap.png", scheme="sequential", n_classes=5)
        for p in palettes:
            print(p.hex_colors, p.score)

    Ablation: pass reranker="mmr" to fall back to the v4.1 MMR top-3.
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

        filtered = []
        for pal in palettes:
            palette_L = pal.lab[:, 0]
            mean_L = float(palette_L.mean())

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

        # Fallback: if all filtered out, return the best-scoring unfiltered palette
        if not filtered and palettes:
            best = max(palettes, key=lambda p: p.score if p.score is not None else 0)
            filtered = [best]

        return filtered

    @staticmethod
    def _palette_distance(pal_a: 'Palette', pal_b: 'Palette') -> float:
        """Unordered mean nearest-colour distance in CIELAB between palettes."""
        a = pal_a.lab
        b = pal_b.lab
        dists = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
        return float(0.5 * (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))))

    @staticmethod
    def _palette_hue_family(pal: 'Palette') -> float:
        """Chroma-weighted mean hue of a palette in degrees."""
        a = pal.lab[:, 1]
        b = pal.lab[:, 2]
        C = np.sqrt(a ** 2 + b ** 2)
        if float(C.sum()) < 1e-6:
            return 0.0
        weights = C / C.sum()
        sin_mean = np.sum(weights * np.sin(np.arctan2(b, a)))
        cos_mean = np.sum(weights * np.cos(np.arctan2(b, a)))
        return float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0)

    # ──────────────────────────────────────────────────────────────────────
    # v4.2 — Adaptive k (decide candidate pool size from basemap properties)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _basemap_difficulty(basemap_lab: np.ndarray, cfg: dict) -> dict:
        """Estimate basemap difficulty from K-Means dominant colors (CIELAB).

        Hard basemaps are those where (a) basemap brightness lies in the
        chromatically dense middle band, (b) basemap saturation is high, or
        (c) basemap lightness has high variance. On these the model benefits
        from a larger candidate pool because the multi-objective problem is
        harder to satisfy with few samples.
        """
        L = basemap_lab[:, 0]
        a = basemap_lab[:, 1]
        b = basemap_lab[:, 2]
        C = np.sqrt(a ** 2 + b ** 2)

        median_L = float(np.median(L))
        mean_C = float(np.mean(C))
        L_std = float(np.std(L))

        lo = cfg.get("medium_L_lo", 35)
        hi = cfg.get("medium_L_hi", 65)
        c_thr = cfg.get("chroma_threshold", 25)
        l_std_thr = cfg.get("L_std_threshold", 20)

        is_hard = (lo <= median_L <= hi) or (mean_C > c_thr) or (L_std > l_std_thr)

        return {
            "median_L": median_L,
            "mean_chroma": mean_C,
            "L_std": L_std,
            "is_hard": bool(is_hard),
        }

    @staticmethod
    def _basemap_hue_summary(basemap_lab: np.ndarray) -> dict:
        """Return dominant hue/chroma/lightness descriptors for generation."""
        L = basemap_lab[:, 0]
        a = basemap_lab[:, 1]
        b = basemap_lab[:, 2]
        C = np.sqrt(a ** 2 + b ** 2)
        angles = np.degrees(np.arctan2(b, a)) % 360.0
        weights = C / (C.sum() + 1e-10)
        sin_mean = np.sum(weights * np.sin(np.radians(angles)))
        cos_mean = np.sum(weights * np.cos(np.radians(angles)))
        mean_hue = float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0)
        high_chroma_hues = angles[C >= max(12.0, float(np.percentile(C, 60)))]
        return {
            "median_L": float(np.median(L)),
            "min_L": float(np.min(L)),
            "max_L": float(np.max(L)),
            "mean_chroma": float(np.mean(C)),
            "mean_hue": mean_hue,
            "high_chroma_hues": high_chroma_hues.tolist(),
        }

    @staticmethod
    def _candidate_hues(summary: dict) -> list:
        """Hue candidates biased away from dominant basemap hues."""
        mean_hue = summary["mean_hue"]
        high_hues = summary["high_chroma_hues"]
        offsets = [60, 90, 120, 150, 180, 210, 240, 270, 300]
        anchors = [10, 35, 65, 105, 145, 185, 220, 260, 300, 330]
        # Fixed anchors come first so a limited candidate budget still covers
        # warm, purple, blue, and teal families instead of only the first
        # complement direction of the basemap hue.
        raw = anchors + [((mean_hue + o) % 360.0) for o in offsets]

        hues = []
        for h in raw:
            if high_hues and min(_angle_distance_deg(h, bh) for bh in high_hues) < 25:
                continue
            if all(_angle_distance_deg(h, existing) >= 12 for existing in hues):
                hues.append(float(h))
        return hues or [float((mean_hue + 120) % 360), float((mean_hue + 240) % 360)]

    @staticmethod
    def _sequential_l_ranges(summary: dict) -> list:
        """Lightness ramps that avoid the basemap's dominant L* band."""
        med = summary["median_L"]
        has_light = summary["max_L"] > 72
        has_dark = summary["min_L"] < 28

        if med < 30 and not has_light:
            return [(45, 92), (55, 96), (35, 88)]
        if med > 70 and not has_dark:
            return [(15, 72), (20, 78), (30, 84)]
        if has_light and has_dark:
            return [(8, 92), (12, 82), (30, 96), (18, 70)]
        return [(12, 88), (18, 92), (25, 78), (35, 96)]

    @staticmethod
    def _generate_cartographic_labs(
        basemap_lab: np.ndarray,
        scheme_type: str,
        n_classes: int,
        max_candidates: int,
    ) -> list:
        """Generate deterministic, basemap-adaptive palettes without the CVAE.

        This is the v4.2 safety net: if the neural sampler misses a good part of
        LAB space, the tool still offers cartographically plausible candidates
        with strong figure-ground contrast.
        """
        summary = CartoPalette._basemap_hue_summary(basemap_lab)
        hues = CartoPalette._candidate_hues(summary)
        out = []

        if scheme_type == "sequential":
            l_ranges = CartoPalette._sequential_l_ranges(summary)
            chromas = [32, 44, 56, 68]
            hue_spans = [0, 18, -18, 36, -36]
            for lo, hi in l_ranges:
                for reverse in [False, True]:
                    for C0 in chromas:
                        for span in hue_spans:
                            for base_hue in hues:
                                L = np.linspace(lo, hi, n_classes)
                                if reverse:
                                    L = L[::-1]
                                H = (base_hue + np.linspace(0, span, n_classes)) % 360.0
                                C = np.full(n_classes, C0, dtype=np.float64)
                                if n_classes >= 7:
                                    C = C * np.linspace(0.92, 1.08, n_classes)
                                out.append(_roundtrip_to_display_lab(_lab_from_lch(L, C, H)))

        else:
            mid = n_classes // 2
            arm_chromas = [38, 52, 66]
            center_chromas = [0, 8, 16]
            center_L_values = [25, 35, 50, 65, 80, 92]
            hue_separations = [90, 120, 150, 180, 210]
            arm_hue_spans = [0, 25, 45]

            for left_hue in hues:
                for sep in hue_separations:
                    right_hue = (left_hue + sep) % 360.0
                    if summary["high_chroma_hues"]:
                        if min(_angle_distance_deg(right_hue, bh) for bh in summary["high_chroma_hues"]) < 25:
                            continue
                    for center_L in center_L_values:
                        for C_arm in arm_chromas:
                            for C_mid in center_chromas:
                                for arm_span in arm_hue_spans:
                                    if center_L >= 60:
                                        edge_L_values = [15, 25, 35, 45]
                                    else:
                                        edge_L_values = [72, 82, 92]
                                    for edge_L in edge_L_values:
                                        left_L = np.linspace(edge_L, center_L, mid + 1)
                                        right_L = np.linspace(center_L, edge_L, n_classes - mid)
                                        L = np.concatenate([left_L, right_L[1:]])
                                        H = np.empty(n_classes, dtype=np.float64)
                                        H[:mid] = (left_hue + np.linspace(-arm_span, 0, mid)) % 360.0
                                        H[mid] = (left_hue + sep / 2.0) % 360.0
                                        n_right = max(n_classes - mid - 1, 0)
                                        if n_right:
                                            H[mid + 1:] = (right_hue + np.linspace(0, arm_span, n_right)) % 360.0
                                        C = np.full(n_classes, C_arm, dtype=np.float64)
                                        if mid > 0:
                                            C[:mid] = np.linspace(C_arm * 1.15, C_arm * 0.85, mid)
                                        if n_right:
                                            C[mid + 1:] = np.linspace(C_arm * 0.85, C_arm * 1.15, n_right)
                                        C[mid] = C_mid
                                        out.append(_roundtrip_to_display_lab(_lab_from_lch(L, C, H)))

        # Deduplicate after RGB roundtrip; many LAB variants collapse to the same sRGB.
        unique = []
        seen = set()
        for pal in out:
            key = tuple(_rgb_to_hex(c) for c in _lab_to_rgb(pal))
            if key in seen:
                continue
            seen.add(key)
            unique.append(pal)
            if len(unique) >= max_candidates:
                break
        return unique

    def _make_quality_floors(self, n_classes: int) -> dict:
        cfg = self.pipeline_config.get("quality_floors",
                                       DEFAULT_PIPELINE_CONFIG["quality_floors"])
        return {
            "distinguishability": cfg.get("distinguishability_by_n", {}).get(n_classes, 0.35),
            "cvd_robustness": cfg.get("cvd_robustness_by_n", {}).get(n_classes, 0.25),
            "perceptual_ordering": cfg.get("perceptual_ordering", 0.85),
            "hue_contrast": cfg.get("hue_contrast", 0.50),
            "lightness_contrast": cfg.get("lightness_contrast", 0.35),
        }

    def _resolve_k(self, basemap_lab: np.ndarray, user_k: Optional[int]) -> tuple:
        """Return (k, difficulty_dict). user_k=None → adaptive choice."""
        adaptive_cfg = self.pipeline_config.get("adaptive_k",
                                                DEFAULT_PIPELINE_CONFIG["adaptive_k"])
        difficulty = self._basemap_difficulty(basemap_lab, adaptive_cfg)

        if user_k is not None:
            return int(user_k), difficulty

        if not adaptive_cfg.get("enabled", True):
            return int(self.pipeline_config.get("k", 20)), difficulty

        k_easy = int(self.pipeline_config.get("k", 20))
        k_hard = int(self.pipeline_config.get("k_hard", 80))
        return (k_hard if difficulty["is_hard"] else k_easy), difficulty

    # ──────────────────────────────────────────────────────────────────────
    # v4.2 — Constrained Reranking (ε-Pareto)
    # ──────────────────────────────────────────────────────────────────────

    def _constrained_rerank(
        self,
        palettes: list,
        n: int,
        beta_basemap: float,
        epsilon: dict,
        quality_floors: Optional[dict] = None,
        diversity_weight: float = 0.15,
    ) -> list:
        """ε-Pareto constrained reranking — v4.2 default.

        Procedure:
          1. Reference candidate = max composite score.
          2. Floors = reference_metric − ε per non-target metric.
          3. Valid set = candidates that meet ALL floors. Two-level fallback if
             the valid set is smaller than n: first relax ε by 1.5×, then
             fall back to all palettes (degenerates to MMR-like selection).
          4. Utility(p) = composite(p) + β · basemap_contrast(p). The β term
             tilts selection toward palettes with stronger basemap contrast
             without changing the underlying Composite Score definition.
          5. Top-1 = argmax utility on valid set.
          6. Picks 2..n add a diversity term to avoid visually redundant
             suggestions (analogous to MMR but on the utility scale).

        Returns ordered list of up to n palettes.
        """
        if not palettes or n <= 0:
            return []

        # Filter out palettes that have no metric breakdown (shouldn't happen,
        # but defensive — fall back to score-only selection in that case).
        scored = [p for p in palettes if p.metrics is not None]
        if not scored:
            scored = list(palettes)
            return scored[:n]

        # Reference candidate = best composite among the survivors
        ref = max(scored, key=lambda p: p.score if p.score is not None else 0.0)
        ref_metrics = ref.metrics or {}
        quality_floors = quality_floors or {}

        def floors_from(eps_mult: float) -> dict:
            floors = {
                m: max(0.0, ref_metrics.get(m, 0.0) - eps_mult * eps)
                for m, eps in epsilon.items()
            }
            for m, floor in quality_floors.items():
                floors[m] = max(floors.get(m, 0.0), float(floor))
            return floors

        def valid_with(floors: dict) -> list:
            out = []
            for p in scored:
                pm = p.metrics or {}
                if all(pm.get(m, 0.0) >= fv for m, fv in floors.items()):
                    out.append(p)
            return out

        # Product mode: absolute quality floors are the hard guard. Relative
        # epsilon floors are useful when no absolute floors are configured, but
        # with large hybrid pools they can overfit to one metric of the best
        # composite candidate and exclude excellent basemap-contrast palettes.
        if quality_floors:
            valid = valid_with(quality_floors)
        else:
            valid = valid_with(floors_from(1.0))
            if not valid:
                valid = valid_with(floors_from(1.5))
        if not valid:
            valid = scored

        # Utility = composite + β · basemap_contrast
        def utility(p):
            bc = (p.metrics or {}).get("basemap_contrast", 0.0)
            return (p.score if p.score is not None else 0.0) + beta_basemap * bc

        if len(valid) < n:
            valid_ids = {id(p) for p in valid}
            relaxed_floors = {
                m: (v if m == "distinguishability" else v * 0.7)
                for m, v in quality_floors.items()
            }
            relaxed = valid_with(relaxed_floors) if quality_floors else scored
            relaxed_ids = {id(p) for p in relaxed}
            backfill_pool = relaxed if len(relaxed) > len(valid) else scored
            backfill = [
                p for p in sorted(backfill_pool, key=utility, reverse=True)
                if id(p) not in valid_ids and (not quality_floors or id(p) in relaxed_ids or backfill_pool is scored)
            ]
            valid = valid + backfill[: n - len(valid)]

        utils = np.array([utility(p) for p in valid])
        u_min, u_max = float(utils.min()), float(utils.max())
        if u_max > u_min:
            u_norm = (utils - u_min) / (u_max - u_min)
        else:
            u_norm = np.ones_like(utils)

        # Top-1: max utility (no diversity term)
        first_idx = int(np.argmax(utils))
        selected_idx = [first_idx]
        remaining = [i for i in range(len(valid)) if i != first_idx]

        # Picks 2..n: utility + diversity
        hue_families = [self._palette_hue_family(p) for p in valid]
        while len(selected_idx) < n and remaining:
            best_score = -np.inf
            best_idx = remaining[0]
            for i in remaining:
                # Diversity = min CIELAB distance to any selected palette, normalised
                min_dist = min(
                    self._palette_distance(valid[i], valid[s]) for s in selected_idx
                )
                if min_dist < 4.0 and len(remaining) > 1:
                    continue
                div_norm = min(min_dist / 50.0, 1.0)
                min_hue_dist = min(
                    _angle_distance_deg(hue_families[i], hue_families[s])
                    for s in selected_idx
                )
                if min_hue_dist < 22.0 and len(remaining) > 1:
                    continue
                hue_norm = min(min_hue_dist / 100.0, 1.0)
                combined = (
                    (1.0 - diversity_weight - 0.12) * u_norm[i]
                    + diversity_weight * div_norm
                    + 0.12 * hue_norm
                )
                if combined > best_score:
                    best_score = combined
                    best_idx = i
            selected_idx.append(best_idx)
            remaining.remove(best_idx)

        return [valid[i] for i in selected_idx]

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
        k: Optional[int] = None,
        lambda_mmr: Optional[float] = None,
        reranker: Optional[str] = None,
        beta_basemap: Optional[float] = None,
        apply_repair: Optional[bool] = None,
        return_diagnostics: bool = False,
    ) -> list:
        """Generate palette suggestions for a basemap.

        v4.2 pipeline:
            1. Extract basemap dominant colours (K-Means)
            2. Decide k adaptively from basemap difficulty (override with k=…)
            3. Generate k candidates from CVAE
            4. Score all candidates with Composite Score v4.1
            5. Guardrail filter (darkness rules)
            6. Reranker:
                 "constrained"  — ε-Pareto reranking with β · basemap_contrast tilt (v4.2 default)
                 "mmr"          — classic MMR (v4.1 legacy, kept for ablation)
            7. LAB contrast repair on top-N (a*b*-priority, accept-gated)

        Args:
            image: File path (str/Path), PIL Image, or numpy array of the basemap.
            scheme: "sequential" or "diverging".
            n_classes: Number of colours (3, 4, 5, 7, or 9).
            scale: Map scale — "overview", "regional", or "local".
            n_suggestions: Number of palettes to return (default 3).
            k: Number of candidates. None → adaptive (k or k_hard depending on basemap).
            lambda_mmr: Only used when reranker="mmr". Default from pipeline_config.
            reranker: "constrained" (default, v4.2) or "mmr" (v4.1 legacy).
            beta_basemap: β weight on basemap_contrast in the constrained reranker.
            apply_repair: Run LAB contrast repair after reranking. None → use pipeline_config.
            return_diagnostics: If True, return (palettes, diagnostics_dict).

        Returns:
            List of Palette objects (or tuple if return_diagnostics=True).
        """
        # Validate inputs
        if scheme not in VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {VALID_SCHEMES}, got '{scheme}'")
        if n_classes not in VALID_NCLASSES:
            raise ValueError(f"n_classes must be one of {VALID_NCLASSES}, got {n_classes}")
        if scale not in VALID_SCALES:
            raise ValueError(f"scale must be one of {VALID_SCALES}, got '{scale}'")

        # Resolve pipeline parameters (CLI args > pipeline_config > defaults)
        reranker = reranker or self.pipeline_config.get("reranker", "constrained")
        if reranker not in ("constrained", "mmr"):
            raise ValueError(f"reranker must be 'constrained' or 'mmr', got '{reranker}'")
        lambda_mmr_eff = (lambda_mmr if lambda_mmr is not None
                          else self.pipeline_config.get("lambda_mmr", 0.85))
        beta_eff = (beta_basemap if beta_basemap is not None
                    else self.pipeline_config.get(
                        "beta_basemap", DEFAULT_PIPELINE_CONFIG["beta_basemap"]
                    ))
        epsilon_cfg = self.pipeline_config.get("epsilon_constraints",
                                               DEFAULT_PIPELINE_CONFIG["epsilon_constraints"])

        # Prepare model inputs
        img_tensor = self._load_image(image)
        metadata = self._build_metadata(scheme, n_classes, scale).unsqueeze(0).to(self.device)

        # ── Step 1: Extract basemap dominant colors EARLY (needed for adaptive k AND scoring) ──
        try:
            from cartopalette.scoring.composite import score_palette_detailed, extract_basemap_colors
            scoring_available = True
        except ImportError:
            scoring_available = False

        basemap_colors = None
        if scoring_available:
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

        # ── Step 2: Decide k adaptively ──
        if basemap_colors is not None:
            k_eff, difficulty = self._resolve_k(basemap_colors, k)
        else:
            # No scoring → no difficulty info → fall back to default k
            k_eff = int(k if k is not None else self.pipeline_config.get("k", 20))
            difficulty = {"is_hard": False, "median_L": None, "mean_chroma": None, "L_std": None}

        # ── Step 3: Generate k candidates (iid from latent space) ──
        cnn_features = self.model.cnn(img_tensor)

        # If the trained model supports the v4.2 dominant-colour block, pack
        # the K-Means centroids and feed them as the third condition component.
        dominant_block = None
        if (getattr(self.model, "dominant_colors_dim", 0) > 0
                and basemap_colors is not None):
            from cartopalette.core.model import pack_dominant_lab
            target_n = self.model.dominant_colors_dim // 3
            dominant_block = pack_dominant_lab(basemap_colors, target_n=target_n)
            dominant_block = dominant_block.unsqueeze(0).to(self.device)

        condition = self.model.build_condition(cnn_features, metadata, dominant_block)
        condition = condition.repeat(k_eff, 1)
        z = torch.randn(k_eff, self.model.latent_dim, device=self.device)
        raw = self.model.decoder(condition, z)
        raw = raw.view(k_eff, 9, 3).cpu().numpy()

        palettes = []
        for i in range(k_eff):
            lab = self._denormalize(raw[i], n_classes)
            palettes.append(Palette(lab, scheme, n_classes))

        # ── Step 4: Score candidates ──
        if scoring_available and basemap_colors is not None:
            for pal in palettes:
                details = score_palette_detailed(pal.lab, basemap_colors, scheme)
                pal.score = details["composite"]
                pal.metrics = details
        else:
            for pal in palettes:
                pal.score = 0.5
                pal.metrics = None

        n_cartographic = 0
        if (reranker == "constrained" and scoring_available and basemap_colors is not None
                and self.pipeline_config.get("use_cartographic_candidates", True)):
            max_cart = int(self.pipeline_config.get("n_cartographic_candidates", 320))
            cart_labs = self._generate_cartographic_labs(
                basemap_colors, scheme, n_classes, max_candidates=max_cart
            )
            for lab in cart_labs:
                pal = Palette(lab, scheme, n_classes)
                details = score_palette_detailed(pal.lab, basemap_colors, scheme)
                pal.score = details["composite"]
                pal.metrics = details
                pal._source = "cartographic"
                palettes.append(pal)
            n_cartographic = len(cart_labs)

        # ── Step 5: Guardrails ──
        basemap_mean_L = (difficulty["median_L"] if difficulty["median_L"] is not None
                          else self._get_basemap_mean_L(image))
        filtered = self._apply_guardrails(
            palettes, scheme, basemap_mean_L,
            basemap_colors if basemap_colors is not None else np.zeros((1, 3)),
        )

        # ── Step 6: Reranking ──
        if reranker == "constrained":
            selected = self._constrained_rerank(
                filtered, n_suggestions, beta_basemap=beta_eff, epsilon=epsilon_cfg,
                quality_floors=self._make_quality_floors(n_classes),
            )
        else:
            selected = self._mmr_select(filtered, n_suggestions, lambda_mmr_eff)

        # ── Step 7: LAB contrast repair (post-selection, surgical) ──
        repair_cfg = self.pipeline_config.get("lab_repair", DEFAULT_PIPELINE_CONFIG["lab_repair"])
        repair_on = repair_cfg.get("enabled", True) if apply_repair is None else apply_repair
        repair_stats = {"applied": False, "n_changed": 0}
        if repair_on and basemap_colors is not None and selected:
            try:
                from cartopalette.core.repair import repair_palette_list
                # Pass repair config minus the "enabled" flag
                rcfg = {k_: v for k_, v in repair_cfg.items() if k_ != "enabled"}
                repaired = repair_palette_list(selected, basemap_colors, config=rcfg)
                n_changed = sum(
                    1 for orig, new in zip(selected, repaired)
                    if orig is not new
                )
                selected = repaired
                repair_stats = {"applied": True, "n_changed": n_changed}
            except ImportError:
                repair_stats = {"applied": False, "n_changed": 0, "reason": "repair module unavailable"}

        if return_diagnostics:
            diag = {
                "k": k_eff,
                "n_cartographic_candidates": n_cartographic,
                "n_total_candidates": len(palettes),
                "k_decision": "adaptive_hard" if (k is None and difficulty["is_hard"])
                              else "adaptive_easy" if k is None
                              else "user_override",
                "basemap_difficulty": difficulty,
                "n_candidates_after_guardrails": len(filtered),
                "n_candidates_rejected": k_eff - len(filtered),
                "reranker": reranker,
                "beta_basemap": beta_eff if reranker == "constrained" else None,
                "lambda_mmr": lambda_mmr_eff if reranker == "mmr" else None,
                "lab_repair": repair_stats,
            }
            return selected, diag

        return selected

    @torch.no_grad()
    def get_cnn_features(self, image) -> np.ndarray:
        """Extract CNN features for a basemap (for debugging/analysis)."""
        img_tensor = self._load_image(image)
        features = self.model.cnn(img_tensor)
        return features.cpu().numpy().flatten()

    def __repr__(self):
        return f"CartoPalette(v4.2, epoch={self.best_epoch})"
