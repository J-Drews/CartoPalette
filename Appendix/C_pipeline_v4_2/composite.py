"""CartoPalette v4 — Composite Score for ranking generated palettes.

Metrics (6 total, 80% basemap-dependent, 20% universal):
    Basemap-dependent (80%):
        1. Basemap Contrast (0.20) — worst-case + mean CIEDE2000 to basemap colors
        2. Lightness Contrast (0.30) — palette L* zone avoidance vs basemap
        3. Hue Contrast (0.30) — chroma-weighted hue separation from basemap

    Universal quality (20%):
        4. Distinguishability (0.10) — min pairwise CIEDE2000 in palette
        5. CVD Robustness (0.05) — distinguishability under simulated CVD
        6. Perceptual Ordering (0.05) — L* monotonicity (seq/div only)

Kartographische Grundlagen:
    Brewer 1994, Harrower & Brewer 2003, Kovesi 2015, Brettel et al. 1997
"""

import math
import numpy as np
from PIL import Image


# ══════════════════════════════════════════════════════════
# CIEDE2000 (scalar, numpy)
# ══════════════════════════════════════════════════════════

def ciede2000(lab1, lab2):
    """CIEDE2000 distance between two CIELAB colors."""
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
    eps = 1e-10

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg**7
    G = 0.5 * (1.0 - math.sqrt(C_avg7 / (C_avg7 + 25.0**7 + eps)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)

    h1p = math.degrees(math.atan2(b1, a1p + eps)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p + eps)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp_diff = h2p - h1p
    if abs(dhp_diff) <= 180.0:
        dhp = dhp_diff
    elif dhp_diff > 180.0:
        dhp = dhp_diff - 360.0
    else:
        dhp = dhp_diff + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p + eps) * math.sin(math.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    hp_diff_abs = abs(h1p - h2p)
    hp_sum = h1p + h2p
    if hp_diff_abs <= 180.0:
        hp_avg = hp_sum / 2.0
    elif hp_sum < 360.0:
        hp_avg = (hp_sum + 360.0) / 2.0
    else:
        hp_avg = (hp_sum - 360.0) / 2.0

    T = (1.0
         - 0.17 * math.cos(math.radians(hp_avg - 30.0))
         + 0.24 * math.cos(math.radians(2.0 * hp_avg))
         + 0.32 * math.cos(math.radians(3.0 * hp_avg + 6.0))
         - 0.20 * math.cos(math.radians(4.0 * hp_avg - 63.0)))

    SL = 1.0 + 0.015 * (Lp_avg - 50.0)**2 / math.sqrt(20.0 + (Lp_avg - 50.0)**2 + eps)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RT = (-2.0 * math.sqrt(Cp_avg7 / (Cp_avg7 + 25.0**7 + eps))
          * math.sin(math.radians(60.0 * math.exp(-((hp_avg - 275.0) / 25.0)**2))))

    dE = math.sqrt(
        (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return dE


# ══════════════════════════════════════════════════════════
# Basemap color extraction
# ══════════════════════════════════════════════════════════

def _rgb_to_lab_pixel(rgb):
    """Convert single sRGB [0,1] to CIELAB."""
    # sRGB -> linear
    def inv_gamma(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = inv_gamma(rgb[0]), inv_gamma(rgb[1]), inv_gamma(rgb[2])

    # Linear RGB -> XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.2591332 * g + 0.9503041 * b

    # Normalize to D65 white
    x /= 0.95047
    z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116

    L = 116.0 * f(y) - 16.0
    a = 500.0 * (f(x) - f(y))
    b_val = 200.0 * (f(y) - f(z))
    return np.array([L, a, b_val])


def extract_basemap_colors(image_path, k=10):
    """Extract k dominant colors from basemap using simple K-Means in CIELAB.

    Uses a lightweight implementation to avoid sklearn dependency.
    """
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float64) / 255.0
    pixels = img.reshape(-1, 3)

    # Subsample
    rng = np.random.RandomState(42)
    if len(pixels) > 5000:
        idx = rng.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    # Convert to LAB
    lab_pixels = np.array([_rgb_to_lab_pixel(p) for p in pixels])

    # Simple K-Means
    centroids = lab_pixels[rng.choice(len(lab_pixels), k, replace=False)]
    for _ in range(20):
        dists = np.linalg.norm(lab_pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            lab_pixels[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=0.5):
            break
        centroids = new_centroids

    return centroids


# ══════════════════════════════════════════════════════════
# CVD Simulation (Brettel)
# ══════════════════════════════════════════════════════════

_RGB_TO_LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
])
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)

_PROTAN_SIM = np.array([
    [0.0, 1.05118294, -0.05116099],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

_DEUTAN_SIM = np.array([
    [1.0, 0.0, 0.0],
    [0.9513092, 0.0, 0.04866992],
    [0.0, 0.0, 1.0],
])


def _simulate_cvd(rgb_colors, cvd_type="protan"):
    """Simulate color vision deficiency on sRGB (N, 3)."""
    lms = rgb_colors @ _RGB_TO_LMS.T
    sim = _PROTAN_SIM if cvd_type == "protan" else _DEUTAN_SIM
    return np.clip(lms @ sim.T @ _LMS_TO_RGB.T, 0, 1)


def _lab_to_rgb_simple(lab):
    """Convert CIELAB (N,3) to sRGB (N,3). Self-contained, no torch dependency."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
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


def _rgb_to_lab_batch(rgb):
    """Convert sRGB (N,3) to CIELAB (N,3)."""
    return np.array([_rgb_to_lab_pixel(c) for c in rgb])


# ══════════════════════════════════════════════════════════
# Individual metrics
# ══════════════════════════════════════════════════════════

def _min_pairwise_dE(palette_lab):
    """Minimum pairwise CIEDE2000 in palette."""
    n = len(palette_lab)
    min_dE = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dE = ciede2000(palette_lab[i], palette_lab[j])
            if dE < min_dE:
                min_dE = dE
    return min_dE


def _distinguishability(palette_lab, threshold=30.0):
    """Score: min_pairwise_dE / threshold, capped at 1.0."""
    return min(_min_pairwise_dE(palette_lab) / threshold, 1.0)


def _basemap_contrast(palette_lab, basemap_lab, threshold=40.0):
    """Basemap contrast emphasizing the WORST palette color.

    For each palette color, compute its min CIEDE2000 to any basemap color.
    Use a blend of min (worst color) and mean to ensure no color blends
    into the basemap while still rewarding overall contrast.
    """
    per_color = []
    for pc in palette_lab:
        min_dE = min(ciede2000(pc, bc) for bc in basemap_lab)
        per_color.append(min_dE)

    worst = min(per_color)
    mean_val = np.mean(per_color)

    # Worst color contributes 60% — heavily penalizes any blending
    blended = 0.6 * worst + 0.4 * mean_val
    return min(blended / threshold, 1.0)


def _cvd_robustness(palette_lab, threshold=20.0):
    """Mean CVD robustness across protanopia and deuteranopia."""
    rgb = _lab_to_rgb_simple(palette_lab)
    scores = []
    for cvd_type in ["protan", "deutan"]:
        sim_rgb = _simulate_cvd(rgb, cvd_type)
        sim_lab = _rgb_to_lab_batch(sim_rgb)
        min_dE = _min_pairwise_dE(sim_lab)
        scores.append(min(min_dE / threshold, 1.0))
    return np.mean(scores)


def _perceptual_ordering(palette_lab, scheme_type):
    """L* monotonicity + step uniformity (must match score_palettes.py).

    Sequential: 70% Spearman monotonicity + 30% step uniformity
    Diverging: 40% arm monotonicity + 30% opposite direction + 30% symmetry
    """
    from scipy import stats as _stats

    L = palette_lab[:, 0]
    n = len(L)
    if n <= 1:
        return 1.0

    if scheme_type == "sequential":
        # Spearman monotonicity
        rho, _ = _stats.spearmanr(np.arange(n), L)
        monotonicity = abs(rho) if not np.isnan(rho) else 0.0

        # Step uniformity (adjacent CIEDE2000 distances)
        adj_dists = np.array([ciede2000(palette_lab[i], palette_lab[i + 1])
                              for i in range(n - 1)])
        if adj_dists.mean() > 0:
            cv = adj_dists.std() / adj_dists.mean()
            uniformity = float(np.exp(-cv))
        else:
            uniformity = 0.0

        return 0.7 * monotonicity + 0.3 * uniformity

    elif scheme_type == "diverging":
        if n < 3:
            return 1.0

        mid = n // 2
        left_L = L[: mid + 1]
        right_L = L[mid:]

        # Arm monotonicity
        mono_scores = []
        for arm_L in [left_L, right_L]:
            if len(arm_L) >= 2:
                rho, _ = _stats.spearmanr(np.arange(len(arm_L)), arm_L)
                mono_scores.append(abs(rho) if not np.isnan(rho) else 0.0)
            else:
                mono_scores.append(1.0)
        monotonicity = np.mean(mono_scores)

        # Opposite direction check
        left_trend = left_L[-1] - left_L[0]
        right_trend = right_L[-1] - right_L[0]
        if abs(left_trend) > 1 and abs(right_trend) > 1:
            opposite = 1.0 if (left_trend * right_trend) < 0 else 0.0
        else:
            opposite = 0.5

        # L* symmetry between arms
        mid_L = L[mid]
        left_dev = np.abs(left_L[::-1] - mid_L)
        right_dev = np.abs(right_L - mid_L)
        min_len = min(len(left_dev), len(right_dev))
        if left_dev[:min_len].sum() + right_dev[:min_len].sum() > 0:
            asymmetry = np.mean(np.abs(left_dev[:min_len] - right_dev[:min_len]))
            symmetry = float(np.exp(-asymmetry / 20.0))
        else:
            symmetry = 1.0

        return 0.4 * monotonicity + 0.3 * opposite + 0.3 * symmetry

    return 1.0


# ══════════════════════════════════════════════════════════
# Composite Score
# ══════════════════════════════════════════════════════════

def _lightness_contrast(palette_lab, basemap_dominant_lab):
    """Proportion of palette colors safely outside the basemap lightness zone.

    Sequential palettes inevitably have one endpoint near any basemap's L*,
    so this metric counts how many colors are SAFELY separated rather than
    harshly penalizing any overlap.

    Returns: score in [0, 1]. Higher = more colors safely separated.
    """
    basemap_L = basemap_dominant_lab[:, 0]
    palette_L = palette_lab[:, 0]
    n = len(palette_L)

    # For each palette color: min L* distance to any basemap dominant color
    safe_count = 0
    for pL in palette_L:
        min_L_dist = min(abs(pL - bL) for bL in basemap_L)
        if min_L_dist > 15.0:  # 15 L* units = clearly different lightness
            safe_count += 1

    # Also add a continuous bonus for HOW far the safe colors are
    total_separation = 0.0
    for pL in palette_L:
        min_L_dist = min(abs(pL - bL) for bL in basemap_L)
        total_separation += min(min_L_dist / 30.0, 1.0)

    # Blend: 60% proportion-based (discrete), 40% distance-based (continuous)
    proportion = safe_count / n
    mean_sep = total_separation / n
    return 0.6 * proportion + 0.4 * mean_sep


def _hue_contrast(palette_lab, basemap_dominant_lab):
    """Penalize palettes whose hues overlap with basemap hues.

    Uses chroma-weighted hue comparison: low-chroma colors (dark/neutral
    basemaps) have unreliable hue, so the metric gracefully degrades to
    a neutral score rather than producing meaningless comparisons.

    Returns: score in [0, 1]. Higher = better hue separation.
    """
    def chroma_weighted_hue(lab_colors):
        """Return mean hue and mean chroma."""
        a, b = lab_colors[:, 1], lab_colors[:, 2]
        C = np.sqrt(a**2 + b**2)
        angles = np.arctan2(b, a)
        # Chroma-weighted circular mean
        weights = C / (C.sum() + 1e-10)
        sin_mean = np.sum(weights * np.sin(angles))
        cos_mean = np.sum(weights * np.cos(angles))
        return np.arctan2(sin_mean, cos_mean), np.mean(C)

    basemap_hue, basemap_C = chroma_weighted_hue(basemap_dominant_lab)
    palette_hue, palette_C = chroma_weighted_hue(palette_lab)

    # Angular difference (0 to pi)
    diff = abs(palette_hue - basemap_hue)
    if diff > np.pi:
        diff = 2 * np.pi - diff

    # Raw hue score: 0 = same hue, 1 = well-separated
    hue_score = min(diff / (np.pi * 0.5), 1.0)

    # Confidence: how much can we trust the hue comparison?
    # Low basemap chroma → hue is meaningless → return neutral 0.5
    # High basemap chroma → hue matters a lot
    confidence = min(basemap_C / 15.0, 1.0)  # C < 15 = near-neutral

    # Blend: confident → use hue_score, not confident → return 0.5 (neutral)
    return confidence * hue_score + (1 - confidence) * 0.5


def score_palette(palette_lab, basemap_dominant_lab, scheme_type):
    """Compute composite quality score for a palette.

    Weights basemap-dependent metrics more heavily so that different basemaps
    produce visibly different palette rankings.

    Args:
        palette_lab: (n_colors, 3) CIELAB palette.
        basemap_dominant_lab: (k, 3) dominant basemap colors in CIELAB.
        scheme_type: "sequential", "diverging", or "qualitative".

    Returns:
        Composite score in [0, 1]. Higher is better.
    """
    dist = _distinguishability(palette_lab)
    contrast = _basemap_contrast(palette_lab, basemap_dominant_lab)
    cvd = _cvd_robustness(palette_lab)
    ordering = _perceptual_ordering(palette_lab, scheme_type)
    l_contrast = _lightness_contrast(palette_lab, basemap_dominant_lab)
    h_contrast = _hue_contrast(palette_lab, basemap_dominant_lab)

    # Weights: 80% basemap-dependent, 20% universal quality
    # v4: Only sequential + diverging (no qualitative)
    w = {"dist": 0.10, "contrast": 0.20, "cvd": 0.05, "order": 0.05,
         "l_contrast": 0.30, "h_contrast": 0.30}

    return (w["dist"] * dist + w["contrast"] * contrast + w["cvd"] * cvd +
            w["order"] * ordering + w["l_contrast"] * l_contrast + w["h_contrast"] * h_contrast)


def score_palette_detailed(palette_lab, basemap_dominant_lab, scheme_type):
    """Compute composite quality score WITH individual metric breakdown.

    Returns:
        dict with keys: 'composite', 'lightness_contrast', 'hue_contrast',
        'basemap_contrast', 'distinguishability', 'cvd_robustness',
        'perceptual_ordering', and 'weights'.
    """
    dist = _distinguishability(palette_lab)
    contrast = _basemap_contrast(palette_lab, basemap_dominant_lab)
    cvd = _cvd_robustness(palette_lab)
    ordering = _perceptual_ordering(palette_lab, scheme_type)
    l_contrast = _lightness_contrast(palette_lab, basemap_dominant_lab)
    h_contrast = _hue_contrast(palette_lab, basemap_dominant_lab)

    w = {"dist": 0.10, "contrast": 0.20, "cvd": 0.05, "order": 0.05,
         "l_contrast": 0.30, "h_contrast": 0.30}

    composite = (w["dist"] * dist + w["contrast"] * contrast + w["cvd"] * cvd +
                 w["order"] * ordering + w["l_contrast"] * l_contrast + w["h_contrast"] * h_contrast)

    return {
        "composite": composite,
        "lightness_contrast": l_contrast,
        "hue_contrast": h_contrast,
        "basemap_contrast": contrast,
        "distinguishability": dist,
        "cvd_robustness": cvd,
        "perceptual_ordering": ordering,
        "weights": w,
    }
