"""
Color science utilities for ChromaMap.

Handles CIELAB conversions, CIEDE2000 color difference, CVD simulation,
and K-Means dominant color extraction.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ──────────────────────────────────────
#  Color Space Conversions
# ──────────────────────────────────────

# sRGB → XYZ D65 reference white
_D65 = np.array([0.95047, 1.00000, 1.08883])


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,255] to CIELAB.

    Args:
        rgb: Array of shape (..., 3) with values in [0, 255]

    Returns:
        Array of shape (..., 3) with (L*, a*, b*) values
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    original_shape = rgb.shape

    # Normalize to [0, 1]
    rgb_norm = rgb / 255.0

    # sRGB to linear
    linear = _srgb_to_linear(rgb_norm)

    # Linear RGB to XYZ (sRGB D65 matrix)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    # Reshape for matrix multiplication
    flat = linear.reshape(-1, 3)
    xyz = flat @ M.T

    # XYZ to Lab
    xyz_norm = xyz / _D65

    epsilon = 0.008856
    kappa = 903.3

    f = np.where(
        xyz_norm > epsilon,
        np.cbrt(xyz_norm),
        (kappa * xyz_norm + 16) / 116,
    )

    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])

    lab = np.stack([L, a, b], axis=-1)
    return lab.reshape(original_shape)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to RGB [0,255].

    Args:
        lab: Array of shape (..., 3) with (L*, a*, b*) values

    Returns:
        Array of shape (..., 3) with values in [0, 255], clipped
    """
    lab = np.asarray(lab, dtype=np.float64)
    original_shape = lab.shape
    flat = lab.reshape(-1, 3)

    L, a, b = flat[:, 0], flat[:, 1], flat[:, 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    x = np.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa)

    xyz = np.stack([x, y, z], axis=-1) * _D65

    # XYZ to linear RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])

    linear = xyz @ M_inv.T
    srgb = _linear_to_srgb(np.clip(linear, 0, 1))
    rgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)

    return rgb.reshape(original_shape)


def lab_to_hex(lab: np.ndarray) -> list[str]:
    """Convert CIELAB values to hex color strings."""
    rgb = lab_to_rgb(lab)
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, 3)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb]


# ──────────────────────────────────────
#  CIEDE2000 Color Difference
# ──────────────────────────────────────

def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float | np.ndarray:
    """Calculate CIEDE2000 color difference between two CIELAB colors.

    Args:
        lab1: First color(s), shape (3,) or (N, 3)
        lab2: Second color(s), shape (3,) or (N, 3)

    Returns:
        CIEDE2000 distance (scalar or array)
    """
    lab1 = np.atleast_2d(np.asarray(lab1, dtype=np.float64))
    lab2 = np.atleast_2d(np.asarray(lab2, dtype=np.float64))

    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    # Step 1: Calculate C'ab and h'ab
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2

    G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))

    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    # Step 2: Calculate delta values
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = np.zeros_like(h1_prime)
    h_diff = h2_prime - h1_prime

    mask1 = (C1_prime * C2_prime) == 0
    mask2 = np.abs(h_diff) <= 180
    mask3 = h_diff > 180
    mask4 = h_diff < -180

    dh_prime[mask1] = 0
    dh_prime[~mask1 & mask2] = h_diff[~mask1 & mask2]
    dh_prime[~mask1 & mask3] = h_diff[~mask1 & mask3] - 360
    dh_prime[~mask1 & mask4] = h_diff[~mask1 & mask4] + 360

    dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))

    # Step 3: Calculate CIEDE2000
    L_avg = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2

    h_avg = np.zeros_like(h1_prime)
    h_sum = h1_prime + h2_prime

    mask_c0 = (C1_prime * C2_prime) == 0
    mask_h180 = np.abs(h1_prime - h2_prime) <= 180

    h_avg[mask_c0] = h_sum[mask_c0]
    h_avg[~mask_c0 & mask_h180] = h_sum[~mask_c0 & mask_h180] / 2
    h_avg[~mask_c0 & ~mask_h180 & (h_sum < 360)] = (h_sum[~mask_c0 & ~mask_h180 & (h_sum < 360)] + 360) / 2
    h_avg[~mask_c0 & ~mask_h180 & (h_sum >= 360)] = (h_sum[~mask_c0 & ~mask_h180 & (h_sum >= 360)] - 360) / 2

    T = (1
         - 0.17 * np.cos(np.radians(h_avg - 30))
         + 0.24 * np.cos(np.radians(2 * h_avg))
         + 0.32 * np.cos(np.radians(3 * h_avg + 6))
         - 0.20 * np.cos(np.radians(4 * h_avg - 63)))

    SL = 1 + 0.015 * (L_avg - 50)**2 / np.sqrt(20 + (L_avg - 50)**2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T

    RT = (-np.sin(2 * np.radians(30 * np.exp(-((h_avg - 275) / 25)**2)))
          * 2 * np.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7)))

    dE = np.sqrt(
        (dL_prime / SL)**2
        + (dC_prime / SC)**2
        + (dH_prime / SH)**2
        + RT * (dC_prime / SC) * (dH_prime / SH)
    )

    return float(dE[0]) if dE.size == 1 else dE


def pairwise_ciede2000(palette_lab: np.ndarray) -> np.ndarray:
    """Calculate pairwise CIEDE2000 distances for a palette (vectorized).

    Args:
        palette_lab: Shape (N, 3) — N colors in CIELAB

    Returns:
        Shape (N, N) distance matrix
    """
    n = len(palette_lab)
    if n < 2:
        return np.zeros((n, n))

    # Build upper-triangle index pairs
    idx_i, idx_j = np.triu_indices(n, k=1)
    lab1 = palette_lab[idx_i]  # (n_pairs, 3)
    lab2 = palette_lab[idx_j]  # (n_pairs, 3)

    # Vectorized CIEDE2000 for all pairs at once
    dists = ciede2000(lab1, lab2)

    dist_matrix = np.zeros((n, n))
    dist_matrix[idx_i, idx_j] = dists
    dist_matrix[idx_j, idx_i] = dists

    return dist_matrix


# ──────────────────────────────────────
#  CVD Simulation (Brettel 1997)
# ──────────────────────────────────────

# Simplified Viénot (1999) simulation matrices for dichromatic vision
# These operate on linear RGB
_CVD_MATRICES = {
    "deuteranopia": np.array([
        [0.625, 0.375, 0.0],
        [0.700, 0.300, 0.0],
        [0.000, 0.300, 0.700],
    ]),
    "protanopia": np.array([
        [0.152, 0.114, -0.004],
        [0.115, 0.886, -0.001],
        [0.004, -0.002, 0.693],
    ]) + np.array([
        [0.567, 0.433, 0.0],
        [0.558, 0.442, 0.0],
        [0.0,   0.242, 0.758],
    ]) * 0,  # Simplified: use Viénot 1999
}

# More accurate Viénot 1999 matrices
_CVD_MATRICES = {
    "deuteranopia": np.array([
        [0.625, 0.375, 0.0],
        [0.700, 0.300, 0.0],
        [0.000, 0.300, 0.700],
    ]),
    "protanopia": np.array([
        [0.567, 0.433, 0.0],
        [0.558, 0.442, 0.0],
        [0.000, 0.242, 0.758],
    ]),
}


def simulate_cvd(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    """Simulate color vision deficiency on RGB colors.

    Args:
        rgb: Array of shape (..., 3) with values in [0, 255]
        cvd_type: Either "deuteranopia" or "protanopia"

    Returns:
        Simulated RGB values [0, 255]
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    shape = rgb.shape

    # Normalize and linearize
    linear = _srgb_to_linear(rgb.reshape(-1, 3) / 255.0)

    # Apply CVD matrix
    matrix = _CVD_MATRICES[cvd_type]
    simulated = linear @ matrix.T

    # Back to sRGB
    srgb = _linear_to_srgb(np.clip(simulated, 0, 1))
    return (np.clip(srgb * 255, 0, 255)).astype(np.uint8).reshape(shape)


# ──────────────────────────────────────
#  Dominant Color Extraction
# ──────────────────────────────────────

def extract_dominant_colors(
    image: Image.Image | np.ndarray,
    k: int = 10,
    sample_pixels: int = 5000,
    random_state: int = 42,
) -> dict:
    """Extract dominant colors from an image using K-Means in CIELAB space.

    Args:
        image: PIL Image or numpy array (H, W, 3) in RGB
        k: Number of clusters
        sample_pixels: Number of pixels to sample (for speed)
        random_state: Random seed for reproducibility

    Returns:
        Dict with:
        - "colors_lab": (k, 3) array of dominant colors in CIELAB
        - "colors_rgb": (k, 3) array of dominant colors in RGB
        - "proportions": (k,) array of cluster proportions
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    # Flatten pixels
    pixels = image.reshape(-1, 3).astype(np.float64)

    # Sample for speed
    rng = np.random.RandomState(random_state)
    if len(pixels) > sample_pixels:
        indices = rng.choice(len(pixels), sample_pixels, replace=False)
        pixels = pixels[indices]

    # Convert to CIELAB
    pixels_lab = rgb_to_lab(pixels)

    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(pixels_lab)

    # Cluster centers and proportions
    centers_lab = kmeans.cluster_centers_
    labels = kmeans.labels_
    proportions = np.bincount(labels, minlength=k).astype(float)
    proportions /= proportions.sum()

    # Sort by proportion (dominant first)
    order = np.argsort(-proportions)
    centers_lab = centers_lab[order]
    proportions = proportions[order]

    # Convert centers to RGB
    centers_rgb = lab_to_rgb(centers_lab)

    return {
        "colors_lab": centers_lab,
        "colors_rgb": centers_rgb,
        "proportions": proportions,
    }
