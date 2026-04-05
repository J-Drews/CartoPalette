"""
ChromaMap v4 — Basemap-Adaptive Palette Candidate Generation (4-Stage Process)

Stage 1: ColorBrewer base palettes (~35 per scheme_type)
Stage 2: Basemap-Adaptive CIELAB variations (~150 per config)
Stage 3: Full contrast optimization (L*, Hue, Chroma) (~50 per config)
Stage 4: Synthetic palette generation from scratch (~100 per config)

Key difference from v3: ALL stages are now based on basemap dominant colors, not independently.

Usage:
    python -m research.data_pipeline.generate_candidates
    python -m research.data_pipeline.generate_candidates --test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.utils.color_utils import rgb_to_lab, lab_to_rgb, ciede2000
from research.utils.io_utils import load_yaml, setup_logging, get_project_root

logger = setup_logging("generate_candidates")


# ──────────────────────────────────────
#  ColorBrewer Palettes (Stage 1)
# ──────────────────────────────────────

# Subset of ColorBrewer palettes organized by scheme type.
# Full palettes from colorbrewer2.org (Brewer 2003)
COLORBREWER = {
    "sequential": {
        "YlGn":    [[255,255,229],[247,252,185],[217,240,163],[173,221,142],[120,198,121],[65,171,93],[35,132,67],[0,104,55],[0,69,41]],
        "YlGnBu":  [[255,255,217],[237,248,177],[199,233,180],[127,205,187],[65,182,196],[29,145,192],[34,94,168],[37,52,148],[8,29,88]],
        "GnBu":    [[247,252,240],[224,243,219],[204,235,197],[168,221,181],[123,204,196],[78,179,211],[43,140,190],[8,104,172],[8,64,129]],
        "BuGn":    [[247,252,253],[229,245,249],[204,236,230],[153,216,201],[102,194,164],[65,174,118],[35,139,69],[0,109,44],[0,68,27]],
        "PuBuGn":  [[255,247,251],[236,226,240],[208,209,230],[166,189,219],[103,169,207],[54,144,192],[2,129,138],[1,108,89],[1,70,54]],
        "PuBu":    [[255,247,251],[236,231,242],[208,209,230],[166,189,219],[116,169,207],[54,144,192],[5,112,176],[4,90,141],[2,56,88]],
        "BuPu":    [[247,252,253],[224,236,244],[191,211,230],[158,188,218],[140,150,198],[140,107,177],[136,65,157],[129,15,124],[77,0,75]],
        "RdPu":    [[255,247,243],[253,224,221],[252,197,192],[250,159,181],[247,104,161],[221,52,151],[174,1,126],[122,1,119],[73,0,106]],
        "PuRd":    [[247,244,249],[231,225,239],[212,185,218],[201,148,199],[223,101,176],[231,41,138],[206,18,86],[152,0,67],[103,0,31]],
        "OrRd":    [[255,247,236],[254,232,200],[253,212,158],[253,187,132],[252,141,89],[239,101,72],[215,48,31],[179,0,0],[127,0,0]],
        "YlOrRd":  [[255,255,204],[255,237,160],[254,217,118],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[189,0,38],[128,0,38]],
        "YlOrBr":  [[255,255,229],[255,247,188],[254,227,145],[254,196,79],[254,153,41],[236,112,20],[204,76,2],[153,52,4],[102,37,6]],
        "Purples": [[252,251,253],[239,237,245],[218,218,235],[188,189,220],[158,154,200],[128,125,186],[106,81,163],[84,39,143],[63,0,125]],
        "Blues":    [[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107]],
        "Greens":  [[247,252,245],[229,245,224],[199,233,192],[161,217,155],[116,196,118],[65,171,93],[35,139,69],[0,109,44],[0,68,27]],
        "Oranges": [[255,245,235],[254,230,206],[253,208,162],[253,174,107],[253,141,60],[241,105,19],[217,72,1],[166,54,3],[127,39,4]],
        "Reds":    [[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],[239,59,44],[203,24,29],[165,15,21],[103,0,13]],
        "Greys":   [[255,255,255],[240,240,240],[217,217,217],[189,189,189],[150,150,150],[115,115,115],[82,82,82],[37,37,37],[0,0,0]],
    },
    "diverging": {
        "RdYlGn":  [[165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,139],[255,255,191],[217,239,139],[166,217,106],[102,189,99],[26,152,80],[0,104,55]],
        "RdBu":    [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[247,247,247],[209,229,240],[146,197,222],[67,147,195],[33,102,172],[5,48,97]],
        "RdYlBu":  [[165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,144],[255,255,191],[224,243,248],[171,217,233],[116,173,209],[69,117,180],[49,54,149]],
        "PiYG":    [[142,1,82],[197,27,125],[222,119,174],[241,182,218],[253,224,239],[247,247,247],[230,245,208],[184,225,134],[127,188,65],[77,146,33],[39,100,25]],
        "PRGn":    [[64,0,75],[118,42,131],[153,112,171],[194,165,207],[231,212,232],[247,247,247],[217,240,211],[166,219,160],[90,174,97],[27,120,55],[0,68,27]],
        "BrBG":    [[84,48,5],[140,81,10],[191,129,45],[223,194,125],[246,232,195],[245,245,245],[199,234,229],[128,205,193],[53,151,143],[1,102,94],[0,60,48]],
        "RdGy":    [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[255,255,255],[224,224,224],[186,186,186],[135,135,135],[77,77,77],[26,26,26]],
    },
}


def get_colorbrewer_palettes(scheme_type: str, n_classes: int) -> list[np.ndarray]:
    """Get all ColorBrewer palettes for a given scheme type and class count.

    Returns:
        List of palettes, each shape (n_classes, 3) in RGB [0,255]
    """
    palettes = []
    schemes = COLORBREWER.get(scheme_type, {})

    for name, colors in schemes.items():
        if scheme_type == "diverging":
            # Diverging palettes: sample symmetrically from full range
            full = np.array(colors)
            if len(full) >= n_classes:
                indices = np.linspace(0, len(full) - 1, n_classes).astype(int)
                palettes.append(full[indices])
        else:
            # Sequential: sample evenly from full range
            full = np.array(colors)
            if len(full) >= n_classes:
                indices = np.linspace(0, len(full) - 1, n_classes).astype(int)
                palettes.append(full[indices])

    return palettes


# ──────────────────────────────────────
#  Basemap Analysis
# ──────────────────────────────────────

def analyze_basemap(basemap_colors_lab: np.ndarray) -> dict:
    """Analyze basemap colors to compute dominant characteristics.

    Args:
        basemap_colors_lab: (n_colors, 3) array in CIELAB space

    Returns:
        dict with keys:
            - median_L: median lightness [0, 100]
            - mean_chroma: mean C* = sqrt(a*^2 + b*^2)
            - mean_hue: mean hue in degrees [0, 360)
            - is_dark: True if median_L < 25
            - is_light: True if median_L > 75
    """
    L = basemap_colors_lab[:, 0]
    a = basemap_colors_lab[:, 1]
    b = basemap_colors_lab[:, 2]

    median_L = np.median(L)
    mean_chroma = np.mean(np.sqrt(a**2 + b**2))

    # Compute hue as chroma-weighted circular mean (consistent with score_palettes.py)
    C = np.sqrt(a**2 + b**2)
    angles = np.arctan2(b, a)
    weights = C / (C.sum() + 1e-10)
    sin_mean = np.sum(weights * np.sin(angles))
    cos_mean = np.sum(weights * np.cos(angles))
    mean_hue_rad = np.arctan2(sin_mean, cos_mean)
    mean_hue = float(np.degrees(mean_hue_rad))
    if mean_hue < 0:
        mean_hue += 360

    return {
        "median_L": float(median_L),
        "mean_chroma": float(mean_chroma),
        "mean_hue": float(mean_hue),
        "is_dark": median_L < 25,
        "is_light": median_L > 75,
    }


# ──────────────────────────────────────
#  Stage 2: Basemap-Adaptive CIELAB Variations
# ──────────────────────────────────────

def generate_basemap_adaptive_variations(
    base_palettes: list[np.ndarray],
    basemap_analysis: dict,
    rng: np.random.RandomState,
    max_variations: int = 150,
) -> list[np.ndarray]:
    """Generate CIELAB variations adapted to basemap characteristics.

    Strategy:
    - L* shifts: push palette lightness AWAY from basemap
    - Hue rotations: maximize distance from dominant basemap hue
    - Chroma scaling: increase on dark basemaps, moderate on light

    Returns:
        List of palettes in LAB space, shape (n_classes, 3)
    """
    variations = []
    median_L = basemap_analysis["median_L"]
    mean_hue = basemap_analysis["mean_hue"]
    is_dark = basemap_analysis["is_dark"]
    is_light = basemap_analysis["is_light"]

    # Determine L* shifts based on basemap darkness
    # v4.1: reduced shift magnitudes to prevent systematic dark bias
    if is_dark:
        l_shifts = [0, 5, 10, 15, 20]  # Lighten; 0 keeps seed palette intact
    elif is_light:
        l_shifts = [-5, -10, -15, 0, 5]  # Moderate darkening + keep original
    else:
        # Medium basemap: symmetric moderate shifts
        l_shifts = [-15, -10, -5, 0, 5, 10, 15]

    # Chroma scaling
    if is_dark:
        chroma_scales = [1.0, 1.15, 1.3]  # Increase
    elif is_light:
        chroma_scales = [0.85, 1.0, 1.15]  # Moderate
    else:
        chroma_scales = [0.9, 1.0, 1.1]

    # Hue rotations: choose angles that maximize distance from mean_hue
    # Generate rotations to cover the hue circle, avoiding basemap dominant hue
    candidate_rotations = [h for h in range(0, 360, 15)]
    rotations = []
    for rot in candidate_rotations:
        dist_to_basemap = min(
            abs(rot - mean_hue),
            360 - abs(rot - mean_hue)
        )
        if dist_to_basemap > 30:  # At least 30° away from basemap hue
            rotations.append(rot)

    # If not enough rotations, use what we have
    if not rotations:
        rotations = [0, 90, 180, 270]

    # Generate variations
    for base_rgb in base_palettes:
        base_lab = rgb_to_lab(np.array(base_rgb, dtype=np.float64))

        # L* shifts
        for l_shift in l_shifts:
            shifted = base_lab.copy()
            shifted[:, 0] = np.clip(shifted[:, 0] + l_shift, 0, 100)
            variations.append(shifted)

        # Chroma scaling
        for scale in chroma_scales:
            if scale == 1.0:
                continue
            scaled = base_lab.copy()
            scaled[:, 1] *= scale  # a*
            scaled[:, 2] *= scale  # b*
            variations.append(scaled)

        # Hue rotations
        for rotation in rotations:
            rotated = base_lab.copy()
            angle_rad = np.radians(rotation)
            a_new = rotated[:, 1] * np.cos(angle_rad) - rotated[:, 2] * np.sin(angle_rad)
            b_new = rotated[:, 1] * np.sin(angle_rad) + rotated[:, 2] * np.cos(angle_rad)
            rotated[:, 1] = a_new
            rotated[:, 2] = b_new
            variations.append(rotated)

    # Random sample to target max_variations
    if len(variations) > max_variations:
        indices = rng.choice(len(variations), max_variations, replace=False)
        variations = [variations[i] for i in indices]

    return variations


# ──────────────────────────────────────
#  Stage 3: Full Contrast Optimization
# ──────────────────────────────────────

def _fast_min_ciede2000_to_basemap(palette_lab: np.ndarray, basemap_lab: np.ndarray):
    """Vectorized: for each palette color, find min CIEDE2000 distance to any basemap color.

    Returns:
        min_dists: (n_palette,) array of minimum distances
        closest_L: (n_palette,) array of L* values of closest basemap colors
        closest_hue: (n_palette,) array of hue values of closest basemap colors
    """
    n_pal = len(palette_lab)
    n_bm = len(basemap_lab)

    pal_expanded = np.repeat(palette_lab, n_bm, axis=0)
    bm_expanded = np.tile(basemap_lab, (n_pal, 1))

    dists = ciede2000(pal_expanded, bm_expanded)
    dists = dists.reshape(n_pal, n_bm)

    min_idx = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(n_pal), min_idx]
    closest_L = basemap_lab[min_idx, 0]

    # Compute hue of closest basemap colors
    a_closest = basemap_lab[min_idx, 1]
    b_closest = basemap_lab[min_idx, 2]
    hue_closest = np.arctan2(b_closest, a_closest)
    hue_closest = np.degrees(hue_closest)
    hue_closest = np.where(hue_closest < 0, hue_closest + 360, hue_closest)

    return min_dists, closest_L, hue_closest


def _compute_palette_hue(palette_lab: np.ndarray) -> np.ndarray:
    """Compute hue for each color in palette (degrees)."""
    a = palette_lab[:, 1]
    b = palette_lab[:, 2]
    hue = np.arctan2(b, a)
    hue = np.degrees(hue)
    hue = np.where(hue < 0, hue + 360, hue)
    return hue


def optimize_contrast_full(
    base_palettes_lab: list[np.ndarray],
    basemap_colors_lab: np.ndarray,
    basemap_analysis: dict,
    n_steps: int = 80,
    lr: float = 1.0,
    max_candidates: int = 50,
    rng: np.random.RandomState = None,
) -> list[np.ndarray]:
    """Optimize palette L*, Hue, and Chroma to maximize contrast against basemap.

    Optimization targets:
    - L*: push palette away from basemap L* zone
    - Hue: rotate palette hue away from dominant basemap hue
    - Chroma: adjust to maximize CIEDE2000 against basemap

    Returns:
        List of contrast-optimized palettes in LAB space
    """
    if rng is None:
        rng = np.random.RandomState(42)

    optimized = []
    basemap_median_L = basemap_analysis["median_L"]
    basemap_mean_hue = basemap_analysis["mean_hue"]

    for base_lab in base_palettes_lab:
        palette = base_lab.copy()

        for step in range(n_steps):
            min_dists, closest_L, closest_hue = _fast_min_ciede2000_to_basemap(
                palette, basemap_colors_lab
            )

            # Early exit: all colors have enough contrast
            too_close = min_dists < 15
            if not too_close.any():
                break

            # Optimize L*: push away from basemap L*
            delta_L = palette[:, 0] - closest_L
            small_delta_L = np.abs(delta_L) < 1
            delta_L[small_delta_L] = rng.choice([-1, 1], size=small_delta_L.sum()) * 5
            palette[too_close, 0] += np.sign(delta_L[too_close]) * lr
            # v4.1: tighter L* bounds to prevent excessively dark palettes
            # Mild version: allows L*=20 minimum for diverging dark arms
            if basemap_analysis["is_light"]:
                palette[:, 0] = np.clip(palette[:, 0], 20, 90)
            elif basemap_analysis["is_dark"]:
                palette[:, 0] = np.clip(palette[:, 0], 15, 95)
            else:
                palette[:, 0] = np.clip(palette[:, 0], 18, 92)

            # Optimize Hue: rotate palette hue away from basemap hue
            pal_hue = _compute_palette_hue(palette)
            delta_hue = pal_hue - closest_hue
            # Normalize delta_hue to [-180, 180]
            delta_hue = np.where(delta_hue > 180, delta_hue - 360, delta_hue)
            delta_hue = np.where(delta_hue < -180, delta_hue + 360, delta_hue)

            # If too close to basemap hue, rotate
            small_delta_hue = np.abs(delta_hue) < 10
            angle_rad = np.radians(10) if small_delta_hue.any() else 0
            if angle_rad > 0:
                mask = too_close & small_delta_hue
                if mask.any():
                    a_rot = palette[mask, 1] * np.cos(angle_rad) - palette[mask, 2] * np.sin(angle_rad)
                    b_rot = palette[mask, 1] * np.sin(angle_rad) + palette[mask, 2] * np.cos(angle_rad)
                    palette[mask, 1] = a_rot
                    palette[mask, 2] = b_rot

            # Optimize Chroma: scale to maximize contrast
            chroma = np.sqrt(palette[:, 1]**2 + palette[:, 2]**2)
            # If too close, increase chroma
            palette[too_close, 1] *= 1.05
            palette[too_close, 2] *= 1.05

        optimized.append(palette)

    # Limit and shuffle
    if len(optimized) > max_candidates:
        indices = rng.choice(len(optimized), max_candidates, replace=False)
        optimized = [optimized[i] for i in indices]

    return optimized


# ──────────────────────────────────────
#  Stage 4: Synthetic Palette Generation
# ──────────────────────────────────────

def clip_to_srgb_gamut(palette_lab: np.ndarray) -> np.ndarray:
    """Clip palette to sRGB gamut by desaturating if necessary."""
    palette_rgb = lab_to_rgb(palette_lab)

    # Check if any values are outside [0, 255]
    out_of_gamut = (palette_rgb < 0) | (palette_rgb > 255)

    if not out_of_gamut.any():
        return palette_lab

    # Reduce chroma iteratively until all colors fit
    palette = palette_lab.copy()
    for iteration in range(20):
        palette_rgb = lab_to_rgb(palette)
        out_of_gamut = (palette_rgb < 0) | (palette_rgb > 255)

        if not out_of_gamut.any():
            break

        # Reduce chroma by 5%
        palette[:, 1] *= 0.95
        palette[:, 2] *= 0.95

    return palette


def generate_synthetic_palettes(
    basemap_analysis: dict,
    scheme_type: str,
    n_classes: int,
    n_palettes: int = 100,
    rng: np.random.RandomState = None,
) -> list[np.ndarray]:
    """Generate synthetic palettes from scratch, adapted to basemap.

    Algorithm:
    1. Choose start hue 90°-180° away from basemap dominant hue
    2. Set L* range to avoid basemap L* zone
    3. Generate n_classes colors with uniform L* spacing
    4. Set moderate chroma and clip to sRGB gamut
    5. Try multiple hue paths and L* ranges for diversity

    Returns:
        List of palettes in LAB space
    """
    if rng is None:
        rng = np.random.RandomState(42)

    basemap_median_L = basemap_analysis["median_L"]
    basemap_mean_hue = basemap_analysis["mean_hue"]
    palettes = []

    # Determine L* range zones
    # v4.1: separate ranges for sequential vs diverging (per methodology PDF)
    # Diverging palettes need: bright center (70-95) + darker arms (down to 20-35)
    # Sequential palettes need: monotonic L* ramp within a plausible zone
    if scheme_type == "diverging":
        if basemap_analysis["is_dark"]:
            # Dark basemap: bright center, arms not too dark
            div_center_range = [70, 95]
            div_arms_min_range = [35, 45]
        elif basemap_analysis["is_light"]:
            # Light basemap: center still bright, arms can go moderately dark
            div_center_range = [70, 90]
            div_arms_min_range = [20, 30]
        else:
            # Mid basemap: flexible
            div_center_range = [55, 80]
            div_arms_min_range = [15, 25]
    else:
        # Sequential: use l_ranges as before
        pass

    if basemap_analysis["is_dark"]:
        l_ranges = [[45, 85], [55, 95]]
    elif basemap_analysis["is_light"]:
        l_ranges = [[30, 75], [35, 80], [40, 85]]
    else:
        l_ranges = [[25, 55], [30, 70], [45, 80], [50, 85]]

    # Hue range: choose directions away from basemap hue
    hue_offsets = [90, 120, 150, 180, -90, -120, -150]

    # Generate palettes
    for _ in range(n_palettes):
        hue_offset = rng.choice(hue_offsets)
        start_hue = (basemap_mean_hue + hue_offset) % 360

        if scheme_type == "diverging" and n_classes >= 3:
            # ── Diverging-specific construction ──
            # Two arms radiating from a neutral/light midpoint, with
            # symmetric L* and two distinct hue arms.
            # v4.1: uses dedicated center/arms ranges from PDF methodology
            mid = n_classes // 2

            # Midpoint: bright center from diverging-specific range
            mid_L = rng.uniform(div_center_range[0], div_center_range[1])

            # Arms minimum L* from diverging-specific range
            l_min = rng.uniform(div_arms_min_range[0], div_arms_min_range[1])

            # Arms go from midpoint L* down to l_min
            left_L = np.linspace(l_min, mid_L, mid + 1)       # ascending to mid
            right_L = np.linspace(mid_L, l_min, n_classes - mid)  # descending from mid

            l_values = np.concatenate([left_L, right_L[1:]])   # avoid duplicating midpoint

            # Pad/trim to exact n_classes
            if len(l_values) < n_classes:
                l_values = np.linspace(l_min, mid_L, n_classes)
                l_values[mid] = mid_L
            elif len(l_values) > n_classes:
                l_values = l_values[:n_classes]

            # Two hue arms: offset from each other by 60°-180°
            hue_spread = rng.uniform(60, 180)
            left_hue = start_hue
            right_hue = (start_hue + hue_spread) % 360

            hues = np.zeros(n_classes)
            hues[:mid] = left_hue
            hues[mid] = (left_hue + right_hue) / 2  # midpoint neutral-ish
            hues[mid + 1:] = right_hue

            # Chroma: lower at midpoint, higher at extremes
            chroma_base = rng.uniform(25, 50)
            chromas = np.full(n_classes, chroma_base)
            chromas[mid] = rng.uniform(5, 15)  # near-neutral midpoint

        else:
            # ── Sequential construction ──
            # Choose L* range from sequential-specific ranges
            l_idx = rng.choice(len(l_ranges))
            l_min, l_max = l_ranges[l_idx]
            ascending = rng.choice([True, False])
            if ascending:
                l_values = np.linspace(l_min, l_max, n_classes)
            else:
                l_values = np.linspace(l_max, l_min, n_classes)

            vary_hue = rng.choice([True, False])
            if vary_hue:
                hue_range = rng.uniform(0, 60)
                hues = start_hue + np.linspace(0, hue_range, n_classes)
                hues = hues % 360
            else:
                hues = np.full(n_classes, start_hue)

            chroma_base = rng.uniform(20, 50)
            chromas = np.full(n_classes, chroma_base)

        # Build palette in LAB
        palette = np.zeros((n_classes, 3))
        palette[:, 0] = l_values

        for i in range(n_classes):
            c = chromas[i] if hasattr(chromas, '__len__') else chroma_base
            a = c * np.cos(np.radians(hues[i]))
            b = c * np.sin(np.radians(hues[i]))
            palette[i, 1] = a
            palette[i, 2] = b

        # Clip to sRGB gamut
        palette = clip_to_srgb_gamut(palette)
        palettes.append(palette)

    return palettes


# ──────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────

def generate_candidates_for_basemap(
    basemap_colors_lab: np.ndarray,
    scheme_type: str,
    n_classes: int,
    config: dict,
    rng: np.random.RandomState,
) -> list[dict]:
    """Generate all palette candidates for one basemap + configuration.

    4-stage process:
    1. ColorBrewer base (~35 palettes)
    2. Basemap-adaptive CIELAB variations (~150 palettes)
    3. Full contrast optimization (~50 palettes)
    4. Synthetic palettes (~100 palettes)

    Returns:
        List of dicts with "palette_lab", "palette_rgb", "source" keys
    """
    candidates = []

    # Analyze basemap
    basemap_analysis = analyze_basemap(basemap_colors_lab)
    logger.info(f"  Basemap: L*={basemap_analysis['median_L']:.1f}, "
                f"Hue={basemap_analysis['mean_hue']:.1f}, "
                f"Dark={basemap_analysis['is_dark']}, Light={basemap_analysis['is_light']}")

    # Stage 1: ColorBrewer base
    logger.info("  Stage 1: ColorBrewer base...")
    cb_palettes = get_colorbrewer_palettes(scheme_type, n_classes)
    for pal_rgb in cb_palettes:
        pal_lab = rgb_to_lab(np.array(pal_rgb, dtype=np.float64))
        candidates.append({
            "palette_lab": pal_lab.tolist(),
            "palette_rgb": pal_rgb.tolist(),
            "source": "colorbrewer",
        })

    # Stage 2: Basemap-adaptive CIELAB variations
    logger.info("  Stage 2: Basemap-adaptive CIELAB variations...")
    variations = generate_basemap_adaptive_variations(
        cb_palettes,
        basemap_analysis,
        rng=rng,
        max_variations=config["candidates"]["n_basemap_adaptive"],
    )
    for pal_lab in variations:
        pal_rgb = lab_to_rgb(pal_lab)
        candidates.append({
            "palette_lab": pal_lab.tolist(),
            "palette_rgb": pal_rgb.astype(int).clip(0, 255).tolist(),
            "source": "cielab_adaptive",
        })

    # Stage 3: Full contrast optimization
    logger.info("  Stage 3: Full contrast optimization...")
    seed_palettes_lab = [np.array(c["palette_lab"]) for c in candidates[:20]]
    contrast_opt = optimize_contrast_full(
        seed_palettes_lab,
        basemap_colors_lab,
        basemap_analysis,
        n_steps=config["candidates"].get("contrast_optimization_steps", 80),
        max_candidates=config["candidates"].get("n_contrast_optimized", 50),
        rng=rng,
    )
    for pal_lab in contrast_opt:
        pal_rgb = lab_to_rgb(pal_lab)
        candidates.append({
            "palette_lab": pal_lab.tolist(),
            "palette_rgb": pal_rgb.astype(int).clip(0, 255).tolist(),
            "source": "contrast_optimized",
        })

    # Stage 4: Synthetic palettes
    if scheme_type in ["sequential", "diverging"]:
        logger.info("  Stage 4: Synthetic palette generation...")
        synthetic = generate_synthetic_palettes(
            basemap_analysis,
            scheme_type,
            n_classes,
            n_palettes=config["candidates"].get("n_synthetic", 100),
            rng=rng,
        )
        for pal_lab in synthetic:
            pal_rgb = lab_to_rgb(pal_lab)
            candidates.append({
                "palette_lab": pal_lab.tolist(),
                "palette_rgb": pal_rgb.astype(int).clip(0, 255).tolist(),
                "source": "synthetic",
            })

    # ── v4.1: Lightness prior — reject implausibly dark/light palettes ──
    # Safety net: remove candidates whose mean L* is extreme for the basemap type
    filtered = []
    for c in candidates:
        pal = np.array(c["palette_lab"])
        mean_L = float(np.mean(pal[:, 0]))
        if basemap_analysis["is_light"] and mean_L < 25:
            continue  # Too dark for a light basemap
        if basemap_analysis["is_dark"] and mean_L > 90:
            continue  # Too light for a dark basemap
        if not basemap_analysis["is_dark"] and not basemap_analysis["is_light"] and mean_L < 20:
            continue  # Too dark for a medium basemap
        filtered.append(c)

    n_rejected = len(candidates) - len(filtered)
    if n_rejected > 0:
        logger.info(f"  Lightness prior: rejected {n_rejected}/{len(candidates)} "
                    f"implausibly extreme candidates")

    return filtered


def run_generation(test_mode: bool = False):
    """Generate palette candidates for all basemaps and configurations."""
    root = get_project_root()
    scoring_config = load_yaml("configs/scoring/composite_score.yaml")

    patch_dir = root / "data" / "interim" / "patches"
    color_dir = root / "data" / "interim" / "basemap_colors"
    candidates_dir = root / "data" / "interim" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Load patch metadata
    with open(patch_dir / "patch_metadata.json") as f:
        patches = json.load(f)

    scheme_types = scoring_config["labeling"]["scheme_types"]
    n_classes_values = scoring_config["labeling"]["n_classes_values"]

    # Filter to sequential and diverging only (no qualitative)
    scheme_types = [s for s in scheme_types if s in ["sequential", "diverging"]]

    if test_mode:
        patches = patches[:2]
        scheme_types = scheme_types[:1]
        n_classes_values = [5]
        logger.info("TEST MODE: limited scope")

    rng = np.random.RandomState(42)
    total_candidates = 0

    for patch_info in patches:
        patch_id = patch_info["patch_id"]
        logger.info(f"Generating candidates for {patch_id}...")

        # Load basemap colors
        color_path = color_dir / f"{patch_id}_colors.json"
        if not color_path.exists():
            logger.warning(f"No color data for {patch_id}, skipping")
            continue

        with open(color_path) as f:
            color_data = json.load(f)
        basemap_colors_lab = np.array(color_data["colors_lab"])

        # Generate candidates for each configuration
        for scheme_type in scheme_types:
            for n_classes in n_classes_values:
                config_id = f"{patch_id}_{scheme_type}_{n_classes}c"
                output_path = candidates_dir / f"{config_id}.json"

                if output_path.exists():
                    continue

                logger.info(f"  {config_id}...")
                candidates = generate_candidates_for_basemap(
                    basemap_colors_lab, scheme_type, n_classes,
                    scoring_config, rng,
                )

                # Save
                output_data = {
                    "patch_id": patch_id,
                    "scheme_type": scheme_type,
                    "n_classes": n_classes,
                    "n_candidates": len(candidates),
                    "candidates": candidates,
                }

                with open(output_path, "w") as f:
                    json.dump(output_data, f)

                total_candidates += len(candidates)
                logger.info(f"    Generated {len(candidates)} candidates")

        logger.info(f"Done with {patch_id}")

    logger.info(f"Pipeline complete! Total candidates generated: {total_candidates}")


def main():
    parser = argparse.ArgumentParser(description="Generate palette candidates (v4)")
    parser.add_argument("--test", action="store_true", help="Test mode with limited scope")
    args = parser.parse_args()
    run_generation(test_mode=args.test)


if __name__ == "__main__":
    main()
