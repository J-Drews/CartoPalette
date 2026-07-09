"""Baseline palette generation: ColorBrewer, Matplotlib/Viridis, Random.

Each palette is returned as a numpy array of shape (n_classes, 3) in CIELAB.
The CartoPalette scoring.composite module uses CIELAB, so we convert everything
to CIELAB upfront to guarantee consistent metric computation.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import the existing LAB conversion from CartoPalette (single source of truth)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from cartopalette.scoring.composite import _rgb_to_lab_pixel  # noqa: E402


# ── ColorBrewer palette families (via matplotlib colormaps) ──
SEQUENTIAL_COLORBREWER = [
    "YlOrRd", "YlGnBu", "Blues", "Greens", "Oranges", "Purples",
    "Reds", "YlGn", "YlOrBr", "BuPu", "GnBu", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "RdPu", "BuGn",
]

DIVERGING_COLORBREWER = [
    "RdBu", "RdYlBu", "RdYlGn", "PiYG", "PRGn", "BrBG",
    "PuOr", "Spectral", "RdGy",
]

# ── Matplotlib / Viridis family (non-ColorBrewer perceptually uniform palettes) ──
SEQUENTIAL_MATPLOTLIB = [
    "viridis", "plasma", "magma", "inferno", "cividis",
]

DIVERGING_MATPLOTLIB = [
    "coolwarm", "RdBu_r", "Spectral_r", "seismic",
]


def _sample_colormap(cmap_name: str, n_classes: int) -> np.ndarray:
    """Sample n_classes equidistant colors from a matplotlib colormap.

    Returns: (n_classes, 3) CIELAB array.
    """
    cmap = plt.get_cmap(cmap_name)
    # Sample at the centers of n_classes bins (conventional for thematic maps)
    positions = np.linspace(0.5 / n_classes, 1.0 - 0.5 / n_classes, n_classes)
    rgba = cmap(positions)  # (n_classes, 4)
    rgb = rgba[:, :3]  # drop alpha

    lab = np.array([_rgb_to_lab_pixel(c) for c in rgb])
    return lab


def get_colorbrewer_palettes(scheme: str, n_classes: int) -> list:
    """Return list of (name, lab_array) for all ColorBrewer palettes of this scheme."""
    names = SEQUENTIAL_COLORBREWER if scheme == "sequential" else DIVERGING_COLORBREWER
    return [(name, _sample_colormap(name, n_classes)) for name in names]


def get_matplotlib_palettes(scheme: str, n_classes: int) -> list:
    """Return list of (name, lab_array) for all Matplotlib/Viridis-family palettes."""
    names = SEQUENTIAL_MATPLOTLIB if scheme == "sequential" else DIVERGING_MATPLOTLIB
    return [(name, _sample_colormap(name, n_classes)) for name in names]


def get_random_palettes(n_classes: int, n_samples: int = 10, seed: int = 42) -> list:
    """Generate n_samples random palettes within the sRGB gamut.

    Strategy:
        Uniform random in CIELAB L∈[20,90], a∈[-60,60], b∈[-60,60], then
        reject samples whose sRGB conversion falls outside [0,1]. Keeps
        the distribution of random palettes realistic (not concentrated
        in unreachable LAB regions).
    """
    rng = np.random.RandomState(seed)
    palettes = []

    from cartopalette.scoring.composite import _lab_to_rgb_simple

    for s in range(n_samples):
        # Sample random LAB colors
        lab = np.zeros((n_classes, 3))
        i = 0
        attempts = 0
        while i < n_classes and attempts < 500:
            L = rng.uniform(20, 90)
            a = rng.uniform(-60, 60)
            b = rng.uniform(-60, 60)
            candidate = np.array([[L, a, b]])
            rgb = _lab_to_rgb_simple(candidate)[0]
            # Accept only colors solidly inside sRGB gamut
            if np.all((rgb > 0.02) & (rgb < 0.98)):
                lab[i] = [L, a, b]
                i += 1
            attempts += 1

        palettes.append((f"random_{s:02d}", lab))

    return palettes


if __name__ == "__main__":
    # Quick sanity check
    print("=== ColorBrewer Sequential (5 classes) ===")
    for name, lab in get_colorbrewer_palettes("sequential", 5):
        L_range = (lab[:, 0].min(), lab[:, 0].max())
        print(f"  {name:10s}  L* range: [{L_range[0]:.1f}, {L_range[1]:.1f}]")

    print("\n=== Matplotlib Sequential (5 classes) ===")
    for name, lab in get_matplotlib_palettes("sequential", 5):
        L_range = (lab[:, 0].min(), lab[:, 0].max())
        print(f"  {name:10s}  L* range: [{L_range[0]:.1f}, {L_range[1]:.1f}]")

    print("\n=== Random (5 classes, 3 samples) ===")
    for name, lab in get_random_palettes(5, n_samples=3):
        L_range = (lab[:, 0].min(), lab[:, 0].max())
        print(f"  {name:10s}  L* range: [{L_range[0]:.1f}, {L_range[1]:.1f}]")
