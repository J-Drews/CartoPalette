"""Run the full CartoPalette benchmark.

For each of 40 test basemaps × 2 schemes × 4 class counts, this script:
    1. Generates CartoPalette's 3 top suggestions, keeps the TOP-1 by composite score.
    2. Evaluates ALL ColorBrewer palettes of matching scheme, keeps BEST-OF by composite.
    3. Evaluates ALL Matplotlib palettes of matching scheme, keeps BEST-OF by composite.
    4. Generates 10 random palettes, keeps the MEDIAN composite score.
    5. Records all 7 metrics (6 components + composite) for each selected palette.

Output:
    output/results_raw.csv — one row per (basemap, scheme, n_classes, baseline, metric)
    output/candidates_raw.json — full CartoPalette top-1 palettes (for reproducibility)

Usage:
    python run_benchmark.py [--limit N] [--device cpu|cuda]

    --limit N      Evaluate only first N basemaps (for quick tests)
    --device       Override CartoPalette inference device
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from cartopalette.core.inference import CartoPalette  # noqa: E402
from cartopalette.scoring.composite import (  # noqa: E402
    score_palette_detailed,
    extract_basemap_colors,
)

from baselines import (  # noqa: E402
    get_colorbrewer_palettes,
    get_matplotlib_palettes,
    get_random_palettes,
    SEQUENTIAL_COLORBREWER, DIVERGING_COLORBREWER,
    SEQUENTIAL_MATPLOTLIB, DIVERGING_MATPLOTLIB,
)


# ── Paths ──
BENCHMARK_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARK_DIR / "output"
TEST_SET_JSON = OUTPUT_DIR / "test_set.json"
RESULTS_CSV = OUTPUT_DIR / "results_raw.csv"
CANDIDATES_JSON = OUTPUT_DIR / "candidates_raw.json"

BASEMAPS_DIR = ROOT / "data" / "processed" / "basemaps"
MODEL_PATH = ROOT / "cartopalette" / "pretrained" / "cartopalette_v4.pt"

# ── Benchmark config ──
SCHEMES = ["sequential", "diverging"]
N_CLASSES_LIST = [3, 5, 7, 9]
METRICS = [
    "composite",
    "basemap_contrast",
    "lightness_contrast",
    "hue_contrast",
    "distinguishability",
    "cvd_robustness",
    "perceptual_ordering",
]


def classify_basemap_brightness(basemap_colors: np.ndarray) -> str:
    """Classify a basemap as 'dark', 'medium', or 'light' based on mean L*."""
    mean_L = basemap_colors[:, 0].mean()
    if mean_L < 45:
        return "dark"
    if mean_L < 70:
        return "medium"
    return "light"


def classify_basemap_chroma(basemap_colors: np.ndarray) -> str:
    """Classify a basemap as 'neutral' or 'chromatic' based on mean chroma."""
    a = basemap_colors[:, 1]
    b = basemap_colors[:, 2]
    mean_C = np.mean(np.sqrt(a**2 + b**2))
    return "neutral" if mean_C < 10 else "chromatic"


def score_single_palette(lab: np.ndarray, basemap_colors: np.ndarray,
                         scheme: str) -> dict:
    """Return the full 7-metric dict for a single palette."""
    details = score_palette_detailed(lab, basemap_colors, scheme)
    return {
        "composite": float(details["composite"]),
        "basemap_contrast": float(details["basemap_contrast"]),
        "lightness_contrast": float(details["lightness_contrast"]),
        "hue_contrast": float(details["hue_contrast"]),
        "distinguishability": float(details["distinguishability"]),
        "cvd_robustness": float(details["cvd_robustness"]),
        "perceptual_ordering": float(details["perceptual_ordering"]),
    }


def best_of_library(palettes: list, basemap_colors: np.ndarray,
                    scheme: str) -> tuple:
    """Score all palettes; return (best_name, best_metrics, best_lab).

    'Best' = highest composite score. This is a pro-baseline choice: we
    let ColorBrewer / Matplotlib pick their best palette for each basemap.
    """
    best_name, best_metrics, best_lab = None, None, None
    best_composite = -1.0
    for name, lab in palettes:
        m = score_single_palette(lab, basemap_colors, scheme)
        if m["composite"] > best_composite:
            best_composite = m["composite"]
            best_name = name
            best_metrics = m
            best_lab = lab
    return best_name, best_metrics, best_lab


def median_of_random(palettes: list, basemap_colors: np.ndarray,
                     scheme: str) -> tuple:
    """Score all random palettes; return median composite and full metric dict.

    The median palette is the one whose composite score is the median
    across all samples — representative of average random performance.
    """
    scored = [
        (name, lab, score_single_palette(lab, basemap_colors, scheme))
        for name, lab in palettes
    ]
    scored.sort(key=lambda x: x[2]["composite"])
    median_idx = len(scored) // 2
    name, lab, m = scored[median_idx]
    return name, m, lab


def cartopalette_top1(cp: CartoPalette, image: Image.Image,
                      scheme: str, n_classes: int) -> tuple:
    """Generate 3 suggestions, return top-1 by composite score.

    This is exactly what the user sees in the Streamlit web app: the first
    suggestion (which is always the highest-scoring palette after MMR).
    """
    suggestions = cp.suggest(image, scheme=scheme, n_classes=n_classes,
                             n_suggestions=3)
    if not suggestions:
        return None, None, None

    top = suggestions[0]
    metrics = {
        "composite": float(top.score),
        "basemap_contrast": float(top.metrics["basemap_contrast"]),
        "lightness_contrast": float(top.metrics["lightness_contrast"]),
        "hue_contrast": float(top.metrics["hue_contrast"]),
        "distinguishability": float(top.metrics["distinguishability"]),
        "cvd_robustness": float(top.metrics["cvd_robustness"]),
        "perceptual_ordering": float(top.metrics["perceptual_ordering"]),
    }
    return "cartopalette_top1", metrics, top.lab


def run(args):
    # Load test set
    with open(TEST_SET_JSON, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    basemaps = test_set["basemaps"]
    if args.limit:
        basemaps = basemaps[: args.limit]

    print(f"Loading CartoPalette model from {MODEL_PATH}")
    cp = CartoPalette(str(MODEL_PATH), device=args.device)
    print(f"Model loaded on {cp.device}")

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    all_candidates = []  # store CartoPalette top-1 palettes for later inspection

    total_combos = len(basemaps) * len(SCHEMES) * len(N_CLASSES_LIST)
    combo_idx = 0
    t_start = time.time()

    for bm in basemaps:
        basemap_path = BASEMAPS_DIR / bm["basemap_filename"]
        image = Image.open(basemap_path).convert("RGB")
        basemap_colors = extract_basemap_colors(str(basemap_path))

        bm_class_brightness = classify_basemap_brightness(basemap_colors)
        bm_class_chroma = classify_basemap_chroma(basemap_colors)

        # Region type for analysis: derive from land cover + climate
        region_type = bm["land_cover"]

        for scheme in SCHEMES:
            for n_classes in N_CLASSES_LIST:
                combo_idx += 1
                elapsed = time.time() - t_start
                print(f"[{combo_idx}/{total_combos}] {bm['basemap_filename']} "
                      f"{scheme}/{n_classes}cls  (elapsed {elapsed:.0f}s)")

                # ── Baseline 1: CartoPalette top-1 ──
                cp_name, cp_metrics, cp_lab = cartopalette_top1(
                    cp, image, scheme, n_classes)

                # ── Baseline 2: ColorBrewer best-of-library ──
                cb_palettes = get_colorbrewer_palettes(scheme, n_classes)
                cb_name, cb_metrics, cb_lab = best_of_library(
                    cb_palettes, basemap_colors, scheme)

                # ── Baseline 3: Matplotlib best-of-library ──
                mpl_palettes = get_matplotlib_palettes(scheme, n_classes)
                mpl_name, mpl_metrics, mpl_lab = best_of_library(
                    mpl_palettes, basemap_colors, scheme)

                # ── Baseline 4: Random median-of-10 ──
                rand_palettes = get_random_palettes(
                    n_classes, n_samples=10, seed=42 + combo_idx)
                rand_name, rand_metrics, rand_lab = median_of_random(
                    rand_palettes, basemap_colors, scheme)

                # ── Write rows: one per (baseline, metric) ──
                for baseline_label, baseline_name, metrics in [
                    ("CartoPalette", cp_name or "none", cp_metrics),
                    ("ColorBrewer", cb_name, cb_metrics),
                    ("Matplotlib", mpl_name, mpl_metrics),
                    ("Random", rand_name, rand_metrics),
                ]:
                    if metrics is None:
                        continue
                    for metric_key in METRICS:
                        raw_rows.append({
                            "basemap_filename": bm["basemap_filename"],
                            "provider": bm["provider"],
                            "location_id": bm["location_id"],
                            "location_name": bm["location_name"],
                            "zoom": bm["zoom"],
                            "zoom_range": bm["zoom_range"],
                            "region_type": region_type,
                            "climate_zone": bm["climate_zone"],
                            "basemap_class": f"{bm_class_brightness}_{bm_class_chroma}",
                            "basemap_brightness": bm_class_brightness,
                            "basemap_chroma": bm_class_chroma,
                            "scheme": scheme,
                            "n_classes": n_classes,
                            "baseline": baseline_label,
                            "baseline_palette_name": baseline_name,
                            "metric": metric_key,
                            "score": metrics[metric_key],
                        })

                # Store the CartoPalette top-1 palette for later inspection
                if cp_lab is not None:
                    all_candidates.append({
                        "basemap_filename": bm["basemap_filename"],
                        "scheme": scheme,
                        "n_classes": n_classes,
                        "cartopalette_top1_lab": cp_lab.tolist(),
                        "cartopalette_top1_metrics": cp_metrics,
                    })

    # ── Write CSV ──
    fieldnames = list(raw_rows[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\nWrote {len(raw_rows)} rows to {RESULTS_CSV}")

    # ── Write candidates JSON ──
    with open(CANDIDATES_JSON, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2)
    print(f"Wrote {len(all_candidates)} CartoPalette candidates to {CANDIDATES_JSON}")

    total_time = time.time() - t_start
    print(f"\nTotal benchmark time: {total_time/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N basemaps (for testing)")
    parser.add_argument("--device", type=str, default=None,
                        choices=[None, "cpu", "cuda"],
                        help="Inference device (default: auto)")
    args = parser.parse_args()

    if not TEST_SET_JSON.exists():
        print("test_set.json not found. Run select_test_set.py first.")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
