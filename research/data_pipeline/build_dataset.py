"""
ChromaMap v4 — Dataset Builder with Diversity-Aware Labeling

Assembles the final training dataset from scored candidates:
1. Applies diversity-aware Top-K selection (greedy algorithm)
   - Selects best-scored candidate first
   - For each subsequent: maximize score + diversity_weight * min_palette_distance_to_selected
2. Only includes sequential and diverging scheme types (no qualitative)
3. Copies basemap images to processed/basemaps/
4. Creates a unified labels.csv file with 6 new metrics
5. Generates train/val/test split files
6. Computes dataset statistics

Output columns:
  patch_id, scheme_type, n_classes, scale_class, split, label_type, rank,
  composite_score, source, palette_lab, palette_rgb, distinguishability,
  basemap_contrast, cvd_robustness, perceptual_ordering, lightness_contrast,
  hue_contrast

Usage:
    python -m research.data_pipeline.build_dataset
    python -m research.data_pipeline.build_dataset --test
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.utils.io_utils import load_yaml, setup_logging, get_project_root
from research.utils.color_utils import lab_to_rgb

logger = setup_logging("build_dataset_v4")


def euclidean_distance_lab(lab1, lab2):
    """Compute Euclidean distance between two LAB colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def compute_min_palette_distance(palette_lab, selected_palettes_lab):
    """
    Compute mean per-color Euclidean distance in CIELAB to nearest selected palette,
    normalized to [0, 1] by a ceiling of 80 dE (CIELAB Euclidean).

    A distance of 80 represents palettes in completely different color regions
    (e.g., max L* range 0-100 plus chroma differences). This ensures the
    diversity term is comparable in scale to the composite score [0, 1].

    Args:
        palette_lab: list of [L, a, b] colors for current candidate
        selected_palettes_lab: list of palettes already selected

    Returns:
        float: normalized mean distance in [0, 1]
    """
    NORM_CEILING = 80.0  # max expected CIELAB Euclidean distance

    if not selected_palettes_lab:
        return 1.0  # maximum diversity for first candidate after seed

    distances = []
    for color in palette_lab:
        min_dist = float('inf')
        for sel_palette in selected_palettes_lab:
            for sel_color in sel_palette:
                dist = euclidean_distance_lab(color, sel_color)
                min_dist = min(min_dist, dist)
        distances.append(min_dist)

    raw = float(np.mean(distances)) if distances else 0.0
    return min(raw / NORM_CEILING, 1.0)


def diversity_aware_top_k_selection(candidates, top_k, diversity_weight=0.3):
    """
    Greedy diversity-aware selection of top-k palettes.

    Selection criterion: score + diversity_weight * min_palette_distance_to_selected

    Args:
        candidates: list of candidate dicts with 'composite_score' and 'palette_lab'
        top_k: number of palettes to select
        diversity_weight: weight for diversity term (default 0.3)

    Returns:
        list: selected candidates in selection order
    """
    if not candidates:
        return []

    # First selection: best score
    selected = [candidates[0]]
    selected_palettes_lab = [candidates[0]["palette_lab"]]
    remaining = candidates[1:]

    # Remaining selections: maximize score + diversity
    for _ in range(min(top_k - 1, len(remaining))):
        best_candidate = None
        best_score = -float('inf')
        best_idx = -1

        for idx, candidate in enumerate(remaining):
            min_dist = compute_min_palette_distance(
                candidate["palette_lab"],
                selected_palettes_lab
            )
            combined_score = (
                candidate["composite_score"] +
                diversity_weight * min_dist
            )

            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
                best_idx = idx

        if best_candidate is not None:
            selected.append(best_candidate)
            selected_palettes_lab.append(best_candidate["palette_lab"])
            remaining.pop(best_idx)
        else:
            break

    return selected[:top_k]


def run_build(test_mode: bool = False):
    """Build the final dataset with diversity-aware labeling."""
    root = get_project_root()
    scoring_config = load_yaml("configs/scoring/composite_score.yaml")
    sampling_config = load_yaml("configs/data/sampling.yaml")

    top_k = scoring_config["labeling"]["top_k"]
    diversity_weight = scoring_config["labeling"].get("diversity_weight", 0.3)

    # Only include sequential and diverging (no qualitative)
    allowed_scheme_types = {"sequential", "diverging"}

    scores_dir = root / "data" / "interim" / "scores"
    patches_dir = root / "data" / "interim" / "patches"
    processed_dir = root / "data" / "processed"
    basemaps_dir = processed_dir / "basemaps"
    labels_dir = processed_dir / "labels"
    splits_dir = processed_dir / "splits"
    stats_dir = processed_dir / "statistics"

    for d in [basemaps_dir, labels_dir, splits_dir, stats_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load patch metadata for split info
    with open(patches_dir / "patch_metadata.json") as f:
        patch_metadata = {p["patch_id"]: p for p in json.load(f)}

    # Build zoom → scale_class mapping
    scale_classes = {}
    for z_info in sampling_config["zoom_levels"]:
        scale_classes[z_info["zoom"]] = z_info["scale_class"]

    # Process all scored files
    score_files = sorted(scores_dir.glob("*.json"))
    if test_mode:
        score_files = score_files[:2]
        logger.info(f"TEST MODE: processing {len(score_files)} files")

    all_labels = []
    basemaps_copied = set()
    split_ids = {"train": [], "val": [], "test": []}

    for score_path in score_files:
        try:
            with open(score_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid score file detected: {score_path}. "
                "Repair by rerunning 'python -m research.data_pipeline.score_palettes' "
                "with the hardened atomic writer."
            ) from exc

        patch_id = data["patch_id"]
        scheme_type = data["scheme_type"]
        n_classes = data["n_classes"]
        candidates = data["candidates"]

        # Skip if scheme type not in allowed list
        if scheme_type not in allowed_scheme_types:
            logger.debug(f"Skipping {patch_id} ({scheme_type}): not sequential or diverging")
            continue

        if not candidates:
            continue

        # Get patch info
        patch_info = patch_metadata.get(patch_id)
        if patch_info is None:
            logger.warning(f"No metadata for {patch_id}")
            continue

        split = patch_info["split"]
        zoom = patch_info["zoom"]
        scale_class = scale_classes.get(zoom, "unknown")

        # Copy basemap image to processed/ (once per patch)
        if patch_id not in basemaps_copied:
            # Normalize Windows backslashes → forward slashes (metadata was
            # generated on Windows, but pipeline may run on Linux/Colab)
            file_rel = patch_info["file"].replace("\\", "/")
            src = root / file_rel
            dst = basemaps_dir / f"{patch_id}.png"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            basemaps_copied.add(patch_id)

        # Apply diversity-aware top-k selection
        selected_candidates = diversity_aware_top_k_selection(
            candidates,
            top_k,
            diversity_weight=diversity_weight
        )

        for rank, candidate in enumerate(selected_candidates):
            label_entry = {
                "patch_id": patch_id,
                "scheme_type": scheme_type,
                "n_classes": n_classes,
                "scale_class": scale_class,
                "split": split,
                "label_type": "positive",
                "rank": rank,
                "composite_score": candidate["composite_score"],
                "source": candidate["source"],
                "palette_lab": json.dumps(candidate["palette_lab"]),
                "palette_rgb": json.dumps(candidate["palette_rgb"]),
                "distinguishability": candidate["metrics"].get("distinguishability", None),
                "basemap_contrast": candidate["metrics"].get("basemap_contrast", None),
                "cvd_robustness": candidate["metrics"].get("cvd_robustness", None),
                "perceptual_ordering": candidate["metrics"].get("perceptual_ordering", None),
                "lightness_contrast": candidate["metrics"].get("lightness_contrast", None),
                "hue_contrast": candidate["metrics"].get("hue_contrast", None),
            }
            all_labels.append(label_entry)

        # Track split membership
        config_id = f"{patch_id}_{scheme_type}_{n_classes}c"
        split_ids[split].append(config_id)

    # Save labels as CSV
    labels_path = labels_dir / "labels.csv"
    if all_labels:
        fieldnames = [
            "patch_id", "scheme_type", "n_classes", "scale_class", "split",
            "label_type", "rank", "composite_score", "source",
            "palette_lab", "palette_rgb", "distinguishability",
            "basemap_contrast", "cvd_robustness", "perceptual_ordering",
            "lightness_contrast", "hue_contrast"
        ]
        with open(labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_labels)
        logger.info(f"Saved {len(all_labels)} labels to {labels_path}")

    # Save split files
    for split_name, ids in split_ids.items():
        split_path = splits_dir / f"{split_name}.txt"
        with open(split_path, "w") as f:
            f.write("\n".join(sorted(set(ids))))
        logger.info(f"Saved {len(set(ids))} {split_name} configs to {split_path}")

    # Compute and save statistics
    if all_labels:
        scores = [l["composite_score"] for l in all_labels if l["label_type"] == "positive"]

        statistics = {
            "total_labels": len(all_labels),
            "positive_labels": len([l for l in all_labels if l["label_type"] == "positive"]),
            "unique_basemaps": len(basemaps_copied),
            "splits": {k: len(set(v)) for k, v in split_ids.items()},
            "score_distribution": {
                "mean": float(np.mean(scores)) if scores else 0,
                "std": float(np.std(scores)) if scores else 0,
                "min": float(np.min(scores)) if scores else 0,
                "max": float(np.max(scores)) if scores else 0,
                "q25": float(np.percentile(scores, 25)) if scores else 0,
                "median": float(np.percentile(scores, 50)) if scores else 0,
                "q75": float(np.percentile(scores, 75)) if scores else 0,
            },
            "scheme_type_distribution": {},
            "source_distribution": {},
            "diversity_weight_used": diversity_weight,
        }

        from collections import Counter
        statistics["scheme_type_distribution"] = dict(Counter(
            l["scheme_type"] for l in all_labels
        ))
        statistics["source_distribution"] = dict(Counter(
            l["source"] for l in all_labels
        ))

        stats_path = stats_dir / "dataset_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset built successfully (v4 diversity-aware)!")
        logger.info(f"{'='*60}")
        logger.info(f"  Total positive labels: {statistics['positive_labels']}")
        logger.info(f"  Unique basemaps: {statistics['unique_basemaps']}")
        logger.info(f"  Splits: {statistics['splits']}")
        logger.info(f"  Score range: {statistics['score_distribution']['min']:.3f} - {statistics['score_distribution']['max']:.3f}")
        logger.info(f"  Score mean: {statistics['score_distribution']['mean']:.3f}")
        logger.info(f"  Scheme types: {statistics['scheme_type_distribution']}")
        logger.info(f"  Diversity weight: {diversity_weight}")
        logger.info(f"{'='*60}\n")
    else:
        logger.warning("No labels generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Build final ChromaMap v4 dataset with diversity-aware labeling"
    )
    parser.add_argument("--test", action="store_true", help="Test mode (processes 2 files)")
    args = parser.parse_args()
    run_build(test_mode=args.test)


if __name__ == "__main__":
    main()