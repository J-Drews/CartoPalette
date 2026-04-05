"""
ChromaMap v4 — Dataset Builder with Diversity-Aware Labeling (Vectorized)

Functionally identical to build_dataset.py but uses NumPy vectorization
for the diversity distance computation, reducing runtime from hours to minutes.

Algorithm (unchanged):
1. For each config: candidates are pre-sorted by composite_score (descending)
2. First pick = candidate with highest composite_score
3. For picks 2..top_k: maximize (composite_score + diversity_weight * normalized_min_dist)
   where normalized_min_dist = min over all already-selected palettes of the
   mean per-color minimum Euclidean distance in CIELAB, divided by 80.0, capped at 1.0

Output columns (unchanged):
  patch_id, scheme_type, n_classes, scale_class, split, label_type, rank,
  composite_score, source, palette_lab, palette_rgb, distinguishability,
  basemap_contrast, cvd_robustness, perceptual_ordering, lightness_contrast,
  hue_contrast

Usage:
    python -m research.data_pipeline.build_dataset_fast
    python -m research.data_pipeline.build_dataset_fast --test
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

logger = setup_logging("build_dataset_v4")

NORM_CEILING = 80.0  # same as original


def _compute_min_palette_distance_vec(candidate_lab, selected_lab_stack):
    """Vectorized: mean per-color min Euclidean distance to any selected palette.

    Args:
        candidate_lab: (n_colors, 3) array — one candidate palette
        selected_lab_stack: (n_selected * n_colors, 3) array — all selected
            palettes concatenated

    Returns:
        float: normalized distance in [0, 1], same as original compute_min_palette_distance
    """
    # candidate_lab: (C, 3),  selected_lab_stack: (S, 3)
    # Broadcast: (C, 1, 3) - (1, S, 3) -> (C, S, 3) -> squared -> sum -> sqrt -> (C, S)
    diff = candidate_lab[:, np.newaxis, :] - selected_lab_stack[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (C, S)
    min_per_color = np.min(dists, axis=1)  # (C,)
    raw = float(np.mean(min_per_color))
    return min(raw / NORM_CEILING, 1.0)


def diversity_aware_top_k_selection_fast(candidates, top_k, diversity_weight=0.3):
    """Greedy diversity-aware selection — vectorized version.

    Identical algorithm to the original:
    1. First pick = highest composite_score (candidates[0], already sorted)
    2. For each subsequent pick: maximize composite_score + diversity_weight * norm_min_dist

    Args:
        candidates: list of dicts with 'composite_score' and 'palette_lab'
        top_k: number to select
        diversity_weight: weight for diversity term

    Returns:
        list of selected candidate dicts
    """
    if not candidates:
        return []

    n_cand = len(candidates)
    n_colors = len(candidates[0]["palette_lab"])

    # Pre-convert all palettes to a single numpy array: (n_cand, n_colors, 3)
    all_palettes = np.array([c["palette_lab"] for c in candidates])
    all_scores = np.array([c["composite_score"] for c in candidates])

    # Track selection
    selected_indices = [0]  # first pick = best score
    remaining_mask = np.ones(n_cand, dtype=bool)
    remaining_mask[0] = False

    # Stack of selected palettes flattened: (n_selected * n_colors, 3)
    selected_stack = all_palettes[0].reshape(-1, 3).copy()

    for _ in range(min(top_k - 1, n_cand - 1)):
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) == 0:
            break

        # Compute distances for all remaining candidates at once
        # remaining_palettes: (R, n_colors, 3)
        remaining_palettes = all_palettes[remaining_idx]
        R = len(remaining_idx)

        # Broadcast: (R, n_colors, 1, 3) vs (1, 1, S_flat, 3) -> (R, n_colors, S_flat)
        diff = remaining_palettes[:, :, np.newaxis, :] - selected_stack[np.newaxis, np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=3))  # (R, n_colors, S_flat)
        min_per_color = np.min(dists, axis=2)  # (R, n_colors)
        raw_mean = np.mean(min_per_color, axis=1)  # (R,)
        norm_dist = np.minimum(raw_mean / NORM_CEILING, 1.0)  # (R,)

        combined = all_scores[remaining_idx] + diversity_weight * norm_dist
        best_local = np.argmax(combined)
        best_global = remaining_idx[best_local]

        selected_indices.append(int(best_global))
        remaining_mask[best_global] = False

        # Append to selected stack
        selected_stack = np.vstack([selected_stack, all_palettes[best_global].reshape(-1, 3)])

    return [candidates[i] for i in selected_indices[:top_k]]


def run_build(test_mode: bool = False):
    """Build the final dataset with diversity-aware labeling (vectorized)."""
    root = get_project_root()
    scoring_config = load_yaml("configs/scoring/composite_score.yaml")
    sampling_config = load_yaml("configs/data/sampling.yaml")

    top_k = scoring_config["labeling"]["top_k"]
    diversity_weight = scoring_config["labeling"].get("diversity_weight", 0.3)

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

    with open(patches_dir / "patch_metadata.json") as f:
        patch_metadata = {p["patch_id"]: p for p in json.load(f)}

    scale_classes = {}
    for z_info in sampling_config["zoom_levels"]:
        scale_classes[z_info["zoom"]] = z_info["scale_class"]

    score_files = sorted(scores_dir.glob("*.json"))
    total_files = len(score_files)

    if test_mode:
        score_files = score_files[:2]
        total_files = len(score_files)
        logger.info(f"TEST MODE: processing {total_files} files")

    logger.info(f"Processing {total_files} scored configurations...")

    all_labels = []
    basemaps_copied = set()
    split_ids = {"train": [], "val": [], "test": []}

    for file_idx, score_path in enumerate(score_files):
        # Progress logging every 1000 files
        if (file_idx + 1) % 1000 == 0 or file_idx == 0:
            logger.info(f"  [{file_idx + 1}/{total_files}] Processing {score_path.stem}...")

        try:
            with open(score_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid score file detected: {score_path}. "
                "Repair by rerunning 'python -m research.data_pipeline.score_palettes'."
            ) from exc

        patch_id = data["patch_id"]
        scheme_type = data["scheme_type"]
        n_classes = data["n_classes"]
        candidates = data["candidates"]

        if scheme_type not in allowed_scheme_types:
            continue

        if not candidates:
            continue

        patch_info = patch_metadata.get(patch_id)
        if patch_info is None:
            logger.warning(f"No metadata for {patch_id}")
            continue

        split = patch_info["split"]
        zoom = patch_info["zoom"]
        scale_class = scale_classes.get(zoom, "unknown")

        if patch_id not in basemaps_copied:
            file_rel = patch_info["file"].replace("\\", "/")
            src = root / file_rel
            dst = basemaps_dir / f"{patch_id}.png"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            basemaps_copied.add(patch_id)

        # Vectorized diversity-aware selection
        selected_candidates = diversity_aware_top_k_selection_fast(
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
        description="Build final ChromaMap v4 dataset with diversity-aware labeling (fast)"
    )
    parser.add_argument("--test", action="store_true", help="Test mode (processes 2 files)")
    args = parser.parse_args()
    run_build(test_mode=args.test)


if __name__ == "__main__":
    main()
