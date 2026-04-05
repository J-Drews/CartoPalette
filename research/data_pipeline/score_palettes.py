"""
ChromaMap v4 — Composite Score Computation

Scores each palette candidate against its basemap using six metrics:
1. Distinguishability (CIEDE2000) — min pairwise distance in palette
2. Basemap Contrast (CIEDE2000) — worst-case + mean to basemap colors
3. Lightness Contrast — L* zone avoidance vs basemap
4. Hue Contrast — chroma-weighted hue separation from basemap
5. CVD Robustness — distinguishability under simulated color blindness
6. Perceptual Ordering — L* monotonicity for sequential/diverging

Weights: dist=0.10, contrast=0.20, cvd=0.05, order=0.05, l_contrast=0.30, h_contrast=0.30
(80% basemap-dependent, 20% universal quality)

Usage:
    python -m research.data_pipeline.score_palettes
    python -m research.data_pipeline.score_palettes --test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.utils.color_utils import (
    ciede2000, pairwise_ciede2000, simulate_cvd,
    rgb_to_lab, lab_to_rgb,
)
from research.utils.io_utils import (
    atomic_write_json,
    load_yaml,
    setup_logging,
    get_project_root,
)

logger = setup_logging("score_palettes")


def has_valid_score_output(path: Path) -> bool:
    """Return True when an existing score file is valid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Detected invalid score file {path.name}: {exc}. Recomputing.")
        return False


# ──────────────────────────────────────
#  Individual Metrics (with absolute thresholds)
# ──────────────────────────────────────

def metric_distinguishability(palette_lab: np.ndarray, threshold: float = 30.0) -> float:
    """Minimum pairwise CIEDE2000 distance within the palette, normalized by threshold.

    Args:
        palette_lab: (n, 3) CIELAB palette
        threshold: dE threshold (30 dE00)

    Returns:
        Score in [0, 1], where 1.0 = min_distance >= threshold
    """
    if len(palette_lab) < 2:
        return 1.0

    dist_matrix = pairwise_ciede2000(palette_lab)
    # Get minimum non-zero distance
    mask = dist_matrix > 0
    if not mask.any():
        return 0.0

    min_dist = float(dist_matrix[mask].min())
    return min(min_dist / threshold, 1.0)


def metric_basemap_contrast(
    palette_lab: np.ndarray,
    basemap_colors_lab: np.ndarray,
    threshold: float = 40.0
) -> float:
    """Basemap contrast using worst-case + mean blend.

    For each palette color, compute its min CIEDE2000 to any basemap color.
    Blend 60% worst (ensures no color blends) + 40% mean (rewards overall contrast).

    Args:
        palette_lab: (n_pal, 3) CIELAB palette
        basemap_colors_lab: (n_bm, 3) dominant basemap colors
        threshold: dE threshold (40 dE00)

    Returns:
        Score in [0, 1], where 1.0 = blend >= threshold
    """
    n_pal = len(palette_lab)
    n_bm = len(basemap_colors_lab)

    # Vectorized: compute all pairwise distances
    pal_expanded = np.repeat(palette_lab, n_bm, axis=0)     # (n_pal*n_bm, 3)
    bm_expanded = np.tile(basemap_colors_lab, (n_pal, 1))   # (n_pal*n_bm, 3)

    dists = ciede2000(pal_expanded, bm_expanded)             # vectorized
    dists = dists.reshape(n_pal, n_bm)

    # Min distance per palette color
    per_color = np.min(dists, axis=1)
    worst = float(np.min(per_color))
    mean_val = float(np.mean(per_color))

    # Blend: 60% worst + 40% mean
    blended = 0.6 * worst + 0.4 * mean_val
    return min(blended / threshold, 1.0)


def metric_lightness_contrast(
    palette_lab: np.ndarray,
    basemap_dominant_lab: np.ndarray,
    safe_threshold: float = 15.0,
    normalization: float = 20.0
) -> float:
    """Lightness contrast: proportion of palette colors safely outside basemap L* zone.

    For each palette color, compute min L* distance to any basemap dominant color.
    Safe if > safe_threshold (15 L* units).
    Continuous bonus: how far safe colors are, normalized by normalization.
    v4.1: normalization reduced from 30 to 20 — saturates reward at moderate
    separation to prevent "darker = always better" on light basemaps.

    Args:
        palette_lab: (n, 3) CIELAB palette
        basemap_dominant_lab: (k, 3) dominant basemap colors
        safe_threshold: L* distance to consider "safe" (15 units)
        normalization: normalization factor for continuous bonus (30)

    Returns:
        Score in [0, 1]
    """
    basemap_L = basemap_dominant_lab[:, 0]
    palette_L = palette_lab[:, 0]
    n = len(palette_L)

    # For each palette color: min L* distance to any basemap color
    safe_count = 0
    total_separation = 0.0

    for pL in palette_L:
        min_L_dist = float(np.min(np.abs(pL - basemap_L)))

        if min_L_dist > safe_threshold:
            safe_count += 1

        # Continuous bonus
        total_separation += min(min_L_dist / normalization, 1.0)

    # Blend: 60% proportion-based (discrete) + 40% distance-based (continuous)
    proportion = safe_count / n
    mean_sep = total_separation / n
    return 0.6 * proportion + 0.4 * mean_sep


def metric_hue_contrast(
    palette_lab: np.ndarray,
    basemap_dominant_lab: np.ndarray,
    confidence_threshold: float = 15.0
) -> float:
    """Hue contrast: chroma-weighted hue separation, confidence-based.

    Uses chroma-weighted hue comparison. Low-chroma basemaps have unreliable hue,
    so metric gracefully degrades to neutral (0.5) rather than producing
    meaningless comparisons.

    Args:
        palette_lab: (n, 3) CIELAB palette
        basemap_dominant_lab: (k, 3) dominant basemap colors
        confidence_threshold: chroma level for full confidence (15)

    Returns:
        Score in [0, 1]. Higher = better hue separation.
    """
    def chroma_weighted_hue(lab_colors):
        """Return (mean_hue_angle, mean_chroma)."""
        a = lab_colors[:, 1]
        b = lab_colors[:, 2]
        C = np.sqrt(a**2 + b**2)
        angles = np.arctan2(b, a)

        # Chroma-weighted circular mean
        weights = C / (C.sum() + 1e-10)
        sin_mean = np.sum(weights * np.sin(angles))
        cos_mean = np.sum(weights * np.cos(angles))
        return np.arctan2(sin_mean, cos_mean), float(np.mean(C))

    basemap_hue, basemap_C = chroma_weighted_hue(basemap_dominant_lab)
    palette_hue, palette_C = chroma_weighted_hue(palette_lab)

    # Angular difference (0 to π)
    diff = abs(palette_hue - basemap_hue)
    if diff > np.pi:
        diff = 2 * np.pi - diff

    # Raw hue score: 0 = same hue, 1 = well-separated (180° apart)
    hue_score = min(diff / (np.pi * 0.5), 1.0)

    # Confidence: can we trust the hue comparison?
    # C < 15 = near-neutral → hue is meaningless
    # C >= 15 = colorful → hue matters
    confidence = min(basemap_C / confidence_threshold, 1.0)

    # Blend: confident → use hue_score, not confident → return 0.5 (neutral)
    return confidence * hue_score + (1 - confidence) * 0.5


def metric_cvd_robustness(
    palette_lab: np.ndarray,
    threshold: float = 20.0
) -> float:
    """CVD robustness: mean distinguishability across deuteranopia and protanopia.

    Args:
        palette_lab: (n, 3) CIELAB palette
        threshold: dE threshold (20 dE00)

    Returns:
        Score in [0, 1], where 1.0 = min_distance >= threshold in both CVD types
    """
    palette_rgb = lab_to_rgb(palette_lab)

    scores = []
    for cvd_type in ["deuteranopia", "protanopia"]:
        simulated_rgb = simulate_cvd(palette_rgb, cvd_type)
        simulated_lab = rgb_to_lab(simulated_rgb.astype(np.float64))

        # Min pairwise distance in simulated palette
        if len(simulated_lab) < 2:
            scores.append(1.0)
            continue

        dist_matrix = pairwise_ciede2000(simulated_lab)
        mask = dist_matrix > 0
        if not mask.any():
            scores.append(0.0)
        else:
            min_dist = float(dist_matrix[mask].min())
            scores.append(min(min_dist / threshold, 1.0))

    return float(np.mean(scores))


def metric_perceptual_ordering(palette_lab: np.ndarray, scheme_type: str) -> float:
    """Perceptual ordering quality for sequential and diverging schemes.

    Combines:
    1. L* monotonicity — Spearman correlation
    2. Step uniformity — evenness of CIEDE2000 between adjacent colors
    3. For diverging: symmetry of arms around midpoint

    For non-sequential/diverging: returns 1.0.

    Args:
        palette_lab: (n, 3) CIELAB palette
        scheme_type: "sequential", "diverging", or other

    Returns:
        Score in [0, 1], where 1.0 = perfect ordering.
    """
    L_values = palette_lab[:, 0]
    n = len(L_values)

    if n < 2:
        return 1.0

    if scheme_type == "sequential":
        # --- Sub-criterion 1: L* monotonicity (Spearman) ---
        indices = np.arange(n)
        rho, _ = stats.spearmanr(indices, L_values)
        monotonicity = abs(rho) if not np.isnan(rho) else 0.0

        # --- Sub-criterion 2: Step uniformity ---
        adjacent_dists = ciede2000(palette_lab[:-1], palette_lab[1:])
        adjacent_dists = np.atleast_1d(adjacent_dists)

        if adjacent_dists.mean() > 0:
            # Coefficient of variation: low = uniform steps, high = uneven
            cv = adjacent_dists.std() / adjacent_dists.mean()
            # Convert to score: cv=0 → 1.0, cv=1 → 0.37, cv=2 → 0.14
            uniformity = float(np.exp(-cv))
        else:
            uniformity = 0.0

        # Weighted combination: 70% monotonicity + 30% uniformity
        return 0.7 * monotonicity + 0.3 * uniformity

    elif scheme_type == "diverging":
        # A diverging scheme has two arms radiating from a neutral midpoint.
        mid = n // 2

        if n < 3:
            return 1.0

        # Split into two arms (both include the midpoint)
        left_arm = palette_lab[:mid + 1]    # start → midpoint
        right_arm = palette_lab[mid:]        # midpoint → end

        left_L = left_arm[:, 0]
        right_L = right_arm[:, 0]

        # --- Sub-criterion 1: Monotonicity of each arm ---
        mono_scores = []
        for arm_L in [left_L, right_L]:
            if len(arm_L) >= 2:
                rho, _ = stats.spearmanr(np.arange(len(arm_L)), arm_L)
                mono_scores.append(abs(rho) if not np.isnan(rho) else 0.0)
            else:
                mono_scores.append(1.0)
        monotonicity = np.mean(mono_scores)

        # --- Sub-criterion 2: Arms should go in OPPOSITE directions ---
        left_trend = left_L[-1] - left_L[0]    # midpoint L* minus start L*
        right_trend = right_L[-1] - right_L[0]  # end L* minus midpoint L*

        if abs(left_trend) > 1 and abs(right_trend) > 1:
            opposite = 1.0 if (left_trend * right_trend) < 0 else 0.0
        else:
            opposite = 0.5

        # --- Sub-criterion 3: L* symmetry between arms ---
        mid_L = palette_lab[mid, 0]
        left_deviations = np.abs(left_L[::-1] - mid_L)
        right_deviations = np.abs(right_L - mid_L)

        # Pad shorter arm if n is even
        min_len = min(len(left_deviations), len(right_deviations))
        left_dev = left_deviations[:min_len]
        right_dev = right_deviations[:min_len]

        if left_dev.sum() + right_dev.sum() > 0:
            asymmetry = np.mean(np.abs(left_dev - right_dev))
            symmetry = float(np.exp(-asymmetry / 20.0))
        else:
            symmetry = 1.0

        # Weighted combination
        return 0.4 * monotonicity + 0.3 * opposite + 0.3 * symmetry

    return 1.0


# ──────────────────────────────────────
#  Composite Score
# ──────────────────────────────────────

def compute_composite_score(
    palette_lab: np.ndarray,
    basemap_colors_lab: np.ndarray,
    scheme_type: str,
    weights: dict,
    source: str = "unknown",
) -> dict:
    """Compute all 6 metrics and weighted composite score.

    Uses absolute thresholds (no min-max normalization).

    Args:
        palette_lab: (n, 3) CIELAB palette
        basemap_colors_lab: (k, 3) dominant basemap colors
        scheme_type: "sequential" or "diverging" (v4 only handles these)
        weights: Dict with metric names → weights
        source: Candidate source tag (colorbrewer, cielab_adaptive, synthetic, etc.)

    Returns:
        Dict with individual metrics (scores in [0,1]) and composite score
    """
    dist = metric_distinguishability(palette_lab, threshold=30.0)
    contrast = metric_basemap_contrast(palette_lab, basemap_colors_lab, threshold=40.0)
    l_contrast = metric_lightness_contrast(palette_lab, basemap_colors_lab)
    h_contrast = metric_hue_contrast(palette_lab, basemap_colors_lab)
    cvd = metric_cvd_robustness(palette_lab, threshold=20.0)
    ordering = metric_perceptual_ordering(palette_lab, scheme_type)

    # Weighted composite (no normalization — all metrics are already in [0,1])
    composite = (
        weights.get("distinguishability", 0.0) * dist +
        weights.get("basemap_contrast", 0.0) * contrast +
        weights.get("lightness_contrast", 0.0) * l_contrast +
        weights.get("hue_contrast", 0.0) * h_contrast +
        weights.get("cvd_robustness", 0.0) * cvd +
        weights.get("perceptual_ordering", 0.0) * ordering
    )

    # v4.1: Dark penalty — prevent excessively dark palettes
    mean_palette_L = float(np.mean(palette_lab[:, 0]))
    mean_basemap_L = float(np.mean(basemap_colors_lab[:, 0]))
    dark_penalty = 0.0
    if mean_basemap_L > 65 and mean_palette_L < 35:
        dark_penalty = min((35 - mean_palette_L) / 20.0, 1.0) * 0.06
    elif mean_basemap_L > 40 and mean_palette_L < 25:
        dark_penalty = min((25 - mean_palette_L) / 15.0, 1.0) * 0.03
    composite = max(composite - dark_penalty, 0.0)

    # v4.1: Source-aware adjustments (mild)
    # Rationale: Pure ColorBrewer palettes are cartographically proven (Brewer 2003)
    # but score low on basemap-dependent metrics (80% weight) since they're not
    # basemap-adapted. Synthetic palettes are over-optimized for the scoring function.
    source_bonus = 0.0
    if source == "colorbrewer":
        source_bonus = 0.02
    elif source == "contrast_optimized":
        source_bonus = 0.005
    elif source == "synthetic":
        source_bonus = -0.010
    composite = max(composite + source_bonus, 0.0)

    return {
        "distinguishability": float(dist),
        "basemap_contrast": float(contrast),
        "lightness_contrast": float(l_contrast),
        "hue_contrast": float(h_contrast),
        "cvd_robustness": float(cvd),
        "perceptual_ordering": float(ordering),
        "composite_score": float(composite),
        "dark_penalty": float(dark_penalty),
        "source_bonus": float(source_bonus),
    }


# ──────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────

def run_scoring(test_mode: bool = False):
    """Score all palette candidates using v4 composite scoring.

    Reads from data/interim/candidates/, writes to data/interim/scores/.
    Skips candidates that have already been scored.
    """
    root = get_project_root()
    # Note: using the v4 config location
    scoring_config = load_yaml("configs/scoring/composite_score.yaml")

    candidates_dir = root / "data" / "interim" / "candidates"
    basemap_colors_dir = root / "data" / "interim" / "basemap_colors"
    scores_dir = root / "data" / "interim" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(candidates_dir.glob("*.json"))

    if test_mode:
        candidate_files = candidate_files[:2]
        logger.info(f"TEST MODE: scoring {len(candidate_files)} files")

    total_scored = 0
    repaired_files = 0
    skipped_valid = 0

    for cand_path in candidate_files:
        output_path = scores_dir / cand_path.name
        output_was_invalid = False

        if output_path.exists():
            if has_valid_score_output(output_path):
                skipped_valid += 1
                logger.info(f"Skipping {cand_path.stem} (already scored)")
                continue
            output_was_invalid = True

        with open(cand_path, encoding="utf-8") as f:
            cand_data = json.load(f)

        patch_id = cand_data["patch_id"]
        scheme_type = cand_data["scheme_type"]

        # v4: only handles sequential and diverging
        if scheme_type not in ["sequential", "diverging"]:
            logger.warning(f"Skipping {cand_path.stem}: unsupported scheme_type '{scheme_type}'")
            continue

        # Load basemap colors
        color_path = basemap_colors_dir / f"{patch_id}_colors.json"
        if not color_path.exists():
            logger.error(f"Missing basemap colors: {color_path}")
            continue

        with open(color_path, encoding="utf-8") as f:
            basemap_colors_lab = np.array(json.load(f)["colors_lab"])

        # Get weights (v4 uses single weight set for seq/div)
        weights = scoring_config["weights"].get("sequential_diverging", {})
        if not weights:
            logger.error(f"Missing weights in config for {cand_path.stem}")
            continue

        # Score each candidate
        scored_candidates = []
        for i, candidate in enumerate(cand_data["candidates"]):
            palette_lab = np.array(candidate["palette_lab"])
            source = candidate.get("source", "unknown")
            score_dict = compute_composite_score(palette_lab, basemap_colors_lab, scheme_type, weights, source=source)

            scored_candidates.append({
                "index": i,
                "source": candidate.get("source", "unknown"),
                "palette_lab": candidate["palette_lab"],
                "palette_rgb": candidate.get("palette_rgb", []),
                "metrics": {
                    "distinguishability": score_dict["distinguishability"],
                    "basemap_contrast": score_dict["basemap_contrast"],
                    "lightness_contrast": score_dict["lightness_contrast"],
                    "hue_contrast": score_dict["hue_contrast"],
                    "cvd_robustness": score_dict["cvd_robustness"],
                    "perceptual_ordering": score_dict["perceptual_ordering"],
                },
                "composite_score": score_dict["composite_score"],
                "dark_penalty": score_dict["dark_penalty"],
                "source_bonus": score_dict["source_bonus"],
            })

        # Sort by composite score (best first)
        scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        output_data = {
            "patch_id": patch_id,
            "scheme_type": scheme_type,
            "n_classes": cand_data.get("n_classes", 0),
            "n_candidates": len(scored_candidates),
            "candidates": scored_candidates,
        }

        atomic_write_json(output_path, output_data)

        if output_was_invalid:
            repaired_files += 1
            logger.info(f"Repaired {cand_path.stem}")
        else:
            total_scored += 1
            logger.info(f"Scored {cand_path.stem} ({len(scored_candidates)} candidates)")

    logger.info(f"Scoring complete! Scored: {total_scored}, Skipped: {skipped_valid}, Repaired: {repaired_files}")


def main():
    parser = argparse.ArgumentParser(description="Score palette candidates (v4)")
    parser.add_argument("--test", action="store_true", help="Test mode with limited scope")
    args = parser.parse_args()
    run_scoring(test_mode=args.test)


if __name__ == "__main__":
    main()
