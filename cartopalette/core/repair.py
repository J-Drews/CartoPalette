"""CartoPalette v4.2 — LAB Contrast Repair.

Surgical post-selection refinement: identify palette colors that are too close
to any dominant basemap color and shift them away in the CIELAB a*b* plane,
preserving the lightness structure (sequential monotonicity, diverging
symmetry). The repair only commits a change when it strictly improves
basemap_contrast AND no other metric drops by more than ε.

Design constraints:
    1. a*b*-priority: never touch L* unless absolutely needed. Sequential and
       diverging palettes depend on a clean L* progression — moving L* risks
       breaking perceptual_ordering. By rotating/translating in the chroma
       plane we keep the lightness ladder intact.
    2. Accept gate: every candidate move is rescored. If basemap_contrast does
       not increase strictly, or any other metric drops by > ε, the move is
       rejected and the previous state retained.
    3. Iterative but bounded: a small step (~3 ΔE per attempt) keeps each move
       reversible. We try at most `max_iter` repair passes per palette.

Realistic expected gain: +0.02 to +0.04 on top-3 mean basemap_contrast, larger
on hard basemaps (satellite imagery, chromatically dense city tiles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from cartopalette.scoring.composite import (
    ciede2000,
    score_palette_detailed,
)


# ── Default accept-gate epsilons (per metric) ──
# A repair pass is accepted only if basemap_contrast strictly increases AND
# every other tracked metric stays within `default_epsilon` of its pre-repair
# value. These mirror the constrained reranker's tolerance budget so the two
# stages stay consistent.
DEFAULT_REPAIR_EPSILON = {
    "distinguishability":  0.02,
    "cvd_robustness":      0.02,
    "perceptual_ordering": 0.02,
    "hue_contrast":        0.02,
    "lightness_contrast":  0.03,
}

# ── Default repair parameters ──
DEFAULT_REPAIR_CONFIG = {
    "target_min_dE":         18.0,  # Stop once worst palette colour is ≥ this from any basemap colour
    "trigger_min_dE":        12.0,  # Skip repair entirely if worst already exceeds this
    "step_size_ab":           3.0,  # ΔE step in a*b* per iteration
    "max_step_size_L":        4.0,  # Maximum L* shift allowed (only if a*b* moves exhaust)
    "max_iterations":          8,
    "out_of_gamut_threshold":  np.inf,  # disabled — clipping handled at RGB stage
    "allow_L_repair":         True,  # Allow tiny L* nudges as last resort
}


@dataclass
class RepairOutcome:
    """Carries the outcome of a repair attempt for diagnostics / debugging."""
    accepted: bool
    new_palette_lab: np.ndarray  # Final palette (either repaired or original)
    new_metrics: dict
    n_iterations: int = 0
    color_indices_touched: list = field(default_factory=list)
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────
# Helper geometry
# ──────────────────────────────────────────────────────────────────────

def _nearest_basemap(
    color_lab: np.ndarray,
    basemap_lab: np.ndarray,
) -> tuple:
    """Return (index, ΔE) of the closest basemap dominant colour."""
    best_idx, best_dE = 0, float("inf")
    for i, bc in enumerate(basemap_lab):
        dE = ciede2000(color_lab, bc)
        if dE < best_dE:
            best_dE = dE
            best_idx = i
    return best_idx, best_dE


def _worst_color(
    palette_lab: np.ndarray,
    basemap_lab: np.ndarray,
) -> tuple:
    """Return (palette index, nearest basemap index, ΔE) of the worst palette colour."""
    worst_pal, worst_bm, worst_dE = 0, 0, float("inf")
    for i, pc in enumerate(palette_lab):
        bm_idx, dE = _nearest_basemap(pc, basemap_lab)
        if dE < worst_dE:
            worst_dE = dE
            worst_pal = i
            worst_bm = bm_idx
    return worst_pal, worst_bm, worst_dE


def _push_ab(
    color_lab: np.ndarray,
    basemap_color_lab: np.ndarray,
    step: float,
) -> np.ndarray:
    """Move a single palette colour `step` ΔE-units away from the basemap colour
    in the a*b* plane while keeping L* fixed.

    If the colour is essentially colocated with the basemap colour in a*b*
    (very rare in practice), we pick a random outward direction.
    """
    da = color_lab[1] - basemap_color_lab[1]
    db = color_lab[2] - basemap_color_lab[2]
    norm = float(np.hypot(da, db))
    if norm < 1e-6:
        # Degenerate: pick an arbitrary outward direction
        da, db, norm = 1.0, 0.0, 1.0

    out = color_lab.copy()
    out[1] = color_lab[1] + step * (da / norm)
    out[2] = color_lab[2] + step * (db / norm)
    # Bound a*, b* to a sensible CIELAB range (well within sRGB-reachable)
    out[1] = float(np.clip(out[1], -100.0, 100.0))
    out[2] = float(np.clip(out[2], -100.0, 100.0))
    return out


def _push_L_safe(
    palette_lab: np.ndarray,
    color_idx: int,
    basemap_color_lab: np.ndarray,
    scheme_type: str,
    step: float,
) -> Optional[np.ndarray]:
    """Move colour at `color_idx` in L* direction that moves AWAY from the
    basemap colour and PRESERVES scheme structure.

    For sequential palettes:
        Allowed direction depends on the palette's overall L* trend and the
        colour's position. We pick the sign that keeps monotonicity intact.

    For diverging palettes:
        Only the centre colour is free to move arbitrarily; arm colours must
        respect the arm's local direction.

    Returns the modified palette or None if no safe L* direction exists.
    """
    n = len(palette_lab)
    L = palette_lab[:, 0]
    pc_L = L[color_idx]
    bm_L = basemap_color_lab[0]

    # Direction away from basemap L*
    if pc_L > bm_L:
        direction = +1.0  # push lighter
    elif pc_L < bm_L:
        direction = -1.0  # push darker
    else:
        return None  # Same L* — no preferred direction

    if scheme_type == "sequential" and n >= 2:
        # Determine overall trend: ascending (L[0] < L[-1]) or descending
        ascending = L[-1] > L[0]
        # Floor/ceiling determined by neighbours so monotonicity is preserved.
        if ascending:
            lo = L[color_idx - 1] + 0.5 if color_idx > 0 else -np.inf
            hi = L[color_idx + 1] - 0.5 if color_idx < n - 1 else np.inf
        else:
            lo = L[color_idx + 1] + 0.5 if color_idx < n - 1 else -np.inf
            hi = L[color_idx - 1] - 0.5 if color_idx > 0 else np.inf
        new_L = pc_L + direction * step
        new_L = float(np.clip(new_L, lo, hi))
        if abs(new_L - pc_L) < 0.25:
            return None  # No room to move

    elif scheme_type == "diverging" and n >= 3:
        mid = n // 2
        if color_idx == mid:
            # Centre — almost free to move, but keep [50, 95] range and not below either arm
            arms_max = max(L[:mid].max() if mid > 0 else 0, L[mid+1:].max() if mid+1 < n else 0)
            new_L = pc_L + direction * step
            new_L = float(np.clip(new_L, max(50.0, arms_max + 1.0), 95.0))
        else:
            # Arm colour — respect arm's local monotonicity towards the centre
            in_left_arm = color_idx < mid
            if in_left_arm:
                lo = L[color_idx - 1] + 0.5 if color_idx > 0 else -np.inf
                hi = L[color_idx + 1] - 0.5
            else:
                lo = L[color_idx - 1] - 0.5
                hi = L[color_idx + 1] + 0.5 if color_idx < n - 1 else np.inf
            # Note: for diverging arms L* may go either way; clip to the local window
            lo_bound, hi_bound = min(lo, hi), max(lo, hi)
            new_L = pc_L + direction * step
            new_L = float(np.clip(new_L, lo_bound, hi_bound))
            if abs(new_L - pc_L) < 0.25:
                return None
    else:
        # No scheme constraints — just push
        new_L = pc_L + direction * step

    out = palette_lab.copy()
    out[color_idx, 0] = new_L
    return out


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def repair_palette(
    palette_lab: np.ndarray,
    basemap_lab: np.ndarray,
    scheme_type: str,
    pre_metrics: Optional[dict] = None,
    config: Optional[dict] = None,
    epsilon: Optional[dict] = None,
) -> RepairOutcome:
    """Run iterative LAB contrast repair on a single palette.

    Args:
        palette_lab: (n_classes, 3) palette in CIELAB.
        basemap_lab: (k_centroids, 3) dominant basemap colours in CIELAB.
        scheme_type: "sequential" or "diverging".
        pre_metrics: Optional pre-computed metrics for the input palette.
            If None, they are computed via score_palette_detailed.
        config: Override DEFAULT_REPAIR_CONFIG. Missing keys fall through to defaults.
        epsilon: Per-metric tolerance budget. Defaults to DEFAULT_REPAIR_EPSILON.

    Returns:
        RepairOutcome with `accepted=True` and the repaired palette if at least
        one repair pass committed, otherwise `accepted=False` with the original
        palette and the reason.
    """
    cfg = {**DEFAULT_REPAIR_CONFIG, **(config or {})}
    eps = {**DEFAULT_REPAIR_EPSILON, **(epsilon or {})}

    if pre_metrics is None:
        pre_metrics = score_palette_detailed(palette_lab, basemap_lab, scheme_type)
    original_metrics = dict(pre_metrics)
    baseline = pre_metrics  # Best-known accepted state

    palette = palette_lab.copy()
    metrics = dict(baseline)
    touched_indices: list = []
    iterations = 0

    # ── Skip-fast: nothing to do if already well-separated ──
    worst_pal_idx, worst_bm_idx, worst_dE = _worst_color(palette, basemap_lab)
    if worst_dE >= cfg["trigger_min_dE"]:
        return RepairOutcome(
            accepted=False,
            new_palette_lab=palette,
            new_metrics=metrics,
            reason=f"skip: worst ΔE {worst_dE:.1f} already ≥ trigger {cfg['trigger_min_dE']:.1f}",
        )

    accepted_any = False

    for it in range(cfg["max_iterations"]):
        iterations = it + 1
        worst_pal_idx, worst_bm_idx, worst_dE = _worst_color(palette, basemap_lab)
        if worst_dE >= cfg["target_min_dE"]:
            break

        # Try an a*b* push first
        candidate = palette.copy()
        candidate[worst_pal_idx] = _push_ab(
            palette[worst_pal_idx], basemap_lab[worst_bm_idx], cfg["step_size_ab"]
        )

        cand_metrics = score_palette_detailed(candidate, basemap_lab, scheme_type)

        if _accept(cand_metrics, baseline, eps, floor_metrics=original_metrics):
            palette = candidate
            baseline = cand_metrics
            metrics = cand_metrics
            if worst_pal_idx not in touched_indices:
                touched_indices.append(worst_pal_idx)
            accepted_any = True
            continue

        # a*b* push rejected — try L* push as fallback, if allowed
        if cfg["allow_L_repair"]:
            candidate_L = _push_L_safe(
                palette, worst_pal_idx, basemap_lab[worst_bm_idx],
                scheme_type, cfg["max_step_size_L"]
            )
            if candidate_L is not None:
                cand_metrics_L = score_palette_detailed(candidate_L, basemap_lab, scheme_type)
                if _accept(cand_metrics_L, baseline, eps, floor_metrics=original_metrics):
                    palette = candidate_L
                    baseline = cand_metrics_L
                    metrics = cand_metrics_L
                    if worst_pal_idx not in touched_indices:
                        touched_indices.append(worst_pal_idx)
                    accepted_any = True
                    continue

        # Neither move accepted — stop early (the palette is locked at a Pareto wall)
        break

    reason = (
        f"accepted {iterations} iter(s), touched {touched_indices}"
        if accepted_any
        else f"no acceptable move after {iterations} iter(s)"
    )
    return RepairOutcome(
        accepted=accepted_any,
        new_palette_lab=palette,
        new_metrics=metrics,
        n_iterations=iterations,
        color_indices_touched=touched_indices,
        reason=reason,
    )


# ──────────────────────────────────────────────────────────────────────
# Accept gate
# ──────────────────────────────────────────────────────────────────────

def _accept(
    new_metrics: dict,
    baseline_metrics: dict,
    epsilon: dict,
    floor_metrics: Optional[dict] = None,
) -> bool:
    """Accept a repair pass only if basemap_contrast strictly increases AND no
    other tracked metric drops by more than its ε tolerance.
    """
    if new_metrics.get("basemap_contrast", 0.0) <= baseline_metrics.get("basemap_contrast", 0.0):
        return False

    floor_metrics = floor_metrics or baseline_metrics
    for metric, eps in epsilon.items():
        before = floor_metrics.get(metric, 0.0)
        after = new_metrics.get(metric, 0.0)
        if after < before - eps:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────
# Batch convenience: repair an entire top-k list in place
# ──────────────────────────────────────────────────────────────────────

def repair_palette_list(
    palettes: Iterable,  # Iterable of Palette objects (from inference.py)
    basemap_lab: np.ndarray,
    config: Optional[dict] = None,
    epsilon: Optional[dict] = None,
) -> list:
    """Apply repair_palette to each item in `palettes`. Returns a new list of
    repaired Palette objects (original ones are not mutated).

    Each Palette is expected to expose .lab, .scheme_type, .metrics.
    Updated Palettes get fresh .lab, .metrics, and .score.
    """
    # Local import keeps this module free of inference.py at import time.
    from cartopalette.core.inference import Palette

    repaired = []
    for p in palettes:
        outcome = repair_palette(
            palette_lab=p.lab,
            basemap_lab=basemap_lab,
            scheme_type=p.scheme_type,
            pre_metrics=p.metrics,
            config=config,
            epsilon=epsilon,
        )
        if outcome.accepted:
            new_p = Palette(
                lab=outcome.new_palette_lab,
                scheme_type=p.scheme_type,
                n_classes=p.n_classes,
                score=outcome.new_metrics.get("composite"),
                metrics=outcome.new_metrics,
            )
            # Tag for diagnostics
            new_p._repair_iterations = outcome.n_iterations
            new_p._repair_touched = outcome.color_indices_touched
            repaired.append(new_p)
        else:
            # Unchanged
            repaired.append(p)
    return repaired
