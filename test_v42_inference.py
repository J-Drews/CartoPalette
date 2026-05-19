"""CartoPalette v4.2 — Smoke + Ablation Test for the inference pipeline.

Runs the v4.1 model checkpoint through:
    A) v4.1 baseline (MMR, k=20, no repair)
    B) v4.2 — constrained reranking only (k=20)
    C) v4.2 — constrained reranking + adaptive k
    D) v4.2 — constrained reranking + adaptive k + LAB repair (full v4.2)

For each variant, prints the top-3 palettes with score and metric breakdown
and computes mean top-1 and top-3 metrics.

Usage:
    python test_v42_inference.py
    python test_v42_inference.py --basemap path/to/your_basemap.png \
                                 --scheme diverging --n-classes 5

Requires: cartopalette package importable from the working directory and the
trained v4.1 checkpoint at cartopalette/pretrained/cartopalette_v4.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cartopalette.core.inference import CartoPalette


# Default test basemaps — try a variety of difficulties
DEFAULT_TEST_BASEMAPS = [
    # (label, relative path, scheme, n_classes)
    ("light_positron_NYC",  "data/processed/basemaps/carto_positron_loc_001_z10.png", "sequential", 5),
    ("dark_carto_NYC",      "data/processed/basemaps/carto_dark_loc_001_z10.png",     "sequential", 5),
    ("osm_Tokyo",           "data/processed/basemaps/osm_loc_003_z10.png",            "diverging",  5),
    ("satellite_Lagos",     "data/processed/basemaps/esri_imagery_loc_002_z10.png",   "diverging",  5),
]


def _fmt_metric(v):
    if v is None:
        return "  -  "
    return f"{v:.3f}"


def _print_palette_row(idx, pal):
    m = pal.metrics or {}
    print(
        f"     #{idx}  "
        f"comp={_fmt_metric(pal.score)}   "
        f"bm={_fmt_metric(m.get('basemap_contrast'))}   "
        f"L={_fmt_metric(m.get('lightness_contrast'))}   "
        f"H={_fmt_metric(m.get('hue_contrast'))}   "
        f"dist={_fmt_metric(m.get('distinguishability'))}   "
        f"cvd={_fmt_metric(m.get('cvd_robustness'))}   "
        f"ord={_fmt_metric(m.get('perceptual_ordering'))}   "
        f"{' '.join(pal.hex_colors)}"
    )


def run_variant(
    cp: CartoPalette,
    image: str,
    scheme: str,
    n_classes: int,
    label: str,
    *,
    reranker: str,
    k: int | None,
    apply_repair: bool,
) -> list:
    palettes, diag = cp.suggest(
        image, scheme=scheme, n_classes=n_classes, scale="regional",
        n_suggestions=3,
        reranker=reranker,
        k=k,
        apply_repair=apply_repair,
        return_diagnostics=True,
    )
    print(f"  [{label}]  k={diag['k']}  ({diag['k_decision']})  "
          f"reranker={diag['reranker']}  repair={diag.get('lab_repair', {})}")
    for i, p in enumerate(palettes, 1):
        _print_palette_row(i, p)
    return palettes


def collect_means(results_by_variant: dict) -> dict:
    summary = {}
    for variant, runs in results_by_variant.items():
        all_top1 = []
        all_top3 = []
        all_bm_top1 = []
        all_bm_top3 = []
        all_dist_top1 = []
        all_lc_top1 = []
        for palettes in runs:
            if not palettes:
                continue
            top1 = palettes[0]
            all_top1.append(top1.score or 0)
            all_bm_top1.append((top1.metrics or {}).get("basemap_contrast", 0))
            all_dist_top1.append((top1.metrics or {}).get("distinguishability", 0))
            all_lc_top1.append((top1.metrics or {}).get("lightness_contrast", 0))
            top3_scores = [p.score or 0 for p in palettes]
            all_top3.append(np.mean(top3_scores))
            top3_bm = [(p.metrics or {}).get("basemap_contrast", 0) for p in palettes]
            all_bm_top3.append(np.mean(top3_bm))
        summary[variant] = {
            "top1_composite":          float(np.mean(all_top1)) if all_top1 else 0,
            "top3_composite_mean":     float(np.mean(all_top3)) if all_top3 else 0,
            "top1_basemap_contrast":   float(np.mean(all_bm_top1)) if all_bm_top1 else 0,
            "top3_basemap_contrast":   float(np.mean(all_bm_top3)) if all_bm_top3 else 0,
            "top1_distinguishability": float(np.mean(all_dist_top1)) if all_dist_top1 else 0,
            "top1_lightness_contrast": float(np.mean(all_lc_top1)) if all_lc_top1 else 0,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cartopalette/pretrained/cartopalette_v4.pt")
    ap.add_argument("--basemap", default=None,
                    help="Run only on a single basemap path (overrides defaults).")
    ap.add_argument("--scheme", default="sequential", choices=["sequential", "diverging"])
    ap.add_argument("--n-classes", type=int, default=5, choices=[3, 4, 5, 7, 9])
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model checkpoint not found at {model_path}")
        sys.exit(1)

    print(f"Loading {model_path}...")
    cp = CartoPalette(str(model_path))
    print(f"  {cp}")

    if args.basemap:
        test_set = [("user_basemap", args.basemap, args.scheme, args.n_classes)]
    else:
        test_set = [
            (lab, p, sc, n) for (lab, p, sc, n) in DEFAULT_TEST_BASEMAPS
            if Path(p).exists()
        ]
        if not test_set:
            print("No default basemaps found in data/processed/basemaps/.")
            print("Pass --basemap path/to/your_basemap.png to test on a custom file.")
            sys.exit(1)

    results = {"A_v41_mmr": [], "B_constr_only": [], "C_constr_adapk": [], "D_full_v42": []}

    for label, bm_path, scheme, n_classes in test_set:
        if not Path(bm_path).exists():
            print(f"  -- skip {label}: {bm_path} not found --")
            continue
        print(f"\n=== Basemap: {label} ({scheme}/{n_classes}) ===")

        # A) v4.1 baseline
        results["A_v41_mmr"].append(
            run_variant(cp, bm_path, scheme, n_classes, "A v4.1 (MMR, k=20, no repair)",
                        reranker="mmr", k=20, apply_repair=False))
        # B) constrained reranking only
        results["B_constr_only"].append(
            run_variant(cp, bm_path, scheme, n_classes, "B constrained only (k=20)",
                        reranker="constrained", k=20, apply_repair=False))
        # C) + adaptive k
        results["C_constr_adapk"].append(
            run_variant(cp, bm_path, scheme, n_classes, "C constrained + adaptive k",
                        reranker="constrained", k=None, apply_repair=False))
        # D) + repair (= full v4.2 default)
        results["D_full_v42"].append(
            run_variant(cp, bm_path, scheme, n_classes, "D full v4.2 (constrained + adapK + repair)",
                        reranker="constrained", k=None, apply_repair=True))

    # ── Summary ──
    print("\n" + "=" * 110)
    print("SUMMARY — mean over all tested basemaps")
    print("=" * 110)
    summary = collect_means(results)
    header = f"{'Variant':<22}  {'comp@1':>8}  {'comp@3':>8}  {'bm@1':>8}  {'bm@3':>8}  {'L@1':>8}  {'dist@1':>8}"
    print(header)
    print("-" * len(header))
    order = [
        ("A_v41_mmr",       "A — v4.1 (MMR)"),
        ("B_constr_only",   "B — +constrained"),
        ("C_constr_adapk",  "C — +adaptive k"),
        ("D_full_v42",      "D — +LAB repair"),
    ]
    for key, lbl in order:
        s = summary[key]
        print(f"{lbl:<22}  "
              f"{s['top1_composite']:>8.3f}  "
              f"{s['top3_composite_mean']:>8.3f}  "
              f"{s['top1_basemap_contrast']:>8.3f}  "
              f"{s['top3_basemap_contrast']:>8.3f}  "
              f"{s['top1_lightness_contrast']:>8.3f}  "
              f"{s['top1_distinguishability']:>8.3f}")

    # Delta vs baseline
    baseline = summary["A_v41_mmr"]
    print()
    print(f"{'Delta vs A':<22}  {'comp@1':>8}  {'comp@3':>8}  {'bm@1':>8}  {'bm@3':>8}  {'L@1':>8}  {'dist@1':>8}")
    print("-" * len(header))
    for key, lbl in order[1:]:
        s = summary[key]
        print(f"{lbl:<22}  "
              f"{s['top1_composite']-baseline['top1_composite']:>+8.3f}  "
              f"{s['top3_composite_mean']-baseline['top3_composite_mean']:>+8.3f}  "
              f"{s['top1_basemap_contrast']-baseline['top1_basemap_contrast']:>+8.3f}  "
              f"{s['top3_basemap_contrast']-baseline['top3_basemap_contrast']:>+8.3f}  "
              f"{s['top1_lightness_contrast']-baseline['top1_lightness_contrast']:>+8.3f}  "
              f"{s['top1_distinguishability']-baseline['top1_distinguishability']:>+8.3f}")

    print()
    print("Reading: positive deltas in the 'bm' columns mean the v4.2 changes lift")
    print("basemap_contrast above the v4.1 baseline. The dist@1 column should stay")
    print("within ~-0.04 of A (the epsilon-Pareto budget). Any larger drop indicates")
    print("that the epsilon constraints need tightening.")


if __name__ == "__main__":
    main()
