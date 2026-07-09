"""Generate key figures for the benchmark report.

Three plots:
    1. plots/radar_mean_scores.png — mean score per metric per baseline
    2. plots/win_rate_heatmap.png  — CartoPalette win rate vs. ColorBrewer per provider & metric
    3. plots/boxplots_per_metric.png — distribution of scores per baseline per metric

All plots use a colorblind-safe palette and are saved at 300 DPI for publication.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARK_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_CSV = OUTPUT_DIR / "results_raw.csv"

METRICS = [
    "composite",
    "basemap_contrast",
    "lightness_contrast",
    "hue_contrast",
    "distinguishability",
    "cvd_robustness",
    "perceptual_ordering",
]
METRIC_LABELS = {
    "composite": "Composite",
    "basemap_contrast": "Basemap\nContrast",
    "lightness_contrast": "Lightness\nContrast",
    "hue_contrast": "Hue\nContrast",
    "distinguishability": "Distinguish.",
    "cvd_robustness": "CVD\nRobustness",
    "perceptual_ordering": "Perceptual\nOrder",
}
BASELINES = ["CartoPalette", "ColorBrewer", "Matplotlib", "Random"]
# Okabe-Ito colorblind-safe palette
COLORS = {
    "CartoPalette": "#0072B2",
    "ColorBrewer": "#D55E00",
    "Matplotlib": "#009E73",
    "Random":     "#999999",
}


def load_raw():
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def radar_mean_scores(rows):
    """Radar chart: mean score for each metric, one line per baseline."""
    # Compute mean per (baseline, metric)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["baseline"], r["metric"])].append(float(r["score"]))
    means = {k: np.mean(v) for k, v in grouped.items()}

    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for baseline in BASELINES:
        values = [means.get((baseline, m), 0) for m in METRICS]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=COLORS[baseline],
                label=baseline)
        ax.fill(angles, values, color=COLORS[baseline], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=10)
    plt.title("Mean Score per Baseline (averaged over all basemaps, schemes, class counts)",
              fontsize=11, pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "radar_mean_scores.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved radar_mean_scores.png")


def win_rate_heatmap_per_provider(rows):
    """Heatmap: provider × metric, cell = CartoPalette win rate vs. ColorBrewer."""
    # Group paired scores by (provider, metric)
    paired = defaultdict(lambda: {"cp": {}, "cb": {}})
    for r in rows:
        if r["metric"] not in METRICS:
            continue
        key = (r["provider"], r["metric"])
        bm_key = (r["basemap_filename"], r["scheme"], r["n_classes"])
        if r["baseline"] == "CartoPalette":
            paired[key]["cp"][bm_key] = float(r["score"])
        elif r["baseline"] == "ColorBrewer":
            paired[key]["cb"][bm_key] = float(r["score"])

    providers = sorted(set(k[0] for k in paired.keys()))

    heatmap = np.zeros((len(providers), len(METRICS)))
    n_matrix = np.zeros_like(heatmap, dtype=int)
    for i, prov in enumerate(providers):
        for j, metric in enumerate(METRICS):
            d = paired.get((prov, metric), {"cp": {}, "cb": {}})
            common = set(d["cp"]) & set(d["cb"])
            if not common:
                heatmap[i, j] = np.nan
                continue
            cp = np.array([d["cp"][k] for k in common])
            cb = np.array([d["cb"][k] for k in common])
            heatmap[i, j] = (cp > cb).mean()
            n_matrix[i, j] = len(common)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers, fontsize=9)

    # Annotate cells
    for i in range(len(providers)):
        for j in range(len(METRICS)):
            if np.isnan(heatmap[i, j]):
                continue
            val = heatmap[i, j]
            color = "white" if (val < 0.3 or val > 0.7) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("CartoPalette win rate", fontsize=10)
    cbar.ax.axhline(0.5, color="black", linewidth=1)

    plt.title("CartoPalette Win Rate vs. ColorBrewer by Basemap Provider",
              fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "win_rate_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved win_rate_heatmap.png")


def boxplots_per_metric(rows):
    """Box plots: distribution of scores per baseline, one subplot per metric."""
    scores = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["metric"] not in METRICS:
            continue
        scores[r["metric"]][r["baseline"]].append(float(r["score"]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        data = [scores[metric][b] for b in BASELINES]
        bp = ax.boxplot(data, labels=BASELINES, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
        for patch, baseline in zip(bp["boxes"], BASELINES):
            patch.set_facecolor(COLORS[baseline])
            patch.set_alpha(0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(METRIC_LABELS[metric].replace("\n", " "), fontsize=11)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplot
    axes[-1].axis("off")
    # Legend in the empty subplot
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[b], alpha=0.7,
                             label=b) for b in BASELINES]
    axes[-1].legend(handles=handles, loc="center", fontsize=11)

    plt.suptitle("Score Distributions per Metric (all basemaps, schemes, class counts)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplots_per_metric.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved boxplots_per_metric.png")


def main():
    if not RESULTS_CSV.exists():
        print(f"{RESULTS_CSV} not found. Run run_benchmark.py first.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_raw()
    print(f"Loaded {len(rows)} raw rows. Generating plots...")

    radar_mean_scores(rows)
    win_rate_heatmap_per_provider(rows)
    boxplots_per_metric(rows)

    print(f"\nAll plots saved in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
