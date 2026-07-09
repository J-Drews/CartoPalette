"""Statistical analysis of benchmark results.

Given results_raw.csv, computes:
    1. Aggregate statistics per (baseline, metric) — mean, median, std.
    2. Pairwise win rates: on how many basemaps does CartoPalette win vs. each baseline?
    3. Wilcoxon signed-rank tests (paired, non-parametric) with p-values.
    4. Effect sizes (rank-biserial correlation r, Cohen's d).
    5. Breakdown by scheme and class count.

Outputs:
    output/results_summary.md        — aggregate tables
    output/win_rates.csv             — one row per (baseline_pair, scheme, n_classes, metric)
    output/statistical_tests.csv     — p-values and effect sizes
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARK_DIR / "output"
RESULTS_CSV = OUTPUT_DIR / "results_raw.csv"
SUMMARY_MD = OUTPUT_DIR / "results_summary.md"
WIN_RATES_CSV = OUTPUT_DIR / "win_rates.csv"
STAT_TESTS_CSV = OUTPUT_DIR / "statistical_tests.csv"

METRICS = [
    "composite",
    "basemap_contrast",
    "lightness_contrast",
    "hue_contrast",
    "distinguishability",
    "cvd_robustness",
    "perceptual_ordering",
]
BASELINES = ["CartoPalette", "ColorBrewer", "Matplotlib", "Random"]


def load_raw() -> list:
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_by(rows, keys):
    """Group rows by tuple of key values."""
    out = defaultdict(list)
    for r in rows:
        k = tuple(r[k] for k in keys)
        out[k].append(r)
    return out


def aggregate_summary(rows: list) -> list:
    """Mean, median, std of score per (baseline, scheme, n_classes, metric)."""
    grouped = index_by(rows, ["baseline", "scheme", "n_classes", "metric"])
    out_rows = []
    for (baseline, scheme, n_classes, metric), group in grouped.items():
        scores = np.array([float(r["score"]) for r in group])
        out_rows.append({
            "baseline": baseline,
            "scheme": scheme,
            "n_classes": int(n_classes),
            "metric": metric,
            "n_basemaps": len(scores),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "min": float(scores.min()),
            "max": float(scores.max()),
        })
    return out_rows


def paired_scores(rows: list, baseline_a: str, baseline_b: str,
                  scheme: str, n_classes: int, metric: str) -> tuple:
    """Extract paired (A, B) scores over matching basemaps."""
    # Index rows by (basemap, baseline)
    a_by_bm = {}
    b_by_bm = {}
    for r in rows:
        if r["scheme"] != scheme or int(r["n_classes"]) != n_classes:
            continue
        if r["metric"] != metric:
            continue
        if r["baseline"] == baseline_a:
            a_by_bm[r["basemap_filename"]] = float(r["score"])
        elif r["baseline"] == baseline_b:
            b_by_bm[r["basemap_filename"]] = float(r["score"])

    common = sorted(set(a_by_bm.keys()) & set(b_by_bm.keys()))
    a_arr = np.array([a_by_bm[bm] for bm in common])
    b_arr = np.array([b_by_bm[bm] for bm in common])
    return a_arr, b_arr


def compute_win_rates_and_tests(rows: list) -> tuple:
    """For every (baseline_B, scheme, n_classes, metric), compute
    how often CartoPalette beats baseline_B plus Wilcoxon p and effect size.
    """
    win_rate_rows = []
    stat_rows = []

    other_baselines = [b for b in BASELINES if b != "CartoPalette"]

    # Also compute "overall" (aggregated across all schemes/class counts)
    # in addition to per-scheme, per-class-count breakdowns.

    scheme_values = sorted(set(r["scheme"] for r in rows))
    ncls_values = sorted(set(int(r["n_classes"]) for r in rows))

    for other in other_baselines:
        for metric in METRICS:
            # Per (scheme, n_classes) breakdown
            for scheme in scheme_values:
                for n_classes in ncls_values:
                    a, b = paired_scores(rows, "CartoPalette", other,
                                         scheme, n_classes, metric)
                    if len(a) < 2:
                        continue
                    wins = int(np.sum(a > b))
                    ties = int(np.sum(a == b))
                    losses = int(np.sum(a < b))
                    n = len(a)

                    win_rate_rows.append({
                        "comparison": f"CartoPalette_vs_{other}",
                        "scheme": scheme,
                        "n_classes": n_classes,
                        "metric": metric,
                        "n": n,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "win_rate": wins / n,
                        "mean_delta": float((a - b).mean()),
                        "median_delta": float(np.median(a - b)),
                    })

                    # Wilcoxon + effect sizes
                    try:
                        stat, p = stats.wilcoxon(a, b, zero_method="wilcox",
                                                 alternative="two-sided")
                    except ValueError:
                        stat, p = np.nan, np.nan

                    # Rank-biserial correlation for Wilcoxon
                    # r = (wins - losses) / n (simplified)
                    r_effect = (wins - losses) / n if n > 0 else 0.0

                    # Cohen's d for paired samples
                    diff = a - b
                    d_cohen = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

                    stat_rows.append({
                        "comparison": f"CartoPalette_vs_{other}",
                        "scheme": scheme,
                        "n_classes": n_classes,
                        "metric": metric,
                        "n": n,
                        "wilcoxon_stat": float(stat) if stat == stat else None,
                        "wilcoxon_p": float(p) if p == p else None,
                        "significant_005": bool(p < 0.05) if p == p else None,
                        "rank_biserial_r": float(r_effect),
                        "cohen_d": float(d_cohen),
                    })

            # Aggregated across ALL schemes and n_classes
            a_all, b_all = paired_scores_any(rows, "CartoPalette", other, metric)
            if len(a_all) >= 2:
                wins = int(np.sum(a_all > b_all))
                ties = int(np.sum(a_all == b_all))
                losses = int(np.sum(a_all < b_all))
                n = len(a_all)
                win_rate_rows.append({
                    "comparison": f"CartoPalette_vs_{other}",
                    "scheme": "ALL",
                    "n_classes": "ALL",
                    "metric": metric,
                    "n": n,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / n,
                    "mean_delta": float((a_all - b_all).mean()),
                    "median_delta": float(np.median(a_all - b_all)),
                })
                try:
                    stat, p = stats.wilcoxon(a_all, b_all, zero_method="wilcox",
                                             alternative="two-sided")
                except ValueError:
                    stat, p = np.nan, np.nan
                r_effect = (wins - losses) / n if n > 0 else 0.0
                diff = a_all - b_all
                d_cohen = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

                stat_rows.append({
                    "comparison": f"CartoPalette_vs_{other}",
                    "scheme": "ALL",
                    "n_classes": "ALL",
                    "metric": metric,
                    "n": n,
                    "wilcoxon_stat": float(stat) if stat == stat else None,
                    "wilcoxon_p": float(p) if p == p else None,
                    "significant_005": bool(p < 0.05) if p == p else None,
                    "rank_biserial_r": float(r_effect),
                    "cohen_d": float(d_cohen),
                })

    return win_rate_rows, stat_rows


def paired_scores_any(rows: list, baseline_a: str, baseline_b: str,
                      metric: str) -> tuple:
    """Paired scores aggregated over all (basemap, scheme, n_classes) combos."""
    a_by_key = {}
    b_by_key = {}
    for r in rows:
        if r["metric"] != metric:
            continue
        key = (r["basemap_filename"], r["scheme"], r["n_classes"])
        if r["baseline"] == baseline_a:
            a_by_key[key] = float(r["score"])
        elif r["baseline"] == baseline_b:
            b_by_key[key] = float(r["score"])
    common = sorted(set(a_by_key.keys()) & set(b_by_key.keys()))
    a_arr = np.array([a_by_key[k] for k in common])
    b_arr = np.array([b_by_key[k] for k in common])
    return a_arr, b_arr


def write_summary_markdown(summary_rows: list):
    """Write a readable aggregate table as markdown."""
    # Group by metric for display
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# CartoPalette Benchmark — Aggregate Results\n\n")
        f.write("Mean score per baseline, metric, scheme, and n_classes.\n")
        f.write("Higher is better for all metrics (all ∈ [0, 1]).\n\n")

        for metric in METRICS:
            f.write(f"## {metric}\n\n")
            for scheme in ["sequential", "diverging"]:
                f.write(f"### {scheme.capitalize()}\n\n")
                f.write("| n_classes | CartoPalette | ColorBrewer | Matplotlib | Random |\n")
                f.write("|-----------|--------------|-------------|------------|--------|\n")
                for n in [3, 5, 7, 9]:
                    row = {b: "—" for b in BASELINES}
                    for s in summary_rows:
                        if (s["scheme"] == scheme and s["n_classes"] == n
                                and s["metric"] == metric):
                            row[s["baseline"]] = f"{s['mean']:.3f}"
                    f.write(f"| {n} | {row['CartoPalette']} | {row['ColorBrewer']} | "
                            f"{row['Matplotlib']} | {row['Random']} |\n")
                f.write("\n")


def write_csv(rows: list, path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not RESULTS_CSV.exists():
        print(f"{RESULTS_CSV} not found. Run run_benchmark.py first.")
        return

    rows = load_raw()
    print(f"Loaded {len(rows)} raw result rows.")

    # Summary
    summary_rows = aggregate_summary(rows)
    write_summary_markdown(summary_rows)
    print(f"Wrote {SUMMARY_MD}")

    # Win rates + statistical tests
    win_rows, stat_rows = compute_win_rates_and_tests(rows)
    write_csv(win_rows, WIN_RATES_CSV)
    write_csv(stat_rows, STAT_TESTS_CSV)
    print(f"Wrote {WIN_RATES_CSV} ({len(win_rows)} rows)")
    print(f"Wrote {STAT_TESTS_CSV} ({len(stat_rows)} rows)")

    # Print headline numbers
    print("\n=== Headline Results (ALL schemes, ALL class counts) ===")
    for r in sorted(stat_rows, key=lambda x: (x["metric"], x["comparison"])):
        if r["scheme"] == "ALL":
            sig = "***" if r["significant_005"] else "   "
            wr = next(w for w in win_rows
                      if w["comparison"] == r["comparison"] and w["scheme"] == "ALL"
                      and w["metric"] == r["metric"])
            print(f"  {r['comparison']:30s}  {r['metric']:22s}  "
                  f"win_rate={wr['win_rate']:.2f}  p={r['wilcoxon_p']:.4g}  "
                  f"r={r['rank_biserial_r']:+.2f}  d={r['cohen_d']:+.2f}  {sig}")


if __name__ == "__main__":
    main()
