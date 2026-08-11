"""Run CartoPalette v4.2 ablations for reviewer response.

This benchmark isolates the contribution of the learned CVAE generator from
candidate-pool size and downstream selection rules. It uses the same held-out
test set and scoring functions as run_benchmark.py.

Outputs are written to research/benchmark/output/:
    ablation_results_raw.csv
    ablation_summary.csv
    ablation_stepwise_deltas.csv
    ablation_budget_summary.csv
    ablation_summary.md

Examples:
    python research/benchmark/run_ablation.py --limit 2 --device cpu
    python research/benchmark/run_ablation.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from cartopalette.core.inference import (  # noqa: E402
    DEFAULT_PIPELINE_CONFIG,
    CartoPalette,
    Palette,
)
from cartopalette.core.model import pack_dominant_lab  # noqa: E402
from cartopalette.core.repair import repair_palette_list  # noqa: E402
from cartopalette.scoring.composite import (  # noqa: E402
    extract_basemap_colors,
    score_palette_detailed,
)
from baselines import (  # noqa: E402
    get_colorbrewer_palettes,
    get_matplotlib_palettes,
    get_random_palettes,
)


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARK_DIR / "output"
TEST_SET_JSON = OUTPUT_DIR / "test_set.json"
BASEMAPS_DIR = ROOT / "data" / "processed" / "basemaps"
MODEL_PATH = ROOT / "cartopalette" / "pretrained" / "cartopalette_v4.pt"

RESULTS_CSV = OUTPUT_DIR / "ablation_results_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "ablation_summary.csv"
DELTAS_CSV = OUTPUT_DIR / "ablation_stepwise_deltas.csv"
BUDGET_CSV = OUTPUT_DIR / "ablation_budget_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "ablation_summary.md"

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

VARIANT_DESCRIPTIONS = {
    "Raw_CVAE_first": "first CVAE sample only; no score-based selection",
    "CVAE_score_best": "best composite among CVAE candidates; no guardrails",
    "CVAE_score_guardrails": "CVAE candidates, guardrails, then best composite",
    "CVAE_guardrails_rerank": "CVAE candidates, guardrails, constrained reranking",
    "CVAE_guardrails_rerank_repair": "previous CVAE-only variant plus LAB repair",
    "NonLearning_budget_score": "deterministic basemap-adaptive candidates, same k as CVAE, best composite",
    "NonLearning_budget_full": "deterministic basemap-adaptive candidates, same k as CVAE, full selector plus repair",
    "Hybrid_score_best": "CVAE plus production deterministic pool, best composite only",
    "Hybrid_guardrails_rerank": "CVAE plus production deterministic pool, guardrails and constrained reranking",
    "Full_CartoPalette_v4_2": "production v4.2 pipeline: hybrid pool, guardrails, reranking, LAB repair",
    "ColorBrewer_best_of_library": "best ColorBrewer palette by composite for each case",
    "Matplotlib_best_of_library": "best Matplotlib palette by composite for each case",
    "Random_median_of_10": "median composite palette among 10 random LAB palettes",
}

VARIANT_ORDER = [
    "Raw_CVAE_first",
    "CVAE_score_best",
    "CVAE_score_guardrails",
    "CVAE_guardrails_rerank",
    "CVAE_guardrails_rerank_repair",
    "NonLearning_budget_score",
    "NonLearning_budget_full",
    "Hybrid_score_best",
    "Hybrid_guardrails_rerank",
    "Full_CartoPalette_v4_2",
    "ColorBrewer_best_of_library",
    "Matplotlib_best_of_library",
    "Random_median_of_10",
]

STEPWISE_COMPARISONS = [
    (
        "Score selection gain within CVAE",
        "Raw_CVAE_first",
        "CVAE_score_best",
    ),
    (
        "Guardrail effect within CVAE",
        "CVAE_score_best",
        "CVAE_score_guardrails",
    ),
    (
        "Constrained reranking effect within CVAE",
        "CVAE_score_guardrails",
        "CVAE_guardrails_rerank",
    ),
    (
        "LAB repair effect within CVAE",
        "CVAE_guardrails_rerank",
        "CVAE_guardrails_rerank_repair",
    ),
    (
        "Learned generator vs budget-matched non-learning full selector",
        "NonLearning_budget_full",
        "CVAE_guardrails_rerank_repair",
    ),
    (
        "Production deterministic pool effect before repair",
        "CVAE_guardrails_rerank",
        "Hybrid_guardrails_rerank",
    ),
    (
        "LAB repair effect in full hybrid pipeline",
        "Hybrid_guardrails_rerank",
        "Full_CartoPalette_v4_2",
    ),
    (
        "Full pipeline vs ColorBrewer best-of-library",
        "ColorBrewer_best_of_library",
        "Full_CartoPalette_v4_2",
    ),
    (
        "Full pipeline vs Matplotlib best-of-library",
        "Matplotlib_best_of_library",
        "Full_CartoPalette_v4_2",
    ),
]


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def score_lab(lab: np.ndarray, basemap_lab: np.ndarray, scheme: str) -> dict:
    details = score_palette_detailed(lab, basemap_lab, scheme)
    return {m: float(details[m]) for m in METRICS}


def scored_palette(
    lab: np.ndarray,
    basemap_lab: np.ndarray,
    scheme: str,
    n_classes: int,
    source: str,
) -> Palette:
    metrics = score_lab(lab, basemap_lab, scheme)
    pal = Palette(
        lab=np.asarray(lab, dtype=np.float64),
        scheme_type=scheme,
        n_classes=n_classes,
        score=metrics["composite"],
        metrics=metrics,
    )
    pal._source = source
    return pal


def scored_palettes(
    labs: Iterable[np.ndarray],
    basemap_lab: np.ndarray,
    scheme: str,
    n_classes: int,
    source: str,
) -> list[Palette]:
    return [
        scored_palette(lab, basemap_lab, scheme, n_classes, source)
        for lab in labs
    ]


def clone_palette(pal: Palette) -> Palette:
    out = Palette(
        lab=pal.lab.copy(),
        scheme_type=pal.scheme_type,
        n_classes=pal.n_classes,
        score=pal.score,
        metrics=dict(pal.metrics or {}),
    )
    out._source = getattr(pal, "_source", "")
    return out


def best_by_composite(palettes: list[Palette]) -> Palette | None:
    if not palettes:
        return None
    return clone_palette(max(palettes, key=lambda p: p.score if p.score is not None else -1.0))


def guardrails(cp: CartoPalette, palettes: list[Palette], basemap_lab: np.ndarray, scheme: str) -> list[Palette]:
    if not palettes:
        return []
    difficulty = cp._basemap_difficulty(
        basemap_lab,
        cp.pipeline_config.get("adaptive_k", DEFAULT_PIPELINE_CONFIG["adaptive_k"]),
    )
    basemap_mean_l = difficulty["median_L"]
    return cp._apply_guardrails(palettes, scheme, basemap_mean_l, basemap_lab)


def constrained_top1(
    cp: CartoPalette,
    palettes: list[Palette],
    n_classes: int,
) -> Palette | None:
    if not palettes:
        return None
    epsilon = cp.pipeline_config.get(
        "epsilon_constraints",
        DEFAULT_PIPELINE_CONFIG["epsilon_constraints"],
    )
    beta = cp.pipeline_config.get(
        "beta_basemap",
        DEFAULT_PIPELINE_CONFIG["beta_basemap"],
    )
    selected = cp._constrained_rerank(
        palettes,
        n=1,
        beta_basemap=beta,
        epsilon=epsilon,
        quality_floors=cp._make_quality_floors(n_classes),
    )
    return clone_palette(selected[0]) if selected else None


def repair_top1(cp: CartoPalette, pal: Palette | None, basemap_lab: np.ndarray) -> Palette | None:
    if pal is None:
        return None
    source = getattr(pal, "_source", "")
    repair_cfg = cp.pipeline_config.get(
        "lab_repair",
        DEFAULT_PIPELINE_CONFIG["lab_repair"],
    )
    rcfg = {k: v for k, v in repair_cfg.items() if k != "enabled"}
    repaired = repair_palette_list([pal], basemap_lab, config=rcfg)[0]
    repaired._source = source
    return repaired


@torch.no_grad()
def generate_cvae_candidates(
    cp: CartoPalette,
    image: Image.Image,
    basemap_lab: np.ndarray,
    scheme: str,
    n_classes: int,
    scale: str,
    k: int,
    seed: int,
) -> list[np.ndarray]:
    set_seed(seed, cp.device)

    img_tensor = cp._load_image(image)
    metadata = cp._build_metadata(scheme, n_classes, scale).unsqueeze(0).to(cp.device)

    cnn_features = cp.model.cnn(img_tensor)
    dominant_block = None
    if getattr(cp.model, "dominant_colors_dim", 0) > 0:
        target_n = cp.model.dominant_colors_dim // 3
        dominant_block = pack_dominant_lab(basemap_lab, target_n=target_n)
        dominant_block = dominant_block.unsqueeze(0).to(cp.device)

    condition = cp.model.build_condition(cnn_features, metadata, dominant_block)
    condition = condition.repeat(k, 1)
    z = torch.randn(k, cp.model.latent_dim, device=cp.device)
    raw = cp.model.decoder(condition, z)
    raw = raw.view(k, 9, 3).cpu().numpy()
    return [cp._denormalize(raw[i], n_classes) for i in range(k)]


def best_of_library(
    palettes: list[tuple[str, np.ndarray]],
    basemap_lab: np.ndarray,
    scheme: str,
) -> tuple[str, dict, np.ndarray]:
    best_name = ""
    best_metrics = None
    best_lab = None
    best_score = -1.0
    for name, lab in palettes:
        metrics = score_lab(lab, basemap_lab, scheme)
        if metrics["composite"] > best_score:
            best_name = name
            best_metrics = metrics
            best_lab = lab
            best_score = metrics["composite"]
    return best_name, best_metrics, best_lab


def median_of_random(
    palettes: list[tuple[str, np.ndarray]],
    basemap_lab: np.ndarray,
    scheme: str,
) -> tuple[str, dict, np.ndarray]:
    scored = []
    for name, lab in palettes:
        metrics = score_lab(lab, basemap_lab, scheme)
        scored.append((name, lab, metrics))
    scored.sort(key=lambda item: item[2]["composite"])
    name, lab, metrics = scored[len(scored) // 2]
    return name, metrics, lab


def classify_basemap_brightness(basemap_lab: np.ndarray) -> str:
    mean_l = float(basemap_lab[:, 0].mean())
    if mean_l < 45:
        return "dark"
    if mean_l < 70:
        return "medium"
    return "light"


def classify_basemap_chroma(basemap_lab: np.ndarray) -> str:
    a = basemap_lab[:, 1]
    b = basemap_lab[:, 2]
    mean_c = float(np.mean(np.sqrt(a**2 + b**2)))
    return "neutral" if mean_c < 10 else "chromatic"


def palette_name(pal: Palette | None, fallback: str = "none") -> str:
    if pal is None:
        return fallback
    return getattr(pal, "_source", fallback) or fallback


def add_variant_rows(
    rows: list[dict],
    case_meta: dict,
    variant: str,
    metrics: dict | None,
    palette_label: str,
    selected_source: str,
    candidate_pool: str,
    candidate_budget: int,
    n_after_guardrails: int | None,
) -> None:
    if metrics is None:
        return
    for metric in METRICS:
        row = dict(case_meta)
        row.update(
            {
                "variant": variant,
                "variant_description": VARIANT_DESCRIPTIONS[variant],
                "palette_label": palette_label,
                "selected_source": selected_source,
                "candidate_pool": candidate_pool,
                "candidate_budget": candidate_budget,
                "n_after_guardrails": "" if n_after_guardrails is None else n_after_guardrails,
                "metric": metric,
                "score": metrics[metric],
            }
        )
        rows.append(row)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["variant"],
            row["scheme"],
            row["n_classes"],
            row["metric"],
        )
        grouped[key].append(float(row["score"]))

        all_key = (row["variant"], "ALL", "ALL", row["metric"])
        grouped[all_key].append(float(row["score"]))

    out = []
    for (variant, scheme, n_classes, metric), values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        out.append(
            {
                "variant": variant,
                "variant_description": VARIANT_DESCRIPTIONS.get(variant, ""),
                "scheme": scheme,
                "n_classes": n_classes,
                "metric": metric,
                "n_cases": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    order = {name: i for i, name in enumerate(VARIANT_ORDER)}
    metric_order = {name: i for i, name in enumerate(METRICS)}
    out.sort(
        key=lambda r: (
            order.get(r["variant"], 999),
            str(r["scheme"]),
            str(r["n_classes"]),
            metric_order.get(r["metric"], 999),
        )
    )
    return out


def paired_values(rows: list[dict], variant_a: str, variant_b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    a_by_key = {}
    b_by_key = {}
    for row in rows:
        if row["metric"] != metric:
            continue
        key = (
            row["basemap_filename"],
            row["scheme"],
            row["n_classes"],
        )
        if row["variant"] == variant_a:
            a_by_key[key] = float(row["score"])
        elif row["variant"] == variant_b:
            b_by_key[key] = float(row["score"])
    common = sorted(set(a_by_key) & set(b_by_key))
    return (
        np.array([a_by_key[k] for k in common], dtype=np.float64),
        np.array([b_by_key[k] for k in common], dtype=np.float64),
    )


def stepwise_deltas(rows: list[dict]) -> list[dict]:
    out = []
    for label, before, after in STEPWISE_COMPARISONS:
        for metric in METRICS:
            a, b = paired_values(rows, before, after, metric)
            if len(a) == 0:
                continue
            diff = b - a
            out.append(
                {
                    "comparison": label,
                    "before_variant": before,
                    "after_variant": after,
                    "metric": metric,
                    "n": len(diff),
                    "mean_before": float(a.mean()),
                    "mean_after": float(b.mean()),
                    "mean_delta": float(diff.mean()),
                    "median_delta": float(np.median(diff)),
                    "wins_after": int(np.sum(b > a)),
                    "ties": int(np.sum(b == a)),
                    "losses_after": int(np.sum(b < a)),
                    "win_rate_after": float(np.sum(b > a) / len(diff)),
                }
            )
    return out


def budget_summary(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        key = (
            row["variant"],
            row["basemap_filename"],
            row["scheme"],
            row["n_classes"],
        )
        if key in seen:
            continue
        seen[key] = row

    grouped = defaultdict(list)
    source_counts = defaultdict(Counter)
    hard_counts = defaultdict(Counter)
    for row in seen.values():
        grouped[row["variant"]].append(row)
        source_counts[row["variant"]][row["selected_source"]] += 1
        hard_counts[row["variant"]][row["k_decision"]] += 1

    out = []
    for variant, group in grouped.items():
        budgets = np.array([int(g["candidate_budget"]) for g in group], dtype=np.float64)
        after = [
            int(g["n_after_guardrails"])
            for g in group
            if str(g["n_after_guardrails"]) != ""
        ]
        after_arr = np.array(after, dtype=np.float64) if after else np.array([])
        out.append(
            {
                "variant": variant,
                "n_cases": len(group),
                "mean_candidate_budget": float(budgets.mean()),
                "median_candidate_budget": float(np.median(budgets)),
                "mean_after_guardrails": float(after_arr.mean()) if len(after_arr) else "",
                "selected_source_counts": json.dumps(dict(source_counts[variant]), sort_keys=True),
                "k_decision_counts": json.dumps(dict(hard_counts[variant]), sort_keys=True),
            }
        )
    order = {name: i for i, name in enumerate(VARIANT_ORDER)}
    out.sort(key=lambda row: order.get(row["variant"], 999))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_summary_md(summary_rows: list[dict], delta_rows: list[dict], budget_rows: list[dict]) -> None:
    overall = [
        r for r in summary_rows
        if r["scheme"] == "ALL" and r["n_classes"] == "ALL"
    ]
    summary_lookup = {
        (r["variant"], r["metric"]): r
        for r in overall
    }
    delta_lookup = {
        (r["comparison"], r["metric"]): r
        for r in delta_rows
    }
    budget_lookup = {r["variant"]: r for r in budget_rows}

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# CartoPalette v4.2 Ablation Benchmark\n\n")
        n_overall_cases = 0
        if ("Full_CartoPalette_v4_2", "composite") in summary_lookup:
            n_overall_cases = summary_lookup[("Full_CartoPalette_v4_2", "composite")]["n_cases"]
        f.write(
            "Higher is better for every metric. Overall means aggregate "
            f"{n_overall_cases} paired benchmark cases.\n\n"
        )

        f.write("## Variants\n\n")
        for variant in VARIANT_ORDER:
            if (variant, "composite") in summary_lookup:
                f.write(f"- **{variant}**: {VARIANT_DESCRIPTIONS[variant]}\n")
        f.write("\n")

        f.write("## Overall Mean Scores\n\n")
        f.write("| Variant | Composite | Basemap contrast | Lightness contrast | Hue contrast | Distinguishability | CVD robustness | Perceptual ordering |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for variant in VARIANT_ORDER:
            if (variant, "composite") not in summary_lookup:
                continue
            values = [fmt(summary_lookup[(variant, m)]["mean"]) for m in METRICS]
            f.write(f"| {variant} | " + " | ".join(values) + " |\n")
        f.write("\n")

        f.write("## Stepwise Mean Deltas\n\n")
        f.write("Positive values mean the later variant improved over the earlier variant.\n\n")
        f.write("| Comparison | Composite | Basemap contrast | Lightness contrast | Hue contrast | Distinguishability | CVD robustness | Perceptual ordering |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, _before, _after in STEPWISE_COMPARISONS:
            if (label, "composite") not in delta_lookup:
                continue
            values = [fmt(delta_lookup[(label, m)]["mean_delta"]) for m in METRICS]
            f.write(f"| {label} | " + " | ".join(values) + " |\n")
        f.write("\n")

        f.write("## Stepwise Win Rates\n\n")
        f.write("Win rate is the proportion of paired cases in which the later variant is higher.\n\n")
        f.write("| Comparison | Composite | Basemap contrast | Lightness contrast | Hue contrast | Distinguishability | CVD robustness | Perceptual ordering |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, _before, _after in STEPWISE_COMPARISONS:
            if (label, "composite") not in delta_lookup:
                continue
            values = [fmt(delta_lookup[(label, m)]["win_rate_after"]) for m in METRICS]
            f.write(f"| {label} | " + " | ".join(values) + " |\n")
        f.write("\n")

        f.write("## Candidate Budgets\n\n")
        f.write("| Variant | Mean budget | Median budget | Mean after guardrails | Selected source counts | k decision counts |\n")
        f.write("|---|---:|---:|---:|---|---|\n")
        for variant in VARIANT_ORDER:
            row = budget_lookup.get(variant)
            if not row:
                continue
            f.write(
                f"| {variant} | {fmt(row['mean_candidate_budget'], 1)} | "
                f"{fmt(row['median_candidate_budget'], 1)} | "
                f"{fmt(row['mean_after_guardrails'], 1)} | "
                f"`{row['selected_source_counts']}` | `{row['k_decision_counts']}` |\n"
            )


def run(args: argparse.Namespace) -> None:
    if not TEST_SET_JSON.exists():
        raise FileNotFoundError(f"Missing test set: {TEST_SET_JSON}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    with open(TEST_SET_JSON, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    basemaps = test_set["basemaps"]
    if args.limit is not None:
        basemaps = basemaps[: args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    cp = CartoPalette(str(MODEL_PATH), device=args.device)
    print(f"Model loaded on {cp.device}")
    print(f"Running {len(basemaps)} basemaps x {len(SCHEMES)} schemes x {len(N_CLASSES_LIST)} class counts")

    rows: list[dict] = []
    total = len(basemaps) * len(SCHEMES) * len(N_CLASSES_LIST)
    combo_idx = 0
    t_start = time.time()

    for bm in basemaps:
        basemap_path = BASEMAPS_DIR / bm["basemap_filename"]
        image = Image.open(basemap_path).convert("RGB")
        basemap_lab = extract_basemap_colors(str(basemap_path))

        brightness = classify_basemap_brightness(basemap_lab)
        chroma = classify_basemap_chroma(basemap_lab)
        difficulty = cp._basemap_difficulty(
            basemap_lab,
            cp.pipeline_config.get("adaptive_k", DEFAULT_PIPELINE_CONFIG["adaptive_k"]),
        )

        for scheme in SCHEMES:
            for n_classes in N_CLASSES_LIST:
                combo_idx += 1
                elapsed = time.time() - t_start
                if combo_idx == 1 or combo_idx % args.log_every == 0:
                    print(
                        f"[{combo_idx}/{total}] {bm['basemap_filename']} "
                        f"{scheme}/{n_classes} classes, elapsed {elapsed/60:.1f} min"
                    )

                scale = bm["zoom_range"] if args.scale_mode == "zoom_range" else args.scale
                k_eff, difficulty_for_k = cp._resolve_k(
                    basemap_lab,
                    args.fixed_k,
                )
                k_decision = (
                    "fixed"
                    if args.fixed_k is not None
                    else "adaptive_hard"
                    if difficulty_for_k["is_hard"]
                    else "adaptive_easy"
                )
                full_cart_budget = int(
                    cp.pipeline_config.get("n_cartographic_candidates", 320)
                )
                seed = args.seed + combo_idx

                case_meta = {
                    "basemap_filename": bm["basemap_filename"],
                    "provider": bm["provider"],
                    "location_id": bm["location_id"],
                    "location_name": bm["location_name"],
                    "zoom": bm["zoom"],
                    "zoom_range": bm["zoom_range"],
                    "scale_used": scale,
                    "region_type": bm["land_cover"],
                    "climate_zone": bm["climate_zone"],
                    "basemap_class": f"{brightness}_{chroma}",
                    "basemap_brightness": brightness,
                    "basemap_chroma": chroma,
                    "basemap_median_L": difficulty["median_L"],
                    "basemap_mean_chroma": difficulty["mean_chroma"],
                    "basemap_L_std": difficulty["L_std"],
                    "scheme": scheme,
                    "n_classes": n_classes,
                    "k_cvae": k_eff,
                    "k_cartographic_budget_matched": k_eff,
                    "k_cartographic_full": full_cart_budget,
                    "k_decision": k_decision,
                    "seed": seed,
                }

                cvae_labs = generate_cvae_candidates(
                    cp,
                    image,
                    basemap_lab,
                    scheme,
                    n_classes,
                    scale,
                    k_eff,
                    seed,
                )
                cvae = scored_palettes(cvae_labs, basemap_lab, scheme, n_classes, "cvae")

                nonlearning_full_labs = cp._generate_cartographic_labs(
                    basemap_lab,
                    scheme,
                    n_classes,
                    max_candidates=full_cart_budget,
                )
                nonlearning_full = scored_palettes(
                    nonlearning_full_labs,
                    basemap_lab,
                    scheme,
                    n_classes,
                    "cartographic_full",
                )
                nonlearning_budget = []
                for pal in nonlearning_full[:k_eff]:
                    cloned = clone_palette(pal)
                    cloned._source = "cartographic_budget"
                    nonlearning_budget.append(cloned)

                cvae_guarded = guardrails(cp, cvae, basemap_lab, scheme)
                nonlearning_guarded = guardrails(cp, nonlearning_budget, basemap_lab, scheme)
                hybrid = cvae + nonlearning_full
                hybrid_guarded = guardrails(cp, hybrid, basemap_lab, scheme)

                raw_cvae = cvae[0] if cvae else None
                cvae_score = best_by_composite(cvae)
                cvae_guarded_score = best_by_composite(cvae_guarded)
                cvae_rerank = constrained_top1(cp, cvae_guarded, n_classes)
                cvae_repair = repair_top1(cp, cvae_rerank, basemap_lab)

                nonlearning_score = best_by_composite(nonlearning_budget)
                nonlearning_rerank = constrained_top1(cp, nonlearning_guarded, n_classes)
                nonlearning_full_selector = repair_top1(cp, nonlearning_rerank, basemap_lab)

                hybrid_score = best_by_composite(hybrid)
                hybrid_rerank = constrained_top1(cp, hybrid_guarded, n_classes)
                full_pipeline = repair_top1(cp, hybrid_rerank, basemap_lab)

                variant_payloads = [
                    (
                        "Raw_CVAE_first",
                        raw_cvae.metrics if raw_cvae else None,
                        "raw_cvae_first",
                        palette_name(raw_cvae, "cvae"),
                        "cvae",
                        len(cvae),
                        None,
                    ),
                    (
                        "CVAE_score_best",
                        cvae_score.metrics if cvae_score else None,
                        "best_composite_cvae",
                        palette_name(cvae_score, "cvae"),
                        "cvae",
                        len(cvae),
                        None,
                    ),
                    (
                        "CVAE_score_guardrails",
                        cvae_guarded_score.metrics if cvae_guarded_score else None,
                        "best_composite_cvae_after_guardrails",
                        palette_name(cvae_guarded_score, "cvae"),
                        "cvae",
                        len(cvae),
                        len(cvae_guarded),
                    ),
                    (
                        "CVAE_guardrails_rerank",
                        cvae_rerank.metrics if cvae_rerank else None,
                        "constrained_top1_cvae",
                        palette_name(cvae_rerank, "cvae"),
                        "cvae",
                        len(cvae),
                        len(cvae_guarded),
                    ),
                    (
                        "CVAE_guardrails_rerank_repair",
                        cvae_repair.metrics if cvae_repair else None,
                        "constrained_top1_cvae_repaired",
                        palette_name(cvae_repair, "cvae"),
                        "cvae",
                        len(cvae),
                        len(cvae_guarded),
                    ),
                    (
                        "NonLearning_budget_score",
                        nonlearning_score.metrics if nonlearning_score else None,
                        "best_composite_nonlearning_budget",
                        palette_name(nonlearning_score, "cartographic_budget"),
                        "cartographic_budget",
                        len(nonlearning_budget),
                        None,
                    ),
                    (
                        "NonLearning_budget_full",
                        nonlearning_full_selector.metrics if nonlearning_full_selector else None,
                        "full_selector_nonlearning_budget",
                        palette_name(nonlearning_full_selector, "cartographic_budget"),
                        "cartographic_budget",
                        len(nonlearning_budget),
                        len(nonlearning_guarded),
                    ),
                    (
                        "Hybrid_score_best",
                        hybrid_score.metrics if hybrid_score else None,
                        "best_composite_hybrid",
                        palette_name(hybrid_score, "hybrid"),
                        "cvae_plus_cartographic_full",
                        len(hybrid),
                        None,
                    ),
                    (
                        "Hybrid_guardrails_rerank",
                        hybrid_rerank.metrics if hybrid_rerank else None,
                        "constrained_top1_hybrid",
                        palette_name(hybrid_rerank, "hybrid"),
                        "cvae_plus_cartographic_full",
                        len(hybrid),
                        len(hybrid_guarded),
                    ),
                    (
                        "Full_CartoPalette_v4_2",
                        full_pipeline.metrics if full_pipeline else None,
                        "full_cartopalette_v4_2",
                        palette_name(full_pipeline, "hybrid"),
                        "cvae_plus_cartographic_full",
                        len(hybrid),
                        len(hybrid_guarded),
                    ),
                ]

                cb_name, cb_metrics, _cb_lab = best_of_library(
                    get_colorbrewer_palettes(scheme, n_classes),
                    basemap_lab,
                    scheme,
                )
                mpl_name, mpl_metrics, _mpl_lab = best_of_library(
                    get_matplotlib_palettes(scheme, n_classes),
                    basemap_lab,
                    scheme,
                )
                rand_name, rand_metrics, _rand_lab = median_of_random(
                    get_random_palettes(n_classes, n_samples=10, seed=42 + combo_idx),
                    basemap_lab,
                    scheme,
                )
                variant_payloads.extend(
                    [
                        (
                            "ColorBrewer_best_of_library",
                            cb_metrics,
                            cb_name,
                            "colorbrewer",
                            "colorbrewer_library",
                            len(get_colorbrewer_palettes(scheme, n_classes)),
                            None,
                        ),
                        (
                            "Matplotlib_best_of_library",
                            mpl_metrics,
                            mpl_name,
                            "matplotlib",
                            "matplotlib_library",
                            len(get_matplotlib_palettes(scheme, n_classes)),
                            None,
                        ),
                        (
                            "Random_median_of_10",
                            rand_metrics,
                            rand_name,
                            "random",
                            "random_lab",
                            10,
                            None,
                        ),
                    ]
                )

                for payload in variant_payloads:
                    add_variant_rows(rows, case_meta, *payload)

    summary_rows = summarize(rows)
    delta_rows = stepwise_deltas(rows)
    budget_rows = budget_summary(rows)

    write_csv(RESULTS_CSV, rows)
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DELTAS_CSV, delta_rows)
    write_csv(BUDGET_CSV, budget_rows)
    write_summary_md(summary_rows, delta_rows, budget_rows)

    total_time = time.time() - t_start
    print(f"Wrote {len(rows)} raw rows to {RESULTS_CSV}")
    print(f"Wrote summary to {SUMMARY_MD}")
    print(f"Total time: {total_time/60:.1f} min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N basemaps")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Inference device")
    parser.add_argument("--seed", type=int, default=20260811, help="Base RNG seed")
    parser.add_argument("--fixed-k", type=int, default=None, help="Override adaptive CVAE candidate budget")
    parser.add_argument(
        "--scale",
        choices=["overview", "regional", "local"],
        default="regional",
        help="Scale metadata used when --scale-mode=fixed",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["fixed", "zoom_range"],
        default="fixed",
        help="Use fixed scale metadata or each basemap's zoom_range",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Progress log interval in cases")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
