# Appendix — CartoPalette v4.2

Supplementary material for the paper **"From Basemap to Color Palette: A Machine-Learning Approach to Context-Adaptive Map Design"** (Drews, Edler, Keil & Dickmann). This folder replaces the in-document appendix. It contains the benchmark results reported in the paper, the code that produced them, the v4.2 inference pipeline with all thresholds referenced in the text, and the training data artifacts that document the location-based train/val/test split.

Reproduction chain, end to end:

```
D (training data)  →  CartoPalette_Training.ipynb  →  cartopalette_v4.pt (C)
C (pipeline)       →  B (benchmark scripts)        →  A (results = Tables 5–7, Figures 5–8)
```

The scripts are copies from the repository structure and import from it (e.g. `from cartopalette.core.inference import CartoPalette`); to execute them, use the repository root layout with `requirements.txt`, not this folder directly. The files here are primarily provided for inspection.

---

## A_benchmark_results — Benchmark output (paper Sections 4–5)

Raw and aggregated results of the v4.2 benchmark run: 40 held-out test basemaps × 2 scheme types (sequential, diverging) × 4 class counts (3, 5, 7, 9) = 320 paired comparison cases against ColorBrewer (best-of-library), Matplotlib and a random baseline.

| File | Content | Paper reference |
|---|---|---|
| `test_set.json` | The 40 stratified test basemaps (provider, location, zoom, land cover, climate zone) selected by `B_benchmark_code/select_test_set.py` | Section 4 |
| `results_raw.csv` | One row per (basemap, scheme, n_classes, baseline, metric): all 7 metric scores (6 components + composite) for every selected palette | Basis of Tables 5–7, Figures 5–7 |
| `candidates_raw.json` | The full CartoPalette top-1 palettes for each test case (RGB/LAB), for exact reproducibility of Figure 8 | Section 5.5, Figure 8 |
| `win_rates.csv` | Pairwise win/tie/loss counts and win rates per (comparison, scheme, n_classes, metric), incl. mean/median paired deltas | Tables 5 and 7 |
| `statistical_tests.csv` | Wilcoxon signed-rank tests (paired, two-sided), rank-biserial r and Cohen's d per comparison and metric | Section 5.2, Figure 5 |

## B_benchmark_code — Benchmark harness (paper Section 4)

| File | Role |
|---|---|
| `select_test_set.py` | Draws the 40 test basemaps, stratified across all 10 tile providers, the 12 test locations, zoom ranges, land cover and climate zones. Reads only from the TEST split (see D). |
| `baselines.py` | Baseline palette generation: full ColorBrewer library, Matplotlib colormaps, random palettes — all converted to CIELAB so that every method is scored identically. |
| `run_benchmark.py` | Runs the full 320-case benchmark: CartoPalette top-1 vs. best-of-library baselines, records all 7 metrics → `results_raw.csv`, `candidates_raw.json`. |
| `analyze.py` | Aggregates `results_raw.csv`: win rates, Wilcoxon signed-rank tests, effect sizes (r, Cohen's d), breakdowns by scheme/class count → `win_rates.csv`, `statistical_tests.csv`. |
| `plot.py` | Publication figures (mean scores, win-rate heatmap by provider, per-metric distributions), 300 DPI, colorblind-safe. |

## C_pipeline_v4_2 — Inference pipeline with all thresholds (paper Sections 3.5–3.7)

| File | Role |
|---|---|
| `inference.py` | The v4.2 pipeline: basemap color extraction → adaptive k → CVAE generation → scoring → guardrail filtering → constrained ε-Pareto reranking with β·basemap_contrast tilt (β = 0.20) → top-3 output. Also preserves the v4.1 MMR reranker used as ablation baseline. |
| `repair.py` | Optional CIELAB contrast repair (Section 3.7): shifts palette colors that sit too close to dominant basemap colors in the a\*b\* plane while preserving the L\* structure; changes are only committed if basemap contrast strictly improves and no other metric drops by more than ε. |
| `model.py` | CVAE architecture (EfficientNet feature encoder + conditional decoder), including the v4.2 condition extension with the 10 dominant basemap colors in CIELAB. |
| `composite.py` | The Composite Score (Section 3.5): Basemap Contrast (0.20), Lightness Contrast (0.30), Hue Contrast (0.30), Distinguishability (0.10), CVD Robustness (0.05, Brettel/Machado simulation), Perceptual Ordering (0.05). |
| `composite_score.yaml` | The exact metric weights and thresholds used in training, inference and benchmark — single source of truth for Section 3.5. |
| `cartopalette_v4.pt` | Trained model weights used for the benchmark run. Without these, the benchmark cannot be recomputed. |
| `sampling.yaml`, `tile_servers.yaml` | Basemap sampling configuration and tile-server definitions (see D). |

## D_training_data_split — Training data and location split (paper Sections 3.2–3.4, Tables 1–2)

| File | Role |
|---|---|
| `locations.csv` | All sampling locations with coordinates, climate zone, land cover — and the `split` column that documents the **location-level** train/val/test split (Section 3.4). Key file: it shows that test locations never occur in training. |
| `train.txt`, `val.txt`, `test.txt` | The resulting per-sample split lists (basemap × scheme × class count identifiers). |
| `labels.csv.xz` | The full training label set: one row per (basemap, scheme, n_classes, candidate) with palettes in LAB/RGB and all six metric scores. XZ-compressed (139 MB → 22.5 MB); readable without unpacking via `pandas.read_csv("labels.csv.xz")`. |
| `dataset_statistics.json` | Aggregate dataset statistics (counts per provider, scheme, split), corresponding to Tables 1–2. |
| `sampling.yaml`, `tile_servers.yaml` | Configuration to re-fetch the basemap images deterministically (locations × providers × zoom levels). The rendered tiles themselves are not redistributed for license reasons. |
| `generate_candidates.py` | Stage 1–4 candidate generation (ColorBrewer seeds, basemap-adaptive CIELAB variations, contrast optimization, synthetic palettes), all conditioned on the basemap's dominant colors (Section 3.3). |
| `score_palettes.py` | Scores every candidate with the six metrics of `composite.py` — produces the training labels. |
| `build_dataset.py` | Diversity-aware top-K label selection, assembles `labels.csv`, the split files and `dataset_statistics.json`. |
| `color_utils.py`, `tile_utils.py`, `io_utils.py` | Shared utilities: CIELAB/CIEDE2000 conversions, tile fetching/stitching, I/O. |
| `CartoPalette_Training.ipynb` | The exact training run (Colab, GPU) that produced `cartopalette_v4.pt`, including hyperparameters and validation curves (Section 3.4). |

---

**Environment:** Python ≥ 3.10; dependencies in the repository root `requirements.txt` (PyTorch, torchvision, NumPy, Pillow, Matplotlib, SciPy, scikit-image, Streamlit for the web app).

**Web application** (Figure 2): https://cartopalette.streamlit.app/ — source in `web/app.py` at the repository root.
