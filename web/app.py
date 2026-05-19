"""CartoPalette v4.2 — Streamlit Web Application."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cartopalette.core.inference import CartoPalette, Palette, VALID_SCHEMES, VALID_NCLASSES, VALID_SCALES


# ── Page config ──
st.set_page_config(
    page_title="CartoPalette",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .palette-container {
        display: flex;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        margin: 0.5rem 0;
        height: 60px;
    }
    .palette-color {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .palette-color:hover {
        transform: scaleY(1.1);
        z-index: 1;
    }
    .score-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .hex-list {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #555;
    }
    .sidebar-info {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    .pipeline-info {
        background: #f0f7ff;
        border-left: 3px solid #2e75b6;
        padding: 0.5rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.8rem;
        color: #444;
        margin: 0.5rem 0 1rem 0;
    }
    .metric-bar-container {
        display: flex;
        align-items: center;
        margin: 2px 0;
        font-size: 0.75rem;
    }
    .metric-label {
        width: 140px;
        color: #555;
        flex-shrink: 0;
    }
    .metric-bar-bg {
        flex: 1;
        background: #e9ecef;
        border-radius: 4px;
        height: 14px;
        overflow: hidden;
        margin: 0 6px;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    .metric-value {
        width: 40px;
        text-align: right;
        font-weight: 600;
        color: #333;
        flex-shrink: 0;
    }
    .metric-weight {
        width: 30px;
        text-align: right;
        color: #999;
        font-size: 0.65rem;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_bar(label: str, value: float, weight: float, color: str = "#4caf50") -> str:
    """Render a single metric as an HTML bar."""
    pct = max(0, min(100, value * 100))
    # Color coding: green (>0.6), orange (0.3-0.6), red (<0.3)
    if value >= 0.6:
        bar_color = "#4caf50"
    elif value >= 0.3:
        bar_color = "#ff9800"
    else:
        bar_color = "#f44336"

    return (
        f'<div class="metric-bar-container">'
        f'<span class="metric-label">{label}</span>'
        f'<div class="metric-bar-bg"><div class="metric-bar-fill" style="width:{pct:.0f}%;background:{bar_color}"></div></div>'
        f'<span class="metric-value">{value:.2f}</span>'
        f'<span class="metric-weight">({weight:.0%})</span>'
        f'</div>'
    )


def render_palette_html(palette: Palette, index: int):
    """Render a palette as colored rectangles with hex codes."""
    colors_html = ""
    for i, hex_color in enumerate(palette.hex_colors):
        rgb = palette.rgb[i]
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = "#fff" if luminance < 0.5 else "#000"
        colors_html += f'<div class="palette-color" style="background:{hex_color};color:{text_color}">{hex_color}</div>'

    score_html = ""
    if palette.score is not None:
        score_html = f' <span class="score-badge">Score: {palette.score:.3f}</span>'

    st.markdown(f"**Suggestion {index + 1}**{score_html}", unsafe_allow_html=True)
    st.markdown(f'<div class="palette-container">{colors_html}</div>', unsafe_allow_html=True)

    # Score breakdown (expandable)
    if palette.metrics is not None:
        m = palette.metrics
        w = m.get("weights", {})
        with st.expander("Score Breakdown", expanded=False):
            metrics_html = ""
            metrics_html += render_metric_bar("Lightness Contrast", m["lightness_contrast"], w.get("l_contrast", 0.30))
            metrics_html += render_metric_bar("Hue Contrast", m["hue_contrast"], w.get("h_contrast", 0.30))
            metrics_html += render_metric_bar("Basemap Contrast", m["basemap_contrast"], w.get("contrast", 0.20))
            metrics_html += render_metric_bar("Distinguishability", m["distinguishability"], w.get("dist", 0.10))
            metrics_html += render_metric_bar("CVD Robustness", m["cvd_robustness"], w.get("cvd", 0.05))
            metrics_html += render_metric_bar("Perceptual Order", m["perceptual_ordering"], w.get("order", 0.05))
            st.markdown(metrics_html, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load model (cached across reruns)."""
    return CartoPalette(model_path)


def find_model():
    """Try to find the model file in common locations."""
    candidates = [
        PROJECT_ROOT / "cartopalette" / "pretrained" / "cartopalette_v4.pt",
        PROJECT_ROOT / "models" / "cartopalette_v4.pt",
        PROJECT_ROOT / "cartopalette_v4.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main():
    # ── Header ──
    st.markdown('<p class="main-title">CartoPalette</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Context-Aware Color Palette Generation for Thematic Cartography</p>', unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("Settings")

        scheme = st.selectbox(
            "Scheme Type",
            options=VALID_SCHEMES,
            index=0,
            help="Sequential: ordered data (e.g. population density). Diverging: data with neutral midpoint (e.g. temperature anomaly)."
        )

        n_classes = st.select_slider(
            "Number of Classes",
            options=VALID_NCLASSES,
            value=5,
            help="How many colors in the palette."
        )

        scale = st.selectbox(
            "Map Scale",
            options=VALID_SCALES,
            index=1,
            help="Overview: country/continent level. Regional: state/province. Local: city/neighborhood."
        )

        st.markdown("---")

        with st.expander("Advanced", expanded=False):
            n_suggestions = st.slider(
                "Suggestions",
                min_value=1, max_value=5, value=3,
                help="Number of palette suggestions (default: 3)."
            )

            auto_k = st.checkbox(
                "Adaptive candidate budget",
                value=True,
                help="Use more neural samples on difficult basemaps."
            )

            k = st.slider(
                "CVAE Candidates (k)",
                min_value=5, max_value=100, value=20,
                disabled=auto_k,
                help="Number of neural candidates. v4.2 also adds deterministic cartographic candidates."
            )

            lambda_mmr = st.slider(
                "Diversity (lambda)",
                min_value=0.5, max_value=1.0, value=0.85, step=0.05,
                help="Only used for the legacy MMR ablation."
            )

        st.markdown("---")

        # Model path
        default_path = find_model()
        model_path = st.text_input(
            "Model Path",
            value=default_path or "",
            help="Path to cartopalette_v4.pt"
        )

        st.markdown("""
        <div class="sidebar-info">
        <strong>About CartoPalette v4.2</strong><br>
        A CVAE-based deep learning system that generates color palettes tailored to your basemap's
        visual context. Unlike static tools like ColorBrewer, CartoPalette analyzes the basemap image
        and produces palettes that contrast well with the background and are accessible under color
        vision deficiency.<br><br>
        <strong>Pipeline:</strong> CVAE + cartographic candidates &rarr; Score &rarr; Guardrails &rarr; constrained Top-3<br><br>
        <em>Part of a PhD dissertation on Generative AI for Cartographic Design.</em>
        </div>
        """, unsafe_allow_html=True)

    # ── Main area ──
    col_upload, col_results = st.columns([1, 1.5])

    with col_upload:
        st.subheader("Upload Basemap")
        uploaded = st.file_uploader(
            "Drop a basemap image here",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Any map image works — screenshots, tiles, exported maps."
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption=f"Basemap ({image.size[0]}x{image.size[1]})", use_container_width=True)

    with col_results:
        st.subheader("Generated Palettes")

        if uploaded is None:
            st.info("Upload a basemap image to generate palette suggestions.")
            return

        if not model_path or not Path(model_path).exists():
            st.error(f"Model not found at: {model_path}\n\nPlace `cartopalette_v4.pt` in `cartopalette/pretrained/`.")
            return

        # Generate
        with st.spinner("Generating palettes..."):
            try:
                cp = load_model(model_path)
                palettes = cp.suggest(
                    image,
                    scheme=scheme,
                    n_classes=n_classes,
                    scale=scale,
                    n_suggestions=n_suggestions if 'n_suggestions' in dir() else 3,
                    k=None if ('auto_k' in dir() and auto_k) else (k if 'k' in dir() else 20),
                    lambda_mmr=lambda_mmr if 'lambda_mmr' in dir() else 0.85,
                    reranker="constrained",
                    apply_repair=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return

        if not palettes:
            st.warning("No palettes generated. Check model and inputs.")
            return

        # Pipeline info
        st.markdown(
            f'<div class="pipeline-info">'
            f'Pipeline: adaptive CVAE + cartographic candidates &rarr; scored &rarr; '
            f'guardrail-filtered &rarr; constrained top-{len(palettes)}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Display palettes
        for i, pal in enumerate(palettes):
            render_palette_html(pal, i)

        # ── Export options ──
        st.markdown("---")
        st.subheader("Export")

        export_format = st.radio("Format", ["Hex Codes", "RGB (0-255)", "CIELAB"], horizontal=True)

        export_lines = []
        for i, pal in enumerate(palettes):
            if export_format == "Hex Codes":
                export_lines.append(f"Palette {i+1}: {', '.join(pal.hex_colors)}")
            elif export_format == "RGB (0-255)":
                rgb_strs = [f"({r},{g},{b})" for r, g, b in pal.rgb_255]
                export_lines.append(f"Palette {i+1}: {', '.join(rgb_strs)}")
            else:
                lab_strs = [f"({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})" for c in pal.lab]
                export_lines.append(f"Palette {i+1}: {', '.join(lab_strs)}")

        export_text = "\n".join(export_lines)
        st.code(export_text, language=None)

        st.download_button(
            "Download as TXT",
            data=export_text,
            file_name=f"cartopalette_{scheme}_{n_classes}cls.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
