"""
Streamlit Image Segmentation App
=================================
Interactive application for SEM/TEM image segmentation and particle analysis.
Supports SAM, MobileSAM, FastSAM, YOLOv8 models.

Author: Dmitry Chezganov
Run: streamlit run app.py

"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import time
import tempfile
import zipfile
import io
import shutil
import subprocess
import matplotlib.pyplot as plt
import cv2
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

APP_DIR = os.path.dirname(__file__)
DOCS_IMAGE_DIR = os.path.join(APP_DIR, "docs", "images")
TUTORIAL_IMAGE_DIR = os.path.join(DOCS_IMAGE_DIR, "tutorial")
APP_LOGO_PATH = os.path.join(DOCS_IMAGE_DIR, "app_logo.svg")

from utils import (
    load_image, load_image_from_bytes, apply_clahe, rebinning,
    get_image_files
)
from segmentation import (
    MODEL_TYPES, get_available_devices, find_models_in_folder,
    load_model, run_inference, extract_particles, resolve_model_path, get_model_filename
)
from analysis import analyze_particles
from visualization import (
    plot_particle_crops, plot_distributions,
    create_summary_stats,
    create_mask_overlay_image, plot_interactive_histogram
)


# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Segmentation & PSD Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 16px;
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }
    .app-hero {
        padding: 1.55rem 1.35rem 1.2rem 1.35rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(249, 178, 51, 0.20), transparent 26%),
            linear-gradient(135deg, rgba(12, 45, 72, 0.10) 0%, rgba(20, 93, 160, 0.05) 100%);
        border: 1px solid rgba(120, 140, 160, 0.18);
        margin-bottom: 1rem;
        overflow: visible;
    }
    .main-header {
        font-size: 3rem;
        line-height: 1.2;
        font-weight: 800;
        margin: 0 0 0.42rem 0;
        padding: 0.08em 0 0.03em 0;
        letter-spacing: -0.04em;
        display: block;
        overflow: visible;
    }
    .sub-header {
        font-size: 1.04rem;
        opacity: 0.82;
        margin: 0;
        max-width: 980px;
    }
    .tutorial-image-frame {
        max-width: 100%;
        margin: 0 0 0.35rem 0;
    }
    .sidebar-shell {
        padding: 0.95rem 1rem 0.35rem 1rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(12, 45, 72, 0.08) 0%, rgba(12, 45, 72, 0.02) 100%);
        border: 1px solid rgba(120, 140, 160, 0.18);
        margin-bottom: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 14px 14px 0 0;
        font-size: 0.98rem;
        font-weight: 600;
        background: rgba(12, 45, 72, 0.04);
    }
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] li,
    .stTabs [data-baseweb="tab-panel"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stMetricValue"] {
        font-size: 0.97rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    h3 {
        font-size: 1.38rem;
        margin-top: 0.4rem;
    }
    .tutorial-card {
        padding: 1.1rem 1.2rem;
        border: 1px solid rgba(120, 140, 160, 0.22);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(18, 39, 56, 0.05) 0%, rgba(18, 39, 56, 0.015) 100%);
        margin-bottom: 1rem;
    }
    .tutorial-step {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .sidebar-logo-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.3rem;
    }
    .sidebar-logo-text {
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .sidebar-kicker {
        font-size: 0.82rem;
        opacity: 0.72;
        margin-top: 0.18rem;
    }
    .empty-state {
        padding: 1.4rem 1.35rem;
        border-radius: 22px;
        border: 1px dashed rgba(120, 140, 160, 0.34);
        background: linear-gradient(180deg, rgba(20, 93, 160, 0.04), rgba(20, 93, 160, 0.015));
        margin: 0.5rem 0 0.8rem 0;
    }
    .empty-state-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: 0.28rem;
    }
    .empty-state-copy {
        font-size: 0.95rem;
        opacity: 0.84;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
def init_session_state():
    defaults = {
        "images_loaded": False,
        "images": [],
        "image_names": [],
        "images_processed": [],       # after CLAHE / rebinning
        "images_rebinned": [],        # intermediate stage (optional)
        "images_clahe": [],           # intermediate stage (optional)
        "images_source_folder": None,
        "segmentation_done": False,
        "seg_results": [],             # raw model results
        "particles_per_image": [],     # list of lists of ParticleData
        "measurements_per_image": [],  # list of DataFrames
        "model_loaded": False,
        "model": None,
        "current_model_name": None,
        "model_device": None,
        "processing_time": 0.0,
        "models_dir_input_pending": None,
        "images_folder_input_pending": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def select_folder_dialog() -> Optional[str]:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                [
                    "osascript",
                    "-e", "try",
                    "-e", 'POSIX path of (choose folder with prompt "Select a folder")',
                    "-e", "on error number -128",
                    "-e", 'return ""',
                    "-e", "end try",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            folder = result.stdout.strip()
            return os.path.normpath(folder) if folder else None

        if not TK_AVAILABLE:
            return None

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory()
        root.destroy()
        return os.path.normpath(folder) if folder else None
    except Exception:
        return None


def folder_picker_is_supported() -> bool:
    return sys.platform == "darwin" or TK_AVAILABLE


def get_default_models_dir() -> str:
    repo_dir = os.path.dirname(__file__)
    candidate_dirs = [
        os.path.join(repo_dir, "Models"),
        os.path.join(os.path.dirname(repo_dir), "Models"),
    ]
    for candidate in candidate_dirs:
        if os.path.isdir(candidate):
            return candidate
    return candidate_dirs[0]


def sync_pending_widget_value(widget_key: str, pending_key: str) -> None:
    pending_value = st.session_state.get(pending_key)
    if pending_value is not None:
        st.session_state[widget_key] = pending_value
        st.session_state[pending_key] = None


def save_results_to_dir(save_dir: str, images, particles_list,
                        measurements_list, names, model_name: str,
                        original_images=None,
                        images_rebinned=None,
                        images_clahe=None) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for subfolder in [
        "images/raw",
        "images/processed",
        "images/rebin",
        "images/clahe",
        "masks_overlay",
        "raw_masks",
        "particle_masks",
        "distributions",
        "interactive_histograms",
        "csv",
        "crops",
        "metadata",
    ]:
        os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)

    for i, (particles, df, name) in enumerate(zip(
        particles_list, measurements_list, names
    )):
        base = os.path.splitext(name)[0]

        # Save original/processed/intermediate images
        if original_images and i < len(original_images):
            cv2.imwrite(
                os.path.join(save_dir, "images", "raw", f"{base}_raw.png"),
                cv2.cvtColor(original_images[i], cv2.COLOR_RGB2BGR)
            )
        if images and i < len(images):
            cv2.imwrite(
                os.path.join(save_dir, "images", "processed", f"{base}_processed.png"),
                cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            )
        if images_rebinned and i < len(images_rebinned) and images_rebinned[i] is not None:
            cv2.imwrite(
                os.path.join(save_dir, "images", "rebin", f"{base}_rebin.png"),
                cv2.cvtColor(images_rebinned[i], cv2.COLOR_RGB2BGR)
            )
        if images_clahe and i < len(images_clahe) and images_clahe[i] is not None:
            cv2.imwrite(
                os.path.join(save_dir, "images", "clahe", f"{base}_clahe.png"),
                cv2.cvtColor(images_clahe[i], cv2.COLOR_RGB2BGR)
            )

        if particles:
            overlay = create_mask_overlay_image(images[i], particles, alpha=0.45, draw_ids=True)
            cv2.imwrite(
                os.path.join(save_dir, "masks_overlay", f"{base}_masks_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

            # Save raw binary mask (full image size)
            h, w = images[i].shape[:2]
            full_mask = np.zeros((h, w), dtype=np.uint8)
            mask_rows = []
            for p in particles:
                x1, y1, x2, y2 = [int(v) for v in p.box_xyxy]
                x1_l = max(0, x1 - 10)
                y1_l = max(0, y1 - 10)
                x2_l = min(w, x2 + 10)
                y2_l = min(h, y2 + 10)
                mask_crop = p.mask_binary
                if mask_crop.size == 0:
                    continue
                full_mask[y1_l:y2_l, x1_l:x2_l] = np.maximum(
                    full_mask[y1_l:y2_l, x1_l:x2_l],
                    mask_crop
                )
                single_mask = np.zeros((h, w), dtype=np.uint8)
                single_mask[y1_l:y2_l, x1_l:x2_l] = np.maximum(
                    single_mask[y1_l:y2_l, x1_l:x2_l], mask_crop
                )
                cv2.imwrite(
                    os.path.join(
                        save_dir, "particle_masks", f"{base}_particle_{int(p.index):04d}_mask.png"
                    ),
                    single_mask
                )
                mask_rows.append({
                    "Particle_ID": int(p.index),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_x1": x1_l,
                    "crop_y1": y1_l,
                    "crop_x2": x2_l,
                    "crop_y2": y2_l,
                    "mask_area_px": int(np.count_nonzero(mask_crop)),
                })
            cv2.imwrite(os.path.join(save_dir, "raw_masks", f"{base}_mask_full.png"), full_mask)
            if mask_rows:
                pd.DataFrame(mask_rows).to_csv(
                    os.path.join(save_dir, "metadata", f"{base}_mask_metadata.csv"),
                    index=False
                )

        if not df.empty:
            fig = plot_distributions(df, model_name, name)
            fig.savefig(os.path.join(save_dir, "distributions", f"{base}_dist.png"),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            hist_cols = [
                ("Equivalent_diameter_nm", "Equivalent Diameter (nm)"),
                ("Area_nm2", "Area (nm²)"),
                ("Major_axis_length_nm", "Major Axis Length (nm)"),
                ("Minor_axis_length_nm", "Minor Axis Length (nm)"),
                ("Max_Feret_diameter_nm", "Max Feret Diameter (nm)"),
                ("Min_Feret_diameter_nm", "Min Feret Diameter (nm)"),
            ]
            for col, label in hist_cols:
                fig_hist = plot_interactive_histogram(
                    df, col, label, n_bins=min(max(8, len(df) // 3), 40)
                )
                if fig_hist is not None:
                    fig_hist.savefig(
                        os.path.join(
                            save_dir, "interactive_histograms", f"{base}_{col}_hist.png"
                        ),
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close(fig_hist)

            df.to_csv(os.path.join(save_dir, "csv", f"{base}_measurements.csv"),
                      index=False)

            if particles:
                for p in particles:
                    cv2.imwrite(
                        os.path.join(save_dir, "crops", f"{base}_particle_{int(p.index):04d}.png"),
                        cv2.cvtColor(p.crop_with_mask, cv2.COLOR_RGB2BGR)
                    )

    all_dfs = []
    for df, name in zip(measurements_list, names):
        if not df.empty:
            df_copy = df.copy()
            df_copy.insert(0, "Image", name)
            all_dfs.append(df_copy)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(save_dir, "csv", "combined_measurements.csv"),
                        index=False)
        fig = plot_distributions(combined, model_name, "Combined")
        fig.savefig(os.path.join(save_dir, "distributions", "combined_dist.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


def tutorial_image_path(filename: str) -> Optional[str]:
    path = os.path.join(TUTORIAL_IMAGE_DIR, filename)
    return path if os.path.isfile(path) else None


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tutorial_tab() -> None:
    st.markdown("### Quick Tutorial")
    st.markdown(
        "Use this tab as the built-in onboarding guide. It mirrors the repository tutorial "
        "and explains the full workflow from model setup to export."
    )

    raw_img = tutorial_image_path("step-1-raw-input.png")
    overlay_img = tutorial_image_path("step-4-mask-overlay.png")
    dist_img = tutorial_image_path("step-5-distributions.png")

    intro_col, model_col = st.columns([1.25, 1.0], gap="large")
    with intro_col:
        st.markdown(
            """
            <div class="tutorial-card">
                <div class="tutorial-step">1. Install and launch</div>
                Clone the repository, create a Python environment, install dependencies, and run the app with <b>streamlit run app.py</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="tutorial-card">
                <div class="tutorial-step">2. Add model weights</div>
                This repository intentionally does not include <b>.pt</b> weights. You can keep checkpoints in the local <b>Models</b> folder or let Ultralytics download the selected official checkpoint automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with model_col:
        st.info(
            "Model setup after downloading the repository:\n\n"
            "- easiest mode: choose a model in the sidebar and press Load model\n"
            "- if the checkpoint is not found locally, Ultralytics can download the official file automatically\n"
            "- manual mode: place .pt files into ./Models with canonical filenames\n"
            "- external mode: point the sidebar to any folder that already contains your checkpoints"
        )

    if raw_img:
        image_col, _, _ = st.columns([1.15, 1.55, 1.55], gap="medium")
        with image_col:
            st.markdown("<div class='tutorial-image-frame'>", unsafe_allow_html=True)
            st.image(raw_img, caption="Step 1: raw microscopy image input", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown(
            """
            <div class="tutorial-card">
                <div class="tutorial-step">3. Load images</div>
                In the Load tab, either upload files directly or point the app to a folder with TIFF, PNG, JPG, or BMP images.
            </div>
            <div class="tutorial-card">
                <div class="tutorial-step">4. Preprocess if needed</div>
                Use CLAHE for local contrast enhancement and rebinning for faster or lower-resolution segmentation runs.
            </div>
            <div class="tutorial-card">
                <div class="tutorial-step">5. Run segmentation</div>
                Load a model in the Segmentation tab, choose CPU, CUDA, or Apple MPS, and run inference on the processed images.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(
            """
            <div class="tutorial-card">
                <div class="tutorial-step">6. Inspect results</div>
                Review the native-resolution overlay, particle crops, and the full measurements table before exporting data.
            </div>
            <div class="tutorial-card">
                <div class="tutorial-step">7. Explore distributions</div>
                The Distributions tab shows particle size statistics such as equivalent diameter, area, and Feret diameters.
            </div>
            <div class="tutorial-card">
                <div class="tutorial-step">8. Export everything</div>
                Download CSV files, a ZIP archive, or save the full local result package with masks, crops, metadata, and figures.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if overlay_img or dist_img:
        img_col1, img_col2 = st.columns(2, gap="large")
        with img_col1:
            if overlay_img:
                st.image(overlay_img, caption="Step 6: segmentation mask overlay", use_container_width=True)
        with img_col2:
            if dist_img:
                st.image(dist_img, caption="Step 7: particle property distributions", use_container_width=True)

    st.markdown("### Supported Model Filenames")
    model_table = pd.DataFrame(
        {
            "Model": [
                "SAM Base",
                "SAM Large",
                "SAM2 Large",
                "SAM2.1 Large",
                "MobileSAM",
                "FastSAM X",
                "FastSAM S",
                "YOLOv8 Nano Seg",
                "YOLOv8 Small Seg",
                "YOLOv8 Medium Seg",
                "YOLOv8 Extra Seg",
            ],
            "Filename": [
                "sam_b.pt",
                "sam_l.pt",
                "sam2_l.pt",
                "sam2.1_l.pt",
                "mobile_sam.pt",
                "FastSAM-x.pt",
                "FastSAM-s.pt",
                "yolov8n-seg.pt",
                "yolov8s-seg.pt",
                "yolov8m-seg.pt",
                "yolov8x-seg.pt",
            ],
        }
    )
    st.dataframe(model_table, use_container_width=True, hide_index=True)


init_session_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='sidebar-shell'>", unsafe_allow_html=True)
        if os.path.isfile(APP_LOGO_PATH):
            logo_col, text_col = st.columns([0.34, 0.66], gap="small")
            with logo_col:
                st.image(APP_LOGO_PATH, width=82)
            with text_col:
                st.markdown(
                    "<div class='sidebar-logo-text'>Segmentation Studio</div><div class='sidebar-kicker'>SEM/TEM particle workflow</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("## Workspace Settings")

        # ── Model section ────────────────────────────────────────────────
        st.markdown("### Model Setup")

        # Auto-detect models folder
        default_models_dir = get_default_models_dir()
        if "models_dir_input" not in st.session_state:
            st.session_state.models_dir_input = default_models_dir
        sync_pending_widget_value("models_dir_input", "models_dir_input_pending")

        model_source_mode = st.radio(
            "Model source",
            options=["Use local folder", "Auto-download official model"],
            horizontal=False,
            help="Use local checkpoints from a folder or let Ultralytics fetch the selected official weight automatically.",
        )

        models_dir = st.session_state.models_dir_input
        available_models = {}
        if model_source_mode == "Use local folder":
            models_dir = st.text_input(
                "Models folder",
                key="models_dir_input",
                help="Path to a folder containing local .pt model files"
            )
            cols_models = st.columns([1, 1])
            with cols_models[0]:
                browse_models = st.button("Browse...", key="browse_models")
            with cols_models[1]:
                st.write("")
            if browse_models:
                folder = select_folder_dialog()
                if folder:
                    st.session_state.models_dir_input_pending = folder
                    st.rerun()
                elif not folder_picker_is_supported():
                    st.warning("Folder picker is not available in this environment.")

            available_models = find_models_in_folder(models_dir)
            if available_models:
                st.success(f"Detected local models: {len(available_models)}")
                model_choice = st.selectbox(
                    "Model",
                    options=list(available_models.keys()),
                    format_func=lambda x: f"{x} — {MODEL_TYPES[x]['description']}",
                    index=0
                )
            else:
                st.warning("No matching local models found in the selected folder.")
                model_choice = st.selectbox(
                    "Model",
                    options=list(MODEL_TYPES.keys()),
                    format_func=lambda x: f"{x} — {MODEL_TYPES[x]['description']}"
                )
                st.caption("Tip: switch to auto-download mode if you want Ultralytics to fetch the checkpoint for you.")
        else:
            model_choice = st.selectbox(
                "Official model",
                options=list(MODEL_TYPES.keys()),
                format_func=lambda x: f"{x} — {MODEL_TYPES[x]['description']}"
            )
            resolved_path = resolve_model_path(model_choice, default_models_dir)
            if os.path.isfile(resolved_path):
                st.caption(f"Local checkpoint already available: {resolved_path}")
            else:
                st.caption(
                    f"If {get_model_filename(model_choice)} is missing locally, Ultralytics will download it automatically during model load."
                )

        # ── Device ───────────────────────────────────────────────────────
        devices = get_available_devices()
        device = st.selectbox(
            "Device",
            options=devices,
            index=0,
            help="CUDA for GPU, MPS for Apple Silicon, CPU by default"
        )

        st.markdown("---")

        # ── Pixel size ───────────────────────────────────────────────────
        st.markdown("### Pixel Size")
        px_size = st.number_input(
            "Pixel size (nm/px)",
            min_value=0.001,
            max_value=10000.0,
            value=0.719,
            step=0.001,
            format="%.3f",
            help="Enter pixel size in nanometers"
        )

        st.markdown("---")

        # ── Preprocessing ────────────────────────────────────────────────
        st.markdown("### Preprocessing")
        use_clahe = st.checkbox("Apply CLAHE", value=False)
        clahe_clip = 2.0
        clahe_tile = 8
        if use_clahe:
            clahe_clip = st.slider("CLAHE clip limit", 1.0, 10.0, 2.0, 0.5)
            clahe_tile = st.slider("CLAHE tile size", 2, 32, 8, 2)

        use_rebin = st.checkbox("Rebinning (downscale)", value=False)
        rebin_factor = 2
        if use_rebin:
            rebin_factor = st.slider("Rebinning factor", 2, 8, 2)

        st.markdown("---")

        # ── Filter settings ──────────────────────────────────────────────
        st.markdown("### Result Filtering")
        size_metric = st.selectbox(
            "Size metric",
            options=[
                "Equivalent_diameter_nm",
                "Max_Feret_diameter_nm",
                "Major_axis_length_nm",
                "Minor_axis_length_nm",
                "Area_nm2",
            ],
            index=0,
            help="Metric used for min/max size filtering"
        )
        min_size_value = st.number_input(
            "Min size value",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Set 0 to disable the min size filter"
        )
        max_size_value = st.number_input(
            "Max size value",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Set 0 to disable the max size filter"
        )

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#9CA3AF; font-size:0.8rem;'>"
            "Image Segmentation App v2.0<br>"
            "Copyright 2026 Dmitry Chezganov<br>"
            "Licensed under AGPL-3.0-only<br>"
            "<a href='https://github.com/DexterChe/NN_image_segmentation_app' target='_blank'>Source code</a>"
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return {
        "model_source_mode": model_source_mode,
        "models_dir": models_dir,
        "model_choice": model_choice,
        "available_models": available_models,
        "device": device,
        "px_size": px_size,
        "use_clahe": use_clahe,
        "clahe_clip": clahe_clip,
        "clahe_tile": clahe_tile,
        "use_rebin": use_rebin,
        "rebin_factor": rebin_factor,
        "size_metric": size_metric,
        "min_size_value": min_size_value,
        "max_size_value": max_size_value,
    }


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        '<div class="app-hero">'
        '<div class="main-header">Image Segmentation & Particle Analysis</div>'
        '<div class="sub-header">Segment microscopy images, measure particles, inspect distributions, and export results from one consistent workspace.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    cfg = render_sidebar()

    if st.session_state.model_loaded and st.session_state.model_device != cfg["device"]:
        st.warning("Device selection changed. Please reload the model.")
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.current_model_name = None
        st.session_state.model_device = None

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_tutorial, tab_load, tab_preprocess, tab_segment, tab_results, tab_distrib, tab_export = st.tabs([
        "📘 Tutorial", "📁 Load", "🔧 Preprocess", "🔬 Segmentation",
        "📊 Results", "📈 Distributions", "💾 Export"
    ])

    with tab_tutorial:
        render_tutorial_tab()

    # ════════════════════════════════════════════════════════════════════
    # TAB 1: Loading images
    # ════════════════════════════════════════════════════════════════════
    with tab_load:
        st.markdown("### Load Images")

        load_method = st.radio(
            "Loading method",
            ["📤 Upload files", "📂 Specify folder path"],
            horizontal=True
        )

        if load_method == "📤 Upload files":
            uploaded_files = st.file_uploader(
                "Select images",
                type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                help="Supported formats: TIF, TIFF, PNG, JPG, BMP (8/16/32 bit)"
            )
            if uploaded_files:
                if st.button("✅ Load images", type="primary"):
                    images = []
                    names = []
                    progress = st.progress(0)
                    for i, f in enumerate(uploaded_files):
                        img = load_image_from_bytes(f.read(), f.name)
                        if img is not None:
                            images.append(img)
                            names.append(f.name)
                        progress.progress((i + 1) / len(uploaded_files))
                    progress.empty()

                    st.session_state.images = images
                    st.session_state.image_names = names
                    st.session_state.images_loaded = True
                    st.session_state.images_processed = images.copy()
                    st.session_state.images_rebinned = [None] * len(images)
                    st.session_state.images_clahe = [None] * len(images)
                    st.session_state.images_source_folder = None
                    # Reset downstream state
                    st.session_state.segmentation_done = False
                    st.session_state.seg_results = []
                    st.session_state.particles_per_image = []
                    st.session_state.measurements_per_image = []
                    st.success(f"Loaded {len(images)} images")
            else:
                render_empty_state(
                    "No images loaded yet",
                    "Upload one or more microscopy images here, or switch to folder mode for batch loading from disk."
                )

        else:  # Folder path
            if "images_folder_input" not in st.session_state:
                st.session_state.images_folder_input = ""
            sync_pending_widget_value("images_folder_input", "images_folder_input_pending")

            folder_path = st.text_input(
                "Path to image folder",
                key="images_folder_input",
                placeholder="/path/to/images/"
            )
            cols_images = st.columns([1, 1])
            with cols_images[0]:
                browse_images = st.button("Browse...", key="browse_images")
            with cols_images[1]:
                st.write("")
            if browse_images:
                folder = select_folder_dialog()
                if folder:
                    st.session_state.images_folder_input_pending = folder
                    st.rerun()
                elif not folder_picker_is_supported():
                    st.warning("Folder picker is not available in this environment.")
            if folder_path and st.button("✅ Load from folder", type="primary"):
                files = get_image_files(folder_path)
                if not files:
                    st.error("No images found in the specified folder")
                else:
                    images = []
                    names = []
                    progress = st.progress(0)
                    for i, fp in enumerate(files):
                        img = load_image(fp)
                        if img is not None:
                            images.append(img)
                            names.append(os.path.basename(fp))
                        progress.progress((i + 1) / len(files))
                    progress.empty()

                    st.session_state.images = images
                    st.session_state.image_names = names
                    st.session_state.images_loaded = True
                    st.session_state.images_processed = images.copy()
                    st.session_state.images_rebinned = [None] * len(images)
                    st.session_state.images_clahe = [None] * len(images)
                    st.session_state.images_source_folder = folder_path
                    st.session_state.segmentation_done = False
                    st.session_state.seg_results = []
                    st.session_state.particles_per_image = []
                    st.session_state.measurements_per_image = []
                    st.success(f"Loaded {len(images)} images from folder")
            elif not folder_path:
                render_empty_state(
                    "Select a source folder",
                    "Use Browse or paste a folder path that contains TIFF, PNG, JPG, or BMP microscopy images."
                )

        # Preview loaded images
        if st.session_state.images_loaded and st.session_state.images:
            st.markdown("---")
            st.markdown(f"**Images loaded: {len(st.session_state.images)}**")

            cols = st.columns(min(4, len(st.session_state.images)))
            for i, col in enumerate(cols):
                if i < len(st.session_state.images):
                    with col:
                        img = st.session_state.images[i]
                        width = max(120, img.shape[1] // 3)
                        st.image(img, caption=st.session_state.image_names[i], width=width)

            if len(st.session_state.images) > 4:
                with st.expander(f"Show all {len(st.session_state.images)} images"):
                    grid_cols = st.columns(4)
                    for i, img in enumerate(st.session_state.images):
                        with grid_cols[i % 4]:
                            width = max(120, img.shape[1] // 3)
                            st.image(img, caption=st.session_state.image_names[i], width=width)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2: Preprocessing
    # ════════════════════════════════════════════════════════════════════
    with tab_preprocess:
        st.markdown("### Preprocess Images")

        if not st.session_state.images_loaded:
            render_empty_state(
                "Preprocessing is waiting for images",
                "Start in the Load tab, then return here to apply CLAHE and rebinning before segmentation."
            )
        else:
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | CLAHE | {'✅ On' if cfg['use_clahe'] else '❌ Off'} |
            | Clip limit | {cfg['clahe_clip']} |
            | Tile size | {cfg['clahe_tile']}×{cfg['clahe_tile']} |
            | Rebinning | {'✅ ×' + str(cfg['rebin_factor']) if cfg['use_rebin'] else '❌ Off'} |
            """)

            if st.button("🔧 Apply preprocessing", type="primary"):
                processed = []
                rebinned_stage = []
                clahe_stage = []
                progress = st.progress(0, text="Processing images...")

                for i, img in enumerate(st.session_state.images):
                    result = img.copy()
                    rebinned_img = None
                    clahe_img = None
                    if cfg['use_rebin']:
                        result = rebinning(result, cfg['rebin_factor'])
                        rebinned_img = result.copy()
                    if cfg['use_clahe']:
                        result = apply_clahe(result, cfg['clahe_clip'],
                                             (cfg['clahe_tile'], cfg['clahe_tile']))
                        clahe_img = result.copy()
                    processed.append(result)
                    rebinned_stage.append(rebinned_img)
                    clahe_stage.append(clahe_img)
                    progress.progress((i + 1) / len(st.session_state.images),
                                      text=f"Processing {i+1}/{len(st.session_state.images)}")

                progress.empty()
                st.session_state.images_processed = processed
                st.session_state.images_rebinned = rebinned_stage
                st.session_state.images_clahe = clahe_stage
                st.session_state.segmentation_done = False  # reset downstream
                st.success("Preprocessing complete!")

            # Show comparison
            if st.session_state.images_processed:
                st.markdown("---")
                st.markdown("#### Comparison: Original vs Processed")

                img_idx = st.selectbox(
                    "Select image for comparison",
                    options=range(len(st.session_state.images)),
                    format_func=lambda x: st.session_state.image_names[x]
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.image(st.session_state.images[img_idx], use_container_width=True)
                    shape = st.session_state.images[img_idx].shape
                    st.caption(f"Size: {shape[1]}×{shape[0]} px")
                with col2:
                    st.markdown("**Processed**")
                    st.image(st.session_state.images_processed[img_idx],
                             use_container_width=True)
                    shape_p = st.session_state.images_processed[img_idx].shape
                    st.caption(f"Size: {shape_p[1]}×{shape_p[0]} px")

    # ════════════════════════════════════════════════════════════════════
    # TAB 3: Segmentation
    # ════════════════════════════════════════════════════════════════════
    with tab_segment:
        st.markdown("### Segmentation")

        if not st.session_state.images_loaded:
            render_empty_state(
                "Segmentation is waiting for input",
                "Load images first, then choose a model source and device before running inference."
            )
        else:
            # Model info
            col_model, col_device, col_images = st.columns(3)
            with col_model:
                st.metric("Model", cfg['model_choice'])
            with col_device:
                st.metric("Device", cfg['device'])
            with col_images:
                st.metric("Images", len(st.session_state.images_processed))

            st.markdown("---")

            # Load model button
            model_name = cfg['model_choice']
            if st.button("📦 Load model", type="secondary"):
                model_path = None
                if cfg["model_source_mode"] == "Use local folder":
                    model_path = cfg['available_models'].get(model_name)
                    if not model_path:
                        st.error("The selected model is not available in the chosen local folder.")
                else:
                    model_path = resolve_model_path(model_name, cfg["models_dir"])

                if model_path:
                    with st.spinner(f"Loading model {model_name}..."):
                        model = load_model(model_name, model_path)
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.session_state.current_model_name = model_name
                        st.session_state.model_device = cfg["device"]
                    if os.path.isfile(model_path):
                        st.success(f"Model {model_name} loaded from local storage.")
                    else:
                        st.success(f"Model {model_name} loaded. If needed, Ultralytics downloaded the checkpoint automatically.")

            # Run segmentation
            if st.session_state.model_loaded:
                st.markdown(f"**Loaded model:** {st.session_state.current_model_name}")

                if st.button("🚀 Run segmentation", type="primary"):
                    images = st.session_state.images_processed
                    names = st.session_state.image_names

                    progress_bar = st.progress(0, text="Segmenting (inference)...")
                    status_text = st.empty()
                    per_image_text = st.empty()
                    start_time = time.time()

                    def seg_progress(current, total):
                        progress_bar.progress(current / total,
                                              text=f"Segmenting: {current}/{total}")
                        if current - 1 < len(names):
                            per_image_text.caption(f"Current image: {names[current - 1]}")

                    # Run inference
                    results = run_inference(
                        st.session_state.model,
                        images,
                        st.session_state.current_model_name,
                        device=cfg['device'],
                        progress_callback=seg_progress
                    )

                    elapsed = time.time() - start_time
                    progress_bar.empty()
                    per_image_text.empty()

                    # Extract particles
                    status_text.text("Extracting particles...")
                    all_particles = []
                    all_measurements = []

                    analyze_progress = st.progress(0, text="Analyzing particles...")

                    for i, (result, img) in enumerate(zip(results, images)):
                        particles = extract_particles(result, img)

                        # Measure particles
                        if particles:
                            df = analyze_particles(particles, cfg['px_size'])

                            # Min/max size filters by chosen metric
                            metric = cfg["size_metric"]
                            if metric in df.columns and not df.empty:
                                if cfg["min_size_value"] > 0:
                                    df = df[df[metric] >= cfg["min_size_value"]]
                                if cfg["max_size_value"] > 0:
                                    df = df[df[metric] <= cfg["max_size_value"]]
                                allowed_ids = set(df["Particle_ID"].astype(int).tolist())
                                particles = [p for p in particles if p.index in allowed_ids]
                                df = df.reset_index(drop=True)

                            all_measurements.append(df)
                        else:
                            df = pd.DataFrame()
                            all_measurements.append(df)

                        all_particles.append(particles)

                        analyze_progress.progress(
                            (i + 1) / len(images),
                            text=f"Analyzing: {i+1}/{len(images)}"
                        )

                    analyze_progress.empty()
                    status_text.empty()

                    # Store results
                    st.session_state.seg_results = results
                    st.session_state.particles_per_image = all_particles
                    st.session_state.measurements_per_image = all_measurements
                    st.session_state.segmentation_done = True
                    st.session_state.processing_time = elapsed

                    total_particles = sum(len(p) for p in all_particles)
                    st.success(
                        f"Segmentation complete in {elapsed:.1f}s | "
                        f"Particles detected: {total_particles}"
                    )

            # Show quick results overview
            if st.session_state.segmentation_done:
                st.markdown("---")
                st.markdown("#### Results Overview")

                total_p = sum(len(p) for p in st.session_state.particles_per_image)
                avg_p = total_p / max(len(st.session_state.particles_per_image), 1)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total particles", total_p)
                c2.metric("Avg per image", f"{avg_p:.1f}")
                c3.metric("Processing time", f"{st.session_state.processing_time:.1f}s")
                c4.metric("Images", len(st.session_state.particles_per_image))

    # ════════════════════════════════════════════════════════════════════
    # TAB 4: Results
    # ════════════════════════════════════════════════════════════════════
    with tab_results:
        st.markdown("### Segmentation Results")

        if not st.session_state.segmentation_done:
            render_empty_state(
                "No segmentation results yet",
                "Load a model and run segmentation to inspect overlays, particle crops, and per-image measurements here."
            )
        else:
            images = st.session_state.images_processed
            particles_list = st.session_state.particles_per_image
            measurements_list = st.session_state.measurements_per_image
            names = st.session_state.image_names

            # Image selector
            img_idx = st.selectbox(
                "Select image",
                options=range(len(images)),
                format_func=lambda x: f"{names[x]} ({len(particles_list[x])} particles)",
                key="results_img_selector"
            )

            particles = particles_list[img_idx]
            df = measurements_list[img_idx]

            # Stats summary
            if not df.empty:
                st.markdown("#### 📋 Summary Statistics")
                summary = create_summary_stats(df)
                st.dataframe(summary, use_container_width=True)

            # Segmentation overlay
            st.markdown("#### 🎨 Segmentation Masks")
            overlay_img = create_mask_overlay_image(images[img_idx], particles, alpha=0.45, draw_ids=True)
            mask_col, mask_meta_col = st.columns([1, 1])
            with mask_col:
                st.image(
                    overlay_img,
                    caption=f"{names[img_idx]} | native resolution {overlay_img.shape[1]}×{overlay_img.shape[0]} px",
                    use_container_width=True
                )
            with mask_meta_col:
                st.markdown("**Display settings**")
                st.markdown("- Native segmentation output resolution")
                st.markdown("- UI width: half page")
                st.markdown(f"- Particles shown: **{len(particles)}**")
                st.markdown(f"- Render standard: **300 dpi export**")

            # Particle crops
            if particles:
                st.markdown("#### 🔎 Particle Crops")
                max_show = st.slider("Max particles to display", 10, 200, 50,
                                     key="max_crops")
                fig_crops = plot_particle_crops(particles, df, max_show)
                if fig_crops:
                    st.pyplot(fig_crops)
                    plt.close(fig_crops)

            # Full measurements table
            if not df.empty:
                st.markdown("#### 📊 Measurements Table")
                st.dataframe(
                    df.style.format({col: "{:.2f}" for col in df.select_dtypes(include='number').columns}),
                    use_container_width=True,
                    height=400
                )

    # ════════════════════════════════════════════════════════════════════
    # TAB 5: Distributions
    # ════════════════════════════════════════════════════════════════════
    with tab_distrib:
        st.markdown("### Particle Property Distributions")

        if not st.session_state.segmentation_done:
            render_empty_state(
                "No distributions available yet",
                "Run segmentation first. This tab will then show publication-ready histograms and summary statistics."
            )
        else:
            measurements_list = st.session_state.measurements_per_image
            names = st.session_state.image_names

            # Option: per image or combined
            view_mode = st.radio(
                "View mode",
                ["🖼️ Per image", "📦 All images"],
                horizontal=True,
                key="distrib_mode"
            )

            st.markdown("#### X-axis limits")
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                x_min = st.number_input("X min", value=0.0, step=1.0, key="x_min_dist")
            with col_x2:
                x_max = st.number_input("X max", value=0.0, step=1.0, key="x_max_dist")
            x_limits = None
            if x_max > x_min and x_max > 0:
                x_limits = (x_min, x_max)

            model_name = st.session_state.current_model_name or ""
            plot_df = pd.DataFrame()
            plot_label = ""
            if view_mode == "🖼️ Per image":
                img_idx = st.selectbox(
                    "Select image",
                    options=range(len(measurements_list)),
                    format_func=lambda x: names[x],
                    key="distrib_img_selector"
                )
                plot_df = measurements_list[img_idx]
                plot_label = names[img_idx]
            else:
                all_dfs = [df for df in measurements_list if not df.empty]
                if all_dfs:
                    plot_df = pd.concat(all_dfs, ignore_index=True)
                    plot_label = "All images"

            dist_col, stat_col = st.columns([1, 1])
            if plot_df.empty:
                st.warning("No data to display")
            else:
                with dist_col:
                    st.markdown("#### Publication Figure (300 dpi)")
                    fig = plot_distributions(
                        plot_df, model_name, plot_label, figsize=(7.0, 4.6), x_limits=x_limits, dpi=300
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption("Display width: half-column layout.")
                with stat_col:
                    st.markdown("#### Summary Statistics")
                    st.markdown(f"**Total particles: {len(plot_df)}**")
                    summary = create_summary_stats(plot_df)
                    st.dataframe(summary, use_container_width=True)

            # Interactive histogram (3 interface pages)
            st.markdown("---")
            st.markdown("#### 🎛️ Interactive Histogram (300 dpi)")

            available_cols = {
                "Equivalent_diameter_nm": "Equivalent Diameter (nm)",
                "Area_nm2": "Area (nm²)",
                "Major_axis_length_nm": "Major Axis Length (nm)",
                "Minor_axis_length_nm": "Minor Axis Length (nm)",
                "Max_Feret_diameter_nm": "Max Feret Diameter (nm)",
                "Min_Feret_diameter_nm": "Min Feret Diameter (nm)",
            }
            if not plot_df.empty:
                tab_cfg, tab_plot, tab_stats = st.tabs(
                    ["1) Histogram Controls", "2) Histogram Figure", "3) Histogram Stats"]
                )
                with tab_cfg:
                    col_int1, col_int2 = st.columns(2)
                    with col_int1:
                        prop = st.selectbox(
                            "Property",
                            options=list(available_cols.keys()),
                            format_func=lambda x: available_cols[x],
                            key="interactive_prop"
                        )
                    with col_int2:
                        n_bins = st.slider("Number of bins", 3, 100, 18, key="interactive_bins")
                    st.caption("Configured for publication-ready rendering at 300 dpi.")

                with tab_plot:
                    hist_col, _ = st.columns([1, 1])
                    with hist_col:
                        fig_int = plot_interactive_histogram(
                            plot_df, prop, available_cols[prop], n_bins=n_bins,
                            figsize=(7.0, 3.8), dpi=300
                        )
                        if fig_int is not None:
                            st.pyplot(fig_int, use_container_width=True)
                            plt.close(fig_int)
                            st.caption("Half-column display, 300 dpi.")
                        else:
                            st.warning("No values available for the selected property.")

                with tab_stats:
                    data = plot_df[prop].dropna()
                    if not data.empty:
                        stats_df = pd.DataFrame({
                            "Metric": ["Count", "Mean", "Median", "Std", "Min", "Max"],
                            "Value": [
                                float(len(data)),
                                float(data.mean()),
                                float(data.median()),
                                float(data.std(ddof=1)) if len(data) > 1 else 0.0,
                                float(data.min()),
                                float(data.max()),
                            ],
                        })
                        st.dataframe(
                            stats_df.style.format({"Value": "{:.4g}"}),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("No data for statistics.")

    # ════════════════════════════════════════════════════════════════════
    # TAB 6: Export
    # ════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown("### Export Results")

        if not st.session_state.segmentation_done:
            render_empty_state(
                "Nothing to export yet",
                "After segmentation finishes, this tab will let you download CSV files, figures, ZIP archives, and a complete local result package."
            )
        else:
            measurements_list = st.session_state.measurements_per_image
            names = st.session_state.image_names
            particles_list = st.session_state.particles_per_image

            st.markdown("#### 📄 Download CSV")

            # Per-image CSV download
            for i, (df, name) in enumerate(zip(measurements_list, names)):
                if not df.empty:
                    csv = df.to_csv(index=False)
                    base = os.path.splitext(name)[0]
                    st.download_button(
                        label=f"📥 {base}_measurements.csv",
                        data=csv,
                        file_name=f"{base}_measurements.csv",
                        mime="text/csv",
                        key=f"csv_download_{i}"
                    )

            # Combined CSV
            all_dfs = []
            for i, (df, name) in enumerate(zip(measurements_list, names)):
                if not df.empty:
                    df_copy = df.copy()
                    df_copy.insert(0, "Image", name)
                    all_dfs.append(df_copy)

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                csv_all = combined.to_csv(index=False)
                st.download_button(
                    label="📥 All measurements (combined_measurements.csv)",
                    data=csv_all,
                    file_name="combined_measurements.csv",
                    mime="text/csv",
                    key="csv_download_all"
                )

            st.markdown("---")
            st.markdown("#### 🖼️ Download Results Archive")

            # Generate and zip full results package
            if st.button("📦 Generate ZIP with results", type="primary"):
                with st.spinner("Generating plots..."):
                    buf = io.BytesIO()
                    tmp_dir = tempfile.mkdtemp(prefix="segmentation_results_")
                    try:
                        save_results_to_dir(
                            tmp_dir,
                            st.session_state.images_processed,
                            particles_list,
                            measurements_list,
                            names,
                            st.session_state.current_model_name or "",
                            original_images=st.session_state.images,
                            images_rebinned=st.session_state.images_rebinned,
                            images_clahe=st.session_state.images_clahe,
                        )
                        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for root, _, files in os.walk(tmp_dir):
                                for filename in files:
                                    full_path = os.path.join(root, filename)
                                    rel_path = os.path.relpath(full_path, tmp_dir)
                                    zf.write(full_path, arcname=rel_path)
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)

                    buf.seek(0)
                    st.download_button(
                        label="📥 Download ZIP archive",
                        data=buf,
                        file_name="segmentation_results.zip",
                        mime="application/zip",
                        key="zip_download"
                    )

            st.markdown("---")
            st.markdown("#### 💾 Save Results")

            source_folder = st.session_state.images_source_folder
            if source_folder:
                default_save_dir = os.path.join(source_folder, "segmentation_results")
            else:
                default_save_dir = os.path.join(os.getcwd(), "segmentation_results")
            st.caption(f"Save path: `{default_save_dir}`")

            if st.button("💾 Save results", type="primary"):
                try:
                    save_results_to_dir(
                        default_save_dir,
                        st.session_state.images_processed,
                        particles_list,
                        measurements_list,
                        names,
                        st.session_state.current_model_name or "",
                        original_images=st.session_state.images,
                        images_rebinned=st.session_state.images_rebinned,
                        images_clahe=st.session_state.images_clahe,
                    )
                    st.success(f"Results saved to {default_save_dir}")
                except Exception as e:
                    st.error(f"Save error: {e}")


if __name__ == "__main__":
    main()
