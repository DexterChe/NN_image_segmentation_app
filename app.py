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

from utils import (
    load_image, load_image_from_bytes, apply_clahe, rebinning,
    get_image_files
)
from segmentation import (
    MODEL_TYPES, get_available_devices, find_models_in_folder,
    load_model, run_inference, extract_particles
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
    /* Main header — inherits text color from theme */
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
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


init_session_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # ── Model section ────────────────────────────────────────────────
        st.markdown("### 🧠 Model")

        # Auto-detect models folder
        default_models_dir = get_default_models_dir()
        if "models_dir_input" not in st.session_state:
            st.session_state.models_dir_input = default_models_dir

        models_dir = st.text_input(
            "Models folder",
            value=st.session_state.models_dir_input,
            key="models_dir_input",
            help="Path to the folder containing .pt model files"
        )
        cols_models = st.columns([1, 1])
        with cols_models[0]:
            browse_models = st.button("Browse...", key="browse_models")
        with cols_models[1]:
            st.write("")
        if browse_models:
            folder = select_folder_dialog()
            if folder:
                st.session_state.models_dir_input = folder
                models_dir = folder
            elif not folder_picker_is_supported():
                st.warning("Folder picker is not available in this environment.")

        available_models = find_models_in_folder(models_dir)
        if available_models:
            st.success(f"Models found: {len(available_models)}")
            model_choice = st.selectbox(
                "Select model",
                options=list(available_models.keys()),
                format_func=lambda x: f"{x} — {MODEL_TYPES[x]['description']}",
                index=0
            )
        else:
            st.warning("No models found in the specified folder")
            model_choice = st.selectbox(
                "Select model",
                options=list(MODEL_TYPES.keys()),
                format_func=lambda x: f"{x} — {MODEL_TYPES[x]['description']}"
            )
            model_path_manual = st.text_input("Path to .pt model", value="")
            if model_path_manual:
                available_models = {model_choice: model_path_manual}

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
        st.markdown("### 📏 Pixel Size")
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
        st.markdown("### 🔧 Preprocessing")
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
        st.markdown("### 🔍 Result Filtering")
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
            "Image Segmentation App v2.0<br>© Dmitry Chezganov</p>",
            unsafe_allow_html=True
        )

    return {
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
    st.markdown('<p class="main-header">🔬 Image Segmentation & Particle Analysis</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">SEM/TEM image segmentation using SAM / FastSAM / YOLO '
        'and particle size distribution analysis</p>',
        unsafe_allow_html=True
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
    tab_load, tab_preprocess, tab_segment, tab_results, tab_distrib, tab_export = st.tabs([
        "📁 Load", "🔧 Preprocess", "🔬 Segmentation",
        "📊 Results", "📈 Distributions", "💾 Export"
    ])

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

        else:  # Folder path
            if "images_folder_input" not in st.session_state:
                st.session_state.images_folder_input = ""

            folder_path = st.text_input(
                "Path to image folder",
                value=st.session_state.images_folder_input,
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
                    st.session_state.images_folder_input = folder
                    folder_path = folder
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
        st.markdown("### Image Preprocessing")

        if not st.session_state.images_loaded:
            st.info("⬅️ First load images in the **Load** tab")
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
            st.info("⬅️ First load images in the **Load** tab")
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
                if model_name in cfg['available_models']:
                    model_path = cfg['available_models'][model_name]
                    with st.spinner(f"Loading model {model_name}..."):
                        model = load_model(model_name, model_path)
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                        st.session_state.current_model_name = model_name
                        st.session_state.model_device = cfg["device"]
                    st.success(f"Model {model_name} loaded!")
                else:
                    st.error("Model not found. Specify the model path in the sidebar.")

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
            st.info("⬅️ First run segmentation in the **Segmentation** tab")
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
            st.info("⬅️ First run segmentation in the **Segmentation** tab")
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
            st.info("⬅️ First run segmentation in the **Segmentation** tab")
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
