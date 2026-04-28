"""
Visualization module — plotting functions for Streamlit app.
Author: Dmitry Chezganov
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from typing import Optional, Tuple
import random
import io

# Use non-interactive backend for thread safety
matplotlib.use("Agg")

# Publication-ready styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _particle_color(seed: int) -> Tuple[int, int, int]:
    """Deterministic RGB color by particle index."""
    rng = random.Random(seed + 17)
    return tuple(int(70 + 185 * rng.random()) for _ in range(3))


def create_mask_overlay_image(image: np.ndarray, particles,
                              alpha: float = 0.45,
                              draw_ids: bool = True) -> np.ndarray:
    """
    Create segmentation overlay image in the native resolution of `image`.
    """
    overlay = np.zeros_like(image)
    out = image.copy()

    for p in particles:
        color = _particle_color(int(p.index))
        contour = np.asarray(p.mask_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [contour], color=color)

    cv2.addWeighted(overlay, alpha, out, 1.0, 0, out)

    if draw_ids:
        for p in particles:
            x1, y1 = int(p.box_xyxy[0]), int(p.box_xyxy[1])
            cv2.putText(
                out,
                str(p.index),
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 40, 40),
                1,
                cv2.LINE_AA,
            )

    return out


def plot_image_with_masks(image: np.ndarray, particles,
                          model_name: str = "",
                          figsize: Tuple[float, float] = (10, 10),
                          dpi: int = 300,
                          show_title: bool = True) -> plt.Figure:
    """
    Plot image with colored mask overlays and particle numbers.

    Args:
        image: original RGB image
        particles: list of ParticleData objects
        model_name: model name for the title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(create_mask_overlay_image(image, particles, alpha=0.45, draw_ids=True))

    n_particles = len(particles)
    if show_title:
        ax.set_title(f'{model_name}: {n_particles} objects detected', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_particle_crops(particles, measurements_df: pd.DataFrame,
                        max_particles: int = 50) -> Optional[plt.Figure]:
    """
    Plot grid of particle crops with measurements.

    Args:
        particles: list of ParticleData
        measurements_df: DataFrame with measurements
        max_particles: max number of particles to show

    Returns:
        matplotlib Figure or None
    """
    n = min(len(particles), max_particles)
    if n == 0:
        return None

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=300)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(particles[idx].crop_with_mask)
        ax.set_title(f'#{particles[idx].index}', fontsize=8)
        ax.axis('off')

        # Add equivalent diameter if available
        pid = particles[idx].index
        match = measurements_df[measurements_df['Particle_ID'] == pid]
        if not match.empty:
            eq_d = match.iloc[0].get('Equivalent_diameter_nm', None)
            if eq_d is not None:
                ax.text(2, 12, f'd={eq_d:.1f} nm', color='yellow',
                        fontsize=6, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))

    # Hide empty axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    plt.suptitle(f'Particle crops (showing {n} of {len(particles)})', fontsize=12)
    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, model_name: str = "",
                       file_name: str = "",
                       figsize: Tuple[float, float] = (11.0, 7.2),
                       x_limits: Optional[Tuple[float, float]] = None,
                       dpi: int = 300) -> plt.Figure:
    """
    Plot distribution histograms for all particle properties.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)

    properties = [
        ("Area_nm2", "Area, nm²", "Area"),
        ("Major_axis_length_nm", "Major axis length, nm", "Major Axis"),
        ("Minor_axis_length_nm", "Minor axis length, nm", "Minor Axis"),
        ("Equivalent_diameter_nm", "Equivalent diameter, nm", "Eq. Diameter"),
        ("Max_Feret_diameter_nm", "Max Feret diameter, nm", "Max Feret"),
        ("Min_Feret_diameter_nm", "Min Feret diameter, nm", "Min Feret"),
    ]

    for idx, (col, xlabel, title) in enumerate(properties):
        row, c = divmod(idx, 3)
        ax = axes[row, c]
        data = df[col].dropna()
        if len(data) > 0:
            n_bins = min(max(8, len(data) // 4), 40)
            ax.hist(
                data,
                bins=n_bins,
                color="#2368a2",
                alpha=0.88,
                edgecolor="#ffffff",
                linewidth=0.6,
                rwidth=0.94
            )
            median_val = data.median()
            mean_val = data.mean()
            ax.axvline(
                mean_val,
                color="#b11f24",
                linestyle="--",
                linewidth=1.4,
                label=f"Mean = {mean_val:.1f}"
            )
            ax.axvline(
                median_val,
                color="#1f7a1f",
                linestyle="-.",
                linewidth=1.4,
                label=f"Median = {median_val:.1f}"
            )
            ax.legend(fontsize=8, frameon=False)
            ax.grid(True, alpha=0.25, linewidth=0.5)
        if x_limits and x_limits[1] > x_limits[0]:
            ax.set_xlim(x_limits)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=11, pad=6)
        ax.tick_params(labelsize=9)

    title_text = f'Particle property distributions'
    if model_name:
        title_text += f' | Model: {model_name}'
    if file_name:
        title_text += f' | {file_name}'
    plt.suptitle(title_text, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_interactive_histogram(df: pd.DataFrame,
                               prop: str,
                               prop_label: str,
                               n_bins: int = 20,
                               figsize: Tuple[float, float] = (8.8, 4.5),
                               dpi: int = 300) -> Optional[plt.Figure]:
    """
    Plot a publication-ready histogram for one property.
    """
    if prop not in df.columns:
        return None

    data = df[prop].dropna()
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(
        data,
        bins=n_bins,
        color="#2368a2",
        alpha=0.88,
        edgecolor="#ffffff",
        linewidth=0.7,
        rwidth=0.94
    )

    mean_v = float(data.mean())
    median_v = float(data.median())
    ax.axvline(mean_v, color="#b11f24", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_v:.2f}")
    ax.axvline(median_v, color="#1f7a1f", linestyle="-.", linewidth=1.8,
               label=f"Median = {median_v:.2f}")

    ax.set_xlabel(prop_label)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: {prop_label}", pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


def plot_comparison(original: np.ndarray, processed: np.ndarray,
                    title_left: str = "Original",
                    title_right: str = "Processed") -> plt.Figure:
    """
    Plot side-by-side comparison of two images.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    ax1.imshow(original)
    ax1.set_title(title_left, fontsize=11)
    ax1.axis('off')
    ax2.imshow(processed)
    ax2.set_title(title_right, fontsize=11)
    ax2.axis('off')
    plt.tight_layout()
    return fig


def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics table from measurements DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    nm_cols = [c for c in df.columns if c.endswith('_nm') or c.endswith('_nm2')]
    if not nm_cols:
        return pd.DataFrame()

    stats = df[nm_cols].describe().T
    stats.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    # Rename index for display
    name_map = {
        "Area_nm2": "Area (nm²)",
        "Major_axis_length_nm": "Major Axis (nm)",
        "Minor_axis_length_nm": "Minor Axis (nm)",
        "Equivalent_diameter_nm": "Eq. Diameter (nm)",
        "Max_Feret_diameter_nm": "Max Feret (nm)",
        "Min_Feret_diameter_nm": "Min Feret (nm)",
    }
    stats.index = [name_map.get(idx, idx) for idx in stats.index]
    return stats.round(2)


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Convert matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
