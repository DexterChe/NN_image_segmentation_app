"""
Analysis module — particle measurements.
Computes area, axis lengths, Feret diameters, equivalent diameter.
Author: Dmitry Chezganov
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from skimage.measure import label, regionprops
from skimage import filters


def image_labeling(image: np.ndarray) -> np.ndarray:
    """
    Convert image to binary using Otsu's threshold and label connected regions.
    """
    if image.ndim > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    threshold = filters.threshold_otsu(gray)
    binary = gray > threshold
    label_image = label(binary)
    return label_image


def minimal_feret_diameter(binary_image: np.ndarray) -> float:
    """
    Compute minimal Feret diameter from a binary image.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_feret = float('inf')
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        min_feret = min(min_feret, min(width, height))
    return min_feret if min_feret != float('inf') else 0.0


def measure_particle(mask_binary: np.ndarray, px_size: float) -> Optional[Dict[str, float]]:
    """
    Measure a single particle from its binary mask.

    Args:
        mask_binary: binary mask (uint8, 0 or 255) of the cropped particle region
        px_size: pixel size in nm

    Returns:
        dict with measurement results or None if no region detected
    """
    label_image = image_labeling(mask_binary)
    props = regionprops(label_image)

    if not props:
        return None

    p = props[0]
    area_px = p.area
    major_px = p.major_axis_length
    minor_px = p.minor_axis_length
    eq_diam_px = p.equivalent_diameter_area
    max_feret_px = p.feret_diameter_max

    # Minimal Feret diameter
    binary_uint8 = np.where(label_image > 0, 255, 0).astype(np.uint8)
    min_feret_px = minimal_feret_diameter(binary_uint8)

    return {
        "Area_px": area_px,
        "Major_axis_length_px": major_px,
        "Minor_axis_length_px": minor_px,
        "Equivalent_diameter_px": eq_diam_px,
        "Max_Feret_diameter_px": max_feret_px,
        "Min_Feret_diameter_px": min_feret_px,
        "Area_nm2": area_px * px_size ** 2,
        "Major_axis_length_nm": major_px * px_size,
        "Minor_axis_length_nm": minor_px * px_size,
        "Equivalent_diameter_nm": eq_diam_px * px_size,
        "Max_Feret_diameter_nm": max_feret_px * px_size,
        "Min_Feret_diameter_nm": min_feret_px * px_size,
    }


def analyze_particles(particles, px_size: float,
                      progress_callback=None) -> pd.DataFrame:
    """
    Analyze all particles and return measurements DataFrame.

    Args:
        particles: list of ParticleData objects
        px_size: pixel size in nm
        progress_callback: optional callable(current, total)

    Returns:
        DataFrame with measurements for all particles
    """
    records = []
    for idx, particle in enumerate(particles):
        measurement = measure_particle(particle.mask_binary, px_size)
        if measurement is not None:
            measurement["Particle_ID"] = particle.index
            records.append(measurement)
        if progress_callback:
            progress_callback(idx + 1, len(particles))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Reorder columns
    cols = ["Particle_ID"] + [c for c in df.columns if c != "Particle_ID"]
    return df[cols]
