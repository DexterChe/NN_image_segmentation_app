"""
Utility functions for image loading, preprocessing, and folder management.
Author: Dmitry Chezganov
"""

import os
import cv2
import numpy as np
import glob
from typing import List, Tuple, Optional


def make_folder(save_path: str, folder_name: str) -> str:
    """
    Create a folder to save the output.

    Args:
        save_path: folder where to create the folder to save results
        folder_name: folder to save results

    Returns:
        path to folder to save results
    """
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path + '/'


def get_image_files(folder_path: str) -> List[str]:
    """
    Get sorted list of image files (.tif, .png, .jpg, .bmp) from a folder.
    """
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort()
    return files


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image handling different bit depths (float32, uint16, uint8).
    Returns RGB image as uint8.
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.dtype == 'float32' or img.dtype == 'uint16':
        img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_8bit = img_normalized.astype('uint8')
    else:
        img_8bit = img

    # Convert to RGB
    if img_8bit.ndim == 2:
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    elif img_8bit.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)

    return img_rgb


def load_image_from_bytes(file_bytes: bytes, filename: str) -> Optional[np.ndarray]:
    """
    Load an image from bytes (uploaded via Streamlit).
    Returns RGB image as uint8.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.dtype == 'float32' or img.dtype == 'uint16':
        img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_8bit = img_normalized.astype('uint8')
    else:
        img_8bit = img

    if img_8bit.ndim == 2:
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    elif img_8bit.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)

    return img_rgb


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an RGB image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return result


def rebinning(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Rebin (downscale) an image by a given factor.
    """
    h, w = image.shape[:2]
    rebinned = cv2.resize(image, (w // factor, h // factor))
    return rebinned
