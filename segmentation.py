"""
Segmentation module — model loading and inference.
Supports SAM, MobileSAM, FastSAM, YOLOv8 segmentation models.
Author: Dmitry Chezganov
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


# Model type mapping
MODEL_TYPES = {
    "SAM_b": {"loader": "SAM", "description": "SAM Base (sam_b.pt)"},
    "SAM_l": {"loader": "SAM", "description": "SAM Large (sam_l.pt)"},
    "SAM2_l": {"loader": "SAM", "description": "SAM2 Large (sam2_l.pt)"},
    "SAM2.1_l": {"loader": "SAM", "description": "SAM2.1 Large (sam2.1_l.pt)"},
    "MobileSAM": {"loader": "SAM", "description": "MobileSAM (mobile_sam.pt)"},
    "FastSAM_x": {"loader": "FastSAM", "description": "FastSAM Extra (FastSAM-x.pt)"},
    "FastSAM_s": {"loader": "FastSAM", "description": "FastSAM Small (FastSAM-s.pt)"},
    "YOLOv8n": {"loader": "YOLO", "description": "YOLOv8 Nano Seg (yolov8n-seg.pt)"},
    "YOLOv8s": {"loader": "YOLO", "description": "YOLOv8 Small Seg (yolov8s-seg.pt)"},
    "YOLOv8m": {"loader": "YOLO", "description": "YOLOv8 Medium Seg (yolov8m-seg.pt)"},
    "YOLOv8x": {"loader": "YOLO", "description": "YOLOv8 Extra Seg (yolov8x-seg.pt)"},
}

# Default model file names
MODEL_FILES = {
    "SAM_b": "sam_b.pt",
    "SAM_l": "sam_l.pt",
    "SAM2_l": "sam2_l.pt",
    "SAM2.1_l": "sam2.1_l.pt",
    "MobileSAM": "mobile_sam.pt",
    "FastSAM_x": "FastSAM-x.pt",
    "FastSAM_s": "FastSAM-s.pt",
    "YOLOv8n": "yolov8n-seg.pt",
    "YOLOv8s": "yolov8s-seg.pt",
    "YOLOv8m": "yolov8m-seg.pt",
    "YOLOv8x": "yolov8x-seg.pt",
}


def get_available_devices() -> List[str]:
    """Get list of available compute devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def find_models_in_folder(models_folder: str) -> Dict[str, str]:
    """
    Scan a folder for known model files and return dict {model_name: full_path}.
    """
    found = {}
    if not os.path.isdir(models_folder):
        return found
    for model_name, filename in MODEL_FILES.items():
        full_path = os.path.join(models_folder, filename)
        if os.path.isfile(full_path):
            found[model_name] = full_path
    return found


def load_model(model_name: str, model_path: str):
    """
    Load a segmentation model by name and path.
    Returns the model object.
    """
    from ultralytics import SAM, YOLO, FastSAM

    loader_type = MODEL_TYPES[model_name]["loader"]
    if loader_type == "SAM":
        model = SAM(model_path)
    elif loader_type == "FastSAM":
        model = FastSAM(model_path)
    elif loader_type == "YOLO":
        model = YOLO(model_path)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
    return model


def run_inference(model, images: List[np.ndarray], model_name: str,
                  device: str = "cpu", progress_callback=None) -> List:
    """
    Run segmentation inference on a list of images.

    Args:
        model: loaded model object
        images: list of RGB numpy arrays
        model_name: name of the model (for determining inference mode)
        device: compute device
        progress_callback: optional callable(current, total) for progress

    Returns:
        list of results (one per image)
    """
    loader_type = MODEL_TYPES[model_name]["loader"]
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    results = []

    for idx, image in enumerate(images):
        if loader_type in ("SAM",):
            result = model(image, device=device)
        elif loader_type == "FastSAM":
            result = model(image, device=device, retina_masks=True,
                           imgsz=max(image.shape[:2]), conf=0.4, iou=0.9)
        elif loader_type == "YOLO":
            result = model(image, device=device)
        else:
            result = model(image)

        results.append(result)

        if progress_callback:
            progress_callback(idx + 1, len(images))

    return results


@dataclass
class ParticleData:
    """Data for a single detected particle."""
    index: int
    mask_xy: np.ndarray       # mask contour coordinates
    box_xyxy: np.ndarray      # bounding box [x1, y1, x2, y2]
    crop_original: np.ndarray  # cropped original image
    crop_with_mask: np.ndarray # cropped image with mask overlay
    mask_binary: np.ndarray   # binary mask of the particle (full image size)


def extract_particles(result, image: np.ndarray) -> List[ParticleData]:
    """
    Extract particle data from a single segmentation result.

    Args:
        result: single image result from model inference
        image: original RGB image

    Returns:
        list of ParticleData objects
    """
    import random

    # Handle result wrapping (some models return list)
    if isinstance(result, list):
        result = result[0]

    if result.masks is None or len(result.masks) == 0:
        return []

    # Move masks to CPU if needed
    if hasattr(result.masks, 'data') and hasattr(result.masks.data, 'cpu'):
        result.masks.data = result.masks.data.cpu()

    particles = []
    h, w = image.shape[:2]

    for i in range(len(result.masks)):
        mask = result.masks[i]
        box = result.boxes[i]

        color = tuple(int(255 * random.random()) for _ in range(3))

        # Extract box coordinates
        box_coords = box.xyxy.cpu().numpy().flatten() if hasattr(box.xyxy, 'cpu') else box.xyxy.numpy().flatten()
        x1, y1, x2, y2 = map(int, box_coords)

        # Expand crop area by 10 px
        x1_l = max(0, x1 - 10)
        y1_l = max(0, y1 - 10)
        x2_l = min(w, x2 + 10)
        y2_l = min(h, y2 + 10)

        # Create mask overlay
        mask_image = np.zeros_like(image)
        contour = np.array(mask.xy[0]).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask_image, [contour], color=color)
        image_with_mask = cv2.addWeighted(image.copy(), 0.5, mask_image, 0.5, 0)

        # Binary mask
        mask_binary = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_binary, [contour], color=255)

        # Crops
        crop_orig = image[y1_l:y2_l, x1_l:x2_l].copy()
        crop_mask = image_with_mask[y1_l:y2_l, x1_l:x2_l].copy()

        particles.append(ParticleData(
            index=i,
            mask_xy=np.array(mask.xy[0]),
            box_xyxy=np.array([x1, y1, x2, y2]),
            crop_original=crop_orig,
            crop_with_mask=crop_mask,
            mask_binary=mask_binary[y1_l:y2_l, x1_l:x2_l],
        ))

    return particles
