import numpy as np
import pytest

from segmentation import extract_particles, run_inference


class _ArrayValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Mask:
    def __init__(self, contour):
        self.xy = [np.asarray(contour, dtype=float)]


class _Masks:
    def __init__(self, contours):
        self.items = [_Mask(contour) for contour in contours]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _Box:
    def __init__(self, coordinates):
        self.xyxy = _ArrayValue(coordinates)


class _Result:
    def __init__(self, contours, boxes):
        self.masks = _Masks(contours)
        self.boxes = [_Box(box) for box in boxes]


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return {"call": len(self.calls)}


@pytest.mark.parametrize(
    ("model_name", "expected_kwargs"),
    [
        ("SAM_b", {"device": "cpu"}),
        ("FastSAM_s", {"device": "cpu", "retina_masks": True, "imgsz": 12, "conf": 0.4, "iou": 0.9}),
        ("YOLOv8n", {"device": "cpu"}),
    ],
)
def test_run_inference_uses_model_specific_arguments(model_name, expected_kwargs):
    model = _RecordingModel()
    image = np.zeros((12, 8, 3), dtype=np.uint8)
    progress = []

    results = run_inference(
        model, [image, image], model_name, progress_callback=lambda current, total: progress.append((current, total))
    )

    assert results == [{"call": 1}, {"call": 2}]
    assert [kwargs for _, kwargs in model.calls] == [expected_kwargs, expected_kwargs]
    assert progress == [(1, 2), (2, 2)]


def test_extract_particles_clips_crops_at_image_boundary_and_preserves_mask():
    image = np.full((20, 20, 3), 100, dtype=np.uint8)
    result = _Result(
        contours=[[(0, 0), (7, 0), (7, 6), (0, 6)]],
        boxes=[[-3, -2, 7, 6]],
    )

    particles = extract_particles(result, image)

    assert len(particles) == 1
    particle = particles[0]
    assert particle.index == 0
    assert particle.box_xyxy.tolist() == [-3, -2, 7, 6]
    assert particle.crop_original.shape == (16, 17, 3)
    assert particle.mask_binary.shape == (16, 17)
    assert particle.mask_binary.max() == 255
    assert not np.array_equal(particle.crop_original, particle.crop_with_mask)


def test_extract_particles_returns_empty_list_when_no_masks_are_present():
    class EmptyResult:
        masks = None

    assert extract_particles(EmptyResult(), np.zeros((5, 5, 3), dtype=np.uint8)) == []
