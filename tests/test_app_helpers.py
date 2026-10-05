import os

import cv2
import numpy as np
import pandas as pd

import app
from segmentation import ParticleData


def _particle() -> ParticleData:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    return ParticleData(
        index=7,
        mask_xy=np.array([[2, 2], [5, 2], [5, 5], [2, 5]]),
        box_xyxy=np.array([2, 2, 6, 6]),
        crop_original=np.zeros((16, 16, 3), dtype=np.uint8),
        crop_with_mask=np.ones((16, 16, 3), dtype=np.uint8) * 100,
        mask_binary=mask,
    )


def _measurements() -> pd.DataFrame:
    values = {
        "Particle_ID": [7],
        "Area_nm2": [16.0],
        "Major_axis_length_nm": [4.0],
        "Minor_axis_length_nm": [4.0],
        "Equivalent_diameter_nm": [4.5],
        "Max_Feret_diameter_nm": [5.0],
        "Min_Feret_diameter_nm": [4.0],
    }
    return pd.DataFrame(values)


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


class _Result:
    def __init__(self, contours, boxes):
        self.masks = _Masks(contours)
        self.boxes = [type("Box", (), {"xyxy": _ArrayValue(box)})() for box in boxes]


def test_default_models_dir_prefers_repository_models_folder():
    assert app.get_default_models_dir() == os.path.join(app.APP_DIR, "Models")


def test_init_session_state_sets_all_defaults_without_overwriting_values(monkeypatch):
    state = {"images_loaded": True}
    monkeypatch.setattr(app.st, "session_state", state)

    app.init_session_state()

    assert state["images_loaded"] is True
    assert state["images"] == []
    assert state["segmentation_done"] is False
    assert state["model_loaded"] is False


def test_folder_picker_support_matches_platform_and_toolkit_availability(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(app, "TK_AVAILABLE", False)
    assert app.folder_picker_is_supported() is False

    monkeypatch.setattr(app, "TK_AVAILABLE", True)
    assert app.folder_picker_is_supported() is True

    monkeypatch.setattr(app.sys, "platform", "darwin")
    assert app.folder_picker_is_supported() is True


def test_select_folder_dialog_normalizes_macos_path(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *args, **kwargs: type("Process", (), {"stdout": "/tmp/images/\n"})(),
    )

    assert app.select_folder_dialog() == "/tmp/images"


def test_preprocess_images_records_stages_without_mutating_inputs():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    original = image.copy()
    progress = []

    processed, rebinned, clahe = app.preprocess_images(
        [image, image],
        use_rebin=True,
        rebin_factor=2,
        use_clahe=True,
        clahe_clip=2.0,
        clahe_tile=8,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert [item.shape for item in processed] == [(32, 32, 3), (32, 32, 3)]
    assert [item.shape for item in rebinned] == [(32, 32, 3), (32, 32, 3)]
    assert all(item.shape == (32, 32, 3) for item in clahe)
    assert np.array_equal(image, original)
    assert progress == [(1, 2), (2, 2)]


def test_preprocess_images_without_options_returns_copies_and_empty_stages():
    image = np.ones((8, 8, 3), dtype=np.uint8)

    processed, rebinned, clahe = app.preprocess_images([image])

    assert processed[0] is not image
    assert np.array_equal(processed[0], image)
    assert rebinned == [None]
    assert clahe == [None]


def test_process_segmentation_results_applies_size_filter_to_particles_and_rows():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = _Result(
        contours=[
            [(10, 10), (19, 10), (19, 19), (10, 19)],
            [(32, 32), (51, 32), (51, 51), (32, 51)],
        ],
        boxes=[[10, 10, 20, 20], [32, 32, 52, 52]],
    )
    progress = []

    particles, measurements = app.process_segmentation_results(
        [result],
        [image],
        px_size=1.0,
        min_size_value=15.0,
        max_size_value=25.0,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert [particle.index for particle in particles[0]] == [1]
    assert measurements[0]["Particle_ID"].tolist() == [1]
    assert progress == [(1, 1)]


def test_process_segmentation_results_keeps_empty_images_aligned():
    class EmptyResult:
        masks = None

    images = [np.zeros((5, 5, 3), dtype=np.uint8)]
    particles, measurements = app.process_segmentation_results([EmptyResult()], images, 1.0)

    assert particles == [[]]
    assert len(measurements) == 1
    assert measurements[0].empty


def test_tutorial_image_path_only_returns_existing_files():
    assert app.tutorial_image_path("step-1-raw-input.png")
    assert app.tutorial_image_path("does-not-exist.png") is None


def test_sync_pending_widget_value_moves_value_and_clears_pending(monkeypatch):
    state = {"pending": "/tmp/models"}
    monkeypatch.setattr(app.st, "session_state", state)

    app.sync_pending_widget_value("models", "pending")

    assert state == {"pending": None, "models": "/tmp/models"}


def test_render_empty_state_writes_html_markup(monkeypatch):
    calls = []
    monkeypatch.setattr(app.st, "markdown", lambda *args, **kwargs: calls.append((args, kwargs)))

    app.render_empty_state("Nothing here", "Load an image to continue")

    assert "Nothing here" in calls[0][0][0]
    assert calls[0][1]["unsafe_allow_html"] is True


def test_save_results_writes_a_complete_export_package(tmp_path):
    image = np.full((20, 20, 3), 120, dtype=np.uint8)
    rebinned = cv2.resize(image, (10, 10))
    clahe = image.copy()

    app.save_results_to_dir(
        str(tmp_path),
        [image],
        [[_particle()]],
        [_measurements()],
        ["sample.png"],
        "SAM_b",
        original_images=[image],
        images_rebinned=[rebinned],
        images_clahe=[clahe],
    )

    expected = [
        "images/raw/sample_raw.png",
        "images/processed/sample_processed.png",
        "images/rebin/sample_rebin.png",
        "images/clahe/sample_clahe.png",
        "masks_overlay/sample_masks_overlay.png",
        "raw_masks/sample_mask_full.png",
        "particle_masks/sample_particle_0007_mask.png",
        "metadata/sample_mask_metadata.csv",
        "csv/sample_measurements.csv",
        "csv/combined_measurements.csv",
        "crops/sample_particle_0007.png",
        "distributions/sample_dist.png",
        "distributions/combined_dist.png",
        "interactive_histograms/sample_Equivalent_diameter_nm_hist.png",
    ]
    for relative_path in expected:
        assert (tmp_path / relative_path).is_file(), relative_path

    metadata = pd.read_csv(tmp_path / "metadata/sample_mask_metadata.csv")
    combined = pd.read_csv(tmp_path / "csv/combined_measurements.csv")
    assert metadata.loc[0, "Particle_ID"] == 7
    assert combined.loc[0, "Image"] == "sample.png"
