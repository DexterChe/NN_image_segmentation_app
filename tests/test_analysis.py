import numpy as np
import pytest

from analysis import analyze_particles, measure_particle, minimal_feret_diameter
from segmentation import ParticleData


def _particle(index: int, mask: np.ndarray) -> ParticleData:
    return ParticleData(
        index=index,
        mask_xy=np.array([[0, 0], [1, 0], [1, 1]]),
        box_xyxy=np.array([0, 0, 1, 1]),
        crop_original=np.zeros((1, 1, 3), dtype=np.uint8),
        crop_with_mask=np.zeros((1, 1, 3), dtype=np.uint8),
        mask_binary=mask,
    )


def test_minimal_feret_diameter_returns_zero_for_empty_mask():
    assert minimal_feret_diameter(np.zeros((8, 8), dtype=np.uint8)) == 0.0


def test_measure_particle_scales_pixel_measurements_to_nanometres():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:7] = 255

    measurement = measure_particle(mask, px_size=2.5)

    assert measurement is not None
    assert measurement["Area_px"] == 16
    assert measurement["Area_nm2"] == 100.0
    assert measurement["Equivalent_diameter_nm"] == pytest.approx(
        measurement["Equivalent_diameter_px"] * 2.5
    )
    assert measurement["Min_Feret_diameter_px"] > 0


def test_analyze_particles_keeps_particle_ids_and_reports_progress():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    calls = []

    result = analyze_particles(
        [_particle(4, mask), _particle(9, np.zeros((8, 8), dtype=np.uint8))],
        px_size=1.0,
        progress_callback=lambda current, total: calls.append((current, total)),
    )

    assert result["Particle_ID"].tolist() == [4]
    assert result.loc[0, "Area_px"] == 16
    assert calls == [(1, 2), (2, 2)]
