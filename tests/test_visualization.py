import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segmentation import ParticleData
from visualization import (
    _particle_color,
    create_mask_overlay_image,
    create_summary_stats,
    fig_to_bytes,
    plot_comparison,
    plot_distributions,
    plot_interactive_histogram,
    plot_particle_crops,
)


def _particle(index=2):
    return ParticleData(
        index=index,
        mask_xy=np.array([[2, 2], [6, 2], [6, 6], [2, 6]]),
        box_xyxy=np.array([2, 2, 6, 6]),
        crop_original=np.zeros((4, 4, 3), dtype=np.uint8),
        crop_with_mask=np.ones((4, 4, 3), dtype=np.uint8) * 120,
        mask_binary=np.ones((4, 4), dtype=np.uint8) * 255,
    )


def test_mask_overlay_is_deterministic_and_changes_masked_pixels():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    particle = _particle()

    overlay_one = create_mask_overlay_image(image, [particle], alpha=1.0, draw_ids=False)
    overlay_two = create_mask_overlay_image(image, [particle], alpha=1.0, draw_ids=False)

    assert _particle_color(2) == _particle_color(2)
    assert np.array_equal(overlay_one, overlay_two)
    assert tuple(overlay_one[4, 4]) == _particle_color(2)
    assert np.array_equal(overlay_one[0, 0], [0, 0, 0])


def test_summary_stats_only_uses_physical_measurements_and_rounds_values():
    frame = pd.DataFrame({"Particle_ID": [0, 1], "Area_nm2": [1.111, 2.222], "Minor_axis_length_nm": [3.0, 5.0]})

    stats = create_summary_stats(frame)

    assert stats.index.tolist() == ["Area (nm²)", "Minor Axis (nm)"]
    assert stats.loc["Area (nm²)", "Mean"] == 1.67
    assert "Particle_ID" not in stats.index


def test_interactive_histogram_handles_missing_and_valid_properties():
    frame = pd.DataFrame({"Equivalent_diameter_nm": [2.0, 4.0, 6.0]})

    assert plot_interactive_histogram(frame, "missing", "Missing") is None
    figure = plot_interactive_histogram(frame, "Equivalent_diameter_nm", "Diameter")
    try:
        assert len(figure.axes) == 1
        assert figure.axes[0].get_xlabel() == "Diameter"
    finally:
        plt.close(figure)


def test_particle_crops_empty_and_figure_export():
    assert plot_particle_crops([], pd.DataFrame()) is None
    figure = plot_particle_crops([_particle()], pd.DataFrame({"Particle_ID": [2], "Equivalent_diameter_nm": [7.5]}))
    try:
        png = fig_to_bytes(figure)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_distribution_and_comparison_figures_render_all_axes():
    measurements = pd.DataFrame({
        "Area_nm2": [1.0, 2.0, 3.0],
        "Major_axis_length_nm": [2.0, 3.0, 4.0],
        "Minor_axis_length_nm": [1.0, 1.5, 2.0],
        "Equivalent_diameter_nm": [1.2, 1.6, 2.0],
        "Max_Feret_diameter_nm": [2.1, 3.1, 4.1],
        "Min_Feret_diameter_nm": [0.9, 1.2, 1.7],
    })
    distribution = plot_distributions(measurements, "SAM_b", "sample.png", x_limits=(0.0, 5.0))
    comparison = plot_comparison(np.zeros((3, 3, 3)), np.ones((3, 3, 3)))
    try:
        assert len(distribution.axes) == 6
        assert all(axis.get_xlim() == (0.0, 5.0) for axis in distribution.axes)
        assert len(comparison.axes) == 2
    finally:
        plt.close(distribution)
        plt.close(comparison)
