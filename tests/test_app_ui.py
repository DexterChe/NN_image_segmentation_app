from streamlit.testing.v1 import AppTest


def test_app_initial_screen_renders_all_workflow_tabs_without_errors():
    at = AppTest.from_file("app.py")

    at.run(timeout=60)

    assert not at.exception
    assert not at.error
    assert [tab.label for tab in at.tabs] == [
        "📘 Tutorial",
        "📁 Load",
        "🔧 Preprocess",
        "🔬 Segmentation",
        "📊 Results",
        "📈 Distributions",
        "💾 Export",
    ]
    assert any("Image Segmentation & Particle Analysis" in markdown.value for markdown in at.markdown)
