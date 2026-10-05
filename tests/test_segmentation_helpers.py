import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from segmentation import (
    MODEL_FILES,
    find_models_in_folder,
    get_model_filename,
    resolve_model_path,
)


class SegmentationHelperTests(unittest.TestCase):
    def test_get_model_filename_returns_expected_name(self):
        self.assertEqual(get_model_filename("SAM_b"), "sam_b.pt")
        self.assertEqual(get_model_filename("YOLOv8s"), "yolov8s-seg.pt")

    def test_find_models_in_folder_detects_known_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sam_path = os.path.join(tmp_dir, MODEL_FILES["SAM_b"])
            yolo_path = os.path.join(tmp_dir, MODEL_FILES["YOLOv8n"])
            with open(sam_path, "wb") as f:
                f.write(b"sam")
            with open(yolo_path, "wb") as f:
                f.write(b"yolo")

            found = find_models_in_folder(tmp_dir)

            self.assertEqual(found["SAM_b"], sam_path)
            self.assertEqual(found["YOLOv8n"], yolo_path)

    def test_resolve_model_path_prefers_local_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, MODEL_FILES["MobileSAM"])
            with open(local_path, "wb") as f:
                f.write(b"model")

            resolved = resolve_model_path("MobileSAM", tmp_dir)

            self.assertEqual(resolved, local_path)

    def test_resolve_model_path_falls_back_to_filename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            resolved = resolve_model_path("FastSAM_x", tmp_dir)
            self.assertEqual(resolved, MODEL_FILES["FastSAM_x"])

    def test_get_model_filename_rejects_unknown_model(self):
        with self.assertRaisesRegex(KeyError, "Unknown model name"):
            get_model_filename("not-a-model")

    def test_load_model_selects_matching_ultralytics_loader(self):
        from segmentation import load_model

        calls = []

        class FakeSAM:
            def __init__(self, path):
                calls.append(("SAM", path))

        class FakeFastSAM:
            def __init__(self, path):
                calls.append(("FastSAM", path))

        class FakeYOLO:
            def __init__(self, path):
                calls.append(("YOLO", path))

        fake_ultralytics = type(
            "FakeUltralytics", (), {"SAM": FakeSAM, "FastSAM": FakeFastSAM, "YOLO": FakeYOLO}
        )
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            load_model("SAM_b", "sam.pt")
            load_model("FastSAM_s", "fastsam.pt")
            load_model("YOLOv8n", "yolo.pt")

        self.assertEqual(calls, [("SAM", "sam.pt"), ("FastSAM", "fastsam.pt"), ("YOLO", "yolo.pt")])

    def test_get_available_devices_reports_cpu_cuda_and_mps(self):
        import segmentation

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        with patch.object(segmentation, "torch", fake_torch):
            self.assertEqual(segmentation.get_available_devices(), ["cpu", "cuda:0", "cuda:1", "mps"])


if __name__ == "__main__":
    unittest.main()
