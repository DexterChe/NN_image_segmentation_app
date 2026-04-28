import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()