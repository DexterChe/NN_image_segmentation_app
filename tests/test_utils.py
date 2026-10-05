import os
import tempfile
import unittest

import cv2
import numpy as np

from utils import (
    apply_clahe,
    get_image_files,
    load_image,
    load_image_from_bytes,
    make_folder,
    rebinning,
)


class UtilsTests(unittest.TestCase):
    def test_get_image_files_returns_sorted_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in ["b.png", "a.tif", "c.jpg", "ignore.txt"]:
                path = os.path.join(tmp_dir, name)
                with open(path, "wb") as f:
                    f.write(b"x")

            files = get_image_files(tmp_dir)

            self.assertEqual([os.path.basename(p) for p in files], ["a.tif", "b.png", "c.jpg"])

    def test_load_image_converts_grayscale_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "gray.png")
            gray = np.full((8, 8), 127, dtype=np.uint8)
            cv2.imwrite(path, gray)

            img = load_image(path)

            self.assertEqual(img.shape, (8, 8, 3))
            self.assertEqual(img.dtype, np.uint8)

    def test_rebinning_reduces_image_size(self):
        img = np.zeros((20, 10, 3), dtype=np.uint8)
        rebinned = rebinning(img, 2)
        self.assertEqual(rebinned.shape[:2], (10, 5))

    def test_load_image_normalizes_uint16_and_converts_bgr_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "high_depth.png")
            source = np.zeros((3, 4, 3), dtype=np.uint16)
            source[:, :, 2] = 65535
            cv2.imwrite(path, source)

            image = load_image(path)

            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(tuple(image[0, 0]), (255, 0, 0))

    def test_load_image_from_bytes_handles_rgba_and_invalid_data(self):
        rgba = np.zeros((4, 5, 4), dtype=np.uint8)
        rgba[:, :, 1] = 120
        rgba[:, :, 3] = 255
        encoded_ok, encoded = cv2.imencode(".png", rgba)

        self.assertTrue(encoded_ok)
        image = load_image_from_bytes(encoded.tobytes(), "image.png")
        self.assertEqual(image.shape, (4, 5, 3))
        self.assertEqual(tuple(image[0, 0]), (0, 120, 0))
        self.assertIsNone(load_image_from_bytes(b"not an image", "broken.png"))

    def test_make_folder_and_clahe_return_usable_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = make_folder(tmp_dir, "results")
            image = np.tile(np.arange(20, dtype=np.uint8), (20, 1))
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            processed = apply_clahe(image)

            self.assertTrue(os.path.isdir(output))
            self.assertEqual(processed.shape, image.shape)
            self.assertEqual(processed.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
