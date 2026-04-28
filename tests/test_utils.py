import os
import tempfile
import unittest

import cv2
import numpy as np

from utils import get_image_files, load_image, rebinning


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


if __name__ == "__main__":
    unittest.main()