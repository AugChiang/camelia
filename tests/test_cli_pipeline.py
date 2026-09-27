"""Regression coverage for retained masks, crop boundaries, and CLI stage behavior."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camelia.artifacts import image_jobs, save_png, save_segmentation, write_manifest
from camelia.inpainting import inpaint_regions
from camelia.masks import crop_with_padding, hard_mask, region_boxes


class MaskTests(unittest.TestCase):
    def test_four_connected_regions_keep_tiny_selections(self):
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:5, 3:8] = 1
        mask[0, 0] = 1
        mask[1, 1] = 1  # Diagonal pixels remain separate with four-way connectivity.
        self.assertEqual(region_boxes(mask), [(3, 2, 8, 5), (0, 0, 1, 1), (1, 1, 2, 2)])

    def test_crop_matches_reflection_without_full_image_padding(self):
        image = np.arange(7 * 9 * 3).reshape(7, 9, 3)
        for box in [(-4, -3, 12, 10), (2, 1, 7, 6), (-20, -20, 25, 23)]:
            x0, y0, x1, y1 = box
            padding = max(abs(x0), abs(y0), x1, y1) + 1
            reference = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
            reference = reference[padding+y0:padding+y1, padding+x0:padding+x1]
            np.testing.assert_array_equal(crop_with_padding(image, box), reference)
        self.assertEqual(crop_with_padding(np.ones((1, 1, 3)), (-32, -32, 32, 32)).shape, (64, 64, 3))

    def test_threshold_and_size_validation(self):
        mask = np.array([[0, 127, 128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(hard_mask(mask, (1, 4)), [[0, 0, 1, 1]])
        with self.assertRaises(ValueError):
            hard_mask(mask, (4, 1))

    def test_inpainting_preserves_all_unselected_pixels(self):
        image = np.random.default_rng(1).integers(0, 256, (19, 23, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[0, 0] = 255
        mask[8:12, 9:15] = 255
        output, debug = inpaint_regions(lambda image, mask: np.ones_like(image, dtype=np.float32), image, mask)
        np.testing.assert_array_equal(output[mask == 0], image[mask == 0])
        self.assertTrue(np.all(output[mask == 255] == 255))
        self.assertIsNone(debug)

    def test_empty_mask_does_not_invoke_model(self):
        image = np.ones((1, 1, 3), dtype=np.uint8) * 87
        output, debug = inpaint_regions(None, image, np.zeros((1, 1), dtype=np.uint8), debug=True)
        np.testing.assert_array_equal(output, image)
        self.assertEqual(debug.shape, (1, 2, 3))

    def test_feathering_and_dilation_do_not_modify_saved_mask(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        original = mask.copy()
        output, _ = inpaint_regions(lambda image, mask: np.ones_like(image, dtype=np.float32),
                                    image, mask, blend_radius=3)
        self.assertTrue(np.all(output[mask == 0] == 0))
        self.assertGreater(int(output[5, 5, 0]), 0)
        self.assertLess(int(output[5, 5, 0]), 255)
        self.assertEqual(int(output[10, 10, 0]), 255)
        inpaint_regions(lambda image, mask: np.zeros_like(image, dtype=np.float32), image, mask, dilation=2)
        np.testing.assert_array_equal(mask, original)


class ArtifactAndCliTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".test-output"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(parent.resolve() in self.directory.parents)

    def tearDown(self):
        self.temporary.cleanup()

    def test_retained_masks_are_binary_and_probabilities_are_absolute(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        probability = np.array([[0.1, 0.4], [0.5, 0.6]], dtype=np.float32)
        relative = Path("chapter/page.png")
        save_segmentation(self.directory, relative, image, probability, 0.5)
        with Image.open(self.directory / "masks" / relative) as mask:
            self.assertEqual(mask.mode, "L")
            np.testing.assert_array_equal(np.array(mask), [[0, 0], [255, 255]])
        with Image.open(self.directory / "probabilities" / relative) as preview:
            self.assertEqual(int(np.array(preview)[1, 1]), 153)  # No per-image max normalization.

    def test_duplicate_stems_and_edited_masks_are_protected(self):
        inputs, output = self.directory / "inputs", self.directory / "output"
        inputs.mkdir()
        (inputs / "page.jpg").touch()
        (inputs / "page.png").touch()
        with self.assertRaisesRegex(ValueError, "same PNG"):
            image_jobs(inputs, output)
        (inputs / "page.jpg").unlink()
        save_png(output / "masks/page.png", np.ones((2, 2), dtype=np.uint8) * 255)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            image_jobs(inputs, output)
        self.assertEqual(len(image_jobs(inputs, output, overwrite=True)), 1)

    def test_manifest_retains_previous_review_pages(self):
        import json
        write_manifest(self.directory, [{"image": "images/first.png", "mask": "masks/first.png"}],
                       "smp", "black_bars", 0.5)
        write_manifest(self.directory, [{"image": "images/second.png", "mask": "masks/second.png"}],
                       "sam3", "white_bars", 0.4)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["images"]), 2)
        self.assertEqual(manifest["images"][0]["backend"], "smp")
        self.assertEqual(manifest["images"][1]["mask_threshold"], 0.4)

    def test_cli_inpaint_only_preserves_intermediates_and_needs_no_model_for_empty_mask(self):
        image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
        intermediates, outputs = self.directory / "intermediates", self.directory / "results"
        save_png(intermediates / "images/chapter/page.png", image)
        save_png(intermediates / "masks/chapter/page.png", np.zeros((8, 9), dtype=np.uint8))
        mask_file = intermediates / "masks/chapter/page.png"
        before = mask_file.read_bytes()
        command = [sys.executable, str(ROOT / "main.py"), "--inpaint-only",
                   "--intermediate-dir", str(intermediates), "--output-dir", str(outputs),
                   "--checkpoint", str(self.directory / "no-checkpoint")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with Image.open(outputs / "chapter/page.png") as actual:
            np.testing.assert_array_equal(np.array(actual), image)
        self.assertEqual(mask_file.read_bytes(), before)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overwrite", result.stderr)

    def test_cli_stage_failure_is_nonzero(self):
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--segment-only",
                                 "--model-type", "black_bars", "--input-dir", str(self.directory / "missing")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Segmentation failed", result.stderr)

    def test_cli_rejects_missing_or_mismatched_masks_before_writing(self):
        intermediates, outputs = self.directory / "intermediates", self.directory / "results"
        save_png(intermediates / "images/page.png", np.zeros((8, 9, 3), dtype=np.uint8))
        command = [sys.executable, str(ROOT / "main.py"), "--inpaint-only",
                   "--intermediate-dir", str(intermediates), "--output-dir", str(outputs)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing mask", result.stderr)
        save_png(intermediates / "masks/page.png", np.zeros((9, 8), dtype=np.uint8))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Mask size", result.stderr)
        self.assertFalse(outputs.exists())

    def test_cli_cannot_write_results_over_edited_masks(self):
        intermediates = self.directory / "intermediates"
        save_png(intermediates / "images/page.png", np.zeros((8, 9, 3), dtype=np.uint8))
        mask_file = intermediates / "masks/page.png"
        save_png(mask_file, np.zeros((8, 9), dtype=np.uint8))
        before = mask_file.read_bytes()
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--inpaint-only",
                                 "--intermediate-dir", str(intermediates),
                                 "--output-dir", str(intermediates / "masks"), "--overwrite"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not overlap", result.stderr)
        self.assertEqual(mask_file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
