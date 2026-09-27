"""Editor regression tests without a desktop window or model weights."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import main as pipeline
from camelia.editor import MaskDocument, editor_pages


ROOT = Path(__file__).resolve().parents[1]


class MaskEditorTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".test-output"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(parent.resolve() in self.directory.parents)
        self.images = self.directory / "images"
        self.masks = self.directory / "masks"
        self.image_path = self.images / "chapter/page.png"
        self.mask_path = self.masks / "chapter/page.png"
        self.image_path.parent.mkdir(parents=True)
        Image.new("RGB", (64, 48), "white").save(self.image_path)
        self.mask_path.parent.mkdir(parents=True)
        Image.new("L", (64, 48), 0).save(self.mask_path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_brush_interpolates_fast_strokes_and_eraser_preserves_other_pixels(self):
        document = MaskDocument(self.image_path, self.mask_path)
        document.brush((5, 24), (58, 24), 5)
        document.end_stroke()
        mask = np.array(document.mask)
        self.assertTrue(np.all(mask[24, 5:59] == 255))
        self.assertTrue(np.all(mask[:20] == 0))
        self.assertEqual(set(np.unique(mask)), {0, 255})
        document.brush((30, 24), (30, 24), 7, erase=True)
        document.end_stroke()
        self.assertEqual(document.mask.getpixel((30, 24)), 0)
        self.assertEqual(document.mask.getpixel((10, 24)), 255)

    def test_one_pixel_and_image_edge_strokes(self):
        document = MaskDocument(self.image_path, self.mask_path)
        document.brush((0, 0), (0, 0), 1)
        document.end_stroke()
        self.assertEqual(np.count_nonzero(document.mask), 1)
        document.brush((63, 47), (63, 47), 20)
        document.end_stroke()
        self.assertEqual(document.mask.size, (64, 48))
        self.assertEqual(document.mask.getpixel((63, 47)), 255)

    def test_undo_redo_restore_strokes_and_saved_dirty_state(self):
        document = MaskDocument(self.image_path, self.mask_path)
        document.brush((20, 20), (30, 20), 9)
        document.end_stroke()
        painted = document.mask.tobytes()
        self.assertTrue(document.dirty)
        document.undo()
        self.assertFalse(document.dirty)
        self.assertFalse(np.any(document.mask))
        document.redo()
        self.assertEqual(document.mask.tobytes(), painted)
        document.save()
        self.assertFalse(document.dirty)
        document.clear()
        self.assertTrue(document.dirty)
        document.undo()
        self.assertEqual(document.mask.tobytes(), painted)
        self.assertFalse(document.dirty)

    def test_overlay_is_transparent_and_saved_mask_is_binary_at_original_size(self):
        original = self.image_path.read_bytes()
        document = MaskDocument(self.image_path, self.mask_path)
        document.brush((10, 10), (40, 30), 8)
        document.end_stroke()
        preview = document.preview()
        self.assertEqual(preview.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(preview.getpixel((10, 10))[0], 255)
        self.assertGreater(preview.getpixel((10, 10))[1], 64)
        self.assertLess(preview.getpixel((10, 10))[1], 255)
        document.save()
        with Image.open(self.mask_path) as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (64, 48))
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertEqual(mask.tobytes(), document.mask.tobytes())
        self.assertEqual(self.image_path.read_bytes(), original)
        self.assertEqual(list(self.mask_path.parent.glob("*.tmp")), [])

    def test_page_discovery_keeps_nested_paths_and_missing_masks_start_empty(self):
        other_image = self.images / "another-page.png"
        Image.new("RGB", (17, 19), "black").save(other_image)
        pages = editor_pages(self.images, self.masks)
        self.assertEqual(len(pages), 2)
        self.assertIn((self.image_path, self.mask_path), pages)
        document = MaskDocument(other_image, self.masks / "another-page.png")
        self.assertFalse(np.any(document.mask))
        self.assertTrue(document.dirty)
        document.save()
        self.assertTrue((self.masks / "another-page.png").is_file())
        self.assertEqual(np.count_nonzero(Image.open(self.mask_path)), 0)

    def test_bad_mask_dimensions_and_original_image_destinations_are_rejected(self):
        Image.new("L", (48, 64), 0).save(self.mask_path)
        with self.assertRaisesRegex(ValueError, "Mask size"):
            MaskDocument(self.image_path, self.mask_path)
        with self.assertRaisesRegex(ValueError, "original image"):
            MaskDocument(self.image_path, self.image_path)
        with self.assertRaisesRegex(ValueError, "overlap"):
            editor_pages(self.images, self.images)

    def test_segment_only_opens_editor_and_no_edit_keeps_batch_mode(self):
        arguments = ["--model-type", "black_bars", "--segment-only",
                     "--intermediate-dir", str(self.directory)]
        with patch.object(pipeline, "run_stage") as stage, patch("camelia.editor.launch_editor") as editor:
            self.assertEqual(pipeline.main(arguments), 0)
            stage.assert_called_once()
            editor.assert_called_once_with(self.images, self.masks)
        with patch.object(pipeline, "run_stage") as stage, patch("camelia.editor.launch_editor") as editor:
            self.assertEqual(pipeline.main(arguments + ["--no-edit"]), 0)
            stage.assert_called_once()
            editor.assert_not_called()

    def test_edit_masks_reopens_retained_pages_without_segmentation_or_inpainting(self):
        custom_masks = self.directory / "edited"
        with patch.object(pipeline, "run_stage") as stage, patch("camelia.editor.launch_editor") as editor:
            self.assertEqual(pipeline.main(["--edit-masks", "--intermediate-dir", str(self.directory),
                                            "--mask-dir", str(custom_masks)]), 0)
            stage.assert_not_called()
            editor.assert_called_once_with(self.images, custom_masks)


if __name__ == "__main__":
    unittest.main()
