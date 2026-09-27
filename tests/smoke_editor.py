"""Optional desktop smoke test: real Tk canvas events, binary saving, and LaMa handoff."""

import argparse
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camelia.editor import MaskEditor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inpaint", action="store_true", help="Also test the saved masks with real LaMa weights.")
    args = parser.parse_args()
    review = ROOT / ".test-output" / f"editor-{uuid.uuid4().hex[:8]}"
    images, masks = review / "images", review / "masks"
    (images / "chapter").mkdir(parents=True)
    (masks / "chapter").mkdir(parents=True)
    page = Image.new("RGB", (192, 128), "white")
    ImageDraw.Draw(page).rectangle((70, 56, 110, 68), fill="black")
    page.save(images / "chapter/page-01.png")
    page.save(images / "chapter/page-02.png")
    Image.new("L", page.size, 0).save(masks / "chapter/page-01.png")
    Image.new("L", page.size, 0).save(masks / "chapter/page-02.png")

    root = tk.Tk()
    root.withdraw()
    # Exercise real mapped widgets without showing a test window on Windows.
    root.attributes("-alpha", 0)
    try:
        editor = MaskEditor(root, images, masks)
        root.deiconify()
        root.update()
        editor.brush_slider.set(17)
        assert editor.brush_size.get() == 17
        editor.set_zoom(2)
        root.update()

        def stroke(start, end, button=1):
            def coordinates(point):
                return (round(point[0] * editor.scale - editor.canvas.canvasx(0)),
                        round(point[1] * editor.scale - editor.canvas.canvasy(0)))
            x0, y0 = coordinates(start)
            x1, y1 = coordinates(end)
            editor.canvas.event_generate(f"<ButtonPress-{button}>", x=x0, y=y0)
            editor.canvas.event_generate(f"<B{button}-Motion>", x=x1, y=y1)
            editor.canvas.event_generate(f"<ButtonRelease-{button}>", x=x1, y=y1)
            root.update()

        stroke((68, 62), (113, 62))
        assert editor.document.mask.getpixel((90, 62)) == 255
        editor.render()
        assert editor._photo is not None
        painted = editor.document.mask.tobytes()
        editor.undo()
        assert not np.any(editor.document.mask)
        editor.redo()
        assert editor.document.mask.tobytes() == painted
        editor.save_button.invoke()
        assert not editor.document.dirty
        editor.next_button.invoke()
        root.update()
        assert editor.index == 1
        assert not np.any(editor.document.mask)
        editor.set_zoom(0.75)
        root.update()
        stroke((68, 62), (113, 62))
        # An unsaved page cannot be lost when navigation is cancelled.
        with patch.object(editor.messagebox, "askyesnocancel", return_value=None):
            editor.previous_button.invoke()
        assert editor.index == 1
        editor.save_button.invoke()
        editor.previous_button.invoke()
        root.update()
        assert editor.index == 0
        assert editor.document.mask.tobytes() == painted
        # Right-button eraser events modify only the mask and are undoable.
        stroke((90, 62), (90, 62), button=3)
        assert editor.document.mask.getpixel((90, 62)) == 0
        editor.undo()
        assert editor.document.mask.tobytes() == painted
        editor.save_button.invoke()
        editor.render()
        assert editor._photo.width() <= editor.canvas.winfo_width() + 16
    finally:
        root.destroy()

    for mask_path in sorted(masks.rglob("*.png")):
        with Image.open(mask_path) as mask:
            assert mask.mode == "L" and mask.size == page.size
            assert set(np.unique(mask)).issubset({0, 255})
    if args.inpaint:
        outputs = review / "results"
        subprocess.run([sys.executable, str(ROOT / "main.py"), "--inpaint-only",
                        "--intermediate-dir", str(review), "--output-dir", str(outputs)],
                       check=True, cwd=ROOT)
        for source in sorted(images.rglob("*.png")):
            relative = source.relative_to(images)
            with Image.open(source) as image, Image.open(masks / relative) as mask, Image.open(outputs / relative) as result:
                before, selected, after = np.array(image), np.array(mask) > 0, np.array(result)
            np.testing.assert_array_equal(before[~selected], after[~selected])
            assert np.any(before[selected] != after[selected])
    print(f"Desktop brush editor smoke passed; artifacts retained at {review}")


if __name__ == "__main__":
    main()
