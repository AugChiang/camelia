"""Optional real-checkpoint smoke run on synthetic geometric pages, never user images."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    directory = ROOT / ".test-output" / f"smoke-{uuid.uuid4().hex[:8]}"
    inputs, intermediate, outputs = directory / "inputs", directory / "review", directory / "results"
    inputs.mkdir(parents=True)
    page = Image.new("RGB", (192, 128), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((8, 8, 183, 119), outline="black", width=2)
    draw.line((24, 32, 170, 90), fill="black", width=2)
    draw.rectangle((70, 56, 110, 68), fill="black")
    for extension in ("png", "jpg", "webp"):
        page.save(inputs / f"page-{extension}.{extension}")
    command = [sys.executable, str(ROOT / "main.py"), "--model-type", "black_bars",
               "--segment-only", "--no-edit", "--input-dir", str(inputs),
               "--intermediate-dir", str(intermediate), "--device", args.device]
    subprocess.run(command, check=True, cwd=ROOT)
    manifest = json.loads((intermediate / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["images"]) == 3
    mask = np.zeros((128, 192), dtype=np.uint8)
    mask[54:71, 68:113] = 255
    saved = {}
    for entry in manifest["images"]:
        filename = Path(entry["image"]).name
        with Image.open(intermediate / entry["mask"]) as automatic:
            assert automatic.mode == "L"
            assert set(np.unique(automatic)).issubset({0, 255})
        # Simulate a user correcting segmentation before inpainting.
        mask_file = intermediate / entry["mask"]
        Image.fromarray(mask).save(mask_file)
        saved[filename] = mask_file.read_bytes()
    command = [sys.executable, str(ROOT / "main.py"), "--inpaint-only",
               "--intermediate-dir", str(intermediate), "--output-dir", str(outputs),
               "--debug-dir", str(directory / "debug"), "--device", args.device]
    subprocess.run(command, check=True, cwd=ROOT)
    for filename, data in saved.items():
        assert (intermediate / "masks" / filename).read_bytes() == data
        with Image.open(intermediate / "images" / filename) as original, Image.open(outputs / filename) as result:
            before, after = np.array(original), np.array(result)
        np.testing.assert_array_equal(before[mask == 0], after[mask == 0])
        assert np.any(before[mask > 0] != after[mask > 0])
    print(f"Real SMP/LaMa smoke passed; artifacts retained at {directory}")


if __name__ == "__main__":
    main()
