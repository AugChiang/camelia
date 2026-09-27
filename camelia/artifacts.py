"""Persistent, editor-friendly segmentation artifacts."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def image_jobs(input_dir, output_dir, overwrite=False, directories=("images", "masks", "probabilities")):
    inputs, outputs = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not inputs.is_dir():
        raise ValueError(f"Input directory does not exist: {inputs}")
    if outputs == inputs or inputs in outputs.parents or outputs in inputs.parents:
        raise ValueError("Input and output directories must not overlap")
    jobs, names = [], set()
    for source in sorted(inputs.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
            continue
        relative = source.relative_to(inputs).with_suffix(".png")
        key = relative.as_posix().casefold()
        if key in names:
            raise ValueError(f"Inputs map to the same PNG filename: {relative}")
        names.add(key)
        for directory in directories:
            target = outputs / directory / relative if directory else outputs / relative
            if target.exists() and not overwrite:
                raise ValueError(f"Refusing to overwrite {target}; choose another directory or use --overwrite")
        jobs.append((source, relative))
    if not jobs:
        raise ValueError(f"No supported images found in {inputs}")
    return jobs


def read_rgb(path):
    # Apply orientation once so the retained image and editable mask use the same coordinates.
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


def save_png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def save_segmentation(output_dir, relative, image, probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != image.shape[:2] or not np.isfinite(probabilities).all():
        raise ValueError("Segmentation probabilities must be finite and match the retained image size")
    probabilities = np.clip(probabilities, 0, 1)
    output_dir = Path(output_dir)
    save_png(output_dir / "images" / relative, image)
    save_png(output_dir / "probabilities" / relative, np.rint(probabilities * 255).astype(np.uint8))
    mask = np.where(probabilities >= threshold, 255, 0).astype(np.uint8)
    save_png(output_dir / "masks" / relative, mask)


def write_manifest(output_dir, entries, backend, model_type, threshold):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    retained = {entry["image"]: entry for entry in previous.get("images", [])}
    for entry in entries:
        retained[entry["image"]] = dict(entry, backend=backend, model_type=model_type,
                                       mask_threshold=threshold)
    manifest = {
        "version": 1, "backend": backend, "model_type": model_type,
        "mask_threshold": threshold,
        "mask_format": {"mode": "L", "background": 0, "inpaint": 255},
        "images": list(retained.values()),
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
