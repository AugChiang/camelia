"""Common argument handling and artifact writing for segmentation adapters."""

import argparse
from pathlib import Path
import sys
import time



MODEL_TYPES = ("black_bars", "white_bars", "transparent_black")


def arguments(backend, root, argv=None):
    parser = argparse.ArgumentParser(description=f"{backend.upper()} segmentation with retained hard masks.")
    parser.add_argument("--model-type", "--model_type", required=True, choices=MODEL_TYPES)
    parser.add_argument("--input-dir", "--input_dir", type=Path,
                        help="Directory containing images, scanned recursively.")
    parser.add_argument("--output-dir", "--output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.mask_threshold < 1:
        parser.error("--mask-threshold must be between 0 and 1 (exclusive)")
    args.input_dir = (args.input_dir or root / "camelia-decensor/input" / args.model_type).resolve()
    args.output_dir = (args.output_dir or root / "camelia-decensor/intermediate" / args.model_type).resolve()
    return args


def run(args, backend, load_predictor):
    """Load a single model per invocation and decode each source only once."""
    try:
        from camelia.artifacts import image_jobs, read_rgb, save_segmentation, write_manifest
        jobs = image_jobs(args.input_dir, args.output_dir, args.overwrite)
        predictor = load_predictor(args)
        entries, errors = [], []
        for index, (source, relative) in enumerate(jobs, 1):
            started = time.perf_counter()
            print(f"[{index}/{len(jobs)}] Segmenting {source}", flush=True)
            try:
                image = read_rgb(source)
                probabilities = predictor(image)
                save_segmentation(args.output_dir, relative, image, probabilities, args.mask_threshold)
                entries.append({"source": str(source), "image": f"images/{relative.as_posix()}",
                                "mask": f"masks/{relative.as_posix()}",
                                "probabilities": f"probabilities/{relative.as_posix()}",
                                "width": image.shape[1], "height": image.shape[0]})
                print(f"Saved {relative} in {time.perf_counter() - started:.2f}s", flush=True)
            except Exception as exc:
                errors.append(source)
                print(f"Error processing {source}: {exc}", file=sys.stderr, flush=True)
        if entries:
            write_manifest(args.output_dir, entries, backend, args.model_type, args.mask_threshold)
        if errors:
            print(f"Segmentation failed for {len(errors)} image(s); successful artifacts are retained.", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"Segmentation error: {exc}", file=sys.stderr, flush=True)
        return 1
