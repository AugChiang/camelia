"""CLI orchestration for segmentation, mask review, and LaMa inpainting."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
MODEL_TYPES = ("black_bars", "white_bars", "transparent_black")


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", "--model_type", choices=MODEL_TYPES)
    parser.add_argument("--seg-model-type", choices=("smp", "sam3"), default="smp")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--segment-only", "--segmentation-only", action="store_true",
                      help="Save segmentation artifacts and open the brush editor; skip inpainting.")
    mode.add_argument("--inpaint-only", action="store_true",
                      help="Use retained images and masks without rerunning segmentation.")
    mode.add_argument("--edit-masks", action="store_true",
                      help="Open the brush editor for retained images and masks, without inference.")
    parser.add_argument("--no-edit", action="store_true",
                        help="Save segmentation artifacts without opening the editor (--segment-only).")
    parser.add_argument("--input-dir", type=Path, help="Directory containing input images (recursive).")
    parser.add_argument("--intermediate-dir", type=Path,
                        help="Persistent images/masks/probabilities directory.")
    parser.add_argument("--mask-dir", type=Path, help="Mask directory for --inpaint-only or --edit-masks.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "lama-inpainting/pretrained/best")
    parser.add_argument("--seg-checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Segmentation probability cutoff; saved masks contain only 0 and 255.")
    parser.add_argument("--mask-dilation", type=int, default=0,
                        help="Expand the inpainting mask by this many pixels; saved masks stay intact.")
    parser.add_argument("--blend-radius", type=int, default=0,
                        help="Feather inside mask edges; 0 replaces the entire selected region.")
    parser.add_argument("--context-scale", type=float, default=1.2,
                        help="Crop side / region's longest side; must be at least 1.")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing masks, intermediate images, and outputs.")
    parser.add_argument("--seg-python", default=sys.executable,
                        help="Segmentation interpreter (allows a separate SAM3 environment).")
    parser.add_argument("--inpaint-python", default=sys.executable,
                        help="LaMa interpreter, defaulting to the current environment.")
    args = parser.parse_args(argv)
    if not (args.inpaint_only or args.edit_masks) and args.model_type is None:
        parser.error("--model-type is required for segmentation")
    if (args.inpaint_only or args.edit_masks) and args.model_type is None and args.intermediate_dir is None:
        parser.error("--inpaint-only and --edit-masks require --intermediate-dir or --model-type")
    if args.mask_dir is not None and not (args.inpaint_only or args.edit_masks):
        parser.error("--mask-dir requires --inpaint-only or --edit-masks")
    if args.no_edit and not args.segment_only:
        parser.error("--no-edit requires --segment-only")
    if not 0 < args.mask_threshold < 1:
        parser.error("--mask-threshold must be between 0 and 1 (exclusive)")
    if args.mask_dilation < 0 or args.blend_radius < 0:
        parser.error("--mask-dilation and --blend-radius must be nonnegative")
    if args.context_scale < 1 or not args.context_scale < float("inf"):
        parser.error("--context-scale must be finite and at least 1")
    return args


def run_stage(name, command, cwd):
    """Inherit console output and isolate stage environments without changing our cwd."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["WORKSPACE_ROOT"] = str(ROOT)
    paths = [str(ROOT), str(ROOT / "lama-inpainting")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    print(f"Running {name}...", flush=True)
    started = time.perf_counter()
    completed = subprocess.run([str(arg) for arg in command], cwd=str(cwd), env=env)
    if completed.returncode:
        raise RuntimeError(f"{name} failed (exit code {completed.returncode})")
    print(f"{name} completed in {time.perf_counter() - started:.2f}s", flush=True)


def main(argv=None):
    args = cli(argv)
    name = args.model_type or args.intermediate_dir.name
    intermediates = (args.intermediate_dir or ROOT / "camelia-decensor/intermediate" / name).resolve()
    outputs = (args.output_dir or ROOT / "camelia-decensor/output" / name).resolve()
    try:
        if not (args.inpaint_only or args.edit_masks):
            inputs = (args.input_dir or ROOT / "camelia-decensor/input" / name).resolve()
            adapter = ROOT / f"{args.seg_model_type}-segmentation/run_segmentation.py"
            command = [args.seg_python, "-u", adapter, "--model-type", args.model_type,
                       "--input-dir", inputs, "--output-dir", intermediates,
                       "--device", args.device, "--mask-threshold", args.mask_threshold]
            if args.seg_checkpoint:
                command.extend(["--checkpoint", args.seg_checkpoint.resolve()])
            if args.overwrite:
                command.append("--overwrite")
            run_stage("Segmentation", command, adapter.parent)
        masks = (args.mask_dir or intermediates / "masks").resolve()
        print(f"Retained masks: {masks}", flush=True)
        if args.edit_masks or (args.segment_only and not args.no_edit):
            print("Opening brush editor. Save masks, then close it to return to the CLI.", flush=True)
            try:
                from camelia.editor import launch_editor
                launch_editor(intermediates / "images", masks)
            except (ImportError, OSError, ValueError) as exc:
                raise RuntimeError(f"Could not open mask editor: {exc}") from exc
        if args.segment_only or args.edit_masks:
            print("Run --inpaint-only to process the saved masks.", flush=True)
            return 0
        command = [args.inpaint_python, "-u", ROOT / "lama-inpainting/bin/uncen.py",
                   "--in-dir", intermediates / "images", "--mask-dir", masks,
                   "--out-dir", outputs, "--checkpoint", args.checkpoint.resolve(),
                   "--device", args.device, "--mask-dilation", args.mask_dilation,
                   "--blend-radius", args.blend_radius, "--context-scale", args.context_scale]
        if args.debug_dir:
            command.extend(["--debug-dir", args.debug_dir.resolve()])
        if args.overwrite:
            command.append("--overwrite")
        run_stage("Inpainting", command, ROOT / "lama-inpainting")
        print(f"Results: {outputs}", flush=True)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Saved intermediate files are retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())