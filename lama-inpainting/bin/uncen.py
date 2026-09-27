"""LaMa CLI using retained or manually edited hard masks."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lama-inpainting"))



def init_inpaint_model(model_path, device):
    """Instantiate only the generator, avoiding training modules, losses, and datasets."""
    import torch
    from omegaconf import OmegaConf
    from saicinpainting.training.modules import make_generator

    model_path = Path(model_path)
    config = OmegaConf.load(model_path / "config.yaml")
    if not config.training_model.get("concat_mask", True) or config.training_model.get("add_noise_kwargs"):
        raise ValueError("This CLI requires a generator trained with an RGB image plus binary mask")
    generator_config = OmegaConf.to_container(config.generator, resolve=True)
    model = make_generator(config, **generator_config)
    checkpoint = torch.load(str(model_path / "models/best.ckpt"), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    generator_state = {key[len("generator."):]: value for key, value in state.items()
                       if key.startswith("generator.")}
    if not generator_state:
        raise ValueError("Checkpoint does not contain generator weights")
    model.load_state_dict(generator_state, strict=True)
    del checkpoint, state, generator_state
    return model.to(device).eval().requires_grad_(False)


def predictor(model, device):
    import numpy as np
    import torch

    @torch.inference_mode()
    def predict(image, mask):
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(device, dtype=torch.float32) / 255
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        inputs = torch.cat([image_tensor * (1 - mask_tensor), mask_tensor], dim=1)
        return model(inputs)[0].permute(1, 2, 0).cpu().numpy()
    return predict


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", "--in_dir", required=True, type=Path)
    parser.add_argument("--mask-dir", "--mask_dir", required=True, type=Path)
    parser.add_argument("--out-dir", "--out_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug-dir", "--debug_dir", type=Path)
    parser.add_argument("--mask-dilation", type=int, default=0)
    parser.add_argument("--blend-radius", type=int, default=0)
    parser.add_argument("--context-scale", type=float, default=1.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.mask_dilation < 0 or args.blend_radius < 0:
        parser.error("Dilation and blend radius must be nonnegative")
    if args.context_scale < 1 or not args.context_scale < float("inf"):
        parser.error("Context scale must be finite and at least 1")
    return args


def read_mask(path, shape):
    import numpy as np
    from PIL import Image
    from camelia.masks import hard_mask
    with Image.open(path) as image:
        mask = np.array(image.convert("L"))
    hard_mask(mask, shape)  # Validate dimensions before inference; never silently resize an edit.
    return mask


def main(argv=None):
    args = cli(argv)
    try:
        import numpy as np
        from PIL import Image
        from camelia.artifacts import image_jobs, read_rgb, save_png
        from camelia.inpainting import inpaint_regions

        jobs = image_jobs(args.in_dir, args.out_dir, args.overwrite, directories=("",))
        masks = args.mask_dir.resolve()
        outputs = args.out_dir.resolve()
        if outputs == masks or masks in outputs.parents or outputs in masks.parents:
            raise ValueError("Output and mask directories must not overlap; retained masks are never outputs")
        if args.debug_dir:
            debug = args.debug_dir.resolve()
            inputs = args.in_dir.resolve()
            for protected in (inputs, masks, outputs):
                if debug == protected or protected in debug.parents or debug in protected.parents:
                    raise ValueError("Debug directory must not overlap images, masks, or final outputs")
            for _, relative in jobs:
                if (debug / relative).exists() and not args.overwrite:
                    raise ValueError(f"Refusing to overwrite debug image: {debug / relative}")
        # Validate every pair before allocating model memory or writing any output.
        for source, relative in jobs:
            mask_path = masks / relative
            if not mask_path.is_file():
                raise ValueError(f"Missing mask for {relative}: {mask_path}")
            with Image.open(source) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Mask size {mask.size} must match image size {image.size}: {mask_path}")
        predict, errors = None, []
        for index, (source, relative) in enumerate(jobs, 1):
            started = time.perf_counter()
            print(f"[{index}/{len(jobs)}] Inpainting {relative}", flush=True)
            try:
                image = read_rgb(source)
                mask = read_mask(masks / relative, image.shape[:2])
                if np.any(mask >= 128) and predict is None:
                    import torch
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
                    predict = predictor(init_inpaint_model(args.checkpoint, device), device)
                output, debug = inpaint_regions(predict, image, mask, args.mask_dilation,
                                                 args.blend_radius, args.context_scale,
                                                 debug=args.debug_dir is not None)
                save_png(args.out_dir / relative, output)
                if debug is not None:
                    save_png(args.debug_dir / relative, debug)
                print(f"Saved {relative} in {time.perf_counter() - started:.2f}s", flush=True)
            except Exception as exc:
                errors.append(relative)
                print(f"Error processing {relative}: {exc}", file=sys.stderr, flush=True)
        return 1 if errors else 0
    except Exception as exc:
        print(f"Inpainting error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
