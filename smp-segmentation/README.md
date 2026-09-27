# SMP segmentation CLI

Run through the main pipeline to retain editable masks:

```bash
python main.py --model-type black_bars --segment-only
```

This opens the local brush editor after segmentation. Paint or erase the magenta
overlay and save the mask, then run `--inpaint-only`. Add `--no-edit` for headless
segmentation, or use `--edit-masks` to reopen existing masks later.

The adapter can also run directly from any working directory:

```bash
python smp-segmentation/run_segmentation.py --model-type black_bars --input-dir ./pages --output-dir ./review
```

`--input-dir` is the actual image directory, scanned recursively. If omitted,
the default is `camelia-decensor/input/<model-type>/`. Its underscore alias has
the same semantics; no model-type subdirectory is appended to an explicit path.

The adapter retains RGB PNGs in `images/`, binary grayscale PNGs in `masks/`,
absolute confidence previews in `probabilities/`, and `manifest.json` metadata.
The default mask threshold is 0.5. Use `--mask-threshold` to change it and
`--overwrite` only when replacing existing masks is intended.

The EfficientNet-B6 Unet++ preprocessing remains grayscale, 1024x1024 nearest
resizing, and ImageNet normalization to match the existing pretrained weights.
Non-PNG inputs are decoded directly without temporary conversion files.
