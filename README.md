# Camelia CLI

Camelia segments selected regions and fills them with LaMa. This branch,
`feat/cli-only`, contains the command-line pipeline and a minimal local brush
editor. The full web interface belongs on `feat/gradio`; this branch does not
depend on Flask, Gradio, Nuxt, or Node.js.

The brush editor lives in the reusable `mask-editor` submodule. After cloning,
initialize submodules before installing dependencies:

```bash
git submodule update --init mask-editor
```

Segmentation artifacts are persistent. `--segment-only` opens the brush editor
after saving them. Edit the masks, save, and run inpainting without regenerating them.

## Installation

The existing SMP and LaMa checkpoints use the legacy Python 3.9 / PyTorch 1.13
environment. Install its inference requirements:

```bash
conda create --name camelia_env python=3.9 -y
conda activate camelia_env
pip install -r requirements.txt
```

The requirements select the CUDA 11.6 PyTorch wheels. For CPU-only installation,
install the corresponding CPU Torch/Torchvision wheels and the remaining listed
dependencies. Runtime `--device auto` selects CUDA when available, otherwise CPU.

Download the existing [SMP and LaMa checkpoints](https://drive.google.com/drive/folders/1AAyv6ms_694VGEtET6TFRLKZ1rx7ue11?usp=sharing):

```text
smp-segmentation/pretrained/best_black_bars_model.pth
smp-segmentation/pretrained/best_white_bars_model.pth
smp-segmentation/pretrained/best_transparent_black_model.pth
lama-inpainting/pretrained/best/config.yaml
lama-inpainting/pretrained/best/models/best.ckpt
```

Legacy checkpoints contain serialized training metadata; use checkpoints from a
trusted source. Only the LaMa generator is instantiated for inference. Lightning
remains a dependency because the legacy checkpoint metadata references its classes.

## Segment, edit, then inpaint

Place images under `camelia-decensor/input/black_bars/`, including chapter
subdirectories, or supply any image directory with `--input-dir`.

```bash
python main.py --model-type black_bars --segment-only
```

The default retained directory is `camelia-decensor/intermediate/black_bars/`:

```text
images/chapter/page.png          # RGB source image, at original dimensions
masks/chapter/page.png           # editable grayscale PNG: black=keep, white=inpaint
probabilities/chapter/page.png   # confidence preview, scaled directly from 0..1
manifest.json                   # source paths, dimensions, and artifact paths
```

The brush editor loads the retained source images and segmentation masks. Paint
directly over the image: a **semi-transparent magenta overlay** shows the selected
inpainting region. Adjust brush diameter with the slider, select **Paint** or
**Erase**, and use **Previous**, **Next**, or the page dropdown to review all images.
Existing detected regions are shown in the same overlay.

Click **Save mask** (or press **Ctrl+S**) to update the PNG under `masks/`, or
**Save and next** to save and advance. Close the editor when finished. The editor
asks before leaving a page or closing with unsaved changes. Saves contain only
black (0, preserve) and white (255, inpaint), at the original dimensions; the
colored overlay is never saved into the source image or mask.

Then reuse your edited masks:

```bash
python main.py --model-type black_bars --inpaint-only
```

Reopen the editor without rerunning segmentation or inpainting:

```bash
python main.py --model-type black_bars --edit-masks
```

The editor includes undo/redo (**Ctrl+Z/Ctrl+Y**), clear mask, zoom buttons and
mouse-wheel zoom, **Fit**, **100%**, scrollbars, and middle-button dragging to pan.
Right-button dragging erases regardless of the selected tool. Brush diameter is
measured in original-image pixels, so it remains consistent when zooming.
The cursor is a black-and-white circle matching the brush diameter at the current
zoom, and updates as you change the brush size or zoom.

Tkinter and a desktop display are required for the local window. Python/Conda on
Windows normally includes Tk support; on other systems install Python's Tk support
if needed. No additional pip GUI package is required. For batch/headless use:

```bash
python main.py --model-type black_bars --segment-only --no-edit
```

To add masks to clean images without running segmentation, launch the editor with:

```bash
python ./mask-editor/main.py --input-dir ./clean-pages
```

Saved masks go to `./clean-pages/mask/`, preserving subdirectories. The editor
also supports draggable, resizable, rotatable rectangles. Its package API and
standalone setup are documented in
[mask-editor/README.md](mask-editor/README.md).

If opening the editor fails, the generated segmentation artifacts are still retained.

To keep several review sessions or use a separate edited-mask directory:

```bash
python main.py --model-type black_bars --segment-only --input-dir ./pages --intermediate-dir ./review/chapter01
python main.py --edit-masks --intermediate-dir ./review/chapter01 --mask-dir ./edited-masks
python main.py --inpaint-only --intermediate-dir ./review/chapter01 --mask-dir ./edited-masks --output-dir ./results/chapter01
```

An empty custom mask directory starts with blank masks; save every desired page
before inpainting. Omit `--mask-dir` to edit the existing segmentation masks directly.

No intermediate images, masks, or probability previews are automatically deleted.
Regenerating existing segmentation artifacts or final outputs requires `--overwrite`.
The editor's **Save mask** button intentionally replaces the selected page's mask.
Be careful:
rerunning segmentation with this flag also replaces edited masks. Use
`--inpaint-only --overwrite` to regenerate only the final outputs.

To run both stages together:

```bash
python main.py --model-type transparent_black
```

The model types are `black_bars`, `white_bars`, and `transparent_black`. Outputs
default to `camelia-decensor/output/<model-type>/`. JPEG/WebP inputs are decoded
directly, and retained images and masks use PNG. Subdirectories are preserved;
ambiguous inputs such as `page.jpg` and `page.png` in one folder are rejected.

## Mask and inpainting controls

- `--mask-threshold 0.5`: cutoff for segmentation probabilities. Lower it to select
  more uncertain pixels; inspect the result before inpainting.
- `--mask-dilation 2`: expand the working mask by two pixels when unwanted borders
  remain. Default is zero; the saved mask is not modified.
- `--blend-radius 3`: optionally soften the replacement inside mask edges. Default
  is zero, which fully replaces selected pixels. Pixels outside the working mask
  remain unchanged.
- `--context-scale 1.2`: surrounding crop context relative to the selected region's
  longest side. Crops are at least 64 pixels wide and aligned to multiples of eight.
  Increase context when nearby lines or textures should guide reconstruction.
- `--debug-dir ./debug`: save mask overlays and output comparisons. Debug images
  are only allocated when requested.
- `--device cpu` or `--device cuda:0`: explicitly select an inference device.
- `--checkpoint PATH` and `--seg-checkpoint PATH`: use alternative model weights.

Masks passed to LaMa are **binary 0/1**. Edited PNG values below 128 become zero;
values at least 128 become one. Mask size mismatches are rejected rather than
silently resized. Empty masks produce an unchanged image without loading LaMa.
Small selections are preserved rather than discarded by area filters.

For manga, a hard mask is the default because it clearly identifies all pixels
that must be reconstructed. A soft alpha can retain parts of an unwanted mark.
Feathering is an optional compositing operation, separate from the model mask.
This matches the binary-mask inference in [upstream LaMa](https://github.com/advimman/lama/blob/main/bin/predict.py).

## SAM3 in a separate environment

SAM3 and the legacy SMP/LaMa dependencies cannot share the pinned NumPy/Torch
environment. Initialize the submodule with `git submodule update --init --recursive`,
create a separate environment, and follow the checked-out [SAM3 installation
instructions](sam3/README.md) for its Python, Torch, and CUDA requirements. Then:

```bash
pip install -r requirements-sam3.txt
```

Put the SAM3 checkpoint at `sam3/pretrained/sam3.pt`. Select its interpreter while
keeping the current environment for LaMa:

```bash
python main.py --model-type black_bars --seg-model-type sam3 --seg-python /path/to/sam3/python --segment-only
python main.py --model-type black_bars --inpaint-only
```

On Windows, use the environment's `python.exe` path. `--inpaint-python` also allows
an explicit LaMa environment. Model loading happens once per stage and models are
released when the stage exits, avoiding simultaneous residency of both models.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover mask thresholds, reflection at crop borders, tiny regions, untouched
pixels, overwrite protection, retained artifacts, and CLI failures. They do not
require model checkpoints or a GPU.

Editor tests additionally cover continuous strokes, erasing, overlays, binary
saving, page paths, undo/redo, and CLI launch behavior. An optional desktop test
exercises real Tk canvas events and page navigation, then checks LaMa handoff:

```bash
python tests/smoke_editor.py --inpaint
```

An optional real-checkpoint test uses synthetic geometric pages in PNG/JPEG/WebP,
simulates mask edits, and checks that LaMa preserves edits and unselected pixels:

```bash
python tests/smoke_inference.py --device cuda
```

It retains its artifacts under `.test-output/` for inspection.

The local editor and future Gradio integration are described in
[docs/mask-editor-plan.md](docs/mask-editor-plan.md). Both use the same hard-mask
artifact contract; inference remains independent of the interface.

## Acknowledgments

[SMP](https://github.com/qubvel-org/segmentation_models.pytorch),
[LaMa](https://github.com/advimman/lama),
[SAM3](https://github.com/facebookresearch/sam3), and
[Er0manga](https://github.com/Er0manga/Er0mangaDemo).
