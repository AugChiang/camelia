# Local brush editor and future Gradio integration

`feat/cli-only` now includes a minimal Tkinter brush editor. It opens after
`--segment-only`, or independently through `--edit-masks`. `--no-edit` preserves
headless segmentation. The full web interface remains separate on `feat/gradio`.

## Implemented local workflow

```bash
python main.py --model-type black_bars --segment-only
# Paint/erase the overlay, save masks, and close the editor.
python main.py --model-type black_bars --inpaint-only
```

The editor loads the retained source images at their original dimensions. It
displays detected and manually painted regions as a magenta overlay at about
40% opacity. Only the mask layer changes; the source image and probability
previews are not modified.

Controls include brush diameter in original-image pixels, paint/erase, previous
and next page, a page dropdown, save, save and next, undo/redo, clear mask, zoom,
fit, and pan. Stroke segments are connected so fast mouse movement leaves no
gaps. Zoom and scrolling are mapped back to source-image coordinates. Only the
visible viewport is rendered, avoiding allocation of an entire enlarged page.

**Save mask** atomically replaces `masks/<relative>.png`, exactly where default
LaMa inpainting reads. Switching pages or closing asks whether to save dirty
masks. The editor also supports an explicit `--mask-dir`, which must be passed
to subsequent `--inpaint-only` calls. A missing mask starts empty and must be saved.

## Shared mask contract

- `images/<relative>.png`: retained RGB source, already oriented, original dimensions.
- `masks/<relative>.png`: one-channel PNG, 0 means preserve and 255 means replace.
- `probabilities/<relative>.png`: read-only segmentation confidence preview.
- `manifest.json`: source paths, artifact paths, dimensions, backend, and threshold.

Binary masks are the authoritative data. The overlay's color and transparency
are display settings, not brush strength. LaMa's model input remains a binary
hole mask, matching [upstream prediction](https://github.com/advimman/lama/blob/main/bin/predict.py).
Optional feathering belongs to output compositing; dilation and feathering are
separate CLI controls and default to zero. Hard masks do not guarantee accurate
linework or reconstruction; checkpoint quality and surrounding context still matter.

The implementation is in `camelia/editor.py`. The document/stroke model is
independent of Tk widgets and model inference. The editor requires Pillow and
Python's Tk support; it has no GPU, Gradio, server, or Node.js dependency.

## Gradio follow-up

Reuse this artifact contract on `feat/gradio`, with the same coordinate mapping
and binary export rules. Additional interface work can include a probability
preview toggle, adjustable overlay opacity, restoring automatic segmentation,
keyboard brush-size controls, and an inpainting preview button with cancellation.

The current editor directly edits `masks/` by default to satisfy the two-command
CLI workflow. A Gradio interface may instead default to `edited-masks/` while
passing that directory explicitly to inpainting. Keep automatic masks if a
separate reset-to-segmentation workflow is required.

The interface may invoke the CLI initially, using argument lists, explicit paths,
and retained process handles. A later shared inference API can cache a model per
worker without changing mask files. Avoid global cleanup or process-name cancellation.

## Validation

The regression suite checks continuous paint/erase strokes, one-pixel and edge
selections, undo/redo, overlay separation, binary saves at original dimensions,
nested page paths, invalid mask sizes, and CLI editor/batch routing.

`python tests/smoke_editor.py --inpaint` exercises real Tk canvas events at
different zoom levels, saves and reloads multiple pages, tests cancellation of
navigation with unsaved edits, and passes the saved masks to real LaMa inference.
It checks that pixels outside the masks remain unchanged. Test artifacts are
retained under `.test-output/` for inspection.
