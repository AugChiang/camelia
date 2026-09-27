"""Crop-based inpainting independent of model loading and interface code."""

import cv2
import numpy as np

from camelia.masks import blend_weights, context_box, crop_with_padding, hard_mask, region_boxes


def inpaint_regions(predict, image, mask, dilation=0, blend_radius=0,
                    context_scale=1.2, debug=False):
    """predict receives RGB uint8 and binary uint8 crops, and returns RGB floats in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Inpainting requires an RGB image")
    selected = hard_mask(mask, image.shape[:2], dilation)
    weights = blend_weights(selected, blend_radius)
    output = image.copy()
    boxes = [context_box(box, context_scale) for box in region_boxes(selected)]
    height, width = selected.shape
    for box in boxes:
        x0, y0, x1, y1 = box
        crop = crop_with_padding(image, box)
        crop_mask = crop_with_padding(selected, box)
        generated = np.asarray(predict(crop, crop_mask), dtype=np.float32)
        if generated.shape != crop.shape or not np.isfinite(generated).all():
            raise ValueError("Model output must be finite RGB values with the crop's dimensions")
        left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        prediction = generated[top-y0:bottom-y0, left-x0:right-x0]
        alpha = weights[top:bottom, left:right, None]
        blended = np.clip(prediction, 0, 1) * 255 * alpha + image[top:bottom, left:right] * (1 - alpha)
        output[top:bottom, left:right] = np.rint(blended).clip(0, 255).astype(np.uint8)
    if not debug:
        return output, None
    overlay = image.copy()
    occupied = selected.astype(bool)
    overlay[occupied] = np.rint(overlay[occupied] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(overlay, (max(0, x0), max(0, y0)),
                      (min(width-1, x1-1), min(height-1, y1-1)), (255, 0, 0), 1)
    return output, np.concatenate([overlay, output], axis=1)
