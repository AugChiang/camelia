"""Hard mask validation and native connected-component region extraction."""

import math

import cv2
import numpy as np


def hard_mask(mask, shape, dilation=0):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != tuple(shape):
        raise ValueError(f"Mask size {mask.shape} must match image size {tuple(shape)}")
    if dilation < 0:
        raise ValueError("Mask dilation must be nonnegative")
    binary = (mask >= 128).astype(np.uint8)
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        binary = cv2.dilate(binary, kernel)
    return binary


def region_boxes(mask):
    """Return occupied bounding boxes, largest components first, with exclusive ends."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), connectivity=4)
    stats = sorted(stats[1:], key=lambda row: int(row[cv2.CC_STAT_AREA]), reverse=True)
    return [(int(x), int(y), int(x + width), int(y + height))
            for x, y, width, height, area in stats]


def context_box(box, scale=1.2, minimum=64):
    if scale < 1 or not math.isfinite(scale):
        raise ValueError("Context scale must be finite and at least 1")
    x0, y0, x1, y1 = box
    side = max(minimum, math.ceil(max(x1 - x0, y1 - y0) * scale / 8) * 8)
    left = (x0 + x1 - side) // 2
    top = (y0 + y1 - side) // 2
    return left, top, left + side, top + side


def crop_with_padding(array, box):
    """Reflect requested coordinates without padding the entire image."""
    x0, y0, x1, y1 = box
    height, width = array.shape[:2]
    if 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height:
        return array[y0:y1, x0:x1]
    def coordinates(start, stop, length):
        indices = np.arange(start, stop)
        if length == 1:
            return np.zeros_like(indices)
        period = 2 * (length - 1)
        folded = indices % period
        return np.where(folded < length, folded, period - folded)
    rows = coordinates(y0, y1, height)
    columns = coordinates(x0, x1, width)
    return array[rows[:, None], columns[None, :]]


def blend_weights(mask, radius=0):
    """Optional feathering confined to selected pixels; preserve the exterior."""
    if radius < 0:
        raise ValueError("Blend radius must be nonnegative")
    if not radius:
        return mask.astype(np.float32)
    padded = np.pad(mask, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return np.minimum(distance / radius, 1).astype(np.float32) * mask
