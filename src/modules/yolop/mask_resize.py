"""Resize YOLOP segmentation masks to original frame dimensions."""

from __future__ import annotations

import cv2
import numpy as np

MaskArray = np.ndarray

__all__ = ["resize_mask_to_frame"]


def resize_mask_to_frame(
    mask: MaskArray | None,
    frame_height: int,
    frame_width: int,
) -> MaskArray | None:
    """Resize a binary mask from model resolution to ``(H, W)`` frame space.

    Args:
        mask: Binary segmentation mask (typically 640×640 from YOLOP).
        frame_height: Original frame height ``frame.shape[0]``.
        frame_width: Original frame width ``frame.shape[1]``.

    Returns:
        Resized mask with shape ``(frame_height, frame_width)``, or ``None``.
    """
    if mask is None:
        return None

    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(frame_width), int(frame_height)),
        interpolation=cv2.INTER_NEAREST,
    )
