"""YOLOP lane segmentation post-processing for inference.

Migrates inference-time lane mask refinement utilities from the YOLOP
reference implementation (``lib/core/postprocess.py``):

- ``morphological_process`` — fill holes and smooth binary masks
- ``connected_components_analysis`` — connected component labeling
- ``connect_lane`` — remove small components and rebuild lane mask

Training-related helpers (e.g. ``build_targets``) are intentionally excluded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .mask_resize import resize_mask_to_frame

logger = logging.getLogger("adas.modules.yolop.postprocess")

ImageArray = np.ndarray
MorphologyOp = int

DEFAULT_MIN_COMPONENT_AREA = 400

__all__ = [
    "ConnectedComponentsResult",
    "connect_lane",
    "connected_components_analysis",
    "morphological_process",
    "postprocess_lane_mask",
    "resize_mask_to_frame",
]


@dataclass(frozen=True)
class ConnectedComponentsResult:
    """Result of connected components analysis on a binary image.

    Attributes:
        num_labels: Total number of labels including background (``0``).
        labels: Label image where each pixel value is its component id.
        stats: Component statistics array from OpenCV.
        centroids: Component centroid coordinates from OpenCV.
    """

    num_labels: int
    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray


def morphological_process(
    image: ImageArray,
    kernel_size: int = 5,
    func_type: MorphologyOp = cv2.MORPH_CLOSE,
) -> ImageArray:
    """Apply morphological processing to a binary segmentation mask.

    Fills small holes and smooths the mask using an elliptical structuring
    element.  Defaults to a closing operation as in the YOLOP reference.

    Args:
        image: Single-channel binary segmentation mask.
        kernel_size: Size of the square structuring element (must be positive).
        func_type: OpenCV morphological operation (e.g. ``cv2.MORPH_CLOSE``,
            ``cv2.MORPH_OPEN``).

    Returns:
        Post-processed binary mask with dtype ``uint8``.

    Raises:
        ValueError: If the image is not single-channel or ``kernel_size`` is
            invalid.
    """
    _validate_binary_image(image)

    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")

    binary_image = _to_uint8(image)
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE,
        ksize=(kernel_size, kernel_size),
    )

    processed = cv2.morphologyEx(binary_image, func_type, kernel, iterations=1)

    logger.debug(
        "morphological_process — kernel=%d, op=%s, shape=%s",
        kernel_size,
        func_type,
        processed.shape,
    )
    return processed


def connected_components_analysis(image: ImageArray) -> ConnectedComponentsResult:
    """Run connected components analysis on a binary lane mask.

    Wrapper around ``cv2.connectedComponentsWithStats`` using 8-connectivity
    and 32-bit integer labels, matching the YOLOP reference.

    Args:
        image: Binary or grayscale segmentation image.

    Returns:
        :class:`ConnectedComponentsResult` with labels, stats, and centroids.

    Raises:
        ValueError: If the input image is invalid.
    """
    gray_image = _to_grayscale(image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gray_image,
        connectivity=8,
        ltype=cv2.CV_32S,
    )

    logger.debug(
        "connected_components_analysis — labels=%d, image_shape=%s",
        num_labels,
        gray_image.shape,
    )

    return ConnectedComponentsResult(
        num_labels=int(num_labels),
        labels=labels,
        stats=stats,
        centroids=centroids,
    )


def connect_lane(
    lane_mask: ImageArray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    shadow_height: int = 0,
) -> ImageArray:
    """Connect and refine lane mask by removing small connected components.

    Adapted from the YOLOP reference ``connect_lane`` for inference use.
    Large connected components are retained and merged into a cleaned binary
    mask.  Polynomial lane fitting (``fitlane`` in the upstream repo) is
    excluded from this module per project scope.

    Args:
        lane_mask: Binary lane segmentation mask.
        min_area: Minimum component area in pixels to retain.
        shadow_height: If positive, zeroes the top ``shadow_height`` rows
            before analysis (removes sky / bonnet shadow region).

    Returns:
        Cleaned binary lane mask containing only large components.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_binary_image(lane_mask)

    if min_area < 0:
        raise ValueError(f"min_area must be non-negative, got {min_area}")

    if shadow_height < 0:
        raise ValueError(f"shadow_height must be non-negative, got {shadow_height}")

    gray_image = _to_grayscale(lane_mask)

    if shadow_height > 0:
        if shadow_height >= gray_image.shape[0]:
            raise ValueError(
                f"shadow_height ({shadow_height}) must be less than image "
                f"height ({gray_image.shape[0]})"
            )
        gray_image = gray_image.copy()
        gray_image[:shadow_height, :] = 0
        logger.debug("Zeroed top %d rows for shadow masking", shadow_height)

    components = connected_components_analysis(gray_image)
    output_mask = np.zeros_like(gray_image, dtype=np.uint8)

    if components.num_labels <= 1:
        logger.warning("connect_lane — no foreground components found")
        return output_mask

    retained_labels: list[int] = []
    for label in range(1, components.num_labels):
        area = int(components.stats[label, cv2.CC_STAT_AREA])
        if area > min_area:
            retained_labels.append(label)
            output_mask[components.labels == label] = 255

    logger.info(
        "connect_lane — retained %d/%d components (min_area=%d)",
        len(retained_labels),
        components.num_labels - 1,
        min_area,
    )

    if not retained_labels:
        logger.warning(
            "connect_lane — all components below min_area=%d; returning empty mask",
            min_area,
        )

    return output_mask


def postprocess_lane_mask(
    lane_mask: ImageArray,
    kernel_size: int = 5,
    morphology_op: MorphologyOp = cv2.MORPH_CLOSE,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    shadow_height: int = 0,
) -> ImageArray:
    """Run the full YOLOP-inspired lane mask post-processing pipeline.

    Pipeline:
        morphological_process → connect_lane

    Args:
        lane_mask: Raw binary lane segmentation mask.
        kernel_size: Morphological kernel size.
        morphology_op: Morphological operation type.
        min_area: Minimum connected-component area to keep.
        shadow_height: Rows to mask at the top before component analysis.

    Returns:
        Refined binary lane mask.
    """
    logger.info("Starting lane mask post-processing pipeline")

    closed = morphological_process(
        lane_mask,
        kernel_size=kernel_size,
        func_type=morphology_op,
    )
    connected = connect_lane(
        closed,
        min_area=min_area,
        shadow_height=shadow_height,
    )

    logger.info("Lane mask post-processing pipeline complete")
    return connected


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _validate_binary_image(image: ImageArray) -> None:
    """Validate that an image is a non-empty 2D or single-channel array."""
    if image is None:
        raise ValueError("image is None")

    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be numpy.ndarray, got {type(image).__name__}")

    if image.ndim == 3:
        if image.shape[2] != 1:
            raise ValueError(
                "Binary segmentation image must be single-channel. "
                f"Received shape {image.shape}"
            )
    elif image.ndim != 2:
        raise ValueError(
            f"image must be 2D or single-channel 3D, got shape {image.shape}"
        )

    if image.size == 0:
        raise ValueError("image is empty")


def _to_uint8(image: ImageArray) -> ImageArray:
    """Convert an array to ``uint8`` for OpenCV processing."""
    if image.dtype != np.uint8:
        return np.asarray(image, dtype=np.uint8)
    return image


def _to_grayscale(image: ImageArray) -> ImageArray:
    """Convert a mask to single-channel ``uint8`` grayscale."""
    _validate_binary_image(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(_to_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = _to_uint8(image)

    return gray
