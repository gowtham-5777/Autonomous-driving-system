"""Vendored YOLOP utilities — inference-minimal exports."""

from .autoanchor import check_anchor_order
from .utils import initialize_weights, is_parallel

__all__ = ["initialize_weights", "is_parallel", "check_anchor_order"]
