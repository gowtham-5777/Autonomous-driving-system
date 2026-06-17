"""Vendored official YOLOP MCnet architecture (hustvl/YOLOP).

Public API for the ADAS integration layer.  Inference wiring lives in
``src.modules.yolop.inference`` — not in this package.
"""

from .models import MCnet, get_net

__all__ = ["get_net", "MCnet"]
