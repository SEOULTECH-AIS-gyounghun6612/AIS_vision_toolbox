"""
vision_toolbox: A library for handling 3D vision data and operations.
"""

# Expose the core data asset classes
from .asset import (
    Asset,
    Camera,
    Image,
    Point_Cloud,
    Scene,
)

# Expose the utility modules under a clear namespace
from . import utils

__all__ = [
    # Assets
    "Asset",
    "Camera",
    "Image",
    "Point_Cloud",
    "Scene",
    # Util modules
    "utils",
]
