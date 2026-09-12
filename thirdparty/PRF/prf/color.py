from __future__ import annotations

import math

import numpy as np

# Absolute bins in m / equiv.pixel. Shared by the 3D viewer and tests.
RESOLUTION_BINS_M = (0.5, 1.0, 2.0, 5.0, 10.0)
RESOLUTION_RGB = (
    (8, 24, 107),
    (0, 72, 168),
    (0, 140, 170),
    (218, 158, 0),
    (215, 72, 0),
    (135, 0, 34),
)

# Display-only R_obs bins reveal its dominant 0.3--1.0 m/equiv.pixel range.
# The absolute bins above remain available for cross-metric comparisons.
OBS_ENHANCED_BINS_M = (0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
OBS_ENHANCED_RGB = (
    (8, 24, 107),
    (0, 59, 143),
    (0, 104, 201),
    (0, 151, 178),
    (224, 160, 0),
    (217, 74, 0),
    (135, 0, 34),
)
INVALID_RGB = (120, 120, 120)


def resolution_bin(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        return -1
    for index, edge in enumerate(RESOLUTION_BINS_M):
        if value <= edge:
            return index
    return len(RESOLUTION_BINS_M)


def resolution_rgb(value: float) -> tuple[int, int, int]:
    index = resolution_bin(value)
    if index < 0:
        return INVALID_RGB
    return RESOLUTION_RGB[index]


def resolution_codes(values: np.ndarray) -> np.ndarray:
    """Vectorized absolute-bin codes; 255 denotes invalid/non-finite values."""
    values = np.asarray(values, dtype=np.float64)
    codes = np.digitize(values, RESOLUTION_BINS_M, right=True).astype(np.uint8)
    codes[~np.isfinite(values) | (values < 0.0)] = np.uint8(255)
    return codes


def obs_enhanced_codes(values: np.ndarray) -> np.ndarray:
    """Display-only R_obs codes with more contrast over its dominant range."""
    values = np.asarray(values, dtype=np.float64)
    codes = np.digitize(values, OBS_ENHANCED_BINS_M, right=True).astype(np.uint8)
    codes[~np.isfinite(values) | (values < 0.0)] = np.uint8(255)
    return codes
