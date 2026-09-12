"""Physical Resolution Field (PRF) for metric 3D Gaussian scenes."""

from prf.config import PRFConfig
from prf.pipeline import build_physical_resolution_field
from prf.types import GaussianScene, PatchResolutionRecord, PhysicalResolutionField

__all__ = [
    "PRFConfig",
    "GaussianScene",
    "PatchResolutionRecord",
    "PhysicalResolutionField",
    "build_physical_resolution_field",
]
