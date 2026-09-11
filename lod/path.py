"""Locate the independent GaussianZoom_distill source tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_GZ_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "gaussianzoom_distill"


def add_gz_src(root: str | os.PathLike[str] | None = None) -> Path:
    """Put ``<root>/src`` on ``sys.path`` so ``gaussianzoom_lod`` can be imported."""

    resolved = Path(root or os.environ.get("GAUSSIANZOOM_ROOT", DEFAULT_GZ_ROOT)).expanduser().resolve()
    model_path = resolved / "src" / "gaussianzoom_lod" / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"GaussianZoom_distill model not found at {model_path}. "
            "Pass --gz_root or set GAUSSIANZOOM_ROOT."
        )
    src = str(resolved / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return resolved
