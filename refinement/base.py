"""Unified refinement interface and generation-result cache."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from .types import RefinementRequest, RefinementResult, canonical_json, to_jsonable


class Refiner(ABC):
    """Backend contract used by ``train_zoom_gen.py``.

    Backends should not update Gaussian parameters or camera courses.  They
    only transform the requested image and report backend metadata.
    """

    name = "refiner"

    @abstractmethod
    def refine(self, request: RefinementRequest) -> RefinementResult:
        """Return an enhanced image and metadata for ``request``."""

    def release_memory(self) -> None:
        """Release optional resident model state before another worker runs."""
        return None


def build_refinement_cache_key(request: RefinementRequest, backend_name: str) -> str:
    """Hash all state that can change a generated image.

    The checkpoint, camera, zoom factor, SR scale, model configuration, prompt
    configuration, and input image digest are all included.  Including the
    input digest is an additional guard against accidentally reusing an image
    produced by a different render with the same nominal experiment settings.
    """

    payload = request.cache_payload(backend_name)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: str | os.PathLike[str], backend_name: str, key: str) -> tuple[Path, Path]:
    safe_backend = "".join(char if char.isalnum() or char in "-_" else "_" for char in backend_name)
    root = Path(cache_dir) / safe_backend
    return root / f"{key}.png", root / f"{key}.json"


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    image.convert("RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def _atomic_save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


class CachedRefiner(Refiner):
    """Transparent on-disk cache wrapper around any refiner backend."""

    def __init__(self, backend: Refiner, cache_dir: str | os.PathLike[str], enabled: bool = True):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.name = getattr(backend, "name", backend.__class__.__name__.lower())

    def release_memory(self) -> None:
        self.backend.release_memory()

    def refine(self, request: RefinementRequest) -> RefinementResult:
        key = build_refinement_cache_key(request, self.name)
        image_path, metadata_path = _cache_paths(self.cache_dir, self.name, key)

        if self.enabled and image_path.is_file() and metadata_path.is_file():
            try:
                with Image.open(image_path) as cached_image:
                    image = cached_image.convert("RGB").copy()
                with metadata_path.open("r", encoding="utf-8") as handle:
                    cached_metadata = json.load(handle)
                metadata = dict(cached_metadata.get("result", {}).get("metadata", {}))
                metadata["cache_path"] = str(image_path)
                metadata["cache_request"] = cached_metadata.get("request", {})
                return RefinementResult(
                    image=image,
                    backend=str(cached_metadata.get("result", {}).get("backend", self.name)),
                    metadata=metadata,
                    cache_key=key,
                    cache_hit=True,
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                # A partial/corrupt cache entry should not make a run unusable.
                print(f"[refinement] ignoring invalid cache entry {image_path}: {exc}")

        result = self.backend.refine(request)
        if not isinstance(result, RefinementResult):
            raise TypeError(
                f"Refiner {self.backend.__class__.__name__} returned {type(result)!r}; "
                "expected RefinementResult."
            )

        result.cache_key = key
        result.cache_hit = False
        result.metadata.setdefault("cache_path", str(image_path))

        if self.enabled:
            _atomic_save_image(result.image, image_path)
            _atomic_save_json(
                {
                    "request": request.cache_payload(self.name),
                    "result": result.to_dict(),
                },
                metadata_path,
            )

        return result


__all__ = ["CachedRefiner", "Refiner", "build_refinement_cache_key"]
