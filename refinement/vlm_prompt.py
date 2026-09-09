"""Structured prompt generation with reusable shared-region caching.

The module does not hard-code a VLM dependency.  ``PromptProvider`` is the
small adapter point for a real VLM; fixed and JSON providers are included so
the zoom pipeline can be evaluated offline and with recorded VLM outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from PIL import Image

from .types import CameraSnapshot, PromptDescription, canonical_json, to_jsonable


class PromptProvider(Protocol):
    """Adapter contract for a VLM or a deterministic prompt source."""

    name: str

    def describe(
        self,
        wide_image: Image.Image | None,
        zoom_image: Image.Image | None,
        *,
        zoom_factor: float,
        level_index: int,
        context: Mapping[str, Any],
    ) -> PromptDescription:
        ...

    def cache_config(self) -> Mapping[str, Any]:
        ...


class PromptCache:
    """Small JSON cache for shared and per-level structured descriptions."""

    def __init__(self, root: str | os.PathLike[str] | None):
        self.root = Path(root) if root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> PromptDescription | None:
        if self.root is None:
            return None
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return PromptDescription.from_dict(json.load(handle))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"[refinement] ignoring invalid prompt cache entry {path}: {exc}")
            return None

    def put(self, key: str, prompt: PromptDescription) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(prompt.to_dict(), handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)


def _cache_key(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

def _image_key(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    digest = hashlib.sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()

def _shared_prompt(prompt: PromptDescription) -> PromptDescription:
    """Keep only fields that are intended to be reused across scales."""

    return PromptDescription(
        shared_region_description=prompt.shared_region_description,
        visible_features=prompt.visible_features,
        preserve_structure=prompt.preserve_structure,
        uncertain_information=prompt.uncertain_information,
        provider=prompt.provider,
        config=prompt.config,
    )


class FixedPromptProvider:
    """Deterministic provider used for baseline and SR-backbone comparisons."""

    name = "fixed"

    def __init__(
        self,
        source_prompt: str,
        target_prompt: str,
        *,
        shared_region_description: str = "",
        current_scale_template: str = "Zoom level {zoom_factor:g}x; preserve the scene layout and camera geometry.",
        visible_features: tuple[str, ...] = tuple(),
        preserve_structure: tuple[str, ...] = tuple(),
        uncertain_information: tuple[str, ...] = tuple(),
        config: Mapping[str, Any] | None = None,
    ):
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt
        self.shared_region_description = shared_region_description
        self.current_scale_template = current_scale_template
        self.visible_features = tuple(visible_features)
        self.preserve_structure = tuple(preserve_structure)
        self.uncertain_information = tuple(uncertain_information)
        self.config = dict(config or {})

    def describe(
        self,
        wide_image: Image.Image | None,
        zoom_image: Image.Image | None,
        *,
        zoom_factor: float,
        level_index: int,
        context: Mapping[str, Any],
    ) -> PromptDescription:
        del wide_image, zoom_image, level_index, context
        return PromptDescription(
            shared_region_description=self.shared_region_description,
            current_scale_description=self.current_scale_template.format(zoom_factor=zoom_factor),
            visible_features=self.visible_features,
            preserve_structure=self.preserve_structure,
            uncertain_information=self.uncertain_information,
            source_prompt=self.source_prompt,
            target_prompt=self.target_prompt,
            provider=self.name,
            config=self.config,
        )

    def cache_config(self) -> Mapping[str, Any]:
        return {
            "provider": self.name,
            "source_prompt": self.source_prompt,
            "target_prompt": self.target_prompt,
            "shared_region_description": self.shared_region_description,
            "current_scale_template": self.current_scale_template,
            "config": self.config,
        }


class JsonPromptProvider:
    """Provider for recorded VLM results.

    The JSON can contain one description for all levels, or a ``levels`` list
    with ``zoom_factor`` and per-level fields.  This lets a real VLM be run in
    a separate process while the training pipeline remains deterministic.
    """

    name = "json"

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fallback_source_prompt: str = "",
        fallback_target_prompt: str = "",
    ):
        self.path = Path(path).resolve()
        self.fallback_source_prompt = fallback_source_prompt
        self.fallback_target_prompt = fallback_target_prompt
        with self.path.open("r", encoding="utf-8") as handle:
            self.data = json.load(handle)
        if not isinstance(self.data, Mapping):
            raise ValueError(f"Prompt JSON must contain an object: {self.path}")

    def _level_data(self, zoom_factor: float, level_index: int) -> dict[str, Any]:
        levels = self.data.get("levels", [])
        if isinstance(levels, list):
            for item in levels:
                if not isinstance(item, Mapping):
                    continue
                if "zoom_factor" in item and abs(float(item["zoom_factor"]) - zoom_factor) < 1e-6:
                    return dict(item)
                if "level_index" in item and int(item["level_index"]) == level_index:
                    return dict(item)
        return dict(self.data)

    def describe(
        self,
        wide_image: Image.Image | None,
        zoom_image: Image.Image | None,
        *,
        zoom_factor: float,
        level_index: int,
        context: Mapping[str, Any],
    ) -> PromptDescription:
        del wide_image, zoom_image, context
        data = dict(self.data)
        data.update(self._level_data(zoom_factor, level_index))
        data.setdefault("source_prompt", self.fallback_source_prompt)
        data.setdefault("target_prompt", self.fallback_target_prompt)
        data.setdefault("provider", self.name)
        data.setdefault("config", {"path": str(self.path)})
        return PromptDescription.from_dict(data)

    def cache_config(self) -> Mapping[str, Any]:
        return {
            "provider": self.name,
            "path": str(self.path),
            "source_prompt": self.fallback_source_prompt,
            "target_prompt": self.fallback_target_prompt,
        }


class CallablePromptProvider:
    """Adapter for an application-owned VLM callable.

    The callable receives the two PIL images and keyword context, and may
    return either a ``PromptDescription`` or a JSON-compatible mapping.
    """

    name = "callable"

    def __init__(self, callback: Callable[..., PromptDescription | Mapping[str, Any]], config: Mapping[str, Any] | None = None):
        self.callback = callback
        self.config = dict(config or {})

    def describe(
        self,
        wide_image: Image.Image | None,
        zoom_image: Image.Image | None,
        *,
        zoom_factor: float,
        level_index: int,
        context: Mapping[str, Any],
    ) -> PromptDescription:
        result = self.callback(
            wide_image,
            zoom_image,
            zoom_factor=zoom_factor,
            level_index=level_index,
            context=context,
        )
        if isinstance(result, PromptDescription):
            return result
        return PromptDescription.from_dict(result)

    def cache_config(self) -> Mapping[str, Any]:
        return {"provider": self.name, **self.config}


class PromptManager:
    """Generate per-level prompts while reusing shared-region semantics."""

    def __init__(self, provider: PromptProvider, cache: PromptCache | None = None):
        self.provider = provider
        self.cache = cache

    def get_prompt(
        self,
        *,
        checkpoint: str,
        camera: CameraSnapshot,
        roi: Mapping[str, Any],
        zoom_factor: float,
        level_index: int,
        wide_image: Image.Image | None,
        zoom_image: Image.Image | None,
        context: Mapping[str, Any] | None = None,
    ) -> PromptDescription:
        provider_config = to_jsonable(dict(self.provider.cache_config()))
        shared_key = _cache_key(
            {
                "schema_version": 1,
                "kind": "shared",
                "checkpoint": os.path.abspath(checkpoint),
                "camera": camera.to_dict(),
                "roi": to_jsonable(dict(roi)),
                "provider": provider_config,
            }
        )
        level_key = _cache_key(
            {
                "schema_version": 1,
                "kind": "level",
                "shared_key": shared_key,
                "zoom_factor": float(zoom_factor),
                "level_index": int(level_index),
                "wide_image_sha256": _image_key(wide_image),
                "zoom_image_sha256": _image_key(zoom_image),
            }
        )

        cached_level = self.cache.get(level_key) if self.cache else None
        if cached_level is not None:
            return cached_level

        shared = self.cache.get(f"shared_{shared_key}") if self.cache else None
        prompt = self.provider.describe(
            wide_image,
            zoom_image,
            zoom_factor=zoom_factor,
            level_index=level_index,
            context={**(context or {}), "shared_prompt": shared.to_dict() if shared else {}},
        )
        if shared is not None:
            prompt = replace(
                prompt,
                shared_region_description=shared.shared_region_description,
                visible_features=shared.visible_features,
                preserve_structure=shared.preserve_structure,
                uncertain_information=shared.uncertain_information,
            )
        elif self.cache:
            self.cache.put(f"shared_{shared_key}", _shared_prompt(prompt))

        prompt = replace(
            prompt,
            config={
                **dict(prompt.config),
                "shared_cache_key": shared_key,
                "level_cache_key": level_key,
            },
        )
        if self.cache:
            self.cache.put(level_key, prompt)
        return prompt


__all__ = [
    "CallablePromptProvider",
    "FixedPromptProvider",
    "JsonPromptProvider",
    "PromptCache",
    "PromptManager",
    "PromptProvider",
]
