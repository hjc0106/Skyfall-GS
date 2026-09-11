"""Mip-NeRF-360 COLMAP dataset: filename-paired LR/HR posed images, read-only.

Scene layout (never written to)::

    <root>/
        images_8/            # LR tier  (1/8 of full res)   -- training input
        images_2/            # HR tier  (1/2 of full res)   -- 4x supervision targets
        sparse/0/            # COLMAP model: {cameras,images,points3D}.{bin|txt}

The COLMAP model is parsed with the vendored RaDe-GS loader
(``gaussianzoom_component/vendor/RaDe-GS/scene/colmap_loader.py``) loaded in
isolation through ``importlib`` so ``scene/__init__.py`` and its heavy
dependencies never execute.

Pairing / protocol
------------------
* LR and HR images are matched across tiers by *exact file basename*: the
  same pose set, same name string.
* ``holdout``: every ``holdout``-th image of the lexicographically sorted
  basenames (indices ``0, holdout, 2*holdout, ...``) is held out as test.
  This fixes an explicit interpretation of the paper's every-eighth-view split.
  The split is made *before* limits and is identical for LR and HR, so the HR
  target of any held-out frame is unreachable through train pairs.
* ``train_limit`` / ``test_limit`` > 0: deterministic even subsampling of the
  already-split lists (round of linspace endpoints, evenly spread).
* Oracle supervision: the default HR tier ``images_2`` holds real captured
  references (``images`` is 5187x3361 originals, ``images_2`` = half).  Fitting
  against them validates the LoD machinery but is **not** generative SR from
  LR alone -- see :meth:`manifest`.
* ``max_width``: smoke-protocol downscale.  When nonzero and the native HR
  width exceeds it, both tiers are scaled by the *same* factor so the HR
  width is at most ``max_width`` while the LR<->HR ratio stays ~4x.
  Intrinsics are re-scaled to the effective pixel grid and images are
  resized (LANCZOS) on load.  This deviates from the full protocol (native
  2594x1681 HR / 648x420 LR) and is only meant for fast smoke runs.

Intrinsics
----------
COLMAP stores one camera per intrinsic group at the resolution COLMAP was run
on (full-res 5187x3361 for Mip-NeRF-360).  For every (name, tier) the actual
on-disk pixel size of that file is measured (PIL header only, no decode) and
fx/fy/cx/cy are rescaled per axis by ``measured/ref`` -- no power-of-two or
exact-factor assumption.  An LR<->HR scale guard (nominal 4x, 2% tolerance)
fails loudly on a mismatched ``hr_dir`` (e.g. wrong tier or generated targets
not ~4x the LR tier); nothing is ever resized to force a match.

Images are decoded on demand into float32 CPU ``CHW`` tensors in [0, 1] and
kept in a bounded LRU CPU cache (``cache_bytes``), so not all HR images are
preloaded.  Cameras are pure header/intrinsic math on the requested device.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .camera import Camera

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# LR (images_8) -> HR (images_2) nominal pixel scale for the Mip-NeRF-360
# 1/8 -> 1/2 zoom protocol, and the accepted relative deviation.  Guard is
# per-axis on the measured effective dimensions of every paired file.
NOMINAL_TIER_SCALE = 4.0
TIER_SCALE_TOLERANCE = 0.02


class DatasetError(RuntimeError):
    """Raised for any invalid/mismatched scene layout. Never resize-fixes."""


class _Intrinsics(NamedTuple):
    model: str
    fx: float
    fy: float
    cx: float
    cy: float
    ref_w: int  # COLMAP reference raster the params are expressed in
    ref_h: int


class _LRUCache:
    """Bounded (bytes) LRU of CPU tensors; oversized items are not cached."""

    def __init__(self, max_bytes: int):
        if max_bytes < 0:
            raise DatasetError(f"cache_bytes must be >= 0, got {max_bytes}")
        self.max_bytes = int(max_bytes)
        self._data: "OrderedDict[Tuple[str, str], torch.Tensor]" = OrderedDict()
        self._used = 0

    def get(self, key: Tuple[str, str]) -> Optional[torch.Tensor]:
        hit = self._data.get(key)
        if hit is not None:
            self._data.move_to_end(key)
        return hit

    def put(self, key: Tuple[str, str], tensor: torch.Tensor) -> None:
        if self.max_bytes == 0:
            return
        size = tensor.numel() * tensor.element_size()
        if size > self.max_bytes:  # single item cannot fit: don't cache it
            return
        if key in self._data:
            self._data.move_to_end(key)
            return
        self._data[key] = tensor
        self._used += size
        while self._used > self.max_bytes and len(self._data) > 1:
            _, victim = self._data.popitem(last=False)
            self._used -= victim.numel() * victim.element_size()

    def __len__(self) -> int:
        return len(self._data)


def _even_indices(n: int, k: int) -> List[int]:
    """Deterministic even spread of k indices over [0, n): round(linspace)."""
    if k <= 0:
        return []
    if k >= n or n == 1:
        return list(range(n))
    if k == 1:
        return [0]
    step = (n - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def _load_colmap_reader():
    """Importlib-isolated load of the vendored RaDe-GS COLMAP reader.

    Avoids importing ``vendor/RaDe-GS/scene/__init__.py`` (and with it torch,
    the gaussian model etc.); the loader module itself only needs numpy.
    The module is intentionally not registered in ``sys.modules``.
    """
    path = Path(__file__).resolve().parents[2] / "vendor" / "RaDe-GS" / "scene" / "colmap_loader.py"
    if not path.is_file():
        raise DatasetError(
            f"vendored COLMAP loader not found at {path}; component tree incomplete"
        )
    spec = importlib.util.spec_from_file_location("_gzl_vendor_colmap_loader", str(path))
    if spec is None or spec.loader is None:
        raise DatasetError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mip360Dataset:
    """Read-only posed LR/HR image pairs of one Mip-NeRF-360 scene."""

    def __init__(
        self,
        root,
        lr_dir: str = "images_8",
        hr_dir: str = "images_2",
        holdout: int = 8,
        train_limit: int = 0,
        test_limit: int = 0,
        max_width: int = 0,
        cache_bytes: int = 1 << 30,
    ):
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise DatasetError(f"scene root not found: {self._root}")
        self._lr_dir = lr_dir
        self._hr_dir = hr_dir
        if isinstance(holdout, bool) or not isinstance(holdout, int) or holdout < 1:
            raise DatasetError(f"holdout must be an int >= 1, got {holdout!r}")
        self._holdout = holdout
        for limit_name, limit in (("train_limit", train_limit), ("test_limit", test_limit)):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise DatasetError(f"{limit_name} must be an int >= 0, got {limit!r}")
        self._train_limit = int(train_limit)
        self._test_limit = int(test_limit)
        if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 0:
            raise DatasetError(f"max_width must be an int >= 0, got {max_width!r}")
        self._max_width = int(max_width)

        self._lr_path = self._root / self._lr_dir
        self._hr_path = self._root / self._hr_dir
        for tier_dir, label in ((self._lr_path, lr_dir), (self._hr_path, hr_dir)):
            if not tier_dir.is_dir():
                raise DatasetError(
                    f"{label}: tier directory not found under {self._root}: {tier_dir}"
                )

        # ---- sparse COLMAP model (bin preferred over txt, per file) ----
        sparse_dir = self._root / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = self._root / "sparse"
        if not sparse_dir.is_dir():
            raise DatasetError(f"no COLMAP model: expected sparse/0 or sparse under {self._root}")
        self._sparse_dir = sparse_dir
        self._colmap = _load_colmap_reader()
        cameras, images, points_path = self._read_sparse()

        # ---- per-name pose + per-camera intrinsics (validated) ----
        self._cam_intr: Dict[int, _Intrinsics] = {}
        self._intrinsics_from_colmap(cameras)
        self._name_cam: Dict[str, int] = {}
        self._name_pose: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for image in images.values():
            base = os.path.basename(image.name)
            if base in self._name_cam:
                raise DatasetError(
                    f"ambiguous COLMAP image name: {base!r} appears more than once "
                    "(basenames must be unique to pair tiers by name)"
                )
            self._name_cam[base] = int(image.camera_id)
            self._name_pose[base] = (np.asarray(image.qvec), np.asarray(image.tvec))
        self._points_path = points_path
        self._points: Optional[torch.Tensor] = None
        self._colors: Optional[torch.Tensor] = None

        # ---- tier file inventories (exact basename pairing) ----
        lr_files = self._image_files(self._lr_path)
        hr_files = self._image_files(self._hr_path)
        self._lr_files = lr_files
        self._hr_files = hr_files
        missing_pose = sorted(set(lr_files) - set(self._name_cam))
        if missing_pose:
            raise DatasetError(
                f"{len(missing_pose)} file(s) in {self._lr_dir} have no COLMAP pose "
                f"(first: {missing_pose[:8]}); every tier image must be registered"
            )
        unposed = sorted(set(self._name_cam) - set(lr_files))
        if unposed:
            raise DatasetError(
                f"COLMAP registers {len(unposed)} image(s) absent from {self._lr_dir} "
                f"(first: {unposed[:8]}); every registered image must exist in both tiers"
            )
        missing_hr = sorted(set(lr_files) - set(hr_files))
        if missing_hr:
            raise DatasetError(
                f"{len(missing_hr)} LR file(s) have no same-basename HR target in "
                f"{self._hr_dir} (first: {missing_hr[:8]}). HR targets must share the "
                "exact LR basename (same case/extension)"
            )
        self._n_hr_extra = len(set(hr_files) - set(lr_files))

        # ---- canonical order + split (every holdout-th, starting at 0) ----
        full = sorted(lr_files)
        self._full_names = full
        test_full = [n for i, n in enumerate(full) if i % self._holdout == 0]
        test_set = set(test_full)
        train_full = [n for n in full if n not in test_set]
        if self._train_limit > 0:
            train_final = [train_full[i] for i in _even_indices(len(train_full), self._train_limit)]
        else:
            train_final = train_full
        if self._test_limit > 0:
            test_final = [test_full[i] for i in _even_indices(len(test_full), self._test_limit)]
        else:
            test_final = test_full
        self._train_names = train_final
        self._test_names = test_final

        # ---- lazy caches ----
        self._lock = threading.RLock()
        self._dims: Dict[Tuple[str, str], Tuple[int, int]] = {}  # (tier, name) -> (w, h)
        self._dims_hist: Dict[str, List[Tuple[int, int, int]]] = {}
        self._images_cache = _LRUCache(cache_bytes)

        # ---- fail-fast: measure every paired file and check the 4x guard ----
        self._validate_pair_geometry()

    # ------------------------------------------------------------------ setup

    def _read_sparse(self):
        cm = self._colmap
        kind = {}
        for stem, loaders in (
            ("cameras", (cm.read_intrinsics_binary, cm.read_intrinsics_text)),
            ("images", (cm.read_extrinsics_binary, cm.read_extrinsics_text)),
            ("points3D", (cm.read_points3D_binary, cm.read_points3D_text)),
        ):
            bin_path = self._sparse_dir / f"{stem}.bin"
            txt_path = self._sparse_dir / f"{stem}.txt"
            if bin_path.is_file():
                kind[stem] = (bin_path, loaders[0])
            elif txt_path.is_file():
                kind[stem] = (txt_path, loaders[1])
            else:
                kind[stem] = (None, None)
        if kind["cameras"][0] is None or kind["images"][0] is None:
            raise DatasetError(
                f"COLMAP model incomplete in {self._sparse_dir}: need cameras and "
                "images (.bin or .txt); points3D optional"
            )
        try:
            cameras = kind["cameras"][1](str(kind["cameras"][0]))
            images = kind["images"][1](str(kind["images"][0]))
        except AssertionError as exc:
            # the vendored text reader asserts PINHOLE; surface it as DatasetError
            raise DatasetError(
                f"cannot read text-format COLMAP model in {self._sparse_dir}: {exc} "
                "(text sparse models are limited to PINHOLE cameras)"
            ) from exc
        points_path = kind["points3D"][0]
        return cameras, images, points_path

    def _intrinsics_from_colmap(self, cameras) -> None:
        if not cameras:
            raise DatasetError(f"COLMAP model has no cameras in {self._sparse_dir}")
        for cam_id, cam in cameras.items():
            params = np.asarray(cam.params, dtype=np.float64)
            model = str(cam.model)
            if model == "PINHOLE" and params.size == 4:
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE" and params.size == 3:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise DatasetError(
                    f"unsupported COLMAP camera model {model!r} (camera id {cam_id}, "
                    f"{params.size} params); only PINHOLE / SIMPLE_PINHOLE are supported"
                )
            if int(cam.width) <= 0 or int(cam.height) <= 0:
                raise DatasetError(f"COLMAP camera {cam_id} has invalid size {cam.width}x{cam.height}")
            self._cam_intr[int(cam_id)] = _Intrinsics(
                model=model,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                ref_w=int(cam.width),
                ref_h=int(cam.height),
            )

    @staticmethod
    def _image_files(tier_dir: Path) -> Dict[str, Path]:
        files = {}
        with os.scandir(tier_dir) as it:
            for entry in it:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in _IMAGE_EXTS:
                    files[entry.name] = Path(entry.path)
        if not files:
            raise DatasetError(f"no images found in {tier_dir}")
        return files

    def _validate_pair_geometry(self) -> None:
        """Fail fast if any LR/HR pair deviates from the nominal 4x scale."""
        bad = []
        for name in self._full_names:
            lr_w, lr_h = self._eff_dims("lr", name)
            hr_w, hr_h = self._eff_dims("hr", name)
            rw = hr_w / lr_w
            rh = hr_h / lr_h
            if abs(rw / NOMINAL_TIER_SCALE - 1.0) > TIER_SCALE_TOLERANCE or abs(
                rh / NOMINAL_TIER_SCALE - 1.0
            ) > TIER_SCALE_TOLERANCE:
                bad.append((name, lr_w, lr_h, hr_w, hr_h, round(rw, 3), round(rh, 3)))
                if len(bad) >= 5:
                    break
        if bad:
            rows = "; ".join(
                f"{n}: lr {w}x{h} vs hr {W}x{H} (ratios {rw}x{rh})" for n, w, h, W, H, rw, rh in bad
            )
            raise DatasetError(
                f"LR<->HR pixel scale mismatch: expected ~{NOMINAL_TIER_SCALE}x "
                f"within {TIER_SCALE_TOLERANCE:.0%} per axis, got {rows}. "
                f"{self._hr_dir} is not the matching ~4x tier of {self._lr_dir} "
                "(wrong tier or mismatched generated targets). No resizing was performed."
            )

    # ------------------------------------------------------- dimension handling

    def _native_dims(self, tier: str, name: str) -> Tuple[int, int]:
        key = (tier, name)
        with self._lock:
            hit = self._dims.get(key)
            if hit is not None:
                return hit
        path = self._tier_files(tier)[name]
        try:
            with Image.open(path) as im:
                dims = im.size  # header only; no pixel decode
        except Exception as exc:  # noqa: BLE001 - surface any read failure clearly
            raise DatasetError(f"cannot read image header {path}: {exc}") from exc
        if dims[0] <= 0 or dims[1] <= 0:
            raise DatasetError(f"image {path} reports invalid size {dims[0]}x{dims[1]}")
        with self._lock:
            self._dims.setdefault(key, dims)
            return self._dims[key]

    def _eff_dims(self, tier: str, name: str) -> Tuple[int, int]:
        """Pixel size actually used for (tier, name) after the max_width cap.

        The downscale factor is derived once from the *HR* native width and
        applied to both tiers, so HR width <= max_width while the LR<->HR
        ratio stays ~4x (rounding aside).
        """
        w, h = self._native_dims(tier, name)
        if self._max_width > 0:
            hr_w = w if tier == "hr" else self._native_dims("hr", name)[0]
            factor = min(1.0, self._max_width / float(hr_w))
            if factor < 1.0:
                w = max(1, int(round(w * factor)))
                h = max(1, int(round(h * factor)))
        return w, h

    def _tier_files(self, tier: str) -> Dict[str, Path]:
        if tier == "lr":
            return self._lr_files
        if tier == "hr":
            return self._hr_files
        raise DatasetError(f"unknown tier {tier!r} (expected 'lr' or 'hr')")

    # ------------------------------------------------------------------ camera

    def _make_camera(self, name: str, tier: str, device) -> Camera:
        if name not in self._name_cam:
            raise DatasetError(
                f"unknown image {name!r}; dataset has {len(self._name_cam)} images "
                f"(train {len(self._train_names)}, test {len(self._test_names)})"
            )
        intr = self._cam_intr[self._name_cam[name]]
        qvec, tvec = self._name_pose[name]
        w, h = self._eff_dims(tier, name)
        rot = self._colmap.qvec2rotmat(qvec)  # world -> camera rotation
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rot
        w2c[:3, 3] = tvec
        return Camera(
            name=name,
            width=w,
            height=h,
            fx=intr.fx * (w / intr.ref_w),
            fy=intr.fy * (h / intr.ref_h),
            cx=intr.cx * (w / intr.ref_w),
            cy=intr.cy * (h / intr.ref_h),
            w2c=torch.tensor(w2c, dtype=torch.float32, device=device),
        )

    def cameras(self, split: str = "train", resolution: str = "lr", device="cuda") -> List[Camera]:
        """Cameras of one split/resolution; dimension headers only, no pixels.

        ``split`` in {'train','test'}, ``resolution`` in {'lr','hr'} (hr is the
        ~4x oracle tier of the same pose).  Images are never decoded here.
        """
        if split == "train":
            names = self._train_names
        elif split == "test":
            names = self._test_names
        else:
            raise DatasetError(f"unknown split {split!r} (expected 'train' or 'test')")
        if resolution not in ("lr", "hr"):
            raise DatasetError(f"unknown resolution {resolution!r} (expected 'lr' or 'hr')")
        return [self._make_camera(n, resolution, device) for n in names]

    # ------------------------------------------------------------------- image

    def _image_tensor(self, name: str, tier: str) -> torch.Tensor:
        key = (tier, name)
        with self._lock:
            hit = self._images_cache.get(key)
            if hit is not None:
                return hit
        eff_w, eff_h = self._eff_dims(tier, name)
        path = self._tier_files(tier)[name]
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise DatasetError(f"cannot decode image {path}: {exc}") from exc
        if rgb.size != (eff_w, eff_h):
            if rgb.size != self._native_dims(tier, name):
                raise DatasetError(
                    f"image changed on disk since dataset construction: {path} was "
                    f"{self._native_dims(tier, name)}, now {rgb.size}"
                )
            rgb = rgb.resize((eff_w, eff_h), Image.Resampling.LANCZOS)
        arr = np.array(rgb)  # writable uint8 HxWx3 copy
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32).div_(255.0)
        with self._lock:
            self._images_cache.put(key, tensor)
        return tensor

    def pair(self, name: str, device="cuda"):
        """(lr_camera, lr_image, hr_camera, hr_image) for one posed frame.

        Images are float32 CPU CHW in [0, 1] (bounded LRU cached); cameras sit
        on ``device``.  Accepts any registered name (train or test); the caller
        decides which split it is optimizing.
        """
        if name not in self._name_cam:
            raise DatasetError(
                f"unknown image {name!r}; dataset has {len(self._name_cam)} images "
                f"(train {len(self._train_names)}, test {len(self._test_names)})"
            )
        return (
            self._make_camera(name, "lr", device),
            self._image_tensor(name, "lr"),
            self._make_camera(name, "hr", device),
            self._image_tensor(name, "hr"),
        )

    def view(self, name: str, resolution: str = "lr", device="cuda"):
        """Load one tier only; L0 training need not decode any HR targets."""
        return self._make_camera(name, resolution, device), self._image_tensor(name, resolution)

    # ------------------------------------------------------------------ points

    def _points_count(self) -> Optional[int]:
        """Sparse point count from the file header only (no decode)."""
        if self._points is not None:
            return int(self._points.shape[0])
        if self._points_path is None:
            return None
        if self._points_path.suffix == ".bin":
            with open(self._points_path, "rb") as fid:
                (n,) = self._colmap.read_next_bytes(fid, 8, "Q")
            return int(n)
        n = 0
        with open(self._points_path, "r") as fid:
            for line in fid:
                line = line.strip()
                if line and not line.startswith("#"):
                    n += 1
        return n

    def _load_points(self) -> None:
        if self._points is not None:
            return
        with self._lock:
            if self._points is not None:
                return
            if self._points_path is None:
                raise DatasetError(f"no points3D (.bin/.txt) in {self._sparse_dir}")
            xyzs, rgbs, _errors = self._colmap.read_points3D_binary(
                str(self._points_path)
            ) if self._points_path.suffix == ".bin" else self._colmap.read_points3D_text(
                str(self._points_path)
            )
            self._points = torch.as_tensor(np.asarray(xyzs, dtype=np.float32))
            self._colors = (torch.as_tensor(np.asarray(rgbs, dtype=np.float32)) / 255.0).clamp_(0.0, 1.0)

    @property
    def points(self) -> torch.Tensor:
        """Sparse COLMAP point positions, CPU float32 [N, 3]."""
        self._load_points()
        return self._points

    @property
    def colors(self) -> torch.Tensor:
        """Sparse COLMAP point colors, CPU float32 [N, 3] in [0, 1]."""
        self._load_points()
        return self._colors

    # ------------------------------------------------------------------ names

    @property
    def names(self) -> List[str]:
        """All registered basenames, lexicographically sorted (pre-limit)."""
        return list(self._full_names)

    @property
    def train_names(self) -> List[str]:
        """Training basenames (post-limit), sorted order."""
        return list(self._train_names)

    @property
    def test_names(self) -> List[str]:
        """Held-out basenames (post-limit), sorted order."""
        return list(self._test_names)

    # ---------------------------------------------------------------- manifest

    def _tier_dims_hist(self, tier: str) -> List[Tuple[int, int, int]]:
        with self._lock:
            hist = self._dims_hist.get(tier)
        if hist is None:
            counts: Dict[Tuple[int, int], int] = {}
            for name in self._full_names:
                d = self._native_dims(tier, name)
                counts[d] = counts.get(d, 0) + 1
            hist = sorted((w, h, c) for (w, h), c in counts.items())
            with self._lock:
                self._dims_hist[tier] = hist
        return hist

    def manifest(self) -> dict:
        """Serializable description of the scene, split and supervision.

        ``supervision.kind`` is *always* "oracle" for the default ``images_2``
        HR tier: HR targets are real captured references, so training against
        them is oracle supervision (validating LoD fine-tuning, layer freeze,
        continuous opacity), not generative LR-only SR.
        """
        lr_hist = self._tier_dims_hist("lr")
        hr_hist = self._tier_dims_hist("hr")
        lr_w, lr_h, _ = max(lr_hist, key=lambda t: t[2])
        hr_w, hr_h, _ = max(hr_hist, key=lambda t: t[2])
        smoke = self._max_width > 0
        return {
            "root": str(self._root),
            "lr_dir": self._lr_dir,
            "hr_dir": self._hr_dir,
            "sparse_dir": str(self._sparse_dir),
            "n_colmap_cameras": len(self._cam_intr),
            "colmap_cameras": [
                {
                    "camera_id": cid,
                    "model": intr.model,
                    "ref_width": intr.ref_w,
                    "ref_height": intr.ref_h,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "cx": intr.cx,
                    "cy": intr.cy,
                }
                for cid, intr in sorted(self._cam_intr.items())
            ],
            "train_names": list(self._train_names),
            "test_names": list(self._test_names),
            "split": {
                "convention": "lexicographic basename order; test indices i % holdout == 0",
                "holdout": self._holdout,
                "n_images": len(self._full_names),
                "n_train": len(self._train_names),
                "n_test": len(self._test_names),
                "train_limit": self._train_limit,
                "test_limit": self._test_limit,
                "limits_applied": self._train_limit > 0 or self._test_limit > 0,
                "test_head": list(self._test_names[:8]),
            },
            "max_width": {
                "enabled": smoke,
                "value": self._max_width,
                "note": (
                    "if enabled, both tiers are downscaled by one common factor so HR "
                    "width <= max_width; smoke-protocol only, not the native "
                    "648x420 -> 2594x1681 full protocol"
                ),
            },
            "tiers": {
                "lr": {"dir": self._lr_dir, "n_files": len(self._full_names), "dims": lr_hist},
                "hr": {
                    "dir": self._hr_dir,
                    "n_files": len(self._full_names) + self._n_hr_extra,
                    "n_paired": len(self._full_names),
                    "n_extra_unpaired": self._n_hr_extra,
                    "dims": hr_hist,
                },
            },
            "pixel_scale_lr_to_hr": {
                "width": round(hr_w / lr_w, 4),
                "height": round(hr_h / lr_h, 4),
                "nominal": NOMINAL_TIER_SCALE,
                "tolerance": TIER_SCALE_TOLERANCE,
            },
            "n_points": self._points_count(),
            "supervision": {
                "kind": "external" if Path(self._hr_dir).is_absolute() else "oracle",
                "detail": (
                    "HR cache paired to fixed-FOV COLMAP views; external caches carry "
                    "user-supplied targets, while images_2 supplies oracle captured HR. "
                    "Oracle optimization is not generative LR-only SR."
                ),
                "split_note": "LR and HR share the identical holdout split; no HR target of any test frame is reachable through train pairs.",
            },
            "read_only": True,
        }

    def __repr__(self) -> str:
        return (
            f"Mip360Dataset(root={str(self._root)!r}, lr={self._lr_dir}, hr={self._hr_dir}, "
            f"train={len(self._train_names)}, test={len(self._test_names)}, "
            f"max_width={self._max_width})"
        )
