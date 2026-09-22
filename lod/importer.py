"""Import a frozen Skyfall-GS Stage1 model as GaussianZoom L0.

Copies xyz, SH, log-scales, rotations, opacity logits, and the already computed
``filter_3D``. Appearance embeddings and the appearance MLP stay attached and
frozen; they are not baked into SH. LoD ``step_scale`` is a focal-length ratio,
never an HR/LR pixel-size ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from .camera import camera_image_stem, skyfall_camera_to_lod, zoom_stage_camera
from .path import add_gz_src


def _require_filter(gaussians) -> torch.Tensor:
    filt = getattr(gaussians, "filter_3D", None)
    if filt is None or not torch.is_tensor(filt) or int(filt.numel()) != int(gaussians._xyz.shape[0]):
        raise ValueError(
            "import_skyfall_l0 requires gaussians.filter_3D already computed "
            "for the L0 camera set; refusing to invent a sampling filter."
        )
    if filt.ndim == 1:
        filt = filt.reshape(-1, 1)
    if tuple(filt.shape) != (int(gaussians._xyz.shape[0]), 1):
        raise ValueError(f"filter_3D must be [N, 1], got {tuple(filt.shape)}")
    return filt


@dataclass
class FrozenAppearance:
    """Skyfall per-image embedding + per-Gaussian Fourier features + MLP."""

    enabled: bool
    n_fourier_freqs: int | None
    embedding_dim: int | None
    gaussian_embeddings: torch.Tensor | None
    image_embeddings: torch.Tensor | None
    mlp: nn.Module | None
    layer_embeddings: list[torch.Tensor] = field(default_factory=list)

    def freeze(self) -> "FrozenAppearance":
        if self.gaussian_embeddings is not None:
            self.gaussian_embeddings.requires_grad_(False)
        if self.image_embeddings is not None:
            self.image_embeddings.requires_grad_(False)
        for emb in self.layer_embeddings:
            if torch.is_tensor(emb):
                emb.requires_grad_(False)
        if self.mlp is not None:
            self.mlp.eval()
            for param in self.mlp.parameters():
                param.requires_grad_(False)
        return self

    def embeddings_for_levels(self, n_levels: int) -> torch.Tensor:
        if self.layer_embeddings:
            parts = self.layer_embeddings[:n_levels]
        elif self.gaussian_embeddings is not None:
            parts = [self.gaussian_embeddings]
        else:
            raise ValueError("appearance has no per-Gaussian embeddings")
        return torch.cat(parts, dim=0)

    def embedding_for_uid(self, uid: int, *, is_train_view: bool) -> torch.Tensor | None:
        if not self.enabled or self.image_embeddings is None:
            return None
        if is_train_view:
            return self.image_embeddings[int(uid)].detach()
        return self.image_embeddings.mean(dim=0).detach()


@dataclass
class SkyfallL0:
    lod: Any
    appearance: FrozenAppearance
    lod_cameras: list[Any]
    step_scale: float
    sh_degree: int
    n_points: int
    filter_source: str = "copied_from_skyfall"
    l1_cameras: list[Any] | None = None
    stage_cameras: list[Any] | None = None

    def layer0(self):
        return self.lod.layers[0]

    def layer1(self):
        return self.layer(1)

    def layer(self, index: int):
        if index < 0 or index >= len(self.lod.layers):
            return None
        return self.lod.layers[index]

    def active_layer(self):
        return self.lod.layers[int(self.lod.active_level)]


def import_skyfall_l0(
    gaussians,
    cameras: Sequence,
    *,
    gz_root: str | None = None,
    step_scale: float = 2.0,
    freeze: bool = True,
) -> SkyfallL0:
    """Build a frozen GaussianLoD L0 from a loaded Skyfall ``GaussianModel``.

    ``cameras`` must be the L0 training cameras (original focal length, original
    raster). Zoom cameras are converted separately for rendering checks and must
    not be mixed into this stage: a 2x Skyfall zoom keeps 2048 pixels.
    """

    add_gz_src(gz_root)
    from gaussianzoom_lod.model import GaussianLayer, GaussianLoD, _creation_psi

    if not cameras:
        raise ValueError("import_skyfall_l0 requires the L0 training cameras.")
    filt = _require_filter(gaussians).detach().to(dtype=torch.float32)
    xyz = gaussians._xyz.detach().to(dtype=torch.float32).contiguous()
    sh = gaussians.get_features.detach().to(dtype=torch.float32).contiguous()
    log_scales = gaussians._scaling.detach().to(dtype=torch.float32).contiguous()
    rotations = gaussians._rotation.detach().to(dtype=torch.float32).contiguous()
    opacity_logits = gaussians._opacity.detach().to(dtype=torch.float32).contiguous()
    degree = int(gaussians.max_sh_degree)
    expected_k = (degree + 1) ** 2
    if int(sh.shape[1]) != expected_k:
        raise ValueError(f"SH width {int(sh.shape[1])} does not match degree {degree}")

    lod_cameras = [skyfall_camera_to_lod(camera, gz_root=gz_root, device=xyz.device) for camera in cameras]
    names = [cam.name for cam in lod_cameras]
    if len(names) != len(set(names)):
        raise ValueError(f"L0 cameras must have unique names, got {names}")

    model = GaussianLoD(degree=degree, step_scale=float(step_scale))
    psi_ref = _creation_psi(xyz, lod_cameras)
    layer = GaussianLayer(
        level=0,
        xyz=xyz.clone(),
        sh=sh.clone(),
        log_scales=log_scales.clone(),
        rotations=rotations.clone(),
        opacity_logits=opacity_logits.clone(),
        psi_ref=psi_ref,
        frozen=bool(freeze),
        filter_3d=filt.clone(),
        node_ids=model._alloc_ids(int(xyz.shape[0]), device=xyz.device),
    )
    model.layers.append(layer)
    model.active_level = 0
    model.bind_base_stage(lod_cameras)
    if freeze:
        layer.freeze()

    l0_embeddings = None
    if gaussians._embeddings is not None:
        l0_embeddings = gaussians._embeddings.detach().clone()
        l0_embeddings.requires_grad_(False)
    appearance = FrozenAppearance(
        enabled=bool(gaussians.appearance_enabled),
        n_fourier_freqs=getattr(gaussians, "appearance_n_fourier_freqs", None),
        embedding_dim=getattr(gaussians, "appearance_embedding_dim", None),
        gaussian_embeddings=l0_embeddings,
        image_embeddings=gaussians.appearance_embeddings,
        mlp=gaussians.appearance_mlp,
        layer_embeddings=[l0_embeddings] if l0_embeddings is not None else [],
    ).freeze()

    return SkyfallL0(
        lod=model,
        appearance=appearance,
        lod_cameras=lod_cameras,
        step_scale=float(step_scale),
        sh_degree=degree,
        n_points=int(xyz.shape[0]),
    )


def assert_l0_tensors_match(gaussians, bundle: SkyfallL0, *, rtol: float = 0.0, atol: float = 0.0) -> dict[str, float]:
    """Check that L0 owns copies of the Skyfall parameters and filter."""

    layer = bundle.layer0()
    checks = {
        "xyz": (gaussians._xyz, layer.xyz),
        "sh": (gaussians.get_features, layer.sh),
        "log_scales": (gaussians._scaling, layer.log_scales),
        "rotations": (gaussians._rotation, layer.rotations),
        "opacity_logits": (gaussians._opacity, layer.opacity_logits),
        "filter_3d": (gaussians.filter_3D, layer.filter_3d),
    }
    report: dict[str, float] = {}
    for name, (src, dst) in checks.items():
        left = src.detach().float().reshape(dst.shape)
        right = dst.detach().float()
        delta = (left - right).abs().max().item()
        report[name] = float(delta)
        if not torch.allclose(left, right, rtol=rtol, atol=atol):
            raise AssertionError(f"L0 {name} differs from Skyfall after import (max abs={delta}).")
    if not layer.frozen:
        raise AssertionError("Imported L0 must be frozen.")
    if bundle.appearance.enabled:
        if bundle.appearance.mlp is None or bundle.appearance.image_embeddings is None:
            raise AssertionError("Skyfall appearance is enabled but the importer dropped the MLP or embeddings.")
        if any(param.requires_grad for param in bundle.appearance.mlp.parameters()):
            raise AssertionError("Imported appearance MLP is not frozen.")
    return report


def copy_l0_into_gaussian_model(bundle: SkyfallL0, gaussians) -> None:
    """Write L0 tensors back into a Skyfall model for the native rasterizer.

    Used only for the import-consistency check. Does not recompute ``filter_3D``.
    """

    layer = bundle.layer0()
    gaussians._xyz = layer.xyz
    sh = layer.sh
    gaussians._features_dc = sh[:, :1, :].contiguous()
    gaussians._features_rest = sh[:, 1:, :].contiguous()
    gaussians._scaling = layer.log_scales
    gaussians._rotation = layer.rotations
    gaussians._opacity = layer.opacity_logits
    gaussians.filter_3D = layer.filter_3d
    if bundle.appearance.enabled:
        gaussians._embeddings = bundle.appearance.gaussian_embeddings
        gaussians.appearance_embeddings = bundle.appearance.image_embeddings
        gaussians.appearance_mlp = bundle.appearance.mlp
        gaussians.appearance_enabled = True


def extend_base_stage(bundle: SkyfallL0, skyfall_cameras: Sequence, *, gz_root: str | None = None):
    """Append extra 1× cameras to the L0 stage record without recomputing ``psi_ref``.

    Synthetic Episode cameras are not in the Stage1 training rig. ``add_level``
    still keys the next stage by name and pose against the base record, so the
    1× Episode camera must be present there. Train-camera ``psi_ref`` stays as
    imported.
    """

    if not skyfall_cameras:
        raise ValueError("extend_base_stage needs at least one 1× camera.")
    if len(bundle.lod.layers) != 1 or int(bundle.lod.active_level) != 0:
        raise ValueError("extend_base_stage only before adding detail levels")
    if not bundle.lod.stage_records:
        raise ValueError("extend_base_stage requires a bound L0 base stage")
    add_gz_src(gz_root)
    from gaussianzoom_lod.stages import capture_stage

    device = bundle.layer0().xyz.device
    extra = [
        skyfall_camera_to_lod(camera, gz_root=gz_root, device=device, use_skyfall_center=True)
        for camera in skyfall_cameras
    ]
    captured = capture_stage(extra, scale=1.0)
    record = bundle.lod.stage_records[0]
    existing = {cam["name"] for cam in record["cameras"]}
    for cam in captured["cameras"]:
        if cam["name"] in existing:
            raise ValueError(f"base stage already has camera {cam['name']!r}")
        record["cameras"].append(cam)
        existing.add(cam["name"])
    return extra


def add_detail_level(
    bundle: SkyfallL0,
    skyfall_cameras: Sequence,
    roi,
    factor: float,
    *,
    rois: Sequence | None = None,
    gz_root: str | None = None,
):
    """Freeze older levels and append an empty detail level.

    ``factor`` is the new level's focal zoom versus L0 (2 for L1, 4 for L2).
    Stage cameras keep L0 names so ``validate_next_stage`` can check the focal step.
    One camera per L0 name: pass a matching ``rois`` sequence for joint coverage.

    The adjacent focal ratio is ``factor / parent_stage_scale``, not the model-wide
    ``step_scale``. Nonuniform intervals (4× then 2×) keep the stored camera scales
    as the LoD reference; do not set ``step_scale=4`` or L2 becomes 16×.
    """

    if not skyfall_cameras:
        raise ValueError("add_detail_level needs the stage cameras (same names as L0).")
    if rois is None:
        rois = [roi] * len(skyfall_cameras)
    if len(rois) != len(skyfall_cameras):
        raise ValueError(f"rois ({len(rois)}) must match cameras ({len(skyfall_cameras)})")
    names = [camera_image_stem(getattr(camera, "image_name", "")) for camera in skyfall_cameras]
    if len(names) != len(set(names)):
        raise ValueError(f"stage cameras must have unique L0 names, got {names}")
    if not bundle.lod.stage_records:
        raise ValueError("add_detail_level requires a bound L0 base stage")
    parent_scale = float(bundle.lod.stage_records[-1]["scale"])
    adjacent = float(factor) / parent_scale
    if not (adjacent > 1.0):
        raise ValueError(
            f"add_detail_level factor={factor} must exceed parent stage scale {parent_scale}; "
            "do not change the global step_scale to invent later stages."
        )
    device = bundle.layer0().xyz.device
    stage_cameras = [
        zoom_stage_camera(camera, tile_roi, factor, gz_root=gz_root, device=device)
        for camera, tile_roi in zip(skyfall_cameras, rois)
    ]
    old_step = float(bundle.lod.step_scale)
    bundle.lod.step_scale = adjacent
    try:
        bundle.lod.add_level(stage_cameras)
    finally:
        bundle.lod.step_scale = old_step
    if bundle.appearance.gaussian_embeddings is not None:
        dim = int(bundle.appearance.gaussian_embeddings.shape[1])
        empty = bundle.appearance.gaussian_embeddings.new_zeros((0, dim))
        while len(bundle.appearance.layer_embeddings) < len(bundle.lod.layers):
            bundle.appearance.layer_embeddings.append(empty)
        bundle.appearance.layer_embeddings[-1] = empty
    bundle.stage_cameras = stage_cameras
    if len(bundle.lod.layers) == 2:
        bundle.l1_cameras = stage_cameras
    return stage_cameras


def _ancestor_embedding_table(bundle: SkyfallL0, child_level: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ancestor node ids onto frozen appearance rows."""

    ids = []
    embs = []
    for level in range(child_level):
        layer = bundle.lod.layers[level]
        if int(layer.xyz.shape[0]) == 0:
            continue
        if level >= len(bundle.appearance.layer_embeddings):
            raise ValueError(f"missing appearance embeddings for frozen level {level}")
        ids.append(layer.node_ids)
        embs.append(bundle.appearance.layer_embeddings[level])
    if not ids:
        raise ValueError("detail levels need at least one ancestor with embeddings")
    all_ids = torch.cat(ids, dim=0)
    all_emb = torch.cat(embs, dim=0)
    mapper_size = int(all_ids.max().item()) + 1
    mapper = torch.full((mapper_size,), -1, dtype=torch.long, device=all_ids.device)
    mapper[all_ids] = torch.arange(int(all_ids.numel()), device=all_ids.device, dtype=torch.long)
    return mapper, all_emb


def sync_layer_embeddings(bundle: SkyfallL0) -> None:
    """Copy ancestor appearance embeddings onto the active detail level. Values are constants.

    L2 may be born from L1 or L0 parents; lookup uses node ids across frozen layers.
    Older layer embeddings are left untouched. Color still flows through the frozen MLP.
    """

    if not bundle.appearance.enabled or bundle.appearance.gaussian_embeddings is None:
        return
    if len(bundle.lod.layers) < 2:
        return
    child_level = int(bundle.lod.active_level)
    layer = bundle.lod.layers[child_level]
    mapper, table = _ancestor_embedding_table(bundle, child_level)
    n = int(layer.xyz.shape[0])
    if n == 0:
        bundle.appearance.layer_embeddings[child_level] = table.new_zeros((0, table.shape[1]))
        return
    parent = layer.parent_ids
    if bool((parent < 0).any().item()):
        raise ValueError(f"L{child_level} rows must inherit a parent_id; found a root.")
    mapper_size = int(mapper.numel())
    if bool((parent >= mapper_size).any().item()):
        raise ValueError(f"L{child_level} parent_id is outside ancestor node ids.")
    rows = mapper[parent]
    if bool((rows < 0).any().item()):
        raise ValueError(f"L{child_level} parent_id does not match any frozen ancestor.")
    inherited = table[rows].detach().clone()
    inherited.requires_grad_(False)
    bundle.appearance.layer_embeddings[child_level] = inherited
    bundle.appearance.gaussian_embeddings.requires_grad_(False)


def densify_with_appearance(bundle: SkyfallL0, optimizer, grads, cameras, **kwargs):
    """``GaussianLoD.densify`` then inherit frozen parent embeddings onto new rows."""

    stats = bundle.lod.densify(optimizer, grads, cameras=cameras, **kwargs)
    sync_layer_embeddings(bundle)
    return stats


def save_bundle(bundle: SkyfallL0, path: str, optimizer=None) -> None:
    bundle.lod.save(path, optimizer=optimizer, metadata={"filter_source": bundle.filter_source})
    side = {
        "layer_embeddings": [emb.detach().cpu() for emb in bundle.appearance.layer_embeddings],
        "n_fourier_freqs": bundle.appearance.n_fourier_freqs,
        "embedding_dim": bundle.appearance.embedding_dim,
        "enabled": bundle.appearance.enabled,
    }
    torch.save(side, path + ".appearance.pt")


def load_lod_onto_bundle(bundle: SkyfallL0, path: str, *, device: str = "cuda"):
    """Restore LoD tensors and layer embeddings; keep the frozen Skyfall MLP in place.

    Returns the saved optimizer state dict, or ``None``.
    """

    add_gz_src()
    from gaussianzoom_lod.model import GaussianLoD

    model, _, opt_state = GaussianLoD.load(path, device=device)
    bundle.lod = model
    side_path = path + ".appearance.pt"
    if bundle.appearance.enabled:
        side = torch.load(side_path, map_location=device, weights_only=False)
        restored = [tensor.to(device=device).requires_grad_(False) for tensor in side["layer_embeddings"]]
        bundle.appearance.layer_embeddings = restored
        if restored:
            bundle.appearance.gaussian_embeddings = restored[0]
    bundle.n_points = int(model.layers[0].xyz.shape[0])
    return opt_state

