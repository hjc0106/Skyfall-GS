#!/usr/bin/env python3
"""Internal influence of a 1px roundtrip gate: pixel mask → feature mask → CFR → image.

Fixes the 4x target, neighbor, prompt, seed, and DLoRAL weights. Does not train 3D
and does not turn the gate on by default. ``target_only`` is the no-neighbor endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lod.warp_diag import (
    GHOSTING_CROPS,
    colorize_scalar,
    contact_sheet,
    crop_image,
    feature_readmission_stats,
    fractional_crop,
)
from refinement.dloral_backend import DLoRALBackend
from refinement.types import CameraSnapshot, MultiViewInput, PromptDescription, RefinementRequest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path)).astype(bool)
    return np.asarray(Image.open(path).convert("L")) > 127


def _l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    residual = np.abs(a - b).mean(axis=-1)
    if mask is None:
        return float(residual.mean())
    if not bool(mask.any()):
        return None
    return float(residual[mask].mean())


def _mean(plane: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is None:
        return float(np.asarray(plane).mean())
    if not bool(mask.any()):
        return None
    return float(np.asarray(plane)[mask].mean())


def _crop_stats(a: np.ndarray, b: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray | None = None) -> float | None:
    x0, y0, x1, y1 = box
    region = np.zeros(a.shape[:2], dtype=bool)
    region[y0:y1, x0:x1] = True
    if mask is not None:
        region = region & mask
    return _l1(a, b, region)


def _prompt_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = str(payload.get("target_prompt") or payload.get("prompt") or "")
    if not text and isinstance(payload.get("config"), dict):
        text = str(payload["config"].get("target_prompt") or "")
    if not text.strip():
        raise ValueError(f"empty prompt in {path}")
    return text


def _as_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None]
    return array


def _l1_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(_as_chw(a) - _as_chw(b)).mean(axis=0)


def _upsample_nearest(plane: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(plane, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    else:
        tensor = tensor[None]
    out = F.interpolate(tensor, size=(height, width), mode="nearest")
    return out[0, 0].numpy() if out.shape[1] == 1 else out[0].numpy()


def _region_stats(plane: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float | None]:
    return {name: _mean(plane, mask) for name, mask in masks.items()}


def _readmission_block(
    name: str,
    region: torch.Tensor,
    feat_valid: torch.Tensor,
    pixel_valid: torch.Tensor | None,
) -> dict:
    stats = feature_readmission_stats(region, feat_valid, pixel_valid=pixel_valid)
    stats["label"] = name
    return stats


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if abs(den) < 1e-12:
        return None if abs(num) < 1e-12 else float("inf")
    return float(num / den)


def _stage_answers(observed: dict) -> dict:
    fused = observed["fused"]
    aligned = observed["aligned"]["baseline_vs_1px"]
    red_readmit = observed["readmission"]["1px"]["red"]
    return {
        "pixel_to_feature": {
            "red_pixels_still_in_valid_feature_cell_frac": red_readmit["pixels_in_valid_feature_cell_frac"],
            "red_overlapping_feature_cells_valid_frac": red_readmit["overlapping_feature_cells_valid_frac"],
            "meaning": "A 100% pixel catch on red does not make those locations invalid at feature resolution.",
        },
        "aligned_to_fused": {
            "aligned_l1_on_red_now_invalid": aligned["red_now_invalid"],
            "aligned_l1_on_red_still_valid": aligned["red_still_valid"],
            "fused_l1_on_red_now_invalid_vs_1px": fused["baseline_vs_1px"]["red_now_invalid"],
            "fused_1px_vs_target_only_on_red_now_invalid": fused["gated_vs_target_only"]["red_now_invalid"],
            "fused_1px_vs_target_only_on_red_still_valid": fused["gated_vs_target_only"]["red_still_valid"],
            "meaning": "Where the feature cell is actually dropped, fusion matches target_only. Remaining valid red cells still carry neighbor features.",
        },
        "fused_to_image": {
            "fused_gate_over_target_only_red": _ratio(
                fused["baseline_vs_1px"]["red_cells"], fused["baseline_vs_target_only"]["red_cells"]
            ),
            "image_gate_over_target_only_red": _ratio(
                observed["image"]["baseline_vs_1px"]["red"],
                observed["image"]["baseline_vs_target_only"]["red"],
            ),
            "red_corr_fused_vs_image_gate": observed["weakening"]["red_corr_fused_vs_image_gate"],
            "meaning": "Image L1 magnitude tracks the target_only endpoint more than it tracks the spatial fused-L1 map.",
        },
        "visual_check": "If target_only still shows the same ghosting structure as baseline, that structure is target input plus generative prior even when RGB L1 is not tiny.",
    }


def _suggest_focus(observed: dict) -> dict:
    """Point to correspondence, fusion, or generation using stage ratios, not image quality."""

    readmit = observed["readmission"]["1px"]["red"]["pixels_in_valid_feature_cell_frac"]
    fused_gate = observed["fused"]["baseline_vs_1px"]["red_cells"]
    fused_to = observed["fused"]["baseline_vs_target_only"]["red_cells"]
    image_gate = observed["image"]["baseline_vs_1px"]["red"]
    image_to = observed["image"]["baseline_vs_target_only"]["red"]
    fused_dropped_vs_to = observed["fused"]["gated_vs_target_only"]["red_now_invalid"]
    fused_ratio = _ratio(fused_gate, fused_to)
    image_ratio = _ratio(image_gate, image_to)
    reasons: list[str] = []
    focus: list[str] = []
    if readmit is not None and readmit > 0.2:
        focus.append("feature_aggregation")
        reasons.append(
            f"After the 1px pixel gate, {readmit:.3f} of original red pixels still sit in a valid feature cell."
        )
    if fused_dropped_vs_to is not None and fused_dropped_vs_to < 1e-6:
        reasons.append(
            "On feature cells the gate actually drops, fused output equals target_only; fusion is not ignoring the mask."
        )
    if image_to is not None and image_to < 0.008:
        focus.append("target_and_generative_prior")
        reasons.append(
            f"target_only vs baseline red L1 is {image_to:.4f}; removing neighbors does not move the image much."
        )
    if fused_ratio is not None and fused_ratio < 0.25 and (readmit is None or readmit < 0.2):
        focus.append("fusion_sensitivity")
        reasons.append(
            f"Fused red-cell L1 gate/endpoint ratio is {fused_ratio:.3f}; fusion barely moves where the pixel gate fired."
        )
    if (
        fused_ratio is not None
        and fused_ratio > 0.4
        and image_ratio is not None
        and image_ratio < 0.25
        and (image_to is None or image_to >= 0.008)
    ):
        focus.append("generation_weakening")
        reasons.append(
            f"Fusion moves (ratio {fused_ratio:.3f}) more than the final image (ratio {image_ratio:.3f})."
        )
    if image_to is not None and image_to >= 0.008 and image_ratio is not None and image_ratio < 0.25:
        if "generation_weakening" not in focus and "fusion_sensitivity" not in focus:
            focus.append("local_gate_or_feature_propagation")
            reasons.append(
                "Neighbor removal changes the image more than the 1px gate; inspect remaining valid cells and fusion."
            )
    if not focus:
        focus.append("mixed")
        reasons.append("Stage ratios do not isolate a single next question; keep the three-stage numbers.")
    reasons.append(
        "RGB L1 cannot decide whether ghosting structure remains; compare baseline vs target_only sheets."
    )
    return {
        "focus": focus,
        "fused_gate_over_target_only": fused_ratio,
        "image_gate_over_target_only": image_ratio,
        "stage_answers": _stage_answers(observed),
        "reasons": reasons,
        "do_not": [
            "treat_closer_to_l1_render_as_quality",
            "treat_red_catch_as_all_errors_caught",
            "treat_warp_fallback_as_repair",
            "sweep_more_thresholds_without_a_focus",
            "change_min_valid_fraction_from_this_run",
        ],
    }


def _run_arm(
    *,
    alignment: str,
    max_roundtrip_error_px: float | None,
    arm_dir: Path,
    backend_kwargs: dict,
    request_kwargs: dict,
    skip_dloral: bool,
) -> Path:
    arm_dir.mkdir(parents=True, exist_ok=True)
    refined = arm_dir / "refined.png"
    fused = arm_dir / "dloral" / "spatial_fused.npy"
    if skip_dloral:
        return arm_dir
    if fused.is_file() and refined.is_file():
        return arm_dir
    backend = DLoRALBackend(
        **backend_kwargs,
        alignment=alignment,
        max_roundtrip_error_px=max_roundtrip_error_px,
        dump_spatial_features=True,
    )
    payload = dict(request_kwargs)
    metadata = dict(payload.pop("metadata"))
    metadata["backend_save_dir"] = str(arm_dir / "dloral")
    result = backend.refine(RefinementRequest(**payload, metadata=metadata))
    result.image.save(refined)
    return arm_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision_dir",
        type=Path,
        default=Path("skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_geometry"),
    )
    parser.add_argument(
        "--ablation_dir",
        type=Path,
        default=Path("skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_geometry/roundtrip_gate_ablation"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_geometry/roundtrip_influence"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_dloral", action="store_true")
    parser.add_argument("--dloral_python", type=str, default=str(Path.home() / "miniconda3/envs/dloral/bin/python"))
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_root", type=str, default="submodules/DLoRAL")
    args = parser.parse_args()

    supervision = args.supervision_dir.resolve()
    ablation = args.ablation_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target = _load_rgb(supervision / "target.png")
    height, width = target.shape[:2]
    archived_refined = _load_rgb(supervision / "refined.png")
    red = _load_mask(ablation / "red_roundtrip_gt_1px.png")
    orange = _load_mask(ablation / "orange_roundtrip_le_1px_rgb.png")
    crops = {name: fractional_crop(width, height, box) for name, box in GHOSTING_CROPS.items()}
    red_t = torch.from_numpy(red)
    orange_t = torch.from_numpy(orange)

    feat_paths = {
        "baseline": supervision / "dloral" / "flow_valid.npy",
        "1px": ablation / "rt_1px" / "dloral" / "flow_valid.npy",
        "2px": ablation / "rt_2px" / "dloral" / "flow_valid.npy",
        "4px": ablation / "rt_4px" / "dloral" / "flow_valid.npy",
    }
    pixel_paths = {
        "1px": ablation / "rt_1px" / "pixel_valid.png",
        "2px": ablation / "rt_2px" / "pixel_valid.png",
        "4px": ablation / "rt_4px" / "pixel_valid.png",
    }
    readmission = {}
    for arm, feat_path in feat_paths.items():
        feat = torch.from_numpy(_load_mask(feat_path))
        pixel = torch.from_numpy(_load_mask(pixel_paths[arm])) if arm in pixel_paths else None
        readmission[arm] = {
            "feature_coverage": float(feat.float().mean().item()),
            "red": _readmission_block("red", red_t, feat, pixel),
            "orange": _readmission_block("orange", orange_t, feat, pixel),
        }
    _write_json(output_dir / "readmission.json", readmission)

    neighbor = Image.open(supervision / "neighbor.png").convert("RGB")
    target_pil = Image.open(supervision / "target.png").convert("RGB")
    forward = torch.from_numpy(np.load(supervision / "target_to_source_flow.npy")).float()
    reverse = torch.from_numpy(np.load(supervision / "source_to_target_flow.npy")).float()
    forward_valid = torch.isfinite(forward).all(dim=-1)
    reverse_valid = torch.isfinite(reverse).all(dim=-1)
    neighbor_input = MultiViewInput(
        name="JAX_068_018_RGB",
        image=neighbor,
        pixel_flow=forward,
        valid_mask=forward_valid,
        source_to_target_flow=reverse,
        reverse_valid_mask=reverse_valid,
        weight=float(forward_valid.float().mean().item()),
        metadata={
            "coverage": float(forward_valid.float().mean().item()),
            "reverse_source": "depth",
            "source_size": (width, height),
            "target_size": (width, height),
        },
    )
    prompt = _prompt_text(supervision / "prompt.json")
    weight_root = Path(os.environ.get("DLORAL_WEIGHT_ROOT", "weights/dloral"))
    camera = CameraSnapshot(
        image_name="zoom4",
        uid=50000,
        colmap_id=None,
        image_width=width,
        image_height=height,
        fov_x=1.0,
        fov_y=1.0,
        cx=0.0,
        cy=0.0,
        R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        T=(0.0, 0.0, 0.0),
    )
    backend_kwargs = dict(
        repo_root=args.dloral_root,
        sd_path=str(weight_root / "stable-diffusion-2-1-base"),
        ckpt_path=str(weight_root / "model_enhanced.pkl"),
        spynet_path=str(weight_root / "spynet_20210409-c6c1bd09.pth"),
        python=args.dloral_python,
        device=args.dloral_device,
        stages=1,
        process_size=512,
        upscale=1,
        align_method="adain",
        latent_tiled_size=96,
    )
    request_base = dict(
        image=target_pil,
        checkpoint=str(supervision),
        camera=camera,
        zoom_factor=4.0,
        sr_scale=1.0,
        prompt=PromptDescription(target_prompt=prompt, provider="archived_qwen3vl"),
        metadata={"seed": int(args.seed), "neighbor_views": [neighbor_input]},
    )
    arms = {
        "baseline": _run_arm(
            alignment="geometry",
            max_roundtrip_error_px=None,
            arm_dir=output_dir / "baseline",
            backend_kwargs=backend_kwargs,
            request_kwargs=request_base,
            skip_dloral=args.skip_dloral,
        ),
        "rt_1px": _run_arm(
            alignment="geometry",
            max_roundtrip_error_px=1.0,
            arm_dir=output_dir / "rt_1px",
            backend_kwargs=backend_kwargs,
            request_kwargs=request_base,
            skip_dloral=args.skip_dloral,
        ),
        "target_only": _run_arm(
            alignment="target_only",
            max_roundtrip_error_px=None,
            arm_dir=output_dir / "target_only",
            backend_kwargs=backend_kwargs,
            request_kwargs=request_base,
            skip_dloral=args.skip_dloral,
        ),
    }

    report = {
        "supervision_dir": str(supervision),
        "ablation_dir": str(ablation),
        "archive": "roundtrip gate mechanism works; generative quality gain is unverified",
        "fixed": ["4x target", "neighbor JAX_068_018_RGB", "prompt", "seed 0", "model_enhanced.pkl"],
        "did_not": ["train_l2", "change_densify", "make_roundtrip_gate_default", "sweep_more_thresholds"],
        "readmission": readmission,
        "caveats": [
            "Red catch rate 100% only matches the >1px definition; it does not prove all bad matches are found or that useful matches were kept.",
            "Warp-fallback ghosting disappearance is construction: rejected pixels copy the target, so residual is zero there.",
            "Closer to the L1 render is an output change, not a quality win; it cannot separate fewer artifacts from weaker enhancement.",
        ],
        "dloral": None,
        "notes": [
            "CFR overlap tiles are averaged into spatial_*.npy; that reconstruction is not a tensor the network used.",
            "feature_diag.json remains last-tile scalars and is not used for spatial comparisons.",
            "min_valid_fraction=0.5 is unchanged; readmission is a post-gate aggregation effect.",
        ],
    }

    fused_files = {name: path / "dloral" / "spatial_fused.npy" for name, path in arms.items()}
    if all(path.is_file() for path in fused_files.values()) and all((path / "refined.png").is_file() for path in arms.values()):
        images = {name: _load_rgb(path / "refined.png") for name, path in arms.items()}
        fused = {name: np.load(path) for name, path in fused_files.items()}
        aligned = {name: np.load(path / "dloral" / "spatial_aligned.npy") for name, path in arms.items()}
        target_feat = {name: np.load(path / "dloral" / "spatial_target.npy") for name, path in arms.items()}
        feat_valid = {
            "baseline": _load_mask(arms["baseline"] / "dloral" / "flow_valid.npy"),
            "rt_1px": _load_mask(arms["rt_1px"] / "dloral" / "flow_valid.npy"),
            "target_only": _load_mask(arms["target_only"] / "dloral" / "flow_valid.npy"),
        }
        feat_h, feat_w = fused["baseline"].shape[-2:]
        red_cells = (
            F.max_pool2d(red_t.float()[None, None], kernel_size=height // feat_h, stride=height // feat_h)[0, 0].numpy() > 0
        )
        orange_cells = (
            F.max_pool2d(orange_t.float()[None, None], kernel_size=height // feat_h, stride=height // feat_h)[0, 0].numpy() > 0
        )
        invalidated = feat_valid["baseline"] & ~feat_valid["rt_1px"]
        red_still_valid = red_cells & feat_valid["rt_1px"]
        red_now_invalid = red_cells & ~feat_valid["rt_1px"]
        feat_masks = {
            "all": np.ones((feat_h, feat_w), dtype=bool),
            "red_cells": red_cells,
            "orange_cells": orange_cells,
            "invalidated_by_1px": invalidated,
            "red_still_valid": red_still_valid,
            "red_now_invalid": red_now_invalid,
            "baseline_valid": feat_valid["baseline"],
            "gated_valid": feat_valid["rt_1px"],
        }
        fused_b1 = _l1_map(fused["baseline"], fused["rt_1px"])
        fused_bt = _l1_map(fused["baseline"], fused["target_only"])
        fused_1t = _l1_map(fused["rt_1px"], fused["target_only"])
        aligned_b1 = _l1_map(aligned["baseline"], aligned["rt_1px"])
        fused_vs_target = {name: _l1_map(fused[name], target_feat[name]) for name in fused}
        image_masks = {
            "full": np.ones((height, width), dtype=bool),
            "red": red,
            "orange": orange,
        }
        for name, box in crops.items():
            x0, y0, x1, y1 = box
            region = np.zeros((height, width), dtype=bool)
            region[y0:y1, x0:x1] = True
            image_masks[name] = region

        fused_up_b1 = _upsample_nearest(fused_b1, height, width)
        fused_up_bt = _upsample_nearest(fused_bt, height, width)
        image_b1 = np.abs(images["baseline"] - images["rt_1px"]).mean(axis=-1)
        image_bt = np.abs(images["baseline"] - images["target_only"]).mean(axis=-1)
        image_1t = np.abs(images["rt_1px"] - images["target_only"]).mean(axis=-1)

        def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
            if not bool(mask.any()):
                return None
            av = a[mask].reshape(-1)
            bv = b[mask].reshape(-1)
            if av.std() < 1e-12 or bv.std() < 1e-12:
                return None
            return float(np.corrcoef(av, bv)[0, 1])

        observed = {
            "rerun_baseline_l1_vs_archived": _l1(images["baseline"], archived_refined),
            "readmission": readmission,
            "fused": {
                "baseline_vs_1px": _region_stats(fused_b1, feat_masks),
                "baseline_vs_target_only": _region_stats(fused_bt, feat_masks),
                "gated_vs_target_only": _region_stats(fused_1t, feat_masks),
                "units": "mean abs over channels at latent resolution",
                "overlap_tiles_averaged": True,
            },
            "aligned": {
                "baseline_vs_1px": _region_stats(aligned_b1, feat_masks),
                "aligned_vs_target_valid_baseline": _mean(
                    _l1_map(aligned["baseline"], target_feat["baseline"]), feat_valid["baseline"]
                ),
                "aligned_vs_target_valid_1px": _mean(
                    _l1_map(aligned["rt_1px"], target_feat["rt_1px"]), feat_valid["rt_1px"]
                ),
            },
            "fused_vs_target_feature": {
                name: _region_stats(plane, feat_masks) for name, plane in fused_vs_target.items()
            },
            "image": {
                "baseline_vs_1px": {name: _mean(image_b1, mask) for name, mask in image_masks.items()},
                "baseline_vs_target_only": {name: _mean(image_bt, mask) for name, mask in image_masks.items()},
                "gated_vs_target_only": {name: _mean(image_1t, mask) for name, mask in image_masks.items()},
                "vs_target_render": {name: _l1(images[name], target, red) for name in images},
            },
            "weakening": {
                "red_corr_fused_vs_image_gate": _corr(fused_up_b1, image_b1, red),
                "red_corr_fused_vs_image_target_only": _corr(fused_up_bt, image_bt, red),
                "note": "Feature L1 and RGB L1 have different units; use the gate/endpoint ratio at each stage, not a cross-stage numeric equality.",
            },
        }
        observed["suggested_focus"] = _suggest_focus(observed)
        report["dloral"] = {
            "arms": {name: str(path / "refined.png") for name, path in arms.items()},
            "observed": observed,
        }

        colorize_scalar(torch.from_numpy(fused_up_b1), vmax=max(float(fused_up_b1.max()), 1e-4)).save(
            output_dir / "fused_l1_baseline_vs_1px.png"
        )
        colorize_scalar(torch.from_numpy(fused_up_bt), vmax=max(float(fused_up_bt.max()), 1e-4)).save(
            output_dir / "fused_l1_baseline_vs_target_only.png"
        )
        colorize_scalar(torch.from_numpy(image_b1), vmax=0.08).save(output_dir / "image_l1_baseline_vs_1px.png")
        colorize_scalar(torch.from_numpy(image_bt), vmax=0.08).save(output_dir / "image_l1_baseline_vs_target_only.png")
        colorize_scalar(torch.from_numpy(image_1t), vmax=0.08).save(output_dir / "image_l1_1px_vs_target_only.png")

        panels = [
            ("target render", Image.fromarray((target * 255).astype(np.uint8))),
            ("baseline", Image.open(arms["baseline"] / "refined.png").convert("RGB")),
            ("rt 1px", Image.open(arms["rt_1px"] / "refined.png").convert("RGB")),
            ("target_only", Image.open(arms["target_only"] / "refined.png").convert("RGB")),
            ("fused Δ 1px", Image.open(output_dir / "fused_l1_baseline_vs_1px.png").convert("RGB")),
            ("fused Δ target_only", Image.open(output_dir / "fused_l1_baseline_vs_target_only.png").convert("RGB")),
            ("image Δ 1px", Image.open(output_dir / "image_l1_baseline_vs_1px.png").convert("RGB")),
            ("image Δ target_only", Image.open(output_dir / "image_l1_baseline_vs_target_only.png").convert("RGB")),
        ]
        contact_sheet(panels, columns=4).save(output_dir / "overview.png")
        for name, box in crops.items():
            contact_sheet([(title, crop_image(image, box)) for title, image in panels], columns=4).save(
                output_dir / f"{name}_sheet.png"
            )
        print(json.dumps({"suggested_focus": observed["suggested_focus"], "image": observed["image"], "fused_red": {
            "baseline_vs_1px": observed["fused"]["baseline_vs_1px"]["red_cells"],
            "baseline_vs_target_only": observed["fused"]["baseline_vs_target_only"]["red_cells"],
        }, "readmission_red_1px": readmission["1px"]["red"]}, indent=2, default=float))
    else:
        print(json.dumps({"output_dir": str(output_dir), "readmission_red_1px": readmission["1px"]["red"], "dloral": "skipped_or_incomplete"}, indent=2))

    _write_json(output_dir / "report.json", report)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
