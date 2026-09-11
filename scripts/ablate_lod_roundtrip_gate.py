#!/usr/bin/env python3
"""2D ablation: full-res roundtrip gate on frozen 4x geometry DLoRAL. Does not train L2.

Reuses the saved dual-depth pair, prompt, seed, and DLoRAL weights. RGB residual is
a report label only and is not a rejection rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lod.warp_diag import GHOSTING_CROPS, colorize_scalar, contact_sheet, crop_image, fractional_crop, upsample_mask
from refinement.dloral_backend import DLoRALBackend
from refinement.dloral_flows import (
    correspondence_from_neighbor,
    correspondence_to_external_flows,
    gate_valid_with_roundtrip,
    roundtrip_error_map,
)
from refinement.types import CameraSnapshot, MultiViewInput, PromptDescription, RefinementRequest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _save_mask(path: Path, mask: torch.Tensor) -> None:
    Image.fromarray((mask.detach().cpu().numpy().astype("uint8") * 255), mode="L").save(path)


def _l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    residual = np.abs(a - b).mean(axis=-1)
    if mask is None:
        return float(residual.mean())
    if not bool(mask.any()):
        return None
    return float(residual[mask].mean())


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervision_dir",
        type=Path,
        default=Path("skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_geometry"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_geometry/roundtrip_gate_ablation"),
    )
    parser.add_argument("--thresholds", type=str, default="1,2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_dloral", action="store_true")
    parser.add_argument("--dloral_python", type=str, default=str(Path.home() / "miniconda3/envs/dloral/bin/python"))
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_root", type=str, default="submodules/DLoRAL")
    args = parser.parse_args()

    supervision = args.supervision_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    red_thresh = 1.0

    target = _load_rgb(supervision / "target.png")
    warped = _load_rgb(supervision / "neighbor_warped.png")
    baseline_refined = _load_rgb(supervision / "refined.png")
    neighbor = Image.open(supervision / "neighbor.png").convert("RGB")
    target_pil = Image.open(supervision / "target.png").convert("RGB")
    height, width = target.shape[:2]
    forward = torch.from_numpy(np.load(supervision / "target_to_source_flow.npy")).float()
    reverse = torch.from_numpy(np.load(supervision / "source_to_target_flow.npy")).float()
    forward_valid = torch.isfinite(forward).all(dim=-1)
    reverse_valid = torch.isfinite(reverse).all(dim=-1)
    error, hit = roundtrip_error_map(forward, forward_valid, reverse, reverse_valid)
    rgb_residual = torch.from_numpy(np.abs(warped - target).mean(axis=-1)).float()
    red = (hit & (error > red_thresh)).cpu().numpy()
    orange = (hit & torch.isfinite(error) & (error <= red_thresh) & (rgb_residual > 0.05)).cpu().numpy()
    geometry_valid = forward_valid.cpu().numpy()

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
    correspondence = correspondence_from_neighbor(neighbor_input)
    prompt = _prompt_text(supervision / "prompt.json")
    crops = {name: fractional_crop(width, height, box) for name, box in GHOSTING_CROPS.items()}

    baseline_payload = correspondence_to_external_flows(correspondence, process_size=512, upscale=1)
    report = {
        "supervision_dir": str(supervision),
        "red_definition": "geometry hit and full-res roundtrip > 1 image px",
        "orange_definition": "geometry hit, roundtrip <= 1 image px, RGB L1 > 0.05 (report only, not a gate)",
        "do_not": [
            "use_rgb_residual_as_gate",
            "tighten_global_depth_tolerance",
            "retrain_l2_before_2d_pass",
            "go_to_8x",
            "change_densify",
            "add_scale_offset_constraint",
        ],
        "baseline": {
            "pixel_coverage": float(forward_valid.float().mean().item()),
            "feature_coverage": baseline_payload["coverage"],
            "image_roundtrip": baseline_payload.get("image_roundtrip"),
            "feature_roundtrip": baseline_payload.get("roundtrip"),
            "approx_image_px_per_feature_px": 8.0,
            "red_frac_of_image": float(red.mean()),
            "orange_frac_of_image": float(orange.mean()),
            "refined": str(supervision / "refined.png"),
        },
        "arms": {},
        "notes": [
            "Alpha near 1 only means accumulated opacity; it does not prove a single surface.",
            "Feature-pixel roundtrip is not comparable to the image-pixel gate. 0.095 feature px ~ 0.76 image px at 8x.",
            "The question is whether 1-4 image-px inconsistencies stay valid after aggregation.",
        ],
    }
    _save_mask(output_dir / "geometry_valid.png", forward_valid)
    _save_mask(output_dir / "red_roundtrip_gt_1px.png", torch.from_numpy(red))
    _save_mask(output_dir / "orange_roundtrip_le_1px_rgb.png", torch.from_numpy(orange))

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

    for thresh in thresholds:
        arm_dir = output_dir / f"rt_{thresh:g}px"
        arm_dir.mkdir(parents=True, exist_ok=True)
        fwd_gated, _, _, _ = gate_valid_with_roundtrip(
            forward, forward_valid, reverse, reverse_valid, max_error_px=thresh,
        )
        payload = correspondence_to_external_flows(
            correspondence, process_size=512, upscale=1, max_roundtrip_error_px=thresh,
        )
        feat_up = upsample_mask(payload["valid_mask"], height, width)
        _save_mask(arm_dir / "pixel_valid.png", fwd_gated)
        _save_mask(arm_dir / "feature_valid_x8.png", feat_up)
        composite = np.where(fwd_gated.numpy()[..., None], warped, target)
        Image.fromarray((composite * 255).clip(0, 255).astype(np.uint8), mode="RGB").save(arm_dir / "warp_fallback.png")
        gated_np = fwd_gated.numpy()
        arm = {
            "max_roundtrip_error_px": thresh,
            "pixel_coverage": float(fwd_gated.float().mean().item()),
            "pixel_coverage_delta": float(fwd_gated.float().mean().item() - forward_valid.float().mean().item()),
            "feature_coverage": payload["coverage"],
            "feature_coverage_delta": payload["coverage"] - baseline_payload["coverage"],
            "red_catch_rate": None if not red.any() else float((~gated_np & red).sum() / red.sum()),
            "orange_catch_rate": None if not orange.any() else float((~gated_np & orange).sum() / orange.sum()),
            "warp_fallback_l1_vs_target": {
                "full": _l1(composite, target),
                "red": _l1(composite, target, red),
                "orange": _l1(composite, target, orange),
                "vehicles": _crop_stats(composite, target, crops["vehicles"]),
                "eaves": _crop_stats(composite, target, crops["eaves"]),
                "trees": _crop_stats(composite, target, crops["trees"]),
            },
            "baseline_warp_l1_vs_target": {
                "full": _l1(warped, target, geometry_valid),
                "red": _l1(warped, target, red),
                "orange": _l1(warped, target, orange),
                "vehicles": _crop_stats(warped, target, crops["vehicles"], geometry_valid),
                "eaves": _crop_stats(warped, target, crops["eaves"], geometry_valid),
                "trees": _crop_stats(warped, target, crops["trees"], geometry_valid),
            },
            "roundtrip_gate": payload.get("roundtrip_gate"),
            "dloral": None,
        }
        if not args.skip_dloral:
            backend = DLoRALBackend(
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
                alignment="geometry",
                latent_tiled_size=96,
                max_roundtrip_error_px=thresh,
            )
            result = backend.refine(
                RefinementRequest(
                    image=target_pil,
                    checkpoint=str(supervision),
                    camera=camera,
                    zoom_factor=4.0,
                    sr_scale=1.0,
                    prompt=PromptDescription(target_prompt=prompt, provider="archived_qwen3vl"),
                    metadata={
                        "seed": int(args.seed),
                        "neighbor_views": [neighbor_input],
                        "backend_save_dir": str(arm_dir / "dloral"),
                    },
                )
            )
            result.image.save(arm_dir / "refined.png")
            gated_refined = _load_rgb(arm_dir / "refined.png")
            arm["dloral"] = {
                "refined": str(arm_dir / "refined.png"),
                "l1_vs_baseline_refined": {
                    "full": _l1(gated_refined, baseline_refined),
                    "red": _l1(gated_refined, baseline_refined, red),
                    "orange": _l1(gated_refined, baseline_refined, orange),
                    "vehicles": _crop_stats(gated_refined, baseline_refined, crops["vehicles"]),
                    "eaves": _crop_stats(gated_refined, baseline_refined, crops["eaves"]),
                    "trees": _crop_stats(gated_refined, baseline_refined, crops["trees"]),
                },
                "l1_vs_target_render": {
                    "full": _l1(gated_refined, target),
                    "red": _l1(gated_refined, target, red),
                    "orange": _l1(gated_refined, target, orange),
                    "vehicles": _crop_stats(gated_refined, target, crops["vehicles"]),
                    "eaves": _crop_stats(gated_refined, target, crops["eaves"]),
                    "trees": _crop_stats(gated_refined, target, crops["trees"]),
                },
                "baseline_l1_vs_target_render": {
                    "full": _l1(baseline_refined, target),
                    "red": _l1(baseline_refined, target, red),
                    "orange": _l1(baseline_refined, target, orange),
                    "vehicles": _crop_stats(baseline_refined, target, crops["vehicles"]),
                    "eaves": _crop_stats(baseline_refined, target, crops["eaves"]),
                    "trees": _crop_stats(baseline_refined, target, crops["trees"]),
                },
                "roundtrip_gate": result.metadata.get("roundtrip_gate"),
            }
            residual = torch.from_numpy(np.abs(gated_refined - baseline_refined).mean(axis=-1)).float()
            colorize_scalar(residual, vmax=0.08).save(arm_dir / "delta_vs_baseline.png")
        panels = [
            ("target", Image.fromarray((target * 255).astype(np.uint8))),
            ("warped", Image.fromarray((warped * 255).astype(np.uint8))),
            ("warp fallback", Image.fromarray((composite * 255).astype(np.uint8))),
            ("pixel valid", Image.fromarray((gated_np.astype("uint8") * 255)).convert("RGB")),
            ("feature valid x8", Image.fromarray((feat_up.numpy().astype("uint8") * 255)).convert("RGB")),
        ]
        if (arm_dir / "refined.png").is_file():
            panels.append(("gated refined", Image.open(arm_dir / "refined.png").convert("RGB")))
            panels.append(("baseline refined", Image.open(supervision / "refined.png").convert("RGB")))
        for name, box in crops.items():
            contact_sheet([(title, crop_image(image, box)) for title, image in panels], columns=4).save(
                arm_dir / f"{name}_sheet.png"
            )
        report["arms"][f"{thresh:g}px"] = arm
        _write_json(output_dir / "report.json", report)
        print(json.dumps({
            "thresh": thresh,
            "pixel_coverage": arm["pixel_coverage"],
            "feature_coverage": arm["feature_coverage"],
            "red_catch_rate": arm["red_catch_rate"],
            "orange_catch_rate": arm["orange_catch_rate"],
            "dloral_red_vs_target": None if arm["dloral"] is None else arm["dloral"]["l1_vs_target_render"]["red"],
            "baseline_red_vs_target": None if arm["dloral"] is None else arm["dloral"]["baseline_l1_vs_target_render"]["red"],
            "dloral_eaves_vs_baseline": None if arm["dloral"] is None else arm["dloral"]["l1_vs_baseline_refined"]["eaves"],
            "dloral_vehicles_vs_baseline": None if arm["dloral"] is None else arm["dloral"]["l1_vs_baseline_refined"]["vehicles"],
        }, indent=2))

    _write_json(output_dir / "report.json", report)
    print(json.dumps({"output_dir": str(output_dir), "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
