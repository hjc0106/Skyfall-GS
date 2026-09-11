#!/usr/bin/env python3
"""CPU tests for geometry pixel_flow → DLoRAL external_flows."""

from __future__ import annotations

import math
import unittest

import torch

from refinement.dloral_flows import (
    ALIGN_CORNERS,
    FLOWS_FORWARD_MEANING,
    FRAME_ORDER,
    VAE_DOWNSAMPLE,
    align_neighbor_latent,
    apply_aligned_feature_fallback,
    correspondence_to_external_flows,
    crop_feature_flow,
    downsample_image_flow,
    fuse_with_valid_mask,
    gate_valid_with_roundtrip,
    hw2_to_nchw,
    invert_feature_flow,
    iter_dloral_latent_tiles,
    latent_is_tiled,
    mask_cross_frame_attention_keys,
    pixel_flow_to_external_flows,
    prepared_image_size,
    resize_pixel_flow,
    roundtrip_diagnostics,
    sanitize_flow,
    zero_invalid_neighbor_features,
)
from refinement.geometry_warp import build_reprojection_grid, correspondence_from_warps
from refinement.types import GeometryCorrespondence


class FakeCamera:
    def __init__(
        self,
        *,
        uid: int = 0,
        image_name: str = "cam",
        width: int = 32,
        height: int = 32,
        fov: float = math.pi / 2.0,
        cx: float = 0.0,
        cy: float = 0.0,
        T=None,
    ):
        self.uid = uid
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.FoVx = fov
        self.FoVy = fov
        self.cx = cx
        self.cy = cy
        self.R = torch.eye(3).numpy()
        self.T = torch.zeros(3).numpy() if T is None else T


class DLoRALFlowAdapterTests(unittest.TestCase):
    def test_prepared_size_keeps_2048_at_upscale_1(self) -> None:
        width, height = prepared_image_size(2048, 2048, process_size=512, upscale=1)
        self.assertEqual((width, height), (2048, 2048))

    def test_cfr_contract_is_neighbor_then_target_forward_flow(self) -> None:
        self.assertEqual(FRAME_ORDER, ("neighbor", "target"))
        self.assertEqual(FLOWS_FORWARD_MEANING, "p_neighbor - p_target at target feature pixels")
        self.assertTrue(ALIGN_CORNERS)

    def test_eight_pixel_shift_becomes_one_feature_pixel(self) -> None:
        height = width = 16
        flow = torch.zeros(height, width, 2)
        flow[..., 0] = 8.0
        payload = pixel_flow_to_external_flows(flow, image_size=(width, height), process_size=8, upscale=1)
        forward = payload["flows_forward"][0, 0]
        self.assertEqual(tuple(forward.shape[-2:]), (2, 2))
        self.assertTrue(torch.isfinite(forward).all())
        self.assertTrue(torch.allclose(forward[0], torch.ones_like(forward[0]), atol=1e-5))
        self.assertTrue(torch.allclose(forward[1], torch.zeros_like(forward[1]), atol=1e-5))
        self.assertAlmostEqual(payload["coverage"], 1.0)

    def test_nan_stays_nan_on_disk_and_zero_in_network(self) -> None:
        height = width = 16
        flow = torch.full((height, width, 2), 8.0)
        flow[:8, :8] = float("nan")
        payload = pixel_flow_to_external_flows(flow, image_size=(width, height), process_size=8, upscale=1)
        stored = payload["flows_forward"]
        self.assertTrue(torch.isnan(stored[..., 0, 0]).all())
        self.assertTrue(torch.isfinite(stored[..., 1, 1]).all())
        safe = sanitize_flow(stored)
        self.assertFalse(torch.isnan(safe).any())
        self.assertEqual(float(safe[..., 0, 0].abs().sum().item()), 0.0)
        self.assertGreater(float(payload["valid_mask"][1, 1].float().item()), 0.0)
        self.assertEqual(float(payload["valid_mask"][0, 0].float().item()), 0.0)

    def test_resize_does_not_bilinear_blend_nan_into_valid(self) -> None:
        flow = torch.zeros(8, 8, 2)
        flow[:, :4] = float("nan")
        flow[:, 4:, 0] = 10.0
        resized = resize_pixel_flow(flow, source_size=(8, 8), target_size=(16, 16))
        valid = torch.isfinite(resized).all(dim=-1)
        self.assertTrue(valid[:, 8:].all())
        self.assertFalse(valid[:, :8].any())
        self.assertTrue(torch.allclose(resized[valid][:, 0], torch.full((int(valid.sum().item()),), 20.0)))

    def test_depth_edge_block_is_invalidated(self) -> None:
        flow = torch.zeros(8, 8, 2)
        flow[:, 4:, 0] = 20.0
        feature_flow, valid = downsample_image_flow(flow, downsample=8, min_valid_fraction=0.5, max_flow_std=4.0)
        self.assertEqual(tuple(feature_flow.shape[:2]), (1, 1))
        self.assertFalse(bool(valid[0, 0].item()))
        self.assertTrue(torch.isnan(feature_flow).all())

    def test_zeroing_invalid_flow_is_not_identity_fallback(self) -> None:
        current = torch.ones(1, 2, 4, 2, 2)
        neighbor_aligned = torch.full((1, 2, 4, 2, 2), 3.0)
        fused = neighbor_aligned.clone()
        valid = torch.tensor([[True, False], [False, True]])
        fused_out, aligned_out = apply_aligned_feature_fallback(fused, neighbor_aligned, current, valid)
        self.assertEqual(float(fused_out[0, 1, 0, 0, 0].item()), 3.0)
        self.assertEqual(float(fused_out[0, 1, 0, 0, 1].item()), 1.0)
        self.assertEqual(float(aligned_out[0, 1, 0, 1, 0].item()), 1.0)
        self.assertEqual(float(aligned_out[0, 1, 0, 1, 1].item()), 3.0)

    def test_all_invalid_falls_back_to_target_feature(self) -> None:
        current = torch.randn(1, 4, 2, 2)
        aligned = torch.ones(1, 4, 2, 2)
        fused = aligned * 5
        valid = torch.zeros(2, 2, dtype=torch.bool)
        out = fuse_with_valid_mask(fused, current, valid)
        self.assertTrue(torch.equal(out, current))
        gated = zero_invalid_neighbor_features(aligned, valid)
        self.assertEqual(float(gated.abs().sum().item()), 0.0)

    def test_resize_scales_displacement_units(self) -> None:
        flow = torch.zeros(8, 8, 2)
        flow[..., 0] = 2.0
        resized = resize_pixel_flow(flow, source_size=(8, 8), target_size=(16, 16))
        self.assertEqual(tuple(resized.shape[:2]), (16, 16))
        self.assertTrue(torch.allclose(resized[..., 0], torch.full_like(resized[..., 0], 4.0)))

    def test_invert_scatter_negates_hit_pixels(self) -> None:
        flow = torch.zeros(4, 4, 2)
        valid = torch.zeros(4, 4, dtype=torch.bool)
        flow[1, 1, 0] = 2.0
        valid[1, 1] = True
        inverse, hit = invert_feature_flow(flow, valid)
        self.assertTrue(bool(hit[1, 3].item()))
        self.assertAlmostEqual(float(inverse[1, 3, 0].item()), -2.0)
        self.assertTrue(torch.isnan(inverse[0, 0]).all())
        full = torch.zeros(8, 8, 2)
        full[..., 0] = 2.0
        payload = pixel_flow_to_external_flows(full, image_size=(8, 8), process_size=8, upscale=1)
        self.assertEqual(payload["reverse_source"], "scatter_invert_diagnostic_only")

    def test_known_translation_flow_and_roundtrip(self) -> None:
        depth_value = 10.0
        target = FakeCamera(uid=0, image_name="target")
        source = FakeCamera(uid=1, image_name="source", T=(0.5, 0.0, 0.0))
        depth = torch.full((32, 32), depth_value)
        forward = build_reprojection_grid(target, source, depth, depth)
        reverse = build_reprojection_grid(source, target, depth, depth)
        self.assertGreater(forward.coverage, 0.85)
        self.assertGreater(reverse.coverage, 0.85)
        valid = forward.valid_mask
        flow_x = forward.pixel_flow[..., 0][valid]
        focal_x = target.image_width / (2.0 * math.tan(target.FoVx / 2.0))
        expected = 0.5 * focal_x / depth_value
        self.assertAlmostEqual(float(flow_x.median().item()), expected, places=4)
        stats = roundtrip_diagnostics(
            forward.pixel_flow, forward.valid_mask, reverse.pixel_flow, reverse.valid_mask
        )
        self.assertGreater(stats["roundtrip_hit_rate"], 0.7)
        self.assertLess(stats["median_roundtrip_error_px"], 0.6)
        correspondence = correspondence_from_warps(forward, reverse)
        payload = correspondence_to_external_flows(correspondence, process_size=8, upscale=1)
        self.assertEqual(payload["reverse_source"], "depth")
        self.assertGreater(payload["coverage"], 0.5)

    def test_depth_reverse_is_not_scatter_inverse(self) -> None:
        target = FakeCamera(uid=0)
        source = FakeCamera(uid=1, T=(0.25, 0.0, 0.0))
        target_depth = torch.full((32, 32), 8.0)
        source_depth = target_depth.clone()
        source_depth[8:24, 8:24] = 2.0
        forward = build_reprojection_grid(
            target, source, target_depth, source_depth, depth_scene_scale=1.0, depth_abs_tolerance=0.05
        )
        reverse = build_reprojection_grid(
            source, target, source_depth, target_depth, depth_scene_scale=1.0, depth_abs_tolerance=0.05
        )
        scatter, scatter_hit = invert_feature_flow(forward.pixel_flow, forward.valid_mask)
        self.assertGreater(float(reverse.valid_mask.float().mean().item()), 0.0)
        disagreement = reverse.valid_mask ^ scatter_hit
        self.assertGreater(int(disagreement.sum().item()), 0)

    def test_tile_crop_keeps_relative_offsets(self) -> None:
        flow = torch.zeros(1, 1, 2, 8, 8)
        flow[..., 0, 4:, 4:] = 3.0
        cropped = crop_feature_flow(flow, origin_hw=(4, 4), size_hw=(4, 4))
        self.assertEqual(tuple(cropped.shape[-2:]), (4, 4))
        self.assertTrue(torch.allclose(cropped[0, 0, 0], torch.full((4, 4), 3.0)))

    def test_full_image_guard_matches_dloral_area_rule(self) -> None:
        self.assertTrue(latent_is_tiled(256, 256, 96))
        self.assertFalse(latent_is_tiled(64, 64, 96))
        self.assertFalse(latent_is_tiled(256, 256, 256))
        self.assertEqual(VAE_DOWNSAMPLE, 8)
        _ = hw2_to_nchw
        self.assertIsInstance(
            GeometryCorrespondence(
                target_to_source_flow=torch.zeros(2, 2, 2),
                valid_mask=torch.ones(2, 2, dtype=torch.bool),
                source_size=(2, 2),
                target_size=(2, 2),
            ),
            GeometryCorrespondence,
        )

    def test_attention_key_mask_zeros_invalid_logits(self) -> None:
        attn = torch.ones(1, 2, 4, 4)
        valid = torch.tensor([[True, False], [True, False]])
        masked = mask_cross_frame_attention_keys(attn, valid, as_logits=False)
        self.assertEqual(float(masked[..., 1].abs().sum().item()), 0.0)
        self.assertEqual(float(masked[..., 3].abs().sum().item()), 0.0)
        self.assertGreater(float(masked[..., 0].abs().sum().item()), 0.0)
        logits = mask_cross_frame_attention_keys(torch.ones(1, 1, 4, 4) * 5, valid, as_logits=True)
        self.assertLess(float(logits[..., 1].max().item()), 0.0)

    def test_depth_reverse_payload_is_not_scatter(self) -> None:
        forward = torch.zeros(16, 16, 2)
        forward[..., 0] = 2.0
        reverse = torch.zeros(16, 16, 2)
        reverse[..., 0] = -2.0
        payload = pixel_flow_to_external_flows(
            forward,
            image_size=(16, 16),
            process_size=8,
            reverse_pixel_flow=reverse,
            source_size=(16, 16),
        )
        self.assertEqual(payload["reverse_source"], "depth")
        self.assertTrue(torch.allclose(payload["flows_backward"][0, 0, 0], torch.full((2, 2), -0.25), atol=1e-5))

    def test_official_tile_grid_for_256_and_96(self) -> None:
        tiles = iter_dloral_latent_tiles(256, 256, 96, 32)
        self.assertEqual(len(tiles), 16)
        self.assertEqual(tiles[0], (0, 0, 96, 96))
        self.assertEqual(tiles[-1][0], 256 - 96)
        self.assertEqual(tiles[-1][1], 256 - 96)
        self.assertFalse(latent_is_tiled(64, 64, 96))
        self.assertTrue(latent_is_tiled(256, 256, 96))

    def test_global_warp_then_tile_crop_keeps_cross_tile_sample(self) -> None:
        neighbor = torch.zeros(2, 1, 8, 8)
        neighbor[0, 0, 0, 7] = 3.0
        flow = torch.zeros(1, 2, 8, 8)
        flow[0, 0, 0, 0] = 7.0
        valid = torch.ones(8, 8, dtype=torch.bool)
        aligned = align_neighbor_latent(neighbor, flow, valid)
        self.assertAlmostEqual(float(aligned[0, 0, 0, 0].item()), 3.0, places=4)
        self.assertEqual(float(aligned[1, 0, 0, 0].item()), 0.0)
        cropped = aligned[:, :, 0:4, 0:4]
        self.assertAlmostEqual(float(cropped[0, 0, 0, 0].item()), 3.0, places=4)

    def test_roundtrip_gate_uses_image_pixels_before_downsample(self) -> None:
        size = 16
        forward = torch.zeros(size, size, 2)
        reverse = torch.zeros(size, size, 2)
        forward[..., 0] = 3.0
        valid = torch.ones(size, size, dtype=torch.bool)
        correspondence = GeometryCorrespondence(
            target_to_source_flow=forward,
            valid_mask=valid,
            source_size=(size, size),
            target_size=(size, size),
            source_to_target_flow=reverse,
            reverse_valid_mask=valid,
        )
        ungated = correspondence_to_external_flows(correspondence, process_size=8, upscale=1)
        self.assertGreater(ungated["coverage"], 0.9)
        self.assertGreater(ungated["image_roundtrip"]["median_roundtrip_error_px"], 2.5)
        self.assertLess(ungated["roundtrip"]["median_roundtrip_error_px"], 0.5)
        gated = correspondence_to_external_flows(
            correspondence, process_size=8, upscale=1, max_roundtrip_error_px=1.0,
        )
        self.assertEqual(gated["coverage"], 0.0)
        self.assertTrue(gated["roundtrip_gate"]["applied_before_feature_downsample"])
        kept = correspondence_to_external_flows(
            correspondence, process_size=8, upscale=1, max_roundtrip_error_px=4.0,
        )
        self.assertGreater(kept["coverage"], 0.9)

    def test_consistent_dual_depth_survives_one_pixel_gate(self) -> None:
        size = 16
        forward = torch.zeros(size, size, 2)
        reverse = torch.zeros(size, size, 2)
        forward[..., 0] = 2.0
        reverse[..., 0] = -2.0
        valid = torch.ones(size, size, dtype=torch.bool)
        fwd_gated, rev_gated, error, _ = gate_valid_with_roundtrip(
            forward, valid, reverse, valid, max_error_px=1.0,
        )
        self.assertGreater(float(fwd_gated.float().mean().item()), 0.85)
        self.assertLess(float(error[fwd_gated].median().item()), 0.1)
        with self.assertRaises(ValueError):
            correspondence_to_external_flows(
                GeometryCorrespondence(
                    target_to_source_flow=forward,
                    valid_mask=valid,
                    source_size=(size, size),
                    target_size=(size, size),
                ),
                process_size=8,
                max_roundtrip_error_px=1.0,
            )

    def test_spatial_tile_average_and_jsonable_dump(self) -> None:
        from refinement.dloral_flows import (
            accumulate_feature_tile,
            finalize_spatial_maps,
            jsonable_feature_dump,
        )

        dump = {"dump_spatial": True, "spatial_hw": (4, 4), "_tile_origin": (0, 0)}
        left = torch.ones(2, 2, 2)
        right = torch.full((2, 2, 2), 3.0)
        accumulate_feature_tile(dump, "fused", left, top=0, left=0, full_hw=(4, 4))
        accumulate_feature_tile(dump, "fused", right, top=0, left=1, full_hw=(4, 4))
        maps = finalize_spatial_maps(dump)
        self.assertEqual(tuple(maps["fused"].shape), (2, 4, 4))
        self.assertTrue(torch.allclose(maps["fused"][:, 0, 0], torch.ones(2)))
        self.assertTrue(torch.allclose(maps["fused"][:, 0, 1], torch.full((2,), 2.0)))
        self.assertTrue(torch.allclose(maps["fused"][:, 0, 2], torch.full((2,), 3.0)))
        dump["coverage"] = 0.5
        jsonable = jsonable_feature_dump(dump)
        self.assertEqual(jsonable["coverage"], 0.5)
        self.assertNotIn("spatial_fused", jsonable)
        self.assertNotIn("_tile_origin", jsonable)


if __name__ == "__main__":
    unittest.main()
