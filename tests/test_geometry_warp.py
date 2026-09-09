#!/usr/bin/env python3
"""CPU tests for spatial ROI projection and RGB reprojection."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from PIL import Image

from refinement.geometry_warp import (
    build_aligned_multiview_inputs,
    build_reprojection_grid,
    estimate_spatial_target,
    project_spatial_roi,
    rank_cameras_by_spatial_overlap,
    warp_image,
)
from refinement.multiview_sr_backend import MultiViewSRBackend
from refinement.sr_backend import UnsharpBackend
from refinement.types import CameraSnapshot, MultiViewInput, PromptDescription, RefinementRequest


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
        R: np.ndarray | None = None,
        T: np.ndarray | None = None,
    ):
        self.uid = uid
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.FoVx = fov
        self.FoVy = fov
        self.cx = cx
        self.cy = cy
        self.R = np.eye(3, dtype=np.float64) if R is None else R
        self.T = np.zeros(3, dtype=np.float64) if T is None else np.asarray(T, dtype=np.float64)


def _request(image: Image.Image) -> RefinementRequest:
    return RefinementRequest(
        image=image,
        checkpoint="/tmp/missing-checkpoint.pth",
        camera=CameraSnapshot(
            image_name="zoom",
            uid=1,
            colmap_id=0,
            image_width=image.width,
            image_height=image.height,
            fov_x=1.0,
            fov_y=1.0,
            cx=0.0,
            cy=0.0,
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            T=(0.0, 0.0, 0.0),
        ),
        zoom_factor=2.0,
        sr_scale=1.0,
        prompt=PromptDescription(target_prompt="test"),
    )


class GeometryWarpTests(unittest.TestCase):
    def test_identity_reprojection_is_almost_complete(self) -> None:
        camera = FakeCamera()
        depth = torch.ones((camera.image_height, camera.image_width))
        rgb = torch.zeros((3, camera.image_height, camera.image_width))
        rgb[0] = torch.linspace(0, 1, camera.image_width).view(1, -1)

        warp = build_reprojection_grid(camera, camera, depth, depth)
        self.assertGreater(warp.coverage, 0.95)
        warped = warp_image(rgb, warp)
        valid = warp.valid_mask
        error = (warped - rgb).abs()[:, valid].mean().item()
        self.assertLess(error, 0.02)

    def test_spatial_roi_is_not_a_copied_normalized_box(self) -> None:
        target = FakeCamera(uid=0, image_name="target")
        depth = torch.zeros((target.image_height, target.image_width))
        depth[:, 22:] = 1.0
        alpha = (depth > 0).float()
        spatial = estimate_spatial_target(target, depth, alpha, min_points=8)
        self.assertGreater(spatial.confidence, 0.0)
        self.assertGreater(spatial.centroid_world[0], 0.0)

        neighbor = FakeCamera(uid=1, image_name="neighbor", T=(-0.35, 0.0, 0.0))
        projected = project_spatial_roi(spatial, neighbor, zoom_factor=2.0)
        self.assertIsNotNone(projected)
        copied_center = 0.5 * (22 + 31) / target.image_width
        self.assertNotAlmostEqual(projected.center_x, copied_center, places=2)
        self.assertLess(projected.center_x, copied_center)
        self.assertAlmostEqual(projected.width, 0.5)
        self.assertAlmostEqual(projected.height, 0.5)

    def test_rank_excludes_base_and_prefers_overlap(self) -> None:
        target = FakeCamera(uid=0, image_name="target")
        depth = torch.ones((target.image_height, target.image_width))
        spatial = estimate_spatial_target(target, depth, min_points=8)
        behind = FakeCamera(uid=2, image_name="out", T=(-5.0, 0.0, 0.0))
        side = FakeCamera(uid=3, image_name="side", T=(-0.15, 0.0, 0.0))
        selected, records = rank_cameras_by_spatial_overlap(
            spatial,
            [target, behind, side],
            zoom_factor=2.0,
            k=1,
            target_camera=target,
            exclude_uids=(0,),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0].uid, 3)
        skipped = {item["uid"]: item for item in records}
        self.assertNotIn(0, skipped)

    def test_multiview_backend_falls_back_on_low_coverage(self) -> None:
        image = Image.fromarray(np.full((16, 16, 3), 40, dtype=np.uint8), mode="RGB")
        backend = MultiViewSRBackend(UnsharpBackend(), min_coverage=0.2)
        result = backend.refine(_request(image))
        self.assertEqual(result.metadata["fallback"], "single_view_low_coverage")
        self.assertFalse(result.metadata["feature_propagation"])

        neighbor = MultiViewInput(
            name="n0",
            image=image,
            warped_image=torch.ones((3, 16, 16)),
            valid_mask=torch.ones((16, 16)),
            weight=1.0,
        )
        request = _request(image)
        request.metadata["neighbor_views"] = [neighbor]
        fused = backend.refine(request)
        self.assertIsNone(fused.metadata["fallback"])
        self.assertGreater(fused.metadata["coverage"], 0.2)

    def _large_depth_pair(self, occluder_delta: float = 0.0, invalid_patch: bool = False):
        depth_value = 1_545_000.0
        target = FakeCamera(uid=0, image_name="sat_target", fov=0.0002)
        source = FakeCamera(uid=1, image_name="sat_source", fov=0.0002)
        target_depth = torch.full((target.image_height, target.image_width), depth_value)
        source_depth = target_depth.clone()
        source_depth[8:24, 8:24] = depth_value - occluder_delta
        if invalid_patch:
            source_depth[4:8, 4:8] = 0.0
        return target, source, target_depth, source_depth

    def test_relative_tolerance_hides_satellite_occlusion_without_scene_cap(self) -> None:
        target, source, target_depth, source_depth = self._large_depth_pair(occluder_delta=100.0)
        uncapped = build_reprojection_grid(
            target,
            source,
            target_depth,
            source_depth,
            depth_abs_tolerance=5.0,
            depth_rel_tolerance=0.02,
            depth_scene_scale=None,
        )
        self.assertGreater(uncapped.metadata["median_depth_tolerance"], 1.0e4)
        occluded = uncapped.valid_mask[8:24, 8:24]
        self.assertGreater(float(occluded.float().mean().item()), 0.9)

    def test_scene_scale_cap_rejects_building_occlusion_at_satellite_range(self) -> None:
        target, source, target_depth, source_depth = self._large_depth_pair(occluder_delta=100.0)
        capped = build_reprojection_grid(
            target,
            source,
            target_depth,
            source_depth,
            depth_abs_tolerance=5.0,
            depth_rel_tolerance=0.02,
            depth_scene_scale=50.0,
        )
        self.assertLessEqual(capped.metadata["max_depth_tolerance"], 50.0 + 1e-5)
        occluded = capped.valid_mask[8:24, 8:24]
        self.assertEqual(float(occluded.float().mean().item()), 0.0)
        visible = capped.valid_mask.clone()
        visible[8:24, 8:24] = False
        self.assertGreater(float(capped.valid_mask[visible].float().mean().item()), 0.9)

    def test_known_visible_correspondence_survives_scene_scale_cap(self) -> None:
        target, source, target_depth, source_depth = self._large_depth_pair(occluder_delta=1.0)
        warp = build_reprojection_grid(
            target,
            source,
            target_depth,
            source_depth,
            depth_abs_tolerance=5.0,
            depth_rel_tolerance=1e-5,
            depth_scene_scale=50.0,
        )
        self.assertGreater(warp.coverage, 0.95)

    def test_invalid_depth_and_flow_nan_not_zero(self) -> None:
        target, source, target_depth, source_depth = self._large_depth_pair(
            occluder_delta=0.0, invalid_patch=True
        )
        warp = build_reprojection_grid(
            target,
            source,
            target_depth,
            source_depth,
            depth_abs_tolerance=5.0,
            depth_rel_tolerance=1e-5,
            depth_scene_scale=50.0,
        )
        invalid = ~torch.isfinite(source_depth) | (source_depth <= 1e-4)
        self.assertFalse(bool(warp.valid_mask[4:8, 4:8].any().item()))
        self.assertTrue(torch.isnan(warp.pixel_flow[4:8, 4:8]).all().item())
        valid_flow = warp.pixel_flow[warp.valid_mask]
        self.assertTrue(torch.isfinite(valid_flow).all().item())
        self.assertFalse(torch.isnan(warp.pixel_flow[warp.valid_mask]).any().item())
        del invalid

    def test_aligned_neighbors_store_depth_based_reverse_flow(self) -> None:
        target = FakeCamera(uid=0, image_name="target")
        source = FakeCamera(uid=1, image_name="neighbor", T=(-0.2, 0.0, 0.0))
        depth = torch.ones((target.image_height, target.image_width))
        rgb = torch.zeros((3, target.image_height, target.image_width))
        aligned, _records = build_aligned_multiview_inputs(
            target,
            depth,
            None,
            [(source, rgb, depth, None)],
            k=1,
        )
        self.assertEqual(len(aligned), 1)
        self.assertIsNotNone(aligned[0].source_to_target_flow)
        self.assertIsNotNone(aligned[0].reverse_valid_mask)
        self.assertEqual(aligned[0].metadata.get("reverse_source"), "depth")
        self.assertGreater(float(aligned[0].reverse_valid_mask.float().mean().item()), 0.5)


if __name__ == "__main__":
    unittest.main()
