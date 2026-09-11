#!/usr/bin/env python3
"""CPU checks for Skyfall → GaussianZoom L0 camera conversion."""

from __future__ import annotations

import math
import unittest

import torch

from lod.camera import (
    lod_camera_diagnostics,
    ndc_principal_to_pixel,
    pixel_principal_to_ndc,
    projection_offset_delta_ndc,
)
from utils.zoom_camera import NormalizedROI, zoom_principal_point


class _FakeSkyfallCamera:
    def __init__(self):
        self.image_name = "JAX_068_011_RGB"
        self.image_width = 2048
        self.image_height = 2048
        self.FoVx = 0.04
        self.FoVy = 0.04
        self.cx = 0.35
        self.cy = -0.12
        self.focal_x = self.image_width / (2.0 * math.tan(self.FoVx / 2.0))
        self.focal_y = self.image_height / (2.0 * math.tan(self.FoVy / 2.0))
        self.znear = 0.01
        self.zfar = 100.0
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, 3] = torch.tensor([1.5, -2.0, 12.0])
        self.world_view_transform = w2c.transpose(0, 1).contiguous()
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class LodCameraConversionTests(unittest.TestCase):
    def test_ndc_pixel_roundtrip(self) -> None:
        width = 2048
        for ndc in (-0.8, 0.0, 0.35):
            pixel = ndc_principal_to_pixel(ndc, width)
            self.assertAlmostEqual(pixel_principal_to_ndc(pixel, width), ndc, places=7)

    def test_filter_projection_matches_skyfall_compute_3d_filter(self) -> None:
        width, ndc = 2048, 0.35
        cx_ori = ndc / 2.0 * width + width / 2.0
        self.assertAlmostEqual(ndc_principal_to_pixel(ndc, width), cx_ori)

    def test_zoom_principal_point_matches_gaussianzoom_pixel_zoom(self) -> None:
        width, height = 2048, 2048
        roi = NormalizedROI(0.592, 0.53, 0.1, 0.1)
        factor = 2.0
        cx_ndc, cy_ndc = 0.35, -0.12
        cx_zoom_ndc, cy_zoom_ndc = zoom_principal_point(cx_ndc, cy_ndc, roi, factor)
        cx0 = ndc_principal_to_pixel(cx_ndc, width)
        cy0 = ndc_principal_to_pixel(cy_ndc, height)
        cx_gz = factor * (cx0 - roi.center_x * width) + width / 2.0
        cy_gz = factor * (cy0 - roi.center_y * height) + height / 2.0
        self.assertAlmostEqual(ndc_principal_to_pixel(cx_zoom_ndc, width), cx_gz, places=6)
        self.assertAlmostEqual(ndc_principal_to_pixel(cy_zoom_ndc, height), cy_gz, places=6)

    def test_half_pixel_projection_gap_is_one_over_width(self) -> None:
        self.assertAlmostEqual(projection_offset_delta_ndc(2048), 1.0 / 2048)

    def test_pose_and_center_survive_glm_transpose(self) -> None:
        from lod.camera import CameraWithStoredCenter, skyfall_camera_to_lod

        cam = _FakeSkyfallCamera()
        lod = skyfall_camera_to_lod(cam, use_skyfall_center=True)
        self.assertIsInstance(lod, CameraWithStoredCenter)
        raw = skyfall_camera_to_lod(cam, use_skyfall_center=False)
        self.assertFalse(isinstance(raw, CameraWithStoredCenter))
        self.assertEqual(lod.name, "JAX_068_011_RGB")
        self.assertEqual(lod.width, 2048)
        self.assertAlmostEqual(lod.fx, cam.focal_x)
        diag = lod_camera_diagnostics(cam, lod)
        self.assertLess(diag["viewmatrix_max_abs"], 1e-6)
        self.assertLess(diag["center_inv_vs_view"], 1e-4)
        self.assertLess(diag["center_used_vs_stored"], 1e-6)
        self.assertGreater(diag["cx_pixel"], 1024.0)
        self.assertAlmostEqual(diag["gz_proj_cx_minus_skyfall_ndc"], 1.0 / 2048)

    def test_off_center_roi_rejects_wide_zoom_window(self) -> None:
        roi = NormalizedROI(0.28, 0.28, 0.1, 0.1)
        self.assertAlmostEqual(roi.min_zoom_factor(), 0.5 / 0.28, places=6)
        self.assertFalse(roi.zoom_is_valid(1.25))
        self.assertTrue(roi.zoom_is_valid(2.0))
        self.assertTrue(roi.zoom_is_valid(4.0))

    def test_named_camera_beats_list_order(self) -> None:
        from types import SimpleNamespace

        from lod.camera import find_named_camera, resolve_scene_camera

        cameras = [
            SimpleNamespace(image_name="JAX_214_011_RGB"),
            SimpleNamespace(image_name="JAX_214_018_RGB"),
        ]
        index, camera = find_named_camera(cameras, "JAX_214_018_RGB.png")
        self.assertEqual(index, 1)
        self.assertEqual(camera.image_name, "JAX_214_018_RGB")
        with self.assertRaises(RuntimeError):
            resolve_scene_camera(cameras, view_index=0, image_name="JAX_214_018_RGB")
        named, named_index = resolve_scene_camera(cameras, view_index=1, image_name="JAX_214_018_RGB")
        self.assertEqual(named_index, 1)
        self.assertEqual(named.image_name, "JAX_214_018_RGB")

    def test_zoom_stage_keeps_name_and_doubles_fx(self) -> None:
        from lod.camera import zoom_stage_camera

        cam = _FakeSkyfallCamera()
        roi = NormalizedROI(0.5, 0.5, 0.1, 0.1)
        zoomed = zoom_stage_camera(cam, roi, 2.0)
        self.assertEqual(zoomed.name, cam.image_name)
        self.assertEqual(zoomed.width, cam.image_width)
        self.assertAlmostEqual(zoomed.fx / cam.focal_x, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
