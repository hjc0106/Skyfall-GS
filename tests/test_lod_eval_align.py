#!/usr/bin/env python3
"""CPU checks for native-crop zoom eval: no GT upsample, integer 2x/4x boxes."""

from __future__ import annotations

import unittest

from lod.eval_protocol import (
    ALIGN_RGB_L1_MAX,
    LOD_GAIN_BASELINE,
    MASK_POLICY,
    PROTOCOL,
    downsample_render_to_native,
    eval_roi_center,
    forbid_gt_upsample,
    native_crop_box,
    native_crop_hw,
)
from lod.jax068 import TEST_VIEWS, TRAIN_VIEW, check_dataset
from utils.zoom_camera import NormalizedROI

import torch


class LodEvalProtocolTests(unittest.TestCase):
    def test_protocol_forbids_upsampled_gt_and_test_leak(self) -> None:
        self.assertFalse(PROTOCOL["upsample_gt"])
        self.assertEqual(
            PROTOCOL["lod_eval"],
            "focal_zoom_render_area_downsampled_to_native_crop",
        )
        self.assertIn("prompt", PROTOCOL["test_images_excluded_from"])
        self.assertIn("supervision", PROTOCOL["test_images_excluded_from"])
        self.assertIn("training", PROTOCOL["test_images_excluded_from"])
        self.assertFalse(PROTOCOL["full_scene_after_this_gate"]["stitch_independent_roi_models"])
        self.assertFalse(PROTOCOL["full_scene_after_this_gate"]["reuse_single_roi_50k_budget"])
        self.assertEqual(MASK_POLICY["ssim"], "mean_ssim_map_on_mask_eroded_by_window_radius")
        self.assertEqual(MASK_POLICY["lpips"], "mean_spatial_lpips_on_downsampled_mask")
        self.assertEqual(MASK_POLICY["l1"], "sum_M_abs_over_3_sum_M")
        self.assertEqual(LOD_GAIN_BASELINE["rasterizer"], "rade_gs")
        self.assertEqual(
            LOD_GAIN_BASELINE["skyfall_diff_gauss"],
            "list_separately_not_in_lod_gain",
        )
        self.assertTrue(PROTOCOL["full_scene_after_this_gate"]["geometry_mainline_only"])

    def test_2048_zoom_crop_is_integer(self) -> None:
        self.assertEqual(native_crop_hw(2048, 2048, 2.0), (1024, 1024))
        self.assertEqual(native_crop_hw(2048, 2048, 4.0), (512, 512))
        with self.assertRaises(ValueError):
            native_crop_hw(2048, 2048, 3.0)

    def test_centered_box_is_symmetric(self) -> None:
        roi = eval_roi_center()
        self.assertEqual(native_crop_box(2048, 2048, roi, 2.0), (512, 512, 1536, 1536))
        self.assertEqual(native_crop_box(2048, 2048, roi, 4.0), (768, 768, 1280, 1280))

    def test_off_center_panel_building_box_stays_inside(self) -> None:
        roi = NormalizedROI(0.38, 0.50, 0.1, 0.1)
        left, top, right, bottom = native_crop_box(2048, 2048, roi, 2.0)
        self.assertEqual(right - left, 1024)
        self.assertEqual(bottom - top, 1024)
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(right, 2048)

    def test_forbid_upsample_gt(self) -> None:
        forbid_gt_upsample(1024, 1024, 1024, 1024)
        with self.assertRaises(ValueError):
            forbid_gt_upsample(1024, 1024, 2048, 2048)
        render = torch.zeros(3, 512, 512)
        with self.assertRaises(ValueError):
            downsample_render_to_native(render, 1024, 1024)

    def test_area_downsample_halves_shape(self) -> None:
        render = torch.zeros(3, 2048, 2048)
        out = downsample_render_to_native(render, 1024, 1024)
        self.assertEqual(tuple(out.shape), (3, 1024, 1024))

    def test_align_gate_is_closed(self) -> None:
        self.assertLessEqual(ALIGN_RGB_L1_MAX, 0.01)

    def test_jax068_split_constants_are_disjoint(self) -> None:
        self.assertEqual(TRAIN_VIEW, "JAX_068_011_RGB")
        self.assertEqual(TEST_VIEWS, ("JAX_068_003_RGB", "JAX_068_012_RGB"))
        self.assertNotIn(TRAIN_VIEW, TEST_VIEWS)


class LodJax068DatasetTests(unittest.TestCase):
    def test_dataset_lock_if_present(self) -> None:
        from lod.jax068 import DATASET_DIR

        if not (DATASET_DIR / "transforms_test.json").is_file():
            self.skipTest("JAX_068 dataset not mounted")
        payload = check_dataset()
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(tuple(payload["test"]), TEST_VIEWS)
        self.assertEqual(payload["train_view0"], TRAIN_VIEW)
        self.assertFalse(payload["train_test_overlap"])


if __name__ == "__main__":
    unittest.main()
