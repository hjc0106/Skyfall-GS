#!/usr/bin/env python3
"""CPU checks for JAX_068 full-scene coverage, budget, and eval-window lock."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from lod.coverage import (
    candidate_grid,
    coverage_payload,
    farthest_probe_ids,
    footprint_occupancy,
    greedy_select_3d,
    image_union_coverage,
    joint_budget,
    locked_eval_windows,
    nested_4x_tiles,
    overlapping_centers,
    primary_tiles_by_view,
    visit_stats,
)
from lod.eval_protocol import (
    LOD_GAIN_BASELINE,
    MASK_POLICY,
    OVERLAP_WINDOWS,
    PROTOCOL,
    apply_mask,
    reconstruction_metrics,
)
from lod.jax068 import TEST_EVAL_WINDOW_CENTERS, TEST_VIEWS
from lod.train_state import advance_joint_mix_rng, draw_joint_mix, mix_rng
from utils.zoom_camera import NormalizedROI


class LodCoverageTests(unittest.TestCase):
    def test_2x_overlap_grid_is_4x_legal(self) -> None:
        centers = overlapping_centers(2.0, 0.5)
        self.assertEqual(centers, (0.25, 0.5, 0.75))
        for cx in centers:
            for cy in centers:
                roi = NormalizedROI(cx, cy, 0.1, 0.1)
                self.assertTrue(roi.zoom_is_valid(2.0))
                self.assertTrue(roi.zoom_is_valid(4.0))

    def test_mask_rejects_empty_corners(self) -> None:
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        rows = candidate_grid(
            view_name="JAX_068_011_RGB", width=64, height=64, mask_hw=mask, min_mask_frac=0.7,
        )
        self.assertTrue(rows)
        ids = {item["tile_id"] for item in rows}
        self.assertIn("JAX_068_011_RGB__z2__0.500_0.500", ids)
        self.assertNotIn("JAX_068_011_RGB__z2__0.250_0.250", ids)

    def test_3d_spread_then_missed_voxel_fill(self) -> None:
        candidates = [
            {"tile_id": "a", "view_name": "v0", "center_x": 0.25, "center_y": 0.5,
             "mask_frac": 0.99, "xyz": (0.0, 0.0, 0.0)},
            {"tile_id": "b", "view_name": "v1", "center_x": 0.5, "center_y": 0.5,
             "mask_frac": 0.98, "xyz": (1.0, 0.0, 0.0)},
            {"tile_id": "c", "view_name": "v2", "center_x": 0.75, "center_y": 0.5,
             "mask_frac": 0.90, "xyz": (100.0, 0.0, 0.0)},
        ]
        selected, occ = greedy_select_3d(candidates, min_world_dist=10.0, voxel_bins=4)
        ids = {item["tile_id"] for item in selected}
        self.assertIn("a", ids)
        self.assertIn("c", ids)
        self.assertNotIn("b", ids)
        self.assertGreaterEqual(occ["coverage_frac"], 1.0 - 1e-9)

        far_cluster = dict(candidates[2])
        far_cluster["tile_id"] = "d"
        far_cluster["xyz"] = (100.0, 80.0, 0.0)
        far_cluster["mask_frac"] = 0.80
        selected2, occ2 = greedy_select_3d(candidates + [far_cluster], min_world_dist=50.0, voxel_bins=2)
        ids2 = {item["tile_id"] for item in selected2}
        self.assertIn("d", ids2)
        self.assertGreaterEqual(occ2["n_selected"], 2)

    def test_probe_picks_spread_tiles(self) -> None:
        tiles = [
            {"tile_id": "near_a", "mask_frac": 0.9, "xyz": (0.0, 0.0, 0.0)},
            {"tile_id": "near_b", "mask_frac": 0.8, "xyz": (1.0, 0.0, 0.0)},
            {"tile_id": "far", "mask_frac": 0.7, "xyz": (50.0, 0.0, 0.0)},
        ]
        ids = farthest_probe_ids(tiles, k=2)
        self.assertEqual(len(ids), 2)
        self.assertIn("near_a", ids)
        self.assertIn("far", ids)

    def test_eval_windows_locked_and_not_a_success_rate(self) -> None:
        rows = locked_eval_windows()
        self.assertEqual(len(TEST_EVAL_WINDOW_CENTERS) * len(TEST_VIEWS) * 2, len(rows))
        self.assertTrue(all(row["split"] == "test" for row in rows))
        self.assertTrue(OVERLAP_WINDOWS["not_independent_samples"])
        self.assertIn("success_rate_over_overlapping_windows", OVERLAP_WINDOWS["do_not_compute"])
        for view in TEST_VIEWS:
            self.assertTrue(any(row["image_name"] == view and row["roi_id"] == "center" for row in rows))

    def test_budget_does_not_copy_single_roi(self) -> None:
        full = joint_budget(24, probe=False)
        probe = joint_budget(24, probe=True)
        self.assertNotEqual(full["steps"], 500)
        self.assertNotEqual(full["max_points"], 50000)
        self.assertEqual(full["steps"], 360)
        self.assertEqual(full["max_points"], 96000)
        self.assertEqual(full["visits_per_tile_expected"], 12)
        self.assertTrue(full["visits_are_expected_not_guaranteed"])
        self.assertEqual(full["sampling"], "random_with_replacement")
        self.assertNotIn("coverage_rounds", full)
        self.assertTrue(full["not_copied_from_single_roi"])
        self.assertEqual(probe["steps"], 40)
        self.assertEqual(probe["max_points"], 8000)
        self.assertEqual(probe["n_tiles_used"], 3)

    def test_one_stage_camera_per_view(self) -> None:
        tiles = [
            {"tile_id": "a", "view_name": "v0", "center_x": 0.5, "center_y": 0.5, "mask_frac": 0.9},
            {"tile_id": "b", "view_name": "v0", "center_x": 0.25, "center_y": 0.5, "mask_frac": 0.8},
            {"tile_id": "c", "view_name": "v1", "center_x": 0.5, "center_y": 0.5, "mask_frac": 0.7},
        ]
        primary = primary_tiles_by_view(tiles)
        self.assertEqual([item["view_name"] for item in primary], ["v0", "v1"])
        self.assertEqual(primary[0]["tile_id"], "a")

    def test_payload_keeps_test_out_of_training(self) -> None:
        tiles = [
            {
                "tile_id": "t0", "view_name": "JAX_068_011_RGB", "center_x": 0.5, "center_y": 0.5,
                "width": 0.1, "height": 0.1, "mask_frac": 0.9, "xyz": (0.0, 0.0, 0.0), "zoom": 2.0,
            }
        ]
        payload = coverage_payload(
            tiles_2x=tiles, occupancy={"coverage_frac": 1.0},
            train_names=["JAX_068_011_RGB"], test_names=list(TEST_VIEWS),
            cameras_extent=128.0, min_world_dist=15.0,
        )
        self.assertNotIn(TEST_VIEWS[0], payload["views_with_tiles"])
        self.assertEqual(payload["split"]["test"], list(TEST_VIEWS))
        self.assertIn("prompt", payload["split"]["test_images_excluded_from"])
        self.assertEqual(payload["lod_gain_baseline"]["rasterizer"], "rade_gs")
        self.assertEqual(payload["eval_windows"][0]["split"], "test")

    def test_nested_4x_box_is_smaller_than_2x(self) -> None:
        tiles = [{
            "tile_id": "v0__z2__0.500_0.500",
            "view_name": "v0", "center_x": 0.5, "center_y": 0.5,
            "width": 0.1, "height": 0.1, "mask_frac": 0.9,
            "box": [512, 512, 1536, 1536], "zoom": 2.0,
        }]
        nested = nested_4x_tiles(tiles)
        self.assertEqual(nested[0]["box"], [768, 768, 1280, 1280])
        self.assertEqual(nested[0]["zoom"], 4.0)

    def test_4x_footprint_can_miss_voxels_that_2x_centers_cover(self) -> None:
        # A 2x-style center at the origin does not cover a far 4x-only footprint.
        candidate = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [(80.0, 0.0, 0.0), (81.0, 0.0, 0.0)]]
        selected = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
        occ = footprint_occupancy(candidate, selected, voxel_bins=4)
        self.assertLess(occ["coverage_frac"], 1.0)
        self.assertTrue(occ["missed_voxels"])

    def test_image_union_reports_uncovered_valid_pixels(self) -> None:
        mask = np.ones((8, 8), dtype=np.float32)
        covered = image_union_coverage(mask, [(0, 0, 4, 8)])
        self.assertAlmostEqual(covered["valid_coverage"], 0.5)
        self.assertEqual(covered["n_uncovered"], 32)

    def test_visit_stats_record_min_median_max_unvisited(self) -> None:
        stats = visit_stats(
            {"a": 15, "b": 12, "c": 0, "__train_1x__": 40},
            expected=12,
        )
        self.assertEqual(stats["min"], 0)
        self.assertEqual(stats["median"], 12)
        self.assertEqual(stats["max"], 15)
        self.assertEqual(stats["n_unvisited"], 1)
        self.assertTrue(stats["expected_is_not_guaranteed"])


class LodMaskAndMixTests(unittest.TestCase):
    def test_ssim_lpips_share_zero_mask_path(self) -> None:
        self.assertEqual(MASK_POLICY["l1"], "sum_M_abs_over_3_sum_M")
        self.assertEqual(MASK_POLICY["ssim"], "mean_ssim_map_on_mask_eroded_by_window_radius")
        self.assertEqual(PROTOCOL["mask_policy"]["name"], "valid_region_mae_psnr_eroded_ssim_lpips")
        self.assertEqual(LOD_GAIN_BASELINE["skyfall_diff_gauss"], "list_separately_not_in_lod_gain")
        rgb = torch.ones(3, 8, 8)
        gt = torch.zeros(3, 8, 8)
        mask = torch.zeros(8, 8)
        mask[:4, :4] = 1.0
        masked = apply_mask(rgb, mask)
        self.assertEqual(float(masked[:, 4, 4].sum()), 0.0)
        self.assertEqual(float(masked[:, 1, 1].sum()), 3.0)
        row = reconstruction_metrics(rgb, gt, mask)
        self.assertAlmostEqual(row["l1"], 1.0, places=5)
        self.assertAlmostEqual(row["zero_filled"]["l1"], 0.25, places=5)
        self.assertAlmostEqual(row["valid_coverage"], 0.25, places=5)
        self.assertIn("ssim", row)
        self.assertIn("psnr", row)
        self.assertLess(row["zero_filled"]["l1"], row["l1"])

    def test_joint_mix_rng_replay(self) -> None:
        seed, mix_ratio, n_train, n_tiles, interrupt = 0, 0.2, 17, 9, 12
        live = mix_rng(seed)
        seq = [draw_joint_mix(live, mix_ratio=mix_ratio, n_train=n_train, n_tiles=n_tiles) for _ in range(20)]
        resumed = mix_rng(seed)
        advance_joint_mix_rng(resumed, interrupt, mix_ratio=mix_ratio, n_train=n_train, n_tiles=n_tiles)
        rest = [draw_joint_mix(resumed, mix_ratio=mix_ratio, n_train=n_train, n_tiles=n_tiles) for _ in range(interrupt, 20)]
        self.assertEqual(rest, seq[interrupt:])
        self.assertTrue(any(kind == "tile" for kind, _ in seq))
        self.assertTrue(any(kind == "train" for kind, _ in seq))


if __name__ == "__main__":
    unittest.main()
