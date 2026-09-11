#!/usr/bin/env python3
"""CPU checks for geometry-warp ghosting labels. Does not run a renderer."""

from __future__ import annotations

import unittest

import torch

from lod.warp_diag import (
    class_id_map,
    classify_allowed_errors,
    fractional_crop,
    masked_stats,
    region_report,
    upsample_mask,
)


class WarpDiagTests(unittest.TestCase):
    def test_mixed_alpha_ghost_is_not_a_roundtrip_failure(self) -> None:
        valid = torch.ones(4, 4, dtype=torch.bool)
        rgb = torch.zeros(4, 4)
        rgb[1, 1] = 0.2
        roundtrip = torch.zeros(4, 4)
        slack = torch.zeros(4, 4)
        alpha = torch.ones(4, 4)
        alpha[1, 1] = 0.45
        sampled = torch.ones(4, 4)
        sampled[1, 1] = 0.9
        masks = classify_allowed_errors(valid, rgb, roundtrip, slack, alpha, sampled)
        self.assertTrue(bool(masks["ghosted_mixed"][1, 1]))
        self.assertFalse(bool(masks["ghosted_roundtrip"][1, 1]))
        ids = class_id_map(masks, valid)
        self.assertEqual(int(ids[1, 1].item()), 2)

    def test_roundtrip_failure_outranks_mixed_alpha(self) -> None:
        valid = torch.ones(2, 2, dtype=torch.bool)
        rgb = torch.full((2, 2), 0.2)
        roundtrip = torch.full((2, 2), 3.0)
        slack = torch.zeros(2, 2)
        alpha = torch.full((2, 2), 0.4)
        sampled = torch.full((2, 2), 0.9)
        masks = classify_allowed_errors(valid, rgb, roundtrip, slack, alpha, sampled)
        ids = class_id_map(masks, valid)
        self.assertTrue(torch.all(ids == 4))

    def test_feature_upsample_and_region_leak(self) -> None:
        feat = torch.zeros(2, 2, dtype=torch.bool)
        feat[0, 0] = True
        up = upsample_mask(feat, 4, 4)
        self.assertEqual(tuple(up.shape), (4, 4))
        self.assertTrue(bool(up[:2, :2].all()))
        self.assertFalse(bool(up[2:, 2:].any()))
        valid = torch.zeros(4, 4, dtype=torch.bool)
        valid[0, 3] = True
        residual = torch.zeros(4, 4)
        classes = classify_allowed_errors(
            valid, residual, torch.zeros(4, 4), torch.zeros(4, 4),
            torch.ones(4, 4), torch.ones(4, 4),
        )
        report = region_report(
            valid=valid, feat_valid=up, rgb_residual=residual,
            roundtrip=torch.zeros(4, 4), slack_ratio=torch.zeros(4, 4),
            dest_alpha=torch.ones(4, 4), classes=classes,
        )
        self.assertGreater(report["feature_leaked_frac"], 0.0)
        self.assertEqual(masked_stats(residual, valid)["count"], 1)

    def test_missing_roundtrip_counts_as_allowed_inconsistency(self) -> None:
        valid = torch.ones(1, 1, dtype=torch.bool)
        rgb = torch.full((1, 1), 0.2)
        roundtrip = torch.full((1, 1), float("nan"))
        masks = classify_allowed_errors(
            valid, rgb, roundtrip, torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1),
        )
        self.assertTrue(bool(masks["ghosted_no_roundtrip"].item()))
        self.assertTrue(bool(masks["ghosted_roundtrip"].item()))
        self.assertFalse(bool(masks["ghosted_opaque"].item()))

    def test_gated_red_pixels_can_be_readmitted_by_a_valid_cell(self) -> None:
        from lod.warp_diag import feature_readmission_stats

        region = torch.zeros(16, 16, dtype=torch.bool)
        region[:8, :8] = True
        feat = torch.tensor([[True, False], [False, False]])
        kept = feature_readmission_stats(region, feat)
        self.assertEqual(kept["pixels_in_valid_feature_cell"], 64)
        self.assertEqual(kept["overlapping_feature_cells_valid"], 1)
        feat[0, 0] = False
        dropped = feature_readmission_stats(region, feat)
        self.assertEqual(dropped["pixels_in_valid_feature_cell"], 0)
        pixel_valid = torch.ones(16, 16, dtype=torch.bool)
        pixel_valid[:8, :8] = False
        feat[0, 0] = True
        readmit = feature_readmission_stats(region, feat, pixel_valid=pixel_valid)
        self.assertEqual(readmit["readmitted_pixels"], 64)
        self.assertEqual(readmit["readmitted_pixel_frac"], 1.0)

    def test_crop_box(self) -> None:
        self.assertEqual(fractional_crop(100, 100, (0.1, 0.2, 0.4, 0.5)), (10, 20, 40, 50))


if __name__ == "__main__":
    unittest.main()
