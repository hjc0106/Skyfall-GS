#!/usr/bin/env python3
"""CPU checks for the JAX_214 one-ROI lock. Does not train."""

from __future__ import annotations

import unittest

from pathlib import Path

from lod.jax214 import CLAIM, ROI, STAGE1_DIR, VIEW_IMAGE, check_dataset, panel_roi, selection_payload
from lod.panel import EXCLUDED_HISTORICAL, FIXED_SETTINGS, PANEL_ROIS


class LodJax214Tests(unittest.TestCase):
    def test_roi_is_not_a_jax068_copy_and_is_2x_4x_legal(self) -> None:
        roi = panel_roi()
        self.assertTrue(roi.zoom_is_valid(2.0))
        self.assertTrue(roi.zoom_is_valid(4.0))
        self.assertGreaterEqual(ROI["center_x"], 0.25)
        self.assertLessEqual(ROI["center_x"], 0.75)
        historical = {(item["center_x"], item["center_y"]) for item in EXCLUDED_HISTORICAL}
        panel = {(item["center_x"], item["center_y"]) for item in PANEL_ROIS}
        chosen = (ROI["center_x"], ROI["center_y"])
        self.assertNotIn(chosen, historical)
        self.assertNotIn(chosen, panel)
        self.assertEqual(VIEW_IMAGE, "JAX_214_018_RGB")
        self.assertEqual(STAGE1_DIR, Path("skyfall-gs_exp/stage1/JAX_214"))

    def test_selection_is_one_scene_one_roi(self) -> None:
        payload = selection_payload(locked_at="t")
        self.assertEqual(payload["claim"], CLAIM)
        self.assertEqual(payload["view"]["image_name"], VIEW_IMAGE)
        self.assertEqual(payload["roi"]["id"], "building_parking")
        self.assertEqual(payload["fixed_settings"], FIXED_SETTINGS)
        self.assertFalse(payload["fixed_settings"]["roundtrip_gate"])
        self.assertFalse(payload["fixed_settings"]["allow_unlineaged_reuse"])
        self.assertIn("cross_scene_generalization", payload["do_not_claim"])
        self.assertIn("mode_ranking", payload["do_not_claim"])

    def test_dataset_inventory(self) -> None:
        data = check_dataset()
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["n_train"], 21)
        self.assertEqual(data["n_test"], 3)
        self.assertEqual(data["view0"], "JAX_214_018_RGB.png")
        self.assertEqual(data["train_test_overlap"], [])


if __name__ == "__main__":
    unittest.main()
