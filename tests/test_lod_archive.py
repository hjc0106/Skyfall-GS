#!/usr/bin/env python3
"""CPU checks for LoD acceptance tables and experiment index entries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lod.archive import build_acceptance, experiment_index_entry


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class LodArchiveTests(unittest.TestCase):
    def test_acceptance_finds_archive_video_and_compacts_recover(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            l1 = root / "l1"
            l2 = root / "l2"
            archive = root / "archive"
            _write(
                l1 / "absorption_curve.json",
                {"curve": [{"step": 0, "n_l1": 0, "target": {"l1_to_refined": 0.04, "hf_l1_to_refined": 0.07}},
                           {"step": 500, "n_l1": 50, "target": {"l1_to_refined": 0.02, "hf_l1_to_refined": 0.05}}]},
            )
            _write(
                l2 / "absorption_curve.json",
                {"curve": [
                    {"step": 0, "n_l2": 0, "target": {"l1_to_refined": 0.03, "hf_l1_to_refined": 0.05},
                     "train_views_mean": {"l1_to_gt_mean": 0.019}, "parent_2x": {"l1_vs_frozen_l1": 0.0}},
                    {"step": 500, "n_l2": 50, "target": {"l1_to_refined": 0.015, "hf_l1_to_refined": 0.04,
                                                         "crops": {"building": {"l1_to_refined": 0.01}}},
                     "train_views_mean": {"l1_to_gt_mean": 0.019},
                     "parent_2x": {"l1_vs_frozen_l1": 0.002}},
                ]},
            )
            _write(l2 / "lod_l2_summary.json", {"acceptance": {"frozen_params_vs_parent_scale": {"l2_weight_at_2x": 0.14}}})
            _write(
                archive / "video" / "zoom.json",
                {"output": str(archive / "video" / "zoom.mp4"), "max_mean_rgb_jump": 0.0046},
            )
            (archive / "renders").mkdir(parents=True)
            (archive / "renders" / "l2_step500.png").symlink_to(root / "missing.png")
            recover = {
                "ok": True,
                "freeze_ok": True,
                "resume_ok": True,
                "stage_link_ok": True,
                "stage_link_rgb_l1": 0.0,
                "stage_link_rgb_l1_float": 0.002,
                "l1_completed": {"ok": True},
                "l2_completed": {"ok": True},
                "appearance": {"too_large": True},
            }
            payload = build_acceptance(
                name="demo",
                scene="JAX_068",
                view_index=0,
                roi={"center_x": 0.5, "center_y": 0.5, "width": 0.1, "height": 0.1},
                l1_dir=l1,
                l2_dir=l2,
                extra={
                    "archive_dir": str(archive),
                    "recover": recover,
                    "recover_path": str(root / "recover.json"),
                    "notes": ["reused supervision"],
                },
            )
            self.assertEqual(payload["checks"]["continuous_zoom"]["video"], str(archive / "video" / "zoom.mp4"))
            self.assertEqual(payload["checks"]["old_scale"]["l2_weight_at_2x"], 0.14)
            self.assertTrue(payload["checks"]["freeze"]["l1_ok"])
            self.assertNotIn("appearance", payload["recover"])
            self.assertEqual(payload["recover"]["path"], str(root / "recover.json"))
            self.assertIn("renders/l2_step500.png", payload["links"])
            entry = experiment_index_entry(payload)
            self.assertEqual(entry["pending"], ["continuous_zoom"])
            self.assertEqual(entry["checks"]["absorption"], "pass")

    def test_archive_without_recover_keeps_stage_link_pending(self) -> None:
        payload = build_acceptance(
            name="empty",
            scene="JAX_068",
            view_index=0,
            roi={"center_x": 0.28, "center_y": 0.28, "width": 0.1, "height": 0.1},
            l1_dir=None,
            l2_dir=None,
        )
        self.assertEqual(payload["checks"]["stage_link"]["status"], "pending")
        self.assertIsNone(payload["recover"])

    def test_merge_appends_without_dropping_historical(self) -> None:
        from lod.archive import merge_experiment_index

        merged = merge_experiment_index(
            [
                {"name": "jax068_ybuilding", "status": "passed_small_scope", "source_complete": False},
                {"name": "jax068_0p28_0p28", "status": "regression_sample", "source_complete": False},
            ],
            {"name": "jax068_panel_building", "status": "small_sample_stability", "source_complete": True},
        )
        self.assertEqual([item["name"] for item in merged], [
            "jax068_ybuilding", "jax068_0p28_0p28", "jax068_panel_building",
        ])
        self.assertFalse(merged[0]["source_complete"])
        self.assertFalse(merged[1]["source_complete"])


if __name__ == "__main__":
    unittest.main()
