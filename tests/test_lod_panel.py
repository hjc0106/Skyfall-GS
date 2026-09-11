#!/usr/bin/env python3
"""CPU checks for the 3-ROI panel lock, index protection, and source completeness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lod.archive import HISTORICAL_INDEX_LOCK, merge_experiment_index
from lod.lineage import write_json, write_train_lineage
from lod.panel import (
    EXCLUDED_HISTORICAL,
    FIXED_SETTINGS,
    PANEL_CLAIM,
    PANEL_DO_NOT_CLAIM,
    PANEL_ROIS,
    archive_gate,
    selection_payload,
    source_side_record,
    summarize_panel,
)


class LodPanelTests(unittest.TestCase):
    def test_selection_is_three_locked_rois_not_historical(self) -> None:
        payload = selection_payload(locked_at="2026-09-11T00:00:00+00:00")
        self.assertTrue(payload["locked_before_training"])
        self.assertEqual(payload["claim"], "small_sample_stability_n=3")
        self.assertEqual(payload["do_not_claim"][0], "general_success_rate")
        self.assertEqual([item["id"] for item in payload["rois"]], ["building", "trees", "parking"])
        self.assertEqual(payload["fixed_settings"]["alignment"], "geometry")
        self.assertFalse(payload["fixed_settings"]["roundtrip_gate"])
        self.assertFalse(payload["fixed_settings"]["spynet_control"])
        self.assertEqual(payload["fixed_settings"]["seed"], 0)
        self.assertEqual(payload["fixed_settings"]["steps"], 500)
        historical = {(item["center_x"], item["center_y"]) for item in EXCLUDED_HISTORICAL}
        chosen = {(item["center_x"], item["center_y"]) for item in PANEL_ROIS}
        self.assertTrue(historical.isdisjoint(chosen))
        self.assertEqual(len(chosen), 3)
        for item in payload["rois"]:
            self.assertTrue(item["zoom_2x_valid"])
            self.assertTrue(item["zoom_4x_valid"])
            self.assertIn(item["dominant"], {"building_boundary", "trees", "parking"})
        from utils.zoom_camera import NormalizedROI
        self.assertFalse(NormalizedROI(0.20, 0.48, 0.1, 0.1).zoom_is_valid(2.0))
        trees = next(item for item in payload["rois"] if item["id"] == "trees")
        self.assertGreaterEqual(trees["center_x"], 0.25)

    def test_index_merge_does_not_upgrade_historical(self) -> None:
        existing = [
            {"name": "jax068_ybuilding", "status": "passed_small_scope", "source_complete": False},
            {"name": "jax068_0p28_0p28", "status": "regression_sample", "source_complete": False},
        ]
        merged = merge_experiment_index(
            existing,
            {"name": "jax068_ybuilding", "status": "small_sample_stability", "source_complete": True},
        )
        by_name = {item["name"]: item for item in merged}
        self.assertEqual(by_name["jax068_ybuilding"]["status"], "passed_small_scope")
        self.assertFalse(by_name["jax068_ybuilding"]["source_complete"])
        merged = merge_experiment_index(
            existing,
            {"name": "jax068_panel_building", "status": "small_sample_stability", "source_complete": True},
        )
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["status"], HISTORICAL_INDEX_LOCK["jax068_ybuilding"]["status"])
        self.assertFalse(merged[1]["source_complete"])

    def test_write_train_lineage_keeps_supervision_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_json(root / "lineage.json", {"kind": "lod_supervision", "seed": 0})
            write_train_lineage(root, {"kind": "lod_train", "level": 1})
            saved = json.loads((root / "lineage.json").read_text(encoding="utf-8"))
            train = json.loads((root / "train_lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["kind"], "lod_supervision")
            self.assertEqual(train["kind"], "lod_train")

    def test_source_complete_needs_supervision_and_train_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_json(root / "lineage.json", {"kind": "lod_supervision"})
            write_json(root / "prompt.json", {"target_prompt": "x"})
            (root / "train_state.pt").write_bytes(b"pt")
            write_json(root / "train_lineage.json", {"kind": "lod_train"})
            record = source_side_record(root, require_prompt=True)
            self.assertTrue(record["complete"])
            (root / "train_state.pt").unlink()
            self.assertFalse(source_side_record(root, require_prompt=True)["complete"])

    def test_summarize_splits_execution_from_visual(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            panel = Path(raw)
            write_json(panel / "SELECTION.json", selection_payload(locked_at="t"))
            payload = summarize_panel(panel)
            self.assertEqual(payload["claim"], PANEL_CLAIM)
            self.assertTrue(payload["closed"])
            self.assertEqual(payload["sample"], "small_sample_stability_n=3")
            self.assertEqual(payload["do_not_claim"], PANEL_DO_NOT_CLAIM)
            self.assertTrue(payload["visual_quality_separate"])
            self.assertEqual(payload["execution_success_count"], 0)
            gate = archive_gate(panel / "building")
            self.assertFalse(gate["ok"])
            self.assertIn("acceptance", gate["missing"])

    def test_summarize_keeps_visual_lists_off_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            panel = Path(raw)
            write_json(panel / "SELECTION.json", selection_payload(locked_at="t"))
            roi = panel / "building"
            roi.mkdir()
            (roi / "stage_log.json").write_text("{}", encoding="utf-8")
            write_json(
                roi / "ACCEPTANCE.json",
                {
                    "checks": {
                        "continuous_zoom": {
                            "status": "visual_checked",
                            "passed": ["silhouette stays connected"],
                            "unresolved": ["flicker not certified"],
                            "do_not_write": "video_fully_passed",
                        }
                    }
                },
            )
            write_json(roi / "l1" / "absorption_curve.json", {"curve": []})
            write_json(roi / "l2" / "absorption_curve.json", {"curve": []})
            write_json(roi / "l2" / "lod_l2_summary.json", {})
            payload = summarize_panel(panel)
            vis = payload["rois"][0]["visual_quality"]
            self.assertEqual(vis["status"], "visual_checked")
            self.assertEqual(vis["passed"], ["silhouette stays connected"])
            self.assertEqual(vis["unresolved"], ["flicker not certified"])
            self.assertTrue(vis["not_used_for_execution_success"])
            self.assertEqual(vis["do_not_write"], "video_fully_passed")
            self.assertFalse(payload["rois"][0]["execution"]["success"])

    def test_fixed_settings_match_entry_defaults(self) -> None:
        self.assertEqual(FIXED_SETTINGS["max_points_per_level"], 50000)
        self.assertEqual(FIXED_SETTINGS["densify_interval"], 10)
        self.assertEqual(FIXED_SETTINGS["mix_ratio"], 0.2)

    def test_memory_accounting_separates_board_use_from_worker_allocated(self) -> None:
        from lod.panel import memory_accounting

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for item in PANEL_ROIS:
                write_json(
                    root / item["id"] / "stage_log.json",
                    {"stages": [{"name": "supervise_l2", "peak_mem_mib": 32291, "elapsed_s": 55}]},
                )
                write_json(
                    root / item["id"] / "l2" / "dloral" / "result.json",
                    {"peak_cuda_memory_bytes": 3438873088, "elapsed_sec": 18.4, "tiled": True},
                )
            mem = memory_accounting(root)
        self.assertEqual(mem["nvidia_smi_memory_used"]["peak_mib"], 32291)
        self.assertEqual(mem["nvidia_smi_memory_used"]["display"], "32,291 MiB ≈ 31.53 GiB")
        self.assertEqual(mem["dloral_worker_allocated"]["peak_bytes"], 3438873088)
        self.assertEqual(mem["dloral_worker_allocated"]["display"], "3,438,873,088 B ≈ 3.20 GiB (3.44 GB)")
        self.assertAlmostEqual(mem["dloral_worker_allocated"]["peak_gb"], 3.438873088)
        self.assertIn("subtract_as_parent_occupancy", mem["do_not"])
        self.assertIn("treat_board_peak_vs_worker_allocated_as_regression", mem["do_not"])
        self.assertGreater(mem["nvidia_smi_memory_used"]["peak_gib"], 10)


if __name__ == "__main__":
    unittest.main()
