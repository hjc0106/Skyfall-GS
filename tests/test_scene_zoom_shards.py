"""Behavioral regressions for complete sharded scene-zoom merges."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRAST = "normalized_float_clamped_v1"
CHECKPOINT = "/run/repaired/chkpnt32000.pth"
FE_IDS = [f"fe_v{index:02d}" for index in range(12)]
REPLAY_IDS = [f"real_cam{index:02d}" for index in range(17)]


def _load_shard_cli():
    candidate = ROOT / "scripts" / "prepare_scene_zoom_shards.py"
    spec = importlib.util.spec_from_file_location("scene_zoom_shards", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SR = _load_shard_cli()


def camera(uid):
    return {
        "image_name": f"cam{uid}", "uid": uid, "colmap_id": uid,
        "image_width": 64, "image_height": 64, "fov_x": 0.8, "fov_y": 0.8,
        "cx": 0.0, "cy": 0.0,
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "T": [float(uid), 0.0, 0.0],
    }


class SrShardMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = self.write(self.root / "protocol.json", {
            "python": "/env/base/bin/python",
            "repo_root": str(ROOT),
            "output_dir": str(self.root / "run-artifacts"),
            "source_path": str(self.root / "dataset"),
            "vlm_python": "/env/vlm/bin/python",
            "vlm_model_path": "/models/vlm",
            "dloral_python": "/env/dloral/bin/python",
            "dloral_root": str(ROOT / "submodules" / "DLoRAL"),
            "dloral_sd_path": "/models/dloral",
            "dloral_ckpt": "/models/dloral/model.pkl",
            "dloral_spynet": "/models/dloral/spynet.pth",
        })
        self.manifests = [
            self.write(self.root / "angle85.json", {
                "kind": "skyfall_flowedit_views", "schema_version": 1, "episode_idx": 85,
                "views": [{"id": view_id} for view_id in FE_IDS[:8]],
            }),
            self.write(self.root / "angle75.json", {
                "kind": "skyfall_flowedit_views", "schema_version": 1, "episode_idx": 75,
                "views": [{"id": view_id} for view_id in FE_IDS[8:]],
            }),
        ]
        self.l1_root = self.root / "z2"
        self.level1 = self.plan(self.l1_root, zoom=2, views_per_shard=5)
        for shard in self.level1["shards"]:
            self.build_shard(self.l1_root, self.level1, shard, 2.0)
        with self.quiet():
            SR.merge_shards(str(self.l1_root / "plan.json"))

    # ---------------------------------------------------------------- helpers
    @contextlib.contextmanager
    def quiet(self):
        """The CLI reports progress on stdout; tests assert artifacts instead."""

        with contextlib.redirect_stdout(io.StringIO()):
            yield

    def write(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return str(path)

    def touch(self, path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return str(path)

    def plan(self, output_dir, *, zoom, views_per_shard, parent=None, previous=None):
        argv = [
            "plan", "--protocol", self.protocol, "--checkpoint", CHECKPOINT,
            "--zoom", str(zoom), "--output-dir", str(output_dir),
            "--flowedit-manifests", *self.manifests,
            "--views-per-shard", str(views_per_shard),
        ]
        if parent:
            argv.extend(["--parent-lod-checkpoint", parent])
        if previous:
            argv.extend(["--previous-supervision", previous])
        with self.quiet():
            SR.main(argv)
        return json.loads((Path(output_dir) / "plan.json").read_text())

    def build_shard(self, root, plan, shard, zoom, parent=None, previous=None):
        """Synthetic shard output mirroring prepare_scene_zoom's artifacts."""
        shard_dir = Path(shard["output_dir"])
        progress = {
            "enabled": True, "render_rgb_contract": CONTRAST, "step_scale": 2.0,
            "start_checkpoint": {"path": CHECKPOINT, "size": 1, "mtime_ns": 1,
                                 "sha256": "c" * 64},
            "parent_lod_checkpoint": (
                None if parent is None
                else {"path": parent, "size": 2, "mtime_ns": 2, "sha256": "p" * 64}
            ),
            "parent_appearance": (
                None if parent is None
                else {"path": parent + ".appearance.pt", "size": 3, "mtime_ns": 3,
                      "sha256": "q" * 64}
            ),
            "carried_from": (
                None if previous is None
                else {"manifest": previous, "base_checkpoint": CHECKPOINT,
                      "levels": [2.0], "samples": len(FE_IDS) * 4}
            ),
        }
        views = [
            {
                "id": view_id,
                "camera": camera(index),
                "image_path": self.touch(Path(plan["output_dir"]) / "fe" / f"{view_id}.png",
                                         f"fe:{view_id}"),
                "source_stage": "stage2", "appearance_uid": index,
            }
            for index, view_id in enumerate(FE_IDS)
        ]
        views.extend(
            {
                "id": view_id,
                "camera": camera(index),
                "image_path": self.touch(shard_dir / "base" / f"{view_id}.png",
                                         f"replay:{view_id}"),
                "source_stage": "real_replay", "appearance_uid": index,
            }
            for index, view_id in enumerate(REPLAY_IDS)
        )

        levels = json.loads(Path(previous).read_text())["levels"] if previous else []
        grid = int(round(zoom))
        samples = [
            self.sample(view_id, zoom, row, col, shard_dir / "targets")
            for view_id in shard["views"]
            for row in range(grid)
            for col in range(grid)
        ]
        self.assertEqual(len(samples), len(shard["views"]) * grid * grid)
        levels = list(levels) + [{"zoom_factor": float(zoom), "samples": samples}]
        manifest_path = self.write(shard_dir / "supervision.json", {
            "schema_version": 1, "kind": "skyfall_scene_zoom",
            "base_checkpoint": CHECKPOINT, "views": views, "levels": levels,
            "progressive": progress,
        })
        self.write(shard_dir / "prepare_scene_zoom_summary.json", {
            "stage": "stage2", "progressive": True, "output_dir": str(shard_dir),
            "start_checkpoint": CHECKPOINT, "zoom_factors": [float(zoom)],
            "dloral_alignment": "geometry", "include_real_replay": True,
            "supervision_manifest": manifest_path,
            "target_view_ids": list(shard["views"]),
            "flowedit_views": len(FE_IDS), "real_replay_views": len(REPLAY_IDS),
            "views_collected": len(FE_IDS) + len(REPLAY_IDS),
            "views_pending": len(shard["views"]),
            "views_skipped_by_max_views": 0,
            "views_skipped_by_target_selection": len(FE_IDS) - len(shard["views"]),
            "samples_generated": len(shard["views"]) * grid * grid,
            "samples_reused": 0, "geometry_neighbor_count": 1,
            "parent_lod_checkpoint": parent, "previous_supervision": previous,
        })

    def sample(self, view_id, zoom, row, col, target_dir):
        sample_id = f"{view_id}__z{zoom:g}__r{row}c{col}"
        return {
            "sample_id": sample_id, "view_id": view_id,
            "tile_row": row, "tile_col": col, "zoom_factor": float(zoom),
            "image_path": self.touch(target_dir / f"{sample_id}.png", f"target:{sample_id}"),
            "lr_image_path": self.touch(target_dir / f"{sample_id}.lr.png",
                                        f"anchor:{sample_id}"),
            "roi": {"center_x": (col + 0.5) / zoom, "center_y": (row + 0.5) / zoom,
                    "width": 1.0 / zoom, "height": 1.0 / zoom},
            "geometry": {"mode": "geometry"},
            "progressive": {"render_rgb_contract": CONTRAST},
        }

    def plan_file(self, root):
        return json.loads((Path(root) / "plan.json").read_text())

    def mutate_shard(self, plan, index, mutate):
        path = Path(plan["shards"][index]["output_dir"]) / "supervision.json"
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2))

    # ----------------------------------------------------------------- tests
    def test_l1_merge_covers_every_planned_teacher_once(self):
        with self.quiet():
            report = SR.merge_shards(str(self.l1_root / "plan.json"))
        merged = json.loads((self.l1_root / "supervision.json").read_text())
        levels = {level["zoom_factor"]: level["samples"] for level in merged["levels"]}
        self.assertEqual(sorted(levels), [2.0])
        self.assertEqual(len(levels[2.0]), len(FE_IDS) * 4)
        self.assertEqual(len({sample["sample_id"] for sample in levels[2.0]}),
                         len(FE_IDS) * 4)
        self.assertEqual(sorted({view["id"] for view in merged["views"]}),
                         sorted(FE_IDS + REPLAY_IDS))
        self.assertEqual(len(merged["views"]), len(FE_IDS) + len(REPLAY_IDS))
        self.assertEqual(report["teachers"]["planned_level"], len(FE_IDS) * 4)
        self.assertEqual(report["views"]["real_replay"], len(REPLAY_IDS))
        for view_id in FE_IDS:
            self.assertEqual(
                sum(1 for sample in levels[2.0] if sample["view_id"] == view_id), 4)

    def test_native_replay_count_is_dataset_dependent(self):
        expected = REPLAY_IDS[:3]
        root = self.root / "three-native-views"
        with patch.dict(globals(), {"REPLAY_IDS": expected}):
            plan = self.plan(root, zoom=2, views_per_shard=5)
            for shard in plan["shards"]:
                self.build_shard(root, plan, shard, 2.0)
            with self.quiet():
                SR.merge_shards(str(root / "plan.json"))
        merged = json.loads((root / "supervision.json").read_text())
        observed = {view["id"] for view in merged["views"] if view["source_stage"] == "real_replay"}
        self.assertEqual(observed, set(expected))

    def test_l2_merge_adds_16x_coverage_and_carries_l1_once(self):
        parent = self.touch(self.root / "l1_final.lod.pt", "parent")
        root = self.root / "z4"
        plan = self.plan(root, zoom=4, views_per_shard=6, parent=parent,
                         previous=str(self.l1_root / "supervision.json"))
        for shard in plan["shards"]:
            self.build_shard(root, plan, shard, 4.0, parent=parent,
                             previous=str(self.l1_root / "supervision.json"))
        with self.quiet():
            SR.merge_shards(str(root / "plan.json"))
        merged = json.loads((root / "supervision.json").read_text())
        levels = {level["zoom_factor"]: level["samples"] for level in merged["levels"]}
        self.assertEqual(sorted(levels), [2.0, 4.0])
        self.assertEqual(len(levels[4.0]), len(FE_IDS) * 16)
        self.assertEqual(len({sample["sample_id"] for sample in levels[4.0]}),
                         len(FE_IDS) * 16)
        self.assertEqual(len(levels[2.0]), len(FE_IDS) * 4)
        self.assertEqual(len({sample["sample_id"] for sample in levels[2.0]}),
                         len(FE_IDS) * 4)

    def test_missing_shard_output_is_refused(self):
        shutil.rmtree(Path(self.l1_root) / "z2_s0001")
        with self.quiet(), self.assertRaisesRegex(ValueError, "missing .*supervision.json"):
            SR.merge_shards(str(self.l1_root / "plan.json"))

    def test_missing_teacher_is_refused(self):
        self.mutate_shard(self.level1, 0,
                          lambda payload: payload["levels"][-1]["samples"].pop())
        with self.quiet(), self.assertRaisesRegex(ValueError, "expected .* teachers"):
            SR.merge_shards(str(self.l1_root / "plan.json"))

    def test_teacher_planned_by_two_shards_is_refused(self):
        donor = json.loads(
            (Path(self.level1["shards"][0]["output_dir"]) / "supervision.json").read_text()
        )["levels"][-1]["samples"][0]
        self.mutate_shard(self.level1, 1,
                          lambda payload: payload["levels"][-1]["samples"].__setitem__(0, donor))
        with self.quiet(), self.assertRaisesRegex(ValueError, "unplanned sample"):
            SR.merge_shards(str(self.l1_root / "plan.json"))

    def test_conflicting_carried_sample_is_refused(self):
        parent = self.touch(self.root / "l1_final.lod.pt", "parent")
        root = self.root / "z4"
        plan = self.plan(root, zoom=4, views_per_shard=6, parent=parent,
                         previous=str(self.l1_root / "supervision.json"))
        for shard in plan["shards"]:
            self.build_shard(root, plan, shard, 4.0, parent=parent,
                             previous=str(self.l1_root / "supervision.json"))
        self.mutate_shard(
            plan, 1,
            lambda payload: payload["levels"][0]["samples"][0].__setitem__(
                "image_path", "/elsewhere/target.png"),
        )
        with self.quiet(), self.assertRaisesRegex(ValueError, "differs from previous supervision"):
            SR.merge_shards(str(root / "plan.json"))

    def test_changed_parent_appearance_is_refused(self):
        parent = self.touch(self.root / "l1_final.lod.pt", "parent")
        root = self.root / "z4"
        plan = self.plan(root, zoom=4, views_per_shard=6, parent=parent,
                         previous=str(self.l1_root / "supervision.json"))
        for shard in plan["shards"]:
            self.build_shard(root, plan, shard, 4.0, parent=parent,
                             previous=str(self.l1_root / "supervision.json"))

        def change_parent(payload):
            payload["progressive"]["parent_appearance"]["sha256"] = "z" * 64

        self.mutate_shard(plan, 1, change_parent)
        with self.quiet(), self.assertRaisesRegex(ValueError, "progressive contract differs"):
            SR.merge_shards(str(root / "plan.json"))


    def test_replay_image_content_mismatch_is_refused(self):
        shard_dir = Path(self.level1["shards"][1]["output_dir"])
        manifest_path = shard_dir / "supervision.json"
        manifest = json.loads(manifest_path.read_text())
        replay = next(
            view for view in manifest["views"] if view["source_stage"] == "real_replay"
        )
        Path(replay["image_path"]).write_text("changed replay bytes")
        with self.quiet(), self.assertRaisesRegex(ValueError, "content differs"):
            SR.merge_shards(str(self.l1_root / "plan.json"))

if __name__ == "__main__":
    unittest.main()
