"""Sharded ``--target_view_ids`` narrows pending work only.

A shard run must keep collecting every view (FlowEdit base evidence plus the
native real replay entries), because the geometry neighbor pool and the parent
renders depend on the full collection; only the new SR targets are restricted
to the requested, already-collected views.
"""
import unittest

from refinement.scene_zoom import (
    REAL_REPLAY_STAGE, SceneZoomConfig, ViewTask, build_tile_plans, build_view_tiles,
)
from refinement.types import CameraSnapshot


def camera(name, uid):
    return CameraSnapshot(
        image_name=name, uid=uid, colmap_id=uid, image_width=64, image_height=64,
        fov_x=0.8, fov_y=0.8, cx=0.0, cy=0.0,
        R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        T=(float(uid), 0.0, 0.0),
    )


def plan_ids(tasks, **config):
    cfg = SceneZoomConfig("/tmp/unused.pth", "/tmp/out", zoom_factors=(2.0,), **config)
    return build_tile_plans(tasks, cfg, level_offset=1)


class TargetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.supervised = [
            ViewTask(f"fe_v{index}", camera(f"fe{index}", 10 + index), "fe.png", "stage2", 6)
            for index in range(3)
        ]
        self.replay = [
            ViewTask(f"real_cam{index}", camera(f"real{index}", index), "real.png",
                     REAL_REPLAY_STAGE, index)
            for index in range(2)
        ]
        self.tasks = self.supervised + self.replay

    def test_no_selection_plans_every_supervised_view(self):
        plans = plan_ids(self.tasks)
        self.assertEqual([view.view_id for view, _ in plans],
                         [view.view_id for view in self.supervised])

    def test_requested_subset_keeps_pool_and_tile_geometry(self):
        view = self.supervised[2]
        plans = plan_ids(self.tasks, target_view_ids=(view.view_id,))
        self.assertEqual([view.view_id for view, _ in plans], [view.view_id])
        # The requested view keeps the exact tiles of an unsharded run.
        self.assertEqual(plans[0][1], build_view_tiles(view, (2.0,), level_offset=1))
        # Collection itself is untouched: every other view stays available for
        # base evidence, replay and neighbor selection.
        self.assertEqual(len(self.tasks), 5)
        self.assertEqual([task.view_id for task in self.tasks],
                         [task.view_id for task in self.supervised + self.replay])

    def test_selection_keeps_collection_order(self):
        plans = plan_ids(
            self.tasks,
            target_view_ids=(self.supervised[2].view_id, self.supervised[0].view_id),
        )
        self.assertEqual([view.view_id for view, _ in plans],
                         [self.supervised[0].view_id, self.supervised[2].view_id])

    def test_unknown_view_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown view id"):
            plan_ids(self.tasks, target_view_ids=("fe_missing",))

    def test_replay_only_view_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "replay-only"):
            plan_ids(self.tasks, target_view_ids=(self.replay[0].view_id,))

    def test_repeated_view_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "repeats view id"):
            plan_ids(self.tasks, target_view_ids=("fe_v1", "fe_v1"))


if __name__ == "__main__":
    unittest.main()
