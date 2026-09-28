"""Replay-only real views must never become super-resolution targets."""
import unittest

from refinement.scene_zoom import REAL_REPLAY_STAGE, SceneZoomConfig, ViewTask, build_tile_plans
from refinement.types import CameraSnapshot


def snapshot(name="v", uid=3, shift=0.0):
    return CameraSnapshot(
        name, uid, None, 64, 64, 0.8, 0.8, 0.0, 0.0,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (shift, 0.0, 0.0),
    )



class ReplayPlanTests(unittest.TestCase):
    def test_replay_views_never_receive_tile_plans(self):
        cfg = SceneZoomConfig("/tmp/unused.pth", "/tmp/out", zoom_factors=(2.0,))
        flowedit = ViewTask("fe_view", snapshot(), "fe.png", "stage2", 6)
        replay = ViewTask("real_cam", snapshot(uid=1), "real.png", REAL_REPLAY_STAGE, 1)
        plans = build_tile_plans([flowedit, replay], cfg, level_offset=1)
        self.assertEqual([view.view_id for view, _ in plans], ["fe_view"])



if __name__ == "__main__":
    unittest.main()
