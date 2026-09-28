"""The SR crop and its zoom camera must describe the same image region."""
import math
import unittest

import numpy as np

from refinement.scene_zoom import ViewTask, build_view_tiles
from refinement.types import CameraSnapshot
from utils.zoom_camera import zoom_fov, zoom_principal_point


class SceneZoomTileTests(unittest.TestCase):
    def test_multiscale_tiles_cover_frame_and_preserve_crop_rays(self):
        width, height = 80, 48
        snapshot = CameraSnapshot(
            image_name="real_view", uid=0, colmap_id=0,
            image_width=width, image_height=height,
            fov_x=0.5, fov_y=0.35, cx=0.35, cy=-0.2,
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            T=(0.0, 0.0, 0.0),
        )
        view = ViewTask(
            view_id="real_view", snapshot=snapshot, base_image_path="unused.png",
            source_stage="stage1", appearance_uid=0,
        )
        base_fx = width / (2 * math.tan(snapshot.fov_x / 2))
        base_fy = height / (2 * math.tan(snapshot.fov_y / 2))
        base_cx, base_cy = width * (snapshot.cx + 1) / 2, height * (snapshot.cy + 1) / 2
        for factor, tiles in build_view_tiles(view, [2.0, 4.0]).items():
            coverage = np.zeros((height, width), dtype=np.int32)
            zoom_fx = width / (2 * math.tan(zoom_fov(snapshot.fov_x, factor) / 2))
            zoom_fy = height / (2 * math.tan(zoom_fov(snapshot.fov_y, factor) / 2))
            for tile in tiles:
                left, top, right, bottom = tile.crop_box
                coverage[top:bottom, left:right] += 1
                tile.roi.validate_for_zoom(factor)
                cx, cy = zoom_principal_point(snapshot.cx, snapshot.cy, tile.roi, factor)
                for fraction_x, fraction_y in ((0.25, 0.75), (0.75, 0.25)):
                    u = left + (right - left) * fraction_x
                    v = top + (bottom - top) * fraction_y
                    ray_x, ray_y = (u - base_cx) / base_fx, (v - base_cy) / base_fy
                    projected_u = zoom_fx * ray_x + width * (cx + 1) / 2
                    projected_v = zoom_fy * ray_y + height * (cy + 1) / 2
                    self.assertAlmostEqual(projected_u, factor * (u - left), places=8)
                    self.assertAlmostEqual(projected_v, factor * (v - top), places=8)
            np.testing.assert_array_equal(coverage, np.ones((height, width), dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
