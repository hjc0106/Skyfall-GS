"""Scene-zoom geometry: raw ``render_depth`` of the effective backend must
become expected camera-space Z, and every target/sample carries an explicit
versioned ``geometry_identity`` so legacy buffers cannot be reinterpreted."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from refinement.scene_zoom import (
    _checkpoint_rasterizer_backend,
    _effective_rasterizer_backend,
    _expected_camera_z,
    _geometry_identity_is_valid,
    geometry_identity,
)


class _OffAxisCamera:
    """Skyfall camera contract with a noncentral NDC principal point."""

    def __init__(self, width, height, fov_x, fov_y, cx, cy):
        self.image_width = width
        self.image_height = height
        self.FoVx = fov_x
        self.FoVy = fov_y
        self.focal_x = width / (2.0 * math.tan(fov_x * 0.5))
        self.focal_y = height / (2.0 * math.tan(fov_y * 0.5))
        self.cx = cx
        self.cy = cy


class SceneZoomDepthConversionTests(unittest.TestCase):
    """Off-axis camera with alpha < 1: both backends must recover camera Z."""

    def setUp(self):
        self.camera = _OffAxisCamera(
            width=40, height=30, fov_x=0.7, fov_y=0.55, cx=0.35, cy=-0.25
        )
        height, width = 30, 40
        ys = torch.arange(height, dtype=torch.float64).reshape(-1, 1)
        xs = torch.arange(width, dtype=torch.float64).reshape(1, -1)
        cx_px = 0.5 * width * (self.camera.cx + 1.0)
        cy_px = 0.5 * height * (self.camera.cy + 1.0)
        # Principal-ray factor through the *pixel* principal point; the exact
        # quantity the rade conversion must invert.
        ray = ((xs - cx_px) / self.camera.focal_x) ** 2
        ray = ray + ((ys - cy_px) / self.camera.focal_y) ** 2 + 1.0
        self.rln = torch.rsqrt(ray)
        self.z_true = 5.0 + 0.1 * ys + 0.2 * xs  # varying per-pixel camera Z

    def _varying_alpha(self) -> torch.Tensor:
        alpha = torch.empty(30, 40, dtype=torch.float64)
        alpha[:, :20] = 0.3
        alpha[:, 20:] = 0.9
        alpha[15, 5] = 1e-6   # empty pixel (below eps)
        alpha[2, 2] = 0.45    # partial coverage
        return alpha

    def test_diff_gauss_accumulated_depth_divides_by_alpha_once(self):
        alpha = self._varying_alpha()
        depth = self.z_true * alpha
        depth[2, 30] = float("nan")
        camera_z = _expected_camera_z(
            "diff_gauss", depth, alpha, self.camera
        ).double()
        valid = alpha > 1e-4
        valid[2, 30] = False  # non-finite accumulated depth stays zero
        self.assertTrue(torch.allclose(camera_z[valid], self.z_true[valid], atol=1e-4))
        self.assertEqual(float(camera_z[15, 5]), 0.0)
        self.assertEqual(float(camera_z[2, 30]), 0.0)

    def test_rade_ray_distance_multiplies_principal_rln_without_alpha_division(self):
        alpha = self._varying_alpha()
        # RaDe ``render_depth`` is the expected distance along the principal ray.
        render_depth = self.z_true / self.rln
        camera_z = _expected_camera_z(
            "rade", render_depth, alpha, self.camera
        ).double()
        # alpha = 0.3/0.45/0.9 must NOT divide the result again: exact Z recovery.
        valid = alpha > 1e-4
        self.assertTrue(torch.allclose(camera_z[valid], self.z_true[valid], atol=1e-4))
        self.assertEqual(float(camera_z[15, 5]), 0.0)

    def test_unsupported_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            _expected_camera_z("made_up", torch.ones(4, 4), torch.ones(4, 4), self.camera)

    def test_plane_shaped_inputs_are_accepted(self):
        alpha = torch.full((1, 30, 40), 0.7, dtype=torch.float64)
        camera_z = _expected_camera_z(
            "diff_gauss", self.z_true * alpha, alpha, self.camera
        ).double()
        self.assertEqual(tuple(camera_z.shape), (30, 40))
        self.assertTrue(torch.allclose(camera_z, self.z_true, atol=1e-4))


class SceneZoomGeometryIdentityTests(unittest.TestCase):
    def test_unknown_backend_has_no_identity(self):
        with self.assertRaises(ValueError):
            geometry_identity("rade_plus")

    def test_identity_survives_json_roundtrip_and_rejects_legacy(self):
        identity = json.loads(json.dumps(geometry_identity("rade")))
        self.assertTrue(_geometry_identity_is_valid(identity))
        self.assertFalse(_geometry_identity_is_valid({}))  # legacy: no contract
        legacy = dict(identity, schema_version=0)
        self.assertFalse(_geometry_identity_is_valid(legacy))
        mismatched = dict(identity, source_depth="accumulated_camera_z")
        self.assertFalse(_geometry_identity_is_valid(mismatched))
        wrong_output = dict(identity, neighbor_depth="ray_distance")
        self.assertFalse(_geometry_identity_is_valid(wrong_output))
        hole = {
            "schema_version": 1, "renderer_backend": "made_up",
            "target_depth": "camera_z", "neighbor_depth": "camera_z",
        }
        self.assertFalse(_geometry_identity_is_valid(hole))
        missing_depth = {"schema_version": 1, "renderer_backend": "rade"}
        self.assertFalse(_geometry_identity_is_valid(missing_depth))


class SceneZoomGenerationCacheIdentityTests(unittest.TestCase):
    """The versioned geometry contract must participate in the cache key."""

    def _request(self, identity):
        from PIL import Image

        from refinement.base import build_refinement_cache_key
        from refinement.types import (
            CameraSnapshot, PromptDescription, RefinementRequest,
        )

        camera = CameraSnapshot(
            image_name="tile", uid=1, colmap_id=0, image_width=64, image_height=48,
            fov_x=0.5, fov_y=0.5, cx=0.0, cy=0.0,
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), T=(0.0, 0.0, 0.0),
        )
        return build_refinement_cache_key(
            RefinementRequest(
                image=Image.new("RGB", (64, 48)),
                checkpoint="synthetic-checkpoint",
                camera=camera,
                zoom_factor=2.0,
                sr_scale=2.0,
                prompt=PromptDescription(),
                cache_context={"geometry_identity": dict(identity)},
            ),
            "dloral",
        )

    def test_legacy_or_foreign_identity_yields_a_different_cache_key(self):
        legacy = self._request({})
        self.assertNotEqual(legacy, self._request(geometry_identity("rade")))
        self.assertNotEqual(
            self._request(geometry_identity("rade")), self._request(geometry_identity("diff_gauss"))
        )


class SceneZoomBackendResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_backend_comes_from_stage1_cfg_next_to_checkpoint(self):
        (self.root / "cfg_args").write_text(
            str(argparse.Namespace(rasterizer_backend="rade"))
        )
        from refinement.scene_zoom import SceneZoomConfig

        cfg = SceneZoomConfig(
            start_checkpoint=str(self.root / "chkpnt.pth"), output_dir=str(self.root)
        )
        self.assertEqual(_checkpoint_rasterizer_backend(cfg.start_checkpoint), "rade")

    def test_cfg_without_backend_defaults_to_diff_gauss(self):
        (self.root / "cfg_args").write_text(str(argparse.Namespace(source_path="x")))
        self.assertEqual(
            _checkpoint_rasterizer_backend(str(self.root / "c.pth")), "diff_gauss"
        )

    def test_missing_cfg_args_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            _checkpoint_rasterizer_backend(str(self.root / "c.pth"))

    def test_effective_backend_is_resolved_from_render_pipeline(self):
        class _Pipe:
            rasterizer_backend = "rade"

        self.assertEqual(_effective_rasterizer_backend({"pipe": _Pipe()}), "rade")

        class _BadPipe:
            rasterizer_backend = "rade_plus"

        with self.assertRaises(ValueError):
            _effective_rasterizer_backend({"pipe": _BadPipe()})


if __name__ == "__main__":
    unittest.main()
