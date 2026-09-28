"""Uncertain boundaries: masked SSIM, camera-Z normals, recursive stage identity."""
from __future__ import annotations

import copy
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from lod.depth import rln_principal_from_skyfall_camera
from lod.path import add_gz_src
from lod.scene_training import (
    RenderCamera, SceneLodTrainer, TrainConfig, extend_parent_base_stage, frequency_residual,
    frequency_support, masked_dssim, masked_l1, rade_depth_normal_consistency, resolve_lr_anchor,
)


def camera(width=64, height=48):
    return RenderCamera(
        image_name="t", uid=0, colmap_id=None,
        R=torch.eye(3), T=[0., 0., 0.], fov_x=.8, fov_y=.6,
        cx=.05, cy=-.03, image_width=width, image_height=height, device="cpu",
    )


class MaskedDssimTests(unittest.TestCase):
    def test_edits_cannot_leak_through_ssim_windows(self):
        target = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0))
        render = target.clone()
        mask = torch.ones(1, 64, 64)
        mask[:, 20:40, 20:40] = 0
        render[:, 20:40, 20:40] = .7
        self.assertAlmostEqual(float(masked_dssim(render, target, mask)), 0., places=7)
        render[:, 8:12, 8:12] = 0
        self.assertGreater(float(masked_dssim(render, target, mask)), 1e-4)

    def test_no_complete_window_has_zero_loss_and_gradient(self):
        render = torch.ones(3, 32, 32, requires_grad=True)
        mask = torch.zeros(1, 32, 32)
        mask[:, 16, 16] = 1
        loss = masked_dssim(render, torch.zeros_like(render), mask)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.equal(render.grad, torch.zeros_like(render)))


class FrequencySupervisionTests(unittest.TestCase):
    @staticmethod
    def detail(image):
        low = F.interpolate(image[None], size=(32, 32), mode="bicubic", align_corners=False, antialias=True)[0]
        return frequency_residual(image, low)

    def test_teacher_brightness_drift_is_not_a_detail_target(self):
        image = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.allclose(self.detail(image), self.detail(image + .2), atol=3e-7))

    def test_blur_has_detail_error_even_when_coarse_brightness_matches(self):
        y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        target = ((x + y) % 2).float().expand(3, -1, -1)
        render = torch.full_like(target, .5, requires_grad=True)
        loss = masked_l1(self.detail(render), self.detail(target), None)
        self.assertGreater(float(loss.detach()), .4)
        loss.backward()
        self.assertGreater(float(render.grad.abs().sum()), .5)

    def test_invalid_pixels_cannot_leak_through_frequency_resampling(self):
        target = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(11))
        render = target.clone()
        mask = torch.ones(1, 64, 64)
        mask[:, 24:40, 24:40] = 0
        render[:, 24:40, 24:40] += 1
        support = frequency_support(mask, (32, 32))
        self.assertAlmostEqual(float(masked_l1(self.detail(render), self.detail(target), support)), 0., places=7)
        render[:, 3:7, 3:7] += .5
        self.assertGreater(float(masked_l1(self.detail(render), self.detail(target), support)), 1e-4)

    def test_point_budget_reserves_late_capacity_and_respects_cap(self):
        trainer = SceneLodTrainer.__new__(SceneLodTrainer)
        trainer.config = TrainConfig(steps_per_level=2000, max_points_per_level=200000,
                                     seed=0, densify_bootstrap_fraction=.25)
        budgets = [trainer.point_budget_for(step, 1200) for step in (0, 25, 600, 1200, 2000)]
        self.assertEqual(budgets, [50000, 53125, 125000, 200000, 200000])


class GeometryLossTests(unittest.TestCase):
    def plane(self):
        cam = camera()
        ray_depth = 5. / rln_principal_from_skyfall_camera(cam)
        normal = torch.zeros(3, 48, 64)
        normal[2] = -1
        return cam, {"render_depth": ray_depth[None], "render_norm": normal,
                     "render_alpha": torch.ones(1, 48, 64)}

    def test_frontal_plane_requires_ray_to_z_conversion(self):
        cam, package = self.plane()
        self.assertAlmostEqual(float(rade_depth_normal_consistency(package, cam)), 0., places=7)
        package["render_norm"] = -package["render_norm"]
        self.assertGreater(float(rade_depth_normal_consistency(package, cam)), 1.9)

    def test_unobserved_nan_buffers_do_not_poison_valid_loss(self):
        cam, package = self.plane()
        package["render_alpha"][:, :8] = 0
        package["render_depth"][:, :8] = float("nan")
        package["render_norm"][:, :8] = float("nan")
        package["render_depth"].requires_grad_(True)
        loss = rade_depth_normal_consistency(package, cam)
        self.assertAlmostEqual(float(loss), 0., places=7)
        loss.backward()
        self.assertTrue(torch.isfinite(package["render_depth"].grad).all())

    def test_geometry_error_backpropagates_to_depth(self):
        cam, package = self.plane()
        package["render_depth"].requires_grad_(True)
        package["render_norm"][0] = .6
        package["render_norm"][2] = -.8
        rade_depth_normal_consistency(package, cam).backward()
        self.assertGreater(float(package["render_depth"].grad.abs().sum()), 0.)

    def test_missing_rasterizer_buffers_fail(self):
        with self.assertRaises(ValueError):
            rade_depth_normal_consistency({"render_depth": torch.ones(1, 48, 64)}, camera())


class LrAnchorTests(unittest.TestCase):
    def test_missing_required_real_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_lr_anchor({}, camera(), 2., "sample", loss_mode="multiscale")

    def test_wrong_image_or_mask_raster_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            image, mask = Path(directory) / "lr.png", Path(directory) / "mask.png"
            Image.new("RGB", (36, 24)).save(image)
            with self.assertRaises(ValueError):
                resolve_lr_anchor({"lr_image_path": str(image)}, camera(), 2., "s", loss_mode="multiscale")
            Image.new("RGB", (32, 24)).save(image)
            Image.new("L", (31, 24)).save(mask)
            with self.assertRaises(ValueError):
                resolve_lr_anchor({"lr_image_path": str(image), "lr_mask_path": str(mask)},
                                  camera(), 2., "s", loss_mode="multiscale")

    def test_negative_rgb_blend_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainConfig(steps_per_level=1, max_points_per_level=100, seed=0,
                        loss_mode="multiscale", loss_dssim=1.1)


def stage_camera(name, zoom=1.):
    return types.SimpleNamespace(name=name, width=64, height=48, fx=100. * zoom,
                                 fy=100. * zoom, cx=32., cy=24., w2c=torch.eye(4))


class ParentStageTests(unittest.TestCase):
    def setUp(self):
        add_gz_src()
        from gaussianzoom_lod.stages import capture_stage
        base = [stage_camera("p0"), stage_camera("p0_z2")]
        records = [capture_stage(base), capture_stage([stage_camera("p0_z2", 2.)], 2.)]
        self.bundle = types.SimpleNamespace(lod=types.SimpleNamespace(stage_records=records))
        self.union = base + [stage_camera("p0_z4")]

    def test_next_level_accepts_new_alias_without_changing_completed_stage(self):
        from gaussianzoom_lod.stages import validate_next_stage
        old = copy.deepcopy(self.bundle.lod.stage_records[1])
        extend_parent_base_stage(self.bundle, self.union, step_scale=2.)
        next_stage = validate_next_stage(self.bundle.lod.stage_records[0],
                                        self.bundle.lod.stage_records[1],
                                        [stage_camera("p0_z4", 4.)], 2.)
        self.assertEqual(next_stage["scale"], 4.)
        self.assertEqual(self.bundle.lod.stage_records[1], old)

    def test_changed_physical_camera_or_missing_parent_alias_is_rejected(self):
        with self.assertRaises(ValueError):
            extend_parent_base_stage(self.bundle, self.union[:1], step_scale=2.)
        self.union[0] = stage_camera("p0", 1.1)
        with self.assertRaises(ValueError):
            extend_parent_base_stage(self.bundle, self.union, step_scale=2.)


if __name__ == "__main__":
    unittest.main()
