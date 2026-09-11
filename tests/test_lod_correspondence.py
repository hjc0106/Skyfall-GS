#!/usr/bin/env python3
"""CPU checks for co-visible metrics, RaDe camera-Z conversion, and L1 outlier flags."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from lod.correspondence import masked_image_metrics, rade_pkg_camera_z
from lod.depth import rln_principal_from_skyfall_camera
from lod.inspect import active_opacity_stats, l1_geometry_report, l1_geometry_tensors


class _FakeCamera:
    def __init__(self):
        self.image_width = 8
        self.image_height = 4
        self.FoVx = 0.4
        self.FoVy = 0.2
        self.cx = 0.35
        self.cy = -0.12
        self.focal_x = self.image_width / (2.0 * math.tan(self.FoVx / 2.0))
        self.focal_y = self.image_height / (2.0 * math.tan(self.FoVy / 2.0))
        self.camera_center = torch.zeros(3)


class _Layer(SimpleNamespace):
    pass


class _Bundle:
    def __init__(self, *layers):
        self._layers = list(layers)

    def layer(self, index: int):
        if index < 0 or index >= len(self._layers):
            return None
        return self._layers[index]

    def layer0(self):
        return self.layer(0)

    def layer1(self):
        return self.layer(1)

    def active_layer(self):
        return self.layer(int(getattr(self.lod, "active_level", 0)))


class CorrespondenceHelperTests(unittest.TestCase):
    def test_masked_metrics_ignore_invalid_and_zero_fill(self) -> None:
        pred = torch.zeros(3, 4, 4)
        target = torch.zeros(3, 4, 4)
        pred[:, 0, 0] = 1.0
        target[:, 0, 0] = 0.0
        pred[:, 1, 1] = 0.4
        target[:, 1, 1] = 0.1
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[1, 1] = True
        stats = masked_image_metrics(pred, target, mask)
        self.assertEqual(stats["valid_pixels"], 1)
        self.assertAlmostEqual(stats["rgb_l1"], 0.3, places=6)
        self.assertAlmostEqual(stats["luma_l1"], 0.3, places=6)

    def test_rade_pkg_converts_ray_distance_to_camera_z(self) -> None:
        camera = _FakeCamera()
        ray = torch.full((4, 8), 10.0)
        pkg = {"render_depth": ray, "render_alpha": torch.ones(4, 8)}
        z = rade_pkg_camera_z(pkg, camera)
        rln = rln_principal_from_skyfall_camera(camera, device=ray.device)
        self.assertTrue(torch.allclose(z, ray * rln))
        self.assertFalse(torch.allclose(z, ray))

    def test_geometry_flags_scale_and_parent_normalized_offset(self) -> None:
        layer0 = _Layer(
            node_ids=torch.tensor([0, 1]),
            xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            log_scales=torch.log(torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
        )
        layer1 = _Layer(
            parent_ids=torch.tensor([0, 0, 1]),
            xyz=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            log_scales=torch.log(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])),
            opacity_logits=torch.tensor([10.0, 10.0, 10.0]),
        )
        tensors = l1_geometry_tensors(_Bundle(layer0, layer1))
        self.assertIsNotNone(tensors)
        self.assertEqual(int(tensors["scale_gt_4x_parent"].sum().item()), 1)
        self.assertEqual(int(tensors["offset_gt_4x_parent"].sum().item()), 1)
        report = l1_geometry_report(_Bundle(layer0, layer1))
        self.assertAlmostEqual(report["frac_scale_gt_4x_parent"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(report["frac_offset_gt_4x_parent"], 1.0 / 3.0, places=6)
        self.assertIn("offset_norm_to_parent", report)

    def test_l2_inherits_l1_parent_embedding(self) -> None:
        from lod.importer import FrozenAppearance, sync_layer_embeddings

        layer0 = _Layer(node_ids=torch.tensor([0, 1]), xyz=torch.zeros(2, 3))
        layer1 = _Layer(
            node_ids=torch.tensor([10, 11]),
            xyz=torch.zeros(2, 3),
            parent_ids=torch.tensor([0, 1]),
        )
        layer2 = _Layer(
            node_ids=torch.tensor([20]),
            xyz=torch.zeros(1, 3),
            parent_ids=torch.tensor([10]),
        )
        l0_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        l1_emb = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
        appearance = FrozenAppearance(
            enabled=True,
            n_fourier_freqs=1,
            embedding_dim=2,
            gaussian_embeddings=l0_emb,
            image_embeddings=None,
            mlp=None,
            layer_embeddings=[l0_emb, l1_emb, l0_emb.new_zeros((0, 2))],
        )
        bundle = _Bundle(layer0, layer1, layer2)
        bundle.lod = type("Lod", (), {"layers": bundle._layers, "active_level": 2})()
        bundle.appearance = appearance
        sync_layer_embeddings(bundle)
        self.assertTrue(torch.equal(bundle.appearance.layer_embeddings[2], torch.tensor([[2.0, 2.0]])))
        self.assertTrue(torch.equal(bundle.appearance.layer_embeddings[1], l1_emb))

    def test_active_opacity_stats_empty_and_opaque(self) -> None:
        empty = _Bundle(_Layer(xyz=torch.zeros(0, 3), opacity_logits=torch.zeros(0, 1)))
        empty.lod = type("Lod", (), {"layers": empty._layers, "active_level": 0})()
        stats = active_opacity_stats(empty)
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["frac_opacity_lt_0.05"], 0.0)
        opaque = _Bundle(_Layer(xyz=torch.zeros(4, 3), opacity_logits=torch.full((4, 1), 8.0)))
        opaque.lod = type("Lod", (), {"layers": opaque._layers, "active_level": 0})()
        stats = active_opacity_stats(opaque)
        self.assertEqual(stats["n"], 4)
        self.assertGreater(stats["opacity_mean"], 0.99)
        self.assertEqual(stats["frac_opacity_lt_0.05"], 0.0)


if __name__ == "__main__":
    unittest.main()
