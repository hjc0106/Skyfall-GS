#!/usr/bin/env python3
"""CPU checks for Skyfall L0 import: copy parameters and filter, freeze appearance."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from lod.importer import assert_l0_tensors_match, import_skyfall_l0


def _fake_camera(name: str, tx: float = 0.0) -> SimpleNamespace:
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[0, 3] = tx
    w2c[2, 3] = 4.0
    view = w2c.transpose(0, 1).contiguous()
    return SimpleNamespace(
        image_name=name,
        image_width=32,
        image_height=32,
        FoVx=0.8,
        FoVy=0.8,
        cx=0.1,
        cy=-0.05,
        focal_x=32.0 / (2.0 * math.tan(0.4)),
        focal_y=32.0 / (2.0 * math.tan(0.4)),
        znear=0.01,
        zfar=100.0,
        world_view_transform=view,
        camera_center=view.inverse()[3, :3],
        R=w2c[:3, :3].T.numpy(),
        T=w2c[:3, 3].numpy(),
    )


class _StubGaussians:
    def __init__(self, n: int = 16):
        self.max_sh_degree = 1
        self.appearance_enabled = True
        self.appearance_n_fourier_freqs = 4
        self.appearance_embedding_dim = 8
        device = torch.device("cpu")
        self._xyz = torch.zeros(n, 3, device=device)
        self._xyz[:, 2] = 1.0
        self._xyz[:, 0] = torch.linspace(-0.2, 0.2, n)
        self._scaling = torch.full((n, 3), -4.0, device=device)
        self._rotation = torch.zeros(n, 4, device=device)
        self._rotation[:, 0] = 1.0
        self._opacity = torch.zeros(n, 1, device=device)
        self._features_dc = torch.zeros(n, 1, 3, device=device)
        self._features_rest = torch.zeros(n, 3, 3, device=device)
        self.filter_3D = torch.full((n, 1), 0.02, device=device)
        self._embeddings = torch.zeros(n, 24, device=device, requires_grad=True)
        self.appearance_embeddings = torch.zeros(2, 8, device=device, requires_grad=True)
        self.appearance_mlp = torch.nn.Linear(8, 8)
        for param in self.appearance_mlp.parameters():
            param.requires_grad_(True)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)


class LodImportTests(unittest.TestCase):
    def test_import_copies_filter_and_freezes(self) -> None:
        gaussians = _StubGaussians()
        cameras = [_fake_camera("a", 0.0), _fake_camera("b", 0.3)]
        bundle = import_skyfall_l0(gaussians, cameras, step_scale=2.0)
        report = assert_l0_tensors_match(gaussians, bundle)
        self.assertEqual(bundle.n_points, 16)
        self.assertEqual(bundle.step_scale, 2.0)
        self.assertEqual(bundle.filter_source, "copied_from_skyfall")
        self.assertEqual(bundle.lod.stage_records[0]["scale"], 1.0)
        self.assertTrue(bundle.layer0().frozen)
        self.assertFalse(any(p.requires_grad for p in bundle.appearance.mlp.parameters()))
        self.assertLess(max(report.values()), 1e-8)
        gaussians.filter_3D.fill_(9.0)
        self.assertAlmostEqual(float(bundle.layer0().filter_3d.max()), 0.02)

    def test_missing_filter_is_rejected(self) -> None:
        gaussians = _StubGaussians()
        gaussians.filter_3D = None
        with self.assertRaises(ValueError):
            import_skyfall_l0(gaussians, [_fake_camera("a")])


if __name__ == "__main__":
    unittest.main()
