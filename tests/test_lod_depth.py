#!/usr/bin/env python3
"""CPU checks for Skyfall vs RaDe-GS depth conversion helpers."""

from __future__ import annotations

import unittest

import torch

from lod.depth import pair_depth_metrics, ray_distance_to_z, skyfall_expected_z, z_to_ray_distance


class DepthConventionTests(unittest.TestCase):
    def test_expected_z_divides_out_alpha(self) -> None:
        accum = torch.tensor([[2.0, 4.0]])
        alpha = torch.tensor([[0.5, 0.8]])
        z = skyfall_expected_z(accum, alpha)
        self.assertAlmostEqual(float(z[0, 0]), 4.0)
        self.assertAlmostEqual(float(z[0, 1]), 5.0)

    def test_z_and_ray_distance_roundtrip(self) -> None:
        z = torch.tensor([[10.0, 20.0]])
        rln = torch.tensor([[0.5, 0.8]])
        dist = z_to_ray_distance(z, rln)
        back = ray_distance_to_z(dist, rln)
        self.assertTrue(torch.allclose(back, z))

    def test_pairing_prefers_expected_z_when_rade_is_normalized(self) -> None:
        alpha = torch.full((4, 4), 0.8)
        z = torch.full((4, 4), 50.0)
        accum = z * alpha
        rade = z.clone()
        rln = torch.ones(4, 4)
        metrics = pair_depth_metrics(accum, alpha, rade, alpha, rln, rln)
        self.assertLess(metrics["expected_z_vs_rade_rel"], metrics["raw_accum_vs_rade_rel"])
        self.assertLess(metrics["expected_z_vs_rade_rel"], 1e-5)

    def test_depth_map_converts_camera_z_and_ray_distance(self) -> None:
        from lod.depth import DepthMap

        rln = torch.full((2, 2), 0.5)
        z = torch.full((2, 2), 10.0)
        depth = DepthMap(value=z, kind="camera_z", rln_principal=rln)
        self.assertTrue(torch.allclose(depth.ray_distance(), torch.full((2, 2), 20.0)))
        ray = DepthMap(value=depth.ray_distance(), kind="ray_distance", rln_principal=rln)
        self.assertTrue(torch.allclose(ray.camera_z(), z))


if __name__ == "__main__":
    unittest.main()
