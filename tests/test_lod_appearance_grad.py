#!/usr/bin/env python3
"""Frozen appearance MLP must still pass color gradients into L1 SH."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from lod.render import appearance_colors_lod


class _PassthroughMLP(nn.Module):
    def forward(self, gembedding, aembedding, color, viewdir=None):
        del gembedding, aembedding, viewdir
        return color.reshape(color.shape[0], -1)


class AppearanceGradientTests(unittest.TestCase):
    def test_frozen_mlp_does_not_block_sh_grad(self) -> None:
        n, k = 8, 4
        xyz = torch.zeros(n, 3)
        xyz[:, 2] = 1.0
        sh = torch.full((n, k, 3), 0.1, requires_grad=True)
        gemb = torch.zeros(n, 6)
        mlp = _PassthroughMLP()
        for param in mlp.parameters():
            param.requires_grad_(False)
        appearance = SimpleNamespace(enabled=True, mlp=mlp)
        center = torch.zeros(3)
        image_emb = torch.zeros(4)
        rgb = appearance_colors_lod(xyz, sh, gemb, appearance, center, image_emb, 1)
        rgb.sum().backward()
        self.assertIsNotNone(sh.grad)
        self.assertGreater(float(sh.grad.abs().sum()), 0.0)
        self.assertFalse(any(p.requires_grad for p in mlp.parameters()))
        self.assertTrue(all(p.grad is None for p in mlp.parameters()))

    def test_frozen_prefix_cache_preserves_active_colors_and_gradients(self) -> None:
        class AppearanceMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(12, 12)

            def forward(self, gembedding, aembedding, color):
                return self.projection(color.reshape(len(color), -1))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            mlp = AppearanceMLP().requires_grad_(False)
            appearance = SimpleNamespace(enabled=True, mlp=mlp)
            fixed_xyz = torch.rand(5, 3) + 1
            active_xyz = (torch.rand(3, 3) + 1).requires_grad_()
            fixed_sh = torch.rand(5, 4, 3) * .1
            active_sh = (torch.rand(3, 4, 3) * .1).requires_grad_()
            xyz, sh = torch.cat((fixed_xyz, active_xyz)), torch.cat((fixed_sh, active_sh))
            embedding = torch.zeros(4)
            center = torch.zeros(3)
            gemb = torch.zeros(8, 6)
            with torch.no_grad():
                cache = appearance_colors_lod(fixed_xyz, fixed_sh, gemb[:5], appearance, center, embedding, 1)
            full = appearance_colors_lod(xyz, sh, gemb, appearance, center, embedding, 1)
            cached = appearance_colors_lod(xyz, sh, gemb, appearance, center, embedding, 1, frozen_colors=cache)
            self.assertTrue(torch.allclose(full, cached, atol=1e-7))
            full_grad = torch.autograd.grad(full.square().sum(), (active_xyz, active_sh), retain_graph=True)
            cache_grad = torch.autograd.grad(cached.square().sum(), (active_xyz, active_sh))
            for expected, actual in zip(full_grad, cache_grad):
                self.assertTrue(torch.allclose(expected, actual, atol=1e-7))
                self.assertGreater(float(actual.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
