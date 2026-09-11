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


if __name__ == "__main__":
    unittest.main()
