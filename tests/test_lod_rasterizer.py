#!/usr/bin/env python3
"""Import-time checks for the RaDe-GS adapter. Skips if the CUDA extension is absent."""

from __future__ import annotations

import unittest

from lod.rasterizer import REQUIRED_SETTINGS, require_rade_gs


class RadeGsAdapterTests(unittest.TestCase):
    def test_require_rade_gs_or_skip(self) -> None:
        try:
            settings, rasterizer = require_rade_gs()
        except ImportError:
            self.skipTest("diff_gaussian_rasterization is not installed")
        fields = settings._fields
        for name in REQUIRED_SETTINGS:
            self.assertIn(name, fields)
        self.assertEqual(rasterizer.__module__, "diff_gaussian_rasterization")

    def test_diff_gauss_still_imports(self) -> None:
        import diff_gauss

        self.assertIsNotNone(diff_gauss)


if __name__ == "__main__":
    unittest.main()
