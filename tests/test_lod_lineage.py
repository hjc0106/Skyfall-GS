#!/usr/bin/env python3
"""CPU checks for LoD run fingerprints and frozen-layer snapshots."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from lod.freeze import (
    appearance_freeze_report,
    assert_snapshot_unchanged,
    completed_layers_frozen,
    snapshot_levels,
    snapshot_mismatches,
)
from lod.lineage import assert_lineage_reusable, file_identity, lineage_mismatches, supervision_lineage


class _Layer:
    def __init__(self, n: int = 4, frozen: bool = True):
        self.xyz = torch.zeros(n, 3)
        self.sh = torch.zeros(n, 4, 3)
        self.log_scales = torch.zeros(n, 3)
        self.rotations = torch.zeros(n, 4)
        self.opacity_logits = torch.zeros(n, 1)
        self.filter_3d = torch.full((n, 1), 0.02)
        self.frozen = frozen


class _Bundle:
    def __init__(self, layers):
        self._layers = layers
        self.lod = SimpleNamespace(active_level=len(layers) - 1)
        self.appearance = SimpleNamespace(
            mlp=torch.nn.Linear(4, 4),
            image_embeddings=torch.zeros(2, 4, requires_grad=False),
            gaussian_embeddings=torch.zeros(layers[0].xyz.shape[0], 4, requires_grad=False),
            layer_embeddings=[torch.zeros(layer.xyz.shape[0], 4, requires_grad=False) for layer in layers],
        )
        for param in self.appearance.mlp.parameters():
            param.requires_grad_(False)

    def layer(self, index: int):
        if index < 0 or index >= len(self._layers):
            return None
        return self._layers[index]


class LodLineageTests(unittest.TestCase):
    def test_supervision_reuse_rejects_parent_checkpoint_change(self) -> None:
        roi = {"center_x": 0.28, "center_y": 0.28, "width": 0.1, "height": 0.1}
        left = supervision_lineage(
            start_checkpoint="/tmp/missing-a.pth",
            parent_lod="/tmp/missing-l1.pt",
            roi=roi, view_index=0, zoom_factor=4.0, step_scale=2.0,
            alignment="geometry", seed=0, max_roundtrip_error_px=None,
            dloral_ckpt="/tmp/missing.pkl", prompt_text="zoom in on the central building",
        )
        right = dict(left)
        right["parent_lod"] = {"path": "/tmp/other-l1.pt", "size": 1, "mtime_ns": 2, "missing": False}
        self.assertIn("parent_lod", lineage_mismatches(left, right))
        with self.assertRaises(ValueError):
            assert_lineage_reusable(left, right)

    def test_view_name_is_recorded_but_not_a_reuse_key(self) -> None:
        roi = {"center_x": 0.66, "center_y": 0.66, "width": 0.1, "height": 0.1}
        named = supervision_lineage(
            start_checkpoint="/tmp/a.pth", parent_lod=None, roi=roi, view_index=0,
            zoom_factor=2.0, step_scale=2.0, alignment="geometry", seed=0,
            max_roundtrip_error_px=None, dloral_ckpt=None, prompt_text="prompt",
            image_name="JAX_214_018_RGB",
        )
        self.assertEqual(named["view_name"], "JAX_214_018_RGB")
        other = dict(named)
        other["view_name"] = "JAX_214_011_RGB"
        self.assertNotIn("view_name", lineage_mismatches(named, other))

    def test_roundtrip_gate_off_is_part_of_lineage(self) -> None:
        roi = {"center_x": 0.592, "center_y": 0.53, "width": 0.1, "height": 0.1}
        base = supervision_lineage(
            start_checkpoint="/tmp/a.pth", parent_lod=None, roi=roi, view_index=0,
            zoom_factor=2.0, step_scale=2.0, alignment="geometry", seed=0,
            max_roundtrip_error_px=None, dloral_ckpt=None, prompt_text="prompt",
        )
        gated = dict(base)
        gated["max_roundtrip_error_px"] = 1.0
        self.assertIn("max_roundtrip_error_px", lineage_mismatches(base, gated))
        self.assertFalse(base["roundtrip_gate_default"])

    def test_file_identity_marks_missing(self) -> None:
        ident = file_identity("/tmp/definitely-not-a-lod-file.pt")
        self.assertTrue(ident["missing"])

    def test_frozen_layer_snapshot_detects_xyz_edit(self) -> None:
        bundle = _Bundle([_Layer(), _Layer(n=3)])
        before = snapshot_levels(bundle, (0, 1))
        self.assertTrue(completed_layers_frozen(bundle, up_to_level=1)["ok"])
        self.assertTrue(appearance_freeze_report(bundle.appearance)["ok"])
        bundle.layer(0).xyz[0, 0] = 1.0
        after = snapshot_levels(bundle, (0, 1))
        self.assertIn("level_0_xyz", snapshot_mismatches(before, after))
        with self.assertRaises(AssertionError):
            assert_snapshot_unchanged(before, after)

    def test_active_layer_appearance_growth_is_not_a_freeze_break(self) -> None:
        bundle = _Bundle([_Layer(), _Layer(n=3)])
        before = snapshot_levels(bundle, (0,))
        bundle.appearance.layer_embeddings[1] = torch.zeros(8, 4, requires_grad=False)
        after = snapshot_levels(bundle, (0,))
        self.assertEqual(snapshot_mismatches(before, after), [])

    def test_frozen_appearance_change_is_detected(self) -> None:
        bundle = _Bundle([_Layer(), _Layer(n=3)])
        before = snapshot_levels(bundle, (0,))
        bundle.appearance.layer_embeddings[0] = bundle.appearance.layer_embeddings[0] + 1
        after = snapshot_levels(bundle, (0,))
        self.assertIn("appearance_layer_0", snapshot_mismatches(before, after))

    def test_unfreezing_completed_layer_fails(self) -> None:
        bundle = _Bundle([_Layer(frozen=False), _Layer()])
        report = completed_layers_frozen(bundle, up_to_level=1)
        self.assertFalse(report["ok"])
        self.assertFalse(report["layers"]["0"])


if __name__ == "__main__":
    unittest.main()
