#!/usr/bin/env python3
"""CPU checks for mix-RNG replay and unlineaged supervision reuse."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lod.lineage import reuse_supervision_dir, supervision_lineage
from lod.train_state import advance_mix_rng, densify_spec, mix_rng, spec_mismatches


class LodTrainStateTests(unittest.TestCase):
    def test_mix_rng_replay_matches_uninterrupted_draws(self) -> None:
        seed, mix_ratio, n_train, interrupt = 0, 0.2, 17, 12
        live = mix_rng(seed)
        seq = []
        for _ in range(20):
            use_train = live.random() < mix_ratio
            choice = live.choice(list(range(n_train))) if use_train else None
            seq.append((use_train, choice))
        resumed = mix_rng(seed)
        advance_mix_rng(resumed, interrupt, mix_ratio=mix_ratio, n_train=n_train)
        rest = []
        for _ in range(interrupt, 20):
            use_train = resumed.random() < mix_ratio
            choice = resumed.choice(list(range(n_train))) if use_train else None
            rest.append((use_train, choice))
        self.assertEqual(rest, seq[interrupt:])

    def test_densify_spec_mismatch_is_named(self) -> None:
        saved = densify_spec(
            densify_from=1, densify_until=250, densify_interval=10,
            densify_grad_threshold=2e-4, max_points=50000, mix_ratio=0.2, seed=0,
        )
        current = dict(saved)
        current["seed"] = 1
        self.assertEqual(spec_mismatches(saved, current), ["seed"])
        self.assertEqual(spec_mismatches(None, current), [])


class LodReuseTests(unittest.TestCase):
    def _lineage(self, directory: Path, seed: int = 0) -> dict:
        ckpt = directory / "stage1.pth"
        if not ckpt.is_file():
            ckpt.write_bytes(b"stage1")
        weights = directory / "dloral.pkl"
        if not weights.is_file():
            weights.write_bytes(b"dloral")
        return supervision_lineage(
            start_checkpoint=str(ckpt),
            parent_lod=None,
            roi={"center_x": 0.592, "center_y": 0.53, "width": 0.1, "height": 0.1},
            view_index=0, zoom_factor=2.0, step_scale=2.0, alignment="geometry",
            seed=seed, max_roundtrip_error_px=None, dloral_ckpt=str(weights),
            prompt_text="zoom in on the central building",
            refined_image=str(directory / "refined.png"),
        )

    def test_cache_hit_and_seed_change_reject(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "refined.png").write_bytes(b"fake")
            lineage = self._lineage(root, seed=0)
            (root / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
            hit = reuse_supervision_dir(root, lineage)
            self.assertTrue(hit["reuse"])
            self.assertTrue(hit["complete"])
            self.assertEqual(hit["source_record"], "lineage_match")
            changed = dict(lineage)
            changed["seed"] = 1
            with self.assertRaises(ValueError):
                reuse_supervision_dir(root, changed)

    def test_unlineaged_reuse_is_explicit_and_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "refined.png").write_bytes(b"fake")
            current = self._lineage(root)
            with self.assertRaises(ValueError):
                reuse_supervision_dir(root, current)
            allowed = reuse_supervision_dir(root, current, allow_unlineaged=True)
            self.assertTrue(allowed["reuse"])
            self.assertFalse(allowed["complete"])
            self.assertEqual(allowed["source_record"], "unlineaged_compat")


if __name__ == "__main__":
    unittest.main()
