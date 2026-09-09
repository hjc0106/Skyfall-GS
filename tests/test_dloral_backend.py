#!/usr/bin/env python3
"""Offline checks for the DLoRAL adapter; does not load weights."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refinement.dloral_backend import (
    CLIP_TOKEN_LIMIT,
    DLORAL_PINNED_COMMIT,
    assert_local_dloral_assets,
)


class DLoRALAdapterTests(unittest.TestCase):
    def test_pinned_commit_is_recorded(self) -> None:
        self.assertEqual(len(DLORAL_PINNED_COMMIT), 40)
        self.assertEqual(CLIP_TOKEN_LIMIT, 77)

    def test_missing_assets_list_every_required_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as raised:
                assert_local_dloral_assets(
                    repo_root=root / "missing-repo",
                    sd_path=root / "missing-sd",
                    ckpt_path=root / "missing.pkl",
                    spynet_path=root / "missing-spynet.pth",
                )
            message = str(raised.exception)
            self.assertIn("no implicit download", message)
            self.assertIn("src/DLoRAL_model.py", message)
            self.assertIn("SD 2.1", message)
            self.assertIn("DLoRAL checkpoint", message)
            self.assertIn("SpyNet", message)
            self.assertIn(DLORAL_PINNED_COMMIT, message)

    def test_incomplete_sd_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            (repo / "src").mkdir(parents=True)
            (repo / "src" / "DLoRAL_model.py").write_text("# stub\n", encoding="utf-8")
            sd = root / "sd"
            (sd / "unet").mkdir(parents=True)
            (sd / "vae").mkdir(parents=True)
            (sd / "model_index.json").write_text("{}", encoding="utf-8")
            ckpt = root / "model.pkl"
            ckpt.write_bytes(b"stub")
            spynet = root / "spynet.pth"
            spynet.write_bytes(b"stub")
            with self.assertRaises(FileNotFoundError) as raised:
                assert_local_dloral_assets(
                    repo_root=repo,
                    sd_path=sd,
                    ckpt_path=ckpt,
                    spynet_path=spynet,
                )
            self.assertIn("download incomplete", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
