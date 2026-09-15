#!/usr/bin/env python3
"""Offline checks for the DLoRAL adapter; does not load weights."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refinement.dloral_backend import assert_local_dloral_assets


class DLoRALAdapterTests(unittest.TestCase):
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
            with self.assertRaises(FileNotFoundError):
                assert_local_dloral_assets(
                    repo_root=repo,
                    sd_path=sd,
                    ckpt_path=ckpt,
                    spynet_path=spynet,
                )


if __name__ == "__main__":
    unittest.main()
