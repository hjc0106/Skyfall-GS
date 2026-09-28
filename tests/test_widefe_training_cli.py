"""Execution-boundary regressions for the portable training controller."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_widefe_gszoom as training


class TrainingExecutionTests(unittest.TestCase):
    def test_selected_virtualenv_is_not_replaced_by_its_system_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)],
                           check=True, capture_output=True)
            executable = environment / "bin/python"
            config = {
                "source_path": str(root / "dataset"), "output_dir": str(root / "run"),
                "python": str(executable), "vlm_python": str(executable),
                "dloral_python": str(executable), "flux_model_path": str(root / "flux"),
                "vlm_model_path": str(root / "vlm"), "dloral_sd_path": str(root / "sd"),
                "dloral_ckpt": str(root / "dloral.pkl"), "dloral_spynet": str(root / "spynet.pth"),
                "lock_dir": str(root / "locks"),
            }
            path = root / "training.json"
            path.write_text(json.dumps(config))
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train_widefe_gszoom.py"),
                 "--config", str(path), "--plan-only"],
                cwd=root, check=True, capture_output=True, text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
            )
            planned = json.loads(result.stdout)["stages"][0]["command"][0]
            observed = subprocess.check_output(
                [planned, "-c", "import sys; print(sys.prefix)"], text=True,
            ).strip()
            self.assertEqual(Path(observed), environment)
            self.assertFalse((root / "run").exists())

    def test_second_controller_cannot_enter_an_owned_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / ".training.lock").open("a") as owner:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(ValueError):
                    training.execute({"output_dir": str(root), "source_path": str(root / "dataset")})
            self.assertFalse((root / "run_status.json").exists())


if __name__ == "__main__":
    unittest.main()
