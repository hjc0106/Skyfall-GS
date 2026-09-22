from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_lod_jax068_c_episodes_l0_108 as runner


class L0Start108RunnerSafetyTests(unittest.TestCase):
    def test_runner_does_not_import_private_experiment_scripts_or_dirty_adapter(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from scripts.run_lod_", source)
        self.assertNotIn("from submodules.FlowEdit.idu_refine", source)
        self.assertIn("from refinement.flowedit_idu import FlowEditRefineIDU", source)

    def test_completed_continuation_archive_is_never_writable(self):
        with self.assertRaises(ValueError):
            runner._assert_output_isolated(runner.PREVIOUS_ARCHIVE)
        with self.assertRaises(ValueError):
            runner._assert_output_isolated(runner.PREVIOUS_ARCHIVE / "episode05")
        runner._assert_output_isolated(runner.DEFAULT_OUTPUT_ROOT)

    def test_missing_adam_state_is_never_silently_reset(self):
        with self.assertRaisesRegex(RuntimeError, "has no Adam state"):
            runner._required_optimizer_state(
                {"lod_opt_state": None},
                Path("/tmp/missing_optimizer.lod.pt"),
            )
        state = {"state": {}, "param_groups": []}
        self.assertIs(
            runner._required_optimizer_state(
                {"lod_opt_state": state},
                Path("/tmp/with_optimizer.lod.pt"),
            ),
            state,
        )

    def test_missing_previous_rng_state_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / runner.TRAIN_STATE_NAME
            with self.assertRaisesRegex(RuntimeError, "sampling-RNG continuation"):
                runner._required_train_state(
                    path,
                    reason="episode 2 sampling-RNG continuation",
                )

    def test_orphan_checkpoint_without_train_state_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=tmp,
                steps=runner.STEPS_PER_EPISODE,
                force=False,
                probe=False,
            )
            train_dir = Path(tmp) / "episode01" / "train"
            train_dir.mkdir(parents=True)
            (train_dir / "l1_step00250.lod.pt").write_bytes(b"orphan")
            with (
                patch.object(runner, "_validate_pool", return_value=[]),
                self.assertRaisesRegex(RuntimeError, "different RNG/Adam trajectory"),
            ):
                runner.phase_train(args, 1, Path(tmp) / "parent.pth")

    def test_preflight_refuses_changed_locked_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "chkpnt30000.pth"
            checkpoint.write_bytes(b"stage1")
            args = SimpleNamespace(
                output_dir=str(root / "run"),
                start_checkpoint=str(checkpoint),
                generation_seed=runner.GENERATION_SEED,
                training_seed=runner.TRAINING_SEED,
                force=False,
            )
            runner.phase_preflight(args)
            protocol = Path(args.output_dir) / "PROTOCOL.json"
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["training"]["mix_ratio"] = 0.5
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SystemExit):
                runner.phase_preflight(args)

    def test_completed_train_resume_still_finishes_eval_and_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=tmp, force=False)
            train_dir = Path(tmp) / "episode01" / "train"
            train_dir.mkdir(parents=True)
            (train_dir / "TRAIN.json").write_text("{}\n", encoding="utf-8")
            checkpoint = train_dir / "l1_final.lod.pt"
            checkpoint.write_bytes(b"lod")
            trained = {
                "checkpoint": str(checkpoint),
                "n_targets": runner.N_SUPERVISION,
            }
            evaluation = {"episode": 1}
            parent = Path(tmp) / "stage1.pth"
            parent.write_bytes(b"parent")

            with (
                patch.object(runner, "_parent_for_episode", return_value=parent),
                patch.object(runner, "phase_supervise") as supervise,
                patch.object(runner, "phase_flowedit") as flowedit,
                patch.object(runner, "phase_train", return_value=trained),
                patch.object(runner, "phase_evaluate", return_value=evaluation) as evaluate,
            ):
                result = runner.run_episode(args, 1)

            supervise.assert_not_called()
            flowedit.assert_not_called()
            evaluate.assert_called_once()
            self.assertEqual(result["eval"], evaluation)
            chain = json.loads((Path(tmp) / "CHAIN.json").read_text(encoding="utf-8"))
            self.assertEqual(chain["episodes"][0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
