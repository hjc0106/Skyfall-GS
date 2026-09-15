import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import archive_dataset_results as archive


class LocalTransport:
    def run(self, command, *, check=True):
        return subprocess.run(command, shell=True, capture_output=True, check=check)


class CleanupAccountingTests(unittest.TestCase):
    def test_repeated_cleanup_counts_only_paths_actually_removed(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / 'remote'
            source.mkdir()
            model = source / "checkpoint's file.pth"
            model.write_bytes(b'model')
            scratch = source / 'scratch'
            scratch.mkdir()
            (scratch / 'payload').write_bytes(b'cache')
            dangling = source / 'dangling'
            dangling.symlink_to(source / 'missing')
            candidates = [
                {'path': str(model), 'bytes': 5, 'gate': 'archive_verified'},
                {'path': str(scratch), 'bytes': 5, 'gate': 'archive_verified', 'directory': True},
                {'path': str(dangling), 'bytes': 0, 'gate': 'archive_verified'},
            ]
            destination = root / 'archive'
            manifest = SimpleNamespace(archive_root=destination, archive_dir=lambda *_: destination)
            gate = {'satisfied': True, 'evaluation_status': 'completed', 'model_load_verified': True,
                    'evaluation_status_path': str(destination / 'evaluation_status.json')}
            replacements = {
                'evaluation_gate': gate,
                'build_selection': [{'source': str(source)}],
                'stage_cleanup_candidates': candidates,
                'approved_scratch_candidates': [],
                'approved_extra_candidates': [],
                'load_cleanup_policy': {'protect_paths': []},
                'scene_evaluation_gates': {'stage1': gate, 'stage2': gate},
                'identity_gate': {'satisfied': True},
                'legacy_parent_guard': None,
                'cleanup_gate_state': (True, []),
                'filesystem_usage': {'used_bytes': None},
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(archive, name, return_value=value))
            destination.mkdir()
            (destination / 'archive_status.json').write_text(json.dumps({'status': 'verified'}))
            first = archive.run_cleanup(manifest, LocalTransport(), 'scene', 'stage2', execute=True)
            self.assertEqual(set(first['deleted']), {str(model), str(scratch), str(dangling)})
            self.assertEqual(first['deleted_bytes'], 10)
            self.assertFalse(model.exists())
            self.assertFalse(scratch.exists())
            self.assertFalse(dangling.is_symlink())
            second = archive.run_cleanup(manifest, LocalTransport(), 'scene', 'stage2', execute=True)
            self.assertEqual(second['deleted'], [])
            self.assertEqual(second['deleted_bytes'], 0)
            self.assertEqual(second['ledger']['deleted_total'], 10)
