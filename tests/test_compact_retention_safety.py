import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.compact_retention import finalize_episode, retire_prior_iteration


class CompactRetentionSafetyTests(unittest.TestCase):
    def make_episode(self, root, indices=(0, 1, 2, 3)):
        episode = root / 'idu' / 'episode_00_e85_r300'
        episode.mkdir(parents=True)
        (episode / 'episode_meta.json').write_text(json.dumps({'n_images': 4}))
        for folder in ('render', 'render_refine', 'render_after_train'):
            (episode / folder).mkdir()
            for index in indices:
                Image.new('RGB', (8, 8), (index * 20, 40, 60)).save(episode / folder / f'{index:05d}.png')
        (episode / 'geometry').mkdir()
        (episode / 'geometry' / 'scratch.npy').write_bytes(b'consumed geometry')
        return episode

    def compact(self, root):
        return finalize_episode(root, episode_dir_name='episode_00_e85_r300',
                                episode_idx=0, elevation=85.0, radius=300.0)

    def test_missing_sample_is_not_hidden_by_an_extra_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = self.make_episode(root, (0, 1, 2, 9))
            report = self.compact(root)
            self.assertEqual(report['status'], 'refused_incomplete')
            self.assertTrue((episode / 'geometry' / 'scratch.npy').is_file())
            self.assertTrue((episode / 'render_refine' / '00009.png').is_file())

    def test_repeated_compaction_preserves_effects_and_missing_effects_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = self.make_episode(root)
            first = self.compact(root)
            panel = Path(first['panels'][0]['path'])
            before = panel.read_bytes()
            self.assertFalse((episode / 'geometry').exists())
            second = self.compact(root)
            self.assertEqual(second['status'], 'completed')
            self.assertEqual(panel.read_bytes(), before)
            with Image.open(episode / 'retained_views' / '00003_render_refine.png') as image:
                self.assertEqual(image.getpixel((0, 0)), (60, 40, 60))
            panel.unlink()
            with self.assertRaises(RuntimeError):
                self.compact(root)

    def test_retirement_never_deletes_the_current_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'chkpnt40000.pth'
            checkpoint.write_bytes(b'current model')
            ply = root / 'point_cloud' / 'iteration_40000' / 'point_cloud.ply'
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b'current filter')
            retire_prior_iteration(root, 40000, 40000, protected_below=30001)
            self.assertEqual(checkpoint.read_bytes(), b'current model')
            self.assertEqual(ply.read_bytes(), b'current filter')
