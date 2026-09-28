"""Concurrent scene readers must see complete committed point-cloud snapshots."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData

from scene.dataset_readers import fetchPly, storePly


class PointCloudCacheTests(unittest.TestCase):
    def test_replacement_keeps_existing_reader_snapshot_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'points3D.ply'
            original = np.arange(8192 * 3, dtype=np.float32).reshape(-1, 3)
            storePly(path, original, np.full_like(original, 127))
            previous_bytes = path.read_bytes()
            replacement = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
            # An open reader must retain its old complete file while another
            # process publishes a smaller replacement. In-place truncation
            # breaks this and can SIGBUS a plyfile memory-mapped reader.
            with path.open('rb') as reader:
                storePly(path, replacement, np.full_like(replacement, 64))
                self.assertEqual(reader.read(), previous_bytes)
            np.testing.assert_array_equal(fetchPly(path).points, replacement)

    def test_failed_write_preserves_last_committed_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'points3D.ply'
            original = np.array([[1., 2., 3.]], dtype=np.float32)
            storePly(path, original, np.full_like(original, 127))
            previous_bytes = path.read_bytes()

            def fail_after_partial_write(_self, destination):
                if hasattr(destination, 'write'):
                    destination.write(b'partial')
                else:
                    Path(destination).write_bytes(b'partial')
                raise OSError('simulated interrupted write')

            with patch.object(PlyData, 'write', fail_after_partial_write):
                with self.assertRaises(OSError):
                    storePly(path, original * 2, np.full_like(original, 64))
            self.assertEqual(path.read_bytes(), previous_bytes)
            np.testing.assert_array_equal(fetchPly(path).points, original)


if __name__ == '__main__':
    unittest.main()
