"""Published fused Gaussian models must not be blurred a second time on load."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement
import torch

import render_video_from_ply as renderer


class FusedPlyRenderTests(unittest.TestCase):
    def test_import_preserves_effective_scale_and_opacity(self):
        fields = ['x', 'y', 'z'] + [f'f_dc_{i}' for i in range(3)]
        fields += [f'f_rest_{i}' for i in range(9)]
        fields += ['opacity'] + [f'scale_{i}' for i in range(3)]
        fields += [f'rot_{i}' for i in range(4)]
        vertices = np.zeros(2, dtype=[(name, 'f4') for name in fields])
        scales = np.array([[.1, .2, .3], [.04, .7, 2.]], dtype=np.float32)
        opacity = np.array([[.8], [.25]], dtype=np.float32)
        vertices['z'] = 2
        vertices['rot_0'] = 1
        vertices['opacity'] = np.log(opacity[:, 0] / (1 - opacity[:, 0]))
        for axis in range(3):
            vertices[f'scale_{axis}'] = np.log(scales[:, axis])
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            # Exercise the real importer and physical-value accessors without
            # allocating a GPU merely to verify file semantics.
            kwargs['device'] = 'cpu'
            return real_tensor(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'published_fused.ply'
            PlyData([PlyElement.describe(vertices, 'vertex')]).write(path)
            with patch.object(renderer.torch, 'tensor', side_effect=cpu_tensor):
                model = renderer.load_ply_gaussians(str(path))
            torch.testing.assert_close(model.get_scaling_with_3D_filter,
                                       real_tensor(scales), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(model.get_opacity_with_3D_filter,
                                       real_tensor(opacity), rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
