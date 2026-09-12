"""Regression checks for the standalone Skyfall depth adapter."""
import numpy as np
import pytest

from prf.gs_depth import render_training_depth_maps
from prf.io_gs import load_gaussians_ply
from tests.test_prf import _nadir_view, _plane_lattice
from tests.test_prf_viewer_invariance import _write_gs_ply


def test_legacy_cache_can_be_used_without_source_ply(tmp_path):
    view = _nadir_view(size=8)
    alpha = np.full((8, 8), 0.5, dtype=np.float32)
    np.savez(tmp_path / 'nadir.npz', depth=alpha * 100, alpha=alpha)
    maps = render_training_depth_maps(tmp_path / 'absent.ply', [view], cache_dir=tmp_path)
    np.testing.assert_allclose(maps['nadir'][0], 100)
    with np.load(tmp_path / 'nadir.npz') as cache:
        assert cache['depth_kind'][0] == 'expected'
    again = render_training_depth_maps(tmp_path / 'absent.ply', [view], cache_dir=tmp_path)
    np.testing.assert_array_equal(again['nadir'][0], maps['nadir'][0])


def test_truncated_ply_header_fails(tmp_path):
    path = tmp_path / 'truncated.ply'
    path.write_bytes(b'ply\nformat binary_little_endian 1.0\n')
    with pytest.raises(ValueError, match='end_header'):
        load_gaussians_ply(path)


def test_gpu_depth_of_plane_and_cache_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('gsplat')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for gsplat rasterization')
    scene = _plane_lattice(spacing=0.5, sigma=0.3, n=12)
    path = tmp_path / 'plane.ply'
    _write_gs_ply(path, scene)
    view = _nadir_view(height=10, focal=50, size=64)
    cache_dir = tmp_path / 'cache'
    depth, alpha = render_training_depth_maps(path, [view], cache_dir=cache_dir)['nadir']
    assert depth.shape == alpha.shape == (64, 64)
    visible = alpha > 0.5
    assert visible.any()
    np.testing.assert_allclose(depth[visible], 10, atol=1e-4)
    cached = render_training_depth_maps(path, [view], cache_dir=cache_dir)['nadir']
    np.testing.assert_array_equal(cached[0], depth)
    np.testing.assert_array_equal(cached[1], alpha)
