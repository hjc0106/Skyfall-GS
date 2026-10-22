import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from scipy.spatial.transform import Rotation

from priors.coordinates import (
    camera_matrices, current_reader_transform, estimate_similarity,
    make_training_camera, project_frame, project_matrices, similarity,
    transform_camera, transform_points,
)
from priors.dsm import read_dsm, sample_dsm
from scripts.prepare_dsm_prior import inventory_rpc, load_observations, read_tracks


def frame(c2w=None):
    return {'fl_x': 800., 'fl_y': 810., 'cx': 320., 'cy': 240.,
            'transform_matrix': (np.eye(4) if c2w is None else c2w).tolist(),
            'file_path': './images/example.png'}


def test_missing_reader_R_T_is_identity():
    points = np.array([[100., 20., 5.], [-3., 8., 17.]])
    np.testing.assert_array_equal(current_reader_transform({}, points), np.eye(4))


def test_reader_normalization_matches_existing_formula():
    points = np.array([[100., 20., 5.], [-3., 8., 17.], [12., 10., 8.]])
    rotation = Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
    contents = {'R': rotation.tolist(), 'T': [2., 3., 4.]}
    expected = points @ rotation.T - contents['T']
    expected *= 256 / np.percentile(np.linalg.norm(expected, axis=1), 99)
    expected -= [0, 0, np.percentile(expected[:, 2], 1)]
    np.testing.assert_allclose(transform_points(points, current_reader_transform(contents, points)), expected)


def test_similarity_roundtrip_and_camera_projection_invariance():
    points = np.random.default_rng(0).uniform([-1, -1, 5], [1, 1, 10], (100, 3))
    matrix = similarity(.03, Rotation.from_rotvec([.1, -.2, .3]).as_matrix(), [1., 2., 3.])
    transformed = transform_points(points, matrix)
    np.testing.assert_allclose(transform_points(transformed, np.linalg.inv(matrix)), points, atol=1e-12)
    a = frame()
    b = frame(transform_camera(np.eye(4), matrix))
    np.testing.assert_allclose(project_frame(points, a, 640, 480)[0],
                               project_frame(transformed, b, 640, 480)[0], atol=1e-3)


def test_large_camera_distance_does_not_imply_bad_projection():
    pose = np.eye(4)
    pose[:3, 3] = [100000., 0., -1000000.]
    f = frame(pose)
    f.update(fl_x=1000000., fl_y=1000000., cx=101024., cy=1024.)
    pixels, depth = project_frame(np.array([[0., 0., 0.]]), f, 2048, 2048, resolution=4)
    np.testing.assert_allclose(pixels[0], [255.5, 255.5], atol=.01)
    assert depth[0] > 100  # Camera.zfar is not the CUDA rasterizer's far clipping rule.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual Camera uses CUDA matrices')
def test_projection_matches_actual_training_Camera():
    points = np.random.default_rng(1).uniform([-1, -1, 5], [1, 1, 10], (100, 3))
    f = frame()
    camera = make_training_camera(f, 640, 480, 4)
    actual, depth = project_matrices(points, camera.world_view_transform, camera.full_proj_transform,
                                     (camera.image_width, camera.image_height))
    predicted, expected_depth = project_frame(points, f, 640, 480, 4)
    np.testing.assert_allclose(actual, predicted, atol=2e-5)
    np.testing.assert_allclose(depth, expected_depth, atol=2e-5)


def test_estimate_similarity_from_verified_multiview_observations():
    rng = np.random.default_rng(10)
    cloud = rng.uniform([-2, -2, 5], [2, 2, 10], (60, 3))
    true_matrix = similarity(1.2, Rotation.from_rotvec([.02, -.01, .03]).as_matrix(), [.3, -.1, .2])
    frames = []
    points, measured, indices = [], [], []
    for index, x in enumerate([-2., 0., 2.]):
        pose = np.eye(4)
        pose[0, 3] = x
        f = frame(pose)
        frames.append(f)
        world = transform_points(cloud, true_matrix)
        camera = world - pose[:3, 3]
        pixels = camera[:, :2] / camera[:, 2:3] * [f['fl_x'], f['fl_y']] + [f['cx'], f['cy']]
        points.extend(cloud)
        measured.extend(pixels)
        indices.extend([index] * len(cloud))
    estimated, info = estimate_similarity(np.array(points), np.array(measured), np.array(indices), frames)
    assert info['success']
    np.testing.assert_allclose(transform_points(cloud, estimated), transform_points(cloud, true_matrix), atol=1e-5)


def test_similarity_rejects_reflection_and_insufficient_observations():
    with pytest.raises(ValueError):
        similarity(1, np.diag([-1, 1, 1]), [0, 0, 0])
    with pytest.raises(ValueError):
        estimate_similarity(np.zeros((5, 3)), np.zeros((5, 2)), np.zeros(5, dtype=int), [frame()])


def test_dsm_sidecar_grid_centers_and_nodata(tmp_path):
    heights = np.arange(16, dtype=np.float32).reshape(4, 4)
    heights[0, 0] = -9999
    tifffile.imwrite(tmp_path / 'dsm.tif', heights)
    (tmp_path / 'dsm.txt').write_text('100\n200\n4\n0.5\n')
    data, valid, info = read_dsm(tmp_path / 'dsm.tif', tmp_path / 'dsm.txt', nodata=-9999)
    assert info['geotiff_tags'] == {}
    assert info['vertical_datum'] == 'unknown'
    assert info['epsg'] == 32617
    samples = sample_dsm(data, valid, info, spacing_m=.5)
    assert samples.shape == (15, 3)
    np.testing.assert_allclose(samples[0], [100.75, 201.75, 1.])
    assert len(sample_dsm(data, valid, info, spacing_m=.5, max_points=3)) == 3


def test_dsm_dimension_mismatch_is_an_error(tmp_path):
    tifffile.imwrite(tmp_path / 'dsm.tif', np.zeros((3, 4), dtype=np.float32))
    (tmp_path / 'dsm.txt').write_text('100 200 4 .5')
    with pytest.raises(ValueError, match='dimensions'):
        read_dsm(tmp_path / 'dsm.tif', tmp_path / 'dsm.txt')


def test_observations_require_exact_point_id_and_feature_index(tmp_path):
    (tmp_path / 'points3D.txt').write_text('42 1 2 8 100 100 100 .5 7 0\n')
    ids, xyz, _, _, tracks = read_tracks(tmp_path / 'points3D.txt')
    obs = tmp_path / 'images.txt'
    obs.write_text('7 1 0 0 0 0 0 0 1 example.png\n420 440 42\n')
    points, pixels, indices = load_observations(obs, ids, xyz, tracks, [frame()])
    np.testing.assert_equal(points, xyz)
    np.testing.assert_equal(pixels, [[420, 440]])
    np.testing.assert_equal(indices, [0])
    obs.write_text('7 1 0 0 0 0 0 0 1 example.png\n420 440 99\n')
    with pytest.raises(ValueError, match='Point ID mismatch'):
        load_observations(obs, ids, xyz, tracks, [frame()])


def test_rpc_crop_inventory_does_not_claim_track_identity(tmp_path):
    (tmp_path / 'example.json').write_text(json.dumps({'img': 'example.tif', 'width': 830, 'height': 832,
         'keypoints': {'2d_coordinates': [[1., 2.]], 'pts3d_indices': [0]}}))
    np.save(tmp_path / 'pts3d.npy', [[700000., -5000000., 3000000.]])
    report = inventory_rpc(tmp_path, [frame()], [np.array([[1, 32441]])])
    assert report['associated_point_count'] == 1
    assert report['colmap_feature_index_max'] == 32441
    assert report['usable_as_current_sfm_tracks'] is False


def test_cli_fails_closed_and_writes_no_training_ready_artifact(tmp_path):
    import subprocess
    import sys
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'points3D.txt').write_text('42 1 2 8 100 100 100 .5 7 0\n')
    (source / 'transforms_train.json').write_text(json.dumps({'w': 640, 'h': 480, 'frames': [frame()]}))
    tifffile.imwrite(tmp_path / 'dsm.tif', np.zeros((4, 4), dtype=np.float32))
    (tmp_path / 'dsm.txt').write_text('100 200 4 .5')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/prepare_dsm_prior.py'),
        '--source', str(source), '--dsm', str(tmp_path / 'dsm.tif'),
        '--dsm-sidecar', str(tmp_path / 'dsm.txt'), '--output', str(tmp_path / 'out'),
        '--device', 'cpu'], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    report = json.loads((tmp_path / 'out/coordinate_report.json').read_text())
    assert report['ready_for_training'] is False
    assert report['current_reprojection_error_px'] is None
    assert 'missing_verified_training_2d_observations' in report['blockers']
    smoke = json.loads((tmp_path / 'out/smoke_report.json').read_text())
    assert smoke['training_started'] is False
    assert all(row['status'] == 'not_started' for row in smoke['abc'].values())
    assert not (tmp_path / 'out/registration.json').exists()
