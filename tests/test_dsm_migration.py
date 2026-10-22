import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arguments import ModelParams, OptimizationParams
from utils.dsm_prior import DSMPrior, add_dsm_losses, load_npz, mix_init_with_dsm
from utils.graphics_utils import BasicPointCloud


@pytest.fixture
def prior_file(tmp_path):
    h, w, step = 8, 8, 2
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = 1
    file = tmp_path / 'dsm_prior.npz'
    np.savez(file, height_world=np.zeros((h, w), np.float32), normal=normals,
             confidence=np.ones((h, w), np.float32), valid=np.ones((h, w), bool),
             holdout=np.zeros((h, w), bool), edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
             x0=0., y0=7., gsd=1., sample_step=step, coarse_h=4, coarse_w=4)
    return file


def parameters(argv=(), sentinel=False):
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel)
    optimization = OptimizationParams(parser)
    args = parser.parse_args(argv)
    return model.extract(args), optimization.extract(args)


def test_defaults_and_negative_switch(prior_file):
    data, opt = parameters()
    assert data.dsm_prior == '' and opt.dsm_surface is True
    data, opt = parameters(['--dsm_prior', str(prior_file)])
    assert data.dsm_init and opt.dsm_surface and opt.lambda_dsm_surface == .05
    data, opt = parameters(['--dsm_prior', str(prior_file), '--no_dsm_surface'])
    assert data.dsm_init and opt.dsm_surface is False
    with pytest.raises(SystemExit):
        parameters(['--dsm_surface', '--no_dsm_surface'])
    with pytest.raises(FileNotFoundError):
        parameters(['--dsm_prior', 'absent.npz', '--no_dsm_surface'])


def test_old_render_config_defaults():
    data, _ = parameters(['-s', '.'], sentinel=True)
    assert data.dsm_prior == '' and data.dsm_init is False


def fake_gaussians(device='cpu', dtype=torch.float32):
    xyz = torch.tensor([[2., 3., .3], [3., 4., .7], [4., 2., -.2]], dtype=dtype, device=device, requires_grad=True)
    return SimpleNamespace(get_xyz=xyz, get_opacity=torch.tensor([[.4], [.6], [.5]], device=device, dtype=torch.float64),
                           dsm_confidence=torch.tensor([1., .5, 1.], device=device))


def test_loss_formula_gradients_and_switch(prior_file):
    prior = DSMPrior.load(prior_file)
    data, opt = parameters(['--dsm_prior', str(prior_file)])
    g = fake_gaussians()
    loss, logs = add_dsm_losses(g.get_xyz.sum() * 0, g, prior, opt, 1000)
    weights = torch.tensor([.4, .3, .5])
    expected = .05 * (weights * torch.log1p((g.get_xyz[:, 2] / .5)**2)).sum() / weights.sum()
    torch.testing.assert_close(loss, expected)
    gradients = torch.autograd.grad(loss, g.get_xyz, retain_graph=True)[0]
    expected_gradients = torch.autograd.grad(expected, g.get_xyz)[0]
    torch.testing.assert_close(gradients, expected_gradients)
    assert logs['dsm_active_points'] == 3 and torch.isfinite(gradients).all()
    opt.dsm_surface = False
    base = g.get_xyz.sum() * 0
    off, logs = add_dsm_losses(base, g, prior, opt, 1000)
    assert off is base and logs == {}
    assert add_dsm_losses(base, g, prior, opt, 1000, is_idu=True)[0] is base


def reference_module():
    root = os.environ.get('DSM_REFERENCE_ROOT')
    if not root:
        pytest.skip('Set DSM_REFERENCE_ROOT to run external source parity tests')
    spec = importlib.util.spec_from_file_location('reference_dsm', Path(root) / 'utils/dsm_prior.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_initializer_parity(prior_file):
    ref = reference_module()
    data, _ = parameters(['--dsm_prior', str(prior_file)])
    rng = np.random.default_rng(3)
    xyz = rng.uniform([0, 0, -1], [7, 7, 1], (80, 3))
    xyz[:5, 2] = 8
    pcd = BasicPointCloud(xyz, rng.random((80, 3)), np.zeros_like(xyz))
    a, am = mix_init_with_dsm(pcd, prior_file, [], data)
    b, bm = ref.mix_init_with_dsm(pcd, prior_file, [], data)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    for key in ('confidence', 'anchor_id', 'rotations', 'is_dsm', 'stats'):
        if isinstance(am[key], dict):
            assert am[key] == bm[key]
        else:
            np.testing.assert_array_equal(am[key], bm[key])


def test_source_surface_value_and_gradient_parity(prior_file):
    ref = reference_module()
    _, opt = parameters()
    opt.lambda_dsm_relative = 0.
    opt.lambda_dsm_normal = 0.
    prior = DSMPrior.load(prior_file)
    a, b = fake_gaussians(), fake_gaussians()
    va, _ = add_dsm_losses(a.get_xyz.sum() * 0, a, prior, opt, 1000)
    vb, _ = ref.add_dsm_losses(b.get_xyz.sum() * 0, b, prior, opt, 1000)
    torch.testing.assert_close(va, vb, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(va, a.get_xyz)[0], torch.autograd.grad(vb, b.get_xyz)[0], rtol=0, atol=0)


def test_invalid_prior_rejected(tmp_path):
    p = tmp_path / 'bad.npz'
    np.savez(p, height_world=np.ones((2, 2)))
    with pytest.raises(ValueError, match='Missing DSM'):
        load_npz(p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Gaussian model requires CUDA')
def test_model_topology_checkpoint_and_reset(prior_file, tmp_path):
    from scene.gaussian_model import GaussianModel
    from utils.dsm_checkpoint import capture_stage1, restore_configuration, restore_aux
    data, opt = parameters(['--dsm_prior', str(prior_file)])
    xyz = np.array([[1., 1., 0.], [1.5, 1., 0.], [2., 1., 0.], [2.5, 1., 0.]])
    pcd = BasicPointCloud(xyz, np.full((4, 3), .5), np.zeros((4, 3)))
    mixed, meta = mix_init_with_dsm(pcd, prior_file, [], data)
    g = GaussianModel(1, False, 4, 32)
    g.create_from_pcd(mixed, 1., meta)
    g.training_setup(opt, num_train_cameras=3)
    g.filter_3D = torch.full((len(g.get_xyz), 1), .01, device='cuda')
    sum(parameter.sum() for group in g.optimizer.param_groups for parameter in group['params']).backward()
    g.optimizer.step()
    g.optimizer.zero_grad(set_to_none=True)
    n = len(g.get_xyz)
    g.densify_and_clone(torch.ones(n, 1, device='cuda'), .1, torch.zeros(n, 1, device='cuda'), 100., 1000.)
    assert len(g.get_xyz) == 2 * n
    g.assert_dsm_alignment()
    n = len(g.get_xyz)
    g.densify_and_split(torch.ones(n, 1, device='cuda'), .1, torch.zeros(n, 1, device='cuda'), 100., 1.)
    assert len(g.get_xyz) == 2 * n
    g.assert_dsm_alignment()
    prune = torch.zeros(len(g.get_xyz), device='cuda', dtype=torch.bool)
    prune[::3] = True
    g.prune_points(prune)
    g.filter_3D = torch.full((len(g.get_xyz), 1), .01, device='cuda')
    g.reset_opacity()
    prior = DSMPrior.load(prior_file, device='cuda')
    loss, _ = add_dsm_losses(g.get_xyz.sum() * 0, g, prior, opt, 1000)
    loss.backward()
    assert torch.isfinite(g.get_xyz.grad).all()
    opt.densify_from_iter = 50
    payload = capture_stage1(g, data, opt, {'test': True})
    path = tmp_path / 'checkpoint.pth'
    torch.save((payload, 100), path)
    loaded, iteration = torch.load(path, weights_only=False)
    restored = GaussianModel(1, False, 4, 32)
    assert restore_configuration(loaded, data, opt)
    assert opt.densify_from_iter == 50 and iteration == 100
    restored.restore(loaded, opt)
    restore_aux(loaded, restored)
    assert restored.xyz_scheduler_args(101) == g.xyz_scheduler_args(101)
    torch.testing.assert_close(restored.get_xyz, g.get_xyz)
    torch.testing.assert_close(restored.get_rotation, g.get_rotation, rtol=0, atol=0)
    torch.testing.assert_close(restored.dsm_confidence, g.dsm_confidence)
    restored.assert_dsm_alignment()
    legacy = GaussianModel(1, False, 4, 32)
    legacy.restore(g.capture()[:15], opt)
    assert (legacy.dsm_confidence == 0).all()
    old_dsm = GaussianModel(1, False, 4, 32)
    old_dsm.restore(g.capture(), opt)
    torch.testing.assert_close(old_dsm.dsm_confidence, g.dsm_confidence)


def test_train_import_does_not_import_model_backends():
    import subprocess, sys
    code = "import train,sys; assert not any(n.startswith(('submodules.MoGe','submodules.FlowEdit')) for n in sys.modules); assert train._moge_standalone is None"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_runner_refuses_unverified_coordinates(prior_file, tmp_path):
    import subprocess, sys, json
    root = Path(__file__).resolve().parents[1]
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'accepted': False, 'ready_for_training': False}))
    result = subprocess.run([sys.executable, str(root / 'scripts/run_stage1_comparison.py'),
        '--source', str(tmp_path), '--prior', str(prior_file), '--output', str(tmp_path / 'out'),
        '--coordinate-report', str(gate), '--smoke'], capture_output=True, text=True)
    assert result.returncode == 2
    report = json.loads((tmp_path / 'out/smoke_report.json').read_text())
    assert not report['training_started'] and not report['accepted']


def test_comparison_commands_only_B_disables_surface(prior_file, tmp_path):
    from scripts.run_stage1_comparison import commands
    args = SimpleNamespace(profile='stage1_dsm', smoke=True, iterations=None, source=tmp_path,
                           output=tmp_path / 'out', prior=prior_file, seed=3, port=6218, max_init_points=100)
    plan, steps = commands(args)
    assert steps == 200
    assert '--dsm_prior' not in plan['A'][0]
    assert '--no_dsm_surface' in plan['B'][0]
    assert '--no_dsm_surface' not in plan['C'][0]
    assert '--stop_after_iteration' in plan['C'][0]
    assert '--start_checkpoint' in plan['C'][1]
    assert plan['B'][0][plan['B'][0].index('--seed') + 1] == '3'


def test_common_grid_uses_identical_cells(prior_file, tmp_path):
    from scripts.evaluate_stage1 import common_grid_reports
    a = np.full((8, 8), 1.)
    b = np.full((8, 8), 2.)
    a[0] = np.nan
    b[:, 0] = np.nan
    np.savez(tmp_path / 'a.npz', pred=a)
    np.savez(tmp_path / 'b.npz', pred=b)
    report = common_grid_reports({'A': tmp_path / 'a.npz', 'B': tmp_path / 'b.npz'}, load_npz(prior_file))
    assert report['common_cells'] == 49
    assert report['methods']['A']['full']['n'] == report['methods']['B']['full']['n'] == 49
    assert report['methods']['A']['full']['mae'] == 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Synthetic render requires CUDA')
def test_checkpoint_restored_model_renders_and_takes_optimizer_step(prior_file):
    from scene.gaussian_model import GaussianModel
    from utils.dsm_checkpoint import capture_stage1, restore_aux
    from priors.coordinates import make_training_camera
    from gaussian_renderer import render
    data, opt = parameters(['--dsm_prior', str(prior_file), '--appearance_enabled'])
    points = np.array([[2., 3., 0.], [3., 3., 0.], [4., 3., 0.], [3., 4., 0.]])
    pcd = BasicPointCloud(points, np.full((4, 3), .7), np.zeros((4, 3)))
    cloud, meta = mix_init_with_dsm(pcd, prior_file, [], data)
    g = GaussianModel(1, True, 4, 32)
    g.create_from_pcd(cloud, 1., meta)
    g.training_setup(opt, num_train_cameras=1)
    g.filter_3D = torch.full((len(g.get_xyz), 1), .01, device='cuda')
    payload = capture_stage1(g, data, opt, {})
    restored = GaussianModel(1, True, 4, 32)
    restored.restore(payload, opt)
    restore_aux(payload, restored)
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [3., 3., 10.]
    f = dict(transform_matrix=pose.tolist(), fl_x=100., fl_y=100., cx=32., cy=32., file_path='synthetic.png')
    camera = make_training_camera(f, 64, 64, 1)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    package = render(camera, restored, pipe, torch.zeros(3, device='cuda'), kernel_size=.1)
    assert package['render'].shape == (3, 64, 64)
    assert torch.isfinite(package['render']).all()
    target = torch.linspace(0., 1., 64, device='cuda').view(1, 1, 64)
    assert package['visibility_filter'].any()
    assert package['render'].detach().max() > 0
    loss = (package['render'] - target).square().mean()
    loss.backward()
    assert torch.isfinite(restored.get_xyz.grad).all()
    before = restored.get_xyz.detach().clone()
    restored.update_learning_rate(101)
    restored.optimizer.step()
    assert not torch.equal(before, restored.get_xyz)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Model module requires CUDA installation')
def test_flat_dsm_appearance_embedding_is_finite():
    from scene.gaussian_model import _get_fourier_features
    xyz = np.array([[0., 0., 0.], [1., 1., 0.], [2., 2., 0.]])
    assert torch.isfinite(_get_fourier_features(xyz, 4)).all()


def test_coordinate_validator_produces_bound_gate_for_verified_tracks(prior_file, tmp_path):
    import json
    from scripts.validate_dsm_coordinates import validate
    from scripts.run_stage1_comparison import check_coordinate_gate
    source = tmp_path / 'source'
    source.mkdir()
    xyz = np.array([[x, y, 0.] for x in (1., 2., 3., 4.) for y in (1., 2., 3., 4.)])
    frames, lines = [], []
    for i, shift in enumerate([-.1, 0., .1]):
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1., -1., -1.])
        pose[:3, 3] = [3.5 + shift, 3.5, 10.]
        f = dict(transform_matrix=pose.tolist(), fl_x=200., fl_y=200., cx=320., cy=240., file_path=f'v{i}.png')
        frames.append(f)
        camera = (xyz - pose[:3, 3]) @ pose[:3, :3]
        pixels = camera[:, :2] / camera[:, 2:3] * 200 + [320, 240]
        lines.append(f'{i + 1} 0 1 0 0 0 0 0 1 v{i}.png\n')
        lines.append(' '.join(f'{u} {v} {j+1}' for j, (u, v) in enumerate(pixels)) + '\n')
    (source / 'transforms_train.json').write_text(json.dumps(dict(w=640, h=480, frames=frames)))
    (source / 'points3D.txt').write_text(''.join(
        f'{j+1} {x} {y} {z} 100 100 100 0 ' + ' '.join(f'{i+1} {j}' for i in range(3)) + '\n'
        for j, (x, y, z) in enumerate(xyz)))
    observations = tmp_path / 'images.txt'
    observations.write_text(''.join(lines))
    prior_file.with_name('registration.json').write_text(json.dumps({'transform': {
        'E': 'E0 + x', 'N': 'N0 + y', 'H': 'h0 + z + dz', 'E0': 100., 'N0': 200., 'h0': 0., 'dz': 0.}}))
    args = SimpleNamespace(source=source, prior=prior_file, observations=observations, resolution=4, device='cpu', max_error=2.)
    report = validate(args)
    assert report['accepted'] and report['ready_for_training']
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(report))
    check_coordinate_gate(path, source, prior_file)
    (source / 'points3D.txt').write_text('# changed dataset\n')
    with pytest.raises(ValueError, match='source hash'):
        check_coordinate_gate(path, source, prior_file)


def test_explicit_surface_switch_overrides_resume_configuration(prior_file):
    from utils.dsm_checkpoint import restore_configuration, file_hash
    data, opt = parameters(['--dsm_prior', str(prior_file), '--no_dsm_surface'])
    saved_opt = vars(opt).copy()
    saved_opt['dsm_surface'] = True
    payload = {'stage1_dsm_version': 1, 'optimization': saved_opt,
               'dataset': vars(data).copy(), 'prior_sha256': file_hash(prior_file)}
    assert restore_configuration(payload, data, opt, {'dsm_surface': False})
    assert opt.dsm_surface is False
    assert restore_configuration(payload, data, opt)
    assert opt.dsm_surface is True
