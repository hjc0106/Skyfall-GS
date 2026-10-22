#!/usr/bin/env python3
"""Evaluate saved Stage 1 geometry and a limited number of held-out RGB views."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from arguments import ModelParams, PipelineParams


def model_configuration(model):
    new = model / 'stage1_config.json'
    if new.is_file():
        return json.loads(new.read_text())['dataset']
    # Read legacy Namespace without executing arbitrary cfg_args code.
    node = ast.parse((model / 'cfg_args').read_text(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace':
        raise ValueError('Unsupported legacy cfg_args')
    return {arg.arg: ast.literal_eval(arg.value) for arg in node.keywords}


def common_grid_reports(predictions, prior):
    from evaluate_dsm_geometry import height_metrics
    arrays = {key: np.load(path)['pred'] for key, path in predictions.items()}
    if any(array.shape != prior['height_world'].shape for array in arrays.values()):
        raise ValueError('Prediction shapes do not match DSM')
    mask = prior['valid'].astype(bool).copy()
    for array in arrays.values():
        mask &= np.isfinite(array)
    holdout = prior['holdout'].astype(bool)
    return {'protocol': 'controlled_same_dsm', 'common_cells': int(mask.sum()),
            'methods': {key: {'full': height_metrics(array, prior['height_world'], mask),
                             'train': height_metrics(array, prior['height_world'], mask & ~holdout),
                             'holdout': height_metrics(array, prior['height_world'], mask & holdout)}
                        for key, array in arrays.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Write evaluation artifacts outside the model directory')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=200)
    parser.add_argument('--rgb-views', type=int, default=2)
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    model = args.model.resolve()
    output = args.output.resolve() if args.output else model
    output.mkdir(parents=True, exist_ok=True)
    from evaluate_dsm_geometry import (load_ply_xyz_opacity, raster_max_z, height_metrics,
                                      relative_error, floater_ratio)
    from utils.dsm_prior import load_npz
    prior = load_npz(args.prior)
    xyz, alpha = load_ply_xyz_opacity(model / 'point_cloud' / f'iteration_{args.iteration}' / 'point_cloud.ply')
    pred = raster_max_z(xyz[alpha >= .05], prior)
    valid, hold = prior['valid'].astype(bool), prior['holdout'].astype(bool)
    report = {'iteration': args.iteration, 'protocol': 'controlled_same_dsm',
              'surface_estimator': 'opacity_filtered_gaussian_centers_max_z',
              'full': height_metrics(pred, prior['height_world'], valid),
              'train': height_metrics(pred, prior['height_world'], valid & ~hold),
              'holdout': height_metrics(pred, prior['height_world'], valid & hold),
              'relative_train': relative_error(pred, prior['height_world'], valid & ~hold),
              'floaters': floater_ratio(xyz, alpha, prior),
              'coverage': float((valid & np.isfinite(pred)).sum() / max(valid.sum(), 1)),
              'rgb': [], 'config': model_configuration(model)}
    if not args.geometry_only:
        from scene import Scene, GaussianModel
        from gaussian_renderer import render
        p = argparse.ArgumentParser()
        mp, pp = ModelParams(p), PipelineParams(p)
        namespace = p.parse_args([])
        for key, value in report['config'].items():
            setattr(namespace, key, value)
        namespace.source_path, namespace.model_path = str(args.source.resolve()), str(model)
        namespace.dsm_prior = str(args.prior.resolve()) if report['config'].get('dsm_prior') else ''
        namespace.eval = True
        dataset = mp.extract(namespace)
        g = GaussianModel(dataset.sh_degree, dataset.appearance_enabled,
                          dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim)
        checkpoint = model / f'chkpnt{args.iteration}.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'RGB evaluation requires a checkpoint: {checkpoint}; use --geometry-only for PLY metrics')
        payload, _ = torch.load(checkpoint, weights_only=False)
        # main's load_ply reads filter metadata and expects geometry loaded first.
        g.load_from_checkpoints(payload)
        scene = Scene(dataset, g, load_iteration=args.iteration)
        # Recover checkpoint prior/aux fields after the optional PLY sidecar.
        g.load_from_checkpoints(payload)
        g.compute_3D_filter(cameras=scene.getTrainCameras())
        background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device='cuda')
        with torch.no_grad():
            for camera in scene.getTestCameras()[:args.rgb_views]:
                rgb = render(camera, g, pp.extract(namespace), background,
                             kernel_size=dataset.kernel_size, testing=True)['render'].clamp(0, 1)
                gt = camera.original_image.to(rgb.device)
                mask = camera.original_mask.to(rgb.device).expand_as(rgb) > 0
                if not mask.any():
                    continue
                difference = (rgb - gt)[mask]
                mse = float(difference.square().mean())
                report['rgb'].append({'image': camera.image_name, 'valid_channel_samples': int(mask.sum()),
                                      'l1': float(difference.abs().mean()), 'psnr': -10 * np.log10(max(mse, 1e-12))})
                Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
                    output / f'eval_{args.iteration}_{camera.image_name}.png')
    np.savez_compressed(output / f'dsm_pred_iter_{args.iteration}.npz', pred=pred)
    registration = args.prior.with_name('registration.json')
    report['coordinates'] = json.loads(registration.read_text()) if registration.exists() else {'status': 'not_supplied'}
    (output / f'stage1_eval_{args.iteration}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('iteration', 'protocol', 'full', 'coverage', 'rgb')}, indent=2))


if __name__ == '__main__':
    main()
