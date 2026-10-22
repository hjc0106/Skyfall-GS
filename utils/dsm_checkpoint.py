"""Versioned Stage 1 state, retaining the outer (model, iteration) checkpoint API."""
from __future__ import annotations
import hashlib
import random
from pathlib import Path
import numpy as np
import torch

VERSION = 1


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def restore_configuration(payload, dataset, opt, explicit_overrides=None):
    if not isinstance(payload, dict) or payload.get('stage1_dsm_version') != VERSION:
        return False
    requested_iterations = opt.iterations
    for key, value in payload['optimization'].items():
        setattr(opt, key, value)
    opt.iterations = requested_iterations
    for key, value in (explicit_overrides or {}).items():
        if key not in ('dsm_surface', 'lambda_dsm_surface'):
            raise ValueError('Unsupported resume override: ' + key)
        setattr(opt, key, value)
    saved = payload['dataset']
    for name, expected in payload.get('source_hashes', {}).items():
        if file_hash(Path(dataset.source_path) / name) != expected:
            raise ValueError('Resume dataset differs from checkpoint: ' + name)
    # Permit relocating the prior on another server only when the content matches.
    prior = getattr(dataset, 'dsm_prior', '') or saved.get('dsm_prior', '')
    if file_hash(prior) != payload['prior_sha256']:
        raise ValueError('Resume DSM prior content differs from checkpoint')
    for key, value in saved.items():
        if key not in ('source_path', 'model_path', 'dsm_prior'):
            setattr(dataset, key, value)
    dataset.dsm_prior = prior
    return True


def capture_stage1(gaussians, dataset, opt, runtime):
    prior = getattr(dataset, 'dsm_prior', '')
    registration = Path(prior).with_name('registration.json') if prior else None
    return {
        'stage1_dsm_version': VERSION, 'model_state': gaussians.capture(),
        'dataset': vars(dataset).copy(), 'optimization': vars(opt).copy(),
        'prior_sha256': file_hash(prior),
        'source_hashes': {name: file_hash(Path(dataset.source_path) / name)
                          for name in ('points3D.txt', 'transforms_train.json')
                          if (Path(dataset.source_path) / name).exists()},
        'coordinates': registration.read_text() if registration and registration.is_file() else None,
        'clone_conf_scale': gaussians.dsm_clone_conf_scale,
        'model_aux': {name: getattr(gaussians, name) for name in
                      ('xyz_gradient_accum_abs', 'xyz_gradient_accum_abs_max', 'filter_3D')},
        'runtime': runtime,
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
    }


def restore_aux(payload, gaussians):
    for key, value in payload['model_aux'].items():
        setattr(gaussians, key, value.to(gaussians.get_xyz.device))
    gaussians.dsm_clone_conf_scale = payload['clone_conf_scale']
    from utils.general_utils import get_expon_lr_func
    opt = payload['optimization']
    gaussians.xyz_scheduler_args = get_expon_lr_func(
        lr_init=opt['position_lr_init'] * gaussians.spatial_lr_scale,
        lr_final=opt['position_lr_final'] * gaussians.spatial_lr_scale,
        lr_delay_mult=opt['position_lr_delay_mult'], max_steps=opt['position_lr_max_steps'])
    gaussians.assert_dsm_alignment()
    rng = payload['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
