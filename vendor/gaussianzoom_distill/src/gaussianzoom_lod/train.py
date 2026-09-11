"""Train L0, add a LoD, or resume; real-image and cached generated targets share one path."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from .data import Mip360Dataset
from .model import GaussianLoD
from .renderer import render
from .losses import dual_scale_loss, geometry_loss, rgb_loss


def save_image(image, path):
    pixels=(image.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def scalar_metrics(image, target):
    mse=(image-target).square().mean().item()
    from fused_ssim import fused_ssim
    return {'psnr':-10*math.log10(max(mse,1e-12)),
            'ssim':float(fused_ssim(image[None],target[None])),
            'l1':float((image-target).abs().mean())}


@torch.no_grad()
def evaluate(model, dataset, output=None, lod=True, max_level=None):
    """Held-out images only. Metrics never enter training or select checkpoint."""
    records=[]
    if output:
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for index,name in enumerate(dataset.test_names):
        lr_cam,lr,hr_cam,hr=dataset.pair(name)
        lr,hr=lr.cuda(),hr.cuda()
        hr_pkg=render(model,hr_cam,lod=lod,max_level=max_level,require_depth=False)
        lr_pkg=render(model,lr_cam,lod=lod,max_level=max_level,require_depth=False)
        records.append({'name':name,'hr':scalar_metrics(hr_pkg['image'],hr),'lr':scalar_metrics(lr_pkg['image'],lr)})
        if output and index<3:
            save_image(hr_pkg['image'],output/f'{index}_render.png')
            save_image(hr,output/f'{index}_target.png')
    if not records:
        raise ValueError('Held-out cameras required for effectiveness evaluation')
    summary={scale:{key:float(np.mean([r[scale][key] for r in records])) for key in ('psnr','ssim','l1')} for scale in ('hr','lr')}
    return {'summary':summary,'views':records}


def frozen_snapshot(model):
    return [{k:v.detach().cpu().clone() for k,v in layer.state_dict().items()} for layer in list(model.layers)[:model.active_level]]


def frozen_equal(model, snapshot):
    return all(torch.equal(old[key], value.detach().cpu()) for layer,old in zip(model.layers,snapshot) for key,value in layer.state_dict().items())


def fit(args):
    config={k:v for k,v in vars(args).items() if k!='func'}
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'checkpoint.pt').exists():
        raise FileExistsError('Use a new output directory; refusing to overwrite a completed run')
    if args.resume and args.base_checkpoint:
        raise ValueError('Choose resume or base-checkpoint, not both')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dataset=Mip360Dataset(args.dataset,lr_dir=args.lr_dir,hr_dir=args.hr_dir,
                          train_limit=args.train_views,test_limit=args.test_views,max_width=args.max_width)
    if args.resume:
        model,prior,opt_state=GaussianLoD.load(args.resume)
        if prior.get('dataset') != dataset.manifest():
            raise ValueError('Resume requires identical data manifest and split')
        if prior.get('lrs') != [args.position_lr,args.feature_lr,args.opacity_lr,args.scaling_lr,args.rotation_lr]:
            raise ValueError('Resume requires matching initial learning rates')
    elif args.base_checkpoint:
        model,prior,_=GaussianLoD.load(args.base_checkpoint)
        opt_state=None
        if prior.get('dataset',{}).get('root') != dataset.manifest()['root']:
            raise ValueError('Base checkpoint must use the same COLMAP scene coordinate frame')
        if set(prior.get('dataset',{}).get('train_names',[])) & set(dataset.test_names):
            raise ValueError('Base checkpoint trained on requested test views')
    else:
        points,colors=dataset.points,dataset.colors
        if len(points)>args.initial_points:
            indices=torch.randperm(len(points))[:args.initial_points]
            points,colors=points[indices],colors[indices]
        model=GaussianLoD.from_points(points.cuda(),colors.cuda(),dataset.cameras(),degree=args.degree,step_scale=args.step_scale)
        opt_state=None; prior={}
    nominal_scale=dataset.manifest()['pixel_scale_lr_to_hr']['nominal']
    if not math.isclose(model.step_scale,nominal_scale,rel_tol=0.02):
        raise ValueError('Model step_scale must match the paired target resolution scale')
    if prior.get('model_lod',{}).get('legacy_unfiltered',False):
        raise ValueError('Unfiltered v1 checkpoints are diagnostic-only; retrain L0 with the corrected pipeline')
    if model.topology_mode != 'tree':
        raise ValueError('This multi-level checkpoint lacks ancestry; start a new tree from filtered L0')
    # Legacy L0 has no ancestry-dependent scale history. Attach its known LR
    # calibration explicitly, after matching the saved calibration manifest.
    if not model.stage_records:
        if len(model.layers)!=1:
            raise ValueError('Legacy multi-level checkpoint has no verified scale history; restart from calibrated L0')
        current_manifest=dataset.manifest()
        old_manifest=prior.get('dataset',{})
        keys=('root','lr_dir','train_names','colmap_cameras','max_width')
        if any(old_manifest.get(k)!=current_manifest.get(k) for k in keys):
            raise ValueError('Legacy L0 calibration cannot be bound: source LR/camera manifest differs')
        if old_manifest.get('tiers',{}).get('lr')!=current_manifest.get('tiers',{}).get('lr'):
            raise ValueError('Legacy L0 LR dimensions differ from the supplied calibration')
        model.bind_base_stage(dataset.cameras(resolution='lr'))
        config['legacy_l0_stage_binding']='explicit matching saved LR/camera manifest'
    model.require_stage_records()
    previous_level=model.active_level
    if args.base_checkpoint:
        if args.densify_every <= 0:
            raise ValueError('A new LoD needs fixed-point gradient densification to create its primitives')
        model.add_level(dataset.cameras(resolution='hr'))
    active_cameras=dataset.cameras(resolution='hr' if model.active_level else 'lr')
    model._check_stage_use(active_cameras)
    reference=evaluate(model,dataset,output/'before',lod=previous_level>0,max_level=previous_level)
    base_records={cam['name']:cam for cam in model.stage_records[0]['cameras']}
    lr_cameras=dataset.cameras(resolution='lr')
    config['lr_reference_scale']=float(np.median([cam.fx/base_records[cam.name]['fx'] for cam in lr_cameras]))
    centers=torch.stack([camera.center for camera in active_cameras])
    scene_extent=float((centers-centers.mean(0)).norm(dim=1).max())*1.1
    if scene_extent<=0:
        xyz=model.tensors()['xyz'].detach()
        scene_extent=float((xyz.amax(0)-xyz.amin(0)).norm())
    if scene_extent<=0:
        raise ValueError('Cannot derive a positive scene extent for density control')
    if args.geometry_from is None:
        args.geometry_from=7000 if model.active_level==0 else 500
    config['geometry_from']=args.geometry_from
    train_names=dataset.train_names
    frozen=frozen_snapshot(model)
    initial_active={k:v.detach().cpu().clone() for k,v in model.active.state_dict().items()}
    optimizer=model.make_optimizer(args.position_lr,args.feature_lr,args.opacity_lr,args.scaling_lr,args.rotation_lr)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
        random.setstate(prior['python_rng'])
        torch.set_rng_state(prior['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in prior['cuda_rng']])
    start_step=prior.get('step',0) if args.resume else 0
    initial_count=len(model.active.xyz)
    grad_sum=torch.zeros(sum(len(layer.xyz) for layer in model.layers),device='cuda')
    grad_count=torch.zeros_like(grad_sum)
    geometry_grad_steps=0
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); log=[]
    for offset in range(args.steps):
        step=start_step+offset
        name=random.choice(train_names)
        if model.active_level:
            lr_cam,lr,hr_cam,hr=dataset.pair(name)
        else:
            lr_cam,lr=dataset.view(name,resolution='lr')
        optimizer.zero_grad(set_to_none=True)
        # Position LR is relative to scene size; caller records exact value.
        rate=args.position_lr*(0.1**(offset/max(args.steps-1,1)))
        for group in optimizer.param_groups:
            if group.get('name')=='xyz': group['lr']=rate
        camera=hr_cam if model.active_level else lr_cam
        geo_weight=args.lambda_geo if step+1>=args.geometry_from else 0.
        package=render(model,camera,lod=model.active_level>0,require_depth=geo_weight>0)
        if model.active_level:
            loss,parts=dual_scale_loss(package,camera,hr.cuda(),lr.cuda(),lambda_geo=geo_weight)
        else:
            color=rgb_loss(package['image'],lr.cuda())
            geo=geometry_loss(package,camera) if geo_weight else color*0
            loss=color+geo_weight*geo
            parts={'hr':color.detach(),'lr':color.detach()*0,'geo':geo.detach()}
        if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss at step {step}')
        loss.backward()
        # Supplement B.1: fixed parameters stay frozen, but their screen proxy
        # gradients drive the birth of new children. Do not slice them away.
        visible=package['radii']>0
        norms=package['means2d'].grad[:,:2].norm(dim=1)
        grad_sum[visible]+=norms[visible]; grad_count[visible]+=1
        if len(model.active.xyz)>0:
            if model.active.xyz.grad is not None and bool(model.active.xyz.grad.abs().sum()>0):
                geometry_grad_steps+=1
            optimizer.step()
        density={}
        bootstrap=model.active_level>0 and len(model.active.xyz)==0
        density_active=args.densify_every and (bootstrap or offset+1<args.steps*args.densify_fraction)
        scheduled=step+1>args.densify_from and args.densify_every and (step+1)%args.densify_every==0
        if density_active and (bootstrap or scheduled):
            density=model.densify(optimizer,grad_sum/grad_count.clamp_min(1),args.grad_threshold,
                                  args.max_points,active_cameras,min_opacity=args.min_opacity,
                                  scene_extent=scene_extent,percent_dense=args.percent_dense)
            grad_sum=torch.zeros(sum(len(layer.xyz) for layer in model.layers),device='cuda')
            grad_count=torch.zeros_like(grad_sum)
        if density or (step+1)%100==0:
            model.update_filter(active_cameras)
        if len(model.active.xyz)>0 and density_active and args.opacity_reset_every and (step+1)%args.opacity_reset_every==0:
            model.reset_opacity(optimizer)
        if offset%50==0 or offset==args.steps-1 or density:
            entry={'step':step+1,'loss':float(loss.detach()),'points':len(model.active.xyz),**{k:float(v) for k,v in parts.items()},'density':density}
            log.append(entry); print(json.dumps(entry),flush=True)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    model.update_filter(active_cameras)
    if not frozen_equal(model,frozen): raise AssertionError('Frozen LoD mutated')
    final_active=model.active.state_dict()
    changed=[key for key,value in final_active.items() if key in initial_active and (value.shape!=initial_active[key].shape or not torch.equal(value.detach().cpu(),initial_active[key]))]
    if not any(k in changed for k in ('xyz','log_scales','rotations','opacity_logits')) or geometry_grad_steps==0:
        raise AssertionError('No optimized primitive geometry; inspect screen-gradient threshold and training duration')
    metadata={'dataset':dataset.manifest(),'step':start_step+args.steps,'lrs':[args.position_lr,args.feature_lr,args.opacity_lr,args.scaling_lr,args.rotation_lr],
              'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
              'args':config,'target_provenance':'oracle HR training views' if not Path(args.hr_dir).is_absolute() else 'external posed HR cache',
              'geometry':'RaDe-GS expected-depth normal consistency after warmup; valid stencil mask; no PGSR multiview loss',
              'filtering':'per-layer RaDe-GS sampling filter and determinant-opacity compensation; older filters frozen',
              'reference_policy':'min visible distance/fx at creation; fallback min across cameras',
              'seed_policy':'supplement B.1 fixed-screen-gradient split/clone; no uniform copied fine layer',
              'hierarchy_policy':'fixed split/clone child; active split/clone sibling under original parent',
              'opacity_policy':'actual-stage parent/cohort intervals; recursive incoming weights; no nominal-gap assumption',
              'scene_extent':scene_extent}
    model.save(output/'checkpoint.pt',optimizer=optimizer,metadata=metadata)
    after=evaluate(model,dataset,output/'after',lod=model.active_level>0)
    loaded,_,_=GaussianLoD.load(output/'checkpoint.pt')
    camera=active_cameras[0]
    with torch.no_grad():
        a=render(model,camera,lod=model.active_level>0)['image']
        b=render(loaded,camera,lod=loaded.active_level>0)['image']
        roundtrip=bool(torch.equal(a,b))
    if not roundtrip: raise AssertionError('Checkpoint render differs after reload')
    result={'before':reference,'after':after,'frozen_unchanged':True,'checkpoint_render_equal':roundtrip,
            'active_level':model.active_level,'initial_active_count':initial_count,'final_active_count':len(model.active.xyz),
            'geometry_grad_steps':geometry_grad_steps,
            'stage_scales':[stage['scale'] for stage in model.stage_records],
            'changed_active_fields':changed,'seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'training_log':log,'configuration':config,'dataset':dataset.manifest()}
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'before':reference['summary'],'after':after['summary'],'seconds':elapsed,'output':str(output)},indent=2),flush=True)


@torch.no_grad()
def sweep(args):
    model,metadata,_=GaussianLoD.load(args.checkpoint)
    cfg=metadata['args']
    dataset=Mip360Dataset(cfg['dataset'],lr_dir=cfg['lr_dir'],hr_dir=cfg['hr_dir'],test_limit=cfg['test_views'],train_limit=cfg['train_views'],max_width=cfg['max_width'])
    camera,_,_,_=dataset.pair(dataset.test_names[args.view_index])
    if model.weight_policy=='interval':
        relative=cfg.get('lr_reference_scale',1.)
        camera=camera.resized(round(camera.width/relative),round(camera.height/relative))
        end_factor=model.stage_records[-1]['scale']
    else:
        end_factor=model.step_scale**model.active_level
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    frames=[]; records=[]; previous=None; widths=[]
    for idx,factor in enumerate(np.geomspace(1,end_factor,args.frames)):
        cam=camera.zoomed(float(factor))
        pkg=render(model,cam,lod=not args.no_lod)
        image=pkg['image']
        if not torch.isfinite(image).all(): raise FloatingPointError('nonfinite sweep image')
        data=model.tensors()
        masses=[float((data['opacity'][sl]*pkg['weights'][sl]).sum()) for sl in pkg['layer_slices']]
        delta=0. if previous is None else float((image-previous).abs().mean())
        records.append({'frame':idx,'focal_factor':float(factor),'opacity_mass':masses,'adjacent_l1':delta})
        path=out/f'{idx:04d}.png'; save_image(image,path)
        frames.append(Image.open(path).copy()); previous=image
    frames[0].save(out/'sweep.gif',save_all=True,append_images=frames[1:],duration=70,loop=0)
    selected=[frames[i] for i in np.linspace(0,len(frames)-1,8,dtype=int)]
    thumbs=[]
    for image in selected:
        image.thumbnail((320,240)); thumbs.append(image)
    strip=Image.new('RGB',(sum(im.width for im in thumbs),max(im.height for im in thumbs)))
    left=0
    for im in thumbs: strip.paste(im,(left,0)); left+=im.width
    strip.save(out/'strip.png')
    (out/'sweep.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'frames':len(frames),'max_adjacent_l1':max(r['adjacent_l1'] for r in records),'output':str(out)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit')
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--lr-dir',default='images_8');p.add_argument('--hr-dir',default='images_2')
    p.add_argument('--base-checkpoint');p.add_argument('--resume')
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--degree',type=int,default=1)
    p.add_argument('--step-scale',type=float,default=4.);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--train-views',type=int,default=0);p.add_argument('--test-views',type=int,default=12)
    p.add_argument('--max-width',type=int,default=0)
    p.add_argument('--initial-points',type=int,default=150000);p.add_argument('--max-points',type=int,default=500000)
    p.add_argument('--position-lr',type=float,default=0.00016);p.add_argument('--feature-lr',type=float,default=0.0025)
    p.add_argument('--opacity-lr',type=float,default=0.05);p.add_argument('--scaling-lr',type=float,default=0.005)
    p.add_argument('--rotation-lr',type=float,default=0.001);p.add_argument('--lambda-geo',type=float,default=0.05)
    p.add_argument('--geometry-from',type=int,default=None,help='Local level iteration: default L0=7000, new LoD=500')
    p.add_argument('--densify-from',type=int,default=500)
    p.add_argument('--opacity-reset-every',type=int,default=3000,help='Active layer only, during densification; 0 disables')
    p.add_argument('--densify-every',type=int,default=100);p.add_argument('--densify-fraction',type=float,default=0.5)
    p.add_argument('--grad-threshold',type=float,default=0.0002);p.add_argument('--min-opacity',type=float,default=0.05)
    p.add_argument('--percent-dense',type=float,default=0.01,help='Split vs clone size threshold as fraction of scene extent')
    p.set_defaults(func=fit)
    p=sub.add_parser('sweep');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=64);p.add_argument('--view-index',type=int,default=0);p.add_argument('--no-lod',action='store_true')
    p.set_defaults(func=sweep)
    args=parser.parse_args()
    if getattr(args,'steps',1)<1 or getattr(args,'frames',2)<2: parser.error('steps >= 1 and frames >= 2 required')
    args.func(args)


if __name__=='__main__': main()
