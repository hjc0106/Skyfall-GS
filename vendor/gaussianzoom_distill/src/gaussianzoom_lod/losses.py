"""GaussianZoom Eq. (10), with explicit RaDe-GS depth-normal consistency."""
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim


def rgb_loss(image, target, dssim=0.2):
    return (1-dssim)*(image-target).abs().mean() + dssim*(1-fused_ssim(image[None],target[None]))


def geometry_loss(package, camera):
    """RaDe-GS camera-space expected-depth normal consistency.

    Same cross(dy,dx) stencil as official utils.graphics_utils.depth_to_normal.
    Excludes borders and unobserved pixels instead of inventing surface normals
    for background. This mask is a documented engineering choice.
    """
    depth, normal = package['depth'], package['normal']
    y, x = torch.meshgrid(torch.arange(camera.height, device=depth.device),
                          torch.arange(camera.width, device=depth.device), indexing='ij')
    rays = torch.stack(((x-camera.cx)/camera.fx, (y-camera.cy)/camera.fy, torch.ones_like(x)), dim=0)
    points = rays*depth
    dy = points[:,2:,1:-1]-points[:,:-2,1:-1]
    dx = points[:,1:-1,2:]-points[:,1:-1,:-2]
    normals = F.normalize(torch.cross(dy,dx,dim=0),dim=0,eps=1e-8)
    valid = (package['alpha'].detach() > 0.05) & (depth.detach()>0) & torch.isfinite(depth.detach())
    # All stencil pixels must be observed.
    valid = valid[:,1:-1,1:-1] & valid[:,2:,1:-1] & valid[:,:-2,1:-1] & valid[:,1:-1,2:] & valid[:,1:-1,:-2]
    error = 1-(normal[:,1:-1,1:-1]*normals).sum(dim=0).clamp(-1,1)
    return (error*valid[0]).sum()/valid.sum().clamp_min(1)


def dual_scale_loss(package, camera, hr_target, lr_target, lambda_hr=0.6,
                    lambda_lr=0.4, lambda_geo=0.05, dssim=0.2):
    """Downsample THIS HR render, not a second render with different LoD weights.

    LR and HR targets must share pose/FOV/crop. All target tensors are detached
    observations. Bicubic resizing uses exact LR shape (noninteger ratios allowed).
    """
    image = package['image']
    if image.shape != hr_target.shape:
        raise ValueError('HR camera/render and target dimensions disagree')
    down = F.interpolate(image[None],size=lr_target.shape[-2:],mode='bicubic',align_corners=False,antialias=True)[0]
    hr = rgb_loss(image,hr_target,dssim)
    lr = rgb_loss(down,lr_target,dssim)
    geo = geometry_loss(package,camera) if lambda_geo else image.sum()*0
    total = lambda_hr*hr + lambda_lr*lr + lambda_geo*geo
    return total, {'hr':hr.detach(),'lr':lr.detach(),'geo':geo.detach()}
