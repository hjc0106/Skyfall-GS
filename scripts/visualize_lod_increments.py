"""CPU-only checkpoint diagnostics; centers/scale proxies are not rendered contribution."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

COLORS=['#9ca9b5','#e48a26','#d3427b']
def stats(v):
 v=np.asarray(v);v=v[np.isfinite(v)]
 return dict(zip(['p50','p90','p99','max'],map(float,np.r_[np.quantile(v,[.5,.9,.99]),v.max()]))) if len(v) else {}
def arr(t):return t.detach().cpu().numpy()
def run(root,out):
 d=torch.load(root/'l2/l2_final.lod.pt',map_location='cpu',weights_only=False);ls=d['layers'];out.mkdir(parents=True,exist_ok=True)
 xyz=[arr(l['xyz']).astype(np.float64) for l in ls];origin=np.median(xyz[0],axis=0);pts=[p-origin for p in xyz]
 rng=np.random.default_rng(0);ids=[rng.choice(len(p),min(40000,len(p)),replace=False) if i==0 else np.arange(len(p)) for i,p in enumerate(pts)]
 report={'checkpoint':str(root/'l2/l2_final.lod.pt'),'origin_world':origin.tolist(),'display_counts':[len(i) for i in ids],'note':'Centers only; no visibility, alpha compositing or LoD weights. Scale is raw max sigma, not raster footprint. All statistics use all points.','layers':[]}
 all_ids=np.concatenate([arr(l['node_ids']) for l in ls]);all_xyz=np.concatenate(xyz);scales=[np.exp(arr(l['log_scales'])).max(1) for l in ls];all_s=np.concatenate(scales);order=np.argsort(all_ids);sorted_ids=all_ids[order]
 fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
 for j,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
  for i,p in enumerate(pts):axes[0,j].scatter(p[ids[i],a],p[ids[i],b],s=.3 if i==0 else .6,c=COLORS[i],alpha=.18 if i==0 else .55,rasterized=True,label=f'L{i}')
  axes[0,j].set(xlabel='XYZ'[a]+' - origin (scene units)',ylabel='XYZ'[b]+' - origin (scene units)',title='All layers: '+'XYZ'[a]+' / '+'XYZ'[b]);axes[0,j].set_aspect('equal');axes[0,j].legend(markerscale=8)
 for i,l in enumerate(ls):
  opacity=torch.sigmoid(l['opacity_logits']).numpy().ravel();s=scales[i];rec={'level':i,'count':len(s),'raw_max_scale':stats(s),'opacity':stats(opacity),'filter_3d':stats(arr(l['filter_3d']).ravel())}
  axes[1,0].hist(np.log10(np.maximum(s,1e-12)),bins=90,histtype='step',density=True,color=COLORS[i],label=f'L{i}')
  if i:
   par=arr(l['parent_ids']);rows=np.searchsorted(sorted_ids,par);assert np.all(rows<len(sorted_ids)) and np.all(sorted_ids[rows]==par)
   ix=order[rows];assert np.all(ix<sum(len(x) for x in xyz[:i]))
   off=np.linalg.norm(xyz[i]-all_xyz[ix],axis=1);ratio=s/all_s[ix];norm=off/all_s[ix]
   parent_level=np.searchsorted(np.cumsum([len(x) for x in xyz]),ix,side='right');u,c=np.unique(par,return_counts=True)
   rec.update(parent_scale_ratio=stats(ratio),offset_scene_units=stats(off),offset_parent_scale=stats(norm),fraction_scale_gt_parent=float(np.mean(ratio>1)),fraction_scale_gt_4parent=float(np.mean(ratio>4)),fraction_offset_gt_4parent=float(np.mean(norm>4)),unique_parents=len(u),children_per_parent=stats(c),parent_levels={str(k):int((parent_level==k).sum()) for k in range(i)})
   axes[1,1].hist(np.log10(np.maximum(ratio,1e-8)),bins=90,histtype='step',density=True,color=COLORS[i],label=f'L{i}')
   axes[1,2].hist(np.log10(np.maximum(norm,1e-8)),bins=90,histtype='step',density=True,color=COLORS[i],label=f'L{i}')
  report['layers'].append(rec)
  # Exact world-coordinate centers plus raw Gaussian attributes, all rows, no decimation.
  dtype=[(n,'<f4') for n in ['x','y','z','scale_x','scale_y','scale_z','opacity']]+[('layer','u1')]
  data=np.empty(len(s),dtype=dtype)
  for k,n in enumerate(['x','y','z']):data[n]=xyz[i][:,k]
  for k,n in enumerate(['scale_x','scale_y','scale_z']):data[n]=np.exp(arr(l['log_scales']))[:,k]
  data['opacity']=opacity;data['layer']=i
  with (out/f'L{i}_centers.ply').open('wb') as f:
   f.write(('ply\nformat binary_little_endian 1.0\ncomment Point centers, raw scales; not Gaussian splat renderer\nelement vertex '+str(len(s))+'\n'+''.join('property float '+n+'\n' for n in ['x','y','z','scale_x','scale_y','scale_z','opacity'])+'property uchar layer\nend_header\n').encode());data.tofile(f)
 for a,title in zip(axes[1],['Raw max sigma: log10(scene units)','Child / parent max sigma: log10 ratio','Parent offset / parent max sigma: log10']):a.set_title(title);a.legend();a.set_ylabel('Density')
 fig.suptitle(root.name+' | Incremental Gaussian centers and raw geometry');fig.savefig(out/'distribution.png',dpi=160);plt.close(fig)
 # Local 3D interactive plot, embedded JS for offline usage.
 fig=go.Figure()
 for i,p in enumerate(pts):
  q=p[ids[i]];fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode='markers',name=f'L{i} ({len(p):,} total; {len(q):,} shown)',marker=dict(size=1.5 if i==0 else 2,color=COLORS[i],opacity=.15 if i==0 else .8)))
 fig.update_layout(title=root.name+' | Toggle layers in legend; point centers only',scene=dict(aspectmode='data',xaxis_title='X - origin',yaxis_title='Y - origin',zaxis_title='Z - origin'),margin=dict(l=0,r=0,b=0,t=55));fig.write_html(out/'interactive_3d.html',include_plotlyjs=True)
 # Full-resolution camera projection: raw max-sigma estimate, no claim of raster footprint.
 fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained');projection=[]
 for row,stage in enumerate([1,2]):
  cam=d['stage_records'][stage]['cameras'][0];w=np.array(cam['w2c'],dtype=np.float64);frame=[]
  for i,p in enumerate(xyz):
   pc=p@w[:3,:3].T+w[:3,3];uv=pc[:,:2]/pc[:,2,None]*[cam['fx'],cam['fy']]+[cam['cx'],cam['cy']];valid=(pc[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cam['width'])&(uv[:,1]>=0)&(uv[:,1]<cam['height'])
   frame.append({'level':i,'center_in_frame_frac':float(valid.mean()),'raw_sigma_pixel_proxy':stats(cam['fx']*scales[i][valid]/pc[valid,2])})
   a=axes[row,i];a.hist2d(uv[valid,0],uv[valid,1],bins=100,range=[[0,2048],[0,2048]],norm=matplotlib.colors.LogNorm(),cmap='magma');a.set_ylim(2048,0);a.set_aspect('equal');a.set_title(f'{2**stage}x L{i}: {valid.sum():,} centers');a.set_xlabel('pixel x');a.set_ylabel('pixel y')
  projection.append({'zoom':2**stage,'camera':cam['name'],'layers':frame})
 fig.suptitle('Projected center density (log count); NOT visibility or contribution');fig.savefig(out/'projected_density.png',dpi=150);plt.close(fig)
 report['projection']=projection;(out/'report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('skyfall-gs_exp/lod_panel_3roi'));p.add_argument('--output',type=Path,default=Path('skyfall-gs_exp/lod_increment_distribution'));a=p.parse_args();reports=[]
 for name in ['building','trees','parking']:
  rec=run((a.root/name).resolve(),(a.output/name).resolve());reports.append(rec);print(name,json.dumps(rec['layers'][1:]),flush=True)
 a.output.mkdir(exist_ok=True,parents=True);(a.output/'summary.json').write_text(json.dumps(reports,indent=2))
 (a.output/'index.html').write_text('<meta charset="utf-8"><title>LoD incremental points</title><h1>JAX_068: LoD 增量点分布</h1><p>L0 灰色、L1 橙色、L2 粉色。统计使用全量点；交互图 L0 固定抽样 40k，L1/L2 全量。点中心密度和原始尺度不是渲染贡献，未计算可见性、LoD 权重或 alpha 合成。</p>'+''.join(f'<h2>{n}</h2><a href="{n}/interactive_3d.html">交互三维：旋转、缩放、图例切换层</a> | <a href="{n}/report.json">统计</a><p>'+ ' '.join(f'<a href="{n}/L{i}_centers.ply">L{i} 全量点中心 PLY</a>' for i in range(3))+f'</p><img width="100%" src="{n}/distribution.png"><img width="100%" src="{n}/projected_density.png">' for n in ['building','trees','parking']))
