# Physical Resolution Field (PRF)

从 ArtiFixer `thirdparty/PRF` 迁移的独立后处理模块，包含多表面 patch、
GS 深度遮挡、冻结几何对照、诊断、NPZ/CSV/PLY 导出和离线 3D 查看器。
不参与训练，也不修改输入高斯场。原始实现思路见 [DESIGN.md](DESIGN.md)，
字段定义见 [SCHEMA_V2.md](SCHEMA_V2.md)。

## 环境与运行

从 Skyfall-GS 根目录运行（Python 3.10+）：

```bash
python -m pip install -r thirdparty/PRF/requirements.txt
export PYTHONPATH="$PWD/thirdparty/PRF${PYTHONPATH:+:$PYTHONPATH}"
python -m prf --help
python -m prf --ply /path/to/fused.ply \
  --transforms /path/to/JAX_068/transforms_train.json \
  --output skyfall-gs_exp/prf/JAX_068/frustum \
  --visibility frustum
```

输入需为标准 binary little-endian float32 3DGS PLY（scale 为 log，opacity 为
logit，四元数为 wxyz），与原始训练相机处于同一米制坐标系。默认按
OpenCV/COLMAP C2W 读取，不翻转 Y/Z；仅 OpenGL 相机使用 `--opengl-c2w`。
尺度检查只是合理性筛查，不会自动恢复米制尺度。

GS 遮挡模式额外需要 PyTorch、CUDA 和 gsplat；这里直接调用 gsplat 1.4.0，
不再依赖 WorldAct 或 ArtiFixer。保留 RGB+D 累积深度除以 alpha 的协议。

```bash
python -m pip install -r thirdparty/PRF/requirements-depth.txt
python -m prf --ply /path/to/fused.ply \
  --transforms /path/to/JAX_068/transforms_train.json \
  --output skyfall-gs_exp/prf/JAX_068/gs_depth \
  --visibility gs-depth --device cuda \
  --depth-cache skyfall-gs_exp/prf/JAX_068/depth_cache
```

缓存按 view_id 命名，必须为不同 PLY、相机或图像分辨率使用不同缓存目录；
现有缓存不会自动检查这些输入是否变化。完整缓存可在没有 GPU/gsplat 时读取。
默认 `--max-anchors 8000 --anchor-voxel-m 4`，每个 anchor 可产生多个表面。

## 冻结几何对照与可视化

在上述计算命令中添加 `--reuse-patches /path/to/baseline/physical_resolution_field.npz`
和 `--compare-to /path/to/baseline/physical_resolution_field.npz`，即可固定 patch、
kernel、spacing，只重算观测项，并输出 `occlusion_delta.json`。
应使用相同 PLY 及匹配的 patch 配置。

```bash
python -m prf.visualize_3d \
  --field skyfall-gs_exp/prf/JAX_068/gs_depth \
  --scene /path/to/fused.ply \
  --transforms /path/to/JAX_068/transforms_train.json \
  --output skyfall-gs_exp/prf/JAX_068/viewer
python -m http.server 8000 --directory skyfall-gs_exp/prf/JAX_068/viewer
```

浏览服务目录中的 HTML 文件；Three.js 及许可证已随模块迁移。
输出包含 summary、实验 manifest、几何诊断与可见性状态。

## 指标解释与验证

`R_phys = max(R_obs, R_kernel, R_spacing)`，单位 m/equiv.pixel，越小越精细。
没有符合夹角条件的可见双视图时观测项为 inf。有限值不代表几何可信；
分析时应同时查看 `geometry_valid`。这些值是分辨率估计，不能作为真实几何误差。
原实验的 B1 重训练 baseline、绝对分辨率标定和浏览器交互不变性尚未闭环；
历史结果不代表本次迁移重新完成了 JAX_068 全量实验。

```bash
python -m pip install pytest
python -m pytest -c thirdparty/PRF/pytest.ini thirdparty/PRF/tests -q
```

保留原有 43 项测试，并补充缓存兼容、截断 PLY 和真实 CUDA 平面深度回归。
CUDA 测试在缺少 torch/gsplat/GPU 时跳过。

迁移验证（2026-09-12）：Skyfall Python 3.10 环境为 45 passed / 1 skipped
（未安装 gsplat）；现有 CUDA 环境（Python 3.12、PyTorch 2.11.0+cu128、
gsplat 1.4.0）为 46 passed。合成 PLY + 双训练视图 CLI 输出 5 个有限 patch，
冻结重算三项指标差值均为 0，3D 查看器成功导出 441 个高斯及 patch 凸包。
本次未重跑 JAX_068 全量实验。
