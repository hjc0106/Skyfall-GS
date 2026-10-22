# Stage 1 DSM 初始化与表面约束：安装和运行

本指南对应 `stage1-dsm` 分支，基于 `main@22134e1`，只增加 Stage 1 的 DSM 混合初始化和表面约束。运行代码不依赖 ArtiFixer 目录。Stage 2 的原有用法见 [主 README](README.md)。

| 组别 | 初始化 | 表面约束 | 参数 |
| --- | --- | --- | --- |
| A | SfM | 无 | 不传 `--dsm_prior` |
| B | DSM + SfM 混合 | 关闭 | `--dsm_prior PATH --no_dsm_surface` |
| C | 与 B 相同 | 默认开启 | `--dsm_prior PATH` |

表面项沿用迁移来源的 `log1p((r/sigma)^2)`，默认系数 0.05、sigma 0.5，预热 500 步；不是死区 Huber。法向用于初始化及表面距离，没有新增法向训练损失。

## 1. 安装环境

需要 Linux、NVIDIA GPU、可用的 NVIDIA 驱动，以及用于编译 CUDA 扩展的 C/C++ 编译器。下面以本地验证使用的 Python 3.10、PyTorch 2.11.0 + CUDA 12.8 为环境参考；新服务器仍需通过第 2 节检查。先用 `nvidia-smi` 确认 GPU 和驱动可用。

```bash
git clone --branch stage1-dsm --recurse-submodules https://github.com/hjc0106/Skyfall-GS.git
cd Skyfall-GS

# 已有仓库则使用：
# git fetch origin
# git switch stage1-dsm
# git submodule update --init --recursive

conda create -y -n skyfall-gs python=3.10
conda activate skyfall-gs
conda install -y -c nvidia cuda-toolkit=12.8 cuda-nvcc=12.8

# 先检查所选 Toolkit 内确实存在 nvcc，再设置 CUDA_HOME。
export CUDA_HOME="$CONDA_PREFIX"
# 如果 Conda 中没有 nvcc，但服务器已安装 CUDA 12.8，可改用：
# export CUDA_HOME=/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc" || { echo "CUDA_HOME 下没有 nvcc，请先安装 Toolkit 或修正路径"; exit 1; }
"$CUDA_HOME/bin/nvcc" --version
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-dsm.txt
python -m pip install matplotlib tqdm tensorboardX ninja

# 补齐来源子模块缺失的标准头文件；可重复执行。
# 修复 std::uintptr_t / uint32_t / uint64_t 未定义的编译错误。
python - <<'PYHEAD'
from pathlib import Path
p = Path("submodules/diff-gaussian-rasterization-depth/cuda_rasterizer/rasterizer_impl.h")
s = p.read_text()
headers = ("#include <cstdint>", "#include <stdint.h>", "#include <cstddef>")
missing = [h for h in headers if h not in s]
if missing:
    assert "#pragma once" in s, "Unexpected rasterizer_impl.h layout"
    p.write_text(s.replace("#pragma once", "#pragma once\n\n" + "\n".join(missing), 1))
print("Rasterizer headers OK")
PYHEAD

# 必须在安装 PyTorch 后编译；更换 PyTorch/CUDA 后需重新编译。
MAX_JOBS=2 python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-depth
MAX_JOBS=2 python -m pip install --no-build-isolation ./submodules/simple-knn
MAX_JOBS=2 python -m pip install --no-build-isolation ./submodules/fused-ssim
```

CUDA Toolkit 的 `nvcc` 用于编译，PyTorch wheel 带运行时，两者都需要核对。安装依赖后再次确认 PyTorch 版本；若被其他包替换，重新安装上述版本并重编译扩展。

冒烟使用 `lambda_pseudo_depth=0`，不会实例化 MoGe；Stage 1 的该设置不需要 FlowEdit 推理。完整训练配置默认保留 MoGe，首次使用需要能够获取对应模型权重。子模块仍需完整拉取。

## 2. 先验证安装

以下命令均从仓库根目录执行，每个新终端先激活环境。

```bash
conda activate skyfall-gs
export GPU=0
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA:", torch.cuda.is_available())'
python -c 'import diff_gauss, diff_gauss._C, simple_knn._C; from fused_ssim import fused_ssim; print("CUDA extensions OK")'

CUDA_VISIBLE_DEVICES="$GPU" python -m pytest -q \
  tests/test_stage1_coordinates.py tests/test_dsm_migration.py

python scripts/smoke_stage1_synthetic.py \
  --output "$PWD/outputs/dsm_synthetic_smoke" --gpu "$GPU"
```

合成冒烟会训练 C 组 4 步、第 2 步保存恢复，并与连续训练比较，最后渲染评估。查看 `outputs/dsm_synthetic_smoke/synthetic_smoke_report.json` 中的 `accepted`。输出目录请为每次运行单独命名。

两个来源数值对照测试需要设置 `DSM_REFERENCE_ROOT` 为原始来源 Skyfall-GS 目录；没有该目录时跳过，不影响独立运行。没有 CUDA 时部分 GPU 测试也会跳过，应结合合成冒烟判断环境是否可用。本地此前通过 27 项测试及合成冒烟；这不代表真实 JAX_068 的坐标和质量已验证。

## 3. 准备真实数据与路径

```text
JAX_068/
├── images/
├── masks/                    # 可选
├── transforms_train.json
├── transforms_test.json      # RGB 测试视角
└── points3D.txt
```

还需要 DSM GeoTIFF、对应 sidecar TXT、定义局部 ENU 原点的 `scene.loc`，以及与 `points3D.txt` 轨迹匹配的 COLMAP `images.txt` 或 `images.bin`。`points3D.txt` 的轨迹索引本身不包含二维观测坐标。

把下面示例改成服务器实际路径；`PRIOR_DIR` 和训练输出必须放在数据目录外。

```bash
export GPU=0
export DATASET=/data/datasets_JAX/JAX_068
export DSM_TIF=/data/Track3-Truth/JAX_068_DSM.tif
export DSM_TXT=/data/Track3-Truth/JAX_068_DSM.txt
export SCENE_LOC=/data/crops_rpcs_raw/JAX_068/scene.loc
export COLMAP_IMAGES=/data/matching_colmap/images.txt
export PRIOR_DIR="$PWD/outputs/priors/JAX_068"
export RUN_ROOT="$PWD/outputs/JAX_068_dsm_smoke_s0"
```

当前支持迁移来源的局部米制坐标路径。若 `transforms_train.json` 包含会触发读取器额外归一化的 R/T，构建器会拒绝，需先实现并验证相应先验坐标转换。不要直接复用别的场景的 `scene.loc` 或高程平移。

## 4. 构建 DSM 先验

```bash
python scripts/build_dsm_prior.py \
  --scene JAX_068 --scene-dir "$DATASET" \
  --dsm "$DSM_TIF" --dsm-txt "$DSM_TXT" \
  --scene-loc "$SCENE_LOC" --output "$PRIOR_DIR" \
  --sample-step 2 --skip-previews
```

输出包括 `dsm_prior.npz`、`registration.json`、`summary.json` 和 `dsm_surface.ply`。已有匹配数据的 NPZ 和配准文件可直接复用。

注意：`--sample-step` 单位为 DSM **像素**；0.5 m DSM 上 `2` 表示约 1 m 间距，要约 2 m 使用 `4`。默认 `2` 沿用来源实现。垂直平移从 SfM 与 DSM 重叠区估计并记录，没有硬编码历史偏移。JAX 的 EPSG:32617 来自 sidecar 数值解释，不能当成 GeoTIFF 内自带 CRS；该工具不是任意 CRS 的通用转换器。

## 5. 坐标检查：通过后才启动 A/B/C

```bash
CUDA_VISIBLE_DEVICES="$GPU" python scripts/validate_dsm_coordinates.py \
  --source "$DATASET" --prior "$PRIOR_DIR/dsm_prior.npz" \
  --observations "$COLMAP_IMAGES" --resolution 4 \
  --output "$RUN_ROOT/coordinate_gate.json"
```

检查包含真实训练轨迹重投影、DSM 正深度及图像内支持、坐标往返，并绑定输入文件哈希。失败退出码为 2；修改数据或先验后必须重新验证。相机平移数值很大本身不能证明不对齐，应使用完整投影。

当前本地 JAX_068 尚缺与 69,027 点对应的已验证二维观测。另一份 5,002 点 RPC 裁剪重建不能直接替代。因此真实 JAX_068 A/B/C 冒烟尚未完成，合成冒烟与历史结果均不能替代此检查。

## 6. 运行 A/B/C 冒烟

```bash
python scripts/run_stage1_comparison.py \
  --source "$DATASET" --prior "$PRIOR_DIR/dsm_prior.npz" \
  --output "$RUN_ROOT" --coordinate-report "$RUN_ROOT/coordinate_gate.json" \
  --profile stage1_dsm --smoke --gpu "$GPU" --seed 0
```

三组依次运行。配置为分辨率 4（2048 → 512）、200 步、初始总点数上限 20,000、MoGe 关闭；50 步开始加密，间隔 50，100 步 opacity reset 并保存恢复，C 组预热 20 步。检查有限损失/梯度、实际加点和裁剪、先验对齐、恢复及渲染。

尚未通过坐标检查时，可加 `--dry-run` 只导出运行命令，不训练。`smoke_report.json` 的 `accepted: true` 表示链路检查通过，不是重建质量结论。

## 7. 完整训练与单独 B/C

坐标检查通过后，在新的输出目录运行完整 30k 三组：

```bash
python scripts/run_stage1_comparison.py \
  --source "$DATASET" --prior "$PRIOR_DIR/dsm_prior.npz" \
  --output "$PWD/outputs/JAX_068_dsm_30k_s0" \
  --coordinate-report "$RUN_ROOT/coordinate_gate.json" \
  --profile stage1_dsm --gpu "$GPU" --seed 0
```

`configs/stage1_dsm.json` 保留主分支 RGB、外观、opacity 正则和 MoGe 设置。`source_bc_replay` 是来源的 10k 筛选协议，与该 30k 配置不同；比较时应统一协议、种子和点数预算。

如只跑 B/C，可先加 `--dry-run`，从输出报告的 `commands.B` / `commands.C` 取对应训练命令。直接调用 `train.py` 的开关示例如下；它使用其自身其他默认参数，**并不等价于上述完整 profile**，且不会自动执行坐标门禁，因此必须先完成第 5 节。

```bash
# C：混合初始化 + 默认开启表面约束；此示例关闭 MoGe。
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  -s "$DATASET" -m "$PWD/outputs/JAX_068_C_custom" \
  --eval --appearance_enabled --sh_degree 1 --iterations 30000 \
  --dsm_prior "$PRIOR_DIR/dsm_prior.npz" --lambda_pseudo_depth 0 \
  --checkpoint_iterations 10000 30000 --save_iterations 10000 30000

# B：复制上面的命令，改用独立输出目录，并增加 --no_dsm_surface。
```

可用 `--lambda_dsm_surface`、`--dsm_surface_sigma`、`--dsm_warmup_iters` 调整表面项。`--lambda_dsm_surface 0` 时实际无表面损失，不能视作有效的 C 组。`--init_max_points` 限制初始总点数，`--dsm_max_init_points` 只限制 DSM 初始化点预算，二者不要混淆。

## 8. 保存与恢复

`--checkpoint_iterations` 保存可恢复的 checkpoint，`--save_iterations` 保存 PLY。需要在中途暂停时使用 `--stop_after_iteration N`，并把 N 包含在两份保存列表中。恢复保留原计划总步数，例如冒烟始终是 `--iterations 200`，不是剩余 100。

```bash
# 将原训练命令的参数与输出目录保留，再增加：
# --start_checkpoint /path/to/model/chkpnt10000.pth
```

新 checkpoint 包含先验、坐标元数据、配置、优化器、随机状态和日程。未显式指定的表面开关沿用保存值；显式传入 `--no_dsm_surface`、`--dsm_surface` 或表面系数会覆盖保存值。主分支旧 tuple checkpoint 仍可加载。

## 9. 评估与查看结果

```bash
export MODEL="$RUN_ROOT/C_s0"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate_stage1.py \
  --model "$MODEL" --source "$DATASET" \
  --prior "$PRIOR_DIR/dsm_prior.npz" --iteration 200 --rgb-views 2
```

完整 30k 运行应将 `MODEL` 和 `--iteration` 改成对应目录及 30000。RGB 评估需要同一步的 checkpoint 和 PLY；只有 PLY 时可加 `--geometry-only`，也可用 `--output DIR` 把评估写入其他目录。

| 文件 | 内容 |
| --- | --- |
| `A_s0/`、`B_s0/`、`C_s0/` | 三组模型、checkpoint、日志 |
| `stage1_config.json` | 实际配置及有效开关 |
| `stage1_events.json` | 有限值、加密、裁剪、恢复检查 |
| `stage1_eval_200.json` | 高程误差、覆盖率、有限测试视角 L1/PSNR |
| `dsm_pred_iter_200.npz` | 预测 DSM 栅格 |
| `eval_200_*.png` | 测试视角渲染图 |
| `common_grid_metrics.json` | 三组共同有效格网上的对比 |
| `smoke_report.json` | 对比流程检查和阻塞原因 |

每组文件位于对应模型目录，最后两个汇总位于 `RUN_ROOT`。几何评估采用过滤 opacity 后的高斯中心最大高程栅格，应同时看 MAE、RMSE 和覆盖率。同一 DSM 用于先验和评估时标为 `controlled_same_dsm`；空间 holdout 不等于独立真值，冒烟也不代替多种子、点数匹配的质量实验。

## 10. 常见问题

### 找不到 nvcc

`nvcc not found at .../envs/skyfall-gs/bin/nvcc` 表示 `CUDA_HOME` 指向的目录没有编译器。PyTorch 能使用 CUDA 不代表已安装 nvcc。不要仅根据环境名设置该路径，先执行：

```bash
ls -l "$CONDA_PREFIX/bin/nvcc" /usr/local/cuda-12.8/bin/nvcc
```

选择确实存在的 CUDA 12.8 Toolkit，例如：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$CUDA_HOME/bin/nvcc" --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

若两处都没有，先执行第 1 节的 Conda Toolkit 安装，再检查 `$CONDA_PREFIX/bin/nvcc`。不要把不含编译器的 PyTorch 运行时目录用作 CUDA_HOME。

### 光栅化扩展的整数类型未定义

若 `rasterizer_impl.h` 报 `namespace std has no member uintptr_t` 或 `uint32_t` / `uint64_t is undefined`，执行第 1 节的头文件修复代码，再重新编译光栅化扩展。这是来源头文件缺少显式 include，与 DSM 训练参数无关。该步骤会使子模块显示本地修改；重新检出或重置子模块后可能需要再次执行。

### 缺少 ninja 与其他问题

`could not find ninja` 本身只是回退到慢速编译的警告；执行 `python -m pip install ninja` 可解决。日志中的普通 CUDA/GLM warning 也不等于构建失败，应定位实际的 `error:`。

- **CUDA 不可用**：确认激活了正确环境、`CUDA_VISIBLE_DEVICES` 是可用卡号，并检查驱动和 PyTorch CUDA 构建。
- **扩展编译失败或 undefined symbol**：确认子模块完整、`CUDA_HOME` 指向所选 Toolkit；在最终 PyTorch 版本下重新安装三个扩展。内存不足时设置 `MAX_JOBS=2` 再编译。
- **显存不足**：先跑分辨率 4、有限点数的冒烟；完整训练可调整分辨率及点数，但对比实验应同步调整并记录。
- **坐标检查拒绝**：查看 `coordinate_gate.json`，核对二维观测和 SfM 点、图像是否属于同一次重建。不要改 `accepted` 字段绕过验证。
- **没有 RGB 指标**：确认测试相机存在且有效；评估默认只取两个测试视角，并非完整图像指标套件。
- **GitHub push 提示无法读取 Username**：先在本机配置 GitHub HTTPS 凭据或 SSH 登录，再执行 `git push -u origin stage1-dsm`。

更多实现细节见 [迁移说明](docs/stage1_dsm_usage.md) 和 [迁移计划](docs/stage1_dsm_migration_plan.md)。
