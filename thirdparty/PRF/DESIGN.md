目前在 Skyfall-GS 上验证的“物理分辨率”，核心是一个独立的 **PRF（Physical Resolution Field，物理分辨率场）后处理模块**：输入已经训练好的高斯场和原始训练相机，为局部表面计算分辨率及其瓶颈。主实现位于 [thirdparty/PRF](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF)，验证场景是 **JAX_068**。

**1. 物理分辨率的定义与计算**

当前实现把局部分辨率分成三个约束，取最差的一项：

\[
R_{\mathrm{phys}}=\max(R_{\mathrm{obs}},R_{\mathrm{kernel}},R_{\mathrm{spacing}})
\]

输出单位为 `m/equiv.pixel` 或 `cm/equiv.pixel`，数值越小表示越精细。这是根据观测和高斯表示推导的分辨率估计，不能直接当作真实几何误差。

| 分量 | 当前计算方法 | 代码 |
|---|---|---|
| 观测分辨率 `R_obs` | 在局部切平面上对相机投影求有限差分 Jacobian，取 `1 / 最小奇异值`；从夹角足够大的可见相机对中，选择较差单视图分辨率最优的一对 | [obs.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/obs.py:13) |
| 高斯核分辨率 `R_kernel` | 把世界坐标协方差投影到切平面，取最大标准差乘 `3.77`，再按透明度和投影面积加权取 90% 分位数 | [kernel.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/kernel.py:12) |
| 采样间距分辨率 `R_spacing` | 高斯中心投影到切平面，计算每点 6 个近邻的平均距离，再取 90% 分位数 | [spacing.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/spacing.py:12) |
| 融合与瓶颈标记 | 三者取最大，标记 `OBS / KERNEL / SPACING`；存在无穷值时标记 `INVALID` | [fusion.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/fusion.py:11) |

其中观测项可以写成：

\[
r_v=\frac{1}{\sigma_{\min}(J_v)},\qquad
R_{\mathrm{obs}}
=\min_{\angle(v_i,v_j)\ge12^\circ}\max(r_{v_i},r_{v_j})
\]

没有合格双视图时，`R_obs = inf`。

**2. 实现链路与关键代码**

入口是 [__main__.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/__main__.py:26)，主流程是 [pipeline.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/pipeline.py:26)：

`高斯 PLY + 训练相机 → 局部表面 patch → 可见性 → 三项分辨率 → 融合 → 导出与可视化`

- **输入与坐标约定**：输入米制高斯位置、尺度、旋转和透明度；检查场景尺度是否合理。JAX 训练相机按 **OpenCV/COLMAP C2W** 读取，默认不翻转 Y/Z。见 [io_gs.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/io_gs.py) 和 [cameras.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/cameras.py:13)。
- **局部表面构建**：默认 4 米体素选 anchor、48 近邻；按无向法向聚类，再按空间连通性拆分，经过两轮加权 PCA、厚度离群点筛选和 patch 去重。当前支持一个 anchor 输出多个表面。见 [patches.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/patches.py:55)。
- **几何可信度**：保存 `geometry_valid`、内点比例、平面残差、支持点数等。需要注意，主流程只筛 `patch.valid`，所以 **分辨率有限不等于几何可信**。见 [assemble_patch_from_members](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/patches.py:114)。
- **GS 深度遮挡**：渲染训练视图 expected depth 和 alpha，在 patch 中心及边界采样；深度容差考虑高斯尺度、表面厚度、入射角和数值精度。只有 `VISIBLE` 参与观测项计算。见 [gs_depth.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/gs_depth.py:29)、[occlusion.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/occlusion.py:83)。
- **冻结几何对照**：`recompute_observation_from_archive()` 复用 patch、`R_kernel` 和 `R_spacing`，只重新计算 `R_obs`，适合隔离遮挡或相机变化的影响。见 [pipeline.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/pipeline.py:94)。
- **输出**：包含 NPZ、CSV、PLY、可见性状态和实验清单；patch 成员和凸包使用 CSR 存储。3D 展示再把 patch 指标映射到高斯上。见 [SCHEMA_V2.md](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/SCHEMA_V2.md)、[export.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/export.py)、[gaussian_field.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/prf/gaussian_field.py)。

**3. JAX_068 已有的验证结果**

下表来自仓库保存的实验结果，`R_phys` 中位数按各版本有限值统计：

| 阶段 | 主要变化 | patch 总数 / 有限值数 | 结果 |
|---|---|---:|---|
| V1 | 视锥筛选 | 2,631 / 2,490 | `R_phys` 中位数约 **7.42 m** |
| GS depth 对照 | 冻结 V1 几何，只加入遮挡 | 2,631 / 2,386 | 104 个 patch 失去有效观测；kernel、spacing 变化均为 0 |
| V2 | 法向聚类，每 anchor 单簇 + GS depth | 4,412 / 4,072 | `R_phys` 中位数约 **6.80 m** |
| V3 | 多表面拆分、几何可信度及去重 | 13,823 / 12,855 | `R_phys` 中位数约 **6.63 m**；几何可信且有限的为 **7,448** |

结果来源：[V1](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/output/summary.json)、[遮挡对照](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/output/prf_gs_depth/occlusion_delta.json)、[V2 对比](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/output/prf_normal_cluster/geometry_delta.json)、[V3](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/output/prf_multisurface_v3/summary.json)。

这里有三个值得保留的结论：

- **当前估计主要受高斯核限制。** V3 有限值中，12,606 / 12,855，约 **98.1%** 的瓶颈是 `KERNEL`；观测项中位数约 **0.44 m**。
- **遮挡对照具有明确的变量隔离。** 固定几何后，只有观测项发生变化，验证了遮挡过滤的影响。
- **版本间中位数下降不能直接解释成重建质量提升。** V2/V3 改变了 patch 定义和覆盖范围，输入仍是同一个 fused PLY。

V3 的 [geometry_acceptance.json](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/output/prf_multisurface_v3/geometry_acceptance.json) 记录了全部验收项通过：可信表面覆盖从约 **56.26% 提高到 58.24%**，空间匹配的稳定平面上 kernel、spacing 比值中位数均为 1。可信集合的非平面率为 0，也与 `geometry_valid` 本身包含平面性筛选有关。

**4. 哪些验证已经完成，哪些还没有闭环**

已有测试覆盖：

- 针孔正视情况下，Jacobian 结果符合 `距离 / 焦距`。
- 固定几何、图像及内参缩小到 1/2、1/4，`R_obs` 分别变为 2、4 倍，kernel 和 spacing 不变。
- 固定成员，高斯尺度翻倍，kernel 翻倍；中心抽稀，spacing 增大。
- 法向符号不变性、墙面/地面拆分、离群点处理、遮挡分类、NPZ 格式及展示参数不修改指标。

对应 [test_prf.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/tests/test_prf.py)、[test_prf_b0.py](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/tests/test_prf_b0.py)、[tests 目录](/home/hongjiacheng/codes/ArtiFixer/thirdparty/PRF/tests)。V2 冻结记录写明当时 **38 项测试通过**；本次梳理没有重跑测试。

尚未闭环的部分：

- **B1 降分辨率重训练对照**：已有 1/2 分辨率 gsplat 实验记录，15,000 次迭代、69,027 个高斯、无 densification；保存的验收记录仍注明缺少同流程 1× baseline。因此还不能宣称验证了重训练后的分辨率变化。见 [retrain/README.md](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/retrain/README.md)。
- **真实分辨率标定**：上述 Skyfall 实验主要验证计算规律和几何处理，尚不足以证明绝对值与真实可分辨细节一致。
- **V3 长尾**：新表面的 `R_obs` 最大值达到约 416.94 m，验收文件明确要求进一步抽查。
- **浏览器交互不变性**：已有构建参数不修改 NPZ 的测试，尚未实现浏览器层 FoV/窗口变化自动验证。

后续阅读建议从 **`pipeline.py → patches.py → obs/kernel/spacing.py → occlusion.py`** 开始；实验对照以冻结的 [V2 baseline](/home/hongjiacheng/codes/ArtiFixer/skyfall-gs_exp/output/baselines/jax068_normalcluster_gsdepth_v2/README.md) 为基准，当前多表面实现看 V3。