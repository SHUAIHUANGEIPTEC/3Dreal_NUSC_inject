# 3Dreal-nusc-inject

把**真实提取的 3D 资产**（3DRealCar 实车扫描 + Argoverse2 动物/OOD 轨迹点云）注入
nuScenes 的 LiDAR 与 6 路相机，产出：

1. **标准 nuScenes 格式注入数据集**（devkit 可加载，供长尾/OOD 检测训练）；
2. **照片级 4D 演示视频**（多相机序列 + freeze-view / freeze-time / multi-view）。

与上一代 `nusc-inject` 的区别：输入不再是干净的 `.glb/.obj/.ply` 实体模型，而是
**扫描网格（带车顶破洞）与稀疏半壳点云**；因此本项目以**高斯泼溅（3DGS）**作为统一
资产表示与渲染器，并以**扩散模型**承担资产补全、场景真实感化、和谐化重绘与生成式插入。

## 状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 脚手架 / 路径 / 卫生门禁 | ✅ |
| P1 | 资产规范化与质量门禁（asset library） | ✅ |
| P2 | 高斯渲染器（CPU EWA + GPU gsplat） | ⏳ |
| P3 | 单场景注入 + 标准导出 | ⏳ |
| P4 | 和谐化 + 视频 | ⏳ |
| P5 | 扩散路线（补全/真实感化/重绘/生成式插入） | ✅（CPU stub + 条件包） |
| P6 | 卫生门禁全量 | ✅（随阶段执行） |
| P7 | 图文文档 + 发布 | ⏳ |

## 快速开始

```bash
conda activate nusc-inject          # 共享环境：numpy/trimesh/cv2/scipy/pytest

cp local.yaml.example local.yaml    # 填本机路径（gitignored）
python scripts/build_asset_library.py   # 资产库 → data/library/
python -m pytest -q                     # 单测（缺数据集自动 skip）
bash scripts/lint.sh                    # 卫生门禁
```

## 目录结构

```
src/realinject/
  paths.py            配置加载 + local.yaml 深合并 + nusc-inject 只读依赖探测
  assets/             资产接入/规范化/质量门禁/CPU 补洞/Gaussian 转换
  gauss/              高斯表示（GaussianCloud）与渲染器（P2）
  inject/             nuScenes 注入管线（P3，复用 nusc-inject 只读）
  stag/               stag3d 4D 管线与 nuScenes 桥接（P4）
  harmonize/          CPU 和谐化（曝光/接触阴影/遮挡重合成，P4）
  diffusion/          可插拔扩散后端（P5）
scripts/              入口脚本 + gpu/ 服务器侧驱动
configs/              可移植配置（机器路径只进 local.yaml）
docs/                 图文资料
tests/                CPU-only 单测（缺数据自动 skip）
```

## 文档

- [图文工作总结](docs/SUMMARY.md)
- [资产库：来源、规范化、质量门禁](docs/ASSETS.md)
- [高斯渲染设计](docs/GAUSSIAN_RENDERING.md)
- [GPU 扩散路线](docs/GPU_DIFFUSION.md)
- [迁移与部署](docs/MIGRATION.md)

## 依赖关系（只读）

| 目录 | 角色 |
|---|---|
| `../nusc-inject` | 几何注入管线（import `anchor`/`lidar_channel`/`occluders`/`validate`/`annotation`）——**不修改** |
| `../stage1` | Stag 官方仓库与 stag3d-repro（4D 重建 + 扩散核参考） |
| `../stag1_inject` | 上一代实现，能力移植来源 |

## 数据来源

- 3DRealCar 实车扫描（199 套裁剪网格，A 档 158）——版权归理想汽车，**资产文件不入库**；
- Argoverse2 物体轨迹点云（24 条，含 DOG/MOTORCYCLE/STROLLER/WHEELED_* 等）；
- 详见 [docs/ASSETS.md](docs/ASSETS.md)。
