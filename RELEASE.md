# 发布说明 — 3Dreal-nusc-inject v0.1.0

**日期**：2026-11-05　｜　**仓库**：https://github.com/SHUAIHUANGEIPTEC/3Dreal_NUSC_inject

## 一句话

把**真实提取的 3D 资产**（3DRealCar 实车扫描网格 + Argoverse2 动物/OOD 轨迹点云）
注入 nuScenes 的 LiDAR 与 6 路相机，产出**标准 nuScenes 格式注入数据集**与
**多相机序列视频**；以**高斯泼溅（3DGS）**作统一资产表示与渲染器，以**扩散模型**
承担资产补全与场景真实感化。

## 与 nusc-inject 的差异

| | nusc-inject | 3Dreal-nusc-inject |
|---|---|---|
| 输入资产 | 干净实体模型（.glb/.gltf/.obj/.fbx/.ply） | **真实提取资产**：扫描网格（车顶破洞）+ 稀疏半壳点云 |
| 相机通道 | Blender Cycles 网格渲染 | **3DGS 高斯泼溅**（CPU EWA + GPU gsplat 接口） |
| 真实感化 | 无 | 和谐化（曝光/阴影/anti-flicker）+ **5 条扩散路线** |
| 上游关系 | — | `nusc-inject` **只读** import（零改动） |

## 能力清单（v0.1.0）

1. **资产库**（P1）：199 套 3DRealCar + 24 条 AV2 轨迹规范化入库，质量门禁
   （158 + 12 通过），canonical npz + `assets_manifest.json`。
2. **高斯渲染器**（P2）：EWA 足迹 + 软 z 累积，120k 高斯 @1600×900 **1.13 s**；
   与 stag3d 点渲染 parity（点核召回 >0.95，深度中位差 <0.01 m）。
3. **单场景注入 + 标准导出**（P3）：scene-0061 5 帧全部 `all_pass`，9 条新标注，
   devkit `verify ok`；高斯深度 vs LiDAR 中位差 ~3 mm。
4. **和谐化 + 视频**（P4）：曝光 |dL| 改善 75%/97%，色度身份 100% 保持；
   6 路相机 mp4 + 2×3 网格（三级帧源回退）。
5. **扩散路线**（P5）：viewcrafter / stagg_b1 / sd_inpaint / complete / magicdrive
   五个 backend 出条件包（CPU stub 全链路可跑），条件源强制追溯 `/gauss/` 帧。

## 快速开始

```bash
conda activate nusc-inject
cp local.yaml.example local.yaml     # 填本机路径
python scripts/build_asset_library.py
python scripts/run_inject_gauss.py --run-id gs01 --scene scene-0061 --samples 5
python scripts/run_harmonize.py --run-id gs01
python scripts/make_videos.py --run-id gs01
bash scripts/lint.sh                 # 85 测试 + 卫生门禁
```

## 验收指标

| 阶段 | 结果 |
|---|---|
| P1 资产库 | 158 realcar A 档 + 12 AV2 轨迹通过门禁 |
| P2 渲染 | 1.13 s / 120k 高斯；解析球 α 误差 <15% |
| P3 注入 | 5/5 帧 all_pass；9 条新标注；devkit verify ok |
| P4 和谐化 | 曝光 75% / 97% 改善；6 路视频 + 网格 |
| P5 扩散 | 5 条件包；3 条 frame 路线 gauss 追溯通过 |
| 质量 | 85 测试全绿；ruff/pyflakes/硬编码路径/docs 门禁全绿 |

## 已知限制

- 3DRealCar 车顶/玻璃缺失是扫描固有缺陷（深度图如实呈现）；补全走 GPU 路线 D。
- AV2 动物为半壳点云，几何补全依赖单视图 3D 生成（TRELLIS/Hunyuan3D-2 类）。
- 扩散真实感化（A/B1/C）与 MagicDrive 生成式插入（E）需 GPU 服务器实跑；
  条件包与驱动已就绪（`scripts/gpu/`），权重沿用 stag1_inject 旧机。
- 4D 演示（freeze-view/time/multi-view）待 stag3d 管线移植（后续版本）。

## 数据来源与许可

- **3DRealCar**：© 理想汽车（数据）；工具包 Apache-2.0 仅覆盖代码。**资产文件不入库**；
  docs 图中的扫描渲染图仅用于研究展示。
- **Argoverse 2**：CC BY-NC-SA 4.0（非商用）。
- **nuScenes**：遵循其官方条款。

## 环境

- Python ≥3.10，conda env `nusc-inject`（numpy 1.26 / trimesh 5.2 / opencv 5.0 /
  scipy 1.15 / nuscenes-devkit 1.2 / pytest 9.2）
- GPU 服务器（可选）：≥24 GB，ViewCrafter + stag_run 权重（复用 stag1_inject）
