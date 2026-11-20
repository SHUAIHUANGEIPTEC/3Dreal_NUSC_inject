# 上下游串联：nusc-inject → 3Dreal-nusc-inject

本工程（下游）依赖上游 [`nusc-inject`](../../projects/nusc-inject) 的几何注入库
（只读 import），并在同一条 conda 环境 + 同一块 GPU 上端到端运行。本文记录
本机（RTX 5090）的完整打通方式与产物。

## 环境

统一使用上游创建的 conda 子环境 **`nusc-inject-ood`**（内含 torch 2.11+cu128、
gsplat、nuscenes-devkit、ruff、uv 全套）：

```bash
conda activate nusc-inject-ood
```

`local.yaml`（gitignored，机器路径唯一入口）：

```yaml
dataset:
  root: /home/ismr01/datasets/nuScenes          # 原始 v1.0-mini（软链）
assets:
  root: /srv/shared_data/3D_assets                # 3D 资产统一根（2026-11-20 迁入，
                                                   #  含 glb_sources -> 共享 road_ood GLB 库软链）
inject_nusc:
  root: /home/ismr01/projects/nusc-inject       # 上游几何注入库（只读）
```

## 3D 资产：网源获取（替代原机不可下载资产）

原 3DRealCar（理想汽车版权）与 AV2 提取件不在本机，改为**两条公开/本地重建路径**，
产物均按 `docs/ASSETS.md` 的文件契约布局：

```bash
# 1) Objaverse 车辆网格（hf-mirror 免注册直链）-> realcar_assets/ 布局
#    门禁：L 3.5-6.5 / W 1.4-2.6 / H 1.0-2.0 m、真贴图、grade A
#    manifest L/W/H 与 canonicalize_realcar 同函数计算 -> bbox 交叉核对零误差
python scripts/fetch_net_assets.py --classes "car,sports_car,truck,cab_(taxi),van,..." \
        --want 15 --max-try 80 --min-mb 0.3
#    -> /srv/shared_data/3D_assets/realcar_assets/ (7 辆过门禁) + manifest

# 2) 本地 nuScenes 聚合物体轨迹点云 -> av2_assets/_merged/ 同 npz 契约
#    (pts/intensity/dims/ego/cat + manifest: n/az_span/pts_per_m2/coverage)
python scripts/make_nusc_track_clouds.py --top-per-class 6
#    -> 6 条 REGULAR_VEHICLE 轨迹全部过 av2 门禁

# 3) 资产库（三源摄入：realcar 网格 + av2 点云 + glb_sources 的 GLB，
#    质量门禁 -> canonical npz + assets_manifest.json）
python scripts/build_asset_library.py
#    -> GLB 首次构建逐个转 canonical/glb/*.npz（~1-2 小时，之后按 mtime 缓存）；
#       GLB 记录 kind=glb_mesh、默认不发牌，须 --assets glb/<stem> 显式选用
#    -> 逐源计数见输出与 data/library/gates_report.json

# 4) P5-D CPU 补全（可选；先重建刷新 az 数值门候选，完成后重建并回 manifest）
python scripts/complete_assets.py
python scripts/build_asset_library.py
#    -> 10/13 completed（5 fan_fill + 5 半壳闭合代理，0 失败）
```

## GPU（gsplat @ RTX 5090）

`configs/inject_gauss.yaml -> gaussian.render.backend: gsplat` 现已**真正生效**：
管线此前硬 import 了 `raster_cpu.render_gaussians`，配置写了 gsplat 也不会用——
已改为 [`get_renderer`](../src/realinject/gauss/render.py) 分发。实测渲染期
nvidia-smi 显存 337→1250 MiB、5 帧全管线 ~65 s（CPU 路径单帧曾达 90 s，
且近距大 footprint 的 COO 展开会 OOM——raster_cpu 现已加 `max_pairs` 预算护栏）。

## 端到端（本机已跑通的命令）

```bash
conda activate nusc-inject-ood

# 单场景注入 + 标准 nuScenes 导出 + 和谐化（GPU，e2e 一条命令）
python scripts/run_inject_gauss.py --run-id gs_e2e --scene scene-0061 --samples 5
# -> 5 frames / 2 tracks / 9 annotations / devkit verify: ok
#    data/exports/gs_e2e/
#    export verify 通过后自动跑 P4 和谐化（覆盖/拒绝时不会跑）:
#      data/outputs/gs_e2e/scene-0061/f00*/refined/<cam>.jpg + harmonize_report.json
#    可选 --skip-harmonize 跳过；--skip-export 只跑 batch（也不和谐化）
#    独立重跑和谐化: python scripts/run_harmonize.py --run-id gs_e2e --scene scene-0061

# 卫生门禁（ruff + pyflakes + pytest + 硬路径 + docs）
PATH=~/miniforge3/envs/nusc-inject-ood/bin:$PATH bash scripts/lint.sh   # ALL CLEAN
```

## 上游 ood_nuscenes 的双层注入（数据层结合）

下游 `dataset.root` 可直接指向上游交付的 OOD 数据集（同为标准 nuScenes 格式，
上游注入帧为实体文件）——在**已含 OOD 材质目标的图像与点云**上叠加注入真实扫描
车辆；上游注入的 `movable_object.anomaly` 标注会被 `scene_occluders` 当作遮挡体，
几何语义自然衔接：

```bash
# 临时把 local.yaml 的 dataset.root 换成：
#   /home/ismr01/projects/nusc-inject/data/exports/ood_nuscenes
python scripts/run_inject_gauss.py --run-id ood_overlay --scene scene-0061 --samples 3
# -> 3 frames / 2 tracks / 6 annotations / devkit verify: ok
#    data/exports/ood_overlay/（跑完把 local.yaml 改回原始 mini）
```

## 本次打通对下游代码的修复清单

| 文件 | 修复 |
|---|---|
| `tests/inject/test_composite_parity.py` | 文件头被 shell 命令污染（`ls .."""` → SyntaxError） |
| `src/realinject/inject/pipeline_gauss.py` | 硬 import CPU 光栅器 → `get_renderer` 分发（gsplat 生效） |
| `src/realinject/gauss/raster_cpu.py` | COO 展开 OOM 护栏（`max_pairs` 预算 + 降采样 + 可操作报错） |
| `src/realinject/inject/validate_ext.py` | `silhouette_density` 加最小命中数（边缘小视野不再误杀）；`gauss_depth_vs_lidar` 限定 visible 像素 + 最小样本量（遮挡边缘不再误杀） |
| `configs/inject_gauss.yaml` | placement 起步 24–36 m 偏轴（5 帧窗口不进入 ego 贴脸区） |
| `local.yaml` | 补 `dataset.root` / `assets.root` 本机路径 |
| 新增 `scripts/fetch_net_assets.py` | 网源 Objaverse 车辆 → realcar 布局 + manifest |
| 新增 `scripts/make_nusc_track_clouds.py` | 本地 nuScenes → av2 npz 契约 + 门禁统计 |
