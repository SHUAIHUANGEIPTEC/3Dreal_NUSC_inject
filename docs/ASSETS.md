# 资产库：来源、规范化与质量门禁

![资产库接触表](figures/fig_asset_grid.png)

*规范坐标系下的资产接触表（前视/侧视/俯视三视图，颜色 = 纹理/反射强度）：上 8 行为
3DRealCar 裁剪网格，下 6 行为 Argoverse2 轨迹点云。车顶空腔与半壳形态一目了然。*

## 来源

| 来源 | 形态 | 数量 | 说明 |
|---|---|---|---|
| 3DRealCar 裁剪资产 | `mesh.obj` + `mesh.jpg`(4096²) + `mesh.mtl` + `meta.json` | 199 套（A 档 158） | iPhone 扫描重建，车顶/玻璃系统性缺失；导出为 Y-up、绕 Y 轴约 45° 斜置 |
| Argoverse2 轨迹点云 | `<CAT>_<id>.npz`（pts/intensity/dims/ego/cat） | 24 条（7 类） | 稀疏半壳聚合，物体局部系、Z-up、bbox 居中；`_merged/manifest.json` 给逐 track 统计 |
| AV2 条件图 | `av2_assets/<scene>/view/<CAT>_<id>.png` | 部分 track | 832×704 RGBA，供 GPU 补全路线做单视图条件 |
| **GLB（road_ood/Objaverse）** | `<name>.glb` + `<name>.json` | 3498 个 / 120 类 | 贴图 PBR 网格；服务器经 `glb_sources/` 软链进库（见 `3D_assets/README.md`），转换后 `kind=glb_mesh` |

> 源资产统一落位（2026-11-20 迁移）：`/srv/shared_data/3D_assets/`
> （`realcar_assets/` + `av2_assets/` + `glb_sources → Objaverse_v1_assets`；旧路径为兼容软链）。

## 规范化（`src/realinject/assets/normalize.py`）

统一规范坐标系 = `nusc-inject` 的 Blender 导入系：Z-up、落地 `z=0`、bbox min 归零、
X=车长 / Y=车宽 / Z=车高。

- **realcar**：UV 缝重复顶点合并（1 mm，34,754→31,879）→ 水平面 PCA 求 yaw 并对齐 +X
  （导出文件里车体斜置，AABB 虚胀为 3.82×4.60 m，真实 OBB 为 4.945×1.945 m）→
  Y-up→Z-up（`(x,y,z)→(x,-z,y)`）→ bbox min 平移归零。纹理必须直读 `mesh.jpg`
  （trimesh 对这套 obj/mtl 返回 2×2 占位材质）。
- **AV2**：点云已 Z-up、bbox 居中 → bbox min 平移归零；`dims`（box LWH）与点云
  extent 分开保存（位姿微调误差会让点云略超出框）。

## 质量门禁（`quality.py`）

realcar：仅 A 档；OBB 尺寸 L∈[3.5,6.5]、W∈[1.4,2.6]、H∈[1.0,2.0] m；纹理非退化
（std>5）且 UV 采样黑区占比 ≤5%（`max_texture_black_frac`——网源 2 图重建的
52% 黑斑车即被此门拒收）；与清单 L/W/H 偏差 ≤20%（不一致记 `bbox_disagrees_*`）。
AV2：点数≥300、方位跨度≥90°、点密度≥150 点/m²、覆盖率≥0.01；方位跨度≥200°
额外标记为**补全候选**（纯数值门——旧实现的 `cat==DOG` 特例把全部车辆挡在
CPU/GPU 补全路线之外）。
GLB（`gates.glb`）：无扫描等级/无外部清单，只查尺寸 sanity 窗 L/W/H∈[0.05,40] m
（单位坏的 4 km "Dog" 死在上限）+ 与 realcar 相同的纹理 std/黑率门。

## 产物

`data/library/`（gitignored）：

```
canonical/realcar/<scene>.npz   # verts/faces/uv/normals/colors + raw_bbox
canonical/av2/<CAT>_<id>.npz    # xyz/intensity/ego/box_lwh
canonical/glb/<stem>.npz        # GLB 转换缓存（mtime 比 glb 新则跳过重转）
completed/realcar/<key>.npz     # P5-D CPU 补洞后的新 canonical（fan_fill）
completed/av2/<key>.npz         # 半壳镜像 + 凸包闭合代理（2 万点，原壳存 shell_xyz）
assets_manifest.json            # schema 3dreal-assets/1（唯一索引）
gates_report.json               # 逐资产门禁结论 + 拒绝原因直方图
completion_report.json          # 补全跑批明细（逐资产方法/环路数/边界边）
```

补全后的 canonical 会**替换** manifest 里的 `canonical_path`，质量门禁在合并视图上
重跑（当前基线：13 条记录、11 过门、10 条已完成补全——5 fan_fill + 5 闭合代理，
az<200° 的 1 条按数值门跳过）。

## 复现

```bash
python scripts/build_asset_library.py            # 全量
python scripts/build_asset_library.py --limit 5  # 冒烟

# P5-D 补全跑批（CPU：realcar 洞填充 + av2 半壳闭合；GPU 生成式路线待权重）
python scripts/build_asset_library.py            # 先重建，刷新补全候选（az 门）
python scripts/complete_assets.py                # 写 completed/ + completion_report
python scripts/build_asset_library.py            # 再重建，把 completed/ 并回 manifest
```
