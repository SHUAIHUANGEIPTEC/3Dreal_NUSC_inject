# 迁移与部署

## 本地（CPU）

```bash
git clone git@github.com:SHUAIHUANGEIPTEC/3Dreal_NUSC_inject.git
cd 3Dreal_NUSC_inject
conda activate nusc-inject          # numpy/trimesh/cv2/scipy/pytest/nuscenes-devkit
cp local.yaml.example local.yaml    # 填 dataset.root / assets.root / inject_nusc.root
python scripts/build_asset_library.py   # 资产库 -> data/library/
python scripts/run_inject_gauss.py --run-id gs01 --scene scene-0061 --samples 5
python scripts/run_harmonize.py --run-id gs01
python scripts/make_videos.py --run-id gs01
python -m pytest -q
bash scripts/lint.sh                # 卫生门禁
```

机器相关路径**只写** `local.yaml`（gitignored）；`configs/*.yaml` 保持可移植
（相对工程根或 `${ENV}`）。

## 只读依赖

| 目录 | 角色 |
|---|---|
| `../nusc-inject` | 几何注入管线（import `anchor`/`lidar_channel`/`occluders`/`validate`/`annotation`）——**不修改**；`bridge.assert_upstream_contract()` 校验上游契约 |
| `../stage1` | Stag 官方仓库与 stag3d-repro（参考） |
| `../stag1_inject` | 上一代实现：harmonize/video 移植来源、ViewCrafter 权重复用 |

## GPU 服务器（扩散路线）

1. 并排拷贝工程（或 clone）+ `local.yaml` 指向服务器上的数据集与权重；
2. 复用 stag1_inject 的 `third_party/ViewCrafter`（22 GB 权重，含 `model.ckpt` /
   `model_sparse.ckpt` / DUSt3R）与 `third_party/stag_run`；
3. CPU 侧生成条件包（可先在本地做好再拷）：

```bash
python scripts/prepare_diffusion.py --run-id gs01 --route all --frame 4
```

4. 服务器上按包内 `command.sh` 执行（各 backend 自带 CUDA/权重预检）：

```bash
bash data/outputs/diffusion/gs01/viewcrafter/command.sh
bash data/outputs/diffusion/gs01/stagg_b1/command.sh
```

5. 回拷 `output/` 后 `collect()` 回填产物；视频走三级回退，未跑扩散也不影响交付。

## 导出注意事项

- `annotation.export_dataset` 开头 `rmtree(export_root)`：**永远用独立 export_dir**，
  勿指向源数据根或已有成品导出（`inject/export.py::guard_export_root` 已拦截）。
- 重导出前若数据集路径变化，用上游的 `fix_export_links.py` 重指软链（或整树重导）。
