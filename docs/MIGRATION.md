# 迁移与部署

## 本地（CPU）

```bash
git clone git@github.com:SHUAIHUANGEIPTEC/3Dreal_NUSC_inject.git
cd 3Dreal_NUSC_inject
conda activate nusc-inject
cp local.yaml.example local.yaml   # 填 dataset.root / assets.root / inject_nusc.root
python scripts/build_asset_library.py
python -m pytest -q
```

机器相关路径**只写** `local.yaml`（gitignored）；`configs/*.yaml` 保持可移植
（相对工程根或 `${ENV}`）。

## GPU 服务器

1. 并排拷贝工程（或 clone）+ `local.yaml` 指向服务器上的数据集与权重；
2. 复用 stag1_inject 的 `third_party/ViewCrafter`（22 GB 权重，含 `model.ckpt` /
   `model_sparse.ckpt` / DUSt3R）；
3. 按需补 TRELLIS / Hunyuan3D-2（见 [GPU_DIFFUSION.md](GPU_DIFFUSION.md)）；
4. 条件包在 CPU 侧生成、`scp` 上服务器执行、结果回拷 `collect`。

## 只读依赖

`../nusc-inject` 只被 import，任何本地改动都会破坏可迁移性；`paths.add_inject_nusc_path`
同时兼容 `nusc_inject` / `inject_nusc` 两个历史包名。
