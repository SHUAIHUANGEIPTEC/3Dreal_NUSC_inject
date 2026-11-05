# GPU 扩散路线（P5，已实现）

统一执行模型：每个 backend 有 3 个 CPU 安全动词 `available / prepare / collect` 与
1 个 GPU 动词 `run_local`。`prepare` 生成条件包（含 `command.sh`），拷贝到 GPU 服务器
`bash command.sh` 执行，`collect` 回读产物并写 provenance。所有 backend 带 stub，
无 GPU 时全链路仍可跑（产物标 `placeholder`，视频三级回退）。

## 条件包结构（`data/outputs/diffusion/<run_id>/<route>/`）

```
inputs/          ref.png / traj.txt / renderings.npy / composite.jpg / mask.png / ...
command.sh       GPU 服务器上的精确命令
backend.json     协议 + condition_sources + renderer + 参数（可追溯）
output/          collect() 回填产物
```

**条件源契约**：frame 路线（viewcrafter / stagg_b1 / sd_inpaint）的
`condition_sources` 必须包含 `/gauss/` 路径、`renderer` 必须是
`realinject.gauss.*`，否则 `prepare()` 直接失败——Blender 时代的批次不可能
悄悄喂给扩散路线。

## 实测（gs01，frame 4）

| 路线 | inputs | gauss 追溯 | run_local |
|---|---|---|---|
| viewcrafter | ref + traj | ✓（gauss/CAM_FRONT.png + refined） | placeholder |
| stagg_b1 | renderings.npy [25,576,1024,3] + ref | ✓（gauss + refined + canonical npz） | placeholder |
| sd_inpaint | composite + mask | ✓（gauss + composite） | placeholder |
| complete | manifest + 候选（DOG_03ce7747: view.png + half_shell.npy） | 资产库 | placeholder |
| magicdrive | specs.json | bbox 条件 | placeholder |

stagg_b1 的 `renderings.npy` 由**高斯渲染器**沿转台轨迹（±15°，25 帧）渲染注入
资产、静态原图作背景、frame0 = 和谐化参考图（官方 `nvs_single_view` 语义）；
预览 `render0.mp4` 随包产出。背景保持静态——stag3d 全场景 4D 渲染接入后替换。

| 路线 | backend | 作用 | 条件输入 | 权重/环境 |
|---|---|---|---|---|
| A | `realism_viewcrafter` | 场景真实感化：单图 + 轨迹 → 环绕视频 | 和谐化后的高斯合成图 1024×576、traj.txt | ViewCrafter（22 GB，已随 stag1_inject 下载） |
| B1 | `realism_stagg_b1` | 官方 Stag 扩散核：我们的 GT 几何渲染帧 → 照片级 | `renderings.npy [25,576,1024,3]` + ref 图 | `model_sparse.ckpt`（官方采样参数） |
| C | `harmonize_sd_inpaint` | 合成区重绘去痕迹 | 合成分 + visible 掩码 | diffusers SD inpaint |
| D | `complete` | 资产补全：AV2 半壳 + 条件图 → 完整资产；realcar 补车顶 | `view/<CAT>_<id>.png` + 半壳点云 | TRELLIS / Hunyuan3D-2 / GINA-3D 类 |
| E | `generate_magicdrive` | 生成式插入（bbox_add） | nuScenes 框 + 相机参数 | MagicDrive（需训练，本期占位） |

## GPU 服务器

沿用 stag1_inject 的同一台机器（≥24 GB）：ViewCrafter 与 stag_run 权重直接复用
（`local.yaml` 指向 `stag1_inject/third_party/ViewCrafter/checkpoints`）；P5 只需补
TRELLIS / Hunyuan3D-2 的仓库与权重。

## 安全约束

扩散重绘不得破坏标注：重绘后须校验投影框 IoU ≥ 0.5（`validate_ext`），否则回退到
未重绘合成图。
