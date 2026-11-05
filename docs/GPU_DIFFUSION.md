# GPU 扩散路线（P5）

统一执行模型：每个 backend 有 3 个 CPU 安全动词 `available / prepare / collect` 与
1 个 GPU 动词 `run_local`。`prepare` 生成条件包（含 `command.sh`），拷贝到 GPU 服务器
`bash command.sh` 执行，`collect` 回读产物并写 provenance。所有 backend 带 stub，
无 GPU 时全链路仍可跑（产物标 `placeholder`，视频三级回退）。

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
