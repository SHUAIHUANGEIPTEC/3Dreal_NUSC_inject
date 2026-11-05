# 图文工作总结（随阶段填充）

本页汇总各阶段的架构图与真实运行产物图。当前处于 P0/P1 阶段，图将在对应阶段补入
`docs/figures/` 后在此引用。

## 1. 总体架构

（P3 完成后补：资产 ingest → 高斯渲染 → 注入 → 扩散真实感化 → 导出流程图）

## 2. 资产库（P1）

![资产库接触表](figures/fig_asset_grid.png)

*199 套 realcar 裁剪网格与 24 条 AV2 轨迹经规范化后，A 档 158 套 / 12 条轨迹通过门禁。
全量统计见 [ASSETS.md](ASSETS.md)。*

## 3. 高斯渲染（P2）

（待补：网格 vs 高斯渲染 A/B、深度/alpha 掩码图）

## 4. 注入与导出（P3）

![注入前后对比](figures/fig_inject_before_after.png)

*scene-0061 f004：原始帧 → 高斯前景（两辆 realcar 扫描资产，车顶破洞如实呈现）→
合成 → 合成 + 注入 LiDAR 回波（红点落在目标上，跨模态一致）。5 帧全部
`validation all_pass=True`，导出 9 条新标注，devkit `verify ok`。*

## 5. 和谐化与视频（P4）

![和谐化 A/B](figures/fig_harmonize_ab.png)

*scene-0061 f004 CAM_FRONT：合成（前）→ 区域（红=目标、蓝=背景环）→ 精修（后）。
两遍法：pass A 逐帧 LAB 曝光统计 + 时域平滑（anti-flicker）→ pass B 精修
（接触阴影 + 边缘软化 + 噪声/锐化）。帧源 = 高斯渲染的 `gauss/<cam>.png`。*

实测（gs01 批次，5 帧）：

| 相机 | 曝光 \|dL\| | 色度身份保持 | 闪烁降低 | dE(target,ring) |
|---|---|---|---|---|
| CAM_FRONT | 56.6 → 14.0（75%） | 100% | 63% | 56.7 → 15.4 |
| CAM_FRONT_LEFT | 14.9 → 0.4（97%） | 99% | — | 15.2 → 3.2 |

视频产出（`data/outputs/videos/gs01/`）：6 路相机 mp4 + `grid_2x3.mp4` 网格，
帧源三级回退 `refined → composite → 原图`（未拍到注入物的相机用原图是物理正确语义）。

## 6. 照片级 4D 演示（P5 扩散路线）

（待补：freeze-view/freeze-time/multi-view 视频帧、扩散前后对比——见
[GPU_DIFFUSION.md](GPU_DIFFUSION.md)）

## 验收指标

| 阶段 | 指标 | 结果 |
|---|---|---|
| P1 | realcar A 档通过数 / AV2 通过数 | **158 / 12**（223 条记录，170 通过） |
| P2 | 解析球 α 误差 / 单资产渲染耗时 | <15% / **1.13 s**（120k 高斯 @1600×900） |
| P3 | 注入样本 all_pass / devkit 导出校验 | **5/5 帧 all_pass** / **verify ok（9 条新标注）** |
| P4 | 曝光改善相机数 / 视频产出 | **2/2 合成相机改善（75%/97%）** / 6 路 mp4 + 网格 |
