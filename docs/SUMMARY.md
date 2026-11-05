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

（待补：注入前后对比、跨模态一致性图、devkit 加载验证截图）

## 5. 照片级 4D 演示（P4/P5）

（待补：freeze-view/freeze-time/multi-view 视频帧、扩散前后对比）

## 验收指标

| 阶段 | 指标 | 结果 |
|---|---|---|
| P1 | realcar A 档通过数 / AV2 通过数 | 运行 `build_asset_library.py` 后填写 |
| P2 | 解析球 α 误差 / 单资产渲染耗时 | 待测 |
| P3 | 注入样本 all_pass / devkit 导出校验 | 待测 |
