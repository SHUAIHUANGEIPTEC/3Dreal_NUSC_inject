# 高斯渲染设计（P2 实现）

## 为什么用高斯

输入资产没有干净的封闭实体：realcar 网格缺车顶/玻璃，AV2 只有稀疏半壳点云。
3DGS 把两者统一成同一种表示（位置 + 各向异性尺度 + 旋转 + 不透明度 + 颜色），
渲染出 RGBA + 深度 + alpha，供合成与遮挡判定直接使用。

## 契约

```python
render(cam: CameraSpec, cloud: GaussianCloud, cfg) -> RenderOut
RenderOut(rgb uint8, alpha f32, depth f32[inf=empty], mask bool, info dict)
```

`CameraSpec` 与 `nusc-inject` 的 `camera_channel.build_job` 输出同构（name/K/
T_cam2global/width/height），相机系为 OpenCV（x 右、y 下、z 前，**不加** Blender 的
`diag(1,-1,-1)`）。

## 后端

- **cpu（默认，本机可用）**：EWA 2D 协方差投影 + 前到后 alpha 混合，numpy 分块。
  起点为 `stag3d/renderer.py` 的 z-buffer 点泼溅，升级为各向异性高斯核；
  空像素深度保持 `inf`（与 `nusc_inject.composite._zbuffer_depth` 约定一致）。
- **gsplat（GPU 服务器）**：`gsplat`/nerfstudio 光栅器，lazy import；接口完全相同，
  下游合成与扩散条件打包不感知后端。

## 高斯初始化

- **网格**：面积加权表面采样；颜色 = 纹理 UV 双线性采样（经重心坐标插值）；
  尺度 = kNN 中位间距 × 0.8（3–150 mm 截断）；旋转 = 面法向（局部 +Z 对齐法向）。
- **点云**：尺度 = kNN 平均间距 × 1.2；颜色 = 反射强度灰度（或密度 turbo 色）；
  不透明度 0.85；旋转单位四元数。

## 与注入管线的关系

`gauss/<cam>.png`（RGBA 前景）与 `depth/<cam>_obj_depth.npy` 与上游 Blender 通道
产物目录语义一致，因此和谐化模块的掩码重建
`visible = (alpha>thr) & (obj_depth<=scene_depth+margin)` 保持逐位同义。
