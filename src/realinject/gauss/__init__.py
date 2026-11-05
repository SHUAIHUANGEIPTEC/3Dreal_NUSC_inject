"""Gaussian asset representation, initialization, rendering and LiDAR adapter."""
from .lidar_splat import cloud_to_lidar_payload, estimate_normals
from .model import GaussianCloud, identity_quat, quat_from_z_axis
from .raster_cpu import RenderOut, render_cfg, render_gaussians
from .render import get_renderer, render, save_rgba_png

__all__ = [
    "GaussianCloud",
    "RenderOut",
    "cloud_to_lidar_payload",
    "estimate_normals",
    "get_renderer",
    "identity_quat",
    "quat_from_z_axis",
    "render",
    "render_cfg",
    "render_gaussians",
    "save_rgba_png",
]
