"""realinject — inject real extracted 3D assets into nuScenes.

Modules:
- ``assets``      asset ingest/normalize/quality/completion + library manifest
- ``gauss``       Gaussian asset representation, initialization, renderers
- ``inject``      nuScenes injection pipeline (read-only nusc-inject reuse)
- ``stag``        vendored stag3d 4D pipeline + nuScenes bridge
- ``harmonize``   CPU realism harmonization (exposure/shadow/occlusion)
- ``diffusion``   pluggable diffusion backends (complete/realism/harmonize/generate)
"""

__version__ = "0.1.0"
