"""Backend registry."""
from __future__ import annotations

from .complete import CompletionBackend
from .generate_magicdrive import MagicDriveBackend
from .harmonize_sd_inpaint import SDInpaintBackend
from .realism_stagg_b1 import StagB1Backend
from .realism_viewcrafter import ViewCrafterBackend

#: route name -> backend class (CLI --route values)
ROUTES = {
    "viewcrafter": ViewCrafterBackend,
    "stagg_b1": StagB1Backend,
    "sd_inpaint": SDInpaintBackend,
    "complete": CompletionBackend,
    "magicdrive": MagicDriveBackend,
}

#: frame-derived routes must trace back to the Gaussian renderer
FRAME_ROUTES = ("viewcrafter", "stagg_b1", "sd_inpaint")


def get_backend(name: str):
    try:
        return ROUTES[name]()
    except KeyError:
        raise ValueError(
            f"unknown route {name!r}; choose from {sorted(ROUTES)}") from None
