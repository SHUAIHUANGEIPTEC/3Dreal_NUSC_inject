"""CPU realism harmonization (exposure match, contact shadow, edge softening)."""
from .pipeline import collect_cam_frames, run_session
from .refine import get_refiner

__all__ = ["collect_cam_frames", "get_refiner", "run_session"]
