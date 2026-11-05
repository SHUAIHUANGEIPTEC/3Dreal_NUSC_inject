"""Standard nuScenes export wrapper (read-only nusc_inject.annotation reuse)."""
from __future__ import annotations

from pathlib import Path


def guard_export_root(export_root, dataset_root) -> Path:
    """Reject exports that would clobber the source dataset or a live tree."""
    export_root = Path(export_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    if export_root == dataset_root:
        raise ValueError("export_root must differ from the dataset root")
    if dataset_root in export_root.parents:
        raise ValueError(f"export_root {export_root} is inside the dataset root")
    return export_root


def export(cfg: dict, specs: list, export_root) -> dict:
    """Export specs to a nuScenes-format dataset and verify it loads."""
    from nusc_inject import annotation

    export_root = guard_export_root(export_root, cfg["dataset"]["root"])
    result = annotation.export_dataset(cfg, specs, export_root)
    names = result.get("category_names") or []
    verify = annotation.verify_export(
        export_root, cfg["dataset"]["version"],
        n_expected=int(result["n_new_annotations"]),
        category_names=names,
        instance_tokens=result.get("instance_tokens"),
    )
    result["verify"] = verify
    return result
