#!/usr/bin/env python
"""Docs hygiene: relative links resolve, code fences balanced, figures exist.

Scans README.md and docs/*.md. External links are skipped. Run via
``scripts/lint.sh`` or directly.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text()
    # balanced code fences
    fences = re.findall(r"^```", text, flags=re.MULTILINE)
    if len(fences) % 2 != 0:
        errors.append(f"{path}: unbalanced code fences ({len(fences)})")
    # relative links / images must resolve
    for target in LINK_RE.findall(text):
        target = target.split("#", 1)[0].strip()
        if not target or target.startswith(SKIP_PREFIXES):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{path}: broken link -> {target}")
    return errors


def main() -> int:
    files = [ROOT / "README.md", ROOT / "RELEASE.md"]
    files += sorted((ROOT / "docs").glob("*.md")) if (ROOT / "docs").is_dir() else []
    errors: list[str] = []
    for f in files:
        if f.exists():
            errors += check_file(f)
    if errors:
        print("\n".join(errors))
        print(f"FAIL: {len(errors)} docs problem(s)")
        return 1
    print(f"docs ok ({len([f for f in files if f.exists()])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
