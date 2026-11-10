"""Tiny staged console logger matching the format in the project spec.

Not using the stdlib ``logging`` module on purpose — we want clean, human-readable
stage banners for a POC that a person is watching interactively.
"""

from __future__ import annotations

import sys


def _emit(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def stage(name: str) -> None:
    """Print a stage banner, e.g. ``[Observer]``."""
    _emit(f"\n[{name}]")


def info(msg: str) -> None:
    _emit(msg)


def warn(msg: str) -> None:
    _emit(f"  ! {msg}")


def error(msg: str) -> None:
    _emit(f"  ✗ {msg}")


def final_summary(result: dict) -> None:
    """Print the final W/H/D + confidence block."""
    dims = result.get("dimensions", {}) or {}
    conf = result.get("confidence", {}) or {}
    ranges = result.get("ranges", {}) or {}

    def fmt(key: str) -> str:
        v = dims.get(key)
        r = ranges.get(key)
        if v is None:
            base = "?"
        else:
            base = f"{v:g} cm"
        if isinstance(r, (list, tuple)) and len(r) == 2 and None not in r:
            base += f"  (range {r[0]:g}–{r[1]:g})"
        return base

    _emit("\nFinal result:")
    if result.get("object"):
        _emit(f"Object: {result['object']}")
    _emit(f"Width:  {fmt('width_cm')}")
    _emit(f"Height: {fmt('height_cm')}")
    _emit(f"Depth:  {fmt('depth_cm')}")
    overall = conf.get("overall")
    if overall is not None:
        _emit(f"Confidence: {overall:g}")
    for w in result.get("warnings", []) or []:
        _emit(f"  ! {w}")
