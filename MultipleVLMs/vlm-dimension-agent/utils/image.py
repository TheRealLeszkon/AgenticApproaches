"""Image loading and (deliberately conservative) preprocessing.

Design note: we do NOT aggressively resize. Dimension estimation depends on being
able to read small cues — labels, handles, outlets, floor tiles, edges. So by default
we keep native resolution and only downscale when the longest side exceeds
``max_image_side`` (to stay within a model's context / memory limits). Set
``max_image_side`` to null in config to disable downscaling entirely.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def expand_image_args(patterns: Iterable[str]) -> list[str]:
    """Expand globs / directories / plain paths into a sorted list of image files.

    Shells usually expand ``*`` already, but we also handle the case where a glob or a
    directory is passed literally (e.g. quoted, or ``./images``).
    """
    paths: list[str] = []
    for pat in patterns:
        if os.path.isdir(pat):
            for name in sorted(os.listdir(pat)):
                full = os.path.join(pat, name)
                if os.path.splitext(name)[1].lower() in VALID_EXTS:
                    paths.append(full)
        elif any(ch in pat for ch in "*?[]"):
            paths.extend(sorted(glob.glob(pat)))
        else:
            paths.append(pat)

    # de-dupe preserving order, keep only valid image extensions that exist
    seen: set[str] = set()
    result: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if os.path.splitext(p)[1].lower() not in VALID_EXTS:
            continue
        if not os.path.isfile(p):
            continue
        result.append(p)
    return result


def load_images(paths: list[str], max_image_side: int | None = 1536) -> list[Image.Image]:
    """Load JPEG/PNG files as RGB PIL images, downscaling only if oversized."""
    images: list[Image.Image] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if max_image_side:
            w, h = img.size
            longest = max(w, h)
            if longest > max_image_side:
                scale = max_image_side / longest
                img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        images.append(img)
    return images
