"""Generic MLX backend for Apple Silicon (via mlx-vlm).

One class serves ALL families: mlx-vlm handles per-architecture chat templating and image
processing internally, so we don't need per-family adapters here. This is the recommended
way to run the 7B/8B models locally on a Mac (4-bit quantized).

    pip install "mlx-vlm>=0.1.0"     # macOS arm64 only
    python main.py --model qwen3-vl-8b --backend mlx --images ...

Repo ids come from REGISTRY[...]["mlx"] (community 4-bit conversions).

NOTE: mlx-vlm's function signatures have shifted across releases; this adapter probes a
couple of known-good call shapes and normalizes the return value. Not exercised by the
local smoke test (which uses the transformers backend + SmolVLM).
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from models.base import VisionModel

if TYPE_CHECKING:
    from PIL.Image import Image
    from utils.config import Config


class MLXVisionModel(VisionModel):
    def __init__(self, repo: str, cfg: "Config"):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.repo = repo
        self.cfg = cfg
        self.model, self.processor = load(repo)
        self.mlx_config = load_config(repo)

    def _write_temp_images(self, images: list["Image"]) -> list[str]:
        paths = []
        tmpdir = tempfile.mkdtemp(prefix="vlm_mlx_")
        for i, img in enumerate(images):
            p = os.path.join(tmpdir, f"img_{i}.png")
            img.save(p)
            paths.append(p)
        return paths

    def generate(self, images: list["Image"], prompt: str) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # mlx-vlm loads images from file paths most reliably across versions.
        image_paths = self._write_temp_images(images)

        formatted = apply_chat_template(
            self.processor, self.mlx_config, prompt, num_images=len(image_paths)
        )

        common = dict(max_tokens=self.cfg.max_tokens, verbose=False)
        # temperature kwarg name varies by version ("temp" vs "temperature").
        temp = self.cfg.temperature or 0.0
        for temp_kwargs in ({"temp": temp}, {"temperature": temp}, {}):
            try:
                result = generate(
                    self.model,
                    self.processor,
                    formatted,
                    image=image_paths,
                    **common,
                    **temp_kwargs,
                )
                break
            except TypeError:
                continue
        else:  # pragma: no cover
            raise RuntimeError("mlx_vlm.generate call failed for all known signatures")

        # Normalize: newer versions return an object with `.text`; older return a str.
        if hasattr(result, "text"):
            return result.text.strip()
        return str(result).strip()
