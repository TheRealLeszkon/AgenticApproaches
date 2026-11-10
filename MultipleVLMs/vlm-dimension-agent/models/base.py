"""Model abstraction: one interface, many backends.

The agent layer only ever sees ``VisionModel.generate(images, prompt) -> str``. Which
concrete runtime is used (HF Transformers on CUDA/CPU/MPS, or MLX on Apple Silicon) is
decided here by ``load_model`` from config, using the REGISTRY below.

Adding a model = one REGISTRY entry (+ a family adapter if it's a new architecture).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from utils.config import Config


class VisionModel(ABC):
    """Common interface every backend/adapter implements."""

    #: friendly name this instance was loaded as (set by load_model)
    name: str = "unknown"

    @abstractmethod
    def generate(self, images: list["Image"], prompt: str) -> str:
        """Run the VLM on ``images`` + a text ``prompt``, return raw text output."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Model registry: friendly name -> {family, hf repo, mlx repo}
#
# `mlx` repo ids target the community 4-bit conversions. If a given repo does not
# exist yet, either quantize it yourself (`mlx_vlm.convert`) or use the transformers
# backend. See README "Downloading & running models".
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, dict[str, str | None]] = {
    "qwen3-vl-8b": {
        "family": "qwen3_vl",
        "hf": "Qwen/Qwen3-VL-8B-Instruct",
        "mlx": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    },
    "qwen3-vl-30b": {
        "family": "qwen3_vl",
        "hf": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "mlx": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
    },
    "internvl3-8b": {
        "family": "internvl3",
        "hf": "OpenGVLab/InternVL3-8B",
        "mlx": "mlx-community/InternVL3-8B-4bit",
    },
    "internvl3-14b": {
        "family": "internvl3",
        "hf": "OpenGVLab/InternVL3-14B",
        "mlx": "mlx-community/InternVL3-14B-4bit",
    },
    "minicpm-v-4.5": {
        "family": "minicpm",
        "hf": "openbmb/MiniCPM-V-4_5",
        "mlx": "mlx-community/MiniCPM-V-4_5-4bit",
    },
    "qwen2.5-vl-7b": {
        "family": "qwen25_vl",
        "hf": "Qwen/Qwen2.5-VL-7B-Instruct",
        "mlx": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    },
    # Tiny smoke-test model — validates the pipeline end-to-end on any hardware.
    "smolvlm-256m": {
        "family": "smolvlm",
        "hf": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "mlx": None,
    },
}


def available_models() -> list[str]:
    return list(REGISTRY.keys())


def load_model(name: str, cfg: "Config") -> VisionModel:
    """Instantiate the right adapter for ``name`` given ``cfg.backend``."""
    if name not in REGISTRY:
        raise ValueError(
            f"unknown model '{name}'. Available: {', '.join(available_models())}"
        )
    entry = REGISTRY[name]
    backend = (cfg.backend or "transformers").lower()

    if backend == "mlx":
        repo = entry.get("mlx")
        if not repo:
            raise ValueError(
                f"model '{name}' has no MLX (4-bit) repo configured; "
                f"use --backend transformers instead."
            )
        from models.mlx_backend import MLXVisionModel

        model = MLXVisionModel(repo, cfg)
        model.name = name
        return model

    if backend == "transformers":
        family = entry["family"]
        repo = entry["hf"]
        adapter_cls = _TRANSFORMERS_ADAPTERS[family]()
        model = adapter_cls(repo, cfg)
        model.name = name
        return model

    raise ValueError(f"unknown backend '{backend}' (expected 'transformers' or 'mlx')")


# Lazy imports so that, e.g., importing the qwen adapter doesn't drag in MiniCPM's
# remote-code deps unless that family is actually requested.
def _load_qwen3_vl():
    from models.qwen3_vl import Qwen3VLModel

    return Qwen3VLModel


def _load_qwen25_vl():
    from models.qwen25_vl import Qwen25VLModel

    return Qwen25VLModel


def _load_internvl3():
    from models.internvl3 import InternVL3Model

    return InternVL3Model


def _load_minicpm():
    from models.minicpm import MiniCPMModel

    return MiniCPMModel


def _load_smolvlm():
    from models.smolvlm import SmolVLMModel

    return SmolVLMModel


_TRANSFORMERS_ADAPTERS = {
    "qwen3_vl": _load_qwen3_vl,
    "qwen25_vl": _load_qwen25_vl,
    "internvl3": _load_internvl3,
    "minicpm": _load_minicpm,
    "smolvlm": _load_smolvlm,
}
