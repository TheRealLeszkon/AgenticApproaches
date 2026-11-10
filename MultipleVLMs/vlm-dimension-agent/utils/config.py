"""Configuration loading and merge precedence.

Precedence (lowest -> highest): built-in defaults < config.yaml < environment
variables < CLI overrides. Env vars are prefixed ``VLM_`` (e.g. ``VLM_MODEL``,
``VLM_BACKEND``, ``VLM_TEMPERATURE``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


@dataclass
class Config:
    model: str = "qwen3-vl-8b"
    backend: str = "transformers"  # "transformers" | "mlx"
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "bfloat16"        # "bfloat16" | "float16" | "float32"
    quantization: str | None = None  # e.g. "4bit" (transformers/bitsandbytes on CUDA)
    max_image_side: int | None = 1536
    temperature: float = 0.2
    max_tokens: int = 4096
    output_dir: str = "runs"
    hf_token: str | None = None

    _KNOWN = None  # set below

    def merge(self, overrides: dict[str, Any]) -> "Config":
        for k, v in overrides.items():
            if v is None:
                continue
            if k in Config._KNOWN:
                setattr(self, k, v)
        return self


Config._KNOWN = {f.name for f in fields(Config) if not f.name.startswith("_")}


_ENV_MAP = {
    "VLM_MODEL": ("model", str),
    "VLM_BACKEND": ("backend", str),
    "VLM_DEVICE": ("device", str),
    "VLM_DTYPE": ("dtype", str),
    "VLM_QUANTIZATION": ("quantization", str),
    "VLM_MAX_IMAGE_SIDE": ("max_image_side", int),
    "VLM_TEMPERATURE": ("temperature", float),
    "VLM_MAX_TOKENS": ("max_tokens", int),
    "VLM_OUTPUT_DIR": ("output_dir", str),
    "HF_TOKEN": ("hf_token", str),
}


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, (attr, caster) in _ENV_MAP.items():
        if env_key in os.environ and os.environ[env_key] != "":
            try:
                out[attr] = caster(os.environ[env_key])
            except ValueError:
                pass
    return out


def load_config(config_path: str | None = None, cli_overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()

    if config_path and os.path.isfile(config_path):
        if yaml is None:
            raise RuntimeError("pyyaml is required to read config.yaml")
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        cfg.merge(data)

    cfg.merge(_env_overrides())

    if cli_overrides:
        cfg.merge(cli_overrides)

    return cfg
