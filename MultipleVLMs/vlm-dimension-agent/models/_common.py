"""Shared helpers for the Transformers-backed adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import Config


def resolve_device(cfg: "Config") -> str:
    """Resolve ``cfg.device`` ('auto'/'cpu'/'cuda'/'mps') to a concrete device."""
    import torch

    dev = (cfg.device or "auto").lower()
    if dev != "auto":
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(cfg: "Config", device: str):
    import torch

    name = (cfg.dtype or "bfloat16").lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(name, torch.bfloat16)
    # CPU has poor/limited bf16/fp16 kernels; fall back to fp32 there for stability.
    if device == "cpu" and dtype is not torch.float32:
        return torch.float32
    return dtype


def quantization_config(cfg: "Config"):
    """Return a BitsAndBytesConfig if 4-bit quant was requested (CUDA only), else None."""
    if (cfg.quantization or "").lower() not in {"4bit", "4-bit", "int4"}:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
